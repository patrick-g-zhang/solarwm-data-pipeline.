#!/usr/bin/env python3
"""Caption raw videos with the local Qwen2.5-VL scene-static captioner.

This is only the caption stage (``solar_wm_data.caption``). It does not
estimate camera pose. Pose, intrinsics, filtering and packaging stay in the
default-mode fleet (``scripts/run_solarwm_fleet.py`` / ``solarwm-pipeline``).

Input: video files, or directories that contain them.
Output, per video::

    <out>/<clip_id>/prompt.txt

Weights: ``$SOLAR_WM_WEIGHTS/qwen25vl7b`` (default ``<repo>/weights/qwen25vl7b``),
the Qwen2.5-VL-7B-Instruct snapshot. ``scripts/setup_real_env.sh`` downloads it
together with the pose models; to fetch only this checkpoint::

    python3 -c "from huggingface_hub import snapshot_download as s; \
s('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='weights/qwen25vl7b')"

    python3 scripts/caption_videos.py videos/ --out captions/
    python3 scripts/caption_videos.py a.mp4 b.mp4 --out captions/ --nframes 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _weights_dir() -> Path:
    root = Path(os.environ.get("SOLAR_WM_WEIGHTS", str(_REPO / "weights")))
    return root / "qwen25vl7b"


def _weights_ready(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    return any(path.glob("*.safetensors")) or (path / "model.safetensors.index.json").is_file()


def collect_videos(inputs: list[Path]) -> list[tuple[str, Path]]:
    """Return (clip_id, video) pairs. clip_id is unique and filesystem-safe.

    A ``<clip_id>/video.mp4`` clip directory is named after the directory, so the
    caption lands next to that clip's poses. Paths with a ``_``-prefixed component
    (``_work``, ``_logs``) are pipeline scratch and are skipped.
    """
    found: list[tuple[str, Path]] = []
    for src in inputs:
        if src.is_dir():
            videos = sorted(
                p for p in src.rglob("*")
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS
                and not any(part.startswith("_") for part in p.relative_to(src).parts)
            )
            if not videos:
                raise SystemExit(f"no videos under {src}")
            for video in videos:
                rel = video.relative_to(src).with_suffix("")
                if video.name == "video.mp4" and len(rel.parts) > 1:
                    rel = rel.parent
                clip_id = "__".join(rel.parts)
                found.append((clip_id, video))
        elif src.is_file() and src.suffix.lower() in VIDEO_EXTS:
            name = src.parent.resolve().name if src.name == "video.mp4" else src.stem
            found.append((name, src))
        else:
            raise SystemExit(f"not a video or directory: {src}")

    seen: dict[str, int] = {}
    unique: list[tuple[str, Path]] = []
    for clip_id, video in found:
        n = seen.get(clip_id, 0)
        seen[clip_id] = n + 1
        if n:
            clip_id = f"{clip_id}__{n + 1}"
        unique.append((clip_id, video))
    return unique


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="video files and/or directories")
    ap.add_argument("--out", type=Path, required=True, help="directory for <clip_id>/prompt.txt")
    ap.add_argument("--nframes", type=int, default=8, help="frames sampled from each video (default 8)")
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value, e.g. 0")
    ap.add_argument("--force", action="store_true", help="re-caption even if prompt.txt already exists")
    args = ap.parse_args(argv)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("SOLAR_WM_WEIGHTS", str(_REPO / "weights"))
    # A prompt.txt sitting next to the source video is a previous caption, not
    # ground truth. This script always asks the model.
    os.environ["SOLAR_WM_NATIVE_CAPTION"] = "0"

    weights = _weights_dir()
    if not _weights_ready(weights):
        raise SystemExit(
            f"Qwen2.5-VL weights not found at {weights}. Download with:\n"
            "  python3 -c \"from huggingface_hub import snapshot_download as s; "
            f"s('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='{weights}')\""
        )

    items = collect_videos(args.inputs)
    args.out.mkdir(parents=True, exist_ok=True)
    _log(f"{len(items)} video(s), weights={weights}")

    from solar_wm_data.caption import caption_clip
    from solar_wm_data.manifest import ClipRecord

    models_cfg = {"dry_run": False, "caption_nframes": args.nframes}
    failed = 0
    for clip_id, video in items:
        dest = args.out / clip_id / "prompt.txt"
        if dest.is_file() and dest.stat().st_size > 0 and not args.force:
            _log(f"skip {clip_id} (exists {dest})")
            continue
        _log(f"caption {clip_id} <- {video}")
        try:
            rec = ClipRecord(
                clip_id=clip_id, source="web", video_path=str(video.resolve()), mode="default",
            )
            text = (caption_clip(rec, models_cfg) or "").strip()
            if not text:
                raise RuntimeError("model returned an empty caption")
        except Exception as exc:  # one bad video must not drop the rest
            failed += 1
            _log(f"  FAIL {clip_id}: {type(exc).__name__}: {exc}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n", encoding="utf-8")
        _log(f"  -> {dest}")

    done = len(items) - failed
    _log(f"CAPTION_VIDEOS_DONE {done}/{len(items)} written, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
