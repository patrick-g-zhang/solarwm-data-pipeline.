#!/usr/bin/env python3
"""Estimate metric camera poses for raw videos with the default-mode pose engine.

This is the fleet's `default` pose path (``solar_wm_data.pose.vipe_cli``): Pi3X +
MoGe-2 fused metric depth, then the patched VIPE SLAM with per-frame intrinsics
BA. Videos are first cut into contiguous spec windows by the fleet's own
``_split_spec_windows``. Filtering, captioning and packaging are not run here;
``scripts/caption_videos.py`` can caption the same output directory afterwards.

Input: video files, or directories that contain them.
Output, one directory per spec window::

    <out>/<clip_id>[_wNNN]/video.mp4        the exact frames the poses describe
                           poses.npy        (N,4,4) camera-to-world, metres
                           intrinsics.npy   (N,4) per-frame fx, fy, cx, cy in pixels
                           meta.json        fps, size, scale_factors, spec, vipe commit

``meta.json`` is written last, so its presence marks a finished window and a rerun
skips it. Run with the Python env that has torch, Pi3 and MoGe (the precompute step
runs under ``sys.executable``); VIPE runs from ``.venv-vipe``.

    python3 scripts/pose_videos.py videos/ --out poses/ --spec 960f --gpus 0,1,2,3
    python3 scripts/pose_videos.py clips/ --out poses/ --spec none      # no cutting
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from caption_videos import collect_videos  # noqa: E402

PINNED_VIPE = "95a8816"
_TAG = ""


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]{_TAG} {msg}", flush=True)


def _check_setup() -> str:
    """Fail before any GPU work if a model or tool is missing. Returns the VIPE commit."""
    wm = Path(os.environ["SOLAR_WM_ROOT"])
    weights = Path(os.environ["SOLAR_WM_WEIGHTS"])
    vipe_bin = Path(os.environ.get("SOLAR_WM_VIPE_BIN") or wm / ".venv-vipe" / "bin" / "vipe")
    need = {
        "Pi3 weights": weights / "pi3" / "model.safetensors",
        "MoGe-2 weights": weights / "moge2" / "model.pt",
        "Pi3 code": wm / "third_party" / "Pi3" / "pi3",
        "VIPE CLI": vipe_bin,
        "VIPE solarwm pipeline": wm / "third_party" / "vipe" / "configs" / "pipeline" / "solarwm.yaml",
        "VIPE pi3xmoge backend": wm / "third_party" / "vipe" / "vipe" / "priors" / "depth" / "pi3x_moge.py",
    }
    missing = [f"{name}: {path}" for name, path in need.items() if not path.exists()]
    if missing:
        raise SystemExit("pose engine is not set up:\n  " + "\n  ".join(missing))
    commit = subprocess.run(
        ["git", "-C", str(wm / "third_party" / "vipe"), "rev-parse", "--short=7", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    if commit != PINNED_VIPE:
        _log(f"WARNING: VIPE is at {commit or 'unknown'}, not the pinned {PINNED_VIPE}")
    return commit


def _load_fleet(spec: str):
    """The fleet module, for its window cutter. Its spec is read at import time."""
    os.environ["SOLAR_WM_SPEC"] = spec
    path = _REPO / "scripts" / "run_solarwm_fleet.py"
    mod_spec = importlib.util.spec_from_file_location("run_solarwm_fleet", path)
    fleet = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(fleet)
    return fleet


def _stage_video(video: Path, dst: Path, fleet) -> None:
    if video.suffix.lower() == ".mp4":
        shutil.copyfile(video, dst)
        return
    ffmpeg = fleet._ffmpeg_bin() if fleet else (shutil.which("ffmpeg") or "ffmpeg")
    base = [ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-an"]
    if subprocess.run(base + ["-c:v", "copy", str(dst)]).returncode != 0:
        subprocess.run(base + ["-c:v", "libx264", "-preset", "fast", "-crf", "18", str(dst)], check=True)


def _windows(clip_id: str, video: Path, out: Path, spec: str, fleet) -> list[Path]:
    cd = out / clip_id
    marker = cd / "_windows.json"
    if marker.is_file():
        plan = json.loads(marker.read_text())
        if plan["spec"] != spec:
            raise RuntimeError(f"{cd} was cut at spec {plan['spec']}, not {spec}; use a new --out")
        return [out / name for name in plan["windows"]]
    cd.mkdir(parents=True, exist_ok=True)
    _stage_video(video, cd / "video.mp4", fleet)
    dirs = fleet._split_spec_windows(cd) if fleet else [cd]
    marker.write_text(json.dumps({"spec": spec, "source": str(video), "windows": [d.name for d in dirs]}))
    return dirs


def _pose_window(wd: Path, gpu: str, spec: str, commit: str, keep_work: bool) -> None:
    import numpy as np

    from solar_wm_data.ingest import _probe_video
    from solar_wm_data.manifest import ClipRecord
    from solar_wm_data.pose.vipe_cli import annotate_pose_vipe_cli

    video = (wd / "video.mp4").resolve()
    rec = ClipRecord(clip_id=wd.name, source="web", video_path=str(video), mode="default")
    for key, value in _probe_video(video).items():
        if value:
            setattr(rec, key, value)
    work = wd / "_work"
    annotate_pose_vipe_cli(rec, work, {"dry_run": False, "vipe": {"gpu": gpu}})

    poses = np.load(rec.pose_path)
    intr = np.load(rec.intrinsics_path)
    if not (len(poses) == len(intr) == rec.num_frames):
        raise RuntimeError(
            f"frame misalignment: video {rec.num_frames}, poses {len(poses)}, intrinsics {len(intr)}")
    if not (np.isfinite(poses).all() and np.isfinite(intr).all()):
        raise RuntimeError("non-finite poses or intrinsics")
    np.save(wd / "poses.npy", poses)
    np.save(wd / "intrinsics.npy", intr)
    rec.pose_path = str(wd / "poses.npy")
    rec.intrinsics_path = str(wd / "intrinsics.npy")
    rec.pose_units = "metric"
    rec.extra.update(spec=spec, vipe_commit=commit)
    (wd / "meta.json").write_text(json.dumps(rec.to_dict(), ensure_ascii=False), encoding="utf-8")
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)


def _run_worker(args, gpu: str, shard: int, num_shards: int) -> int:
    global _TAG
    _TAG = f" [gpu{gpu}]"
    commit = _check_setup()
    fleet = _load_fleet(args.spec) if args.spec != "none" else None
    items = collect_videos(args.inputs)[shard::num_shards]
    _log(f"{len(items)} video(s), spec={args.spec}, vipe={commit}")
    done = skipped = failed = 0
    for clip_id, video in items:
        try:
            windows = _windows(clip_id, video, args.out, args.spec, fleet)
        except Exception as exc:  # one bad video must not stop the shard
            failed += 1
            _log(f"FAIL {clip_id}: cutting windows: {type(exc).__name__}: {exc}")
            continue
        if not windows:
            _log(f"SKIP {clip_id}: no complete {args.spec} window (too short, or fps below spec)")
            continue
        for wd in windows:
            if (wd / "meta.json").is_file() and (wd / "poses.npy").is_file() and not args.force:
                skipped += 1
                continue
            t0 = time.time()
            _log(f"pose {wd.name} <- {video}")
            try:
                _pose_window(wd, gpu, args.spec, commit, args.keep_work)
            except Exception as exc:
                failed += 1
                _log(f"  FAIL {wd.name}: {type(exc).__name__}: {str(exc)[:300]}")
                continue
            done += 1
            _log(f"  -> {wd} ({time.time() - t0:.0f}s)")
    _log(f"POSE_VIDEOS_WORKER_DONE posed={done} skipped={skipped} failed={failed}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="video files and/or directories")
    ap.add_argument("--out", type=Path, required=True, help="output root for per-window clip dirs")
    ap.add_argument("--spec", default=os.environ.get("SOLAR_WM_SPEC", "5s"),
                    help="window spec (5s, 60s, 81f, 160f, 960f, or <frames>@<fps>); "
                         "'none' poses each video whole (default: $SOLAR_WM_SPEC or 5s)")
    ap.add_argument("--gpus", default="0", help="comma-separated physical GPU ids, one worker each")
    ap.add_argument("--force", action="store_true", help="re-pose windows that already have meta.json")
    ap.add_argument("--keep-work", action="store_true", help="keep the fused depth and raw VIPE output")
    ap.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    os.environ.setdefault("SOLAR_WM_ROOT", str(_REPO))
    os.environ.setdefault("SOLAR_WM_WEIGHTS", str(_REPO / "weights"))
    if args.spec != "none":
        from solar_wm_data import spec as spec_mod
        spec_mod.parse_spec(args.spec)
    args.out.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    if args.shard is not None:
        return _run_worker(args, gpus[0], args.shard, args.num_shards)
    if len(gpus) == 1:
        return _run_worker(args, gpus[0], 0, 1)

    _check_setup()
    passthrough = [str(p) for p in args.inputs] + ["--out", str(args.out), "--spec", args.spec]
    passthrough += ["--force"] * args.force + ["--keep-work"] * args.keep_work
    procs = [
        subprocess.Popen([sys.executable, __file__, *passthrough, "--gpus", gpu,
                          "--shard", str(i), "--num-shards", str(len(gpus))])
        for i, gpu in enumerate(gpus)
    ]
    codes = [p.wait() for p in procs]
    finished = sum(1 for _ in args.out.glob("*/meta.json"))
    _log(f"POSE_VIDEOS_DONE {finished} window(s) with poses under {args.out}; "
         f"worker exit codes {codes}")
    return 1 if any(codes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
