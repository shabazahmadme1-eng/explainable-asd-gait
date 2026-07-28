"""
Build a MediaPipe skeleton reference for domain alignment.

Pools per-channel (25 joints x xyz) statistics of the *centered NTU skeleton* and
its *bone vectors* over a corpus of MediaPipe-derived clips, so the inference-time
aligner can map MediaPipe skeletons into the Kinect distribution the embedding net
expects. Mirrors the spirit of asd_backend's build_mp_reference.py, but in the
raw-skeleton space the MS-G3D embedding stream is sensitive to.

Use a representative corpus (ideally your TYPICAL videos / their keypoint CSVs —
the population baseline), not just a couple of clips. More clips => more stable
reference => better alignment.

Usage:
  # from keypoint CSVs (H:M:S:MS) + 75 coords):
  python -m scripts.build_mp_reference --csv "C:/path/to/keypoint_csvs/*.csv"
  # from videos (runs MediaPipe extraction first):
  python -m scripts.build_mp_reference --video "C:/path/to/typical/**/*.mp4"
  # output (default): v40_artifacts/mp_skeleton_reference.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config as C
from backend import ntu_features as F


def _accumulate(raw, jacc, bacc):
    """raw (TARGET_FRAMES,75) -> append (T,25,3) joint & bone frames to accumulators."""
    ntu = F.raw_to_ntu_tensor(raw)                       # (3,T,25,1) centered
    bone = F.compute_bone_data(ntu)                      # bones from RAW (unaligned) joints
    jacc.append(ntu[:, :, :, 0].transpose(1, 2, 0))      # (T,25,3)
    bacc.append(bone[:, :, :, 0].transpose(1, 2, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="glob of keypoint CSV/XLSX files")
    ap.add_argument("--video", help="glob of video files (MediaPipe extracted first)")
    ap.add_argument("--out", default=C.MP_REFERENCE_PATH, help="output .npz path")
    args = ap.parse_args()

    jacc, bacc = [], []
    n_clips = 0

    if args.csv:
        files = sorted(glob.glob(args.csv, recursive=True))
        print(f"Building reference from {len(files)} keypoint CSV(s)...")
        for fp in files:
            raw, _ = F.parse_skeleton_file(fp)
            if raw is None:
                print(f"  skip (parse failed): {fp}")
                continue
            _accumulate(raw, jacc, bacc)
            n_clips += 1
            print(f"  [{n_clips}] {os.path.basename(fp)}")

    if args.video:
        from backend.video_pose import extract_skeleton_from_video
        vids = sorted(glob.glob(args.video, recursive=True))
        print(f"Building reference from {len(vids)} video(s)...")
        for vp in vids:
            try:
                raw, meta = extract_skeleton_from_video(vp)
            except Exception as exc:
                print(f"  skip ({exc}): {vp}")
                continue
            _accumulate(raw, jacc, bacc)
            n_clips += 1
            print(f"  [{n_clips}] {os.path.basename(vp)}  det_rate={meta['detection_rate']}")

    if n_clips == 0:
        raise SystemExit("No clips processed. Pass --csv and/or --video globs.")

    J = np.concatenate(jacc, axis=0)   # (sumT,25,3)
    B = np.concatenate(bacc, axis=0)
    ref = dict(
        joint_mu=J.mean(0), joint_sd=J.std(0) + 1e-6,
        bone_mu=B.mean(0), bone_sd=B.std(0) + 1e-6,
        n_frames=len(J), n_clips=n_clips,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **ref)
    print(f"\nSaved {args.out}")
    print(f"  clips={n_clips}  frames={len(J)}")
    if n_clips < 20:
        print("  WARNING: small corpus — reference may be unstable. Aim for >=20 clips "
              "(ideally your typical-population set) for production use.")


if __name__ == "__main__":
    main()
