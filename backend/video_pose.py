"""
RGB video -> Kinect-style 75-column skeleton via MediaPipe Pose.

The joint mapping and coordinate convention here are copied verbatim from the
project's established extractor (asd_backend `services/pose_extraction.py` /
`train/make_keypoints_csv.py`), so a video processed here yields the SAME
skeleton as the `*_keypoints.csv` files the rest of the pipeline was built on.

Coordinate convention (MediaPipe world landmarks, metres, hip-centred):
    x' = -x,  y' = -y,  z' = z
Joint definitions (MediaPipe 33-landmark indices):
    Head=0, Shoulders=11/12, Elbows=13/14, Wrists=15/16, Index=19/20,
    Thumb=21/22, Hips=23/24, Knees=25/26, Ankles=27/28, FootIndex=31/32
    SpineShoulder = Neck = mid(11,12);  SpineBase = mid(23,24)
    SpineMid (Midspain) = mid(SpineShoulder, SpineBase)
    HandLeft=Wrist(15), HandRight=Wrist(16), HandTip=Index(19/20)

Output column layout matches `config.USER_JOINT_COL` exactly, so the resulting
(TARGET_FRAMES, 75) array is byte-compatible with the CSV path, the handcrafted
feature extractor, and the NTU re-order used to feed MS-G3D.

⚠️  Real phone-video accuracy of the v40 model is unmeasured (see the model's
deployment note); treat video predictions as indicative, not diagnostic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import ntu_features as F

# Cap how many frames we run through MediaPipe (latency bound); the downstream
# resample to TARGET_FRAMES makes denser sampling unnecessary.
MAX_PROC_FRAMES = 600


def _kinect_joints_from_world(lm) -> dict:
    """Map MediaPipe world landmarks (list) -> {joint_name: (x,y,z)} in Kinect axes."""
    def g(i):
        return (lm[i].x * -1.0, lm[i].y * -1.0, lm[i].z)

    def gm(a, b):
        return (((lm[a].x + lm[b].x) / 2.0) * -1.0,
                ((lm[a].y + lm[b].y) / 2.0) * -1.0,
                (lm[a].z + lm[b].z) / 2.0)

    ss = gm(11, 12)   # SpineShoulder
    sb = gm(23, 24)   # SpineBase
    mid = ((ss[0] + sb[0]) / 2.0, (ss[1] + sb[1]) / 2.0, (ss[2] + sb[2]) / 2.0)
    return {
        'SpineMid': mid, 'AnkleLeft': g(27), 'AnkleRight': g(28),
        'ElbowLeft': g(13), 'ElbowRight': g(14), 'FootLeft': g(31),
        'FootRight': g(32), 'HandLeft': g(15), 'HandRight': g(16),
        'HandTipLeft': g(19), 'HandTipRight': g(20), 'Head': g(0),
        'HipLeft': g(23), 'HipRight': g(24), 'KneeLeft': g(25),
        'KneeRight': g(26), 'Neck': gm(11, 12), 'ShoulderLeft': g(11),
        'ShoulderRight': g(12), 'SpineBase': sb, 'SpineShoulder': ss,
        'ThumbLeft': g(21), 'ThumbRight': g(22), 'WristLeft': g(15),
        'WristRight': g(16),
    }


def _frame_to_row(lm) -> np.ndarray:
    """One MediaPipe frame -> flat 75-vector in USER_JOINT_COL order."""
    out = np.full(75, np.nan, dtype=np.float32)
    for name, (x, y, z) in _kinect_joints_from_world(lm).items():
        col = C.USER_JOINT_COL[name]
        out[col:col + 3] = (x, y, z)
    return out


def _collect_frames(path: str, model_complexity: int):
    """Run MediaPipe Pose over a video -> (rows list of 75-vectors, fps, meta)."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not fps or fps != fps:  # NaN guard
        fps = 30.0
    if total and total > MAX_PROC_FRAMES:
        proc_idx = set(np.linspace(0, total - 1, MAX_PROC_FRAMES).astype(int).tolist())
        fps = fps * MAX_PROC_FRAMES / total   # effective fps after subsampling
    else:
        proc_idx = None

    rows: list[np.ndarray] = []
    n_detected = 0
    frame_i = -1
    pose = mp.solutions.pose.Pose(static_image_mode=False,
                                  model_complexity=model_complexity,
                                  enable_segmentation=False,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1
            if proc_idx is not None and frame_i not in proc_idx:
                continue
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_world_landmarks:
                rows.append(_frame_to_row(res.pose_world_landmarks.landmark))
                n_detected += 1
            else:
                rows.append(np.full(75, np.nan, dtype=np.float32))
    finally:
        pose.close()
        cap.release()

    if not rows or n_detected == 0:
        raise ValueError("No pose detected in any frame of the video.")
    meta = {
        "total_frames": total,
        "processed_frames": len(rows),
        "detected_frames": n_detected,
        "detection_rate": round(n_detected / max(1, len(rows)), 3),
        "fps": round(float(fps), 2),
        "model_complexity": model_complexity,
    }
    return rows, float(fps), meta


def extract_skeleton_from_video(path: str, model_complexity: int = 1):
    """Video -> (raw (TARGET_FRAMES, 75), meta). Single 150-frame clip."""
    rows, _fps, meta = _collect_frames(path, model_complexity)
    raw, n_clipped = F.finalize_coords(pd.DataFrame(np.vstack(rows)))
    meta["n_clipped_coords"] = n_clipped
    return raw, meta


def extract_skeleton_full_from_video(path: str, model_complexity: int = 1):
    """Video -> (cleaned full-length (T,75), fps, meta) for sliding-window reports."""
    rows, fps, meta = _collect_frames(path, model_complexity)
    raw_full = F._clean_coords(pd.DataFrame(np.vstack(rows)))
    return raw_full, fps, meta
