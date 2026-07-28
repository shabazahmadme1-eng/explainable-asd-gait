"""
Skeleton parsing + handcrafted (HC) biomechanical feature extraction.

The functions here are copied verbatim from the v40 training script so that the
534-dim HC vector and the (C, T, V, M) skeleton tensor are byte-for-byte
compatible with what the deployed scalers / ensembles / EBM expect.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from . import config as C

J_H = C.USER_JOINT_COL


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_skeleton_file(filepath: str):
    """Read a Kinect skeleton CSV/XLSX -> (raw (TARGET_FRAMES, 75) array, n_clipped).

    Returns (None, 0) if the file cannot be parsed or has too few columns.
    Identical logic to training: NaN-fill zero joints, interpolate, rolling
    median smooth, resample to TARGET_FRAMES, clip to +/-COORD_CLIP.
    """
    try:
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath, engine='openpyxl')
        else:
            df = pd.read_csv(filepath)
    except Exception:
        return None, 0

    if df.shape[1] < 76:
        return None, 0

    coords_df = df.iloc[:, 1:76].apply(pd.to_numeric, errors='coerce')
    for j in range(25):
        cols = [j * 3, j * 3 + 1, j * 3 + 2]
        all_zero = (coords_df.iloc[:, cols] == 0.0).all(axis=1)
        if all_zero.any():
            coords_df.iloc[all_zero.values, cols] = np.nan

    return finalize_coords(coords_df)


def fps_from_timestamps(ts_series) -> float:
    """Derive fps from an 'H:M:S:MS)' timestamp column; default 30 if unparseable."""
    import re as _re

    def to_sec(s):
        a = _re.split("[:.]", str(s).replace(")", ""))
        try:
            return int(a[0]) * 3600 + int(a[1]) * 60 + int(a[2]) + int(a[3]) / 1000.0
        except Exception:
            return None

    t = [x for x in (to_sec(v) for v in ts_series) if x is not None]
    if len(t) < 2:
        return 30.0
    d = np.diff(t)
    d = d[d > 0]
    return float(1.0 / np.median(d)) if len(d) else 30.0


def _clean_coords(coords_df: pd.DataFrame) -> np.ndarray:
    """Interpolate gaps -> rolling-median smooth -> clip. No frame resampling."""
    coords_df = (coords_df.interpolate(method='linear', limit_direction='both')
                          .rolling(window=3, min_periods=1, center=True).median())
    raw = np.nan_to_num(coords_df.values, nan=0.0)
    return np.clip(raw, -C.COORD_CLIP, C.COORD_CLIP)


def parse_skeleton_full(filepath: str):
    """Read a keypoint CSV/XLSX -> (cleaned full-length (T,75) array, fps).

    Like parse_skeleton_file but keeps the original frame count (no 150 resample),
    so callers can slide time windows over the walk. Returns (None, 30.0) on failure.
    """
    try:
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath, engine='openpyxl')
        else:
            df = pd.read_csv(filepath)
    except Exception:
        return None, 30.0
    if df.shape[1] < 76:
        return None, 30.0
    fps = fps_from_timestamps(df.iloc[:, 0])
    coords_df = df.iloc[:, 1:76].apply(pd.to_numeric, errors='coerce')
    for j in range(25):
        cols = [j * 3, j * 3 + 1, j * 3 + 2]
        all_zero = (coords_df.iloc[:, cols] == 0.0).all(axis=1)
        if all_zero.any():
            coords_df.iloc[all_zero.values, cols] = np.nan
    if len(coords_df) == 0:
        return None, fps
    return _clean_coords(coords_df), fps


def load_feature_windows(filepath: str):
    """If `filepath` is a PRE-ENGINEERED 448-feature-per-window export (columns
    like ``Window_Timestamp, Frame_0_Left_Arm_Angle, ...`` = the spatiotemporal
    448-vector already computed, one row per window), return
    ``(X_full (n,448), mids_seconds)``. Otherwise return ``None`` and the caller
    should treat the file as a raw-coordinate CSV.

    Such a file has no raw skeleton, so only the handcrafted stream can score it
    (the MS-G3D embedding needs joint coordinates).
    """
    try:
        df = pd.read_excel(filepath, engine='openpyxl') if filepath.endswith('.xlsx') \
            else pd.read_csv(filepath)
    except Exception:
        return None
    low = [str(c).lower() for c in df.columns]
    is_feat = (any('window_timestamp' in c for c in low)
               or any(c.startswith('frame_0_') or c.startswith('frame0_') for c in low))
    if not is_feat:
        return None
    ts_idx = next((i for i, c in enumerate(low) if 'timestamp' in c), None)
    feat_cols = [c for i, c in enumerate(df.columns) if i != ts_idx]
    X = df[feat_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    X = np.nan_to_num(X)
    if X.shape[1] < 448:
        return None
    X = X[:, :448].astype(np.float32)
    if ts_idx is not None:
        try:
            mids = pd.to_numeric(df.iloc[:, ts_idx], errors='coerce').fillna(
                0.0).to_numpy(dtype=float).tolist()
        except Exception:
            mids = [i * 0.33 + 0.5 for i in range(len(X))]
    else:
        mids = [i * 0.33 + 0.5 for i in range(len(X))]
    return X, mids


def resample_clip(raw: np.ndarray, n: int = None) -> np.ndarray:
    """Resample/pad a (W,75) window to exactly n frames (default TARGET_FRAMES)."""
    n = n or C.TARGET_FRAMES
    W = len(raw)
    if W == n:
        return raw.astype(np.float32)
    if W < n:
        pad = np.repeat(raw[-1:], n - W, axis=0)
        return np.vstack([raw, pad]).astype(np.float32)
    idx = np.linspace(0, W - 1, n).astype(int)
    return raw[idx].astype(np.float32)


def finalize_coords(coords_df: pd.DataFrame):
    """Shared tail of the skeleton pipeline (CSV *and* video both use this):
    interpolate gaps -> rolling-median smooth -> pad/resample to TARGET_FRAMES
    -> clip to +/-COORD_CLIP. Input is a (T, 75) frame DataFrame that may hold
    NaNs for missing joints/frames. Returns (raw (TARGET_FRAMES,75), n_clipped).
    """
    coords_df = (coords_df.interpolate(method='linear', limit_direction='both')
                          .rolling(window=3, min_periods=1, center=True).median())
    if len(coords_df) == 0:
        return None, 0

    if len(coords_df) < C.TARGET_FRAMES:
        pad = pd.concat([coords_df.iloc[-1:]] * (C.TARGET_FRAMES - len(coords_df)))
        coords_df = pd.concat([coords_df, pad], ignore_index=True)
    elif len(coords_df) > C.TARGET_FRAMES:
        idx = np.linspace(0, len(coords_df) - 1, C.TARGET_FRAMES).astype(int)
        coords_df = coords_df.iloc[idx].reset_index(drop=True)

    raw = np.nan_to_num(coords_df.values, nan=0.0)
    n_clipped = int(np.sum(np.abs(raw) > C.COORD_CLIP))
    return np.clip(raw, -C.COORD_CLIP, C.COORD_CLIP), n_clipped


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def center_skeleton(data: np.ndarray) -> np.ndarray:
    """data shape (C, T, V, M) -> centered on CENTER_JOINT."""
    return data - data[:, :, C.CENTER_JOINT:C.CENTER_JOINT + 1, :]


def compute_bone_data(skeleton_data: np.ndarray) -> np.ndarray:
    """skeleton_data shape (C, T, V, M) -> bone vectors via parent differencing."""
    parent_indices = np.array(C.NTU_PARENTS)
    return skeleton_data - skeleton_data[:, :, parent_indices, :]


def raw_to_ntu_tensor(raw: np.ndarray) -> np.ndarray:
    """raw (TARGET_FRAMES, 75) -> centered NTU tensor (3, TARGET_FRAMES, 25, 1)."""
    ntu = (raw[:, C.NTU_REORDER_COLS]
           .reshape(C.TARGET_FRAMES, 25, 3)
           .transpose(2, 0, 1)[:, :, :, np.newaxis]
           .astype(np.float32))
    return center_skeleton(ntu)


def _get_vec_h(f, j):
    return f[J_H[j]:J_H[j] + 3]


def _calc_angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)))


def _calc_dist(p1, p2):
    return np.linalg.norm(p1 - p2)


# --------------------------------------------------------------------------- #
# Handcrafted features (534-dim) — verbatim from v37u/v40
# --------------------------------------------------------------------------- #
def extract_handcrafted_features(raw_frames: np.ndarray) -> np.ndarray:
    T = len(raw_frames)
    frames = (raw_frames[np.linspace(0, T - 1, 30).astype(int)]
              if T != 30 else raw_frames)
    bio = np.zeros((30, 12))
    g = _get_vec_h
    for i, f in enumerate(frames):
        tl = max(_calc_dist(g(f, 'SpineShoulder'), g(f, 'SpineBase')), 0.01)
        bio[i] = [
            _calc_angle(g(f, 'ShoulderLeft'),  g(f, 'ElbowLeft'),  g(f, 'WristLeft')),
            _calc_angle(g(f, 'ShoulderRight'), g(f, 'ElbowRight'), g(f, 'WristRight')),
            _calc_angle(g(f, 'HipLeft'),       g(f, 'KneeLeft'),   g(f, 'AnkleLeft')),
            _calc_angle(g(f, 'HipRight'),      g(f, 'KneeRight'),  g(f, 'AnkleRight')),
            _calc_angle(g(f, 'SpineShoulder'), g(f, 'ShoulderLeft'),  g(f, 'ElbowLeft')),
            _calc_angle(g(f, 'SpineShoulder'), g(f, 'ShoulderRight'), g(f, 'ElbowRight')),
            _calc_angle(g(f, 'SpineBase'),     g(f, 'HipLeft'),    g(f, 'KneeLeft')),
            _calc_angle(g(f, 'SpineBase'),     g(f, 'HipRight'),   g(f, 'KneeRight')),
            _calc_dist(g(f, 'WristLeft'),  g(f, 'WristRight')) / tl,
            _calc_dist(g(f, 'AnkleLeft'),  g(f, 'AnkleRight')) / tl,
            _calc_dist(g(f, 'Head'), g(f, 'WristLeft'))  / tl,
            _calc_dist(g(f, 'Head'), g(f, 'WristRight')) / tl,
        ]
    vel = np.diff(bio, axis=0)
    acc = np.diff(vel, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau_time = np.zeros(12)
        for c in range(12):
            v, _ = kendalltau(np.arange(30), bio[:, c])
            tau_time[c] = 0.0 if np.isnan(v) else v
        tau_sym = np.nan_to_num(np.array([
            kendalltau(bio[:, 0], bio[:, 1])[0],
            kendalltau(bio[:, 2], bio[:, 3])[0],
            kendalltau(bio[:, 4], bio[:, 5])[0],
            kendalltau(bio[:, 6], bio[:, 7])[0],
        ]), nan=0.0)
        orig = np.concatenate([
            bio.flatten(),
            np.nanmean(bio, axis=0), np.nanstd(bio, axis=0),
            np.nanmean(np.abs(vel), axis=0), np.nanmax(np.abs(vel), axis=0),
            np.nanmean(np.abs(acc), axis=0), np.nanmax(np.abs(acc), axis=0),
            tau_time, tau_sym,
        ])

    ntu_raw = frames[:, C.NTU_REORDER_COLS].reshape(30, 25, 3)
    ntu_c = ntu_raw - ntu_raw[:, C.CENTER_JOINT, :][:, np.newaxis, :]
    js_dist = np.linalg.norm(ntu_c, axis=2)
    spine_unit = (ntu_c[:, 1, :] /
                  np.maximum(np.linalg.norm(ntu_c[:, 1, :], axis=1, keepdims=True),
                             0.01))[:, np.newaxis, :]
    js_ang = np.degrees(np.arccos(
        np.clip(np.sum(ntu_c * spine_unit, axis=2) / np.maximum(js_dist, 0.01),
                -1.0, 1.0)))
    j_mot = np.linalg.norm(np.diff(ntu_c, axis=0), axis=2)

    a_asym = [np.abs(np.mean(js_ang[:, l]) - np.mean(js_ang[:, r]))
              for l, r in C.LR_JOINT_PAIRS]
    d_asym = [np.abs(np.mean(js_dist[:, l]) - np.mean(js_dist[:, r]))
              for l, r in C.LR_JOINT_PAIRS]
    m_asym = [np.abs(np.mean(j_mot[:, l]) - np.mean(j_mot[:, r]))
              for l, r in C.LR_JOINT_PAIRS]
    L_idx = [l for l, r in C.LR_JOINT_PAIRS]
    R_idx = [r for l, r in C.LR_JOINT_PAIRS]
    fa_ang = np.abs(js_ang[:, L_idx] - js_ang[:, R_idx]).mean(axis=1)
    fa_dist = np.abs(js_dist[:, L_idx] - js_dist[:, R_idx]).mean(axis=1)

    return np.nan_to_num(np.concatenate([
        orig, a_asym, d_asym, m_asym,
        np.mean(js_ang, axis=0), np.std(js_ang, axis=0),
        [np.mean(fa_ang), np.std(fa_ang), np.max(fa_ang),
         np.mean(fa_dist), np.std(fa_dist), np.max(fa_dist)],
    ]), nan=0.0)
