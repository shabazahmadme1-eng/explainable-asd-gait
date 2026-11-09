"""
Kinetiq handcrafted stream — the deployed window-level XGBoost HC model.

On phone/MediaPipe input this model is *far* stronger than the v40 Kinect-trained
HC ensemble (86% vs 43% ASD sensitivity on a 13-child labelled cohort), so it is
the preferred HC stream for RGB deployment. Self-contained copy of the
asd_backend `services/feature_engineering.py` pipeline (448 spatiotemporal
features per 30-frame window -> Mann-Whitney selection to 376 -> XGBoost),
faithful to how the reference `report_*.pdf` results were produced.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

from . import config as C

WINDOW_SIZE = 30
STEP_SIZE = 10
TARGET_FPS = 30.0
ASSET_DIR = os.path.join(C.BACKEND_DIR, "kinetiq_assets")
MODEL_PATH = os.path.join(ASSET_DIR, "xgboost_asd_spatiotemporal_optimized.json")
MW_PATH = os.path.join(ASSET_DIR, "mann_whitney_indices.npy")

J = C.USER_JOINT_COL  # Midspain=0, AnkleLeft=3, ... (same layout as the CSVs)


def _vec(f, name):
    return f[J[name]:J[name] + 3]


def _angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)))


def _dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def _resample_fps(arr: np.ndarray, src_fps: float) -> np.ndarray:
    """Linearly resample (T,75) to TARGET_FPS."""
    if arr is None or len(arr) < 2 or not src_fps or src_fps <= 0 \
            or abs(src_fps - TARGET_FPS) < 0.1:
        return arr
    n = len(arr)
    n2 = max(1, int(round(n / src_fps * TARGET_FPS)))
    si = np.linspace(0.0, n - 1, n)
    ti = np.linspace(0.0, n - 1, n2)
    return np.stack([np.interp(ti, si, arr[:, c]) for c in range(arr.shape[1])], axis=1)


def _savgol(arr: np.ndarray) -> np.ndarray:
    if len(arr) < 13:
        return arr
    from scipy.signal import savgol_filter
    out = arr.copy()
    for c in range(arr.shape[1]):
        try:
            out[:, c] = savgol_filter(arr[:, c], window_length=13, polyorder=3)
        except Exception:
            pass
    return out


def _spatiotemporal_448(window: np.ndarray) -> np.ndarray:
    """One 30-frame window (>=? frames, uses first 30) -> 448-vector (verbatim)."""
    from scipy.stats import kendalltau
    bio = np.zeros((30, 12))
    limit = min(30, len(window))
    for i in range(limit):
        f = window[i]
        torso = _dist(_vec(f, 'SpineShoulder'), _vec(f, 'SpineBase'))
        if torso < 0.01:
            torso = 1.0
        bio[i] = [
            _angle(_vec(f, 'ShoulderLeft'), _vec(f, 'ElbowLeft'), _vec(f, 'WristLeft')),
            _angle(_vec(f, 'ShoulderRight'), _vec(f, 'ElbowRight'), _vec(f, 'WristRight')),
            _angle(_vec(f, 'HipLeft'), _vec(f, 'KneeLeft'), _vec(f, 'AnkleLeft')),
            _angle(_vec(f, 'HipRight'), _vec(f, 'KneeRight'), _vec(f, 'AnkleRight')),
            _angle(_vec(f, 'SpineShoulder'), _vec(f, 'ShoulderLeft'), _vec(f, 'ElbowLeft')),
            _angle(_vec(f, 'SpineShoulder'), _vec(f, 'ShoulderRight'), _vec(f, 'ElbowRight')),
            _angle(_vec(f, 'SpineBase'), _vec(f, 'HipLeft'), _vec(f, 'KneeLeft')),
            _angle(_vec(f, 'SpineBase'), _vec(f, 'HipRight'), _vec(f, 'KneeRight')),
            _dist(_vec(f, 'WristLeft'), _vec(f, 'WristRight')) / torso,
            _dist(_vec(f, 'AnkleLeft'), _vec(f, 'AnkleRight')) / torso,
            _dist(_vec(f, 'Head'), _vec(f, 'WristLeft')) / torso,
            _dist(_vec(f, 'Head'), _vec(f, 'WristRight')) / torso,
        ]
    vel = np.diff(bio, axis=0)
    acc = np.diff(vel, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau_time = np.zeros(12)
        for c in range(12):
            v, _ = kendalltau(np.arange(30), bio[:, c])
            tau_time[c] = v if not np.isnan(v) else 0.0
        tau_sym = np.nan_to_num(np.array([
            kendalltau(bio[:, 0], bio[:, 1])[0], kendalltau(bio[:, 2], bio[:, 3])[0],
            kendalltau(bio[:, 4], bio[:, 5])[0], kendalltau(bio[:, 6], bio[:, 7])[0],
        ]), nan=0.0)
        feats = np.concatenate([
            bio.flatten(),
            np.nanmean(bio, axis=0), np.nanstd(bio, axis=0),
            np.nanmean(np.abs(vel), axis=0), np.nanmax(np.abs(vel), axis=0),
            np.nanmean(np.abs(acc), axis=0), np.nanmax(np.abs(acc), axis=0),
            tau_time, tau_sym,
        ])
    return np.nan_to_num(feats, nan=0.0)


class KinetiqHC:
    """Deployed window-XGBoost HC scorer. Loaded lazily; .available tells callers."""

    # sustained-spike decision rule (asd_backend app.py)
    WINDOW_RISK_THRESHOLD = 0.30
    MIN_ATYPICAL_WINDOWS = 2
    MIN_ATYPICAL_FRACTION = 0.10

    def __init__(self):
        self.model = None
        self.mw = None
        try:
            import xgboost as xgb
            m = xgb.XGBClassifier()
            m.load_model(MODEL_PATH)
            self.model = m
            self.mw = np.load(MW_PATH).astype(np.int64)
        except Exception as exc:  # pragma: no cover
            print(f"[kinetiq_hc] not available: {exc}")

    @property
    def available(self) -> bool:
        return self.model is not None and self.mw is not None

    def window_risks(self, raw75: np.ndarray, src_fps: float = 30.0):
        """raw (T,75) coords + fps -> (per-window risk array, window mid-times sec)."""
        if not self.available:
            raise RuntimeError("Kinetiq HC model not loaded.")
        arr = _savgol(_resample_fps(np.asarray(raw75, dtype=np.float64), src_fps))
        n = len(arr)
        if n < WINDOW_SIZE:                       # pad short clips
            arr = np.vstack([arr, np.repeat(arr[-1:], WINDOW_SIZE - n, axis=0)])
            n = len(arr)
        feats, mids = [], []
        for s in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
            feats.append(_spatiotemporal_448(arr[s:s + WINDOW_SIZE])[self.mw])
            mids.append((s + WINDOW_SIZE / 2.0) / TARGET_FPS)
        X = np.asarray(feats, dtype=np.float32)
        risk = self.model.predict_proba(X)[:, 1]
        return risk, mids

    def flag(self, risk: np.ndarray) -> bool:
        """Sustained-spike rule: enough windows above the per-window threshold."""
        n_at = int((risk >= self.WINDOW_RISK_THRESHOLD).sum())
        n = max(1, len(risk))
        return n_at >= self.MIN_ATYPICAL_WINDOWS and (n_at / n) >= self.MIN_ATYPICAL_FRACTION
