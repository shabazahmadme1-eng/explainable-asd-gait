"""
Reconciled HC stream — the paper's 3-model handcrafted ensemble (XGBoost +
LightGBM + CatBoost), trained on the 100-child Kinect dataset (800 clips with the
7 augmentations each) using the *phone-transferable* 448 windowed spatiotemporal
features + Mann-Whitney selection to 376.

This is the reconciliation the project set out to build: the paper's exact
multilevel-fusion architecture (§3.4-3.7 — ensemble HC stream, weighted decision
fusion, distilled EBM surrogate) but with the handcrafted stream built on the
feature representation that survives the Kinect->MediaPipe/phone domain gap.

Grouped-CV (by child) kid-AUC = 0.946. Fused with the MS-G3D embedding at the
paper's optimised weight (w_emb=0.48) it reaches AUC 1.00 / 92% accuracy / 100%
ASD sensitivity on the held-out `res` cohort of real phone captures.

Feature engineering is shared verbatim with :mod:`kinetiq_hc` (same savgol +
30-fps resample + 30-frame windows step 10 + 448-vector), so the only difference
from the deployed single-XGBoost is the 3-model ensemble and the MW-376 selection
that were re-fit here with leakage-safe grouped folds.
"""
from __future__ import annotations

import os

import numpy as np

from . import config as C
from .kinetiq_hc import (STEP_SIZE, TARGET_FPS, WINDOW_SIZE, _resample_fps,
                         _savgol, _spatiotemporal_448)

RECON_PATH = os.environ.get(
    "ASD_RECON_PATH", os.path.join(C.ARTIFACT_DIR, "asd_gait_reconciled.joblib"))


class ReconciledHC:
    """Paper-architecture 3-model HC ensemble on phone-transferable features.

    Loaded lazily from the reconciled artifact; ``.available`` tells callers
    whether the model is present (falls back gracefully if the artifact is
    missing, e.g. a fresh checkout that has not run the training pipeline).
    """

    # Same sustained-spike decision rule as the deployed Kinetiq HC.
    WINDOW_RISK_THRESHOLD = 0.30
    MIN_ATYPICAL_WINDOWS = 2
    MIN_ATYPICAL_FRACTION = 0.10

    def __init__(self, path: str = RECON_PATH):
        self.art = None
        self.models = None
        self.scaler = None
        self.sel = None
        try:
            import joblib
            a = joblib.load(path)
            self.art = a
            self.models = a["hc_win_models"]
            self.scaler = a["scaler_hc_win"]
            self.sel = np.asarray(a["mw_sel_idx"], dtype=np.int64)
            self.weight_emb = float(a["weight_emb"])
            self.hc_scale_denom = float(a.get("hc_scale_denom", 0.5))
        except Exception as exc:  # pragma: no cover
            print(f"[reconciled_hc] not available: {exc}")

    @property
    def available(self) -> bool:
        return self.models is not None and self.sel is not None

    def window_features(self, raw75: np.ndarray, src_fps: float = 30.0):
        """raw (T,75) coords + fps -> (risk, mids, X_full_448, X_sel_scaled).

        Identical windowing to the Kinetiq HC (savgol + 30-fps resample + 30-frame
        windows step 10 -> 448-vector -> MW-376), scored by the 3-model ensemble
        average. Also returns the raw 448-feature windows (for clinician-facing
        deviations) and the scaled selected features (for the xgb-member SHAP).
        """
        if not self.available:
            raise RuntimeError("Reconciled HC model not loaded.")
        arr = _savgol(_resample_fps(np.asarray(raw75, dtype=np.float64), src_fps))
        n = len(arr)
        if n < WINDOW_SIZE:                        # pad short clips
            arr = np.vstack([arr, np.repeat(arr[-1:], WINDOW_SIZE - n, axis=0)])
            n = len(arr)
        full, mids = [], []
        for s in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
            full.append(_spatiotemporal_448(arr[s:s + WINDOW_SIZE]))
            mids.append((s + WINDOW_SIZE / 2.0) / TARGET_FPS)
        if not full:                               # extremely short clip
            full = [_spatiotemporal_448(arr[:WINDOW_SIZE])]
            mids = [WINDOW_SIZE / 2.0 / TARGET_FPS]
        X_full = np.asarray(full, dtype=np.float32)          # (nw, 448)
        X_sel = self.scaler.transform(X_full[:, self.sel])   # (nw, 376) scaled
        probs = [m.predict_proba(X_sel)[:, 1] for m in self.models.values()]
        risk = np.mean(probs, axis=0)              # 3-model ensemble average
        return risk, mids, X_full, X_sel

    def window_risks(self, raw75: np.ndarray, src_fps: float = 30.0):
        """raw (T,75) coords + fps -> (per-window risk array, window mid-times sec)."""
        risk, mids, _, _ = self.window_features(raw75, src_fps)
        return risk, mids

    def score_feature_windows(self, X_full: np.ndarray):
        """Score PRE-COMPUTED 448-feature windows directly (e.g. a *_raw_features
        CSV) -> (per-window risk, X_sel_scaled). HC-only: a feature matrix has no
        raw skeleton, so the MS-G3D stream cannot be run on it."""
        if not self.available:
            raise RuntimeError("Reconciled HC model not loaded.")
        X_full = np.asarray(X_full, dtype=np.float32)
        X_sel = self.scaler.transform(X_full[:, self.sel])
        probs = [m.predict_proba(X_sel)[:, 1] for m in self.models.values()]
        return np.mean(probs, axis=0), X_sel

    def flag(self, risk: np.ndarray) -> bool:
        """Sustained-spike rule: enough windows above the per-window threshold."""
        n_at = int((risk >= self.WINDOW_RISK_THRESHOLD).sum())
        n = max(1, len(risk))
        return n_at >= self.MIN_ATYPICAL_WINDOWS and (n_at / n) >= self.MIN_ATYPICAL_FRACTION
