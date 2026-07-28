"""
Clinician-facing report detail for the `/api/analyze` endpoint — ported from the
Kinetiq deployment (`asd_backend`) so the existing React frontend renders
unchanged, but driven by the **reconciled** model:

  * regional attribution + top drivers via SHAP on the ensemble's XGBoost member
    (the frontend expects the single-model SHAP shape; the xgb member is a
    faithful, cheap proxy for the 3-model ensemble's feature attribution);
  * per-metric deviations + bilateral symmetry vs the typical-population baseline
    (model-agnostic — pure feature-vs-reference z-scores);
  * a clinical summary/narrative rewritten to reflect the reconciled *fused*
    decision (HC ensemble + MS-G3D embedding), not the raw sustained-spike rule.

Feature layout of the 448-vector (see kinetiq_hc._spatiotemporal_448):
  [0:360]   flattened bio    30 frames x 12 metrics   (metric = idx % 12)
  [360:444] 7 stat blocks    means/stds/mean_vel/max_vel/mean_acc/max_acc/tau_time
  [444:448] tau_sym          4 bilateral-symmetry features
"""
from __future__ import annotations

import os

import numpy as np

from . import config as C

REF_DIR = os.path.join(C.ARTIFACT_DIR, "clinical_refs")

REGION_NAMES = ["Upper Body", "Lower Body", "Symmetry"]
REGION_DESCRIPTIONS = {
    "Upper Body": "Arm/shoulder kinematics - elbow & shoulder angles, wrist excursion and inter-wrist distance. Sensitive to repetitive arm movements (stereotypies) and reduced/asymmetric arm swing.",
    "Lower Body": "Gait & lower-limb kinematics - hip & knee flexion, ankle base-of-support. Sensitive to atypical gait, toe-walking and posture.",
    "Symmetry": "Left-right coordination - Kendall's-tau correlation between paired limbs. Sensitive to breakdowns in bilateral motor symmetry.",
}
DISCLAIMER = ("This is an automated screening aid, NOT a diagnosis. Results quantify motor "
              "kinematics only and must be interpreted by a qualified clinician alongside history, "
              "observation and validated diagnostic instruments.")

BIO_METRICS = [
    "Left Arm Angle", "Right Arm Angle", "Left Leg Angle", "Right Leg Angle",
    "Left Shoulder Posture", "Right Shoulder Posture", "Left Hip Posture", "Right Hip Posture",
    "Wrist-to-Wrist Distance", "Ankle-to-Ankle Distance", "Head-to-Left-Wrist", "Head-to-Right-Wrist",
]
SYM_METRICS = ["Arms", "Legs", "Shoulders", "Hips"]
_ASPECTS = ["average", "variability", "velocity", "peak velocity",
            "acceleration (jerk)", "peak acceleration", "temporal trend"]
_METRIC_REGION = {
    0: "Upper Body", 1: "Upper Body", 2: "Lower Body", 3: "Lower Body",
    4: "Upper Body", 5: "Upper Body", 6: "Lower Body", 7: "Lower Body",
    8: "Upper Body", 9: "Lower Body", 10: "Upper Body", 11: "Upper Body",
}


def _region_for_raw_index(k: int) -> str:
    if k >= 444:
        return "Symmetry"
    if k >= 360:
        return _METRIC_REGION[(k - 360) % 12]
    return _METRIC_REGION[k % 12]


def _metric_region(metric: str) -> str:
    m = metric.lower()
    if any(w in m for w in ["arm", "shoulder", "wrist", "head"]):
        return "Upper Body"
    if any(w in m for w in ["leg", "hip", "ankle"]):
        return "Lower Body"
    return "Symmetry"


def _feature_label(k: int):
    if k >= 444:
        return SYM_METRICS[k - 444] + " symmetry", "bilateral symmetry"
    if k >= 360:
        return BIO_METRICS[(k - 360) % 12], _ASPECTS[(k - 360) // 12]
    return BIO_METRICS[k % 12], "instantaneous"


def _load_ref(name):
    try:
        r = np.load(os.path.join(REF_DIR, name))
        return (r["mu"], r["sd"])
    except Exception as e:  # pragma: no cover
        print(f"[clinical] reference {name} not loaded ({e}).")
        return None


TD_REF_KINECT = _load_ref("td_reference_448.npz")
TD_REF_MP = _load_ref("mediapipe_td_reference_448.npz")


def pick_reference(source_kind: str):
    """MediaPipe-derived inputs (video, phone CSV) compare against the MediaPipe
    baseline; genuine Kinect coordinate exports use the Kinect one."""
    mp_like = source_kind in ("video", "csv-coordinates", "csv-features", "mediapipe")
    return (TD_REF_MP or TD_REF_KINECT) if mp_like else (TD_REF_KINECT or TD_REF_MP)


class ClinicalDetail:
    """Builds the frontend's rich report fields from reconciled-model outputs.

    `sel` is the reconciled model's 376 selected indices into the 448-vector, so
    the SHAP region/label mapping aligns to the ensemble's feature columns.
    """

    def __init__(self, sel: np.ndarray, xgb_member):
        self.sel = np.asarray(sel, dtype=int)
        self.xgb = xgb_member
        self.feature_regions = [_region_for_raw_index(int(k)) for k in self.sel]
        self.feature_labels = [_feature_label(int(k)) for k in self.sel]

    # ---- SHAP on the ensemble's xgb member (proxy for ensemble attribution) ----
    def _shap_pos(self, X_sel_scaled: np.ndarray):
        try:
            import xgboost as xgb
            contribs = self.xgb.get_booster().predict(
                xgb.DMatrix(X_sel_scaled), pred_contribs=True)[:, :-1]
            return np.clip(contribs, 0, None).sum(axis=0)   # push-toward-ASD only
        except Exception as e:
            print(f"[clinical] SHAP unavailable ({e}); using feature importances.")
            imp = getattr(self.xgb, "feature_importances_", np.ones(len(self.sel)))
            return np.asarray(imp, dtype=float)

    def regional_drivers(self, X_sel_scaled: np.ndarray) -> dict:
        out = {k: 0.0 for k in REGION_NAMES}
        pos = self._shap_pos(X_sel_scaled)
        for j, r in enumerate(self.feature_regions):
            out[r] += float(pos[j])
        total = sum(out.values())
        return {k: (v / total * 100 if total > 0 else 0.0) for k, v in out.items()}

    def kinematic_markers(self, X_sel_scaled: np.ndarray, top: int = 6) -> list:
        pos = self._shap_pos(X_sel_scaled)
        agg: dict = {}
        for j, (metric, aspect) in enumerate(self.feature_labels):
            agg[(metric, aspect)] = agg.get((metric, aspect), 0.0) + float(pos[j])
        total = sum(agg.values()) or 1.0
        ordered = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top]
        return [{
            "marker": f"{metric} - {aspect}", "metric": metric, "aspect": aspect,
            "region": _metric_region(metric),
            "contribution_pct": round(v / total * 100, 1),
        } for (metric, aspect), v in ordered if v > 0]

    # ---- deviations vs typical population (model-agnostic) ----
    @staticmethod
    def kinematic_findings(X_full: np.ndarray, ref) -> list:
        if ref is None or X_full is None or len(X_full) == 0:
            return []
        mu, sd = ref
        pmean = np.asarray(X_full).mean(axis=0)
        blocks = {"posture": 360, "variability": 372, "movement speed": 384}
        found = []
        for label, base in blocks.items():
            for mi in range(12):
                k = base + mi
                z = float((pmean[k] - mu[k]) / (sd[k] if sd[k] else 1.0))
                if abs(z) >= 1.5:
                    found.append({
                        "marker": BIO_METRICS[mi], "aspect": label,
                        "region": _metric_region(BIO_METRICS[mi]),
                        "patient_value": round(float(pmean[k]), 2),
                        "typical_value": round(float(mu[k]), 2),
                        "z_score": round(z, 1),
                        "direction": "above typical" if z > 0 else "below typical",
                    })
        found.sort(key=lambda d: abs(d["z_score"]), reverse=True)
        return found[:8]

    @staticmethod
    def symmetry_indices(X_full: np.ndarray, ref) -> list:
        if ref is None or X_full is None or len(X_full) == 0:
            return []
        mu, sd = ref
        pmean = np.asarray(X_full).mean(axis=0)
        out = []
        for i, pair in enumerate(SYM_METRICS):
            k = 444 + i
            z = float((pmean[k] - mu[k]) / (sd[k] if sd[k] else 1.0))
            out.append({
                "pair": pair, "coordination": round(float(pmean[k]), 2),
                "typical": round(float(mu[k]), 2),
                "status": "reduced" if z < -1.5 else "typical",
            })
        return out

    @staticmethod
    def ood_score(X_full: np.ndarray, ref) -> float | None:
        if ref is None or X_full is None or len(X_full) == 0:
            return None
        mu, sd = ref
        sd = np.where(sd == 0, 1.0, sd)
        return float(np.abs((np.asarray(X_full).mean(axis=0) - mu) / sd).mean())


EMB_CONTEXT_HIGH = 0.80   # embedding value considered "elevated" for the context note


def build_clinical_summary(is_atypical, predictions, atypical_count, min_windows,
                           regions_pct, window_threshold, emb, hc_avg,
                           mode="fusion", fused_score=None) -> dict:
    """Clinical summary. ``mode='fusion'`` (default, best measured results): the
    decision is the paper's weighted fusion of the handcrafted ensemble and the
    MS-G3D movement embedding. ``mode='hc-primary'``: the flag comes from the
    handcrafted ensemble alone and the embedding is reported as context only
    (also used automatically when ``emb is None`` — a feature CSV with no skeleton).
    """
    n = len(predictions)
    peak = float(np.max(predictions)) if n else 0.0
    mean = float(np.mean(predictions)) if n else 0.0
    frac = atypical_count / max(n, 1)
    primary = max(regions_pct, key=regions_pct.get) if regions_pct else "Upper Body"
    emb_high = emb is not None and float(emb) >= EMB_CONTEXT_HIGH
    hc_only = emb is None
    fusion = (mode == "fusion") and (emb is not None)

    if fusion:
        fs = float(fused_score if fused_score is not None else mean)
        if not is_atypical:
            severity = "Within typical range"
            narrative = (f"No atypical motor signature detected. Fused risk {fs*100:.0f}% is "
                         f"below the screening threshold (handcrafted-gait ensemble "
                         f"{hc_avg*100:.0f}%, MS-G3D movement embedding {float(emb)*100:.0f}%, "
                         f"weighted). The motor profile aligns with neurotypical development.")
        else:
            severity = ("Pronounced" if fs >= 0.75 else "Moderate" if fs >= 0.55
                        else "Mild / low-confidence flag")
            narrative = (f"Atypical motor kinematics flagged - fused risk {fs*100:.0f}% "
                         f"(handcrafted-gait ensemble {hc_avg*100:.0f}%, MS-G3D movement "
                         f"embedding {float(emb)*100:.0f}%), most strongly in "
                         f"{primary.lower()}. Peak window {peak*100:.0f}%. Clinical "
                         f"correlation recommended.")
        decision_rule = (f"Flagged: fused risk {fs*100:.0f}% >= screening threshold "
                         f"(HC ensemble {hc_avg*100:.0f}% + MS-G3D {float(emb)*100:.0f}%)."
                         if is_atypical else
                         f"Not flagged: fused risk {fs*100:.0f}% below screening threshold "
                         f"(HC ensemble {hc_avg*100:.0f}% + MS-G3D {float(emb)*100:.0f}%).")
        fused_val = round(fs, 4)
    else:
        # HC-primary (or feature-CSV HC-only): embedding is context, not a trigger
        if hc_only:
            emb_ctx = (" The MS-G3D movement model was not run (a pre-engineered feature "
                       "CSV has no raw skeleton).")
        elif emb_high:
            emb_ctx = (f" MS-G3D movement model: {float(emb)*100:.0f}% (elevated, but "
                       f"out-of-distribution on phone/RGB input - low confidence, not used "
                       f"for the screen decision).")
        else:
            emb_ctx = (f" MS-G3D movement model: {float(emb)*100:.0f}% (context only).")
        if not is_atypical:
            severity = "Within typical range"
            narrative = (f"Handcrafted gait screen is typical: {atypical_count} of {n} "
                         f"one-second windows crossed the {int(window_threshold*100)}% risk "
                         f"threshold (peak {peak*100:.0f}%), below the sustained-spike flag."
                         + emb_ctx)
            if emb_high:
                narrative += (" Note: the movement model read elevated motion - consider a "
                              "repeat recording or clinician review if other concerns exist.")
        else:
            severity = ("Pronounced" if peak >= 0.80 else "Moderate" if peak >= 0.55
                        else "Mild / low-confidence flag")
            corro = " Corroborated by the movement model." if emb_high else ""
            narrative = (f"Handcrafted gait screen flagged {atypical_count} of {n} windows "
                         f"({frac*100:.0f}%), peak {peak*100:.0f}%, most strongly in "
                         f"{primary.lower()}." + corro + emb_ctx)
        decision_rule = (f"Screen decision = handcrafted 3-model ensemble (sustained-spike: "
                         f"{atypical_count}/{n} windows >= {int(window_threshold*100)}%, flag "
                         f"needs >= {min_windows}). MS-G3D shown as context only"
                         + ("" if emb is not None else " (not run for feature CSVs)") + ".")
        fused_val = None

    return {
        "classification": "Atypical Kinematics Flagged" if is_atypical else "Typical Motor Development",
        "severity": severity,
        "peak_risk": round(peak, 4),
        "mean_risk": round(mean, 4),
        "fused_risk": fused_val,
        "embedding_risk": (round(float(emb), 4) if emb is not None else None),
        "embedding_context_only": (not fusion),
        "hc_risk": round(float(hc_avg), 4),
        "hc_only": hc_only,
        "atypical_windows": int(atypical_count),
        "total_windows": int(n),
        "atypical_fraction_pct": round(frac * 100, 1),
        "flag_threshold_windows": int(min_windows),
        "window_risk_threshold_pct": int(window_threshold * 100),
        "primary_driver": primary if is_atypical else None,
        "decision_rule": decision_rule,
        "narrative": narrative,
    }
