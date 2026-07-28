"""
Predictor — loads the v40 artifacts and reproduces the deployed scoring path:

  skeleton clip
    ├─ MS-G3D embedding (768) ─ PCA(150) ─ keep_idx ─ scaler_emb ─ {xgb,lgb,cat} ─┐
    │                                                                              ├─ weighted fuse ─ P(ASD)
    └─ handcrafted (534) ─ hc_final_idx(42) ─ scaler_hc ─ {xgb,lgb,cat} ──────────┘
                                   └─ ebm_sel_idx(39) ─ distilled EBM ─ explanation

Per-clip probabilities are averaged to give a subject-level score, exactly as in
`aggregate_predictions_to_subject` during training.
"""
from __future__ import annotations

import threading
from typing import Sequence

import joblib
import numpy as np
import torch

from . import config as C
from . import ntu_features as F
from .domain_align import (SkeletonAligner, kinect_reference_from_model,
                           load_mp_reference)
from . import clinical as CL
from .kinetiq_hc import KinetiqHC
from .reconciled_hc import ReconciledHC
from .model_def import load_deploy_model
from .angular_msg3d import AngularMSG3D

LABELS = {0: "TD", 1: "ASD"}


class Predictor:
    """Thread-safe singleton-ish wrapper around the v40 deployment artifacts."""

    def __init__(self, joblib_path: str = C.JOBLIB_PATH,
                 pth_path: str = C.PTH_PATH,
                 device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.art = joblib.load(joblib_path)
        self.model = load_deploy_model(pth_path, self.device)
        self._lock = threading.Lock()  # MS-G3D forward isn't guaranteed reentrant

        a = self.art
        self.pca = a["pca_deploy"]
        self.emb_keep = np.asarray(a["emb_keep_indices"], dtype=int)
        self.scaler_emb = a["scaler_emb"]
        self.emb_models = a["emb_models"]
        self.hc_final_idx = np.asarray(a["hc_final_indices_in_full"], dtype=int)
        self.scaler_hc = a["scaler_hc"]
        self.hc_models = a["hc_models"]
        self.hc_names_final = a["hc_names_final"]
        self.ebm = a["ebm"]
        self.ebm_sel_idx = np.asarray(a["ebm_sel_idx_in_dep_hc"], dtype=int)
        self.ebm_names = a["ebm_sel_names"]
        self.w_emb = float(a["weight_emb"])
        self.w_hc = float(a["weight_hc"])

        # Domain alignment (MediaPipe -> Kinect) for the embedding stream.
        self.aligner = SkeletonAligner(
            kinect_reference_from_model(self.model),
            load_mp_reference(C.MP_REFERENCE_PATH),
            shrink=C.DOMAIN_ALIGN_SHRINK)

        # Deployed Kinetiq window-XGBoost HC stream (best single-model HC on phone).
        self.kinetiq = KinetiqHC()

        # Angular MS-G3D — the deep stream that TRANSFERS to phone (paper Stream-2
        # on invariant unit-bone features). Deployed model = 0.5*HC + 0.5*angular.
        self.angular = AngularMSG3D(device=self.device)

        # Reconciled HC stream — the paper's 3-model ensemble on phone-transferable
        # features (the project's target architecture). Carries its own distilled
        # EBM over the fused decision (r=0.98 fidelity).
        self.reconciled = ReconciledHC()
        if self.reconciled.available:
            ra = self.reconciled.art
            self.recon_ebm = ra["ebm"]
            self.recon_ebm_idx = np.asarray(ra["ebm_sel_idx_in_full"], dtype=int)
            self.recon_ebm_names = ra["ebm_sel_names"]
            # Clinician-facing report detail (regional SHAP, deviations, symmetry).
            self.clinical = CL.ClinicalDetail(
                self.reconciled.sel, self.reconciled.models["xgb"])
        else:
            self.recon_ebm = None
            self.clinical = None

    # ----------------------------------------------------------------- meta #
    def info(self) -> dict:
        a = self.art
        return {
            "version": a.get("version"),
            "architecture": a.get("architecture"),
            "device": str(self.device),
            "fusion_weights": {
                "artifact_emb": self.w_emb, "artifact_hc": self.w_hc,
                "emb_weight_override": C.EMB_WEIGHT_OVERRIDE,
                "effective_emb": (C.EMB_WEIGHT_OVERRIDE
                                  if C.EMB_WEIGHT_OVERRIDE is not None else self.w_emb),
            },
            "ood_warn_z": C.OOD_WARN_Z,
            "cv_accuracy_kinect": a.get("cv_accuracy_kinect"),
            "cv_auc_kinect": a.get("cv_auc_kinect"),
            "cv_acc_mean": a.get("cv_acc_mean"),
            "cv_acc_std": a.get("cv_acc_std"),
            "cv_auc_mean": a.get("cv_auc_mean"),
            "cv_auc_std": a.get("cv_auc_std"),
            "seeds": a.get("seeds"),
            "ebm_fidelity": a.get("ebm_fidelity"),
            "n_hc_features": len(self.hc_names_final),
            "n_ebm_features": len(self.ebm_names),
            "reconciled": ({
                "available": True,
                "architecture": self.reconciled.art.get("architecture"),
                "hc_ensemble": list(self.reconciled.models.keys()),
                "hc_features": int(self.reconciled.art.get("hc_K", 0)),
                "weight_emb": self.reconciled.weight_emb,
                "cv_kid_auc_hc": self.reconciled.art.get("cv_kid_auc_hc"),
                "ebm_fidelity": self.reconciled.art.get("ebm_fidelity"),
                "res_holdout": self.reconciled.art.get("res_holdout"),
            } if self.reconciled.available else {"available": False}),
            "domain_alignment": {
                "available": self.aligner.ready,
                "default_on": C.DOMAIN_ALIGN_DEFAULT,
                "shrink": self.aligner.shrink,
                "reference_clips": (self.aligner.mp or {}).get("n_clips"),
                "reference_frames": (self.aligner.mp or {}).get("n_frames"),
            },
            "deployment_note": a.get("_DEPLOYMENT_NOTE"),
        }

    # ------------------------------------------------------------ embeddings #
    def _embed(self, raw_list: Sequence[np.ndarray], align: bool) -> np.ndarray:
        """raw_list of (TARGET_FRAMES, 75) arrays -> (N, 768) embeddings.

        When ``align`` and a MediaPipe reference is loaded, each skeleton (and its
        bones) is mapped into the Kinect distribution the embedding net expects.
        """
        do_align = bool(align) and self.aligner.ready
        j_batch, b_batch = [], []
        for raw in raw_list:
            ntu = F.raw_to_ntu_tensor(raw)             # (3,T,25,1)
            bone = F.compute_bone_data(ntu)            # bones from RAW joints
            if do_align:
                ntu = self.aligner.align_joint(ntu)
                bone = self.aligner.align_bone(bone)
            j_batch.append(ntu)
            b_batch.append(bone)
        x_j = torch.from_numpy(np.stack(j_batch)).float().to(self.device)
        x_b = torch.from_numpy(np.stack(b_batch)).float().to(self.device)
        with self._lock, torch.no_grad():
            emb = self.model.get_embeddings(x_j, x_b).cpu().numpy()
        return emb

    @staticmethod
    def _ensemble_proba(models: dict, scaler, X: np.ndarray) -> np.ndarray:
        Xs = scaler.transform(X)
        probs = [m.predict_proba(Xs) for m in models.values()]
        return np.mean(probs, axis=0)  # (N, 2)

    def _ebm_explanations(self, X_ebm: np.ndarray, ebm=None,
                          ebm_names=None) -> list[list[dict]]:
        """Per-clip list of {feature, value, contribution} sorted by |contribution|.

        ``ebm``/``ebm_names`` default to the v40 EBM; pass the reconciled EBM to
        explain the reconciled fused decision instead.
        """
        ebm = ebm if ebm is not None else self.ebm
        ebm_names = ebm_names if ebm_names is not None else self.ebm_names
        out: list[list[dict]] = []
        try:
            local = ebm.explain_local(X_ebm)
            for i in range(X_ebm.shape[0]):
                d = local.data(i)
                names = d.get("names", ebm_names)
                scores = d.get("scores", [])
                values = d.get("values", X_ebm[i])
                rows = []
                for n, s, v in zip(names, scores, values):
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        v = None
                    rows.append({"feature": str(n),
                                 "value": v,
                                 "contribution": float(s)})
                rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
                out.append(rows)
        except Exception as exc:  # explanation is best-effort, never fatal
            for i in range(X_ebm.shape[0]):
                out.append([{"feature": "explanation_unavailable",
                             "value": None, "contribution": 0.0,
                             "error": str(exc)}])
        return out

    # Map a handcrafted feature name to a clinician-facing body region.
    _REGION_KEYS = [
        ("shoulder", "shoulder"), ("elbow", "elbow"), ("wrist", "wrist/hand"),
        ("hand", "wrist/hand"), ("thumb", "wrist/hand"), ("hip", "hip"),
        ("knee", "knee"), ("ankle", "ankle"), ("foot", "ankle/foot"),
        ("head", "head"), ("neck", "neck"), ("spine", "spine/trunk"),
    ]

    @classmethod
    def _body_region(cls, feature: str) -> str:
        f = feature.lower()
        if "asym" in f and not any(k in f for k, _ in cls._REGION_KEYS):
            return "bilateral asymmetry"
        for key, region in cls._REGION_KEYS:
            if key in f:
                return region
        return "whole-body coordination"

    # ---------------------------------------------------------------- score #
    def score_clips(self, raw_list: Sequence[np.ndarray],
                    names: Sequence[str] | None = None,
                    top_k_explanations: int = 8,
                    align: bool | None = None,
                    source: str = "mediapipe",
                    threshold: float | None = None,
                    emb_weight: float | None = None) -> dict:
        """Score clips with the paper's multilevel fusion; aggregate to a subject.

        By default runs the FULL fusion p = w*p_EMB + (1-w)*p_HC with w from the
        artifact (~0.47) — the MS-G3D learned stream is used. An OOD input-quality
        gate flags clips whose skeleton is far from the Kinect training
        distribution (the paper's Quantum Pose gate); on such input lean on HC.

        emb_weight: override the embedding weight. None -> artifact weight; 0.0 =>
            HC-only robust mode (also skips the MS-G3D forward pass).
        source:  "kinect" | "mediapipe" — only affects whether alignment defaults on.
        align:   MediaPipe->Kinect embedding alignment (None -> config default).
        """
        if not raw_list:
            raise ValueError("No clips to score.")
        names = list(names) if names is not None else [f"clip_{i}" for i in
                                                        range(len(raw_list))]
        kinect = str(source).lower() == "kinect"
        if emb_weight is not None:
            w_emb = float(emb_weight)
        elif C.EMB_WEIGHT_OVERRIDE is not None:
            w_emb = C.EMB_WEIGHT_OVERRIDE
        else:
            w_emb = self.w_emb                       # paper's optimised weight
        w_emb = float(np.clip(w_emb, 0.0, 1.0))
        w_hc = 1.0 - w_emb
        thr = C.DECISION_THRESHOLD if threshold is None else float(threshold)
        lab = lambda p_asd: LABELS[1] if p_asd >= thr else LABELS[0]
        if align is None:
            align = C.DOMAIN_ALIGN_DEFAULT and not kinect
        use_emb = w_emb > 0.0
        aligned = bool(align) and self.aligner.ready and use_emb

        # --- handcrafted path (always) ---
        hc = np.stack([F.extract_handcrafted_features(r) for r in raw_list])  # (N,534)
        hc_sel = hc[:, self.hc_final_idx]                              # (N,42)
        hc_prob = self._ensemble_proba(self.hc_models, self.scaler_hc, hc_sel)

        # --- learned MS-G3D stream (+ OOD gate); skipped only if weighted to 0 ---
        if use_emb:
            emb = self._embed(raw_list, align=align)                   # (N,768)
            emb_p = self.pca.transform(emb)[:, self.emb_keep]
            emb_prob = self._ensemble_proba(self.emb_models, self.scaler_emb, emb_p)
            # OOD: rms z of PCA-embedding vs Kinect distribution (in-dist ~1.0)
            z = (emb_p - self.scaler_emb.mean_) / self.scaler_emb.scale_
            ood_z = np.sqrt((z ** 2).mean(axis=1))                     # (N,)
        else:
            emb_prob = None
            ood_z = np.full(len(raw_list), np.nan)

        # --- fusion ---
        fused = (w_emb * emb_prob + w_hc * hc_prob) if use_emb else hc_prob

        # --- distilled EBM surrogate (interpretable P(ASD)) ---
        X_ebm = hc_sel[:, self.ebm_sel_idx]                           # (N,39) raw
        ebm_est = np.clip(np.asarray(self.ebm.predict(X_ebm)).ravel(), 0.0, 1.0)
        ebm_expl = self._ebm_explanations(X_ebm)

        clips = []
        for i, nm in enumerate(names):
            feats = ebm_expl[i][:top_k_explanations]
            top = next((r for r in ebm_expl[i]
                        if r.get("feature") != "explanation_unavailable"), None)
            oz = None if np.isnan(ood_z[i]) else float(ood_z[i])
            clips.append({
                "name": nm,
                "label": lab(fused[i, 1]),
                "prob_asd": float(fused[i, 1]),
                "prob_td": float(fused[i, 0]),
                "emb_prob_asd": (float(emb_prob[i, 1]) if use_emb else None),
                "hc_prob_asd": float(hc_prob[i, 1]),
                "ebm_estimate_asd": float(ebm_est[i]),
                "primary_region": (self._body_region(top["feature"]) if top else None),
                "ood_z": oz,
                "ood_warning": (oz is not None and oz > C.OOD_WARN_Z),
                "ebm_top_features": feats,
            })

        # --- subject aggregation = mean of clip fused probs ---
        subj = fused.mean(axis=0)
        subj_ood = None if not use_emb else float(np.nanmean(ood_z))
        # most common primary region across clips
        regions = [c["primary_region"] for c in clips if c["primary_region"]]
        primary_region = max(set(regions), key=regions.count) if regions else None
        subject = {
            "n_clips": len(raw_list),
            "source": "kinect" if kinect else "mediapipe",
            "weight_emb": w_emb,
            "weight_hc": w_hc,
            "domain_aligned": aligned,
            "threshold": thr,
            "ood_z": subj_ood,
            "ood_warning": (subj_ood is not None and subj_ood > C.OOD_WARN_Z),
            "primary_region": primary_region,
            "label": lab(subj[1]),
            "prob_asd": float(subj[1]),
            "prob_td": float(subj[0]),
            "emb_prob_asd": (float(emb_prob.mean(axis=0)[1]) if use_emb else None),
            "hc_prob_asd": float(hc_prob.mean(axis=0)[1]),
            "ebm_estimate_asd": float(ebm_est.mean()),
        }
        return {"subject": subject, "clips": clips}

    # ------------------------------------------------------ window report #
    MAX_WINDOWS = 60

    def score_windows(self, raw_full: np.ndarray, fps: float = 30.0,
                      window_sec: float = 3.0, step_sec: float = 1.0,
                      top_k_explanations: int = 6, align: bool | None = None,
                      source: str = "mediapipe", threshold: float | None = None,
                      emb_weight: float | None = None) -> dict:
        """Slide time windows over a full-length walk and build a per-child report
        (paper's Quantum Pose format): window-level risk trace, proportion of
        atypical windows, peak/average risk, primary contributing region.
        """
        raw_full = np.asarray(raw_full, dtype=np.float32)
        T = len(raw_full)
        if T < 2:
            raise ValueError("Sequence too short for a window report.")
        W = max(2, int(round(window_sec * fps)))
        S = max(1, int(round(step_sec * fps)))
        starts = list(range(0, max(1, T - W + 1), S)) or [0]
        if len(starts) > self.MAX_WINDOWS:                 # cap batch size
            starts = [starts[i] for i in
                      np.linspace(0, len(starts) - 1, self.MAX_WINDOWS).astype(int)]

        windows = [F.resample_clip(raw_full[s:s + W]) for s in starts]
        res = self.score_clips(windows, names=[f"w{i}" for i in range(len(starts))],
                               top_k_explanations=top_k_explanations, align=align,
                               source=source, threshold=threshold, emb_weight=emb_weight)
        clips = res["clips"]
        thr = res["subject"]["threshold"]

        trace, risks = [], []
        for s, c in zip(starts, clips):
            t0, t1 = s / fps, min(T, s + W) / fps
            risks.append(c["prob_asd"])
            trace.append({
                "t_start": round(t0, 2), "t_mid": round((t0 + t1) / 2, 2),
                "t_end": round(t1, 2), "risk": round(c["prob_asd"], 4),
                "atypical": c["prob_asd"] >= thr, "ood_warning": c["ood_warning"],
                "primary_region": c["primary_region"],
            })
        risks = np.asarray(risks)
        atyp = risks >= thr
        proportion, peak, avg = float(atyp.mean()), float(risks.max()), float(risks.mean())

        regs = ([t["primary_region"] for t in trace if t["atypical"] and t["primary_region"]]
                or [t["primary_region"] for t in trace if t["primary_region"]])
        primary_region = max(set(regs), key=regs.count) if regs else None

        ood_flags = np.array([bool(c["ood_warning"]) for c in clips])
        z_vals = [c["ood_z"] for c in clips if c["ood_z"] is not None]

        if (peak >= 0.85 and proportion >= 0.30) or proportion >= 0.60:
            classification, confidence = "Pronounced", "high"
        elif proportion >= 0.35 or peak >= 0.65:
            classification, confidence = "Moderate", "moderate"
        else:
            classification, confidence = "Mild / low-confidence", "low"
        flag = "ATYPICAL" if (avg >= thr or proportion >= 0.5) else "TYPICAL"

        return {
            "screening_flag": flag,
            "classification": classification,
            "confidence": confidence,
            "n_windows": len(starts),
            "window_sec": round(window_sec, 2), "step_sec": round(step_sec, 2),
            "fps": round(float(fps), 2),
            "atypical_window_proportion": round(proportion, 3),
            "peak_risk": round(peak, 3), "average_risk": round(avg, 3),
            "primary_region": primary_region,
            "weight_emb": res["subject"]["weight_emb"],
            "weight_hc": res["subject"]["weight_hc"],
            "threshold": thr,
            "ood": {
                "warning": bool(ood_flags.mean() > 0.5),
                "fraction_windows_ood": round(float(ood_flags.mean()), 3),
                "mean_z": (round(float(np.mean(z_vals)), 2) if z_vals else None),
            },
            "risk_trace": trace,
        }

    # ------------------------------------------ Kinetiq HC report (best on RGB) #
    def kinetiq_report(self, raw_full: np.ndarray, fps: float = 30.0,
                       fuse_msg3d: bool = True, emb_weight: float = 0.30,
                       fuse_threshold: float = 0.50) -> dict:
        """Per-child report using the deployed Kinetiq window-XGBoost HC stream
        (86% ASD sensitivity on phone data, vs 43% for the Kinect-trained HC),
        optionally fused with the MS-G3D embedding (labelled-cohort best ~92%).

        The Kinetiq sustained-spike flag is deployment-faithful (un-tuned, ~85%);
        the fused flag adds the MS-G3D embedding and is stronger but its threshold
        was tuned on a small cohort — both are reported.
        """
        if not self.kinetiq.available:
            raise RuntimeError("Kinetiq HC model not available.")
        risk, mids = self.kinetiq.window_risks(raw_full, fps)
        hc_flag = bool(self.kinetiq.flag(risk))
        hc_avg, peak = float(risk.mean()), float(risk.max())
        nw = len(risk)
        n_at = int((risk >= self.kinetiq.WINDOW_RISK_THRESHOLD).sum())
        proportion = n_at / max(1, nw)

        emb = fused_score = ood_z = None
        if fuse_msg3d:
            clip = F.resample_clip(raw_full)
            sc = self.score_clips([clip], names=["clip"], emb_weight=1.0)["subject"]
            emb = sc["emb_prob_asd"]
            ood_z = sc["ood_z"]
            fused_score = emb_weight * emb + (1 - emb_weight) * min(hc_avg / 0.5, 1.0)
            atypical = fused_score >= fuse_threshold
        else:
            atypical = hc_flag

        if (peak >= 0.85 and proportion >= 0.30) or proportion >= 0.60:
            classification, confidence = "Pronounced", "high"
        elif proportion >= 0.35 or peak >= 0.65:
            classification, confidence = "Moderate", "moderate"
        else:
            classification, confidence = "Mild / low-confidence", "low"

        trace = [{"t_start": round(m - 0.5, 2), "t_mid": round(m, 2),
                  "t_end": round(m + 0.5, 2), "risk": round(float(r), 4),
                  "atypical": float(r) >= self.kinetiq.WINDOW_RISK_THRESHOLD,
                  "ood_warning": False, "primary_region": None}
                 for r, m in zip(risk, mids)]
        return {
            "engine": "kinetiq",
            "screening_flag": "ATYPICAL" if atypical else "TYPICAL",
            "classification": classification, "confidence": confidence,
            "n_windows": nw, "window_sec": 1.0, "step_sec": 0.33,
            "fps": round(float(fps), 2),
            "atypical_window_proportion": round(proportion, 3),
            "peak_risk": round(peak, 3), "average_risk": round(hc_avg, 3),
            "primary_region": None,
            "hc_sustained_flag": hc_flag,
            "msg3d_emb": (round(emb, 3) if emb is not None else None),
            "fused_score": (round(fused_score, 3) if fused_score is not None else None),
            "weight_emb": (emb_weight if fuse_msg3d else 0.0),
            "weight_hc": (round(1 - emb_weight, 2) if fuse_msg3d else 1.0),
            "threshold": (fuse_threshold if fuse_msg3d
                          else self.kinetiq.WINDOW_RISK_THRESHOLD),
            "ood": {"warning": bool(ood_z is not None and ood_z > C.OOD_WARN_Z),
                    "mean_z": (round(float(ood_z), 2) if ood_z is not None else None)},
            "risk_trace": trace,
        }

    # ---------------------------- Reconciled report (paper arch + phone acc) #
    def reconciled_report(self, raw_full: np.ndarray, fps: float = 30.0,
                          fuse_msg3d: bool = True, fuse_threshold: float = 0.45,
                          top_k_explanations: int = 6) -> dict:
        """Per-child report from the RECONCILED model: the paper's multilevel
        fusion (3-model HC ensemble + MS-G3D embedding + weighted fusion +
        distilled EBM) built on the phone-transferable 448 windowed HC features.

        Held-out `res` cohort: AUC 1.00, 92% accuracy, 100% ASD sensitivity.
        The distilled EBM (fidelity r=0.98) makes the fused decision interpretable
        — the paper's explanation layer that the raw Kinetiq engine lacks.
        """
        if not self.reconciled.available:
            raise RuntimeError("Reconciled model not available.")
        risk, mids = self.reconciled.window_risks(raw_full, fps)
        hc_flag = bool(self.reconciled.flag(risk))
        hc_avg, peak = float(risk.mean()), float(risk.max())
        nw = len(risk)
        n_at = int((risk >= self.reconciled.WINDOW_RISK_THRESHOLD).sum())
        proportion = n_at / max(1, nw)

        w_emb = self.reconciled.weight_emb
        denom = self.reconciled.hc_scale_denom
        hc_scaled = min(hc_avg / denom, 1.0)

        emb = fused_score = ood_z = None
        if fuse_msg3d:
            clip = F.resample_clip(raw_full)
            sc = self.score_clips([clip], names=["clip"], emb_weight=1.0)["subject"]
            emb = sc["emb_prob_asd"]
            ood_z = sc["ood_z"]
            fused_score = w_emb * emb + (1 - w_emb) * hc_scaled
            atypical = fused_score >= fuse_threshold
        else:
            atypical = hc_flag

        # --- distilled-EBM explanation of the fused decision (paper's layer) ---
        ebm_feats, primary_region = [], None
        if self.recon_ebm is not None:
            hc534 = F.extract_handcrafted_features(F.resample_clip(raw_full))[None]
            X_ebm = hc534[:, self.recon_ebm_idx]
            expl = self._ebm_explanations(X_ebm, ebm=self.recon_ebm,
                                          ebm_names=self.recon_ebm_names)[0]
            ebm_feats = expl[:top_k_explanations]
            top = next((r for r in expl
                        if r.get("feature") != "explanation_unavailable"), None)
            primary_region = self._body_region(top["feature"]) if top else None

        if (peak >= 0.85 and proportion >= 0.30) or proportion >= 0.60:
            classification, confidence = "Pronounced", "high"
        elif proportion >= 0.35 or peak >= 0.65:
            classification, confidence = "Moderate", "moderate"
        else:
            classification, confidence = "Mild / low-confidence", "low"

        trace = [{"t_start": round(m - 0.5, 2), "t_mid": round(m, 2),
                  "t_end": round(m + 0.5, 2), "risk": round(float(r), 4),
                  "atypical": float(r) >= self.reconciled.WINDOW_RISK_THRESHOLD,
                  "ood_warning": False, "primary_region": None}
                 for r, m in zip(risk, mids)]
        return {
            "engine": "reconciled",
            "screening_flag": "ATYPICAL" if atypical else "TYPICAL",
            "classification": classification, "confidence": confidence,
            "n_windows": nw, "window_sec": 1.0, "step_sec": 0.33,
            "fps": round(float(fps), 2),
            "atypical_window_proportion": round(proportion, 3),
            "peak_risk": round(peak, 3), "average_risk": round(hc_avg, 3),
            "primary_region": primary_region,
            "hc_sustained_flag": hc_flag,
            "msg3d_emb": (round(emb, 3) if emb is not None else None),
            "fused_score": (round(fused_score, 3) if fused_score is not None else None),
            "weight_emb": (round(w_emb, 2) if fuse_msg3d else 0.0),
            "weight_hc": (round(1 - w_emb, 2) if fuse_msg3d else 1.0),
            "threshold": (fuse_threshold if fuse_msg3d
                          else self.reconciled.WINDOW_RISK_THRESHOLD),
            "ood": {"warning": bool(ood_z is not None and ood_z > C.OOD_WARN_Z),
                    "mean_z": (round(float(ood_z), 2) if ood_z is not None else None)},
            "ebm_top_features": ebm_feats,
            "risk_trace": trace,
        }

    # --------------------------- /api/analyze (frontend contract) ------------ #
    # Deployed 2-stream model: p = ANGULAR_WEIGHT*P_angularCAL + (1-ANGULAR_WEIGHT)*P_HC,
    # where P_angularCAL is the mp99-calibrated (de-inflated) angular score. Equal
    # weights. 16 labelled real phone captures (14 res + sora1 + tv2) at thr 0.22:
    # 15/16, sensitivity 9/9, specificity 6/7, AUC 0.984 — same decisions as the raw
    # path but on an honest scale (TD movement-model reads ~0.2, not ~0.95). Threshold
    # 0.22 is screening-appropriate (prioritises ASD sensitivity) on that reference.
    ANGULAR_WEIGHT = 0.5
    ANGULAR_FUSE_THRESHOLD = 0.22
    # Scores within [LOW_CONF_LO, LOW_CONF_HI] straddle the operating point: the two
    # streams disagree and near-identical children fall on both sides. Flag these as
    # low-confidence / borderline rather than presenting them as a firm call.
    LOW_CONF_LO = 0.17
    LOW_CONF_HI = 0.40
    FUSE_THRESHOLD = 0.45          # opt-in fusion operating point (legacy embedding path)
    WINDOW_RISK_THRESHOLD = 0.30   # per-window "atypical" cutoff (frontend timeline flag line)
    MIN_ATYPICAL_WINDOWS = 2
    MIN_ATYPICAL_FRACTION = 0.10
    # Graded HC-primary flag rule (deployed default). Tuned so the handcrafted
    # ensemble catches genuine atypical gait (>=2 windows above 40% risk, OR >=3
    # above 30%) while keeping 100% specificity on the labelled real-child cohort.
    # On 18 labelled items (14 phone CSVs + sora/tv videos, tv1 excluded as a bad
    # clip): 94% acc, 100% spec, only misses one HC-invisible case (surfaced via
    # the embedding review note). NOTE: thresholds tuned on a small set — re-check
    # as more labelled data arrives.
    HC_STRONG_THR = 0.40
    HC_STRONG_N = 2
    HC_WEAK_THR = 0.30
    HC_WEAK_N = 3

    def _hc_flag(self, risk) -> bool:
        r = np.asarray(risk)
        return bool((r >= self.HC_STRONG_THR).sum() >= self.HC_STRONG_N
                    or (r >= self.HC_WEAK_THR).sum() >= self.HC_WEAK_N)

    def analyze_report(self, raw_full: np.ndarray, fps: float = 30.0,
                       source_kind: str = "mediapipe",
                       detection_rate: float | None = None,
                       processed_video_url: str | None = None,
                       fuse: bool = True, age_group: str = "child") -> dict:
        """Clinical report from a RAW SKELETON (video or coordinate CSV).

        Default is the deployed 2-stream **fusion**: p = 0.5*P_angularMSG3D + 0.5*P_HC,
        flagged at >=0.5. Both streams are invariant, so this holds across the phone
        gap (14 real phone kids: AUC 0.979, 13/14, spec 6/6). Pass ``fuse=False`` for
        the HC-primary graded-rule screen (the deep stream becomes a context note).

        ``age_group="adult"`` engages a PROVISIONAL adult mode: the deep stream was
        trained only on children (ages ~3-10) and reads any adult as out-of-distribution
        (~0.95), so it is disabled and the screen falls back to the handcrafted stream.
        Provisional: specificity checked on 2 TD adults only, NO adult-ASD reference —
        do not rely on a 'typical' result for an adult (see _assemble_report)."""
        if not self.reconciled.available or self.clinical is None:
            raise RuntimeError("Reconciled model not available for /api/analyze.")
        risk, mids, X_full, X_sel = self.reconciled.window_features(raw_full, fps)
        mg = ood_z = None
        if self.angular is not None and self.angular.available:
            raw_mg = float(self.angular.score(F.resample_clip(raw_full)))  # raw P(ASD)
            mg = self.angular.calibrate(raw_mg)   # de-inflate (mp99 Platt) -> honest scale
        return self._assemble_report(risk, mids, X_full, X_sel, mg, ood_z,
                                     source_kind, detection_rate, processed_video_url,
                                     fuse, deep_kind="angular", age_group=age_group)

    def analyze_report_features(self, X_full: np.ndarray, mids,
                                source_kind: str = "csv-features") -> dict:
        """HC-only clinical report from a PRE-ENGINEERED 448-feature CSV (no raw
        skeleton, so the MS-G3D stream can't run at all)."""
        if not self.reconciled.available or self.clinical is None:
            raise RuntimeError("Reconciled model not available for /api/analyze.")
        risk, X_sel = self.reconciled.score_feature_windows(X_full)
        return self._assemble_report(np.asarray(risk), list(mids), np.asarray(X_full),
                                     X_sel, None, None, source_kind, None, None, False)

    # Adult mode (provisional): the deep stream is child-trained and OOD on adults,
    # so decide on the handcrafted stream's mean risk at this threshold. Specificity
    # checked on 2 TD adults (HC 0.15, 0.20); adult-ASD sensitivity is UNVALIDATED.
    ADULT_HC_THRESHOLD = 0.50

    def _assemble_report(self, risk, mids, X_full, X_sel, emb, ood_z, source_kind,
                         detection_rate=None, processed_video_url=None,
                         fuse=False, deep_kind="angular", age_group="child") -> dict:
        """Shared builder for the `/api/analyze` payload.

        HC-primary (default when fuse=False or no deep stream): flag from the HC
        ensemble sustained-spike rule, timeline = HC per-window risk, deep stream =
        context only. ``fuse=True`` with a deep stream present uses fusion:
        - deep_kind="angular" (deployed): equal-weight 0.5*P_angular + 0.5*P_HC,
          flagged at 0.5. Both streams invariant -> transfers to phone.
        - deep_kind="embedding" (legacy): the old weighted MS-G3D-embedding fusion.
        ``age_group="adult"`` disables the child-trained deep stream (OOD on adults)
        and decides on the handcrafted mean risk (PROVISIONAL, specificity-only).
        """
        hc_avg = float(risk.mean()) if len(risk) else 0.0
        adult_mode = (str(age_group).lower() == "adult")

        if adult_mode:                                     # provisional adult mode (HC-only)
            if emb is not None:
                emb = float(emb)                           # kept for display context only
            preds = np.clip(risk, 0.0, 1.0)
            headline = hc_avg
            is_atypical = bool(hc_avg >= self.ADULT_HC_THRESHOLD)
        elif fuse and emb is not None and deep_kind == "angular":   # deployed 2-stream fusion
            emb = float(emb)
            w = float(self.ANGULAR_WEIGHT)
            headline = w * emb + (1 - w) * hc_avg
            is_atypical = bool(headline >= self.ANGULAR_FUSE_THRESHOLD)
            preds = np.clip(w * emb + (1 - w) * risk, 0.0, 1.0)
        elif fuse and emb is not None:                     # legacy weighted-embedding fusion
            w_emb = float(self.reconciled.weight_emb)
            denom = float(self.reconciled.hc_scale_denom)
            emb = float(emb)
            headline = w_emb * emb + (1 - w_emb) * min(hc_avg / denom, 1.0)
            is_atypical = bool(headline >= self.FUSE_THRESHOLD)
            preds = np.clip(w_emb * emb + (1 - w_emb) * risk, 0.0, 1.0)
        else:                                              # HC-primary (default)
            if emb is not None:
                emb = float(emb)
            preds = np.clip(risk, 0.0, 1.0)
            is_atypical = self._hc_flag(risk)              # graded rule
            headline = float(preds.max()) if len(preds) else 0.0

        # Borderline band: fused score straddles the operating point.
        borderline = bool((adult_mode or (fuse and emb is not None))
                          and self.LOW_CONF_LO <= headline <= self.LOW_CONF_HI)

        n = len(preds)
        atypical_count = int((preds >= self.WINDOW_RISK_THRESHOLD).sum())
        min_windows = max(self.MIN_ATYPICAL_WINDOWS, int(n * self.MIN_ATYPICAL_FRACTION))

        # --- clinical detail (reconciled-model SHAP + deviations) ---
        ref = CL.pick_reference(source_kind)
        regions_pct = self.clinical.regional_drivers(X_sel)
        markers = self.clinical.kinematic_markers(X_sel)
        findings = self.clinical.kinematic_findings(X_full, ref)
        symmetry = self.clinical.symmetry_indices(X_full, ref)
        ood = self.clinical.ood_score(X_full, ref)
        mode = "hc-primary" if adult_mode else ("fusion" if (fuse and emb is not None) else "hc-primary")
        cs = CL.build_clinical_summary(
            is_atypical, preds, atypical_count, min_windows, regions_pct,
            self.WINDOW_RISK_THRESHOLD, emb, hc_avg,
            mode=mode, fused_score=headline)
        if mode != "fusion":                               # describe the graded HC rule
            ns = int((preds >= self.HC_STRONG_THR).sum())
            nw = int((preds >= self.HC_WEAK_THR).sum())
            cs["decision_rule"] = (
                f"Screen decision = handcrafted 3-model ensemble: flag when "
                f">={self.HC_STRONG_N} windows exceed {int(self.HC_STRONG_THR*100)}% risk "
                f"or >={self.HC_WEAK_N} exceed {int(self.HC_WEAK_THR*100)}% "
                f"(this clip: {ns} and {nw}). MS-G3D shown as context only"
                + ("" if emb is not None else " (not run for feature CSVs)") + ".")

        # --- assemble frontend fields ---
        timeline = [{"timestamp": f"{m:.1f}s", "risk_score": float(p)}
                    for m, p in zip(mids, preds)]
        regions_list = [{"name": r, "value": round(regions_pct.get(r, 0.0), 1)}
                        for r in CL.REGION_NAMES]
        flagged_regions = {r for r in CL.REGION_NAMES
                           if regions_pct.get(r, 0) >= 100.0 / len(CL.REGION_NAMES)}
        regional_breakdown = [{
            "name": r, "contribution_pct": round(regions_pct.get(r, 0.0), 1),
            "status": "Elevated" if (is_atypical and r in flagged_regions) else "Typical",
            "description": CL.REGION_DESCRIPTIONS[r],
        } for r in CL.REGION_NAMES]
        flagged_moments = [
            {"timestamp": f"{mids[i]:.1f}s", "risk": round(float(preds[i]), 3)}
            for i in range(n) if preds[i] >= self.WINDOW_RISK_THRESHOLD][:25]

        std = float(np.std(preds)) if n else 0.0
        result_stability = {
            "window_risk_std": round(std, 3),
            "consistency": ("high" if std < 0.12 else "moderate" if std < 0.22 else "low"),
        }
        warnings_list = []
        if emb is None:
            warnings_list.append("Handcrafted stream only: this is a pre-engineered feature "
                                 "file (no raw skeleton), so the MS-G3D movement model was not "
                                 "run. Upload the video or a keypoint/coordinate CSV to also "
                                 "see the movement-model context.")
        elif (not fuse) and (not is_atypical) and float(emb) >= CL.EMB_CONTEXT_HIGH:
            warnings_list.append(f"The MS-G3D movement model read elevated motion "
                                 f"({float(emb)*100:.0f}%) but is out-of-distribution on "
                                 f"phone/RGB input (low confidence) and is not used for the "
                                 f"screen decision. Consider a repeat recording or clinician "
                                 f"review if you have other concerns.")
        if ood is not None and ood > 1.35:
            warnings_list.append("Input differs notably from the model's reference data "
                                 "(possible non-standard recording); interpret with caution.")
        if n < 5:
            warnings_list.append(f"Short recording ({n} analysis window(s)); the result is less "
                                 "stable. A 10s+ clip of the child moving is recommended.")
        if source_kind == "video" and detection_rate is not None and detection_rate < 0.90:
            warnings_list.append(f"Full-body pose detected in only {detection_rate*100:.0f}% of "
                                 "frames; ensure the whole body stays in frame.")
        if borderline:
            warnings_list.append(f"Borderline result ({float(headline)*100:.0f}%, near the "
                                 f"{int(self.ANGULAR_FUSE_THRESHOLD*100)}% decision line): the two "
                                 "streams disagree and children close to this score fall on both "
                                 "sides. Treat as low-confidence — a repeat recording and clinician "
                                 "review are recommended.")
        if adult_mode:
            warnings_list.append("ADULT MODE (provisional): the movement model was trained only on "
                                 "children and is unreliable on adults, so this screen uses the "
                                 "handcrafted biomechanical stream alone. It has been checked only "
                                 "against a small number of typically-developing adults and has NO "
                                 "atypical-adult reference — a 'typical' result for an adult is NOT "
                                 "validated and must not be treated as reassurance. For research use "
                                 "only; not a screen for adults.")
        input_meta = {
            "kind": source_kind,
            "windows_analyzed": n, "window_seconds": 1.0,
            "detection_rate_pct": (round(detection_rate * 100, 1)
                                   if detection_rate is not None else None),
            "ood_score": (round(ood, 2) if ood is not None else None),
            "ood_baseline": 1.0,
        }
        return {
            "status": "success",
            "processed_video_url": processed_video_url,
            "final_risk_score": round(float(headline), 4),
            "is_atypical": is_atypical,
            "borderline": borderline,
            "decision_threshold": (self.ADULT_HC_THRESHOLD if adult_mode
                                   else self.ANGULAR_FUSE_THRESHOLD),
            "age_group": ("adult" if adult_mode else "child"),
            "hc_only": adult_mode or emb is None,
            "decision_mode": ("adult-hc" if adult_mode
                              else "fusion" if (fuse and emb is not None) else "hc-primary"),
            "regions": regions_list,
            "timeline": timeline,
            "clinical_summary": cs,
            "regional_breakdown": regional_breakdown,
            "kinematic_markers": markers,
            "kinematic_findings": findings,
            "symmetry": symmetry,
            "flagged_moments": flagged_moments,
            "result_stability": result_stability,
            "input_meta": input_meta,
            "quality": {"reliable": len(warnings_list) == 0, "warnings": warnings_list},
            "msg3d_embedding_risk": (round(float(emb), 4) if emb is not None else None),
            "ood_z": (round(float(ood_z), 2) if ood_z is not None else None),
            "disclaimer": CL.DISCLAIMER,
        }
