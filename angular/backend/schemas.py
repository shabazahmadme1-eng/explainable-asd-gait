"""Pydantic response models for the API (drive the auto-generated OpenAPI docs)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class EBMFeature(BaseModel):
    feature: str
    value: Optional[float] = None
    contribution: float


class ClipResult(BaseModel):
    name: str
    label: str                      # "ASD" | "TD"
    prob_asd: float
    prob_td: float
    emb_prob_asd: Optional[float] = None   # None when embedding is down-weighted out
    hc_prob_asd: float
    ebm_estimate_asd: float
    primary_region: Optional[str] = None   # top contributing body region (EBM)
    ood_z: Optional[float] = None          # embedding distance from Kinect dist (sigma)
    ood_warning: bool = False              # input-quality gate: learned stream is OOD
    ebm_top_features: list[EBMFeature] = []
    meta: Optional[dict] = None     # populated for video clips (detection rate, …)


class SubjectResult(BaseModel):
    n_clips: int
    source: str = "mediapipe"       # "mediapipe" | "kinect"
    weight_emb: float = 0.0
    weight_hc: float = 1.0
    domain_aligned: bool = False
    threshold: float = 0.5
    ood_z: Optional[float] = None
    ood_warning: bool = False
    primary_region: Optional[str] = None
    label: str
    prob_asd: float
    prob_td: float
    emb_prob_asd: Optional[float] = None
    hc_prob_asd: float
    ebm_estimate_asd: float


class PredictResponse(BaseModel):
    subject: SubjectResult
    clips: list[ClipResult]
    errors: list[dict] = []
    disclaimer: str


class WindowPoint(BaseModel):
    t_start: float
    t_mid: float
    t_end: float
    risk: float
    atypical: bool
    ood_warning: bool = False
    primary_region: Optional[str] = None


class ReportResponse(BaseModel):
    """Per-child window-level screening report (paper's Quantum Pose format)."""
    name: str
    engine: str = "reconciled"      # "reconciled" (paper arch + phone acc) | "kinetiq" | "v40"
    screening_flag: str             # "ATYPICAL" | "TYPICAL"
    classification: str             # "Pronounced" | "Moderate" | "Mild / low-confidence"
    confidence: str                 # "high" | "moderate" | "low"
    hc_sustained_flag: Optional[bool] = None   # kinetiq deployment-faithful flag
    msg3d_emb: Optional[float] = None          # MS-G3D embedding P(ASD), if fused
    fused_score: Optional[float] = None        # fused HC+embedding score, if fused
    n_windows: int
    window_sec: float
    step_sec: float
    fps: float
    atypical_window_proportion: float
    peak_risk: float
    average_risk: float
    primary_region: Optional[str] = None
    weight_emb: float
    weight_hc: float
    threshold: float
    ood: dict
    ebm_top_features: list[EBMFeature] = []   # distilled-EBM explanation (reconciled/v40)
    risk_trace: list[WindowPoint] = []
    meta: Optional[dict] = None
    disclaimer: str


class InfoResponse(BaseModel):
    version: Optional[str] = None
    architecture: Optional[str] = None
    device: str
    fusion_weights: dict
    cv_accuracy_kinect: Optional[float] = None
    cv_auc_kinect: Optional[float] = None
    cv_acc_mean: Optional[float] = None
    cv_acc_std: Optional[float] = None
    cv_auc_mean: Optional[float] = None
    cv_auc_std: Optional[float] = None
    seeds: Optional[list] = None
    ebm_fidelity: Optional[float] = None
    n_hc_features: int
    n_ebm_features: int
    reconciled: Optional[dict] = None
    ood_warn_z: Optional[float] = None
    domain_alignment: Optional[dict] = None
    deployment_note: Optional[str] = None
