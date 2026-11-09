"""
FastAPI service for the ASD-GAIT v40 model.

Endpoints
---------
GET  /health                  liveness + which device/model is loaded
GET  /info                    model metadata, CV metrics, fusion weights
POST /predict/csv             one or more Kinect skeleton CSV/XLSX files
POST /predict/video           one or more RGB videos (MediaPipe, experimental)

All uploaded files in a single request are treated as clips of ONE subject:
the response gives a subject-level aggregate plus per-clip results (so a single
file simply yields a one-clip subject).

Run:  uvicorn backend.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

from . import config as C
from . import ntu_features as F
from .inference import Predictor
from .schemas import InfoResponse, PredictResponse, ReportResponse
from .video_pose import (extract_skeleton_from_video,
                         extract_skeleton_full_from_video)

DISCLAIMER = (
    "Research prototype. Trained on Kinect skeletons (Kinect->Kinect CV "
    "~0.93 acc). RGB-video accuracy is unmeasured and the video path uses an "
    "approximate MediaPipe->NTU joint mapping. NOT a medical device; not for "
    "diagnostic use."
)
CSV_EXTS = (".csv", ".xlsx")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model + artifacts once at startup.
    app.state.predictor = Predictor()
    print(f"[startup] Predictor ready on {app.state.predictor.device} "
          f"(version={app.state.predictor.info().get('version')})")
    yield


app = FastAPI(
    title="ASD-GAIT v40 Inference API",
    description="MS-G3D + handcrafted-feature fusion with a distilled EBM "
                "explanation layer, for ASD-vs-TD movement screening.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _predictor(app_) -> Predictor:
    p = getattr(app_.state, "predictor", None)
    if p is None:  # pragma: no cover
        raise HTTPException(503, "Model not loaded yet.")
    return p


def _save_temp(upload: UploadFile, data: bytes) -> str:
    suffix = os.path.splitext(upload.filename or "")[1] or ".bin"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index():
    path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(404, "UI not found.")


@app.get("/health")
def health():
    p = _predictor(app)
    return {"status": "ok", "device": str(p.device),
            "version": p.info().get("version")}


@app.get("/info", response_model=InfoResponse)
def info():
    return _predictor(app).info()


@app.post("/predict/csv", response_model=PredictResponse)
async def predict_csv(
    files: list[UploadFile] = File(..., description="Kinect skeleton CSV/XLSX"),
    top_k: int = Query(8, ge=0, le=42, description="EBM features per clip"),
    source: str = Query(
        "mediapipe", description="'kinect' or 'mediapipe' — only controls whether "
        "embedding alignment defaults on. Fusion runs the full paper method either way."),
    emb_weight: float | None = Query(
        None, ge=0.0, le=1.0, description="Override embedding fusion weight "
        "(None=paper's artifact weight ~0.47; 0.0=HC-only robust mode)"),
    align: bool | None = Query(
        None, description="MediaPipe->Kinect domain alignment of the embedding "
        "stream (None=config default; only used when embedding weight > 0)"),
    threshold: float | None = Query(
        None, ge=0.0, le=1.0, description="P(ASD) decision threshold for the label "
        "(None=config default 0.5; ~0.31 was used for MediaPipe in asd_backend)"),
):
    p = _predictor(app)
    raws, names, errors = [], [], []
    for up in files:
        ext = os.path.splitext(up.filename or "")[1].lower()
        if ext not in CSV_EXTS:
            errors.append({"file": up.filename, "error": f"unsupported ext {ext}"})
            continue
        path = _save_temp(up, await up.read())
        try:
            raw, n_clip = F.parse_skeleton_file(path)
            if raw is None:
                errors.append({"file": up.filename, "error": "could not parse "
                               "(need >=76 columns: index + 75 coords)"})
                continue
            raws.append(raw)
            names.append(up.filename)
        finally:
            os.unlink(path)

    if not raws:
        raise HTTPException(422, {"message": "No valid skeleton files.",
                                  "errors": errors})
    result = p.score_clips(raws, names=names, top_k_explanations=top_k,
                           align=align, source=source, threshold=threshold,
                           emb_weight=emb_weight)
    return {**result, "errors": errors, "disclaimer": DISCLAIMER}


@app.post("/predict/video", response_model=PredictResponse)
async def predict_video(
    files: list[UploadFile] = File(..., description="RGB video files"),
    top_k: int = Query(8, ge=0, le=42, description="EBM features per clip"),
    model_complexity: int = Query(1, ge=0, le=2,
                                  description="MediaPipe Pose complexity (0/1/2)"),
    emb_weight: float | None = Query(
        None, ge=0.0, le=1.0, description="Override embedding fusion weight "
        "(None=paper's artifact weight; 0.0=HC-only robust mode)"),
    align: bool | None = Query(
        None, description="MediaPipe->Kinect domain alignment of the embedding "
        "stream (None=config default)"),
    threshold: float | None = Query(
        None, ge=0.0, le=1.0, description="P(ASD) decision threshold for the label"),
):
    p = _predictor(app)
    raws, names, metas, errors = [], [], [], []
    for up in files:
        ext = os.path.splitext(up.filename or "")[1].lower()
        if ext not in VIDEO_EXTS:
            errors.append({"file": up.filename, "error": f"unsupported ext {ext}"})
            continue
        path = _save_temp(up, await up.read())
        try:
            raw, meta = extract_skeleton_from_video(
                path, model_complexity=model_complexity)
            raws.append(raw)
            names.append(up.filename)
            metas.append(meta)
        except Exception as exc:
            errors.append({"file": up.filename, "error": str(exc)})
        finally:
            os.unlink(path)

    if not raws:
        raise HTTPException(422, {"message": "No pose could be extracted.",
                                  "errors": errors})
    result = p.score_clips(raws, names=names, top_k_explanations=top_k,
                           align=align, source="mediapipe", threshold=threshold,
                           emb_weight=emb_weight)
    for clip, meta in zip(result["clips"], metas):
        clip["meta"] = meta
    return {**result, "errors": errors, "disclaimer": DISCLAIMER}


# --------------------------------------------------------------------------- #
# Per-child window-level report (paper's Quantum Pose format)
# --------------------------------------------------------------------------- #
@app.post("/report/csv", response_model=ReportResponse)
async def report_csv(
    file: UploadFile = File(..., description="A single child's keypoint CSV/XLSX"),
    engine: str = Query("reconciled", description="'reconciled' (paper's 3-model HC "
                        "ensemble + MS-G3D fusion + EBM, phone-transferable — AUC 1.0 "
                        "on held-out cohort), 'kinetiq' (deployed single-XGBoost HC), "
                        "or 'v40' (Kinect-HC fusion)"),
    fuse_msg3d: bool = Query(True, description="[reconciled/kinetiq] fuse the MS-G3D "
                            "embedding with the HC stream (paper's multilevel fusion)"),
    window_sec: float = Query(3.0, gt=0, le=30, description="[v40] window length (s)"),
    step_sec: float = Query(1.0, gt=0, le=30, description="[v40] window step (s)"),
    emb_weight: float | None = Query(None, ge=0.0, le=1.0),
    threshold: float | None = Query(None, ge=0.0, le=1.0),
    align: bool | None = Query(None),
    source: str = Query("mediapipe"),
):
    p = _predictor(app)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in CSV_EXTS:
        raise HTTPException(422, f"Unsupported extension {ext}; expected CSV/XLSX.")
    path = _save_temp(file, await file.read())
    try:
        raw_full, fps = F.parse_skeleton_full(path)
    finally:
        os.unlink(path)
    if raw_full is None:
        raise HTTPException(422, "Could not parse keypoint file (need index + 75 coords).")
    if engine == "reconciled" and p.reconciled.available:
        report = p.reconciled_report(raw_full, fps=fps, fuse_msg3d=fuse_msg3d)
    elif engine == "kinetiq" and p.kinetiq.available:
        report = p.kinetiq_report(raw_full, fps=fps, fuse_msg3d=fuse_msg3d)
    else:
        report = p.score_windows(raw_full, fps=fps, window_sec=window_sec,
                                 step_sec=step_sec, align=align, source=source,
                                 threshold=threshold, emb_weight=emb_weight)
        report["engine"] = "v40"
    return {"name": file.filename, **report,
            "meta": {"total_frames": len(raw_full), "fps": round(fps, 2)},
            "disclaimer": DISCLAIMER}


# --------------------------------------------------------------------------- #
# Drop-in endpoint for the Kinetiq React frontend (asd-frontend).
# Same request/response contract as the deployed asd_backend `/api/analyze`, but
# powered by the reconciled model (HC ensemble + MS-G3D fusion + EBM).
# --------------------------------------------------------------------------- #
@app.post("/api/analyze")
async def api_analyze(
    file: UploadFile = File(...),
    file_type: str = Form("video"),
    fuse: bool = Form(True),   # default: paper's weighted MS-G3D fusion. False = HC-primary graded rule.
    age_group: str = Form("child"),  # "adult" -> provisional adult mode (deep stream disabled)
):
    p = _predictor(app)
    if not p.reconciled.available:
        raise HTTPException(500, "Reconciled model is not loaded.")
    safe_name = os.path.basename(file.filename or "upload")
    is_csv = file_type == "csv" or safe_name.lower().endswith(CSV_EXTS)
    path = _save_temp(file, await file.read())
    try:
        if is_csv:
            # A pre-engineered 448-feature export (e.g. *_raw_features.csv) has no
            # raw skeleton -> score HC-only. Otherwise parse as raw coordinates.
            feat = F.load_feature_windows(path)
            if feat is not None:
                X_full, mids = feat
                report = p.analyze_report_features(X_full, mids, source_kind="csv-features")
            else:
                raw_full, fps = F.parse_skeleton_full(path)
                if raw_full is None:
                    raise HTTPException(400, "Could not parse keypoint CSV (need a "
                                        "coordinate export with index + 75 columns, or a "
                                        "pre-engineered 448-feature CSV).")
                report = p.analyze_report(raw_full, fps=fps, source_kind="csv-coordinates",
                                          fuse=fuse, age_group=age_group)
        else:
            try:
                raw_full, fps, meta = extract_skeleton_full_from_video(path)
            except Exception as exc:
                raise HTTPException(400, f"Pose extraction failed: {exc}")
            det = float(meta.get("detection_rate", 1.0)) if isinstance(meta, dict) else 1.0
            if det < 0.60:
                raise HTTPException(422, f"Low pose-detection quality ({det:.0%} of frames). "
                                    "Use a clearer, full-body, single-person video.")
            report = p.analyze_report(raw_full, fps=fps, source_kind="video",
                                      detection_rate=det, fuse=fuse, age_group=age_group)
    finally:
        os.unlink(path)
    return report


@app.post("/report/video", response_model=ReportResponse)
async def report_video(
    file: UploadFile = File(..., description="A single child's RGB video"),
    engine: str = Query("reconciled", description="'reconciled' (paper arch + phone "
                        "acc), 'kinetiq' (deployed HC), or 'v40'"),
    fuse_msg3d: bool = Query(True),
    window_sec: float = Query(3.0, gt=0, le=30),
    step_sec: float = Query(1.0, gt=0, le=30),
    model_complexity: int = Query(1, ge=0, le=2),
    emb_weight: float | None = Query(None, ge=0.0, le=1.0),
    threshold: float | None = Query(None, ge=0.0, le=1.0),
    align: bool | None = Query(None),
):
    p = _predictor(app)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(422, f"Unsupported extension {ext}; expected a video.")
    path = _save_temp(file, await file.read())
    try:
        raw_full, fps, meta = extract_skeleton_full_from_video(
            path, model_complexity=model_complexity)
    except Exception as exc:
        raise HTTPException(422, f"Pose extraction failed: {exc}")
    finally:
        os.unlink(path)
    if engine == "reconciled" and p.reconciled.available:
        report = p.reconciled_report(raw_full, fps=fps, fuse_msg3d=fuse_msg3d)
    elif engine == "kinetiq" and p.kinetiq.available:
        report = p.kinetiq_report(raw_full, fps=fps, fuse_msg3d=fuse_msg3d)
    else:
        report = p.score_windows(raw_full, fps=fps, window_sec=window_sec,
                                 step_sec=step_sec, align=align, source="mediapipe",
                                 threshold=threshold, emb_weight=emb_weight)
        report["engine"] = "v40"
    return {"name": file.filename, **report, "meta": meta, "disclaimer": DISCLAIMER}
