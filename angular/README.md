# Angular MS-G3D screening prototype (not part of the paper)

A smartphone-video gait screening prototype and the app that serves it. Its angular
MS-G3D stream is trained from scratch on unit-bone direction vectors and is **not**
the pretrained embedding pipeline evaluated in the paper (see
[`../paper/`](../paper/)); the paper lists validating transfer to independent
smartphone recordings as future work.

> **Screening aid, not a diagnostic device.** Outputs indicate movement features
> that differ from typically-developing peers and warrant a closer clinical look —
> never a diagnosis. The clinician remains the decision-maker.

## Model

```
skeleton (T × 75)
  ├─ Handcrafted    — 448 windowed spatiotemporal features → MW-376 → XGB+LGB+CatBoost
  └─ Angular MS-G3D — unit-bone (direction-cosine) vectors → multi-scale graph conv
                        p = 0.5·P_hc + 0.5·P_angular  →  distilled EBM explanation
```

Both streams are scale/coordinate-invariant, which is what the prototype relies on to
transfer off Kinect. The angular score is Platt-calibrated (fit on MediaPipe pose) so
the displayed movement-model number is not inflated. A borderline band around the
operating point flags low-confidence cases for review.

**Development-stage check (grouped by child)**

| | AUC | accuracy |
|---|---|---|
| In-domain (Kinect, grouped 10-fold) | 0.980 | 92% |

This is a prototype check on a small sample and is not reported in the manuscript.
The prototype has no independent smartphone validation.

`age_group=adult` engages a **provisional** adult mode: the movement model is
child-trained, so adult mode screens on handcrafted biomechanics only and is not
validated for detecting atypical adults.

## Layout

| path | what |
|---|---|
| `backend/` | FastAPI inference — fusion, angular MS-G3D, reconciled HC, clinical detail, EBM |
| `frontend/` | React (Vite) app — upload, child/adult toggle, OT report |
| `ms-g3d/` | MS-G3D graph-network code (Liu et al., 2020) |
| `training/` | angular-stream trainer (`train_msg3d_colab.py`) and experiment scripts — see [`training/README.md`](training/README.md) |
| `scripts/` | `build_mp_reference.py` — builds the MediaPipe skeleton reference |
| `artifacts/` | the weights and references the backend loads (below) |

`artifacts/`:

| file | what |
|---|---|
| `msg3d_angular.pt` | angular MS-G3D weights (the deployed deep stream) |
| `asd_gait_reconciled.joblib` | handcrafted 3-model ensemble on phone-transferable features + its distilled EBM |
| `mp_skeleton_reference.npz`, `clinical_refs/` | MediaPipe skeleton reference and typical-development reference statistics for the report |
| `asd_gait_v40_3seed.joblib`, `msg3d_v40.pth` | copies of the paper's v40 model. The backend loads them at start-up for the older `/predict/*` and `/report/*` endpoints; the deployed `/api/analyze` path does not use them. |

Raw gait datasets are **not** included — the phone cohort is identifiable paediatric
data.

## Run

**Backend** (from this `angular/` folder; loads models on startup, `/health` reports ready):
```bash
pip install -r requirements.txt
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8172
```
Weights are read from `artifacts/`; set `ASD_ARTIFACT_DIR` to use another folder.

**Frontend**:
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173  (.env → VITE_API_URL=http://127.0.0.1:8172)
```

Upload a video or a 3D-coordinate CSV and pick Child / Adult. The report returns
movement-domain attribution, measured kinematic findings with plain-language
interpretation, OT focus areas, and a progress-monitoring table.

`POST /api/analyze` — `file` (video or coordinate CSV), `file_type`, `fuse`,
`age_group`. Returns the screening flag, calibrated fused score, per-region
attribution, kinematic findings, and quality/borderline notes.
