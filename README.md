# Interpretable Gait Screening for Autism

An interpretable, cross-device screening pipeline that reads 3D skeletal gait and
returns a clinician-facing, occupational-therapy-oriented report. Two invariant
streams are fused — a handcrafted biomechanical ensemble and an angular MS-G3D
graph network — so the same model holds across the Kinect → phone (MediaPipe)
domain gap.

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

Both streams are scale/coordinate-invariant, which is what lets the model transfer
off Kinect. The angular score is Platt-calibrated (fit on MediaPipe pose) so the
displayed movement-model number is honest rather than inflated. A borderline band
around the operating point flags low-confidence cases for review.

**Validation (out-of-sample, grouped by child)**

| | AUC | accuracy |
|---|---|---|
| In-domain (Kinect, grouped 10-fold) | 0.980 | 92% |
| Held-out real phone captures | 0.979 | 15/16 |

`age_group=adult` engages a **provisional** adult mode: the movement model is
child-trained, so adult mode screens on handcrafted biomechanics only and is not
validated for detecting atypical adults.

## Layout

| path | what |
|---|---|
| `backend/` | FastAPI inference — fusion, angular MS-G3D, reconciled HC, clinical detail, EBM |
| `frontend/` | React (Vite) app — upload, child/adult toggle, OT report |
| `ms-g3d/` | MS-G3D graph-network code (Liu et al., 2020) |
| `colab/` | training script for the angular stream (`train_msg3d_colab.py`) |
| `v40_artifacts/` | trained model weights + references |

Raw gait datasets are **not** included — the deployment cohort is identifiable
paediatric data. The public source is the Al-Jubouri Kinect gait dataset
(Dryad `10.5061/dryad.s7h44j150`).

## Run

**Backend** (loads models on startup; `/health` reports ready):
```bash
ASD_ARTIFACT_DIR=v40_artifacts python -m uvicorn backend.app:app --host 127.0.0.1 --port 8172
```

**Frontend**:
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173  (.env → VITE_API_URL=http://127.0.0.1:8172)
```

Upload a video or a 3D-coordinate CSV and pick Child / Adult. The report returns
movement-domain attribution, measured kinematic findings with plain-language
interpretation, OT focus areas, and a progress-monitoring table.

### Main endpoint

`POST /api/analyze` — `file` (video or coordinate CSV), `file_type`, `fuse`,
`age_group`. Returns the screening flag, calibrated fused score, per-region
attribution, kinematic findings, and quality/borderline notes.

## Reference

Puppala, Hasib, Aashna, Tej — *Interpretable Multilevel Fusion of Skeletal Gait
Signals for Autism Screening* (manuscript).
