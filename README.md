# Interpretable Gait Screening for Autism

An interpretable, cross-device screening pipeline that reads 3D skeletal gait and
returns a clinician-facing, occupational-therapy-oriented report. Two invariant
streams are fused — a handcrafted biomechanical ensemble and an angular MS-G3D
graph network — so the same model holds across the Kinect → phone (MediaPipe)
domain gap.

> **Screening aid, not a diagnostic device.** Outputs indicate movement features
> that differ from typically-developing peers and warrant a closer clinical look —
> never a diagnosis. The clinician remains the decision-maker.

## Paper version (Neuroscience Informatics submission, 2026)

The manuscript *Interpretable fusion of learned and biomechanical skeletal gait
representations for autism-related motor atypicality* evaluates the **v40 model**,
which differs from the deployed app described further below:

- **Learned stream:** two-stream MS-G3D (SpineBase-centred joints + parent-relative
  bones), NTU RGB+D 60 initialisation, 768-D embedding → PCA 150 → XGBoost / LightGBM /
  CatBoost.
- **Handcrafted stream:** 534 biomechanical features (joint angles, torso-normalised
  distances, Kendall correlations, left–right asymmetry) → fold-wise filtering →
  XGBoost / LightGBM / CatBoost.
- **Fusion:** decision-level fusion with a per-seed weight; a distilled EBM explains the
  fused probability.
- **Validation:** child-grouped 5-fold CV × 3 seeds on the Al-Jubouri Kinect v2
  dataset (50 ASD / 50 TD): 93.3 ± 1.2 % accuracy, AUC 0.979 ± 0.007. No cross-device
  or phone results are part of the paper.
- **Saved models and fold summaries:** `v40_artifacts/asd_gait_v40_3seed.joblib`,
  `v40_artifacts/v40_3seed_summary.json`.
- **Paper figures:** `colab/make_figs_nsi_v2.py` regenerates Figures 1–5 and the
  graphical abstract from the saved artifacts (`v40_artifacts/v40_3seed_summary.json`
  and `colab/nsi_model_facts.json`, the latter produced from the saved model by
  `colab/inspect_nsi_model.py`).

The angular stream, phone calibration and phone results in the sections below
describe the deployed screening app, not the paper.

## Model (deployed app)

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

Puppala, Gorthi, Hasib, Tej, Aashna, Kalyan — *Interpretable fusion of learned and
biomechanical skeletal gait representations for autism-related motor atypicality*
(manuscript submitted to Neuroscience Informatics, 2026).
