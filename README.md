# Interpretable Gait Screening for Autism

Code, saved models and figure scripts for the manuscript *Interpretable fusion of
learned and biomechanical skeletal gait representations for autism-related motor
atypicality* (submitted to **Neuroscience Informatics**, 2026), plus the smartphone
screening-app prototype the project grew out of. These are two different systems —
this README keeps them apart.

> **Screening aid, not a diagnostic device.** Outputs indicate movement features
> that differ from typically-developing peers and warrant a closer clinical look —
> never a diagnosis. The clinician remains the decision-maker.

## The paper (Neuroscience Informatics submission)

Does fusing a learned skeletal embedding with biomechanical features help separate
autistic from typically-developing (TD) gait, and can the fused score be explained,
when validation is done at the level of the child rather than the clip?

```
Kinect v2 skeleton (150 frames × 25 joints)
  ├─ Learned      two-stream MS-G3D (SpineBase-centred joints + parent-relative bones),
  │               NTU RGB+D 60 initialisation, 768-D embedding → PCA 150 → XGBoost / LightGBM / CatBoost
  └─ Handcrafted  534 biomechanical features (joint angles, torso-normalised distances,
                  left–right asymmetry, Kendall rank correlations) → fold-wise filtering
                  → XGBoost / LightGBM / CatBoost
        child-level probabilities → weighted decision-level fusion → threshold 0.5
                                  └→ distilled EBM (39 main effects + 10 pairwise terms) explains the fused probability
```

**Data and protocol.** The public Al-Jubouri Kinect v2 gait dataset (Dryad
`10.5061/dryad.s7h44j150`): 100 children (50 ASD / 50 TD), 800 pre-generated
150-frame clips (about eight views per child, including augmentation-derived views).
All views of a child stay in the same partition. Evaluation is stratified,
child-grouped 5-fold cross-validation repeated over seeds 42, 123 and 2024 (15
fold-runs). Every data-dependent step is fitted on the training children of each
fold; only the fusion-weight search uses pooled out-of-fold predictions.

**Results** (child level, threshold 0.5; SD is the population SD across seeds):

| | seed 42 | seed 123 | seed 2024 | mean ± SD |
|---|---|---|---|---|
| Handcrafted stream, accuracy (%) | 87.0 | 82.0 | 85.0 | 84.7 ± 2.1 |
| MS-G3D stream, accuracy (%) | 94.0 | 91.0 | 93.0 | 92.7 ± 1.2 |
| **Fusion, accuracy (%)** | 95.0 | 92.0 | 93.0 | **93.3 ± 1.2** |
| **Fusion, AUC** | 0.982 | 0.970 | 0.986 | **0.979 ± 0.007** |

- The learned stream carries most of the signal: MS-G3D was more accurate than the
  handcrafted stream in 12 of 15 matched fold-runs; fusion added about 0.7 points.
- A three-pass out-of-fold ensemble reached 95.0 % accuracy (AUC 0.986; 49/50 TD and
  46/50 ASD correct), against 92.0 % (AUC 0.981) with a GCN+Transformer backbone in
  the same pipeline. This ensemble is reported separately from the run means; the
  manuscript states exactly how it is formed (streams averaged across seeds, then the
  mean learned-stream weight 0.4667 applied).
- The EBM reproduces the fused probability with Pearson r = 0.959, measured on the
  same children used to distil it (in-sample).

**How to read these numbers.** They are internal development evidence, not a
screening-performance claim. The fusion weight was selected on the same pooled
out-of-fold predictions that score it (not nested); held-out views are augmented and
view-averaged rather than single raw recordings; EBM fidelity is in-sample; and there
is no external-device, clinical-control or smartphone validation in the paper.

### Reproducing and inspecting the paper's artefacts

- **Saved model and fold summaries:** `v40_artifacts/asd_gait_v40_3seed.joblib`,
  `v40_artifacts/v40_3seed_summary.json`.
- **Figures 1–5 and the graphical abstract:** `colab/make_figs_nsi_v2.py` redraws them
  from `v40_artifacts/v40_3seed_summary.json` and `colab/nsi_model_facts.json`; the
  latter (model parameters, EBM terms, PCA variance) is produced from the saved model
  by `colab/inspect_nsi_model.py`.
- **Not included:** the v40 fine-tuning / cross-validation training script, the
  original participant-level out-of-fold predictions and the split manifests. The
  paper's headline numbers are therefore read from the saved summary rather than
  re-derived here, and cannot be independently reconstructed from this repository.
- **Dataset:** not redistributed; download it from Dryad (link above).

## Screening-app prototype (not part of the paper)

A separate, earlier system built for smartphone video. Its angular MS-G3D stream is
trained from scratch on unit-bone direction vectors and is **not** the pretrained
embedding pipeline evaluated in the paper; the paper lists validating transfer to
independent smartphone recordings as future work.

### Model

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

**Development-stage checks (grouped by child)**

| | AUC | accuracy |
|---|---|---|
| In-domain (Kinect, grouped 10-fold) | 0.980 | 92% |
| Real phone captures | 0.979 | 15/16 |

These are prototype checks on small samples. The phone row is not an independent,
frozen-model validation and is not reported in the manuscript.

`age_group=adult` engages a **provisional** adult mode: the movement model is
child-trained, so adult mode screens on handcrafted biomechanics only and is not
validated for detecting atypical adults.

### Layout

| path | what |
|---|---|
| `backend/` | FastAPI inference — fusion, angular MS-G3D, reconciled HC, clinical detail, EBM |
| `frontend/` | React (Vite) app — upload, child/adult toggle, OT report |
| `ms-g3d/` | MS-G3D graph-network code (Liu et al., 2020) |
| `colab/` | angular-stream trainer (`train_msg3d_colab.py`), experiment scripts, and the paper's figure scripts |
| `v40_artifacts/` | saved v40 model + summaries used by the paper, and the app's weights and references |

Raw gait datasets are **not** included — the deployment cohort is identifiable
paediatric data.

### Run

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

`POST /api/analyze` — `file` (video or coordinate CSV), `file_type`, `fuse`,
`age_group`. Returns the screening flag, calibrated fused score, per-region
attribution, kinematic findings, and quality/borderline notes.

## Citation

S. Puppala, G. Gorthi, S. A. Hasib, G. P. V. R. Tej, V. Aashna, V. S. Kalyan —
*Interpretable fusion of learned and biomechanical skeletal gait representations for
autism-related motor atypicality* (manuscript submitted to Neuroscience Informatics,
2026).

Dataset: A. Al-Jubouri, I. Hadi, Y. Rajihy, *Three dimensional dataset combining gait
and full body movement of children with autism spectrum disorders collected by Kinect
v2 camera*, Dryad, 2020. https://doi.org/10.5061/dryad.s7h44j150
