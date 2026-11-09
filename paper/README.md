# The paper model (Neuroscience Informatics submission)

Everything for the manuscript *Interpretable fusion of learned and biomechanical
skeletal gait representations for autism-related motor atypicality* (submitted to
**Neuroscience Informatics**, 2026). The smartphone prototype is a different system
and lives in [`../angular/`](../angular/).

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

## Results

Child level, threshold 0.5; SD is the population SD across seeds.

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

## Layout

| path | what |
|---|---|
| `training/model6_msg3d.py` | the v40 three-seed trainer that produced the saved model and summaries below |
| `training/model5_gcnt.py` | the GCN+Transformer backbone run in the same pipeline (comparison model) |
| `v40_artifacts/asd_gait_v40_3seed.joblib` | the saved paper model: PCA, scalers, both streams' tree ensembles, fusion weights, EBM, per-fold details |
| `v40_artifacts/msg3d_v40.pth` | the fine-tuned two-stream MS-G3D used to compute the embedding |
| `v40_artifacts/v40_3seed_summary.json` | the run's summary (per-fold and per-seed results), readable without Python |
| `v40_artifacts/asd_gait_v40.joblib`, `v40_summary.json` | an earlier single-seed run of the same pipeline; not reported in the paper |
| `figures/make_figs_nsi_v2.py` | redraws Figures 1–5 and the graphical abstract from `v40_3seed_summary.json` and `figures/nsi_model_facts.json` |
| `figures/inspect_nsi_model.py` | produces `figures/nsi_model_facts.json` (model parameters, EBM terms, PCA variance) from the saved model |

See [`training/README.md`](training/README.md) for how to run the trainers.

**Not included:** the run's participant-level out-of-fold predictions. The trainer
computes them but does not save them, so the headline numbers above are read from the
saved summary rather than re-derived here. Re-running the trainer repeats the whole
experiment; GPU fine-tuning is not bit-for-bit deterministic, so expect small
differences. The dataset itself is not redistributed; download it from Dryad.
