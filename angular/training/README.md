# Angular model training — two invariant streams (HC + angular MS-G3D)

Training and experiment scripts for the **angular MS-G3D smartphone prototype**. This
is not the model in the paper; the paper's trainer is in
[`../../paper/training/`](../../paper/training/).

The deployed model is **`p = 0.5·P_HC + 0.5·P_angularMSG3D`**, flagged at 0.5.
Both streams are scale/coordinate-invariant, so the *same* model holds across the
Kinect→phone domain gap. ShuffleNet-Skepxels was dropped: excellent in-domain
(0.983) but Kinect-only — it saturates on phone, and three rescues (recalibration,
angular input, MediaPipe fine-tuning) all failed.

## Truthful numbers (grouped 10-fold by child / held-out cohort)
| | AUC | acc | sens | spec |
|---|---|---|---|---|
| **In-domain** (Kinect dataset) | 0.980 | 92% | 47/50 | 45/50 |
| **Phone** (14 held-out real captures) | 0.979 | 13/14 | 7/8 | 6/6 |
| — HC alone | 0.946 | 85% | | |
| — angular MS-G3D alone | 0.972 | 91% | | |

## Files here
In the repo:
| file | what |
|---|---|
| `train_msg3d_colab.py` | trains the angular MS-G3D stream (grouped 10-fold, from scratch, leakage-free) |
| `train_msg3d_ensemble.py` | MS-G3D ensemble / DANN experiment script |
| `train_skepxels_dann.py` | Skepxels DANN experiment script |
| `skepxels_indomain_3stream.py` | in-domain 3-stream Skepxels experiment |
| `test_dann_field_local.py` | local DANN field check |

Local-only (not committed — data, weights and run outputs):
| file | what |
|---|---|
| `dataset_800.npz` | 800 Kinect clips (100 kids × orig+7 aug), `X150`/`y`/`groups`/`clip_ids` |
| `res_val_14.npz` | 14 held-out real phone kids (6 TD / 8 ASD) — the phone validation set |
| `field_hc_ang_14.npz` | HC and angular probabilities for those 14 kids (input to the DANN field checks) |
| `mp_train_99.npz` | 99 MediaPipe skeletons from the dataset videos (used for the rescue experiment) |
| `msg3d_pkg.zip` | the MS-G3D code packaged for DGX/Colab (locally the scripts can use `../ms-g3d/` instead) |
| `final_msg3d_bone.pt` | deployed angular MS-G3D weights (also staged at `../artifacts/msg3d_angular.pt`) |
| `final_skepxels_dann.pt` | Skepxels DANN weights from `train_skepxels_dann.py` |
| `oof_msg3d.npy` | angular MS-G3D out-of-fold P(ASD) per clip |
| `hc_oof_800.npy` | handcrafted-ensemble out-of-fold P(ASD) per clip |

## Retrain the deep stream
On a GPU (DGX/Colab), with `dataset_800.npz` + `msg3d_pkg.zip` alongside (or run
from this folder, where `../ms-g3d/` is found automatically):
```
python train_msg3d_colab.py      # -> oof_msg3d.npy + final_msg3d_*.pt
```
Then copy `final_msg3d_bone.pt` → `../artifacts/msg3d_angular.pt` (the backend loads it there).

## Where it's wired
Backend: `../backend/angular_msg3d.py` (loads the classifier, invariant preprocessing) +
`../backend/inference.py` `analyze_report` (the 0.5/0.5 fusion). Frontend contract:
`POST /api/analyze`.
