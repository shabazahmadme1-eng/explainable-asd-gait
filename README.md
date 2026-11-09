# Interpretable Gait Screening for Autism

This repository holds two separate systems built on the same skeletal-gait problem.
Each lives in its own folder with its own code, weights and instructions. They are
**not** interchangeable, and results from one should never be quoted for the other.

| folder | what it is |
|---|---|
| [`paper/`](paper/) | The model evaluated in the manuscript *Interpretable fusion of learned and biomechanical skeletal gait representations for autism-related motor atypicality* (submitted to **Neuroscience Informatics**, 2026): the v40 pretrained two-stream MS-G3D embedding plus 534 biomechanical features, decision-level fusion and a distilled EBM. Trainers, saved model, fold summaries and figure scripts. |
| [`angular/`](angular/) | The angular MS-G3D smartphone screening prototype the project grew out of: an MS-G3D trained from scratch on unit-bone direction vectors, fused with a handcrafted stream, plus the FastAPI backend and React app that serve it. **Not part of the paper.** |

> **Screening aid, not a diagnostic device.** Outputs indicate movement features
> that differ from typically-developing peers and warrant a closer clinical look —
> never a diagnosis. The clinician remains the decision-maker.

## Data

Both systems are developed on the public Al-Jubouri Kinect v2 gait dataset (100
children, 50 ASD / 50 TD), which is not redistributed here — download it from Dryad:
https://doi.org/10.5061/dryad.s7h44j150. The prototype's phone recordings are
identifiable paediatric data and are not published.

## Citation

S. Puppala, G. Gorthi, S. A. Hasib, G. P. V. R. Tej, V. Aashna, V. S. Kalyan —
*Interpretable fusion of learned and biomechanical skeletal gait representations for
autism-related motor atypicality* (manuscript submitted to Neuroscience Informatics,
2026).

Dataset: A. Al-Jubouri, I. Hadi, Y. Rajihy, *Three dimensional dataset combining gait
and full body movement of children with autism spectrum disorders collected by Kinect
v2 camera*, Dryad, 2020. https://doi.org/10.5061/dryad.s7h44j150
