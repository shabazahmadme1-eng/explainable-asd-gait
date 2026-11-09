# Paper trainers

Both scripts are the exact code that was run for the paper, written for Google Colab
with a GPU. They are kept unchanged, so their header comments still use the
project's internal version names (v40 and v41).

| script | what it trains | writes to |
|---|---|---|
| `model6_msg3d.py` | the paper model: fine-tuned two-stream MS-G3D embedding + 534 biomechanical features, three tree ensembles per stream, decision-level fusion, distilled EBM, under child-grouped 5-fold CV × seeds 42 / 123 / 2024 | `OUTPUT_DIR` = `.../v40_artifacts` (the files in [`../v40_artifacts/`](../v40_artifacts/)) |
| `model5_gcnt.py` | the GCN+Transformer backbone in the same pipeline, for the backbone comparison | `OUTPUT_DIR` = `.../v41_paper_artifacts` (not included here) |

## Running them

1. Download the Al-Jubouri Kinect v2 dataset from Dryad
   (https://doi.org/10.5061/dryad.s7h44j150).
2. Point `DATA_DIR` at the dataset folder and `OUTPUT_DIR` at where results should
   go. Both are set near the top of each script and default to the authors' Google
   Drive paths (`/content/drive/MyDrive/ASD_Project/...`).
3. Run on a GPU, for example `!python model6_msg3d.py` in a Colab cell. The scripts
   install their own Python packages, and `model6_msg3d.py` clones the official
   MS-G3D code (https://github.com/kenziyuliu/ms-g3d) itself.

The default paths assume Google Drive is already mounted at `/content/drive` (mount it
in an earlier cell). Outside Colab, set the two paths to local folders.
