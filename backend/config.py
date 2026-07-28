"""
Static configuration + constants for the ASD-GAIT v40 inference backend.

Everything here is copied verbatim from the training script (model6_msg3d.py)
so that feature ordering, joint indices and the NTU re-order columns line up
*exactly* with how the deployed artifacts were produced. Do not "tidy" these
lists — their order is load-bearing.
"""
from __future__ import annotations

import math
import os
import re

# --------------------------------------------------------------------------- #
# Paths (override via environment variables if you move things around)
# --------------------------------------------------------------------------- #
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)

ARTIFACT_DIR = os.environ.get(
    "ASD_ARTIFACT_DIR", os.path.join(PROJECT_DIR, "v40_artifacts")
)
# The 3-seed bundle is the headline model; single-seed is available as a fallback.
JOBLIB_PATH = os.environ.get(
    "ASD_JOBLIB_PATH", os.path.join(ARTIFACT_DIR, "asd_gait_v40_3seed.joblib")
)
PTH_PATH = os.environ.get(
    "ASD_PTH_PATH", os.path.join(ARTIFACT_DIR, "msg3d_v40.pth")
)
# Local clone of https://github.com/kenziyuliu/ms-g3d (provides model.msg3d.Model)
MSG3D_REPO = os.environ.get("ASD_MSG3D_REPO", os.path.join(PROJECT_DIR, "ms-g3d"))

# Angular MS-G3D classifier (paper Stream-2 on invariant unit-bone features).
# Trained from scratch, grouped 10-fold. This is the deep stream that TRANSFERS to
# phone/MediaPipe (unlike the raw-coordinate embedding + Skepxels, which are Kinect
# only). Deployed model = 0.5*P_HC + 0.5*P_angular. In-domain AUC 0.980 / 92%;
# 14 held-out real phone kids AUC 0.979 / 13-of-14 / spec 6-of-6.
MSG3D_ANGULAR_PATH = os.environ.get(
    "ASD_MSG3D_ANGULAR_PATH", os.path.join(ARTIFACT_DIR, "msg3d_angular.pt"))

# --- Domain alignment (MediaPipe -> Kinect) for the embedding stream ---
# MediaPipe reference built by scripts/build_mp_reference.py (per-channel skeleton
# stats). The Kinect target is read from the model's data_bn buffers at load time.
MP_REFERENCE_PATH = os.environ.get(
    "ASD_MP_REFERENCE", os.path.join(ARTIFACT_DIR, "mp_skeleton_reference.npz"))
# How much MediaPipe variance is mapped onto Kinect variance (1.0 = full match).
DOMAIN_ALIGN_SHRINK = float(os.environ.get("ASD_ALIGN_SHRINK", "0.5"))
# Alignment is an EXPERIMENTAL add-on (not part of the paper); OFF by default so
# the default pipeline is the paper's raw MS-G3D + HC fusion. Enable per request
# (?align=true) or globally (ASD_ALIGN_DEFAULT=1) to experiment.
DOMAIN_ALIGN_DEFAULT = os.environ.get("ASD_ALIGN_DEFAULT", "0") not in ("0", "false", "False")

# --- Fusion weighting --------------------------------------------------------
# Default is the paper's full multilevel fusion: p = w*p_EMB + (1-w)*p_HC with w
# from the artifact (~0.47). Override the embedding weight via ASD_EMB_WEIGHT
# (e.g. 0.0 = HC-only "robust" mode for degraded input, which also skips the
# MS-G3D forward pass). None => use the artifact's optimised weight.
_emb_w = os.environ.get("ASD_EMB_WEIGHT")
EMB_WEIGHT_OVERRIDE = float(_emb_w) if _emb_w not in (None, "") else None

# Decision threshold on P(ASD) for the reported label. The fused model is
# calibrated near 0.5; a HC-leaning operating point sits lower (asd_backend used
# ~0.31 for MediaPipe). Tune against labelled data for your operating point.
DECISION_THRESHOLD = float(os.environ.get("ASD_DECISION_THRESHOLD", "0.5"))

# --- OOD input-quality gate (paper's Quantum Pose input-quality gate) ---------
# The embedding is standardised against the Kinect distribution (scaler_emb). We
# flag a clip when its embedding sits this many sigma out (rms z); in-distribution
# Kinect clips are ~1.0, MediaPipe/phone clips measured ~4.3. A warning means the
# learned stream is extrapolating and the result should lean on HC / be reviewed.
OOD_WARN_Z = float(os.environ.get("ASD_OOD_WARN_Z", "2.5"))

# --------------------------------------------------------------------------- #
# Model / skeleton geometry
# --------------------------------------------------------------------------- #
NUM_JOINTS = 25
CENTER_JOINT = 0                 # SpineBase in NTU order — used to center the skeleton
TARGET_FRAMES = 150
USE_BONE_STREAM = True
EMB_DIM = 768                    # stream1(384) + stream2(384) after fc->Identity

NTU60_NUM_CLASSES = 60
NTU60_NUM_GCN_SCALES = 13
NTU60_NUM_G3D_SCALES = 6

# Bone graph (child -> parent) in NTU order.
NTU_PARENTS = [0, 0, 24, 2, 24, 4, 5, 6, 7, 6, 24, 10, 11, 12, 13, 12,
               0, 16, 17, 18, 0, 20, 21, 22, 1]
assert len(NTU_PARENTS) == NUM_JOINTS

# --------------------------------------------------------------------------- #
# CSV parsing helpers
# --------------------------------------------------------------------------- #
INTERMEDIATE_FOLDERS = frozenset({
    "children with ASD", "Severe level of ASD",
    "Mild", "Moderate", "Severe",
    "Level 1", "Level 2", "Level 3",
})
CLIP_FILENAME_RE = re.compile(r"^\d+$")

# --------------------------------------------------------------------------- #
# Joint column maps (original CSV layout -> NTU order)
# --------------------------------------------------------------------------- #
# Column offset of each joint's (x,y,z) triple inside the 75-wide coord block.
USER_JOINT_COL = {
    'SpineMid': 0, 'AnkleLeft': 3, 'AnkleRight': 6, 'ElbowLeft': 9,
    'ElbowRight': 12, 'FootLeft': 15, 'FootRight': 18, 'HandLeft': 21,
    'HandRight': 24, 'HandTipLeft': 27, 'HandTipRight': 30, 'Head': 33,
    'HipLeft': 36, 'HipRight': 39, 'KneeLeft': 42, 'KneeRight': 45,
    'Neck': 48, 'ShoulderLeft': 51, 'ShoulderRight': 54, 'SpineBase': 57,
    'SpineShoulder': 60, 'ThumbLeft': 63, 'ThumbRight': 66, 'WristLeft': 69,
    'WristRight': 72,
}
NTU_JOINT_NAMES = [
    'SpineBase', 'SpineMid', 'Neck', 'Head', 'ShoulderLeft', 'ElbowLeft',
    'WristLeft', 'HandLeft', 'HandTipLeft', 'ThumbLeft', 'ShoulderRight',
    'ElbowRight', 'WristRight', 'HandRight', 'HandTipRight', 'ThumbRight',
    'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft', 'HipRight', 'KneeRight',
    'AnkleRight', 'FootRight', 'SpineShoulder',
]
# Column indices that re-order the raw 75-wide block into NTU joint order.
NTU_REORDER_COLS: list[int] = []
for _name in NTU_JOINT_NAMES:
    _cs = USER_JOINT_COL[_name]
    NTU_REORDER_COLS.extend([_cs, _cs + 1, _cs + 2])

# Left/right joint pairs (indices into NTU order) for asymmetry features.
LR_JOINT_PAIRS = [(4, 10), (5, 11), (6, 12), (7, 13), (8, 14), (9, 15),
                  (16, 20), (17, 21), (18, 22), (19, 23)]
LR_PAIR_NAMES = ['Shoulder', 'Elbow', 'Wrist', 'Hand', 'HandTip', 'Thumb',
                 'Hip', 'Knee', 'Ankle', 'Foot']

# --------------------------------------------------------------------------- #
# Handcrafted feature names (534 total) — order must match training exactly.
# --------------------------------------------------------------------------- #
BIO_SIGNAL_NAMES = [
    'L_elbow_angle', 'R_elbow_angle', 'L_hip_angle', 'R_hip_angle',
    'L_shoulder_angle', 'R_shoulder_angle', 'L_hipext_angle', 'R_hipext_angle',
    'wrist_dist_norm', 'ankle_dist_norm', 'head_Lwrist_dist_norm',
    'head_Rwrist_dist_norm',
]
STAT_SUFFIXES = ['mean', 'std', 'mean_vel', 'max_vel', 'mean_acc', 'max_acc']


def _build_hc_feature_names() -> list[str]:
    names: list[str] = []
    for t in range(30):
        for sig in BIO_SIGNAL_NAMES:
            names.append(f'bio_{sig}_t{t}')
    for stat in STAT_SUFFIXES:
        for sig in BIO_SIGNAL_NAMES:
            names.append(f'{stat}_{sig}')
    for sig in BIO_SIGNAL_NAMES:
        names.append(f'tau_time_{sig}')
    names.extend(['tau_sym_elbow', 'tau_sym_hip', 'tau_sym_shoulder',
                  'tau_sym_hipext'])
    for pn in LR_PAIR_NAMES:
        names.append(f'angle_asym_{pn}')
        names.append(f'dist_asym_{pn}')
        names.append(f'motion_asym_{pn}')
    for jn in NTU_JOINT_NAMES:
        names.append(f'spine_angle_mean_{jn}')
        names.append(f'spine_angle_std_{jn}')
    names.extend(['frame_asym_angle_mean', 'frame_asym_angle_std',
                  'frame_asym_angle_max', 'frame_asym_dist_mean',
                  'frame_asym_dist_std', 'frame_asym_dist_max'])
    return names


HC_FEATURE_NAMES = _build_hc_feature_names()
assert len(HC_FEATURE_NAMES) == 534, len(HC_FEATURE_NAMES)

# Geometric clip range applied during parsing.
COORD_CLIP = 10.0
