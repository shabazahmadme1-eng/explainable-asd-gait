# =============================================================================
# ASD-GAIT v40 TRAIN: Kinect CSVs → RGB-deployable model  [3-SEED EDITION]
# =============================================================================
# This is your v40 with ONE structural addition: the 5-fold CV is now wrapped
# in a 3-seed loop so v40 reports a mean ± std comparable to v41's protocol.
#
# Nothing about the model, fine-tuning, domain randomization, feature
# extraction, ensembles, weight tuning, or EBM was changed. The ONLY edits:
#   • Added SEEDS = [42, 123, 2024].
#   • run_pipeline() split into run_one_seed(seed, ...) + a multi-seed driver.
#   • The seed is threaded through the 4 random_state values that were
#     previously hardcoded (StratifiedKFold=42, ft split=42, PCA=42,
#     inner-ensemble split=fold_num) so each seed yields a genuinely
#     different CV. The model/training code is untouched.
#   • OOF predictions averaged across seeds for the EBM teacher (this is what
#     v41 does too, so the explainability layer is built on the same basis).
#   • Deployment teacher still trained once (seed 42), exactly as before.
#
# Original bridging strategy (unchanged) — three layered defenses against
# Kinect→phone domain shift:
#   1. DOMAIN-RANDOMIZATION AUGMENTATION (z-jitter, joint dropout, temporal blur)
#   2. SHIFTED EMB/HC WEIGHTING (lean on view-invariant HC at deployment)
#   3. DISTILLED EBM WHITEBOX (faithful per-feature explanations of the fusion)
#
# IMPORTANT CAVEAT (unchanged): without paired Kinect+phone data on the same
# children, real-world phone-video accuracy is unknown. Reported CV is
# Kinect→Kinect.
# =============================================================================

import os
os.system('pip install -q xgboost lightgbm catboost openpyxl shap joblib interpret')

import re, math, copy, json, random, shutil, sys, warnings
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from scipy.stats import kendalltau, mannwhitneyu
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     train_test_split, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import ElasticNet, LogisticRegression

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingRegressor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# Clone MS-G3D
OFFICIAL_CODE_PATH = "/content/ms-g3d"
if os.path.exists(OFFICIAL_CODE_PATH):
    shutil.rmtree(OFFICIAL_CODE_PATH)
os.system("git clone https://github.com/kenziyuliu/ms-g3d.git /content/ms-g3d")
if OFFICIAL_CODE_PATH not in sys.path:
    sys.path.append(OFFICIAL_CODE_PATH)
from model.msg3d import Model as OfficialMSG3D


# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR     = "/content/drive/MyDrive/ASD_Project/Kinect_3D_Dataset/Dataset"
WEIGHTS_PATH = "/content/drive/MyDrive/ASD-STGCN/weights/ntu60-xsub-joint-better.pt"
OUTPUT_DIR   = "/content/drive/MyDrive/ASD_Project/v40_artifacts"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [42, 123, 2024]      # ← multi-seed protocol (matches v41)

LR_FINETUNE         = 3e-5
LR_STREAM1_RATIO    = 0.1
LR_CLASSIFIER_RATIO = 5.0
WARMUP_EPOCHS       = 5
NUM_EPOCHS_FINETUNE = 80
FINETUNE_PATIENCE   = 20
FT_VAL_SPLIT        = 0.10
DROPOUT_RATE        = 0.5
BATCH_SIZE          = 32
TARGET_FRAMES       = 150
USE_BONE_STREAM     = True
PCA_COMPONENTS      = 150

# v37u-style augmentation (now joined by domain-randomization below)
AUG_ROTATION_MAX    = math.pi / 6
AUG_JITTER_STD      = 0.02
AUG_TEMPORAL_SHIFT  = 15

# Domain randomization (NEW in v40) — simulates MediaPipe noise on Kinect data
DR_PROB             = 0.5            # probability of applying any DR per sample
DR_Z_NOISE_STD      = 0.04           # z-axis is the noisiest in monocular pose
DR_XY_NOISE_STD     = 0.015          # smaller xy noise
DR_JOINT_DROPOUT    = 0.10           # prob of dropping a joint per sample
DR_TEMPORAL_BLUR_P  = 0.30           # prob of applying temporal-blur smoothing

CORRELATION_THRESHOLD = 0.90
MIN_VARIANCE          = 1e-8
WHITEBOX_MAX_FEATURES = 50

INTERMEDIATE_FOLDERS = frozenset({
    "children with ASD", "Severe level of ASD",
    "Mild", "Moderate", "Severe",
    "Level 1", "Level 2", "Level 3",
})

CLIP_FILENAME_RE = re.compile(r"^\d+$")
NTU60_NUM_CLASSES = 60
NTU60_NUM_GCN_SCALES = 13
NTU60_NUM_G3D_SCALES = 6

NUM_JOINTS = 25
CENTER_JOINT = 0
NTU_PARENTS = [0,0,24,2,24,4,5,6,7,6,24,10,11,12,13,12,0,16,17,18,0,20,21,22,1]
assert len(NTU_PARENTS) == NUM_JOINTS

print(f"Using device: {DEVICE}")


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_bone_data(skeleton_data):
    """skeleton_data shape (C, T, V, M)."""
    parent_indices = np.array(NTU_PARENTS)
    return skeleton_data - skeleton_data[:, :, parent_indices, :]


# =============================================================================
# MODEL (unchanged from v37u)
# =============================================================================
class TwoStreamOfficialMSG3D(nn.Module):
    def __init__(self, num_classes=2, weights_path=WEIGHTS_PATH,
                 use_bone_stream=True, dropout=0.5):
        super().__init__()
        self.use_bone_stream = use_bone_stream
        model_args = dict(
            num_class=NTU60_NUM_CLASSES, num_point=NUM_JOINTS, num_person=1,
            num_gcn_scales=NTU60_NUM_GCN_SCALES, num_g3d_scales=NTU60_NUM_G3D_SCALES,
            graph="graph.ntu_rgb_d.AdjMatrixGraph",
        )
        self.stream1 = OfficialMSG3D(**model_args)
        self._load_official_weights(self.stream1, weights_path)
        self.stream1.fc = nn.Identity()
        if use_bone_stream:
            self.stream2 = OfficialMSG3D(**model_args)
            self._load_official_weights(self.stream2, weights_path)
            self.stream2.fc = nn.Identity()
        with torch.no_grad():
            actual_dim = self.stream1(torch.zeros(1, 3, TARGET_FRAMES, 25, 1)).shape[1]
        self.emb_dim = actual_dim * 2 if use_bone_stream else actual_dim
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.emb_dim, num_classes)

    def _load_official_weights(self, stream, path):
        if not os.path.exists(path):
            print(f"  ⚠️  Weights not found at {path} — random init")
            return
        try:
            state = torch.load(path, map_location=DEVICE, weights_only=True)
        except (TypeError, RuntimeError):
            state = torch.load(path, map_location=DEVICE, weights_only=False)
        clean = {k.replace("module.", ""): v for k, v in state.items()}
        md = stream.state_dict()
        pre = {k: v for k, v in clean.items() if k in md and v.shape == md[k].shape}
        md.update(pre); stream.load_state_dict(md)

    def forward(self, x_j, x_b=None):
        feat1 = self.stream1(x_j)
        if self.use_bone_stream and x_b is not None:
            features = torch.cat([feat1, self.stream2(x_b)], dim=1)
        else:
            features = feat1
        return self.classifier(self.dropout(features)), features

    def get_embeddings(self, x_j, x_b=None):
        with torch.no_grad():
            return self.forward(x_j, x_b)[1]


# =============================================================================
# HANDCRAFTED FEATURES (unchanged from v37u — feature names preserved exactly)
# =============================================================================
USER_JOINT_COL = {'SpineMid':0,'AnkleLeft':3,'AnkleRight':6,'ElbowLeft':9,'ElbowRight':12,'FootLeft':15,'FootRight':18,'HandLeft':21,'HandRight':24,'HandTipLeft':27,'HandTipRight':30,'Head':33,'HipLeft':36,'HipRight':39,'KneeLeft':42,'KneeRight':45,'Neck':48,'ShoulderLeft':51,'ShoulderRight':54,'SpineBase':57,'SpineShoulder':60,'ThumbLeft':63,'ThumbRight':66,'WristLeft':69,'WristRight':72}
NTU_JOINT_NAMES = ['SpineBase','SpineMid','Neck','Head','ShoulderLeft','ElbowLeft','WristLeft','HandLeft','HandTipLeft','ThumbLeft','ShoulderRight','ElbowRight','WristRight','HandRight','HandTipRight','ThumbRight','HipLeft','KneeLeft','AnkleLeft','FootLeft','HipRight','KneeRight','AnkleRight','FootRight','SpineShoulder']
NTU_REORDER_COLS = []
for name in NTU_JOINT_NAMES:
    cs = USER_JOINT_COL[name]; NTU_REORDER_COLS.extend([cs, cs+1, cs+2])
LR_JOINT_PAIRS = [(4,10),(5,11),(6,12),(7,13),(8,14),(9,15),(16,20),(17,21),(18,22),(19,23)]
LR_PAIR_NAMES = ['Shoulder','Elbow','Wrist','Hand','HandTip','Thumb','Hip','Knee','Ankle','Foot']
J_H = USER_JOINT_COL
BIO_SIGNAL_NAMES = ['L_elbow_angle','R_elbow_angle','L_hip_angle','R_hip_angle','L_shoulder_angle','R_shoulder_angle','L_hipext_angle','R_hipext_angle','wrist_dist_norm','ankle_dist_norm','head_Lwrist_dist_norm','head_Rwrist_dist_norm']
STAT_SUFFIXES = ['mean','std','mean_vel','max_vel','mean_acc','max_acc']
HC_FEATURE_NAMES = []
for t in range(30):
    for sig in BIO_SIGNAL_NAMES: HC_FEATURE_NAMES.append(f'bio_{sig}_t{t}')
for stat in STAT_SUFFIXES:
    for sig in BIO_SIGNAL_NAMES: HC_FEATURE_NAMES.append(f'{stat}_{sig}')
for sig in BIO_SIGNAL_NAMES: HC_FEATURE_NAMES.append(f'tau_time_{sig}')
HC_FEATURE_NAMES.extend(['tau_sym_elbow','tau_sym_hip','tau_sym_shoulder','tau_sym_hipext'])
for pn in LR_PAIR_NAMES:
    HC_FEATURE_NAMES.append(f'angle_asym_{pn}')
    HC_FEATURE_NAMES.append(f'dist_asym_{pn}')
    HC_FEATURE_NAMES.append(f'motion_asym_{pn}')
for jn in NTU_JOINT_NAMES:
    HC_FEATURE_NAMES.append(f'spine_angle_mean_{jn}')
    HC_FEATURE_NAMES.append(f'spine_angle_std_{jn}')
HC_FEATURE_NAMES.extend(['frame_asym_angle_mean','frame_asym_angle_std','frame_asym_angle_max','frame_asym_dist_mean','frame_asym_dist_std','frame_asym_dist_max'])


def parse_skeleton_file(filepath):
    try:
        df = pd.read_excel(filepath, engine='openpyxl') if filepath.endswith('.xlsx') else pd.read_csv(filepath)
    except Exception:
        return None, 0
    if df.shape[1] < 76: return None, 0
    coords_df = df.iloc[:, 1:76].apply(pd.to_numeric, errors='coerce')
    for j in range(25):
        cols = [j*3, j*3+1, j*3+2]
        all_zero = (coords_df.iloc[:, cols] == 0.0).all(axis=1)
        if all_zero.any():
            coords_df.iloc[all_zero.values, cols] = np.nan
    coords_df = (coords_df.interpolate(method='linear', limit_direction='both')
                          .rolling(window=3, min_periods=1, center=True).median())
    if len(coords_df) == 0: return None, 0
    if len(coords_df) < TARGET_FRAMES:
        coords_df = pd.concat([coords_df,
                               pd.concat([coords_df.iloc[-1:]]*(TARGET_FRAMES-len(coords_df)))],
                              ignore_index=True)
    elif len(coords_df) > TARGET_FRAMES:
        coords_df = coords_df.iloc[np.linspace(0, len(coords_df)-1, TARGET_FRAMES).astype(int)].reset_index(drop=True)
    raw = np.nan_to_num(coords_df.values, nan=0.0)
    n_clipped = int(np.sum(np.abs(raw) > 10.0))
    return np.clip(raw, -10.0, 10.0), n_clipped


def center_skeleton(data):
    return data - data[:, :, CENTER_JOINT:CENTER_JOINT+1, :]


def get_vec_h(f, j): return f[J_H[j]:J_H[j]+3]


def calc_angle(p1, p2, p3):
    v1, v2 = p1-p2, p3-p2
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return 0.0 if m1 == 0 or m2 == 0 else np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(m1*m2), -1.0, 1.0)))


def calc_dist(p1, p2): return np.linalg.norm(p1-p2)


def extract_handcrafted_features(raw_frames):
    """Identical to v37u — preserved for feature-name stability."""
    T = len(raw_frames)
    frames = raw_frames[np.linspace(0, T-1, 30).astype(int)] if T != 30 else raw_frames
    bio = np.zeros((30, 12))
    for i, f in enumerate(frames):
        tl = max(calc_dist(get_vec_h(f, 'SpineShoulder'), get_vec_h(f, 'SpineBase')), 0.01)
        bio[i] = [
            calc_angle(get_vec_h(f, 'ShoulderLeft'),  get_vec_h(f, 'ElbowLeft'),  get_vec_h(f, 'WristLeft')),
            calc_angle(get_vec_h(f, 'ShoulderRight'), get_vec_h(f, 'ElbowRight'), get_vec_h(f, 'WristRight')),
            calc_angle(get_vec_h(f, 'HipLeft'),       get_vec_h(f, 'KneeLeft'),   get_vec_h(f, 'AnkleLeft')),
            calc_angle(get_vec_h(f, 'HipRight'),      get_vec_h(f, 'KneeRight'),  get_vec_h(f, 'AnkleRight')),
            calc_angle(get_vec_h(f, 'SpineShoulder'), get_vec_h(f, 'ShoulderLeft'),  get_vec_h(f, 'ElbowLeft')),
            calc_angle(get_vec_h(f, 'SpineShoulder'), get_vec_h(f, 'ShoulderRight'), get_vec_h(f, 'ElbowRight')),
            calc_angle(get_vec_h(f, 'SpineBase'),     get_vec_h(f, 'HipLeft'),    get_vec_h(f, 'KneeLeft')),
            calc_angle(get_vec_h(f, 'SpineBase'),     get_vec_h(f, 'HipRight'),   get_vec_h(f, 'KneeRight')),
            calc_dist(get_vec_h(f, 'WristLeft'),  get_vec_h(f, 'WristRight')) / tl,
            calc_dist(get_vec_h(f, 'AnkleLeft'),  get_vec_h(f, 'AnkleRight')) / tl,
            calc_dist(get_vec_h(f, 'Head'), get_vec_h(f, 'WristLeft'))  / tl,
            calc_dist(get_vec_h(f, 'Head'), get_vec_h(f, 'WristRight')) / tl,
        ]
    vel = np.diff(bio, axis=0); acc = np.diff(vel, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau_time = np.zeros(12)
        for c in range(12):
            v, _ = kendalltau(np.arange(30), bio[:, c])
            tau_time[c] = 0.0 if np.isnan(v) else v
        tau_sym = np.nan_to_num(np.array([
            kendalltau(bio[:, 0], bio[:, 1])[0],
            kendalltau(bio[:, 2], bio[:, 3])[0],
            kendalltau(bio[:, 4], bio[:, 5])[0],
            kendalltau(bio[:, 6], bio[:, 7])[0],
        ]), nan=0.0)
        orig = np.concatenate([bio.flatten(),
                               np.nanmean(bio, axis=0), np.nanstd(bio, axis=0),
                               np.nanmean(np.abs(vel), axis=0), np.nanmax(np.abs(vel), axis=0),
                               np.nanmean(np.abs(acc), axis=0), np.nanmax(np.abs(acc), axis=0),
                               tau_time, tau_sym])
    ntu_raw = frames[:, NTU_REORDER_COLS].reshape(30, 25, 3)
    ntu_c = ntu_raw - ntu_raw[:, CENTER_JOINT, :][:, np.newaxis, :]
    js_dist = np.linalg.norm(ntu_c, axis=2)
    spine_unit = (ntu_c[:, 1, :] /
                  np.maximum(np.linalg.norm(ntu_c[:, 1, :], axis=1, keepdims=True), 0.01))[:, np.newaxis, :]
    js_ang = np.degrees(np.arccos(
        np.clip(np.sum(ntu_c * spine_unit, axis=2) / np.maximum(js_dist, 0.01), -1.0, 1.0)))
    j_mot = np.linalg.norm(np.diff(ntu_c, axis=0), axis=2)
    a_asym = [np.abs(np.mean(js_ang[:, l]) - np.mean(js_ang[:, r])) for l, r in LR_JOINT_PAIRS]
    d_asym = [np.abs(np.mean(js_dist[:, l]) - np.mean(js_dist[:, r])) for l, r in LR_JOINT_PAIRS]
    m_asym = [np.abs(np.mean(j_mot[:, l])  - np.mean(j_mot[:, r]))   for l, r in LR_JOINT_PAIRS]
    L_idx = [l for l, r in LR_JOINT_PAIRS]; R_idx = [r for l, r in LR_JOINT_PAIRS]
    fa_ang  = np.abs(js_ang[:, L_idx]  - js_ang[:, R_idx]).mean(axis=1)
    fa_dist = np.abs(js_dist[:, L_idx] - js_dist[:, R_idx]).mean(axis=1)
    return np.nan_to_num(np.concatenate([
        orig, a_asym, d_asym, m_asym,
        np.mean(js_ang, axis=0), np.std(js_ang, axis=0),
        [np.mean(fa_ang), np.std(fa_ang), np.max(fa_ang),
         np.mean(fa_dist), np.std(fa_dist), np.max(fa_dist)],
    ]), nan=0.0)


# =============================================================================
# DOMAIN-RANDOMIZED DATASET (NEW in v40)
# =============================================================================
class SkeletonDataset(Dataset):
    """Adds domain-randomization on top of v37u's geometric augmentations.
    DR is only applied during training (augment=True)."""

    def __init__(self, joint_data, bone_data, labels, augment=False, domain_random=False):
        self.joint_data = joint_data
        self.bone_data = bone_data
        self.labels = np.asarray(labels)
        self.use_bone = bone_data is not None
        self.augment = augment
        self.domain_random = domain_random

    def __len__(self): return len(self.joint_data)

    @staticmethod
    def _temporal_shift(x, shift, axis=1):
        if shift == 0: return x.copy()
        n = x.shape[axis]; out = np.empty_like(x)
        if shift > 0:
            sl1 = [slice(None)]*x.ndim; sl1[axis] = slice(0, shift)
            sl2 = [slice(None)]*x.ndim; sl2[axis] = slice(0, 1)
            out[tuple(sl1)] = np.broadcast_to(x[tuple(sl2)], out[tuple(sl1)].shape)
            sl3 = [slice(None)]*x.ndim; sl3[axis] = slice(shift, None)
            sl4 = [slice(None)]*x.ndim; sl4[axis] = slice(0, n-shift)
            out[tuple(sl3)] = x[tuple(sl4)]
        else:
            sa = -shift
            sl1 = [slice(None)]*x.ndim; sl1[axis] = slice(n-sa, None)
            sl2 = [slice(None)]*x.ndim; sl2[axis] = slice(-1, None)
            out[tuple(sl1)] = np.broadcast_to(x[tuple(sl2)], out[tuple(sl1)].shape)
            sl3 = [slice(None)]*x.ndim; sl3[axis] = slice(0, n-sa)
            sl4 = [slice(None)]*x.ndim; sl4[axis] = slice(sa, None)
            out[tuple(sl3)] = x[tuple(sl4)]
        return out

    @staticmethod
    def _temporal_blur(x, sigma=1.5):
        """Simulates MediaPipe's frame-to-frame jitter being smoothed by tracking."""
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(x, sigma=sigma, axis=1).astype(np.float32)

    def _domain_randomize(self, x_j, x_b):
        """Simulate MediaPipe-style noise on top of clean Kinect data.

        Effects in order:
          1. Anisotropic noise — z-axis (depth) is much noisier in monocular
             pose estimation than xy.
          2. Random joint dropout — low-visibility joints get masked and
             interpolated, similar to what extract_pose_from_video does.
          3. Temporal blur — tracking smoothing introduces low-pass artifacts.
        """
        # Anisotropic per-coord noise
        if random.random() < DR_PROB:
            noise = np.zeros_like(x_j)
            noise[0] = np.random.normal(0, DR_XY_NOISE_STD, x_j.shape[1:]).astype(np.float32)
            noise[1] = np.random.normal(0, DR_XY_NOISE_STD, x_j.shape[1:]).astype(np.float32)
            noise[2] = np.random.normal(0, DR_Z_NOISE_STD,  x_j.shape[1:]).astype(np.float32)
            x_j = x_j + noise
            if x_b is not None:
                # Bones inherit noise via differencing — recompute for consistency
                x_b = compute_bone_data(x_j)

        # Joint dropout — set joint position to nearest-neighbor in time, simulating
        # the linear interpolation MediaPipe does for occluded joints.
        if random.random() < DR_PROB:
            for v in range(NUM_JOINTS):
                if random.random() < DR_JOINT_DROPOUT:
                    # pick random consecutive frames to "drop"
                    T = x_j.shape[1]
                    drop_len = random.randint(2, max(2, T // 10))
                    drop_start = random.randint(0, T - drop_len)
                    # replace with the value at drop_start (held constant)
                    held = x_j[:, drop_start:drop_start+1, v:v+1, :]
                    x_j[:, drop_start:drop_start+drop_len, v:v+1, :] = held
            if x_b is not None:
                x_b = compute_bone_data(x_j)

        # Temporal blur
        if random.random() < DR_TEMPORAL_BLUR_P:
            x_j = self._temporal_blur(x_j, sigma=random.uniform(0.8, 1.8))
            if x_b is not None:
                x_b = compute_bone_data(x_j)

        return x_j, x_b

    def _augment(self, x_j, x_b):
        # Geometric augmentations (v37u-style)
        if random.random() < 0.5:
            a = random.uniform(-AUG_ROTATION_MAX, AUG_ROTATION_MAX)
            R = np.array([[math.cos(a), 0, math.sin(a)],
                          [0, 1, 0],
                          [-math.sin(a), 0, math.cos(a)]], dtype=np.float32)
            x_j = np.einsum('ij,jtvm->itvm', R, x_j)
            if x_b is not None:
                x_b = np.einsum('ij,jtvm->itvm', R, x_b)
        if random.random() < 0.5:
            x_j = x_j + np.random.normal(0, AUG_JITTER_STD, x_j.shape).astype(np.float32)
            if x_b is not None:
                x_b = x_b + np.random.normal(0, AUG_JITTER_STD*0.25, x_b.shape).astype(np.float32)
        if random.random() < 0.3:
            s = random.randint(-AUG_TEMPORAL_SHIFT, AUG_TEMPORAL_SHIFT)
            if s != 0:
                x_j = self._temporal_shift(x_j, s, axis=1)
                if x_b is not None:
                    x_b = self._temporal_shift(x_b, s, axis=1)

        # Domain randomization (v40 NEW)
        if self.domain_random:
            x_j, x_b = self._domain_randomize(x_j, x_b)

        return x_j, x_b

    def __getitem__(self, idx):
        x_j = self.joint_data[idx].copy()
        x_b = self.bone_data[idx].copy() if self.use_bone else None
        y = self.labels[idx]
        if self.augment:
            x_j, x_b = self._augment(x_j, x_b)
        return (torch.tensor(x_j, dtype=torch.float32),
                torch.tensor(x_b, dtype=torch.float32) if x_b is not None else None,
                torch.tensor(y, dtype=torch.long))


def collate_fn(batch):
    j, b, y = zip(*batch)
    return torch.stack(j), (torch.stack(b) if b[0] is not None else None), torch.stack(y)


def _gather(data_list, mask):
    assert np.issubdtype(mask.dtype, np.bool_)
    return None if data_list is None else [data_list[i] for i in range(len(data_list)) if mask[i]]


# =============================================================================
# DATASET BUILDING (Kinect CSVs)
# =============================================================================
def build_dataset(data_dir):
    print(f"\n📁 Scanning: {data_dir}")
    files_to_process = []
    for root, dirs, files in os.walk(data_dir):
        path = os.path.normpath(root).split(os.sep)
        if "Autism" in path:
            cls, lbl, b_idx = "ASD", 1, path.index("Autism")
        elif "Typical" in path:
            cls, lbl, b_idx = "TD", 0, path.index("Typical")
        else:
            continue
        s_idx = next(
            (i for i in range(b_idx+1, len(path))
             if path[i] not in INTERMEDIATE_FOLDERS
             and re.match(r'^([A-Za-z]+)?\d+$', path[i].strip())),
            -1,
        )
        if s_idx == -1: continue
        for f in files:
            basename = os.path.splitext(f)[0].strip()
            if CLIP_FILENAME_RE.match(basename) and f.endswith(('.xlsx', '.csv')):
                files_to_process.append(
                    (os.path.join(root, f), lbl, f"{cls}_Subj_{path[s_idx].strip()}"))

    j_data, b_data, hc, labels, groups = [], [], [], [], []
    total_clipped = 0
    for fp, lbl, gid in tqdm(files_to_process, desc="🚀 Extracting Clips"):
        res, n_clip = parse_skeleton_file(fp)
        if res is None: continue
        total_clipped += n_clip
        ntu = center_skeleton(
            res[:, NTU_REORDER_COLS].reshape(TARGET_FRAMES, 25, 3)
               .transpose(2, 0, 1)[:, :, :, np.newaxis].astype(np.float32))
        j_data.append(ntu)
        hc.append(extract_handcrafted_features(res))
        labels.append(lbl); groups.append(gid)
        if USE_BONE_STREAM:
            b_data.append(compute_bone_data(ntu))

    print(f"\n✅ Extracted: {len(j_data)} clips (ASD: {sum(labels)}, TD: {len(labels)-sum(labels)})")
    if total_clipped > 0:
        print(f"  ⚠️  Total clipped coord values: {total_clipped}")
    return j_data, (b_data if USE_BONE_STREAM else None), np.array(hc), np.array(labels), np.array(groups)


# =============================================================================
# FINE-TUNING + EMBEDDINGS
# =============================================================================
def fine_tune_model(model, train_dataset, val_dataset,
                    epochs=NUM_EPOCHS_FINETUNE, patience=FINETUNE_PATIENCE):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn)
    pg = [{"params": list(model.stream1.parameters()),
           "lr": LR_FINETUNE * LR_STREAM1_RATIO}]
    if model.use_bone_stream:
        pg.append({"params": list(model.stream2.parameters()), "lr": LR_FINETUNE})
    pg.append({"params": list(model.classifier.parameters()),
               "lr": LR_FINETUNE * LR_CLASSIFIER_RATIO})
    opt = torch.optim.Adam(pg, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda e: (e+1)/WARMUP_EPOCHS if e < WARMUP_EPOCHS
        else 0.5 * (1.0 + math.cos(math.pi * (e - WARMUP_EPOCHS) /
                                   max(1, epochs - WARMUP_EPOCHS))))
    cw = compute_class_weight("balanced", classes=np.unique(train_dataset.labels),
                              y=train_dataset.labels)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE))

    best_acc, best_state, pat = 0.0, None, 0
    for epoch in range(epochs):
        model.train(); tc = tt = 0
        for x_j, x_b, y in train_loader:
            x_j = x_j.to(DEVICE)
            x_b = x_b.to(DEVICE) if x_b is not None else None
            y = y.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(x_j, x_b)
            loss = crit(logits, y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tc += (logits.argmax(1) == y).sum().item(); tt += x_j.size(0)
        sched.step()
        model.eval(); vc = vt = 0
        with torch.no_grad():
            for x_j, x_b, y in val_loader:
                x_j = x_j.to(DEVICE)
                x_b = x_b.to(DEVICE) if x_b is not None else None
                y = y.to(DEVICE)
                logits, _ = model(x_j, x_b)
                vc += (logits.argmax(1) == y).sum().item(); vt += x_j.size(0)
        va = vc / vt if vt > 0 else 0.0
        if va > best_acc:
            best_acc, best_state, pat = va, copy.deepcopy(model.state_dict()), 0
        else:
            pat += 1
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}: TrAcc={tc/tt:.4f} | ValAcc={va:.4f}")
        if pat >= patience:
            print(f"  ⏹️  Early stop @ {epoch+1}")
            break
    if best_state:
        model.load_state_dict(best_state)
    print(f"  ✅ Best val acc: {best_acc:.4f}")
    return model


def extract_all_embeddings(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    model.eval(); out = []
    with torch.no_grad():
        for x_j, x_b, _ in loader:
            x_j = x_j.to(DEVICE)
            x_b = x_b.to(DEVICE) if x_b is not None else None
            out.append(model.get_embeddings(x_j, x_b).cpu().numpy())
    return np.concatenate(out, axis=0)


# =============================================================================
# FILTERING + ENSEMBLES (from v37u)
# =============================================================================
def variance_filter(X, mn=MIN_VARIANCE):
    return np.where(np.var(X, axis=0) > mn)[0].tolist()


def mann_whitney_filter(X, y, alpha=0.01):
    pv = []
    for i in range(X.shape[1]):
        x0, x1 = X[y == 0, i], X[y == 1, i]
        if np.std(x0) == 0 and np.std(x1) == 0:
            pv.append(1.0); continue
        try:
            _, p = mannwhitneyu(x0, x1, alternative="two-sided")
            pv.append(p if not np.isnan(p) else 1.0)
        except Exception:
            pv.append(1.0)
    pv = np.array(pv)
    sig = np.where(np.nan_to_num(pv) <= alpha)[0].tolist()
    if not sig:
        if np.any(pv <= 0.05):
            sig = np.where(pv <= 0.05)[0].tolist()
        else:
            sig = np.argsort(pv)[:min(50, len(pv))].tolist()
    return sig


def remove_correlated_features(X, names, threshold=CORRELATION_THRESHOLD):
    upper = pd.DataFrame(X, columns=names).corr(numeric_only=True).abs().where(
        np.triu(np.ones((X.shape[1], X.shape[1])), k=1).astype(bool))
    drop = {c for c in upper.columns if any(upper[c] > threshold)}
    return [i for i, n in enumerate(names) if n not in drop]


XGB_PARAMS = dict(objective="binary:logistic", eval_metric="logloss",
                  n_estimators=300, max_depth=4, learning_rate=0.03,
                  min_child_weight=3, subsample=0.7, colsample_bytree=0.5,
                  reg_alpha=2.0, reg_lambda=2.0, gamma=0,
                  importance_type="gain", random_state=42, verbosity=0,
                  early_stopping_rounds=30)
LGB_PARAMS = dict(objective="binary", n_estimators=300, max_depth=4,
                  learning_rate=0.03, min_child_samples=10,
                  subsample=0.7, colsample_bytree=0.5,
                  reg_alpha=2.0, reg_lambda=2.0, random_state=42, verbose=-1)
CAT_PARAMS = dict(loss_function="Logloss", iterations=300, depth=4,
                  learning_rate=0.03, l2_leaf_reg=10, subsample=0.7,
                  bootstrap_type="Bernoulli", random_seed=42,
                  verbose=0, early_stopping_rounds=30)


def train_ensemble_clip(X_tr, y_tr, X_val, y_val):
    sc = StandardScaler()
    Xt = sc.fit_transform(X_tr); Xv = sc.transform(X_val)
    bi = {}
    m = xgb.XGBClassifier(**XGB_PARAMS).fit(Xt, y_tr, eval_set=[(Xv, y_val)], verbose=False)
    bi["xgb"] = m.best_iteration + 1
    m = lgb.LGBMClassifier(**LGB_PARAMS).fit(Xt, y_tr, eval_set=[(Xv, y_val)],
                                              callbacks=[lgb.early_stopping(30, verbose=False)])
    bi["lgb"] = m.best_iteration_
    m = CatBoostClassifier(**CAT_PARAMS).fit(Xt, y_tr, eval_set=(Xv, y_val), verbose=False)
    bi["cat"] = m.best_iteration_ + 1

    sc_f = StandardScaler()
    X_all = sc_f.fit_transform(np.vstack([X_tr, X_val]))
    y_all = np.hstack([y_tr, y_val])
    fm = {}
    p = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    p["n_estimators"] = max(10, int(round(bi["xgb"])))
    fm["xgb"] = xgb.XGBClassifier(**p).fit(X_all, y_all)
    p = {k: v for k, v in LGB_PARAMS.items() if k != "n_estimators"}
    p["n_estimators"] = max(10, int(round(bi["lgb"])))
    fm["lgb"] = lgb.LGBMClassifier(**p).fit(X_all, y_all)
    p = {k: v for k, v in CAT_PARAMS.items() if k not in ("iterations", "early_stopping_rounds")}
    p["iterations"] = max(10, int(round(bi["cat"])))
    fm["cat"] = CatBoostClassifier(**p).fit(X_all, y_all)
    return fm, sc_f


def predict_with_ensemble(models, scaler, X):
    Xs = scaler.transform(X)
    probs = {n: m.predict_proba(Xs) for n, m in models.items()}
    probs["ensemble"] = sum(probs.values()) / len(probs)
    return probs


def aggregate_predictions_to_subject(clip_probs, y_clips, g_clips):
    subjects = np.unique(g_clips)
    sp, sl, sg = [], [], []
    for s in subjects:
        m = g_clips == s
        sp.append(clip_probs[m].mean(axis=0))
        sl.append(y_clips[m][0]); sg.append(s)
    return np.array(sp), np.array(sl), np.array(sg)


def tune_ensemble_weights(oof_emb, oof_hc, oof_y, oof_g):
    """Wider search range than v37u (down to EMB=0.30) since for RGB deployment
    we may want HC to dominate. The Kinect-CV optimum is reported anyway —
    you may need to manually shift weights toward HC for actual phone use."""
    print("\n  ⚖️  Tuning EMB/HC weight on OOF subject-level data...")
    best_w, best_acc, best_auc = 0.5, 0.0, 0.0
    for w in np.arange(0.30, 0.86, 0.05):
        comb = w*oof_emb + (1-w)*oof_hc
        sp, sl, _ = aggregate_predictions_to_subject(comb, oof_y, oof_g)
        acc = accuracy_score(sl, sp.argmax(axis=1))
        auc = roc_auc_score(sl, sp[:, 1]) if len(np.unique(sl)) == 2 else 0.0
        print(f"    EMB={w:.2f}/HC={1-w:.2f}: Acc={acc:.4f}  AUC={auc:.4f}")
        if acc > best_acc or (acc == best_acc and auc > best_auc):
            best_w, best_acc, best_auc = w, acc, auc
    print(f"  ✅ Kinect-CV optimal: EMB={best_w:.2f}/HC={1-best_w:.2f}")
    print(f"     ⚠️  For RGB deployment, consider shifting toward HC (e.g. 0.4/0.6)")
    return best_w, 1 - best_w


# =============================================================================
# DISTILLED EBM
# =============================================================================
def build_distilled_ebm(X_hc, p_teacher_oof, hc_names, max_features=WHITEBOX_MAX_FEATURES):
    print("\n" + "="*60)
    print(" DISTILLED EBM WHITEBOX (faithful to deployed model)")
    print("="*60)

    sc_sel = StandardScaler(); Xs = sc_sel.fit_transform(X_hc)
    print(f"  🔍 ElasticNet feature selection (target=p_teacher, budget≤{max_features}):")
    best_alpha, best_r2, best_nz = None, -np.inf, 0
    for alpha in [0.001, 0.01, 0.05, 0.1, 0.5]:
        en = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42, max_iter=5000)
        en.fit(Xs, p_teacher_oof)
        nz = int((en.coef_ != 0).sum())
        r2 = en.score(Xs, p_teacher_oof)
        within = nz <= max_features
        if within and r2 > best_r2:
            best_r2, best_alpha, best_nz = r2, alpha, nz
        print(f"    α={alpha:>6.3f}: R²={r2:.4f}  nz={nz}/{X_hc.shape[1]}  {'✓' if within else '✗'}")

    en_sel = ElasticNet(alpha=best_alpha, l1_ratio=0.5, random_state=42, max_iter=5000)
    en_sel.fit(Xs, p_teacher_oof)
    sel_idx = np.where(en_sel.coef_ != 0)[0]
    if len(sel_idx) == 0:
        sel_idx = np.argsort(np.abs(en_sel.coef_))[::-1][:min(20, len(hc_names))]
    sel_names = [hc_names[i] for i in sel_idx]
    X_sel = X_hc[:, sel_idx]
    print(f"  ✅ Selected {len(sel_names)} HC features (α={best_alpha}, R²={best_r2:.4f})")

    ebm_params = dict(
        max_bins=32, max_interaction_bins=4, interactions=10,
        inner_bags=8, outer_bags=8, learning_rate=0.01, max_rounds=500,
        min_hessian=1e-4, reg_alpha=0.1, reg_lambda=0.1, random_state=42,
    )
    ebm = ExplainableBoostingRegressor(feature_names=sel_names, **ebm_params)
    ebm.fit(X_sel, p_teacher_oof)
    fidelity = float(np.corrcoef(ebm.predict(X_sel), p_teacher_oof)[0, 1])
    print(f"  📐 Surrogate-teacher fidelity (Pearson r): {fidelity:.4f}")

    n_main = len(sel_names)
    importances = np.array([np.mean(np.abs(ebm.term_scores_[i])) for i in range(n_main)])
    print(f"\n  Top 15 EBM features:")
    for r, idx in enumerate(np.argsort(importances)[::-1][:15], 1):
        print(f"    {r:>2}. {sel_names[idx]:<45} {importances[idx]:.6f}")

    interactions = []
    for i in range(n_main, len(ebm.term_names_)):
        nm = str(ebm.term_names_[i])
        if " & " in nm:
            interactions.append((nm, float(np.mean(np.abs(ebm.term_scores_[i])))))
    if interactions:
        interactions.sort(key=lambda t: t[1], reverse=True)
        print(f"\n  Top discovered interactions:")
        for nm, imp in interactions[:5]:
            print(f"    • {nm}  (importance={imp:.6f})")

    return ebm, sc_sel, sel_idx.tolist(), sel_names, fidelity


# =============================================================================
# RUN ONE SEED — one full 5-fold subject CV. Body is the original STEP B
# loop + OOF reconstruction + weight tuning, with the seed threaded through
# the four random_state values. Nothing else changed.
# =============================================================================
def run_one_seed(seed, j_data, b_data, hc_features, labels, groups, N_CLASSES):
    set_seed(seed)
    IS_BINARY = N_CLASSES == 2

    unique_subjs = np.unique(groups)
    subj_y = pd.DataFrame({"g": groups, "y": labels}).groupby("g")["y"].first()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)   # was 42

    oof_clip_emb, oof_clip_hc = {}, {}
    fold_details = []
    fold_num = 0
    model = None

    for tr_idx, te_idx in skf.split(unique_subjs, subj_y.reindex(unique_subjs).values):
        fold_num += 1
        print(f"\n{'─'*60}\n  🔄 Seed {seed} | Fold {fold_num}/5\n{'─'*60}")
        train_subjs, test_subjs = unique_subjs[tr_idx], unique_subjs[te_idx]
        if model is not None:
            del model; torch.cuda.empty_cache()

        ft_tr_s, ft_vl_s = train_test_split(
            train_subjs, test_size=FT_VAL_SPLIT, random_state=seed,   # was 42
            stratify=subj_y.reindex(train_subjs).values)
        ft_tr_m = np.isin(groups, ft_tr_s); ft_vl_m = np.isin(groups, ft_vl_s)

        print("  🏗️  Fine-tuning MS-G3D (with domain randomization)...")
        model = TwoStreamOfficialMSG3D(num_classes=N_CLASSES,
                                       use_bone_stream=USE_BONE_STREAM,
                                       dropout=DROPOUT_RATE).to(DEVICE)
        # DOMAIN RANDOMIZATION ON during training
        model = fine_tune_model(
            model,
            SkeletonDataset(_gather(j_data, ft_tr_m), _gather(b_data, ft_tr_m),
                             labels[ft_tr_m], augment=True, domain_random=True),
            SkeletonDataset(_gather(j_data, ft_vl_m), _gather(b_data, ft_vl_m),
                             labels[ft_vl_m], augment=False, domain_random=False),
        )

        tree_tr_m = np.isin(groups, train_subjs) & ~np.isin(groups, ft_vl_s)
        test_m = np.isin(groups, test_subjs)
        y_tr, g_tr = labels[tree_tr_m], groups[tree_tr_m]
        y_te, g_te = labels[test_m], groups[test_m]
        te_indices_global = np.where(test_m)[0]

        # NOTE: embeddings extracted WITHOUT augmentation/DR for stable features
        print("  🧠 Extracting embeddings...")
        emb_tr = extract_all_embeddings(
            model, SkeletonDataset(_gather(j_data, tree_tr_m), _gather(b_data, tree_tr_m), y_tr))
        emb_te = extract_all_embeddings(
            model, SkeletonDataset(_gather(j_data, test_m), _gather(b_data, test_m), y_te))
        hc_tr, hc_te = hc_features[tree_tr_m], hc_features[test_m]

        pca_n = min(PCA_COMPONENTS, emb_tr.shape[0]-1, emb_tr.shape[1])
        pca = PCA(n_components=pca_n, random_state=seed)   # was 42
        emb_tr_p = pca.fit_transform(emb_tr); emb_te_p = pca.transform(emb_te)

        keep_emb = variance_filter(emb_tr_p)
        Xe_tr, Xe_te = emb_tr_p[:, keep_emb], emb_te_p[:, keep_emb]

        keep_hc1 = variance_filter(hc_tr)
        Xh_tr1, Xh_te1 = hc_tr[:, keep_hc1], hc_te[:, keep_hc1]
        names1 = [HC_FEATURE_NAMES[i] for i in keep_hc1]
        sx, sy = [], []
        for g in np.unique(g_tr):
            m = g_tr == g; sx.append(Xh_tr1[m].mean(axis=0)); sy.append(y_tr[m][0])
        sx, sy = np.array(sx), np.array(sy)
        keep_hc2 = mann_whitney_filter(sx, sy, alpha=0.01)
        Xh_tr2, Xh_te2 = Xh_tr1[:, keep_hc2], Xh_te1[:, keep_hc2]
        names2 = [names1[i] for i in keep_hc2]
        keep_hc3 = remove_correlated_features(sx[:, keep_hc2], names2)
        Xh_tr_f = Xh_tr2[:, keep_hc3]; Xh_te_f = Xh_te2[:, keep_hc3]

        tr_subjs_local = np.unique(g_tr)
        strat = pd.DataFrame({"g":g_tr, "y":y_tr}).groupby("g")["y"].first().reindex(tr_subjs_local).values
        try:
            tr_s, vl_s = train_test_split(tr_subjs_local, test_size=0.15,
                                          random_state=seed*100+fold_num, stratify=strat)  # was fold_num
        except ValueError:
            tr_s, vl_s = train_test_split(tr_subjs_local, test_size=0.15,
                                          random_state=seed*100+fold_num)
        m_tr, m_vl = np.isin(g_tr, tr_s), np.isin(g_tr, vl_s)

        emb_models, sc_emb = train_ensemble_clip(Xe_tr[m_tr], y_tr[m_tr],
                                                  Xe_tr[m_vl], y_tr[m_vl])
        hc_models, sc_hc   = train_ensemble_clip(Xh_tr_f[m_tr], y_tr[m_tr],
                                                  Xh_tr_f[m_vl], y_tr[m_vl])

        emb_te_probs = predict_with_ensemble(emb_models, sc_emb, Xe_te)
        hc_te_probs  = predict_with_ensemble(hc_models, sc_hc, Xh_te_f)

        for i, gi in enumerate(te_indices_global):
            oof_clip_emb[gi] = emb_te_probs["ensemble"][i]
            oof_clip_hc[gi]  = hc_te_probs["ensemble"][i]

        sp_e, sl_e, _ = aggregate_predictions_to_subject(emb_te_probs["ensemble"], y_te, g_te)
        sp_h, _, _    = aggregate_predictions_to_subject(hc_te_probs["ensemble"], y_te, g_te)
        em_acc = accuracy_score(sl_e, sp_e.argmax(axis=1))
        hm_acc = accuracy_score(sl_e, sp_h.argmax(axis=1))
        print(f"  🏆 Seed {seed} Fold {fold_num}: EMB={em_acc:.4f}  HC={hm_acc:.4f}")
        fold_details.append({"seed": seed, "fold": fold_num,
                             "emb_subj_acc": em_acc, "hc_subj_acc": hm_acc})

    # OOF reconstruction
    N = len(labels)
    oof_emb = np.zeros((N, N_CLASSES), dtype=np.float32)
    oof_hc  = np.zeros((N, N_CLASSES), dtype=np.float32)
    for i in range(N):
        oof_emb[i] = oof_clip_emb[i]; oof_hc[i] = oof_clip_hc[i]

    w_emb, w_hc = tune_ensemble_weights(oof_emb, oof_hc, labels, groups)
    oof_combined = w_emb*oof_emb + w_hc*oof_hc
    sp, sl, sg = aggregate_predictions_to_subject(oof_combined, labels, groups)
    cv_acc = accuracy_score(sl, sp.argmax(axis=1))
    cv_auc = roc_auc_score(sl, sp[:, 1]) if IS_BINARY else None
    print(f"\n  🏆 Seed {seed} | 5-Fold Subject CV (Kinect→Kinect): "
          f"Acc={cv_acc:.4f}" + (f", AUC={cv_auc:.4f}" if cv_auc else "") +
          f" | EMB/HC={w_emb:.2f}/{w_hc:.2f}")

    if model is not None:
        del model; torch.cuda.empty_cache()

    return {"seed": seed, "cv_acc": cv_acc, "cv_auc": cv_auc,
            "oof_emb": oof_emb, "oof_hc": oof_hc,
            "w_emb": w_emb, "w_hc": w_hc, "fold_details": fold_details}


# =============================================================================
# MAIN — multi-seed CV + single deployment training
# =============================================================================
def run_pipeline():
    print("="*60); print(" STEP A: BUILD DATASET (Kinect CSVs)"); print("="*60)
    set_seed(42)
    j_data, b_data, hc_features, labels, groups = build_dataset(DATA_DIR)
    if len(j_data) == 0:
        print("❌ No clips processed.")
        return {}
    N_CLASSES = len(np.unique(labels))
    IS_BINARY = N_CLASSES == 2

    # ---- Multi-seed CV ----
    print(f"\n{'='*60}\n  STEP B: MULTI-SEED 5-FOLD CV ({len(SEEDS)} seeds)\n{'='*60}")
    seed_results = []
    for seed in SEEDS:
        print(f"\n{'#'*60}\n  SEED {seed}\n{'#'*60}")
        r = run_one_seed(seed, j_data, b_data, hc_features, labels, groups, N_CLASSES)
        seed_results.append(r)

    # ---- Aggregate across seeds ----
    print(f"\n{'='*60}\n  STEP C: AGGREGATED RESULTS ACROSS {len(SEEDS)} SEEDS\n{'='*60}")
    accs = np.array([r["cv_acc"] for r in seed_results])
    aucs_present = [r["cv_auc"] for r in seed_results if r["cv_auc"] is not None]
    aucs = np.array(aucs_present) if aucs_present else None
    weights_emb = [r["w_emb"] for r in seed_results]

    print(f"\n  Per-seed CV accuracy: {[f'{a:.4f}' for a in accs]}")
    print(f"  Mean ± std:           {accs.mean():.4f} ± {accs.std():.4f}")
    if aucs is not None:
        print(f"\n  Per-seed CV AUC:      {[f'{a:.4f}' for a in aucs]}")
        print(f"  Mean ± std:           {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"\n  Per-seed EMB weight:  {[f'{w:.2f}' for w in weights_emb]}")

    # ---- Average OOF across seeds for the EBM teacher (matches v41) ----
    print(f"\n  📊 Averaging OOF predictions across {len(SEEDS)} seeds for EBM teacher...")
    oof_emb_avg = np.mean([r["oof_emb"] for r in seed_results], axis=0)
    oof_hc_avg  = np.mean([r["oof_hc"]  for r in seed_results], axis=0)
    w_emb_avg = float(np.mean(weights_emb)); w_hc_avg = 1.0 - w_emb_avg
    oof_combined_avg = w_emb_avg*oof_emb_avg + w_hc_avg*oof_hc_avg
    sp, sl, sg = aggregate_predictions_to_subject(oof_combined_avg, labels, groups)
    cv_acc_avg = accuracy_score(sl, sp.argmax(axis=1))
    cv_auc_avg = roc_auc_score(sl, sp[:, 1]) if IS_BINARY else None
    print(f"\n  Subject-aggregated CV on AVERAGED OOF: Acc={cv_acc_avg:.4f}" +
          (f", AUC={cv_auc_avg:.4f}" if cv_auc_avg else ""))
    print(classification_report(sl, sp.argmax(axis=1), digits=4))
    print(f"\n  ⚠️  This is Kinect-test-on-Kinect performance. RGB-video performance")
    print(f"     is unmeasured until you collect a phone-camera validation set.")

    p_teacher_oof_clip = oof_combined_avg[:, 1]

    # ---- STEP D: Deployment teacher (trained once, seed 42) — original logic ----
    print("\n" + "="*60); print(f" STEP D: TRAIN DEPLOYMENT TEACHER (with DR, seed={SEEDS[0]})"); print("="*60)
    set_seed(SEEDS[0])
    all_subjs = np.unique(groups)
    dep_tr, dep_vl = train_test_split(
        all_subjs, test_size=0.10, random_state=SEEDS[0],
        stratify=pd.DataFrame({"g":groups,"y":labels}).groupby("g")["y"].first().reindex(all_subjs).values)
    deploy_model = TwoStreamOfficialMSG3D(num_classes=N_CLASSES,
                                          use_bone_stream=USE_BONE_STREAM,
                                          dropout=DROPOUT_RATE).to(DEVICE)
    deploy_model = fine_tune_model(
        deploy_model,
        SkeletonDataset(_gather(j_data, np.isin(groups, dep_tr)),
                         _gather(b_data, np.isin(groups, dep_tr)),
                         labels[np.isin(groups, dep_tr)],
                         augment=True, domain_random=True),
        SkeletonDataset(_gather(j_data, np.isin(groups, dep_vl)),
                         _gather(b_data, np.isin(groups, dep_vl)),
                         labels[np.isin(groups, dep_vl)],
                         augment=False, domain_random=False),
    )

    print("  🧠 Embedding all clips...")
    emb_all = extract_all_embeddings(deploy_model, SkeletonDataset(j_data, b_data, labels))
    pca_n = min(PCA_COMPONENTS, emb_all.shape[0]-1, emb_all.shape[1])
    pca_dep = PCA(n_components=pca_n, random_state=SEEDS[0])
    emb_all_p = pca_dep.fit_transform(emb_all)

    keep_emb_d = variance_filter(emb_all_p); Xe_dep = emb_all_p[:, keep_emb_d]
    keep_hc_d1 = variance_filter(hc_features); Xh_d1 = hc_features[:, keep_hc_d1]
    names_d1 = [HC_FEATURE_NAMES[i] for i in keep_hc_d1]
    sx, sy = [], []
    for g in np.unique(groups):
        m = groups == g; sx.append(Xh_d1[m].mean(axis=0)); sy.append(labels[m][0])
    sx, sy = np.array(sx), np.array(sy)
    keep_hc_d2 = mann_whitney_filter(sx, sy, alpha=0.01)
    Xh_d2 = Xh_d1[:, keep_hc_d2]; names_d2 = [names_d1[i] for i in keep_hc_d2]
    keep_hc_d3 = remove_correlated_features(sx[:, keep_hc_d2], names_d2)
    Xh_dep = Xh_d2[:, keep_hc_d3]
    hc_names_dep = [names_d2[i] for i in keep_hc_d3]

    print("  🌳 Training deployment EMB ensemble...")
    tr_s, vl_s = train_test_split(
        np.unique(groups), test_size=0.15, random_state=SEEDS[0],
        stratify=pd.DataFrame({"g":groups,"y":labels}).groupby("g")["y"].first().reindex(np.unique(groups)).values)
    m_tr, m_vl = np.isin(groups, tr_s), np.isin(groups, vl_s)
    emb_models_dep, sc_emb_dep = train_ensemble_clip(Xe_dep[m_tr], labels[m_tr],
                                                      Xe_dep[m_vl], labels[m_vl])
    print("  🌳 Training deployment HC ensemble...")
    hc_models_dep, sc_hc_dep = train_ensemble_clip(Xh_dep[m_tr], labels[m_tr],
                                                    Xh_dep[m_vl], labels[m_vl])

    # Distilled EBM (built on the seed-averaged OOF teacher)
    ebm, sc_ebm, ebm_idx_in_dep, ebm_names, fidelity = build_distilled_ebm(
        Xh_dep, p_teacher_oof_clip, hc_names_dep, max_features=WHITEBOX_MAX_FEATURES)

    # Save artifacts (everything infer_video.py needs)
    print("\n" + "="*60); print(" STEP E: SAVING ARTIFACTS"); print("="*60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    idx_after_var  = np.array(keep_hc_d1)
    idx_after_mw   = idx_after_var[np.array(keep_hc_d2)]
    idx_after_corr = idx_after_mw[np.array(keep_hc_d3)]

    artifact = {
        "version":           "v40_3seed",
        "architecture":      "MS-G3D + DR + distilled EBM (Kinect-trained, RGB-deployable)",
        "deploy_model_path": os.path.join(OUTPUT_DIR, "msg3d_v40.pth"),
        "pca_deploy":        pca_dep,
        "emb_keep_indices":  keep_emb_d,
        "hc_final_indices_in_full": idx_after_corr.tolist(),
        "hc_names_final":    hc_names_dep,
        "scaler_emb":        sc_emb_dep,
        "scaler_hc":         sc_hc_dep,
        "scaler_ebm":        sc_ebm,
        "emb_models":        emb_models_dep,
        "hc_models":         hc_models_dep,
        "ebm":               ebm,
        "ebm_sel_idx_in_dep_hc": ebm_idx_in_dep,
        "ebm_sel_names":     ebm_names,
        "ebm_fidelity":      fidelity,
        "weight_emb":        w_emb_avg,
        "weight_hc":         w_hc_avg,
        # ---- single-seed-compatible keys (kept for infer_video.py) ----
        "cv_accuracy_kinect": cv_acc_avg,
        "cv_auc_kinect":     cv_auc_avg,
        "fold_details":      [d for r in seed_results for d in r["fold_details"]],
        # ---- multi-seed reporting ----
        "seeds":             SEEDS,
        "per_seed_acc":      accs.tolist(),
        "per_seed_auc":      [r["cv_auc"] for r in seed_results],
        "cv_acc_mean":       float(accs.mean()),
        "cv_acc_std":        float(accs.std()),
        "cv_auc_mean":       float(aucs.mean()) if aucs is not None else None,
        "cv_auc_std":        float(aucs.std()) if aucs is not None else None,
        "feature_names_full": list(HC_FEATURE_NAMES),
        "_DEPLOYMENT_NOTE":   ("v40 multi-seed. Trained on Kinect CSVs with domain "
                               "randomization. RGB-video accuracy unmeasured until paired "
                               f"validation collected. CV over {len(SEEDS)} seeds: "
                               f"{accs.mean():.4f} ± {accs.std():.4f}."),
    }
    joblib.dump(artifact, os.path.join(OUTPUT_DIR, "asd_gait_v40_3seed.joblib"))
    torch.save(deploy_model.state_dict(), artifact["deploy_model_path"])

    summary = {k: v for k, v in artifact.items()
               if k not in ("pca_deploy", "scaler_emb", "scaler_hc", "scaler_ebm",
                            "emb_models", "hc_models", "ebm")}
    with open(os.path.join(OUTPUT_DIR, "v40_3seed_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  ✅ Saved to {OUTPUT_DIR}")
    print(f"\n  📊 PAPER NUMBERS (v40 / MS-G3D):")
    print(f"     CV Acc (mean ± std over {len(SEEDS)} seeds): {accs.mean():.4f} ± {accs.std():.4f}")
    if aucs is not None:
        print(f"     CV AUC (mean ± std):                      {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"     EMB/HC weights (avg):                     {w_emb_avg:.2f} / {w_hc_avg:.2f}")
    print(f"     EBM fidelity (in-sample r):               {fidelity:.4f}")
    print(f"\n  Next step: use infer_video.py to test on phone videos.")

    return dict(cv_acc_mean=float(accs.mean()), cv_acc_std=float(accs.std()),
                cv_auc_mean=float(aucs.mean()) if aucs is not None else None,
                cv_auc_std=float(aucs.std()) if aucs is not None else None,
                weight_emb=w_emb_avg, weight_hc=w_hc_avg, ebm_fidelity=fidelity)


if __name__ == "__main__":
    results = run_pipeline()
