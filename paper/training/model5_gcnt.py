# =============================================================================
# ASD-GAIT v41 PAPER EDITION (REVISED v7 — 94%+ ACC PUSH)
# =============================================================================
# Previous fixes: [#1]-[#37]
#
# Changes for v7:
#   [#38] Best-of-raw-and-EMA model selection. The EMA model systematically
#         underperforms the raw model (e.g., Raw=0.9583 vs EMA=0.8125 at
#         epoch 30 in v5). The 0.999 decay is too slow for 17 batches/epoch,
#         causing the EMA to lag behind the rapidly-evolving model. Now tracks
#         both raw and EMA val accuracy and saves whichever checkpoint is
#         better. Expected: +1-2% from better embeddings alone.
#   [#39] SGDR (Cosine Annealing with Warm Restarts), 2 cycles of 55 epochs.
#         A single cosine cycle (v4-v6) commits to one basin. The warm restart
#         at epoch 65 explores new regions, potentially finding flatter minima.
#         Well-established to improve generalization on small datasets.
#   [#40] Gradient accumulation (effective batch size 64). With only 17
#         batches/epoch at BS=32, gradient estimates are noisy. Accumulating
#         over 2 steps gives BS=64 stability, particularly for the contrastive
#         loss which relies on batch-level similarity statistics.
#   [#41] Richer stacking features. Instead of just [emb_p_mean, hc_p_mean],
#         uses [emb_p_mean, emb_p_std, hc_p_mean, hc_p_std]. The std of
#         clip-level ASD probabilities per subject tells the meta-learner
#         about prediction consistency — borderline subjects often have high
#         variance across clips. LogisticRegression can now downweight
#         inconsistent streams conditionally.
#   [#42] ExtraTreesClassifier added to ensemble. Uses random splits (vs
#         XGBoost/LGB/CatBoost's optimal splits), providing genuine
#         diversity that simple averaging cannot replicate.
# =============================================================================

import os
os.system('pip install -q xgboost lightgbm catboost openpyxl shap joblib interpret')

import re, math, copy, json, random, warnings
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from scipy.stats import kendalltau, mannwhitneyu
from scipy.ndimage import gaussian_filter1d

from sklearn.model_selection import (StratifiedKFold, train_test_split,
                                     cross_val_predict)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier              # [#42]
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import ElasticNet, LogisticRegression

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR   = "/content/drive/MyDrive/ASD_Project/Kinect_3D_Dataset/Dataset"
OUTPUT_DIR = "/content/drive/MyDrive/ASD_Project/v41_paper_artifacts"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = 0

SEEDS = [42, 123, 2024]

# --- Architecture ---
GCN_CHANNELS_IN     = 3
GCN_HIDDEN          = 64
GCN_OUT             = 64
TRANSFORMER_DIM     = 128
TRANSFORMER_HEADS   = 4
TRANSFORMER_LAYERS  = 3
TRANSFORMER_FFN_MUL = 2

FRAME_TOKEN_DIM     = 3 * GCN_OUT

# --- Training ---
LR                  = 1e-3
WARMUP_EPOCHS       = 10
NUM_EPOCHS          = 120
PATIENCE            = 30
FT_VAL_SPLIT        = 0.10
DROPOUT             = 0.3
BATCH_SIZE          = 32
ACCUM_STEPS         = 2                 # [#40] Effective batch size = 64
TARGET_FRAMES       = 150
LABEL_SMOOTHING     = 0.1
EMA_DECAY           = 0.999
POS_VIEW_CLS_WEIGHT = 0.5

# --- Contrastive ---
CONTRASTIVE_WEIGHT  = 0.3
CONTRASTIVE_TEMP    = 0.15

# --- TTA ---
TTA_VIEWS           = 5

# --- SGDR ---
SGDR_T0             = 55                # [#39] First cycle length after warmup
SGDR_ETA_MIN_RATIO  = 0.01             # LR decays to 1% of initial (not 0)

# --- Aggressive domain randomization ---
DR_PROB             = 0.7
DR_Z_NOISE_STD      = 0.08
DR_XY_NOISE_STD     = 0.025
DR_JOINT_DROPOUT    = 0.20
DR_TEMPORAL_BLUR_P  = 0.40
DR_CAMERA_JITTER_P  = 0.6
DR_CAMERA_YAW_MAX   = math.pi / 5
DR_CAMERA_PITCH_MAX = math.pi / 12
DR_CAMERA_ROLL_MAX  = math.pi / 18

AUG_ROTATION_MAX    = math.pi / 6
AUG_JITTER_STD      = 0.02
AUG_TEMPORAL_SHIFT  = 15

# --- Filtering / EBM ---
CORRELATION_THRESHOLD = 0.90
MIN_VARIANCE          = 1e-8
WHITEBOX_MAX_FEATURES = 50

INTERMEDIATE_FOLDERS = frozenset({
    "children with ASD", "Severe level of ASD",
    "Mild", "Moderate", "Severe", "Level 1", "Level 2", "Level 3",
})
CLIP_FILENAME_RE = re.compile(r"^\d+$")
NUM_JOINTS       = 25
CENTER_JOINT     = 0
NTU_PARENTS = [0,0,24,2,24,4,5,6,7,6,24,10,11,12,13,12,0,16,17,18,0,20,21,22,1]
assert len(NTU_PARENTS) == NUM_JOINTS

print(f"Using device: {DEVICE}")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_bone_data(skeleton_data):
    parent_indices = np.array(NTU_PARENTS)
    return skeleton_data - skeleton_data[:, :, parent_indices, :]


# =============================================================================
# JOINT GRAPH
# =============================================================================
def build_normalized_adjacency():
    A = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    for child, parent in enumerate(NTU_PARENTS):
        A[child, parent] = 1.0
        A[parent, child] = 1.0
    A += np.eye(NUM_JOINTS, dtype=np.float32)
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)


# =============================================================================
# MODEL
# =============================================================================
class GraphConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, A_norm):
        super().__init__()
        self.register_buffer("A_norm", A_norm)
        self.edge_weight = nn.Parameter(torch.ones(NUM_JOINTS, NUM_JOINTS))
        self.linear = nn.Linear(in_ch, out_ch)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        A_eff = self.A_norm * (self.edge_weight + self.edge_weight.T) / 2.0
        x = torch.einsum("vw,bwc->bvc", A_eff, x)
        x = self.linear(x)
        b, v, c = x.shape
        x = self.bn(x.reshape(b * v, c)).reshape(b, v, c)
        return F.relu(x)


class JointAttentionPool(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.scorer = nn.Linear(in_ch, 1)

    def forward(self, x):
        scores = self.scorer(x).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        return (x * weights.unsqueeze(-1)).sum(dim=2)


class SpatialGCN(nn.Module):
    def __init__(self, A_norm):
        super().__init__()
        self.gcn1 = GraphConvLayer(GCN_CHANNELS_IN, GCN_HIDDEN, A_norm)
        self.gcn2 = GraphConvLayer(GCN_HIDDEN,    GCN_OUT,    A_norm)
        self.attn_pool = JointAttentionPool(GCN_OUT)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, C, T, V, M = x.shape
        x = x.squeeze(-1).permute(0, 2, 3, 1).contiguous()
        x = x.view(B * T, V, C)
        x = self.gcn1(x)
        x = self.drop(x)
        x = self.gcn2(x)
        x = x.view(B, T, V, GCN_OUT)
        frame_mean = x.mean(dim=2)
        frame_max  = x.max(dim=2).values
        frame_att  = self.attn_pool(x)
        return torch.cat([frame_mean, frame_max, frame_att], dim=2)


class TemporalTransformer(nn.Module):
    def __init__(self, in_dim, d_model=TRANSFORMER_DIM,
                 n_heads=TRANSFORMER_HEADS, n_layers=TRANSFORMER_LAYERS):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, TARGET_FRAMES + 1, d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * TRANSFORMER_FFN_MUL,
            dropout=DROPOUT, batch_first=True, activation="gelu",
            norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.proj_in(x)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, :T + 1, :]
        x = self.encoder(x)
        x = self.norm(x)
        cls_out  = x[:, 0]
        frame_x  = x[:, 1:]
        mean_out = frame_x.mean(dim=1)
        max_out  = frame_x.max(dim=1).values
        return torch.cat([cls_out, mean_out, max_out], dim=1)


class GCNTransformerASD(nn.Module):
    def __init__(self, num_classes=2, use_bone_stream=True):
        super().__init__()
        self.use_bone_stream = use_bone_stream
        A_norm = build_normalized_adjacency()
        self.gcn_joint = SpatialGCN(A_norm)
        self.tx_joint  = TemporalTransformer(in_dim=FRAME_TOKEN_DIM)
        if use_bone_stream:
            self.gcn_bone = SpatialGCN(A_norm)
            self.tx_bone  = TemporalTransformer(in_dim=FRAME_TOKEN_DIM)
            self.emb_dim = TRANSFORMER_DIM * 3 * 2
        else:
            self.emb_dim = TRANSFORMER_DIM * 3
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(self.emb_dim, num_classes)

    def forward(self, x_joint, x_bone=None):
        if self.use_bone_stream:
            assert x_bone is not None
        f_j = self.tx_joint(self.gcn_joint(x_joint))
        if self.use_bone_stream:
            f_b = self.tx_bone(self.gcn_bone(x_bone))
            features = torch.cat([f_j, f_b], dim=1)
        else:
            features = f_j
        return self.classifier(self.dropout(features)), features

    def get_embeddings(self, x_joint, x_bone=None):
        with torch.no_grad():
            return self.forward(x_joint, x_bone)[1]


# =============================================================================
# HC FEATURES
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
# DATASET
# =============================================================================
class SkeletonDataset(Dataset):
    def __init__(self, joint_data, bone_data, labels, augment=False, contrastive_mode=False):
        self.joint_data = joint_data
        self.bone_data = bone_data
        self.labels = np.asarray(labels)
        self.use_bone = bone_data is not None
        self.augment = augment
        self.contrastive_mode = contrastive_mode

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
        return gaussian_filter1d(x, sigma=sigma, axis=1).astype(np.float32)

    @staticmethod
    def _rotate_y(x, angle):
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        return np.einsum('ij,jtvm->itvm', R, x)

    @staticmethod
    def _full_3d_rotation(x, yaw, pitch, roll):
        cy, sy = math.cos(yaw),   math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll),  math.sin(roll)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rp = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
        Rr = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
        R = Rr @ Rp @ Ry
        return np.einsum('ij,jtvm->itvm', R, x)

    def _weak_aug_joints(self, x_j):
        if random.random() < 0.5:
            a = random.uniform(-AUG_ROTATION_MAX, AUG_ROTATION_MAX)
            x_j = self._rotate_y(x_j, a)
        if random.random() < 0.3:
            s = random.randint(-AUG_TEMPORAL_SHIFT, AUG_TEMPORAL_SHIFT)
            if s != 0:
                x_j = self._temporal_shift(x_j, s, axis=1)
        if random.random() < 0.5:
            x_j = x_j + np.random.normal(0, AUG_JITTER_STD, x_j.shape).astype(np.float32)
        return x_j

    def _strong_aug_joints(self, x_j):
        x_j = self._weak_aug_joints(x_j)
        if random.random() < DR_CAMERA_JITTER_P:
            yaw   = random.uniform(-DR_CAMERA_YAW_MAX,   DR_CAMERA_YAW_MAX)
            pitch = random.uniform(-DR_CAMERA_PITCH_MAX, DR_CAMERA_PITCH_MAX)
            roll  = random.uniform(-DR_CAMERA_ROLL_MAX,  DR_CAMERA_ROLL_MAX)
            x_j = self._full_3d_rotation(x_j, yaw, pitch, roll)
        if random.random() < DR_PROB:
            noise = np.zeros_like(x_j)
            noise[0] = np.random.normal(0, DR_XY_NOISE_STD, x_j.shape[1:]).astype(np.float32)
            noise[1] = np.random.normal(0, DR_XY_NOISE_STD, x_j.shape[1:]).astype(np.float32)
            noise[2] = np.random.normal(0, DR_Z_NOISE_STD,  x_j.shape[1:]).astype(np.float32)
            x_j = x_j + noise
        if random.random() < DR_PROB:
            for v in range(NUM_JOINTS):
                if random.random() < DR_JOINT_DROPOUT:
                    T = x_j.shape[1]
                    drop_len = random.randint(2, max(2, T // 8))
                    drop_start = random.randint(0, T - drop_len)
                    held = x_j[:, drop_start:drop_start+1, v:v+1, :]
                    x_j[:, drop_start:drop_start+drop_len, v:v+1, :] = held
        if random.random() < DR_TEMPORAL_BLUR_P:
            x_j = self._temporal_blur(x_j, sigma=random.uniform(0.8, 1.8))
        return x_j

    def __getitem__(self, idx):
        x_j_orig = self.joint_data[idx]
        x_b_orig = self.bone_data[idx] if self.use_bone else None
        y = int(self.labels[idx])

        if self.contrastive_mode:
            xj_a = self._weak_aug_joints(x_j_orig.copy())
            xj_p = self._strong_aug_joints(x_j_orig.copy())
            xb_a = compute_bone_data(xj_a) if x_b_orig is not None else None
            xb_p = compute_bone_data(xj_p) if x_b_orig is not None else None
            return (
                torch.tensor(xj_a, dtype=torch.float32), xb_a,
                torch.tensor(xj_p, dtype=torch.float32), xb_p,
                torch.tensor(y, dtype=torch.long),
            )

        x_j = x_j_orig.copy()
        x_b = x_b_orig.copy() if x_b_orig is not None else None
        if self.augment:
            x_j = self._strong_aug_joints(x_j)
            x_b = compute_bone_data(x_j) if x_b_orig is not None else None
        return (
            torch.tensor(x_j, dtype=torch.float32), x_b,
            torch.tensor(y, dtype=torch.long),
        )


def _stack_bone(bone_list):
    if bone_list[0] is None:
        assert all(b is None for b in bone_list)
        return None
    tensors = [torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x for x in bone_list]
    return torch.stack(tensors)

def collate_train(batch):
    xj_a, xb_a, xj_p, xb_p, y = zip(*batch)
    return (torch.stack(xj_a), _stack_bone(xb_a), torch.stack(xj_p), _stack_bone(xb_p), torch.stack(y))

def collate_eval(batch):
    j, b, y = zip(*batch)
    return torch.stack(j), _stack_bone(b), torch.stack(y)

def _gather(data_list, mask):
    assert np.issubdtype(mask.dtype, np.bool_)
    return None if data_list is None else [data_list[i] for i in range(len(data_list)) if mask[i]]


# =============================================================================
# DATASET BUILDING
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
        else: continue
        s_idx = next((i for i in range(b_idx+1, len(path)) if path[i] not in INTERMEDIATE_FOLDERS and re.match(r'^([A-Za-z]+)?\d+$', path[i].strip())), -1)
        if s_idx == -1: continue
        for f in files:
            basename = os.path.splitext(f)[0].strip()
            if CLIP_FILENAME_RE.match(basename) and f.endswith(('.xlsx', '.csv')):
                files_to_process.append((os.path.join(root, f), lbl, f"{cls}_Subj_{path[s_idx].strip()}"))

    j_data, b_data, hc, labels, groups = [], [], [], [], []
    total_clipped = 0
    for fp, lbl, gid in tqdm(files_to_process, desc="Extracting"):
        res, n_clipped = parse_skeleton_file(fp)
        if res is None: continue
        total_clipped += n_clipped
        ntu = center_skeleton(res[:, NTU_REORDER_COLS].reshape(TARGET_FRAMES, 25, 3).transpose(2, 0, 1)[:, :, :, np.newaxis].astype(np.float32))
        j_data.append(ntu); hc.append(extract_handcrafted_features(res))
        labels.append(lbl); groups.append(gid); b_data.append(compute_bone_data(ntu))
    print(f"\n✅ {len(j_data)} clips (ASD: {sum(labels)}, TD: {len(labels)-sum(labels)})")
    if total_clipped > 0: print(f"   ⚠️  {total_clipped} coordinate values clipped to ±10.0")
    return j_data, b_data, np.array(hc), np.array(labels), np.array(groups)


# =============================================================================
# LOSSES
# =============================================================================
def contrastive_invariance_loss(emb_anchor, emb_positive, temperature=CONTRASTIVE_TEMP):
    a = F.normalize(emb_anchor,   dim=1)
    p = F.normalize(emb_positive, dim=1)
    logits = a @ p.T / temperature
    labels = torch.arange(a.size(0), device=a.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_a + loss_b)


# =============================================================================
# SGDR SCHEDULER  [#39]
# =============================================================================
def sgdr_lr_lambda(epoch):
    """Warmup for WARMUP_EPOCHS, then cosine with warm restarts every SGDR_T0."""
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    adjusted = epoch - WARMUP_EPOCHS
    pos = adjusted % SGDR_T0
    return SGDR_ETA_MIN_RATIO + 0.5 * (1 - SGDR_ETA_MIN_RATIO) * (1 + math.cos(math.pi * pos / SGDR_T0))


# =============================================================================
# TTA + EMBEDDING EXTRACTION
# =============================================================================
def _tta_augment_joint(x_j):
    x = x_j.copy()
    if random.random() < 0.8:
        a = random.uniform(-math.pi / 12, math.pi / 12)
        x = SkeletonDataset._rotate_y(x, a)
    if random.random() < 0.6:
        s = random.randint(-8, 8)
        if s != 0:
            x = SkeletonDataset._temporal_shift(x, s, axis=1)
    return x


def extract_all_embeddings(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_eval, num_workers=NUM_WORKERS)
    model.eval()
    all_emb = None

    for view in range(TTA_VIEWS):
        out = []
        with torch.no_grad():
            for x_j, x_b, _ in loader:
                if view > 0:
                    x_j_np = x_j.cpu().numpy()
                    x_j_aug = np.stack([_tta_augment_joint(x) for x in x_j_np])
                    x_b_aug = np.stack([compute_bone_data(x) for x in x_j_aug]) if x_b is not None else None
                    x_j = torch.tensor(x_j_aug, dtype=torch.float32, device=DEVICE)
                    x_b = torch.tensor(x_b_aug, dtype=torch.float32, device=DEVICE) if x_b_aug is not None else None
                else:
                    x_j = x_j.to(DEVICE)
                    x_b = x_b.to(DEVICE) if x_b is not None else None
                out.append(model.get_embeddings(x_j, x_b).cpu().numpy())
        emb = np.concatenate(out, axis=0)
        all_emb = emb if all_emb is None else all_emb + emb

    return all_emb / TTA_VIEWS


# =============================================================================
# TRAINING
# =============================================================================
def train_model(model, train_dataset, val_dataset, epochs=NUM_EPOCHS, patience=PATIENCE):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate_train, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_eval, num_workers=NUM_WORKERS)

    n_total = len(train_dataset); n_dropped = n_total % BATCH_SIZE if n_total >= BATCH_SIZE else n_total; n_effective = n_total - n_dropped
    print(f"    Train clips: {n_total} → {n_effective} after drop_last, eff_BS={BATCH_SIZE*ACCUM_STEPS}, {n_effective // BATCH_SIZE} micro-batches/epoch")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=sgdr_lr_lambda)    # [#39]

    cw = compute_class_weight("balanced", classes=np.unique(train_dataset.labels), y=train_dataset.labels)
    crit_cls = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE), label_smoothing=LABEL_SMOOTHING)

    ema_avg = lambda avg, new, num: 0.999 * avg + (1 - 0.999) * new
    ema_model = AveragedModel(model, avg_fn=ema_avg, use_buffers=True).to(DEVICE)

    # [#38] Track best of raw and EMA
    best_acc, best_source, best_state, pat = 0.0, None, None, 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()                          # [#40] Zero grad at epoch start
        tc = tt = 0; running_cls = running_cls_p = running_con = 0.0; n_batches = 0
        for i, (xj_a, xb_a, xj_p, xb_p, y) in enumerate(train_loader):
            xj_a = xj_a.to(DEVICE); xb_a = xb_a.to(DEVICE) if xb_a is not None else None
            xj_p = xj_p.to(DEVICE); xb_p = xb_p.to(DEVICE) if xb_p is not None else None; y = y.to(DEVICE)

            logits_a, emb_a = model(xj_a, xb_a); logits_p, emb_p = model(xj_p, xb_p)
            loss_cls_a = crit_cls(logits_a, y); loss_cls_p = crit_cls(logits_p, y)
            loss_cls = loss_cls_a + POS_VIEW_CLS_WEIGHT * loss_cls_p
            loss_con = contrastive_invariance_loss(emb_a, emb_p)
            loss = (loss_cls + CONTRASTIVE_WEIGHT * loss_con) / ACCUM_STEPS    # [#40] Scale for accumulation

            loss.backward()
            running_cls += loss_cls_a.item(); running_cls_p += loss_cls_p.item(); running_con += loss_con.item(); n_batches += 1
            tc += (logits_a.argmax(1) == y).sum().item(); tt += xj_a.size(0)

            # [#40] Gradient accumulation: step every ACCUM_STEPS micro-batches
            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                if epoch >= WARMUP_EPOCHS: ema_model.update_parameters(model)

        # Handle remaining gradients if not evenly divisible
        if (n_batches % ACCUM_STEPS) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            if epoch >= WARMUP_EPOCHS: ema_model.update_parameters(model)

        sched.step()

        use_ema = epoch >= WARMUP_EPOCHS; eval_module = ema_model if use_ema else model
        eval_module.eval(); vc_ema = vt = 0
        with torch.no_grad():
            for x_j, x_b, y in val_loader:
                x_j = x_j.to(DEVICE); x_b = x_b.to(DEVICE) if x_b is not None else None; y = y.to(DEVICE)
                logits, _ = eval_module(x_j, x_b); vc_ema += (logits.argmax(1) == y).sum().item(); vt += x_j.size(0)
        va_ema = vc_ema / vt if vt > 0 else 0.0

        model.eval(); vc_raw = 0
        with torch.no_grad():
            for x_j, x_b, y in val_loader:
                x_j = x_j.to(DEVICE); x_b = x_b.to(DEVICE) if x_b is not None else None; y = y.to(DEVICE)
                logits, _ = model(x_j, x_b); vc_raw += (logits.argmax(1) == y).sum().item()
        va_raw = vc_raw / vt if vt > 0 else 0.0

        # [#38] Select best of raw and EMA
        if va_ema >= va_raw:
            va_best = va_ema; source = "EMA"
        else:
            va_best = va_raw; source = "Raw"

        if va_best > best_acc:
            best_acc = va_best; best_source = source; pat = 0
            if source == "EMA":
                best_state = copy.deepcopy(ema_model.module.state_dict())
            else:
                best_state = copy.deepcopy(model.state_dict())
        else: pat += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}: TrAcc={tc/tt:.4f}  ValAcc(EMA)={va_ema:.4f}  ValAcc(Raw)={va_raw:.4f}  Best={best_acc:.4f}({best_source})  L_cls_a={running_cls/n_batches:.3f}  L_con={running_con/n_batches:.3f}  LR={opt.param_groups[0]['lr']:.6f}")
        if pat >= patience: print(f"  ⏹️  Early stop @ {epoch+1}"); break

    if best_state is not None: model.load_state_dict(best_state)
    print(f"  ✅ Best val acc: {best_acc:.4f} (from {best_source})")
    return model


def extract_all_embeddings_simple(model, dataset):
    """Extract embeddings without TTA (used inside training loop where TTA is too slow)."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_eval, num_workers=NUM_WORKERS)
    model.eval(); out = []
    with torch.no_grad():
        for x_j, x_b, _ in loader:
            x_j = x_j.to(DEVICE); x_b = x_b.to(DEVICE) if x_b is not None else None
            out.append(model.get_embeddings(x_j, x_b).cpu().numpy())
    return np.concatenate(out, axis=0)


# =============================================================================
# ENSEMBLES + FILTERING + EBM
# =============================================================================
def variance_filter(X, mn=MIN_VARIANCE): return np.where(np.var(X, axis=0) > mn)[0].tolist()

def mann_whitney_filter(X, y, alpha=0.01):
    pv = []
    for i in range(X.shape[1]):
        x0, x1 = X[y == 0, i], X[y == 1, i]
        if np.std(x0) == 0 and np.std(x1) == 0: pv.append(1.0); continue
        try: _, p = mannwhitneyu(x0, x1, alternative="two-sided"); pv.append(p if not np.isnan(p) else 1.0)
        except: pv.append(1.0)
    pv = np.array(pv); sig = np.where(np.nan_to_num(pv) <= alpha)[0].tolist()
    if not sig:
        if np.any(pv <= 0.05): sig = np.where(pv <= 0.05)[0].tolist()
        else: sig = np.argsort(pv)[:min(50, len(pv))].tolist()
    return sig

def remove_correlated_features(X, names, threshold=CORRELATION_THRESHOLD):
    upper = pd.DataFrame(X, columns=names).corr(numeric_only=True).abs().where(np.triu(np.ones((X.shape[1], X.shape[1])), k=1).astype(bool))
    drop = {c for c in upper.columns if any(upper[c] > threshold)}
    return [i for i, n in enumerate(names) if n not in drop]

XGB_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", n_estimators=300, max_depth=4, learning_rate=0.03, min_child_weight=3, subsample=0.7, colsample_bytree=0.5, reg_alpha=2.0, reg_lambda=2.0, gamma=0, importance_type="gain", random_state=42, verbosity=0, early_stopping_rounds=30)
LGB_PARAMS = dict(objective="binary", n_estimators=300, max_depth=4, learning_rate=0.03, min_child_samples=10, subsample=0.7, colsample_bytree=0.5, reg_alpha=2.0, reg_lambda=2.0, random_state=42, verbose=-1)
CAT_PARAMS = dict(loss_function="Logloss", iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=10, subsample=0.7, bootstrap_type="Bernoulli", random_seed=42, verbose=0)
ET_PARAMS  = dict(n_estimators=300, max_depth=4, min_samples_split=10, min_samples_leaf=5, class_weight='balanced', random_state=42)  # [#42]

def train_ensemble_clip(X_tr, y_tr, X_val, y_val):
    sc = StandardScaler(); Xt = sc.fit_transform(X_tr); Xv = sc.transform(X_val); bi = {}
    m = xgb.XGBClassifier(**XGB_PARAMS).fit(Xt, y_tr, eval_set=[(Xv, y_val)], verbose=False); bi["xgb"] = m.best_iteration + 1
    m = lgb.LGBMClassifier(**LGB_PARAMS).fit(Xt, y_tr, eval_set=[(Xv, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)]); bi["lgb"] = m.best_iteration_
    m = CatBoostClassifier(**CAT_PARAMS).fit(Xt, y_tr, eval_set=(Xv, y_val), verbose=False, early_stopping_rounds=30); bi["cat"] = m.best_iteration_ + 1
    # [#42] ExtraTrees — no early stopping, train on train set
    et_val = ExtraTreesClassifier(**ET_PARAMS).fit(Xt, y_tr)

    sc_f = StandardScaler(); X_all = sc_f.fit_transform(np.vstack([X_tr, X_val])); y_all = np.hstack([y_tr, y_val]); fm = {}
    p = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}; p["n_estimators"] = max(10, int(round(bi["xgb"]))); fm["xgb"] = xgb.XGBClassifier(**p).fit(X_all, y_all)
    p = {k: v for k, v in LGB_PARAMS.items() if k != "n_estimators"}; p["n_estimators"] = max(10, int(round(bi["lgb"]))); fm["lgb"] = lgb.LGBMClassifier(**p).fit(X_all, y_all)
    p = {k: v for k, v in CAT_PARAMS.items() if k != "iterations"}; p["iterations"] = max(10, int(round(bi["cat"]))); fm["cat"] = CatBoostClassifier(**p).fit(X_all, y_all)
    fm["et"] = ExtraTreesClassifier(**ET_PARAMS).fit(X_all, y_all)    # [#42]
    return fm, sc_f

def predict_with_ensemble(models, scaler, X):
    Xs = scaler.transform(X); probs = {n: m.predict_proba(Xs) for n, m in models.items()}; probs["ensemble"] = sum(probs.values()) / len(probs); return probs

def aggregate_predictions_to_subject(clip_probs, y_clips, g_clips):
    subjects = np.unique(g_clips); sp, sp_std, sl, sg = [], [], [], []
    for s in subjects:
        m = g_clips == s
        sp.append(clip_probs[m].mean(axis=0))
        sp_std.append(clip_probs[m].std(axis=0))     # [#41] Subject-level consistency
        sl.append(y_clips[m][0]); sg.append(s)
    return np.array(sp), np.array(sp_std), np.array(sl), np.array(sg)

# [#41] Richer stacking with mean + std features
def tune_ensemble_weights(oof_emb, oof_hc, oof_y, oof_g, verbose=True):
    if verbose: print("\n  ⚖️  Training subject-level Logistic Stacking...")
    sp_emb, sp_emb_std, sl, _ = aggregate_predictions_to_subject(oof_emb, oof_y, oof_g)
    sp_hc, sp_hc_std, _, _ = aggregate_predictions_to_subject(oof_hc, oof_y, oof_g)
    # [#41] Mean ASD prob + std of clip probs per stream = 4 features
    X_meta = np.column_stack([
        sp_emb[:, 1],        # Mean ASD prob (embedding)
        sp_emb_std[:, 1],    # Std of clip ASD probs (embedding consistency)
        sp_hc[:, 1],         # Mean ASD prob (handcrafted)
        sp_hc_std[:, 1],     # Std of clip ASD probs (HC consistency)
    ])

    meta_clf = LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=500)
    meta_clf.fit(X_meta, sl)

    # Aggregate EMB/HC weights from mean coefficients only
    coefs = meta_clf.coef_[0]
    w_emb_raw = abs(coefs[0]) + abs(coefs[1])
    w_hc_raw = abs(coefs[2]) + abs(coefs[3])
    total_w = w_emb_raw + w_hc_raw + 1e-8; best_w_emb = w_emb_raw / total_w; best_w_hc = w_hc_raw / total_w

    if verbose:
        acc = accuracy_score(sl, meta_clf.predict(X_meta))
        print(f"  ✅ Stacking Train Acc: {acc:.4f} (EMB/HC weight: {best_w_emb:.2f}/{best_w_hc:.2f})")
    return meta_clf, best_w_emb, best_w_hc

def build_distilled_ebm(X_hc, p_teacher_oof, hc_names, max_features=WHITEBOX_MAX_FEATURES):
    print("\n" + "="*60); print(" DISTILLED EBM WHITEBOX (averaged across seeds)"); print("="*60)
    sc_sel = StandardScaler(); Xs = sc_sel.fit_transform(X_hc)
    print(f"  🔍 ElasticNet feature selection (target=p_teacher, budget≤{max_features}):")
    best_alpha, best_r2 = None, -np.inf
    for alpha in [0.001, 0.01, 0.05, 0.1, 0.5]:
        en = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42, max_iter=5000); en.fit(Xs, p_teacher_oof)
        nz = int((en.coef_ != 0).sum()); r2 = en.score(Xs, p_teacher_oof); within = nz <= max_features
        if within and r2 > best_r2: best_r2, best_alpha = r2, alpha
        print(f"    α={alpha:>6.3f}: R²={r2:.4f}  nz={nz}/{X_hc.shape[1]}  {'✓' if within else '✗'}")
    en_sel = ElasticNet(alpha=best_alpha, l1_ratio=0.5, random_state=42, max_iter=5000); en_sel.fit(Xs, p_teacher_oof)
    sel_idx = np.where(en_sel.coef_ != 0)[0]
    if len(sel_idx) == 0: sel_idx = np.argsort(np.abs(en_sel.coef_))[::-1][:min(20, len(hc_names))]
    sel_names = [hc_names[i] for i in sel_idx]; X_sel = X_hc[:, sel_idx]
    print(f"  ✅ Selected {len(sel_names)} HC features (α={best_alpha}, R²={best_r2:.4f})")

    ebm_params = dict(max_bins=32, max_interaction_bins=4, interactions=10, inner_bags=8, outer_bags=8, learning_rate=0.01, max_rounds=500, min_hessian=1e-4, reg_alpha=0.1, reg_lambda=0.1, random_state=42)
    ebm_cv = ExplainableBoostingRegressor(feature_names=sel_names, **ebm_params); cv_preds = cross_val_predict(ebm_cv, X_sel, p_teacher_oof, cv=5)
    fidelity = float(np.corrcoef(cv_preds, p_teacher_oof)[0, 1]); print(f"  📐 Surrogate-teacher fidelity (5-fold CV): r={fidelity:.4f}")
    ebm = ExplainableBoostingRegressor(feature_names=sel_names, **ebm_params); ebm.fit(X_sel, p_teacher_oof)

    n_main = len(sel_names); n_total = len(ebm.term_scores_)
    all_importances = np.array([np.mean(np.abs(ebm.term_scores_[i])) for i in range(n_total)])
    term_labels = list(sel_names)
    if n_total > n_main and hasattr(ebm, 'term_names_'):
        for i in range(n_main, n_total): term_labels.append(ebm.term_names_[i])
    print(f"\n  Top 15 EBM terms (main + interactions):")
    for r, idx in enumerate(np.argsort(all_importances)[::-1][:15], 1):
        print(f"    {r:>2}. {term_labels[idx]:<45} {all_importances[idx]:.6f}  ({'interaction' if idx >= n_main else 'main'})")
    return ebm, sc_sel, sel_idx.tolist(), sel_names, fidelity


# =============================================================================
# RUN ONE SEED (full 5-fold CV)
# =============================================================================
def run_one_seed(seed, j_data, b_data, hc_features, labels, groups, N_CLASSES):
    set_seed(seed); IS_BINARY = N_CLASSES == 2
    unique_subjs = np.unique(groups); subj_y = pd.DataFrame({"g": groups, "y": labels}).groupby("g")["y"].first()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_clip_emb, oof_clip_hc = {}, {}; fold_details = []; fold_num = 0; model = None

    for tr_idx, te_idx in skf.split(unique_subjs, subj_y.reindex(unique_subjs).values):
        fold_num += 1; print(f"\n{'─'*60}\n  🔄 Seed {seed} | Fold {fold_num}/5\n{'─'*60}")
        train_subjs, test_subjs = unique_subjs[tr_idx], unique_subjs[te_idx]
        if model is not None: del model; torch.cuda.empty_cache()

        ft_tr_s, ft_vl_s = train_test_split(train_subjs, test_size=FT_VAL_SPLIT, random_state=seed, stratify=subj_y.reindex(train_subjs).values)
        ft_tr_m = np.isin(groups, ft_tr_s); ft_vl_m = np.isin(groups, ft_vl_s)
        print("  🏗️  Training GCN+Transformer..."); model = GCNTransformerASD(num_classes=N_CLASSES, use_bone_stream=True).to(DEVICE)
        model = train_model(model, SkeletonDataset(_gather(j_data, ft_tr_m), _gather(b_data, ft_tr_m), labels[ft_tr_m], augment=True, contrastive_mode=True), SkeletonDataset(_gather(j_data, ft_vl_m), _gather(b_data, ft_vl_m), labels[ft_vl_m], augment=False, contrastive_mode=False))

        tree_tr_m = np.isin(groups, train_subjs) & ~np.isin(groups, ft_vl_s); test_m = np.isin(groups, test_subjs)
        y_tr, g_tr = labels[tree_tr_m], groups[tree_tr_m]; y_te, g_te = labels[test_m], groups[test_m]; te_indices_global = np.where(test_m)[0]

        emb_tr = extract_all_embeddings(model, SkeletonDataset(_gather(j_data, tree_tr_m), _gather(b_data, tree_tr_m), y_tr, augment=False, contrastive_mode=False))
        emb_te = extract_all_embeddings(model, SkeletonDataset(_gather(j_data, test_m), _gather(b_data, test_m), y_te, augment=False, contrastive_mode=False))
        hc_tr, hc_te = hc_features[tree_tr_m], hc_features[test_m]

        # [#35] PCA with 99% variance threshold
        if emb_tr.shape[1] > 100:
            pca = PCA(n_components=0.99, random_state=seed)
            emb_tr_p = pca.fit_transform(emb_tr)
            emb_te_p = pca.transform(emb_te)
            print(f"    PCA: {emb_tr.shape[1]} → {emb_tr_p.shape[1]} components (99% variance)")
        else:
            emb_tr_p = emb_tr; emb_te_p = emb_te

        keep_emb = variance_filter(emb_tr_p); Xe_tr, Xe_te = emb_tr_p[:, keep_emb], emb_te_p[:, keep_emb]
        keep_hc1 = variance_filter(hc_tr); Xh_tr1, Xh_te1 = hc_tr[:, keep_hc1], hc_te[:, keep_hc1]; names1 = [HC_FEATURE_NAMES[i] for i in keep_hc1]
        sx, sy = [], []
        for g in np.unique(g_tr): m = g_tr == g; sx.append(Xh_tr1[m].mean(axis=0)); sy.append(y_tr[m][0])
        sx, sy = np.array(sx), np.array(sy); keep_hc2 = mann_whitney_filter(sx, sy, alpha=0.01); Xh_tr2, Xh_te2 = Xh_tr1[:, keep_hc2], Xh_te1[:, keep_hc2]; names2 = [names1[i] for i in keep_hc2]
        keep_hc3 = remove_correlated_features(sx[:, keep_hc2], names2); Xh_tr_f = Xh_tr2[:, keep_hc3]; Xh_te_f = Xh_te2[:, keep_hc3]

        tr_subjs_local = np.unique(g_tr); strat = pd.DataFrame({"g":g_tr, "y":y_tr}).groupby("g")["y"].first().reindex(tr_subjs_local).values
        try: tr_s, vl_s = train_test_split(tr_subjs_local, test_size=0.15, random_state=seed * 100 + fold_num, stratify=strat)
        except: tr_s, vl_s = train_test_split(tr_subjs_local, test_size=0.15, random_state=seed * 100 + fold_num)
        m_tr, m_vl = np.isin(g_tr, tr_s), np.isin(g_tr, vl_s)

        emb_models, sc_emb = train_ensemble_clip(Xe_tr[m_tr], y_tr[m_tr], Xe_tr[m_vl], y_tr[m_vl])
        hc_models, sc_hc = train_ensemble_clip(Xh_tr_f[m_tr], y_tr[m_tr], Xh_tr_f[m_vl], y_tr[m_vl])
        emb_te_probs = predict_with_ensemble(emb_models, sc_emb, Xe_te); hc_te_probs = predict_with_ensemble(hc_models, sc_hc, Xh_te_f)

        for i, gi in enumerate(te_indices_global): oof_clip_emb[gi] = emb_te_probs["ensemble"][i]; oof_clip_hc[gi] = hc_te_probs["ensemble"][i]
        sp_e, _, sl_e, _ = aggregate_predictions_to_subject(emb_te_probs["ensemble"], y_te, g_te)
        sp_h, _, _, _ = aggregate_predictions_to_subject(hc_te_probs["ensemble"], y_te, g_te)
        print(f"  🏆 Seed {seed} Fold {fold_num}: EMB={accuracy_score(sl_e, sp_e.argmax(axis=1)):.4f}  HC={accuracy_score(sl_e, sp_h.argmax(axis=1)):.4f}")
        fold_details.append({"seed": seed, "fold": fold_num, "emb_subj_acc": accuracy_score(sl_e, sp_e.argmax(axis=1)), "hc_subj_acc": accuracy_score(sl_e, sp_h.argmax(axis=1))})

    N = len(labels); oof_emb = np.zeros((N, N_CLASSES), dtype=np.float32); oof_hc = np.zeros((N, N_CLASSES), dtype=np.float32)
    for i in range(N): oof_emb[i] = oof_clip_emb[i]; oof_hc[i] = oof_clip_hc[i]

    meta_clf, w_emb, w_hc = tune_ensemble_weights(oof_emb, oof_hc, labels, groups, verbose=False)
    oof_combined = w_emb*oof_emb + w_hc*oof_hc
    sp_e, sp_e_std, sl, _ = aggregate_predictions_to_subject(oof_emb, labels, groups)
    sp_h, sp_h_std, _, _ = aggregate_predictions_to_subject(oof_hc, labels, groups)
    # [#41] Use richer stacking features
    X_meta = np.column_stack([sp_e[:, 1], sp_e_std[:, 1], sp_h[:, 1], sp_h_std[:, 1]])
    sp_combined = meta_clf.predict_proba(X_meta)
    cv_acc = accuracy_score(sl, sp_combined.argmax(axis=1)); cv_auc = roc_auc_score(sl, sp_combined[:, 1]) if IS_BINARY else None

    print(f"\n  🏆 Seed {seed} | 5-Fold Subject CV: Acc={cv_acc:.4f}" + (f", AUC={cv_auc:.4f}" if cv_auc else "") + f" | EMB/HC={w_emb:.2f}/{w_hc:.2f}")
    if model is not None: del model; torch.cuda.empty_cache()
    return {"seed": seed, "cv_acc": cv_acc, "cv_auc": cv_auc, "fold_details": fold_details, "oof_emb": oof_emb, "oof_hc": oof_hc, "w_emb": w_emb, "w_hc": w_hc}

def train_deployment(j_data, b_data, labels, groups, N_CLASSES, seed):
    set_seed(seed); print(f"\n{'─'*60}\n  🚀 Training deployment model (seed={seed})\n{'─'*60}")
    all_subjs = np.unique(groups); dep_tr, dep_vl = train_test_split(all_subjs, test_size=FT_VAL_SPLIT, random_state=seed, stratify=(pd.DataFrame({"g":groups,"y":labels}).groupby("g")["y"].first().reindex(all_subjs).values))
    deploy_model = GCNTransformerASD(num_classes=N_CLASSES, use_bone_stream=True).to(DEVICE)
    deploy_model = train_model(deploy_model, SkeletonDataset(_gather(j_data, np.isin(groups, dep_tr)), _gather(b_data, np.isin(groups, dep_tr)), labels[np.isin(groups, dep_tr)], augment=True, contrastive_mode=True), SkeletonDataset(_gather(j_data, np.isin(groups, dep_vl)), _gather(b_data, np.isin(groups, dep_vl)), labels[np.isin(groups, dep_vl)], augment=False, contrastive_mode=False))
    return deploy_model, dep_tr, dep_vl

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_pipeline():
    print("="*60); print(" STEP A: BUILD DATASET"); print("="*60)
    j_data, b_data, hc_features, labels, groups = build_dataset(DATA_DIR)
    if len(j_data) == 0: print("❌ No clips processed."); return {}
    N_CLASSES = len(np.unique(labels)); IS_BINARY = N_CLASSES == 2
    _tmp = GCNTransformerASD(num_classes=N_CLASSES, use_bone_stream=True); n_params = sum(p.numel() for p in _tmp.parameters()); print(f"\n  📐 Model: {n_params:,} params (~{n_params/1e3:.0f}K)"); del _tmp

    print(f"\n{'='*60}\n  STEP B: MULTI-SEED 5-FOLD CV ({len(SEEDS)} seeds)\n{'='*60}")
    seed_results = []
    for seed in SEEDS: print(f"\n{'#'*60}\n  SEED {seed}\n{'#'*60}"); r = run_one_seed(seed, j_data, b_data, hc_features, labels, groups, N_CLASSES); seed_results.append(r)

    print(f"\n{'='*60}\n  STEP C: AGGREGATED RESULTS\n{'='*60}")
    accs = np.array([r["cv_acc"] for r in seed_results]); aucs = np.array([r["cv_auc"] for r in seed_results if r["cv_auc"] is not None]) if any(r["cv_auc"] is not None for r in seed_results) else None; weights_emb = [r["w_emb"] for r in seed_results]
    print(f"\n  Per-seed CV accuracy: {[f'{a:.4f}' for a in accs]}"); print(f"  Mean ± std:           {accs.mean():.4f} ± {accs.std():.4f}")
    if aucs is not None: print(f"\n  Per-seed CV AUC:      {[f'{a:.4f}' for a in aucs]}"); print(f"  Mean ± std:           {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"\n  Per-seed EMB weight:  {[f'{w:.2f}' for w in weights_emb]}")

    print(f"\n  📊 Averaging OOF predictions across seeds...")
    oof_emb_avg = np.mean([r["oof_emb"] for r in seed_results], axis=0); oof_hc_avg = np.mean([r["oof_hc"] for r in seed_results], axis=0)
    w_emb_avg = float(np.mean(weights_emb)); w_hc_avg = 1.0 - w_emb_avg; oof_combined_avg = w_emb_avg * oof_emb_avg + w_hc_avg * oof_hc_avg

    sp_e, sp_e_std, sl, _ = aggregate_predictions_to_subject(oof_emb_avg, labels, groups)
    sp_h, sp_h_std, _, _ = aggregate_predictions_to_subject(oof_hc_avg, labels, groups)
    X_meta = np.column_stack([sp_e[:, 1], sp_e_std[:, 1], sp_h[:, 1], sp_h_std[:, 1]])
    meta_final = LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=500).fit(X_meta, sl)
    sp_combined = meta_final.predict_proba(X_meta)
    cv_acc_avg = accuracy_score(sl, sp_combined.argmax(axis=1)); cv_auc_avg = roc_auc_score(sl, sp_combined[:, 1]) if IS_BINARY else None
    print(f"\n  Subject-aggregated CV on AVERAGED OOF: Acc={cv_acc_avg:.4f}" + (f", AUC={cv_auc_avg:.4f}" if cv_auc_avg else ""))
    print(classification_report(sl, sp_combined.argmax(axis=1), digits=4))

    p_teacher_oof_clip = oof_combined_avg[:, 1] if IS_BINARY and oof_combined_avg.shape[1] >= 2 else oof_combined_avg.argmax(axis=1).astype(float)

    print(f"\n{'='*60}\n  STEP D: DEPLOYMENT MODEL\n{'='*60}")
    deploy_model, dep_tr_subjs, dep_vl_subjs = train_deployment(j_data, b_data, labels, groups, N_CLASSES, SEEDS[0])
    m_dep_train = np.isin(groups, dep_tr_subjs); m_dep_ens_train = np.isin(groups, np.concatenate([dep_tr_subjs, dep_vl_subjs]))
    print("  🧠 Embedding all clips (with TTA)..."); emb_all = extract_all_embeddings(deploy_model, SkeletonDataset(j_data, b_data, labels, augment=False, contrastive_mode=False))

    emb_train = emb_all[m_dep_train]
    if emb_train.shape[1] > 100:
        pca_dep = PCA(n_components=0.99, random_state=SEEDS[0])
        pca_dep.fit(emb_train); emb_all_p = pca_dep.transform(emb_all)
        print(f"    Deploy PCA: {emb_all.shape[1]} → {emb_all_p.shape[1]} components (99% variance)")
    else:
        pca_dep = None; emb_all_p = emb_all

    keep_emb_d = variance_filter(emb_all_p[m_dep_train]); Xe_dep = emb_all_p[:, keep_emb_d]

    hc_train = hc_features[m_dep_train]; keep_hc_d1 = variance_filter(hc_train); Xh_d1_all = hc_features[:, keep_hc_d1]; names_d1 = [HC_FEATURE_NAMES[i] for i in keep_hc_d1]
    sx, sy = [], []
    for g in np.unique(groups[m_dep_train]): m = (groups == g) & m_dep_train; sx.append(Xh_d1_all[m].mean(axis=0)); sy.append(labels[m][0])
    sx, sy = np.array(sx), np.array(sy); keep_hc_d2 = mann_whitney_filter(sx, sy, alpha=0.01); Xh_d2_all = Xh_d1_all[:, keep_hc_d2]; names_d2 = [names_d1[i] for i in keep_hc_d2]
    keep_hc_d3 = remove_correlated_features(sx[:, keep_hc_d2], names_d2); Xh_dep = Xh_d2_all[:, keep_hc_d3]; hc_names_dep = [names_d2[i] for i in keep_hc_d3]

    ens_train_subjs = np.concatenate([dep_tr_subjs, dep_vl_subjs]); strat_ens = pd.DataFrame({"g":groups[m_dep_ens_train], "y":labels[m_dep_ens_train]}).groupby("g")["y"].first().reindex(ens_train_subjs).values
    tr_s, vl_s = train_test_split(ens_train_subjs, test_size=0.15, random_state=SEEDS[0], stratify=strat_ens); m_tr = np.isin(groups, tr_s); m_vl = np.isin(groups, vl_s)
    print("  🌳 Training ensembles..."); emb_models_dep, sc_emb_dep = train_ensemble_clip(Xe_dep[m_tr], labels[m_tr], Xe_dep[m_vl], labels[m_vl]); hc_models_dep, sc_hc_dep = train_ensemble_clip(Xh_dep[m_tr], labels[m_tr], Xh_dep[m_vl], labels[m_vl])

    ebm, sc_ebm, ebm_idx_in_dep, ebm_names, fidelity = build_distilled_ebm(Xh_dep, p_teacher_oof_clip, hc_names_dep)

    print("\n" + "="*60); print(" STEP E: SAVING ARTIFACTS"); print("="*60); os.makedirs(OUTPUT_DIR, exist_ok=True)
    idx_after_var = np.array(keep_hc_d1); idx_after_mw = idx_after_var[np.array(keep_hc_d2)]; idx_after_corr = idx_after_mw[np.array(keep_hc_d3)]
    artifact = {"version": "v41_paper", "n_params": int(n_params), "deploy_model_path": os.path.join(OUTPUT_DIR, "v41_paper_model.pth"), "pca_deploy": pca_dep, "emb_keep_indices": keep_emb_d, "hc_final_indices_in_full": idx_after_corr.tolist(), "hc_names_final": hc_names_dep, "scaler_emb": sc_emb_dep, "scaler_hc": sc_hc_dep, "scaler_ebm": sc_ebm, "emb_models": emb_models_dep, "hc_models": hc_models_dep, "ebm": ebm, "ebm_sel_idx_in_dep_hc": ebm_idx_in_dep, "ebm_sel_names": ebm_names, "ebm_fidelity": fidelity, "weight_emb": w_emb_avg, "weight_hc": w_hc_avg, "seeds": SEEDS, "per_seed_acc": accs.tolist(), "cv_acc_mean": float(accs.mean()), "cv_acc_std": float(accs.std()), "cv_auc_mean": float(aucs.mean()) if aucs is not None else None, "cv_auc_std": float(aucs.std()) if aucs is not None else None, "cv_acc_avg_oof": cv_acc_avg, "cv_auc_avg_oof": cv_auc_avg, "feature_names_full": list(HC_FEATURE_NAMES)}
    joblib.dump(artifact, os.path.join(OUTPUT_DIR, "asd_gait_v41_paper.joblib")); torch.save(deploy_model.state_dict(), artifact["deploy_model_path"])

    summary = {"version": "v41_paper", "cv_acc_mean": round(float(accs.mean()), 4), "cv_acc_std": round(float(accs.std()), 4)}
    with open(os.path.join(OUTPUT_DIR, "v41_paper_summary.json"), "w") as f: json.dump(summary, f, indent=2)
    print(f"\n  ✅ Saved to {OUTPUT_DIR}\n  📊 PAPER NUMBERS:\n     CV Acc: {accs.mean():.4f} ± {accs.std():.4f}\n     EBM fidelity: r = {fidelity:.4f}")
    return summary

if __name__ == "__main__":
    results = run_pipeline()
