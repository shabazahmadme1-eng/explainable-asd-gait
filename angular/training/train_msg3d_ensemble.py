"""
Angular MS-G3D, MULTI-SEED ENSEMBLE  (the safe accuracy lever).
================================================================
Trains the angular (unit-bone) MS-G3D over several random seeds, each with strict
grouped 10-fold CV by child, then AVERAGES the out-of-fold predictions across seeds.
Ensembling reduces init/augmentation variance -> smoother, usually higher AUC and
steadier accuracy, with NO change to the input representation (so it stays
phone-deployable) and NO leakage.

If  hc_oof_800.npy  is next to this file, it also prints the FINAL FUSED number
(0.5*HC + 0.5*angular-ensemble) directly, so you see the real accuracy in one run.

Put next to this file:  dataset_800.npz , msg3d_pkg.zip , (optional) hc_oof_800.npy
Run on DGX/Colab GPU. ~SEEDS x single-seed time (3 seeds ~ 50 min on an A100 MIG;
lower SEEDS or EPOCHS to go faster). Saves: oof_msg3d_ens.npy
"""
import os, sys, zipfile, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# --- locate + import the MS-G3D package ------------------------------------ #
def _locate_pkg():
    for c in ('msg3d_pkg', './msg3d_pkg'):
        if os.path.isdir(os.path.join(c, 'model')): return c
    if os.path.exists('msg3d_pkg.zip'):
        with zipfile.ZipFile('msg3d_pkg.zip') as z: z.extractall('msg3d_pkg')
        return 'msg3d_pkg'
    for c in ('ms-g3d', '../ms-g3d'):
        if os.path.isdir(os.path.join(c, 'model')): return c
    raise FileNotFoundError('Put msg3d_pkg.zip (or ms-g3d/) next to this script.')
sys.path.insert(0, _locate_pkg())
from model.msg3d import Model                                    # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'; _CUDA = (DEVICE == 'cuda')
try:
    from torch.amp import autocast as _ac, GradScaler as _GS
    def make_scaler(): return _GS('cuda', enabled=_CUDA)
    def amp_ctx():     return _ac('cuda', enabled=_CUDA)
except Exception:
    def make_scaler(): return torch.cuda.amp.GradScaler(enabled=_CUDA)
    def amp_ctx():     return torch.cuda.amp.autocast(enabled=_CUDA)

# ----------------------------- config ------------------------------------- #
SEEDS   = [0, 1, 2]     # <- ensemble members. drop to [0,1] for a faster run.
T_LEN   = 100
EPOCHS  = 30
BATCH   = 64
LR      = 1e-3
WD      = 5e-4
N_FOLDS = 10
WORKERS = 4
try: torch.multiprocessing.set_sharing_strategy('file_system')
except Exception: pass
def _find(n):
    for c in (n, f'./{n}', f'data/{n}'):
        if os.path.exists(c): return c
    raise FileNotFoundError(n)
DATA = _find('dataset_800.npz')
print('device:', DEVICE, '| torch', torch.__version__, '| seeds', SEEDS)

# --- NTU geometry (verbatim from backend/config.py) ------------------------ #
USER_JOINT_COL = {
    'SpineMid':0,'AnkleLeft':3,'AnkleRight':6,'ElbowLeft':9,'ElbowRight':12,
    'FootLeft':15,'FootRight':18,'HandLeft':21,'HandRight':24,'HandTipLeft':27,
    'HandTipRight':30,'Head':33,'HipLeft':36,'HipRight':39,'KneeLeft':42,
    'KneeRight':45,'Neck':48,'ShoulderLeft':51,'ShoulderRight':54,'SpineBase':57,
    'SpineShoulder':60,'ThumbLeft':63,'ThumbRight':66,'WristLeft':69,'WristRight':72}
NTU_JOINT_NAMES = ['SpineBase','SpineMid','Neck','Head','ShoulderLeft','ElbowLeft',
    'WristLeft','HandLeft','HandTipLeft','ThumbLeft','ShoulderRight','ElbowRight',
    'WristRight','HandRight','HandTipRight','ThumbRight','HipLeft','KneeLeft',
    'AnkleLeft','FootLeft','HipRight','KneeRight','AnkleRight','FootRight','SpineShoulder']
NTU_REORDER_COLS = []
for _n in NTU_JOINT_NAMES:
    c = USER_JOINT_COL[_n]; NTU_REORDER_COLS += [c, c+1, c+2]
NTU_PARENTS = [0,0,24,2,24,4,5,6,7,6,24,10,11,12,13,12,0,16,17,18,0,20,21,22,1]
CENTER, SPINE_SHLD = 0, 24
LR_PAIRS = [(4,10),(5,11),(6,12),(7,13),(8,14),(9,15),(16,20),(17,21),(18,22),(19,23)]

d = np.load(DATA, allow_pickle=True)
Xraw = d['X150'].astype(np.float32); y = (d['y'] == 'ASD').astype(np.int64); groups = d['groups']
print('clips', Xraw.shape, '| kids', len(set(groups.tolist())),
      '| TD', int((y==0).sum()), 'ASD', int((y==1).sum()))

def to_ntu(x):
    ntu = x[:, NTU_REORDER_COLS].reshape(150, 25, 3)
    ntu = ntu[np.linspace(0, 149, T_LEN).astype(int)]
    ntu = ntu - ntu[:, CENTER:CENTER+1, :]
    torso = np.linalg.norm(ntu[:, SPINE_SHLD, :], axis=-1).mean() + 1e-6
    return ntu / torso

def augment(ntu):
    ntu = ntu.copy(); a = np.random.uniform(-0.30, 0.30); ca, sa = np.cos(a), np.sin(a)
    x, z = ntu[..., 0].copy(), ntu[..., 2].copy(); ntu[..., 0] = ca*x + sa*z; ntu[..., 2] = -sa*x + ca*z
    ntu *= np.random.uniform(0.90, 1.10); ntu += np.random.normal(0, 0.01, ntu.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        ntu[..., 0] *= -1
        for a_, b_ in LR_PAIRS: ntu[:, [a_, b_]] = ntu[:, [b_, a_]]
    return ntu

def bone(ntu):
    b = ntu - ntu[:, NTU_PARENTS, :]
    return b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-6)

def to_tensor(ntu):
    t = np.transpose(bone(ntu), (2, 0, 1))[:, :, :, None]
    return torch.from_numpy(np.ascontiguousarray(t)).float()

class DS(Dataset):
    def __init__(self, idx, train): self.idx, self.train = idx, train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]; ntu = to_ntu(Xraw[j])
        if self.train: ntu = augment(ntu)
        return to_tensor(ntu), y[j], j

def make_model():
    return Model(num_class=2, num_point=25, num_person=1, num_gcn_scales=13,
                 num_g3d_scales=6, graph='graph.ntu_rgb_d.AdjMatrixGraph', in_channels=3).to(DEVICE)

def train_one(m, idx):
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1); sc = make_scaler()
    dl = DataLoader(DS(idx, True), batch_size=BATCH, shuffle=True, num_workers=WORKERS,
                    pin_memory=_CUDA, drop_last=False)
    for _ in range(EPOCHS):
        m.train()
        for xb, yb, _ in dl:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with amp_ctx(): loss = crit(m(xb), yb)
            sc.scale(loss).backward(); sc.step(opt); sc.update()
        sch.step()

@torch.no_grad()
def predict(m, idx):
    m.eval(); out = np.zeros((len(y), 2), np.float32)
    for xb, yb, jb in DataLoader(DS(idx, False), batch_size=BATCH, num_workers=WORKERS, pin_memory=_CUDA):
        with amp_ctx(): p = torch.softmax(m(xb.to(DEVICE)).float(), 1)
        out[jb.numpy()] = p.cpu().numpy()
    return out

kids = sorted(set(groups.tolist()))
ky = np.array([y[groups == k][0] for k in kids])
def kagg(p): return np.array([p[groups == k].mean() for k in kids])
def kid_metrics(p_kid):
    pr = (p_kid >= 0.5).astype(int); tn, fp, fn, tp = confusion_matrix(ky, pr, labels=[0,1]).ravel()
    return roc_auc_score(ky, p_kid), accuracy_score(ky, pr), tp, fn, tn, fp

def grouped_cv(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if _CUDA: torch.cuda.manual_seed_all(seed)
    gkf = GroupKFold(N_FOLDS); oof = np.zeros((len(y), 2), np.float32); t0 = time.time()
    for f, (tr, te) in enumerate(gkf.split(Xraw, y, groups)):
        m = make_model(); train_one(m, tr); oof[te] = predict(m, te)[te]
        del m
        if _CUDA: torch.cuda.empty_cache()
    return oof, time.time() - t0

# ----------------- run each seed, then ensemble --------------------------- #
oof_seeds = []
for s in SEEDS:
    oof, dt = grouped_cv(s)
    oof_seeds.append(oof)
    auc, acc, tp, fn, tn, fp = kid_metrics(kagg(oof[:, 1]))
    print(f'  seed {s}: angular kid-AUC {auc:.3f} acc {acc:.3f}  ({dt:.0f}s)')

ens = np.mean(oof_seeds, axis=0)                     # (800,2) ensemble OOF
np.save('oof_msg3d_ens.npy', ens)
auc, acc, tp, fn, tn, fp = kid_metrics(kagg(ens[:, 1]))
print(f'\n=== angular ENSEMBLE ({len(SEEDS)} seeds), kid-level ===')
print(f'  AUC {auc:.3f}  acc {acc:.3f}  (single-seed was ~0.972 / 0.910)')

# ----------------- fuse with HC if the OOF is here ------------------------ #
if os.path.exists('hc_oof_800.npy'):
    hc = np.load('hc_oof_800.npy')                   # per-clip HC OOF, dataset order
    for w, tag in [(0.5, 'equal 0.5/0.5')]:
        fused = kagg(w * hc + (1 - w) * ens[:, 1])
        a, ac, tp, fn, tn, fp = kid_metrics(fused)
        print(f'\n=== FUSED  HC + angular-ensemble  ({tag}) ===')
        print(f'  AUC {a:.3f}  acc {ac:.3f}  sens {tp}/{tp+fn}  spec {tn}/{tn+fp}')
        print(f'  (single-seed fusion baseline was AUC 0.980 / acc 0.920)')
    print('\n--> if this beats 0.980 / 0.92, the ensemble is the honest higher number.')
else:
    print('\n(hc_oof_800.npy not found -> download oof_msg3d_ens.npy and fuse locally.)')
print('\nDone. Saved oof_msg3d_ens.npy.')
