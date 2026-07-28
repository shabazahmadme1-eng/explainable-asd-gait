"""
Paper Stream-2 : MS-G3D on angular (unit-bone) features  —  DGX / Jupyter.
=========================================================================
Reproduces the Shin et al. 2025 second stream: coordinate-invariant angular
features -> MS-G3D graph network, on OUR 800-clip cache, with STRICT grouped
10-fold CV by child. Trained FROM SCRATCH each fold (no pretrained init, so no
leakage from the Kinect-fine-tuned weights). Honest OOF, no in-sample numbers.

WHY "angular" and not raw coordinates
-------------------------------------
The paper feeds MS-G3D angular features (cosine similarity between normalized
joint vectors), which are translation/scale invariant. We implement that as
UNIT-normalized bone vectors (each joint's direction-cosine to its parent) on a
spine-centered, torso-scaled skeleton. This is the invariant input the paper
uses — NOT the raw-coordinate MS-G3D that went out-of-distribution off Kinect.

HOW TO RUN (DGX / Jupyter)
--------------------------
Put these 3 files in the SAME folder as your notebook (or point the paths below):
    dataset_800.npz          the 800-clip cache
    msg3d_pkg.zip            the MS-G3D code  (or copy the whole `ms-g3d/` folder)
    train_msg3d_colab.py     this file        (or just paste its code into a cell)
Then run the cell / `python train_msg3d_colab.py`. It auto-locates the package
and data, runs all 10 folds, and saves:
    oof_msg3d.npy            out-of-fold P(ASD) per clip, for fusion
    final_msg3d_*.pt         weights trained on all 800 clips

On an A100/H100 the defaults below (BATCH=64) finish 10 folds in ~15-30 min.
Pick one GPU with e.g.  os.environ['CUDA_VISIBLE_DEVICES']='0'  before torch runs.
The paper flags this stream as the weaker/overfitting one — its value shows up
after we FUSE it with the ShuffleNet(Skepxels) and handcrafted streams locally.
"""
import os, sys, zipfile, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# --- locate + import the MS-G3D package (zip, unpacked dir, or full repo) --- #
def _locate_pkg():
    for c in ('msg3d_pkg', './msg3d_pkg'):
        if os.path.isdir(os.path.join(c, 'model')): return c
    if os.path.exists('msg3d_pkg.zip'):
        with zipfile.ZipFile('msg3d_pkg.zip') as z: z.extractall('msg3d_pkg')
        return 'msg3d_pkg'
    for c in ('ms-g3d', '../ms-g3d', './ms-g3d'):
        if os.path.isdir(os.path.join(c, 'model')): return c
    raise FileNotFoundError('Put msg3d_pkg.zip (or the ms-g3d/ folder) next to this script.')
sys.path.insert(0, _locate_pkg())
from model.msg3d import Model                    # noqa: E402

# --- AMP API that works on both old and new PyTorch ------------------------ #
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_USE_CUDA = (DEVICE == 'cuda')
try:                                             # torch >= 2.4 preferred API
    from torch.amp import autocast as _ac, GradScaler as _GS
    def make_scaler(): return _GS('cuda', enabled=_USE_CUDA)
    def amp_ctx():     return _ac('cuda', enabled=_USE_CUDA)
except Exception:                                # older torch fallback
    def make_scaler(): return torch.cuda.amp.GradScaler(enabled=_USE_CUDA)
    def amp_ctx():     return torch.cuda.amp.autocast(enabled=_USE_CUDA)

# ----------------------------- config ------------------------------------- #
T_LEN    = 100        # temporal length after resample (raw is 150). DGX can afford it.
EPOCHS   = 30
BATCH    = 64         # A100/H100-sized. Drop to 24/16 on smaller GPUs / if OOM.
LR       = 1e-3
WD       = 5e-4
N_FOLDS  = 10
WORKERS  = 8          # DGX has plenty of CPU; raise/lower to taste.
STREAMS  = ['bone']   # 'bone' = angular (paper Stream-2). add 'joint' for the
                      # full MS-G3D two-stream ensemble (doubles the runtime).
def _find(name, extra=()):
    for c in (name, f'./{name}', f'data/{name}', *extra):
        if os.path.exists(c): return c
    raise FileNotFoundError(f'{name} not found next to this script.')
DATA = _find('dataset_800.npz')
print('device:', DEVICE, '| torch', torch.__version__,
      '| gpu', torch.cuda.get_device_name(0) if _USE_CUDA else '-')

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
CENTER      = 0       # SpineBase (NTU order)
SPINE_SHLD  = 24      # SpineShoulder (NTU order) -> torso vector after centering
LR_PAIRS    = [(4,10),(5,11),(6,12),(7,13),(8,14),(9,15),
               (16,20),(17,21),(18,22),(19,23)]

# --------------------------- load the data -------------------------------- #
d = np.load(DATA, allow_pickle=True)
Xraw = d['X150'].astype(np.float32)              # (800,150,75)
y = (d['y'] == 'ASD').astype(np.int64)
groups = d['groups']
print('clips', Xraw.shape, '| kids', len(set(groups.tolist())),
      '| TD', int((y==0).sum()), 'ASD', int((y==1).sum()))

def to_ntu(x):                                   # (150,75) -> (T_LEN,25,3) invariant
    ntu = x[:, NTU_REORDER_COLS].reshape(150, 25, 3)
    idx = np.linspace(0, 149, T_LEN).astype(int)
    ntu = ntu[idx]
    ntu = ntu - ntu[:, CENTER:CENTER+1, :]                    # center on spine base
    torso = np.linalg.norm(ntu[:, SPINE_SHLD, :], axis=-1).mean() + 1e-6
    return ntu / torso                                        # scale by torso length

def augment(ntu):                                # rot-y / scale / jitter / mirror
    ntu = ntu.copy()
    a = np.random.uniform(-0.30, 0.30); ca, sa = np.cos(a), np.sin(a)
    x, z = ntu[..., 0].copy(), ntu[..., 2].copy()
    ntu[..., 0] = ca*x + sa*z; ntu[..., 2] = -sa*x + ca*z
    ntu *= np.random.uniform(0.90, 1.10)
    ntu += np.random.normal(0, 0.01, ntu.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        ntu[..., 0] *= -1
        for a_, b_ in LR_PAIRS:
            ntu[:, [a_, b_]] = ntu[:, [b_, a_]]
    return ntu

def bone(ntu):                                   # (T,25,3) -> unit bone vectors (angular)
    b = ntu - ntu[:, NTU_PARENTS, :]
    return b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-6)

def to_tensor(ntu, stream):                      # (T,25,3) -> (3,T,25,1) for MS-G3D
    js = bone(ntu) if stream == 'bone' else ntu
    t = np.transpose(js, (2, 0, 1))[:, :, :, None]            # (C,T,V,M=1)
    return torch.from_numpy(np.ascontiguousarray(t)).float()

class SkelDS(Dataset):
    def __init__(self, idx, stream, train):
        self.idx, self.stream, self.train = idx, stream, train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]; ntu = to_ntu(Xraw[j])
        if self.train: ntu = augment(ntu)
        return to_tensor(ntu, self.stream), y[j], j

def make_model():
    return Model(num_class=2, num_point=25, num_person=1,
                 num_gcn_scales=13, num_g3d_scales=6,
                 graph='graph.ntu_rgb_d.AdjMatrixGraph', in_channels=3).to(DEVICE)

def train_one(model, idx, stream, epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = make_scaler()
    dl = DataLoader(SkelDS(idx, stream, True), batch_size=BATCH, shuffle=True,
                    num_workers=WORKERS, pin_memory=_USE_CUDA, drop_last=False)
    for ep in range(epochs):
        model.train()
        for xb, yb, _ in dl:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with amp_ctx():
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()

@torch.no_grad()
def predict(model, idx, stream):
    model.eval(); out = np.zeros((len(y), 2), np.float32)
    for xb, yb, jb in DataLoader(SkelDS(idx, stream, False), batch_size=BATCH,
                                 num_workers=WORKERS, pin_memory=_USE_CUDA):
        with amp_ctx():
            p = torch.softmax(model(xb.to(DEVICE)).float(), 1)
        out[jb.numpy()] = p.cpu().numpy()
    return out

def kid_scores(p):
    kids = sorted(set(groups.tolist()))
    ky = np.array([y[groups == k][0] for k in kids])
    kp = np.array([p[groups == k].mean() for k in kids])
    return ky, kp

# ------------------- grouped 10-fold CV per stream ------------------------ #
def run(stream):
    print(f'\n===== MS-G3D [{stream}] : grouped {N_FOLDS}-fold CV =====')
    gkf = GroupKFold(N_FOLDS); oof = np.zeros((len(y), 2), np.float32); t0 = time.time()
    for f, (tr, te) in enumerate(gkf.split(Xraw, y, groups)):
        m = make_model()
        train_one(m, tr, stream, EPOCHS)
        pr = predict(m, te, stream)
        oof[te] = pr[te]
        del m
        if _USE_CUDA: torch.cuda.empty_cache()
        print(f'  fold {f+1}/{N_FOLDS} done  ({time.time()-t0:.0f}s)')
    p = oof[:, 1]; ky, kp = kid_scores(p)
    print(f'  CLIP-level : AUC {roc_auc_score(y, p):.3f}  acc {accuracy_score(y, p>=.5):.3f}')
    print(f'  KID-level  : AUC {roc_auc_score(ky, kp):.3f}  acc {accuracy_score(ky, kp>=.5):.3f}')
    np.save(f'oof_msg3d_{stream}.npy', oof)
    m = make_model(); train_one(m, np.arange(len(y)), stream, EPOCHS)
    torch.save(m.state_dict(), f'final_msg3d_{stream}.pt'); del m
    if _USE_CUDA: torch.cuda.empty_cache()
    return oof

oofs = {s: run(s) for s in STREAMS}

# combine streams -> the single "MS-G3D stream" OOF
p = np.mean([o[:, 1] for o in oofs.values()], axis=0)
np.save('oof_msg3d.npy', np.stack([1-p, p], 1))
ky, kp = kid_scores(p)
print(f'\n===== MS-G3D stream ({"+".join(oofs)}) =====')
print(f'  KID-level : AUC {roc_auc_score(ky, kp):.3f}  acc {accuracy_score(ky, kp>=.5):.3f}')
print('\nSaved oof_msg3d.npy + final_msg3d_*.pt. Bring oof_msg3d.npy back to fuse locally.')
