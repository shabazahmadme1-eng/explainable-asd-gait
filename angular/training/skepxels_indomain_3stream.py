"""
IN-DOMAIN 3-STREAM ensemble  (reproduce the ~96% headline).
===========================================================
This is the Kinect in-domain number that shows the ACCURACY GAIN of adding
Skepxels as a third stream -- separate from the phone/field test (train_skepxels_dann.py).

  stream 1  HC            -> hc_oof_800.npy   (per-clip P, already on disk)
  stream 2  angular MSG3D -> oof_msg3d.npy    (per-clip softmax, already on disk)
  stream 3  Skepxels CNN  -> trained HERE, grouped 10-fold CV, saved as skepxels_oof_800.npy

Then it aggregates every stream to kid level and fuses:
  2-stream = 0.5*HC + 0.5*angular            (the deployed baseline, ~92%)
  3-stream = (HC + angular + Skepxels)/3      (equal weight, honest)   -> expect ~96%
It also reports the best simplex weight, clearly labelled as in-sample/optimistic.

NOTE: this Skepxels stream is Kinect-only (does NOT transfer to phone on its own).
The 3-stream is deployable on phone ONLY if train_skepxels_dann.py shows the
domain-adversarial Skepxels de-saturates on the 14 field kids. This script proves
the in-domain gain; that script proves it's safe to ship.

Put next to this file:  dataset_800.npz , hc_oof_800.npy , oof_msg3d.npy
Run on DGX/Colab GPU (~30-45 min: 10 folds x 224x224). Saves: skepxels_oof_800.npy
"""
import os, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import torchvision.models as tvm
try: torch.multiprocessing.set_sharing_strategy('file_system')
except Exception: pass

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; _CUDA=(DEVICE=='cuda')
IMG=224; EPOCHS=25; BATCH=48; LR=3e-4; WD=1e-4; N_FOLDS=10; WORKERS=0; SEED=0
torch.manual_seed(SEED); np.random.seed(SEED)
if _CUDA: torch.cuda.manual_seed_all(SEED)
def _find(n):
    for c in (n,f'./{n}',f'data/{n}'):
        if os.path.exists(c): return c
    raise FileNotFoundError(n)
try:
    from torch.amp import autocast as _ac, GradScaler as _GS
    def scaler(): return _GS('cuda',enabled=_CUDA)
    def amp(): return _ac('cuda',enabled=_CUDA)
except Exception:
    def scaler(): return torch.cuda.amp.GradScaler(enabled=_CUDA)
    def amp(): return torch.cuda.amp.autocast(enabled=_CUDA)
print('device:',DEVICE,'| torch',torch.__version__)

d=np.load(_find('dataset_800.npz'),allow_pickle=True)
Xraw=d['X150'].astype(np.float32); y=(d['y']=='ASD').astype(np.int64); groups=d['groups']
print('clips',Xraw.shape,'| kids',len(set(groups.tolist())),'| TD',int((y==0).sum()),'ASD',int((y==1).sum()))

# --- coordinate-Skepxels image (raw joint order) --------------------------- #
LR_PAIRS=[(1,2),(3,4),(5,6),(7,8),(9,10),(12,13),(14,15),(17,18),(21,22),(23,24)]
def normalize(js):
    js=js-js[:,19:20,:]; torso=np.linalg.norm(js[:,20,:],axis=-1).mean()+1e-6; return js/torso
def aug_geo(js):
    js=js.copy(); a=np.random.uniform(-.30,.30); ca,sa=np.cos(a),np.sin(a)
    x,z=js[...,0].copy(),js[...,2].copy(); js[...,0]=ca*x+sa*z; js[...,2]=-sa*x+ca*z
    js*=np.random.uniform(.90,1.10); js+=np.random.normal(0,.01,js.shape).astype(np.float32)
    if np.random.rand()<.5:
        js[...,0]*=-1
        for a_,b_ in LR_PAIRS: js[:,[a_,b_]]=js[:,[b_,a_]]
    return js
def skel_image(js):
    img=np.transpose(js,(2,1,0)); t=torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0)
    t=torch.nn.functional.interpolate(t,size=(IMG,IMG),mode='bilinear',align_corners=False)[0]
    for c in range(3): t[c]=(t[c]-t[c].mean())/(t[c].std()+1e-6)
    return t

class DS(Dataset):
    def __init__(self,idx,train): self.idx,self.train=idx,train
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        j=self.idx[i]; js=normalize(Xraw[j].reshape(150,25,3))
        if self.train: js=aug_geo(js)
        return skel_image(js), y[j], j

def make_model():
    net=tvm.shufflenet_v2_x1_0(weights='IMAGENET1K_V1'); net.fc=nn.Linear(net.fc.in_features,2); return net.to(DEVICE)

def train_one(m,idx):
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    crit=nn.CrossEntropyLoss(label_smoothing=0.1); sc=scaler()
    dl=DataLoader(DS(idx,True),batch_size=BATCH,shuffle=True,num_workers=WORKERS,pin_memory=_CUDA,drop_last=False)
    for _ in range(EPOCHS):
        m.train()
        for xb,yb,_ in dl:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
            with amp(): loss=crit(m(xb),yb)
            sc.scale(loss).backward(); sc.step(opt); sc.update()
        sch.step()

@torch.no_grad()
def predict(m,idx):
    m.eval(); out=np.zeros(len(y),np.float32)
    for xb,yb,jb in DataLoader(DS(idx,False),batch_size=BATCH,num_workers=WORKERS,pin_memory=_CUDA):
        with amp(): p=torch.softmax(m(xb.to(DEVICE)).float(),1)[:,1]
        out[jb.numpy()]=p.cpu().numpy()
    return out

# --- grouped 10-fold CV -> Skepxels OOF ------------------------------------ #
gkf=GroupKFold(N_FOLDS); skep_oof=np.zeros(len(y),np.float32)
print('training Skepxels (grouped 10-fold)...')
for f,(tr,te) in enumerate(gkf.split(Xraw,y,groups)):
    m=make_model(); train_one(m,tr); skep_oof[te]=predict(m,te)[te]
    del m
    if _CUDA: torch.cuda.empty_cache()
    print(f'  fold {f+1}/{N_FOLDS} done')
np.save('skepxels_oof_800.npy',skep_oof)

# --- fuse all three at kid level ------------------------------------------- #
hc=np.load(_find('hc_oof_800.npy')).astype(np.float32)
mg=np.load(_find('oof_msg3d.npy')).astype(np.float32); mg=mg[:,1] if mg.ndim==2 else mg
kids=sorted(set(groups.tolist())); ky=np.array([y[groups==k][0] for k in kids])
def kagg(p): return np.array([p[groups==k].mean() for k in kids])
HC,ANG,SKEP=kagg(hc),kagg(mg),kagg(skep_oof)
def report(tag,p):
    pr=(p>=0.5).astype(int); tn,fp,fn,tp=confusion_matrix(ky,pr,labels=[0,1]).ravel()
    print(f'  {tag:26s} AUC {roc_auc_score(ky,p):.3f}  acc {accuracy_score(ky,pr):.3f}  sens {tp}/{tp+fn}  spec {tn}/{tn+fp}')

print('\n=== IN-DOMAIN, kid-level (100 children) ===')
report('Skepxels only',SKEP)
report('2-stream HC+ANG (deployed)',0.5*HC+0.5*ANG)
report('3-stream equal (HC+ANG+SKEP)',(HC+ANG+SKEP)/3.0)

# best simplex weight -- OPTIMISTIC (fit on the same kids); shown for reference only
best=(-1,None)
for a in np.linspace(0,1,21):
    for b in np.linspace(0,1-a,int(round((1-a)*20))+1):
        c=1-a-b
        if c<-1e-9: continue
        p=a*HC+b*ANG+c*SKEP; au=roc_auc_score(ky,p)
        if au>best[0]: best=(au,(round(a,2),round(b,2),round(max(c,0),2)))
print(f'\n  best simplex weight (HC,ANG,SKEP)={best[1]}  AUC {best[0]:.3f}   [in-sample/optimistic]')
print('\n--> headline = the 3-stream EQUAL row. If it clears the 2-stream, that is the honest gain.')
print('Done. Saved skepxels_oof_800.npy.')
