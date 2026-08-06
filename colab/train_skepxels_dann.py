"""
Skepxels rescue #3 : DOMAIN-ADVERSARIAL (DANN) + domain randomization.
======================================================================
Goal: make the Skepxels ShuffleNet SENSOR-INVARIANT so it transfers to phone.
Unlike plain MediaPipe fine-tuning (which memorised lab-MediaPipe and still
flooded real-phone TDs), DANN uses a gradient-reversal domain head to force the
feature extractor to learn features a Kinect-vs-MediaPipe classifier CANNOT tell
apart -- i.e. invariant to the sensor -- which has a real shot at generalising to
UNSEEN field phone captures.

  domains:  Kinect (dataset_800, labelled)   vs   MediaPipe (mp_train_99, labelled)
  heads:    label head (ASD/TD)  +  domain head via Gradient Reversal Layer
  extra:    domain randomisation on the Kinect source (simulate MediaPipe artifacts)
  test:     the 14 field phone kids (res_val_14.npz) -- does Skepxels DE-SATURATE,
            and does adding it as a 3RD STREAM beat the deployed 2-stream (HC+angular)?

This is a FEASIBILITY run (single train on all data), not CV: the question is only
"do the field TDs stop reading ~1.0, and does the 3-stream AUC clear the 2-stream?".
If yes, we move to a proper grouped-CV run.

Put next to this file:
    dataset_800.npz , mp_train_99.npz , res_val_14.npz   (required)
    field_hc_ang_14.npz                                  (HC+angular field probs;
        pre-computed locally so the run can print the 3-STREAM ENSEMBLE. Optional:
        without it you still get the Skepxels-alone de-saturation check.)
Run on DGX/Colab GPU (~10-15 min). Saves: final_skepxels_dann.pt
"""
import os, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm
try: torch.multiprocessing.set_sharing_strategy('file_system')
except Exception: pass

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; _CUDA=(DEVICE=='cuda')
IMG=224; EPOCHS=30; BATCH=48; LR=3e-4; WORKERS=0
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

# --- data: source = Kinect (labelled), target = MediaPipe (labelled) ------- #
k=np.load(_find('dataset_800.npz'),allow_pickle=True)
Xs=k['X150'].astype(np.float32); ys=(k['y']=='ASD').astype(np.int64)
m=np.load(_find('mp_train_99.npz'),allow_pickle=True)
Xt=m['X150'].astype(np.float32); yt=(m['y']=='ASD').astype(np.int64)
print(f'source Kinect {len(Xs)} | target MediaPipe {len(Xt)}')

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
def aug_mpsim(js):
    """Domain randomisation: make a Kinect skeleton look MediaPipe-degraded."""
    js=js.copy()
    js+=np.random.normal(0,0.03,js.shape).astype(np.float32)          # heavier jitter
    js[...,2]*=np.random.uniform(0.7,1.3)                              # depth-axis scale wobble
    js[...,2]+=np.random.normal(0,0.05,js[...,2].shape).astype(np.float32)  # depth noise
    for _ in range(np.random.randint(0,3)):                           # transient joint dropout
        j=np.random.randint(0,25); t=np.random.randint(0,len(js))
        js[t,j]=js[max(t-1,0),j]
    return js
def skel_image(js):
    img=np.transpose(js,(2,1,0)); t=torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0)
    t=torch.nn.functional.interpolate(t,size=(IMG,IMG),mode='bilinear',align_corners=False)[0]
    for c in range(3): t[c]=(t[c]-t[c].mean())/(t[c].std()+1e-6)
    return t

class DS(Dataset):
    def __init__(self,X,y,dom,train,mpsim): self.X,self.y,self.dom,self.train,self.mpsim=X,y,dom,train,mpsim
    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        js=normalize(self.X[i].reshape(150,25,3))
        if self.train:
            js=aug_geo(js)
            if self.mpsim and np.random.rand()<0.5: js=aug_mpsim(js)   # source only
        return skel_image(js), self.y[i], self.dom

# --- DANN model: ShuffleNet features + label head + GRL domain head -------- #
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,lambd): ctx.lambd=lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return -ctx.lambd*g, None

class DANN(nn.Module):
    def __init__(self):
        super().__init__()
        net=tvm.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')
        self.dim=net.fc.in_features; net.fc=nn.Identity()
        self.backbone=net
        self.label =nn.Linear(self.dim,2)
        self.domain=nn.Sequential(nn.Linear(self.dim,256),nn.ReLU(),nn.Linear(256,2))
    def forward(self,x,lambd=0.0):
        f=self.backbone(x)
        return self.label(f), self.domain(GRL.apply(f,lambd))

model=DANN().to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
ce=nn.CrossEntropyLoss(); sc=scaler()
# oversample the smaller target to balance domains
reps=max(1,round(len(Xs)/len(Xt)))
src=DataLoader(DS(Xs,ys,0,True,True),batch_size=BATCH,shuffle=True,num_workers=WORKERS,pin_memory=_CUDA,drop_last=True)
tgt=DataLoader(DS(np.repeat(Xt,reps,0),np.repeat(yt,reps,0),1,True,False),batch_size=BATCH,shuffle=True,num_workers=WORKERS,pin_memory=_CUDA,drop_last=True)

import itertools
print('training DANN...')
for ep in range(EPOCHS):
    model.train(); p=ep/max(1,EPOCHS-1); lambd=2.0/(1.0+math.exp(-10*p))-1.0
    for (xs,ls,ds),(xt,lt,dt) in zip(src,itertools.cycle(tgt)):
        xs,ls,ds=xs.to(DEVICE),ls.to(DEVICE),ds.to(DEVICE)
        xt,lt,dt=xt.to(DEVICE),lt.to(DEVICE),dt.to(DEVICE)
        opt.zero_grad()
        with amp():
            ysrc,dsrc=model(xs,lambd); ytgt,dtgt=model(xt,lambd)
            loss=ce(ysrc,ls)+ce(ytgt,lt) + ce(dsrc,ds)+ce(dtgt,dt)
        sc.scale(loss).backward(); sc.step(opt); sc.update()
    if (ep+1)%5==0: print(f'  epoch {ep+1}/{EPOCHS}  lambda {lambd:.2f}')
torch.save(model.state_dict(),'final_skepxels_dann.pt')

# --- FIELD TEST: Skepxels-alone de-saturation + the 3-STREAM ENSEMBLE ------ #
@torch.no_grad()
def skepxels_prob(x):
    js=normalize(np.asarray(x,dtype=np.float32).reshape(150,25,3))
    t=skel_image(js).unsqueeze(0).to(DEVICE)
    with amp(): return torch.softmax(model(t)[0].float(),1)[0,1].item()

def best_acc(p,yv):
    return max((((p>=t).astype(int)==yv).mean()) for t in np.linspace(0,1,101))

if os.path.exists('res_val_14.npz'):
    model.eval()
    from sklearn.metrics import roc_auc_score, confusion_matrix
    d=np.load('res_val_14.npz',allow_pickle=True); X=d['X150']; yl=list(d['y']); names=[str(n) for n in d['keys']]
    sk={}; Y={}
    for i in range(len(X)):
        sk[names[i]]=skepxels_prob(X[i]); Y[names[i]]=1 if yl[i]=='ASD' else 0
    order=names; yv=np.array([Y[n] for n in order]); skv=np.array([sk[n] for n in order])

    print('\n=== FIELD TEST (14 phone kids) ===')
    have3 = os.path.exists('field_hc_ang_14.npz')
    if have3:
        f=np.load('field_hc_ang_14.npz',allow_pickle=True)
        hc={str(k):float(v) for k,v in zip(f['keys'],f['hc'])}
        ang={str(k):float(v) for k,v in zip(f['keys'],f['ang'])}
        hcv=np.array([hc[n] for n in order]); angv=np.array([ang[n] for n in order])
        print('  kid            lab   HC   ANG  SKEP | 2str 3str')
        for n in order:
            two=0.5*hc[n]+0.5*ang[n]; three=(hc[n]+ang[n]+sk[n])/3
            print(f'  {n:14s}{"ASD" if Y[n] else "TD":4s} {hc[n]:.2f} {ang[n]:.2f} {sk[n]:.2f} | {two:.2f} {three:.2f}')
    else:
        for n in order: print(f'  {n:14s}{"ASD" if Y[n] else "TD":4s}  skepxels {sk[n]:.2f}')

    print(f'\n  [Skepxels-DANN alone]  TD mean {skv[yv==0].mean():.2f} (was ~0.95=flooded)  '
          f'ASD mean {skv[yv==1].mean():.2f}  AUC {roc_auc_score(yv,skv):.3f}')
    if have3:
        two=0.5*hcv+0.5*angv; three=(hcv+angv+skv)/3.0
        print(f'  [2-stream HC+ANG]      AUC {roc_auc_score(yv,two):.3f}  best-acc {best_acc(two,yv):.2f}   <- current deployed')
        print(f'  [3-stream +Skepxels]   AUC {roc_auc_score(yv,three):.3f}  best-acc {best_acc(three,yv):.2f}')
        print('  --> Skepxels EARNS its place only if the 3-stream AUC clears the 2-stream AUC.')
        print('      (best-acc is in-sample/optimistic on 14 kids — judge by AUC.)')
    else:
        print('  (field_hc_ang_14.npz not next to script -> 3-stream not shown; upload it to see the ensemble.)')
else:
    print('\n(res_val_14.npz not found -> download final_skepxels_dann.pt to test locally.)')
print('\nDone. Saved final_skepxels_dann.pt.')
