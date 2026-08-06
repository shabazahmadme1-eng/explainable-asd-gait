"""
Local field verdict for the DANN Skepxels model (no retrain needed).
Loads final_skepxels_dann.pt, scores the 14 phone kids, and prints the 3-stream.
Run AFTER copying final_skepxels_dann.pt from the DGX into this folder.
"""
import os, numpy as np, torch, torch.nn as nn
import torchvision.models as tvm
from sklearn.metrics import roc_auc_score
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; IMG=224
if not os.path.exists('final_skepxels_dann.pt'):
    raise SystemExit('Put final_skepxels_dann.pt in this folder first (download from DGX).')

def normalize(js):
    js=js-js[:,19:20,:]; torso=np.linalg.norm(js[:,20,:],axis=-1).mean()+1e-6; return js/torso
def skel_image(js):
    img=np.transpose(js,(2,1,0)); t=torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0)
    t=torch.nn.functional.interpolate(t,size=(IMG,IMG),mode='bilinear',align_corners=False)[0]
    for c in range(3): t[c]=(t[c]-t[c].mean())/(t[c].std()+1e-6)
    return t

class DANN(nn.Module):
    def __init__(self):
        super().__init__()
        net=tvm.shufflenet_v2_x1_0(weights=None); self.dim=net.fc.in_features; net.fc=nn.Identity()
        self.backbone=net; self.label=nn.Linear(self.dim,2)
        self.domain=nn.Sequential(nn.Linear(self.dim,256),nn.ReLU(),nn.Linear(256,2))
    def forward(self,x): f=self.backbone(x); return self.label(f)

model=DANN().to(DEVICE); model.load_state_dict(torch.load('final_skepxels_dann.pt',map_location=DEVICE)); model.eval()

@torch.no_grad()
def skep(x):
    js=normalize(np.asarray(x,dtype=np.float32).reshape(150,25,3))
    return torch.softmax(model(skel_image(js).unsqueeze(0).to(DEVICE)).float(),1)[0,1].item()

d=np.load('res_val_14.npz',allow_pickle=True); X=d['X150']; yl=list(d['y']); names=[str(n) for n in d['keys']]
Y={n:(1 if l=='ASD' else 0) for n,l in zip(names,yl)}; SK={n:skep(X[i]) for i,n in enumerate(names)}
order=names; yv=np.array([Y[n] for n in order]); skv=np.array([SK[n] for n in order])
f=np.load('field_hc_ang_14.npz',allow_pickle=True)
hc={str(k):float(v) for k,v in zip(f['keys'],f['hc'])}; ang={str(k):float(v) for k,v in zip(f['keys'],f['ang'])}
hcv=np.array([hc[n] for n in order]); angv=np.array([ang[n] for n in order])
def bestacc(p): return max((((p>=t).astype(int)==yv).mean()) for t in np.linspace(0,1,101))

print('  kid            lab   HC   ANG  SKEP | 2str 3str')
for n in order:
    print(f'  {n:14s}{"ASD" if Y[n] else "TD":4s} {hc[n]:.2f} {ang[n]:.2f} {SK[n]:.2f} | {0.5*hc[n]+0.5*ang[n]:.2f} {(hc[n]+ang[n]+SK[n])/3:.2f}')
two=0.5*hcv+0.5*angv; three=(hcv+angv+skv)/3
print(f'\n  [Skepxels-DANN alone]  TD mean {skv[yv==0].mean():.2f} (was ~0.95=flooded)  ASD mean {skv[yv==1].mean():.2f}  AUC {roc_auc_score(yv,skv):.3f}')
print(f'  [2-stream HC+ANG]      AUC {roc_auc_score(yv,two):.3f}  best-acc {bestacc(two):.2f}   <- current deployed')
print(f'  [3-stream +Skepxels]   AUC {roc_auc_score(yv,three):.3f}  best-acc {bestacc(three):.2f}')
print('  --> Skepxels is safe to ship only if the 3-stream AUC stays at the 2-stream ceiling (1.000).')
