import json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import joblib, numpy as np
r=Path(__file__).resolve().parents[1]
a=joblib.load(r/'v40_artifacts/asd_gait_v40_3seed.joblib')
e=a['ebm']; names=a['ebm_sel_names']; imp=e.term_importances()
out={'parameters':{s:{k:v.get_params() for k,v in a[s].items()} for s in ['emb_models','hc_models']},'ebm_params':e.get_params(),'ebm_names':names,'top_terms':[{'name':' x '.join(names[int(x.split('_')[-1])] if x.startswith('feature_') else x for x in e.term_names_[j].split(' & ')), 'importance':float(imp[j])} for j in np.argsort(imp)[::-1][:15]],'pca_variance':float(a['pca_deploy'].explained_variance_ratio_.sum()),'fidelity':a['ebm_fidelity']}
(r/'colab/nsi_model_facts.json').write_text(json.dumps(out,indent=2,default=str))
print(json.dumps(out,indent=2,default=str))
