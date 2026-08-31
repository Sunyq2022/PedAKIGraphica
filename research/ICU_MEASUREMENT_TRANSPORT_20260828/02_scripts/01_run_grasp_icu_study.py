"""GRASP-aligned ICU semantic-transfer analysis across three EHR systems.

Four prespecified models mirror Kirchler et al.: demographic-only, random-
embedding transformer, XGBoost, and semantic-embedding transformer. Because
this ICU contract contains fixed-window binary endpoints rather than valid
time-to-event outcomes, masked multi-task BCE replaces the original Cox loss.
"""
from __future__ import annotations

import argparse, copy, gc, hashlib, importlib.metadata as metadata, json, os, platform, random, subprocess
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT=Path(__file__).resolve().parents[1]
TR=Path(os.environ.get("TRANSITION_ROOT",str(PROJECT.parent/"outputs"/"trust_aki"/"coarse_clinical_v1"/"adult_full_v002_contractfix")))
DB={"MIMIC-III":TR/"mimic3_coarse_clinical_v1_transitions.csv","MIMIC-IV":TR/"mimic4_coarse_clinical_v1_transitions.csv","eICU":TR/"eicu_coarse_clinical_v1_transitions.csv"}
# The audited, boundary-corrected files match the concept hashes recorded in
# the locked semantic manifest. The pre-audit directory is retained only as
# provenance input to that correction.
CONCEPT_ROOT=Path(os.environ.get("CONCEPT_ROOT",str(PROJECT/"03_data/01_analysis_data/concepts_audit_corrected")))
CF={"MIMIC-III":CONCEPT_ROOT/"mimiciii_concept_events_4h.csv.gz","MIMIC-IV":CONCEPT_ROOT/"mimiciv_concept_events_4h.csv.gz","eICU":CONCEPT_ROOT/"eicu_concept_events_4h.csv.gz"}
MIMIC3_ROOT=Path(os.environ["MIMIC3_ROOT"])
MIMIC4_ROOT=Path(os.environ["MIMIC4_ROOT"])
EICU_ROOT=Path(os.environ["EICU_ROOT"])
RAW={
 "MIMIC-III":(MIMIC3_ROOT/"PATIENTS.csv.gz",MIMIC3_ROOT/"ADMISSIONS.csv.gz"),
 "MIMIC-IV":(MIMIC4_ROOT/"hosp/patients.csv.gz",MIMIC4_ROOT/"hosp/admissions.csv.gz"),
 "eICU":(EICU_ROOT/"patient.csv.gz",None)}
STATE=[f"s_{i}" for i in range(15)]
ENDPOINTS=["next_aki_progression","next_aki_stage2plus","next_aki_stage3","hospital_death","icu_death"]
MODELS=["demographic","random_transformer","xgboost","semantic_transformer"]
SEED=20260828; MAXTOK=64

def sha256(path):
 h=hashlib.sha256()
 with open(path,"rb") as handle:
  for chunk in iter(lambda:handle.read(1024*1024),b""): h.update(chunk)
 return h.hexdigest()

def package_versions():
 names=["numpy","pandas","scipy","scikit-learn","xgboost","torch","transformers"]
 out={}
 for name in names:
  try:
   value=metadata.version(name)
   if value is None and name=="scikit-learn":
    import sklearn; value=sklearn.__version__
   out[name]=value if value is not None else "unknown"
  except metadata.PackageNotFoundError: out[name]="not-found"
 return out

def git_state():
 try:
  head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT,text=True,stderr=subprocess.DEVNULL).strip()
  dirty=bool(subprocess.check_output(["git","status","--porcelain","--",str(PROJECT)],cwd=PROJECT,text=True,stderr=subprocess.DEVNULL).strip())
  return {"head":head,"study_path_dirty":dirty}
 except Exception as exc: return {"unavailable":str(exc)}

def seed(s=SEED):
 random.seed(s); np.random.seed(s); import torch; torch.manual_seed(s)

def transitions(path):
 cols=["subject_id","hospital_admission_id","episode_id","decision_time","action_valid",*STATE,"next_s_12","hospital_death","icu_death"]
 f=pd.read_csv(path,usecols=cols,low_memory=False); f=f[pd.to_numeric(f.action_valid,errors="coerce").eq(1)].copy()
 f=f.sort_values(["episode_id","decision_time"],kind="mergesort").drop_duplicates("episode_id")
 ok=f.s_12.between(0,3)&f.next_s_12.between(0,3)
 f["next_aki_progression"]=np.where(ok,(f.next_s_12>f.s_12).astype(float),np.nan)
 f["next_aki_stage2plus"]=np.where(ok,(f.next_s_12>=2).astype(float),np.nan); f["next_aki_stage3"]=np.where(ok,(f.next_s_12>=3).astype(float),np.nan)
 for c in ENDPOINTS: f[c]=pd.to_numeric(f[c],errors="coerce").where(pd.to_numeric(f[c],errors="coerce").isin([0,1]))
 for c in ["subject_id","hospital_admission_id","episode_id"]: f[c]=f[c].astype(str)
 return f.reset_index(drop=True)

def demographics(name):
 pth,ath=RAW[name]
 if name=="eICU":
  d=pd.read_csv(pth,usecols=["patientunitstayid","gender","age","hospitalid"])
  age=d.age.astype(str).str.strip().replace({"> 89":"90",">89":"90"}); age=pd.to_numeric(age,errors="coerce")
  return pd.DataFrame({"icu_stay_id":d.patientunitstayid.astype(str),"age_years":age.clip(18,90),"female":d.gender.eq("Female").astype(float),"hospital_id":d.hospitalid.astype("Int64")}).drop_duplicates("icu_stay_id")
 if name=="MIMIC-III":
  p=pd.read_csv(pth,usecols=["SUBJECT_ID","GENDER","DOB"]); a=pd.read_csv(ath,usecols=["SUBJECT_ID","HADM_ID","ADMITTIME"]); d=a.merge(p,on="SUBJECT_ID",validate="many_to_one")
  # MIMIC-III date shifting can make otherwise valid timestamp differences
  # overflow pandas' nanosecond timedelta representation on some versions.
  # Python date subtraction preserves the same day-based age definition.
  admitted=pd.to_datetime(d.ADMITTIME).dt.date; born=pd.to_datetime(d.DOB).dt.date
  age=pd.Series([(x-y).days/365.2425 for x,y in zip(admitted,born)],index=d.index)
  return pd.DataFrame({"subject_id":d.SUBJECT_ID.astype(str),"hospital_admission_id":d.HADM_ID.astype(str),"age_years":age.clip(18,91.4),"female":d.GENDER.eq("F").astype(float)}).drop_duplicates(["subject_id","hospital_admission_id"])
 p=pd.read_csv(pth,usecols=["subject_id","gender","anchor_age","anchor_year"]); a=pd.read_csv(ath,usecols=["subject_id","hadm_id","admittime"]); d=a.merge(p,on="subject_id",validate="many_to_one")
 age=d.anchor_age+pd.to_datetime(d.admittime).dt.year-d.anchor_year
 return pd.DataFrame({"subject_id":d.subject_id.astype(str),"hospital_admission_id":d.hadm_id.astype(str),"age_years":age.clip(18,91.4),"female":d.gender.eq("F").astype(float)}).drop_duplicates(["subject_id","hospital_admission_id"])

def concepts(path,mode="all"):
 f=pd.read_csv(path,usecols=lambda c:c in ["episode_id","code_system","code","description","event_type","event_time_precision","window_id"],dtype="string",low_memory=False); f.window_id=pd.to_numeric(f.window_id,errors="coerce"); f=f[f.window_id.eq(0)].copy()
 if mode=="medication_only": f=f[f.event_type.eq("medication")].copy()
 elif mode=="precise_time": f=f[f.event_time_precision.isin(["timestamp","relative_minute"])].copy()
 elif mode=="label_proximal_excluded":
  proximal=r"dialys|renal replacement|hemofiltration|hemodia|haemodia|cvvh|crrt|do not resuscitate|\bdnr\b|palliative|comfort care|expired|death"
  f=f[~f.description.fillna("").str.contains(proximal,case=False,regex=True)].copy()
 elif mode!="all": raise ValueError(f"Unknown concept sensitivity: {mode}")
 for c in ["episode_id","code_system","code","description","event_type"]: f[c]=f[c].fillna("").str.strip()
 f=f[f.episode_id.ne("")&f.code.ne("")]; f["token"]=f.code_system+":"+f.code; f["norm"]=f.description.str.lower().str.replace(r"[^a-z0-9]+"," ",regex=True).str.strip(); return f

def frames(concept_mode="all"):
 out={}; allc=[]
 for name in DB:
  f=transitions(DB[name]); demo=demographics(name)
  if name=="eICU":
   f["icu_stay_id"]=f.episode_id.str.rsplit("|",n=1).str[-1]
   f=f.merge(demo,on="icu_stay_id",how="left",validate="many_to_one")
  else:
   f=f.merge(demo,on=["subject_id","hospital_admission_id"],how="left",validate="many_to_one"); f["hospital_id"]=pd.NA
  c=concepts(CF[name],concept_mode); c["database"]=name; allc.append(c)
  c["concept_pair"]=list(zip(c.token,c.description,c.norm))
  g=c.groupby("episode_id",sort=False).agg(concept_pairs=("concept_pair",lambda x:list(dict.fromkeys(x))),n_events=("token","size")).reset_index(); f=f.merge(g,on="episode_id",how="left",validate="one_to_one")
  def as_pairs(x): return list(x) if isinstance(x,(list,tuple,np.ndarray)) else []
  f["concept_pairs"]=f.concept_pairs.apply(as_pairs)
  f["tokens"]=f.concept_pairs.apply(lambda x:[p[0] for p in x])
  f["descriptions"]=f.concept_pairs.apply(lambda x:[p[1] for p in x if p[1]])
  f["norms"]=f.concept_pairs.apply(lambda x:[p[2] for p in x if p[2]])
  f["database"]=name; out[name]=f; print(name,len(f),"demographics complete",f[["age_years","female"]].notna().all(axis=1).mean())
 return out,pd.concat(allc,ignore_index=True)

def embed(texts,model_name,cache,batch=64):
 existing={}
 if cache.exists():
  with np.load(cache,allow_pickle=False) as z:
   cached_keys=z["keys"].tolist(); cached_vectors=np.asarray(z["vectors"],dtype=np.float32).copy()
  existing=dict(zip(cached_keys,cached_vectors))
 missing=[text for text in texts if text not in existing]
 if not missing: return {text:existing[text] for text in texts}
 old=metadata.version; import numpy as _np
 try:
  metadata.version=lambda n:_np.__version__ if n=="numpy" else old(n)
  import torch; from transformers import AutoModel,AutoTokenizer
 finally: metadata.version=old
 tok=AutoTokenizer.from_pretrained(model_name,local_files_only=True); model=AutoModel.from_pretrained(model_name,local_files_only=True).eval(); vv=[]
 for s in range(0,len(missing),batch):
  x=tok(missing[s:s+batch],padding=True,truncation=True,max_length=64,return_tensors="pt")
  with torch.no_grad(): h=model(**x).last_hidden_state
  m=x["attention_mask"].unsqueeze(-1); v=(h*m).sum(1)/m.sum(1).clamp(min=1); vv.append(torch.nn.functional.normalize(v,dim=1).numpy().astype(np.float32))
 for text,vector in zip(missing,np.vstack(vv)): existing[text]=vector
 mat=np.stack([existing[text] for text in texts]).astype(np.float32); tmp=cache.with_suffix(".tmp.npz"); np.savez_compressed(tmp,keys=np.asarray(texts),vectors=mat); tmp.replace(cache)
 print(f"Semantic cache: {len(texts)} descriptions ({len(missing)} newly encoded)",flush=True)
 return dict(zip(texts,mat))

def demo_xy(train,test):
 # Fixed clinically interpretable scaling makes source weights reusable during
 # target-domain fine-tuning; all linked demographic fields are complete.
 def transform(f): return np.column_stack([(f.age_years.to_numpy(float)-60.)/20.,f.female.to_numpy(float)]).astype("float32")
 return transform(train),transform(test)

def mlp_pred(train,test,epochs,s):
 import torch; from torch import nn
 seed(s); a,b=demo_xy(train,test); y=train[ENDPOINTS].to_numpy(float); m=nn.Sequential(nn.Linear(a.shape[1],32),nn.GELU(),nn.Linear(32,len(ENDPOINTS))); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4); x=torch.tensor(a); yy=torch.tensor(np.nan_to_num(y),dtype=torch.float32); mask=torch.tensor(np.isfinite(y),dtype=torch.float32); rng=np.random.default_rng(s)
 for ep in range(epochs):
  for ii in np.array_split(rng.permutation(len(a)),max(1,int(np.ceil(len(a)/512)))):
   loss=torch.nn.functional.binary_cross_entropy_with_logits(m(x[ii]),yy[ii],reduction="none"); loss=(loss*mask[ii]).sum()/mask[ii].sum().clamp(min=1); opt.zero_grad(); loss.backward(); opt.step()
 m.eval()
 with torch.no_grad(): return torch.sigmoid(m(torch.tensor(b))).numpy()

def xgb_pred(train,test,s,field="tokens"):
 docs=train[field].apply(" ".join); test_docs=test[field].apply(" ".join)
 if docs.str.strip().any():
  try:
   vec=CountVectorizer(token_pattern=r"(?u)\S+",lowercase=False,binary=True,min_df=2); a=vec.fit_transform(docs); b=vec.transform(test_docs); vocab=set(vec.vocabulary_)
  except ValueError:
   a=sparse.csr_matrix((len(train),0)); b=sparse.csr_matrix((len(test),0)); vocab=set()
 else:
  a=sparse.csr_matrix((len(train),0)); b=sparse.csr_matrix((len(test),0)); vocab=set()
 da,db=demo_xy(train,test); a=sparse.hstack([da,a],format="csr"); b=sparse.hstack([db,b],format="csr"); y=train[ENDPOINTS].to_numpy(float); pred=np.zeros((len(test),len(ENDPOINTS)))
 for j in range(len(ENDPOINTS)):
  ok=np.isfinite(y[:,j]); pos=y[ok,j].sum(); neg=ok.sum()-pos; model=XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.04,subsample=.8,colsample_bytree=.8,min_child_weight=5,reg_lambda=1,objective="binary:logistic",eval_metric="logloss",tree_method="hist",n_jobs=4,random_state=s,scale_pos_weight=max(1,neg/max(pos,1))); model.fit(a[ok],y[ok,j]); pred[:,j]=model.predict_proba(b)[:,1]
 return pred,vocab

def net(input_dim):
 import torch; from torch import nn
 class Net(nn.Module):
  def __init__(self):
   super().__init__(); d=64; self.proj=nn.Linear(input_dim,d); self.demo=nn.Linear(2,d); self.cls=nn.Parameter(torch.zeros(1,1,d)); layer=nn.TransformerEncoderLayer(d,8,256,.1,"gelu",batch_first=True,norm_first=True); self.enc=nn.TransformerEncoder(layer,4,enable_nested_tensor=False); self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,len(ENDPOINTS))
  def forward(self,x,pad,demo):
   b=x.shape[0]; z=torch.cat([self.cls.expand(b,-1,-1)+self.demo(demo).unsqueeze(1),self.proj(x)],1); pm=torch.cat([torch.zeros((b,1),dtype=torch.bool,device=x.device),pad],1); return self.head(self.norm(self.enc(z,src_key_padding_mask=pm)[:,0]))
 return Net()

def token_arrays(frame,lookup,field,dim=None):
 dim=dim or (len(next(iter(lookup.values()))) if lookup else 1); arrays=[]; names=[]
 for row in frame[field]:
  n=list(dict.fromkeys(v for v in row if v in lookup))[:MAXTOK]; z=np.zeros((len(n),dim),np.float32)
  if n:z[:len(n)]=np.stack([lookup[v] for v in n])
  arrays.append(z); names.append(n)
 return arrays,names

def transformer_pred(train,test,semantic,kind,epochs,s,field="descriptions",pretrained=None):
 import torch
 seed(s); dim=len(next(iter(semantic.values()))) if semantic else 1; vocab=sorted({v for row in train[field] for v in row})
 if kind=="random":
  rng=np.random.default_rng(s); lookup={v:rng.normal(0,1/np.sqrt(dim),dim).astype(np.float32) for v in vocab}
 else: lookup=semantic
 aa,_=token_arrays(train,lookup,field,dim); bb,names=token_arrays(test,lookup,field,dim); da,db=demo_xy(train,test); y=train[ENDPOINTS].to_numpy(float); model=net(dim) if pretrained is None else copy.deepcopy(pretrained); opt=torch.optim.AdamW(model.parameters(),lr=1e-3 if pretrained is None else 1e-4,weight_decay=1e-4); rng=np.random.default_rng(s)
 def batch(arrays,demos,idx):
  rows=[arrays[i] for i in idx]; L=max(1,max(len(v) for v in rows)); z=np.zeros((len(rows),L,dim),np.float32); pm=np.ones((len(rows),L),bool)
  for k,v in enumerate(rows): z[k,:len(v)]=v; pm[k,:len(v)]=False
  return torch.tensor(z),torch.tensor(pm),torch.tensor(demos[idx],dtype=torch.float32)
 model.train()
 for ep in range(epochs):
  for ii in np.array_split(rng.permutation(len(train)),max(1,int(np.ceil(len(train)/256)))):
   z,pm,d=batch(aa,da,ii); logits=model(z,pm,d); yy=torch.tensor(np.nan_to_num(y[ii]),dtype=torch.float32); mm=torch.tensor(np.isfinite(y[ii]),dtype=torch.float32); loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,yy,reduction="none"); loss=(loss*mm).sum()/mm.sum().clamp(min=1); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
 model.eval(); out=[]
 with torch.no_grad():
  for st in range(0,len(test),256):
   ii=np.arange(st,min(st+256,len(test))); z,pm,d=batch(bb,db,ii); out.append(torch.sigmoid(model(z,pm,d)).numpy())
 return np.vstack(out),set(vocab),model,(bb,names,db)

def transformer_infer(model,test,lookup,field="descriptions"):
 """Apply an already-trained transformer without changing its parameters."""
 import torch
 arrays,_=token_arrays(test,lookup,field); demos=demo_xy(test,test)[1]; dim=len(next(iter(lookup.values())))
 def batch(idx):
  rows=[arrays[i] for i in idx]; length=max(1,max(len(v) for v in rows)); z=np.zeros((len(rows),length,dim),np.float32); pm=np.ones((len(rows),length),bool)
  for k,v in enumerate(rows): z[k,:len(v)]=v; pm[k,:len(v)]=False
  return torch.tensor(z),torch.tensor(pm),torch.tensor(demos[idx],dtype=torch.float32)
 model.eval(); out=[]
 with torch.no_grad():
  for start in range(0,len(test),256):
   idx=np.arange(start,min(start+256,len(test))); z,pm,d=batch(idx); out.append(torch.sigmoid(model(z,pm,d)).numpy())
 return np.vstack(out)

def group_split(frame,train_fraction,s,group="subject_id"):
 """Split complete patients, never episodes, between development and test sets."""
 groups=frame[group].astype(str); unique=groups.drop_duplicates().to_numpy(); rng=np.random.default_rng(s); rng.shuffle(unique)
 cut=max(1,min(len(unique)-1,int(train_fraction*len(unique)))); train_groups=set(unique[:cut]); take=groups.isin(train_groups)
 train=frame.loc[take].copy(); test=frame.loc[~take].copy()
 assert set(train[group].astype(str)).isdisjoint(set(test[group].astype(str)))
 return train,test

def nested_group_subset(frame,fraction,ordered_groups,group="subject_id"):
 """Select an increasing, patient-grouped subset from a fixed random ordering."""
 n=max(1,min(len(ordered_groups),int(fraction*len(ordered_groups)))); selected=set(ordered_groups[:n])
 out=frame.loc[frame[group].astype(str).isin(selected)].copy()
 assert out[group].astype(str).nunique()==n
 return out,n

def metrics(y,p,target,model,evaluation,source):
 rows=[]
 for j,e in enumerate(ENDPOINTS):
  ok=np.isfinite(y[:,j]); yy=y[ok,j].astype(int); pp=p[ok,j]
  if np.unique(yy).size<2:continue
  ece=0.; bins=np.linspace(0,1,11)
  for lo,hi in zip(bins[:-1],bins[1:]):
   take=(pp>=lo)&(pp<hi if hi<1 else pp<=hi)
   if take.any(): ece+=take.mean()*abs(pp[take].mean()-yy[take].mean())
  rows.append(dict(evaluation=evaluation,source_database=source,target_database=target,model=model,endpoint=e,n=len(yy),event_rate=yy.mean(),auroc=roc_auc_score(yy,pp),average_precision=average_precision_score(yy,pp),brier=brier_score_loss(yy,pp),ece_10bin=ece))
 return rows

def bootstrap(y,preds,reps,source,target,s,evaluation):
 rng=np.random.default_rng(s); rows=[]
 for j,e in enumerate(ENDPOINTS):
  ok=np.isfinite(y[:,j]); yy=y[ok,j].astype(int); vals={k:[] for k in preds}; delta={k:[] for k in preds if k!="semantic_transformer"}
  for _ in range(reps):
   ii=rng.integers(0,len(yy),len(yy))
   if np.unique(yy[ii]).size<2:continue
   scores={k:roc_auc_score(yy[ii],v[ok,j][ii]) for k,v in preds.items()}
   for k,v in scores.items():vals[k].append(v)
   for k in delta:delta[k].append(scores["semantic_transformer"]-scores[k])
  for k in preds:rows.append(dict(evaluation=evaluation,source_database=source,target_database=target,endpoint=e,model=k,auroc_lo=np.quantile(vals[k],.025),auroc_hi=np.quantile(vals[k],.975),delta_vs_semantic_lo=np.quantile(delta[k],.025) if k in delta else np.nan,delta_vs_semantic_hi=np.quantile(delta[k],.975) if k in delta else np.nan))
 return rows

def mean_auroc_bootstrap(y,preds,reps,source,target,s):
 """Bootstrap the unweighted mean AUROC across available endpoints."""
 rng=np.random.default_rng(s); rows=[]; values={k:[] for k in preds}; n=len(y)
 for _ in range(reps):
  idx=rng.integers(0,n,n)
  for model,pred in preds.items():
   scores=[]
   for j in range(len(ENDPOINTS)):
    ok=np.isfinite(y[idx,j]); yy=y[idx,j][ok].astype(int)
    if np.unique(yy).size>=2: scores.append(roc_auc_score(yy,pred[idx,j][ok]))
   if scores: values[model].append(float(np.mean(scores)))
 for model,v in values.items(): rows.append(dict(source_database=source,target_database=target,model=model,uncertainty_unit="target episode",mean_auroc_lo=np.quantile(v,.025),mean_auroc_hi=np.quantile(v,.975),bootstrap_reps=len(v)))
 return rows

def cluster_bootstrap(y,preds,clusters,reps,source,target,s,evaluation,cluster_unit):
 rng=np.random.default_rng(s); rows=[]; groups=pd.Series(clusters).fillna(-1).astype(str).to_numpy(); unique=np.unique(groups)
 for j,e in enumerate(ENDPOINTS):
  ok=np.isfinite(y[:,j]); yy=y[ok,j].astype(int); gg=groups[ok]; pp={k:v[ok,j] for k,v in preds.items()}; vals={k:[] for k in preds}
  for _ in range(reps):
   sampled=rng.choice(unique,len(unique),replace=True); idx=np.concatenate([np.flatnonzero(gg==g) for g in sampled])
   if np.unique(yy[idx]).size<2: continue
   for k,v in pp.items(): vals[k].append(roc_auc_score(yy[idx],v[idx]))
  for k,v in vals.items(): rows.append(dict(evaluation=evaluation,cluster_unit=cluster_unit,source_database=source,target_database=target,endpoint=e,model=k,n_clusters=len(unique),auroc_lo=np.quantile(v,.025),auroc_hi=np.quantile(v,.975)))
 return rows

def balanced_pool(fs,names,s):
 n=min(len(fs[name]) for name in names); parts=[]
 for i,name in enumerate(names):
  q=fs[name].sample(n=n,random_state=s+i).copy(); q["training_database"]=name; parts.append(q)
 return pd.concat(parts,ignore_index=True)

def occlusion_rows(model,payload,pred,target,limit):
 import torch
 arrays,names,demos=payload; model.eval(); freq=Counter(); imp=defaultdict(list)
 eligible=[i for i,name_list in enumerate(names) if name_list][:limit]
 for i in eligible:
  arr=arrays[i]; pm=torch.zeros((1,len(arr)),dtype=torch.bool); d=torch.tensor(demos[i:i+1]); base=pred[i,1]
  alt=np.repeat(arr[None],len(names[i]),axis=0); alt[np.arange(len(names[i])),np.arange(len(names[i]))]=0
  with torch.no_grad(): qs=torch.sigmoid(model(torch.tensor(alt),pm.expand(len(names[i]),-1),d.expand(len(names[i]),-1)))[:,1].numpy()
  for name,q in zip(names[i],qs): imp[name].append(float(base-q)); freq[name]+=1
 return [dict(database=target,description=name,importance=np.mean(v),frequency=freq[name],endpoint="next_aki_stage2plus") for name,v in imp.items()]

def run(a):
 seed(); out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()) and not a.overwrite:raise FileExistsError(f"{out} not empty; use --overwrite")
 fs,allc=frames(a.concept_sensitivity); texts=sorted({v for f in fs.values() for row in f.descriptions for v in row}); semantic=embed(texts,a.semantic_model,out/"semantic_embedding_lookup.npz",a.semantic_batch_size)
 if a.max_episodes:
  fs={k:v.sample(min(a.max_episodes,len(v)),random_state=SEED).reset_index(drop=True) for k,v in fs.items()}
 result=[]; boot=[]; cluster=[]; patient_cluster=[]; cover=[]; sample=[]; sample_boot=[]; adapt=[]; coding=[]; coding_boot=[]; importance=[]; availability=[]; lodo=[]
 directions=[(s,t) for s in DB for t in DB if s!=t]
 explained=set()
 for di,(sn,tn) in enumerate(directions):
  print(f"Direction {sn} -> {tn}",flush=True); src,tgt=fs[sn],fs[tn]; y=tgt[ENDPOINTS].to_numpy(float); preds={}
  preds["demographic"]=mlp_pred(src,tgt,a.epochs,SEED+di)
  preds["xgboost"],_=xgb_pred(src,tgt,SEED+di)
  preds["random_transformer"],_,_,_=transformer_pred(src,tgt,semantic,"random",a.epochs,SEED+10+di)
  preds["semantic_transformer"],_,sm,payload=transformer_pred(src,tgt,semantic,"semantic",a.epochs,SEED+20+di)
  for m,p in preds.items(): result+=metrics(y,p,tn,m,"external",sn)
  boot+=bootstrap(y,preds,a.bootstrap_reps,sn,tn,SEED+di,"external")
  patient_cluster+=cluster_bootstrap(y,preds,tgt.subject_id,a.bootstrap_reps,sn,tn,SEED+50+di,"external","patient")
  if tn=="eICU": cluster+=cluster_bootstrap(y,preds,tgt.hospital_id,a.bootstrap_reps,sn,tn,SEED+100+di,"external","hospital")
  sc={v for row in src.tokens for v in row}; tc={v for row in tgt.tokens for v in row}; cover.append(dict(source_database=sn,target_database=tn,n_target_unique_codes=len(tc),n_exactly_seen=len(tc&sc),exact_code_seen_fraction=len(tc&sc)/max(1,len(tc)),semantic_representable_fraction=1.0,target_episode_concept_coverage=np.mean(tgt.descriptions.apply(len)>0)))
  if tn=="eICU":
   has=tgt.descriptions.apply(len)>0
   for label,mask in [("all",np.ones(len(tgt),bool)),("with_concepts",has.to_numpy()),("without_concepts",(~has).to_numpy())]:
    for m,p in preds.items(): availability+=metrics(y[mask],p[mask],tn,m,f"concept_availability_{label}",sn)
   ad,te=group_split(tgt,.1,SEED+di)
   p,_,_,_=transformer_pred(ad,te,semantic,"semantic",a.adapt_epochs,SEED+40+di); adapt+=metrics(te[ENDPOINTS].to_numpy(float),p,tn,"target_trained","adaptation",sn)
   p,_,_,_=transformer_pred(ad,te,semantic,"semantic",a.adapt_epochs,SEED+41+di,pretrained=sm); adapt+=metrics(te[ENDPOINTS].to_numpy(float),p,tn,"source_to_target_finetuned","adaptation",sn)
   restricted=tgt.copy(); restricted["descriptions"]=restricted.concept_pairs.apply(lambda pairs:[d for t,d,_ in pairs if t in sc and d]); p=transformer_infer(sm,restricted,semantic); coding+=metrics(y,preds["demographic"],tn,"demographic","coding_stress",sn)+metrics(y,p,tn,"exact_code_supported_semantic","coding_stress",sn)+metrics(y,preds["semantic_transformer"],tn,"all_description_semantic","coding_stress",sn)
   coding_preds={"demographic":preds["demographic"],"exact_code_supported_semantic":p,"semantic_transformer":preds["semantic_transformer"]}; coding_boot+=bootstrap(y,coding_preds,a.bootstrap_reps,sn,tn,SEED+130+di,"coding_stress")
  if tn not in explained:
   importance+=occlusion_rows(sm,payload,preds["semantic_transformer"],tn,a.occlusion_patients); explained.add(tn)
  gc.collect()
 # Leave-one-database-out with equal source-database episode counts.
 for li,tn in enumerate(DB):
  sources=[x for x in DB if x!=tn]; sn=" + ".join(sources); src=balanced_pool(fs,sources,SEED+li); tgt=fs[tn]; y=tgt[ENDPOINTS].to_numpy(float); preds={}
  print(f"LODO {sn} -> {tn}",flush=True)
  preds["demographic"]=mlp_pred(src,tgt,a.epochs,SEED+200+li); preds["xgboost"],_=xgb_pred(src,tgt,SEED+200+li); preds["random_transformer"],_,_,_=transformer_pred(src,tgt,semantic,"random",a.epochs,SEED+210+li); preds["semantic_transformer"],_,sm,payload=transformer_pred(src,tgt,semantic,"semantic",a.epochs,SEED+220+li)
  for m,p in preds.items(): lodo+=metrics(y,p,tn,m,"lodo",sn)
  boot+=bootstrap(y,preds,a.bootstrap_reps,sn,tn,SEED+230+li,"lodo")
  patient_cluster+=cluster_bootstrap(y,preds,tgt.subject_id,a.bootstrap_reps,sn,tn,SEED+235+li,"lodo","patient")
  if tn=="eICU": cluster+=cluster_bootstrap(y,preds,tgt.hospital_id,a.bootstrap_reps,sn,tn,SEED+240+li,"lodo","hospital")
 # Figure-4-aligned sample-size curves: one development database evaluated
 # internally and in both external databases, matching the reference design.
 sample_scenarios=[]; internal_train,internal_test=group_split(fs["MIMIC-IV"],.8,SEED); sample_scenarios.append(("MIMIC-IV (development)",internal_train,"MIMIC-IV (internal test)",internal_test)); sample_scenarios.append(("MIMIC-IV",fs["MIMIC-IV"],"MIMIC-III",fs["MIMIC-III"])); sample_scenarios.append(("MIMIC-IV",fs["MIMIC-IV"],"eICU",fs["eICU"]))
 for si,(sn,src,tn,tgt) in enumerate(sample_scenarios):
  y=tgt[ENDPOINTS].to_numpy(float)
  source_patients=src.subject_id.astype(str).drop_duplicates().to_numpy(); np.random.default_rng(SEED+si).shuffle(source_patients)
  for frac in [.1,.25,.5,1.]:
   sub,n_patients=nested_group_subset(src,frac,source_patients); xp,_=xgb_pred(sub,tgt,SEED+300+si); sp,_,_,_=transformer_pred(sub,tgt,semantic,"semantic",a.sample_epochs,SEED+310+si)
   for m,p in [("xgboost",xp),("semantic_transformer",sp)]:
    mr=metrics(y,p,tn,m,"sample_size",sn); sample.append(dict(source_database=sn,target_database=tn,model=m,fraction=frac,n_source_patients=n_patients,n_source_episodes=len(sub),mean_auroc=np.mean([r["auroc"] for r in mr]),mean_average_precision=np.mean([r["average_precision"] for r in mr])))
   sb=mean_auroc_bootstrap(y,{"xgboost":xp,"semantic_transformer":sp},a.bootstrap_reps,sn,tn,SEED+320+si+int(frac*100))
   for row in sb: row.update(fraction=frac,n_source_patients=n_patients,n_source_episodes=len(sub))
   sample_boot+=sb
 outputs={"external_results.csv":result,"external_bootstrap_intervals.csv":boot,"hospital_cluster_bootstrap.csv":cluster,"patient_cluster_bootstrap.csv":patient_cluster,"coding_transfer_coverage.csv":cover,"sample_size_curves.csv":sample,"sample_size_bootstrap.csv":sample_boot,"target_adaptation_results.csv":adapt,"coding_stress_results.csv":coding,"coding_stress_bootstrap.csv":coding_boot,"concept_importance.csv":importance,"concept_availability_results.csv":availability,"lodo_results.csv":lodo}
 for name,rows in outputs.items(): pd.DataFrame(rows).to_csv(out/name,index=False)
 manifest={"analysis":"GRASP-aligned three-system ICU semantic transfer","status":"SMOKE_TEST" if a.max_episodes else "MODEL_RESULTS_BUILT","completed_at_utc":datetime.now(timezone.utc).isoformat(),"databases":list(DB),"database_versions":{"MIMIC-III":"1.4","MIMIC-IV":"3.1","eICU":"2.0"},"concept_sensitivity":a.concept_sensitivity,"n_episodes":{k:len(v) for k,v in fs.items()},"directions":directions,"lodo_targets":list(DB),"lodo_sampling":"equal episode count per source database","split_unit":"patient for target adaptation and internal development/test analyses","uncertainty_units":["episode","patient cluster","eICU hospital cluster"],"max_episodes_test_only":a.max_episodes,"models":MODELS,"endpoints":ENDPOINTS,"architecture":{"depth":4,"heads":8,"hidden_dim":64,"max_tokens":64,"epochs":a.epochs,"loss":"masked multi-task binary cross-entropy","empty_concept_handling":"fully masked; CLS plus demographics only"},"semantic_model":a.semantic_model,"bootstrap_reps":a.bootstrap_reps,"input_sha256":{"transitions":{k:sha256(v) for k,v in DB.items()},"concepts":{k:sha256(v) for k,v in CF.items()}},"script_sha256":{Path(__file__).name:sha256(Path(__file__))},"software":{"python":platform.python_version(),"packages":package_versions()},"git":git_state(),"reference_alignment":"scientific design adaptation, not numerical replication","interpretation_boundary":["predictive transportability only","no causal, policy-value, or clinical-utility claim","MIMIC-III/MIMIC-IV cross-version patient overlap cannot be excluded"]}; (out/"study_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); print(json.dumps(manifest,indent=2))

def main():
 p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,default=PROJECT/"05_results_derived/03_semantic_rerun"); p.add_argument("--semantic-model",default="sentence-transformers/all-MiniLM-L6-v2"); p.add_argument("--semantic-batch-size",type=int,default=64); p.add_argument("--epochs",type=int,default=8); p.add_argument("--sample-epochs",type=int,default=4); p.add_argument("--adapt-epochs",type=int,default=4); p.add_argument("--bootstrap-reps",type=int,default=200); p.add_argument("--occlusion-patients",type=int,default=1000); p.add_argument("--concept-sensitivity",choices=["all","medication_only","precise_time","label_proximal_excluded"],default="all"); p.add_argument("--max-episodes",type=int,default=None,help="Test-only cap; never use for formal results"); p.add_argument("--overwrite",action="store_true"); run(p.parse_args())
if __name__=="__main__":main()
