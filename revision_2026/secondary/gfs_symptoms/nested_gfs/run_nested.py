"""Nested country-held-out PHQ-4 benchmark (PROTOCOL.md). Stage 'inner' scores all candidates in inner folds (parallel);
stage 'outer' refits selected settings and scores held-out countries; stage 'summary' writes tables."""
from pathlib import Path
import os,sys,json,random,hashlib,itertools,platform
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MPLCONFIGDIR','/tmp/violence-mpl')
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1];sys.path.insert(0,str(BASE.parent/'vendor'))
import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA

SRC=ROOT/'04_mental_health/gfs_public/analysis/prediction_cohort.csv'
CATS=['education','employment','marital','income','urbanicity','attendance','covid_loss'];NUMS=['age10','age10_sq']
GRID={'ridge':[{'alpha':a} for a in [1,10,100]],
      'hgb':[{'max_leaf_nodes':l,'l2_regularization':r} for l in [7,15] for r in [1,10]],
      'pca':[{'n_components':k} for k in [3,8]],
      'cebra':[{'iterations':i,'dim':d,'temperature':t} for i in [1000,5000] for d in [3,8] for t in [.1,1.]]}
OUT=BASE/'results'

def data():
    x=pd.read_csv(SRC)
    for c in CATS:x[c]=x[c].map(lambda v:'Missing' if pd.isna(v) else str(int(v)))
    return x
def capped(x,exclude):
    t=x[~x.country.isin(exclude)]
    return pd.concat([g.sample(n=min(len(g),1000),random_state=2026) for _,g in t.groupby('country')])
def weights(t):w=(t.weight/t.groupby('country').weight.transform('sum')).to_numpy();return w*len(w)/w.sum()
def prep(a,b):
    pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),NUMS),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATS)])
    return pre.fit_transform(a).astype(np.float32),pre.transform(b).astype(np.float32)
def wmae(y,p,w):return float(np.sum(w*np.abs(y-p))/np.sum(w))

def fit_predict(model,par,A,B,y,w,seed=42,shuffle=False,loss_out=None):
    if model=='mean':return np.full(len(B),np.average(y,weights=w))
    if model=='ridge':return Ridge(alpha=par['alpha']).fit(A,y,sample_weight=w).predict(B)
    if model=='hgb':return HistGradientBoostingRegressor(max_iter=100,min_samples_leaf=30,random_state=42,**par).fit(A,y,sample_weight=w).predict(B)
    if model=='pca':
        pc=PCA(n_components=par['n_components'],random_state=42).fit(A);s=StandardScaler().fit(pc.transform(A))
        return Ridge(alpha=10).fit(s.transform(pc.transform(A)),y,sample_weight=w).predict(s.transform(pc.transform(B)))
    if model=='cebra':
        import torch,cebra;torch.set_num_threads(2);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        lab=((y-y.mean())/y.std()).astype(np.float32)[:,None]
        if shuffle:lab=np.random.default_rng(seed).permutation(lab)
        m=cebra.CEBRA(model_architecture='offset1-model',conditional='delta',delta=.3,distance='cosine',temperature=par['temperature'],batch_size=128,
            learning_rate=.001,max_iterations=par['iterations'],output_dimension=par['dim'],num_hidden_units=32,device='cpu',verbose=False).fit(A,lab)
        if loss_out is not None:loss_out.extend(np.asarray(m.state_dict_['loss'],dtype=float).reshape(-1).tolist())
        za,zb=m.transform(A),m.transform(B);s=StandardScaler().fit(za)
        return Ridge(alpha=10).fit(s.transform(za),y,sample_weight=w).predict(s.transform(zb))

def inner_job(job):
    held,model,ci=job;f=OUT/'inner'/f'{held}__{model}__{ci}.json'
    if f.exists():return
    x=data();tr=capped(x,[held]);par=GRID[model][ci];scores={}
    for inner in sorted(tr.country.unique()):
        a,b=tr[tr.country.ne(inner)],tr[tr.country.eq(inner)];A,B=prep(a,b)
        p=fit_predict(model,par,A,B,a.phq4_w1.to_numpy(),weights(a));scores[inner]=wmae(b.phq4_w1.to_numpy(),p,b.weight.to_numpy())
    f.write_text(json.dumps({'held_out':held,'model':model,'candidate':ci,'params':par,'inner_MAE':scores,'mean_inner_MAE':float(np.mean(list(scores.values())))}))

def stage_inner(workers):
    from concurrent.futures import ProcessPoolExecutor
    (OUT/'inner').mkdir(parents=True,exist_ok=True);x=data();countries=sorted(x.country.unique());assert len(countries)==9
    jobs=[(h,m,i) for m in ['cebra','hgb','pca','ridge'] for h in countries for i in range(len(GRID[m]))]
    with ProcessPoolExecutor(workers) as ex:
        for k,_ in enumerate(ex.map(inner_job,jobs)):
            if (k+1)%10==0:print('inner jobs',k+1,'/',len(jobs),flush=True)

def outer_job(job):
    held,model,seed,shuffle=job;f=OUT/'outer'/f'{held}__{model}__{seed}__{int(shuffle)}.json'
    if f.exists():return
    x=data();tr=capped(x,[held]);te=x[x.country.eq(held)];assert held not in set(tr.country)
    sel=json.loads((OUT/'selected.json').read_text())[held].get(model,{})
    A,B=prep(tr,te);loss=[];y=tr.phq4_w1.to_numpy()
    p=fit_predict(model,sel,A,B,y,weights(tr),seed=seed,shuffle=shuffle,loss_out=loss)
    yt,wt=te.phq4_w1.to_numpy(),te.weight.to_numpy();e=yt-p
    r2=1-np.sum(wt*e**2)/np.sum(wt*(yt-np.average(yt,weights=wt))**2)
    f.write_text(json.dumps({'held_out':held,'model':model,'seed':seed,'shuffled':shuffle,'params':sel,'train_n':len(tr),'test_n':len(te),
        'MAE':wmae(yt,p,wt),'RMSE':float(np.sqrt(np.sum(wt*e**2)/np.sum(wt))),'R2':float(r2),'loss':loss}))

def stage_outer(workers):
    from concurrent.futures import ProcessPoolExecutor
    (OUT/'outer').mkdir(parents=True,exist_ok=True);x=data();countries=sorted(x.country.unique());sel={}
    rows=[json.loads(f.read_text()) for f in (OUT/'inner').glob('*.json')];assert len(rows)==9*sum(len(v) for v in GRID.values()),len(rows)
    inner=pd.DataFrame([{k:v for k,v in r.items() if k!='inner_MAE'} for r in rows]);inner['params']=inner.params.map(json.dumps)
    inner.to_csv(OUT/'inner_scores.csv',index=False)
    for h in countries:
        sel[h]={m:json.loads(inner[(inner.held_out==h)&(inner.model==m)].sort_values(['mean_inner_MAE','candidate']).iloc[0].params) for m in GRID}
    (OUT/'selected.json').write_text(json.dumps(sel,indent=2))
    jobs=[(h,'mean',42,False) for h in countries]+[(h,m,42,False) for h in countries for m in ['ridge','hgb','pca']]
    jobs+=[(h,'cebra',s,sh) for h in countries for s in [42,43,44] for sh in [False,True]]
    with ProcessPoolExecutor(workers) as ex:
        for k,_ in enumerate(ex.map(outer_job,jobs)):
            if (k+1)%10==0:print('outer jobs',k+1,'/',len(jobs),flush=True)

def stage_summary():
    rows=[json.loads(f.read_text()) for f in (OUT/'outer').glob('*.json')];assert len(rows)==9*4+9*6
    d=pd.DataFrame(rows);d['label']=np.where(d.model.eq('cebra')&d.shuffled,'cebra_shuffled',d.model)
    loss=[]
    for r in rows:
        if r['model']=='cebra':
            L=np.array(r['loss']);loss.append(dict(held_out=r['held_out'],seed=r['seed'],shuffled=r['shuffled'],iterations=len(L),
                last500=L[-500:].mean(),prev500=L[-1000:-500].mean(),relative_change=(L[-500:].mean()-L[-1000:-500].mean())/abs(L[-1000:-500].mean())))
            np.save(OUT/'outer'/f"loss_{r['held_out'].replace(' ','_')}_{r['seed']}_{int(r['shuffled'])}.npy",L)
    pd.DataFrame(loss).to_csv(OUT/'cebra_loss_convergence.csv',index=False)
    per=d.groupby(['held_out','label'],as_index=False)[['MAE','RMSE','R2']].mean();per.to_csv(OUT/'per_country_metrics.csv',index=False)
    summ=per.groupby('label',as_index=False).agg(countries=('held_out','size'),mean_MAE=('MAE','mean'),mean_RMSE=('RMSE','mean'),mean_R2=('R2','mean'))
    summ.to_csv(OUT/'summary.csv',index=False)
    piv=per.pivot(index='held_out',columns='label',values='MAE')
    old=pd.read_csv(BASE.parent/'gfs_public/per_country_metrics.csv').query("features=='individual_covariates'").groupby(['held_out_country','model']).MAE.mean().unstack()
    comp=dict(cebra_minus_ridge_mean=float((piv.cebra-piv.ridge).mean()),countries_cebra_better_than_ridge=int((piv.cebra<piv.ridge).sum()),
        cebra_minus_shuffled_mean=float((piv.cebra-piv.cebra_shuffled).mean()),countries_cebra_better_than_shuffled=int((piv.cebra<piv.cebra_shuffled).sum()),
        tuned_minus_fixed_ridge_mean=float((piv.ridge-old.loc[piv.index,'ridge']).mean()),tuned_minus_fixed_cebra_mean=float((piv.cebra-old.loc[piv.index,'CEBRA3_ridge']).mean()),
        best_model=summ.sort_values('mean_MAE').iloc[0].label,selected_cebra=pd.Series([json.dumps(v['cebra']) for v in json.loads((OUT/'selected.json').read_text()).values()]).value_counts().to_dict(),
        cebra_fits_with_relative_last500_change_below_1pct=int((pd.DataFrame(loss).relative_change.abs()<.01).sum()),cebra_final_fits=len(loss))
    (OUT/'comparisons.json').write_text(json.dumps(comp,indent=2)+'\n')
    (OUT/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'input_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256((BASE/'PROTOCOL.md').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print(summ.round(4).to_string());print(json.dumps(comp,indent=2))

if __name__=='__main__':
    stage=sys.argv[1];workers=int(sys.argv[2]) if len(sys.argv)>2 else 5
    {'inner':lambda:stage_inner(workers),'outer':lambda:stage_outer(workers),'summary':stage_summary}[stage]()
