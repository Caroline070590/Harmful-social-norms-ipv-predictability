"""Exploratory country-held-out stress prediction with supervised CEBRA and baselines.
No held-out country stress labels enter preprocessing, representation fitting or readout.
Fixed settings are not tuned against these test results. Six countries imply a feasibility pilot.
This is not an unsupervised subgroup analysis or a convergence-validated CEBRA benchmark.
See ../method_audit_2026_10_19/ and the manuscript's Communications Psychology plan.
"""
from pathlib import Path
import os,sys,json,random,hashlib,itertools,platform
os.environ.setdefault('MPLCONFIGDIR','/tmp/violence-mpl')
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OMP_NUM_THREADS','2')
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parent
sys.path.insert(0,str(BASE/'vendor'))
import numpy as np
import pandas as pd
import torch
import cebra
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from scipy.spatial import procrustes
import matplotlib.pyplot as plt

CONFIG={'seeds':[42,43,44],'cebra_iterations':1000,'embedding_dimensions':3,'hidden_units':32,'batch_size':128,'learning_rate':0.001,'conditional':'delta','delta':0.3,'distance':'cosine','device':'cpu','ridge_alpha':10.,'evaluation':'leave one country out; primary output macro average over six countries','design_status':'exploratory fixed-setting pilot; no confirmatory inference; no causal claims','null_control':'shuffle only representation-training labels; readout still fits true training outcomes'}

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def main():
 torch.set_num_threads(2)
 datafile=ROOT/'04_mental_health/covid19/analysis/cohort.csv';x=pd.read_csv(datafile)
 out=BASE/'covid_pilot';out.mkdir(exist_ok=True)
 CONFIG['input_sha256']=hashlib.sha256(datafile.read_bytes()).hexdigest()
 (out/'config.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
 env={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'sklearn':sklearn.__version__,'cebra':cebra.__version__}
 (out/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
 numeric=['Dem_age','Dem_dependents','day']
 categorical=['Dem_edu','Dem_employment','Dem_maritalstatus','Dem_isolation']
 contextual=[c for c in x if c.startswith('norm_')]
 countries=sorted(x.country.unique());assert len(countries)>=3 and x.ID.is_unique
 rows=[];predictions=[];losses=[];stability=[];embeddings={}
 for held in countries:
  train=x.country.ne(held);test=~train
  assert set(x.loc[train,'country']).isdisjoint(set(x.loc[test,'country']))
  ytr=x.loc[train,'pss_sum_0_40'].to_numpy();yte=x.loc[test,'pss_sum_0_40'].to_numpy()
  def record(name,pred,spec,seed=-1):
   rows.append({'held_out_country':held,'features':spec,'model':name,'seed':seed,'train_n':int(train.sum()),'test_n':int(test.sum()),'MAE':mean_absolute_error(yte,pred),'RMSE':float(np.sqrt(mean_squared_error(yte,pred))),'R2':r2_score(yte,pred)})
   predictions.extend({'ID':int(i),'country':held,'features':spec,'model':name,'seed':seed,'observed':float(y),'predicted':float(p)} for i,y,p in zip(x.loc[test,'ID'],yte,pred))
  record('training_mean',np.full(len(yte),ytr.mean()),'baseline')
  for spec in ['demographics','demographics_plus_historical_context']:
   nums=numeric+(contextual if spec!='demographics' else [])
   pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nums),('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),categorical)])
   a=pre.fit_transform(x.loc[train]).astype(np.float32);b=pre.transform(x.loc[test]).astype(np.float32)
   assert np.isfinite(a).all() and np.isfinite(b).all()
   ridge=Ridge(alpha=CONFIG['ridge_alpha']).fit(a,ytr);record('ridge',ridge.predict(b),spec)
   hgb=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=10,random_state=42).fit(a,ytr);record('hist_gradient_boosting',hgb.predict(b),spec)
   pca=PCA(n_components=3,random_state=42).fit(a);pa=pca.transform(a);pb=pca.transform(b)
   readout=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(pa,ytr);record('PCA3_ridge',readout.predict(pb),spec)
   if held==countries[0] and spec!='demographics':
    emb=pd.DataFrame(np.vstack([pa,pb]),columns=['z1','z2','z3']);emb['split']=['train']*len(a)+['held_out_country']*len(b);emb['pss']=np.r_[ytr,yte];emb.to_csv(out/'PCA_example_fold.csv',index=False)
   label=(ytr-ytr.mean())/ytr.std()
   labels=label.astype(np.float32)[:,None]
   for seed in CONFIG['seeds']:
    for null in ([False,True] if seed==42 else [False]):
     seed_all(seed)
     aux=labels.copy()
     if null:aux=np.random.default_rng(seed).permutation(aux)
     # One independent respondent per input, with label-distance sampling rather than row-time sampling.
     # Make the previously implicit 0.5.0 default explicit; this preserves the recorded pilot settings.
     model=cebra.CEBRA(model_architecture='offset1-model',conditional=CONFIG['conditional'],delta=CONFIG['delta'],distance=CONFIG['distance'],temperature=1.0,batch_size=128,learning_rate=CONFIG['learning_rate'],max_iterations=CONFIG['cebra_iterations'],output_dimension=3,num_hidden_units=32,device='cpu',verbose=False)
     model.fit(a,aux)
     za=model.transform(a);zb=model.transform(b)
     assert za.shape==(len(a),3) and zb.shape==(len(b),3)
     ro=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(za,ytr)
     name='CEBRA3_ridge_shuffled_aux' if null else 'CEBRA3_ridge'
     record(name,ro.predict(zb),spec,seed)
     # Extract training losses without using held-out outcomes.
     loss=np.asarray(model.state_dict_['loss'],dtype=float).reshape(-1)
     losses.extend({'country':held,'features':spec,'seed':seed,'shuffled_aux':null,'iteration':i+1,'loss':float(v)} for i,v in enumerate(loss))
     if not null:embeddings[(held,spec,seed)]=zb
     if held==countries[0] and spec!='demographics' and seed==42 and not null:
      model.save(str(out/'example_fold_CEBRA.pt'))
      emb=pd.DataFrame(np.vstack([za,zb]),columns=['z1','z2','z3']);emb['split']=['train']*len(a)+['held_out_country']*len(b);emb['pss']=np.r_[ytr,yte];emb.to_csv(out/'CEBRA_example_fold.csv',index=False)
     print(held,spec,name,seed,'MAE',round(rows[-1]['MAE'],3),flush=True)
   # Save intermediate metrics so an interruption does not erase completed folds.
   pd.DataFrame(rows).to_csv(out/'per_country_metrics.csv',index=False)
 for (held,spec),group in itertools.groupby(sorted(embeddings),key=lambda k:k[:2]):
  keys=list(group)
  for akey,bkey in itertools.combinations(keys,2):
   _,_,dis=procrustes(embeddings[akey],embeddings[bkey]);stability.append({'country':held,'features':spec,'seed_a':akey[2],'seed_b':bkey[2],'heldout_procrustes_disparity':dis})
 pd.DataFrame(stability).to_csv(out/'embedding_seed_stability.csv',index=False)
 metrics=pd.DataFrame(rows);metrics.to_csv(out/'per_country_metrics.csv',index=False)
 pd.DataFrame(predictions).to_csv(out/'heldout_predictions.csv',index=False)
 pd.DataFrame(losses).to_csv(out/'training_losses.csv',index=False)
 # Average seeds within country first, then average countries equally.
 per=metrics.groupby(['features','model','held_out_country'])[['MAE','RMSE','R2']].mean().reset_index()
 summ=per.groupby(['features','model']).agg(country_count=('held_out_country','size'),macro_MAE=('MAE','mean'),macro_RMSE=('RMSE','mean'),macro_R2=('R2','mean')).reset_index();summ.to_csv(out/'macro_country_summary.csv',index=False)
 fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
 subset=summ[summ.features!='baseline'];labels=[r.features.replace('demographics_plus_historical_context','Demo + context').replace('demographics','Demo')+' / '+r.model for r in subset.itertuples()]
 axs[0].barh(labels,subset.macro_MAE,color=['#3a768b' if 'CEBRA' not in s else '#c26944' for s in labels]);axs[0].set(xlabel='Country-macro MAE, PSS 0–40 (lower is better)',title='A. Six-country held-out feasibility pilot')
 for name,grp in metrics.query("model=='CEBRA3_ridge'").groupby('features'):
  axs[1].scatter(grp.held_out_country,grp.MAE,label=name,alpha=.65)
 axs[1].tick_params(axis='x',rotation=45);axs[1].set(ylabel='Held-out MAE across three seeds',title='B. Variation across countries and seeds');axs[1].legend(fontsize=7)
 fig.savefig(out/'benchmark.png',dpi=600);fig.savefig(out/'benchmark.pdf');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for ax,name in zip(axs,['PCA','CEBRA']):
  e=pd.read_csv(out/f'{name}_example_fold.csv');tr=e.split.eq('train');ax.scatter(e.loc[tr,'z1'],e.loc[tr,'z2'],s=5,c='lightgray',alpha=.3);sc=ax.scatter(e.loc[~tr,'z1'],e.loc[~tr,'z2'],c=e.loc[~tr,'pss'],s=12,cmap='viridis');ax.set(title=f'{name}: held-out {countries[0]}',xlabel='Embedding dimension 1',ylabel='Embedding dimension 2');fig.colorbar(sc,ax=ax,label='Observed held-out PSS (display only)')
 fig.suptitle('One training fold only; CEBRA uses training stress labels, PCA does not')
 fig.savefig(out/'embeddings_example.png',dpi=600);fig.savefig(out/'embeddings_example.pdf');plt.close(fig)
 print(summ.to_string(index=False),flush=True)
if __name__=='__main__':main()
