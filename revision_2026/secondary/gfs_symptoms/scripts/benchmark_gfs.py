"""Fixed-setting PHQ-4 prototype; no test labels used in fitting.
Convergence and hyperparameter adequacy are unresolved; see the dated method audit.
"""
from pathlib import Path
import os,sys,json,random,hashlib,itertools
os.environ.setdefault('MPLCONFIGDIR','/tmp/violence-mpl');os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('OMP_NUM_THREADS','2')
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parent
sys.path.insert(0,str(BASE/'vendor'))
import numpy as np
import pandas as pd
import torch,cebra,sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from scipy.spatial import procrustes
import matplotlib.pyplot as plt

CONFIG={'iterations':1000,'cebra_seeds':[42,43,44],'training_sample_seed':2026,'max_train_per_country':1000,'output_dimension':3,'hidden_units':32,'ridge_alpha':10,'test':'leave-one-country-out','weights':'country-normalized training weights; survey-weighted held-out metrics; CEBRA representation sampling unweighted','context':'one official historical national female any-reason percentage; no individual norms','status':'exploratory fixed-setting protocol, no held-out test tuning'}

def main():
 torch.set_num_threads(2)
 out=BASE/'gfs_public';out.mkdir(exist_ok=True)
 src=ROOT/'04_mental_health/gfs_public/analysis/prediction_cohort.csv';x=pd.read_csv(src)
 CONFIG['input_sha256']=hashlib.sha256(src.read_bytes()).hexdigest();CONFIG['protocol_sha256']=hashlib.sha256((ROOT/'04_mental_health/gfs_public/PROTOCOL.md').read_bytes()).hexdigest()
 (out/'config.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
 (out/'environment.json').write_text(json.dumps({'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'cebra':cebra.__version__,'sklearn':sklearn.__version__},indent=2)+'\n')
 cats=['education','employment','marital','income','urbanicity','attendance','covid_loss']
 # Treat documented codes as categories; missing is a separate explicit level.
 for c in cats:x[c]=x[c].map(lambda v:'Missing' if pd.isna(v) else str(int(v)))
 countries=sorted(x.country.unique());assert len(countries)==9 and x.ID.is_unique
 rows=[];lossrows=[];sampleledger=[];embeds={}
 for held in countries:
  training=x[x.country.ne(held)]
  training=pd.concat([g.sample(n=min(len(g),CONFIG['max_train_per_country']),random_state=2026) for _,g in training.groupby('country')])
  test=x[x.country.eq(held)]
  assert set(training.ID).isdisjoint(set(test.ID)) and held not in training.country.unique()
  sampleledger.extend({'held_out_country':held,'training_country':cn,'available':int((x.country.eq(cn)).sum()),'selected':len(g)} for cn,g in training.groupby('country'))
  y=training.phq4_w1.to_numpy();yt=test.phq4_w1.to_numpy();w=(training.weight/training.groupby('country').weight.transform('sum')).to_numpy();w=w*len(w)/w.sum();wt=test.weight.to_numpy()
  def record(model,pred,features,seed=-1):
   assert np.isfinite(pred).all()
   rows.append({'held_out_country':held,'features':features,'model':model,'seed':seed,'train_n':len(training),'test_n':len(test),'MAE':mean_absolute_error(yt,pred,sample_weight=wt),'RMSE':float(np.sqrt(mean_squared_error(yt,pred,sample_weight=wt))),'R2':r2_score(yt,pred,sample_weight=wt)})
  record('training_mean',np.full(len(test),np.average(y,weights=w)),'baseline')
  for spec in ['individual_covariates','individual_plus_historical_context']:
   nums=['age10','age10_sq']+(['historical_norm'] if spec.endswith('context') else [])
   pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nums),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
   a=pre.fit_transform(training).astype(np.float32);b=pre.transform(test).astype(np.float32)
   assert np.isfinite(a).all() and np.isfinite(b).all()
   ridge=Ridge(alpha=10).fit(a,y,sample_weight=w);record('ridge',ridge.predict(b),spec)
   hgb=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=10,random_state=42).fit(a,y,sample_weight=w);record('hist_gradient_boosting',hgb.predict(b),spec)
   pc=PCA(n_components=3,random_state=42).fit(a);pa=pc.transform(a);pb=pc.transform(b)
   scaler=StandardScaler().fit(pa);readout=Ridge(alpha=10).fit(scaler.transform(pa),y,sample_weight=w);record('PCA3_ridge',readout.predict(scaler.transform(pb)),spec)
   labels=((y-y.mean())/y.std()).astype(np.float32)[:,None]
   for seed in CONFIG['cebra_seeds']:
    for null in ([False,True] if seed==42 else [False]):
     random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
     aux=np.random.default_rng(seed).permutation(labels) if null else labels
     # Single-respondent receptive field and non-temporal auxiliary-label sampler.
     # Explicit temperature preserves the previously implicit CEBRA 0.5.0 default.
     model=cebra.CEBRA(model_architecture='offset1-model',conditional='delta',delta=.3,distance='cosine',temperature=1.0,batch_size=128,learning_rate=.001,max_iterations=1000,output_dimension=3,num_hidden_units=32,device='cpu',verbose=False)
     model.fit(a,aux);za=model.transform(a);zb=model.transform(b)
     scaler=StandardScaler().fit(za);ro=Ridge(alpha=10).fit(scaler.transform(za),y,sample_weight=w)
     name='CEBRA3_ridge_shuffled_aux' if null else 'CEBRA3_ridge';record(name,ro.predict(scaler.transform(zb)),spec,seed)
     loss=np.asarray(model.state_dict_['loss'],dtype=float).reshape(-1)
     assert len(loss)==1000 and np.isfinite(loss).all()
     lossrows.append({'country':held,'features':spec,'seed':seed,'shuffled_aux':null,'first100_mean':loss[:100].mean(),'last100_mean':loss[-100:].mean(),'last300_slope':np.polyfit(np.arange(300),loss[-300:],1)[0]})
     if not null:embeds[(held,spec,seed)]=zb
     if held==countries[0] and spec.endswith('context') and seed==42 and not null:
      # Plot one fold only, with a bounded uniform display sample; do not align unrelated fold spaces.
      rng=np.random.default_rng(2026);it=rng.choice(len(a),min(2000,len(a)),replace=False)
      e=pd.DataFrame(np.vstack([za[it],zb]),columns=['z1','z2','z3']);e['split']=['training']*len(it)+['held_out_country']*len(b);e['observed_phq4']=np.r_[y[it],yt];e.to_csv(out/'example_embedding.csv',index=False)
     print(held,spec,name,seed,'weighted MAE',round(rows[-1]['MAE'],3),flush=True)
   pd.DataFrame(rows).to_csv(out/'per_country_metrics.csv',index=False)
 metrics=pd.DataFrame(rows);metrics.to_csv(out/'per_country_metrics.csv',index=False)
 pd.DataFrame(sampleledger).to_csv(out/'training_sample_ledger.csv',index=False)
 pd.DataFrame(lossrows).to_csv(out/'loss_diagnostics.csv',index=False)
 stability=[]
 for held in countries:
  for spec in ['individual_covariates','individual_plus_historical_context']:
   for s1,s2 in itertools.combinations(CONFIG['cebra_seeds'],2):
    _,_,dis=procrustes(embeds[(held,spec,s1)],embeds[(held,spec,s2)]);stability.append({'country':held,'features':spec,'seed_a':s1,'seed_b':s2,'procrustes_disparity':dis})
 pd.DataFrame(stability).to_csv(out/'embedding_seed_stability.csv',index=False)
 per=metrics.groupby(['features','model','held_out_country'])[['MAE','RMSE','R2']].mean().reset_index()
 summary=per.groupby(['features','model']).agg(countries=('held_out_country','size'),macro_MAE=('MAE','mean'),macro_RMSE=('RMSE','mean'),macro_R2=('R2','mean')).reset_index();summary.to_csv(out/'macro_country_summary.csv',index=False)
 p=per.query("features!='baseline'").pivot(index=['held_out_country','model'],columns='features',values='MAE').reset_index();p['context_minus_individual_MAE']=p.individual_plus_historical_context-p.individual_covariates;p.to_csv(out/'paired_context_increment.csv',index=False)
 from plot_gfs_benchmark import plot_benchmark
 plot_benchmark(out)
 e=pd.read_csv(out/'example_embedding.csv');tr=e.split.eq('training');fig,ax=plt.subplots(figsize=(7,5),layout='constrained');ax.scatter(e.loc[tr,'z1'],e.loc[tr,'z2'],s=3,c='lightgray',alpha=.3);sc=ax.scatter(e.loc[~tr,'z1'],e.loc[~tr,'z2'],s=8,c=e.loc[~tr,'observed_phq4'],cmap='viridis');fig.colorbar(sc,ax=ax,label='Observed held-out PHQ-4 (display only)');ax.set(title=f'One CEBRA fold: held-out {countries[0]}',xlabel='Embedding dimension 1',ylabel='Embedding dimension 2');fig.savefig(out/'embedding.pdf');fig.savefig(out/'embedding.png',dpi=600);plt.close(fig)
 assert len(metrics)==135 and len(lossrows)==72 and len(stability)==54
 (out/'qc_summary.json').write_text(json.dumps({'metric_rows':len(metrics),'held_out_countries':len(countries),'cebra_fits':len(lossrows),'seed_comparisons':len(stability),'outcomes_finite':bool(np.isfinite(metrics[['MAE','RMSE','R2']]).all().all()),'test_labels_used_to_fit':False},indent=2)+'\n');print(summary.to_string(index=False),flush=True)
if __name__=='__main__':main()
