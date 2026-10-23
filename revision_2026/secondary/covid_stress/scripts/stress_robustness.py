"""Country-count threshold and Turkey-exclusion sensitivities; no test-set tuning."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/violence-mpl');os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
BASE=Path(__file__).resolve().parents[1]

def main():
 out=BASE/'robustness';out.mkdir(exist_ok=True)
 config={'status':'post-pilot exploratory robustness; not a fresh confirmatory evaluation','thresholds':[50,100,200],'additional_cohort':'minimum100 excluding Turkey','models':'training mean, ridge10, HGB fixed100/7leaves, PCA3+ridge10; no additional CEBRA tuning','independence':'country-held-out, convenience sample; country contexts eight or fewer'}
 (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 rows=[];counts=[]
 for label,folder,exclude in [('minimum50','analysis_min50',False),('minimum100','analysis',False),('minimum200','analysis_min200',False),('minimum100_without_Turkey','analysis',True)]:
  x=pd.read_csv(BASE/folder/'cohort.csv')
  if exclude:x=x[x.country.ne('Turkey')].copy()
  counts.append({'cohort':label,'women':len(x),'countries':x.country.nunique(),'country_list':'; '.join(sorted(x.country.unique()))})
  for held in sorted(x.country.unique()):
   tr=x[x.country.ne(held)];te=x[x.country.eq(held)];y=tr.pss_sum_0_40;yt=te.pss_sum_0_40
   def record(name,pred,features):rows.append({'cohort':label,'country':held,'features':features,'model':name,'train_n':len(tr),'test_n':len(te),'MAE':mean_absolute_error(yt,pred),'RMSE':float(np.sqrt(mean_squared_error(yt,pred))),'R2':r2_score(yt,pred)})
   record('training_mean',np.full(len(te),y.mean()),'baseline')
   for spec in ['demographics','demographics_plus_historical_context']:
    nums=['Dem_age','Dem_dependents','day']+([c for c in x if c.startswith('norm_')] if spec.endswith('context') else [])
    cats=['Dem_edu','Dem_employment','Dem_maritalstatus','Dem_isolation']
    pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nums),('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),cats)])
    a=pre.fit_transform(tr);b=pre.transform(te)
    m=Ridge(alpha=10).fit(a,y);record('ridge',m.predict(b),spec)
    m=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=10,random_state=42).fit(a,y);record('hist_gradient_boosting',m.predict(b),spec)
    pc=PCA(n_components=3,random_state=42).fit(a);m=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(pc.transform(a),y);record('PCA3_ridge',m.predict(pc.transform(b)),spec)
  print(label,'complete',flush=True)
 z=pd.DataFrame(rows);z.to_csv(out/'per_country_metrics.csv',index=False)
 z.groupby(['cohort','features','model']).agg(countries=('country','size'),macro_MAE=('MAE','mean'),macro_RMSE=('RMSE','mean'),macro_R2=('R2','mean')).reset_index().to_csv(out/'summary.csv',index=False)
 pd.DataFrame(counts).to_csv(out/'cohort_inventory.csv',index=False)
 assert np.isfinite(z[['MAE','RMSE','R2']]).all().all()
 print('DONE',len(z),flush=True)
if __name__=='__main__':main()
