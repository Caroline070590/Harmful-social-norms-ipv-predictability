"""Post-primary sensitivity: latest official national context before baseline fieldwork.
Selection uses dates and indicator availability, never mental-health outcome values.
"""
from pathlib import Path
import os,json,hashlib
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np,pandas as pd,requests
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,r2_score
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parents[1]
def main():
 out=BASE/'temporal_context';out.mkdir(exist_ok=True);raw=out/'api_snapshots';raw.mkdir(exist_ok=True)
 x=pd.read_csv(BASE/'analysis/prediction_cohort.csv',low_memory=False)
 surveys=json.loads((ROOT/'01_provenance_validation/expanded_api/raw/survey_metadata.json').read_text())['data']
 dates=pd.to_datetime(x.interview_date);cutoffs=dates.groupby(x.country).min()
 old=pd.read_csv(BASE/'analysis/official_national_context.csv').set_index('country')
 protocol={'status':'post_primary_exploratory_sensitivity','selection':'latest DHS survey with fieldwork ending before earliest eligible baseline interview, available national female any-reason indicator; fall back to earlier survey if missing','not_assumed':'publication before baseline or real-time forecasting; release dates recorded','no_outcome_based_selection':True,'models':'same capped samples, covariates, weights and fixed ridge/HGB settings as primary benchmark; no additional CEBRA tuning'}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 contexts=[];attempts=[]
 for country in sorted(x.country.unique()):
  sid=old.loc[country,'survey_id'];code=sid[:2]
  candidates=[s for s in surveys if s['DHS_CountryCode']==code and s['SurveyType']=='DHS' and s['FieldworkEnd'] and pd.Timestamp(s['FieldworkEnd'])<cutoffs[country]]
  for s in sorted(candidates,key=lambda z:z['FieldworkEnd'],reverse=True):
   p=raw/(s['SurveyId']+'.json')
   if not p.exists():
    r=requests.get('https://api.dhsprogram.com/rest/dhs/data',params={'indicatorIds':'WE_AWBT_W_AGR','surveyIds':s['SurveyId'],'breakdown':'all','perpage':1000,'f':'json'},timeout=(30,90));r.raise_for_status();j=r.json()
    assert int(j.get('TotalPages',1))<=1,'Unexpected pagination; stop rather than truncate'
    p.write_text(json.dumps({'url':r.url,'response':j},indent=2))
   j=json.loads(p.read_text())['response'];found=[v for v in j.get('Data',[]) if v['IsTotal']==1 and v['CharacteristicCategory']=='Total']
   attempts.append({'country':country,'survey_id':s['SurveyId'],'national_indicator_found':len(found)})
   if not found:continue
   assert len(found)==1
   v=found[0]
   original=next(m for m in surveys if m['SurveyId']==sid)
   contexts.append({'country':country,'earliest_baseline':str(cutoffs[country].date()),'old_survey':sid,'old_norm':old.loc[country,'historical_norm'],'new_survey':s['SurveyId'],'new_norm':float(v['Value']),'new_year_label':s['SurveyYearLabel'],'new_fieldwork_end':s['FieldworkEnd'],'new_publication_date':s['PublicationDate'],'new_release_date':s['ReleaseDate'],'old_women_universe':original['UniverseOfWomen'],'new_women_universe':s['UniverseOfWomen'],'old_min_age':original['MinAgeWomen'],'old_max_age':original['MaxAgeWomen'],'new_min_age':s['MinAgeWomen'],'new_max_age':s['MaxAgeWomen'],'indicator_data_id':v['DataId'],'same_survey':sid==s['SurveyId']})
   break
  else:raise ValueError('No eligible national indicator '+country)
 context=pd.DataFrame(contexts);context.to_csv(out/'context_inventory.csv',index=False);pd.DataFrame(attempts).to_csv(out/'availability_attempts.csv',index=False)
 (out/'download_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.glob('*.json')},indent=2)+'\n')
 cats=['education','employment','marital','income','urbanicity','attendance','covid_loss']
 for c in cats:x[c]=x[c].map(lambda v:'Missing' if pd.isna(v) else str(int(v)))
 x['new_context']=x.country.map(context.set_index('country').new_norm)
 rows=[]
 for held in sorted(x.country.unique()):
  tr=pd.concat([g.sample(n=min(1000,len(g)),random_state=2026) for _,g in x[x.country.ne(held)].groupby('country')]);te=x[x.country.eq(held)]
  w=tr.weight/tr.groupby('country').weight.transform('sum');w=w*len(w)/w.sum()
  for spec,col in [('individual_only',None),('historical_original','historical_norm'),('latest_prebaseline','new_context')]:
   nums=['age10','age10_sq']+([col] if col else [])
   pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nums),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
   a=pre.fit_transform(tr).astype(np.float32);b=pre.transform(te).astype(np.float32)
   models={'ridge':Ridge(alpha=10),'hist_gradient_boosting':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=10,random_state=42)}
   for name,model in models.items():
    model.fit(a,tr.phq4_w1,sample_weight=w);pred=model.predict(b);rows.append({'country':held,'context':spec,'model':name,'MAE':mean_absolute_error(te.phq4_w1,pred,sample_weight=te.weight),'R2':r2_score(te.phq4_w1,pred,sample_weight=te.weight),'train_n':len(tr),'test_n':len(te)})
 z=pd.DataFrame(rows);assert len(z)==54 and np.isfinite(z[['MAE','R2']]).all().all();z.to_csv(out/'per_country_metrics.csv',index=False)
 z.groupby(['context','model']).agg(countries=('country','size'),macro_MAE=('MAE','mean'),macro_R2=('R2','mean')).reset_index().to_csv(out/'summary.csv',index=False)
 print(context.to_string(index=False),flush=True);print('DONE 54 metric rows',flush=True)
if __name__=='__main__':main()
