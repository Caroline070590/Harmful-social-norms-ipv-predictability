"""Prepare a documented exploratory cohort and country-linked contextual predictors."""
from pathlib import Path
import json,re,unicodedata,hashlib,argparse
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[1]

def norm(x):return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode().lower())
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--minimum',type=int,default=100);parser.add_argument('--output-folder',default='analysis');args=parser.parse_args()
 out=BASE/args.output_folder;out.mkdir(exist_ok=True)
 path=BASE/'data/COVIDiSTRESS_May_30_cleaned_final.csv'
 x=pd.read_csv(path,encoding='latin1',low_memory=False)
 assert x.ID.is_unique
 raw=pd.read_csv(ROOT/'legacy/data/violence_against_women_girls_original.csv')
 mapping={norm(v):v for v in raw.Country.unique()}
 mapping.update({'thegambia':'Gambia','democraticrepublicofthecongo':'Congo Democratic Republic','congokinshasa':'Congo Democratic Republic','kyrgyzstan':'Kyrgyz Republic','swaziland':'Eswatini'})
 x['country']=x.Country.map(lambda v:mapping.get(norm(v)))
 items=[f'Scale_PSS10_UCLA_{i}' for i in range(1,11)]
 vals=x[items].apply(pd.to_numeric,errors='coerce')
 assert vals.stack().between(1,5).all()
 for i in [4,5,7,8]:vals[f'Scale_PSS10_UCLA_{i}']=6-vals[f'Scale_PSS10_UCLA_{i}']
 complete=vals.notna().all(axis=1)
 # Cleaning code uses 1–5 item codes, reversing items 4/5/7/8 with 6-x.
 # Convert the complete-item mean to the conventional 0–40 PSS sum.
 x['pss_sum_0_40']=(vals.mean(axis=1)-1)*10
 assert np.max(np.abs(vals.loc[complete].mean(axis=1)-x.loc[complete,'PSS10_avg']))<1e-10
 dates=pd.to_datetime(x.RecordedDate,errors='coerce')
 x['day']=((dates-pd.Timestamp('2020-03-30')).dt.total_seconds()/86400)
 ledger=[]
 def retain(mask,label):
  nonlocal x,complete,dates
  before=len(x);x=x.loc[mask].copy();ledger.append({'step':label,'before':before,'retained':len(x),'excluded':before-len(x)})
 retain(x.Dem_gender.eq('Female'),'female respondents in author-cleaned dataset')
 retain(x.Dem_age.between(18,100),'age 18–100')
 retain(complete.loc[x.index],'all 10 PSS items present')
 retain(dates.loc[x.index].between('2020-03-30','2020-05-30 23:59:59'),'recorded within documented collection window')
 # Include non-overlap countries in a separate descriptive coverage table, not model data.
 x.groupby(['Country','country'],dropna=False).agg(n_women=('ID','size'),mean_pss=('pss_sum_0_40','mean')).reset_index().to_csv(out/'all_country_women_coverage.csv',index=False)
 retain(x.country.notna(),'exact or explicitly aliased legacy country match')
 country_counts=x.groupby('country').size();country_counts.rename('n_complete_women').to_csv(out/'overlap_country_counts.csv')
 selected=country_counts[country_counts>=args.minimum].index
 retain(x.country.isin(selected),f'at least {args.minimum} eligible women per country; feasibility threshold')
 audit=pd.read_csv(ROOT/'01_provenance_validation/expanded_api/results/matched_cells.csv')
 # Context consists of ten separate official female urban/rural scenario estimates.
 # No national prevalence is created by averaging demographic margins.
 a=audit[(audit.gender=='F') & (audit.dimension=='Residence') & audit.question_key.ne('any_reason')].copy()
 assert not a.duplicated(['Country','response','question_key']).any()
 a['feature']='norm_'+a.response.str.lower()+'_'+a.question_key
 context=a.pivot(index='Country',columns='feature',values='official_value')
 context=context.reindex(selected)
 context['context_survey_year']=a.groupby('Country').year.first()
 context.to_csv(out/'official_country_context_features.csv')
 ctx=[c for c in context if c.startswith('norm_')]
 if len(ctx)!=10:raise ValueError('Expected five scenarios × urban/rural')
 no_context=context[ctx].isna().any(axis=1)
 if no_context.any():raise ValueError('Incomplete official contextual profiles: '+str(context.index[no_context].tolist()))
 x=x.merge(context,left_on='country',right_index=True,validate='many_to_one')
 x['survey_year_gap']=2020-x.context_survey_year
 cols=['ID','Country','country','UserLanguage','Dem_age','Dem_edu','Dem_employment','Dem_maritalstatus','Dem_dependents','Dem_isolation','day','pss_sum_0_40','survey_year_gap']+ctx
 cohort=x[cols].copy();cohort.to_csv(out/'cohort.csv',index=False)
 cohort.groupby('country').agg(n=('ID','size'),mean_pss=('pss_sum_0_40','mean'),sd_pss=('pss_sum_0_40','std'),historical_gap=('survey_year_gap','first')).reset_index().to_csv(out/'cohort_country_summary.csv',index=False)
 pd.DataFrame(ledger).to_csv(out/'exclusion_ledger.csv',index=False)
 pd.DataFrame({'variable':cols,'missing':cohort[cols].isna().sum().values}).to_csv(out/'cohort_missingness.csv',index=False)
 report={'source_rows':125306,'downloaded_rows':int(pd.read_csv(path,encoding='latin1',usecols=['ID']).shape[0]),'eligible_overlap_countries_before_threshold':len(country_counts),'pilot_countries':len(selected),'pilot_women':len(cohort),'pss_complete_item_scoring_matches_published_mean':True,'pss_range':[0,40],'minimum_country_women':100,'readiness':'exploratory_pilot_only; few independent country contexts','country_context_source':'independently retrieved official DHS API, female urban/rural five scenarios; any_reason excluded','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'country_aliases':{'The Gambia':'Gambia','Kyrgyzstan':'Kyrgyz Republic','Swaziland':'Eswatini','Democratic Republic of the Congo':'Congo Democratic Republic'},'not_established':['national representativeness','causal effect of COVID-19','individual linkage to historical respondents','individual wife-beating beliefs or IPV in COVID respondents']}
 report['minimum_country_women']=args.minimum
 (out/'cohort_readiness.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
