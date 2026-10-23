"""Codebook-based public GFS cohort, preserving full samples for domain estimation."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[1]
COUNTRIES={1:'Argentina',2:'Australia',3:'Brazil',4:'Egypt',5:'Germany',6:'India',7:'Indonesia',8:'Israel',9:'Japan',10:'Kenya',11:'Mexico',12:'Nigeria',13:'Philippines',14:'Poland',16:'South Africa',17:'Spain',18:'Tanzania',19:'Turkey',20:'United Kingdom',22:'United States',23:'Sweden',24:'Hong Kong',25:'China'}

def valid(s,lo,hi):
    z=pd.to_numeric(s,errors='coerce');return z.where(z.between(lo,hi))

def main():
    out=BASE/'analysis';out.mkdir(exist_ok=True)
    src=BASE/'data/gfs_all_countries_wave2.csv'
    x=pd.read_csv(src,low_memory=False);assert x.ID.is_unique
    d=x[['ID','COUNTRY','PSU','STRATA','DOI_ANNUAL_Y1','DOI_ANNUAL_Y2']].copy()
    d['country']=x.COUNTRY.map(COUNTRIES);assert d.country.notna().all()
    d['female']=pd.to_numeric(x.GENDER,errors='coerce').eq(2).astype(int)
    d['age']=valid(x.AGE_Y1,18,99);d['age10']=d.age/10;d['age10_sq']=d.age10**2
    for short,field,hi in [('education','EDUCATION_3',3),('employment','EMPLOYMENT',8),('marital','MARITAL_STATUS',6),('income','INCOME_FEELINGS',4),('attendance','ATTEND_SVCS',5),('urbanicity','URBAN_RURAL',4)]:d[short]=valid(x[field+'_Y1'],1,hi)
    d['covid_loss']=valid(x.COVID_DEATH_Y1,1,2).map({1:1,2:0})
    for wave in [1,2]:
        z=pd.DataFrame({k:4-valid(x[k+f'_Y{wave}'],1,4) for k in ['DEPRESSED','INTEREST','FEEL_ANXIOUS','CONTROL_WORRY']})
        for label,items in [('phq2',['DEPRESSED','INTEREST']),('gad2',['FEEL_ANXIOUS','CONTROL_WORRY']),('phq4',list(z.columns))]:
            s=z[items].sum(axis=1,min_count=len(items));d[f'{label}_w{wave}']=s
            assert s.dropna().between(0,6 if label!='phq4' else 12).all()
            if label!='phq4':d[f'{label}_positive_w{wave}']=s.ge(3).astype(float).where(s.notna())
    for label,col in [('weight','ANNUAL_WEIGHT_C1'),('weight_c2','ANNUAL_WEIGHT_C2'),('weight_l2','ANNUAL_WEIGHT_L2')]:
        w=pd.to_numeric(x[col],errors='coerce');d[label]=w.where(w.gt(0)&np.isfinite(w))
    d['interview_date']=pd.to_datetime(x.DOI_ANNUAL_Y1,errors='coerce')
    d['followup_date']=pd.to_datetime(x.DOI_ANNUAL_Y2,errors='coerce')
    inventory=pd.read_csv(ROOT/'01_provenance_validation/expanded_api/survey_inventory.csv')
    contexts=[]
    for r in inventory.itertuples():
        records=json.loads((ROOT/f'01_provenance_validation/expanded_api/raw/{r.SurveyId}.json').read_text())['data']
        found=[v for v in records if v['IndicatorId']=='WE_AWBT_W_AGR' and v['CharacteristicCategory']=='Total' and v['IsTotal']==1]
        assert len(found)==1,(r.Country,len(found))
        v=found[0];contexts.append({'country':r.Country,'historical_norm':float(v['Value']),'context_year':v['SurveyYear'],'context_year_label':v['SurveyYearLabel'],'survey_id':v['SurveyId'],'indicator_id':v['IndicatorId'],'data_id':v['DataId'],'weighted_denominator':v['DenominatorWeighted'],'unweighted_denominator':v['DenominatorUnweighted']})
    ctx=pd.DataFrame(contexts);ctx.to_csv(out/'official_national_context.csv',index=False)
    d=d.merge(ctx[['country','historical_norm','context_year','survey_id']],on='country',how='left',validate='many_to_one')
    d['year_gap']=d.interview_date.dt.year-d.context_year
    # All enrolled respondents are adults. Retain unknown-age respondents in the
    # design and exclude them only from the analysis domain, preserving variance.
    d['valid_design']=d[['weight','STRATA','PSU']].notna().all(axis=1)
    d['eligible']=d.female.eq(1)&d.age.notna()&d.phq4_w1.notna()&d.historical_norm.notna()&d.valid_design
    cov=['covid_loss','education','employment','marital','income','attendance','urbanicity','age10','age10_sq']
    d['model_complete']=d.eligible&d[cov].notna().all(axis=1)
    # Keep all sexes and non-domain rows: R creates full design, then subsets domains.
    d.to_csv(out/'design_input.csv',index=False)
    cohort=d[d.eligible].copy();cohort.to_csv(out/'prediction_cohort.csv',index=False)
    assert cohort.country.nunique()==9 and cohort.ID.is_unique
    coverage=d.groupby('country').agg(all_rows=('ID','size'),women=('female','sum'),eligible=('eligible','sum'),complete_model=('model_complete','sum'),first_date=('interview_date','min'),last_date=('interview_date','max'),first_followup=('followup_date','min'),last_followup=('followup_date','max'),context_year=('context_year','first'),historical_norm=('historical_norm','first'),min_year_gap=('year_gap','min'),max_year_gap=('year_gap','max'))
    coverage.to_csv(out/'country_coverage.csv')
    cohort[cov].isna().sum().rename('missing').to_csv(out/'covariate_missingness.csv')
    ledger=[];mask=pd.Series(True,index=d.index)
    for label,keep in [('female',d.female.eq(1)),('age_18_99plus',d.age.notna()),('four_complete_symptom_items',d.phq4_w1.notna()),('nine_historical_overlap_countries',d.historical_norm.notna()),('valid_baseline_design',d.valid_design),('complete_model_covariates',d[cov].notna().all(axis=1))]:
        before=int(mask.sum());mask&=keep;ledger.append({'criterion':label,'before':before,'retained':int(mask.sum()),'excluded':before-int(mask.sum())})
    pd.DataFrame(ledger).to_csv(out/'exclusion_ledger.csv',index=False)
    report={'release_rows':len(x),'countries_in_release':int(d.country.nunique()),'baseline_prediction_women':len(cohort),'overlap_countries':sorted(cohort.country.unique()),'complete_case_model_women':int(d.model_complete.sum()),'followup_complete_eligible_women':int((d.eligible&d.phq4_w2.notna()&d.weight_l2.notna()).sum()),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'age_99_is_valid_top_code':True,'symptom_item_scoring':'4 minus codebook original value in 1..4; otherwise missing','clinical_diagnoses_available':False,'country_context_is_individual_exposure':False,'protocol_sha256':hashlib.sha256((BASE/'PROTOCOL.md').read_bytes()).hexdigest()}
    (out/'cohort_qc.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
