"""Code the public women-only release for the separately documented pilot."""
from pathlib import Path
import hashlib,json,sys
import numpy as np,pandas as pd

OUT=Path(__file__).resolve().parent;BASE=OUT.parent
raw=BASE/'data_local/unwomen_vaw_rga_public.xlsx'
fields=['Country','Respondent_Serial','weight','AgeCat','Education','Locality','Marital_Status','C08','C11','C20','C22','C26','C26_1','C26_2','mental_stress','B02_1_Phase2','B02_2_Phase2','B02_3_Phase2']
d=pd.read_excel(raw,usecols=fields)
assert len(d)==16154 and d.Country.nunique()==13 and not d.duplicated(['Country','Respondent_Serial']).any()
maps={
 'local_violence_increased':('C08',{'Yes, increased':1,'Yes, decreased':0,'No, stayed the same':0}),
 'vignette_worse':('C11',{'Worse':1,'Better':0,'Same':0}),
 'conflict_more_frequent':('C20',{'More frequent':1,'Less frequent':0,'About the same':0}),
 'home_safety_worse':('C22',{'Worse':1,'Better':0,'Same':0}),
 'direct_ipv':('C26',{'Often':1,'A few times':1,'Rarely':1,'Never':0}),
 'direct_ipv_frequency_increased':('C26_2',{'Yes – increased':1,'Yes – decreased':0,'No – stayed the same':0}),
 'negative_mental_impact':('mental_stress',{'Yes, negatively':1,'Yes, both negatively and positively':1,'Yes, positively':0,'No':0})}
unknown={'Refused',"Don't know",'Not applicable'}
coding=[]
for target,(src,mapping) in maps.items():
    original=d[src].astype('string').str.strip();bad=set(original.dropna())-set(mapping)-unknown
    assert not bad,(src,bad)
    d[target]=original.map(mapping).astype(float)
    for value,n in original.fillna('<missing>').value_counts().items():coding.append({'source':src,'target':target,'original_value':value,'n':int(n),'coded_value':mapping.get(value,'excluded_or_structural_skip')})
symptom_map={'Not at all':0,'Several days':1,'More than half the days':2,'Nearly every day':3}
for i in [1,2,3]:
    src='B02_'+str(i)+'_Phase2';original=d[src].astype('string').str.strip()
    bad=set(original.dropna())-set(symptom_map)-unknown;assert not bad,(src,bad)
    d['symptom_'+str(i)]=original.map(symptom_map).astype(float)
    for value,n in original.fillna('<missing>').value_counts().items():coding.append({'source':src,'target':'symptom_'+str(i),'original_value':value,'n':int(n),'coded_value':symptom_map.get(value,'excluded_or_structural_skip')})
d['worry_item']=d.symptom_1
d['depressive_item_sum']=d[['symptom_2','symptom_3']].sum(axis=1,min_count=2)
covars=['AgeCat','Education','Locality','Marital_Status']
for c in covars:
    d[c]=d[c].astype('string').str.strip();d.loc[d[c].isin(unknown),c]=pd.NA
d['valid_weight']=np.isfinite(d.weight)&d.weight.gt(0)
d['direct_complete']=d.Country.eq('Colombia')&d.direct_ipv.notna()
d['symptoms_complete']=d[['depressive_item_sum','worry_item']].notna().all(axis=1)
d['covariates_complete']=d[covars].notna().all(axis=1)
d['model_complete']=d.direct_complete&d.symptoms_complete&d.covariates_complete&d.valid_weight
assert d.loc[d.Country.ne('Colombia'),'direct_ipv'].isna().all()
assert d.loc[d.direct_complete,'direct_ipv'].eq(1).sum()==102
assert d.loc[d.direct_complete,'direct_ipv'].eq(0).sum()==704
local=BASE/'data_local/pilot_coded_input.csv';d.to_csv(local,index=False)
pd.DataFrame(coding).to_csv(OUT/'coding_ledger.csv',index=False)
rows=[]
for cn,g in d.groupby('Country'):
    rows.append({'country':cn,'released_women':len(g),'valid_weight':int(g.valid_weight.sum()),'valid_direct_ipv':int(g.direct_complete.sum()),'symptoms_complete':int(g.symptoms_complete.sum()),'complete_primary_models':int(g.model_complete.sum())})
pd.DataFrame(rows).to_csv(OUT/'sample_coverage.csv',index=False)
col=d[d.Country.eq('Colombia')]
exclusions={'colombia_released':len(col),'direct_not_applicable_or_invalid':int((~col.direct_complete).sum()),'direct_eligible':int(col.direct_complete.sum()),'direct_eligible_missing_symptoms':int((col.direct_complete&~col.symptoms_complete).sum()),'direct_symptom_complete_missing_covariates':int((col.direct_complete&col.symptoms_complete&~col.covariates_complete).sum()),'direct_symptom_covariate_complete_bad_weight':int((col.direct_complete&col.symptoms_complete&col.covariates_complete&~col.valid_weight).sum()),'complete_models':int(col.model_complete.sum()),'direct_positive_complete_models':int(col.loc[col.model_complete,'direct_ipv'].eq(1).sum()),'before_only_among_direct_positive':int((col.direct_ipv.eq(1)&col.C26_1.eq('Only before')).sum()),'no_fits_in_preparation':True}
assert exclusions['complete_models']+exclusions['direct_not_applicable_or_invalid']+exclusions['direct_eligible_missing_symptoms']+exclusions['direct_symptom_complete_missing_covariates']+exclusions['direct_symptom_covariate_complete_bad_weight']==len(col)
(OUT/'COHORT_QC.json').write_text(json.dumps(exclusions,indent=2)+'\n')
(OUT/'INPUT_MANIFEST.json').write_text(json.dumps({'raw_file_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'coded_local_file_sha256':hashlib.sha256(local.read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest(),'python':sys.version,'pandas':pd.__version__},indent=2)+'\n')
print(json.dumps(exclusions,indent=2))
