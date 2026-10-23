"""Post-primary country-profile extension. Run prepare, then R, then analyze."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/violence-mpl')
import argparse, hashlib, itertools, json, math, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr, rankdata
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
GFS=OUT.parent
EXT=ROOT/'02_aggregate_landscape/original_paper_extension'
FEATURES=['education_gap','rural_gap','national_acceptance']
LABELS={'education_gap':'Acceptance gap: no education − higher education (pp)', 'rural_gap':'Rural − urban (pp)', 'national_acceptance':'National any-reason acceptance (%)'}
PERM={}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(name,value): (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def prepare():
    inventory=pd.read_csv(GFS/'temporal_context/context_inventory.csv')
    source=pd.read_csv(EXT/'results/demographic_country_contrasts.csv')
    source=source[source.gender.eq('F') & source.Country.isin(inventory.country)].copy()
    source.rename(columns={'Country':'country'},inplace=True)
    assert source[['country','contrast']].duplicated().sum()==0
    table=source.pivot(index='country',columns='contrast',values='mean_component_contrast_pp')
    features=inventory[['country','old_survey','old_norm','old_women_universe','old_min_age','old_max_age','earliest_baseline']].copy()
    features['education_gap']=-features.country.map(table['education_higher_minus_none'])
    features['rural_gap']=features.country.map(table['rural_minus_urban'])
    features['national_acceptance']=features.old_norm
    assert len(features)==9 and features[FEATURES].notna().all().all()
    assert source.loc[source.contrast.isin(['education_higher_minus_none','rural_minus_urban']),'items'].eq(5).all()
    features.to_csv(OUT/'country_features.csv',index=False)
    contrasts=['education_higher_minus_none','rural_minus_urban','age35_49_minus15_24','cash_minus_unemployed','kind_minus_unemployed','married_minus_never']
    ledger=pd.MultiIndex.from_product([sorted(inventory.country),contrasts],names=['country','contrast']).to_frame(index=False)
    ledger=ledger.merge(source,on=['country','contrast'],how='left',validate='one_to_one')
    ledger['available']=ledger.mean_component_contrast_pp.notna()
    ledger.to_csv(OUT/'feature_availability.csv',index=False)
    inputs=[GFS/'analysis/design_input.csv',GFS/'temporal_context/context_inventory.csv',EXT/'results/demographic_country_contrasts.csv',OUT/'PROTOCOL.md']
    write_json('INPUT_MANIFEST.json',{'saved_utc':datetime.now(timezone.utc).isoformat(),'files':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'status':'exploratory; before extension outcome models','primary_feature':'education_gap','countries':sorted(features.country),'seed':2026})
    print(features.to_string(index=False))

def exact_rank(x,y):
    n=len(x); assert 5<=n<=9
    rx=rankdata(x);rx=rx-rx.mean();ry=rankdata(y);ry=ry-ry.mean()
    denom=np.linalg.norm(rx)*np.linalg.norm(ry); assert denom>0
    rho=float(rx@ry/denom)
    if n not in PERM: PERM[n]=np.array(list(itertools.permutations(range(n))),dtype=np.uint8)
    null=ry[PERM[n]]@rx/denom
    p=float(np.mean(np.abs(null)>=abs(rho)-1e-12))
    deleted=[float(spearmanr(np.delete(x,i),np.delete(y,i)).statistic) for i in range(n)]
    assert np.isclose(rho,spearmanr(x,y).statistic)
    return rho,p,min(deleted),max(deleted)

def holm(p):
    p=np.array(p);order=np.argsort(p);result=np.empty(len(p))
    result[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    return result

def correlations(features):
    means=pd.read_csv(OUT/'country_symptom_estimates.csv')
    rows=[]
    for symptom in ['gad2','phq2']:
        for when,target in [('baseline',symptom+'_w1'),('followup',symptom+'_w2'),('change','change_'+symptom)]:
            z=features.merge(means[means.outcome.eq(target)],on='country',validate='one_to_one')
            for feature in FEATURES:
                rho,p,lo,hi=exact_rank(z[feature].to_numpy(),z.estimate.to_numpy())
                primary=feature=='education_gap' and when=='followup'
                rows.append(dict(feature=feature,outcome=symptom,time=when,countries=len(z),rho=rho,exact_p=p,deletion_rho_min=lo,deletion_rho_max=hi,family='primary_two' if primary else 'secondary_sixteen'))
    result=pd.DataFrame(rows);result['holm_p']=np.nan
    for _,idx in result.groupby('family').groups.items():result.loc[idx,'holm_p']=holm(result.loc[idx,'exact_p'])
    result.to_csv(OUT/'country_correlations.csv',index=False)
    effects=pd.read_csv(OUT/'country_bereavement_coefficients.csv');rows=[]
    for symptom in ['gad2','phq2']:
        z=features.merge(effects[effects.outcome.eq(symptom)],on='country',validate='one_to_one')
        assert len(z)==9
        rho,p,lo,hi=exact_rank(z.education_gap.to_numpy(),z.coefficient.to_numpy())
        rows.append(dict(feature='education_gap',outcome=symptom,countries=len(z),rho=rho,exact_p=p,deletion_rho_min=lo,deletion_rho_max=hi))
    er=pd.DataFrame(rows);er['holm_p']=holm(er.exact_p);er.to_csv(OUT/'bereavement_profile_correlations.csv',index=False)
    return result,er

def prediction(features):
    d=pd.read_csv(GFS/'analysis/design_input.csv',low_memory=False)
    for col in ['eligible','model_complete','valid_design']:d[col]=d[col].astype(str).eq('True')
    d=d[d.eligible & d.model_complete & d.valid_design & d.phq4_w2.notna() & d.weight_l2.gt(0)].copy()
    d=d.merge(features[['country']+FEATURES],on='country',validate='many_to_one')
    assert len(d)==16926 and d.ID.is_unique and d[FEATURES].notna().all().all()
    cats=['education','employment','marital','income','urbanicity','attendance','covid_loss']
    assert d[cats].notna().all().all()
    for c in cats:d[c]=d[c].map(lambda z:str(int(z)))
    metrics=[]; splits=[];per_country=sorted(d.country.unique())
    for held in per_country:
        train=d[d.country.ne(held)];test=d[d.country.eq(held)]
        assert set(train.ID).isdisjoint(test.ID) and held not in set(train.country)
        w=(train.weight_l2/train.groupby('country').weight_l2.transform('sum')).to_numpy();w*=len(w)/w.sum()
        wt=test.weight_l2.to_numpy()
        for cn,g in train.groupby('country'):splits.append(dict(held_out_country=held,training_country=cn,n=len(g),weight_mass=float(w[train.country.eq(cn)].sum())))
        for outcome in ['gad2','phq2']:
            y=train[outcome+'_w2'].to_numpy();yt=test[outcome+'_w2'].to_numpy()
            def record(model,pred):
                assert np.isfinite(pred).all()
                metrics.append(dict(held_out_country=held,outcome=outcome,model=model,train_n=len(train),test_n=len(test),MAE=mean_absolute_error(yt,pred,sample_weight=wt),RMSE=np.sqrt(mean_squared_error(yt,pred,sample_weight=wt)),R2=r2_score(yt,pred,sample_weight=wt)))
            record('training_mean',np.repeat(np.average(y,weights=w),len(test)))
            record('baseline_persistence',test[outcome+'_w1'].to_numpy())
            for spec in ['individual']+FEATURES:
                nums=['age10','age10_sq',outcome+'_w1']+([] if spec=='individual' else [spec])
                assert train[nums].notna().all().all() and test[nums].notna().all().all()
                # All numeric predictors observed. Scaling is training-only, unweighted;
                # ridge risk and evaluation use longitudinal weights.
                pre=ColumnTransformer([('numeric',StandardScaler(),nums),('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
                a=pre.fit_transform(train);b=pre.transform(test)
                model=Ridge(alpha=10).fit(a,y,sample_weight=w)
                record(spec,model.predict(b))
    result=pd.DataFrame(metrics);result.to_csv(OUT/'prediction_country_metrics.csv',index=False)
    pd.DataFrame(splits).to_csv(OUT/'prediction_split_ledger.csv',index=False)
    summary=result.groupby(['outcome','model'],as_index=False).agg(countries=('held_out_country','nunique'),macro_MAE=('MAE','mean'),macro_RMSE=('RMSE','mean'),macro_R2=('R2','mean'),median_R2=('R2','median'))
    summary.to_csv(OUT/'prediction_summary.csv',index=False)
    individual=result[result.model.eq('individual')][['held_out_country','outcome','MAE']].rename(columns={'MAE':'individual_MAE'})
    increment=result[result.model.isin(FEATURES)].merge(individual,on=['held_out_country','outcome'],validate='many_to_one')
    increment['MAE_change']=increment.MAE-increment.individual_MAE
    increment.to_csv(OUT/'prediction_context_increment.csv',index=False)
    increment['improved']=increment.MAE_change.lt(0)
    incsum=increment.groupby(['outcome','model'],as_index=False).agg(countries=('held_out_country','nunique'),mean_MAE_change=('MAE_change','mean'),countries_improved=('improved','sum'),min_MAE_change=('MAE_change','min'),max_MAE_change=('MAE_change','max'))
    incsum.to_csv(OUT/'prediction_context_summary.csv',index=False)
    return summary,incsum,increment

def figures(features,corr,er,increment):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':'black','axes.linewidth':1,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    colors=dict(zip(sorted(features.country),['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00','#6A51A3','#7A8B00','#555555']))
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        for ext in ['pdf','svg','png']:fig.savefig(folder/(name+'.'+ext),dpi=600,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    means=pd.read_csv(OUT/'country_symptom_estimates.csv');effects=pd.read_csv(OUT/'country_bereavement_coefficients.csv')
    for kind in ['means','effects']:
        fig,axes=plt.subplots(1,2,figsize=(10,4.3))
        for ax,target,panel in zip(axes,['gad2','phq2'],['a','b']):
            table=means[means.outcome.eq(target+'_w2')] if kind=='means' else effects[effects.outcome.eq(target)]
            z=features.merge(table,on='country');v='estimate' if kind=='means' else 'coefficient'
            for row in z.itertuples():
                y=getattr(row,v)
                ax.errorbar(row.education_gap,y,yerr=[[y-row.lower],[row.upper-y]],fmt='o',ms=6,color=colors[row.country],markeredgecolor='black',markeredgewidth=.65,ecolor='#555555',elinewidth=.8,capsize=2,zorder=3)
                offset=(4,5)
                if kind=='effects':
                    offsets={'gad2':{'Indonesia':(-5,-19),'Philippines':(6,11),'India':(-8,-18),'Turkey':(5,9)},'phq2':{'Indonesia':(-6,16),'Philippines':(6,-18)}}
                    offset=offsets[target].get(row.country,offset)
                ax.annotate(row.country,(row.education_gap,y),xytext=offset,textcoords='offset points',fontsize=6.5)
            r=(corr[(corr.outcome.eq(target)) & corr.time.eq('followup') & corr.feature.eq('education_gap')].iloc[0] if kind=='means' else er[er.outcome.eq(target)].iloc[0])
            ax.set_title(f"{panel}   {'Anxiety (GAD-2)' if target=='gad2' else 'Depressive symptoms (PHQ-2)'}",loc='left',weight='bold')
            ax.set_xlabel(LABELS['education_gap']);ax.set_ylabel('Follow-up mean score (0–6)' if kind=='means' else 'Adjusted bereavement difference (points)')
            ax.text(.03,.97,f"9 countries; Spearman ρ = {r.rho:.2f}\nExact p = {r.exact_p:.3f}; Holm p = {r.holm_p:.3f}",transform=ax.transAxes,va='top',fontsize=8)
            ax.margins(x=.22,y=.22)
            if kind=='effects':ax.axhline(0,color='black',lw=.8,ls='--',zorder=0)
        save(fig,'CP1_country_followup_scores' if kind=='means' else 'CP3_bereavement_profile_diagnostic')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    jitter=dict(zip(sorted(features.country),np.linspace(-.16,.16,9)))
    for ax,target,panel in zip(axes,['gad2','phq2'],['a','b']):
        for i,spec in enumerate(FEATURES):
            z=increment[increment.outcome.eq(target)&increment.model.eq(spec)]
            for row in z.itertuples():ax.scatter(i+jitter[row.held_out_country],row.MAE_change,s=34,c=colors[row.held_out_country],edgecolors='black',linewidth=.6,zorder=3)
            ax.plot([i-.2,i+.2],[z.MAE_change.mean()]*2,color='black',lw=2,zorder=4)
        ax.axhline(0,color='black',lw=.8,ls='--');ax.set_xticks(range(3));ax.set_xticklabels(['Education gap','Rural–urban gap','National level']);ax.set_ylabel('Context − individual MAE (points)')
        ax.set_title(f"{panel}   {'Anxiety (GAD-2)' if target=='gad2' else 'Depressive symptoms (PHQ-2)'}",loc='left',weight='bold')
    handles=[plt.Line2D([],[],marker='o',ls='',markerfacecolor=color,markeredgecolor='black',label=cn,markersize=5) for cn,color in colors.items()]
    fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=7,bbox_to_anchor=(.5,-.08),frameon=False)
    save(fig,'CP2_country_profile_prediction')

def report(corr,er,summary,incsum):
    cov=pd.read_csv(OUT/'sample_coverage.csv');primary=corr[corr.family.eq('primary_two')]
    lines=['# Country attitude profiles and later anxiety/depressive symptoms','',
    'Executed post-primary exploratory extension; protocol saved before its outcome fits. These are new analyses of the existing GFS cohort, not independent replication. Manuscript integration is pending.','',
    f"Nine countries; {cov.paired_women.sum():,} paired women for design-aware country means and {cov.complete_model_women.sum():,} complete cases for prediction and adjusted bereavement models.",'',
    'The main feature is the historical female no-education minus higher-education difference in acceptance, averaged across five reasons (percentage points). This is between-group variation in published margins, not respondent variability. Primary outcomes are follow-up 2024 GAD-2 and PHQ-2 scores (0–6); country sample size is nine.','',
    '## Main country associations','',
    '| Outcome | Spearman rho | Exact permutation p | Holm p (2 tests) | Country-deletion rho range |','|---|---:|---:|---:|---|']
    for r in primary.itertuples():lines.append(f'| {r.outcome} | {r.rho:.3f} | {r.exact_p:.4f} | {r.holm_p:.4f} | {r.deletion_rho_min:.3f} to {r.deletion_rho_max:.3f} |')
    lines+=['','Country means and intervals use longitudinal weights and the full adult survey design before the women-domain subset. These rank tests give countries equal weight and do not propagate uncertainty in country means or DHS contrasts. Secondary baseline/change/other-feature correlations form a separate 16-test family in `country_correlations.csv`; do not select one as a new primary finding.','',
    '## Held-out-country prediction','',
    'Each outcome model includes its baseline symptom score and individual covariates. Each context model adds only one country feature. Fixed ridge alpha=10; all preprocessing is training-only, all women are used, and training countries have equal total longitudinal weight. MAE is weighted within held-out country and averaged equally across countries. Negative MAE change means improvement.','',
    '| Outcome | Added feature | MAE change | Countries improved / 9 |','|---|---|---:|---:|']
    for r in incsum.itertuples():lines.append(f'| {r.outcome} | {r.model} | {r.mean_MAE_change:+.4f} | {r.countries_improved} |')
    lines+=['','| Outcome | Reference/model | Mean MAE | Mean R2 |','|---|---|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {r.outcome} | {r.model} | {r.macro_MAE:.4f} | {r.macro_R2:.4f} |')
    lines+=['','Fold training samples overlap. These differences are descriptive; there is no fold-level significance test. This is retrospective transfer to a withheld country, not a deployment-time forecast. A small average change is not evidence of useful prediction without examining countries and baselines.','',
    '## Secondary bereavement diagnostic','',
    'Country-specific follow-up symptom coefficients adjust for the matching baseline score and measured individual covariates. The table correlates those estimated bereavement coefficients with the education contrast. This is a two-stage exploratory diagnostic, not a definitive pooled interaction test. Coefficient uncertainty is shown in the figure and source table but is not propagated through the rank test.','',
    '| Outcome | Spearman rho | Exact p | Holm p (2 tests) | Country-deletion rho range |','|---|---:|---:|---:|---|']
    for r in er.itertuples():lines.append(f'| {r.outcome} | {r.rho:.3f} | {r.exact_p:.4f} | {r.holm_p:.4f} | {r.deletion_rho_min:.3f} to {r.deletion_rho_max:.3f} |')
    lines+=['','## What these results support','',
    'The primary education-contrast correlations provide no clear evidence of a country-level relationship with later anxiety or depressive symptoms. Every tested one-feature context addition worsens average held-out-country MAE relative to its corresponding individual model. The secondary bereavement diagnostic is also inconclusive. These findings do not support making a positive country-norms-to-mental-health prediction claim. They do not establish absence of an individual relationship, or rule out other well-measured contextual features. With only nine country units and historical, heterogeneous context measurements, the extension is best treated as a transparent exploratory boundary analysis.','',
    '## Interpretation limits','',
    '- Positive country associations cannot show that an individual who accepts violence has worse mental health. GFS has no linked individual attitude or IPV measurement here.',
    '- COVID exposure is reported bereavement, not infection or a dated event. This cannot estimate pandemic-caused anxiety, depression, or clinical diagnoses.',
    '- Historical DHS source years differ; Egypt covers ever-married women and the other source inventories all women aged 15–49. GFS covers adult women, with different populations, translations and follow-up selection.',
    '- Baseline symptom adjustment concerns residual follow-up differences and may condition on symptoms that already reflect bereavement. It does not estimate the total causal effect.',
    '- Employment/marital contrasts with unavailable categories were excluded before fitting, not filled with substitute categories. No community or respondent variability is inferred from published margins.',
    '- Only nine selected country units support the country-level question. Deep embeddings cannot increase that number. These analyses cannot by themselves establish journal suitability.','',
    '## Reproduction','', 'Run from the repository root with the recorded Python/R environments:','',
    '```bash','python revision_2026/04_mental_health/gfs_public/country_profile_extension/run_extension.py prepare',
    'Rscript revision_2026/04_mental_health/gfs_public/country_profile_extension/estimate_country_outcomes.R',
    'python revision_2026/04_mental_health/gfs_public/country_profile_extension/run_extension.py analyze','```','',
    'Do not include respondent-level `design_input.csv` in a review archive. All outputs in this directory are country-level summaries or code; the input manifest records local source hashes.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def analyze():
    manifest=json.loads((OUT/'INPUT_MANIFEST.json').read_text())
    for path,digest in manifest['files'].items():assert sha(ROOT/path)==digest,path
    features=pd.read_csv(OUT/'country_features.csv')
    corr,er=correlations(features);summary,incsum,increment=prediction(features)
    figures(features,corr,er,increment);report(corr,er,summary,incsum)
    coefficients=pd.read_csv(OUT/'country_bereavement_coefficients.csv')
    assert len(coefficients)==18 and coefficients.converged.all() and (coefficients.residual_df>0).all()
    assert len(corr)==18 and corr.family.eq('primary_two').sum()==2 and summary.countries.eq(9).all()
    assert pd.read_csv(OUT/'model_failures.csv').empty
    assert np.isclose(exact_rank(np.arange(5),np.arange(5))[1],2/math.factorial(5))
    write_json('QC.json',{'status':'passed','paired_women':17186,'model_women':16926,'country_units':9,'country_correlations':18,'primary_country_tests':2,'secondary_country_tests':16,'bereavement_models':18,'model_failures':0,'prediction_outer_folds':9,'prediction_metric_rows':108,'respondent_rows_exported':False,'input_hashes_verified':True,'exact_permutation_sanity_check':'perfect five-country rank association p=2/5!','claims':'exploratory; no individual norms, diagnoses, pandemic effects or definitive interactions'})
    write_json('python_environment.json',{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__})
    write_json('OUTPUT_MANIFEST.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='OUTPUT_MANIFEST.json' and '__pycache__' not in str(p) and not p.name.endswith('.log')})
    print((OUT/'RESULTS.md').read_text())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','analyze']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else analyze()
