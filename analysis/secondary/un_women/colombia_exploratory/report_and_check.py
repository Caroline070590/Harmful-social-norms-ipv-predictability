"""Independent weighted numerical checks, figures and an exploratory review report."""
from pathlib import Path
import os,hashlib,json,sys
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR','/tmp/violence-mpl')
import numpy as np,pandas as pd
from scipy.stats import t
import scipy,matplotlib
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent;BASE=OUT.parent
d=pd.read_csv(BASE/'data_local/pilot_coded_input.csv')
manifest=json.loads((OUT/'INPUT_MANIFEST.json').read_text())
assert hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest()==manifest['protocol_sha256']
assert hashlib.sha256((BASE/'data_local/pilot_coded_input.csv').read_bytes()).hexdigest()==manifest['coded_local_file_sha256']
m=pd.read_csv(OUT/'adjusted_symptom_associations.csv');desc=pd.read_csv(OUT/'reported_worsening.csv');means=pd.read_csv(OUT/'unadjusted_symptom_means.csv')
cohort=d[d.model_complete.eq(True)].copy();assert len(cohort)==779
x=pd.get_dummies(cohort[['AgeCat','Education','Locality','Marital_Status']],drop_first=True,dtype=float)
x.insert(0,'direct_ipv',cohort.direct_ipv);x.insert(0,'intercept',1.)
x=x.to_numpy();w=cohort.weight.to_numpy();bread=np.linalg.inv(x.T@(w[:,None]*x))
full_n=int(d.valid_weight.sum());errors=[];se_errors=[];mean_errors=[]
for r in m.itertuples():
    y=cohort[r.outcome].to_numpy()
    coef=np.linalg.lstsq(x*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
    residual=y-x@coef
    meat=x.T@(((w*residual)**2)[:,None]*x)
    # Independent full-design IDs: domain linearized contributions are zero
    # outside the model cohort; survey's correction uses the full sample n.
    variance=bread@meat@bread*full_n/(full_n-1)
    errors.append(abs(coef[1]-r.difference));se_errors.append(abs(np.sqrt(variance[1,1])-r.SE))
    assert r.n==len(cohort) and r.direct_positive==101 and r.converged
    assert np.isclose(r.lower,r.difference-t.ppf(.975,r.residual_df)*r.SE)
    assert np.isclose(r.upper,r.difference+t.ppf(.975,r.residual_df)*r.SE)
    assert np.isclose(r.p,2*t.sf(abs(r.difference/r.SE),r.residual_df))
for r in means.itertuples():
    g=cohort[cohort.direct_ipv.eq(r.direct_ipv)]
    mean_errors.append(abs(np.average(g[r.outcome],weights=g.weight)-r.estimate));assert len(g)==r.n
for r in desc.itertuples():
    g=d[d.Country.eq(r.country)&d.valid_weight.eq(True)&d[r.outcome].notna()]
    mean_errors.append(abs(np.average(g[r.outcome],weights=g.weight)-r.estimate))
    assert len(g)==r.n_valid and g[r.outcome].eq(1).sum()==r.n_yes
assert max(errors+se_errors+mean_errors)<1e-10
expected=np.minimum(1,np.maximum.accumulate(m.sort_values('p').p.to_numpy()*[2,1]))
assert np.allclose(expected,m.sort_values('p').holm_p)
log=(OUT/'run.log').read_text();assert 'Warning' not in log and 'Error' not in log
qc={'status':'passed','primary_models':2,'common_model_women':779,'direct_IPV_positive_model_women':101,'adjusted_coefficients_independently_recomputed':True,'approximate_independent_ID_covariance_independently_recomputed':True,'maximum_coefficient_difference':max(errors),'maximum_SE_difference':max(se_errors),'maximum_weighted_mean_difference':max(mean_errors),'descriptive_rows_checked':len(desc),'R_warnings':0,'protocol_and_coded_input_hashes_verified':True,'original_sampling_design_recovered':False,'causal_effect_estimated':False,'standard_PHQ_or_GAD_score_estimated':False}
timing_path=OUT/'post_primary_timing_sensitivity.csv'
timing=pd.read_csv(timing_path) if timing_path.exists() else pd.DataFrame()
if not timing.empty:
    sc=cohort[cohort.C26_1.ne('Only before')].copy();assert len(sc)==759
    sx=pd.get_dummies(sc[['AgeCat','Education','Locality','Marital_Status']],drop_first=True,dtype=float)
    sx.insert(0,'direct_ipv',sc.direct_ipv);sx.insert(0,'intercept',1.)
    sx=sx.to_numpy();sw=sc.weight.to_numpy();sbread=np.linalg.inv(sx.T@(sw[:,None]*sx))
    for r in timing.itertuples():
        sy=sc[r.outcome].to_numpy();coef=np.linalg.lstsq(sx*np.sqrt(sw[:,None]),sy*np.sqrt(sw),rcond=None)[0]
        residual=sy-sx@coef;sv=sbread@(sx.T@(((sw*residual)**2)[:,None]*sx))@sbread*full_n/(full_n-1)
        assert np.isclose(coef[1],r.difference,atol=1e-10) and np.isclose(np.sqrt(sv[1,1]),r.SE,atol=1e-10)
        assert np.isclose(r.lower,r.difference-t.ppf(.975,r.residual_df)*r.SE) and np.isclose(r.upper,r.difference+t.ppf(.975,r.residual_df)*r.SE)
        assert np.isclose(r.p_unadjusted,2*t.sf(abs(r.difference/r.SE),r.residual_df)) and r.direct_positive==81 and r.converged
    qc.update({'post_primary_timing_models_checked':2,'timing_sensitivity_women':759,'timing_sensitivity_IPV_positive':81,'timing_amendment_sha256':hashlib.sha256((OUT/'TIMING_SENSITIVITY_AMENDMENT.md').read_bytes()).hexdigest()})
(OUT/'QC.json').write_text(json.dumps(qc,indent=2)+'\n')
(OUT/'python_environment.json').write_text(json.dumps({'python':sys.version,'pandas':pd.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':1,'axes.spines.top':False,'axes.spines.right':False})
figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
countries=sorted(desc.country.unique());colors=dict(zip(countries,plt.get_cmap('viridis')(np.linspace(.08,.9,len(countries)))))
def save(fig,name):
    fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(figdir/(name+'.'+ext),dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,5.2),sharey=True)
for ax,outcome,title in zip(axes,['local_violence_increased','vignette_worse'],['a   Reported local violence increase','b   COVID worsened the domestic-violence vignette']):
    z=desc[desc.outcome.eq(outcome)].set_index('country').loc[countries]
    for i,(cn,r) in enumerate(z.iterrows()):
        ax.errorbar(r.estimate*100,i,xerr=[[(r.estimate-r.lower)*100],[(r.upper-r.estimate)*100]],fmt='o',color=colors[cn],markeredgecolor='black',markeredgewidth=.65,ecolor='#555555',elinewidth=.8,capsize=2,ms=5)
    ax.set_title(title,loc='left',weight='bold',fontsize=9);ax.set_xlim(0,100);ax.set_xlabel('Women giving this response (%)');ax.set_yticks(range(len(countries)));ax.set_yticklabels(countries)
axes[0].invert_yaxis();fig.text(.5,-.015,'Reported perceptions; weighted approximate pointwise 95% intervals. These are not observed incidence changes.',ha='center',fontsize=7);save(fig,'V1_reports_of_worsening')
fig,axes=plt.subplots(1,2,figsize=(9,3.1))
for ax,r,title,color in zip(axes,m.itertuples(),['a   Depressive-symptom item sum (0–6)','b   Worry item (0–3)'],['#0072B2','#D55E00']):
    ax.axvline(0,color='black',lw=.9,ls='--')
    ax.errorbar(r.difference,0,xerr=[[r.difference-r.lower],[r.upper-r.difference]],fmt='o',ms=8,color=color,markeredgecolor='black',ecolor='black',elinewidth=1.1,capsize=4)
    ax.set_yticks([]);ax.set_ylim(-.65,.7);ax.margins(x=.2);ax.set_title(title,loc='left',weight='bold',fontsize=9)
    ax.set_xlabel('Adjusted difference: reported IPV − never (points)')
    ax.text(.03,.95,f'{r.difference:.2f} ({r.lower:.2f} to {r.upper:.2f})\nHolm p = {r.holm_p:.3f}',transform=ax.transAxes,va='top',bbox={'facecolor':'white','edgecolor':'none','pad':2})
    ax.text(.03,.06,f'n = {r.n}; reported IPV: {r.direct_positive}',transform=ax.transAxes)
fig.text(.5,-.035,'Exploratory, modified symptom recall; approximate intervals assume independent respondents.',ha='center',fontsize=7);save(fig,'V2_colombia_symptom_associations')

q=json.loads((OUT/'COHORT_QC.json').read_text())
lines=['# Exploratory UN Women results: both author questions','',
'These are executed analyses of a newly downloaded public source. They are not changes to the manuscript, a causal COVID/IPV effect, validated clinical PHQ/GAD scores, or an independent replication of the source report. The protocol was saved before the new fits.','',
'## 1. Reports that violence or safety worsened','',
'There are 16,154 released women across 13 countries. The table reports country-weighted proportions among valid answers. It measures reported perceptions/retrospective change, not observed incidence before and after COVID. The domestic-violence-vignette question differs from the broad local-violence question and must not be presented as the same indicator.','',
'| Country | Local violence increased (%) | Valid n | Vignette worsened (%) | Valid n |','|---|---:|---:|---:|---:|']
for cn in countries:
    a=desc[desc.country.eq(cn)&desc.outcome.eq('local_violence_increased')].iloc[0]
    b=desc[desc.country.eq(cn)&desc.outcome.eq('vignette_worse')].iloc[0]
    lines.append(f'| {cn} | {100*a.estimate:.1f} | {a.n_valid} | {100*b.estimate:.1f} | {b.n_valid} |')
direct=desc[desc.outcome.eq('direct_ipv_frequency_increased')].iloc[0]
lines+=['',f"In Colombia, **{100*direct.estimate:.1f}%** of the **{direct.n_valid}** valid respondents to the conditional direct-IPV frequency-change question reported increased frequency (approximate 95% interval {100*direct.lower:.1f}–{100*direct.upper:.1f}%). The raw yes count was {direct.n_yes}. This is not the fraction of all Colombian women whose IPV increased. Refused/unknown responses were excluded and structural skips were not coded as no increase.",'',
f"A timing flag remains: {q['before_only_among_direct_positive']} direct-positive women reported before-only histories despite the direct question's last-12-month framing. Therefore these reports cannot establish incident post-COVID IPV or a causal change. No prospective violence time series has been constructed.",'',
'## 2. Direct reported violence and symptom items in Colombia','',
f"Of {q['colombia_released']} released Colombian women, {q['direct_eligible']} had a valid, non-inapplicable direct-IPV response. Excluding {q['direct_eligible_missing_symptoms']} incomplete symptom records and {q['direct_symptom_complete_missing_covariates']} additional incomplete covariate records left **{q['complete_models']} women**, including **{q['direct_positive_complete_models']} reporting direct IPV**. Full exclusions and coding are saved.",'',
'| Outcome | Adjusted difference, IPV versus never | Approximate 95% interval | Holm p, two primary tests |','|---|---:|---:|---:|']
for r in m.itertuples():lines.append(f'| {r.outcome} | {r.difference:.3f} | {r.lower:.3f} to {r.upper:.3f} | {r.holm_p:.4f} |')
if not timing.empty:
    lines+=['','### Explicitly post-primary timing sensitivity','',
    'A separately saved amendment, written after inspecting the primary estimates and before the sensitivity fits, excludes the 20 complete women reporting before-only histories. Both models retain 759 women, including 81 reporting IPV. The depressive-item association remains positive and the worry-item interval still includes zero. This is a sensitivity in the same cohort, not a new primary discovery.','',
    '| Outcome | Adjusted difference | Approximate 95% interval | Unadjusted descriptive p |','|---|---:|---:|---:|']
    for r in timing.itertuples():lines.append(f'| {r.outcome} | {r.difference:.3f} | {r.lower:.3f} to {r.upper:.3f} | {r.p_unadjusted:.4f} |')
lines+=['','The depressive-symptom item sum is higher among women reporting direct partner violence after adjustment for age category, education, locality and marital category. The worry-item estimate is imprecise and does not provide clear evidence of a difference. It is not evidence that violence has no relationship with anxiety.','',
'These Phase II questions use a since-COVID reference period. The two depressive items are summed 0–6 and the worry item ranges 0–3. This is not a standard two-week PHQ-2, a GAD-2, a clinical diagnosis or a change score. Phase I contains different last-month stress items and was not pooled into these outcomes. Report the actual item wording/reference periods in any manuscript use.','',
'The direct Colombian exposure question concerns specific physical partner-violence acts; it is not a complete inventory of emotional, sexual and physical IPV. No religion/faith/attendance field was identified in the released column inventory, so this model does not claim religion adjustment.','',
'## Uncertainty and interpretation','',
'Supplied country weights are used. Original PSU/strata are not released, so uncertainty assumes independent respondent IDs and does not recover the complete original survey design or nonresponse uncertainty. Intervals are approximate and country intervals pointwise. The two primary model p values are Holm-adjusted. No global pooled estimate is calculated.','',
'The cross-sectional association cannot establish whether violence preceded symptoms, whether COVID caused violence, or whether an individual treatment gap explains the association. Exposure and symptom recall periods differ, and income/food insecurity, pre-existing mental health and other unmeasured factors may matter. The direct results concern one selected Colombian analysis domain, not the full 13-country sample or all Colombian women.','',
'These findings concern different measurements and units from the earlier GFS bereavement/country-context analyses. Keep earlier negative and inconclusive results. Do not claim the present individual association validates earlier country-level prediction or establishes CEBRA superiority. Additional algorithms were not used.','',
'## Files and checks','',
'`reported_worsening.csv`, `adjusted_symptom_associations.csv`, `all_model_coefficients.csv`, `unadjusted_symptom_means.csv`, `broad_mental_impact_descriptive.csv`, source/coding ledgers, `QC.json`, `R_sessionInfo.txt` and figures provide traceability. The independent Python check reproduces both adjusted coefficients and approximate covariance from weighted estimating equations, all descriptive weighted means, intervals and Holm correction. Respondent rows remain local.','',
'Run `prepare_pilot.py`, then `analyze_pilot.R`, then `timing_sensitivity.R`, then `report_and_check.py` from the repository root using their project-relative paths. Source: [UN Women VAW rapid gender assessments](https://data.unwomen.org/publications/vaw-rga).']
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
(OUT/'FIGURE_LEGENDS.md').write_text('''# Pilot review figure legends

V1: Country-weighted reported local increase in violence/abuse/harassment (a), and COVID-related worsening for the domestic-violence vignette (b). These are distinct perception questions with valid-response denominators; they are not two estimates of actual IPV prevalence. Bars show approximate pointwise 95% intervals assuming independent respondents and using released weights. Original survey PSUs/strata are unavailable. All thirteen country points are shown.

V2: Adjusted differences associated with the specific direct physical partner-violence question versus never among 779 eligible, complete Colombian women. This is not a complete IPV inventory. Left: sum of two depressive-symptom items, 0–6. Right: one worry item, 0–3. The recall period is since COVID onset; these are not standard PHQ-2/GAD-2 scores or diagnoses. Models adjust original age, education, locality and marital categories. Bars show approximate 95% t intervals with independent-respondent weighted uncertainty. Holm correction covers the two displayed primary coefficients. No temporal or causal effect is identified. Points, lines and text are vector PDF/SVG; PNG files use 600 dpi.
''')
print(json.dumps(qc,indent=2));print(m.to_string(index=False));print('Colombia conditional increase:',direct.estimate,'n=',direct.n_valid)
