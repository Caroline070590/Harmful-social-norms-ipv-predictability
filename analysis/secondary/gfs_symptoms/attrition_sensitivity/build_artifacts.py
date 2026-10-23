"""Generate descriptive figure and an explicit code/aggregate-only archive."""
from pathlib import Path
import hashlib
import json
import os
import zipfile
os.environ.setdefault('MPLCONFIGDIR','/tmp/gfs_attrition_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]


def main():
    assert json.loads((OUT/'INDEPENDENT_QC.json').read_text())['all_checks_passed']
    ret = pd.read_csv(OUT/'retention_by_country.csv').sort_values('baseline_weighted_retention')
    results = pd.read_csv(OUT/'followup_attrition_sensitivity.csv').set_index('specification')
    names = ['released_longitudinal_reference','baseline_complete_case','ipw_uncapped']
    labels = ['Released longitudinal weights','Baseline weights, retained women','Baseline weights + response model']
    colors = ['#D55E00','#777777','#0072B2']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':1.05,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig, axes = plt.subplots(1,2,figsize=(10.8,4.2),gridspec_kw={'width_ratios':[1,1.45]})
    a,b = axes
    yy = np.arange(len(ret))
    a.hlines(yy,0,ret.baseline_weighted_retention*100,color='#CCCCCC',linewidth=1.2)
    a.scatter(ret.baseline_weighted_retention*100,yy,s=42,color='#0072B2',edgecolor='black',linewidth=.5,zorder=3)
    a.set_yticks(yy);a.set_yticklabels(ret.country)
    a.set_xlabel('Follow-up retention (%)\nbaseline survey-weighted')
    a.set_xlim(0,80);a.set_xticks([0,20,40,60,80])
    a.set_title('A   Observed follow-up',loc='left',fontweight='bold',pad=12)
    for i,(name,label,col) in enumerate(zip(names,labels,colors)):
        row = results.loc[name]
        y = len(names)-1-i
        b.errorbar(row.coefficient_phq4_points,y,
                   xerr=[[row.coefficient_phq4_points-row.lower],[row.upper-row.coefficient_phq4_points]],
                   fmt='o',color=col,ecolor=col,elinewidth=1.6,capsize=3,markersize=6,
                   markeredgecolor='black',markeredgewidth=.5)
    b.axvline(0,color='black',linewidth=.8,linestyle='--',zorder=0)
    b.set_yticks([2,1,0]);b.set_yticklabels(labels)
    b.set_ylim(-.6,2.6);b.set_xlim(-.25,.65)
    b.set_xlabel('Adjusted COVID-bereavement difference\nin follow-up PHQ-4 points (95% CI)')
    b.set_title('B   Same follow-up outcome and covariates',loc='left',fontweight='bold',pad=12)
    fig.subplots_adjust(left=.12,right=.98,bottom=.23,top=.86,wspace=1.05)
    n = int(results.loc['ipw_uncapped','n'])
    target = int(results.loc['ipw_uncapped','baseline_target_n'])
    fig.text(.12,.04,f'{target:,} baseline-complete women; {n:,} observed at follow-up. '
             'Response-model intervals propagate fitted-weight uncertainty.',fontsize=8)
    dest = OUT/'figures';dest.mkdir(exist_ok=True)
    for ext in ['pdf','svg','png']:
        fig.savefig(dest/f'Fig_GFS_attrition_sensitivity.{ext}',dpi=600,facecolor='white',bbox_inches='tight')
    plt.close(fig)
    (dest/'CAPTION.md').write_text('''Country differences in observed follow-up and sensitivity to analysis-specific response weighting. Panel A shows baseline-survey-weighted proportions with complete later PHQ-4 and valid released longitudinal weight among baseline-complete women. Panel B shows the baseline-symptom-adjusted reported COVID-bereavement coefficient under released longitudinal weights, baseline weights alone and baseline weights divided by modeled retention probability, on the identical follow-up cohort. Horizontal bars are 95% t intervals. Response-model intervals propagate estimated retention probabilities by stacked estimating equations and PSU/stratum Taylor linearization; reference intervals use the original survey procedure. Released longitudinal weights already adjust attrition; the new response model is an alternative sensitivity with a baseline-complete target. The fixed .05 probability floor produces the same estimates because no fitted probability is below that floor. No interval identifies a causal bereavement or pandemic effect.\n''')
    # Archive code and aggregate outputs only. No respondent rows, raw source or libraries.
    own = ['PROTOCOL.md','README.md','RESULTS.md','EXECUTION_NOTES.md','PRE_FIT_MANIFEST.json',
           'INDEPENDENT_QC.json','run_attrition.R','check_and_report.py','build_artifacts.py','R_sessionInfo.txt',
           'followup_attrition_sensitivity.csv','retention_model_coefficients.csv','outcome_coefficients.csv',
           'estimating_equation_checks.csv','retention_by_country.csv','probability_quantiles.csv',
           'weight_diagnostics.csv','baseline_balance.csv','balance_summary.csv','model_terms.csv',
           'ipw_uncapped_covariance.csv','ipw_floor_005_covariance.csv','run_checks.csv','warnings.csv']
    files = [OUT/name for name in own]
    files += sorted(dest.glob('Fig_GFS_attrition_sensitivity.*')) + [dest/'CAPTION.md']
    extras = ['scripts/download_inputs.py','scripts/prepare_gfs.py','scripts/install_r_dependencies.R',
              'scripts/longitudinal_extension.R','PROTOCOL.md','LONGITUDINAL_AMENDMENT_2026_10_19.md',
              'longitudinal_extension/followup_bereavement_associations.csv']
    files += [OUT.parent/name for name in extras]
    manifest = {'description':'Code, aggregate diagnostics and figures only; public respondent inputs must be obtained separately',
                'respondent_data_included':False,
                'files':[{'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in files]}
    (OUT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files += [OUT/'ARTIFACT_MANIFEST.json']
    archive = OUT/'GFS_attrition_sensitivity_code_aggregate.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            relative = str(path.relative_to(ROOT))
            assert not any(v in relative for v in ['/data_local/','/data/','/analysis/'])
            z.write(path,relative)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(files)
    print(json.dumps({'archive':str(archive),'files':len(files),'bytes':archive.stat().st_size,
                      'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':
    main()
