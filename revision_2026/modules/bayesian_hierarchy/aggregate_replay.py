"""Independent-formula public aggregate score replay; no fitting or upstream downloads."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=BASE);args=parser.parse_args();base=args.root
    config=json.loads((base/'CONFIG.json').read_text());p=pd.read_csv(base/'results/predictions.csv');m=pd.read_csv(base/'results/country_metrics.csv')
    keys=['case','evaluation','target_survey','prior','model'];replayed=[]
    for name,g in p.groupby(keys,sort=True):
        g=g.set_index('item').loc[config['items']]
        y=g.observed_pp.to_numpy(float)
        for kind,column in [('raw','raw_prediction_pp'),('bounded','prediction_pp')]:
            pred=g[column].to_numpy(float);delta=pred-y;level=delta.mean();profile=(pred-pred.mean())-(y-y.mean())
            wins=[]
            for j in range(5):
                for k in range(j+1,5):
                    if y[j]==y[k]:continue
                    wins.append(.5 if abs(pred[j]-pred[k])<=1e-10 else float(np.sign(pred[j]-pred[k])==np.sign(y[j]-y[k])))
            row=dict(zip(keys,name));row.update(prediction_kind=kind,MAE_pp=np.abs(delta).mean(),MSE_pp2=np.mean(delta**2),level_MAE_pp=abs(level),level_MSE_pp2=level**2,profile_MAE_pp=np.abs(profile).mean(),profile_MSE_pp2=np.mean(profile**2),ordering_accuracy=np.mean(wins))
            replayed.append(row)
    result=pd.DataFrame(replayed);merged=result.merge(m,on=keys+['prediction_kind'],suffixes=('_new','_old'),validate='one_to_one')
    losses=['MAE_pp','MSE_pp2','level_MAE_pp','level_MSE_pp2','profile_MAE_pp','profile_MSE_pp2','ordering_accuracy']
    maximum=max(float(np.max(np.abs(merged[f'{v}_new']-merged[f'{v}_old']))) for v in losses)
    assert len(merged)==len(m) and maximum<1e-8
    means=result.groupby(['case','evaluation','prior','model','prediction_kind'])[losses].mean().reset_index()
    old=pd.read_csv(base/'results/model_summary.csv');j=means.merge(old,on=['case','evaluation','prior','model','prediction_kind'],suffixes=('_new','_old'),validate='one_to_one')
    summary_error=max(float(np.max(np.abs(j[f'{v}_new']-j[f'{v}_old']))) for v in losses)
    assert len(j)==len(old) and summary_error<1e-8
    comparisons=pd.read_csv(base/'results/paired_comparisons.csv')
    differences=pd.read_csv(base/'results/paired_country_differences.csv')
    refs=pd.read_csv(base/'results/reference_country_metrics.csv')
    rng=np.random.default_rng(config['bootstrap_seed']);ci_records=[];difference_records=[]
    plook={name:g.set_index('item').loc[config['items']] for name,g in p.groupby(['case','evaluation','prior','target_survey','model'],sort=True)}
    for (case,evaluation),domain in result.groupby(['case','evaluation'],sort=True):
        targets=sorted(domain.target_survey.unique());indices=rng.integers(0,len(targets),(config['bootstrap_draws'],len(targets)))
        for prior in config['priors']:
            for kind in ['raw','bounded']:
                groups={model:domain.loc[domain.prior.eq(prior)&domain.model.eq(model)&domain.prediction_kind.eq(kind)].set_index('target_survey').loc[targets] for model in config['models']}
                pairs=[('B1','B0'),('B2','B1'),('B2','B0')]
                if case=='questionnaire_compatible' and prior=='primary':
                    for model in ['R0','S','R1']:groups[model]=refs.loc[refs.evaluation.eq(evaluation)&refs.model.eq(model)&refs.prediction_kind.eq(kind)].set_index('target_survey').loc[targets]
                    pairs += [('B0','R0'),('B1','S'),('B2','R1')]
                for candidate,reference in pairs:
                    for metric in losses:
                        delta=groups[candidate][metric].to_numpy()-groups[reference][metric].to_numpy()
                        if (candidate,reference) in [('B1','B0'),('B2','B1')]:
                            for tindex,target in enumerate(targets):
                                ca=plook[(case,evaluation,prior,target,candidate)];re=plook[(case,evaluation,prior,target,reference)]
                                cr,rr=ca.raw_prediction_pp.to_numpy(float),re.raw_prediction_pp.to_numpy(float)
                                cb,rb=ca.prediction_pp.to_numpy(float),re.prediction_pp.to_numpy(float)
                                noclip=bool(np.all((cr>=0)&(cr<=100)) and np.all((rr>=0)&(rr<=100)) and np.array_equal(cr,cb) and np.array_equal(rr,rb))
                                allowed=kind=='raw' or noclip
                                identity=allowed and ((candidate=='B1' and reference=='B0' and metric in ['profile_MAE_pp','profile_MSE_pp2','ordering_accuracy']) or (candidate=='B2' and reference=='B1' and metric in ['level_MAE_pp','level_MSE_pp2']))
                                if allowed and candidate=='B2' and reference=='B1' and metric=='MAE_pp':
                                    obs=ca.observed_pp.to_numpy(float);ce=(cr if kind=='raw' else cb)-obs;re=(rr if kind=='raw' else rb)-obs
                                    identity=bool((np.all(ce>=0) and np.all(re>=0)) or (np.all(ce<=0) and np.all(re<=0)))
                                    if identity:assert abs(cr.mean()-rr.mean())<=1e-10
                                if identity:
                                    assert abs(delta[tindex])<=1e-10
                                    delta[tindex]=0.0
                        boot=delta[indices].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
                        row=dict(case=case,evaluation=evaluation,prior=prior,prediction_kind=kind,candidate=candidate,reference=reference,metric=metric)
                        ci_records.append({**row,'mean_difference':float(delta.mean()),'lower':float(lo),'upper':float(hi),'countries_improved':int(np.sum(delta>0 if metric=='ordering_accuracy' else delta<0))})
                        difference_records += [{**row,'target_survey':target,'difference':float(value)} for target,value in zip(targets,delta)]
    contrast_keys=['case','evaluation','prior','prediction_kind','candidate','reference','metric']
    j=pd.DataFrame(ci_records).merge(comparisons,on=contrast_keys,suffixes=('_new','_old'),validate='one_to_one')
    contrast_error=max(float(np.max(np.abs(j[f'{v}_new']-j[f'{v}_old']))) for v in ['mean_difference','lower','upper'])
    assert len(j)==len(comparisons) and contrast_error<1e-8
    assert np.array_equal(j.countries_improved_new,j.countries_improved_old)
    j=pd.DataFrame(difference_records).merge(differences,on=contrast_keys+['target_survey'],suffixes=('_new','_old'),validate='one_to_one')
    difference_error=float(np.max(np.abs(j.difference_new-j.difference_old)))
    assert len(j)==len(differences) and difference_error<1e-8
    intervals=pd.read_csv(base/'results/predictive_intervals.csv')
    coverage=intervals.groupby(['case','evaluation','prior','model','space','nominal_coverage']).agg(cells=('covered','size'),coverage=('covered','mean'),mean_width_pp=('width_pp','mean'),mean_out_of_range_probability=('out_of_range_probability','mean')).reset_index()
    stored=pd.read_csv(base/'results/predictive_coverage.csv')
    j=coverage.merge(stored,on=['case','evaluation','prior','model','space','nominal_coverage'],suffixes=('_new','_old'),validate='one_to_one')
    coverage_error=max(float(np.max(np.abs(j[f'{v}_new']-j[f'{v}_old']))) for v in ['cells','coverage','mean_width_pp','mean_out_of_range_probability'])
    assert len(j)==len(stored) and coverage_error<1e-8
    report={'status':'passed','prediction_cells':len(p),'country_score_rows':len(m),'summary_rows':len(old),'paired_comparisons':len(comparisons),'paired_country_differences':len(differences),'coverage_rows':len(stored),'maximum_country_metric_difference':maximum,'maximum_summary_difference':summary_error,'maximum_comparison_difference':contrast_error,'maximum_country_difference_discrepancy':difference_error,'maximum_coverage_difference':coverage_error,'scope':'Public aggregate point scores, model summaries, all fixed bootstrap comparisons, exact-identity normalization and saved-interval coverage aggregation. No original source extraction, posterior/model refitting or predictive-draw regeneration.'}
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
