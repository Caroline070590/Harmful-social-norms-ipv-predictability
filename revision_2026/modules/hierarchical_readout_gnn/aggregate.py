"""Country-equal evaluation and compact exact-mean/conditional-interval summaries."""
from __future__ import annotations
import gzip,json
import numpy as np
import pandas as pd
import common as c
ROOT=c.ROOT;OUT=ROOT/'results'
METRICS=['MAE_pp','MSE_pp2','level_MAE_pp','level_MSE_pp2','profile_MAE_pp','profile_MSE_pp2','ordering_accuracy']

def loadgz(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)

def aggregate():
    assert (OUT/'FIT_COMPLETION.json').exists()
    tasks=loadgz(ROOT/'inputs/tasks.json.gz');predictions=[];scores=[];uncertainty=[];precision=[];identities=[];modelpoints=[]
    counter=0
    for task in tasks:
        key={k:task[k] for k in ['uid','evaluation','target_country','target_iso3','target_survey']};z=np.array(task['z']);zq=np.array(task['zquery']);yy=np.array(task['coordinates']);countries=task['countries'];obs=np.array(task['observed']);obscoords=np.r_[obs.mean(),obs@c.Q]
        baseline=c.readout(z,zq,yy,countries);basepoint=np.array(baseline['point']);baseblocks=c.conditional_coordinate_blocks(baseline,range(5))
        models=[('linear',0,'B2',basepoint,baseblocks)]
        for arm in c.CONFIG['arms']:
            for seed in c.CONFIG['seeds']:
                state=loadgz(OUT/'states'/f'{task["uid"]}_{arm}_{seed}.json.gz');mean=state['readouts']['mean'];profile=state['readouts']['profile']
                g1=np.r_[mean['point'],basepoint[1:]];g2=np.r_[mean['point'],profile['point']]
                g1blocks=c.conditional_coordinate_blocks(mean,[0])+baseblocks[1:]
                g2blocks=c.conditional_coordinate_blocks(mean,[0])+c.conditional_coordinate_blocks(profile,range(4))
                assert g1[0]==g2[0] and np.array_equal(g1[1:],basepoint[1:])
                identities.append(dict(**key,arm=arm,seed=seed,G1_G2_mean_identical=True,G1_B2_raw_profile_identical=True,max_mean_difference=float(abs(g1[0]-g2[0])),max_profile_difference=float(np.max(abs(g1[1:]-basepoint[1:])))))
                models.extend([(arm,seed,'G1',g1,g1blocks),(arm,seed,'G2',g2,g2blocks)])
        for arm,seed,model,point,blocks in models:
            full={**key,'arm':arm,'seed':seed,'model':model};raw=point@c.RECONSTRUCT.T;bounded=np.clip(raw,0,100)
            modelpoints.append({**full,**{f'coordinate_{j}':float(v) for j,v in enumerate(point)}})
            for j,item in enumerate(c.ITEMS):predictions.append(dict(**full,item=item,observed_pp=float(obs[j]),raw_prediction_pp=float(raw[j]),prediction_pp=float(bounded[j]),clipped=bool(raw[j]!=bounded[j])))
            for kind,p in [('raw',raw),('bounded',bounded)]:scores.append(dict(**full,prediction_kind=kind,**c.losses(obs,p,point if kind=='raw' else None)))
            intervals=c.predictive_intervals(blocks,c.CONFIG['posterior_seed']+counter*1009);counter+=1
            precision.append({**full,**{k:v for k,v in intervals.items() if k not in ['scenario','coordinate']}})
            for space,observed in [('scenario',obs),('coordinate',obscoords)]:
                for level,ends in intervals[space].items():
                    for j,(lo,hi,value) in enumerate(zip(*ends,observed)):
                        uncertainty.append(dict(**full,space=space,coordinate=j,coverage_level=float(level),lower=float(lo),upper=float(hi),observed=float(value),covered=bool(lo<=value<=hi),width=float(hi-lo),out_of_range_fraction=intervals['out_of_range_fraction']))
    pf=pd.DataFrame(predictions);mf=pd.DataFrame(scores);uf=pd.DataFrame(uncertainty)
    pf.to_csv(OUT/'predictions.csv',index=False);mf.to_csv(OUT/'country_metrics.csv',index=False);uf.to_csv(OUT/'predictive_intervals.csv.gz',index=False,compression='gzip');pd.DataFrame(modelpoints).to_csv(OUT/'coordinate_predictions.csv',index=False)
    c.save(OUT/'PREDICTIVE_PRECISION.json',dict(status='passed' if all(r['precision_passed'] for r in precision) else 'precision_limit_disclosed',records=precision));c.save(OUT/'HIERARCHY_IDENTITIES.json',dict(status='passed',raw_canonical_profile_evaluation=True,predicted_tie_tolerance_pp=1e-10,records=identities))
    keys=['evaluation','arm','seed','model','prediction_kind'];summary=mf.groupby(keys)[METRICS].mean().reset_index();summary['country_targets']=mf.groupby(keys).size().values;summary.to_csv(OUT/'model_summary.csv',index=False)
    cov=uf.groupby(['evaluation','arm','seed','model','space','coverage_level']).agg(coverage=('covered','mean'),width=('width','mean'),out_of_range_fraction=('out_of_range_fraction','mean')).reset_index();cov.to_csv(OUT/'predictive_coverage.csv',index=False)
    pairs=[];diffrows=[]
    for domain,domainrows in mf.groupby('evaluation'):
        for seed in c.CONFIG['seeds']:
            comparisons=[('real_G2_vs_linear_B2',('real',seed,'G2'),('linear',0,'B2')),('real_G1_vs_linear_B2',('real',seed,'G1'),('linear',0,'B2')),('real_G2_vs_self_G2',('real',seed,'G2'),('self_only',seed,'G2')),('real_G2_vs_random_G2',('real',seed,'G2'),('random',seed,'G2')),('real_G2_vs_G1',('real',seed,'G2'),('real',seed,'G1'))]
            for kind,rows in domainrows.groupby('prediction_kind'):
                targetids=sorted(rows.uid.unique());rng=np.random.default_rng(c.CONFIG['bootstrap_seed']+(0 if domain=='chronological42' else 1));inds=rng.integers(0,len(targetids),(c.CONFIG['bootstrap_draws'],len(targetids)))
                for name,left,right in comparisons:
                    ll=rows.loc[rows.arm.eq(left[0])&rows.seed.eq(left[1])&rows.model.eq(left[2])].set_index('uid').loc[targetids]
                    rr=rows.loc[rows.arm.eq(right[0])&rows.seed.eq(right[1])&rows.model.eq(right[2])].set_index('uid').loc[targetids]
                    for metric in METRICS:
                        diff=ll[metric].to_numpy()-rr[metric].to_numpy();boot=diff[inds].mean(1);lo,hi=np.quantile(boot,[.025,.975])
                        pairs.append(dict(evaluation=domain,seed=seed,prediction_kind=kind,comparison=name,metric=metric,difference=float(diff.mean()),lower=float(lo),upper=float(hi),targets=len(diff),conditional_fixed_prediction_bootstrap=True))
                        for uid,dd in zip(targetids,diff):diffrows.append(dict(evaluation=domain,seed=seed,prediction_kind=kind,comparison=name,metric=metric,uid=uid,difference=float(dd)))
    pd.DataFrame(pairs).to_csv(OUT/'paired_comparisons.csv',index=False);pd.DataFrame(diffrows).to_csv(OUT/'paired_country_differences.csv.gz',index=False,compression='gzip')
    previous=pd.read_csv(ROOT.parent/'bayesian_country_scenario_2026_10_22/results/predictions.csv');old=previous.loc[previous.case.eq('questionnaire_compatible')&previous.prior.eq('primary')&previous.model.eq('B2')];new=pf.loc[pf.model.eq('B2')]
    check=new.merge(old,on=['evaluation','target_survey','item'],suffixes=('_new','_old'));assert len(check)==305
    maximum=float(np.max(abs(check.raw_prediction_pp_new-check.raw_prediction_pp_old)));assert maximum<1e-8
    c.save(OUT/'AGGREGATION_COMPLETION.json',dict(status='completed_pending_independent_QC',models=len(modelpoints),encoder_fits=732,metric_rows=len(mf),prediction_rows=len(pf),interval_rows=len(uf),comparison_rows=len(pairs),linear_B2_match_max_abs_pp=maximum,precision_failures=sum(not r['precision_passed'] for r in precision),larger_streams=sum(len(r['attempts'])>1 for r in precision),representation_uncertainty_included=False))
    print(json.dumps(json.loads((OUT/'AGGREGATION_COMPLETION.json').read_text()),indent=2))

if __name__=='__main__':aggregate()
