"""Portable replay of all published point scores and paired saved-error ranges."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'

def main():
    cfg=json.loads((ROOT/'CONFIG.json').read_text());pred=pd.read_csv(OUT/'predictions.csv');coord=pd.read_csv(OUT/'coordinate_predictions.csv');metrics=pd.read_csv(OUT/'country_metrics.csv');pairs=pd.read_csv(OUT/'paired_comparisons.csv');diffs=pd.read_csv(OUT/'paired_country_differences.csv.gz');interval=pd.read_csv(OUT/'predictive_intervals.csv.gz');cov=pd.read_csv(OUT/'predictive_coverage.csv')
    from scipy.linalg import helmert
    q=helmert(5,full=False).T;keys=['uid','evaluation','target_country','target_iso3','target_survey','arm','seed','model'];checks=0;maximum=0
    for key,rows in pred.groupby(keys):
        rows=rows.set_index('item').loc[cfg['items']];y=rows.observed_pp.to_numpy();cc=coord
        for col,value in zip(keys,key):cc=cc.loc[cc[col].eq(value)]
        co=cc[[f'coordinate_{j}' for j in range(5)]].iloc[0].to_numpy();assert np.max(abs(rows.raw_prediction_pp.to_numpy()-(co[0]+co[1:]@q.T)))<1e-9
        for kind,p in [('raw',rows.raw_prediction_pp.to_numpy()),('bounded',rows.prediction_pp.to_numpy())]:
            e=p-y
            if kind=='raw':level=co[0]-y.mean();prof=(co[1:]-y@q)@q.T;order=co[1:]@q.T
            else:level=e.mean();prof=e-level;order=p
            w=[]
            for j in range(5):
                for k in range(j+1,5):
                    if y[j]==y[k]:continue
                    w.append(.5 if abs(order[j]-order[k])<=1e-10 else float(np.sign(order[j]-order[k])==np.sign(y[j]-y[k])))
            calc=dict(MAE_pp=np.abs(e).mean(),MSE_pp2=(e**2).mean(),level_MAE_pp=abs(level),level_MSE_pp2=level**2,profile_MAE_pp=np.abs(prof).mean(),profile_MSE_pp2=(prof**2).mean(),ordering_accuracy=np.mean(w),ordering_pairs=len(w),observed_ties=10-len(w))
            rr=metrics
            for col,value in zip(keys,key):rr=rr.loc[rr[col].eq(value)]
            rr=rr.loc[rr.prediction_kind.eq(kind)];assert len(rr)==1
            for col,val in calc.items():
                error=abs(float(rr.iloc[0][col])-val);maximum=max(maximum,error);assert error<1e-8,(key,kind,col,error);checks+=1
    bootchecks=0
    for r in pairs.itertuples():
        dd=diffs.loc[diffs.evaluation.eq(r.evaluation)&diffs.seed.eq(r.seed)&diffs.prediction_kind.eq(r.prediction_kind)&diffs.comparison.eq(r.comparison)&diffs.metric.eq(r.metric)].sort_values('uid');a=dd.difference.to_numpy();rng=np.random.default_rng(cfg['bootstrap_seed']+(0 if r.evaluation=='chronological42' else 1));inds=rng.integers(0,len(a),(cfg['bootstrap_draws'],len(a)));lo,hi=np.quantile(a[inds].mean(1),[.025,.975])
        for calc,expected in [(a.mean(),r.difference),(lo,r.lower),(hi,r.upper)]:assert abs(calc-expected)<1e-9;bootchecks+=1
    coverage=interval.groupby(['evaluation','arm','seed','model','space','coverage_level']).agg(coverage=('covered','mean'),width=('width','mean'),out_of_range_fraction=('out_of_range_fraction','mean')).reset_index();merged=coverage.merge(cov,on=['evaluation','arm','seed','model','space','coverage_level'],suffixes=('_replay','_stored'));assert len(merged)==len(cov)
    for col in ['coverage','width','out_of_range_fraction']:assert np.max(abs(merged[col+'_replay']-merged[col+'_stored']))<1e-9
    result=dict(status='passed',metric_bindings=checks,max_score_error=maximum,paired_bootstrap_bindings=bootchecks,coverage_rows=len(cov),scope='aggregate table replay; excludes independent source recovery, encoder retraining and posterior interval resimulation')
    (ROOT/'AGGREGATE_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.parse_args();main()
