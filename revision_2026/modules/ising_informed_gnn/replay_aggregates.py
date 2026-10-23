"""Portable aggregate-only consistency checks; does not recreate respondent models."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.parse_args()
    m=pd.read_csv(BASE/'results/heldout_metrics.csv');c=pd.read_csv(BASE/'results/paired_comparisons.csv');b=pd.read_csv(BASE/'results/bootstrap_contrasts.csv.gz')
    assert np.isfinite(m.value).all() and np.isfinite(c[['difference','conditional_95_low','conditional_95_high']]).all().all()
    rows=0
    for r in c.itertuples(index=False):
        a=m[(m.weighting==r.weighting)&(m.partition==r.partition)&(m.model=='real_ising')&(m.scope=='all_complete')&(m.item=='mean')&(m.metric==r.metric)]
        z=m[(m.weighting==r.weighting)&(m.partition==r.partition)&(m.model==r.reference)&(m.scope=='all_complete')&(m.item=='mean')&(m.metric==r.metric)]
        np.testing.assert_allclose(r.difference,a.value.iloc[0]-z.value.iloc[0],rtol=0,atol=1e-14)
        d=b[(b.weighting==r.weighting)&(b.partition==r.partition)&(b.reference==r.reference)&(b.metric==r.metric)]
        assert len(d)==2000
        np.testing.assert_allclose([r.conditional_95_low,r.conditional_95_high],np.quantile(d.value,[.025,.975]),rtol=0,atol=1e-14);rows+=1
    assert m.model.nunique()==16 and len(c)==84
    report=dict(status='passed',scope='Aggregate score differences and all stored conditional percentile ranges only; no original-source, individual-prediction or model retraining replay.',metric_rows=len(m),paired_rows=rows,bootstrap_rows=len(b))
    (BASE/'AGGREGATE_REPLAY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
