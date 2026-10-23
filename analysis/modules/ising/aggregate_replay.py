"""Portable verification of public aggregate calculations; no individual data needed."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent

def main():
 out=BASE/'results'; checks={}
 metric=pd.read_csv(out/'heldout_metrics.csv'); comp=pd.read_csv(out/'paired_comparisons.csv'); draws=pd.read_csv(out/'prediction_bootstrap_draws.csv.gz')
 assert len(metric)==252 and len(comp)==28 and len(draws)==8000
 assert not metric.duplicated(['weighting','seed','model','metric','scope']).any()
 wide=metric.pivot(index=['weighting','seed','model','scope'],columns='metric',values='value')
 checks['mean_profile_decomposition_max_error']=float(np.max(abs(wide.item_Brier-wide.mean_Brier-wide.profile_Brier)))
 assert checks['mean_profile_decomposition_max_error']<1e-12
 worst=0.
 for row in comp.itertuples():
  scope=metric.query('weighting==@row.weighting and seed==42 and metric==@row.metric and scope=="all_complete_respondents"').set_index('model').value
  point=scope.ising-scope[row.reference]
  samples=draws.query('weighting==@row.weighting and reference==@row.reference')[row.metric]
  lo,hi=np.quantile(samples,[.025,.975])
  worst=max(worst,abs(point-row.difference),abs(lo-row.conditional_95_low),abs(hi-row.conditional_95_high))
 assert worst<1e-12;checks['contrast_and_quantile_max_error']=float(worst)
 pattern_tables=list(out.glob('full_pattern_probabilities_*.csv'));assert len(pattern_tables)==6
 for path in pattern_tables:
  table=pd.read_csv(path); assert len(table)==32 and table.pattern_id.tolist()==list(range(32)) and table.probability.gt(0).all()
  assert abs(table.probability.sum()-1)<1e-12
 counts=pd.read_csv(out/'joint_pattern_feasibility.csv'); assert len(counts)==32 and counts.women.sum()==29102 and counts.women.gt(0).sum()==23
 assert counts.loc[counts.pattern_id.ne(0),'women'].sum()==433
 edges=pd.read_csv(out/'network_edges.csv');assert len(edges)==10
 assert np.max(abs(np.exp(edges.log_conditional_odds_ratio)-edges.conditional_odds_ratio))<1e-11
 assert ((edges.bootstrap_low<=edges.bootstrap_high)&(edges.positive_fraction.between(0,1))).all()
 fit=json.loads((out/'FIT_SUMMARY.json').read_text());assert len(fit)==96 and all(x['success'] for x in fit)
 common=[x for x in fit if x['model']=='common_acceptance'];assert len(common)==32 and not any(x['scale_near_bound'] for x in common)
 assert max(x['quadrature_max_abs'] for x in common)<1e-8
 bindings=pd.read_csv(BASE/'NUMBER_TRACEABILITY.csv');assert len(bindings)==24
 for r in bindings.itertuples():
  path=BASE/r.source
  if path.suffix=='.json': value=json.loads(path.read_text())[r.field]
  else:
   table=pd.read_csv(path);filters=r.field.split(',');value_field='value'
   for clause in filters:
    if '=' in clause:
     key,value=clause.split('=',1)
     if key=='seed':table=table.loc[table[key].eq(int(value))]
     else:table=table.loc[table[key].eq(value)]
    else:value_field=clause
   assert len(table)==1;value=table[value_field].iloc[0]
  assert abs(float(value)-float(r.value))<1e-12,(r.name,value,r.value)
 checks.update(status='passed',metric_rows=len(metric),comparisons=len(comp),bootstrap_draw_rows=len(draws),joint_pattern_rows=len(counts),full_distribution_rows=192,edge_rows=len(edges),numerical_bindings=len(bindings),scope='Public aggregate identities, contrast points, saved-draw quantiles and report bindings only. Does not rerun source linkage, participant folds, model optimization or raw-data preparation.')
 print(json.dumps(checks,indent=2))
 return checks
if __name__=='__main__':main()
