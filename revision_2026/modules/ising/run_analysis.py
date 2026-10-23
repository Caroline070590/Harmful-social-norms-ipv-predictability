"""Exact finite-state response network; individual artifacts remain local."""
from pathlib import Path
import hashlib, importlib.util, json, platform, sys
from itertools import combinations
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logsumexp, roots_hermitenorm
from scipy.optimize import minimize

BASE=Path(__file__).resolve().parent
LOCAL=BASE/'data_local'; OUT=BASE/'results'
ITEMS=['V744'+x for x in 'ABCDE']
STATES=((np.arange(32)[:,None]>>np.arange(5))&1).astype(float)
PAIRS=list(combinations(range(5),2))
FEATURES=np.column_stack([STATES]+[STATES[:,i]*STATES[:,j] for i,j in PAIRS])
METRICS=['joint_NLL','pattern_Brier','item_Brier','mean_Brier','profile_Brier','conditional_NLL','conditional_Brier']


def save(path,value):
 path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def states_probability(theta,model,cfg,nodes=None):
 if model!='common_acceptance':
  phi=STATES if model=='independent' else FEATURES
  logits=phi@theta
  return np.exp(logits-logsumexp(logits))
 z,w=roots_hermitenorm(nodes or cfg['quadrature_points']); w=w/np.sqrt(2*np.pi)
 eta=theta[:5][None,:]+np.exp(theta[5])*z[:,None]
 logterms=eta@STATES.T-np.logaddexp(0,eta).sum(1)[:,None]
 return (w[:,None]*np.exp(logterms)).sum(0)


def make_objective(mass,model,cfg,nodes=None):
 n=mass.sum(); hprec=1/cfg['intercept_prior_sd']**2
 if model!='common_acceptance':
  phi=STATES if model=='independent' else FEATURES
  observed=mass@phi
  prec=np.r_[np.full(5,hprec),np.full(10,1/cfg['ising_interaction_prior_sd']**2)] if model=='ising' else np.full(5,hprec)
  def objective(theta):
   logits=phi@theta; logz=logsumexp(logits); p=np.exp(logits-logz)
   return float(n*logz-observed@theta+.5*np.sum(prec*theta**2)), n*(p@phi)-observed+prec*theta
  return objective
 z,w=roots_hermitenorm(nodes or cfg['quadrature_points']); lw=np.log(np.maximum(w/np.sqrt(2*np.pi),1e-300))
 def objective(theta):
  sig=np.exp(theta[5]); eta=theta[:5][None,:]+sig*z[:,None]
  probitem=expit(eta)
  logterms=eta@STATES.T-np.logaddexp(0,eta).sum(1)[:,None]+lw[:,None]
  logp=logsumexp(logterms,axis=0)
  # Responsibilities integrate each pattern's latent-score derivatives.
  resp=np.exp(logterms-logp[None,:]); delta=STATES[None,:,:]-probitem[:,None,:]
  d_h=(resp[:,:,None]*delta).sum(0)
  d_logsig=(resp*((delta.sum(2))*sig*z[:,None])).sum(0)
  penalty=.5*hprec*np.sum(theta[:5]**2)+.5*(sig/cfg['common_acceptance_scale_prior_sd'])**2
  grad=np.r_[-mass@d_h+hprec*theta[:5], -mass@d_logsig+(sig/cfg['common_acceptance_scale_prior_sd'])**2]
  return float(-mass@logp+penalty),grad
 return objective


def fit(mass,model,cfg):
 marg=mass@STATES/mass.sum(); init=np.log(np.clip(marg,1e-10,1-1e-10)/(1-np.clip(marg,1e-10,1-1e-10)))
 objective=make_objective(mass,model,cfg)
 starts=[init] if model=='independent' else ([np.r_[init,np.zeros(10)]] if model=='ising' else [np.r_[init,np.log(s)] for s in cfg['common_initial_scales']])
 results=[]
 for initial in starts:
  bounds=None if model!='common_acceptance' else [(None,None)]*5+[(np.log(cfg['common_scale_bounds'][0]),np.log(cfg['common_scale_bounds'][1]))]
  ans=minimize(objective,initial,method='L-BFGS-B',jac=True,bounds=bounds,options=dict(maxiter=cfg['maxiter'],ftol=cfg['ftol'],gtol=cfg['gtol'],maxls=100))
  results.append(ans)
 best=min(results,key=lambda a:a.fun)
 assert best.success and np.max(abs(best.jac))<.01,(model,best.message,best.jac)
 p=states_probability(best.x,model,cfg)
 assert p.min()>0 and abs(p.sum()-1)<1e-10
 info=dict(objective=float(best.fun),gradient_max_abs=float(np.max(abs(best.jac))),iterations=int(best.nit),success=bool(best.success),starts=[dict(objective=float(a.fun),success=bool(a.success),iterations=int(a.nit)) for a in results])
 if model=='common_acceptance':
  fine=states_probability(best.x,model,cfg,nodes=cfg['quadrature_check_points'])
  info['quadrature_max_abs']=float(max(abs(p-fine)))
  info['quadrature_max_relative']=float(max(abs(p-fine)/fine))
  info['quadrature_NLL_difference']=float(abs(mass@np.log(p/fine)/mass.sum()))
  info['scale']=float(np.exp(best.x[5])); info['scale_near_bound']=bool(np.exp(best.x[5])>=cfg['common_scale_bounds'][1]*.999)
  assert info['quadrature_max_abs']<cfg['quadrature_absolute_tolerance'] and info['quadrature_max_relative']<cfg['quadrature_relative_tolerance'] and info['quadrature_NLL_difference']<cfg['quadrature_NLL_tolerance'],info
 return best.x,p,info


def losses(pattern,p):
 # P may be a single32-vector or a woman-by32 array.
 p=np.broadcast_to(p,(len(pattern),32)); y=STATES[pattern]
 q=p[np.arange(len(pattern)),pattern]
 marg=p@STATES
 err=marg-y
 meanerr=err.mean(1)
 flip=pattern[:,None]^(1<<np.arange(5))[None,:]
 other=p[np.arange(len(pattern))[:,None],flip]
 actual=q[:,None]/(q[:,None]+other)
 condyes=np.where(y==1,actual,1-actual)
 return np.column_stack([-np.log(q),1+np.sum(p*p,axis=1)-2*q,np.mean(err*err,axis=1),meanerr**2,np.mean((err-meanerr[:,None])**2,axis=1),-np.log(actual).mean(1),np.mean((condyes-y)**2,axis=1)])


def prior_assignments(f,seed,k):
 path=BASE.parent/'endes_public/prediction/run_prediction.py'
 spec=importlib.util.spec_from_file_location('prior_endes_fold',path); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 return mod.assignments(f,seed,k)


def bootstrap_counts(f,cfg,draws):
 units=f[['stratum','psu']].drop_duplicates().sort_values(['stratum','psu']).reset_index(drop=True)
 units['unit']=np.arange(len(units))
 ix=f[['stratum','psu']].merge(units,on=['stratum','psu'],validate='many_to_one').unit.to_numpy()
 counts=np.zeros((draws,len(units)),dtype=np.int16); rng=np.random.default_rng(cfg['bootstrap_seed'])
 for _,g in units.groupby('stratum',sort=True):
  ids=g.unit.to_numpy(); counts[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=draws)
 return units,ix,counts


def main():
 cfg=json.loads((BASE/'CONFIG.json').read_text()); source=(BASE/cfg['source_cohort']).resolve()
 assert sha(source)==cfg['source_cohort_sha256']
 freeze=json.loads((BASE/'PROTOCOL_FREEZE.json').read_text())
 amend=json.loads((BASE/'PRE_FIT_NUMERICAL_AMENDMENT.json').read_text())
 assert sha(BASE/'CONFIG.json')==amend['CONFIG_sha256']
 f=pd.read_csv(source,dtype={'psu':str,'stratum':str,'V007':str,'row_id':str})
 y=f[ITEMS].to_numpy(int); pattern=(y*(2**np.arange(5))).sum(1)
 assert len(f)==29102 and y.any(1).sum()==433
 wraw=f.weight.to_numpy(float)
 state_records=[]; metric_rows=[]; folds_all=[]; prediction_losses={}; fold_mass_rows=[]
 for weighting in cfg['weightings']:
  w=wraw if weighting=='released_weight' else np.ones(len(f))
  for seed in cfg['assignment_seeds']:
   foldframe=prior_assignments(f,seed,cfg['folds']); folds=foldframe.fold.to_numpy(int)
   if weighting==cfg['weightings'][0]: folds_all.append(foldframe.assign(seed=seed))
   all_probs={m:np.zeros((len(f),32)) for m in cfg['models']}
   for fold in range(cfg['folds']):
    test=folds==fold; train=~test
    assert set(f.loc[train,'psu']).isdisjoint(f.loc[test,'psu'])
    mass=np.bincount(pattern[train],weights=w[train]/w[train].mean(),minlength=32)
    fold_mass_rows.append(dict(weighting=weighting,seed=seed,fold=fold,train_women=int(train.sum()),test_women=int(test.sum()),train_PSUs=int(f.loc[train,'psu'].nunique()),test_PSUs=int(f.loc[test,'psu'].nunique()),raw_training_weight_sum=float(w[train].sum()),normalization_factor=float(1/w[train].mean())))
    for model in cfg['models']:
     theta,p,info=fit(mass,model,cfg); key=f'{weighting}_seed{seed}_fold{fold}_{model}'
     np.savez_compressed(LOCAL/(key+'.npz'),theta=theta,probabilities=p,mass=mass)
     state_records.append(dict(key=key,weighting=weighting,seed=seed,fold=fold,model=model,scope='whole_PSU_heldout',**info))
     all_probs[model][test]=p
    print(f'{weighting} seed{seed} fold{fold+1}/5 complete',flush=True)
   for model,p in all_probs.items():
    ll=losses(pattern,p); prediction_losses[(weighting,seed,model)]=ll
    np.savez_compressed(LOCAL/f'losses_{weighting}_seed{seed}_{model}.npz',losses=ll)
    vals=np.average(ll,axis=0,weights=w)
    for name,val in zip(METRICS,vals): metric_rows.append(dict(weighting=weighting,seed=seed,model=model,metric=name,value=float(val),scope='all_complete_respondents'))
    for name,val in zip(METRICS,np.average(ll[y.any(1)],axis=0,weights=w[y.any(1)])): metric_rows.append(dict(weighting=weighting,seed=seed,model=model,metric=name,value=float(val),scope='any_endorser_descriptive_subset'))
  mass=np.bincount(pattern,weights=w/w.mean(),minlength=32)
  for model in cfg['models']:
   theta,p,info=fit(mass,model,cfg); key=f'{weighting}_full_{model}'
   np.savez_compressed(LOCAL/(key+'.npz'),theta=theta,probabilities=p,mass=mass)
   state_records.append(dict(key=key,weighting=weighting,model=model,scope='full_cohort_descriptive',**info))
   pd.DataFrame(dict(pattern_id=np.arange(32),probability=p)).assign(model=model,weighting=weighting).to_csv(OUT/f'full_pattern_probabilities_{weighting}_{model}.csv',index=False)
 pd.concat(folds_all).to_csv(LOCAL/'fold_assignments.csv.gz',index=False)
 save(LOCAL/'FIT_RECORDS.json',state_records); pd.DataFrame(fold_mass_rows).to_csv(OUT/'fold_counts.csv',index=False)
 metrics=pd.DataFrame(metric_rows); metrics.to_csv(OUT/'heldout_metrics.csv',index=False)
 units,ix,counts=bootstrap_counts(f,cfg,cfg['prediction_bootstrap_draws'])
 np.savez_compressed(LOCAL/'prediction_bootstrap_counts.npz',counts=counts)
 contrast_rows=[]; draw_frames=[]
 for weighting in cfg['weightings']:
  w=wraw if weighting=='released_weight' else np.ones(len(f)); denom=counts@np.bincount(ix,weights=w,minlength=len(units))
  for reference in ['independent','common_acceptance']:
   first=prediction_losses[(weighting,cfg['primary_partition'],'ising')]; second=prediction_losses[(weighting,cfg['primary_partition'],reference)]
   diff=first-second
   summed=np.column_stack([np.bincount(ix,weights=w*diff[:,j],minlength=len(units)) for j in range(len(METRICS))])
   draws=(counts@summed)/denom[:,None]
   draw_frames.append(pd.DataFrame(draws,columns=METRICS).assign(draw=np.arange(len(draws)),weighting=weighting,reference=reference))
   for j,name in enumerate(METRICS):
    lo,hi=np.quantile(draws[:,j],[.025,.975]); point=np.average(diff[:,j],weights=w)
    contrast_rows.append(dict(weighting=weighting,reference=reference,model='ising',metric=name,difference=float(point),conditional_95_low=float(lo),conditional_95_high=float(hi),bootstrap_draws=len(draws)))
 pd.DataFrame(contrast_rows).to_csv(OUT/'paired_comparisons.csv',index=False)
 pd.concat(draw_frames).to_csv(OUT/'prediction_bootstrap_draws.csv.gz',index=False)
 units,ix,counts=bootstrap_counts(f,cfg,cfg['stability_bootstrap_draws'])
 np.savez_compressed(LOCAL/'stability_bootstrap_counts.npz',counts=counts)
 unitmass=np.zeros((len(units),32)); np.add.at(unitmass,(ix,pattern),wraw)
 unitn=np.bincount(ix,minlength=len(units))
 edge_samples=[]; boot_meta=[]
 for d,c in enumerate(counts):
  mass=c@unitmass; norm=float((c@unitn)/mass.sum()); mass*=norm
  theta,p,info=fit(mass,'ising',cfg)
  np.savez_compressed(LOCAL/f'stability{d:03d}.npz',theta=theta,probabilities=p,mass=mass)
  edge_samples.append(theta[5:]); boot_meta.append(dict(draw=d,normalization=norm,**info))
  if (d+1)%50==0: print(f'Edge stability {d+1}/{len(counts)} complete',flush=True)
 edge_samples=np.asarray(edge_samples); np.savez_compressed(LOCAL/'edge_stability_samples.npz',coefficients=edge_samples)
 save(LOCAL/'STABILITY_FIT_RECORDS.json',boot_meta)
 full=np.load(LOCAL/'released_weight_full_ising.npz')['theta'][5:]
 edges=[]
 for k,(i,j) in enumerate(PAIRS):
  lo,hi=np.quantile(edge_samples[:,k],[.025,.975])
  edges.append(dict(item_i=ITEMS[i],item_j=ITEMS[j],log_conditional_odds_ratio=float(full[k]),conditional_odds_ratio=float(np.exp(full[k])),bootstrap_low=float(lo),bootstrap_high=float(hi),positive_fraction=float(np.mean(edge_samples[:,k]>0)),negative_fraction=float(np.mean(edge_samples[:,k]<0))))
 pd.DataFrame(edges).to_csv(OUT/'network_edges.csv',index=False)
 public_fit=[{k:v for k,v in r.items() if k!='key'} for r in state_records]
 save(OUT/'FIT_SUMMARY.json',public_fit)
 save(OUT/'BOOTSTRAP_SCOPE.json',dict(predictive_draws=cfg['prediction_bootstrap_draws'],edge_refit_draws=cfg['stability_bootstrap_draws'],PSUs=len(units),strata=units.stratum.nunique(),singleton_strata=int(units.groupby('stratum').size().eq(1).sum()),predictions_fixed=True,edges_refitted=True,singletons_fixed=True,national_design_uncertainty=False,full_retraining_predictive_uncertainty=False))
 save(OUT/'ENVIRONMENT.json',dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,platform=platform.platform()))
 save(OUT/'RUN_COMPLETE.json',dict(status='fitted_pending_independent_QC',CV_fits=90,full_descriptive_fits=6,edge_bootstrap_fits=200,source_sha256=sha(source),CONFIG_sha256=sha(BASE/'CONFIG.json'),main_manuscript_edited=False))
 print('All prespecified fits and aggregate scores complete.',flush=True)

if __name__=='__main__': main()
