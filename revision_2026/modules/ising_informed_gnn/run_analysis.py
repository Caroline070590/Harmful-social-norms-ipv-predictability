"""Fixed Ising-informed masked-item comparison; writes only this study."""
from pathlib import Path
import hashlib,json,time
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from core import STATES,CONFIG_ID,ZERO_STATES,TARGET,fit_neural,conditional_from_joint,losses
BASE=Path(__file__).resolve().parent;LOCAL=BASE/'data_local';OUT=BASE/'results'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def fit_logistic(mass,cfg):
    output=np.zeros((32,5));coef=[];intercept=[];checks=[]
    good=mass>0
    for j in range(5):
        x=STATES[:,np.arange(5)!=j]
        model=LogisticRegression(C=cfg['logistic_C'],max_iter=cfg['logistic_maxiter'],tol=cfg['logistic_tol'],solver='lbfgs')
        model.fit(x[good],STATES[good,j],sample_weight=mass[good])
        output[:,j]=model.predict_proba(x)[:,1]
        coef.append(model.coef_[0]);intercept.append(model.intercept_[0]);checks.append(int(model.n_iter_[0]))
    return output,np.array(coef),np.array(intercept),checks

def bootstrap(f,cfg):
    units=f[['stratum','psu']].drop_duplicates().sort_values(['stratum','psu']).reset_index(drop=True);units['unit']=np.arange(len(units))
    ix=f[['stratum','psu']].merge(units,on=['stratum','psu'],validate='many_to_one').unit.to_numpy()
    counts=np.zeros((cfg['bootstrap_draws'],len(units)),np.int16);rng=np.random.default_rng(cfg['bootstrap_seed'])
    for _,g in units.groupby('stratum',sort=True):
        ids=g.unit.to_numpy();counts[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=cfg['bootstrap_draws'])
    return units,ix,counts

def main():
    cfg=json.loads((BASE/'CONFIG.json').read_text());freeze=json.loads((BASE/'FREEZE.json').read_text())
    for name,h in freeze['files'].items():assert sha(BASE/name)==h,(name,'changed after freeze')
    gate=json.loads((BASE/'SOURCE_AND_ALGORITHM_GATE.json').read_text())
    for path,h in gate['bindings'].items():assert sha(path)==h
    source=(BASE/cfg['cohort_relative']).resolve();prior=(BASE/cfg['prior_study_relative']).resolve()
    f=pd.read_csv(source,dtype={'psu':str,'stratum':str,'row_id':str});folds=pd.read_csv(prior/'data_local/fold_assignments.csv.gz',dtype={'psu':str,'stratum':str,'row_id':str})
    y=f[cfg['items']].to_numpy(int);pattern=(y*(2**np.arange(5))).sum(1);metricrows=[];records=[];allpred={};start=time.time()
    for weighting in cfg['weightings']:
        w=f.weight.to_numpy(float) if weighting=='released_weight' else np.ones(len(f))
        for seed in cfg['assignment_seeds']:
            foldvec=folds.loc[folds.seed==seed,'fold'].to_numpy(int)
            names=['independent','common_acceptance','exact_ising','conditional_logistic']+[f'{a}_init{i}' for a in cfg['neural_arms'] for i in cfg['neural_seeds']]
            pred={name:np.zeros((len(f),5)) for name in names}
            for fold in range(cfg['folds']):
                test=foldvec==fold;train=~test;mass=np.bincount(pattern[train],weights=w[train]/w[train].mean(),minlength=32)
                ising=np.load(prior/'data_local'/f'{weighting}_seed{seed}_fold{fold}_ising.npz')
                for model,name in [('independent','independent'),('common_acceptance','common_acceptance'),('ising','exact_ising')]:
                    p=np.load(prior/'data_local'/f'{weighting}_seed{seed}_fold{fold}_{model}.npz')['probabilities'];q=conditional_from_joint(p)
                    pred[name][test]=q[pattern[test]]
                q,coef,intercept,niter=fit_logistic(mass,cfg);pred['conditional_logistic'][test]=q[pattern[test]]
                key=f'{weighting}_seed{seed}_fold{fold}_conditional_logistic';np.savez_compressed(LOCAL/(key+'.npz'),probabilities=q,coef=coef,intercept=intercept,mass=mass)
                records.append(dict(key=key,weighting=weighting,partition=seed,fold=fold,model='conditional_logistic',initialization=None,equations=5,iterations=niter))
                for arm in cfg['neural_arms']:
                    for initialization in cfg['neural_seeds']:
                        q,state,trace,info,a,offset=fit_neural(mass,ising['theta'],arm,initialization,cfg)
                        pred[f'{arm}_init{initialization}'][test]=q[pattern[test]]
                        key=f'{weighting}_seed{seed}_fold{fold}_{arm}_init{initialization}'
                        np.savez_compressed(LOCAL/(key+'.npz'),probabilities=q,mass=mass,adjacency=a,offset=offset,trace=trace,**state)
                        records.append(dict(key=key,weighting=weighting,partition=seed,fold=fold,model=arm,initialization=initialization,**info))
                print(f'{weighting} partition{seed} fold{fold+1}/5 complete; elapsed {time.time()-start:.1f}s',flush=True)
            for arm in cfg['neural_arms']:pred[arm]=np.mean([pred[f'{arm}_init{i}'] for i in cfg['neural_seeds']],axis=0)
            np.savez_compressed(LOCAL/f'predictions_{weighting}_seed{seed}.npz',**pred)
            allpred[(weighting,seed)]=pred
            for model,q in pred.items():
                ll=losses(pattern,q)
                for scope,selection in [('all_complete',np.ones(len(f),bool)),('any_endorser_descriptive',y.any(1))]:
                    for mi,metric in enumerate(['conditional_Brier','conditional_NLL']):
                        for item in ['mean']+cfg['items']:
                            v=ll[:,:,mi].mean(1) if item=='mean' else ll[:,cfg['items'].index(item),mi]
                            metricrows.append(dict(weighting=weighting,partition=seed,model=model,scope=scope,item=item,metric=metric,value=float(np.average(v[selection],weights=w[selection]))))
    pd.DataFrame(metricrows).to_csv(OUT/'heldout_metrics.csv',index=False);save(LOCAL/'FIT_RECORDS.json',records)
    units,ix,counts=bootstrap(f,cfg);np.savez_compressed(LOCAL/'bootstrap_counts.npz',counts=counts)
    refs=['exact_ising','common_acceptance','conditional_logistic','conditional_MLP','permuted_ising','self_message','independent'];rows=[];drawrows=[]
    for weighting in cfg['weightings']:
        w=f.weight.to_numpy(float) if weighting=='released_weight' else np.ones(len(f));denom=counts@np.bincount(ix,weights=w,minlength=len(units))
        for seed in cfg['assignment_seeds']:
            pred=allpred[(weighting,seed)];real=losses(pattern,pred['real_ising']).mean(1)
            for ref in refs:
                delta=real-losses(pattern,pred[ref]).mean(1)
                sums=np.column_stack([np.bincount(ix,weights=w*delta[:,j],minlength=len(units)) for j in range(2)])
                draws=(counts@sums)/denom[:,None]
                for j,metric in enumerate(['conditional_Brier','conditional_NLL']):
                    lo,hi=np.quantile(draws[:,j],[.025,.975]);point=np.average(delta[:,j],weights=w)
                    rows.append(dict(weighting=weighting,partition=seed,model='real_ising',reference=ref,metric=metric,difference=float(point),conditional_95_low=float(lo),conditional_95_high=float(hi)))
                    drawrows.append(pd.DataFrame(dict(draw=np.arange(len(draws)),value=draws[:,j])).assign(weighting=weighting,partition=seed,reference=ref,metric=metric))
    pd.DataFrame(rows).to_csv(OUT/'paired_comparisons.csv',index=False);pd.concat(drawrows).to_csv(OUT/'bootstrap_contrasts.csv.gz',index=False)
    save(OUT/'RUN_COMPLETE.json',dict(status='fitted_pending_independent_QC',neural_fits=240,logistic_blocks=30,logistic_equations=150,prior_states_reused=90,participants=len(f),masked_tasks_per_woman=5,neural_parameters=137,elapsed_seconds=time.time()-start,main_and_final_unchanged=True))
    print('Fixed producer fitting complete',flush=True)
if __name__=='__main__':main()
