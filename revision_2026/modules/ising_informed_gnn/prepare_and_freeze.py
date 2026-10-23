"""Source/fold/state and algorithm checks only; no outcome model fitting."""
from pathlib import Path
import hashlib,json,platform
import numpy as np,pandas as pd, scipy, sklearn, torch
from core import STATES,PAIRS,mask_invariance,graph,MaskedNet
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def prepare():
    cfg=json.loads((BASE/'CONFIG.json').read_text());src=(BASE/cfg['cohort_relative']).resolve();prior=(BASE/cfg['prior_study_relative']).resolve()
    assert sha(src)==cfg['cohort_sha256']
    f=pd.read_csv(src,dtype={'psu':str,'stratum':str,'row_id':str});folds=pd.read_csv(prior/'data_local/fold_assignments.csv.gz',dtype={'psu':str,'stratum':str,'row_id':str})
    y=f[cfg['items']].to_numpy(int);pattern=(y*(2**np.arange(5))).sum(1)
    assert len(f)==29102 and np.isin(y,[0,1]).all() and y.any(1).sum()==433
    bound={str(src):sha(src),str(prior/'data_local/fold_assignments.csv.gz'):sha(prior/'data_local/fold_assignments.csv.gz')};rows=[];minimum=np.inf
    for weighting in cfg['weightings']:
        w=f.weight.to_numpy(float) if weighting=='released_weight' else np.ones(len(f))
        for seed in cfg['assignment_seeds']:
            sf=folds.loc[folds.seed==seed].reset_index(drop=True)
            assert sf.row_id.tolist()==f.row_id.tolist() and sf.psu.tolist()==f.psu.tolist() and sf.stratum.tolist()==f.stratum.tolist()
            for fold in range(cfg['folds']):
                test=sf.fold.to_numpy()==fold;train=~test
                assert set(f.loc[train,'psu']).isdisjoint(set(f.loc[test,'psu']))
                mass=np.bincount(pattern[train],weights=w[train]/w[train].mean(),minlength=32)
                for model in ['independent','common_acceptance','ising']:
                    p=prior/'data_local'/f'{weighting}_seed{seed}_fold{fold}_{model}.npz';saved=np.load(p)
                    np.testing.assert_allclose(mass,saved['mass'],rtol=0,atol=1e-10)
                    assert saved['probabilities'].min()>0 and abs(saved['probabilities'].sum()-1)<1e-10
                    if model=='ising':
                        minimum=min(minimum,float(saved['theta'][5:].min()))
                        for arm in cfg['neural_arms']:
                            if arm!='conditional_MLP':
                                a=graph(saved['theta'],arm,cfg['node_permutation'])
                                np.testing.assert_allclose(np.abs(a).sum(1),1,atol=1e-14)
                    bound[str(p)]=sha(p)
                rows.append(dict(weighting=weighting,seed=seed,fold=fold,train_women=int(train.sum()),test_women=int(test.sum()),train_PSUs=int(f.loc[train,'psu'].nunique()),test_PSUs=int(f.loc[test,'psu'].nunique())))
    pd.DataFrame(rows).to_csv(BASE/'results/fold_counts.csv',index=False)
    env=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__,torch_threads=torch.get_num_threads())
    save(BASE/'ENVIRONMENT.json',env)
    params={arm:sum(p.numel() for p in MaskedNet(arm,42).parameters()) for arm in cfg['neural_arms']}
    assert set(params.values())=={137}
    gate=dict(status='passed',women=len(f),any_endorsers=int(y.any(1).sum()),PSUs=int(f.psu.nunique()),strata=int(f.stratum.nunique()),actual_2024=int((f.V007==2024).sum()),fold_sets=30,bound_prior_model_states=90,minimum_training_Ising_interaction=minimum,mask_gate=mask_invariance(),trainable_parameters=params,bindings=bound,scope='Reuses previously independently audited original-source cohort; exact source hash, fold rows, PSU separation and all90training masses/states checked. No original-archive reconstruction or outcome fitting.')
    save(BASE/'SOURCE_AND_ALGORITHM_GATE.json',gate)
    frozen=['PROTOCOL.md','CONFIG.json','core.py','run_analysis.py','prepare_and_freeze.py','SOURCE_AND_ALGORITHM_GATE.json','PRE_FIT_OBJECTIVE_AMENDMENT.json']
    save(BASE/'FREEZE.json',dict(status='frozen_before_new_neural_and_logistic_fits',files={name:sha(BASE/name) for name in frozen},upstream_bound_files=len(bound),outcome_models_fitted=False))
    print(json.dumps({k:v for k,v in gate.items() if k!='bindings'},indent=2))
if __name__=='__main__':prepare()
