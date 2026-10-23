"""Fixed PSU-held-out, equal-person ENDES prediction; respondent rows stay local."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import helmert
import scipy
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
import sklearn

BASE = Path(__file__).resolve().parent
STUDY = BASE.parent
REV = STUDY.parents[1]
LOCAL = BASE/'data_local'
RESULTS = BASE/'results'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load():
    cfg = json.loads((BASE/'CONFIG.json').read_text())
    ready = json.loads((STUDY/'MODEL_READY.json').read_text())
    assert ready['status'] == 'ready_for_equal_person_conditional_prediction_only'
    path = STUDY/ready['cohort_path']
    assert sha(path) == ready['summary']['prepared_cohort_sha256']
    dtype = {name:str for name in ready['categorical_covariates']+[ready['id_column'],ready['group_column'],ready['stratum_column']]}
    frame = pd.read_csv(path,dtype=dtype,keep_default_na=False)
    assert len(frame) == ready['summary']['prepared_records'] and frame.row_id.is_unique
    assert frame[ready['attitude_items']].isin([0,1]).all().all()
    assert frame.phq9_total.between(0,27).all() and frame.age.between(18,49).all()
    assert frame.groupby('psu').stratum.nunique().max() == 1
    assert frame.analysis_weight_equal.eq(1).all()
    assert np.array_equal(frame[ready['attitude_items']].sum(1),frame.norm_score_0_5)
    h = helmert(5,full=False)
    assert np.allclose(h@h.T,np.eye(4)) and np.allclose(h.sum(1),0)
    profile = frame[ready['attitude_items']].to_numpy(float)@h.T
    frame['age_squared'] = frame.age**2
    for i in range(4):
        frame['attitude_profile_'+str(i+1)] = profile[:,i]
    reconstruction = frame.norm_score_0_5.to_numpy()[:,None]/5 + profile@h
    assert np.allclose(reconstruction,frame[ready['attitude_items']].to_numpy(),atol=1e-14)
    return cfg,ready,frame


def blocks(ready):
    demo = ['age','age_squared']
    ipv = ready['IPV_primary_recent']
    total = ['norm_score_0_5']
    pattern = ['attitude_profile_'+str(i+1) for i in range(4)]
    return dict(M0=demo,M1=demo+ipv,M2=demo+ipv+total,
                M3=demo+ipv+total+pattern,M4=demo+total+pattern)


def assignments(frame,seed,k):
    """Outcome-free balance of whole PSUs within strata and across folds."""
    units = frame.groupby(['stratum','psu'],sort=True).size().reset_index(name='women')
    rng = np.random.default_rng(seed)
    global_load = np.zeros(k,dtype=int)
    rows = []
    for stratum, group in units.groupby('stratum',sort=True):
        group = group.iloc[rng.permutation(len(group))].sort_values('women',ascending=False,kind='stable')
        local_load = np.zeros(k,dtype=int)
        for row in group.itertuples():
            candidate = np.flatnonzero(local_load == local_load.min())
            candidate = candidate[global_load[candidate] == global_load[candidate].min()]
            fold = int(rng.choice(candidate))
            rows.append(dict(stratum=stratum,psu=row.psu,fold=fold))
            local_load[fold] += row.women
            global_load[fold] += row.women
    mapping = pd.DataFrame(rows)
    assigned = frame[['row_id','stratum','psu']].merge(mapping,on=['stratum','psu'],validate='many_to_one',how='left')
    assert np.array_equal(assigned.row_id,frame.row_id) and assigned.fold.notna().all()
    assert assigned.groupby('psu').fold.nunique().max() == 1
    assert assigned.fold.nunique() == k
    return assigned


def encode(train,test,numeric,categorical):
    a,b = train[numeric].to_numpy(float),test[numeric].to_numpy(float)
    mean,sd = a.mean(0),a.std(0)
    sd[sd<1e-12] = 1.
    pieces_a,pieces_b = [(a-mean)/sd],[(b-mean)/sd]
    categories,unseen = {},{}
    for name in categorical:
        levels = sorted(train[name].astype(str).unique().tolist())
        categories[name] = levels
        pieces_a.append(np.column_stack([train[name].astype(str).eq(level).to_numpy(float) for level in levels]))
        pieces_b.append(np.column_stack([test[name].astype(str).eq(level).to_numpy(float) for level in levels]))
        unseen[name] = int((~test[name].astype(str).isin(levels)).sum())
    x,xq = np.column_stack(pieces_a),np.column_stack(pieces_b)
    assert np.isfinite(x).all() and np.isfinite(xq).all()
    return x,xq,dict(numeric=numeric,categorical=categorical,mean=mean.tolist(),scale=sd.tolist(),categories=categories,
                     unseen_test_category_counts=unseen,encoded_columns=x.shape[1])


def metrics(y,p):
    error = np.asarray(p)-np.asarray(y)
    variance = np.mean((y-np.mean(y))**2)
    return dict(RMSE=float(np.sqrt(np.mean(error**2))),MAE=float(np.mean(np.abs(error))),
                bias=float(np.mean(error)),R2=float(1-np.mean(error**2)/variance))


def bootstrap(frame,predictions,cfg):
    """Conditional cluster-resampling uncertainty, holding OOF predictions fixed."""
    keys = list(predictions)
    units = frame[['stratum','psu']].drop_duplicates().sort_values(['stratum','psu']).reset_index(drop=True)
    units['unit'] = np.arange(len(units))
    which = frame[['stratum','psu']].merge(units,on=['stratum','psu'],validate='many_to_one').unit.to_numpy()
    counts = np.zeros((cfg['bootstrap_draws'],len(units)),dtype=np.int16)
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    singletons = 0
    for _,group in units.groupby('stratum',sort=True):
        ix = group.unit.to_numpy();m = len(ix)
        singletons += int(m==1)
        counts[:,ix] = rng.multinomial(m,np.ones(m)/m,size=cfg['bootstrap_draws'])
    y = frame.phq9_total.to_numpy(float)
    cn = np.bincount(which,minlength=len(units))
    sse = np.column_stack([np.bincount(which,weights=(predictions[key]-y)**2,minlength=len(units)) for key in keys])
    sae = np.column_stack([np.bincount(which,weights=np.abs(predictions[key]-y),minlength=len(units)) for key in keys])
    denominator = counts@cn
    estimates = dict(RMSE=np.sqrt((counts@sse)/denominator[:,None]),MAE=(counts@sae)/denominator[:,None])
    pairs = [('M3','M1'),('M2','M1'),('M3','M2'),('M1','M0'),('M4','M0')]
    summaries,draws = [],[]
    for algorithm in cfg['algorithms']:
        for model,reference in pairs:
            a,b = keys.index((algorithm,model)),keys.index((algorithm,reference))
            for metric,values in estimates.items():
                delta = values[:,a]-values[:,b]
                observed = metrics(y,predictions[(algorithm,model)])[metric]-metrics(y,predictions[(algorithm,reference)])[metric]
                lo,hi = np.quantile(delta,[.025,.975])
                summaries.append(dict(algorithm=algorithm,model=model,reference=reference,metric=metric,
                                      difference=observed,lower=float(lo),upper=float(hi),draws=cfg['bootstrap_draws'],
                                      uncertainty_scope='descriptive PSU-resampling of fixed OOF predictions in observed cohort; not population-design or full-refit inference'))
                draws.extend(dict(algorithm=algorithm,model=model,reference=reference,metric=metric,draw=i,difference=float(value)) for i,value in enumerate(delta))
    pd.DataFrame(summaries).to_csv(RESULTS/'paired_comparisons.csv',index=False)
    pd.DataFrame(draws).to_csv(RESULTS/'bootstrap_draws.csv',index=False)
    save(RESULTS/'BOOTSTRAP_METADATA.json',dict(draws=cfg['bootstrap_draws'],seed=cfg['bootstrap_seed'],PSUs=len(units),
         strata=units.stratum.nunique(),singleton_strata=singletons,order=keys,resampling='multinomial PSUs within observed strata',fixed_predictions=True))


def run():
    cfg,ready,frame = load()
    for path in [LOCAL,RESULTS]:
        path.mkdir(parents=True,exist_ok=True)
    assert not (RESULTS/'RUN_METADATA.json').exists(), 'Preserve the completed run; use a fresh analysis directory.'
    preparation_qc = STUDY/'INDEPENDENT_PREPARATION_QC.json'
    assert json.loads(preparation_qc.read_text())['status'] == 'passed'
    inputs = [BASE/'PROTOCOL.md',BASE/'CONFIG.json',Path(__file__),STUDY/'MODEL_READY.json',
              STUDY/ready['cohort_path'],preparation_qc]
    hashes = {str(p.relative_to(REV)):sha(p) for p in inputs}
    freeze = BASE/'FIT_INPUT_MANIFEST.json'
    assert not freeze.exists(), 'Existing fit freeze must not be overwritten.'
    save(freeze,dict(saved_utc=datetime.now(timezone.utc).isoformat(),before_fitting=True,files=hashes))
    y = frame.phq9_total.to_numpy(float)
    predictors = blocks(ready)
    for numeric in predictors.values():
        assert cfg['outcome'] not in numeric and not set(ready['depression_items_are_not_predictors'])&set(numeric)
    all_predictions,records,foldcounts,foldmetrics,splitrecords = {},[],[],[],[]
    for seed in cfg['assignment_seeds']:
        assigned = assignments(frame,seed,cfg['outer_folds'])
        splitrecords.append(assigned.assign(seed=seed))
        for algorithm in cfg['algorithms']:
            for model in cfg['feature_models']:
                all_predictions[(algorithm,model,str(seed))] = np.full(len(frame),np.nan)
        all_predictions[('reference','training_mean',str(seed))] = np.full(len(frame),np.nan)
        for fold in range(cfg['outer_folds']):
            test = assigned.fold.eq(fold).to_numpy();train = ~test
            assert not set(frame.loc[train,'psu'])&set(frame.loc[test,'psu'])
            foldcounts.append(dict(seed=seed,fold=fold,training_women=int(train.sum()),test_women=int(test.sum()),
                                   training_PSUs=frame.loc[train,'psu'].nunique(),test_PSUs=frame.loc[test,'psu'].nunique()))
            all_predictions[('reference','training_mean',str(seed))][test] = y[train].mean()
            for model in cfg['feature_models']:
                x,xq,transform = encode(frame.loc[train],frame.loc[test],predictors[model],ready['categorical_covariates'])
                for algorithm in cfg['algorithms']:
                    if algorithm == 'ridge':
                        estimator = Ridge(alpha=cfg['ridge_alpha'],solver='cholesky').fit(x,y[train])
                    else:
                        estimator = HistGradientBoostingRegressor(**cfg['boosting'],random_state=42).fit(x,y[train])
                    raw = estimator.predict(xq)
                    all_predictions[(algorithm,model,str(seed))][test] = raw
                    key = f'{algorithm}_{model}_seed{seed}_fold{fold}'
                    joblib.dump(estimator,LOCAL/(key+'.joblib'),compress=3)
                    record = dict(algorithm=algorithm,model=model,seed=seed,fold=fold,transform=transform,
                                  training_women=int(train.sum()),test_women=int(test.sum()),model_file=key+'.joblib')
                    if algorithm == 'ridge':
                        record.update(coefficient=estimator.coef_.tolist(),intercept=float(estimator.intercept_))
                    else:
                        record.update(iterations=estimator.n_iter_,tree_count=sum(len(v) for v in estimator._predictors))
                    records.append(record)
                    for variant,p in [('bounded',np.clip(raw,*cfg['bounds'])),('raw',raw)]:
                        foldmetrics.append(dict(algorithm=algorithm,model=model,seed=seed,fold=fold,variant=variant,n=int(test.sum()),**metrics(y[test],p)))
            print('Finished assignment',seed,'fold',fold+1,'/',cfg['outer_folds'],flush=True)
    pd.concat(splitrecords).to_csv(LOCAL/'fold_assignments.csv.gz',index=False,compression='gzip')
    metrics_rows=[]
    for algorithm,model,_ in list(all_predictions):
        if (algorithm,model,'ensemble') in all_predictions:
            continue
        all_predictions[(algorithm,model,'ensemble')] = np.mean([all_predictions[(algorithm,model,str(seed))] for seed in cfg['assignment_seeds']],axis=0)
    output = frame[['row_id','psu','stratum','phq9_total']].copy()
    for (algorithm,model,seed),raw in all_predictions.items():
        assert np.isfinite(raw).all()
        output['__'.join([algorithm,model,seed,'raw'])] = raw
        for variant,p in [('bounded',np.clip(raw,*cfg['bounds'])),('raw',raw)]:
            metrics_rows.append(dict(algorithm=algorithm,model=model,seed=seed,variant=variant,n=len(y),
                                     clipped=int(np.count_nonzero(raw!=np.clip(raw,*cfg['bounds']))),**metrics(y,p)))
    output.to_csv(LOCAL/'oof_predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(metrics_rows).to_csv(RESULTS/'metrics.csv',index=False)
    pd.DataFrame(foldcounts).to_csv(RESULTS/'fold_counts.csv',index=False)
    pd.DataFrame(foldmetrics).to_csv(RESULTS/'fold_metrics.csv',index=False)
    save(RESULTS/'model_records.json',records)
    ensemble = {(a,m):np.clip(raw,*cfg['bounds']) for (a,m,s),raw in all_predictions.items() if s=='ensemble' and a!='reference'}
    bootstrap(frame,ensemble,cfg)
    assert all(sha(REV/name)==value for name,value in hashes.items()), 'Frozen fitting inputs changed.'
    summary = dict(status='fitted_pending_independent_qc',women=len(frame),PSUs=frame.psu.nunique(),strata=frame.stratum.nunique(),
                   fitted_models=len(records),ridge_fits=sum(v['algorithm']=='ridge' for v in records),
                   boosting_fits=sum(v['algorithm']=='hist_gradient_boosting' for v in records),
                   estimator='equal-person observed-linked-cohort prediction, not nationally weighted inference',
                   frozen_sources=hashes,finished_utc=datetime.now(timezone.utc).isoformat(),
                   environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                                    scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__))
    save(RESULTS/'RUN_METADATA.json',summary)
    print(pd.DataFrame(metrics_rows).loc[lambda d:d.seed.eq('ensemble')&d.variant.eq('bounded')].to_string(index=False),flush=True)


if __name__ == '__main__':
    run()
