"""Fixed exact-conjugate hierarchical country/scenario comparison; new-folder writes only."""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import cho_factor, cho_solve, helmert
from scipy.special import gammaln, logsumexp
from scipy.stats import t

BASE = Path(__file__).resolve().parent
REV = BASE.parents[1]
T = BASE.parent
INPUT, OUT = BASE / 'inputs', BASE / 'results'
CONFIG = json.loads((BASE / 'CONFIG.json').read_text())
ITEMS, FEATURES = CONFIG['items'], CONFIG['features']
Q = helmert(5, full=False).T
A = np.column_stack([np.ones(5), Q])
LOSS_NAMES = ['MAE_pp','MSE_pp2','level_MAE_pp','level_MSE_pp2','profile_MAE_pp','profile_MSE_pp2','ordering_accuracy']
SOURCES = {
    'rounds.csv': T / 'context_blocks_2026_10_22/inputs/rounds.csv',
    'external_targets.csv': T / 'context_blocks_2026_10_22/inputs/external_targets.csv',
    'chronological_reference_membership.csv': T / 'shared_context_shift/results/training_membership.csv',
    'chronological_reference_predictions.csv': T / 'shared_context_shift/results/predictions.csv',
    'external_reference_membership.csv': T / 'mics_external_evaluation/results/training_membership.csv',
    'external_reference_predictions.csv': T / 'mics_external_evaluation/results/predictions.csv',
    'national_source_cells.csv': T / 'context_blocks_2026_10_22/inputs/national_source_cells.csv',
    'structural_source_ledger.csv': T / 'context_blocks_2026_10_22/inputs/structural_source_ledger.csv',
    'external_source_cells.csv': T / 'context_blocks_2026_10_22/inputs/external_source_cells.csv',
    'external_structural_source_ledger.csv': T / 'context_blocks_2026_10_22/inputs/external_structural_source_ledger.csv',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def read(path):
    return json.loads(Path(path).read_text())

def source_check():
    checked = {}
    original = read(T / 'joint_country_time/prepared/INPUT_MANIFEST.json')
    for category in ('source_sha256','raw_api_sha256'):
        for name, expected in original[category].items():
            assert sha(REV / name) == expected, name
            checked[name] = expected
    for name, expected in original['prepared_sha256'].items():
        path = T / 'joint_country_time/prepared' / name
        assert sha(path) == expected, path
        checked[str(path.relative_to(REV))] = expected
    shared = read(T / 'shared_context_shift/inputs/SOURCE_MANIFEST.json')
    for name, expected in shared['input_sha256'].items():
        assert sha(T / 'shared_context_shift/inputs' / name) == expected, name
    external = read(T / 'mics_external_evaluation/results/RUN_METADATA.json')
    for name, expected in external['source_sha256'].items():
        assert sha(REV / name) == expected, name
        checked[name] = expected
    for path in SOURCES.values():
        checked[str(path.relative_to(REV))] = sha(path)
    for directory in ('shared_context_shift', 'mics_external_evaluation'):
        assert read(T / directory / 'results/INDEPENDENT_QC.json')['status'] == 'passed'
    assert sha(SOURCES['rounds.csv']) == sha(T / 'joint_country_time/prepared/rounds.csv')
    return checked

def prior_check():
    assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-14)
    assert np.allclose(Q.T @ np.ones(5), 0, atol=1e-14)
    demo = np.array([[0,1,9,12,18],[90,15,27,81,70]], float)
    transformed = np.column_stack([demo.mean(axis=1), demo @ Q])
    assert np.allclose(transformed @ A.T, demo, atol=1e-13)
    rng = np.random.default_rng(202610229)
    records = []
    n = 300000
    lambdas = np.array(CONFIG['lambda_grid'])
    probs = np.array(CONFIG['lambda_prior_probabilities'])
    for prior_name, prior in CONFIG['priors'].items():
        sig2 = 1.0 / rng.gamma(prior['a0'], 1.0 / prior['b0'], (n, 5))
        lam = rng.choice(lambdas, (n,5), p=probs)
        expected_coordinate_variance = prior['b0'] / (prior['a0'] - 1) * (1 + np.dot(lambdas, probs) + 25)
        means = np.array(CONFIG['prior_intercept_means'])
        coords = means + rng.normal(size=(n,5)) * np.sqrt(sig2 * (1 + lam + 25))
        scenarios = coords @ A.T
        # IG shape 2 has an infinite variance for sigma²; empirical variance is a diagnostic,
        # not a strict Gaussian-SE check. Mean checks use exact finite predictive variance.
        se_mean = np.sqrt(expected_coordinate_variance / n)
        max_mean_error = float(np.max(np.abs(coords.mean(axis=0) - means)))
        assert max_mean_error <= 8 * se_mean
        relative_variance_error = np.abs(coords.var(axis=0) / expected_coordinate_variance - 1)
        assert np.isfinite(relative_variance_error).all()
        assert relative_variance_error.max() <= 0.20, 'Fixed prior-predictive second-moment tolerance failed'
        records.append({'prior':prior_name,'draws':n,'analytic_coordinate_variance_at_center':float(expected_coordinate_variance),
                        'coordinate_mean_max_abs_error':max_mean_error,'mean_eight_SE_tolerance':float(8*se_mean),
                        'coordinate_variance_relative_errors':relative_variance_error.tolist(),
                        'variance_check_scope':'fixed 20% empirical-versus-analytic second-moment tolerance; IG shape 2 has no fourth moment, so no Gaussian variance-SE claim','variance_relative_tolerance':0.20,
                        'scenario_out_of_range_fraction':float(np.mean((scenarios<0)|(scenarios>100)))})
    result = {'status':'passed','basis_Q':Q.tolist(),'reconstruction_max_error':float(np.max(np.abs(transformed @ A.T-demo))),
              'basis_scope':'fixed Helmert basis; independent block hierarchy is basis dependent','checks':records}
    dump(BASE / 'PRIOR_PREDICTIVE_CHECK.json', result)
    return result

def freeze():
    assert not (BASE / 'FREEZE.json').exists()
    assert not (OUT / 'predictions.csv').exists()
    checked = source_check()
    INPUT.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    for name, path in SOURCES.items():
        shutil.copyfile(path, INPUT / name)
    prior_check()
    protected = {}
    for folder in (REV / '07_manuscript/latex', REV / 'FINAL_Scientific_Reports_2026_10_22'):
        for path in folder.rglob('*'):
            if path.is_file(): protected[str(path.relative_to(REV))] = sha(path)
    for path in (REV/'FINAL_Scientific_Reports_2026_10_22.zip',):
        if path.exists(): protected[str(path.relative_to(REV))] = sha(path)
    for folder in (T/'joint_country_time',T/'shared_context_shift',T/'mics_external_evaluation',T/'context_blocks_2026_10_22'):
        for path in folder.rglob('*'):
            if path.is_file(): protected[str(path.relative_to(REV))] = sha(path)
    files = [BASE/name for name in ('PROTOCOL.md','CONFIG.json','run_analysis.py','report_results.py','PRIOR_PREDICTIVE_CHECK.json')]
    files += list(INPUT.glob('*'))
    dump(BASE/'FREEZE.json',{'status':'frozen_before_new_fits_retrospective_exploratory','date_utc':datetime.now(timezone.utc).isoformat(),
                            'files':{str(p.relative_to(BASE)):sha(p) for p in files},'source_sha256':checked,'protected_sha256':protected})
    dump(BASE/'STATUS.json',{'status':'frozen_not_fitted_waiting_Ising_completion_and_root_start'})
    print('Protocol, priors, algorithms, aggregate sources and prior-predictive check frozen; NO fits run', flush=True)

def verify_freeze():
    f=read(BASE/'FREEZE.json')
    for key,root in (('files',BASE),('source_sha256',REV),('protected_sha256',REV)):
        for name,h in f[key].items(): assert sha(root/name)==h, name
    return f

def preprocess(train, target):
    weight=1.0/train.groupby('country').survey_id.transform('size').to_numpy(float)
    x=train[FEATURES].to_numpy(float); xt=np.array([getattr(target,f) for f in FEATURES],float)
    assert np.isfinite(x).any(axis=0).all()
    impute=np.array([np.average(x[np.isfinite(x[:,j]),j],weights=weight[np.isfinite(x[:,j])]) for j in range(x.shape[1])])
    filled=np.where(np.isfinite(x),x,impute)
    mean=np.average(filled,weights=weight,axis=0)
    scale=np.sqrt(np.average((filled-mean)**2,weights=weight,axis=0));scale[scale==0]=1
    z=(filled-mean)/scale;zt=(np.where(np.isfinite(xt),xt,impute)-mean)/scale
    return z,zt,{'features':FEATURES,'impute':impute.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),
                 'donor_missing':(~np.isfinite(x)).sum(axis=0).tolist(),'target_missing':(~np.isfinite(xt)).tolist(),
                 'preprocessing_weights':'one total per donor country; likelihood remains proper unweighted round-level Gaussian'}

def coordinate_cache(train, target, z,zt):
    y=train[ITEMS].to_numpy(float)
    yy=np.column_stack([y.mean(axis=1), y@Q])
    country=pd.Categorical(train.country)
    C=np.eye(len(country.categories))[country.codes]
    cc=C@C.T
    cache={}
    for design in ('calendar','context'):
        cols=1 if design=='calendar' else len(FEATURES)
        X=np.column_stack([np.ones(len(train)),z[:,:cols]])
        xs=np.r_[1.0,zt[:cols]]
        V=np.diag([25.0]+[1.0]*cols)
        for j,lam in enumerate(CONFIG['lambda_grid']):
            K=np.eye(len(train))+lam*cc+X@V@X.T
            factor=cho_factor(K,lower=True,check_finite=True)
            logdet=2*np.log(np.diag(factor[0])).sum()
            means=np.array(CONFIG['prior_intercept_means'])
            centered=yy-np.ones((len(train),1))*means
            solve=cho_solve(factor,centered)
            cross=X@V@xs
            solve_cross=cho_solve(factor,cross)
            prediction_means=means+cross@solve
            variance_factor=1+lam+xs@V@xs-cross@solve_cross
            assert variance_factor>1+lam-1e-9
            vbeta=V-V@X.T@cho_solve(factor,X@V)
            assert np.linalg.eigvalsh(vbeta).min()>-1e-10
            beta_means=V@X.T@solve
            beta_means[0]+=means
            cache[(design,j)]={'logdet':float(logdet),'quadratic':np.sum(centered*solve,axis=0),
                                  'prediction_mean':prediction_means,'variance_factor':float(variance_factor),
                                  'beta_means':beta_means,'beta_variance_factor':vbeta,
                                  'features':['intercept']+FEATURES[:cols],
                                  'country_posterior_mean':lam*C.T@solve,'donor_country_order':country.categories.tolist()}
    return cache

def posterior_blocks(cache,n,model,prior):
    means,scales,weights,records=[],[],[],[]
    for q in range(5):
        design='context' if model=='B2' or (model=='B1' and q==0) else 'calendar'
        an=prior['a0']+n/2
        bns=np.array([prior['b0']+.5*cache[(design,j)]['quadratic'][q] for j in range(len(CONFIG['lambda_grid']))])
        loge=np.array([gammaln(an)-gammaln(prior['a0'])+prior['a0']*np.log(prior['b0'])-an*np.log(bns[j])
                       -.5*cache[(design,j)]['logdet']-n/2*np.log(2*np.pi) for j in range(len(CONFIG['lambda_grid']))])
        logw=loge+np.log(CONFIG['lambda_prior_probabilities']);ww=np.exp(logw-logsumexp(logw))
        mu=np.array([cache[(design,j)]['prediction_mean'][q] for j in range(len(CONFIG['lambda_grid']))])
        ss=np.sqrt(np.array([cache[(design,j)]['variance_factor'] for j in range(len(CONFIG['lambda_grid']))])*bns/an)
        assert np.isclose(ww.sum(),1,atol=1e-12)
        means.append(mu);scales.append(ss);weights.append(ww)
        for j,lam in enumerate(CONFIG['lambda_grid']):
            block=cache[(design,j)]
            records.append({'coordinate':q,'design':design,'lambda':float(lam),'posterior_weight':float(ww[j]),
                            'log_marginal_likelihood':float(loge[j]),'a_n':float(an),'b_n':float(bns[j]),'degrees_freedom':float(2*an),
                            'predictive_location':float(mu[j]),'predictive_scale':float(ss[j]),
                            'predictive_variance_factor':block['variance_factor'],
                            'coefficient_features':block['features'],'coefficient_posterior_means':block['beta_means'][:,q].tolist(),
                            'coefficient_variance_factor':block['beta_variance_factor'].tolist(),
                            'donor_country_order':block['donor_country_order'],'donor_country_posterior_means':block['country_posterior_mean'][:,q].tolist()})
    return np.array(means),np.array(scales),np.array(weights),2*an,records

def draw_predictive(mu,sc,ww,df,seed,n):
    rng=np.random.default_rng(seed)
    values=np.empty((n,5))
    for q in range(5):
        qrng=np.random.default_rng(seed + q*1000003)
        mixture=qrng.choice(mu.shape[1],size=n,p=ww[q])
        trng=np.random.default_rng(seed + q*1000003 + 5000011)
        values[:,q]=mu[q,mixture]+sc[q,mixture]*trng.standard_t(df,n)
    return values

def intervals(mu,sc,ww,df,seed):
    n=CONFIG['posterior_predictive_draws']
    attempts=[]
    while True:
        coords=draw_predictive(mu,sc,ww,df,seed,n)
        scenarios=coords@A.T
        both=np.column_stack([scenarios,coords])
        first=np.quantile(both[:n//2],[.025,.975],axis=0)
        second=np.quantile(both[n//2:],[.025,.975],axis=0)
        discrepancy=float(np.max(np.abs(first-second)))
        attempts.append({'draws':n,'max_half_stream_endpoint_difference_pp':discrepancy})
        if discrepancy<=CONFIG['predictive_interval_endpoint_tolerance_pp'] or n==CONFIG['maximum_posterior_predictive_draws']:break
        n=CONFIG['maximum_posterior_predictive_draws']
    return coords,scenarios,{'seed':int(seed),'draws':n,'attempts':attempts,'endpoint_precision_check_passed':bool(discrepancy<=CONFIG['predictive_interval_endpoint_tolerance_pp'])}

def losses(y,p):
    error=p-y;le=float(error.mean());pr=error-le
    mse=float(np.square(error).mean());profile=float(np.square(pr).mean())
    assert np.isclose(mse,le**2+profile,atol=1e-9)
    wins=[]
    for j in range(5):
        for k in range(j+1,5):
            if y[j]==y[k]:continue
            wins.append(.5 if abs(p[j]-p[k]) <= 1e-10 else float(np.sign(p[j]-p[k])==np.sign(y[j]-y[k])))
    return {'MAE_pp':float(np.abs(error).mean()),'MSE_pp2':mse,'level_MAE_pp':abs(le),'level_MSE_pp2':le**2,
            'profile_MAE_pp':float(np.abs(pr).mean()),'profile_MSE_pp2':profile,'ordering_accuracy':float(np.mean(wins)),
            'ordering_pairs':len(wins),'ordering_observed_ties':10-len(wins),'bias_pp':le}

def fit():
    verify_freeze()
    permission=read(BASE/'START_AUTHORISATION.json')
    assert permission['ising_stage_completed'] is True and permission['start_bayesian_fits_authorized'] is True
    assert not (OUT/'FIT_COMPLETION.json').exists(),'Never overwrite completed fits'
    rounds=pd.read_csv(INPUT/'rounds.csv')
    external=pd.read_csv(INPUT/'external_targets.csv')
    assert len(rounds)==169 and rounds.country.nunique()==62
    for frame in (rounds,external):
        for col in ('fieldwork_start','fieldwork_end'):frame[col]=pd.to_datetime(frame[col])
    rounds['conservative_end']=rounds.fieldwork_end.dt.to_period('M').dt.end_time.dt.normalize()
    targets=rounds.loc[rounds.is_target.astype(str).str.lower().eq('true')].sort_values('country')
    frames={'chronological42':targets,'external19':external}
    ref_members={name:pd.read_csv(INPUT/file) for name,file in [('chronological42','chronological_reference_membership.csv'),('external19','external_reference_membership.csv')]}
    references={name:pd.read_csv(INPUT/file) for name,file in [('chronological42','chronological_reference_predictions.csv'),('external19','external_reference_predictions.csv')]}
    predictions=[];metric_rows=[];members=[];excluded=[];preprocessing=[];block_rows=[];uncertainty=[];pair_rows=[];precision=[]
    references_used=[];invariants=[];counter=0
    for case in CONFIG['donor_cases']:
        for evaluation in CONFIG['evaluations']:
            target_frame=frames[evaluation]
            assert len(target_frame)==CONFIG['expected_target_counts'][evaluation]
            old_members=ref_members[evaluation].loc[ref_members[evaluation].case.eq(CONFIG['reference_case'])]
            old_prediction=references[evaluation].loc[references[evaluation].case.eq(CONFIG['reference_case'])]
            for target in target_frame.itertuples():
                key={'case':case,'evaluation':evaluation,'target_country':target.country,'target_iso3':target.iso3,'target_survey':target.survey_id}
                same=rounds.country.eq(target.country) if evaluation=='chronological42' else rounds.iso3.eq(target.iso3)
                ends=rounds.fieldwork_end if evaluation=='chronological42' else rounds.conservative_end
                late=ends.ge(target.fieldwork_start)
                omitted=rounds.survey_id.isin(CONFIG['omitted_surveys_all_cases'])
                india=rounds.iso3.eq('IND') if case=='omit_india' else pd.Series(False,index=rounds.index)
                eligible=~same&~late&~omitted&~india
                train=rounds.loc[eligible].copy()
                previous=old_members.loc[old_members.target_survey.eq(target.survey_id)]
                expected=previous.training_survey.tolist()
                if case=='omit_india':expected=[s for s in expected if s not in set(rounds.loc[rounds.iso3.eq('IND')].survey_id)]
                assert expected==train.survey_id.tolist(),(key,expected,train.survey_id.tolist())
                assert not train.iso3.eq(target.iso3).any()
                assert (ends[eligible]<target.fieldwork_start).all()
                for row in train.itertuples():members.append({**key,'training_country':row.country,'training_iso3':row.iso3,'training_survey':row.survey_id,'donor_end':str((row.fieldwork_end if evaluation=='chronological42' else row.conservative_end).date()),'target_start':str(target.fieldwork_start.date())})
                for i in np.flatnonzero(~eligible.to_numpy()):
                    why=[]
                    if same.iloc[i]:why.append('all_target_country_history')
                    if late.iloc[i]:why.append('not_strictly_earlier')
                    if omitted.iloc[i]:why.append('questionnaire_filter_Nepal2011')
                    if india.iloc[i]:why.append('fixed_India_donor_omission')
                    excluded.append({**key,'survey_id':rounds.iloc[i].survey_id,'reasons':'|'.join(why)})
                z,zt,prep=preprocess(train,target)
                preprocessing.append({**key,'training_rounds':len(train),'training_countries':train.country.nunique(),**prep})
                cache=coordinate_cache(train,target,z,zt)
                obs=np.array([getattr(target,item) for item in ITEMS],float)
                for prior_index,(prior_name,prior) in enumerate(CONFIG['priors'].items()):
                    invariant_vectors={}
                    for model_index,model in enumerate(CONFIG['models']):
                        fullkey={**key,'prior':prior_name,'model':model}
                        mu,sc,ww,df,records=posterior_blocks(cache,len(train),model,prior)
                        for record in records:block_rows.append({**fullkey,**record})
                        coord_mean=(mu*ww).sum(axis=1)
                        raw=coord_mean@A.T;bounded=np.clip(raw,0,100)
                        seed=CONFIG['prediction_seed']+counter*100+prior_index*3
                        invariant_vectors[model]=(raw,mu,sc,ww)
                        coords,draws,prec=intervals(mu,sc,ww,df,seed)
                        precision.append({**fullkey,**prec})
                        for j,item in enumerate(ITEMS):
                            predictions.append({**fullkey,'item':item,'observed_pp':float(obs[j]),'raw_prediction_pp':float(raw[j]),'prediction_pp':float(bounded[j]),'clipped':bool(raw[j]!=bounded[j]),'training_rounds':len(train),'training_countries':train.country.nunique()})
                        obscoords=np.r_[obs.mean(),obs@Q]
                        for kind,p in [('raw',raw),('bounded',bounded)]:metric_rows.append({**fullkey,'prediction_kind':kind,**losses(obs,p)})
                        for space,values,observed in [('scenario',draws,obs),('coordinate',coords,obscoords)]:
                            for j in range(5):
                                for coverage in [.5,.8,.95]:
                                    lo,hi=np.quantile(values[:,j],[(1-coverage)/2,(1+coverage)/2])
                                    uncertainty.append({**fullkey,'space':space,'index':j,'name':ITEMS[j] if space=='scenario' else 'national_mean' if j==0 else f'Helmert_{j}',
                                                        'nominal_coverage':coverage,'observed':float(observed[j]),'lower':float(lo),'upper':float(hi),'covered':bool(lo<=observed[j]<=hi),'width_pp':float(hi-lo),'draws':len(draws),
                                                        'out_of_range_probability':float(np.mean((values[:,j]<0)|(values[:,j]>100))) if space=='scenario' else 0.0})
                        for j in range(5):
                            for k in range(j+1,5):
                                prob=float(np.mean(draws[:,j]>draws[:,k]));diff=obs[j]-obs[k]
                                pair_rows.append({**fullkey,'item_a':ITEMS[j],'item_b':ITEMS[k],'observed_a_minus_b':float(diff),'observed_tie':bool(diff==0),'posterior_probability_a_gt_b':prob,'probability_MCSE':float(np.sqrt(prob*(1-prob)/len(draws))),'posterior_probability_positive_correct':prob if diff>0 else 1-prob if diff<0 else .5})
                    v0,v1,v2=[invariant_vectors[m] for m in ('B0','B1','B2')]
                    discrepancy=float(np.max(np.abs((v0[0]-v0[0].mean())-(v1[0]-v1[0].mean()))))
                    assert discrepancy<1e-10
                    assert abs(v1[0].mean()-v2[0].mean())<1e-10
                    for position in (1,2,3):
                        assert np.array_equal(v0[position][1:],v1[position][1:])
                        assert np.array_equal(v1[position][0],v2[position][0])
                    assert losses(obs,v0[0])['ordering_accuracy']==losses(obs,v1[0])['ordering_accuracy']
                    invariants.append({**key,'prior':prior_name,'B0_B1_raw_profile_max_difference_pp':discrepancy,
                                       'B1_B2_raw_mean_difference_pp':float(abs(v1[0].mean()-v2[0].mean())),
                                       'B0_B1_contrast_posterior_blocks_identical':True,'B1_B2_mean_posterior_blocks_identical':True})
                if case=='questionnaire_compatible':
                    for refmodel in ('R0','S','R1'):
                        rr=old_prediction.loc[old_prediction.target_survey.eq(target.survey_id)&old_prediction.model.eq(refmodel)].set_index('item').loc[ITEMS]
                        assert np.allclose(rr.observed_pp,obs,atol=1e-14)
                        if refmodel=='R0':assert rr.training_rounds.iloc[0]==len(train)
                        for kind,column in [('raw','raw_prediction_pp'),('bounded','prediction_pp')]:
                            references_used.append({**key,'prior':'saved_reference','model':refmodel,'prediction_kind':kind,**losses(obs,rr[column].to_numpy(float))})
                counter+=1
                print(f'{case} {evaluation} {target.survey_id}: {len(train)} donor rounds, exact 9 model/prior fits',flush=True)
    assert counter==122
    for name,rows in [('predictions',predictions),('country_metrics',metric_rows),('training_membership',members),('exclusions',excluded),('predictive_intervals',uncertainty),('ordering_probabilities',pair_rows),('reference_country_metrics',references_used)]:pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    dump(OUT/'HIERARCHY_INVARIANTS.json',invariants);dump(OUT/'preprocessing.json',preprocessing);dump(OUT/'posterior_blocks.json',block_rows);dump(OUT/'predictive_precision.json',precision)
    verify_freeze()
    dump(OUT/'FIT_COMPLETION.json',{'status':'producer_exact_fits_complete_independent_QC_pending','target_case_folds':counter,'bayesian_target_model_prior_fits':counter*9,'coordinate_blocks':counter*9*5,'conditional_lambda_blocks':len(block_rows),'posterior_prediction_cells':len(predictions),'interval_precision_failures':sum(not p['endpoint_precision_check_passed'] for p in precision),'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},'platform':platform.platform(),'inference':'exact conjugacy/finite prior integration; no MCMC or ADVI'})
    summarize()

def summarize():
    metrics=pd.read_csv(OUT/'country_metrics.csv')
    refs=pd.read_csv(OUT/'reference_country_metrics.csv')
    summary=metrics.groupby(['case','evaluation','prior','model','prediction_kind'])[LOSS_NAMES].mean().reset_index()
    counts=metrics.groupby(['case','evaluation','prior','model','prediction_kind']).size().rename('countries').reset_index()
    summary=summary.merge(counts)
    summary.to_csv(OUT/'model_summary.csv',index=False)
    comps=[];differences=[]
    rng=np.random.default_rng(CONFIG['bootstrap_seed'])
    for (case,evaluation),domain in metrics.groupby(['case','evaluation'],sort=True):
        targets=sorted(domain.target_survey.unique());idx=rng.integers(0,len(targets),(CONFIG['bootstrap_draws'],len(targets)))
        for prior in CONFIG['priors']:
            for kind in ('raw','bounded'):
                groups={m:domain.loc[domain.prior.eq(prior)&domain.model.eq(m)&domain.prediction_kind.eq(kind)].set_index('target_survey').loc[targets] for m in CONFIG['models']}
                pairs=[('B1','B0'),('B2','B1'),('B2','B0')]
                if case=='questionnaire_compatible' and prior=='primary':
                    for m in ('R0','S','R1'):groups[m]=refs.loc[refs.evaluation.eq(evaluation)&refs.model.eq(m)&refs.prediction_kind.eq(kind)].set_index('target_survey').loc[targets]
                    pairs += [('B0','R0'),('B1','S'),('B2','R1')]
                for candidate,reference in pairs:
                    for loss in LOSS_NAMES:
                        d=groups[candidate][loss].to_numpy()-groups[reference][loss].to_numpy()
                        draws=d[idx].mean(axis=1);lo,hi=np.quantile(draws,[.025,.975])
                        direction=d>0 if loss=='ordering_accuracy' else d<0
                        comps.append({'case':case,'evaluation':evaluation,'prior':prior,'prediction_kind':kind,'candidate':candidate,'reference':reference,'metric':loss,'countries':len(targets),'mean_difference':float(d.mean()),'lower':float(lo),'upper':float(hi),'countries_improved':int(direction.sum()),'interpretation':'fixed-prediction country bootstrap; not model posterior or survey design interval'})
                        for target,value in zip(targets,d):differences.append({'case':case,'evaluation':evaluation,'prior':prior,'prediction_kind':kind,'candidate':candidate,'reference':reference,'metric':loss,'target_survey':target,'difference':float(value)})
    pd.DataFrame(comps).to_csv(OUT/'paired_comparisons.csv',index=False)
    pd.DataFrame(differences).to_csv(OUT/'paired_country_differences.csv',index=False)
    ints=pd.read_csv(OUT/'predictive_intervals.csv')
    ints.groupby(['case','evaluation','prior','model','space','nominal_coverage']).agg(cells=('covered','size'),coverage=('covered','mean'),mean_width_pp=('width_pp','mean'),mean_out_of_range_probability=('out_of_range_probability','mean')).reset_index().to_csv(OUT/'predictive_coverage.csv',index=False)
    dump(OUT/'SUMMARY_COMPLETION.json',{'status':'producer_summaries_complete_independent_QC_pending','summary_rows':len(summary),'paired_comparisons':len(comps),'paired_country_rows':len(differences),'bootstrap_resamples_per_domain':CONFIG['bootstrap_draws']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--fit',action='store_true');p.add_argument('--summarize',action='store_true');args=p.parse_args()
    if args.freeze:freeze()
    if args.fit:fit()
    if args.summarize:summarize()
