"""Prepare, freeze and execute the fixed hybrid GNN/hierarchical-readout study."""
from __future__ import annotations
import argparse,gzip,json,platform,shutil,time
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import scipy,torch
import common as c
ROOT=c.ROOT;REV=ROOT.parents[1];PREV=ROOT.parent/'bayesian_country_scenario_2026_10_22';OUT=ROOT/'results'

def load(path):return json.loads(Path(path).read_text())
def dump_gz(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt',compresslevel=9) as stream:json.dump(value,stream,allow_nan=False,separators=(',',':'))
def load_gz(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)

def synthetic_check():
    rng=np.random.default_rng(c.CONFIG['runtime_synthetic_seed']);n=162
    x=rng.normal(size=(n,5));xq=rng.normal(size=5);countries=np.array([f'c{i%61:02}' for i in range(n)])
    a,q,g=c.graph(x,xq,countries,42);y=rng.normal(size=(n,4))
    start=time.perf_counter();p,diag=c.train_encoder(x,y,a['real'],countries,42)
    elapsed=time.perf_counter()-start
    h,pred,hq,pq=c.forward(p,x,a['real'],xq,q['real'])
    wt=c.country_weights(countries);ys,_,_=c.standardize(y,wt)
    pt={k:torch.tensor(v,dtype=torch.float64,requires_grad=True) for k,v in p.items()}
    tensors=[torch.tensor(v,dtype=torch.float64) for v in [x,a['real']@x,ys,wt]]
    obj,err,pen=c.torch_objective(pt,*tensors);obj.backward();epsilon=1e-5;checks=[]
    for key,index in [('W_self',(0,0)),('W_neighbour',(2,1)),('W_hidden',(1,2)),('W_linear',(0,1)),('b_hidden',(2,)),('b_output',(1,))]:
        states=[]
        for sign in (1,-1):
            pp={k:np.asarray(v,float).copy() for k,v in p.items()};pp[key][index]+=sign*epsilon
            _,ppred=c.forward(pp,x,a['real']);ee=np.sum(wt*np.mean((ppred-ys)**2,axis=1));rr=c.CONFIG['l2_mean_all_parameters']*sum((v**2).sum() for v in pp.values())/sum(v.size for v in pp.values());states.append(ee+rr)
        numeric=(states[0]-states[1])/(2*epsilon);analytic=float(pt[key].grad[index]);checks.append(dict(parameter=key,index=list(index),numeric=numeric,analytic=analytic,absolute_error=abs(numeric-analytic)))
    assert max(v['absolute_error'] for v in checks)<1e-7
    ap,qp,gp=c.graph(x,xq+4,countries,42)
    assert all(np.array_equal(a[name],ap[name]) for name in a)
    hdirect=c.forward(p,x,ap['real'],xq+4,qp['real'])[0]
    assert np.array_equal(h,hdirect)
    yy=np.column_stack([y[:,0],y]);yy=np.column_stack([x[:,0]+rng.normal(size=n),y])
    fit=c.readout(x,xq,yy,countries)
    assert all(np.isclose(sum(w),1) for w in fit['weights'])
    c.save(ROOT/'SYNTHETIC_CHECK.json',dict(status='passed',no_real_outcomes_used=True,nodes=n,steps=c.CONFIG['steps'],single_fit_seconds=elapsed,upper_732_fit_serial_seconds=732*elapsed,rough_two_worker_minutes=732*elapsed/120,gradient_checks=checks,query_invariance_max_abs_error=float(np.max(abs(h-hdirect))),parameters=diag['parameters'],torch=torch.__version__,scope='synthetic gradients/one fixed runtime; independent review remains separate'))

def prepare():
    assert not (ROOT/'FREEZE.json').exists()
    (ROOT/'inputs').mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    sources={}
    names=['rounds.csv','external_targets.csv','chronological_reference_membership.csv','external_reference_membership.csv']
    for name in names:
        src=PREV/'inputs'/name;dst=ROOT/'inputs'/name;shutil.copyfile(src,dst);sources[str(src.relative_to(REV))]=c.sha(src)
    for name in ['CONFIG.json','PROTOCOL.md','FREEZE.json','results/predictions.csv','results/country_metrics.csv']:
        src=PREV/name
        if src.exists():sources[str(src.relative_to(REV))]=c.sha(src)
    rounds=pd.read_csv(ROOT/'inputs/rounds.csv');external=pd.read_csv(ROOT/'inputs/external_targets.csv');assert len(rounds)==169 and rounds.country.nunique()==62
    for d in (rounds,external):
        for col in ['fieldwork_start','fieldwork_end']:d[col]=pd.to_datetime(d[col])
    rounds['conservative_end']=rounds.fieldwork_end.dt.to_period('M').dt.end_time.dt.normalize()
    targets={'chronological42':rounds.loc[rounds.is_target.astype(str).str.lower().eq('true')].sort_values('country'),'external19':external}
    tasks=[];membership=[];preps=[];edge_rows=[]
    for domain,frame in targets.items():
        old=pd.read_csv(ROOT/'inputs'/('chronological_reference_membership.csv' if domain=='chronological42' else 'external_reference_membership.csv'));old=old.loc[old.case.eq('questionnaire_compatible')]
        assert len(frame)==(42 if domain=='chronological42' else 19)
        for target in frame.itertuples():
            ends=rounds.fieldwork_end if domain=='chronological42' else rounds.conservative_end
            eligible=~rounds.iso3.eq(target.iso3)&ends.lt(target.fieldwork_start)&~rounds.survey_id.eq('NP2011DHS')
            train=rounds.loc[eligible].copy();expected=old.loc[old.target_survey.eq(target.survey_id)].training_survey.tolist();assert train.survey_id.tolist()==expected
            uid=f'target_{len(tasks):03}';countries=train.country.tolist();x=train[c.FEATURES].to_numpy(float);xq=np.array([getattr(target,f) for f in c.FEATURES],float);z,zq,prep=c.preprocess(x,xq,countries)
            graphseed=c.CONFIG['graph_seed']+len(tasks);a,q,g=c.graph(z,zq,countries,graphseed)
            y=train[c.ITEMS].to_numpy(float);coords=np.column_stack([y.mean(1),y@c.Q]);obs=np.array([getattr(target,f) for f in c.ITEMS],float)
            task=dict(uid=uid,evaluation=domain,target_country=target.country,target_iso3=target.iso3,target_survey=target.survey_id,training_survey=train.survey_id.tolist(),countries=countries,z=z.tolist(),zquery=zq.tolist(),y=y.tolist(),coordinates=coords.tolist(),observed=obs.tolist(),graphs={k:v.tolist() for k,v in a.items()},query_weights={k:v.tolist() for k,v in q.items()},graph_seed=graphseed,bandwidth=g['bandwidth'],preprocessing=prep)
            tasks.append(task);preps.append(dict(uid=uid,evaluation=domain,target_survey=target.survey_id,training_rounds=len(train),training_countries=train.country.nunique(),**prep))
            for row in train.itertuples():membership.append(dict(uid=uid,evaluation=domain,target_survey=target.survey_id,target_country=target.country,training_country=row.country,training_survey=row.survey_id,donor_end=str(ends.loc[row.Index].date()),target_start=str(target.fieldwork_start.date())))
            for edge in g['edges']:edge_rows.append(dict(uid=uid,**edge))
    dump_gz(ROOT/'inputs/tasks.json.gz',tasks);pd.DataFrame(membership).to_csv(OUT/'training_membership.csv',index=False);pd.DataFrame(edge_rows).to_csv(OUT/'graph_edges.csv.gz',index=False,compression='gzip');c.save(OUT/'preprocessing.json',preps)
    c.save(ROOT/'SOURCE_BINDINGS.json',dict(status='prepared_before_model_fits',sources=sources,targets=61,rounds=169,countries=62,item_order=c.ITEMS,basis_Q=c.Q.tolist(),task_sha256=c.sha(ROOT/'inputs/tasks.json.gz')))
    synthetic_check()
    print(json.dumps(load(ROOT/'SYNTHETIC_CHECK.json'),indent=2),flush=True)

def freeze():
    assert not (ROOT/'FREEZE.json').exists()
    protected={}
    for folder in [REV/'07_manuscript/latex',REV/'FINAL_Scientific_Reports_2026_10_22',PREV,REV/'04_mental_health/endes_ising_2026_10_22']:
        for p in folder.rglob('*'):
            if p.is_file():protected[str(p.relative_to(REV))]=c.sha(p)
    finalzip=REV/'FINAL_Scientific_Reports_2026_10_22.zip';protected[str(finalzip.relative_to(REV))]=c.sha(finalzip)
    files={str(p.relative_to(ROOT)):c.sha(p) for p in ROOT.rglob('*') if p.is_file() and 'independent_review' not in p.parts and '__pycache__' not in p.parts}
    c.save(ROOT/'FREEZE.json',dict(status='frozen_before_new_scientific_fits_retrospective',created_utc=datetime.now(timezone.utc).isoformat(),files=files,protected=protected,source_bindings=load(ROOT/'SOURCE_BINDINGS.json')))
    print('Frozen; no scientific GNN fits started',flush=True)

def verify_freeze():
    f=load(ROOT/'FREEZE.json')
    for name,h in f['files'].items():assert c.sha(ROOT/name)==h,name
    for name,h in f['protected'].items():assert c.sha(REV/name)==h,name
    return f

def fit_worker(arg):
    task,arm,seed=arg;uid=task['uid'];path=OUT/'states'/f'{uid}_{arm}_{seed}.json.gz'
    if path.exists():raise RuntimeError('Preserve existing fit '+str(path))
    z=np.array(task['z']);zq=np.array(task['zquery']);a=np.array(task['graphs'][arm]);q=np.array(task['query_weights'][arm]);countries=task['countries'];yy=np.array(task['coordinates'])
    states={};readouts={};diagnostics={}
    for branch,idx in [('mean',[0]),('profile',[1,2,3,4])]:
        pars,diag=c.train_encoder(z,yy[:,idx],a,countries,seed);h,training,hq,qp=c.forward(pars,z,a,zq,q)
        fit=c.readout(z,zq,yy[:,idx],countries,h,hq,idx)
        states[branch]=pars;readouts[branch]=fit;diagnostics[branch]=diag
    record=dict(uid=uid,arm=arm,seed=seed,states=states,readouts=readouts,diagnostics=diagnostics,finished_utc=datetime.now(timezone.utc).isoformat(),task_sha256=c.sha(ROOT/'inputs/tasks.json.gz'),freeze_sha256=c.sha(ROOT/'FREEZE.json'))
    dump_gz(path,record);return dict(uid=uid,arm=arm,seed=seed,seconds=sum(d['elapsed_seconds'] for d in diagnostics.values()))

def fit(resume=False):
    verify_freeze();permission=load(ROOT/'START_AUTHORISATION.json');assert permission['start_new_fits_authorized'] is True
    tasks=load_gz(ROOT/'inputs/tasks.json.gz');args=[]
    for t in tasks:
        for a in c.CONFIG['arms']:
            for s in c.CONFIG['seeds']:
                p=OUT/'states'/f'{t["uid"]}_{a}_{s}.json.gz'
                if resume and p.exists():continue
                args.append((t,a,s))
    start=time.perf_counter();print(f'Starting {len(args)} target/arm/seed jobs, two encoders each',flush=True)
    with ProcessPoolExecutor(max_workers=c.CONFIG['workers']) as pool:
        fs=[pool.submit(fit_worker,arg) for arg in args]
        for i,f in enumerate(as_completed(fs),1):
            r=f.result()
            if i%10==0 or i==len(fs):print(json.dumps(dict(completed=i,total=len(fs),last=r,elapsed_seconds=time.perf_counter()-start)),flush=True)
    c.save(OUT/'FIT_COMPLETION.json',dict(status='all_fixed_encoder_and_conditional_readout_fits_completed_pending_QC',states=len(list((OUT/'states').glob('*.json.gz'))),encoder_fits=732,elapsed_seconds=time.perf_counter()-start,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,torch=torch.__version__,neural_weights_bayesian=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--freeze',action='store_true');p.add_argument('--fit',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.prepare:prepare()
    elif args.freeze:freeze()
    elif args.fit:fit(args.resume)
    else:p.error('Choose --prepare, --freeze or --fit')
