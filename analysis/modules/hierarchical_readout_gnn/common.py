"""Fixed graph encoder and exact conditional conjugate country readout."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
import torch
from scipy.linalg import cho_factor,cho_solve,helmert
from scipy.special import gammaln,logsumexp
ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'CONFIG.json').read_text())
FEATURES=CONFIG['features'];ITEMS=CONFIG['items'];Q=helmert(5,full=False).T
RECONSTRUCT=np.column_stack([np.ones(5),Q])

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def country_weights(countries):
    _,inv,counts=np.unique(countries,return_inverse=True,return_counts=True)
    w=1.0/counts[inv];return w/w.sum()
def standardize(x,w):
    x=np.asarray(x,float);w=np.asarray(w,float)
    mean=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale[scale<1e-12]=1
    return (x-mean)/scale,mean,scale

def preprocess(x,xquery,countries):
    w=country_weights(countries);x=np.asarray(x,float);xquery=np.asarray(xquery,float)
    assert np.isfinite(x).any(axis=0).all()
    impute=np.array([np.average(x[np.isfinite(x[:,j]),j],weights=w[np.isfinite(x[:,j])]) for j in range(x.shape[1])])
    filled=np.where(np.isfinite(x),x,impute);z,mean,scale=standardize(filled,w)
    zq=(np.where(np.isfinite(xquery),xquery,impute)-mean)/scale
    return z,zq,dict(impute=impute.tolist(),mean=mean.tolist(),scale=scale.tolist(),donor_missing=(~np.isfinite(x)).sum(axis=0).tolist(),target_missing=(~np.isfinite(xquery)).tolist())

def graph(z,zq,countries,seed):
    n=len(z);k=CONFIG['neighbours'];countries=np.asarray(countries)
    distances=np.linalg.norm(z[:,None,1:]-z[None,:,1:],axis=2)
    chosen=[];selected=[]
    for i in range(n):
        eligible=np.flatnonzero(countries!=countries[i]);assert len(eligible)>=k
        ids=eligible[np.argsort(distances[i,eligible],kind='stable')[:k]]
        chosen.append(ids);selected.extend(distances[i,ids].tolist())
    positive=np.array(selected)[np.array(selected)>0];band=float(np.median(positive)) if len(positive) else 1.
    a=np.zeros((n,n));random=np.zeros((n,n));rng=np.random.default_rng(seed)
    edges=[]
    for i,ids in enumerate(chosen):
        weights=np.exp(-.5*(distances[i,ids]/band)**2);weights/=weights.sum()
        a[i,ids]=weights
        rnd=rng.choice(np.flatnonzero(countries!=countries[i]),size=k,replace=False)
        random[i,rnd]=weights
        for kind,jj in [('real',ids),('random',rnd)]:
            for j,ww in zip(jj,weights):edges.append(dict(arm=kind,node=i,donor=int(j),weight=float(ww)))
    qdist=np.linalg.norm(z[:,1:]-zq[None,1:],axis=1);qi=np.argsort(qdist,kind='stable')[:k]
    qw=np.exp(-.5*(qdist[qi]/band)**2);qw/=qw.sum();q=np.zeros(n);q[qi]=qw
    rq=np.zeros(n);rqi=rng.choice(n,size=k,replace=False);rq[rqi]=qw
    for kind,jj in [('real',qi),('random',rqi)]:
        for j,ww in zip(jj,qw):edges.append(dict(arm=kind,node=-1,donor=int(j),weight=float(ww)))
    assert np.allclose(a.sum(1),1) and np.allclose(random.sum(1),1)
    assert np.isfinite(q).all() and np.isfinite(rq).all() and np.isclose(q.sum(),1) and np.isclose(rq.sum(),1)
    assert not np.any(a[countries[:,None]==countries[None,:]])
    assert not np.any(random[countries[:,None]==countries[None,:]])
    return dict(real=a,self_only=np.zeros_like(a),random=random),dict(real=q,self_only=np.zeros_like(q),random=rq),dict(bandwidth=band,edges=edges,seed=seed)

def initial_parameters(seed,outputs):
    rng=np.random.default_rng(seed);d=len(FEATURES);u=CONFIG['units'];shapes={'W_self':(d,u),'W_neighbour':(d,u),'b_hidden':(u,),'W_linear':(d,outputs),'W_hidden':(u,outputs),'b_output':(outputs,)}
    pars={}
    for key,shape in shapes.items():
        if key.startswith('b_'):pars[key]=np.zeros(shape)
        else:
            bound=np.sqrt(6/(shape[0]+shape[1]));pars[key]=rng.uniform(-bound,bound,size=shape)
    return pars

def forward(pars,z,a,zq=None,q=None):
    p={k:np.asarray(v,float) for k,v in pars.items()}
    h=np.tanh(z@p['W_self']+(a@z)@p['W_neighbour']+p['b_hidden'])
    out=z@p['W_linear']+h@p['W_hidden']+p['b_output']
    if zq is None:return h,out
    hq=np.tanh(zq@p['W_self']+(q@z)@p['W_neighbour']+p['b_hidden'])
    oq=zq@p['W_linear']+hq@p['W_hidden']+p['b_output']
    return h,out,hq,oq

def torch_objective(pars,z,az,y,w):
    h=torch.tanh(z@pars['W_self']+az@pars['W_neighbour']+pars['b_hidden'])
    prediction=z@pars['W_linear']+h@pars['W_hidden']+pars['b_output']
    error=((prediction-y).square().mean(1)*w).sum()
    penalty=CONFIG['l2_mean_all_parameters']*sum(v.square().sum() for v in pars.values())/sum(v.numel() for v in pars.values())
    return error+penalty,error,penalty

def train_encoder(z,y,a,countries,seed,steps=None):
    start=time.perf_counter();torch.set_num_threads(1)
    w=country_weights(countries);ys,ym,yscale=standardize(y,w)
    p={k:torch.tensor(v,dtype=torch.float64,requires_grad=True) for k,v in initial_parameters(seed,y.shape[1]).items()}
    optimizer=torch.optim.Adam(list(p.values()),lr=CONFIG['learning_rate'],betas=tuple(CONFIG['adam_betas']),eps=CONFIG['adam_epsilon'])
    zt=torch.tensor(z,dtype=torch.float64);az=torch.tensor(a@z,dtype=torch.float64);yt=torch.tensor(ys,dtype=torch.float64);wt=torch.tensor(w,dtype=torch.float64)
    checkpoints=[];steps=CONFIG['steps'] if steps is None else steps
    for step in range(steps):
        optimizer.zero_grad();obj,err,pen=torch_objective(p,zt,az,yt,wt)
        if not torch.isfinite(obj):raise RuntimeError('nonfinite objective')
        obj.backward()
        if not all(torch.isfinite(v.grad).all() for v in p.values()):raise RuntimeError('nonfinite gradient')
        optimizer.step()
        if step in (0,99,199,steps-1):
            with torch.no_grad():vv,ee,pp=torch_objective(p,zt,az,yt,wt)
            checkpoints.append(dict(step=step+1,objective=float(vv),mse=float(ee),penalty=float(pp)))
    state={k:v.detach().numpy().tolist() for k,v in p.items()}
    return state,dict(ymean=ym.tolist(),yscale=yscale.tolist(),checkpoints=checkpoints,parameters=sum(v.numel() for v in p.values()),elapsed_seconds=time.perf_counter()-start)

def readout(z,zq,y,countries,hidden=None,hidden_query=None,coordinate_indices=None):
    w=country_weights(countries);hprep=None
    if hidden is not None:
        hh,hm,hs=standardize(hidden,w);hq=(hidden_query-hm)/hs
        zz=np.column_stack([z,hh]);zzq=np.r_[zq,hq];hprep=dict(mean=hm.tolist(),scale=hs.tolist())
    else:zz=z;zzq=zq
    x=np.column_stack([np.ones(len(z)),zz]);xs=np.r_[1.,zzq]
    cat=pd.Categorical(countries);c=np.eye(len(cat.categories))[cat.codes];cc=c@c.T
    v=np.diag([CONFIG['prior_intercept_variance']]+[CONFIG['prior_slope_variance']]*zz.shape[1]);n=len(z)
    coords=list(range(y.shape[1])) if coordinate_indices is None else coordinate_indices
    m0=np.array(CONFIG['prior_intercept_means'])[coords]
    centered=y-m0;an=CONFIG['a0']+n/2
    mu=[];scale=[];logev=[];blocks=[]
    for lam in CONFIG['lambda_grid']:
        k=np.eye(n)+lam*cc+x@v@x.T;cf=cho_factor(k,lower=True);solve=cho_solve(cf,centered)
        logdet=2*np.log(np.diag(cf[0])).sum();cross=x@v@xs;solve_cross=cho_solve(cf,cross)
        loc=m0+cross@solve;vf=1+lam+xs@v@xs-cross@solve_cross
        assert vf>=1+lam-1e-8
        bn=CONFIG['b0']+.5*np.sum(centered*solve,axis=0)
        le=gammaln(an)-gammaln(CONFIG['a0'])+CONFIG['a0']*np.log(CONFIG['b0'])-an*np.log(bn)-.5*logdet-n/2*np.log(2*np.pi)
        mu.append(loc);scale.append(np.sqrt(bn/an*vf));logev.append(le)
        bm=v@x.T@solve;bm[0]+=m0
        blocks.append(dict(lambda_value=float(lam),a_n=float(an),b_n=bn.tolist(),logdet=float(logdet),log_marginal_likelihood=le.tolist(),predictive_location=loc.tolist(),predictive_scale=np.sqrt(bn/an*vf).tolist(),predictive_variance_factor=float(vf),coefficient_mean=bm.tolist(),hidden_scaling=hprep))
    mu=np.array(mu).T;scale=np.array(scale).T;logev=np.array(logev).T
    logweights=logev+np.log(CONFIG['lambda_prior_probabilities']);weights=np.exp(logweights-logsumexp(logweights,axis=1)[:,None]);point=(mu*weights).sum(1)
    for j,b in enumerate(blocks):b['posterior_weight']=weights[:,j].tolist()
    return dict(point=point.tolist(),mu=mu.tolist(),scale=scale.tolist(),weights=weights.tolist(),df=float(2*an),blocks=blocks,hidden_scaling=hprep)

def draw_coords(blocks,seed,n):
    values=np.empty((n,5))
    for j in range(5):
        block=blocks[j];mu=np.array(block['mu']);sc=np.array(block['scale']);ww=np.array(block['weights'])
        rng=np.random.default_rng(seed+j*1000003);mix=rng.choice(len(mu),size=n,p=ww)
        rng2=np.random.default_rng(seed+j*1000003+5000011)
        values[:,j]=mu[mix]+sc[mix]*rng2.standard_t(block['df'],n)
    return values

def conditional_coordinate_blocks(fit,indices):
    return [dict(mu=fit['mu'][j],scale=fit['scale'][j],weights=fit['weights'][j],df=fit['df']) for j in indices]

def predictive_intervals(blocks,seed):
    n=CONFIG['posterior_draws'];attempts=[]
    while True:
        coords=draw_coords(blocks,seed,n);scenarios=coords@RECONSTRUCT.T;combined=np.column_stack([scenarios,coords])
        discrepancy=float(np.max(np.abs(np.quantile(combined[:n//2],[.025,.975],axis=0)-np.quantile(combined[n//2:],[.025,.975],axis=0))))
        attempts.append(dict(draws=n,half_stream_max_endpoint_difference_pp=discrepancy))
        if discrepancy<=CONFIG['interval_endpoint_tolerance_pp'] or n==CONFIG['posterior_max_draws']:break
        n=CONFIG['posterior_max_draws']
    result=dict(attempts=attempts,precision_passed=discrepancy<=CONFIG['interval_endpoint_tolerance_pp'],draws=n,seed=seed,out_of_range_fraction=float(np.mean((scenarios<0)|(scenarios>100))))
    for name,values in [('scenario',scenarios),('coordinate',coords)]:
        result[name]={str(level):np.quantile(values,[(1-level)/2,1-(1-level)/2],axis=0).tolist() for level in (.5,.8,.95)}
    return result

def losses(y,p,coordinates=None):
    e=p-y
    if coordinates is None:
        mean=float(e.mean());profile=e-mean;ordering_values=p
    else:
        coordinates=np.asarray(coordinates,float)
        mean=float(coordinates[0]-np.mean(y));profile=(coordinates[1:]-y@Q)@Q.T
        ordering_values=coordinates[1:]@Q.T
    wins=[]
    for j in range(5):
        for k in range(j+1,5):
            if y[j]==y[k]:continue
            wins.append(.5 if abs(ordering_values[j]-ordering_values[k])<=1e-10 else float(np.sign(ordering_values[j]-ordering_values[k])==np.sign(y[j]-y[k])))
    out=dict(MAE_pp=float(np.abs(e).mean()),MSE_pp2=float((e**2).mean()),level_MAE_pp=abs(mean),level_MSE_pp2=mean**2,profile_MAE_pp=float(np.abs(profile).mean()),profile_MSE_pp2=float((profile**2).mean()),ordering_accuracy=float(np.mean(wins)),ordering_pairs=len(wins),observed_ties=10-len(wins),clipped_components=int(np.sum((p<0)|(p>100))))
    assert np.isclose(out['MSE_pp2'],out['level_MSE_pp2']+out['profile_MSE_pp2'],atol=1e-9)
    return out
