"""Two warm synthetic branch timings in each of two initialized one-thread workers."""
from concurrent.futures import ProcessPoolExecutor
import json,time
import numpy as np
import common as c

def worker(seed):
    rng=np.random.default_rng(seed);x=rng.normal(size=(162,5));countries=np.array([f'c{i%61:02}' for i in range(162)]);y=rng.normal(size=(162,5));a,q,g=c.graph(x,rng.normal(size=5),countries,seed)
    _,cold=c.train_encoder(x,y[:,:1],a['real'],countries,42)
    _,mean=c.train_encoder(x,y[:,:1],a['real'],countries,42)
    _,profile=c.train_encoder(x,y[:,1:],a['real'],countries,42)
    return dict(worker_synthetic_seed=seed,startup_including_first_fit_seconds=cold['elapsed_seconds'],warm_mean_fit_seconds=mean['elapsed_seconds'],warm_profile_fit_seconds=profile['elapsed_seconds'])
if __name__=='__main__':
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:records=list(pool.map(worker,[2026102217,2026102218]))
    avg=float(np.mean([r['warm_mean_fit_seconds']+r['warm_profile_fit_seconds'] for r in records]))
    result=dict(status='passed',real_outcomes_used=False,workers=2,threads_per_worker=1,steps=400,nodes=162,records=records,estimated_366_job_two_worker_neural_minutes=366*avg/120,wall_seconds=time.perf_counter()-start,scope='warm synthetic encoder timing; Bayesian readout, workers, storage and QC add overhead; no target-based tuning')
    c.save(c.ROOT/'WARM_RUNTIME.json',result);print(json.dumps(result,indent=2))
