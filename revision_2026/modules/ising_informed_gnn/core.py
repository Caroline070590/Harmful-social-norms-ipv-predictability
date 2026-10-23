"""Five-node masked-response GNN; no respondent records in public outputs."""
from itertools import combinations
import numpy as np
from scipy.special import expit
import torch

torch.set_num_threads(1)
DTYPE=torch.float64
STATES=((np.arange(32)[:,None]>>np.arange(5))&1).astype(float)
PAIRS=list(combinations(range(5),2))
TARGET=np.repeat(np.arange(5),16)
ZERO_STATES=[]
for j in range(5):
    for code in range(16):
        bits=np.zeros(5);bits[np.arange(5)!=j]=(code>>np.arange(4))&1
        ZERO_STATES.append(bits)
ZERO_STATES=np.array(ZERO_STATES)
OBSERVED=np.ones((80,5));OBSERVED[np.arange(80),TARGET]=0
NODE_INPUT=np.concatenate([ZERO_STATES[:,:,None],OBSERVED[:,:,None],np.broadcast_to(np.eye(5),(80,5,5))],axis=2)
FLAT_INPUT=np.concatenate([ZERO_STATES,OBSERVED,np.eye(5)[TARGET]],axis=1)
ZERO_ID=(ZERO_STATES*(2**np.arange(5))).sum(1).astype(int)
ONE_ID=ZERO_ID+(1<<TARGET)
CONFIG_ID=np.zeros((32,5),int)
for j in range(5):
    others=np.arange(5)!=j
    CONFIG_ID[:,j]=16*j+(STATES[:,others]*(2**np.arange(4))).sum(1).astype(int)

def configurations(mass):
    """Exact weighted counts; every woman contributes five hidden-item tasks."""
    return np.asarray(mass)[ZERO_ID],np.asarray(mass)[ONE_ID]

def graph(theta,arm,permutation):
    a=np.zeros((5,5))
    for value,(j,k) in zip(theta[5:],PAIRS):a[j,k]=a[k,j]=value
    if arm=='permuted_ising':a=a[np.ix_(permutation,permutation)]
    if arm=='self_message':return np.eye(5)
    denominator=np.abs(a).sum(1)
    assert (denominator>0).all()
    return a/denominator[:,None]

def offsets(mass):
    p=np.clip(np.asarray(mass)@STATES/np.sum(mass),1e-6,1-1e-6)
    return np.log(p/(1-p))

def conditional_from_joint(p):
    p=np.asarray(p)
    q=p[ONE_ID]/(p[ONE_ID]+p[ZERO_ID])
    return q[CONFIG_ID]

class MaskedNet(torch.nn.Module):
    def __init__(self,arm,seed,hidden=8):
        super().__init__();self.arm=arm
        torch.manual_seed(int(seed))
        self.layer1=torch.nn.Linear(15 if arm=='conditional_MLP' else 14,hidden,dtype=DTYPE)
        self.layer2=torch.nn.Linear(hidden if arm=='conditional_MLP' else 2*hidden,1,dtype=DTYPE)
    def forward(self,a,offset):
        if self.arm=='conditional_MLP':
            x=torch.tensor(FLAT_INPUT,dtype=DTYPE)
            z=self.layer2(torch.relu(self.layer1(x))).squeeze(-1)
        else:
            x=torch.tensor(NODE_INPUT,dtype=DTYPE)
            ax=torch.einsum('ij,bjf->bif',a,x)
            h=torch.relu(self.layer1(torch.cat([x,ax],dim=-1)))
            ah=torch.einsum('ij,bjf->bif',a,h)
            z=self.layer2(torch.cat([h,ah],dim=-1)).squeeze(-1)[torch.arange(80),torch.tensor(TARGET)]
        return z+offset[torch.tensor(TARGET)]

def objective(net,a,offset,negative,positive,penalty):
    logits=net(a,offset)
    nll=(negative*torch.nn.functional.softplus(logits)+positive*torch.nn.functional.softplus(-logits)).sum()/(negative.sum()+positive.sum())
    prior=.5*penalty*(net.layer1.weight.square().sum()+net.layer2.weight.square().sum())
    return nll+prior,nll,prior

def fit_neural(mass,theta,arm,seed,cfg):
    aa=np.eye(5) if arm=='conditional_MLP' else graph(theta,arm,cfg['node_permutation'])
    a=torch.tensor(aa,dtype=DTYPE);off=torch.tensor(offsets(mass),dtype=DTYPE)
    n,p=configurations(mass);n=torch.tensor(n,dtype=DTYPE);p=torch.tensor(p,dtype=DTYPE)
    net=MaskedNet(arm,seed,cfg['hidden_units']);opt=torch.optim.Adam(net.parameters(),lr=cfg['learning_rate'])
    penalty=cfg['weight_prior_precision']/np.sum(mass)
    trace=[]
    for step in range(cfg['updates']):
        opt.zero_grad();loss,nll,prior=objective(net,a,off,n,p,penalty);loss.backward();opt.step()
        trace.append([step,float(loss.detach()),float(nll.detach()),float(prior.detach())])
    net.zero_grad();loss,nll,prior=objective(net,a,off,n,p,penalty);loss.backward()
    with torch.no_grad():q=torch.sigmoid(net(a,off)).numpy()
    states={k:v.detach().numpy() for k,v in net.state_dict().items()}
    info=dict(final_objective=float(loss.detach()),final_NLL=float(nll.detach()),final_penalty=float(prior.detach()),penalty_coefficient=float(penalty),gradient_max_abs=max(float(x.grad.abs().max()) for x in net.parameters()),parameters=sum(x.numel() for x in net.parameters()),updates=cfg['updates'])
    return q[CONFIG_ID],states,np.array(trace),info,aa,off.numpy()

def losses(pattern,q):
    y=STATES[pattern];p=np.asarray(q)
    p=np.clip(p,1e-15,1-1e-15)
    return np.stack([(p-y)**2,-(y*np.log(p)+(1-y)*np.log1p(-p))],axis=-1)

def mask_invariance():
    for pattern in range(32):
        for j in range(5):
            flip=pattern^(1<<j)
            assert CONFIG_ID[pattern,j]==CONFIG_ID[flip,j]
            index=CONFIG_ID[pattern,j]
            assert ZERO_STATES[index,j]==0 and OBSERVED[index,j]==0
    assert NODE_INPUT.shape==(80,5,7) and FLAT_INPUT.shape==(80,15)
    return dict(status='passed',hidden_flip_pairs_checked=160,unique_masked_configurations=80)
