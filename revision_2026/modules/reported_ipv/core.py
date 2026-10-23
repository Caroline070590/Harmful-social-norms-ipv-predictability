"""Fixed capacity-matched one-message mean GraphSAGE and controls.

This module receives only prepared training outcomes and supplied predictor
aggregations. No held-out outcome argument or model-selection routine exists.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Classifier(nn.Module):
    def __init__(self, dimension, hidden, seed, prevalence):
        super().__init__()
        self.lin = nn.Linear(dimension, hidden, dtype=torch.float64)
        self.out = nn.Linear(hidden, 1, dtype=torch.float64)
        rng = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for layer in (self.lin, self.out):
                layer.weight.normal_(0., layer.in_features ** -.5, generator=rng)
                layer.bias.zero_()
            self.out.bias.fill_(float(np.log(prevalence / (1 - prevalence))))

    def forward(self, x):
        return self.out(F.relu(self.lin(x))).reshape(-1)


def state_numpy(model):
    return {k.replace('.', '_'): v.detach().numpy().copy()
            for k, v in model.state_dict().items()}


def forward_numpy(inputs, state):
    h = np.maximum(inputs @ state['lin_weight'].T + state['lin_bias'], 0.)
    return (h @ state['out_weight'].T + state['out_bias']).reshape(-1)


def fit(x_train, y_train, x_test, seed, config):
    """Fixed updates; controls differ only in the externally supplied inputs."""
    assert len(y_train) == len(x_train) and x_train.shape[1] == x_test.shape[1]
    prevalence = float(np.mean(y_train))
    assert 0 < prevalence < 1
    model = Classifier(x_train.shape[1], config['hidden'], seed, prevalence)
    initial = state_numpy(model)
    x = torch.as_tensor(np.ascontiguousarray(x_train, dtype=np.float64))
    query = torch.as_tensor(np.ascontiguousarray(x_test, dtype=np.float64))
    y = torch.as_tensor(np.asarray(y_train, dtype=np.float64))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    trace = []

    def objective():
        bce = F.binary_cross_entropy_with_logits(model(x), y)
        penalty = config['penalty_lambda'] / 2 * (model.lin.weight.square().sum() + model.out.weight.square().sum())
        return bce + penalty, bce, penalty

    def snapshot(step):
        optimizer.zero_grad(set_to_none=True)
        value, bce, penalty = objective()
        value.backward()
        trace.append(dict(step=step, objective=float(value.detach()), bce=float(bce.detach()),
            penalty=float(penalty.detach()), max_abs_gradient=max(float(p.grad.abs().max()) for p in model.parameters())))

    snapshot(0)
    for step in range(1, config['epochs'] + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective()
        loss.backward()
        optimizer.step()
        if step in (100, 200, config['epochs']):
            snapshot(step)
    with torch.no_grad():
        train_logit = model(x).numpy()
        test_logit = model(query).numpy()
    state = state_numpy(model)
    assert np.isfinite(test_logit).all() and np.isfinite(train_logit).all()
    assert np.allclose(forward_numpy(x_test, state), test_logit, atol=1e-12, rtol=1e-12)
    return train_logit, test_logit, state, dict(initial=initial, trace=trace,
        allocated_parameters=int(sum(p.numel() for p in model.parameters())),
        updates=config['epochs'], dtype='float64', threads=1, convergence_certified=False)
