"""Fixed one-message classifier core; no source loading or target labels accepted.

Mean GraphSAGE: a learned nonlinear transform of own and mean-neighbour raw
predictor vectors. Graphs and aggregation are explicitly supplied by the runner.
This separate implementation does not import an earlier producer's fit routine.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Classifier(nn.Module):
    def __init__(self, dimensions, hidden, seed, prevalence):
        super().__init__()
        self.lin = nn.Linear(dimensions, hidden, dtype=torch.float64)
        self.out = nn.Linear(hidden, 1, dtype=torch.float64)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            for layer in (self.lin, self.out):
                layer.weight.normal_(0., layer.in_features ** -.5, generator=generator)
                layer.bias.zero_()
            self.out.bias.fill_(float(np.log(prevalence / (1 - prevalence))))

    def forward(self, inputs):
        return self.out(F.relu(self.lin(inputs))).reshape(-1)


def parameters_numpy(model):
    return {name.replace(".", "_"): tensor.detach().numpy().copy()
            for name, tensor in model.state_dict().items()}


def parameter_hash(state):
    digest = hashlib.sha256()
    for name in ("lin_weight", "lin_bias", "out_weight", "out_bias"):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(state[name], dtype=np.float64).tobytes())
    return digest.hexdigest()


def fit_classifier(x_train, means_train, y_train, weight_train,
                   x_test, means_test, arm, seed, config):
    """Return train/test logits, four-array state and diagnostic trace.

    No test outcome argument exists. Graph-self/MLP inputs must be passed
    explicitly; the runner is responsible for donor-only mean construction.
    """
    assert arm in config["neural_arms"]
    train_input = x_train if arm == "MLP" else np.column_stack([x_train, means_train])
    test_input = x_test if arm == "MLP" else np.column_stack([x_test, means_test])
    y = torch.as_tensor(np.asarray(y_train, dtype=np.float64))
    weight = torch.as_tensor(np.asarray(weight_train, dtype=np.float64))
    assert np.isclose(weight_train.sum(), 1) and np.all(weight_train > 0)
    prevalence = float(weight_train @ y_train)
    assert 0 < prevalence < 1
    hidden = config["hidden_mlp"] if arm == "MLP" else config["hidden_graph"]
    model = Classifier(train_input.shape[1], hidden, seed, prevalence)
    initial_hash = parameter_hash(parameters_numpy(model))
    train_tensor, test_tensor = (torch.as_tensor(np.ascontiguousarray(v)) for v in (train_input, test_input))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0.)
    trace = []

    def objective():
        logits = model(train_tensor)
        bce = (weight * F.binary_cross_entropy_with_logits(logits, y, reduction="none")).sum()
        penalty = config["penalty_lambda"] / 2 * (model.lin.weight.square().sum() + model.out.weight.square().sum())
        return bce + penalty, bce, penalty

    def snapshot(step):
        optimizer.zero_grad(set_to_none=True)
        total, bce, penalty = objective()
        total.backward()
        gradient = max(float(p.grad.detach().abs().max()) for p in model.parameters())
        trace.append({"step": int(step), "total": float(total.detach()), "bce": float(bce.detach()),
                      "penalty": float(penalty.detach()), "gradient_infinity_norm": gradient})
        assert all(np.isfinite(list(trace[-1].values())))

    snapshot(0)
    for step in range(1, config["epochs"] + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective()
        loss.backward()
        optimizer.step()
        if step in (100, config["epochs"]):
            snapshot(step)
    with torch.no_grad():
        train_logit, test_logit = model(train_tensor).numpy(), model(test_tensor).numpy()
        hidden_variance = float(F.relu(model.lin(train_tensor)).var(unbiased=False))
    state = parameters_numpy(model)
    assert all(np.isfinite(v).all() for v in state.values())
    assert np.isfinite(train_logit).all() and np.isfinite(test_logit).all()
    diagnostics = {"arm": arm, "seed": int(seed), "initial_parameter_sha256": initial_hash,
                   "final_parameter_sha256": parameter_hash(state), "trace": trace,
                   "allocated_parameters": int(sum(p.numel() for p in model.parameters())),
                   "hidden_variance": hidden_variance, "training_prevalence": prevalence,
                   "dtype": "float64", "threads": 1, "updates": config["epochs"],
                   "finite": True, "convergence_certified": False}
    return train_logit, test_logit, state, diagnostics
