"""Exact finite-state information summaries. No estimation or model fitting."""
from __future__ import annotations

import numpy as np

STATES = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(int)
ITEMS = ["V744" + letter for letter in "ABCDE"]
LABELS = ["Going out", "Neglecting children", "Arguing", "Refusing sex", "Burning food"]


def entropy_bits(probabilities):
    p = np.asarray(probabilities, dtype=float)
    positive = p > 0
    return float(-np.sum(p[positive] * np.log2(p[positive])))


def distribution_metrics(probabilities):
    p = np.asarray(probabilities, dtype=float).copy()
    assert p.shape == (32,) and np.isfinite(p).all() and (p >= 0).all()
    assert abs(p.sum() - 1) < 1e-10
    p /= p.sum()
    joint = entropy_bits(p)
    item_rows = []
    for j, (item, label) in enumerate(zip(ITEMS, LABELS)):
        marginal = np.bincount(STATES[:, j], weights=p, minlength=2)
        keep = [i for i in range(5) if i != j]
        codes = (STATES[:, keep] * 2 ** np.arange(4)).sum(axis=1)
        others = np.bincount(codes, weights=p, minlength=16)
        conditional = joint - entropy_bits(others)
        direct_conditional = 0.0
        for code in range(16):
            selected = codes == code
            if others[code] > 0:
                direct_conditional += others[code] * entropy_bits(p[selected] / others[code])
        marginal_h = entropy_bits(marginal)
        assert abs(conditional - direct_conditional) < 1e-12
        assert -1e-12 <= conditional <= marginal_h + 1e-12
        item_rows.append(dict(item=item, label=label, endorsement_probability=float(marginal[1]),
                              marginal_entropy_bits=marginal_h, others_entropy_bits=entropy_bits(others),
                              conditional_entropy_bits=conditional,
                              information_from_other_items_bits=marginal_h - conditional))
    marginal_sum = sum(row["marginal_entropy_bits"] for row in item_rows)
    tc = marginal_sum - joint
    own_marginals = p @ STATES
    product = np.prod(np.where(STATES, own_marginals, 1 - own_marginals), axis=1)
    selected = p > 0
    assert (product[selected] > 0).all()
    direct_tc = float(np.sum(p[selected] * np.log2(p[selected] / product[selected])))
    assert abs(tc - direct_tc) < 1e-12 and tc >= -1e-12
    any_p = float(p[1:].sum())
    any_entropy = entropy_bits([p[0], any_p])
    nonzero_entropy = entropy_bits(p[1:] / any_p) if any_p > 0 else 0.0
    within = any_p * nonzero_entropy
    assert abs(joint - any_entropy - within) < 1e-12
    summary = dict(joint_entropy_bits=joint, normalized_joint_entropy=joint / 5,
                   effective_response_patterns=float(2 ** joint),
                   marginal_entropy_sum_bits=marginal_sum, total_correlation_bits=tc,
                   any_endorsement_probability=any_p, any_endorsement_entropy_bits=any_entropy,
                   nonzero_pattern_entropy_bits=nonzero_entropy,
                   normalized_nonzero_pattern_entropy=nonzero_entropy / np.log2(31),
                   within_endorser_contribution_bits=within,
                   mean_conditional_item_entropy_bits=float(np.mean([r["conditional_entropy_bits"] for r in item_rows])))
    assert 0 <= joint <= 5 + 1e-12 and nonzero_entropy <= np.log2(31) + 1e-12
    return summary, item_rows
