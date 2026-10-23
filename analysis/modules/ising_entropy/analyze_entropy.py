"""Summarize existing probability distributions; never fit a model."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys

from pathlib import Path
import numpy as np
import pandas as pd

from entropy import distribution_metrics

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / "endes_ising_2026_10_22"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    freeze = json.loads((BASE / "FREEZE.json").read_text())
    for name, expected in freeze["sha256"].items():
        assert sha(BASE / name) == expected, name
    sources = {}
    summaries, items, boot, boot_items = [], [], [], []

    def record(path):
        sources[str(path.relative_to(SOURCE))] = sha(path)

    def add(p, metadata, target, item_target):
        summary, per_item = distribution_metrics(p)
        target.append({**metadata, **summary})
        item_target.extend({**metadata, **row} for row in per_item)

    # Public full-model probabilities and empirical counts, no person-level source.
    empirical_path = SOURCE / "results/joint_pattern_feasibility.csv"
    record(empirical_path)
    empirical = pd.read_csv(empirical_path)
    assert np.array_equal(empirical.pattern_id, np.arange(32))
    assert empirical.women.sum() == 29102
    shutil.copy2(empirical_path, BASE / "public_inputs" / empirical_path.name)
    for weighting in ("released_weight", "equal_person"):
        empirical_p = empirical.weighted_share.to_numpy() if weighting == "released_weight" else empirical.women.to_numpy() / empirical.women.sum()
        add(empirical_p, dict(scope="full_cohort_descriptive", model="empirical_plugin", weighting=weighting, partition=-1, fold=-1), summaries, items)
        for model in ("independent", "common_acceptance", "ising"):
            path = SOURCE / f"results/full_pattern_probabilities_{weighting}_{model}.csv"
            record(path)
            frame = pd.read_csv(path)
            assert np.array_equal(frame.pattern_id, np.arange(32))
            p = frame.probability.to_numpy()
            shutil.copy2(path, BASE / "public_inputs" / path.name)
            statepath = SOURCE / f"data_local/{weighting}_full_{model}.npz"
            record(statepath)
            with np.load(statepath) as state:
                assert np.max(abs(p - state["probabilities"])) < 1e-14
                assert np.max(abs(empirical_p - state["mass"] / state["mass"].sum())) < 1e-13
            add(p, dict(scope="full_cohort_descriptive", model=model, weighting=weighting, partition=-1, fold=-1), summaries, items)
            for partition in (42, 43, 44):
                for fold in range(5):
                    path = SOURCE / f"data_local/{weighting}_seed{partition}_fold{fold}_{model}.npz"
                    record(path)
                    with np.load(path) as state:
                        add(state["probabilities"], dict(scope="training_fold_distribution", model=model,
                            weighting=weighting, partition=partition, fold=fold), summaries, items)
    for draw in range(200):
        path = SOURCE / f"data_local/stability{draw:03d}.npz"
        record(path)
        with np.load(path) as state:
            add(state["probabilities"], dict(draw=draw), boot, boot_items)
    for name, records in (("distribution_metrics.csv", summaries), ("item_entropy.csv", items),
                          ("bootstrap_metrics.csv", boot), ("bootstrap_item_entropy.csv", boot_items)):
        pd.DataFrame(records).to_csv(BASE / "results" / name, index=False, float_format="%.17g")
    point = next(row for row in summaries if row["scope"] == "full_cohort_descriptive" and row["weighting"] == "released_weight" and row["model"] == "ising")
    point_items = [row for row in items if row["scope"] == "full_cohort_descriptive" and row["weighting"] == "released_weight" and row["model"] == "ising"]
    ranges = []
    for metric, value in point.items():
        if metric in ("scope", "model", "weighting", "partition", "fold"):
            continue
        lo, hi = np.quantile([row[metric] for row in boot], [.025, .975])
        ranges.append(dict(item="joint", metric=metric, estimate=value, lower=lo, upper=hi, refits=200))
    for row in point_items:
        for metric in ("endorsement_probability", "marginal_entropy_bits", "others_entropy_bits", "conditional_entropy_bits", "information_from_other_items_bits"):
            lo, hi = np.quantile([r[metric] for r in boot_items if r["item"] == row["item"]], [.025, .975])
            ranges.append(dict(item=row["item"], metric=metric, estimate=row[metric], lower=lo, upper=hi, refits=200))
    pd.DataFrame(ranges).to_csv(BASE / "results/bootstrap_ranges.csv", index=False, float_format="%.17g")
    for name in ("CONFIG.json", "results/BOOTSTRAP_SCOPE.json", "results/FEASIBILITY.json"):
        record(SOURCE / name)
    (BASE / "SOURCE_BINDINGS.json").write_text(json.dumps(dict(
        source_study=str(SOURCE.relative_to(BASE.parent.parent)),
        source_files_sha256=sources,
        public_inputs_are_existing_aggregate_tables=True,
        new_models_fitted=0,
        cohort_rows_reopened=False,
    ), indent=2) + "\n")
    (BASE / "ENVIRONMENT.json").write_text(json.dumps(dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__), indent=2) + "\n")
    (BASE / "CALCULATION_COMPLETION.json").write_text(json.dumps(dict(
        status="computed_pending_independent_review", distribution_rows=len(summaries), item_rows=len(items),
        bootstrap_rows=len(boot), bootstrap_item_rows=len(boot_items), percentile_rows=len(ranges),
        input_distributions_evaluated=len(summaries) + len(boot), new_models_fitted=0,
        manuscript_integration=False, uncertainty_scope="200 existing weighted Ising observed-PSU refits; 18 singleton strata fixed; not national-design intervals",
    ), indent=2) + "\n")
    print(pd.DataFrame(summaries).query("scope == 'full_cohort_descriptive'")[["model", "weighting", "joint_entropy_bits", "total_correlation_bits", "any_endorsement_probability", "nonzero_pattern_entropy_bits"]].to_string(index=False))
    print("Computed summaries for 298 saved/empirical distributions; no models refitted.")


if __name__ == "__main__":
    main()
