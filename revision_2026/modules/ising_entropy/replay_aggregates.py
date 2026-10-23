"""Portable replay of public full-distribution summaries and bootstrap ranges.

Training-fold distributions and bootstrap probability vectors are deliberately
excluded. This script does not claim their original-source or fit replay.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from entropy import distribution_metrics

BASE = Path(__file__).resolve().parent


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    summary = pd.read_csv(BASE / "results/distribution_metrics.csv")
    item = pd.read_csv(BASE / "results/item_entropy.csv")
    empirical = pd.read_csv(BASE / "public_inputs/joint_pattern_feasibility.csv")
    bindings = 0
    maximum_error = 0.0
    for weighting in ("released_weight", "equal_person"):
        for model in ("empirical_plugin", "independent", "common_acceptance", "ising"):
            if model == "empirical_plugin":
                p = empirical.weighted_share.to_numpy() if weighting == "released_weight" else empirical.women.to_numpy() / empirical.women.sum()
            else:
                p = pd.read_csv(BASE / f"public_inputs/full_pattern_probabilities_{weighting}_{model}.csv").probability.to_numpy()
            expected, per_item = distribution_metrics(p)
            row = summary[(summary.scope == "full_cohort_descriptive") & (summary.weighting == weighting) & (summary.model == model)]
            assert len(row) == 1
            for key, value in expected.items():
                error = abs(value - float(row.iloc[0][key]))
                assert error < 1e-12, (weighting, model, key, error)
                maximum_error = max(maximum_error, error)
                bindings += 1
            for expected_item in per_item:
                row = item[(item.scope == "full_cohort_descriptive") & (item.weighting == weighting) & (item.model == model) & (item.item == expected_item["item"])]
                assert len(row) == 1
                for key, value in expected_item.items():
                    if key in ("item", "label"):
                        assert row.iloc[0][key] == value
                    else:
                        error = abs(value - float(row.iloc[0][key]))
                        assert error < 1e-12
                        maximum_error = max(maximum_error, error)
                        bindings += 1
    draws = pd.read_csv(BASE / "results/bootstrap_metrics.csv")
    item_draws = pd.read_csv(BASE / "results/bootstrap_item_entropy.csv")
    ranges = pd.read_csv(BASE / "results/bootstrap_ranges.csv")
    for _, row in ranges.iterrows():
        values = draws[row["metric"]] if row["item"] == "joint" else item_draws[item_draws["item"] == row["item"]][row["metric"]]
        assert len(values) == 200 and values.notna().all()
        lower, upper = np.quantile(values, [.025, .975])
        assert abs(lower - row.lower) < 1e-12 and abs(upper - row.upper) < 1e-12
    result = dict(status="passed", full_distributions_replayed=8, scalar_bindings=bindings,
                  bootstrap_range_rows=len(ranges), max_error=maximum_error,
                  scope="Public full-distribution summaries, conditional item entropies and bootstrap summary percentiles only; no training-fold source reconstruction, original model refitting or original raw respondent replay.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
