"""Replay public country-level Afrobarometer classifier summaries.

Recompute equal-country means and paired saved-error bootstrap intervals.
This does not reconstruct respondent predictions or refit a classifier.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent


def main():
    cfg = json.loads((BASE / "CONFIG.json").read_text())
    metrics = pd.read_csv(BASE / "results/country_metrics.csv")
    summary = pd.read_csv(BASE / "results/model_summary.csv").set_index("model")
    pairs = pd.read_csv(BASE / "results/paired_summary.csv")
    differences = pd.read_csv(BASE / "results/paired_country_differences.csv")
    countries = sorted(metrics.iso3.unique())
    assert len(countries) == cfg["countries"] == 39
    assert len(metrics) == 234 and not metrics.duplicated(["iso3", "model"]).any()
    averaged = ["brier", "log_loss", "roc_auc", "average_precision", "observed_endorsement",
                "predicted_endorsement", "mean_prediction_bias", "weighted_sensitivity",
                "weighted_specificity", "weighted_accuracy", "weighted_balanced_accuracy",
                "raw_sensitivity", "raw_specificity", "raw_accuracy", "raw_balanced_accuracy"]
    for model, rows in metrics.groupby("model"):
        saved = summary.loc[model]
        assert len(rows) == saved.countries == 39
        assert rows.n.sum() == saved.women == cfg["women"]
        assert rows.positive.sum() == saved.positive
        np.testing.assert_allclose(rows[averaged].mean(), saved[averaged].astype(float), atol=1e-12, rtol=0)
        np.testing.assert_allclose(rows.mean_prediction_bias.abs().mean(), saved.mean_absolute_prediction_bias, atol=1e-12, rtol=0)
        for column in ["tp", "tn", "fp", "fn"]:
            assert rows[column].sum() == saved[column]
    assert len(pairs) == 10 and len(differences) == 390
    draws = np.random.default_rng(cfg["bootstrap_seed"]).integers(0, 39, (cfg["bootstrap_draws"], 39))
    for row in pairs.itertuples():
        wide = metrics.pivot(index="iso3", columns="model", values=row.metric).loc[countries]
        delta = (wide[row.first] - wide[row.second]).to_numpy()
        saved = differences.loc[differences["first"].eq(row.first) &
                                differences["second"].eq(row.second) &
                                differences.metric.eq(row.metric)].set_index("iso3").loc[countries]
        np.testing.assert_allclose(delta, saved.difference, atol=1e-12, rtol=0)
        low, high = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])
        np.testing.assert_allclose([delta.mean(), low, high], [row.difference, row.lower, row.upper], atol=1e-12, rtol=0)
        assert (int((delta < 0).sum()), int((delta > 0).sum()), int((delta == 0).sum())) == (
            row.improved_countries, row.worse_countries, row.equal_countries)
    print(json.dumps(dict(status="passed", country_score_rows=len(metrics), model_summaries=len(summary),
                         paired_country_rows=len(differences), conditional_bootstrap_comparisons=len(pairs),
                         scope="Public aggregate country summaries and fixed-error resampling; no individual reconstruction, model refitting or neural training."), indent=2))


if __name__ == "__main__":
    main()
