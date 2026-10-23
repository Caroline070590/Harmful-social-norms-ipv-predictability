#!/usr/bin/env python3
"""Independent, public-only consistency replay for the ENDES IPV classifiers.

Reads public CSVs and CONFIG.json only. Does not import producer/core modules or
read individual inputs, predictions, graphs, transformations or fitted states.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
DOMAINS = ("primary", "all_2024", "new_psu_2024")
ALGORITHMS = ("logistic", "boosting", "real", "self", "random")
MODELS = ("prevalence",) + tuple(a + "_" + b for a in ALGORITHMS for b in ("D", "A"))
LOSSES = ("Brier", "log_loss")
INT_FIELDS = ("TP", "FP", "TN", "FN", "women", "positives")
TOL = 3e-11
METRIC_FIELDS = {"Brier", "log_loss", "ROC_AUC", "average_precision", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "PPV", "TP", "FP", "TN", "FN", "women", "positives", "prevalence", "mean_prediction", "threshold"}
SCHEMAS = {
    "metrics.csv": METRIC_FIELDS | {"domain", "model"},
    "initialization_metrics.csv": METRIC_FIELDS | {"domain", "model", "seed"},
    "fold_metrics.csv": METRIC_FIELDS | {"domain", "fold", "model"},
    "calibration.csv": {"domain", "model", "bin", "lower", "upper", "women", "mean_prediction", "observed_prevalence"},
    "paired_comparisons.csv": {"domain", "model", "reference", "metric", "difference", "lower", "upper", "draws", "primary", "interval_scope"},
    "bootstrap_draws.csv.gz": {"domain", "model", "reference", "metric", "draw", "difference"},
    "class_counts.csv": {"domain", "women", "positive", "negative", "PSUs", "strata", "any_attitude", "actual_individual_years", "actual_health_years"},
    "fold_inventory.csv": {"fold", "train_women", "test_women", "train_positive", "test_positive", "train_PSUs", "test_PSUs", "shared_PSUs", "demo_dimensions", "attitude_dimensions", "neighbours", "self_or_ownPSU_donors", "heldout_donors"},
    "REPORT_BINDINGS.csv": {"source", "selectors", "field", "value", "rendered"},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, context):
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise AssertionError((context, actual, expected))
        return
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)) or not math.isclose(float(actual), float(expected), rel_tol=TOL, abs_tol=TOL):
        raise AssertionError((context, actual, expected))


def num(value):
    return None if value in (None, "", "nan", "NaN") else float(value)


def integer(value, context):
    answer = num(value)
    if answer is None or answer < 0 or int(answer) != answer:
        raise AssertionError((context, value))
    return int(answer)


def read_csv(path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert set(reader.fieldnames) == SCHEMAS[path.name], ("unexpected public schema", path.name, reader.fieldnames)
        return list(reader)


def keyed(rows, names):
    answer = {}
    for row in rows:
        key = tuple(row[name] for name in names)
        if key in answer:
            raise AssertionError(("duplicate aggregate key", names, key))
        answer[key] = row
    return answer


def metric_checks(row, threshold):
    label = tuple(row.get(k) for k in ("domain", "fold", "model", "seed"))
    values = {name: integer(row[name], (label, name)) for name in INT_FIELDS}
    tp, fp, tn, fn, n, positive = (values[name] for name in INT_FIELDS)
    assert n > 0 and tp + fp + tn + fn == n
    assert tp + fn == positive and tn + fp == n - positive
    sensitivity = tp / positive if positive else None
    specificity = tn / (n - positive) if n > positive else None
    checks = dict(accuracy=(tp + tn) / n, sensitivity=sensitivity, specificity=specificity,
        balanced_accuracy=(sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None,
        PPV=tp / (tp + fp) if tp + fp else None, prevalence=positive / n, threshold=threshold)
    for name, expected in checks.items():
        close(num(row[name]), expected, (label, name))
    for name in ("ROC_AUC", "average_precision", "mean_prediction", "Brier"):
        value = num(row[name])
        assert value is not None and 0 <= value <= 1, (label, name, value)
    assert num(row["log_loss"]) >= 0
    # Jensen's lower bound is recoverable from aggregate means, not exact loss.
    assert num(row["Brier"]) + TOL >= (num(row["mean_prediction"]) - positive / n) ** 2
    return values


def pairs():
    result = [(a + "_A", a + "_D") for a in ALGORITHMS]
    result += [("real_" + b, ref + "_" + b) for b in ("D", "A") for ref in ("logistic", "self", "random")]
    result.append(("boosting_A", "logistic_A"))
    result += [(a + "_" + b, "prevalence") for a in ALGORITHMS for b in ("D", "A")]
    assert len(result) == len(set(result)) == 22
    return result


def quantile(sorted_values, proportion):
    position = (len(sorted_values) - 1) * proportion
    lo = int(math.floor(position)); hi = int(math.ceil(position))
    return sorted_values[lo] + (position - lo) * (sorted_values[hi] - sorted_values[lo])


def replay():
    cfg = json.loads((BASE / "CONFIG.json").read_text())
    assert cfg["folds"] == 5 and cfg["bootstrap_draws"] == 2000
    assert tuple(cfg["neural_arms"]) == ("real", "self", "random")
    assert cfg["initializations"] == [11, 29, 47]
    assert cfg["threshold"] == 0.5
    names = ("metrics.csv", "initialization_metrics.csv", "fold_metrics.csv", "calibration.csv",
        "paired_comparisons.csv", "bootstrap_draws.csv.gz", "class_counts.csv", "fold_inventory.csv")
    for name in names:
        if not (BASE / "results" / name).is_file():
            raise FileNotFoundError(BASE / "results" / name)
    rows = {name: read_csv(BASE / "results" / name) for name in names if not name.endswith(".gz")}
    expected_counts = {"metrics.csv": 33, "initialization_metrics.csv": 54, "fold_metrics.csv": 55,
        "calibration.csv": 330, "paired_comparisons.csv": 132, "class_counts.csv": 4, "fold_inventory.csv": 6}
    for name, count in expected_counts.items():
        assert len(rows[name]) == count, (name, len(rows[name]), count)

    metrics = keyed(rows["metrics.csv"], ("domain", "model"))
    assert set(metrics) == {(domain, model) for domain in DOMAINS for model in MODELS}
    counts = keyed(rows["class_counts.csv"], ("domain",))
    assert set(counts) == {(name,) for name in ("primary", "strict_training_2023", "all_2024", "new_psu_2024")}
    expected_n = dict(primary=10267, strict_training_2023=10259, all_2024=9847, new_psu_2024=3250)
    for (domain,), row in counts.items():
        n = integer(row["women"], domain)
        assert n == expected_n[domain]
        assert integer(row["positive"], domain) + integer(row["negative"], domain) == n
        assert 0 < integer(row["PSUs"], domain) <= n
        assert 0 < integer(row["strata"], domain) <= integer(row["PSUs"], domain)
        assert 0 <= integer(row["any_attitude"], domain) <= n
    for field in ("women", "positive", "negative", "PSUs", "any_attitude"):
        assert integer(counts[("new_psu_2024",)][field], field) <= integer(counts[("all_2024",)][field], field)

    for (domain, model), row in metrics.items():
        metric_checks(row, cfg["threshold"])
        assert integer(row["women"], domain) == expected_n[domain]
        assert integer(row["positives"], domain) == integer(counts[(domain,)]["positive"], domain)

    inventory = keyed(rows["fold_inventory.csv"], ("fold",))
    primary_folds = tuple("primary_" + str(i) for i in range(cfg["folds"]))
    assert set(inventory) == {(name,) for name in primary_folds + ("temporal",)}
    for (fold,), row in inventory.items():
        assert integer(row["neighbours"], fold) == cfg["neighbours"]
        assert integer(row["heldout_donors"], fold) == integer(row["self_or_ownPSU_donors"], fold) == 0
        assert integer(row["attitude_dimensions"], fold) == integer(row["demo_dimensions"], fold) + len(cfg["attitudes"])
        if fold.startswith("primary"):
            assert integer(row["shared_PSUs"], fold) == 0
            for suffix, class_field in (("women", "women"), ("positive", "positive"), ("PSUs", "PSUs")):
                assert integer(row["train_" + suffix], fold) + integer(row["test_" + suffix], fold) == integer(counts[("primary",)][class_field], fold)
    for suffix, class_field in (("women", "women"), ("positive", "positive"), ("PSUs", "PSUs")):
        assert sum(integer(inventory[(fold,)]["test_" + suffix], fold) for fold in primary_folds) == integer(counts[("primary",)][class_field], suffix)
        assert integer(inventory[("temporal",)]["train_" + suffix], suffix) == integer(counts[("strict_training_2023",)][class_field], suffix)
        assert integer(inventory[("temporal",)]["test_" + suffix], suffix) == integer(counts[("all_2024",)][class_field], suffix)

    folds = keyed(rows["fold_metrics.csv"], ("domain", "fold", "model"))
    assert set(folds) == {("primary", fold, model) for fold in primary_folds for model in MODELS}
    for (domain, fold, model), row in folds.items():
        metric_checks(row, cfg["threshold"])
        assert integer(row["women"], fold) == integer(inventory[(fold,)]["test_women"], fold)
        assert integer(row["positives"], fold) == integer(inventory[(fold,)]["test_positive"], fold)
    for model in MODELS:
        groups = [folds[("primary", fold, model)] for fold in primary_folds]
        full = metrics[("primary", model)]
        for name in INT_FIELDS:
            assert sum(integer(row[name], name) for row in groups) == integer(full[name], name)
        for name in ("Brier", "log_loss", "mean_prediction", "prevalence", "accuracy"):
            pooled = sum(num(row[name]) * integer(row["women"], name) for row in groups) / expected_n["primary"]
            close(pooled, num(full[name]), (model, "pooled", name))
    # Prevalence predictions are constant within each primary fold and temporal domain.
    prevalence_sets = [(folds[("primary", fold, "prevalence")],
        integer(inventory[(fold,)]["train_positive"], fold) / integer(inventory[(fold,)]["train_women"], fold)) for fold in primary_folds]
    training_rate = integer(counts[("strict_training_2023",)]["positive"], "training") / expected_n["strict_training_2023"]
    prevalence_sets += [(metrics[(domain, "prevalence")], training_rate) for domain in ("all_2024", "new_psu_2024")]
    for row, probability in prevalence_sets:
        actual_rate = num(row["prevalence"])
        close(num(row["mean_prediction"]), probability, "constant prevalence mean")
        close(num(row["Brier"]), probability ** 2 - 2 * probability * actual_rate + actual_rate, "constant prevalence Brier")
        close(num(row["log_loss"]), -actual_rate * math.log(probability) - (1 - actual_rate) * math.log1p(-probability), "constant prevalence log loss")

    initialization = keyed(rows["initialization_metrics.csv"], ("domain", "model", "seed"))
    neural_models = tuple(arm + "_" + block for arm in cfg["neural_arms"] for block in ("D", "A"))
    assert set(initialization) == {(domain, model, str(seed)) for domain in DOMAINS for model in neural_models for seed in cfg["initializations"]}
    ensemble_bounds = []
    for (domain, model, seed), row in initialization.items():
        metric_checks(row, cfg["threshold"])
        for name in ("women", "positives"):
            assert integer(row[name], name) == integer(metrics[(domain, model)][name], name)
    for domain in DOMAINS:
        for model in neural_models:
            members = [initialization[(domain, model, str(seed))] for seed in cfg["initializations"]]
            ensemble = metrics[(domain, model)]
            close(num(ensemble["mean_prediction"]), sum(num(row["mean_prediction"]) for row in members) / len(members), (domain, model, "ensemble probability mean"))
            for loss in LOSSES:
                seed_average = sum(num(row[loss]) for row in members) / len(members)
                assert num(ensemble[loss]) <= seed_average + TOL, (domain, model, loss, "ensemble convexity bound")
                ensemble_bounds.append({"domain": domain, "model": model, "loss": loss, "ensemble_loss": num(ensemble[loss]), "mean_initialization_loss": seed_average})

    calibration = keyed(rows["calibration.csv"], ("domain", "model", "bin"))
    assert set(calibration) == {(domain, model, str(binno)) for domain in DOMAINS for model in MODELS for binno in range(10)}
    for domain in DOMAINS:
        for model in MODELS:
            n = predicted_sum = observed_sum = lower_bound = 0.0
            confusion = dict(TP=0, FP=0, TN=0, FN=0)
            for binno in range(10):
                row = calibration[(domain, model, str(binno))]
                close(num(row["lower"]), binno / 10, "calibration lower")
                close(num(row["upper"]), (binno + 1) / 10, "calibration upper")
                count = integer(row["women"], "calibration women")
                prediction, observed = num(row["mean_prediction"]), num(row["observed_prevalence"])
                if not count:
                    assert prediction is None and observed is None
                    continue
                assert binno / 10 - TOL <= prediction <= (binno + 1) / 10 + TOL
                assert 0 <= observed <= 1
                positive = round(count * observed)
                close(count * observed, positive, "bin positive count")
                n += count; predicted_sum += count * prediction; observed_sum += positive
                lower_bound += count * (prediction - observed) ** 2
                if binno >= 5:
                    confusion["TP"] += positive; confusion["FP"] += count - positive
                else:
                    confusion["FN"] += positive; confusion["TN"] += count - positive
            full = metrics[(domain, model)]
            assert n == integer(full["women"], "calibration total")
            assert observed_sum == integer(full["positives"], "calibration positives")
            close(predicted_sum / n, num(full["mean_prediction"]), "calibration weighted prediction")
            close(observed_sum / n, num(full["prevalence"]), "calibration weighted outcome")
            assert lower_bound / n <= num(full["Brier"]) + TOL
            for key, value in confusion.items():
                assert value == integer(full[key], (domain, model, key))

    contrasts = keyed(rows["paired_comparisons.csv"], ("domain", "model", "reference", "metric"))
    expected_pairs = {(domain, model, ref, loss) for domain in DOMAINS for model, ref in pairs() for loss in LOSSES}
    assert set(contrasts) == expected_pairs
    boot = {key: [None] * cfg["bootstrap_draws"] for key in expected_pairs}
    draw_rows = 0
    with gzip.open(BASE / "results/bootstrap_draws.csv.gz", "rt", newline="") as f:
        reader = csv.DictReader(f)
        assert set(reader.fieldnames) == SCHEMAS["bootstrap_draws.csv.gz"]
        for row in reader:
            key = tuple(row[name] for name in ("domain", "model", "reference", "metric"))
            assert key in boot
            ordinal = integer(row["draw"], "bootstrap draw")
            assert ordinal < cfg["bootstrap_draws"] and boot[key][ordinal] is None
            value = num(row["difference"])
            assert value is not None and math.isfinite(value)
            if key[-1] == "Brier":
                assert -1 <= value <= 1
            boot[key][ordinal] = value
            draw_rows += 1
    assert draw_rows == len(expected_pairs) * cfg["bootstrap_draws"] == 264000
    primary_pairs = {tuple(cfg["primary_attitude_contrast"])} | {tuple(pair) for pair in cfg["primary_graph_contrasts"]}
    primary_brier = []
    for key, values in boot.items():
        assert all(value is not None for value in values), key
        domain, model, ref, loss = key
        observed = num(metrics[(domain, model)][loss]) - num(metrics[(domain, ref)][loss])
        row = contrasts[key]
        close(num(row["difference"]), observed, (key, "observed contrast"))
        ordered = sorted(values)
        close(num(row["lower"]), quantile(ordered, 0.025), (key, "percentile lower"))
        close(num(row["upper"]), quantile(ordered, 0.975), (key, "percentile upper"))
        assert integer(row["draws"], key) == cfg["bootstrap_draws"]
        expected_primary = (model, ref) in primary_pairs
        assert row["primary"].lower() == str(expected_primary).lower()
        assert row["interval_scope"] == "paired observed-PSU bootstrap; fixed predictions/training/graph; not population or full-refit inference"
        if expected_primary and loss == "Brier":
            primary_brier.append(dict(domain=domain, model=model, reference=ref,
                difference=observed, lower=num(row["lower"]), upper=num(row["upper"])))

    source_files = {"CONFIG.json": sha(BASE / "CONFIG.json")}
    source_files.update({"results/" + name: sha(BASE / "results" / name) for name in names})
    binding_count = 0
    bindings_path = BASE / "REPORT_BINDINGS.csv"
    if bindings_path.is_file():
        prose_path = BASE / "RESULTS.md"
        prose = prose_path.read_text()
        for binding in read_csv(bindings_path):
            source_name = binding["source"]
            assert source_name.startswith("results/") and source_name[len("results/"):] in rows
            selectors = json.loads(binding["selectors"])
            chosen = [row for row in rows[source_name[len("results/"):]] if all(row[key] == str(value) for key, value in selectors.items())]
            assert len(chosen) == 1, ("report binding source", binding)
            value = num(chosen[0][binding["field"]])
            close(num(binding["value"]), value, ("report value", binding))
            rendered = binding["rendered"]
            expected = f"{value:.{len(rendered.partition('.')[2])}f}" if "." in rendered else str(int(value))
            assert expected == rendered and rendered in prose, ("report rendering", binding)
            binding_count += 1
        assert binding_count > 0
        source_files.update({"REPORT_BINDINGS.csv": sha(bindings_path), "RESULTS.md": sha(prose_path)})
    return {"status": "passed", "public_inputs_sha256": source_files,
        "verified_rows": dict(metrics=33, initialization_metrics=54, fold_metrics=55, calibration=330,
            paired_comparisons=132, bootstrap_draws=264000, class_counts=4, fold_inventory=6),
        "checks": dict(unique_expected_keys=True, metric_confusion_arithmetic=True,
            exact_public_table_schemas=True,
            primary_losses_pooled_by_women=True, fold_count_margins=True,
            constant_prevalence_losses=True, calibration_weighted_means=True,
            calibration_confusion_at_fixed_threshold=True, initialization_probability_mean=True,
            initialization_loss_convexity_only=True, observed_pair_differences=True,
            paired_percentile_intervals_from_all_public_draws=True, config_primary_flags=True),
        "report_numeric_bindings_verified": binding_count,
        "report_binding_scope": "Manifest-selected public table values, decimal formatting and presence in RESULTS.md only; not a complete prose, semantic or figure-content audit." if binding_count else "No report bindings were present; no prose-binding audit claimed.",
        "primary_Brier_contrasts": sorted(primary_brier, key=lambda row: (row["domain"], row["model"], row["reference"])),
        "ensemble_convexity_bounds": sorted(ensemble_bounds, key=lambda row: (row["domain"], row["model"], row["loss"])),
        "verification_limits": [
            "Public aggregate arithmetic only; no independent respondent-source preparation, cohort identities, temporal overlap or leakage audit.",
            "No model optimization, prediction/state replay, graph construction or neighbour-identity verification.",
            "Fold Brier/log loss and probability means are pooled by participant counts; AUC and average precision are not recovered from fold averages.",
            "Exact ensemble losses and rankings cannot be reconstructed from seed-level aggregate losses; mean probability and convexity bounds alone are checked.",
            "Calibration bins recover observed/predicted marginal means and fixed-threshold confusion; exact probability-level losses and ranking metrics are not recovered.",
            "Percentile intervals are recomputed from stored public bootstrap contrasts; original PSU resampling and individual loss contributions are not reconstructed.",
            "Fixed-prediction intervals do not include model refitting, graph selection, national sampling or prospective uncertainty.",
            "The new-PSU 2024 evaluation is a subset of all-2024 evaluation, not an independent cohort; no within-person follow-up or future-risk validation is established.",
            "No manuscript prose audit is claimed; report binding scope is limited to the explicit manifest-selected values and renderings."
        ], "reads_individual_data": False, "imports_producer_or_core": False,
        "source_preparation_and_model_fitting_replayed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    args = parser.parse_args()
    result = replay()
    (BASE / "AGGREGATE_REPLAY.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "verified_rows": result["verified_rows"], "source_preparation_and_model_fitting_replayed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
