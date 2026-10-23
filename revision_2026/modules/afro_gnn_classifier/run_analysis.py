"""Bounded retrospective country-held-out Afrobarometer graph classification.

Only new owned paths are written. Individual matrices, memberships, predictions
and fitted states stay private in data_local/. No raw data are downloaded.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = Path(__file__).resolve().parent
REV = BASE.parents[1]
REPO = REV.parent
SOURCE = BASE.parent / "individual_replication_2026_10_22"
LOCAL = BASE / "data_local"
RESULTS = BASE / "results"
FROZEN_PRODUCERS = ["PROTOCOL.md", "CONFIG.json", "core.py", "run_analysis.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ah(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def js(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cfg():
    return js(BASE / "CONFIG.json")


def cohort():
    path = SOURCE / "data_local/cohort.csv"
    assert sha(path) == cfg()["source_cohort_sha256"]
    frame = pd.read_csv(path)
    assert len(frame) == 26602 and frame.iso3.nunique() == 39
    assert np.array_equal(frame.row_key, np.arange(len(frame)))
    assert (frame.Q100 == 2).all() and frame.Q52B.isin([1, 2, 3]).all()
    assert np.array_equal(frame.y, frame.Q52B.isin([1, 2]).astype(int))
    assert frame.withinwt_hh.gt(0).all() and frame.RESPNO.is_unique
    return frame


def weights(frame):
    w = frame.withinwt_hh.to_numpy(float)
    totals = frame.groupby("iso3").withinwt_hh.transform("sum").to_numpy(float)
    w = w / totals / frame.iso3.nunique()
    assert np.isclose(w.sum(), 1)
    return w


def transform(frame, train):
    config = cfg()
    w = weights(frame.iloc[train])
    numeric = config["numeric_variables"]
    values = frame[numeric].to_numpy(float)
    missing = ~np.isfinite(values)
    z = values.copy()
    means, scales, no_data = [], [], []
    for j in range(len(numeric)):
        available = ~missing[train, j]
        empty = not available.any()
        mean = 0. if empty else float(np.average(values[train[available], j], weights=w[available]))
        z[missing[:, j], j] = mean
        scale = float(np.sqrt(w @ ((z[train, j] - mean) ** 2)))
        if scale < 1e-12:
            scale = 1.
        z[:, j] = (z[:, j] - mean) / scale
        means.append(mean); scales.append(scale); no_data.append(empty)
    arrays, names, graph_arrays, graph_names = [], [], [], []
    for variable in config["categorical_variables"]:
        legal = config["categorical_valid_codes"][variable]
        values = frame[variable].to_numpy(float)
        invalid = ~np.isin(values, legal)
        for code in legal[1:]:
            arrays.append((values == code).astype(float))
            names.append(variable + "_code_" + str(code))
        arrays.append(invalid.astype(float)); names.append(variable + "_missing")
        for code in legal:
            graph_arrays.append((values == code).astype(float) / np.sqrt(2.))
            graph_names.append(variable + "_full_code_" + str(code))
        graph_arrays.append(invalid.astype(float) / np.sqrt(2.)); graph_names.append(variable + "_full_missing")
    x = np.column_stack([z, missing.astype(float), *arrays])
    graph_x = np.column_stack([z, missing.astype(float), *graph_arrays])
    info = {"numeric": numeric, "means": means, "scales": scales, "all_training_missing": no_data,
            "columns": numeric + [v + "_missing" for v in numeric] + names,
            "graph_columns": numeric + [v + "_missing" for v in numeric] + graph_names,
            "categorical_variables": config["categorical_variables"],
            "vocabulary_source": "completed source-study CONFIG native codebook categories; no target-derived vocabulary",
            "training_weight_sha256": ah(w), "training_rows_sha256": ah(np.asarray(train, dtype=np.int64))}
    assert np.isfinite(x).all() and np.isfinite(graph_x).all()
    return x, graph_x, info


def tie_key(source_row):
    return hashlib.sha256((str(cfg()["tie_seed"]) + ":" + str(int(source_row))).encode()).hexdigest()


def neighbours(query, donors, donor_rows, self_local=None):
    config = cfg()
    order = np.array(sorted(range(len(donor_rows)), key=lambda i: tie_key(donor_rows[i])), dtype=np.int64)
    reference = donors[order]
    result = np.empty((len(query), config["neighbours"]), dtype=np.int32)
    norms = (reference * reference).sum(axis=1)
    for first in range(0, len(query), config["distance_block_rows"]):
        block = query[first:first + config["distance_block_rows"]]
        d = (block * block).sum(axis=1)[:, None] + norms[None, :] - 2 * (block @ reference.T)
        np.maximum(d, 0, out=d)
        np.round(d, config["squared_distance_decimal_places"], out=d)
        if self_local is not None:
            inverse = np.argsort(order)
            rows = np.arange(len(block))
            d[rows, inverse[self_local[first:first + len(block)]]] = np.inf
        choice = np.argsort(d, axis=1, kind="stable")[:, :config["neighbours"]]
        result[first:first + len(block)] = order[choice]
    return result


def random_neighbours(size, count, generator, self_local=None):
    k = cfg()["neighbours"]
    out = np.empty((size, k), dtype=np.int32)
    for j in range(size):
        if self_local is None:
            out[j] = generator.choice(count, k, replace=False)
        else:
            # Uniform subset from n-1 donors, remapped around excluded own index.
            chosen = generator.choice(count - 1, k, replace=False)
            out[j] = chosen + (chosen >= self_local[j])
    return out


def prepare():
    assert not (BASE / "FREEZE.json").exists(), "Prepare before final freeze"
    assert not (LOCAL / "PREPARED.json").exists(), "Preserve prepared inputs"
    frame = cohort()
    for folder in [LOCAL / "folds", LOCAL / "states", RESULTS]:
        folder.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(cfg()["cap_seed"])
    selections = []
    for iso, group in frame.groupby("iso3", sort=True):
        selected = generator.choice(group.index.to_numpy(), cfg()["donors_per_country"], replace=False)
        selections.extend({"iso3": iso, "row_key": int(row)} for row in sorted(selected))
    caps = pd.DataFrame(selections)
    caps.to_csv(LOCAL / "donor_selection.csv", index=False)
    inventories = []
    started = time.monotonic()
    for fold, iso in enumerate(sorted(frame.iso3.unique())):
        train = np.sort(caps.loc[caps.iso3.ne(iso), "row_key"].to_numpy(np.int64))
        test = np.flatnonzero(frame.iso3.eq(iso)).astype(np.int64)
        assert len(train) == 3800 and not len(np.intersect1d(train, test))
        assert frame.iloc[train].iso3.nunique() == 38 and frame.iloc[test].iso3.nunique() == 1
        x, graph_x, info = transform(frame, train)
        real_train = neighbours(graph_x[train], graph_x[train], train, np.arange(len(train)))
        real_test = neighbours(graph_x[test], graph_x[train], train)
        rng = np.random.default_rng(cfg()["random_graph_seed"] + fold)
        random_train = random_neighbours(len(train), len(train), rng, np.arange(len(train)))
        random_test = random_neighbours(len(test), len(train), rng)
        assert not (real_train == np.arange(len(train))[:, None]).any()
        assert not (random_train == np.arange(len(train))[:, None]).any()
        w = weights(frame.iloc[train])
        wt = frame.iloc[test].withinwt_hh.to_numpy(float); wt /= wt.sum()
        np.savez_compressed(LOCAL / "folds" / (iso + ".npz"), train_rows=train, test_rows=test,
            X_train=x[train], X_test=x[test], weight_train=w, weight_test=wt,
            y_train=frame.y.to_numpy(float)[train], y_test=frame.y.to_numpy(float)[test],
            real_train=real_train, real_test=real_test, random_train=random_train, random_test=random_test)
        info.update({"held_out_iso3": iso, "fold": fold, "train_countries": sorted(frame.iloc[train].iso3.unique()),
                     "train_design_sha256": ah(x[train]), "test_design_sha256": ah(x[test]),
                     "graph_train_design_sha256": ah(graph_x[train]), "graph_test_design_sha256": ah(graph_x[test]),
                     "neighbours_real_train_sha256": ah(real_train), "neighbours_real_test_sha256": ah(real_test),
                     "neighbours_random_train_sha256": ah(random_train), "neighbours_random_test_sha256": ah(random_test)})
        dump(LOCAL / "folds" / (iso + ".json"), info)
        inventories.append({"iso3": iso, "fold": fold, "train_women": len(train), "test_women": len(test),
            "donor_countries": 38, "donors_each": 100, "features": x.shape[1], "graph_features": graph_x.shape[1],
            "k": cfg()["neighbours"], "train_self_edges": 0, "held_out_training_nodes": 0,
            "allocated_graph_parameters": 16 * x.shape[1] + 17, "allocated_mlp_parameters": 16 * x.shape[1] + 33})
        print(iso, "predictor-only fold prepared", fold + 1, "of39", flush=True)
    pd.DataFrame(inventories).to_csv(RESULTS / "fold_inventory.csv", index=False)
    # Existing source summaries are checked references, not neural-model results.
    frame.groupby("iso3").agg(n=("y", "size"), positive=("y", "sum"), country=("country", "first")).reset_index().to_csv(RESULTS / "class_summary.csv", index=False)
    paths = [LOCAL / "donor_selection.csv"] + sorted((LOCAL / "folds").glob("*"))
    dump(LOCAL / "PREPARED.json", {"status": "predictor_only_preparation_before_new_optimization", "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {str(p.relative_to(BASE)): sha(p) for p in paths}, "source_cohort_sha256": cfg()["source_cohort_sha256"],
        "source_outcomes_previously_examined": True, "new_classifier_fits": 0,
        "elapsed_seconds": time.monotonic() - started})


def freeze():
    assert not (BASE / "FREEZE.json").exists(), "Never overwrite a design freeze"
    prepared = js(LOCAL / "PREPARED.json")
    for name, expected in prepared["files"].items():
        assert sha(BASE / name) == expected
    source_paths = [SOURCE / "CONFIG.json", SOURCE / "ANALYTIC_INPUT_FREEZE.json", SOURCE / "results/INDEPENDENT_QC.json",
                    SOURCE / "data_local/cohort.csv", SOURCE / "data_local/selected_source.csv", SOURCE / "data_local/source_column_metadata.json"]
    protected = js(REV / "07_manuscript/scientific_reports_2026_10_22/BASE_PRE_CLASSIFIER_MANUSCRIPT.json")["files"]
    for name, expected in protected.items():
        assert sha(REPO / name) == expected
    dump(BASE / "FREEZE.json", {"status": "frozen_before_any_new_classifier_optimization", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer_files": {name: sha(BASE / name) for name in FROZEN_PRODUCERS},
        "source_files": {str(p.relative_to(REPO)): sha(p) for p in source_paths},
        "prepared_manifest_sha256": sha(LOCAL / "PREPARED.json"), "protected_manuscript_files": protected,
        "earlier_outcomes_and_logistic_performance_seen": True, "new_neural_or_capped_logistic_fits": 0})
    print("Producer, protocol, fixed source, capped donors, graphs and protected manuscript frozen")


def verify():
    frozen = js(BASE / "FREEZE.json")
    for name, expected in frozen["producer_files"].items():
        assert sha(BASE / name) == expected, ("Producer changed", name)
    for name, expected in frozen["source_files"].items():
        assert sha(REPO / name) == expected, ("Source changed", name)
    for name, expected in frozen["protected_manuscript_files"].items():
        assert sha(REPO / name) == expected, ("Protected manuscript changed", name)
    assert sha(LOCAL / "PREPARED.json") == frozen["prepared_manifest_sha256"]
    for name, expected in js(LOCAL / "PREPARED.json")["files"].items():
        assert sha(BASE / name) == expected, ("Prepared input changed", name)


def logistic_objective(theta, x, y, w, penalty):
    eta = theta[0] + x @ theta[1:]
    value = w @ (np.logaddexp(0., eta) - y * eta) + penalty / 2 * (theta[1:] @ theta[1:])
    residual = w * (expit(eta) - y)
    gradient = np.r_[residual.sum(), x.T @ residual + penalty * theta[1:]]
    return float(value), gradient


def scores(y, p, weight):
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    weight = weight / weight.sum()
    q = np.clip(p, 1e-15, 1 - 1e-15)
    prediction = p >= cfg()["decision_threshold"]
    positive, negative = y == 1, y == 0
    tp = int((prediction & positive).sum()); tn = int((~prediction & negative).sum())
    fp = int((prediction & negative).sum()); fn = int((~prediction & positive).sum())
    sensitivity = float(weight[prediction & positive].sum() / weight[positive].sum())
    specificity = float(weight[~prediction & negative].sum() / weight[negative].sum())
    return {"n": len(y), "positive": int(positive.sum()), "negative": int(negative.sum()),
        "brier": float(weight @ ((p - y) ** 2)),
        "log_loss": float(weight @ (-y * np.log(q) - (1 - y) * np.log1p(-q))),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=weight)),
        "average_precision": float(average_precision_score(y, p, sample_weight=weight)),
        "observed_endorsement": float(weight @ y), "predicted_endorsement": float(weight @ p),
        "mean_prediction_bias": float(weight @ (p - y)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "weighted_sensitivity": sensitivity, "weighted_specificity": specificity,
        "weighted_accuracy": float(weight @ (prediction == positive)),
        "weighted_balanced_accuracy": (sensitivity + specificity) / 2,
        "raw_sensitivity": tp / (tp + fn), "raw_specificity": tn / (tn + fp),
        "raw_accuracy": (tp + tn) / len(y), "raw_balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
        "threshold": cfg()["decision_threshold"]}


def fit():
    verify()
    from core import fit_classifier
    assert not (RESULTS / "FIT_COMPLETION.json").exists(), "Completed fit is immutable"
    frame, config = cohort(), cfg()
    completed = []
    for iso in sorted(frame.iso3.unique()):
        folder = LOCAL / "predictions"; folder.mkdir(exist_ok=True)
        completion = folder / (iso + ".json")
        if completion.exists():
            entry = js(completion)
            for name, expected in entry["artifacts_sha256"].items():
                assert sha(BASE / name) == expected
            completed.append(entry)
            print(iso, "completed fold preserved", flush=True)
            continue
        f = np.load(LOCAL / "folds" / (iso + ".npz"))
        x, xt, y, w = f["X_train"], f["X_test"], f["y_train"], f["weight_train"]
        prior = float(w @ y)
        prediction = {"P_cap": np.full(len(xt), prior)}
        initial = np.zeros(x.shape[1] + 1); initial[0] = np.log(prior / (1 - prior))
        result = minimize(logistic_objective, initial, args=(x, y, w, config["penalty_lambda"]), jac=True,
            method="L-BFGS-B", options={"maxiter": config["logistic_maxiter"], "ftol": config["logistic_ftol"], "gtol": config["logistic_gtol"]})
        value, gradient = logistic_objective(result.x, x, y, w, config["penalty_lambda"])
        assert result.success and abs(gradient).max() < config["logistic_max_gradient"]
        prediction["D1_cap"] = expit(result.x[0] + xt @ result.x[1:])
        dump(LOCAL / "states" / (iso + "_D1_cap.json"), {"held_out_iso3": iso, "model": "D1_cap", "parameters": result.x.tolist(),
            "objective": value, "gradient_infinity_norm": float(abs(gradient).max()), "iterations": int(result.nit),
            "solver_message": str(result.message), "training_prevalence": prior, "fold_npz_sha256": sha(LOCAL / "folds" / (iso + ".npz"))})
        initialization_rows = []
        for arm in config["neural_arms"]:
            if arm in ("MLP", "G_self"):
                mean, mean_test = x, xt
            else:
                key = "real" if arm == "G_real" else "random"
                mean = x[f[key + "_train"]].mean(axis=1)
                mean_test = x[f[key + "_test"]].mean(axis=1)
            restarted = []
            for seed in config["initializations"]:
                stem = iso + "_" + arm + "_" + str(seed)
                state_path, diagnostic_path = LOCAL / "states" / (stem + ".npz"), LOCAL / "states" / (stem + ".json")
                assert not state_path.exists() and not diagnostic_path.exists(), "Do not overwrite partial states; investigate a stopped fold"
                train_logit, test_logit, state, diagnostic = fit_classifier(x, mean, y, w, xt, mean_test, arm, seed, config)
                np.savez_compressed(state_path, **state, train_logit=train_logit, test_logit=test_logit)
                diagnostic.update({"held_out_iso3": iso, "state_sha256": sha(state_path), "fold_npz_sha256": sha(LOCAL / "folds" / (iso + ".npz"))})
                dump(diagnostic_path, diagnostic)
                p = expit(test_logit); restarted.append(p)
                initialization_rows.append({"iso3": iso, "model": arm, "seed": seed, **scores(f["y_test"], p, f["weight_test"]),
                    "final_objective": diagnostic["trace"][-1]["total"], "final_bce": diagnostic["trace"][-1]["bce"],
                    "final_penalty": diagnostic["trace"][-1]["penalty"], "allocated_parameters": diagnostic["allocated_parameters"]})
            prediction[arm] = np.mean(restarted, axis=0)
        out = pd.DataFrame({"row_key": f["test_rows"], "y": f["y_test"], "weight_test": f["weight_test"]})
        for model in config["models"]:
            out[model] = prediction[model]
        out.to_csv(folder / (iso + ".csv"), index=False)
        pd.DataFrame(initialization_rows).to_csv(folder / (iso + "_initializations.csv"), index=False)
        artifacts = [folder / (iso + ".csv"), folder / (iso + "_initializations.csv")] + sorted((LOCAL / "states").glob(iso + "_*"))
        entry = {"status": "fold_fits_completed", "iso3": iso, "neural_fits": 12, "logistic_fits": 1,
            "held_out_women": len(xt), "artifacts_sha256": {str(p.relative_to(BASE)): sha(p) for p in artifacts}}
        dump(completion, entry); completed.append(entry)
        print(iso, "completed", len(completed), "of39 folds;", len(completed) * 12, "neural fits", flush=True)
    verify()
    dump(RESULTS / "FIT_COMPLETION.json", {"status": "all_fixed_fits_complete_pending_independent_QC", "countries": len(completed),
        "women": len(frame), "neural_fits": sum(x["neural_fits"] for x in completed), "logistic_fits": len(completed),
        "manuscript_integrated": False, "no_new_downloads": True})


def summarize():
    verify()
    assert js(RESULTS / "FIT_COMPLETION.json")["neural_fits"] == 468
    assert not (RESULTS / "PRODUCER_COMPLETION.json").exists(), "Never overwrite final aggregate tables"
    frame, config = cohort(), cfg()
    country_metrics, all_predictions, restart_rows = [], [], []
    for iso in sorted(frame.iso3.unique()):
        p = pd.read_csv(LOCAL / "predictions" / (iso + ".csv"))
        p["iso3"] = iso; all_predictions.append(p)
        for model in config["models"]:
            country_metrics.append({"iso3": iso, "model": model, **scores(p.y.to_numpy(float), p[model].to_numpy(float), p.weight_test.to_numpy(float))})
        restart_rows.append(pd.read_csv(LOCAL / "predictions" / (iso + "_initializations.csv")))
    metric = pd.DataFrame(country_metrics)
    assert len(metric) == 234 and not metric.duplicated(["iso3", "model"]).any()
    metric.to_csv(RESULTS / "country_metrics.csv", index=False)
    pd.concat(all_predictions).sort_values("row_key").to_csv(LOCAL / "held_out_predictions.csv", index=False)
    pd.concat(restart_rows).to_csv(RESULTS / "initialization_metrics.csv", index=False)
    summaries = []
    mean_columns = ["brier", "log_loss", "roc_auc", "average_precision", "observed_endorsement", "predicted_endorsement",
                    "mean_prediction_bias", "weighted_sensitivity", "weighted_specificity", "weighted_accuracy", "weighted_balanced_accuracy",
                    "raw_sensitivity", "raw_specificity", "raw_accuracy", "raw_balanced_accuracy"]
    for model in config["models"]:
        part = metric.loc[metric.model.eq(model)]
        summaries.append({"model": model, "countries": len(part), "women": int(part.n.sum()), "positive": int(part.positive.sum()),
            **{column: float(part[column].mean()) for column in mean_columns},
            "mean_absolute_prediction_bias": float(part.mean_prediction_bias.abs().mean()),
            **{column: int(part[column].sum()) for column in ("tp", "tn", "fp", "fn")}})
    pd.DataFrame(summaries).to_csv(RESULTS / "model_summary.csv", index=False)
    countries = sorted(frame.iso3.unique())
    draws = np.random.default_rng(config["bootstrap_seed"]).integers(0, 39, (config["bootstrap_draws"], 39))
    np.save(LOCAL / "country_bootstrap_draws.npy", draws)
    differences, contrasts = [], []
    for first, second in config["contrasts"]:
        for column in ("brier", "log_loss"):
            wide = metric.pivot(index="iso3", columns="model", values=column).reindex(countries)
            delta = (wide[first] - wide[second]).to_numpy(float)
            interval = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])
            differences += [{"iso3": iso, "first": first, "second": second, "metric": column, "difference": float(v)} for iso, v in zip(countries, delta)]
            contrasts.append({"first": first, "second": second, "metric": column, "difference": float(delta.mean()), "lower": float(interval[0]), "upper": float(interval[1]),
                "countries": 39, "improved_countries": int((delta < 0).sum()), "worse_countries": int((delta > 0).sum()), "equal_countries": int((delta == 0).sum()),
                "uncertainty": "paired_country_bootstrap_conditional_on_fixed_fits_and_cap"})
    pd.DataFrame(differences).to_csv(RESULTS / "paired_country_differences.csv", index=False)
    pd.DataFrame(contrasts).to_csv(RESULTS / "paired_summary.csv", index=False)
    # Only fixed-grid aggregated operating rows; no per-person thresholds exported.
    curves = []
    for block in all_predictions:
        iso = block.iso3.iloc[0]
        y, w = block.y.to_numpy(float), block.weight_test.to_numpy(float)
        for model in config["models"]:
            probability = block[model].to_numpy(float)
            for threshold in np.linspace(0, 1, 101):
                yes = probability >= threshold
                tp = float(w[yes & (y == 1)].sum()); fp = float(w[yes & (y == 0)].sum())
                curves.append({"iso3": iso, "model": model, "threshold": float(threshold),
                    "tpr": tp / w[y == 1].sum(), "fpr": fp / w[y == 0].sum(),
                    "precision": tp / (tp + fp) if tp + fp > 0 else 1., "recall": tp / w[y == 1].sum()})
    pd.DataFrame(curves).to_csv(RESULTS / "curves.csv", index=False)
    import torch
    dump(BASE / "ENVIRONMENT.json", {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__, "threads": 1})
    dump(RESULTS / "PRODUCER_COMPLETION.json", {**js(RESULTS / "FIT_COMPLETION.json"), "status": "producer_complete_pending_independent_QC",
        "country_metric_rows": len(metric), "model_summary_rows": len(summaries), "comparison_rows": len(contrasts),
        "paired_country_rows": len(differences), "initialization_metric_rows": sum(len(x) for x in restart_rows), "fixed_grid_curve_rows": len(curves),
        "freeze_sha256": sha(BASE / "FREEZE.json"), "aggregate_sha256": {p.name: sha(p) for p in sorted(RESULTS.glob("*.csv"))},
        "no_held_out_training_nodes_or_labels": True, "protected_manuscript_unchanged": True, "individual_artifacts_publicly_exported": False})
    print("All aggregate tables completed; independent QC required", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "freeze", "fit", "summarize", "verify"])
    action = parser.parse_args().action
    try:
        globals()[action]()
    except Exception as error:
        dump(BASE / "READINESS_OR_FAILURE.json", {"status": "stopped_without_placeholder_results", "action": action,
            "error_type": type(error).__name__, "error": str(error), "affected_computation_complete": False})
        raise
