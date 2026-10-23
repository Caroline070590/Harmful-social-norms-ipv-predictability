"""Fixed exploratory shared-context ablation; see PROTOCOL.md saved before fits."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import platform
import shutil
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import solve
import sklearn
from sklearn.linear_model import Ridge

BASE = Path(__file__).resolve().parent
REV = BASE.parents[1]
ORIGINAL = BASE.parent / "joint_country_time"
INPUT = BASE / "inputs"
OUT = BASE / "results"
ITEMS = ["WE_AWBT_W_BFD", "WE_AWBT_W_ARG", "WE_AWBT_W_OUT", "WE_AWBT_W_NEG", "WE_AWBT_W_REF"]
FEATURES = ["calendar", "hdi", "electoral_democracy", "liberal_democracy", "women_political_empowerment"]
CASES = ["primary", "anomaly_removed", "common_2020", "questionnaire_compatible"]
ALPHA, SEED, NBOOT = 10.0, 20261020, 10000
METRICS = ["MAE_pp", "level_MAE_pp", "profile_MAE_pp", "MSE_pp2", "level_MSE_pp2", "profile_MSE_pp2"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def prepare():
    INPUT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    original = json.loads((ORIGINAL / "prepared/INPUT_MANIFEST.json").read_text())
    checks = {}
    for category in ["source_sha256", "raw_api_sha256"]:
        for rel, expected in original[category].items():
            actual = sha(REV / rel)
            assert actual == expected, rel
            checks[rel] = actual
    for name, expected in original["prepared_sha256"].items():
        assert sha(ORIGINAL / "prepared" / name) == expected, name
    files = {"rounds.csv": "prepared/rounds.csv",
             "national_source_cells.csv": "prepared/national_source_cells.csv",
             "structural_source_ledger.csv": "prepared/structural_source_ledger.csv",
             "population_inventory.csv": "prepared/population_inventory.csv",
             "original_predictions.csv": "results/predictions.csv",
             "original_country_metrics.csv": "results/country_metrics.csv",
             "original_training_membership.csv": "results/training_membership.csv",
             "original_INPUT_MANIFEST.json": "prepared/INPUT_MANIFEST.json",
             "original_INDEPENDENT_QC.json": "results/INDEPENDENT_QC.json"}
    for name, rel in files.items():
        shutil.copyfile(ORIGINAL / rel, INPUT / name)
    guard = {str(p.relative_to(REV)): sha(p) for p in ORIGINAL.rglob("*") if p.is_file()}
    save_json(INPUT / "SOURCE_MANIFEST.json", {
        "prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256": sha(BASE / "PROTOCOL.md"),
        "questionnaire_amendment_sha256": sha(BASE / "QUESTIONNAIRE_AMENDMENT.md"),
        "verified_original_source_sha256": checks,
        "input_sha256": {p.name: sha(p) for p in INPUT.iterdir() if p.is_file() and p.name != "SOURCE_MANIFEST.json"},
        "original_analysis_guard_sha256": guard,
        "historical_manuscript_hashes_not_rechecked": "Original INPUT_MANIFEST protected an earlier manuscript; it is preserved as historical metadata. This extension never writes manuscript files."})
    return pd.read_csv(INPUT / "rounds.csv")


def metrics(y, pred):
    error = pred - y
    level = float(error.mean())
    profile = pred - pred.mean() - y + y.mean()
    full_mse = float(np.mean(error ** 2))
    profile_mse = float(np.mean(profile ** 2))
    assert np.isclose(full_mse, level ** 2 + profile_mse, atol=1e-10)
    return dict(MAE_pp=float(np.abs(error).mean()), MSE_pp2=full_mse,
                RMSE_pp=float(np.sqrt(full_mse)), bias_pp=level,
                level_MAE_pp=abs(level), level_MSE_pp2=level ** 2,
                profile_MAE_pp=float(np.abs(profile).mean()), profile_MSE_pp2=profile_mse,
                observed_level_pp=float(y.mean()), predicted_level_pp=float(pred.mean()))


def fit():
    start = dt.datetime.now(dt.timezone.utc).isoformat()
    rounds = prepare()
    assert len(rounds) == 169 and rounds.country.nunique() == 62
    for col in ["fieldwork_start", "fieldwork_end"]:
        rounds[col] = pd.to_datetime(rounds[col], errors="raise")
    targets = rounds[rounds.is_target].sort_values("country")
    assert len(targets) == 42
    pred_rows, met_rows, membership, exclusions, params, missingness, identities = [], [], [], [], [], [], []
    for case in CASES:
        case_targets = targets if case != "common_2020" else targets[targets.fieldwork_start.ge("2020-01-01")]
        assert len(case_targets) == (20 if case == "common_2020" else 42)
        for target in case_targets.itertuples():
            reasons = {}
            for row in rounds.itertuples():
                r = []
                if row.country == target.country:
                    r.append("target_country")
                if row.fieldwork_end >= target.fieldwork_start:
                    r.append("not_strictly_earlier")
                if case == "anomaly_removed" and row.survey_id in {"NP2011DHS", "BJ2012DHS"}:
                    r.append("flagged_official_round")
                if case == "questionnaire_compatible" and row.survey_id == "NP2011DHS":
                    r.append("incompatible_nepal_2011_questionnaire_filter")
                if case == "common_2020" and row.fieldwork_end >= pd.Timestamp("2020-01-01"):
                    r.append("after_common_training_cutoff")
                if r:
                    reasons[row.survey_id] = "|".join(r)
                    exclusions.append(dict(case=case, target_country=target.country, target_survey=target.survey_id,
                                           excluded_survey=row.survey_id, excluded_country=row.country, reasons="|".join(r)))
            train = rounds[~rounds.survey_id.isin(reasons)].copy()
            weights = 1.0 / train.groupby("country").survey_id.transform("size").to_numpy()
            assert np.isclose(weights.sum(), train.country.nunique())
            key = dict(case=case, target_country=target.country, target_survey=target.survey_id)
            for row, weight in zip(train.itertuples(), weights):
                membership.append(dict(**key, training_country=row.country, training_survey=row.survey_id, weight=float(weight)))
            y = train[ITEMS].to_numpy(float)
            observed = np.array([getattr(target, i) for i in ITEMS], float)
            x = train[FEATURES].to_numpy(float)
            test_x = np.array([getattr(target, f) for f in FEATURES], float)
            impute = np.array([np.average(x[np.isfinite(x[:, j]), j], weights=weights[np.isfinite(x[:, j])])
                               for j in range(len(FEATURES))])
            filled = np.where(np.isfinite(x), x, impute)
            mean = np.average(filled, weights=weights, axis=0)
            scale = np.sqrt(np.average((filled - mean) ** 2, weights=weights, axis=0))
            scale[scale == 0] = 1.0
            z = (filled - mean) / scale
            test_z = (np.where(np.isfinite(test_x), test_x, impute) - mean) / scale
            for j, feature in enumerate(FEATURES):
                missingness.append(dict(**key, feature=feature, training_missing=int(np.isnan(x[:, j]).sum()),
                                        target_missing=bool(np.isnan(test_x[j])), imputed_value=float(impute[j])))
            raw = {}
            for model, nfeature in [("R0", 1), ("R1", 5)]:
                estimator = Ridge(alpha=ALPHA, fit_intercept=True, solver="cholesky")
                estimator.fit(z[:, :nfeature], y, sample_weight=weights)
                raw[model] = estimator.predict(test_z[None, :nfeature])[0]
                params.append(dict(**key, model=model, features=FEATURES[:nfeature], alpha=ALPHA,
                                   intercept=estimator.intercept_.tolist(), coefficients=estimator.coef_.tolist()))
            # Five balanced outcome tasks; context coefficients are shared.
            n = len(train)
            design = np.zeros((n * 5, 14), float)
            target_design = np.zeros((5, 14), float)
            for j in range(5):
                design[j*n:(j+1)*n, j] = 1.0
                design[j*n:(j+1)*n, 5+j] = z[:, 0]
                design[j*n:(j+1)*n, 10:] = z[:, 1:]
                target_design[j, j] = 1.0
                target_design[j, 5+j] = test_z[0]
                target_design[j, 10:] = test_z[1:]
            stacked_weight = np.tile(weights, 5)
            penalty = np.diag([0.0] * 5 + [ALPHA] * 5 + [5 * ALPHA] * 4)
            normal = design.T @ (stacked_weight[:, None] * design) + penalty
            rhs = design.T @ (stacked_weight * y.T.reshape(-1))
            theta = solve(normal, rhs, assume_a="pos")
            raw["S"] = target_design @ theta
            params.append(dict(**key, model="S", features=FEATURES, alpha=ALPHA,
                               shared_context_penalty_multiplier=5, expanded_parameter_order="5 intercepts, 5 calendar slopes, 4 shared context slopes",
                               expanded_coefficients=theta.tolist(),
                               intercept=theta[:5].tolist(), calendar_coefficients=theta[5:10].tolist(),
                               shared_context_coefficients=theta[10:].tolist()))
            params.append(dict(**key, model="preprocessing", features=FEATURES, training_rounds=n,
                               training_countries=train.country.nunique(), total_weight=float(weights.sum()),
                               impute_weighted_mean=impute.tolist(), weighted_mean=mean.tolist(), weighted_scale=scale.tolist()))
            expected = raw["R1"].mean() + raw["R0"] - raw["R0"].mean()
            error = float(np.max(np.abs(expected - raw["S"])))
            assert error < 1e-8
            identities.append(dict(**key, raw_shared_identity_max_abs_error=error,
                                   raw_level_identity_abs_error=float(abs(raw["S"].mean()-raw["R1"].mean())),
                                   raw_profile_identity_max_abs_error=float(np.max(np.abs(raw["S"]-raw["S"].mean()-raw["R0"]+raw["R0"].mean()))),
                                   normal_equation_max_abs_residual=float(np.max(np.abs(normal @ theta-rhs)))))
            for model in ["R0", "S", "R1"]:
                bounded = np.clip(raw[model], 0, 100)
                for j, item in enumerate(ITEMS):
                    pred_rows.append(dict(**key, item=item, observed_pp=float(observed[j]), model=model,
                                          raw_prediction_pp=float(raw[model][j]), prediction_pp=float(bounded[j]),
                                          clipped=bool(raw[model][j] != bounded[j]), training_rounds=n,
                                          training_countries=train.country.nunique()))
                for kind, prediction in [("bounded", bounded), ("raw", raw[model])]:
                    met_rows.append(dict(**key, model=model, prediction_kind=kind, **metrics(observed, prediction)))
        print("Completed", case, len(case_targets), "targets", flush=True)
    predictions, met = pd.DataFrame(pred_rows), pd.DataFrame(met_rows)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    met.to_csv(OUT / "country_metrics.csv", index=False)
    pd.DataFrame(membership).to_csv(OUT / "training_membership.csv", index=False)
    pd.DataFrame(exclusions).to_csv(OUT / "exclusions.csv", index=False)
    pd.DataFrame(missingness).to_csv(OUT / "fold_missingness.csv", index=False)
    pd.DataFrame(identities).to_csv(OUT / "identity_checks.csv", index=False)
    save_json(OUT / "model_parameters.json", params)
    prior = pd.read_csv(INPUT / "original_predictions.csv")
    keys = ["case", "target_country", "target_survey", "item", "model"]
    matched = predictions[predictions.model.isin(["R0", "R1"])].merge(prior, on=keys, validate="one_to_one", suffixes=("_new", "_old"))
    assert len(matched) == len(prior) == 1040
    replay_error = float(np.max(np.abs(matched.raw_prediction_pp_new - matched.raw_prediction_pp_old)))
    assert replay_error < 1e-8
    old_membership = pd.read_csv(INPUT / "original_training_membership.csv")
    new_membership = pd.DataFrame(membership)
    new_membership = new_membership[new_membership.case.isin(old_membership.case.unique())].reset_index(drop=True)
    assert len(new_membership) == len(old_membership)
    for col in old_membership:
        if col != "weight":
            assert old_membership[col].equals(new_membership[col]), col
    assert np.max(abs(old_membership.weight-new_membership.weight)) < 1e-14
    summaries, comparisons, paired_diffs, draws_saved = [], [], [], []
    rng = np.random.default_rng(SEED)
    for case in CASES:
        for kind in ["bounded", "raw"]:
            part = met[met.case.eq(case) & met.prediction_kind.eq(kind)]
            for model, group in part.groupby("model"):
                summaries.append(dict(case=case, prediction_kind=kind, model=model, countries=len(group),
                                      **{"mean_"+m: float(group[m].mean()) for m in METRICS+["bias_pp"]}))
            pivot = part.pivot(index="target_country", columns="model", values=METRICS).sort_index()
            draws = rng.integers(0, len(pivot), size=(NBOOT, len(pivot)))
            for model, reference in [("S", "R0"), ("R1", "S"), ("R1", "R0")]:
                for metric in METRICS:
                    delta = pivot[(metric, model)] - pivot[(metric, reference)]
                    sampled = delta.to_numpy()[draws].mean(axis=1)
                    low, high = np.quantile(sampled, [.025, .975])
                    comparisons.append(dict(case=case, prediction_kind=kind, model=model, reference=reference,
                                            metric=metric, countries=len(delta), mean_difference=float(delta.mean()),
                                            median_difference=float(delta.median()), countries_improved=int((delta < -1e-10).sum()),
                                            countries_tied_numerically=int((delta.abs() <= 1e-10).sum()),
                                            descriptive_low=float(low), descriptive_high=float(high), bootstrap_resamples=NBOOT))
                    for country, value in delta.items():
                        paired_diffs.append(dict(case=case, prediction_kind=kind, model=model, reference=reference,
                                                 metric=metric, target_country=country, difference=float(value)))
                    draws_saved.extend(dict(case=case, prediction_kind=kind, model=model, reference=reference,
                                            metric=metric, resample=i, mean_difference=float(v)) for i,v in enumerate(sampled))
    pd.DataFrame(summaries).to_csv(OUT / "model_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_comparisons.csv", index=False)
    pd.DataFrame(paired_diffs).to_csv(OUT / "paired_country_differences.csv", index=False)
    pd.DataFrame(draws_saved).to_csv(OUT / "bootstrap_draws.csv", index=False)
    item_results = []
    for kind, column in [("bounded", "prediction_pp"), ("raw", "raw_prediction_pp")]:
        temp = predictions.copy()
        temp["error"] = temp[column]-temp.observed_pp
        temp["absolute_error"] = abs(temp.error)
        temp["squared_error"] = temp.error**2
        grouped = temp.groupby(["case", "model", "item"], as_index=False).agg(countries=("target_country", "size"),
                   MAE_pp=("absolute_error", "mean"), MSE_pp2=("squared_error", "mean"), bias_pp=("error", "mean"))
        grouped["prediction_kind"] = kind
        item_results.append(grouped)
    pd.concat(item_results, ignore_index=True).to_csv(OUT / "item_metrics.csv", index=False)
    manifest = json.loads((INPUT / "SOURCE_MANIFEST.json").read_text())
    assert sha(BASE / "PROTOCOL.md") == manifest["protocol_sha256"]
    for rel, expected in manifest["original_analysis_guard_sha256"].items():
        assert sha(REV / rel) == expected, rel
    summary = pd.DataFrame(summaries)
    save_json(OUT / "key_numbers.json", {
        "rounds": 169, "countries_in_source_pool": 62, "case_target_counts": {"primary":42,"anomaly_removed":42,"common_2020":20,"questionnaire_compatible":42},
        "primary_bounded_summary": summary[summary.case.eq("primary") & summary.prediction_kind.eq("bounded")].to_dict("records"),
        "clipping_counts": predictions.groupby(["case", "model"]).clipped.sum().reset_index().to_dict("records"),
        "original_raw_prediction_replay_max_abs_error": replay_error})
    save_json(OUT / "RUN_METADATA.json", {
        "started_utc":start, "finished_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256":sha(BASE/"PROTOCOL.md"), "fitting_code_sha256":sha(__file__),
        "questionnaire_amendment_sha256":sha(BASE/"QUESTIONNAIRE_AMENDMENT.md"),
        "source_manifest_sha256":sha(INPUT/"SOURCE_MANIFEST.json"), "seed":SEED,"bootstrap_resamples":NBOOT,"alpha":ALPHA,
        "environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,
                       "scipy":scipy.__version__,"sklearn":sklearn.__version__},
        "source_analysis_preserved":True, "original_prediction_replay_max_abs_error":replay_error,
        "output_sha256":{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name != "RUN_METADATA.json"}})
    print(summary[summary.prediction_kind.eq("bounded")].to_string(index=False))


if __name__ == "__main__":
    fit()
