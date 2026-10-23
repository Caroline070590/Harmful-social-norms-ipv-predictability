"""Locked ENDES 2023-to-2024 prediction; no test-outcome model selection."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import platform

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import helmert
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

BASE = Path(__file__).resolve().parent
STUDY = BASE.parent
REV = STUDY.parents[1]
OLD = REV / "04_mental_health/endes_public"
LOCAL = BASE / "data_local"
OUT = BASE / "results"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def legacy():
    path = OLD / "prediction/run_prediction.py"
    spec = importlib.util.spec_from_file_location("frozen_endes_2023_prediction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load():
    cfg = json.loads((BASE / "CONFIG.json").read_text())
    fixed = legacy()
    old_cfg, old_ready, full_train = fixed.load()
    for key in ["outcome", "bounds", "feature_models", "algorithms", "ridge_alpha", "boosting"]:
        assert cfg[key] == old_cfg[key], ("Unplanned specification change", key)
    strict = (pd.to_numeric(full_train.interview_year).eq(cfg["strict_training_interview_year"])
              & pd.to_numeric(full_train.health_interview_year).eq(cfg["strict_training_interview_year"]))
    train = full_train.loc[strict].copy()
    train.attrs["calendar_gate"] = dict(release_cohort_women=len(full_train),
        strict_training_women=len(train), later_or_other_interview_exclusions=int((~strict).sum()),
        rule="Both actual individual and health interview years equal2023; outcome-independent restriction",
        original_cohort_and_previous_results_preserved=True)
    ready = json.loads((STUDY / "MODEL_READY.json").read_text())
    assert ready["status"] == "ready_for_equal_person_conditional_prediction_only"
    target_path = STUDY / ready["cohort_path"]
    assert sha(target_path) == ready["summary"]["prepared_cohort_sha256"]
    cats = old_ready["categorical_covariates"]
    dtype = {name: str for name in cats + ["row_id", "psu", "stratum"]}
    target = pd.read_csv(target_path, dtype=dtype, keep_default_na=False)
    assert len(target) == ready["summary"]["prepared_records"] and target.row_id.is_unique
    assert ready["attitude_items"] == old_ready["attitude_items"]
    assert ready["IPV_primary_recent"] == old_ready["IPV_primary_recent"]
    assert ready["categorical_covariates"] == cats
    assert target[ready["attitude_items"]].isin([0, 1]).all().all()
    assert target.phq9_total.between(0, 27).all() and target.age.between(18, 49).all()
    assert np.array_equal(target[ready["attitude_items"]].sum(axis=1), target.norm_score_0_5)
    for frame, year in [(train, cfg["strict_training_interview_year"]), (target, cfg["strict_target_interview_year"])]:
        assert pd.to_numeric(frame.interview_year).eq(year).all()
        assert pd.to_numeric(frame.health_interview_year).eq(year).all()
        assert frame.analysis_weight_equal.eq(1).all()
        assert frame.groupby("psu").stratum.nunique().max() == 1
    h = helmert(5, full=False)
    contrast = target[ready["attitude_items"]].to_numpy(float) @ h.T
    target["age_squared"] = target.age ** 2
    for i in range(4):
        target["attitude_profile_" + str(i + 1)] = contrast[:, i]
    assert np.allclose(target.norm_score_0_5.to_numpy()[:, None] / 5 + contrast @ h,
                       target[ready["attitude_items"]].to_numpy(), atol=1e-14)
    flags_path = STUDY / "data_local/temporal_test_flags.csv"
    assert sha(flags_path) == ready["temporal_validation"]["test_flags_sha256"]
    flags = pd.read_csv(flags_path, dtype={"row_id": str})
    assert flags.row_id.is_unique and set(flags.row_id) == set(target.row_id)
    target = target.merge(flags, on="row_id", validate="one_to_one", how="left")
    assert target[cfg["new_psu_flag"]].isin([0, 1]).all()
    new = target[cfg["new_psu_flag"]].eq(1)
    assert new.any() and not set(target.loc[new, "psu"]) & set(train.psu)
    independent = json.loads((STUDY / "INDEPENDENT_PREPARATION_QC.json").read_text())
    assert independent["model_ready_sha256"] == sha(STUDY / "MODEL_READY.json")
    assert independent["cohort_sha256"] == sha(target_path)
    assert independent["source_inventory_sha256"] == sha(STUDY / "SOURCE_INVENTORY.json")
    assert len(train) == independent["training_cohort"]["women"]
    assert int(new.sum()) == independent["cross_year_PSU"]["test_women_without_any_2023_household_source_PSU"]
    return cfg, fixed, old_ready, ready, train, target, flags_path


def metrics(y, p):
    e = np.asarray(p) - np.asarray(y)
    variance = np.mean((y - np.mean(y)) ** 2)
    return dict(RMSE=float(np.sqrt(np.mean(e * e))), MAE=float(np.mean(abs(e))),
                bias=float(np.mean(e)), R2=float(1 - np.mean(e * e) / variance) if variance > 0 else None)


def domains(frame, cfg):
    return dict(all_2024=np.ones(len(frame), dtype=bool),
                new_psu_2024=frame[cfg["new_psu_flag"]].eq(1).to_numpy(),
                no_endorsement=frame.norm_score_0_5.eq(0).to_numpy(),
                any_endorsement=frame.norm_score_0_5.gt(0).to_numpy())


def cluster_uncertainty(frame, predictions, cfg, masks):
    keys = [(a, m) for a in cfg["algorithms"] for m in cfg["feature_models"]]
    pairs = [tuple(cfg["primary_comparison"])] + [tuple(p) for p in cfg["secondary_comparisons"]]
    units = frame[["stratum", "psu"]].drop_duplicates().sort_values(["stratum", "psu"]).reset_index(drop=True)
    units["unit"] = np.arange(len(units))
    mapping = frame[["stratum", "psu"]].merge(units, on=["stratum", "psu"], validate="many_to_one")
    assert np.array_equal(mapping.psu, frame.psu)
    which = mapping.unit.to_numpy()
    count = np.zeros((cfg["bootstrap_draws"], len(units)), dtype=np.int16)
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    singletons = 0
    for _, group in units.groupby("stratum", sort=True):
        ix = group.unit.to_numpy()
        count[:, ix] = rng.multinomial(len(ix), np.ones(len(ix)) / len(ix), size=len(count))
        singletons += int(len(ix) == 1)
    y = frame.phq9_total.to_numpy(float)
    bounded = np.column_stack([np.clip(predictions[k], *cfg["bounds"]) for k in keys])
    errors = bounded - y[:, None]
    rows, draws, eligibility = [], [], []
    for domain, mask in masks.items():
        n, psus = int(mask.sum()), frame.loc[mask, "psu"].nunique()
        eligible = n >= cfg["bootstrap_min_women"] and psus >= cfg["bootstrap_min_PSUs"]
        eligibility.append(dict(domain=domain, women=n, PSUs=psus, intervals_available=bool(eligible),
                                reason="fixed count thresholds passed" if eligible else "insufficient observed women or PSUs for planned interval"))
        if not eligible:
            continue
        cn = np.bincount(which, weights=mask.astype(float), minlength=len(units))
        sse = np.column_stack([np.bincount(which, weights=errors[:, j] ** 2 * mask, minlength=len(units)) for j in range(len(keys))])
        sae = np.column_stack([np.bincount(which, weights=abs(errors[:, j]) * mask, minlength=len(units)) for j in range(len(keys))])
        denominator = count @ cn
        valid = denominator > 0
        assert valid.any()
        values = dict(RMSE=np.sqrt((count[valid] @ sse) / denominator[valid, None]),
                      MAE=(count[valid] @ sae) / denominator[valid, None])
        for algorithm in cfg["algorithms"]:
            for model, reference in pairs:
                i, j = keys.index((algorithm, model)), keys.index((algorithm, reference))
                for metric, estimates in values.items():
                    delta = estimates[:, i] - estimates[:, j]
                    difference = metrics(y[mask], bounded[mask, i])[metric] - metrics(y[mask], bounded[mask, j])[metric]
                    lo, hi = np.quantile(delta, [.025, .975])
                    rows.append(dict(domain=domain, algorithm=algorithm, model=model, reference=reference,
                        metric=metric, difference=difference, lower=float(lo), upper=float(hi),
                        valid_draws=len(delta), excluded_zero_denominator_draws=int((~valid).sum()),
                        uncertainty_scope=cfg["interval_scope"]))
                    draws.extend(dict(domain=domain, algorithm=algorithm, model=model, reference=reference,
                        metric=metric, draw=int(k), difference=float(v)) for k, v in zip(np.flatnonzero(valid), delta))
    pd.DataFrame(rows).to_csv(OUT / "paired_comparisons.csv", index=False)
    pd.DataFrame(draws).to_csv(OUT / "bootstrap_draws.csv", index=False)
    pd.DataFrame(eligibility).to_csv(OUT / "interval_readiness.csv", index=False)
    save(OUT / "BOOTSTRAP_METADATA.json", dict(draws=cfg["bootstrap_draws"], seed=cfg["bootstrap_seed"],
        all_test_PSUs=len(units), strata=units.stratum.nunique(), singleton_strata=singletons,
        order=keys, resampling="Whole observed 2024 PSUs within released strata; same draws across domains/models",
        predictions_fixed=True, training_fixed=True, no_population_or_full_refitting_inference=True))


def main():
    for path in [LOCAL, OUT]:
        path.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "RUN_METADATA.json").exists(), "Preserve completed runs; use a new directory"
    assert not (BASE / "FIT_INPUT_MANIFEST.json").exists(), "Do not overwrite a frozen fitting manifest"
    for path in [STUDY / "INDEPENDENT_PREPARATION_QC.json", OLD / "INDEPENDENT_PREPARATION_QC.json"]:
        assert json.loads(path.read_text())["status"] == "passed", path
    cfg, fixed, old_ready, ready, train, target, flags_path = load()
    inputs = [BASE / "CONFIG.json", Path(__file__), STUDY / "PROTOCOL.md", STUDY / "MODEL_READY.json",
              STUDY / "INDEPENDENT_PREPARATION_QC.json", STUDY / ready["cohort_path"], flags_path,
              OLD / "prediction/run_prediction.py", OLD / "prediction/CONFIG.json", OLD / "MODEL_READY.json",
              OLD / old_ready["cohort_path"], OLD / "INDEPENDENT_PREPARATION_QC.json"]
    for name in ["SOURCE_INVENTORY.json", "SOURCE_AND_DESIGN_REVIEW.md", "READINESS.json", "TEMPORAL_DATE_AMENDMENT.md"]:
        if (STUDY / name).exists():
            inputs.append(STUDY / name)
    frozen = {str(p.relative_to(REV)): sha(p) for p in inputs}
    save(BASE / "FIT_INPUT_MANIFEST.json", dict(before_fitting=True,
        saved_utc=datetime.now(timezone.utc).isoformat(), source_sha256=frozen,
        test_outcomes_not_used_for_specification_selection=True))
    save(OUT / "TRAINING_DATE_GATE.json", train.attrs["calendar_gate"])
    blocks = fixed.blocks(old_ready)
    train_y, target_y = train.phq9_total.to_numpy(float), target.phq9_total.to_numpy(float)
    predictions = {("reference", "training_mean"): np.full(len(target), train_y.mean())}
    records, metrics_rows, domain_counts = [], [], []
    masks = domains(target, cfg)
    for name, mask in masks.items():
        domain_counts.append(dict(domain=name, women=int(mask.sum()), PSUs=target.loc[mask, "psu"].nunique(),
            strata=target.loc[mask, "stratum"].nunique(), endorsers=int(target.loc[mask, "norm_score_0_5"].gt(0).sum())))
    for model in cfg["feature_models"]:
        x, query, transform = fixed.encode(train, target, blocks[model], old_ready["categorical_covariates"])
        assert not {"row_id", "psu", "stratum", "phq9_total"}.intersection(blocks[model])
        for algorithm in cfg["algorithms"]:
            if algorithm == "ridge":
                estimator = Ridge(alpha=cfg["ridge_alpha"], solver="cholesky").fit(x, train_y)
            else:
                estimator = HistGradientBoostingRegressor(**cfg["boosting"], random_state=cfg["estimator_seed"]).fit(x, train_y)
            key = algorithm + "_" + model
            raw = estimator.predict(query)
            assert np.isfinite(raw).all()
            predictions[(algorithm, model)] = raw
            joblib.dump(estimator, LOCAL / (key + ".joblib"), compress=3)
            record = dict(algorithm=algorithm, model=model, transform=transform,
                training_women=len(train), test_women=len(target), model_file=key + ".joblib")
            if algorithm == "ridge":
                record.update(coefficient=estimator.coef_.tolist(), intercept=float(estimator.intercept_))
            else:
                record.update(iterations=estimator.n_iter_, tree_count=sum(len(t) for t in estimator._predictors))
            records.append(record)
            print("Completed", key, flush=True)
    private = target[["row_id", "psu", "stratum", "phq9_total", "norm_score_0_5", cfg["new_psu_flag"]]].copy()
    for (algorithm, model), raw in predictions.items():
        private[algorithm + "__" + model + "__raw"] = raw
        bounded = np.clip(raw, *cfg["bounds"])
        for domain, mask in masks.items():
            if not mask.any():
                continue
            for variant, predicted in [("raw", raw), ("bounded", bounded)]:
                metrics_rows.append(dict(domain=domain, algorithm=algorithm, model=model, variant=variant,
                    women=int(mask.sum()), PSUs=target.loc[mask, "psu"].nunique(),
                    clipped_rows=int(np.count_nonzero(raw[mask] != bounded[mask])),
                    **metrics(target_y[mask], predicted[mask])))
    private.to_csv(LOCAL / "temporal_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(metrics_rows).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(domain_counts).to_csv(OUT / "target_domain_counts.csv", index=False)
    save(OUT / "model_records.json", records)
    cluster_uncertainty(target, predictions, cfg, masks)
    assert all(sha(REV / p) == digest for p, digest in frozen.items()), "Frozen input changed during fitting"
    save(OUT / "RUN_METADATA.json", dict(status="fitted_pending_independent_qc", fitted_models=len(records),
        training_women=len(train), target_women=len(target), actual_interview_years=[2023, 2024],
        all_preprocessing_fitted_on_2023=True, target_year_used_for_tuning=False,
        no_PSU_ID_predictor=True, source_sha256=frozen,
        finished_utc=datetime.now(timezone.utc).isoformat(), environment=dict(python=platform.python_version(),
            numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__)))
    print(pd.DataFrame(metrics_rows).query("domain == 'all_2024' and variant == 'bounded'").to_string(index=False))


if __name__ == "__main__":
    main()
