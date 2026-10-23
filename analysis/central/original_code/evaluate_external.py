"""Fixed external MICS evaluation from report-verified aggregate cells only."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import Ridge

BASE = Path(__file__).resolve().parent
REV = BASE.parents[1]
AUDIT = BASE / "source_audit/results"
OUT = BASE / "results"
STRUCT = REV / "02_aggregate_landscape/original_paper_extension/structural/raw"
DHS = REV / "05_transportability/joint_country_time/prepared/rounds.csv"
ITEMS = ["WE_AWBT_W_BFD", "WE_AWBT_W_ARG", "WE_AWBT_W_OUT", "WE_AWBT_W_NEG", "WE_AWBT_W_REF"]
FEATURES = ["hdi", "electoral_democracy", "liberal_democracy", "women_political_empowerment"]
FILES = {"electoral_democracy":"electoral-democracy-index.csv", "liberal_democracy":"liberal-democracy-index.csv", "women_political_empowerment":"women-political-empowerment-index.csv"}
ALPHA, SEED, NBOOT = 10.0, 20261020, 10000
NUMERICAL_TIE_TOLERANCE = 1e-9
SOURCE_HASHES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    SOURCE_HASHES[str(path.relative_to(REV))] = sha(path)
    return path


def csv(path, **kwargs):
    return pd.read_csv(record(path), **kwargs)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def truth(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    inv_path = AUDIT / "verified_survey_inventory.csv"
    cells_path = AUDIT / "verified_national_components.csv"
    missing = [str(p) for p in [inv_path, cells_path] if not p.exists()]
    if missing:
        save_json(OUT / "READINESS.json", {"status":"missing_verified_sources", "missing":missing, "numerical_results_generated":False})
        raise FileNotFoundError("Source verification incomplete; no placeholder results generated: " + ", ".join(missing))
    inv, cells = csv(inv_path), csv(cells_path)
    frozen = json.loads(record(AUDIT.parent / "SOURCE_AUDIT_FREEZE.json").read_text())
    frozen_files = {item["path"]:item["sha256"] for item in frozen["files"]}
    for path in [inv_path, cells_path]:
        # Source auditor freezes the exact inputs before parent-model evaluation.
        assert sha(path) == frozen_files[str(path.relative_to(AUDIT.parent))], ("Source inventory differs from frozen audit", path)
    for row in inv[["source_file", "report_sha256"]].drop_duplicates().itertuples(index=False):
        report = AUDIT.parent / row.source_file
        assert sha(record(report)) == row.report_sha256
    required = {"iso3", "country", "survey_id", "survey_year", "fieldwork_start", "fieldwork_end", "eligible_primary", "all_five_verified", "programme_class", "documented_geographic_omission"}
    assert required.issubset(inv), required - set(inv)
    assert {"survey_id", "item", "observed_pp"}.issubset(cells)
    eligible = inv.loc[truth(inv.eligible_primary) & truth(inv.all_five_verified)].copy()
    eligible["documented_geographic_omission"] = truth(eligible.documented_geographic_omission)
    eligible["start_date_reported"] = eligible.fieldwork_start
    eligible["fieldwork_start"] = pd.to_datetime(eligible.fieldwork_start, format="ISO8601").dt.to_period("M").dt.start_time
    eligible["fieldwork_end"] = pd.to_datetime(eligible.fieldwork_end, format="ISO8601").dt.to_period("M").dt.end_time.dt.normalize()
    assert eligible.fieldwork_start.notna().all() and eligible.fieldwork_end.ge(eligible.fieldwork_start).all()
    eligible = eligible.sort_values(["iso3", "fieldwork_start", "survey_id"]).groupby("iso3", as_index=False).tail(1).sort_values("iso3")
    assert eligible.iso3.is_unique and eligible.survey_id.is_unique and len(eligible) > 0
    cells = cells.loc[cells.survey_id.isin(eligible.survey_id) & cells.item.isin(ITEMS)].copy()
    assert not cells.duplicated(["survey_id", "item"]).any()
    y = cells.pivot(index="survey_id", columns="item", values="observed_pp").reindex(columns=ITEMS)
    assert y.notna().all().all() and len(y) == len(eligible)
    assert y.ge(0).all().all() and y.le(100).all().all()
    eligible = eligible.merge(y.reset_index(), on="survey_id", validate="one_to_one")
    eligible["calendar"] = eligible.fieldwork_start.dt.year + (eligible.fieldwork_start.dt.month - 1) / 12
    hdr = csv(STRUCT / "undp_hdr.csv", encoding="latin1").set_index("iso3")
    lookups = {}
    for feature, filename in FILES.items():
        d = csv(STRUCT / filename)
        value_cols = [c for c in d if c not in ["Entity", "Code", "Year", "World region according to OWID"]]
        assert len(value_cols) == 1
        d = d.loc[d.Code.str.fullmatch("[A-Z]{3}", na=False)]
        assert not d.duplicated(["Code", "Year"]).any()
        lookups[feature] = {(r.Code, int(r.Year)):float(v) for r, v in zip(d.itertuples(), d[value_cols[0]]) if pd.notna(v)}
    ledger = []
    for index, row in eligible.iterrows():
        for feature in FEATURES:
            value, selected_year = np.nan, None
            for year in range(row.fieldwork_start.year - 1, row.fieldwork_start.year - 4, -1):
                candidate = (hdr.loc[row.iso3, "hdi_" + str(year)] if feature == "hdi" and row.iso3 in hdr.index and "hdi_" + str(year) in hdr else lookups.get(feature, {}).get((row.iso3, year)))
                candidate = pd.to_numeric(pd.Series([candidate]), errors="coerce").iloc[0]
                if pd.notna(candidate):
                    value, selected_year = float(candidate), year
                    break
            eligible.loc[index, feature] = value
            ledger.append(dict(iso3=row.iso3, country=row.country, survey_id=row.survey_id, feature=feature,
                value=value, source_year=selected_year, target_start_year=row.fieldwork_start.year,
                lag_years=row.fieldwork_start.year - selected_year if selected_year else np.nan,
                missing=selected_year is None))
    donors = csv(DHS)
    for col in ["fieldwork_start", "fieldwork_end"]:
        donors[col] = pd.to_datetime(donors[col])
    donors["conservative_fieldwork_end"] = donors.fieldwork_end.dt.to_period("M").dt.end_time.dt.normalize()
    eligible["in_DHS_training_inventory"] = eligible.iso3.isin(set(donors.iso3))
    eligible["complete_context"] = eligible[FEATURES].notna().all(axis=1)
    eligible.to_csv(OUT / "eligible_targets.csv", index=False)
    cells.to_csv(OUT / "source_cells.csv", index=False)
    pd.DataFrame(ledger).to_csv(OUT / "structural_source_ledger.csv", index=False)
    return eligible, donors


def metric(y, p):
    error = p - y
    level = error.mean()
    profile = error - level
    assert np.isclose(np.mean(error ** 2), level ** 2 + np.mean(profile ** 2), atol=1e-10)
    return {"MAE_pp":float(np.abs(error).mean()), "MSE_pp2":float(np.mean(error ** 2)), "bias_pp":float(level),
        "level_MAE_pp":float(abs(level)), "level_MSE_pp2":float(level ** 2),
        "profile_MAE_pp":float(np.abs(profile).mean()), "profile_MSE_pp2":float(np.mean(profile ** 2))}


def fit(targets, donors):
    pred, met, membership, exclusions, parameters = [], [], [], [], []
    for case in ["primary", "anomaly_removed", "questionnaire_compatible"]:
        removed = {"NP2011DHS", "BJ2012DHS"} if case == "anomaly_removed" else {"NP2011DHS"} if case == "questionnaire_compatible" else set()
        for target in targets.itertuples():
            keep = donors.iso3.ne(target.iso3) & donors.conservative_fieldwork_end.lt(target.fieldwork_start) & ~donors.survey_id.isin(removed)
            train = donors.loc[keep].copy()
            if train.country.nunique() < 2:
                exclusions.append(dict(case=case, target_survey=target.survey_id, excluded_survey="TARGET", reason="fewer_than_two_eligible_donor_countries"))
                continue
            weights = 1 / train.groupby("country").survey_id.transform("size").to_numpy(float)
            assert np.isclose(weights.sum(), train.country.nunique())
            for row, weight in zip(train.itertuples(), weights):
                membership.append(dict(case=case, target_iso3=target.iso3, target_survey=target.survey_id,
                    training_iso3=row.iso3, training_country=row.country, training_survey=row.survey_id, weight=float(weight),
                    conservative_training_end=str(row.conservative_fieldwork_end.date()), target_start=str(target.fieldwork_start.date())))
            for row in donors.loc[~keep].itertuples():
                reason = []
                if row.iso3 == target.iso3: reason.append("target_country")
                if row.conservative_fieldwork_end >= target.fieldwork_start: reason.append("not_strictly_earlier")
                if row.survey_id in removed: reason.append("fixed_source_exclusion")
                assert reason
                exclusions.append(dict(case=case, target_survey=target.survey_id, excluded_survey=row.survey_id, reason="|".join(reason)))
            y_train, y_test = train[ITEMS].to_numpy(float), np.array([getattr(target, f) for f in ITEMS])
            raw = {}
            for model, features in [("R0", ["calendar"]), ("R1", ["calendar"] + FEATURES)]:
                x = train[features].to_numpy(float)
                x_test = np.array([getattr(target, f) for f in features], float)
                if not np.isfinite(x).any(axis=0).all():
                    raise ValueError("Required feature entirely missing within training fold: " + target.survey_id)
                impute = np.array([np.average(x[np.isfinite(x[:, j]), j], weights=weights[np.isfinite(x[:, j])]) for j in range(len(features))])
                filled = np.where(np.isfinite(x), x, impute)
                mean = np.average(filled, weights=weights, axis=0)
                scale = np.sqrt(np.average((filled - mean) ** 2, weights=weights, axis=0)); scale[scale == 0] = 1
                z = (filled - mean) / scale
                z_test = (np.where(np.isfinite(x_test), x_test, impute) - mean) / scale
                estimator = Ridge(alpha=ALPHA, fit_intercept=True, solver="cholesky").fit(z, y_train, sample_weight=weights)
                raw[model] = estimator.predict(z_test.reshape(1, -1))[0]
                parameters.append(dict(case=case, target_survey=target.survey_id, model=model, features=features, alpha=ALPHA,
                    training_rounds=len(train), training_countries=train.country.nunique(), impute=impute.tolist(), weighted_mean=mean.tolist(),
                    weighted_scale=scale.tolist(), target_feature_missing=np.isnan(x_test).tolist(), intercept=estimator.intercept_.tolist(), coefficients=estimator.coef_.tolist()))
            raw["S"] = raw["R1"].mean() + raw["R0"] - raw["R0"].mean()
            assert np.isclose(raw["S"].mean(), raw["R1"].mean())
            assert np.allclose(raw["S"] - raw["S"].mean(), raw["R0"] - raw["R0"].mean())
            for model in ["R0", "S", "R1"]:
                bounded = np.clip(raw[model], 0, 100)
                for j, item in enumerate(ITEMS):
                    pred.append(dict(case=case, iso3=target.iso3, country=target.country, target_survey=target.survey_id, item=item,
                        model=model, observed_pp=float(y_test[j]), raw_prediction_pp=float(raw[model][j]), prediction_pp=float(bounded[j]),
                        clipped=bool(raw[model][j] != bounded[j]), complete_context=bool(target.complete_context), in_DHS_training_inventory=bool(target.in_DHS_training_inventory),
                        programme_class=target.programme_class, documented_geographic_omission=bool(target.documented_geographic_omission),
                        training_rounds=len(train), training_countries=train.country.nunique()))
                for kind, prediction in [("raw", raw[model]), ("bounded", bounded)]:
                    met.append(dict(case=case, iso3=target.iso3, country=target.country, target_survey=target.survey_id, model=model,
                        prediction_kind=kind, complete_context=bool(target.complete_context), in_DHS_training_inventory=bool(target.in_DHS_training_inventory),
                        programme_class=target.programme_class, documented_geographic_omission=bool(target.documented_geographic_omission), **metric(y_test, prediction)))
    pred, met = pd.DataFrame(pred), pd.DataFrame(met)
    assert not pred.empty
    pred.to_csv(OUT / "predictions.csv", index=False)
    met.to_csv(OUT / "country_metrics.csv", index=False)
    pd.DataFrame(membership).to_csv(OUT / "training_membership.csv", index=False)
    pd.DataFrame(exclusions).to_csv(OUT / "exclusions.csv", index=False)
    save_json(OUT / "model_parameters.json", parameters)
    summarize(met)
    save_json(OUT / "RUN_METADATA.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(BASE / "PROTOCOL.md"),
        fitting_script_sha256=sha(Path(__file__)), source_sha256=SOURCE_HASHES, predictions=len(pred), targets=pred.iso3.nunique(),
        alpha=ALPHA, seed=SEED, bootstrap_replicates=NBOOT, training_programme="DHS only", evaluation_programme="report-verified MICS",
        source_compatible_exclusion=["NP2011DHS"], survey_sampling_uncertainty_reconstructed=False,
        environment={"python":platform.python_version(), "numpy":np.__version__, "pandas":pd.__version__, "scipy":scipy.__version__, "sklearn":sklearn.__version__}))


def summarize(met):
    summaries, comparisons, differences = [], [], []
    names = ["MAE_pp", "level_MAE_pp", "profile_MAE_pp", "MSE_pp2", "level_MSE_pp2", "profile_MSE_pp2", "bias_pp"]
    rng = np.random.default_rng(SEED)
    for (case, kind), all_rows in met.groupby(["case", "prediction_kind"], sort=True):
        for cohort, group in [("all_eligible", all_rows), ("complete_context", all_rows.loc[all_rows.complete_context]),
                ("outside_DHS_inventory", all_rows.loc[~all_rows.in_DHS_training_inventory]),
                ("nonhybrid_MICS", all_rows.loc[~all_rows.programme_class.eq("hybrid_DHS_MICS")]),
                ("no_documented_geographic_omission", all_rows.loc[~all_rows.documented_geographic_omission])]:
            if group.empty: continue
            for model, rows in group.groupby("model"):
                summaries.append(dict(case=case, prediction_kind=kind, cohort=cohort, model=model, countries=rows.iso3.nunique(), **{"mean_" + n:float(rows[n].mean()) for n in names}))
            p = group.pivot(index="iso3", columns="model", values=names).sort_index()
            draws = rng.integers(0, len(p), size=(NBOOT, len(p))) if len(p) >= 10 else None
            for model, ref in [("R1", "R0"), ("S", "R0"), ("R1", "S")]:
                for name in names[:-1]:
                    delta = p[(name, model)] - p[(name, ref)]
                    intervals = np.quantile(delta.to_numpy()[draws].mean(axis=1), [.025, .975]) if draws is not None else [np.nan, np.nan]
                    comparisons.append(dict(case=case, prediction_kind=kind, cohort=cohort, model=model, reference=ref, metric=name,
                        countries=len(delta), mean_difference=float(delta.mean()), median_difference=float(delta.median()),
                        countries_improved=int(delta.lt(-NUMERICAL_TIE_TOLERANCE).sum()), countries_tied_numerically=int(delta.abs().le(NUMERICAL_TIE_TOLERANCE).sum()),
                        descriptive_low=intervals[0], descriptive_high=intervals[1], bootstrap_replicates=NBOOT if draws is not None else 0))
                    for iso3, value in delta.items():
                        differences.append(dict(case=case, prediction_kind=kind, cohort=cohort, model=model, reference=ref, metric=name, iso3=iso3, difference=float(value)))
    pd.DataFrame(summaries).to_csv(OUT / "model_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_comparisons.csv", index=False)
    pd.DataFrame(differences).to_csv(OUT / "paired_country_differences.csv", index=False)
    print(pd.DataFrame(summaries).query("prediction_kind=='bounded' and cohort=='all_eligible'").to_string(index=False), flush=True)


if __name__ == "__main__":
    prefit = json.loads((BASE / "PREFIT_MANIFEST.json").read_text())
    assert not prefit["model_fitting_started"]
    for name, digest in prefit["source_sha256"].items():
        assert sha(REV.parent / name) == digest, name
    # The audit classifications were fixed before this sensitivity was fitted.
    record(REV / "05_transportability/measurement_comparability/results/FROZEN_INITIAL_CLASSIFICATION.csv")
    record(BASE / "PROTOCOL.md")
    record(BASE / "AMENDMENT_SOURCE_PROVENANCE.md")
    record(BASE / "AMENDMENT_NUMERICAL_TIES.md")
    targets, donors = prepare()
    fit(targets, donors)
