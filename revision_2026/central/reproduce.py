"""Reproduce the current national mean-versus-profile comparison from aggregates.

This portable implementation refits the fixed R0/S/R1 models for all 42 dated
country holdouts and 19 external MICS-family targets. It checks predictions,
training membership and scores against the original saved results. It also
replays the original conditional country-bootstrap summaries, preserving the
original random-number sequence across historical sensitivity cases.

It does not retrieve source reports, reconstruct survey uncertainty, tune a
model, or fit respondent data. Run with ``python reproduce.py --output-dir DIR``.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
CASE = "questionnaire_compatible"
ITEMS = ["WE_AWBT_W_BFD", "WE_AWBT_W_ARG", "WE_AWBT_W_OUT", "WE_AWBT_W_NEG", "WE_AWBT_W_REF"]
FEATURES = ["calendar", "hdi", "electoral_democracy", "liberal_democracy", "women_political_empowerment"]
LOSSES = ["MAE_pp", "level_MAE_pp", "profile_MAE_pp", "MSE_pp2", "level_MSE_pp2", "profile_MSE_pp2"]


def read(name):
    return pd.read_csv(BASE / name)


def check_frame(actual, expected, keys, columns):
    assert not actual.duplicated(keys).any()
    assert not expected.duplicated(keys).any()
    merged = actual.merge(expected, on=keys, how="outer", indicator=True,
                          validate="one_to_one", suffixes=("_new", "_saved"))
    assert merged._merge.eq("both").all(), keys
    maximum = 0.0
    for column in columns:
        new, old = merged[column + "_new"].to_numpy(float), merged[column + "_saved"].to_numpy(float)
        assert np.array_equal(np.isnan(new), np.isnan(old)), column
        error = float(np.nanmax(np.abs(new-old))) if np.isfinite(new).any() else 0.0
        maximum = max(maximum, error)
        assert error < 1e-8, (column, error)
    return {"rows": len(merged), "maximum_absolute_difference": maximum}


def fit_models(train, target):
    weight = 1.0 / train.groupby("country").survey_id.transform("size").to_numpy(float)
    x, xt = train[FEATURES].to_numpy(float), target[FEATURES].to_numpy(float)
    impute = np.array([np.average(x[np.isfinite(x[:, j]), j], weights=weight[np.isfinite(x[:, j])]) for j in range(5)])
    x = np.where(np.isfinite(x), x, impute)
    center = np.average(x, axis=0, weights=weight)
    scale = np.sqrt(np.average((x-center)**2, axis=0, weights=weight))
    scale[scale == 0] = 1
    z, zt = (x-center)/scale, (np.where(np.isfinite(xt), xt, impute)-center)/scale
    y = train[ITEMS].to_numpy(float)
    mean = np.average(y, axis=0, weights=weight)
    prediction = {}
    for model, count in [("R0", 1), ("R1", 5)]:
        zz = z[:, :count]
        coef = np.linalg.solve(zz.T @ (weight[:, None]*zz) + 10*np.eye(count), zz.T @ (weight[:, None]*(y-mean)))
        prediction[model] = mean + zt[:count] @ coef
    # Explicit shared-context system: five unpenalized intercepts, five calendar
    # slopes (penalty10), four shared context slopes (penalty5*10).
    count = len(train)
    design, test = np.zeros((5*count, 14)), np.zeros((5, 14))
    for item in range(5):
        sl = slice(item*count, (item+1)*count)
        design[sl, item], design[sl, 5+item], design[sl, 10:] = 1, z[:, 0], z[:, 1:]
        test[item, item], test[item, 5+item], test[item, 10:] = 1, zt[0], zt[1:]
    ww = np.tile(weight, 5)
    normal = design.T @ (ww[:, None]*design) + np.diag([0.0]*5+[10.0]*5+[50.0]*4)
    theta = np.linalg.solve(normal, design.T @ (ww*y.T.reshape(-1)))
    prediction["S"] = test @ theta
    np.testing.assert_allclose(prediction["S"], prediction["R1"].mean()+prediction["R0"]-prediction["R0"].mean(), atol=1e-8, rtol=0)
    return prediction, weight


def score(observed, predicted):
    error = predicted-observed
    level, profile = float(error.mean()), error-error.mean()
    out = dict(MAE_pp=float(np.abs(error).mean()), level_MAE_pp=abs(level),
               profile_MAE_pp=float(np.abs(profile).mean()), MSE_pp2=float(np.square(error).mean()),
               level_MSE_pp2=level**2, profile_MSE_pp2=float(np.square(profile).mean()), bias_pp=level)
    assert abs(out["MSE_pp2"]-out["level_MSE_pp2"]-out["profile_MSE_pp2"]) < 1e-8
    return out


def replay_bootstraps(domain, stored_metrics):
    """Historical cases are retained only to preserve their original RNG order."""
    rng, comparisons = np.random.default_rng(20261020), []
    if domain == "chronological42":
        groups = [(case, kind, stored_metrics.loc[stored_metrics.case.eq(case)&stored_metrics.prediction_kind.eq(kind)])
                  for case in ["primary", "anomaly_removed", "common_2020", CASE] for kind in ["bounded", "raw"]]
    else:
        groups = list(stored_metrics.groupby(["case", "prediction_kind"], sort=True))
        groups = [(case, kind, frame) for (case, kind), frame in groups]
    for case, kind, frame in groups:
        cohorts = [("all_eligible", frame)] if domain == "chronological42" else [
            ("all_eligible", frame), ("complete_context", frame.loc[frame.complete_context]),
            ("outside_DHS_inventory", frame.loc[~frame.in_DHS_training_inventory]),
            ("nonhybrid_MICS", frame.loc[~frame.programme_class.eq("hybrid_DHS_MICS")]),
            ("no_documented_geographic_omission", frame.loc[~frame.documented_geographic_omission])]
        for cohort, part in cohorts:
            if part.empty:
                continue
            pivot = part.pivot(index="target_country" if domain == "chronological42" else "iso3", columns="model", values=LOSSES).sort_index()
            count = len(pivot)
            draws = rng.integers(0, count, size=(10000, count)) if domain == "chronological42" or count >= 10 else None
            if case != CASE:
                continue
            for model, reference in [("R1", "R0"), ("S", "R0"), ("R1", "S")]:
                for loss in LOSSES:
                    delta = (pivot[(loss, model)]-pivot[(loss, reference)]).to_numpy(float)
                    low, high = np.quantile(delta[draws].mean(axis=1), [.025, .975]) if draws is not None else (np.nan, np.nan)
                    comparisons.append(dict(case=case, prediction_kind=kind, cohort=cohort, model=model, reference=reference,
                                            metric=loss, mean_difference=float(delta.mean()), descriptive_low=low, descriptive_high=high))
    return pd.DataFrame(comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE / "reproduced")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BASE / "SOURCE_MANIFEST.json").read_text())
    for relative, expected in manifest["packaged_sha256"].items():
        assert hashlib.sha256((BASE/relative).read_bytes()).hexdigest() == expected, relative
    rounds, external = read("inputs/dhs_rounds.csv"), read("inputs/mics_targets.csv")
    assert len(rounds) == 169 and rounds.country.nunique() == 62 and len(external) == 19
    for frame in [rounds, external]:
        for field in ["fieldwork_start", "fieldwork_end"]:
            frame[field] = pd.to_datetime(frame[field], errors="raise")
    rounds["month_end"] = rounds.fieldwork_end.dt.to_period("M").dt.end_time.dt.normalize()
    targets = rounds.loc[rounds.is_target.astype(str).str.lower().isin(["true", "1"])].sort_values("country")
    assert len(targets) == 42
    checks, summaries = {}, []
    for domain, target_frame in [("chronological42", targets), ("external19", external)]:
        predictions, metrics, members = [], [], []
        for _, target in target_frame.iterrows():
            end = rounds.fieldwork_end if domain == "chronological42" else rounds.month_end
            keep = rounds.iso3.ne(target.iso3) & end.lt(target.fieldwork_start) & rounds.survey_id.ne("NP2011DHS")
            train = rounds.loc[keep].copy()
            assert train.iso3.nunique() >= 2
            fitted, weights = fit_models(train, target)
            key = dict(case=CASE, target_survey=target.survey_id)
            members.extend(dict(**key, training_survey=row.survey_id, weight=float(weight)) for row, weight in zip(train.itertuples(), weights))
            observed = target[ITEMS].to_numpy(float)
            for model, raw in fitted.items():
                bounded = np.clip(raw, 0, 100)
                predictions.extend(dict(**key, model=model, item=item, observed_pp=float(observed[j]), raw_prediction_pp=float(raw[j]), prediction_pp=float(bounded[j]), training_rounds=len(train), training_countries=train.country.nunique()) for j, item in enumerate(ITEMS))
                metrics.extend(dict(**key, model=model, prediction_kind=kind, **score(observed, values)) for kind, values in [("raw", raw), ("bounded", bounded)])
        expected = {name: read(f"reference_results/{domain}_{name}.csv") for name in ["predictions", "country_metrics", "training_membership", "model_summary", "paired_comparisons"]}
        p, m, member = pd.DataFrame(predictions), pd.DataFrame(metrics), pd.DataFrame(members)
        pkey, mkey = ["case", "target_survey", "model", "item"], ["case", "target_survey", "model", "prediction_kind"]
        checks[domain] = dict(predictions=check_frame(p, expected["predictions"].query("case == @CASE"), pkey, ["observed_pp", "raw_prediction_pp", "prediction_pp", "training_rounds", "training_countries"]),
                              metrics=check_frame(m, expected["country_metrics"].query("case == @CASE"), mkey, LOSSES+["bias_pp"]),
                              membership=check_frame(member, expected["training_membership"].query("case == @CASE"), ["case", "target_survey", "training_survey"], ["weight"]))
        boot = replay_bootstraps(domain, expected["country_metrics"])
        original = expected["paired_comparisons"].query("case == @CASE").copy()
        if "cohort" not in original:
            original["cohort"] = "all_eligible"
        checks[domain]["conditional_bootstrap"] = check_frame(boot, original, ["case", "prediction_kind", "cohort", "model", "reference", "metric"], ["mean_difference", "descriptive_low", "descriptive_high"])
        summary = m.groupby(["model", "prediction_kind"])[LOSSES+["bias_pp"]].mean().reset_index()
        saved_summary = expected["model_summary"].query("case == @CASE").copy()
        if "cohort" in saved_summary:
            saved_summary = saved_summary.loc[saved_summary.cohort.eq("all_eligible")]
        saved_summary = saved_summary.rename(columns={"mean_"+name: name for name in LOSSES+["bias_pp"]})
        checks[domain]["equal_country_summary"] = check_frame(summary, saved_summary, ["model", "prediction_kind"], LOSSES+["bias_pp"])
        summary.insert(0, "evaluation", domain)
        summaries.append(summary)
        for name, frame in [("predictions", p), ("country_metrics", m), ("training_membership", member), ("paired_comparisons", boot)]:
            frame.to_csv(output / f"{domain}_{name}.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(output / "model_summary.csv", index=False)
    report = {"status": "passed", "case": CASE, "targets": {"chronological42": 42, "external19": 19}, "checks": checks,
              "scope": "Fixed aggregate model refitting from supplied prepared national inputs; complete current predictions, donor membership, point losses and original conditional fixed-prediction bootstrap summaries checked. No raw-source retrieval, original survey-design uncertainty, hyperparameter search or respondent/graph-neural retraining."}
    (output / "REPRODUCTION_QC.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
