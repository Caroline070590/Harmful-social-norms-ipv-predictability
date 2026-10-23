#!/usr/bin/env python3
"""Prepare source-verified ENDES2024 later-year observed intersection; no fits.

Adapted from the frozen ENDES2023 preparer; source guard records provenance.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import platform
import zipfile
import pandas as pd

BASE = Path(__file__).resolve().parent
REV = BASE.parents[1]
DEFAULT_ARCHIVE = BASE / "data_local/2024.zip"
ATT_RAW = ["V744" + c for c in "ABCDE"]
ATT_OUT = ["att_goes_out", "att_neglects_children", "att_argues", "att_refuses_sex", "att_burns_food"]
PHQ = ["QS700" + c for c in "ABCDEFGHI"]
EMO = ["D103A", "D103B", "D103D"]
PHYS = ["D105" + c for c in "ABCDEFG"]
SEX = ["D105H", "D105I"]
COVARS = {"age": "V012", "education": "V106", "wealth": "V190", "work": "V714", "marital": "V501", "residence": "V025", "region": "V024"}
VALID_COVARS = {"V106": set("0123"), "V190": set("12345"), "V714": set("01"), "V501": set("12345"), "V025": set("12"), "V024": {str(i) for i in range(1, 26)}}


def digest(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = ap.parse_args()
    private = BASE / "data_local"
    private.mkdir(exist_ok=True)
    fields = {
        "REC0111": ["ID1", "CASEID", "HHID", "V001", "V003", "V007", "V008", "V012", "V015", "V021", "V022", "V023", "V024", "V025", "V044", "V005", "V106", "V130", "V190"],
        "RE516171": ["CASEID", "V501", "V502", "V714", "V731"] + ATT_RAW,
        "REC84DV": ["CASEID", "D103C"] + EMO + PHYS + SEX,
        "CSALUD01": ["HHID", "QHCLUSTER", "QSNUMERO", "QS20C", "QSSEXO", "QS23", "QSRESULT", "QSRESINF", "QSMEF", "PESO15_AMAS", "QSINTY", "QSINTM"] + PHQ,
        "RECH0": ["HHID", "HV001", "HV022", "UBIGEO", "NCONGLOME1"],
    }
    frames, sources = {}, []
    locations = json.loads((BASE / "MODULE_LOCATIONS.json").read_text())
    with zipfile.ZipFile(args.archive) as z:
        assert z.testzip() is None
        for module, requested in fields.items():
            match = [x for x in locations if x["module"] == module]
            assert len(match) == 1, module
            location = match[0]
            outer_raw = z.read(location["outer_member"])
            with zipfile.ZipFile(io.BytesIO(outer_raw)) as inner:
                assert inner.testzip() is None
                raw = inner.read(location["inner_member"])
            frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, encoding="utf-8-sig")
            frame.columns = frame.columns.str.upper()
            assert set(requested) <= set(frame), (module, set(requested) - set(frame))
            for col in requested:
                frame[col] = frame[col].str.strip()
            frames[module] = frame[requested].copy()
            sources.append({"module": module, **location, "outer_member_sha256": digest(outer_raw), "source_rows": len(frame), "source_columns": list(frame), "member_sha256": digest(raw), "read_fields": requested})
    core, health = frames["REC0111"], frames["CSALUD01"]
    keys = []
    for name in ["REC0111", "RE516171", "REC84DV"]:
        x = frames[name]
        assert x.CASEID.ne("").all() and not x.CASEID.duplicated().any()
        keys.append({"module": name, "key": "CASEID", "rows": len(x), "duplicate_rows": 0, "missing_keys": 0})
    for x, field, label in [(core, "V003", "REC0111"), (health, "QSNUMERO", "CSALUD01")]:
        assert x[field].str.fullmatch(r"\d+").all()
        x["PERSON_LINE"] = x[field].map(lambda v: str(int(v)))
        assert x.HHID.ne("").all() and not x.duplicated(["HHID", "PERSON_LINE"]).any()
        keys.append({"module": label, "key": "HHID + PERSON_LINE", "rows": len(x), "duplicate_rows": 0, "missing_keys": 0})
    household = frames["RECH0"]
    assert household.HHID.ne("").all() and not household.HHID.duplicated().any()
    joined = core
    link = []
    for name in ["RE516171", "REC84DV"]:
        joined = joined.merge(frames[name], how="left", on="CASEID", validate="one_to_one", indicator="JOIN_" + name)
        link.extend({"join": name, "status": str(k), "rows": int(v)} for k, v in joined["JOIN_" + name].value_counts().items())
    joined = joined.merge(health, on=["HHID", "PERSON_LINE"], how="left", validate="one_to_one", indicator="JOIN_HEALTH")
    link.extend({"join": "HEALTH", "status": str(k), "rows": int(v)} for k, v in joined.JOIN_HEALTH.value_counts().items())
    joined = joined.merge(household, on="HHID", how="left", validate="many_to_one", indicator="JOIN_HOUSEHOLD")
    link.extend({"join": "HOUSEHOLD", "status": str(k), "rows": int(v)} for k, v in joined.JOIN_HOUSEHOLD.value_counts().items())
    ages = pd.to_numeric(joined.V012, errors="coerce")
    hages = pd.to_numeric(joined.QS23, errors="coerce")
    filters = [
        ("adult_individual_age_18_49", ages.between(18, 49)),
        ("individual_interview_year_strictly_2024", joined.V007.eq("2024")),
        ("completed_individual_V015_1", joined.V015.eq("1")),
        ("unique_person_matched_health_female_18_49", joined.JOIN_HEALTH.eq("both") & joined.QSSEXO.eq("2") & hages.between(18, 49)),
        ("health_interview_year_strictly_2024", joined.QSINTY.eq("2024")),
        ("completed_health_respondent_QSRESINF_1", joined.QSRESINF.eq("1")),
        ("completed_health_questionnaire_QSRESULT_1", joined.QSRESULT.eq("1")),
        ("selected_and_interviewed_DV_privacy_V044_1", joined.V044.eq("1")),
        ("ever_married_or_cohabited_V502_1_or_2", joined.V502.isin(["1", "2"])),
        ("all_nine_PHQ14day_codes_0_to_3", joined[PHQ].isin(["0", "1", "2", "3"]).all(axis=1)),
        ("all_five_binary_attitudes_unknown8_excluded", joined[ATT_RAW].isin(["0", "1"]).all(axis=1)),
        ("all_three_observed_emotional_items_valid", joined[EMO].isin(["0", "1", "2", "3"]).all(axis=1)),
        ("all_seven_physical_items_valid", joined[PHYS].isin(["0", "1", "2", "3"]).all(axis=1)),
        ("both_sexual_items_valid", joined[SEX].isin(["0", "1", "2", "3"]).all(axis=1)),
        ("all_native_baseline_categories_valid", pd.concat([joined[c].isin(v) for c, v in VALID_COVARS.items()], axis=1).all(axis=1)),
        ("PSU_strata_and_household_match_available", joined.JOIN_HOUSEHOLD.eq("both") & joined.V001.ne("") & joined.V022.ne("")),
    ]
    keep = pd.Series(True, index=joined.index)
    flow = [{"stage": "all_individual_module_records", "remaining": len(joined), "new_exclusions": 0}]
    flags = pd.DataFrame({"CASEID": joined.CASEID})
    for label, condition in filters:
        before = int(keep.sum()); keep &= condition
        flow.append({"stage": label, "remaining": int(keep.sum()), "new_exclusions": before - int(keep.sum())})
        flags[label] = condition.astype(int)
    flags.to_csv(private / "exclusion_flags.csv", index=False)
    cohort = joined.loc[keep].copy()
    assert len(cohort) > 0, "No eligible complete observed-respondent cohort"
    assert pd.to_numeric(cohort.V012).equals(pd.to_numeric(cohort.QS23))
    for a, b in [("V001", "QHCLUSTER"), ("V001", "HV001"), ("V022", "HV022")]:
        assert (pd.to_numeric(cohort[a]) == pd.to_numeric(cohort[b])).all(), (a, b)
    assert frames["REC0111"].V130.eq("").all()
    assert frames["REC84DV"].D103C.eq("").all()
    out = pd.DataFrame(index=cohort.index)
    out["row_id"] = cohort.CASEID.map(lambda v: hashlib.sha256(("ENDES2024:" + v).encode()).hexdigest())
    out["source_caseid"] = cohort.CASEID
    out["psu"] = cohort.V001
    out["stratum"] = cohort.V022
    out["survey_release_year"] = 2024
    out["source_hhid"] = cohort.HHID
    out["source_person_line"] = cohort.PERSON_LINE
    out["household_ubigeo"] = cohort.UBIGEO
    out["household_nconglome1"] = cohort.NCONGLOME1
    out["individual_interview_CMC"] = cohort.V008
    out["health_interview_month"] = cohort.QSINTM
    out["interview_year"] = cohort.V007
    out["health_interview_year"] = cohort.QSINTY
    for label, source in COVARS.items():
        out[label] = cohort[source]
    out["age"] = out.age.astype(int)
    out["work_12month_native"] = cohort.V731
    for raw, label in zip(ATT_RAW, ATT_OUT):
        out[label] = cohort[raw].astype(int)
    out["norm_score_0_5"] = out[ATT_OUT].sum(axis=1)
    out["phq9_total"] = cohort[PHQ].astype(int).sum(axis=1)
    for label, raw_fields in [("emotional", EMO), ("physical", PHYS), ("sexual", SEX)]:
        out[label + "_recent"] = cohort[raw_fields].isin(["1", "2"]).any(axis=1).astype(int)
        out[label + "_lifetime"] = cohort[raw_fields].isin(["1", "2", "3"]).any(axis=1).astype(int)
    out["analysis_weight_equal"] = 1.0
    out["released_health_weight"] = pd.to_numeric(cohort.PESO15_AMAS, errors="coerce") / 1_000_000
    out["released_woman_weight"] = pd.to_numeric(cohort.V005, errors="coerce") / 1_000_000
    for col in PHQ + EMO + PHYS + SEX + ["V044", "V015", "V502", "QSRESULT", "QSRESINF"]:
        out[col] = cohort[col]
    assert out.row_id.is_unique and out.phq9_total.between(0, 27).all() and out.norm_score_0_5.between(0, 5).all()
    target = private / "prepared_cohort.csv"
    out.to_csv(target, index=False)
    pd.DataFrame(flow).to_csv(BASE / "cohort_flow.csv", index=False)
    pd.DataFrame(keys).to_csv(BASE / "key_inventory.csv", index=False)
    pd.DataFrame(link).to_csv(BASE / "linkage_counts.csv", index=False)
    categories = []
    category_fields = list(VALID_COVARS) + ["V044", "V015", "V502", "V130", "D103C", "V731", "QSRESULT", "QSRESINF", "V007", "V008", "QSINTY", "QSINTM"] + ATT_RAW + EMO + PHYS + SEX + PHQ
    for scope, f in [("all_joined_individual_records", joined), ("prepared_conditional_prediction_cohort", cohort)]:
        for col in category_fields:
            for value, n in f[col].fillna("[unmatched_module]").value_counts(dropna=False).items():
                categories.append({"scope": scope, "field": col, "native_code": value, "records": int(n)})
    pd.DataFrame(categories).to_csv(BASE / "category_inventory.csv", index=False)
    missingness = []
    for col in category_fields + ["V001", "V022", "HV022", "PESO15_AMAS", "V005"]:
        for scope, f in [("all_joined_individual_records", joined), ("prepared_conditional_prediction_cohort", cohort)]:
            s = f[col]
            missingness.append({"scope": scope, "field": col, "records": len(s), "blank_or_unmatched": int((s.isna() | s.eq("")).sum())})
    pd.DataFrame(missingness).to_csv(BASE / "missingness.csv", index=False)
    summary = {"prepared_records": len(out), "PSUs": out.psu.nunique(), "strata": out.stratum.nunique(), "regions": out.region.nunique(), "source_archive_sha256": digest(args.archive.read_bytes()), "prepared_cohort_sha256": digest(target.read_bytes()), "health_weight_positive": int(out.released_health_weight.gt(0).sum()), "woman_weight_positive": int(out.released_woman_weight.gt(0).sum()), "religion_all_missing": True, "D103C_all_missing": True, "no_models_fitted": True}
    ready = {"status": "ready_for_equal_person_conditional_prediction_only", "population_inference_ready": False,
             "cohort_path": "data_local/prepared_cohort.csv",
             "temporal_status": "Both individual and selected-health interviews strictly2024; different selected dwellings by documented cluster-rotation design; not longitudinal people",
             "cross_year_PSU_independence": "Not established: design retains two thirds of clusters; numeric cross-year code equality alone is not verified geographic identity", "summary": summary,
             "target": "Observed complete linked women18–49, ever married/cohabiting, completed individual/health/DVprivacy interviews, native baseline/item responses valid; not nationally representative joint-module population",
             "numeric_covariates": ["age"], "categorical_covariates": ["education", "wealth", "work", "marital", "residence", "region"],
             "attitude_items": ATT_OUT, "norm_score": "norm_score_0_5", "outcome": "phq9_total", "IPV_primary_recent": ["emotional_recent", "physical_recent", "sexual_recent"], "IPV_lifetime_sensitivity": ["emotional_lifetime", "physical_lifetime", "sexual_lifetime"],
             "group_column": "psu", "stratum_column": "stratum", "region_column": "region", "id_column": "row_id",
             "primary_weight_column": "analysis_weight_equal", "primary_weight_estimand": "equal-person error conditional on observed eligible complete respondent intersection",
             "released_weight_sensitivity": {"columns": ["released_health_weight", "released_woman_weight"], "joint_population_weight_validated": False, "restriction": "Only labelled reweighting sensitivity for this conditional cohort; not national prevalence/inference or proven population prediction performance; never multiply weights"},
             "depression_items_are_not_predictors": PHQ, "religion": "V130 entirely blank; omitted with explicit availability limitation", "unused_emotional_item": "D103C never filled with zero",
             "source_field_mappings": {"covariates": COVARS, "attitudes": dict(zip(ATT_OUT, ATT_RAW)), "PHQ": PHQ, "emotional": EMO, "physical": PHYS, "sexual": SEX},
             "IPV_coding": {"recent": "Any item1/2; all observed0/3 yields0", "lifetime": "Any item1/2/3; all0 yields0", "missing": "Exclude any required item outside0–3; unusedD103C excluded from exposuredefinition, not imputed"}}
    (BASE / "MODEL_READY.json").write_text(json.dumps(ready, indent=2) + "\n")
    (BASE / "SOURCE_INVENTORY.json").write_text(json.dumps({"archive": str(args.archive.relative_to(REV)), "archive_sha256": summary["source_archive_sha256"], "modules": sources, "environment": {"python": platform.python_version(), "pandas": pd.__version__}, "script_sha256": digest(Path(__file__).read_bytes())}, indent=2) + "\n")
    (BASE / "PREPARATION_QC.json").write_text(json.dumps({"status": "passed", "scope": "Deterministic source/key/eligibility/item/covariate construction; no model fit or population-weight validation", "summary": summary, "checks": {"unique_keys": True, "all_required_source_codes_valid_in_cohort": True, "health_core_age_agree": True, "PSU_and_stratum_modules_agree": True, "PHQ_score_bounds": True, "attitude_score_bounds": True, "privacy_eligibility_filters": True, "source_ids_only_data_local": True}}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
