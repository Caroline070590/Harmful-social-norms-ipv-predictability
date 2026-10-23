#!/usr/bin/env python3
"""Read actual release years and documented cross-year PSU overlap; no fits."""
from pathlib import Path
import hashlib
import io
import json
import zipfile
import pandas as pd

BASE = Path(__file__).resolve().parent
REV = BASE.parents[1]
OLD = REV / "04_mental_health/endes_public"
OLD_ARCHIVE = REV / "04_mental_health/psychological_violence_readiness_2026_10_20/endes2023_linkage/data_local/2023_2.zip"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source(module, year):
    archive = OLD_ARCHIVE if year == 2023 else BASE / "data_local/2024.zip"
    with zipfile.ZipFile(archive) as z:
        if year == 2023:
            names = [n for n in z.namelist() if n.endswith(module + "_2023.csv")]
            assert len(names) == 1
            raw = z.read(names[0])
        else:
            location, = [x for x in json.loads((BASE / "MODULE_LOCATIONS.json").read_text()) if x["module"] == module]
            with zipfile.ZipFile(io.BytesIO(z.read(location["outer_member"]))) as inner:
                raw = inner.read(location["inner_member"])
    f = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    f.columns = f.columns.str.upper()
    for c in f:
        f[c] = f[c].str.strip()
    return f


def main():
    years = []
    loaded = {}
    for year in [2023, 2024]:
        for module, field in [("REC0111", "V007"), ("CSALUD01", "QSINTY")]:
            f = source(module, year)
            loaded[(year, module)] = f
            for value, n in f[field].value_counts(dropna=False).items():
                years.append({"release_year": year, "module": module, "year_field": field, "interview_year": value, "records": int(n)})
        loaded[(year, "RECH0")] = source("RECH0", year)
    pd.DataFrame(years).to_csv(BASE / "interview_year_inventory.csv", index=False)
    old_house = loaded[(2023, "RECH0")]
    new_house = loaded[(2024, "RECH0")]
    old_core = loaded[(2023, "REC0111")]
    new_core = loaded[(2024, "REC0111")]
    old_prepared = pd.read_csv(OLD / "data_local/prepared_cohort.csv", dtype=str, keep_default_na=False)
    new_prepared = pd.read_csv(BASE / "data_local/prepared_cohort.csv", dtype=str, keep_default_na=False)
    psu23 = set(old_house.HV001)
    psu24 = set(new_house.HV001)
    analytic23 = set(old_prepared.psu)
    analytic24 = set(new_prepared.psu)
    assert "" not in psu23 | psu24
    flags = pd.DataFrame({"row_id": new_prepared.row_id,
                          "new_psu_vs_all_2023_households": (~new_prepared.psu.isin(psu23)).astype(int),
                          "new_psu_vs_analytic_2023": (~new_prepared.psu.isin(analytic23)).astype(int)})
    flags.to_csv(BASE / "data_local/temporal_test_flags.csv", index=False)
    train_flags = pd.DataFrame({"row_id": old_prepared.row_id,
                                "strict_2023_individual_and_health": (old_prepared.interview_year.eq("2023") & old_prepared.health_interview_year.eq("2023")).astype(int)})
    train_flags.to_csv(BASE / "data_local/temporal_training_flags.csv", index=False)
    # These comparisons check public released keys; they are not personal identity records.
    home_overlap = set(old_house.HHID) & set(new_house.HHID)
    case_overlap = set(old_core.CASEID) & set(new_core.CASEID)
    def personkeys(f):
        return set(zip(f.HHID, f.V003.map(lambda x: str(int(x)))))
    person_overlap = personkeys(old_core) & personkeys(new_core)
    old_match = old_core.loc[old_core.CASEID.isin(case_overlap), ["CASEID", "V009", "V010"]]
    new_match = new_core.loc[new_core.CASEID.isin(case_overlap), ["CASEID", "V009", "V010"]]
    birth_check = old_match.merge(new_match, on="CASEID", validate="one_to_one", suffixes=("_2023", "_2024"))
    matching_birth = birth_check.V009_2023.eq(birth_check.V009_2024) & birth_check.V010_2023.eq(birth_check.V010_2024)
    same_birth_cases = set(birth_check.loc[matching_birth, "CASEID"])
    same_analytic_keys = set(old_prepared.source_caseid) & set(new_prepared.source_caseid)
    # Verify source-record agreement for represented shared PSU geography, without
    # exporting geographic coordinates or respondent/household identifiers.
    geo23 = old_house.groupby("HV001").UBIGEO.agg(lambda s: set(s))
    geo24 = new_house.groupby("HV001").UBIGEO.agg(lambda s: set(s))
    common = psu23 & psu24
    disjoint_geo = [k for k in common if not geo23[k] & geo24[k]]
    counts = {
        "source_2023_household_PSUs": len(psu23), "source_2024_household_PSUs": len(psu24),
        "source_shared_PSUs_documented_stable_HV001": len(common),
        "shared_source_PSU_without_any_common_UBIGEO": len(disjoint_geo),
        "analytic_2023_records": len(old_prepared), "analytic_2024_records": len(new_prepared),
        "analytic_2023_PSUs": len(analytic23), "analytic_2024_PSUs": len(analytic24),
        "analytic_shared_PSUs": len(analytic23 & analytic24),
        "raw_household_key_overlap": len(home_overlap), "raw_CASEID_overlap": len(case_overlap),
        "raw_household_person_line_overlap": len(person_overlap),
        "same_raw_CASEID_and_reported_birth_month_year": int(matching_birth.sum()),
        "same_CASEID_in_both_prepared_cohorts": len(same_analytic_keys),
        "same_CASEID_and_birth_month_year_in_both_prepared_cohorts": len(same_analytic_keys & same_birth_cases),
        "strict_2023_training_records": int(train_flags.strict_2023_individual_and_health.sum()),
        "2023_release_training_records_excluded_for_later_interview": int((1 - train_flags.strict_2023_individual_and_health).sum()),
        "2024_records_in_new_PSUs_vs_all2023households": int(flags.new_psu_vs_all_2023_households.sum()),
        "2024_new_PSUs_vs_all2023households": len(analytic24 - psu23),
        "2024_records_in_new_PSUs_vs_analytic2023": int(flags.new_psu_vs_analytic_2023.sum()),
        "2024_new_PSUs_vs_analytic2023": len(analytic24 - analytic23),
    }
    pd.DataFrame([{"quantity": k, "value": v} for k, v in counts.items()]).to_csv(BASE / "cross_year_overlap_inventory.csv", index=False)
    strict = new_prepared.loc[flags.new_psu_vs_all_2023_households.eq(1)]
    strict_counts = {"records": len(strict), "PSUs": strict.psu.nunique(), "strata": strict.stratum.nunique(),
                     "regions": strict.region.nunique(), "any_attitude_endorsers": int(pd.to_numeric(strict.norm_score_0_5).gt(0).sum())}
    assert new_prepared.interview_year.eq("2024").all()
    assert new_prepared.health_interview_year.eq("2024").all()
    assert set(old_prepared.interview_year) <= {"2023", "2024"}
    assert set(old_prepared.health_interview_year) <= {"2023", "2024"}
    assert not disjoint_geo, "A same-number PSU has no shared UBIGEO: investigate stable-identity mapping"
    assert flags.row_id.is_unique
    summary = {
        "status": "ready_for_fixed_retrospective_temporal_conditional_prediction",
        "no_outcome_models_fitted_in_readiness": True,
        "strictly_later_individual_and_health_interviews": True,
        "training_date_filter_mandatory": "Exclude old2023release records unless both individual and selected-health actualinterviewyear equal2023; eight previouslyprepared records are2024interviews",
        "later_test_release_year": 2024, "training_release_year": 2023,
        "national_joint_module_population_weight_verified": False,
        "target": "Equal-person prediction among observed complete linked women18–49 ever married/cohabiting; no national population inference",
        "measurement_review": "Same verified five attitudes, nine14-dayPHQ9 items and recent12month emotional/physical/sexual act blocks; native demographic categories retained; religion and unusedD103C allblank in both releases",
        "cross_year_HV001_stability_documented": True,
        "stable_PSU_evidence": {"document": "docs/FichaTecnica2024.pdf", "pdf_page": 3,
                                "meaning": "Same-numberHV001 clusters in2023/2024 are panels"},
        "dwellings_rotation_evidence": {"document": "docs/FichaTecnica2024.pdf", "pdf_page": 7,
                                       "meaning": "Two thirds clusters retained; selected dwellings differ throughout a cluster's sample tenure"},
        "person_follow_up_or_person_identity_verification": False,
        "independence_scope": "Documented different selected dwellings, no person panel; retainedPSUs create contextual dependence. Later interviews are distinct by dates. Public keys alone cannot prove personal identity non-overlap or rule out movers.",
        "overlap_counts": counts,
        "strict_new_PSU_sensitivity": {"rule": "2024HV001 absent from every released2023householdHV001", **strict_counts,
                                       "private_flags_path": "data_local/temporal_test_flags.csv"},
        "prepared_cohort_sha256": sha(BASE / "data_local/prepared_cohort.csv"),
        "temporal_test_flags_sha256": sha(BASE / "data_local/temporal_test_flags.csv"),
        "temporal_training_flags_sha256": sha(BASE / "data_local/temporal_training_flags.csv"),
        "training_cohort_sha256": sha(OLD / "data_local/prepared_cohort.csv"),
        "script_sha256": sha(Path(__file__)),
        "require_before_fitting": ["Independent source preparationQCpassed", "Separate fixed evaluationPROTOCOL.md saved", "Source/design review complete", "No tuning on2024outcomes"]
    }
    (BASE / "READINESS.json").write_text(json.dumps(summary, indent=2) + "\n")
    ready_path = BASE / "MODEL_READY.json"
    ready = json.loads(ready_path.read_text())
    ready["temporal_status"] = "Both actual2024interviews; later-yearobservedcohort; no personpanel or guaranteedpersonalidentitydeduplication"
    ready["cross_year_PSU_independence"] = "DocumentedstableHV001 panels: primaryretains sharedPSUs; fixedsecondary excludes every2023householdsourcePSU"
    ready["temporal_validation"] = {"training_year_filter": "Both individual and health actualyear2023;10259of10267oldpreparedrows", "test_actual_year": 2024,
                                     "test_flags_path": "data_local/temporal_test_flags.csv", "test_flags_sha256": summary["temporal_test_flags_sha256"],
                                     "training_flags_path": "data_local/temporal_training_flags.csv", "training_flags_sha256": summary["temporal_training_flags_sha256"],
                                     "primary_test": "All9847eligible2024rows", "new_PSU_test_flag": "new_psu_vs_all_2023_households",
                                     "new_PSU_test_counts": strict_counts, "source_household_2023_archive_sha256": sha(OLD_ARCHIVE),
                                     "source_household_2023_PSUs": len(psu23), "readiness_path": "READINESS.json"}
    ready_path.write_text(json.dumps(ready, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
