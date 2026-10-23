# Public-source readiness: violence, COVID and women's mental health

The author confirmed both questions on 19 October 2026. Read `BOTH_QUESTIONS_DESIGN.md` for the source/measurement choices and `READINESS.json` for executed checks. The public XLSX is downloaded and audited. A subsequent exploratory pilot is executed in `colombia_exploratory/`; read its `RESULTS.md`, `PROTOCOL.md` and `QC.json`. It reports retrospective perceptions in 13 countries and direct-IPV/symptom associations in 779 eligible Colombian women, without claiming a causal effect or replacing the original attitude study.

## Commands

Run from the repository root:

```bash
python revision_2026/04_mental_health/women_covid_vaw/audit_public_sources.py
python revision_2026/04_mental_health/women_covid_vaw/download_microdata_for_readiness.py
python revision_2026/04_mental_health/women_covid_vaw/inspect_release.py
```

Network downloads need access to the publicly linked provider files. The initial CSV attempt transferred too slowly and was interrupted without a complete saved data file. The downloader now uses the public XLSX alternative with a strict two-minute wall-time limit and a file-size bound. Check its actual access/download manifest before assuming completion. No source login, author contact or restricted-data request was made. The second Harvard candidate returned HTTP403 for metadata; it remains unaudited locally.

The inspection saves country counts, released-variable value counts, source-label inconsistencies and hashes. Counts are availability diagnostics: CSV/XLSX nonmissing values may include refusals or inapplicable codes. They are not valid-outcome sample sizes or prevalence estimates. No person rows are printed or exported to review tables.

`source_documents/` holds the public questionnaires, technical note and codebook with provenance. `data_local/` holds downloaded public respondent data only; exclude the entire folder from code/review ZIPs. A public release does not make a restricted or unidentified study interchangeable with this source. No merged DHS/GFS/UN Women respondent file is constructed.

UN Women cross-country measures often combine own/known-woman experiences and use a broad mental-impact outcome. The full raw release also contains phase-specific stress/symptom items omitted from the short codebook; they cannot be pooled into a common score. Colombia-specific direct partner-violence questions support the separate pilot. Do not call the modified symptom items standard PHQ/GAD scales, clinical diagnoses or prospective causal effects. Peru's official ENDES annual survey is another candidate for actual violence and PHQ-9 observations; no Peru microdata were downloaded or linked here.

Pilot commands, from the repository root after the downloads:

```bash
python revision_2026/04_mental_health/women_covid_vaw/colombia_exploratory/prepare_pilot.py
Rscript revision_2026/04_mental_health/women_covid_vaw/colombia_exploratory/analyze_pilot.R > revision_2026/04_mental_health/women_covid_vaw/colombia_exploratory/run.log 2>&1
Rscript revision_2026/04_mental_health/women_covid_vaw/colombia_exploratory/timing_sensitivity.R > revision_2026/04_mental_health/women_covid_vaw/colombia_exploratory/timing_sensitivity_run.log 2>&1
python revision_2026/04_mental_health/women_covid_vaw/colombia_exploratory/report_and_check.py
```

The local R library is inherited from the existing GFS analysis; use the full `Public_data_analysis_and_figure_code.zip` companion and its recorded dependency/install instructions. Python requires pandas/openpyxl, NumPy, SciPy, requests and Matplotlib. Code, estimates, approximate covariance, intervals and figure exports were checked independently; original sampling PSUs/strata are not released, and the pilot assumes independent respondents for uncertainty. The 759-woman timing sensitivity is explicitly post-primary, separately documented, and does not replace the primary analysis.

Subsequent country-context work is executed in `country_profile_linkage/` and `context_prediction/`. Eleven countries have historical national female context, eight have compatible Phase II symptoms and six also have complete education/rural contrasts. Fixed held-out-country comparisons use 9,898 and 7,524 complete women respectively; all eight context/outcome additions worsen average MAE. See `context_prediction/RESULTS.md`, its original pre-fit protocol and independent QC. These fits also require scikit-learn; versions are recorded locally. Earlier review ZIPs do not include this subsequent work unless explicitly refreshed. No main-manuscript integration or new CEBRA fit has been performed.
