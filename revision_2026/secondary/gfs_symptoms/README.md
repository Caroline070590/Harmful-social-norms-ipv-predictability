# Public GFS depression/anxiety and COVID-bereavement extension

This directory uses the public non-sensitive wave-2 wide release of the Global Flourishing Study (GFS), including baseline and follow-up columns. It is a separate cohort from COVIDiSTRESS. Baseline dates in the nine overlapping country samples extend from November 2022 into January 2024; follow-up was in 2024. This cannot estimate a 2020 lockdown effect. COVID-related exposure means self-reported death of a family member or close friend. Timing and cause verification are unavailable.

The actual release contains 207,919 adults across 23 coded countries/territories. The baseline analysis has 33,421 women with four complete symptom items in nine countries overlapping the historical source; primary association models have 32,788 complete cases. PHQ-2/GAD-2 positive screens are not diagnoses. The dataset has these two-item subscales, not PHQ-9/GAD-7. The original reversed response codes are decoded from the supplied codebook.

Public release/access policy: https://www.cos.io/gfs-access-data and https://www.cos.io/gfs-faqs. Repository: https://osf.io/c8hbk/. Methods: https://doi.org/10.1186/s44263-026-00287-6. PHQ-4 original validation: https://pubmed.ncbi.nlm.nih.gov/19996233/; PHQ-2 validation: https://pubmed.ncbi.nlm.nih.gov/14583691/. Country comparisons can reflect survey mode, translation, population coverage and response differences; the screen threshold does not establish diagnostic comparability in every country.

## Reproduction from repository root

Use the existing Python environment recorded in `../../06_optional_cebra/gfs_public/environment.json` and `../../06_optional_cebra/requirements_analysis.txt`. The environment available for this analysis uses Python 3.9.7; this is the documented compatibility exception to the repository's preferred Python version. CEBRA 0.5.0 is installed locally under `../../06_optional_cebra/vendor/`, excluded from the code archive.

```bash
python revision_2026/04_mental_health/gfs_public/scripts/download_inputs.py
python revision_2026/04_mental_health/gfs_public/scripts/prepare_gfs.py
Rscript revision_2026/04_mental_health/gfs_public/scripts/install_r_dependencies.R
Rscript revision_2026/04_mental_health/gfs_public/scripts/survey_analysis.R
Rscript revision_2026/04_mental_health/gfs_public/scripts/religion_sensitivity.R
Rscript revision_2026/04_mental_health/gfs_public/scripts/association_sensitivity.R
python revision_2026/06_optional_cebra/scripts/benchmark_gfs.py
```

R 3.6.3 was supplied by the local environment; the current CRAN survey release does not support it. The compatible archived survey 3.37 and dependencies were installed in this project only. `analysis/R_sessionInfo.txt` records versions. Survey designs retain the full adult sample before subsetting to women, complete cases and paired follow-up respondents. Primary inference uses equal total country weight in the full adult design, country fixed effects, and linearized PSU/stratum variance. It is conditional on these nine countries. Singleton-stratum adjustment is explicit, with sensitivity alternatives recorded.

`PROTOCOL.md` was saved before fitting outcome models; this is an exploratory fixed analysis plan, not a registered study. Current religious-affiliation adjustment, country-deletion and singleton-variance diagnostics were added after viewing the primary association and are labelled post-primary sensitivities. Religious attendance was included in the original primary model. Religion at age 12 is not mislabelled current religion.

## Outputs

The 19 October post-primary follow-up extension is in `longitudinal_extension/`, with its pre-fit exploratory amendment in `LONGITUDINAL_AMENDMENT_2026_10_19.md`. Run `Rscript revision_2026/04_mental_health/gfs_public/scripts/longitudinal_extension.R` with output redirected to `revision_2026/04_mental_health/gfs_public/longitudinal_extension_run.log`, then `python revision_2026/04_mental_health/gfs_public/scripts/report_longitudinal_extension.py`. The generated `RESULTS.md` and `QC.json` report 16,926 complete model cases and an inconclusive baseline-symptom-adjusted bereavement coefficient. This does not alter the original baseline outputs or claim a total causal effect.

- `analysis/cohort_qc.json`, `country_coverage.csv`, `exclusion_ledger.csv`, `covariate_missingness.csv`: cohort, dates and missingness.
- `analysis/official_national_context.csv`: official national totals and identifiers; no national percentage is manufactured by averaging demographic margins.
- `analysis/weighted_country_estimates.csv`: PHQ-2/GAD-2 screening rates, mean PHQ-4 and COVID-bereavement prevalence, with design-aware uncertainty.
- `analysis/covid_bereavement_associations.csv`: two primary coefficients and age/attendance sensitivities; Holm adjustment across the two primary tests.
- `analysis/current_affiliation_sensitivity.csv`, `leave_country_out_association.csv`, `singleton_variance_sensitivity.csv`: labelled sensitivity analyses.
- `analysis/calendar_time_sensitivity.csv`: post-primary calendar-month adjustment for the later baseline period.
- `temporal_context/`: date-selected newer official context in six countries and identical-sample comparator replay; no additional CEBRA tuning.
- `analysis/longitudinal_paired_descriptives.csv`: paired baseline/follow-up PHQ-4 in 17,186 women using longitudinal weights. This is descriptive; attrition and measurement comparability limit interpretation. Country intervals are pointwise, not familywise discoveries.
- `../../06_optional_cebra/gfs_public/`: held-out-country predictive benchmarks, seed/null controls, embedding and diagnostics. CEBRA receives only training outcomes; the held-out country is never used for fitting or tuning. CEBRA representation training is unweighted, while baselines, readouts and held-out metrics use weights.

Raw public respondent data, local libraries, model objects and respondent-level derived files are kept locally and excluded from the review archive. The downloader permits reproduction from the public sources. No authorized original DHS respondent data were supplied. These contextual analyses do not replace missing individual attitudes, IPV exposure or cluster normative-climate analyses.
