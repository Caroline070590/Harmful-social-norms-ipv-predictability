# Separate Afrobarometer inductive graph classifier

This retrospective exploratory study reuses the independently checked public-source cohort in `../individual_replication_2026_10_22/`. Its single broad approval item is separate from the five-scenario national profile target. No new download or manuscript edit is performed here.

Start with `PROTOCOL.md`, `CONFIG.json` and `FREEZE.json`. New optimization is permitted only after final producer/input freeze. The workflow is:

```sh
python run_analysis.py prepare
python run_analysis.py freeze
python run_analysis.py fit
python run_analysis.py summarize
```

Preparation constructs an outcome-blind fixed cap of100 women in each country, donor-only transformations, and sparse nearest/random-neighbour indices. Each country holdout uses3,800 donor women and evaluates every eligible target woman. All individual source indices, prepared matrices/graphs, fitted states and predictions remain in excluded `data_local/`. The complete original source study and active manuscript files are hash-protected. Public release must explicitly allowlist aggregate files and code, after independent QC.

The fixed comparisons are capped training prevalence, capped D1 logistic, MLP, one-message mean GraphSAGE, random-neighbour GraphSAGE and self-message GraphSAGE. Three neural initializations are averaged; no architecture, threshold or sample-cap search is allowed. Constructed edges represent predictor similarity, not observed social ties. Equal-country household-weighted losses, within-country weighted ROC/AP and weighted operating rates are distinguished from raw integer confusion counts. Fixed-prediction country bootstrap intervals omit retraining, cap-selection and full survey-design uncertainty.

The unchanged uncapped logistic study remains an unmatched contextual reference. The EEG/fMRI classifier cohorts, labels and measurement scales differ; these are not paired observations or modality comparisons. No neuroscience, violence-exposure, five-scenario replication, clinical or causal claim is supported by this graph classifier alone.

Status and empirical conclusions must be read from generated completion/QC/reports after the fixed analyses finish. This README is not a claim of completed model verification.

The original public source is [Afrobarometer Round 9, merged39country release](https://www.afrobarometer.org/survey-resource/merged-round-9-data-39-countries-2023/), with its [codebook](https://www.afrobarometer.org/survey-resource/merged-round-9-codebook-39-countries-2023/) and [usage/citation policy](https://www.afrobarometer.org/data/data-usage-and-access-policy/). Source copyright is retained. Source access/extraction, native recodes and cohort preparation are documented in the completed earlier source-study folder; this release supplies no respondent records or original SAV archive.

After independent source/model gates pass, `python complete_review.py combine` binds them and separately verifies public fold/class metadata. `python build_report.py build` and `check` create/check the author report, numerical ledger, staged S31 and vector/600dpi figure. `python complete_review.py review` completes numerical/preservation checks. Only then may `python package_release.py create` build an explicitly allowlisted code/aggregate archive. It checks ZIP CRC/member hashes/privacy and reconstructs aggregate arithmetic, report, staged TeX and all figure files after clean extraction. These aggregate checks are self-contained; full source/model optimization replay requires excluded inputs and is not self-contained.
