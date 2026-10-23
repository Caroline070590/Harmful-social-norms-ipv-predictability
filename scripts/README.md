# Historical notebook exports

These scripts preserve the earlier notebook analyses. The current manuscript
workflow and the scope of its reproducibility checks are described in the
[repository README](../README.md). The numbered exports below are not a single
command-line pipeline for the current paper.

| File | Historical purpose |
| --- | --- |
| `01_data_explanation.py` | Notebook explanation cell; no numerical analysis. |
| `02_country_record_maps.py` | Counts of aggregate source records and country maps. |
| `03_part1_full_model_RF_XGBoost_optimized.py` | Within-country aggregate regression with scenario and demographic predictors. |
| `04_part1_regression_random_search_cv.py` | **Unrelated OULAD student-withdrawal regression**, retained from the original export; not an IPV analysis and not required for this paper. |
| `05_part1_model_agreement_heatmaps.py` | Earlier model-comparison and feature-ranking agreement plots. |
| `06_part1_feature_importance_barplots.py` | Earlier within-country feature-importance plots. |
| `07_part2_demographics_only_xgboost.py` | Within-country aggregate regression with demographic predictors. |
| `08_part2_barplots.py` | Plots of the earlier demographic-predictor analysis. |
| `09_part3_hofstede_macro_analysis.py` | Earlier country summaries and exploratory cultural-dimension analyses. |
| `10_part3_external_hierarchy_analysis.py` | Earlier country-summary associations with external structural indicators. |
| `11_country_clustering_macro_indicators.py` | Earlier country clustering and descriptive plots. |
| `12_download_hofstede_data.py` | Live KaggleHub dataset acquisition; not a version-frozen replay. |
| `13_web_scraping_unwomen_measures.py` | Live acquisition of UNDP and V-Dem/Our World in Data indicators. Despite the exported filename, it does not download UN Women respondent data. |
| `legacy/` | Earlier exploratory notebook exports, retained for provenance. |

## Execution limits

- The original notebooks remain in `notebooks/`. Several Python exports contain
  multiple notebook cells, successive definitions of `main`, top-level reads,
  plotting commands and output-directory creation. Running one file may run
  several historical stages. Importing these files may also perform work.
- Input and output paths retain the original working-directory layout, including
  names such as `Violence Against Women  Girls Data.csv` and
  `results_paper_ready/`. They are not automatically mapped to the organized
  `data/` and `results/` folders. Inspect each stage and provide its original
  inputs in a separate working directory before any intentional rerun.
- Some map stages download Natural Earth data if their cache is absent. The
  acquisition scripts use live upstream resources; a newer download may differ
  from the historical input. Installing their dependencies does not guarantee
  network access, authentication or unchanged source files.
- Historical within-country row splits answer a different question from the
  current country-and-time validation. They do not establish performance in a
  previously unseen country or a later survey.
- The earlier country-summary analyses must be interpreted using their original
  population and aggregation definitions. In particular, the historical
  `Violence_part2_all` outcome is a median across retained aggregate rows,
  including different sex and justification labels. It is not the current
  women-only, five-scenario national mean. The original external-indicator merge
  also converts missing ISO codes to the string `NAN`, allowing an invalid
  missing-code match for the Democratic Republic of the Congo. The current
  manuscript reports the source audit and corresponding sensitivity analyses;
  these preserved scripts must not be treated as a corrected national analysis.

The October 2026 maintenance update only repairs the placement of repeated
`from __future__ import annotations` statements in four concatenated exports and
declares missing download dependencies. It does not alter historical model
settings, inputs, target construction, country linkage, results or figures.

## Syntax smoke check

From the repository root:

```bash
python scripts/check_syntax.py
```

This compiles the Python files in memory without importing them. A successful
check means they are syntactically valid; it does not demonstrate a complete
analysis replay or scientific verification.
