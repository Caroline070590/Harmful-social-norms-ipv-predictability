# COVIDiSTRESS sampling and country-count sensitivities

These are post-pilot robustness checks, not independent confirmation. Minimum country sample sizes 50, 100 and 200 produce eight, six and three country contexts respectively. A separate sensitivity omits Turkey from the original six-country pilot. All are adult women's convenience samples with complete PSS-10 items and documented 2020 collection dates.

The eight-country threshold-50 demographic PCA benchmark has macro MAE 5.414 and mean R² −0.025. Adding ten historical urban/rural contextual estimates yields PCA MAE 5.588 and mean R² −0.079. Threshold and model differences do not establish useful contextual transfer. At threshold 100, contextual ridge extrapolation is especially poor; this is retained, not repaired by tuning against the held-out countries. The original float32 pilot and the float64 baseline replay can differ slightly under this unstable extrapolation.

`cohort_inventory.csv`, `summary.csv`, `per_country_metrics.csv` and `config.json` record the exact populations and models. No CEBRA parameters were tuned or rerun based on these results. The original three-seed CEBRA/null-control pilot is preserved in `../../../06_optional_cebra/covid_pilot/`.

Run the preparation script with `--minimum 50 --output-folder analysis_min50` and `--minimum 200 --output-folder analysis_min200`, then run `scripts/stress_robustness.py` from this directory's parent project context (see the full review-package instructions). PSS-10 is perceived stress, not a depression/anxiety diagnosis. The nine-country GFS PHQ-4 analysis is a separate dataset and period.
