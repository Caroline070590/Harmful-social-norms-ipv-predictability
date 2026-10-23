# Nested country-held-out benchmark for baseline PHQ-4 (GFS): protocol

Saved 19 October 2026 before any fit in this directory. Post-primary exploratory revision of the fixed-setting benchmark (`../gfs_public/`, results known: ridge MAE 2.644, CEBRA 2.729). Purpose: resolve the stated limitation that CEBRA and comparators were not tuned with training-only validation and CEBRA convergence was not assessed. Not preregistered. Not a search for a favourable result: the grid below is fixed and all results are reported.

## Data and target

Same GFS prediction cohort (`04_mental_health/gfs_public/analysis/prediction_cohort.csv`): eligible women, nine countries, baseline PHQ-4 (0–12). Individual covariates only (age, age², education, employment, marital status, income difficulty, urbanicity, attendance, reported COVID bereavement); historical national context is not added (its lack of gain is already reported). Same training cap: 1000 women per training country, sample seed 2026. Missing categorical values form an explicit level.

## Outer and inner validation

Outer: leave one of nine countries out; all eligible women in the held-out country are scored with survey weights. Inner: leave-one-training-country-out among the eight training countries (capped samples). Preprocessing (median imputation, scaling, one-hot) is fitted inside every inner and outer training set. The hyperparameter with the lowest mean inner weighted MAE is refitted on the full outer training set. The outer country never enters selection.

## Fixed candidate grids

- Ridge: alpha ∈ {1, 10, 100}.
- Histogram gradient boosting: max_leaf_nodes ∈ {7, 15} × l2_regularization ∈ {1, 10}; 100 iterations, min_samples_leaf 30, seed 42.
- PCA + ridge readout (alpha 10): components ∈ {3, 8}.
- CEBRA + ridge readout (alpha 10): offset1-model, delta conditioning 0.3, cosine distance, batch 128, learning rate 0.001, 32 hidden units; iterations ∈ {1000, 5000} × output dimension ∈ {3, 8} × temperature ∈ {0.1, 1.0} (8 candidates). Inner fits use seed 42.
- Reference: weighted training mean.

Final outer CEBRA: selected setting, seeds 42, 43, 44 (seed-averaged metrics), plus shuffled-auxiliary-label controls at the selected setting with the same three seeds (true training outcomes in the readout). Full loss trajectories saved for every final fit; convergence summarized by last-500 vs preceding-500 mean loss.

Weights: training weights normalized to equal country mass for readouts/regressors; CEBRA representation sampling unweighted (documented limitation); held-out metrics survey-weighted.

## Metrics and comparisons

Per held-out country weighted MAE, RMSE, R²; equal-country mean. Paired country MAE differences: tuned CEBRA vs tuned ridge, tuned CEBRA vs its shuffled control, tuned vs fixed-setting ridge/CEBRA (from the earlier benchmark). Nine countries: report sign counts; no p values are interpreted as confirmatory.

## Limits

Nine countries; inner selection uses eight-country folds; capped training samples; one bounded grid. Results cannot establish the best achievable CEBRA performance outside this grid.
