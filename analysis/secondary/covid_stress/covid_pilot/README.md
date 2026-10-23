# CEBRA: exploratory COVID-19 stress benchmark

This is a supervised representation-learning pilot on the real COVIDiSTRESS cohort. CEBRA receives demographics (and, in the extended specification, historical country context) as features and only **training-country** stress scores as its continuous auxiliary variable. It is not an unsupervised discovery of mental health groups.

## Fixed design
- Six countries; 2,510 eligible women, minimum 100 per country.
- Leave one country out: every held-out country is absent from feature preprocessing, representation learning and readout fitting.
- Baseline features: age, dependents, collection day, education, employment, marital status and isolation.
- Extended features: baseline plus ten official historical female urban/rural attitude estimates (five scenarios). No COVID respondent is assigned personal norm beliefs from these estimates.
- Models: training mean, ridge, histogram gradient boosting, three-component PCA with ridge readout, three-dimensional CEBRA with ridge readout.
- CEBRA uses non-temporal `offset1-model` and continuous-label `conditional='delta'`; row order is not interpreted as time.
- Three CEBRA seeds (42, 43, 44), 1,000 training iterations per fit. One shuffled-auxiliary-label control (seed 42) per country/specification. The shuffled control still trains its readout on true training scores.
- Hyperparameters fixed before the benchmark. No held-out-country tuning or favorable-seed selection.
- Metrics: MAE and RMSE on a 0–40 stress score, plus R². Average seeds within country and then countries equally. No independent-respondent confidence interval or p-value for the six country contexts.

## Interpretation limits
Six countries are too few to establish a reliable cross-country relationship between historical norms and pandemic stress. Online respondents are not assumed nationally representative. Models can fail on new contexts; negative R² is meaningful. A seed-stability diagnostic does not establish substantive validity. Training losses are saved; 1,000 iterations is a pilot budget, not a proof of convergence or an exhaustive CEBRA search.

The embedding display shows **one fold**, not a concatenation of unaligned fold coordinates. Its held-out points are colored by measured stress only after fitting. Color gradients in training embeddings reflect supervised training and cannot be treated as independent evidence. The same fixed settings can be rerun from `scripts/benchmark_covid_embeddings.py`; the local environment and input hash are recorded.

## Outputs
- `macro_country_summary.csv`: equal-country performance, averaging CEBRA seeds within country.
- `per_country_metrics.csv`: every country/model/specification/seed result.
- `training_losses.csv`: training diagnostics for all fitted representations.
- `embedding_seed_stability.csv`: held-out Procrustes disparities across seeds.
- `benchmark.pdf` and `embeddings_example.pdf`: saved numerical-result visualizations.
- `heldout_predictions.csv`: local prediction audit.

The manuscript subsequently integrated this pilot as supplementary evidence. The 19 October 2026 method audit confirms non-temporal sampling and records the limits of this 1,000-iteration prototype. It does not establish convergence or an adequate hyperparameter choice. `../method_audit_2026_10_19/` contains the audit evidence; the manuscript's `COMMUNICATIONS_PSYCHOLOGY_PLAN.md` describes the proposed stronger evaluation. Making temperature 1.0 explicit in the code preserves the original CEBRA 0.5.0 default and does not constitute a new model fit. No broader CEBRA superiority or inferiority is established.

## Primary method documentation
https://cebra.ai/docs/usage.html
https://cebra.ai/docs/api/sklearn/cebra.html
https://cebra.ai/docs/demo_notebooks/CEBRA_best_practices.html
