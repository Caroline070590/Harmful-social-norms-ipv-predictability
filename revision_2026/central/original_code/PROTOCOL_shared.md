# Shared structural shift versus item-specific context: fixed exploratory protocol

Saved 20 October 2026 before this extension's fits. Existing target outcomes and R0/R1 predictions were already inspected. This is a retrospective post-primary **ablation and interpretation analysis**, not independent external replication, prospective validation, an item-response measurement model, or a latent psychological trait test.

## Fixed sample, source chain, and validation

Reuse the source-hashed prepared observations of `../joint_country_time/`: 169 complete all-women age 15–49 DHS rounds in 62 countries, with five national component percentages, fieldwork month dates, and four lagged structural predictors. No new respondent data, survey SEs, or outcomes are generated. Check the earlier raw/API/context and prepared-file hashes before fitting. Copy the aggregate prepared inputs and original R0/R1 predictions to this new directory for explicit provenance; preserve every old file.

Use exactly the earlier three cases: `primary` (42 latest country targets); `anomaly_removed` (same 42 targets, excluding NP2011DHS and BJ2012DHS donors); and `common_2020` (20 target surveys starting on/after 1 January 2020, donors ending before that date). Exclude all target-country attitude rounds and all donors whose fieldwork did not end strictly before target fieldwork began. Preserve five-item order and country-round weights w = 1/(eligible rounds in that donor country), with each donor country total weight one. Retain original missingness/exclusion logs and recompute fold membership. Retrospective source releases remain a limit on historical publication availability.

## Fixed models and objective

R0 is the previous item-specific intercept/calendar ridge reference. R1 adds item-specific HDI, electoral democracy, liberal democracy, and women's political empowerment coefficients. Repeat R0/R1 without retuning and require every raw prediction to match the original saved predictions within 1e-8.

S constrains the four standardized structural-context coefficients to be shared across all five outcomes, while retaining an intercept and calendar coefficient for each item. Fit by weighted penalized least squares with unpenalized intercepts and alpha = 10:

    sum_j sum_r w_r (y_rj - a_j - b_j t_r - x_r' gamma)^2
      + alpha sum_j b_j^2 + 5 alpha ||gamma||^2.

The factor five on the shared-context penalty matches the sum of five separate ridge penalties. It must not be omitted by stacking rows or changing weight normalization. No hyperparameter search, moderator, threshold, outcome selection, or alternative definition is permitted.

Impute and standardize every feature using the earlier training-only weighted means and weighted population SDs, with scale one for constant features. Calendar uses identical values in both feature sets. All outcomes use the same fully observed training rounds, weights and alpha. These conditions imply, for unbounded predictions:

    S = mean(R1) + R0 - mean(R0).

Thus S has R1's mean predicted endorsement and exactly R0's centered justification profile. Verify both identities numerically and independently verify the shared model's normal equations. For unbounded predictions, R1-minus-S full MSE must equal R1-minus-R0 profile MSE. This identity is a model property, not empirical evidence of a single psychological dimension.

## Evaluation and uncertainty

Bound all model predictions to [0,100] for the primary comparison. Save raw predictions, clipping flags, and raw metrics separately; clipping can break the exact mean/profile identities. Calculate equal-country component MAE/MSE, mean-level absolute/squared error, and centered-profile MAE/MSE. Verify full MSE = level MSE + profile MSE for all models/countries/kinds. Export predictions, parameters, membership, exclusions, missingness, country/item/model summaries, and paired country differences for S−R0, R1−S and R1−R0 in all cases.

For six fixed loss metrics use 10,000 paired country bootstrap resamples, seed 20261020, averaging country errors and never refitting models. Within case and prediction kind use the same country draws for every contrast and metric. Report descriptive 2.5/97.5 percentiles and country counts. These are not survey sampling CIs, calibrated prediction intervals, p-values, or causal evidence; correlated outer training samples and survey sampling variability remain unaccounted for. No multiple significance tests or success thresholds are introduced.

## Interpretation and deliverables

Evaluate whether allowing different structural-context effects for each justification contributes beyond a common additive shift. A shared model performing similarly to R1 supports a parsimonious interpretation of **these aggregate prediction results**, not construct invariance, individual homogeneity, mechanism, causation, or external validity. Keep all three cases, adverse results, and bounded/raw distinctions. Do not add algorithms to obtain a favorable result.

Save source/protocol/code/output hashes, tested environment and fixed seeds. Independently check the weighted shared normal equations and the derived identities; match R0/R1 to saved originals. Generate a compact CSV-based figure with vector PDF/SVG, 600-dpi PNG, readable type, and bold black panel letters. Do not edit active LaTeX, old analysis folders, or archives. Stop the affected computation and save a readiness report if required inputs fail verification.
