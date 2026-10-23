# Separate inductive Afrobarometer graph classifier

**The fixed country-held-out classifier and its scoped independent checks are complete.** This retrospective exploratory extension tests whether a constructed predictor-similarity graph improves prediction of a woman's broad approval of wife discipline beyond the same personal predictors in logistic and non-graph controls. Earlier outcomes and logistic results were already seen; this is not preregistration or independent confirmation of the manuscript's national scenario-profile question.

All 26602 eligible women in 39 countries receive held-out-country predictions; 7251 approve in the native always/sometimes categories. Never justified is the negative category, and unknown/refused/missing responses remain excluded. This is a single general approval item, not measured violence exposure, five-scenario endorsement, depression, COVID or a brain outcome.

## Fixed design and controls

A fixed outcome-blind sample of 100 women from each donor country gives 3800 training women per holdout. Every eligible woman in the omitted country is evaluated. Donor household weights are normalized within country, then donor countries contribute equally. The capped logistic and all neural arms use identical donors, targets, predictor transformations and weighting. The unchanged earlier full-donor logistic analysis is an unmatched contextual reference, not a matched graph comparator.

D1 predictors are age, age squared, actual interview year, native education/religion/employment/residence and separate gender-role beliefs about leadership, jobs and land rights. Missingness is explicit. Numerical imputation/scaling uses capped donors only. Native source-code vocabularies are fixed before fitting; no country identifier or target label enters the predictors or graph distance.

Nearest-neighbour distance uses standardized continuous features plus full one-hot native categories (including missing), with a fixed categorical-block scale. A training node's own index is excluded. Held-out women query only the capped donors; no held-out nodes, target-to-target edges or target labels enter training. These edges represent constructed predictor similarity, not observed social contacts, causal influence or anatomical connectivity.

The node design has 83 features. The one-message mean GraphSAGE concatenates own and mean-neighbour features before a hidden ReLU and readout. Real, random-neighbour and self-message arms each allocate 1345 parameters. The own-feature MLP allocates 1361; it is approximately parameter-matched and has a different functional form. Random neighbours preserve outgoing count, not degree or distance distributions; this single null graph is not a permutation test. Self-message inputs duplicate a woman's own vector and contain no neighbour information.

All neural arms use the same fixed initializations, probability averaging, full-batch float64 Adam objective, L2 weight penalty and fixed update count. No model, neighbour metric, cap, decision threshold or restart was selected from predictive performance. Fixed updates do not certify global optimization or convergence. See PROTOCOL.md, CONFIG.json and the pre-optimization FREEZE.json.

## Held-out-country predictive scores

Lower Brier and log loss are better. AUC and AP are equal-country means of household-weighted within-country scores, not pooled global discrimination. AUC gives ties half credit; weighted AP groups equal probability scores. Country means, fit initializations and donor-cap selections are not independent new samples.

| Model | Mean country Brier | Mean country log loss | Mean within-country AUC | Mean within-country AP |
|---|---:|---:|---:|---:|
| Capped training prevalence | 0.2043 | 0.5992 | 0.5000 | 0.2820 |
| Capped D1 logistic | 0.2029 | 0.5971 | 0.5731 | 0.3335 |
| Own-feature MLP | 0.2022 | 0.5922 | 0.5763 | 0.3371 |
| Similarity GraphSAGE | 0.2021 | 0.5917 | 0.5776 | 0.3363 |
| Random-neighbour GraphSAGE | 0.2025 | 0.5932 | 0.5765 | 0.3378 |
| Self-message GraphSAGE | 0.2027 | 0.5937 | 0.5739 | 0.3360 |

Differences are first model minus second; negative values favour the first model. The intervals are pointwise paired country-bootstrap percentiles, conditional on the fitted predictions and the chosen cap/graph. They omit retraining, model development, cap-selection, graph-definition and complete survey-design/population uncertainty.

| Contrast | ΔBrier [95% conditional interval] | ΔLog loss [95% conditional interval] | Countries with lower Brier |
|---|---:|---:|---:|
| G_real − D1_cap | -0.0008 [-0.0074, 0.0065] | -0.0055 [-0.0233, 0.0122] | 23 |
| G_real − MLP | -0.0001 [-0.0009, 0.0008] | -0.0005 [-0.0025, 0.0015] | 21 |
| G_real − G_random | -0.0005 [-0.0018, 0.0007] | -0.0016 [-0.0050, 0.0011] | 24 |
| G_real − G_self | -0.0006 [-0.0018, 0.0005] | -0.0021 [-0.0058, 0.0007] | 21 |
| MLP − D1_cap | -0.0007 [-0.0068, 0.0058] | -0.0050 [-0.0211, 0.0111] | 24 |

## What the graph comparison establishes

**Matched capped logistic:** the real graph has lower mean Brier loss (difference -0.0008); the conditional interval includes zero. This comparison does not establish a graph-specific prediction gain. An inconclusive interval is not evidence that all possible graph models are equivalent.

**Own-feature nonlinear control:** the real graph has lower mean Brier loss (difference -0.0001); the conditional interval includes zero. This comparison does not establish a graph-specific prediction gain. An inconclusive interval is not evidence that all possible graph models are equivalent.

**Random-neighbour control:** the real graph has lower mean Brier loss (difference -0.0005); the conditional interval includes zero. This comparison does not establish a graph-specific prediction gain. An inconclusive interval is not evidence that all possible graph models are equivalent.

**Self-message control:** the real graph has lower mean Brier loss (difference -0.0006); the conditional interval includes zero. This comparison does not establish a graph-specific prediction gain. An inconclusive interval is not evidence that all possible graph models are equivalent.

## Fixed-threshold operating results

The fixed decision rule is approval if probability is at least 0.5. TP/TN/FP/FN below are raw pooled integer counts. Sensitivity, specificity and accuracy are separately the equal-country means of household-weighted target-country rates; they cannot be reconstructed by dividing these pooled raw counts. Prevalence and accuracy alone do not establish clinical or policy utility.

| Model | Raw TP | Raw TN | Raw FP | Raw FN | Mean weighted sensitivity | Mean weighted specificity | Mean weighted accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Capped training prevalence | 0 | 19351 | 0 | 7251 | 0.0000 | 1.0000 | 0.7180 |
| Capped D1 logistic | 54 | 19185 | 166 | 7197 | 0.0110 | 0.9912 | 0.7136 |
| Own-feature MLP | 464 | 18316 | 1035 | 6787 | 0.0485 | 0.9617 | 0.7034 |
| Similarity GraphSAGE | 532 | 18131 | 1220 | 6719 | 0.0571 | 0.9541 | 0.7005 |
| Random-neighbour GraphSAGE | 485 | 18275 | 1076 | 6766 | 0.0518 | 0.9598 | 0.7033 |
| Self-message GraphSAGE | 608 | 18110 | 1241 | 6643 | 0.0652 | 0.9506 | 0.7017 |

Country-level counts/rates, unweighted rates, fixed-grid aggregate curves and every initialization are preserved in results/. Mean prediction bias is a mean-probability diagnostic, not full calibration validation. Public curves use a fixed probability grid rather than individual probability thresholds.

## Verification, storage and scope

The independent QC report defines the checked source/cohort, donor cap, transformations, neighbour construction, stored neural forward/objective, logistic-solution and aggregate/resampling scope. Exactly one prespecified Angola/real-graph/seed101 fit is independently coded and optimized; all neural optimization paths are not independently rerun. Intermediate-step traces are not a claim that every historical optimization step was independently reconstructed. Source/raw-survey extraction verification remains in the completed source study.

Rank scores can be sensitive to machine-precision ties. Final checks score exact saved logits after verifying independent forward agreement. The documented independent-checker precision amendment changes no producer, fitted model, prediction, reported score or verification tolerance; it preserves the original one-fit replay and does not repeat neural optimization.

The donor cap constrains neural training-data exposure and limits conclusions about larger-data performance. The similarity metric, finite model family, fixed optimizers and single random-neighbour realization further bound interpretation. Native category coding does not establish cross-country psychological measurement invariance. The observed respondents and household-selection weights do not supply complete national design uncertainty.

The analysis does not alter the original attitude manuscript or earlier results. It supplies a separate computational benchmark; it cannot validate a violence–brain pathway, educational causal effect, clinical diagnosis or paired EEG/fMRI fusion. Individual memberships, matrices, neighbours, predictions and states remain in excluded data_local/. Any public archive must contain only explicitly reviewed code/aggregate files.

[Two-panel figure](figures/afrobarometer_graph_validation.pdf): colored country Brier strips with black equal-country means, and primary conditional Brier contrasts. [Earlier individual psychological complement](../individual_replication_2026_10_22/RESULTS.md). See REPORT_BINDINGS.csv for every empirical numerical mention in this report.
