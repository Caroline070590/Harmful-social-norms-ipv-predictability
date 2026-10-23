# ENDES concurrent reported IPV classification

Frozen specification dated 22 October 2026, after author authorization and before any fit in this study. This is retrospective exploratory post-primary work. Previous ENDES outcomes and analyses have already been examined; the present cohort's outcome count was inspected before freezing. No confirmatory preregistration or prospective risk prediction is claimed.

The psychological question is whether five personal violence-justification responses add discrimination and probabilistic accuracy for reported past-year physical or sexual intimate partner violence (IPV), beyond demographics. A separate computational comparison tests whether predictor-similarity aggregation contributes beyond simpler classifiers and matched graph controls. The similarity graph is constructed by the analyst; it is not an observed friendship or contact network, an anatomical network or evidence of influence.

## Cohort, outcome and estimand

Retain the existing independently source-verified 10,267 complete linked ENDES release-2023 women, aged 18–49 and ever married/cohabiting. Preserve all earlier filters, including health/PHQ completeness; those restrictions make this a conditional observed cohort and may select respondents. Eight members were actually interviewed in 2024. The target is one if any of nine D105A–I physical/sexual items has native code 1 or 2 (past-year violence); all codes 0 or 3 form the negative group. Every item must have a valid native 0–3 value. Code 3 means lifetime-only violence and is negative for this recent target. Emotional abuse is not included in the target. The prepared cohort contains 954 positive respondents and 169 endorsing any justification; these counts were inspected before model fitting.

Primary performance gives equal weight to each observed woman. No validated joint DV–health population weight exists; national prevalence or national model-performance inference is not supported. Religion is entirely unavailable. Age, education and work retain native distinctions; no category merging is performed.

## Predictor blocks and excluded information

D: age, education, wealth, work, marital status, residence and region. A: D plus the five binary justification responses, without an endorsement-total duplicate. Age is centered/scaled using training participants only; categorical levels are the sorted levels present in training. Unknown held-out levels receive all-zero indicators and their counts are disclosed. Attitude inputs remain binary 0/1.

Exclude all IPV items and summaries, PHQ items and totals, other health variables, target labels, respondent/household/PSU/stratum identifiers, interview dates and survey weights from model and graph inputs. PSU is used only for splitting, donor exclusion and uncertainty clustering. Strata are descriptive only. The same demographic-only graph is reused in D and A comparisons; attitudes do not choose neighbours.

## Fixed validation and graph construction

For the primary analysis, sort unique PSU strings lexicographically, permute with NumPy default_rng(20261022), split the permuted units using array_split into five parts and assign each part a fold. All women in a PSU stay in one fold. No label is used for fold assignment, preprocessing or neighbour selection. Every woman is tested once.

Fit demographic transforms on training women only. The graph uses standardized age and full one-hot demographic categories divided by sqrt(2), so each categorical mismatch contributes one unit of squared distance. Eight nearest donors are selected by squared Euclidean distance rounded to 12 decimal places, with deterministic SHA256-derived donor ordering for ties. All donor nodes belong to the training fold; training queries exclude their entire own PSU (and hence self). Test queries connect only to training nodes. No test–test edges or target label propagation are used. Graphs are directed query-to-donor mean aggregations and have eight donors per query.

The random-neighbour control preserves eight training donors per query, excludes every query's own PSU (also for temporal test queries), and samples without replacement with the fixed graph seed plus fold. Real-neighbour selection also excludes every query's own PSU. The control preserves query degree, not donor in-degree. Its graph is common to D and A. The self-message control receives [x,x] and is a capacity-matched ordinary multilayer perceptron. Real and random GraphSAGE receive [x,mean(x_donors)]. These three neural arms have identical layers, allocated parameter counts, initial seeds, objectives and updates within each block.

## Fixed classifiers

Logistic: sklearn LogisticRegression(C=1, penalty=l2, solver=lbfgs, max_iter=2000, tol=1e-9), no class weighting. Boosting: HistGradientBoostingClassifier(max_iter=150, learning_rate=.05, max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=10, early_stopping=False), fixed seed 20261022. No hyperparameter tuning or early stopping uses held-out labels.

One small one-message mean GraphSAGE: concatenated own/mean-neighbour vectors, linear layer with eight ReLU units and logistic readout. Full-batch float64 Adam, learning rate .01, exactly 300 updates; objective is mean binary cross-entropy plus .01/2 times the sum of squared layer weights (biases unpenalized). Three initializations: 11, 29, 47. Output probabilities are averaged across those initializations before computing the primary neural performance. Save all three initialization metrics, states and objective traces. No convergence certification is implied by a fixed update count. A training-prevalence-only classifier is also evaluated. No resampling/class-balancing creates artificial prevalence.

Five folds × two blocks × three neural arms × three initializations = 90 neural fits, plus 10 logistic and 10 boosting fits.

## Fixed secondary actual-year transport

Train once on the 10,259 earlier-cohort women whose individual and health interviews both occurred in actual 2023, excluding the eight later interviews before preprocessing, graph construction and fitting. Apply the same frozen classifiers, settings and three initializations to the independently prepared actual-2024 cohort (9,847 women). Evaluate all 2024 women and the prespecified subgroup with PSUs absent from the entire earlier household source (3,250 women), using the existing independently checked membership flags. All model fitting and graph donors use strict actual-2023 records only. These are successive observed cohorts, not a verified panel or a personal-identity deduplication. The all-2024 domain may share PSUs with training; the new-PSU domain is the stricter geographic sensitivity. No model choice depends on either domain's outcomes. This adds 18 neural, two logistic and two boosting fits.

## Metrics and fixed comparisons

Primary loss: Brier score (lower is better). Secondary: log loss, ROC-AUC, average precision (AP; a stepwise precision–recall summary, not trapezoidal PR-AUC), accuracy, balanced accuracy, sensitivity, specificity, positive predictive value and confusion counts at the fixed probability threshold .5, plus mean predicted probability. Zero-denominator positive predictive value is explicitly unavailable, not zero. Domain prevalence is the descriptive no-skill AP benchmark. The separately scored primary training-prevalence model varies between folds, so its pooled AP need not equal domain prevalence and its ROC-AUC need not equal .5; temporal constant-prior predictions have AP equal to test prevalence and ROC-AUC .5. Descriptive ten-bin calibration tables use fixed probability bins, with no model recalibration.

Primary attitude contrast is logistic A minus D. Primary graph contrasts are real A minus logistic A, self A and random A. Also report A minus D for every architecture, real minus self/random in D, boosting A minus logistic A and every fitted model minus the prevalence reference. No favorable metric or comparator replaces the prespecified primary contrasts.

For Brier and log loss, 2,000 paired whole-PSU bootstrap draws (seed 20261022) resample observed PSUs globally with replacement; all models share draws within each dataset/domain. Percentile intervals condition on the fixed training sets, preprocessing, fitted models, selected graphs and predictions. They omit fitting/selection uncertainty and are not national survey-design intervals. No significance screen, optimized threshold, selected subgroup or post-hoc architecture search will be used.

## Storage and review

Only this new directory may be written by model code. Individual records, membership flags, fold IDs, matrices, graphs, predictions, states and detailed fit records remain excluded under data_local/. Public outputs are code/configuration/protocol, aggregate counts, model/domain/initialization metrics, paired loss intervals and calibration tables. No new download or installation. Independent source/graph and stored-state/score reviews have separately declared scopes; do not claim full raw/model replay merely from aggregate checking.
