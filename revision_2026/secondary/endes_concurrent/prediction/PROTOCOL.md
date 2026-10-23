# ENDES 2023: incremental concurrent depression prediction

20 October 2026. Post-primary exploratory extension authorized by the author. This protocol is saved before fitting outcome models. Previous aggregate attitudes and separate COVID/mental-health studies remain preserved.

## Question and estimand

Do the five individual justification responses add out-of-sample prediction of concurrent PHQ-9 depressive symptom burden beyond demographics and measured IPV? Separate overall endorsement from the relative pattern of scenarios. Endorsement, experiencing abuse and symptoms remain distinct constructs. No causal attitude-to-victimisation or pandemic-effect claim is supported.

Primary target: eligible, observed, complete linked women aged 18–49 in the 2023 public INEI release. The health and violence modules select respondents separately; no official joint-intersection population weight has been verified. Therefore the primary loss gives each included respondent equal weight and targets the observed linked cohort. It is not national prevalence, national predictive performance or a population effect. Published individual and selected-health weights must not be multiplied or silently substituted for a joint weight. No nationally weighted inference will proceed without a separately documented design justification.

## Cohort and coding gate

Use the separately documented preparation workflow, official questionnaire/manual/dictionaries and source archive hash. Require unique same-person linkage, adult female eligibility, completed individual/health interviews and violence selection/privacy, ever-partnered status, all nine valid 14-day PHQ items, five binary attitude answers and complete verified emotional/physical/sexual item responses. Save exclusions, exact flags, native categories and missingness before fitting. Do not recode unknown attitudes as no, unused emotional `D103C` as zero, or lifetime-versus-recent code3 as never exposed. Religion `V130` is empty in this release and cannot enter the model.

PHQ-9 outcome: sum the nine validated 0–3 responses, range 0–27; this is symptom burden, not a clinical diagnosis. Recent IPV indicators use the verified preceding-12-month item codes; lifetime definitions remain distinct. Demographic predictors use source-native age, education, wealth, current work, partnership status, residence and region, with source-supported unavailable/unknown categorical levels explicit. Age and age squared enter numerical models. Any required change to this list must be documented before outcome fitting.

## Fixed feature blocks

- M0: demographics.
- M1: M0 plus measured recent emotional, physical and sexual IPV indicators.
- M2: M1 plus overall attitude endorsement sum (0–5).
- M3: M2 plus four fixed orthonormal Helmert contrasts of the five binary attitude answers. Together the sum and four contrasts preserve all five responses without a redundant copy of their sum.
- M4: M0 plus the same attitude sum and contrasts, omitting the IPV block.

Primary comparison: ridge M3 versus ridge M1, testing all attitude information beyond demographics/IPV. Boosting is a fixed secondary algorithm check. Secondary comparisons: M2 versus M1 (overall endorsement), M3 versus M2 (pattern beyond total), M1 versus M0 (IPV block), M4 versus M0 (attitudes before IPV adjustment). No hypothesis about effects' direction is assumed. Adding variables does not establish clinical usefulness. Intervals are descriptive paired uncertainty estimates; no unadjusted multi-model significance claim is planned.

## Fixed algorithms and validation

Fit ridge regression with alpha10 and a fixed shallow histogram gradient-boosting regressor (200 iterations, learning rate0.05, seven leaves, minimum40 observations per leaf, L2 regularization10, no early stopping). No outer-outcome architecture or hyperparameter selection. All numerical scaling, categorical encoding and missing-category treatment are fitted using training observations only. Record unseen test categories rather than treating them as known categories.

Use five grouped outer folds with three pre-set assignment seeds42/43/44. A PSU remains wholly within one fold, so women from the same household/PSU cannot occur in training and test. Build folds using PSU/stratum/region membership and sample counts, never depressive outcomes. Reuse identical folds and women for every feature block and algorithm. Record membership privately and export only aggregate fold counts. There is no later-year or other-country validation in this one-year analysis.

Primary metric: equal-person RMSE of out-of-fold predictions, bounded to the outcome's 0–27 range. Also report unbounded predictions, MAE, bias and out-of-sample R2 against each test set's mean. Average each respondent's three genuine out-of-fold predictions for the main fixed ensemble; retain each assignment's metrics. Add a training-mean reference. Never compare these errors directly with aggregate percentage-point attitude errors.

For descriptive uncertainty in paired differences, resample observed PSUs within released strata, keeping whole clusters and the already-generated out-of-fold predictions fixed (2000 bootstrap draws, seed20261020). Report percentile intervals and their conditional scope; these do not include full refitting uncertainty or establish population-design inference. If observed strata contain a singleton PSU, preserve it and report the count rather than inventing variability. No significance-only conclusion or unplanned p-value search.

## Verification and outputs

Freeze preparation sources/cohort/config/code before fitting. Independently check raw-source recoding and sample flow, all fold separations, Helmert decomposition, ridge weighted-normal-equation solutions, saved predictions/metrics and bootstrap summaries. Verify representative boosting predictions with an independent traversal of the saved trees if feasible; do not claim all boosting fits independently reoptimized. Keep respondent rows, keys, models containing sample-sensitive state and individual predictions in excluded `data_local/`.

Export aggregate numerical CSVs, source/config/environment hashes, protocol amendments if required, vector PDF/SVG plus600-dpi PNG and a plain-language result report. Integrate only supported findings into the paper, with GNN comparisons in the supplement. No placeholders or fabricated numerical results.
