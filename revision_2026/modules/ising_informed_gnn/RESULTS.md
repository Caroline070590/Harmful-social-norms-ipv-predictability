# Ising-informed GNN for missing justification responses

22 October 2026. Retrospective exploratory analysis; existing manuscripts and final export remain unchanged.

## Question and population

Can a small item graph neural network (GNN) predict one missing justification answer after the other four answers are known? Five nodes represent the questionnaire items, not women, countries or brain regions. This is not prediction before questionnaire administration or forecasting violence.

The complete-response cohort from Peru's Encuesta Demográfica y de Salud Familiar (ENDES) contains **29,102 women**. They contribute five masked tasks each, with all tasks from one woman kept in the same whole-primary-sampling-unit (PSU) validation fold. Only **433 women** endorse any justification. Masks are computational tasks, not additional independent respondents. Matching training-only Ising associations define each fold's graph; all held-out target answers are hidden before any input or message is constructed.

## Primary held-out results

Released-weighted partition 42 is primary. Lower Brier and negative log likelihood (NLL) are better. An information-matched multilayer perceptron (MLP) receives the same four known answers as the graph models. Neural predictions average two fixed initializations; no validation-based architecture or threshold selection occurs.

| Model | Conditional-item Brier | Conditional-item NLL |
|---|---:|---:|
| Independent items | 0.006022 | 0.036834 |
| Common acceptance | 0.001912 | 0.010868 |
| Exact Ising | 0.001949 | 0.011401 |
| Conditional logistic | 0.001942 | 0.011431 |
| Ising GNN | 0.001929 | 0.011248 |
| Permuted Ising GNN | 0.001892 | 0.011087 |
| Self-message neural | 0.006022 | 0.036834 |
| Conditional MLP | 0.001901 | 0.011042 |

Prespecified Ising-GNN-minus-reference contrasts:

| Reference | Brier difference | Conditional 95% range | NLL difference |
|---|---:|---:|---:|
| Exact Ising | -0.000020 | -0.000051 to +0.000001 | -0.000152 |
| Common acceptance | +0.000017 | -0.000067 to +0.000117 | +0.000380 |
| Conditional logistic | -0.000013 | -0.000030 to +0.000002 | -0.000182 |
| Conditional MLP | +0.000028 | -0.000007 to +0.000077 | +0.000207 |
| Permuted Ising GNN | +0.000037 | -0.000016 to +0.000119 | +0.000162 |
| Self-message neural | -0.004092 | -0.005705 to -0.002929 | -0.025586 |
| Independent items | -0.004092 | -0.005705 to -0.002929 | -0.025586 |

Against the information-matched conditional MLP, the primary Brier difference is +0.000028. The conditional range includes zero. Against the permuted Ising alignment, the difference is +0.000037. The conditional range includes zero.

The real-versus-permuted comparison tests alignment of the training Ising weights to the questionnaire identities. The permutation retains weighted topology up to relabelling. The conditional MLP and logistic models see all four known answers, providing information-matched non-graph references. The self-message missing node cannot see those answers, so a favorable comparison against that control alone would not establish a topology advantage.

## Weighting and partition sensitivity

| Weighting | Partition | GNN minus exact Ising Brier | GNN minus permuted Brier | GNN minus conditional MLP Brier |
|---|---:|---:|---:|---:|
| Released women’s weights | 42 | -0.000020 | +0.000037 | +0.000028 |
| Released women’s weights | 43 | -0.000037 | +0.000021 | +0.000027 |
| Released women’s weights | 44 | -0.000025 | +0.000002 | +0.000011 |
| Equal person | 42 | +0.000021 | -0.000038 | -0.000024 |
| Equal person | 43 | +0.000022 | -0.000042 | -0.000012 |
| Equal person | 44 | +0.000025 | -0.000048 | -0.000027 |

The real-versus-MLP and real-versus-permuted directions reverse under equal-person refits, leaving no stable advantage of the Ising alignment. The alternate partitions reuse the same women and are not independent replications. Equal-person sensitivities refit every new model. Item-specific, initialization-specific and endorser-only descriptive scores remain in the complete aggregate tables. Endorser-only subsets use outcomes to define the subset and do not replace the full-cohort estimand.

## Model and inferential limits

All four neural arms have 137 trainable parameters and use the same fixed 400 float64 Adam updates, training-marginal offsets and unit-SD Gaussian weight prior scaled by training respondent count. Fixed-step training does not certify convergence or global optimality. The original penalty proposal was corrected by a documented amendment before any fitting; it was not tuned against performance. The graph is signed and row-normalized by absolute weight sum. All reused training interactions are positive; the minimum is 0.841115.

The neural and separate conditional-logistic outputs need not define a compatible normalized joint distribution. Only missing-item conditional scores are reported; no neural whole-pattern likelihood is claimed. The masks are artificial and deterministic among observed complete responders. They do not validate imputation for actual unknown answers or nonresponse, whose respondents were excluded, or identify a missingness mechanism. This conditional reconstruction task is separate from national mean/profile transportability. A dense five-node graph, sparse endorsements and a single-country complete-response cohort limit generalization. All published comparisons concern fixed learned associations and architectures, not causal influence or a psychological mechanism.

Paired ranges use 2,000 resamples of observed PSUs within released strata with fitted predictions fixed. Eighteen singleton strata remain fixed. These conditional ranges omit fitting uncertainty and unidentified singleton-stratum national variance; they are not national confidence intervals or clinical uncertainty. The 2023 release includes 34 actual-2024 records; this is not a temporal test.

## Verification and reuse

Producer fitting and independent verification have separately documented scopes. Stored-state replay does not itself reproduce every optimization. Individual records, folds, source states and predictions remain excluded under data_local/. The public archive supports aggregate replay; full original-source/model replay requires upstream cohort and prior Ising dependencies. No new download, installation, manuscript integration or journal action occurred.
