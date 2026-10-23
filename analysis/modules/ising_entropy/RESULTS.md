# Entropy of violence-justification response patterns

This post hoc descriptive extension summarizes previously fitted distributions; it introduces no new model fits or predictive test. The same **29,102** complete ENDES respondents include **433** women endorsing at least one justification. ENDES denotes Peru’s Encuesta Demográfica y de Salud Familiar. These participants, their validation partitions and bootstrap refits are not separate samples.

## What entropy describes

Joint Shannon entropy H(Y) measures uncertainty across the possible combinations of five binary answers. Total correlation is the sum of the five marginal entropies minus the joint entropy, using marginals from the **same** distribution. It describes response dependence; it is not an edge weight, a causal influence measure or the difference between entropies of separately fitted models. All entropy values use bits.

| Weighting | Distribution | Joint entropy, bits | Sum of marginal entropies, bits | Total correlation, bits | Any-endorsement probability |
|---|---|---:|---:|---:|---:|
| Released women’s weights | Empirical plug-in | 0.120545 | 0.265224 | 0.144679 | 0.011469 |
| Released women’s weights | Independent items | 0.265486 | 0.265486 | 0.000000 | 0.029979 |
| Released women’s weights | Common acceptance | 0.141443 | 0.258048 | 0.116605 | 0.012183 |
| Released women’s weights | Ising | 0.128544 | 0.265578 | 0.137034 | 0.012475 |
| Equal person | Empirical plug-in | 0.153413 | 0.330404 | 0.176991 | 0.014879 |
| Equal person | Independent items | 0.330639 | 0.330639 | 0.000000 | 0.038873 |
| Equal person | Common acceptance | 0.175817 | 0.327828 | 0.152011 | 0.015599 |
| Equal person | Ising | 0.160031 | 0.330728 | 0.170697 | 0.015736 |

The common-acceptance model also produces substantial dependence. Consequently, positive total correlation does not distinguish a pairwise network mechanism from shared endorsement propensity. Empirical plug-in entropy is a descriptive finite-sample estimate with sparse-cell bias, rather than an unbiased population reference.

## Primary weighted Ising distribution

| Quantity | Estimate | Conditional refit 2.5–97.5% range |
|---|---:|---:|
| Joint entropy H(Y), bits | 0.128544 | 0.107957 to 0.145484 |
| Total correlation, bits | 0.137034 | 0.103040 to 0.179911 |
| Any-endorsement entropy H(A), bits | 0.096790 | 0.082554 to 0.107266 |
| Within-endorser contribution qH(Y|A=1), bits | 0.031754 | 0.025646 to 0.037273 |
| Nonzero-pattern entropy H(Y|A=1), bits | 2.545322 | 2.325896 to 2.776082 |
| Any-endorsement probability q | 0.012475 | 0.010266 to 0.014162 |

The fitted joint entropy decomposes as **0.128544 = 0.096790 + 0.031754 bits**. Here A indicates any endorsement, q=P(A=1)=0.012475, and the entropy among nonzero response patterns is 2.545322 bits. Thus H(Y)=H(A)+qH(Y|A=1). The all-rejection pattern is the only response when A=0; rare endorsement makes the within-endorser contribution small in the full distribution. Conditioning on endorsement is outcome-based selection and does not define a new population effect.

## Uncertainty of each item

The marginal entropy describes an item before the other answers are known. Conditional entropy describes its remaining uncertainty given the other four answers. Their difference is mutual information, which describes association and redundancy rather than causal importance. Items remain in questionnaire order; no entropy ranking is proposed.

| Justification scenario | Marginal entropy, bits | Conditional entropy, bits | Conditional-entropy refit range | Information from other items, bits |
|---|---:|---:|---:|---:|
| Going out | 0.052356 | 0.015384 | 0.010770 to 0.018892 | 0.036972 |
| Neglecting children | 0.076443 | 0.043523 | 0.034040 to 0.052356 | 0.032920 |
| Arguing | 0.047872 | 0.010708 | 0.006138 to 0.015068 | 0.037164 |
| Refusing sex | 0.041937 | 0.005895 | 0.004427 to 0.007414 | 0.036042 |
| Burning food | 0.046971 | 0.010453 | 0.007203 to 0.012540 | 0.036518 |

## Scope and interpretation

The ranges reuse **200 existing weighted Ising refits**, with observed primary sampling units (PSUs) resampled within released strata and singleton strata fixed. They include the corresponding refit variation but cannot recover unidentified national-design variance. They are conditional sensitivity summaries, not national confidence intervals or Bayesian credible intervals. They concern Ising quantities alone: no paired comparator-model interval was calculated. Equal-person and training-fold distributions provide descriptive sensitivities, not independent replications.

Entropy is uncertainty about responses, not the amount or severity of violence, psychological health, temporal volatility or brain complexity. Low entropy can reflect dominant rejection, dominant endorsement or an overconfident model. Model-implied entropy averages under that model’s own distribution; it differs from held-out negative log likelihood, which averages over observed test responses. **Lower entropy does not demonstrate better prediction.** These summaries do not strengthen a predictive-superiority claim or change the earlier predictive findings.

The source contains complete respondents; unknown or missing responders were excluded. These distribution summaries do not validate imputation for actual nonresponse or establish a missingness mechanism. The fitted ENDES distributions do not identify country-level joint response entropy from national scenario marginals, and no brain or causal inference is supported. The separate graph neural network outputs are not assumed to form a normalized joint distribution. Existing manuscript, supplementary material and final submission folder remain unchanged.
