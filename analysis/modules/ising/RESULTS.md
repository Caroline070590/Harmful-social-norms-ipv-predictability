# Exact Ising network of five violence-justification responses

22 October 2026. This retrospective exploratory study is separate from the active manuscript and independent final submission folder. Numerical estimates below are generated from aggregate result tables. Independent review has its own recorded scope; fitting completion alone is not certification.

## What was tested

Can pairwise conditional response associations explain held-out combinations of five justification answers beyond independent items and a single common acceptance tendency? Nodes are questionnaire items. None of these models predicts future violence, identifies interpersonal influence or measures a neural pathway.

The previously audited broader ENDES women’s-module cohort contains **29,102 complete adult respondents**, **3,249 observed PSUs** and **240 released strata**. Only **433 women** endorse any item. **23 of the 32 possible patterns** occur, including the all-zero pattern; nine patterns have no observed respondent. There are 34 interviews coded actual 2024 within the 2023 release. Unknown responses were excluded rather than recoded as rejection. The broader attitude cohort is distinct from the 10,267-woman health-linked cohort used previously.

The models use the exact 32-state joint distribution: independent items (five intercepts), one common logistic-normal acceptance factor (five intercepts and one common scale), and regularized Ising (five intercepts and ten pairwise interactions). Fixed penalties and all validation choices were frozen before fitting. The common-factor alternative is essential: response co-occurrence alone does not distinguish a network mechanism from a shared tendency.

## Held-out results

All models use the same whole-PSU fivefold split. Seed 42 is primary; seeds 43/44 are sensitivity partitions of the same women. Primary losses use released women’s weights; the equal-person sensitivity refits every model. Lower loss is better.

| Model | Joint-pattern negative log likelihood | 32-pattern Brier | Marginal item Brier | Conditional item Brier |
|---|---:|---:|---:|---:|
| Independent items | 0.184171 | 0.023260 | 0.006022 | 0.006022 |
| Common acceptance factor | 0.087525 | 0.022784 | 0.006022 | 0.001912 |
| Ising network | 0.087117 | 0.022782 | 0.006022 | 0.001949 |

Primary paired Ising-minus-reference contrasts:

| Reference | Joint-NLL difference | 95% conditional range | Conditional-Brier difference |
|---|---:|---:|---:|
| Independent items | -0.097054 | -0.130624 to -0.073480 | -0.004073 |
| Common acceptance factor | -0.000408 | -0.002609 to +0.002464 | +0.000037 |

The Ising model represents response co-occurrence much better than independent items. The crucial comparison against one common acceptance factor is less decisive: its weighted primary joint-NLL interval includes zero, and conditional-item scores are slightly worse. Equal-person refits show a more favorable joint-NLL contrast, so the weighting sensitivity must remain visible. These results support dependent endorsement responses while leaving a network-specific advantage over shared acceptance uncertain.

The primary comparison concerns whole response-pattern probability. Conditional item scores deliberately observe the other four responses when predicting the remaining answer; they evaluate missing-response prediction after partial questionnaire completion. They are not pre-questionnaire prediction or future-violence forecasting. Marginal item scores predict the overall item probabilities for women in each held-out fold and carry no person-specific demographic information.

### Sensitivity partitions and weighting

| Weighting | Partition | Ising minus independent NLL | Ising minus common-factor NLL |
|---|---:|---:|---:|
| released_weight | 42 | -0.097054 | -0.000408 |
| released_weight | 43 | -0.096704 | -0.000150 |
| released_weight | 44 | -0.097030 | -0.000416 |
| equal_person | 42 | -0.121021 | -0.001678 |
| equal_person | 43 | -0.120941 | -0.001698 |
| equal_person | 44 | -0.121027 | -0.001760 |

These partitions reuse the same participants and are not independent replications. All scores, including endorser-only descriptive subsets and unfavorable contrasts, are retained in the tables. Rare endorsement dominates absolute average losses; a small number or high rejection accuracy does not establish useful identification of women who endorse a justification.

## Conditional response network

Item key: V744A, going out without telling him; V744B, neglecting children; V744C, arguing with him; V744D, refusing sex; V744E, burning food.

All ten Ising interactions are reported, with full-cohort weighted coefficients and 200 stratified observed-PSU refits. The log conditional odds ratio is on the binary 0/1 coding scale. Positive values indicate co-endorsement conditional on the other three answers; unmeasured shared dispositions or measurement processes can still generate these associations. No edge is selected by a significance threshold and no centrality ranking is claimed.

| Items | Log conditional odds ratio | Refitted bootstrap 2.5–97.5% range | Fraction positive |
|---|---:|---:|---:|
| V744A–V744B | 3.172 | 1.298 to 4.568 | 1.000 |
| V744A–V744C | 4.171 | 0.704 to 6.459 | 1.000 |
| V744A–V744D | 1.117 | 0.332 to 2.472 | 0.990 |
| V744A–V744E | 3.402 | 1.903 to 4.980 | 1.000 |
| V744B–V744C | 2.915 | 1.434 to 4.661 | 1.000 |
| V744B–V744D | 2.604 | 1.513 to 3.512 | 1.000 |
| V744B–V744E | 3.032 | 1.538 to 3.965 | 1.000 |
| V744C–V744D | 3.562 | 2.697 to 4.399 | 1.000 |
| V744C–V744E | 1.132 | 0.091 to 2.926 | 0.975 |
| V744D–V744E | 3.929 | 2.672 to 4.867 | 1.000 |

## Numerical and inferential limits

All 96 held-out/full-cohort fits and 200 edge-refit models completed under the fixed design. The common-factor scale ranges from 7.122 to 8.076; 0 fits reach the prespecified upper-bound diagnostic. Every common-factor fit is checked using 3,841 quadrature nodes against the fitted 1,921-node calculation. Maximum probability discrepancy is 2e-15. The earlier 61-node proposal was corrected by a documented numerical amendment before any outcome fit.

Primary predictive uncertainty uses 2,000 paired observed-PSU resamples with fitted predictions held fixed. Edge ranges instead refit the weighted full network in 200 resamples. In both procedures 18 singleton strata remain fixed. Neither procedure recovers their unidentified national-design variance; the ranges are conditional sensitivity summaries for these observed complete respondents, not national confidence intervals, causal tests or calibrated Bayesian credible intervals. Predictive ranges do not include full model retraining.

Survey weights are normalized within each training set to sum to its respondent count before adding fixed Gaussian penalties. This is a working weighted likelihood/MAP objective; the regularized Ising analysis does not sample a Bayesian posterior. The common-factor model is a bounded one-factor comparator, not proof of a unitary psychological construct. Better Ising scores would establish added representation of these responses relative to these fixed alternatives, not causal item interaction.

Nine unobserved patterns, sparse endorsement, complete-case eligibility and one-country scope limit transportability. No education-specific network, network intervention or time transition is estimated. The graph layout is schematic. Existing national-context, mental-health and separate EEG/fMRI analyses are unchanged.

## Primary methodological and theoretical references

The Ising precedent is [van Borkulo et al., Scientific Reports (2014)](https://doi.org/10.1038/srep05918), which introduced eLasso network estimation for binary data. Here the five-item space is small enough to use an exact 32-state joint likelihood with fixed Gaussian regularization, rather than eLasso or data-selected sparsity. The [Causal Attitude Network model, Dalege et al., Psychological Review (2016)](https://doi.org/10.1037/a0039802) provides a theoretical motivation for studying attitude relationships. Our cross-sectional response likelihood does not identify its proposed causal interactions or dynamics, and the common-acceptance comparator remains an observational alternative.

## Figure legend

Exact response-pattern prediction and the descriptive Ising network. (a) Weighted empirical probability by number of endorsed scenarios and full-cohort fitted distributions; the vertical axis is logarithmic to show rare endorsements. (b) Held-out joint-NLL differences for Ising relative to independent items (upper axis) and the common acceptance factor (lower zoomed axis), separately for released-weighted and equal-person fits; the two axes have different horizontal scales. Black points and bars show seed 42 and conditional fixed-prediction 95% ranges, with colored alternate-partition points. (c) All ten weighted full-cohort interactions with 200-refit observed-PSU percentile ranges. (d) Schematic item network; edge width reflects absolute fitted interaction and color denotes sign. Layout and edges do not represent social ties, anatomy or causal influence. Abbreviations: ENDES, Encuesta Demográfica y de Salud Familiar; Ising, pairwise binary-response probability model; NLL, negative log likelihood; PSU, primary sampling unit; Out, going out without telling him; Children, neglecting children; Arguing, arguing with him; Sex, refusing sex; Food, burning food.
