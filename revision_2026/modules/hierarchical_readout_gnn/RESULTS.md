# Graph representations with a conditional Bayesian country readout

This completed fixed retrospective comparison tests national female endorsement percentages. The learned encoder is deterministic; only the subsequent conditional readout has Bayesian parameter, residual-variance and new-country uncertainty. These are not individual risk predictions, observed social ties or a fully Bayesian neural model.

The real-neighbour model has higher observed bounded component error than the linear hierarchy in both target domains for both fixed initializations. The primary conditional contrasts do not establish an advantage over self-only messages or a consistent improvement from the additional centered-profile encoder. These comparisons provide no basis for claiming graph-specific superiority.

## Primary observed errors

MAE means mean absolute error; pp means percentage points. This table scores outputs bounded to 0–100%. Clipping can change the displayed G1/B2 centered profiles and G1/G2 means even though their corresponding raw coordinates obey the exact identities described below.

| Domain | Model | Graph | Component MAE (pp) | Mean MAE (pp) | Centered-profile MAE (pp) |
|---|---|---|---:|---:|---:|
| chronological42 | B2 | linear | 8.916 | 8.333 | 3.773 |
| chronological42 | G1 | random | 10.721 | 9.953 | 3.725 |
| chronological42 | G2 | random | 10.672 | 9.949 | 3.767 |
| chronological42 | G1 | real | 10.278 | 9.730 | 3.786 |
| chronological42 | G2 | real | 10.196 | 9.733 | 3.822 |
| chronological42 | G1 | self_only | 9.639 | 8.998 | 3.766 |
| chronological42 | G2 | self_only | 9.630 | 8.985 | 3.779 |
| external19 | B2 | linear | 7.140 | 6.791 | 2.874 |
| external19 | G1 | random | 6.740 | 6.159 | 2.859 |
| external19 | G2 | random | 6.908 | 6.159 | 3.045 |
| external19 | G1 | real | 7.875 | 7.135 | 2.911 |
| external19 | G2 | real | 7.893 | 7.148 | 2.843 |
| external19 | G1 | self_only | 8.028 | 7.630 | 2.892 |
| external19 | G2 | self_only | 8.014 | 7.624 | 2.737 |

Before clipping, G1 learns the national mean and preserves the original linear B2 profile. G2 uses exactly the same raw learned mean and additionally learns the four centered-profile coordinates. Real, self-only and random controls share the fixed architecture and initialization. The self-only model has inactive neighbour parameters. Lower error is better.

## Primary fixed-prediction contrasts

| Domain | Comparison | Raw loss | Difference [conditional 95% range] |
|---|---|---|---|
| chronological42 | real_G2_vs_linear_B2 | MAE_pp | +1.330 [-0.349, +3.461] |
| chronological42 | real_G2_vs_linear_B2 | level_MAE_pp | +1.449 [-0.314, +3.635] |
| chronological42 | real_G2_vs_linear_B2 | profile_MAE_pp | +0.058 [-0.182, +0.306] |
| chronological42 | real_G2_vs_self_G2 | MAE_pp | +0.592 [-0.696, +1.964] |
| chronological42 | real_G2_vs_self_G2 | level_MAE_pp | +0.773 [-0.601, +2.176] |
| chronological42 | real_G2_vs_self_G2 | profile_MAE_pp | +0.035 [-0.229, +0.307] |
| chronological42 | real_G2_vs_random_G2 | MAE_pp | -0.625 [-2.165, +0.933] |
| chronological42 | real_G2_vs_G1 | profile_MAE_pp | +0.058 [-0.182, +0.306] |
| external19 | real_G2_vs_linear_B2 | MAE_pp | +0.697 [-0.696, +2.203] |
| external19 | real_G2_vs_linear_B2 | level_MAE_pp | +0.370 [-1.365, +2.156] |
| external19 | real_G2_vs_linear_B2 | profile_MAE_pp | -0.102 [-0.399, +0.196] |
| external19 | real_G2_vs_self_G2 | MAE_pp | -0.115 [-1.555, +1.402] |
| external19 | real_G2_vs_self_G2 | level_MAE_pp | -0.452 [-2.115, +1.269] |
| external19 | real_G2_vs_self_G2 | profile_MAE_pp | +0.109 [-0.211, +0.412] |
| external19 | real_G2_vs_random_G2 | MAE_pp | +0.822 [-1.262, +2.963] |
| external19 | real_G2_vs_G1 | profile_MAE_pp | -0.102 [-0.399, +0.196] |

Differences subtract the named comparator: negative loss changes favour the candidate. These ranges resample saved country errors without refitting; they do not include encoder-training variability, survey design uncertainty or independent outer-training sets. The ordering metric has the opposite direction. Seed 43 remains a separate optimization sensitivity in the complete tables, with no selection by target performance.

## Conditional uncertainty and numerical identities

| Domain | Model | Graph | 95% scenario coverage | Mean width (pp) | Predictive mass outside 0–100 |
|---|---|---|---:|---:|---:|
| chronological42 | B2 | linear | 0.976 | 57.8 | 0.157 |
| chronological42 | G2 | real | 0.938 | 54.4 | 0.144 |
| external19 | B2 | linear | 0.989 | 57.8 | 0.229 |
| external19 | G2 | real | 0.989 | 54.8 | 0.191 |

The reconstructed linear B2 baseline differs from the completed hierarchy by at most 1.3e-12 percentage points. Raw G1/G2 mean coordinates and raw G1/B2 profile coordinates are identical by construction. Canonical coordinate evaluation and the fixed 1e-10 predicted-gap tolerance prevent floating-point roundoff being reported as a learned profile or ordering effect.

All 732 fixed encoder fits and 793 target/model evaluations are retained. The predictive endpoint precision check leaves 0 unresolved failures; 3 evaluations used the prespecified larger draw stream. Conditional draws are discarded after interval summaries.

The readout integrates a previously unseen country random intercept, coefficients, residual variance and the fixed variance-ratio prior. It conditions on learned hidden features, and therefore omits representation uncertainty and the effect of supervised feature learning on the posterior. Broad Gaussian predictive support permits impossible percentages. Survey comparability, source-release timing and national sampling uncertainty remain upstream limitations. No gain should be attributed specifically to graph structure without agreement against both self-only and random controls.

The existing manuscript, supplement, final export and earlier Ising/Bayesian studies remain preserved. This study is analysis-only.

See [protocol](PROTOCOL.md), [all model summaries](results/model_summary.csv), [all paired contrasts](results/paired_comparisons.csv), [conditional coverage](results/predictive_coverage.csv), and the independent review for the exact verification scope.
