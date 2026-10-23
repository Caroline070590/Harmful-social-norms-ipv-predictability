# Nested country-held-out PHQ-4 benchmark: results

Executed 19–20 October 2026 under `PROTOCOL.md` (saved before fitting). Post-primary exploratory revision of the fixed-setting benchmark. Independent check passed: selections reproduced from inner scores; all nine outer ridge fits reproduced with separately implemented preprocessing and normal equations (max MAE difference 2.4e-07); seed means reproduced.

## Outer held-out-country performance (equal-country mean, PHQ-4 0–12)

| Model (tuned within training countries) | MAE | Mean R² |
|---|---:|---:|
| Training mean | 2.744 | -0.084 |
| Ridge | 2.646 | -0.019 |
| Gradient boosting | 2.653 | -0.024 |
| PCA + ridge | 2.715 | -0.068 |
| CEBRA + ridge | 2.687 | -0.053 |
| CEBRA shuffled | 2.742 | -0.086 |

- Tuned CEBRA minus tuned ridge: +0.0410 MAE; CEBRA better in 0/9 countries.
- Tuned CEBRA minus shuffled-label control: -0.0556; better in 7/9.
- Tuned minus fixed-setting: ridge +0.0020; CEBRA -0.0423.
- Lowest mean MAE: Ridge.

## Selection and convergence

Inner selection chose 5000 iterations for CEBRA in every fold (settings: {'{"iterations": 5000, "dim": 3, "temperature": 1.0}': 4, '{"iterations": 5000, "dim": 8, "temperature": 1.0}': 4, '{"iterations": 5000, "dim": 8, "temperature": 0.1}': 1}). Longer training lowered inner error relative to 1000 iterations for every dimension/temperature combination. For supervised final fits, the mean loss of the last 500 iterations differed from the preceding 500 by a median -0.02% (range -0.10% to 0.07%); 54/54 final fits (supervised and shuffled) changed by less than 1%. This indicates approximate stabilization within the tested budget, not proof of a global optimum.

## Interpretation

With training-only tuning and a longer training budget, CEBRA improved relative to its fixed-setting prototype but still did not outperform ridge regression in withheld countries; all models had small or negative R². The earlier conclusion therefore no longer rests on an untuned prototype: within this bounded grid, supervised embeddings added no predictive value over a linear model for cross-country PHQ-4 prediction. Representation sampling remained unweighted. Nine countries.

Figure: `figures/FigS6_nested_benchmark` (A: inner MAE by CEBRA setting, dashed = tuned ridge inner MAE; B: outer country MAE; C: loss curves).
