# Exact justification-response network

22 October 2026. Start with [RESULTS.md](RESULTS.md) for the scientific findings and [COMPLETION_REPORT.md](COMPLETION_REPORT.md) for the completed work and review limits.

This analysis tests whether associations among five responses add held-out joint-pattern prediction beyond independent items and a shared acceptance factor. It is separate from the active manuscript and the independent final submission export. It does not predict future violence or identify social influence, brain connectivity or causal interactions.

The weighted primary comparison does not establish a clear Ising advantage over a common acceptance factor. The Ising network substantially improves joint-pattern prediction over independent items; equal-person refits also have a more favorable common-factor comparison. These distinctions and all adverse conditional-item comparisons are retained.

## Files

- `PROTOCOL.md`, `PROTOCOL_FREEZE.json`, `CONFIG.json`: fixed exploratory design. `PRE_FIT_NUMERICAL_AMENDMENT.json` documents a numerical quadrature refinement made before fitting, superseding the original 61-node proposal.
- `results/`: aggregate source feasibility, model metrics, paired comparisons, bootstrap summaries and all ten interactions.
- `figures/ising_justification_network.pdf`, `.svg`, `.png`: vector and 600-dpi versions of the same four-panel figure, with bold lowercase labels.
- `independent_review/`: original-source, design, mathematical and presentation review, each with its stated scope.
- `NUMBER_TRACEABILITY.csv`: machine-readable bindings for the main reported estimates. Additional result tables are generated directly from their aggregate CSVs.
- `aggregate_replay.py`: verifies the portable public aggregate calculations.
- `run_analysis.py`, `report_and_figure.py`: producer fitting and report/figure code. Scientific fitting requires the original authorized public ENDES release or its audited compact cohort, plus the earlier fold-assignment helper in this repository. The small aggregate package is not a self-contained original-data/model-training replay.

`data_local/` contains individual fold identities, losses, fitted states and resampling matrices. It is excluded from the public ZIP. No individual respondent file, original source archive, EEG/fMRI file, confidential manuscript or previous large archive is included.

## Reproduction and uncertainty

Run `python aggregate_replay.py` from an extracted code/aggregate ZIP to verify aggregate identities and reported contrasts. It requires NumPy and pandas. Recreating the figure requires Matplotlib. The complete producer run additionally requires SciPy and the earlier repository source/fold workflow; use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for the recorded numerical setting.

The source is the official INEI ENDES 2023 release. The existing audited adult complete-item cohort includes 29,102 women, 433 endorsing at least one justification, across 3,249 observed PSUs and 240 released strata. Interview year 2024 occurs for 34 women in this release; no temporal prediction is performed. Documented women's weights are used for weighted conditional point/loss comparisons. Eighteen singleton strata have unresolved national variance contributions. Paired loss-difference ranges hold fitted predictions fixed; edge ranges refit the network but still hold singleton units fixed. Neither is a verified national design confidence interval.

The exact joint likelihood sums all 32 binary response patterns. It is not pairwise pseudolikelihood or eLasso. Fixed Gaussian penalties yield a weighted working maximum-a-posteriori objective; no Ising posterior or calibrated Bayesian credible interval was sampled.

## Primary references

[Van Borkulo et al., *Scientific Reports* (2014)](https://doi.org/10.1038/srep05918) presents eLasso for binary networks. Our five-item analysis instead computes the exact finite-state likelihood, with fixed regularization and a common-factor comparator. [Dalege et al., *Psychological Review* (2016)](https://doi.org/10.1037/a0039802) develops the Causal Attitude Network theory. Cross-sectional response associations here cannot identify its proposed causal interactions or dynamics.

Abbreviations: ENDES, Encuesta Demográfica y de Salud Familiar; INEI, Instituto Nacional de Estadística e Informática; PSU, primary sampling unit; NLL, negative log likelihood; MAP, maximum a posteriori; eLasso, the referenced regularized logistic network-estimation method; CAN, Causal Attitude Network.
