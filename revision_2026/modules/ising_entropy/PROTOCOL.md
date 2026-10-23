# Entropy of fitted justification-response distributions

22 October 2026. This is a post hoc descriptive extension of the already fitted ENDES Ising experiment. The original predictive results were known before proposing entropy. This document fixes the calculations before computing the new summaries; it is not preregistration or a new model-selection exercise. No additional fits or individual-data analysis are planned.

## Inputs and estimands

Use the six saved full-cohort distributions (released women's weights or equal-person fits, each with independent items, common acceptance or Ising), all 90 saved training-fold distributions, and the 200 existing primary weighted Ising observed-PSU bootstrap refits. All distributions have 32 probabilities for five binary responses, with V744A as the least significant pattern bit. The public aggregate feasibility table supplies separate weighted and equal-person empirical plug-in distributions. These are the same 29,102 complete responders, including 433 women endorsing at least one item; partitions/refits are not additional participants.

The probability sum is checked at numerical tolerance before normalization for floating-point accumulation. No probabilities are truncated or added to empty empirical cells. Use base-two logarithms and the convention 0 log2 0 = 0. Exact enumeration means summing the 32 saved probabilities; the common-factor input retains the earlier verified numerical quadrature accuracy.

For each distribution calculate:

- Joint Shannon entropy H(Y) = -sum p(y) log2 p(y), between 0 and 5 bits; normalized entropy H(Y)/5; effective response-pattern count 2^H(Y).
- Sum of the five marginal binary entropies. Total correlation is that sum minus H(Y), equivalently KL divergence from the product of **this distribution's own marginals**. It is not the difference between entropies of separately regularized model fits.
- For each item j, H(Yj|Y-other) = H(Y)-H(Y-other), and mutual information H(Yj)-H(Yj|Y-other). Verify the conditional calculation directly as a probability-weighted binary entropy.
- With A indicating any endorsement, calculate P(A=1), H(A), H(Y|A=1), and their exact decomposition H(Y) = H(A) + P(A=1) H(Y|A=1). There is only one A=0 pattern. Normalize the conditional nonzero entropy by log2(31), not by five. Conditioning on any endorsement selects on the outcome and can itself induce dependence; this is descriptive and does not identify a new population effect.

## Sensitivity and interpretation

For the primary weighted full Ising distribution, reuse the 200 saved bootstrap **refits**, computing 2.5th/97.5th percentile summaries for each entropy metric and item summary. They are conditional observed-PSU/refit sensitivity ranges, not calibrated national-design confidence intervals. The original 18 singleton strata remain fixed. Do not use the 2,000 fixed-prediction loss resamples as entropy refits, and do not claim paired model-comparison intervals without refitting comparator models. Alternate weighting and training-fold distributions are descriptive sensitivities; folds and partitions are not independent replications. No entropy-based item ranking, causal effect or new predictive superiority will be asserted.

Low entropy may reflect the dominant all-rejection pattern. Entropy is a property of a response distribution, not a measure of moral quality, psychological health, temporal volatility, brain complexity or future violence. Model-implied entropy averages under the model's own distribution and is different from held-out negative log likelihood, which averages over held-out observed responses. Lower entropy does not establish better prediction. The common-acceptance comparison remains essential: dependence does not prove a pairwise network mechanism. Empirical plug-in entropy has sparse-cell and finite-sample bias; it is not a population truth or a bias-corrected estimator. No neural information is used. Separate GNN conditional outputs are not assumed to define a normalized 32-state joint distribution.

## Outputs and boundaries

Produce compact CSVs, a generated Markdown results report, a vector/600-dpi figure, recorded source hashes, independent mathematical/source/report checks, and a code/aggregate ZIP. Existing full-cohort public probability tables may be copied into public_inputs for replay. Original individual records, private fitted NPZs, bootstrap probability/state vectors and source cohort rows are excluded from the ZIP. Public bootstrap outputs contain derived scalar summaries only. Original-source/full-refit replay remains dependent on the earlier study; the new calculations introduce no fit.

All earlier studies, main and supplementary sources/PDFs, and the independent final submission folder are protected. No downloads, installations, manuscript integration, contact edits or journal actions.

## Method references

Shannon's [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) defines entropy and its logarithmic units. Schneidman and colleagues' [Network Information and Connected Correlations](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.91.238701) provides the information-theoretic dependence context. These established measures are applied here; no new entropy metric or neural mechanism is claimed.

Abbreviations: ENDES, Encuesta Demográfica y de Salud Familiar; GNN, graph neural network; KL, Kullback–Leibler divergence; PSU, primary sampling unit.
