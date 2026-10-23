# Bayesian country–scenario hierarchy: fixed retrospective test

Prepared 22 October 2026 before these new fits. Earlier outcomes and competing model results have been examined. This is a retrospective exploratory model comparison, not prospective confirmation or an estimate of individual effects. The preceding Ising study must be completed before fitting starts, according to the author's requested order. A root-issued start record is required by the fitting script.

## Data and folds

Reuse the source-corrected, independently checked national female five-component inventory from the shared-context/context-block studies: 169 DHS rounds in 62 countries, 42 latest-country targets and 19 availability-selected MICS country targets. Copy published aggregate inputs, exact questionnaire-compatible reference memberships/predictions and source ledgers into this new directory. Check the original API/context, prepared-file and external-source hashes before fitting. All target-country attitude history is excluded. Donors must finish strictly before target fieldwork; the existing conservative month-end convention is retained for external folds. Nepal 2011 is excluded for its known questionnaire filter. A fixed sensitivity omits every India donor in addition; all target outcomes remain fixed.

The national percentages have no reconstructed sample sizes or survey standard errors. Do not model percentages as binomial counts. Measurement equivalence remains unresolved for many rounds; known restricted populations, modified wording, report availability and geographic coverage remain upstream limitations. Mean endorsement means the arithmetic average of the five scenario percentages, not any-reason prevalence.

## Fixed basis and Gaussian hierarchy

Let Y have five published scenario percentages in the CONFIG order. The first modeled coordinate is m = mean(Y). The remaining four are Y Q, where Q is the transpose of SciPy's four-row Helmert matrix. Thus Q'Q = I and Q'1 = 0, and Y = m 1' + (Y Q)Q'. These describe centered national marginal profiles, not respondent-level response patterns.

For each of the five coordinates separately, with n eligible donor rounds and G donor countries:

    y = X beta + C u + epsilon
    epsilon | sigma² ~ Normal(0, sigma² I_n)
    beta | sigma² ~ Normal(m0, sigma² V0)
    u | sigma², lambda ~ Normal(0, lambda sigma² I_G)
    sigma² ~ InverseGamma(a0, b0)
    lambda ~ DiscreteUniform{0, 0.25, 1, 4, 16}.

C is the country indicator matrix. The mean-coordinate prior intercept is 25 percentage points; all four contrast prior intercepts are zero. All slope means are zero. V0 has diagonal 25 for the intercept and 1 for every standardized slope. The primary residual prior is a0 = 2, b0 = 100, using density proportional to (sigma²)^(-a0-1) exp(-b0/sigma²). Fixed sensitivities use b0 = 50 and 200. These are disclosed modeling choices, not priors estimated from held-out outcomes. Different coordinates have independent prior blocks; the diagonal contrast covariance and finite variance-ratio support limit the model.

B0 includes intercept and calendar in all coordinates. B1 adds HDI, electoral democracy, liberal democracy and women's political empowerment to the mean coordinate only. B2 also adds those four structural variables to all four contrast coordinates. No new feature, penalty grid, country cutoff, result-dependent model choice or additional architecture is introduced.

Recompute all imputation and scaling using donor rows only, with the prior study's one-total-per-donor-country weights for preprocessing. A constant donor feature has scale one. The proper Gaussian likelihood includes each round once; repeated countries are represented through country covariance. Consequently, earlier weighted-ridge fits are references on identical targets/donor rows but impose a different fitting assumption. This is not survey-design weighting or multilevel regression and poststratification.

## Exact inference and unseen-country predictions

For each fixed lambda, integrate beta and u to obtain K = I + lambda C C' + X V0 X'. Set v = y - X m0, a_n = a0 + n/2 and b_n = b0 + v'K^(-1)v/2. The marginal likelihood is evaluated in log space using Cholesky solves, the determinant of K and normal–inverse-gamma normalization. Normalize the five prior-weighted marginal likelihoods to integrate lambda uncertainty; do not select its best value.

For target design x*, no target-country random effect has been estimated. The train–target covariance is k* = X V0 x*, and the new country has prior random-effect variance lambda sigma². Conditional-lambda predictive mean is x* m0 + k*' K^(-1)v. Its Student-t predictive scale² is b_n/a_n times [1 + lambda + x*'V0x* - k*'K^(-1)k*], with 2a_n degrees of freedom. Thus prediction explicitly integrates the unseen country's random effect and a new round residual. It does not set the target-country effect to zero without uncertainty, use target history or estimate an intercept from its observed responses.

All coefficient, variance and finite-mixture integrations use conjugate algebra; there is no MCMC, ADVI or convergence claim. The PyMC import was checked but is incompatible with the installed ArviZ/SciPy combination; no environment mutation is required for this exact NumPy/SciPy implementation. Draw 20,000 independent posterior predictive samples per target/model/prior with fixed deterministic seeds only to transform predictive uncertainty into five scenario intervals and pair-order probabilities; point predictions use the exact mixture mean. Saved interval endpoints are Monte Carlo summaries, not analytic exact quantiles. Prespecified numerical precision check: compare the independent first-half and second-half 2.5/97.5% endpoints for all five scenarios, the mean and four contrast coordinates. If their maximum absolute discrepancy exceeds 2 percentage points, repeat that target/model/prior using a fixed 100,000-draw stream, without changing its fitted posterior. Record any residual failure explicitly. This cannot change point predictions, model choice or profile gains. Before fitting, fixed prior-predictive draws at centered predictors must match analytic coordinate means within eight finite-variance mean standard errors and empirical second moments within a fixed 20% relative tolerance (the inverse-gamma shape of two gives no fourth moment, so no Gaussian variance-SE claim) and report the broad Gaussian prior’s out-of-range mass. The primary point forecast is clipped to [0,100]; retain raw outputs and both kinds of losses. Clipping is a deterministic sensitivity, not a beta likelihood or calibration proof. Record raw predictive out-of-range probability.

Normal–inverse-gamma conjugacy is described in [Murphy's original technical report](https://cbmm.mit.edu/sites/default/files/learning-hub/Murphy07_gaussian.pdf). The extension from its Gaussian updates to this covariance model follows by analytic Gaussian marginalization; the full formulas above define the implemented model.

## Fixed evaluation and interpretation

Export training/excluded memberships, preprocessing, all conditional-lambda posterior parameters/weights, predictive means/intervals, component and country scores, pairwise-order probabilities, mean/profile summaries, priors, seeds and runtime versions. Independently verify source identities, fieldwork exclusions, train-only scaling, prior support/normalization, posterior algebra and every final metric.

Report equal-country component MAE/MSE, mean absolute/squared error and centered-profile MAE/MSE for all models, priors, donor cases and target sets. Check exact MSE = squared mean error + centered-profile MSE for every vector. Verify that B0 and B1 have identical raw centered point profiles and ordering, and identical contrast-coordinate posterior blocks; any bounded-profile difference is a clipping artifact. B1 and B2 have identical raw mean predictions and mean-coordinate posterior blocks. Report pair-order accuracy excluding observed ties with half credit for numerical predicted ties (absolute gap at most 1e-10 percentage points); report the exact denominator and raw/bounded differences. Posterior pair probabilities have Monte Carlo error and are model conditional. Assess 50%, 80% and 95% marginal predictive coverage/width and 95% mean/profile-coordinate coverage; these are retrospective adequacy summaries, not nationally design-calibrated confidence intervals.

Compare B1−B0, B2−B1, B2−B0 and primary Bayesian models against saved calendar, shared-shift and scenario-specific ridge references. A fixed 10,000 equal-country bootstrap with one common resampling index per domain describes paired saved-prediction loss differences. It does not refit the hierarchy, account for shared outer-training sets or recover DHS design uncertainty. Retain all directions and all fixed sensitivities without a success threshold.

## Outputs and preservation

Write only inside this new study. Preserve the active manuscript/SI, independent final folder and all earlier source/results/archives. Produce a restrained black-framed figure with bold lowercase panel letters, vector PDF/SVG and 600-dpi PNG, a concise scientific interpretation, methods, completion/release records and a compact code/public-aggregate package. Never include individual records, confidential PDFs or claims of individual causation, mental-health linkage, social influence or brain connectivity. No journal action is authorized.
