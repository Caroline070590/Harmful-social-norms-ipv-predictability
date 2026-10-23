# Bayesian country–scenario comparison — 22 October 2026

Read [RESULTS.md](RESULTS.md) for scientific interpretation, [PROTOCOL.md](PROTOCOL.md) for the fixed likelihood/priors and [COMPLETION_REPORT.md](COMPLETION_REPORT.md) for exact execution and verification scope. The author requested this stage after the completed separate ENDES Ising model test; [ROOT_START.json](ROOT_START.json) records the authorization and binds its completion.

The analysis predicts national female endorsement means separately from four fixed centered scenario contrasts. The outcomes come from the checked official aggregate DHS/MICS inputs, rather than respondent records or the legacy Kaggle benchmark. It is a proper Gaussian hierarchical model with donor-country random intercepts, normal–inverse-gamma priors and an integrated finite prior over country variance ratios. New-country uncertainty is integrated; no held-out history is used. Analytic posterior means produce scores, while fixed seeded simulations summarize intervals and order probabilities. No MCMC, ADVI or convergence assertion is involved.

All 42 strict country/time targets and 19 external MICS targets represent 61 target evaluations across 58 distinct countries; three countries occur in both domains. Both primary/India-omission donor cases, three fixed residual priors and three models remain in the outputs. These are previously inspected aggregate outcomes, so the new comparisons are retrospective exploratory. Gaussian support, broad prior predictive tails, a fixed finite variance-ratio grid, independent fixed-basis coordinate blocks, source/measurement uncertainty and unavailable national design variances limit interpretation. The primary B1-versus-shared-ridge and B2-versus-scenario-specific-ridge component-MAE intervals include zero in both domains; this does not demonstrate superiority over the existing context baselines. Earlier ridge comparisons differ in likelihood weighting and regularization; a contrast cannot isolate an effect of hierarchy alone. High retrospective scenario coverage accompanies roughly 58-percentage-point interval widths and 16–23% predictive probability outside 0–100 for the structural-context models. Thus panel c does not establish useful calibration.

The PyMC distribution was present but its import failed because installed ArviZ expects a removed SciPy symbol. Exact NumPy/SciPy conjugacy required no package installation or environment change. The implemented formulas are stated in full. A verified original technical reference is [Murphy, *Conjugate Bayesian analysis of the Gaussian distribution* (2007)](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf); the frozen protocol retains an additional university-hosted mirror.

The public aggregate replay needs Python, NumPy and pandas:

```bash
python aggregate_replay.py
```

It recomputes saved prediction scores, model summaries, all fixed country-bootstrap contrasts and saved-interval coverage aggregation. It does not refit posterior parameters or regenerate posterior-predictive streams. Full source extraction/posterior refitting requires the repository's upstream public-source files and expected relative paths:

```bash
python run_analysis.py --fit
python report_results.py
```

The fitting command deliberately refuses to overwrite completed fits. It verifies the frozen source and preserved-file hashes before fitting. To perform a fresh original-source/model replay, create a separate owned replay directory/workflow preserving provenance; do not delete or overwrite the authoritative completed outputs. Report/figure generation alone reads saved aggregate tables. No national sampling counts/SEs are reconstructed, no respondent records are included, and no original report PDFs or old archives are duplicated.

The active main/supplement sources/PDFs, earlier figures/analyses/archives and independent final folder are preserved. This is a separate analysis, with no automatic manuscript integration, contact change, download, installation, public deposit or journal action.

Posterior parameters are stored losslessly as `results/posterior_blocks.json.gz` after independent numerical QC. [POSTERIOR_STORAGE.json](POSTERIOR_STORAGE.json) binds the stored gzip and exact original uncompressed bytes; this storage change does not alter frozen inference code or any result. Verify storage and rerun aggregate scores with:

```bash
python storage_and_replay.py --verify --scores
```

The independent checker transparently reads compressed posterior parameters. No predictive-draw arrays are retained. Original uncompressed parameter bytes can be regenerated with standard gzip tools in a separate replay directory.

Independent review found that mathematically identical raw profiles/means had floating-point differences which could be miscounted as improvements. The new `correct_identity_summaries.py` normalizes only the proven B1−B0 raw profile/ordering and B2−B1 raw mean contrasts to zero after a fixed 1e−10 tolerance check. All unrelated CSV rows remain byte-identical; original CSV bytes are preserved losslessly under `postfit_identity_correction/pre_correction/`. Frozen fitting code and every parameter, prediction and figure remain unchanged. A fresh original workflow should apply this documented correction after producing its summaries; no generic small-effect screening is used.

A second proof-only correction, `complete_identity_summaries.py`, extends the same identities to bounded rows only when both full five-scenario vectors are wholly unchanged by bounding. It also handles B2−B1 component-MAE ties when both models' five residuals share the same sign, so MAE equals their absolute shared mean error. Both rounds retain exact original CSV bytes and enumerate every corrected case/field. All other rows are byte-identical; no generic small-effect tolerance selects results. The public aggregate replay implements these exact mathematical identities before reconstructing bootstrap summaries and counts.
