# ENDES reported-IPV classification — 22 October 2026

This study compares demographics with demographics plus five violence-justification responses for classification of reported past-year physical or sexual intimate partner violence. One fixed GraphSAGE model is compared with logistic regression, gradient boosting, self-messages and random neighbours. Graph edges represent demographic similarity, not observed social ties.

Start with [RESULTS.md](RESULTS.md), the [figure](figures/endes_ipv_classification.pdf), [frozen protocol](PROTOCOL.md), [source/design review](source_review/SOURCE_DESIGN_REVIEW.md), and [completion report](COMPLETION_REPORT.md). Final results and verification statuses are recorded separately; filenames alone do not imply a completed check.

The public archive contains code, configuration, aggregate scores, conditional loss comparisons, calibration summaries, figures and computational review reports. It excludes respondent records, individual predictions, graph memberships and model states. Earlier analyses and manuscript releases remain unchanged in their recorded stages. The new result is integrated into the later independent final manuscript export after its checks pass.

Run the public aggregate check from this directory:

```bash
python replay_aggregates.py --check
```

This uses the Python standard library and public aggregate files only. With NumPy, pandas and Matplotlib installed, the report and figure can be regenerated using:

```bash
python build_report.py
python build_supplement.py
python replay_aggregates.py --check
```

These commands do not refit models or verify outcomes from individual records. Original-source/model replay requires the provider's releases, the existing source-preparation workflows, the pinned prepared cohorts and the documented directory layout. See [SOURCE_ACQUISITION.md](SOURCE_ACQUISITION.md). The original repository workflow runs `run_analysis.py` in ordered preparation, freeze, fit and aggregation stages and refuses silent replacement of completed fits. It depends on its source and independent graph gates; it is not advertised as a one-command raw-data replay from the public aggregate ZIP.

This is retrospective exploratory classification of reported preceding-year violence within a selected linked cohort. It is not future-assault forecasting, a clinical diagnosis, a national prevalence analysis, a causal attitude effect, an observed-network influence model or a neural mechanism. No journal action is performed by this code.
