# Entropy of the fitted Ising response network

[Read the scientific results](RESULTS.md) · [Figure](figures/ising_entropy.pdf) · [Calculation protocol](PROTOCOL.md)

This post hoc analysis uses the earlier fitted response-pattern distributions. It adds no model fits or participant records. The main manuscript, supplement and existing submission folder remain preserved.

[Completion and code archive](RELEASE_INDEX.json) · [Code and aggregate ZIP](Ising_entropy_code_aggregate_2026_10_22.zip)

For a portable public aggregate replay:

```bash
python replay_aggregates.py
```

The replay calculates full-model and empirical entropy from the seven public aggregate input tables and checks the derived bootstrap percentile summaries. The 90 training-fold and 200 bootstrap distributions were independently checked against their saved upstream states locally. Their source probability/state vectors remain outside the public ZIP. The same complete responders underlie all fits, folds and bootstrap summaries.

To regenerate every new entropy table in the repository, use `python analyze_entropy.py`; this requires the earlier audited Ising study and its saved private states. No original Ising fitting or raw respondent processing is performed by that command. Figure/report regeneration additionally uses `python report_entropy.py` and requires Matplotlib. Runtime package versions are recorded in ENVIRONMENT.json.

The independent review distinguishes mathematical/source checks from report/figure checks and clean-extraction public aggregate replay. Bootstrap ranges are conditional sensitivity summaries from the existing weighted Ising refits, not national-design confidence intervals or evidence of better prediction.
