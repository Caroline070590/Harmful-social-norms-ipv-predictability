# Country–scenario GNN with a conditional Bayesian hierarchical readout

This separate analysis tests the author's requested graph-based extension. It retains the existing 42 strict country/time and 19 external targets, uses a fixed learned message-passing encoder and then fits an exact conjugate hierarchical readout. The neural weights are point estimates: this is a **Bayesian-readout GNN**, not a fully Bayesian neural network.

Start with [RESULTS.md](RESULTS.md), the [protocol](PROTOCOL.md), and [COMPLETION_REPORT.md](COMPLETION_REPORT.md). The [figure legend](FIGURE_LEGEND.md) defines the model labels. Outside the ZIP, `RELEASE_INDEX.json` records the final files and archive digest.

The complete comparison preserves real, self-only and random neighbours and both fixed optimization seeds. Mean and centered-profile encoders are trained separately; G1/G2 share exactly the same mean. ENDES Ising edges are not national country edges and are not used here. All inputs are public national aggregates; no individual responses are included.

`python replay_aggregates.py --check` verifies the published point-score and conditional country-resampling tables from the included aggregate predictions. It does not retrain neural representations or recover original-source survey data. Original preparation/fitting entry points remain in `run_analysis.py`; their preservation and upstream-source checks require the repository and original releases. Independent checks have separately stated scopes.

The active manuscript, supplement, independent final export and earlier studies remain unchanged. No download, installation, author-contact edit, manuscript integration, journal submission or public deposit occurred.
