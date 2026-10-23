# Current analysis and reproduction scope

The focused working manuscript asks: **Can structural context predict women’s national average endorsement of violence justifications and the relative pattern across those justifications?** Official aggregate estimates lead. Historical benchmarks and separate respondent studies have supporting roles; they do not share participants.

The national mean is the average of the five published endorsement percentages. The centered profile is each situation’s deviation from that mean. A model can therefore predict whether a country has generally higher endorsement while still failing to predict which situations receive relatively greater endorsement.

## Public reproduction entry point

Run from the repository root:

```bash
python -m pip install -r revision_2026/requirements.txt
python revision_2026/reproduce.py check
python revision_2026/reproduce.py national --output reproduced/national
python revision_2026/reproduce.py aggregates --output reproduced/aggregate_checks
```

The national command refits the fixed central models from prepared published national estimates/contextual inputs. It checks saved predictions, country scores and donor membership for the 42/19 target domains. Source API/report retrieval and harmonization are not repeated. Its conditional bootstrap stage replays saved country-error tables with the original random-number sequence; it does not bootstrap refitting or national survey sampling.

The aggregates command runs the seven supporting public-result replays in temporary module copies, without provider records or neural retraining. Module-specific instructions and upstream requirements govern additional fits. See the [central workflow](../revision_2026/central/README.md) for the implemented model/refitting scope.

## Scientific analysis map

| Analysis | Question and permitted interpretation |
|---|---|
| Official country-and-time prediction | Predict 42 latest-survey targets using earlier surveys from other countries. Evaluate the national mean separately from the centered five-situation profile. |
| External survey evaluation | Evaluate 19 MICS targets under the documented source mapping: 61 target evaluations in 58 countries across both domains. |
| Context/questionnaire checks | Compare development/institutional blocks, preserve native categories and use directly documented questionnaires in the specified restricted analyses. |
| Country similarity/network models | Compare context-based networks with matched non-network, self-only and random/shuffled controls. A constructed similarity graph is not an observed social-influence network. |
| Bayesian country–scenario hierarchy | Separate mean/profile prediction under documented priors; the analytic hierarchy does not establish a fully Bayesian GNN. |
| Separate Afrobarometer analysis | Classify one general acceptance/rejection item among adult women with whole-country holdouts, in a separate sample/questionnaire. |
| Separate ENDES analyses | Examine linked attitudes, reported IPV and depressive symptoms in the applicable Peruvian cohorts, with documented temporal, weighting and sampling limits. |
| ENDES Ising/joint-response analyses | Examine answer dependence, descriptive entropy and artificial answer masking, not future violence or causal response mechanisms. |
| International symptom/context analyses | Supply secondary COVID-era/exposure/symptom context from separate studies, without pairing respondents to national attitude samples. |

The public commands cover the analysis actually implemented under `revision_2026/`. This table describes the scientific structure; it does not promise that every row has an original-source fitting command in this checkout. The current module inventory states the delivered scope. EEG/fMRI and confidential artwork are outside this focused release.

## Included supporting code/results

The [machine-readable module inventory](../revision_2026/module_index.json) identifies supplied files, provenance and replay limits. The following compact modules retain original scientific code and public aggregates; their README states the portable replay scope:

- [Afrobarometer acceptance classifier](../revision_2026/modules/afro_gnn_classifier/README.md): country aggregate loss/bootstrap comparisons and original classifier code; original fitting needs provider records and upstream preparation.
- [Reported-IPV classifier](../revision_2026/modules/reported_ipv/README.md): public aggregate classification comparisons.
- [Bayesian country–scenario hierarchy](../revision_2026/modules/bayesian_hierarchy/README.md): saved-prediction scores, conditional country resampling and interval summaries.
- [Ising joint-response model](../revision_2026/modules/ising/README.md): public joint distributions, score decomposition and aggregate ranges.
- [Ising entropy](../revision_2026/modules/ising_entropy/README.md): entropy/dependence calculations from released aggregate distributions.
- [Ising-informed GNN](../revision_2026/modules/ising_informed_gnn/README.md): saved aggregate losses and paired comparisons.
- [Hierarchical-readout GNN](../revision_2026/modules/hierarchical_readout_gnn/README.md): saved country predictions and conditional comparisons.

[Secondary symptom code and public summaries](../revision_2026/secondary/README.md) cover ENDES, COVID-19, GFS and UN Women. This complement supports inspection of the original code and aggregate results; provider releases and upstream preparation remain necessary for fitting.

These scopes do not make original respondent/source fitting self-contained. The historical producer code is provided for inspection with its original upstream requirements, separately from the portable replay entry points.

## Interpretation and validation

Context improves mean-endorsement prediction more consistently than relative-profile prediction. Added profile gains remain uncertain. Previously inspected outcomes make validation retrospective exploratory. Conditional bootstrap comparisons do not reconstruct unavailable national survey-design uncertainty.

GNN and hierarchy comparisons retain adverse and uncertain results. More layers, attention scores, a Bayesian readout or lower entropy do not establish a new psychological mechanism, useful clinical prediction or graph-specific superiority. Future violence, intervention effects and neural changes are not measured prediction targets.

## Historical workflow

Top-level `notebooks/` and `scripts/` preserve the earlier preprint workflow. Labels such as “Part I–III” and old figure numbers refer to that historical analysis. Interactive history and broad dependencies do not constitute a newly verified reproduction of the revised paper.

Original-source/refitting workflows require provider releases, source documentation, module-specific environments and upstream paths. The code update must not be described as reproducing fits that were not rerun.
