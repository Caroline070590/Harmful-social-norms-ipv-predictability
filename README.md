# Structural context and women’s endorsement of justifications for intimate partner violence

Analysis code and public aggregate results for the current working manuscript by **Caroline L. Alves**, Department of Basic Neurosciences, University of Geneva, Switzerland.

The central question is whether structural context predicts both **women’s national average endorsement** of violence justifications and **the relative pattern across five situations**. The official-source analyses more consistently support gains in mean-endorsement prediction; added value for the relative scenario profile remains uncertain. The results do not establish causal effects, forecast future violence or demonstrate consistent graph-neural-network superiority.

## Start with the current analysis

The [`revision_2026/`](revision_2026/) directory supplies the updated, explicitly scoped workflow. The existing top-level notebooks, result folders and manuscript PDFs document the earlier analysis and are retained separately.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r revision_2026/requirements.txt
python revision_2026/reproduce.py check
python revision_2026/reproduce.py national --output reproduced/national
python revision_2026/reproduce.py aggregates --output reproduced/aggregate_checks
```

The check command validates the supplied inventory, Python syntax and declared replay entry points. The national command refits the fixed models from prepared official national inputs and checks predictions, scores and donor membership. Its bootstrap comparison replays saved country-error tables. The aggregates command checks supporting result calculations in temporary copies without retraining neural models. See [`docs/CURRENT_ANALYSES.md`](docs/CURRENT_ANALYSES.md) for the exact scope. These commands do not download respondent records or refit every model in the manuscript.

| Reproduction task | Scope |
|---|---|
| Current national mean/profile analysis | Official aggregate inputs and the documented public workflow in `revision_2026/`. |
| Supporting aggregate comparisons | Public summaries and code identified in the current module inventory; saved-summary replay is distinguished from model fitting. |
| Respondent-level model refitting | Requires original provider releases, source documentation and module-specific environments; individual records and private fitted states are not newly distributed here. |
| Historical benchmark notebooks | Earlier interactive workflow under `notebooks/` and extracted scripts under `scripts/`; these are not the revised national-analysis entry point. |

The primary official test contains **42 country-and-time targets**, predicted using earlier surveys from other countries. External evaluation contains **19 MICS targets**. These are **61 target evaluations in 58 distinct countries**, because three countries occur in both domains. Outcomes had been examined during earlier work, so the validation is retrospective and exploratory.

## Why the historical Kaggle input is retained

The earlier workflow used the Kaggle-distributed *Violence Against Women & Girls* compilation. Each row is a published country/survey/sex/demographic-group/item **percentage**, not an individual woman or an observed violence event. The identified input permits reproduction and evaluation of the historical benchmark.

Official estimates have authority for the revised source-based analyses. The complete comparison matched **11,187 nonmissing historical cells** to unique official DHS API records: **11,132 (99.51%)** agreed at one-decimal precision, and **55** differed, all in India or Turkey. Agreement does not establish questionnaire or population equivalence. The distributor’s extraction procedure and the original local download date remain unverified. Native age and employment categories are preserved rather than silently pooled.

The historical repository already contains the compilation and processed aggregate tables. The current additions distinguish those existing assets from provider-retrieved respondent inputs. See [`docs/DATA_README.md`](docs/DATA_README.md) and [`docs/DATA_AVAILABILITY_STATEMENT.md`](docs/DATA_AVAILABILITY_STATEMENT.md) for provenance, access and reuse scope.

## Current and historical materials

| Location | Purpose |
|---|---|
| `revision_2026/` | Current public code, aggregate inputs/results and scoped reproduction commands. |
| `docs/CURRENT_ANALYSES.md` | Scientific analysis map and reproduction limits. |
| `data/`, `results/`, `figures/` | Existing historical tables, outputs and artwork; earlier figure numbers do not identify current figures. |
| `notebooks/`, `scripts/` | Earlier interactive analyses and extracted Python code, including legacy explorations. |
| `manuscript/` | Historical submitted/supporting documents; these are not automatically the current focused draft. |

Separate respondent studies supply individual-level checks and mental-health context. Their participants are not linked to the aggregate attitude samples. EEG and fMRI are outside the current focused paper.

Top-level `requirements.txt` and `environment.yml` describe broad historical dependencies, not a pinned numerical environment for all current analyses. Use the current minimal requirements for the quick start and each module’s documented dependencies for additional fits.

## Citation and status

This is a working research-code repository, not a claim of journal acceptance or a new numbered software release. Machine-readable citation metadata are in [`CITATION.cff`](CITATION.cff).

The earlier manuscript is:

Alves, C. L. *When Are Harmful Social Norms Predictable? Cross-National Structure in Attitudes Justifying Intimate Partner Violence*. medRxiv, 2026. [Earlier preprint](https://www.medrxiv.org/content/10.64898/2026.01.25.26344795v1.full).

The repository previously identified that historical package as v1.0.0, dated 16 June 2026. That metadata does not describe a new release of the revised manuscript. The current focused title appears at the top of this page.

## Reuse

Earlier repository documentation declares MIT terms for code and computational notebooks and CC BY 4.0 for documentation and generated figures. Corresponding standalone license-text files are not presently included. These declarations do not relicense third-party datasets, dependencies, source documents or third-party artwork. Those materials retain their providers’ terms; public availability and aggregate format alone do not grant unrestricted reuse.
