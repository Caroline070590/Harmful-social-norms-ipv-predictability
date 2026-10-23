# Structural context and women’s endorsement of justifications for intimate partner violence

Analysis code and public aggregate results for the manuscript by **Caroline L. Alves**, Department of Basic Neurosciences, University of Geneva, Switzerland.

The study asks whether structural context predicts both **women’s national average endorsement** of justifications for intimate partner violence and **the relative pattern across five situations** (going out without telling the husband, neglecting the children, arguing, refusing sex and burning the food). Structural context more consistently improves prediction of the national average; additional prediction of the relative pattern remains uncertain. The results do not establish causal effects, forecast future violence or demonstrate consistent graph-neural-network superiority.

## Quick start

Tested with Python 3.9.7, NumPy 1.26, pandas 2.1 and SciPy 1.13. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
python analysis/reproduce.py check
python analysis/reproduce.py national --output reproduced/national
python analysis/reproduce.py aggregates --output reproduced/aggregate_checks
```

| Command | What it does |
|---|---|
| `check` | Verifies every supplied file against `analysis/MANIFEST.json`, compiles all Python code and confirms the replay entry points. No fitting or downloads. |
| `national` | Refits the fixed national models from the official aggregate inputs for all 42 country-and-time and 19 external targets, and checks the predictions, country scores and donor membership against the saved results. The bootstrap stage replays saved country-error tables with the original random-number sequence. |
| `aggregates` | Recomputes the published aggregate results of the seven supporting model analyses in temporary copies, without individual records or neural retraining. |

Outputs are written to the folder you choose (it must be outside `analysis/`).

## Main analysis

The **national mean** is the average of the five published endorsement percentages; the **centered profile** is each situation’s deviation from that mean. A model can predict that a country has generally higher endorsement while still failing to predict which situations receive relatively greater endorsement.

- **Country-and-time validation:** 42 latest-survey targets, each predicted using only earlier surveys from other countries and excluding the target country’s own attitude history.
- **External validation:** 19 MICS-family targets. Together these are 61 target evaluations in 58 distinct countries.
- Outcomes had been examined during earlier work, so the validation is retrospective and exploratory.

Official estimates from the [DHS Program API](https://api.dhsprogram.com/) are the source for all national inputs. A historical aggregate compilation distributed on Kaggle (*Violence Against Women & Girls*) was compared with these official estimates: 11,187 nonmissing cells matched unique official records, of which 11,132 (99.51%) agreed at one-decimal precision and 55 differed (all in India or Turkey). Numerical agreement does not establish questionnaire or population equivalence. The compilation is not redistributed here; it can be obtained from its provider.

## Repository contents

| Location | Contents |
|---|---|
| `analysis/central/` | National mean/profile models: inputs, source ledgers, original fitting code and the reproduction runner. |
| `analysis/modules/` | Supporting analyses: Bayesian country–scenario hierarchy, Afrobarometer acceptance classifier, reported-IPV classifier, Ising joint-response model and entropy, Ising-informed GNN and hierarchical-readout GNN. |
| `analysis/secondary/` | Separate mental-health and COVID-19 analyses (ENDES, COVIDiSTRESS, Global Flourishing Study, UN Women): original code and public summaries. |
| `analysis/MANIFEST.json`, `analysis/module_index.json` | File hashes and the module inventory with each module’s scope. |
| `docs/` | [Analysis map](docs/ANALYSES.md), [data availability](docs/DATA_AVAILABILITY.md) and [declarations](docs/DECLARATIONS.md). |

All supplied data are country-, fold- or model-level aggregates. No respondent records, individual predictions or fitted respondent-level model states are included. Refitting the individual-level analyses requires the original provider releases (INEI ENDES, Afrobarometer, Global Flourishing Study, UN Women, COVIDiSTRESS) under their access conditions; each module documents its own requirements.

## Citation

Please cite the repository and the commit used; machine-readable metadata are in [`CITATION.cff`](CITATION.cff).

An earlier version of this work was posted as a preprint with a different title and scope: Alves, C. L. *When Are Harmful Social Norms Predictable? Cross-National Structure in Attitudes Justifying Intimate Partner Violence*. medRxiv, 2026. [Preprint](https://www.medrxiv.org/content/10.64898/2026.01.25.26344795v1.full).

## Reuse

Code is provided under MIT terms and documentation under CC BY 4.0, as previously declared for this repository. These terms do not relicense third-party datasets, dependencies or source documents, which retain their providers’ terms.
