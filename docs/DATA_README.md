# Data sources and analytical units

## Current official aggregate analysis

The current workflow under `revision_2026/` uses official published attitude percentages and documented national structural indicators. It evaluates women’s **national average endorsement** separately from the **centered profile across five situations**. These are country/survey estimates, not linked individual participants or observed violence events.

The strict country-and-time evaluation contains 42 targets; external evaluation contains 19 MICS targets. Together these are 61 target evaluations in 58 distinct countries. Current source/membership documentation governs those analyses; historical compilation row counts do not describe the revised prediction sample. See [CURRENT_ANALYSES.md](CURRENT_ANALYSES.md) for reproduction scope.

## Historical Kaggle-distributed compilation

`data/raw/violence_against_women_girls_original.csv` is the historical compilation distributed as [*Violence Against Women & Girls*](https://www.kaggle.com/datasets/whenamancodes/violence-against-women-girls), version 1. Earlier data documentation records 12,600 rows and eight columns:

- `RecordID`: compilation record identifier.
- `Country`: country name.
- `Gender`: published sex category.
- `Demographics Question`: stratification dimension.
- `Demographics Response`: native subgroup label.
- `Question`: justification situation.
- `Survey Year`: source survey-year label.
- `Value`: percentage endorsing that justification in the specified group.

Rows are aggregate published cells. Subgroups/items can derive from the same survey and are not independent respondents. The compilation includes female and male categories; the current primary question concerns women. An endorsement percentage is not the prevalence of experienced violence.

The compilation supports historical benchmark reproducibility. Its original local download date and distributor extraction procedure remain unverified. An identified public version-1 download reproduced the historical local input exactly. All 11,187 nonmissing historical cells matched unique official DHS API records; 11,132 agreed at one-decimal precision, while 55 differed in India or Turkey. This is a numerical source comparison, not universal survey equivalence.

Native age ranges, employment groups, population eligibility, survey dates and questionnaire wording remain relevant. Incompatible groups must not be silently merged to create apparent harmonization.

## Historical processed tables and external indicators

`data/processed/clean_modeling_table.csv` is the earlier modeling table. Earlier documentation records 11,187 nonmissing valid aggregate cells. Fields include `Country`, `Gender`, `Demographics Question`, `Demographics Response`, `Question`, parsed `SurveyYear`, `Value`, `ok_*` and `is_valid_row`. These flags concern historical compilation checks, not eligibility in a new respondent study.

Historical external inputs include Hofstede dimensions, UNDP/HDR development/gender indices and V-Dem/Our World in Data indicators. They retain provider terms and release scope. Earlier merged tables/exploratory outputs are not automatically current official-source model inputs.

[DATA_DICTIONARY.csv](DATA_DICTIONARY.csv) and [FILE_MANIFEST.csv](FILE_MANIFEST.csv) describe the earlier package. The current revision has its own inventory; the historical manifest is not a complete current-release manifest.

## Separate respondent sources

| Source | Role and access |
|---|---|
| [INEI ENDES](https://proyectos.inei.gob.pe/microdatos/) | Peru’s separate releases/module documentation; linked attitudes, reported IPV and depressive symptoms require original files. |
| [Afrobarometer](https://www.afrobarometer.org/data/) | Separate Round 9 acceptance/rejection analyses; the general discipline item does not replicate all five national scenarios in individuals. |
| [Global Flourishing Study](https://osf.io/c8hbk/) | Separate symptom/context analyses under the public release’s access conditions. |
| [UN Women rapid gender assessments](https://data.unwomen.org/publications/vaw-rga) | Separate COVID-era exposure/symptom context with its own sampling and questions. |

New additions distribute aggregate results and code rather than respondent rows, private fitted states or provider source archives. Samples are not paired across studies. Country linkage cannot establish a participant-level association, causal education effect or attitude–brain pathway. EEG and fMRI are outside the current focused manuscript.

Access conditions, reuse rights and ethics determinations are separate questions. See [DATA_AVAILABILITY_STATEMENT.md](DATA_AVAILABILITY_STATEMENT.md) for distribution scope.
