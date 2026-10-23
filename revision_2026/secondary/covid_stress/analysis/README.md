# COVID-19 stress cohort: observed feasibility

This cohort uses the corrected author-cleaned COVIDiSTRESS 2020 release downloaded from the official OSF project. It is a new respondent dataset, not DHS microdata. Source file: `https://osf.io/download/cjxua/`. The downloaded file has 125,306 rows; this is the cleaned release, not the descriptor's raw survey count. Exact hash and cleaning documents are saved locally.

## Scoring and corrections
The source uses item values 1–5. Items 4, 5, 7 and 8 are reversed with `6 - item`; the complete-item mean is converted to a 0–40 sum by `10 * (mean - 1)`. Recomputed complete-item means agree exactly with the published `PSS10_avg`. PSS measures perceived stress, not clinical depression or anxiety.

The official cleaning code and correction note document prior language/coding errors, including a Bengali item swap. The corrected release already addresses these; this workflow does not apply those corrections twice. The input encoding is Latin-1. ID uniqueness and observed item ranges were checked.

## Cohort and linkage
The exploratory cohort retains women aged 18–100, with all ten PSS items, recorded during 30 March–30 May 2020. Exact or explicitly aliased country linkage is used. The full exclusion ledger and country coverage are exported.

There are 2,510 women across six overlapping countries meeting the preset feasibility threshold of at least 100 eligible women: Bangladesh, Colombia, Indonesia, Pakistan, Philippines and Turkey. This threshold is a feasibility rule, not a power calculation. Full overlap counts, including countries below the threshold, remain available.

Contextual predictors are ten separate official DHS Program estimates: female urban/rural endorsement under five component scenarios. They come from the independently retrieved official API, not an unweighted average of demographic margins. The any-reason composite is excluded from this feature set. These historical country measures do not describe the personal beliefs or IPV exposure of COVID survey respondents. The six historical surveys precede 2020 by 3–8 years. No population representativeness is assumed.

## What this permits
A six-country leave-one-country-out pilot can test computational feasibility and reveal poor transfer. It cannot supply strong evidence for a general contextual effect. Country-level information remains limited to six contexts even with thousands of people. No causal pandemic effect, clinical diagnosis, individual norm effect or national stress prevalence is estimated.

## Reproduce
From the repository root:
```bash
python revision_2026/04_mental_health/covid19/scripts/download_covidistress.py
python revision_2026/04_mental_health/covid19/scripts/prepare_cohort.py
python revision_2026/06_optional_cebra/scripts/benchmark_covid_embeddings.py
```
The cohort step requires the expanded official API audit first. Dependencies used are recorded in the CEBRA analysis requirements and environment JSON. The current executable environment is Python 3.9.7, retained for compatibility; no global environment was changed. CEBRA and its added dependency were installed in the project's local vendor directory.

The manuscript text and supplement have deliberately not been updated with these analysis results.
