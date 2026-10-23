# Both questions: women, violence and COVID-era mental health

19 October 2026. The author confirmed both research questions and a women-focused primary population. This confirms the direction, not any result. Preserve the original attitude-paper core; the additional cohorts remain separate extensions.

**Subsequent execution:** the public XLSX was downloaded and inspected: 16,154 women, 13 countries, 142 columns, unique country/respondent IDs. It contains additional phase-specific items omitted from the short codebook. Phase I measures last-month stress; Phase II measures one worry item and two depressive items with a since-COVID reference period. These are not interchangeable standard PHQ/GAD scales. A separately documented pre-fit exploratory protocol, country reports of worsening, and two direct-IPV/symptom association models in 779 eligible Colombian women are now executed in `colombia_exploratory/`. Read its generated `RESULTS.md` and independent `QC.json`. This advances individual association and retrospective-report questions; actual incidence trends, pandemic causation and temporal prediction remain unestablished.

## Questions and the closest public sources

| Question | Candidate | What can be tested | Boundary |
|---|---|---|---|
| Did violence against women worsen around COVID? | Public UN Women VAW Rapid Gender Assessments | Women's retrospective reports/perceptions of worsening, with country and phase distinctions | Not prospective incidence, not a monthly time series, and often not the respondent's own violence exposure |
| Is violence associated with worse mental health in the same women? | UN Women Colombia-specific direct partner-violence questions and the public mental_stress item | Direct-violence/mental-impact association, subject to question eligibility and available codes | Cross-sectional broad emotional impact; cannot establish later symptoms, separate anxiety/depression, or causation |
| Are there actual repeated annual violence observations and validated depression items? | Peru INEI/ENDES public annual files | Candidate for survey-weighted annual trends and linked within-woman violence/depression association | Modules, IDs, design weights, year comparability and access not yet audited locally; repeated cross-sections, not a respondent panel |

The [UN Women release](https://data.unwomen.org/publications/vaw-rga) provides public microdata and documentation. Documentation confirms country/pooled weights and Colombia-specific direct questions. Most cross-country exposure questions combine own experience with knowing another woman. The public codebook lists a broad mental-impact item rather than separate PHQ/GAD scores. Raw-release availability, identifiers and columns are audited separately; a questionnaire item alone does not prove an item is released.

Important coding issues already identified:

- Public `C26` refers to direct partner violence; it maps to questionnaire `C23`. Public `C23_*` instead contains reasons for feeling unsafe. Use labels and wording, not coincident variable numbers.
- The grouped frequency variable combines rarely/never. For an any-violence definition, rarely cannot be treated as no violence. Recode the original frequency only after checking valid responses and applicability.
- Retrospective change questions have conditional denominators. Do not assign “no violence” or “no increase” to structural skips.
- Phase I/II questionnaires differ. The technical note records a Bangladesh timing exception. Do not pool measures as interchangeable without a crosswalk.
- The released codebook repeats a combined-experience variable name for several distinct items. `codebook_duplicate_names.csv` records the inconsistency; do not silently repair it without checking actual columns.

## A stronger candidate for the depression/time component

Peru's [official ENDES annual series](https://www.gob.pe/institucion/inei/informes-publicaciones/6813742-peru-series-anuales-de-principales-indicadores-de-la-endes-1986-2024) includes violence indicators. Its [2024 public data catalogue](https://www.datosabiertos.gob.pe/dataset/encuesta-demogr%C3%A1fica-y-de-salud-familiar-endes-2024-instituto-nacional-de-estad%C3%ADstica-e) lists survey data and dictionaries. Public INEI files are a distinct access route from applying for original DHS respondent recodes; check release terms rather than assuming national public files are unavailable.

A [published ENDES analysis](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2026.1939992/full) reports linked violence and PHQ-9 measures across 2015–2024, with pandemic-period interview changes. This establishes a candidate, not verified inputs for our analysis. A simple period association would overlap existing work. Any new contribution needs a defined question, e.g. temporal predictive validation against simple baselines, with a separate source/selection and measurement audit. Do not copy its sample counts or effect sizes into our results.

Before fitting: retrieve annual dictionaries and permitted files; map individual/household keys, sex, age, union/violence-module eligibility, violence recall periods, PHQ-9 response coding, survey weights/PSU/strata, privacy exclusions and pandemic interview modes. Count linked eligible women and valid design records for each year. Never perform a many-to-many person merge or infer mental health from national averages.

If harmonization succeeds, compare survey-weighted annual violence estimates and the violence–depression association before, during and after the pandemic. Keep a change in prevalence distinct from a change in association strength. For forecasting, reserve later observed years and compare against persistence/simple trend baselines. Annual observations with fieldwork changes do not justify a causal pandemic-effect claim; anxiety is not available merely because PHQ-9 is present. No temporal CEBRA or deep forecasting model is justified by a short annual country series alone.

## Status and reproducibility

`audit_public_sources.py` retrieved the UN Women page, codebook, two questionnaires and technical note, with hashes in `DOWNLOAD_MANIFEST.json`. `download_microdata_for_readiness.py` retrieves only the explicitly public release; respondent rows stay under `data_local/` and are excluded from review/code archives. The first CSV transfer was interrupted without a complete file; the XLSX alternative uses a two-minute limit. `inspect_release.py` generates measurement/availability checks, not regression or prevalence estimates. Read `READINESS.json` for the actual raw-release status; do not infer completion from the presence of a downloader.

A second three-wave Bangladesh/Kenya/Nigeria candidate (doi:10.7910/DVN/ODM6I0) was identified, but the Harvard metadata request returned HTTP403. It is not an available, audited dataset here, and all three documented waves are pandemic-era rather than a pre-pandemic baseline.

No causal violence effect, separate clinical anxiety/depression estimate, model improvement or prospective time-series result is claimed. The exploratory pilot above supplies separately labelled individual associations and reported perceptions. Existing GFS outcomes are retained with their original definitions and negative/inconclusive results. WHO Atlas indicators remain optional country service context; they cannot substitute for any woman's exposure or treatment. Source checks continue to determine which question is answerable with which data.
