# Supporting mental-health and COVID analyses

This directory provides original analysis code, study specifications and selected
public aggregate result tables for the manuscript's separate mental-health
studies. It is a **code-inspection and results complement**. These folders do not
provide a self-contained participant-level fitting workflow, and they are not
included in the seven model-module replay commands.

| Folder | Current manuscript role | Included summaries |
|---|---|---|
| `endes_concurrent/` | Same-woman justification, recent intimate partner violence and concurrent depressive symptoms; main mental-health complement and Supplementary Section S21 | Fold counts and errors, score summaries, attitude counts and conditional comparisons |
| `endes_interview_year/` | Prediction from actual 2023 interviews to actual 2024 interviews; main complement and Section S25 | Full-cohort, new-PSU and endorsement-domain errors and conditional comparisons |
| `covid_stress/` | COVIDiSTRESS 2020 perceived stress and algorithm comparisons; Sections S8 and S10 | Country coverage, country errors, sensitivity and benchmark summaries |
| `gfs_symptoms/` | Separate Global Flourishing Study symptoms, COVID-related bereavement, follow-up selection and contextual prediction; Sections S5, S6, S9, S10 and S14 | Country estimates, association summaries, retention, country-held-out errors and nested-model summaries |
| `un_women/` | Separate rapid gender assessments, symptom/context prediction and Colombia violence-exposure comparison; Sections S6 and S7 | Country/domain summaries and adjusted/unadjusted symptom-association summaries |

Section S37 synthesizes the secondary exposure/symptom findings. The national
mean-versus-justification-profile question remains the main study. ENDES, GFS,
COVIDiSTRESS and UN Women respondents are separate people from the DHS national
attitude samples and from each other. These sources must not be joined as if
they contained linked individual observations.

`source_manifest.json` records the original checked public archive, archive member,
SHA-256 hash, byte size and table schema for every included file. All 107
original files are byte-identical to their current local study versions; their
combined size is 606,889 bytes. Original Results/protocol documents describe the
workflow at the time of analysis. Their references to historical journal targets,
manuscript integration stages or omitted figures do not describe the current
BMC draft. Current section mappings above govern this compact complement.

The supplied tables contain country, fold, model or study-domain summaries,
rather than participant records or individual predictions. Provider source
files, prepared cohorts, source-input tables, individual identifiers, individual
embeddings/probabilities, fitted model states, posterior blocks, bootstrap arrays
and neuroscience data are omitted. Full fitting requires the original provider
releases, their access conditions and the original workflow's documented paths
and dependencies. Relative paths in the unchanged original scripts retain that
original workflow; running those scripts directly from this compact selection
will not reconstruct omitted inputs. No acquisition or fitting command is
advertised here.

Interpretation also depends on the outcome. The ENDES and GFS symptom scores do
not establish clinical diagnoses. COVIDiSTRESS measures perceived stress. The
UN Women questions distinguish reported exposure or worsening from endorsement
of justifications. The ENDES interview-year comparison uses different cohorts,
rather than within-person prognosis; it is not an estimate of a COVID effect.
Conditional intervals based on fixed predictions omit model-refitting and full
survey-design uncertainty. These results do not establish a causal
attitude–violence–mental-health pathway.
