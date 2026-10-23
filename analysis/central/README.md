# National mean versus justification profile

This is the reproduction of the manuscript's central aggregate
prediction comparison. It uses 169 prepared DHS national rounds in 62 countries,
42 latest-country targets and 19 external MICS-family targets. Inputs are
published national estimates and national structural indicators, not individual
respondent records.

From this folder, with Python, NumPy and pandas installed, run:

```bash
python reproduce.py --output-dir reproduced
```

The command independently solves the fixed weighted normal equations and checks
all 915 component predictions, 366 country-score rows and 8,182 donor
membership rows against the original results. It also checks all 216 conditional
country-bootstrap comparisons, including the external sensitivity cohorts.
Outputs are written only into the supplied output folder. No data download,
neural retraining or parameter selection occurs.

R0 contains item-specific intercepts and calendar slopes. R1 adds item-specific
slopes for lagged HDI, electoral democracy, liberal democracy and women's
political empowerment. S restricts those four contextual slopes to be shared
between scenarios. The ridge penalty is 10; shared-context terms have the
equivalent five-task penalty of 50. Training countries receive equal total
weight. Imputation and scaling use donors only. Every target country's entire
history is withheld; donors must finish before target fieldwork begins. External
donor end dates conservatively use the last day of the reported month. Nepal
2011 is excluded because of its documented questionnaire filter. The protocols
and unchanged original fitting implementations are in `original_code/`.

The bounded component MAE is 11.968 versus 9.015 percentage points for
R0 versus R1 in the country/time comparison and 15.126 versus 8.043 externally.
The distinction between national average endorsement and the centered relative
scenario profile must be retained. Lower component error does not itself prove
better scenario-specific prediction, an individual mechanism, violence risk or
causal prevention effects.

`reference_results/` preserves complete original aggregate tables, including
earlier sensitivity cases. Earlier cases are used only to preserve the exact
original bootstrap random-number sequence; the reproduction refits and reports
the `questionnaire_compatible` case. Intervals resample fixed country
prediction errors without refitting and omit original survey-design uncertainty.
This remains retrospective exploratory validation.

`SOURCE_MANIFEST.json` records byte-preserving paths and hashes. Source-cell and
structural-year ledgers identify official published inputs. This compact replay
starts from those prepared inputs; it does not repeat API/report extraction or
independently establish questionnaire invariance. Full original-source pipelines
remain available in the source repository/workflow and require their separately
documented cached public source files.
