# Analysis code

The national mean-versus-profile question is the primary analysis. The central runner refits all 42 country-and-time and 19 external targets from the prepared official national inputs. These are retrospective exploratory comparisons.

From the repository root:

```bash
python -m pip install -r analysis/requirements.txt
python analysis/reproduce.py check
python analysis/reproduce.py national --output reproduced/national
python analysis/reproduce.py aggregates --output reproduced/aggregate_checks
```

`check` validates supplied-file hashes, Python syntax and declared replay entry points without fitting or downloading data. `national` refits the fixed central models and checks saved predictions, country scores and donor membership. Its conditional bootstrap stage replays the saved country-error tables, preserving the original random-number sequence; it does not bootstrap model refitting or survey-design variance. `aggregates` recomputes the supporting public result tables in temporary copies without accessing individual records or retraining neural models.

See the [central methods and inputs](central/README.md), the [module index](module_index.json) and the [analysis map](../docs/ANALYSES.md). Each supporting module contains its original code and protocol; refitting from original sources requires the provider files and dependencies documented in that module. Dates and stage labels inside protocol and results documents record when each analysis was specified and run.

The primary runner starts from the extracted official national estimates and contextual covariates; it does not repeat the API/report retrieval and harmonization, which are documented in the source manifests.

[Secondary symptom analyses](secondary/README.md) contain the original code and public summaries for ENDES, COVID-19 (COVIDiSTRESS), the Global Flourishing Study and UN Women. Model fitting for these requires the providers' participant releases; they are outside the seven portable aggregate replays.
