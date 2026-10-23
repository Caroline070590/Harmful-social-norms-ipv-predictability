# Current analysis code

The national mean-versus-profile question is the primary analysis. The current central runner refits all 42 country/time and 19 external targets from prepared published national inputs. These are retrospective exploratory comparisons.

From the repository root:

```bash
python -m pip install -r revision_2026/requirements.txt
python revision_2026/reproduce.py check
python revision_2026/reproduce.py national --output reproduced/national
python revision_2026/reproduce.py aggregates --output reproduced/aggregate_checks
```

`check` validates supplied-file hashes, Python syntax and declared replay entry points without fitting or downloading data. `national` refits the fixed central models and checks saved predictions, country scores and donor membership. Its conditional bootstrap stage replays verified saved country-error tables, preserving the original random-number sequence; it does not bootstrap model refitting or survey-design variance. `aggregates` checks the additional public result tables in temporary copies without accessing individual records or retraining neural models.

See [central methods and inputs](central/README.md), [module index](module_index.json), and [data-source scope](../docs/CURRENT_ANALYSES.md). Each supplementary module retains its original code and protocol, but full original-source fitting may require upstream provider files and workflow helpers that are not part of this compact update. Historical flags and section labels within these preserved modules describe their original analysis stage; the module index identifies their current scientific role. No EEG/fMRI study is integrated.

The input/source manifests document exact public aggregate files. The primary runner starts from already extracted national estimates and contextual covariates; it does not repeat the API/report retrieval and harmonization. All original historical results remain in the existing repository folders.

[Secondary symptom analyses](secondary/README.md) supply original code and public summaries for ENDES, COVID-19, GFS and UN Women. These source-inspection complements require upstream participant releases for model fitting and are outside the seven portable aggregate replays.
