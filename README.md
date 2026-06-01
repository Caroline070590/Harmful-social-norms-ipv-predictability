# When Are Harmful Social Norms Predictable? Reproducibility Package

This repository contains the data tables, analysis notebooks, extracted Python scripts, model outputs, and manuscript figures for the manuscript:

**When Are Harmful Social Norms Predictable? Cross-National Structure in Attitudes Justifying Intimate Partner Violence**


## Repository structure

```text
data/
  raw/                     Original public violence-attitudes CSV used as input.
  processed/               Cleaned modeling tables and merged macro-structural panels.
  external/                External indicators from Hofstede, UNDP/HDR, V-Dem/OWID, and Natural Earth.
notebooks/                 Original analysis notebooks, renamed by analysis stage.
scripts/                   Python scripts auto-extracted from the notebooks for easier inspection.
figures/manuscript_figures/ Figure files mapped to manuscript figure panels.
results/                   Model summaries, feature-importance outputs, macro-indicator analyses, and supplementary figures.
docs/                      Data availability, declarations, file manifest, and response-letter draft.
manuscript/                Submitted manuscript PDF and supporting PDFs from the analysis workflow.
```

## Main data files

- `data/raw/violence_against_women_girls_original.csv`: original public survey compilation used as the source table.
- `data/processed/clean_modeling_table.csv`: cleaned country-year-stratum-item modeling table used in Parts I-II.
- `data/processed/MASTER_unwomen_HDR_VDEM_panel_CLEAN.csv`: merged country-year macro panel used for contextual analyses.
- `data/external/Hofstede/hofstede_country_scores.csv`: Hofstede cultural-dimension country scores used in Part III.
- `data/external/undp_hdr/HDR_composite_indices_complete_time_series.csv`: UNDP/HDR composite index data.
- `data/external/vdem_owid/*.csv`: V-Dem/OWID democracy and women-political-empowerment indicators.

## Reproducing the analyses

1. Create the environment:

```bash
conda env create -f environment.yml
conda activate vaw-hssc-revision
```

or install with pip:

```bash
pip install -r requirements.txt
```

2. Run the notebooks in this approximate order:

```text
01_data_explanation.ipynb
02_country_record_maps.ipynb
03_part1_full_model_RF_XGBoost_optimized.ipynb
05_part1_model_agreement_heatmaps.ipynb
06_part1_feature_importance_barplots.ipynb
07_part2_demographics_only_xgboost.ipynb
08_part2_barplots.ipynb
09_part3_hofstede_macro_analysis.ipynb
10_part3_external_hierarchy_analysis.ipynb
11_country_clustering_macro_indicators.ipynb
```

The `scripts/` folder contains auto-extracted `.py` versions of these notebooks to help editors/referees inspect the code outside Jupyter. The notebooks are retained as the authoritative executed workflow because figures and results were generated interactively from those analyses.

## Notes for peer review

- The original public input data, cleaned modeling data, external indicator tables, and result summaries are included.
- The package includes both raw/processed CSV files and model-output CSVs so reviewers can verify data transformations and downstream statistical outputs.
- Some legacy exploratory notebooks are retained in `notebooks/legacy/` for transparency but are not required for reproducing the manuscript figures.
- Duplicate `.ipynb_checkpoints` and redundant nested zip archives were intentionally removed from this organized repository.

## Suggested citation

Alves, C. L. *When Are Harmful Social Norms Predictable? Cross-National Structure in Attitudes Justifying Intimate Partner Violence*. medRxiv preprint. Available at: https://www.medrxiv.org/content/10.64898/2026.01.25.26344795v1.full

## License

This repository uses a mixed-license structure:

- Code, scripts, and computational notebooks are released under the MIT License.
- Documentation, README files, manuscript-supporting text, and generated figures are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
- Data files are shared for transparency, peer review, and reproducibility. Reuse of third-party datasets remains subject to the terms and conditions of the original data providers, including the DHS-derived source dataset and external country-level indicator sources.

Please cite the associated preprint and this repository when reusing the code, documentation, figures, or processed materials.
