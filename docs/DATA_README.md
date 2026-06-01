# Data README

## Raw data

`data/raw/violence_against_women_girls_original.csv` contains the original public survey compilation. It has 12,600 rows and 8 columns in the uploaded version:

- `RecordID`
- `Country`
- `Gender`
- `Demographics Question`
- `Demographics Response`
- `Question`
- `Survey Year`
- `Value`

## Cleaned modeling table

`data/processed/clean_modeling_table.csv` is the analysis-ready table used for Parts I-II. It contains 11,187 valid records after cleaning and validity filtering. Important fields include:

- `Country`: country identifier.
- `Gender`: respondent gender category.
- `Demographics Question`: demographic stratification dimension.
- `Demographics Response`: response category within the demographic dimension.
- `Question`: attitudinal item/scenario.
- `SurveyYear`: parsed numeric survey year.
- `Value`: prevalence percentage used as the regression outcome.
- `ok_*` and `is_valid_row`: cleaning/validity flags.

## External macro-structural indicators

The external files include Hofstede cultural dimensions, UNDP/HDR development and gender indices, and V-Dem/OWID democratic-quality and women-political-empowerment indicators. These are used in Part III to relate country-level violence-justification prevalence to macro-structural conditions.

## File manifest

See `docs/FILE_MANIFEST.csv` for the full list of files included in this package.
