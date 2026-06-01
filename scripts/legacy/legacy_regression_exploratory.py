# Auto-extracted from notebooks/legacy/legacy_regression_exploratory.ipynb

# Review/reproducibility helper script. Original notebook is retained in notebooks/.



# %% Cell 0
import kagglehub

# Download latest version
path = kagglehub.dataset_download("whenamancodes/violence-against-women-girls")

print("Path to dataset files:", path)


# %% Cell 1
# ============================================================
# VAWEG: One regression model per country (Ridge + one-hot)
# - Reads CSV
# - Cleans / parses Survey Year (stored as dates like 01/01/2015)
# - Trains a RidgeCV model per country
# - Evaluates with K-fold CV (MAE, RMSE, R2)
# - Exports summary CSV + cleaned modeling table
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# USER SETTINGS
# -----------------------------
INPUT_CSV = "Violence Against Women  Girls Data.csv"   # change if needed
OUT_DIR   = "vawg_country_models"                     # output folder
MIN_ROWS_PER_COUNTRY = 30

# If your Survey Year parsing is ambiguous, set this to True/False explicitly:
# - True  -> interpret strings like 01/01/2015 as day/month/year
# - False -> interpret as month/day/year
DAYFIRST = True


# -----------------------------
# HELPERS
# -----------------------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def parse_year(series: pd.Series, dayfirst: bool = True) -> pd.Series:
    """Parse date-like strings to year integer."""
    dt = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    # fallback: if parsing failed for many rows, try the opposite
    if dt.isna().mean() > 0.5:
        dt2 = pd.to_datetime(series, errors="coerce", dayfirst=not dayfirst)
        dt = dt.fillna(dt2)
    return dt.dt.year


# -----------------------------
# 1) READ
# -----------------------------
in_path = Path(INPUT_CSV)
if not in_path.exists():
    raise FileNotFoundError(f"Could not find: {in_path.resolve()}")

df = pd.read_csv(in_path)

# Expected columns
cols_keep = [
    "Country",
    "Gender",
    "Demographics Question",
    "Demographics Response",
    "Question",
    "Survey Year",
    "Value",
]
missing = [c for c in cols_keep if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

df = df[cols_keep].copy()


# -----------------------------
# 2) CLEAN / PREPROCESS
# -----------------------------
# Parse "Survey Year" into a numeric year
df["SurveyYear"] = parse_year(df["Survey Year"], dayfirst=DAYFIRST)

# Make Value numeric
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# Strip strings
for c in ["Country", "Gender", "Demographics Question", "Demographics Response", "Question"]:
    df[c] = df[c].astype(str).str.strip()

# Drop rows missing essentials
df_clean = df.dropna(
    subset=[
        "Country", "Gender", "Demographics Question", "Demographics Response",
        "Question", "SurveyYear", "Value"
    ]
).copy()

# Optionally remove empty strings
df_clean = df_clean[
    (df_clean["Country"] != "") &
    (df_clean["Question"] != "") &
    (df_clean["Gender"] != "")
].copy()


# -----------------------------
# 3) MODEL PIPELINE
# -----------------------------
X_cols = ["Gender", "Demographics Question", "Demographics Response", "Question", "SurveyYear"]
y_col  = "Value"

categorical = ["Gender", "Demographics Question", "Demographics Response", "Question"]
numeric     = ["SurveyYear"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ],
    remainder="drop"
)

alphas = np.logspace(-3, 6, 50)
model  = RidgeCV(alphas=alphas)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("model", model),
])

scorers = {
    "MAE":  make_scorer(mean_absolute_error, greater_is_better=False),
    "RMSE": make_scorer(rmse, greater_is_better=False),
    "R2":   make_scorer(r2_score),
}


# -----------------------------
# 4) TRAIN/EVALUATE PER COUNTRY
# -----------------------------
countries = sorted(df_clean["Country"].unique())

rows = []
for country in countries:
    d = df_clean[df_clean["Country"] == country].copy()
    n = len(d)

    out = {
        "Country": country,
        "n_rows": n,
        "cv_folds": np.nan,
        "MAE_mean": np.nan,  "MAE_std": np.nan,
        "RMSE_mean": np.nan, "RMSE_std": np.nan,
        "R2_mean": np.nan,   "R2_std": np.nan,
        "note": "",
    }

    if n < MIN_ROWS_PER_COUNTRY:
        out["note"] = f"Skipped (<{MIN_ROWS_PER_COUNTRY} rows)"
        rows.append(out)
        continue

    # choose folds based on sample size
    n_splits = 5 if n >= 60 else 3
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    X = d[X_cols]
    y = d[y_col].values

    cv_out = cross_validate(pipe, X, y, cv=cv, scoring=scorers, return_train_score=False)

    mae = -cv_out["test_MAE"]
    rms = -cv_out["test_RMSE"]
    r2  =  cv_out["test_R2"]

    out["cv_folds"]  = n_splits
    out["MAE_mean"]  = float(np.mean(mae))
    out["MAE_std"]   = float(np.std(mae, ddof=1)) if len(mae) > 1 else 0.0
    out["RMSE_mean"] = float(np.mean(rms))
    out["RMSE_std"]  = float(np.std(rms, ddof=1)) if len(rms) > 1 else 0.0
    out["R2_mean"]   = float(np.mean(r2))
    out["R2_std"]    = float(np.std(r2, ddof=1)) if len(r2) > 1 else 0.0

    rows.append(out)

summary = pd.DataFrame(rows).sort_values(["R2_mean", "Country"], ascending=[False, True]).reset_index(drop=True)


# -----------------------------
# 5) EXPORT
# -----------------------------
out_dir = Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

summary_path = out_dir / "per_country_regression_cv_summary.csv"
clean_path   = out_dir / "clean_modeling_table.csv"
meta_path    = out_dir / "data_overview.txt"

summary.to_csv(summary_path, index=False)
df_clean.to_csv(clean_path, index=False)

with open(meta_path, "w", encoding="utf-8") as f:
    f.write(f"Input file: {in_path.resolve()}\n")
    f.write(f"Rows (raw): {len(df)}\n")
    f.write(f"Rows (clean, used): {len(df_clean)}\n")
    f.write(f"Countries: {len(countries)}\n")
    f.write("Target: Value\n")
    f.write("Predictors: Gender, Demographics Question, Demographics Response, Question, SurveyYear\n")
    f.write("Preprocess: OneHotEncoder(handle_unknown='ignore') for categorical; SurveyYear numeric\n")
    f.write("Model: RidgeCV (alphas logspace -3..6)\n")
    f.write("CV: 5-fold if n>=60 else 3-fold; shuffled, random_state=42\n")
    f.write("Metrics: MAE, RMSE, R2 (test folds)\n")

print("Done.")
print("Saved:", summary_path)
print("Saved:", clean_path)
print("Saved:", meta_path)
print("\nTop countries by CV R2:")
print(summary.head(10).to_string(index=False))



# %% Cell 2

