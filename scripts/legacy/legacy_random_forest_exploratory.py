# Auto-extracted from notebooks/legacy/legacy_random_forest_exploratory.ipynb

# Review/reproducibility helper script. Original notebook is retained in notebooks/.



# %% Cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# =========================
# USER SETTINGS
# =========================
INPUT_CSV = "Violence Against Women  Girls Data.csv"

OUT_ROOT = Path("results")
RF_DIR = OUT_ROOT / "random_forest"
DT_DIR = OUT_ROOT / "decision_tree"

MIN_ROWS_PER_COUNTRY = 40

# Cross-validation
N_FOLDS = 5
RANDOM_STATE = 42

# Random Forest parameters (increase for better stability, decrease for speed)
RF_N_ESTIMATORS = 600
RF_MIN_SAMPLES_LEAF = 2
RF_N_JOBS = -1

# Decision Tree parameters
DT_MIN_SAMPLES_LEAF = 2


# =========================
# HELPERS
# =========================
def parse_year(series: pd.Series) -> pd.Series:
    """
    Your 'Survey Year' column looks like '01/01/2015' (date string).
    We parse it and extract the year as an integer.
    """
    dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    # fallback if parsing fails for many entries
    if dt.isna().mean() > 0.5:
        dt2 = pd.to_datetime(series, errors="coerce", dayfirst=False)
        dt = dt.fillna(dt2)
    return dt.dt.year

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in str(s)).strip()

def cv_metrics(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, n_folds: int, seed: int):
    """
    Manual CV to compute mean/std for R2, MAE, RMSE.
    """
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    r2s, maes, rmses = [], [], []
    for tr, te in cv.split(X):
        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[te])
        r2s.append(r2_score(y[te], pred))
        maes.append(mean_absolute_error(y[te], pred))
        rmses.append(rmse(y[te], pred))

    return {
        "R2_mean": float(np.mean(r2s)),
        "R2_std": float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
    }


# =========================
# 1) READ + CLEAN
# =========================
df = pd.read_csv(INPUT_CSV)

expected = ["Country", "Gender", "Demographics Question", "Demographics Response",
            "Question", "Survey Year", "Value"]
missing = [c for c in expected if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

df = df[expected].copy()

df["SurveyYear"] = parse_year(df["Survey Year"])
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

for c in ["Country", "Gender", "Demographics Question", "Demographics Response", "Question"]:
    df[c] = df[c].astype(str).str.strip()

df_clean = df.dropna(subset=[
    "Country", "Gender", "Demographics Question", "Demographics Response",
    "Question", "SurveyYear", "Value"
]).copy()

df_clean = df_clean[(df_clean["Country"] != "") & (df_clean["Question"] != "")].copy()

# Save clean table for reproducibility
OUT_ROOT.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(OUT_ROOT / "clean_modeling_table.csv", index=False)


# =========================
# 2) PREPROCESSING
# =========================
X_cols = ["Gender", "Demographics Question", "Demographics Response", "Question", "SurveyYear"]
y_col = "Value"

categorical = ["Gender", "Demographics Question", "Demographics Response", "Question"]
numeric = ["SurveyYear"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ],
    remainder="drop"
)

# =========================
# 3) MODELS
# =========================
rf_model = RandomForestRegressor(
    n_estimators=RF_N_ESTIMATORS,
    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
    n_jobs=RF_N_JOBS
)

dt_model = DecisionTreeRegressor(
    min_samples_leaf=DT_MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE
)

rf_pipe = Pipeline([("prep", preprocess), ("model", rf_model)])
dt_pipe = Pipeline([("prep", preprocess), ("model", dt_model)])


# =========================
# 4) OUTPUT FOLDERS
# =========================
(RF_DIR / "feature_rankings").mkdir(parents=True, exist_ok=True)
(DT_DIR / "feature_rankings").mkdir(parents=True, exist_ok=True)


# =========================
# 5) PER-COUNTRY LOOP
# =========================
countries = sorted(df_clean["Country"].unique())

rf_rows = []
dt_rows = []

for country in countries:
    d = df_clean[df_clean["Country"] == country].copy()
    n = len(d)

    X = d[X_cols]
    y = d[y_col].to_numpy()

    # ---- Random Forest ----
    rf_row = {
        "Country": country,
        "n_rows": n,
        "cv_folds": N_FOLDS,
        "note": ""
    }

    # ---- Decision Tree ----
    dt_row = {
        "Country": country,
        "n_rows": n,
        "cv_folds": N_FOLDS,
        "note": ""
    }

    if n < MIN_ROWS_PER_COUNTRY:
        rf_row["note"] = f"Skipped (<{MIN_ROWS_PER_COUNTRY} rows)"
        dt_row["note"] = f"Skipped (<{MIN_ROWS_PER_COUNTRY} rows)"
        rf_rows.append(rf_row)
        dt_rows.append(dt_row)
        continue

    # CV metrics
    rf_row.update(cv_metrics(rf_pipe, X, y, N_FOLDS, RANDOM_STATE))
    dt_row.update(cv_metrics(dt_pipe, X, y, N_FOLDS, RANDOM_STATE))

    rf_rows.append(rf_row)
    dt_rows.append(dt_row)

    # Fit full data and export feature rankings
    # Random Forest
    rf_pipe.fit(X, y)
    rf_features = rf_pipe.named_steps["prep"].get_feature_names_out()
    rf_imp = rf_pipe.named_steps["model"].feature_importances_
    rf_rank = pd.DataFrame({"feature": rf_features, "importance": rf_imp}).sort_values("importance", ascending=False)
    rf_rank.to_csv(RF_DIR / "feature_rankings" / f"{safe_filename(country)}_feature_importance.csv", index=False)

    # Decision Tree
    dt_pipe.fit(X, y)
    dt_features = dt_pipe.named_steps["prep"].get_feature_names_out()
    dt_imp = dt_pipe.named_steps["model"].feature_importances_
    dt_rank = pd.DataFrame({"feature": dt_features, "importance": dt_imp}).sort_values("importance", ascending=False)
    dt_rank.to_csv(DT_DIR / "feature_rankings" / f"{safe_filename(country)}_feature_importance.csv", index=False)


# =========================
# 6) SAVE SUMMARY CSVs
# =========================
rf_summary = pd.DataFrame(rf_rows).sort_values(["R2_mean", "Country"], ascending=[False, True])
dt_summary = pd.DataFrame(dt_rows).sort_values(["R2_mean", "Country"], ascending=[False, True])

rf_summary.to_csv(RF_DIR / "per_country_cv_summary_random_forest.csv", index=False)
dt_summary.to_csv(DT_DIR / "per_country_cv_summary_decision_tree.csv", index=False)

combined = rf_summary.rename(columns={
    "R2_mean": "R2_RF", "R2_std": "R2_RF_std",
    "MAE_mean": "MAE_RF", "MAE_std": "MAE_RF_std",
    "RMSE_mean": "RMSE_RF", "RMSE_std": "RMSE_RF_std",
    "note": "note_RF"
}).merge(
    dt_summary[[
        "Country",
        "R2_mean", "R2_std",
        "MAE_mean", "MAE_std",
        "RMSE_mean", "RMSE_std",
        "note"
    ]].rename(columns={
        "R2_mean": "R2_DT", "R2_std": "R2_DT_std",
        "MAE_mean": "MAE_DT", "MAE_std": "MAE_DT_std",
        "RMSE_mean": "RMSE_DT", "RMSE_std": "RMSE_DT_std",
        "note": "note_DT"
    }),
    on="Country", how="left"
)

combined.to_csv(OUT_ROOT / "per_country_cv_summary_combined.csv", index=False)

print("Done.")
print("Saved:")
print(" -", RF_DIR / "per_country_cv_summary_random_forest.csv")
print(" -", DT_DIR / "per_country_cv_summary_decision_tree.csv")
print(" -", OUT_ROOT / "per_country_cv_summary_combined.csv")
print("Feature rankings saved under:")
print(" -", RF_DIR / "feature_rankings/")
print(" -", DT_DIR / "feature_rankings/")



# %% Cell 1
import os
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# pip install xgboost
from xgboost import XGBRegressor

# pip install geopandas
import geopandas as gpd


# =========================
# USER SETTINGS
# =========================
INPUT_CSV = "Violence Against Women  Girls Data.csv"

OUT_ROOT = Path("results")
RF_DIR = OUT_ROOT / "random_forest"
DT_DIR = OUT_ROOT / "decision_tree"
XGB_DIR = OUT_ROOT / "xgboost"

MIN_ROWS_PER_COUNTRY = 40

# Cross-validation
N_FOLDS = 5
RANDOM_STATE = 42

# Random Forest parameters
RF_N_ESTIMATORS = 600
RF_MIN_SAMPLES_LEAF = 2
RF_N_JOBS = -1

# Decision Tree parameters
DT_MIN_SAMPLES_LEAF = 2

# XGBoost parameters
XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

# Natural Earth admin0 countries zip
NE_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
NE_ZIP = OUT_ROOT / "ne_110m_admin_0_countries.zip"
NE_SHP_IN_ZIP = "ne_110m_admin_0_countries.shp"


# =========================
# HELPERS
# =========================
def parse_year(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if dt.isna().mean() > 0.5:
        dt2 = pd.to_datetime(series, errors="coerce", dayfirst=False)
        dt = dt.fillna(dt2)
    return dt.dt.year

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in str(s)).strip()

def cv_metrics(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, n_folds: int, seed: int):
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    r2s, maes, rmses = [], [], []
    for tr, te in cv.split(X):
        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[te])
        r2s.append(r2_score(y[te], pred))
        maes.append(mean_absolute_error(y[te], pred))
        rmses.append(rmse(y[te], pred))

    return {
        "R2_mean": float(np.mean(r2s)),
        "R2_std": float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
    }

def ensure_naturalearth_admin0(zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"Downloading Natural Earth countries to: {zip_path}")
        urllib.request.urlretrieve(NE_URL, zip_path)

def load_world_admin0(zip_path: Path) -> gpd.GeoDataFrame:
    ensure_naturalearth_admin0(zip_path)
    world = gpd.read_file(f"zip://{zip_path}!{NE_SHP_IN_ZIP}")
    if "ADMIN" not in world.columns:
        raise ValueError(f"'ADMIN' column not found. Available columns: {list(world.columns)}")
    return world

def plot_r2_map(world: gpd.GeoDataFrame, results_df: pd.DataFrame, value_col: str, title: str,
                out_png: Path, out_pdf: Path):
    """
    Choropleth map colored by mean R². Missing data shown in hatched grey.
    """
    gdf = world.merge(results_df, how="left", left_on="ADMIN", right_on="Country")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

    gdf.plot(
        column=value_col,
        ax=ax,
        cmap="viridis",
        linewidth=0.25,
        edgecolor="black",
        legend=True,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "black",
            "hatch": "///",
            "label": "No data / skipped"
        },
        legend_kwds={"label": "Mean CV R²", "shrink": 0.6}
    )

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# =========================
# 1) READ + CLEAN (WITH AUDIT EXPORTS)
# =========================
df_raw = pd.read_csv(INPUT_CSV)

expected = ["Country", "Gender", "Demographics Question", "Demographics Response",
            "Question", "Survey Year", "Value"]
missing = [c for c in expected if c not in df_raw.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

df = df_raw[expected].copy()

# Keep originals for audit
df["SurveyYear_raw"] = df["Survey Year"]
df["Value_raw"] = df["Value"]

# Parse year and numeric value
df["SurveyYear"] = parse_year(df["Survey Year"])
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# Strip strings
for c in ["Country", "Gender", "Demographics Question", "Demographics Response", "Question"]:
    df[c] = df[c].astype(str).str.strip()

# Validity flags (helps debug “why did I lose rows?”)
df["ok_country"] = df["Country"].notna() & (df["Country"] != "") & (df["Country"].str.lower() != "nan")
df["ok_gender"] = df["Gender"].notna() & (df["Gender"] != "") & (df["Gender"].str.lower() != "nan")
df["ok_demo_q"] = df["Demographics Question"].notna() & (df["Demographics Question"] != "") & (df["Demographics Question"].str.lower() != "nan")
df["ok_demo_r"] = df["Demographics Response"].notna() & (df["Demographics Response"] != "") & (df["Demographics Response"].str.lower() != "nan")
df["ok_question"] = df["Question"].notna() & (df["Question"] != "") & (df["Question"].str.lower() != "nan")
df["ok_year"] = df["SurveyYear"].notna()
df["ok_value"] = df["Value"].notna()

df["is_valid_row"] = (
    df["ok_country"] &
    df["ok_gender"] &
    df["ok_demo_q"] &
    df["ok_demo_r"] &
    df["ok_question"] &
    df["ok_year"] &
    df["ok_value"]
)

df_clean = df[df["is_valid_row"]].copy()

# Export clean + dropped + audit
OUT_ROOT.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(OUT_ROOT / "clean_modeling_table.csv", index=False)
df[~df["is_valid_row"]].to_csv(OUT_ROOT / "dropped_rows.csv", index=False)

audit = pd.DataFrame({
    "n_raw_rows": [len(df_raw)],
    "n_after_column_select": [len(df)],
    "n_clean_rows": [len(df_clean)],
    "n_dropped": [len(df) - len(df_clean)],
    "dropped_bad_country": [int((~df["ok_country"]).sum())],
    "dropped_bad_gender": [int((~df["ok_gender"]).sum())],
    "dropped_bad_demo_question": [int((~df["ok_demo_q"]).sum())],
    "dropped_bad_demo_response": [int((~df["ok_demo_r"]).sum())],
    "dropped_bad_question": [int((~df["ok_question"]).sum())],
    "dropped_bad_year": [int((~df["ok_year"]).sum())],
    "dropped_bad_value": [int((~df["ok_value"]).sum())],
})
audit.to_csv(OUT_ROOT / "cleaning_audit_summary.csv", index=False)

print("Cleaning summary:")
print(audit.to_string(index=False))
print("Saved cleaning outputs to:", OUT_ROOT.resolve())


# =========================
# 2) PREPROCESSING
# =========================
X_cols = ["Gender", "Demographics Question", "Demographics Response", "Question", "SurveyYear"]
y_col = "Value"

categorical = ["Gender", "Demographics Question", "Demographics Response", "Question"]
numeric = ["SurveyYear"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ],
    remainder="drop"
)


# =========================
# 3) MODELS
# =========================
rf_model = RandomForestRegressor(
    n_estimators=RF_N_ESTIMATORS,
    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
    n_jobs=RF_N_JOBS
)

dt_model = DecisionTreeRegressor(
    min_samples_leaf=DT_MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE
)

xgb_model = XGBRegressor(
    n_estimators=XGB_N_ESTIMATORS,
    max_depth=XGB_MAX_DEPTH,
    learning_rate=XGB_LEARNING_RATE,
    subsample=XGB_SUBSAMPLE,
    colsample_bytree=XGB_COLSAMPLE_BYTREE,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=RF_N_JOBS
)

rf_pipe = Pipeline([("prep", preprocess), ("model", rf_model)])
dt_pipe = Pipeline([("prep", preprocess), ("model", dt_model)])
xgb_pipe = Pipeline([("prep", preprocess), ("model", xgb_model)])


# =========================
# 4) OUTPUT FOLDERS
# =========================
for d in [RF_DIR, DT_DIR, XGB_DIR]:
    (d / "feature_rankings").mkdir(parents=True, exist_ok=True)
    (d / "maps").mkdir(parents=True, exist_ok=True)


# =========================
# 5) PER-COUNTRY LOOP
# =========================
countries = sorted(df_clean["Country"].unique())

rf_rows, dt_rows, xgb_rows = [], [], []

for country in countries:
    d = df_clean[df_clean["Country"] == country].copy()
    n = len(d)

    X = d[X_cols]
    y = d[y_col].to_numpy()

    rf_row = {"Country": country, "n_rows": n, "cv_folds": N_FOLDS, "note": ""}
    dt_row = {"Country": country, "n_rows": n, "cv_folds": N_FOLDS, "note": ""}
    xgb_row = {"Country": country, "n_rows": n, "cv_folds": N_FOLDS, "note": ""}

    if n < MIN_ROWS_PER_COUNTRY:
        note = f"Skipped (<{MIN_ROWS_PER_COUNTRY} rows)"
        rf_row["note"] = note
        dt_row["note"] = note
        xgb_row["note"] = note
        rf_rows.append(rf_row); dt_rows.append(dt_row); xgb_rows.append(xgb_row)
        continue

    # CV metrics
    rf_row.update(cv_metrics(rf_pipe, X, y, N_FOLDS, RANDOM_STATE))
    dt_row.update(cv_metrics(dt_pipe, X, y, N_FOLDS, RANDOM_STATE))
    xgb_row.update(cv_metrics(xgb_pipe, X, y, N_FOLDS, RANDOM_STATE))

    rf_rows.append(rf_row)
    dt_rows.append(dt_row)
    xgb_rows.append(xgb_row)

    # Fit full data + feature rankings
    safe_c = safe_filename(country)

    # Random Forest
    rf_pipe.fit(X, y)
    rf_features = rf_pipe.named_steps["prep"].get_feature_names_out()
    rf_imp = rf_pipe.named_steps["model"].feature_importances_
    pd.DataFrame({"feature": rf_features, "importance": rf_imp}) \
        .sort_values("importance", ascending=False) \
        .to_csv(RF_DIR / "feature_rankings" / f"{safe_c}_feature_importance.csv", index=False)

    # Decision Tree
    dt_pipe.fit(X, y)
    dt_features = dt_pipe.named_steps["prep"].get_feature_names_out()
    dt_imp = dt_pipe.named_steps["model"].feature_importances_
    pd.DataFrame({"feature": dt_features, "importance": dt_imp}) \
        .sort_values("importance", ascending=False) \
        .to_csv(DT_DIR / "feature_rankings" / f"{safe_c}_feature_importance.csv", index=False)

    # XGBoost
    xgb_pipe.fit(X, y)
    xgb_features = xgb_pipe.named_steps["prep"].get_feature_names_out()
    xgb_imp = xgb_pipe.named_steps["model"].feature_importances_
    pd.DataFrame({"feature": xgb_features, "importance": xgb_imp}) \
        .sort_values("importance", ascending=False) \
        .to_csv(XGB_DIR / "feature_rankings" / f"{safe_c}_feature_importance.csv", index=False)


# =========================
# 6) SAVE SUMMARY CSVs
# =========================
rf_summary = pd.DataFrame(rf_rows).sort_values(["R2_mean", "Country"], ascending=[False, True])
dt_summary = pd.DataFrame(dt_rows).sort_values(["R2_mean", "Country"], ascending=[False, True])
xgb_summary = pd.DataFrame(xgb_rows).sort_values(["R2_mean", "Country"], ascending=[False, True])

rf_summary.to_csv(RF_DIR / "per_country_cv_summary_random_forest.csv", index=False)
dt_summary.to_csv(DT_DIR / "per_country_cv_summary_decision_tree.csv", index=False)
xgb_summary.to_csv(XGB_DIR / "per_country_cv_summary_xgboost.csv", index=False)

combined = rf_summary.rename(columns={
    "R2_mean": "R2_RF", "R2_std": "R2_RF_std",
    "MAE_mean": "MAE_RF", "MAE_std": "MAE_RF_std",
    "RMSE_mean": "RMSE_RF", "RMSE_std": "RMSE_RF_std",
    "note": "note_RF"
}).merge(
    dt_summary[[
        "Country", "R2_mean", "R2_std", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "note"
    ]].rename(columns={
        "R2_mean": "R2_DT", "R2_std": "R2_DT_std",
        "MAE_mean": "MAE_DT", "MAE_std": "MAE_DT_std",
        "RMSE_mean": "RMSE_DT", "RMSE_std": "RMSE_DT_std",
        "note": "note_DT"
    }),
    on="Country", how="left"
).merge(
    xgb_summary[[
        "Country", "R2_mean", "R2_std", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "note"
    ]].rename(columns={
        "R2_mean": "R2_XGB", "R2_std": "R2_XGB_std",
        "MAE_mean": "MAE_XGB", "MAE_std": "MAE_XGB_std",
        "RMSE_mean": "RMSE_XGB", "RMSE_std": "RMSE_XGB_std",
        "note": "note_XGB"
    }),
    on="Country", how="left"
)

combined.to_csv(OUT_ROOT / "per_country_cv_summary_combined.csv", index=False)


# =========================
# 7) MAPS (R² colorbar) for EACH MODEL
# =========================
world = load_world_admin0(NE_ZIP)

plot_r2_map(
    world=world,
    results_df=rf_summary[["Country", "R2_mean"]].rename(columns={"R2_mean": "R2"}),
    value_col="R2",
    title="Mean CV R² per country (Random Forest)",
    out_png=RF_DIR / "maps" / "r2_map_random_forest.png",
    out_pdf=RF_DIR / "maps" / "r2_map_random_forest.pdf",
)

plot_r2_map(
    world=world,
    results_df=dt_summary[["Country", "R2_mean"]].rename(columns={"R2_mean": "R2"}),
    value_col="R2",
    title="Mean CV R² per country (Decision Tree)",
    out_png=DT_DIR / "maps" / "r2_map_decision_tree.png",
    out_pdf=DT_DIR / "maps" / "r2_map_decision_tree.pdf",
)

plot_r2_map(
    world=world,
    results_df=xgb_summary[["Country", "R2_mean"]].rename(columns={"R2_mean": "R2"}),
    value_col="R2",
    title="Mean CV R² per country (XGBoost)",
    out_png=XGB_DIR / "maps" / "r2_map_xgboost.png",
    out_pdf=XGB_DIR / "maps" / "r2_map_xgboost.pdf",
)

print("\nDone.")
print("Saved cleaning outputs:")
print(" -", OUT_ROOT / "clean_modeling_table.csv")
print(" -", OUT_ROOT / "dropped_rows.csv")
print(" -", OUT_ROOT / "cleaning_audit_summary.csv")
print("\nSaved summaries:")
print(" -", RF_DIR / "per_country_cv_summary_random_forest.csv")
print(" -", DT_DIR / "per_country_cv_summary_decision_tree.csv")
print(" -", XGB_DIR / "per_country_cv_summary_xgboost.csv")
print(" -", OUT_ROOT / "per_country_cv_summary_combined.csv")
print("\nSaved maps under:")
print(" -", RF_DIR / "maps/")
print(" -", DT_DIR / "maps/")
print(" -", XGB_DIR / "maps/")
print("\nFeature rankings saved under:")
print(" -", RF_DIR / "feature_rankings/")
print(" -", DT_DIR / "feature_rankings/")
print(" -", XGB_DIR / "feature_rankings/")



# %% Cell 2
# run_country_models_vaw.py
# ------------------------------------------------------------
# Per-country regression with Train/Val/Test + light tuning
# Models: RandomForest, DecisionTree, XGBoost
# Parallelized across countries (fast), avoids nested parallelism.
# Exports: summary CSV + feature rankings + R2 maps + cleaning audit.
# ------------------------------------------------------------

import os
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# pip install xgboost
from xgboost import XGBRegressor

# pip install geopandas
import geopandas as gpd

# pip install pycountry (optional but helps map name matching)
try:
    import pycountry
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# =========================
# USER SETTINGS
# =========================
INPUT_CSV = "Violence Against Women  Girls Data.csv"

OUT_ROOT = Path("results_paper_ready")
RF_DIR = OUT_ROOT / "random_forest"
DT_DIR = OUT_ROOT / "decision_tree"
XGB_DIR = OUT_ROOT / "xgboost"

MIN_ROWS_PER_COUNTRY = 40
RANDOM_STATE = 42

# Parallelization across countries:
# -1 uses all cores; if your machine gets sluggish, try 4 or 6
N_JOBS_COUNTRIES = -1

# RandomizedSearch budgets (keep small for speed; increase later)
RF_N_ITER  = 12
DT_N_ITER  = 12
XGB_N_ITER = 12

# Internal CV folds used inside RandomizedSearch on TRAIN only
TUNING_CV_FOLDS = 3

# Map data (Natural Earth Admin-0 countries)
NE_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
NE_ZIP = OUT_ROOT / "ne_110m_admin_0_countries.zip"
NE_SHP_IN_ZIP = "ne_110m_admin_0_countries.shp"


# =========================
# HELPERS
# =========================
def parse_year(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if dt.isna().mean() > 0.5:
        dt2 = pd.to_datetime(series, errors="coerce", dayfirst=False)
        dt = dt.fillna(dt2)
    return dt.dt.year

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in str(s)).strip()

def ensure_naturalearth_admin0(zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"[INFO] Downloading Natural Earth countries to: {zip_path}")
        urllib.request.urlretrieve(NE_URL, zip_path)

def load_world_admin0(zip_path: Path) -> gpd.GeoDataFrame:
    ensure_naturalearth_admin0(zip_path)
    world = gpd.read_file(f"zip://{zip_path}!{NE_SHP_IN_ZIP}")
    if "ADMIN" not in world.columns:
        raise ValueError(f"'ADMIN' column not found. Available columns: {list(world.columns)}")
    return world

def normalize_country_name(name: str) -> str:
    """
    Attempt to normalize dataset country names to NaturalEarth ADMIN names.
    This is best-effort; we also export unmatched names to a CSV.
    """
    name = str(name).strip()

    # common overrides (extend as needed)
    overrides = {
        "Congo Democratic Republic": "Democratic Republic of the Congo",
        "Congo, Dem. Rep.": "Democratic Republic of the Congo",
        "Congo Republic": "Republic of the Congo",
        "Congo, Rep.": "Republic of the Congo",
        "Cote d'Ivoire": "Ivory Coast",
        "Côte d’Ivoire": "Ivory Coast",
        "Côte d'Ivoire": "Ivory Coast",
        "Russian Federation": "Russia",
        "Viet Nam": "Vietnam",
        "Lao PDR": "Laos",
        "Iran": "Iran",
        "Sao Tome and Principe": "São Tomé and Príncipe",
        "Timor-Leste": "East Timor",
        "United States of America": "United States",
        "USA": "United States",
        "United Kingdom": "United Kingdom",
        "Bolivia (Plurinational State of)": "Bolivia",
        "Venezuela (Bolivarian Republic of)": "Venezuela",
        "Syrian Arab Republic": "Syria",
        "Türkiye": "Turkey",
    }
    if name in overrides:
        return overrides[name]

    if HAS_PYCOUNTRY:
        try:
            return pycountry.countries.lookup(name).name
        except Exception:
            return name

    return name

def plot_r2_map(world: gpd.GeoDataFrame, results_df: pd.DataFrame,
                value_col: str, title: str, out_png: Path, out_pdf: Path):
    """
    Choropleth map colored by R². Missing data shown in hatched grey.
    """
    gdf = world.merge(results_df, how="left", left_on="ADMIN", right_on="Country_NE")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

    gdf.plot(
        column=value_col,
        ax=ax,
        cmap="viridis",
        linewidth=0.25,
        edgecolor="black",
        legend=True,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "black",
            "hatch": "///",
            "label": "No data / unmatched / skipped"
        },
        legend_kwds={"label": "Test R²", "shrink": 0.6}
    )

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# =========================
# 1) READ + CLEAN (WITH AUDIT EXPORTS)
# =========================
def read_and_clean(input_csv: str, out_root: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(input_csv)

    expected = ["Country", "Gender", "Demographics Question", "Demographics Response",
                "Question", "Survey Year", "Value"]
    missing = [c for c in expected if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df_raw[expected].copy()

    # Keep originals for audit
    df["SurveyYear_raw"] = df["Survey Year"]
    df["Value_raw"] = df["Value"]

    # Parse year and numeric value
    df["SurveyYear"] = parse_year(df["Survey Year"])
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # Strip strings
    for c in ["Country", "Gender", "Demographics Question", "Demographics Response", "Question"]:
        df[c] = df[c].astype(str).str.strip()

    # Validity flags
    df["ok_country"] = df["Country"].notna() & (df["Country"] != "") & (df["Country"].str.lower() != "nan")
    df["ok_gender"] = df["Gender"].notna() & (df["Gender"] != "") & (df["Gender"].str.lower() != "nan")
    df["ok_demo_q"] = df["Demographics Question"].notna() & (df["Demographics Question"] != "") & (df["Demographics Question"].str.lower() != "nan")
    df["ok_demo_r"] = df["Demographics Response"].notna() & (df["Demographics Response"] != "") & (df["Demographics Response"].str.lower() != "nan")
    df["ok_question"] = df["Question"].notna() & (df["Question"] != "") & (df["Question"].str.lower() != "nan")
    df["ok_year"] = df["SurveyYear"].notna()
    df["ok_value"] = df["Value"].notna()

    df["is_valid_row"] = (
        df["ok_country"] &
        df["ok_gender"] &
        df["ok_demo_q"] &
        df["ok_demo_r"] &
        df["ok_question"] &
        df["ok_year"] &
        df["ok_value"]
    )

    df_clean = df[df["is_valid_row"]].copy()

    # Exports
    out_root.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_root / "clean_modeling_table.csv", index=False)
    df[~df["is_valid_row"]].to_csv(out_root / "dropped_rows.csv", index=False)

    audit = pd.DataFrame({
        "n_raw_rows": [len(df_raw)],
        "n_after_column_select": [len(df)],
        "n_clean_rows": [len(df_clean)],
        "n_dropped": [len(df) - len(df_clean)],
        "dropped_bad_country": [int((~df["ok_country"]).sum())],
        "dropped_bad_gender": [int((~df["ok_gender"]).sum())],
        "dropped_bad_demo_question": [int((~df["ok_demo_q"]).sum())],
        "dropped_bad_demo_response": [int((~df["ok_demo_r"]).sum())],
        "dropped_bad_question": [int((~df["ok_question"]).sum())],
        "dropped_bad_year": [int((~df["ok_year"]).sum())],
        "dropped_bad_value": [int((~df["ok_value"]).sum())],
    })
    audit.to_csv(out_root / "cleaning_audit_summary.csv", index=False)

    print("[INFO] Cleaning summary:")
    print(audit.to_string(index=False))
    print("[INFO] Saved cleaning outputs to:", out_root.resolve())
    return df_clean


# =========================
# 2) PREPROCESSING
# =========================
def build_preprocess():
    categorical = ["Gender", "Demographics Question", "Demographics Response", "Question"]
    numeric = ["SurveyYear"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop"
    )
    return preprocess


# =========================
# 3) PER-COUNTRY FIT/TUNE/EVAL
# =========================
def save_importance(best_pipe: Pipeline, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    feat = best_pipe.named_steps["prep"].get_feature_names_out()
    model = best_pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        pd.DataFrame({"feature": feat, "importance": imp}) \
            .sort_values("importance", ascending=False) \
            .to_csv(out_csv, index=False)

def tune_model(pipe: Pipeline, param_space: dict,
               X_train, y_train, X_val, y_val, X_test, y_test,
               n_iter: int, seed: int) -> dict:
    rs = RandomizedSearchCV(
        pipe,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring="r2",
        cv=TUNING_CV_FOLDS,
        random_state=seed,
        n_jobs=1,           # IMPORTANT: avoid nested parallelism
        verbose=0
    )
    rs.fit(X_train, y_train)
    best = rs.best_estimator_

    pred_val = best.predict(X_val)
    val_r2 = float(r2_score(y_val, pred_val))
    val_mae = float(mean_absolute_error(y_val, pred_val))
    val_rmse = rmse(y_val, pred_val)

    # Refit on Train+Val
    X_tv = pd.concat([X_train, X_val], axis=0)
    y_tv = np.concatenate([y_train, y_val])
    best.fit(X_tv, y_tv)

    pred_test = best.predict(X_test)
    test_r2 = float(r2_score(y_test, pred_test))
    test_mae = float(mean_absolute_error(y_test, pred_test))
    test_rmse = rmse(y_test, pred_test)

    return {
        "best_estimator": best,
        "best_params": rs.best_params_,
        "val_R2": val_r2,
        "val_MAE": val_mae,
        "val_RMSE": val_rmse,
        "test_R2": test_r2,
        "test_MAE": test_mae,
        "test_RMSE": test_rmse
    }

def process_country(country: str, d: pd.DataFrame, preprocess: ColumnTransformer,
                    min_rows: int, seed: int,
                    rf_dir: Path, dt_dir: Path, xgb_dir: Path) -> dict:
    X_cols = ["Gender", "Demographics Question", "Demographics Response", "Question", "SurveyYear"]
    y_col = "Value"

    n = len(d)
    summary = {
        "Country": country,
        "n_rows": n,
        "note": ""
    }

    if n < min_rows:
        summary["note"] = f"Skipped (<{min_rows} rows)"
        return summary

    X = d[X_cols]
    y = d[y_col].to_numpy()

    # Train/Val/Test: 70/15/15
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=seed)

    # Models (n_jobs=1 to avoid nested parallelism)
    rf = RandomForestRegressor(random_state=seed, n_jobs=1)
    dt = DecisionTreeRegressor(random_state=seed)
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=1,
        tree_method="hist"
    )

    rf_pipe = Pipeline([("prep", preprocess), ("model", rf)])
    dt_pipe = Pipeline([("prep", preprocess), ("model", dt)])
    xgb_pipe = Pipeline([("prep", preprocess), ("model", xgb)])

    # Parameter spaces (small/fast; expand later for a stronger paper)
    rf_space = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.5, None],
    }
    dt_space = {
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
        "model__min_samples_split": [2, 5, 10],
    }
    xgb_space = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
        "model__reg_lambda": [1.0, 5.0, 10.0],
    }

    rf_res = tune_model(rf_pipe, rf_space, X_train, y_train, X_val, y_val, X_test, y_test, RF_N_ITER, seed)
    dt_res = tune_model(dt_pipe, dt_space, X_train, y_train, X_val, y_val, X_test, y_test, DT_N_ITER, seed)
    xgb_res = tune_model(xgb_pipe, xgb_space, X_train, y_train, X_val, y_val, X_test, y_test, XGB_N_ITER, seed)

    # Save feature rankings (best estimator refit on Train+Val)
    safe_c = safe_filename(country)
    save_importance(rf_res["best_estimator"], rf_dir / "feature_rankings" / f"{safe_c}_feature_importance.csv")
    save_importance(dt_res["best_estimator"], dt_dir / "feature_rankings" / f"{safe_c}_feature_importance.csv")
    save_importance(xgb_res["best_estimator"], xgb_dir / "feature_rankings" / f"{safe_c}_feature_importance.csv")

    # Save best params too (handy for paper)
    pd.DataFrame([rf_res["best_params"]]).to_csv(rf_dir / "best_params" / f"{safe_c}_best_params.csv", index=False)
    pd.DataFrame([dt_res["best_params"]]).to_csv(dt_dir / "best_params" / f"{safe_c}_best_params.csv", index=False)
    pd.DataFrame([xgb_res["best_params"]]).to_csv(xgb_dir / "best_params" / f"{safe_c}_best_params.csv", index=False)

    # Summary
    summary.update({
        "RF_val_R2": rf_res["val_R2"], "RF_val_MAE": rf_res["val_MAE"], "RF_val_RMSE": rf_res["val_RMSE"],
        "RF_test_R2": rf_res["test_R2"], "RF_test_MAE": rf_res["test_MAE"], "RF_test_RMSE": rf_res["test_RMSE"],

        "DT_val_R2": dt_res["val_R2"], "DT_val_MAE": dt_res["val_MAE"], "DT_val_RMSE": dt_res["val_RMSE"],
        "DT_test_R2": dt_res["test_R2"], "DT_test_MAE": dt_res["test_MAE"], "DT_test_RMSE": dt_res["test_RMSE"],

        "XGB_val_R2": xgb_res["val_R2"], "XGB_val_MAE": xgb_res["val_MAE"], "XGB_val_RMSE": xgb_res["val_RMSE"],
        "XGB_test_R2": xgb_res["test_R2"], "XGB_test_MAE": xgb_res["test_MAE"], "XGB_test_RMSE": xgb_res["test_RMSE"],
    })

    return summary


# =========================
# MAIN
# =========================
def main():
    # Output folders
    for d in [RF_DIR, DT_DIR, XGB_DIR]:
        (d / "feature_rankings").mkdir(parents=True, exist_ok=True)
        (d / "best_params").mkdir(parents=True, exist_ok=True)
        (d / "maps").mkdir(parents=True, exist_ok=True)

    # Read + clean
    df_clean = read_and_clean(INPUT_CSV, OUT_ROOT)

    # Build preprocess
    preprocess = build_preprocess()

    # Per-country parallel processing
    countries = sorted(df_clean["Country"].unique())
    tasks = [(c, df_clean[df_clean["Country"] == c].copy()) for c in countries]

    print(f"[INFO] Countries found: {len(countries)}")
    print(f"[INFO] Parallel jobs (countries): {N_JOBS_COUNTRIES}")

    rows = Parallel(n_jobs=N_JOBS_COUNTRIES, backend="loky")(
        delayed(process_country)(
            country=c,
            d=d,
            preprocess=preprocess,
            min_rows=MIN_ROWS_PER_COUNTRY,
            seed=RANDOM_STATE,
            rf_dir=RF_DIR,
            dt_dir=DT_DIR,
            xgb_dir=XGB_DIR
        )
        for c, d in tasks
    )

    summary = pd.DataFrame(rows)

    # Save combined summary
    summary_path = OUT_ROOT / "per_country_train_val_test_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("[INFO] Saved:", summary_path)

    # =========================
    # MAPS: Test R²
    # =========================
    world = load_world_admin0(NE_ZIP)

    # Normalize names for matching
    summary["Country_NE"] = summary["Country"].apply(normalize_country_name)

    # Report unmatched (for debugging)
    admin_set = set(world["ADMIN"].astype(str).unique())
    used = summary.loc[summary["note"].fillna("") == "", "Country_NE"].astype(str)
    unmatched = sorted(set(used.unique()) - admin_set)

    match_report = pd.DataFrame({
        "Country_in_dataset": summary["Country"],
        "Country_NE": summary["Country_NE"],
        "is_in_NaturalEarth_ADMIN": summary["Country_NE"].astype(str).isin(admin_set)
    }).drop_duplicates()

    match_report.to_csv(OUT_ROOT / "country_name_match_report.csv", index=False)
    pd.DataFrame({"unmatched_Country_NE": unmatched}).to_csv(OUT_ROOT / "unmatched_country_names.csv", index=False)

    print(f"[INFO] Unmatched country names for map: {len(unmatched)} (see unmatched_country_names.csv)")

    # Create R2 maps (Test R2)
    # Only include countries not skipped
    ok = summary[summary["note"].fillna("") == ""].copy()

    plot_r2_map(
        world=world,
        results_df=ok[["Country_NE", "RF_test_R2"]].rename(columns={"RF_test_R2": "R2"}),
        value_col="R2",
        title="Test R² per country (Random Forest)",
        out_png=RF_DIR / "maps" / "r2_map_random_forest_test.png",
        out_pdf=RF_DIR / "maps" / "r2_map_random_forest_test.pdf",
    )

    plot_r2_map(
        world=world,
        results_df=ok[["Country_NE", "DT_test_R2"]].rename(columns={"DT_test_R2": "R2"}),
        value_col="R2",
        title="Test R² per country (Decision Tree)",
        out_png=DT_DIR / "maps" / "r2_map_decision_tree_test.png",
        out_pdf=DT_DIR / "maps" / "r2_map_decision_tree_test.pdf",
    )

    plot_r2_map(
        world=world,
        results_df=ok[["Country_NE", "XGB_test_R2"]].rename(columns={"XGB_test_R2": "R2"}),
        value_col="R2",
        title="Test R² per country (XGBoost)",
        out_png=XGB_DIR / "maps" / "r2_map_xgboost_test.png",
        out_pdf=XGB_DIR / "maps" / "r2_map_xgboost_test.pdf",
    )

    print("\n[INFO] Done.")
    print("[INFO] Outputs:")
    print(" -", OUT_ROOT / "clean_modeling_table.csv")
    print(" -", OUT_ROOT / "dropped_rows.csv")
    print(" -", OUT_ROOT / "cleaning_audit_summary.csv")
    print(" -", OUT_ROOT / "per_country_train_val_test_summary.csv")
    print(" -", OUT_ROOT / "country_name_match_report.csv")
    print(" -", OUT_ROOT / "unmatched_country_names.csv")
    print(" -", RF_DIR / "feature_rankings/")
    print(" -", DT_DIR / "feature_rankings/")
    print(" -", XGB_DIR / "feature_rankings/")
    print(" -", RF_DIR / "maps/")
    print(" -", DT_DIR / "maps/")
    print(" -", XGB_DIR / "maps/")


if __name__ == "__main__":
    main()



# %% Cell 3

