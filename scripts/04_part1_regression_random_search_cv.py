# Auto-extracted from notebooks/04_part1_regression_random_search_cv.ipynb

# Review/reproducibility helper script. Original notebook is retained in notebooks/.



# %% Cell 0
# results_oulad_regression_days_to_withdraw_pub_v7_MULTI_HALVING.py
# ------------------------------------------------------------
# OULAD — Regression (Publication v7, MULTI-model, FAST tuning)
#
# Target: days_until_unregistration for students who withdrew (is_unregistered==1)
#
# This script is designed to be "paper-ready" and fix the issues you mentioned:
#  - Train & compare 4 regressors globally: RF, DT, HGB, XGB
#  - Fast hyperparameter optimization (Successive Halving if available; else RandomizedSearchCV)
#  - Global performance: repeated stratified splits -> mean ± SD for R² and RMSE
#    -> barplot with error bars (SVG+PDF), R² dark grey, RMSE light grey
#  - Pick best model (max mean R²; tie-break min mean RMSE), refit on train+val, evaluate on held-out TEST
#  - Best model: per-region and per-module TEST metrics -> CSV + barplots (R² and RMSE)
#  - Best model: REAL map choropleth (UK regions) of TEST R² by region (requires GeoJSON + matching labels)
#  - Best model: feature ranking using permutation importance (encoded + aggregated):
#       * Global
#       * Per-region (top-k long + best feature)
#       * Per-module (top-k long + best feature)
#
# INPUT OPTIONS:
#   A) If you already have the preprocessed modeling table:
#        /mnt/data/clean_modeling_table_days_to_withdraw.csv   (recommended here)
#   B) Or build from original OULAD zip (slower):
#        /mnt/data/open-university-learning-analytics-dataset.zip
#
# OUTPUT:
#   results_oulad_regression_days_to_withdraw_pub_v7_MULTI_HALVING/
# ------------------------------------------------------------

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
)

# --- optional: successive halving (MUCH faster) ---
try:
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV
    HAS_HALVING = True
except Exception:
    HAS_HALVING = False
    HalvingRandomSearchCV = None

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

import joblib

# ---- XGBoost ----
try:
    from xgboost import XGBRegressor
except Exception as e:
    raise ImportError(
        "xgboost is required.\n"
        "Install with: pip install xgboost\n"
        f"Original error:\n{e}"
    )

# ---- Geo stack for REAL map ----
# pip install geopandas shapely pyproj fiona
try:
    import geopandas as gpd
except Exception:
    gpd = None


# =========================
# USER SETTINGS
# =========================

# Option A: use already-built modeling table (fast)
INPUT_MODELING_CSV = Path("open-university-learning-analytics-dataset/clean_modeling_table_days_to_withdraw.csv")

# Option B: build from original OULAD zip (slow)
ZIP_PATH = Path("open-university-learning-analytics-dataset")
ZIP_BASE = "open-university-learning-analytics-dataset/versions/1/anonymiseddata/"

# Output
OUT_ROOT = Path("results_oulad_regression_days_to_withdraw_pub_v7_MULTI_HALVING")
PLOTS_DIR = OUT_ROOT / "plots"
BEST_DIR = OUT_ROOT / "best_model"
BY_REGION_DIR = OUT_ROOT / "by_region"
BY_MODULE_DIR = OUT_ROOT / "by_module"
MAPS_DIR = OUT_ROOT / "maps"
FEATURES_DIR = OUT_ROOT / "feature_importance"

RANDOM_STATE = 42

# Enriched engagement (only used if building from ZIP)
STUDENTVLE_CHUNKSIZE = 700_000
TOP_ACTIVITY_TYPES = 12
EARLY_WINDOW_DAYS = 30

# Repeated split evaluation (global)
N_REPEATS = 10
REPEAT_TEST_SIZE = 0.15
N_BINS_STRATIFY = 10  # stratify regression by quantile bins

# Tuning
TUNING_CV_FOLDS = 3

# RandomizedSearch fallback budgets (if halving unavailable)
N_ITER_FALLBACK = {"RF": 25, "DT": 25, "HGB": 45, "XGB": 45}

# Subgroup stability
MIN_TEST_SAMPLES_PER_REGION = 60
MIN_TEST_SAMPLES_PER_MODULE = 100

# Feature ranking
TOPK_FEATURES_GLOBAL = 25
TOPK_FEATURES_GROUP = 15
PERM_N_REPEATS = 10
PERM_SCORING = "neg_mean_absolute_error"

# Export
EXPORT_SVG = True
EXPORT_PDF = True
SVG_DPI = 600

# Grayscale requirement
COLOR_R2 = "#4d4d4d"    # darker grey
COLOR_RMSE = "#bdbdbd"  # lighter grey
ERR_COLOR = "#8c8c8c"

# REAL UK map input (you must provide a correct GeoJSON with polygons)
UK_REGIONS_GEOJSON = Path("uk_regions.geojson")  # put your file next to the script
REGION_KEY_COL = "region"                        # column name inside GeoJSON matching df['region'] exactly


# =========================
# SMALL UTILITIES
# =========================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def set_pub_style_white():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    ensure_dir(out_base.parent)
    if EXPORT_SVG:
        fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", dpi=SVG_DPI)
    if EXPORT_PDF:
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =========================
# ZIP READ HELPERS (if building from ZIP)
# =========================
def read_csv_from_zip(zf: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    with zf.open(name) as f:
        return pd.read_csv(f, **kwargs)


def iter_csv_chunks_from_zip(zf: zipfile.ZipFile, name: str, chunksize: int, **kwargs):
    with zf.open(name) as f:
        for chunk in pd.read_csv(f, chunksize=chunksize, **kwargs):
            yield chunk


# =========================
# PLOTTING HELPERS
# =========================
def save_grouped_bar_mean_sd(means: pd.DataFrame, sds: pd.DataFrame, out_base: Path, title: str):
    """
    One figure: mean±SD for R2 and RMSE per model.
    R2 bars: dark grey; RMSE bars: light grey.
    """
    set_pub_style_white()
    ensure_dir(out_base.parent)

    models = means.index.tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)

    ax.bar(
        x - width/2,
        means["r2_mean"].values,
        width=width,
        yerr=sds["r2_sd"].values,
        capsize=4,
        label="R²",
        color=COLOR_R2,
        edgecolor=COLOR_R2,
        error_kw={"ecolor": ERR_COLOR, "elinewidth": 1.2, "capthick": 1.2}
    )

    ax.bar(
        x + width/2,
        means["rmse_mean"].values,
        width=width,
        yerr=sds["rmse_sd"].values,
        capsize=4,
        label="RMSE",
        color=COLOR_RMSE,
        edgecolor=COLOR_RMSE,
        error_kw={"ecolor": ERR_COLOR, "elinewidth": 1.2, "capthick": 1.2}
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, out_base)


def save_barplot(values: pd.Series,
                 title: str,
                 ylabel: str,
                 out_base: Path,
                 rotate_xticks: int = 25,
                 ylim: Optional[Tuple[float, float]] = None,
                 bar_color: str = "#4d4d4d"):
    set_pub_style_white()
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
    x = np.arange(len(values))

    ax.bar(x, values.values, color=bar_color, edgecolor=bar_color)

    ax.set_xticks(x)
    ax.set_xticklabels(values.index.tolist(), rotation=rotate_xticks, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)

    fig.tight_layout()
    save_figure(fig, out_base)


def save_real_map_choropleth_r2(per_region_df: pd.DataFrame, out_base: Path, title: str):
    """
    TRUE UK regions choropleth for R² (requires geopandas + uk_regions.geojson).
    Expects per_region_df columns: ['region','test_R2'].
    GeoJSON must contain REGION_KEY_COL matching OULAD region strings exactly.
    """
    if gpd is None:
        print("[WARN] geopandas not installed; skipping REAL map.")
        return
    if not UK_REGIONS_GEOJSON.exists():
        print(f"[WARN] Missing {UK_REGIONS_GEOJSON}; skipping REAL map.")
        return

    set_pub_style_white()
    ensure_dir(out_base.parent)

    gdf = gpd.read_file(UK_REGIONS_GEOJSON)
    if REGION_KEY_COL not in gdf.columns:
        raise ValueError(
            f"GeoJSON does not contain REGION_KEY_COL='{REGION_KEY_COL}'. "
            f"Available columns: {list(gdf.columns)}"
        )

    m = per_region_df[["region", "test_R2"]].copy()
    m["region"] = m["region"].astype(str).str.strip()
    gdf[REGION_KEY_COL] = gdf[REGION_KEY_COL].astype(str).str.strip()

    # Debug lists to diagnose mismatched labels
    print("[DEBUG] Regions in TEST:", sorted(m["region"].unique()))
    print("[DEBUG] Regions in GEOJSON:", sorted(gdf[REGION_KEY_COL].unique()))

    gdfm = gdf.merge(m, left_on=REGION_KEY_COL, right_on="region", how="left")

    # Greys: higher R² darker
    fig, ax = plt.subplots(figsize=(7.8, 9.5), dpi=300)
    gdfm.plot(
        column="test_R2",
        ax=ax,
        cmap="Greys",
        legend=True,
        edgecolor="black",
        linewidth=0.6,
        missing_kwds={"color": "lightgrey", "edgecolor": "black", "hatch": "///", "label": "Missing"},
        vmin=-1.0,
        vmax=1.0,
    )
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, out_base)


# =========================
# 1) OPTIONAL: BUILD MODELING TABLE FROM ZIP (if needed)
# =========================
def pick_top_activity_types(vle: pd.DataFrame, top_k: int) -> List[str]:
    vc = vle["activity_type"].astype(str).value_counts()
    return vc.head(top_k).index.tolist()


def build_engagement_enriched(
    zf: zipfile.ZipFile,
    studentvle_name: str,
    vle_name: str,
    chunksize: int,
    top_activity_types: List[str],
    early_days: int
) -> pd.DataFrame:
    vle = read_csv_from_zip(zf, vle_name, usecols=["id_site", "activity_type"])
    vle["activity_type"] = vle["activity_type"].astype(str)
    mapping = dict(zip(vle["id_site"].astype(int), vle["activity_type"]))

    total_clicks: Dict[Tuple[int, str, str], float] = {}
    n_records: Dict[Tuple[int, str, str], int] = {}
    day_sets: Dict[Tuple[int, str, str], set] = {}
    distinct_sites: Dict[Tuple[int, str, str], set] = {}
    at_clicks: Dict[Tuple[int, str, str], Dict[str, float]] = {}
    early_clicks: Dict[Tuple[int, str, str], float] = {}
    late_clicks: Dict[Tuple[int, str, str], float] = {}

    usecols = ["id_student", "code_module", "code_presentation", "id_site", "date", "sum_click"]

    for chunk in iter_csv_chunks_from_zip(zf, studentvle_name, chunksize=chunksize, usecols=usecols):
        chunk["sum_click"] = safe_float_series(chunk["sum_click"])
        chunk["date"] = safe_float_series(chunk["date"])
        chunk["id_site"] = safe_float_series(chunk["id_site"])

        chunk = chunk.dropna(subset=["id_student", "code_module", "code_presentation", "sum_click", "date", "id_site"])
        chunk["id_student"] = chunk["id_student"].astype(int)
        chunk["id_site"] = chunk["id_site"].astype(int)

        chunk["activity_type"] = chunk["id_site"].map(mapping).fillna("Unknown")
        chunk.loc[~chunk["activity_type"].isin(top_activity_types), "activity_type"] = "Other"

        for row in chunk.itertuples(index=False):
            key = (int(row.id_student), str(row.code_module), str(row.code_presentation))
            sc = float(row.sum_click)
            d = int(row.date)
            site = int(row.id_site)
            at = str(row.activity_type)

            total_clicks[key] = total_clicks.get(key, 0.0) + sc
            n_records[key] = n_records.get(key, 0) + 1
            day_sets.setdefault(key, set()).add(d)
            distinct_sites.setdefault(key, set()).add(site)

            at_clicks.setdefault(key, {})
            at_clicks[key][at] = at_clicks[key].get(at, 0.0) + sc

            if d <= early_days:
                early_clicks[key] = early_clicks.get(key, 0.0) + sc
            else:
                late_clicks[key] = late_clicks.get(key, 0.0) + sc

    rows = []
    for key in total_clicks.keys():
        days = day_sets.get(key, set())
        sites = distinct_sites.get(key, set())

        tot = float(total_clicks.get(key, 0.0))
        early = float(early_clicks.get(key, 0.0))
        late = float(late_clicks.get(key, 0.0))

        rec = {
            "id_student": key[0],
            "code_module": key[1],
            "code_presentation": key[2],
            "total_clicks": tot,
            "n_vle_records": int(n_records.get(key, 0)),
            "n_active_days": int(len(days)),
            "n_distinct_sites": int(len(sites)),
            "clicks_per_active_day": (tot / max(len(days), 1)),
            "early_clicks": early,
            "late_clicks": late,
            "early_late_ratio": (early / max(late, 1.0)),
        }

        dct = at_clicks.get(key, {})
        for at in top_activity_types + ["Other", "Unknown"]:
            rec[f"clicks_at_{at}"] = float(dct.get(at, 0.0))

        rows.append(rec)

    eng = pd.DataFrame(rows)

    # log1p for heavy tails
    log_cols = ["total_clicks", "early_clicks", "late_clicks", "clicks_per_active_day"] + \
               [c for c in eng.columns if c.startswith("clicks_at_")]
    for c in log_cols:
        eng[f"log1p_{c}"] = np.log1p(pd.to_numeric(eng[c], errors="coerce").fillna(0.0))

    return eng


def build_registration_features(reg: pd.DataFrame) -> pd.DataFrame:
    r = reg.copy()
    for c in ["date_registration", "date_unregistration"]:
        r[c] = safe_float_series(r[c])

    r["is_unregistered"] = r["date_unregistration"].notna().astype(int)
    r["days_until_unregistration"] = (r["date_unregistration"] - r["date_registration"])
    r["days_until_unregistration"] = r["days_until_unregistration"].fillna(-1)

    keep = [
        "id_student", "code_module", "code_presentation",
        "date_registration", "date_unregistration",
        "is_unregistered", "days_until_unregistration"
    ]
    return r[keep]


def read_and_build_modeling_table_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        student_info = read_csv_from_zip(zf, ZIP_BASE + "studentInfo.csv")
        student_reg = read_csv_from_zip(zf, ZIP_BASE + "studentRegistration.csv")
        vle = read_csv_from_zip(zf, ZIP_BASE + "vle.csv", usecols=["id_site", "activity_type"])

        top_ats = pick_top_activity_types(vle, TOP_ACTIVITY_TYPES)

        eng = build_engagement_enriched(
            zf=zf,
            studentvle_name=ZIP_BASE + "studentVle.csv",
            vle_name=ZIP_BASE + "vle.csv",
            chunksize=STUDENTVLE_CHUNKSIZE,
            top_activity_types=top_ats,
            early_days=EARLY_WINDOW_DAYS
        )

    reg_feat = build_registration_features(student_reg)

    df = (
        student_info.merge(eng, on=["id_student", "code_module", "code_presentation"], how="left")
                   .merge(reg_feat, on=["id_student", "code_module", "code_presentation"], how="left")
    )

    # withdrawn-only
    df = df[df["region"].notna()].copy()
    df = df[df["is_unregistered"] == 1].copy()

    df["days_until_unregistration"] = pd.to_numeric(df["days_until_unregistration"], errors="coerce")
    df = df[df["days_until_unregistration"].notna() & (df["days_until_unregistration"] >= 0)].copy()

    # fill engineered numeric cols
    numeric_fill_cols = [c for c in df.columns if (
        c.startswith("total_clicks") or c.startswith("n_") or c.startswith("early_") or c.startswith("late_")
        or c.startswith("clicks_") or c.startswith("log1p_")
        or c in ["date_registration"]
    )]
    for c in numeric_fill_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in ["studied_credits", "num_of_prev_attempts"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[c for c in ["studied_credits", "num_of_prev_attempts"] if c in df.columns])

    return df


# =========================
# 2) PREPROCESS
# =========================
def build_preprocess(df: pd.DataFrame) -> ColumnTransformer:
    target = "days_until_unregistration"
    id_col = "id_student"

    cat_cols = [c for c in [
        "code_module", "code_presentation", "gender", "region",
        "highest_education", "imd_band", "age_band", "disability"
    ] if c in df.columns]

    leak = {target, "is_unregistered", "date_unregistration", id_col}
    if "final_result" in df.columns:
        leak.add("final_result")

    numeric_cols = [
        c for c in df.columns
        if c not in leak and c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop"
    )
    return preprocess


def get_cat_cols_from_preprocessor(prep: ColumnTransformer) -> List[str]:
    try:
        return list(prep.transformers_[0][2])
    except Exception:
        return []


def get_feature_names_from_pipe(pipe: Pipeline, X_ref: pd.DataFrame) -> List[str]:
    """
    Robustly recover transformed feature names aligned to transformed X.
    Prevents "feature ranking not showing" due to length mismatches.
    """
    prep: ColumnTransformer = pipe.named_steps["prep"]
    try:
        names = [str(x) for x in prep.get_feature_names_out()]
    except Exception:
        names = None

    Xt = prep.transform(X_ref)
    n_out = Xt.shape[1]

    if names is None:
        return [f"feature_{i:04d}" for i in range(n_out)]
    if len(names) == n_out:
        return names
    if len(names) > n_out:
        return names[:n_out]
    return names + [f"extra_feature_{i:04d}" for i in range(n_out - len(names))]


def encoded_to_base_factory(cat_cols: List[str]):
    def encoded_to_base(name: str) -> str:
        if name.startswith("num__"):
            return name.replace("num__", "", 1)
        if name.startswith("cat__"):
            rest = name.replace("cat__", "", 1)
            matches = [c for c in cat_cols if rest.startswith(str(c) + "_") or rest == str(c)]
            if matches:
                return sorted(matches, key=len, reverse=True)[0]
            return rest.split("_", 1)[0]
        return name.split("__", 1)[-1].split("_", 1)[0]
    return encoded_to_base


# =========================
# 3) MODEL SPECS + SPACES
# =========================
def model_spaces():
    models = {
        "RF": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        "DT": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "HGB": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "XGB": XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0
        ),
    }

    # Halving-friendly: include smaller n_estimators values
    spaces = {
        "RF": {
            "model__n_estimators": [150, 300, 600, 900],
            "model__max_depth": [None, 12, 24, 36],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.5, None],
        },
        "DT": {
            "model__max_depth": [None, 6, 12, 24, 36],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__min_samples_split": [2, 5, 10],
        },
        "HGB": {
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_depth": [3, 5, 7, None],
            "model__max_leaf_nodes": [63, 127, 255],
            "model__min_samples_leaf": [10, 20, 50, 100],
            "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
        },
        "XGB": {
            "model__n_estimators": [200, 400, 800, 1200],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__min_child_weight": [1, 2, 5, 10],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_alpha": [0.0, 0.1, 1.0, 5.0],
            "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "model__gamma": [0.0, 0.1, 0.3, 1.0],
        },
    }
    return models, spaces


# =========================
# 4) FAST TUNER (HALVING > RANDOMIZED)
# =========================
def tune_model(name: str,
               preprocess: ColumnTransformer,
               X_train, y_train,
               spaces: Dict[str, Dict],
               fallback_n_iter: int) -> Tuple[Pipeline, Dict]:
    """
    Prefer HalvingRandomSearchCV if available; else RandomizedSearchCV.
    - Tuning metric: R² (consistent with selection rule)
    - Avoid nested parallelism: n_jobs=1
    """
    models, _spaces = model_spaces()
    pipe = Pipeline([("prep", preprocess), ("model", models[name])])

    if HAS_HALVING:
        if name in ("RF", "XGB") and "model__n_estimators" in spaces[name]:
            resource = "model__n_estimators"
            tuner = HalvingRandomSearchCV(
                estimator=pipe,
                param_distributions=spaces[name],
                scoring="r2",
                cv=TUNING_CV_FOLDS,
                factor=3,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=0,
                error_score="raise",
                resource=resource,
                max_resources=max(spaces[name]["model__n_estimators"]),
                min_resources=min(spaces[name]["model__n_estimators"]),
            )
        else:
            tuner = HalvingRandomSearchCV(
                estimator=pipe,
                param_distributions=spaces[name],
                scoring="r2",
                cv=TUNING_CV_FOLDS,
                factor=3,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=0,
                error_score="raise",
                resource="n_samples",
                max_resources="auto",
            )
        tuner.fit(X_train, y_train)
        return tuner.best_estimator_, tuner.best_params_

    rs = RandomizedSearchCV(
        pipe,
        param_distributions=spaces[name],
        n_iter=fallback_n_iter,
        scoring="r2",
        cv=TUNING_CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
        error_score="raise",
    )
    rs.fit(X_train, y_train)
    return rs.best_estimator_, rs.best_params_


# =========================
# 5) GLOBAL REPEATED SPLIT EVAL (mean±SD)
# =========================
def repeated_split_eval(best_pipes: Dict[str, Pipeline], X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Refit each best_pipe on each split (no re-tuning), compute R² and RMSE on test.
    Returns means_df, sds_df indexed by model name.
    """
    y_s = pd.Series(y)
    y_bins = pd.qcut(y_s.rank(method="first"), q=N_BINS_STRATIFY, labels=False, duplicates="drop")

    splitter = StratifiedShuffleSplit(
        n_splits=N_REPEATS,
        test_size=REPEAT_TEST_SIZE,
        random_state=RANDOM_STATE
    )

    store = {m: {"r2": [], "rmse": [], "mae": []} for m in best_pipes.keys()}

    for tr_idx, te_idx in splitter.split(X, y_bins):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        for m, pipe in best_pipes.items():
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            store[m]["r2"].append(float(r2_score(y_te, pred)))
            store[m]["rmse"].append(float(rmse(y_te, pred)))
            store[m]["mae"].append(float(mean_absolute_error(y_te, pred)))

    means = pd.DataFrame({
        m: {"r2_mean": np.mean(d["r2"]), "rmse_mean": np.mean(d["rmse"]), "mae_mean": np.mean(d["mae"])}
        for m, d in store.items()
    }).T

    sds = pd.DataFrame({
        m: {"r2_sd": np.std(d["r2"], ddof=1), "rmse_sd": np.std(d["rmse"], ddof=1), "mae_sd": np.std(d["mae"], ddof=1)}
        for m, d in store.items()
    }).T

    return means, sds


def pick_best_model(means: pd.DataFrame) -> str:
    tmp = means.copy()
    tmp["rmse_mean"] = pd.to_numeric(tmp["rmse_mean"])
    tmp["r2_mean"] = pd.to_numeric(tmp["r2_mean"])
    tmp = tmp.sort_values(["r2_mean", "rmse_mean"], ascending=[False, True])
    return str(tmp.index[0])


# =========================
# 6) GROUP METRICS + FEATURE IMPORTANCE
# =========================
def per_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray, group_name: str) -> pd.DataFrame:
    df = pd.DataFrame({"group": group.astype(str), "y_true": y_true, "y_pred": y_pred})
    rows = []
    for g, sub in df.groupby("group"):
        if len(sub) < 2:
            continue
        rows.append({
            group_name: g,
            "n_test": int(len(sub)),
            "test_R2": float(r2_score(sub["y_true"], sub["y_pred"])),
            "test_RMSE": float(rmse(sub["y_true"].values, sub["y_pred"].values)),
            "test_MAE": float(mean_absolute_error(sub["y_true"].values, sub["y_pred"].values)),
        })
    return pd.DataFrame(rows)


def permutation_importance_aggregated(pipe: Pipeline, X_sub: pd.DataFrame, y_sub: np.ndarray) -> Tuple[pd.Series, pd.Series]:
    prep: ColumnTransformer = pipe.named_steps["prep"]
    cat_cols = get_cat_cols_from_preprocessor(prep)
    enc2base = encoded_to_base_factory(cat_cols)

    perm = permutation_importance(
        pipe,
        X_sub, y_sub,
        scoring=PERM_SCORING,
        n_repeats=PERM_N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    names = get_feature_names_from_pipe(pipe, X_sub)
    imp_encoded = pd.Series(perm.importances_mean, index=names).sort_values(ascending=False)

    base_index = [enc2base(n) for n in imp_encoded.index]
    imp_agg = imp_encoded.groupby(base_index).sum().sort_values(ascending=False)

    return imp_encoded, imp_agg


def save_feature_ranking_plots(imp_agg: pd.Series, out_base: Path, title: str, topk: int):
    top = imp_agg.sort_values(ascending=True).tail(topk)
    set_pub_style_white()
    fig, ax = plt.subplots(figsize=(10.0, 7.0), dpi=300)
    y = np.arange(len(top))
    ax.barh(y, top.values, color=COLOR_R2, edgecolor=COLOR_R2)
    ax.set_yticks(y)
    ax.set_yticklabels(top.index.tolist())
    ax.set_xlabel(f"Permutation importance (Δ {PERM_SCORING}) — aggregated")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, out_base)


# =========================
# MAIN
# =========================
def main():
    # Output folders
    for d in [OUT_ROOT, PLOTS_DIR, BEST_DIR, BY_REGION_DIR, BY_MODULE_DIR, MAPS_DIR, FEATURES_DIR]:
        ensure_dir(d)

    # -------------------------
    # Load / build modeling data
    # -------------------------
    if INPUT_MODELING_CSV.exists():
        df = pd.read_csv(INPUT_MODELING_CSV)
        print("[INFO] Loaded modeling table:", INPUT_MODELING_CSV)
    else:
        if not ZIP_PATH.exists():
            raise FileNotFoundError(
                f"Neither modeling CSV nor ZIP found.\n"
                f"Missing: {INPUT_MODELING_CSV} and {ZIP_PATH}"
            )
        print("[INFO] Building modeling table from ZIP (slow):", ZIP_PATH)
        df = read_and_build_modeling_table_from_zip(ZIP_PATH)
        df.to_csv(OUT_ROOT / "clean_modeling_table_days_to_withdraw.csv", index=False)

    # Ensure target exists
    if "days_until_unregistration" not in df.columns:
        raise ValueError("Expected 'days_until_unregistration' in modeling table.")

    # Clean minimal expectations
    df = df.copy()
    df["days_until_unregistration"] = pd.to_numeric(df["days_until_unregistration"], errors="coerce")
    df = df.dropna(subset=["days_until_unregistration"]).copy()

    # Global dataset
    y = df["days_until_unregistration"].to_numpy()
    X = df.drop(columns=["days_until_unregistration"])

    preprocess = build_preprocess(df)

    # Single split for final TEST reporting + subgroup analyses
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE)

    # -------------------------
    # Tune each model (fast)
    # -------------------------
    models, spaces = model_spaces()
    tuned: Dict[str, Pipeline] = {}
    params_out: Dict[str, Dict] = {}

    for name in ["RF", "DT", "HGB", "XGB"]:
        print(f"[INFO] Tuning {name} ... (Halving={HAS_HALVING})")
        best_pipe, best_params = tune_model(
            name=name,
            preprocess=preprocess,
            X_train=X_train,
            y_train=y_train,
            spaces=spaces,
            fallback_n_iter=N_ITER_FALLBACK[name]
        )
        tuned[name] = best_pipe
        params_out[name] = best_params
        pd.DataFrame([best_params]).to_csv(OUT_ROOT / f"best_params_{name}.csv", index=False)

    # -------------------------
    # Repeated split eval (mean ± SD)
    # -------------------------
    means_df, sds_df = repeated_split_eval(tuned, X, y)
    means_df.to_csv(OUT_ROOT / "global_repeated_split_means.csv")
    sds_df.to_csv(OUT_ROOT / "global_repeated_split_sds.csv")

    global_summary = means_df.join(sds_df).copy()
    global_summary.to_csv(OUT_ROOT / "global_models_summary_means_sds.csv")

    # Global barplot
    save_grouped_bar_mean_sd(
        means=means_df.loc[["RF", "DT", "HGB", "XGB"]],
        sds=sds_df.loc[["RF", "DT", "HGB", "XGB"]],
        out_base=PLOTS_DIR / "global_models_R2_RMSE_meanSD",
        title=f"OULAD regression (withdrawn only) — global performance across {N_REPEATS} splits (mean ± SD)"
    )

    # Choose best model
    best_name = pick_best_model(means_df)
    print("[INFO] Best model (max mean R²; tie-break min mean RMSE):", best_name)

    # -------------------------
    # Fit best model on TRAIN+VAL, evaluate on TEST
    # -------------------------
    best_pipe: Pipeline = tuned[best_name]
    X_tv = pd.concat([X_train, X_val], axis=0)
    y_tv = np.concatenate([y_train, y_val])

    best_pipe.fit(X_tv, y_tv)
    test_pred = best_pipe.predict(X_test)

    test_metrics = {
        "test_R2": float(r2_score(y_test, test_pred)),
        "test_RMSE": float(rmse(y_test, test_pred)),
        "test_MAE": float(mean_absolute_error(y_test, test_pred)),
    }
    pd.DataFrame([{"model": best_name, **test_metrics}]).to_csv(
        BEST_DIR / "best_model_single_split_test_metrics.csv", index=False
    )

    # Save best model + metadata
    joblib.dump(best_pipe, BEST_DIR / "best_model_pipeline.joblib")
    meta = {
        "best_model": best_name,
        "selection_rule": "max(mean R2), tie-break min(mean RMSE) over repeated splits",
        "repeated_split": {
            "n_repeats": N_REPEATS,
            "test_size": REPEAT_TEST_SIZE,
            "means": means_df.loc[best_name].to_dict(),
            "sds": sds_df.loc[best_name].to_dict(),
        },
        "single_split_test_metrics": test_metrics,
        "split": {"train": 0.70, "val": 0.15, "test": 0.15},
        "tuning": {
            "halving_used": bool(HAS_HALVING),
            "cv_folds": TUNING_CV_FOLDS,
            "fallback_randomized_n_iter": N_ITER_FALLBACK,
        },
        "perm_scoring": PERM_SCORING,
        "perm_n_repeats": PERM_N_REPEATS,
        "best_params_all_models": params_out,
        "notes": [
            "Withdrawn students only (is_unregistered==1).",
            "days_until_unregistration = date_unregistration - date_registration.",
            "Real choropleth requires uk_regions.geojson with region labels matching OULAD exactly."
        ],
    }
    with open(BEST_DIR / "best_model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # -------------------------
    # Best model: per-region + per-module TEST metrics
    # -------------------------
    if "region" not in X_test.columns:
        raise ValueError("Expected 'region' column in X_test but not found.")
    if "code_module" not in X_test.columns:
        raise ValueError("Expected 'code_module' column in X_test but not found.")

    per_region_df = per_group_metrics(y_test, test_pred, X_test["region"].astype(str).values, "region")
    per_module_df = per_group_metrics(y_test, test_pred, X_test["code_module"].astype(str).values, "code_module")

    per_region_df.to_csv(BY_REGION_DIR / "best_model_test_metrics_by_region.csv", index=False)
    per_module_df.to_csv(BY_MODULE_DIR / "best_model_test_metrics_by_module.csv", index=False)

    # Barplots: R2 (dark), RMSE (light)
    r2_region = per_region_df.dropna(subset=["test_R2"]).set_index("region")["test_R2"].sort_values(ascending=False)
    rmse_region = per_region_df.set_index("region")["test_RMSE"].sort_values(ascending=True)

    save_barplot(
        values=r2_region,
        title=f"Best model ({best_name}) — TEST R² by region",
        ylabel="R²",
        out_base=BY_REGION_DIR / "best_model_test_R2_by_region",
        rotate_xticks=35,
        ylim=(-1.0, 1.0),
        bar_color=COLOR_R2
    )
    save_barplot(
        values=rmse_region,
        title=f"Best model ({best_name}) — TEST RMSE by region",
        ylabel="RMSE (days)",
        out_base=BY_REGION_DIR / "best_model_test_RMSE_by_region",
        rotate_xticks=35,
        bar_color=COLOR_RMSE
    )

    r2_mod = per_module_df.dropna(subset=["test_R2"]).set_index("code_module")["test_R2"].sort_values(ascending=False)
    rmse_mod = per_module_df.set_index("code_module")["test_RMSE"].sort_values(ascending=True)

    save_barplot(
        values=r2_mod,
        title=f"Best model ({best_name}) — TEST R² by module",
        ylabel="R²",
        out_base=BY_MODULE_DIR / "best_model_test_R2_by_module",
        rotate_xticks=35,
        ylim=(-1.0, 1.0),
        bar_color=COLOR_R2
    )
    save_barplot(
        values=rmse_mod,
        title=f"Best model ({best_name}) — TEST RMSE by module",
        ylabel="RMSE (days)",
        out_base=BY_MODULE_DIR / "best_model_test_RMSE_by_module",
        rotate_xticks=35,
        bar_color=COLOR_RMSE
    )

    # REAL map of R² by region
    if len(per_region_df) > 0:
        save_real_map_choropleth_r2(
            per_region_df=per_region_df,
            out_base=MAPS_DIR / "best_model_test_R2_by_region_REALMAP",
            title=f"OULAD — Best model ({best_name}): TEST R² by region"
        )

    # -------------------------
    # Feature importance: GLOBAL
    # -------------------------
    ensure_dir(FEATURES_DIR / "global")
    imp_encoded, imp_agg = permutation_importance_aggregated(best_pipe, X_test, y_test)
    imp_encoded.to_csv(FEATURES_DIR / "global" / "perm_importance_encoded.csv", header=["importance_mean"])
    imp_agg.to_csv(FEATURES_DIR / "global" / "perm_importance_aggregated.csv", header=["importance_sum_encoded_columns"])
    save_feature_ranking_plots(
        imp_agg=imp_agg,
        out_base=FEATURES_DIR / "global" / f"perm_importance_top{TOPK_FEATURES_GLOBAL}",
        title=f"Best model ({best_name}) — permutation importance on TEST (aggregated)",
        topk=TOPK_FEATURES_GLOBAL
    )

    # -------------------------
    # Feature importance: BY REGION
    # -------------------------
    ensure_dir(FEATURES_DIR / "by_region")
    region_best_rows = []
    region_long_rows = []

    for reg, idx in X_test.groupby("region").groups.items():
        idx = list(idx)
        if len(idx) < MIN_TEST_SAMPLES_PER_REGION:
            continue

        Xr = X_test.loc[idx]
        pos = [X_test.index.get_loc(i) for i in idx]
        yr = y_test[pos]

        _, imp_agg_r = permutation_importance_aggregated(best_pipe, Xr, yr)

        out_reg_dir = FEATURES_DIR / "by_region" / str(reg).replace("/", "_")
        ensure_dir(out_reg_dir)
        imp_agg_r.to_csv(out_reg_dir / "perm_importance_aggregated.csv", header=["importance_sum_encoded_columns"])

        top = imp_agg_r.head(TOPK_FEATURES_GROUP)
        for feat, val0 in top.items():
            region_long_rows.append({"region": str(reg), "feature": str(feat), "importance_sum": float(val0)})

        region_best_rows.append({
            "region": str(reg),
            "n_test": len(idx),
            "best_feature": str(imp_agg_r.index[0]) if len(imp_agg_r) else "NA",
            "importance_sum": float(imp_agg_r.iloc[0]) if len(imp_agg_r) else np.nan
        })

    pd.DataFrame(region_best_rows).to_csv(FEATURES_DIR / "best_feature_by_region.csv", index=False)
    pd.DataFrame(region_long_rows).to_csv(FEATURES_DIR / "feature_ranking_topk_by_region_long.csv", index=False)

    # -------------------------
    # Feature importance: BY MODULE
    # -------------------------
    ensure_dir(FEATURES_DIR / "by_module")
    mod_best_rows = []
    mod_long_rows = []

    for mod, idx in X_test.groupby("code_module").groups.items():
        idx = list(idx)
        if len(idx) < MIN_TEST_SAMPLES_PER_MODULE:
            continue

        Xm = X_test.loc[idx]
        pos = [X_test.index.get_loc(i) for i in idx]
        ym = y_test[pos]

        _, imp_agg_m = permutation_importance_aggregated(best_pipe, Xm, ym)

        out_mod_dir = FEATURES_DIR / "by_module" / str(mod).replace("/", "_")
        ensure_dir(out_mod_dir)
        imp_agg_m.to_csv(out_mod_dir / "perm_importance_aggregated.csv", header=["importance_sum_encoded_columns"])

        top = imp_agg_m.head(TOPK_FEATURES_GROUP)
        for feat, val0 in top.items():
            mod_long_rows.append({"code_module": str(mod), "feature": str(feat), "importance_sum": float(val0)})

        mod_best_rows.append({
            "code_module": str(mod),
            "n_test": len(idx),
            "best_feature": str(imp_agg_m.index[0]) if len(imp_agg_m) else "NA",
            "importance_sum": float(imp_agg_m.iloc[0]) if len(imp_agg_m) else np.nan
        })

    pd.DataFrame(mod_best_rows).to_csv(FEATURES_DIR / "best_feature_by_module.csv", index=False)
    pd.DataFrame(mod_long_rows).to_csv(FEATURES_DIR / "feature_ranking_topk_by_module_long.csv", index=False)

    # -------------------------
    # Final logs
    # -------------------------
    print("\n[INFO] Done.")
    print("[INFO] Global repeated-split means:\n", means_df)
    print("[INFO] Global repeated-split SDs:\n", sds_df)
    print("[INFO] Best model:", best_name)
    print("[INFO] Best model single-split TEST:", test_metrics)
    print("[INFO] Outputs saved to:", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()



# %% Cell 1

