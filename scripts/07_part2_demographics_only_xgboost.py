# Auto-extracted from notebooks/07_part2_demographics_only_xgboost.ipynb

# Review/reproducibility helper script. Original notebook is retained in notebooks/.



# %% Cell 0
# run_country_models_vaw_part2.py
# ------------------------------------------------------------
# PART II — Demographics-only predictors of violence acceptance
# Model: XGBoost (regression)
# Per-country Train / Val / Test + light tuning
#
# Exports:
#   - results_paper_ready/part2_demographics_only_xgb/clean_modeling_table.csv
#   - results_paper_ready/part2_demographics_only_xgb/dropped_rows.csv
#   - results_paper_ready/part2_demographics_only_xgb/cleaning_audit_summary.csv
#   - results_paper_ready/part2_demographics_only_xgb/per_country_train_val_test_summary.csv
#   - results_paper_ready/part2_demographics_only_xgb/xgboost/feature_rankings/<Country>_feature_importance.csv
#   - results_paper_ready/part2_demographics_only_xgb/xgboost/best_params/<Country>_best_params.csv
#   - results_paper_ready/part2_demographics_only_xgb/xgboost/maps/r2_map_xgboost_test.(png/pdf)
#   - results_paper_ready/part2_demographics_only_xgb/paper_plots/heatmap_country_x_dimension.(png/pdf)
#   - results_paper_ready/part2_demographics_only_xgb/paper_plots/heatmap_country_x_top_demoresp.(png/pdf)
# ------------------------------------------------------------

import re
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

import geopandas as gpd

try:
    import pycountry
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# =========================
# USER SETTINGS
# =========================
INPUT_CSV = "Violence Against Women  Girls Data.csv"

OUT_ROOT = Path("results_paper_ready/part2_demographics_only_xgb")
XGB_DIR  = OUT_ROOT / "xgboost"
PLOT_DIR = OUT_ROOT / "paper_plots"

MIN_ROWS_PER_COUNTRY = 40
RANDOM_STATE = 42

# Parallelization across countries
N_JOBS_COUNTRIES = -1

# RandomizedSearch budget
XGB_N_ITER = 12
TUNING_CV_FOLDS = 3

# Natural Earth Admin-0
NE_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
NE_ZIP = OUT_ROOT / "ne_110m_admin_0_countries.zip"
NE_SHP_IN_ZIP = "ne_110m_admin_0_countries.shp"

DPI = 600
HEATMAP_CMAP = "magma"

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
    name = str(name).strip()

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
        "Syrian Arab Republic": "Syria",
        "Türkiye": "Turkey",
        "United States of America": "United States",
        "USA": "United States",
        "United Kingdom": "United Kingdom",
        "Bolivia (Plurinational State of)": "Bolivia",
        "Venezuela (Bolivarian Republic of)": "Venezuela",
        "Sao Tome and Principe": "São Tomé and Príncipe",
        "Timor-Leste": "East Timor",
        "Iran": "Iran",
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
        legend_kwds={"label": value_col, "shrink": 0.6}
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def clean_raw_feature_name(f: str) -> str:
    f = str(f)
    return re.sub(r"^(cat__|num__)", "", f).strip()


# =========================
# 1) READ + CLEAN (audit-lite)
# =========================
def read_and_clean(input_csv: str, out_root: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(input_csv)

    expected = ["Country", "Gender", "Demographics Question", "Demographics Response",
                "Survey Year", "Value"]
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
    for c in ["Country", "Gender", "Demographics Question", "Demographics Response"]:
        df[c] = df[c].astype(str).str.strip()

    # Validity mask
    ok = (
        df["Country"].notna() & (df["Country"] != "") & (df["Country"].str.lower() != "nan") &
        df["Gender"].notna() & (df["Gender"] != "") & (df["Gender"].str.lower() != "nan") &
        df["Demographics Question"].notna() & (df["Demographics Question"] != "") & (df["Demographics Question"].str.lower() != "nan") &
        df["Demographics Response"].notna() & (df["Demographics Response"] != "") & (df["Demographics Response"].str.lower() != "nan") &
        df["SurveyYear"].notna() &
        df["Value"].notna()
    )

    df_clean = df[ok].copy()

    out_root.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_root / "clean_modeling_table.csv", index=False)
    df[~ok].to_csv(out_root / "dropped_rows.csv", index=False)

    audit = pd.DataFrame({
        "n_raw_rows": [len(df_raw)],
        "n_clean_rows": [len(df_clean)],
        "n_dropped": [int((~ok).sum())],
    })
    audit.to_csv(out_root / "cleaning_audit_summary.csv", index=False)

    print("[INFO] Cleaning summary:")
    print(audit.to_string(index=False))
    print("[INFO] Saved cleaning outputs to:", out_root.resolve())
    return df_clean


# =========================
# 2) PREPROCESSING (NO QUESTION)
# =========================
def build_preprocess():
    categorical = ["Gender", "Demographics Question", "Demographics Response"]
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
# 3) MODEL FIT/TUNE/EVAL
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
        n_jobs=1,           # avoid nested parallelism
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
                    min_rows: int, seed: int, xgb_dir: Path) -> dict:
    X_cols = ["Gender", "Demographics Question", "Demographics Response", "SurveyYear"]
    y_col = "Value"

    n = len(d)
    summary = {"Country": country, "n_rows": n, "note": ""}

    if n < min_rows:
        summary["note"] = f"Skipped (<{min_rows} rows)"
        return summary

    X = d[X_cols]
    y = d[y_col].to_numpy()

    # Train/Val/Test: 70/15/15
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed
    )

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=1,
        tree_method="hist"
    )
    pipe = Pipeline([("prep", preprocess), ("model", xgb)])

    xgb_space = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
        "model__reg_lambda": [1.0, 5.0, 10.0],
    }

    res = tune_model(
        pipe, xgb_space,
        X_train, y_train, X_val, y_val, X_test, y_test,
        n_iter=XGB_N_ITER, seed=seed
    )

    safe_c = safe_filename(country)

    save_importance(
        res["best_estimator"],
        xgb_dir / "feature_rankings" / f"{safe_c}_feature_importance.csv"
    )

    pd.DataFrame([res["best_params"]]).to_csv(
        xgb_dir / "best_params" / f"{safe_c}_best_params.csv",
        index=False
    )

    summary.update({
        "XGB_val_R2": res["val_R2"],
        "XGB_val_MAE": res["val_MAE"],
        "XGB_val_RMSE": res["val_RMSE"],
        "XGB_test_R2": res["test_R2"],
        "XGB_test_MAE": res["test_MAE"],
        "XGB_test_RMSE": res["test_RMSE"],
    })
    return summary


# =========================
# 4) HEATMAPS (no seaborn)
# =========================
def plot_heatmap(matrix: pd.DataFrame, title: str, out_png: Path, out_pdf: Path,
                 xlabel: str, ylabel: str, vmax: Optional[float] = None):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    # Dynamic height: ~0.25 inch per row, minimum 6 inches
    fig_h = max(6, 0.25 * matrix.shape[0])
    fig_w = max(10, 0.35 * matrix.shape[1])  # also scale with columns a bit

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    im = ax.imshow(
        matrix.to_numpy(),
        cmap=HEATMAP_CMAP,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=vmax
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=45, ha="right")

    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.tolist())

    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("Importance share")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def build_heatmaps_from_feature_rankings(fi_dir: Path, countries_ok: list, out_dir: Path):
    """
    Uses saved feature_importance.csv files.
    Produces:
      (A) Country x Demographics Question (dimension-level) share
      (B) Country x Top Demographics Response categories share
    """
    rows_dim = []
    rows_resp = []

    for c in countries_ok:
        p = fi_dir / f"{safe_filename(c)}_feature_importance.csv"
        if not p.exists():
            continue

        df = pd.read_csv(p)
        if not {"feature", "importance"}.issubset(df.columns):
            continue

        df["feature"] = df["feature"].map(clean_raw_feature_name)
        df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
        df = df.dropna(subset=["importance"])

        s = df["importance"].sum()
        if not np.isfinite(s) or s <= 0:
            continue

        df["imp_norm"] = df["importance"] / s

        # Dimension-level: Demographics Question_*
        dqd = df[df["feature"].str.startswith("Demographics Question_", na=False)].copy()
        if not dqd.empty:
            dqd["dim"] = dqd["feature"].str.replace("Demographics Question_", "", regex=False)
            agg = dqd.groupby("dim", as_index=False)["imp_norm"].sum()
            agg["Country"] = c
            rows_dim.append(agg)

        # Response-level: Demographics Response_*
        dr = df[df["feature"].str.startswith("Demographics Response_", na=False)].copy()
        if not dr.empty:
            dr["resp"] = dr["feature"].str.replace("Demographics Response_", "", regex=False)
            agg = dr.groupby("resp", as_index=False)["imp_norm"].sum()
            agg["Country"] = c
            rows_resp.append(agg)

    # (A) Country x Dimension heatmap
    if rows_dim:
        d_dim = pd.concat(rows_dim, ignore_index=True)
        mat_dim = d_dim.pivot_table(
            index="Country", columns="dim", values="imp_norm", aggfunc="sum"
        ).fillna(0.0)

        # Normalize per country so row sums = 1 (shares across dimensions)
        mat_dim = mat_dim.div(mat_dim.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

        # Stable ordering
        mat_dim = mat_dim.loc[sorted(mat_dim.index.tolist())]
        mat_dim = mat_dim[sorted(mat_dim.columns.tolist())]

        plot_heatmap(
            matrix=mat_dim,
            title="Part II — Importance share by demographic dimension (XGBoost, demographics-only)",
            out_png=out_dir / "heatmap_country_x_dimension.png",
            out_pdf=out_dir / "heatmap_country_x_dimension.pdf",
            xlabel="Demographics Question (dimension)",
            ylabel="Country",
            vmax=float(mat_dim.to_numpy().max()) if mat_dim.size else None
        )
    else:
        print("[WARN] No Demographics Question_* features found in feature rankings (no dimension heatmap).")

    # (B) Country x Top Responses heatmap
    if rows_resp:
        d_resp = pd.concat(rows_resp, ignore_index=True)

        # Pick global top-K responses by mean across countries
        top_resp = (
            d_resp.groupby("resp")["imp_norm"].mean()
            .sort_values(ascending=False)
            .head(20)
            .index.tolist()
        )
        d_resp = d_resp[d_resp["resp"].isin(top_resp)].copy()

        mat_resp = d_resp.pivot_table(
            index="Country", columns="resp", values="imp_norm", aggfunc="sum"
        ).fillna(0.0)

        # Normalize per country so row sums = 1 (shares across selected top responses)
        mat_resp = mat_resp.div(mat_resp.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

        mat_resp = mat_resp.loc[sorted(mat_resp.index.tolist())]
        # Keep columns in top_resp order (more interpretable)
        mat_resp = mat_resp[top_resp]

        plot_heatmap(
            matrix=mat_resp,
            title="Part II — Importance share by top demographic responses (XGBoost, demographics-only)",
            out_png=out_dir / "heatmap_country_x_top_demoresp.png",
            out_pdf=out_dir / "heatmap_country_x_top_demoresp.pdf",
            xlabel="Top Demographics Response categories",
            ylabel="Country",
            vmax=float(mat_resp.to_numpy().max()) if mat_resp.size else None
        )
    else:
        print("[WARN] No Demographics Response_* features found in feature rankings (no response heatmap).")


# =========================
# MAIN
# =========================
def main():
    # Create folders
    (XGB_DIR / "feature_rankings").mkdir(parents=True, exist_ok=True)
    (XGB_DIR / "best_params").mkdir(parents=True, exist_ok=True)
    (XGB_DIR / "maps").mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Read + clean
    df_clean = read_and_clean(INPUT_CSV, OUT_ROOT)

    # Preprocess
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
    # MAP: Test R² (XGBoost)
    # =========================
    world = load_world_admin0(NE_ZIP)

    summary["Country_NE"] = summary["Country"].apply(normalize_country_name)

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

    ok = summary[summary["note"].fillna("") == ""].copy()

    if not ok.empty and "XGB_test_R2" in ok.columns:
        plot_r2_map(
            world=world,
            results_df=ok[["Country_NE", "XGB_test_R2"]].rename(columns={"XGB_test_R2": "Test_R2"}),
            value_col="Test_R2",
            title="Part II — Test R² per country (XGBoost, demographics-only)",
            out_png=XGB_DIR / "maps" / "r2_map_xgboost_test.png",
            out_pdf=XGB_DIR / "maps" / "r2_map_xgboost_test.pdf",
        )
        print("[INFO] Saved map to:", (XGB_DIR / "maps").resolve())
    else:
        print("[WARN] No valid countries for mapping (all skipped or missing XGB_test_R2).")

    # =========================
    # HEATMAPS from feature_rankings
    # =========================
    countries_ok = ok["Country"].tolist()
    build_heatmaps_from_feature_rankings(
        fi_dir=XGB_DIR / "feature_rankings",
        countries_ok=countries_ok,
        out_dir=PLOT_DIR
    )

    print("\n[INFO] Done.")
    print("[INFO] Outputs:")
    print(" -", (OUT_ROOT / "clean_modeling_table.csv").resolve())
    print(" -", (OUT_ROOT / "dropped_rows.csv").resolve())
    print(" -", (OUT_ROOT / "cleaning_audit_summary.csv").resolve())
    print(" -", (OUT_ROOT / "per_country_train_val_test_summary.csv").resolve())
    print(" -", (OUT_ROOT / "country_name_match_report.csv").resolve())
    print(" -", (OUT_ROOT / "unmatched_country_names.csv").resolve())
    print(" -", (XGB_DIR / "feature_rankings").resolve())
    print(" -", (XGB_DIR / "best_params").resolve())
    print(" -", (XGB_DIR / "maps").resolve())
    print(" -", (PLOT_DIR / "heatmap_country_x_dimension.png").resolve())
    print(" -", (PLOT_DIR / "heatmap_country_x_top_demoresp.png").resolve())


if __name__ == "__main__":
    main()



# %% Cell 1

