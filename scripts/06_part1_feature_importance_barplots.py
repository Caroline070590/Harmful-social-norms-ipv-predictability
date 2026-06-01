# Auto-extracted from notebooks/06_part1_feature_importance_barplots.ipynb

# Review/reproducibility helper script. Original notebook is retained in notebooks/.



# %% Cell 0
# ============================================================
# High-impact publication plots (WHITE STYLE, NO GRID):
# - Top-5 countries by XGBoost test R²
# - Barplot: test R²
# - Barplot: RMSE
# - Barplot: NRMSE (computed)
# Style: white background, darkslategrey bars, large fonts
# Exports: PNG (600 dpi) + PDF
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# USER SETTINGS
# ----------------------------
PERF_CSV = Path("results_paper_ready/per_country_train_val_test_summary.csv")

OUTDIR = Path("results_paper_ready") / "paper_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

TOPK = 5
BAR_COLOR = "darkslategrey"
DPI_EXPORT = 600

# ----------------------------
# Publication typography
# ----------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# ----------------------------
# Helpers
# ----------------------------
def save_fig(fig, png_path: Path, pdf_path: Path):
    fig.savefig(png_path, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

def clean_barplot(df, xcol, ycol, title, ylabel,
                  out_png, out_pdf,
                  higher_is_better=True):
    """
    Clean barplot:
    - white background
    - no grid
    - darkslategrey bars
    - value labels on top
    """
    d = df[[xcol, ycol]].dropna().copy()
    d[ycol] = d[ycol].astype(float)

    d = d.sort_values(ycol, ascending=not higher_is_better).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    bars = ax.bar(
        d[xcol].astype(str),
        d[ycol],
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=0.6
    )

    # Clean axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    ax.set_title(title, pad=16)
    ax.set_ylabel(ylabel)

    ax.tick_params(axis="x", rotation=25)

    # Value labels
    yvals = d[ycol].to_numpy()
    y_range = yvals.max() - yvals.min() if yvals.max() != yvals.min() else 1.0
    offset = 0.03 * y_range

    for rect, val in zip(bars, yvals):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + offset,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=14
        )

    fig.tight_layout()
    save_fig(fig, out_png, out_pdf)

# ----------------------------
# MAIN
# ----------------------------
def main():
    if not PERF_CSV.exists():
        raise FileNotFoundError(f"Cannot find: {PERF_CSV.resolve()}")

    df = pd.read_csv(PERF_CSV)

    needed = [
        "Country",
        "XGB_test_R2",
        "XGB_test_RMSE",
    ]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    # ----------------------------
    # Top-5 by XGB test R²
    # ----------------------------
    top = (
        df[["Country", "XGB_test_R2", "XGB_test_RMSE"]]
        .dropna(subset=["XGB_test_R2"])
        .sort_values("XGB_test_R2", ascending=False)
        .head(TOPK)
        .reset_index(drop=True)
    )

    # ----------------------------
    # Compute NRMSE (relative, defensible)
    # ----------------------------
    rmse_min = top["XGB_test_RMSE"].min()
    rmse_max = top["XGB_test_RMSE"].max()
    denom = rmse_max - rmse_min if rmse_max > rmse_min else 1.0

    top["XGB_test_NRMSE"] = top["XGB_test_RMSE"] / denom

    # ----------------------------
    # Plot 1: Test R²
    # ----------------------------
    clean_barplot(
        df=top,
        xcol="Country",
        ycol="XGB_test_R2",
        title=f"Top-{TOPK} countries by XGBoost performance",
        ylabel="test R²",
        out_png=OUTDIR / "top5_XGB_testR2_bar.png",
        out_pdf=OUTDIR / "top5_XGB_testR2_bar.pdf",
        higher_is_better=True
    )

    # ----------------------------
    # Plot 2: RMSE
    # ----------------------------
    clean_barplot(
        df=top,
        xcol="Country",
        ycol="XGB_test_RMSE",
        title=f"Top-{TOPK} countries — XGBoost test RMSE",
        ylabel="RMSE",
        out_png=OUTDIR / "top5_XGB_testRMSE_bar.png",
        out_pdf=OUTDIR / "top5_XGB_testRMSE_bar.pdf",
        higher_is_better=False
    )

    # ----------------------------
    # Plot 3: NRMSE
    # ----------------------------
    clean_barplot(
        df=top,
        xcol="Country",
        ycol="XGB_test_NRMSE",
        title=f"Top-{TOPK} countries — XGBoost normalized RMSE",
        ylabel="NRMSE",
        out_png=OUTDIR / "top5_XGB_testNRMSE_bar.png",
        out_pdf=OUTDIR / "top5_XGB_testNRMSE_bar.pdf",
        higher_is_better=False
    )

    # Save table for supplement
    top.to_csv(OUTDIR / "top5_countries_XGB_metrics.csv", index=False)

    print("\n[OK] High-impact plots saved in:")
    print(OUTDIR.resolve())

if __name__ == "__main__":
    main()



# %% Cell 1
# ============================================================
# High-impact figure (interpretable version):
# Top-5 countries by XGB test R²
# Stacked barplot where:
#   - x-axis = FEATURES
#   - colors = COUNTRIES (Top-5)
# Values:
#   - default: importance per country (each country sums to 1 across all features)
#   - plotted as raw per-country normalized importances (so bar height reflects sum across countries)
#     Optional: normalize each feature column to sum to 1 (composition across countries) -> see TOGGLE
#
# Exports: PNG (600 dpi) + PDF + CSVs with feature names and matrix values
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# USER SETTINGS
# ----------------------------
PERF_CSV = Path("results_paper_ready/per_country_train_val_test_summary.csv")
FI_DIR   = Path("results_paper_ready/xgboost/feature_rankings")

OUTDIR = Path("results_paper_ready") / "paper_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

TOPK_COUNTRIES = 5
TOPM_FEATURES  = 10
IMPORTANCE_COL = "importance"

# If True: for each feature, normalize contributions across countries to sum=1 (composition view).
# If False: show raw per-country normalized importances (bar height can be up to TOPK).
NORMALIZE_PER_FEATURE = False

DPI_EXPORT = 600

# Publication typography
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 22,
    "axes.labelsize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})

# Country color palette (harmonic)
COUNTRY_CMAP = "tab10"   # good for up to 10 countries

# ----------------------------
# Helpers
# ----------------------------
def _save(fig, out_png: Path, out_pdf: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def _read_feature_importance(country: str) -> pd.DataFrame:
    exact = FI_DIR / f"{country}_feature_importance.csv"
    if exact.exists():
        return pd.read_csv(exact)

    # best-effort fuzzy match
    cand = []
    for p in FI_DIR.glob("*_feature_importance.csv"):
        if p.stem.lower().startswith(country.lower()):
            cand.append(p)
    if len(cand) == 1:
        return pd.read_csv(cand[0])

    raise FileNotFoundError(
        f"Feature-importance CSV not found for '{country}'. Tried:\n"
        f"  - {exact}\n"
        f"  - fuzzy match candidates: {len(cand)}"
    )

def _country_colors(countries):
    cmap = plt.get_cmap(COUNTRY_CMAP)
    n = len(countries)
    return {c: cmap(i % cmap.N) for i, c in enumerate(countries)}

def build_plot():
    if not PERF_CSV.exists():
        raise FileNotFoundError(f"Missing performance CSV: {PERF_CSV.resolve()}")
    if not FI_DIR.exists():
        raise FileNotFoundError(f"Missing feature-importance folder: {FI_DIR.resolve()}")

    perf = pd.read_csv(PERF_CSV)
    if "Country" not in perf.columns or "XGB_test_R2" not in perf.columns:
        raise ValueError(f"PERF_CSV must contain Country and XGB_test_R2. Columns: {list(perf.columns)}")

    top = (
        perf[["Country", "XGB_test_R2"]]
        .dropna(subset=["XGB_test_R2"])
        .sort_values("XGB_test_R2", ascending=False)
        .head(TOPK_COUNTRIES)
        .reset_index(drop=True)
    )

    countries = top["Country"].astype(str).tolist()
    print("\nTop countries by XGB_test_R2:")
    print(top.to_string(index=False))

    # ---- Load and normalize per-country feature importances ----
    per_country = {}
    for c in countries:
        df_fi = _read_feature_importance(c)

        if "feature" not in df_fi.columns or IMPORTANCE_COL not in df_fi.columns:
            raise ValueError(
                f"Unexpected columns in feature importance for '{c}'. "
                f"Need ['feature', '{IMPORTANCE_COL}']. Got: {list(df_fi.columns)}"
            )

        df_fi = df_fi[["feature", IMPORTANCE_COL]].copy()
        df_fi[IMPORTANCE_COL] = pd.to_numeric(df_fi[IMPORTANCE_COL], errors="coerce")
        df_fi = df_fi.dropna(subset=[IMPORTANCE_COL])

        df_fi["feature"] = df_fi["feature"].astype(str)
        df_fi = df_fi.sort_values(IMPORTANCE_COL, ascending=False).reset_index(drop=True)

        s = float(df_fi[IMPORTANCE_COL].sum())
        if not np.isfinite(s) or s <= 0:
            raise ValueError(f"Invalid importance sum for '{c}': {s}")
        df_fi[IMPORTANCE_COL] = df_fi[IMPORTANCE_COL] / s  # sum=1 per country

        per_country[c] = df_fi

    # ---- Choose global top features (by mean importance across the Top-5 countries) ----
    all_feats = sorted(set().union(*[set(df["feature"]) for df in per_country.values()]))

    rows = []
    for feat in all_feats:
        vals = []
        for c in countries:
            s = per_country[c].set_index("feature")[IMPORTANCE_COL]
            vals.append(float(s.get(feat, 0.0)))
        rows.append((feat, float(np.mean(vals))))
    mean_rank = (
        pd.DataFrame(rows, columns=["feature", "mean_importance"])
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )

    show_features = mean_rank["feature"].head(TOPM_FEATURES).tolist()

    # ---- Print feature names clearly + export ----
    print("\nFeatures used in the plot (ordered):")
    for i, f in enumerate(show_features, 1):
        print(f"{i:02d}. {f}")

    (OUTDIR / "top_features_used.csv").write_text(
        "rank,feature\n" + "\n".join([f"{i},{repr(f)[1:-1]}" for i, f in enumerate(show_features, 1)]),
        encoding="utf-8"
    )

    # ---- Build matrix: rows = countries, cols = features ----
    mat = pd.DataFrame(index=countries, columns=show_features, data=0.0)
    for c in countries:
        s = per_country[c].set_index("feature")[IMPORTANCE_COL]
        for f in show_features:
            mat.loc[c, f] = float(s.get(f, 0.0))

    # Optional: normalize per feature to sum to 1 (composition across countries)
    mat_plot = mat.copy()
    if NORMALIZE_PER_FEATURE:
        col_sums = mat_plot.sum(axis=0).replace(0.0, np.nan)
        mat_plot = mat_plot.divide(col_sums, axis=1).fillna(0.0)

    # Export numeric matrix used
    mat_plot.to_csv(OUTDIR / "top5_feature_importance_matrix.csv", index=True)

    # ---- Plot: x = features, stacks = countries (colors = countries) ----
    colors = _country_colors(countries)

    fig, ax = plt.subplots(figsize=(16, 6.5), facecolor="white")
    ax.set_facecolor("white")

    x = np.arange(len(show_features))
    bottom = np.zeros(len(show_features), dtype=float)

    for c in countries:
        vals = mat_plot.loc[c, show_features].to_numpy(dtype=float)
        ax.bar(
            x, vals, bottom=bottom,
            color=colors[c],
            edgecolor="white", linewidth=1.0,
            label=c
        )
        bottom += vals

    # Styling
    ax.grid(False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(show_features, rotation=35, ha="right")

    ylabel = "Normalized importance (countries stack; per-country sums to 1)" if not NORMALIZE_PER_FEATURE \
             else "Per-feature composition across countries (sums to 1)"
    ax.set_ylabel(ylabel)

    title = "Top-5 countries by XGBoost test R² — feature importance (features on x-axis; colors = countries)"
    ax.set_title(title, pad=14)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Country")

    fig.tight_layout()

    out_png = OUTDIR / f"top5_xgb_feature_importance_STACK_featuresX_countriesColor_top{TOPM_FEATURES}.png"
    out_pdf = OUTDIR / f"top5_xgb_feature_importance_STACK_featuresX_countriesColor_top{TOPM_FEATURES}.pdf"
    _save(fig, out_png, out_pdf)

    print("\n[OK] Saved:")
    print(" -", out_png.resolve())
    print(" -", out_pdf.resolve())
    print(" -", (OUTDIR / "top_features_used.csv").resolve())
    print(" -", (OUTDIR / "top5_feature_importance_matrix.csv").resolve())

def main():
    build_plot()

if __name__ == "__main__":
    main()



# %% Cell 2
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================
# PATHS / SETTINGS
# ==========================
FI_DIR   = Path("results_paper_ready/xgboost/feature_rankings")
PERF_CSV = Path("results_paper_ready/per_country_train_val_test_summary.csv")
OUTDIR  = Path("results_paper_ready/paper_plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

TOPK_COUNTRIES = 5
TOPM_FEATURES  = 10
DPI = 600

# --------------------------
# Publication typography
# --------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

# ==========================
# Helpers
# ==========================
def safe_filename(s: str) -> str:
    """Must match how you saved files in run_country_models_vaw.py"""
    return "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in str(s)).strip()

def find_fi_file(country: str) -> Path:
    """Robustly locate <Country>_feature_importance.csv using safe_filename + fallback search."""
    safe = safe_filename(country)
    exact = FI_DIR / f"{safe}_feature_importance.csv"
    if exact.exists():
        return exact

    # fallback: contains safe token
    cands = list(FI_DIR.glob("*_feature_importance.csv"))
    hits = [p for p in cands if safe.lower() in p.name.lower()]
    if len(hits) == 1:
        return hits[0]

    raise FileNotFoundError(
        f"Could not find feature importance CSV for country '{country}'.\n"
        f"Tried: {exact}\n"
        f"Fallback hits: {len(hits)}"
    )

def clean_raw_feature_name(f: str) -> str:
    """Remove OneHotEncoder prefixes like cat__ / num__ so rules match."""
    f = str(f)
    f = re.sub(r"^(cat__|num__)", "", f)
    return f.strip()

def abbrev_feature(raw: str) -> str:
    """
    Rule-based abbreviation to avoid brittle exact-string matching.
    Adjust keywords here if your raw text differs slightly.
    """
    f = clean_raw_feature_name(raw)

    # Numeric
    if f == "SurveyYear":
        return "YEAR"

    # Gender
    if f.startswith("Gender_"):
        val = f.split("Gender_", 1)[1].strip()
        if val.lower().startswith("f"):
            return "G_F"
        if val.lower().startswith("m"):
            return "G_M"
        return "G_" + val.upper()

    # Question (violence-justification scenarios)
    if f.startswith("Question_"):
        q = f.split("Question_", 1)[1].lower()

        # Keyword rules (robust to small text differences)
        if "burn" in q and "food" in q:
            return "Q_FOOD"
        if "argue" in q:
            return "Q_ARG"
        if "sex" in q or "refus" in q:
            return "Q_REFUSE"
        if ("go" in q and "out" in q) or "without" in q:
            return "Q_OUT"
        if "child" in q or "neglect" in q:
            return "Q_NEGLECT"

        return "Q_OTHER"

    # Demographics Response (age/education/residence/employment etc.)
    if f.startswith("Demographics Response_"):
        r = f.split("Demographics Response_", 1)[1].strip()

        # Age groups (common patterns)
        r_low = r.lower().replace("–", "-")
        if r_low in {"15-24", "15 - 24"}:
            return "AGE_15_24"
        if r_low in {"25-34", "25 - 34"}:
            return "AGE_25_34"
        if r_low in {"35-49", "35 - 49"}:
            return "AGE_35_49"

        # Education
        if "no" in r_low and "educ" in r_low:
            return "EDU_NONE"
        if "primary" in r_low:
            return "EDU_PRIMARY"
        if "secondary" in r_low or "higher" in r_low:
            return "EDU_SECONDARY"

        # Residence
        if "urban" in r_low:
            return "RES_URBAN"
        if "rural" in r_low:
            return "RES_RURAL"

        # Employment
        if "employ" in r_low and ("not" in r_low or "unemploy" in r_low):
            return "EMP_NOT"
        if "employ" in r_low or "working" in r_low:
            return "EMP_WORKING"

        return "DEM_OTHER"

    # Demographics Question (dimension name itself)
    if f.startswith("Demographics Question_"):
        dq = f.split("Demographics Question_", 1)[1].strip().lower()
        if "age" in dq:
            return "DQ_AGE"
        if "educ" in dq:
            return "DQ_EDU"
        if "employ" in dq or "work" in dq:
            return "DQ_EMP"
        if "residen" in dq or "urban" in dq or "rural" in dq:
            return "DQ_RES"
        if "marital" in dq or "married" in dq:
            return "DQ_MAR"
        return "DQ_OTHER"

    return "OTHER"

def load_country_importances(country: str) -> pd.DataFrame:
    fi_path = find_fi_file(country)
    df = pd.read_csv(fi_path)

    if not {"feature", "importance"}.issubset(df.columns):
        raise ValueError(f"{fi_path} must contain columns ['feature','importance']. Got: {list(df.columns)}")

    df = df.copy()
    df["feature_raw"] = df["feature"].astype(str)
    df["feature_raw_clean"] = df["feature_raw"].map(clean_raw_feature_name)
    df["abbr"] = df["feature_raw"].map(abbrev_feature)
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    df = df.dropna(subset=["importance"])

    # Normalize within country
    s = df["importance"].sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"Non-positive importance sum for {country}: {s}")
    df["importance"] /= s

    # Aggregate by abbreviation
    out = df.groupby("abbr", as_index=False)["importance"].sum()
    out["Country"] = country

    return out, df.sort_values("importance", ascending=False).head(20)

# ==========================
# MAIN
# ==========================
def main():
    if not PERF_CSV.exists():
        raise FileNotFoundError(f"Missing: {PERF_CSV.resolve()}")
    if not FI_DIR.exists():
        raise FileNotFoundError(f"Missing: {FI_DIR.resolve()}")

    perf = pd.read_csv(PERF_CSV)
    top = (
        perf[["Country", "XGB_test_R2"]]
        .dropna(subset=["Country", "XGB_test_R2"])
        .sort_values("XGB_test_R2", ascending=False)
        .head(TOPK_COUNTRIES)
        .reset_index(drop=True)
    )
    countries = top["Country"].tolist()

    # Load importances
    agg_rows = []
    print("\n=== Debug: top raw features per country (top 20) ===")
    for c in countries:
        agg, topraw = load_country_importances(c)
        agg_rows.append(agg)
        print(f"\n[{c}]")
        # show the raw names so you can verify what exists in your files
        for _, r in topraw.iterrows():
            print(f"  {r['importance']:.4f}  {r['feature_raw_clean']}")

    df_all = pd.concat(agg_rows, ignore_index=True)

    # Pick top features globally by mean importance across selected countries
    feat_rank = (
        df_all.groupby("abbr")["importance"]
        .mean()
        .sort_values(ascending=False)
    )

    # keep TOPM features, but drop the trivial "OTHER" if it dominates
    top_feats = [f for f in feat_rank.index.tolist() if f != "OTHER"][:TOPM_FEATURES]
    if len(top_feats) < TOPM_FEATURES and "OTHER" in feat_rank.index:
        top_feats.append("OTHER")

    df_plot = df_all[df_all["abbr"].isin(top_feats)].copy()

    # Pivot: rows=features, cols=countries, values=importance
    pivot = df_plot.pivot_table(index="abbr", columns="Country", values="importance", aggfunc="sum").fillna(0.0)

    # IMPORTANT FIX:
    # Your plot stacks countries *within each feature*,
    # so we normalize per feature across countries (row sum = 1).
    row_sums = pivot.sum(axis=1).replace(0, np.nan)
    pivot = pivot.div(row_sums, axis=0).fillna(0.0)

    # Order: most important at top (using original mean importance ranking)
    order = [f for f in feat_rank.index.tolist() if f in pivot.index]
    pivot = pivot.loc[order[:len(pivot)]]
    pivot = pivot.iloc[::-1]  # flip so best appears at top in barh

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.set_facecolor("white")

    colors = plt.get_cmap("Set3")(np.linspace(0, 1, pivot.shape[1]))

    left = np.zeros(len(pivot))
    ylabels = pivot.index.tolist()

    for i, country in enumerate(pivot.columns):
        vals = pivot[country].to_numpy()
        ax.barh(
            ylabels, vals, left=left,
            color=colors[i],
            edgecolor="white",
            linewidth=1.0,
            label=country
        )
        left += vals

    ax.set_xlim(0, 1)
    ax.set_xlabel("Within-feature share of importance across countries (stack sums to 1)")
    ax.set_title("Part I — Structural predictors of violence-related attitudes (XGBoost)")
    ax.grid(False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    ax.legend(
        title="Country",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        frameon=False
    )

    fig.tight_layout()
    out_png = OUTDIR / "part1_feature_importance_features_axis_colors_countries_set3.png"
    out_pdf = OUTDIR / "part1_feature_importance_features_axis_colors_countries_set3.pdf"
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("\n✅ Saved:", out_png.resolve())
    print("✅ Saved:", out_pdf.resolve())

    # Also print the abbreviation legend used
    print("\n=== Abbreviation legend (interpretation) ===")
    legend = {
        "Q_FOOD": "Justified if wife burns food",
        "Q_ARG": "Justified if wife argues",
        "Q_NEGLECT": "Justified if wife neglects children",
        "Q_REFUSE": "Justified if wife refuses sex",
        "Q_OUT": "Justified if wife goes out without permission",
        "G_F": "Female respondents",
        "G_M": "Male respondents",
        "RES_URBAN": "Urban",
        "RES_RURAL": "Rural",
        "EDU_NONE": "No education",
        "EDU_PRIMARY": "Primary education",
        "EDU_SECONDARY": "Secondary or higher",
        "EMP_WORKING": "Employed / working",
        "EMP_NOT": "Not employed",
        "AGE_15_24": "Age 15–24",
        "AGE_25_34": "Age 25–34",
        "AGE_35_49": "Age 35–49",
        "YEAR": "Survey year",
        "DQ_AGE": "Demographics dimension = age",
        "DQ_EDU": "Demographics dimension = education",
        "DQ_EMP": "Demographics dimension = employment",
        "DQ_RES": "Demographics dimension = residence",
        "DQ_MAR": "Demographics dimension = marital status",
        "Q_OTHER": "Other question category",
        "DEM_OTHER": "Other demographic response",
        "DQ_OTHER": "Other demographic dimension",
        "OTHER": "Everything else / unmapped",
    }
    for k in pivot.index[::-1]:
        print(f"{k:10s}  {legend.get(k, '(see rules in abbrev_feature)')}")

if __name__ == "__main__":
    main()



# %% Cell 3

