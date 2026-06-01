# Auto-extracted from notebooks/08_part2_barplots.ipynb

# Review/reproducibility helper script. Original notebook is retained in notebooks/.



# %% Cell 0
import pandas as pd

perf = pd.read_csv("results_paper_ready/part2_demographics_only_xgb/per_country_train_val_test_summary.csv")
perf_ok = perf[perf["note"].fillna("") == ""].copy()

top = perf_ok.sort_values("XGB_test_R2", ascending=False).head(10)
print(top[["Country","n_rows","XGB_test_R2","XGB_test_MAE","XGB_test_RMSE"]].to_string(index=False))

best = top.iloc[0]
print("\nBEST COUNTRY:", best["Country"], "Test R2 =", best["XGB_test_R2"])



# %% Cell 1
# ============================================================
# PART II — High-impact publication plots (WHITE STYLE, NO GRID):
# - Top-5 countries by XGBoost test R²
# - Barplot: test R²
# - Barplot: RMSE
# - Barplot: NRMSE (per-country y-range)
#
# Inputs (Part II outputs):
#   - per_country_train_val_test_summary.csv
#   - clean_modeling_table.csv  (used to compute y-range per country for NRMSE)
#
# Exports: PNG (600 dpi) + PDF + CSV table
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# USER SETTINGS (PART II)
# ----------------------------
PART2_ROOT = Path("results_paper_ready/part2_demographics_only_xgb")

PERF_CSV  = PART2_ROOT / "per_country_train_val_test_summary.csv"
CLEAN_CSV = PART2_ROOT / "clean_modeling_table.csv"

OUTDIR = PART2_ROOT / "paper_plots"
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
    png_path.parent.mkdir(parents=True, exist_ok=True)
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
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.dropna(subset=[ycol])

    d = d.sort_values(ycol, ascending=not higher_is_better).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    bars = ax.bar(
        d[xcol].astype(str),
        d[ycol].astype(float),
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
    yvals = d[ycol].to_numpy(dtype=float)
    y_min, y_max = float(np.min(yvals)), float(np.max(yvals))
    y_range = (y_max - y_min) if y_max != y_min else (abs(y_max) if y_max != 0 else 1.0)
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

def compute_country_y_range(clean_csv: Path) -> pd.DataFrame:
    """
    Returns a dataframe with:
      Country, y_min, y_max, y_range
    based on clean_modeling_table.csv (Part II).
    """
    df = pd.read_csv(clean_csv)
    if "Country" not in df.columns or "Value" not in df.columns:
        raise ValueError(f"{clean_csv} must contain columns ['Country','Value']. Got: {list(df.columns)}")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Country", "Value"]).copy()

    g = df.groupby("Country")["Value"]
    out = g.agg(y_min="min", y_max="max").reset_index()
    out["y_range"] = out["y_max"] - out["y_min"]

    # If a country has zero range (all values identical), keep range=NaN to avoid misleading NRMSE
    out.loc[out["y_range"] <= 0, "y_range"] = np.nan
    return out

# ----------------------------
# MAIN
# ----------------------------
def main():
    if not PERF_CSV.exists():
        raise FileNotFoundError(f"Cannot find: {PERF_CSV.resolve()}")
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find: {CLEAN_CSV.resolve()}\n"
            f"Needed to compute per-country NRMSE = RMSE / (max(y)-min(y))."
        )

    perf = pd.read_csv(PERF_CSV)

    needed = ["Country", "XGB_test_R2", "XGB_test_RMSE"]
    miss = [c for c in needed if c not in perf.columns]
    if miss:
        raise ValueError(f"Missing columns in PERF_CSV: {miss}. Columns: {list(perf.columns)}")

    # Remove skipped countries if note exists
    if "note" in perf.columns:
        perf_ok = perf[perf["note"].fillna("") == ""].copy()
    else:
        perf_ok = perf.copy()

    # Top-K by XGB test R²
    top = (
        perf_ok[["Country", "XGB_test_R2", "XGB_test_RMSE"]]
        .dropna(subset=["XGB_test_R2"])
        .sort_values("XGB_test_R2", ascending=False)
        .head(TOPK)
        .reset_index(drop=True)
    )

    # Compute NRMSE using per-country target range from clean_modeling_table.csv
    ranges = compute_country_y_range(CLEAN_CSV)
    top = top.merge(ranges[["Country", "y_range"]], on="Country", how="left")

    top["XGB_test_NRMSE"] = top["XGB_test_RMSE"] / top["y_range"]

    # If y_range is missing/zero, NRMSE is undefined -> keep NaN (so plot will drop those rows)
    # This is better than fabricating a denominator.
    # You still get R² and RMSE plots for those countries.

    # Plot 1: Test R²
    clean_barplot(
        df=top,
        xcol="Country",
        ycol="XGB_test_R2",
        title=f"Part II (demographics-only) — Top-{TOPK} countries by XGBoost test performance",
        ylabel="test R²",
        out_png=OUTDIR / f"part2_top{TOPK}_XGB_testR2_bar.png",
        out_pdf=OUTDIR / f"part2_top{TOPK}_XGB_testR2_bar.pdf",
        higher_is_better=True
    )

    # Plot 2: RMSE
    clean_barplot(
        df=top,
        xcol="Country",
        ycol="XGB_test_RMSE",
        title=f"Part II — Top-{TOPK} countries: XGBoost test RMSE",
        ylabel="RMSE",
        out_png=OUTDIR / f"part2_top{TOPK}_XGB_testRMSE_bar.png",
        out_pdf=OUTDIR / f"part2_top{TOPK}_XGB_testRMSE_bar.pdf",
        higher_is_better=False
    )

    # Plot 3: NRMSE (may drop countries with undefined y_range)
    clean_barplot(
        df=top,
        xcol="Country",
        ycol="XGB_test_NRMSE",
        title=f"Part II — Top-{TOPK} countries: XGBoost normalized RMSE (RMSE / value-range)",
        ylabel="NRMSE",
        out_png=OUTDIR / f"part2_top{TOPK}_XGB_testNRMSE_bar.png",
        out_pdf=OUTDIR / f"part2_top{TOPK}_XGB_testNRMSE_bar.pdf",
        higher_is_better=False
    )

    # Save table for supplement
    top.to_csv(OUTDIR / f"part2_top{TOPK}_countries_XGB_metrics.csv", index=False)

    print("\n[OK] Part II high-impact plots saved in:")
    print(OUTDIR.resolve())
    print("\nSaved table:")
    print((OUTDIR / f"part2_top{TOPK}_countries_XGB_metrics.csv").resolve())

if __name__ == "__main__":
    main()



# %% Cell 2

