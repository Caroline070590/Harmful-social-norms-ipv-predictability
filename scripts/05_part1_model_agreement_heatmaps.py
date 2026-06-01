# Auto-extracted from notebooks/05_part1_model_agreement_heatmaps.ipynb

# Review/reproducibility helper script. Original notebook is retained in notebooks/.



# %% Cell 0
import zipfile
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# =========================
# USER SETTINGS
# =========================
ZIP_PATH = "results_paper_ready.zip"   # your uploaded zip
OUT_DIR  = Path("results_agreement")  # where outputs will be written
TOPN     = 20                         # compare Top-N features

# Model folders expected inside the zip:
# random_forest/feature_rankings/*.csv
# decision_tree/feature_rankings/*.csv
# xgboost/feature_rankings/*.csv


# =========================
# HELPERS
# =========================
def country_from_filename(p: Path) -> str:
    # e.g., Afghanistan_feature_importance.csv -> Afghanistan
    name = p.stem
    return name.replace("_feature_importance", "").strip()

def read_rank_file(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if "feature" not in df.columns:
        raise ValueError(f"Missing 'feature' column in: {p.name} | columns={list(df.columns)}")

    if "importance" not in df.columns:
        # try common alternatives
        cand = next((c for c in df.columns if c.lower().startswith("imp")), None)
        if cand is None:
            raise ValueError(f"Missing 'importance' column in: {p.name} | columns={list(df.columns)}")
        df = df.rename(columns={cand: "importance"})

    df = df[["feature", "importance"]].copy()
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df

def rank_map(df: pd.DataFrame, topn: int) -> dict:
    d = df.head(topn).reset_index(drop=True)
    return {feat: i + 1 for i, feat in enumerate(d["feature"].tolist())}

def agreement_spearman(rankA: dict, rankB: dict, topn: int) -> float:
    """
    Compare ranks over the UNION of Top-N features.
    Missing features get rank (topn+1).
    """
    feats = sorted(set(rankA) | set(rankB))
    if len(feats) < 3:
        return np.nan
    a = [rankA.get(f, topn + 1) for f in feats]
    b = [rankB.get(f, topn + 1) for f in feats]
    rho, _ = spearmanr(a, b)
    return float(rho)

def agreement_jaccard(rankA: dict, rankB: dict) -> float:
    A, B = set(rankA), set(rankB)
    if not A and not B:
        return np.nan
    return float(len(A & B) / len(A | B))

def find_first(base: Path, pattern: str):
    hits = glob.glob(str(base / "**" / pattern), recursive=True)
    return Path(hits[0]) if hits else None


# =========================
# 1) UNZIP
# =========================
OUT_DIR.mkdir(parents=True, exist_ok=True)
extract_dir = OUT_DIR / "results_paper_ready_extracted"
if extract_dir.exists():
    import shutil
    shutil.rmtree(extract_dir)
extract_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(extract_dir)

# locate model ranking directories
rf_rank_dir  = find_first(extract_dir, "random_forest/feature_rankings")
dt_rank_dir  = find_first(extract_dir, "decision_tree/feature_rankings")
xgb_rank_dir = find_first(extract_dir, "xgboost/feature_rankings")

if not (rf_rank_dir and dt_rank_dir and xgb_rank_dir):
    raise ValueError(
        "Could not find one or more expected folders inside the zip:\n"
        "  random_forest/feature_rankings\n"
        "  decision_tree/feature_rankings\n"
        "  xgboost/feature_rankings\n"
        f"Found:\n  RF={rf_rank_dir}\n  DT={dt_rank_dir}\n  XGB={xgb_rank_dir}\n"
    )

rf_rank_dir, dt_rank_dir, xgb_rank_dir = Path(rf_rank_dir), Path(dt_rank_dir), Path(xgb_rank_dir)


# =========================
# 2) LOAD TOP-N RANK MAPS
# =========================
def load_model_rankmaps(rank_dir: Path, topn: int):
    out = {}
    files = sorted(rank_dir.glob("*.csv"))
    for p in files:
        c = country_from_filename(p)
        df = read_rank_file(p)
        out[c] = rank_map(df, topn)
    return out

rf_maps  = load_model_rankmaps(rf_rank_dir, TOPN)
dt_maps  = load_model_rankmaps(dt_rank_dir, TOPN)
xgb_maps = load_model_rankmaps(xgb_rank_dir, TOPN)

countries = sorted(set(rf_maps) & set(dt_maps) & set(xgb_maps))
if len(countries) == 0:
    raise ValueError("No common countries found across RF/DT/XGB feature_rankings.")


# =========================
# 3) AGREEMENT TABLE
# =========================
rows = []
for c in countries:
    rf = rf_maps[c]
    dt = dt_maps[c]
    xg = xgb_maps[c]

    rows.append({
        "Country": c,

        f"Spearman_RF_DT_top{TOPN}": agreement_spearman(rf, dt, TOPN),
        f"Spearman_RF_XGB_top{TOPN}": agreement_spearman(rf, xg, TOPN),
        f"Spearman_DT_XGB_top{TOPN}": agreement_spearman(dt, xg, TOPN),

        f"Jaccard_RF_DT_top{TOPN}": agreement_jaccard(rf, dt),
        f"Jaccard_RF_XGB_top{TOPN}": agreement_jaccard(rf, xg),
        f"Jaccard_DT_XGB_top{TOPN}": agreement_jaccard(dt, xg),
    })

agree_df = pd.DataFrame(rows).sort_values("Country")
agree_csv = OUT_DIR / f"feature_ranking_agreement_top{TOPN}.csv"
agree_df.to_csv(agree_csv, index=False)


# =========================
# 4) HEATMAP EXPORTS
# =========================
def export_heatmap(mat_df: pd.DataFrame, title: str, cbar_label: str, out_png: Path, out_pdf: Path):
    fig, ax = plt.subplots(figsize=(8, max(6, 0.22 * len(mat_df))), dpi=300)

    im = ax.imshow(mat_df.to_numpy(), aspect="auto", interpolation="nearest")

    ax.set_yticks(np.arange(len(mat_df.index)))
    ax.set_yticklabels(mat_df.index.tolist(), fontsize=6)

    ax.set_xticks(np.arange(mat_df.shape[1]))
    ax.set_xticklabels(["RF vs DT", "RF vs XGB", "DT vs XGB"], rotation=0)

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# Spearman heatmap
spearman_cols = [f"Spearman_RF_DT_top{TOPN}", f"Spearman_RF_XGB_top{TOPN}", f"Spearman_DT_XGB_top{TOPN}"]
spearman_mat = agree_df.set_index("Country")[spearman_cols]
export_heatmap(
    spearman_mat,
    title=f"Agreement of feature rankings (Spearman ρ on Top-{TOPN} features)",
    cbar_label="Spearman ρ",
    out_png=OUT_DIR / f"heatmap_feature_ranking_agreement_spearman_top{TOPN}.png",
    out_pdf=OUT_DIR / f"heatmap_feature_ranking_agreement_spearman_top{TOPN}.pdf",
)

# Jaccard heatmap
jacc_cols = [f"Jaccard_RF_DT_top{TOPN}", f"Jaccard_RF_XGB_top{TOPN}", f"Jaccard_DT_XGB_top{TOPN}"]
jacc_mat = agree_df.set_index("Country")[jacc_cols]
export_heatmap(
    jacc_mat,
    title=f"Overlap of Top-{TOPN} features (Jaccard index)",
    cbar_label="Jaccard",
    out_png=OUT_DIR / f"heatmap_feature_ranking_overlap_jaccard_top{TOPN}.png",
    out_pdf=OUT_DIR / f"heatmap_feature_ranking_overlap_jaccard_top{TOPN}.pdf",
)


# =========================
# 5) OPTIONAL: MERGE WITH YOUR TRAIN/VAL/TEST SUMMARY
# =========================
summary_path = find_first(extract_dir, "per_country_train_val_test_summary.csv")
if summary_path and summary_path.exists():
    summary = pd.read_csv(summary_path)
    merged = agree_df.merge(summary, on="Country", how="left")
    merged_path = OUT_DIR / f"agreement_plus_train_val_test_summary_top{TOPN}.csv"
    merged.to_csv(merged_path, index=False)


print("Done ✅")
print("Wrote:")
print(" -", agree_csv)
print(" -", OUT_DIR / f"heatmap_feature_ranking_agreement_spearman_top{TOPN}.png")
print(" -", OUT_DIR / f"heatmap_feature_ranking_agreement_spearman_top{TOPN}.pdf")
print(" -", OUT_DIR / f"heatmap_feature_ranking_overlap_jaccard_top{TOPN}.png")
print(" -", OUT_DIR / f"heatmap_feature_ranking_overlap_jaccard_top{TOPN}.pdf")
if summary_path and summary_path.exists():
    print(" -", OUT_DIR / f"agreement_plus_train_val_test_summary_top{TOPN}.csv")



# %% Cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

IN_CSV = "results_agreement/feature_ranking_agreement_top20.csv"  # adjust path
OUTDIR = Path("results_agreement/clear_plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

TOPN = 20
df = pd.read_csv(IN_CSV)

spearman_cols = [f"Spearman_RF_DT_top{TOPN}", f"Spearman_RF_XGB_top{TOPN}", f"Spearman_DT_XGB_top{TOPN}"]
jacc_cols     = [f"Jaccard_RF_DT_top{TOPN}",  f"Jaccard_RF_XGB_top{TOPN}",  f"Jaccard_DT_XGB_top{TOPN}"]

df["Mean_Spearman"] = df[spearman_cols].mean(axis=1)
df["Mean_Jaccard"]  = df[jacc_cols].mean(axis=1)

# Sort so patterns pop out
df_s = df.sort_values("Mean_Spearman", ascending=False).reset_index(drop=True)

def heatmap(mat, title, cbar_label, out_png, out_pdf, vmin, vmax):
    fig, ax = plt.subplots(figsize=(8, max(6, 0.22*len(mat))), dpi=300)
    im = ax.imshow(mat.to_numpy(), aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)

    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index.tolist(), fontsize=6)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["RF vs DT", "RF vs XGB", "DT vs XGB"])

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# --- Spearman heatmap (absolute scale -1..1) ---
spearman_mat = df_s.set_index("Country")[spearman_cols]
heatmap(
    spearman_mat,
    title=f"Feature-ranking agreement (Spearman ρ, Top-{TOPN}) — sorted by mean agreement",
    cbar_label="Spearman ρ",
    out_png=OUTDIR / f"spearman_heatmap_top{TOPN}_sorted.png",
    out_pdf=OUTDIR / f"spearman_heatmap_top{TOPN}_sorted.pdf",
    vmin=-1, vmax=1
)

# --- Jaccard heatmap (absolute scale 0..1) ---
jacc_mat = df_s.set_index("Country")[jacc_cols]
heatmap(
    jacc_mat,
    title=f"Top-{TOPN} feature overlap (Jaccard) — sorted by mean Spearman agreement",
    cbar_label="Jaccard",
    out_png=OUTDIR / f"jaccard_heatmap_top{TOPN}_sorted.png",
    out_pdf=OUTDIR / f"jaccard_heatmap_top{TOPN}_sorted.pdf",
    vmin=0, vmax=1
)

# Also export a tiny summary table to read quickly
summary = pd.DataFrame({
    "pair": ["RF vs DT", "RF vs XGB", "DT vs XGB"],
    "spearman_mean": [df[c].mean() for c in spearman_cols],
    "spearman_median": [df[c].median() for c in spearman_cols],
    "jaccard_mean": [df[c].mean() for c in jacc_cols],
})
summary.to_csv(OUTDIR / f"agreement_summary_top{TOPN}.csv", index=False)

print("Saved to:", OUTDIR.resolve())



# %% Cell 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

IN_CSV = "results_agreement/feature_ranking_agreement_top20.csv"  # adjust path
OUTDIR = Path("results_agreement/clear_plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

TOPN = 20
df = pd.read_csv(IN_CSV)

spearman_cols = [f"Spearman_RF_DT_top{TOPN}", f"Spearman_RF_XGB_top{TOPN}", f"Spearman_DT_XGB_top{TOPN}"]
jacc_cols     = [f"Jaccard_RF_DT_top{TOPN}",  f"Jaccard_RF_XGB_top{TOPN}",  f"Jaccard_DT_XGB_top{TOPN}"]

df["Mean_Spearman"] = df[spearman_cols].mean(axis=1)
df["Mean_Jaccard"]  = df[jacc_cols].mean(axis=1)

# Sort so patterns pop out
df_s = df.sort_values("Mean_Spearman", ascending=False).reset_index(drop=True)


# =========================================================
# ONLY MODIFIED PART: bigger, clearer heatmap
# =================================================
def heatmap(mat, title, cbar_label, out_png, out_pdf, vmin, vmax):

    nrows = len(mat)

    # Bigger, publication-ready size
    fig_w = 11.5
    fig_h = max(10, 0.30 * nrows)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300, facecolor="white")

    im = ax.imshow(
        mat.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(mat.index.tolist(), fontsize=7)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["RF vs DT", "RF vs XGB", "DT vs XGB"], fontsize=11)

    ax.set_title(title, fontsize=14)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# --- Spearman heatmap (absolute scale -1..1) ---
spearman_mat = df_s.set_index("Country")[spearman_cols]
heatmap(
    spearman_mat,
    title=f"Feature-ranking agreement (Spearman ρ, Top-{TOPN}) — sorted by mean agreement",
    cbar_label="Spearman ρ",
    out_png=OUTDIR / f"spearman_heatmap_top{TOPN}_sorted.png",
    out_pdf=OUTDIR / f"spearman_heatmap_top{TOPN}_sorted.pdf",
    vmin=-1, vmax=1
)

# --- Jaccard heatmap (absolute scale 0..1) ---
jacc_mat = df_s.set_index("Country")[jacc_cols]
heatmap(
    jacc_mat,
    title=f"Top-{TOPN} feature overlap (Jaccard) — sorted by mean Spearman agreement",
    cbar_label="Jaccard",
    out_png=OUTDIR / f"jaccard_heatmap_top{TOPN}_sorted.png",
    out_pdf=OUTDIR / f"jaccard_heatmap_top{TOPN}_sorted.pdf",
    vmin=0, vmax=1
)

# Summary table
summary = pd.DataFrame({
    "pair": ["RF vs DT", "RF vs XGB", "DT vs XGB"],
    "spearman_mean": [df[c].mean() for c in spearman_cols],
    "spearman_median": [df[c].median() for c in spearman_cols],
    "jaccard_mean": [df[c].mean() for c in jacc_cols],
})
summary.to_csv(OUTDIR / f"agreement_summary_top{TOPN}.csv", index=False)

print("Saved to:", OUTDIR.resolve())



# %% Cell 3
# ============================================================
# Paper-ready plots:
# 1) Performance overview heatmap (RF/DT/XGB test R²)
# 2) Circular (polar) top-20 countries per model (test R²)
# 3) Agreement heatmaps (Spearman + Jaccard) from agreement CSV
# Exports: PNG (600 dpi) + PDF
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# USER PATHS (EDIT IF NEEDED)
# =========================
# IMPORTANT: keep these as Path(...) OR they will be strings.
PERF_CSV  = r"results_paper_ready/per_country_train_val_test_summary.csv"
AGREE_CSV = r"results_agreement/feature_ranking_agreement_top20.csv"

OUTDIR = Path("results_agreement") / "paper_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

TOPN_COUNTRIES = 20
TOPN_FEATURES  = 20  # must match columns like ..._top20 in agreement CSV

DPI_SCREEN = 300
DPI_EXPORT = 600

# =========================
# FONT SIZES (bigger letters)
# =========================
FS_TITLE   = 18
FS_XTICK   = 14
FS_YTICK   = 10
FS_CBAR_L  = 14
FS_CBAR_T  = 12
FS_POLAR_LAB = 10
FS_POLAR_ANN = 12


# =========================
# HELPERS
# =========================
def save_fig(fig, png_path: Path, pdf_path: Path):
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def polar_barplot_top_countries(df_top: pd.DataFrame, value_col: str, title: str,
                               out_png: Path, out_pdf: Path):
    df_top = df_top.sort_values(value_col, ascending=False).reset_index(drop=True)

    labels = df_top["Country"].astype(str).tolist()
    vals = df_top[value_col].astype(float).to_numpy()

    vals_clip = np.clip(vals, 0.0, None)
    n = len(vals_clip)
    if n == 0:
        print(f"[WARN] No rows to plot for: {title}")
        return

    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    width = (2*np.pi / n) * 0.85

    vmax = max(float(np.max(vals_clip)), 1e-12)
    radii = vals_clip / vmax

    fig = plt.figure(figsize=(10, 10), dpi=DPI_SCREEN, facecolor="white")
    ax = plt.subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines["polar"].set_visible(False)

    ax.bar(angles, radii, width=width, bottom=0.0, linewidth=0.8, edgecolor="white")

    for ang, r, lab in zip(angles, radii, labels):
        rr = min(1.10, r + 0.06)
        rot = np.degrees(ang)
        if 90 < rot < 270:
            rot += 180
            ha = "right"
        else:
            ha = "left"
        ax.text(ang, rr, lab, rotation=rot, rotation_mode="anchor",
                ha=ha, va="center", fontsize=FS_POLAR_LAB, color="#333")

    ax.set_title(title, fontsize=FS_TITLE, pad=18)
    ax.text(np.deg2rad(90), 1.14, f"max {value_col} = {vmax:.3f}",
            ha="center", va="center", fontsize=FS_POLAR_ANN, color="#444")

    fig.tight_layout()
    save_fig(fig, out_png, out_pdf)


def performance_overview_plot(df_perf: pd.DataFrame, out_png: Path, out_pdf: Path):
    needed = ["Country", "RF_test_R2", "DT_test_R2", "XGB_test_R2"]
    miss = [c for c in needed if c not in df_perf.columns]
    if miss:
        raise ValueError(f"Performance CSV is missing columns: {miss}\nAvailable: {list(df_perf.columns)}")

    d = df_perf[needed].copy()
    d = d.dropna(subset=["RF_test_R2", "DT_test_R2", "XGB_test_R2"], how="all").copy()

    d["Mean_test_R2"] = d[["RF_test_R2", "DT_test_R2", "XGB_test_R2"]].mean(axis=1, skipna=True)
    d = d.sort_values("Mean_test_R2", ascending=False).reset_index(drop=True)

    mat = d.set_index("Country")[["RF_test_R2", "DT_test_R2", "XGB_test_R2"]]
    arr = mat.to_numpy().astype(float)

    vals = arr.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No finite numeric values found in test R² columns.")

    vmin = np.percentile(vals, 5)
    vmax = np.percentile(vals, 95)
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if vmin == vmax:
            vmin, vmax = 0.0, 1.0

    # NOTE: keep size logic, but labels are now bigger
    fig, ax = plt.subplots(figsize=(7.5, max(6, 0.22*len(mat))), dpi=DPI_SCREEN, facecolor="white")
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)

    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index.tolist(), fontsize=FS_YTICK)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["RF test R²", "DT test R²", "XGB test R²"], fontsize=FS_XTICK)

    ax.set_title("Per-country predictive performance (test R²) — sorted by mean performance",
                 fontsize=FS_TITLE)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("test R²", fontsize=FS_CBAR_L)
    cbar.ax.tick_params(labelsize=FS_CBAR_T)

    fig.tight_layout()
    save_fig(fig, out_png, out_pdf)


def agreement_heatmaps(agree_csv: Path, outdir: Path, topn_features: int):
    df = pd.read_csv(agree_csv)

    if "Country" not in df.columns:
        raise ValueError(f"Agreement CSV missing 'Country'. Available: {list(df.columns)}")

    spearman_cols = [
        f"Spearman_RF_DT_top{topn_features}",
        f"Spearman_RF_XGB_top{topn_features}",
        f"Spearman_DT_XGB_top{topn_features}",
    ]
    jacc_cols = [
        f"Jaccard_RF_DT_top{topn_features}",
        f"Jaccard_RF_XGB_top{topn_features}",
        f"Jaccard_DT_XGB_top{topn_features}",
    ]

    miss = [c for c in (spearman_cols + jacc_cols) if c not in df.columns]
    if miss:
        raise ValueError(f"Agreement CSV missing columns: {miss}\nAvailable: {list(df.columns)}")

    df["Mean_Spearman"] = df[spearman_cols].mean(axis=1)
    df["Mean_Jaccard"]  = df[jacc_cols].mean(axis=1)

    df_s = df.sort_values("Mean_Spearman", ascending=False).reset_index(drop=True)

    def heatmap(mat: pd.DataFrame, title: str, cbar_label: str,
                out_png: Path, out_pdf: Path, vmin: float, vmax: float):
        fig, ax = plt.subplots(figsize=(8, max(6, 0.22*len(mat))), dpi=DPI_SCREEN, facecolor="white")
        im = ax.imshow(mat.to_numpy(), aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)

        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index.tolist(), fontsize=FS_YTICK)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["RF vs DT", "RF vs XGB", "DT vs XGB"], fontsize=FS_XTICK)

        ax.set_title(title, fontsize=FS_TITLE)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label, fontsize=FS_CBAR_L)
        cbar.ax.tick_params(labelsize=FS_CBAR_T)

        fig.tight_layout()
        save_fig(fig, out_png, out_pdf)

    spearman_mat = df_s.set_index("Country")[spearman_cols]
    heatmap(
        spearman_mat,
        title=f"Feature-ranking agreement (Spearman ρ, Top-{topn_features}) — sorted by mean agreement",
        cbar_label="Spearman ρ",
        out_png=outdir / f"spearman_heatmap_top{topn_features}_sorted.png",
        out_pdf=outdir / f"spearman_heatmap_top{topn_features}_sorted.pdf",
        vmin=-1, vmax=1
    )

    jacc_mat = df_s.set_index("Country")[jacc_cols]
    heatmap(
        jacc_mat,
        title=f"Top-{topn_features} feature overlap (Jaccard) — sorted by mean Spearman agreement",
        cbar_label="Jaccard overlap",
        out_png=outdir / f"jaccard_heatmap_top{topn_features}_sorted.png",
        out_pdf=outdir / f"jaccard_heatmap_top{topn_features}_sorted.pdf",
        vmin=0, vmax=1
    )

    summary = pd.DataFrame({
        "pair": ["RF vs DT", "RF vs XGB", "DT vs XGB"],
        "spearman_mean": [df[c].mean() for c in spearman_cols],
        "spearman_median": [df[c].median() for c in spearman_cols],
        "jaccard_mean": [df[c].mean() for c in jacc_cols],
        "jaccard_median": [df[c].median() for c in jacc_cols],
    })
    summary.to_csv(outdir / f"agreement_summary_top{topn_features}.csv", index=False)
    df_s.to_csv(outdir / f"feature_ranking_agreement_top{topn_features}_sorted.csv", index=False)

    print("[OK] Agreement plots saved to:", outdir.resolve())


# =========================
# MAIN
# =========================
def main():
    perf_csv = Path(PERF_CSV)
    agree_csv = Path(AGREE_CSV)

    # ---------- Performance ----------
    if not perf_csv.exists():
        raise FileNotFoundError(f"Cannot find performance CSV: {perf_csv.resolve()}")

    df_perf = pd.read_csv(perf_csv)

    needed_any = ["Country", "RF_test_R2", "DT_test_R2", "XGB_test_R2"]
    miss_any = [c for c in needed_any if c not in df_perf.columns]
    if miss_any:
        raise ValueError(
            "Your performance CSV columns don’t match what this script expects.\n"
            f"Missing: {miss_any}\n"
            f"Available columns: {list(df_perf.columns)}"
        )

    performance_overview_plot(
        df_perf=df_perf,
        out_png=OUTDIR / "performance_overview_testR2_all_countries.png",
        out_pdf=OUTDIR / "performance_overview_testR2_all_countries.pdf"
    )

    top_rf  = df_perf[["Country", "RF_test_R2"]].dropna().sort_values("RF_test_R2", ascending=False).head(TOPN_COUNTRIES)
    top_dt  = df_perf[["Country", "DT_test_R2"]].dropna().sort_values("DT_test_R2", ascending=False).head(TOPN_COUNTRIES)
    top_xgb = df_perf[["Country", "XGB_test_R2"]].dropna().sort_values("XGB_test_R2", ascending=False).head(TOPN_COUNTRIES)

    polar_barplot_top_countries(
        df_top=top_rf.rename(columns={"RF_test_R2": "test_R2"}),
        value_col="test_R2",
        title=f"Top-{TOPN_COUNTRIES} countries by test R² (Random Forest)",
        out_png=OUTDIR / f"circular_top{TOPN_COUNTRIES}_RF_testR2.png",
        out_pdf=OUTDIR / f"circular_top{TOPN_COUNTRIES}_RF_testR2.pdf",
    )
    polar_barplot_top_countries(
        df_top=top_dt.rename(columns={"DT_test_R2": "test_R2"}),
        value_col="test_R2",
        title=f"Top-{TOPN_COUNTRIES} countries by test R² (Decision Tree)",
        out_png=OUTDIR / f"circular_top{TOPN_COUNTRIES}_DT_testR2.png",
        out_pdf=OUTDIR / f"circular_top{TOPN_COUNTRIES}_DT_testR2.pdf",
    )
    polar_barplot_top_countries(
        df_top=top_xgb.rename(columns={"XGB_test_R2": "test_R2"}),
        value_col="test_R2",
        title=f"Top-{TOPN_COUNTRIES} countries by test R² (XGBoost)",
        out_png=OUTDIR / f"circular_top{TOPN_COUNTRIES}_XGB_testR2.png",
        out_pdf=OUTDIR / f"circular_top{TOPN_COUNTRIES}_XGB_testR2.pdf",
    )

    model_summary = pd.DataFrame({
        "model": ["RF", "DT", "XGB"],
        "mean_test_R2": [
            df_perf["RF_test_R2"].mean(skipna=True),
            df_perf["DT_test_R2"].mean(skipna=True),
            df_perf["XGB_test_R2"].mean(skipna=True),
        ],
        "median_test_R2": [
            df_perf["RF_test_R2"].median(skipna=True),
            df_perf["DT_test_R2"].median(skipna=True),
            df_perf["XGB_test_R2"].median(skipna=True),
        ],
    }).sort_values("mean_test_R2", ascending=False)
    model_summary.to_csv(OUTDIR / "model_summary_testR2.csv", index=False)

    # ---------- Agreement ----------
    if agree_csv.exists():
        agreement_heatmaps(agree_csv, OUTDIR, TOPN_FEATURES)
    else:
        print("[WARN] Agreement CSV not found, skipping agreement plots:", agree_csv.resolve())

    print("\nAll outputs saved in:", OUTDIR.resolve())


if __name__ == "__main__":
    main()



# %% Cell 4
# ============================================================
# NEW FIGURE ONLY (Circos-style, FIXED):
# - 3 sectors: RF / DT / XGB
# - TOPN blocks per sector (Top-N countries by model performance)
# - Blocks are numbered (1..TOPN) to avoid label clutter
# - Links connect the same country across sectors (if present in both)
# - Link width/alpha ∝ |Spearman rho| for that country + that pair
# - Also exports a mapping CSV: number -> country per sector
# Exports: PNG (600 dpi) + PDF into results_agreement/paper_plots
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path as MplPath
from pathlib import Path

# -------------------------
# USER PATHS
# -------------------------
PERF_CSV  = Path(r"results_paper_ready/per_country_train_val_test_summary.csv")
AGREE_CSV = Path(r"results_agreement/feature_ranking_agreement_top20.csv")

OUTDIR = Path("results_agreement") / "paper_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

TOPN = 10                 # <<<<<<<<<< change here (10 recommended)
TOPN_FEATURES = 20        # must match ..._top20 columns in AGREE_CSV
DPI_SCREEN = 300
DPI_EXPORT = 600

OUT_PNG = OUTDIR / f"spearman_circos_top{TOPN}_per_model_top{TOPN_FEATURES}.png"
OUT_PDF = OUTDIR / f"spearman_circos_top{TOPN}_per_model_top{TOPN_FEATURES}.pdf"
OUT_MAP = OUTDIR / f"spearman_circos_top{TOPN}_country_mapping.csv"

# -------------------------
# Style / geometry
# -------------------------
SECTOR_COLORS = {"RF": "#4C78A8", "DT": "#F58518", "XGB": "#54A24B"}
POS_COLOR = "#2C7BB6"   # positive rho
NEG_COLOR = "#D7191C"   # negative rho

R_OUT = 1.00
R_IN  = 0.84
R_LINK = 0.78

GAP_DEG = 18.0  # bigger gaps -> cleaner
SECTOR_LABEL_FONTSIZE = 14
BLOCKNUM_FONTSIZE = 9

# Only show block numbers, not country names
SHOW_COUNTRY_NAMES = False  # keep False for publication


# -------------------------
# Geometry helpers
# -------------------------
def pol2cart(theta, r):
    return np.array([r*np.cos(theta), r*np.sin(theta)], float)

def bezier_link(theta_a, theta_b, r=R_LINK, bend=0.55):
    """
    Cubic bezier path connecting two angles on a circle.
    Larger bend pulls the curve inward -> nicer circos.
    """
    p0 = pol2cart(theta_a, r)
    p3 = pol2cart(theta_b, r)
    c0 = p0 * (1 - bend)
    c1 = p3 * (1 - bend)
    verts = [tuple(p0), tuple(c0), tuple(c1), tuple(p3)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    return MplPath(verts, codes)

def angle_for_block(sector_start, sector_end, i, n):
    frac = (i + 0.5) / n
    return sector_start + frac * (sector_end - sector_start)


# -------------------------
# Plot
# -------------------------
def plot_circos(top_rf, top_dt, top_xgb, agree_df, out_png, out_pdf, out_map, title):
    sectors = [("RF", top_rf), ("DT", top_dt), ("XGB", top_xgb)]

    total_gap = GAP_DEG * len(sectors)
    usable = 360.0 - total_gap
    span = usable / len(sectors)

    # Put RF at right-ish, DT bottom-ish, XGB left-ish (pleasant reading)
    theta0 = np.deg2rad(40.0)  # start angle reference
    sector_ranges = {}
    cursor = theta0

    for name, _ in sectors:
        start = cursor
        end = cursor - np.deg2rad(span)  # clockwise
        sector_ranges[name] = (start, end)
        cursor = end - np.deg2rad(GAP_DEG)

    fig = plt.figure(figsize=(10, 10), dpi=DPI_SCREEN, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")

    # Give generous limits so everything stays centered
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)

    block_pos = {}   # (sector, country) -> theta
    block_rank = {}  # (sector, country) -> rank (1..TOPN)

    # ----- draw sectors + blocks -----
    for sec_name, df_top in sectors:
        start, end = sector_ranges[sec_name]

        # Sector base ring
        base = Wedge((0, 0), R_OUT,
                     np.degrees(end), np.degrees(start),
                     width=(R_OUT - R_IN),
                     facecolor=SECTOR_COLORS[sec_name],
                     edgecolor="white", lw=2)
        ax.add_patch(base)

        # Sector label
        mid = 0.5 * (start + end)
        p = pol2cart(mid, 1.12)
        rot = np.degrees(mid) - 90
        # keep readable
        if rot < -90: rot += 180
        if rot > 90:  rot -= 180
        ax.text(p[0], p[1], sec_name, fontsize=SECTOR_LABEL_FONTSIZE,
                fontweight="bold", ha="center", va="center", rotation=rot)

        countries = df_top["Country"].astype(str).tolist()
        n = len(countries)

        for i, ctry in enumerate(countries):
            th_center = angle_for_block(start, end, i, n)
            block_pos[(sec_name, ctry)] = th_center
            block_rank[(sec_name, ctry)] = i + 1

            th1 = start + (i / n) * (end - start)
            th2 = start + ((i + 1) / n) * (end - start)

            # "block" wedge with white separator edge
            blk = Wedge((0, 0), R_OUT,
                        np.degrees(th2), np.degrees(th1),
                        width=(R_OUT - R_IN),
                        facecolor=SECTOR_COLORS[sec_name],
                        edgecolor="white", lw=1.4)
            ax.add_patch(blk)

            # label = number (1..TOPN) placed just outside
            label = str(i + 1) if not SHOW_COUNTRY_NAMES else ctry
            ptxt = pol2cart(th_center, 1.04)
            rrot = np.degrees(th_center) - 90
            if rrot < -90: rrot += 180
            if rrot > 90:  rrot -= 180
            ax.text(ptxt[0], ptxt[1], label, fontsize=BLOCKNUM_FONTSIZE,
                    ha="center", va="center", rotation=rrot, color="#111")

    # ----- agreement lookup -----
    c12 = f"Spearman_RF_DT_top{TOPN_FEATURES}"
    c13 = f"Spearman_RF_XGB_top{TOPN_FEATURES}"
    c23 = f"Spearman_DT_XGB_top{TOPN_FEATURES}"

    agree_df = agree_df.copy()
    agree_df["Country"] = agree_df["Country"].astype(str)
    agree_map = agree_df.set_index("Country")[[c12, c13, c23]].to_dict(orient="index")

    def get_rho(country, pair):
        if country not in agree_map:
            return np.nan
        d = agree_map[country]
        if pair == ("RF", "DT"):
            return float(d.get(c12, np.nan))
        if pair == ("RF", "XGB"):
            return float(d.get(c13, np.nan))
        if pair == ("DT", "XGB"):
            return float(d.get(c23, np.nan))
        return np.nan

    # Countries in each sector
    sector_sets = {
        "RF": set(top_rf["Country"].astype(str)),
        "DT": set(top_dt["Country"].astype(str)),
        "XGB": set(top_xgb["Country"].astype(str)),
    }

    # Scale normalization based only on links that will be drawn
    abs_rhos = []
    for country in set.union(*sector_sets.values()):
        for a, b in [("RF", "DT"), ("RF", "XGB"), ("DT", "XGB")]:
            if (country in sector_sets[a]) and (country in sector_sets[b]):
                rho = get_rho(country, (a, b))
                if np.isfinite(rho):
                    abs_rhos.append(abs(rho))
    max_abs = max(abs_rhos) if abs_rhos else 1.0
    max_abs = max(max_abs, 1e-9)

    # ----- links -----
    for country in set.union(*sector_sets.values()):
        for a, b in [("RF", "DT"), ("RF", "XGB"), ("DT", "XGB")]:
            if (country in sector_sets[a]) and (country in sector_sets[b]):
                th_a = block_pos.get((a, country), None)
                th_b = block_pos.get((b, country), None)
                if th_a is None or th_b is None:
                    continue

                rho = get_rho(country, (a, b))
                if not np.isfinite(rho):
                    continue

                # Slightly toned down widths to avoid clutter
                lw = 0.6 + 6.0 * (abs(rho) / max_abs)
                alpha = 0.08 + 0.45 * (abs(rho) / max_abs)
                col = POS_COLOR if rho >= 0 else NEG_COLOR

                path = bezier_link(th_a, th_b, r=R_LINK, bend=0.60)
                ax.add_patch(PathPatch(path, facecolor="none", edgecolor=col,
                                       lw=lw, alpha=alpha, capstyle="round"))

    # ----- title + legend -----
    ax.text(0, 1.30, title, ha="center", va="center", fontsize=13)

    ax.text(0, -1.18, "Blocks are ranked (1 = best test R² within model).",
            ha="center", va="center", fontsize=10, color="#444")
    ax.text(0, -1.28, "Links connect the same country across models (only if present in both Top-N lists).",
            ha="center", va="center", fontsize=10, color="#444")
    ax.text(0, -1.38, "Link thickness/opacity ∝ |Spearman ρ| (country-wise feature ranking agreement).",
            ha="center", va="center", fontsize=10, color="#444")
    ax.text(0, -1.48, "Blue: positive agreement   •   Red: negative agreement",
            ha="center", va="center", fontsize=10, color="#444")

    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # ----- mapping table export -----
    max_len = max(len(top_rf), len(top_dt), len(top_xgb))
    def pad(lst, n):
        return lst + ["" for _ in range(n - len(lst))]

    rf_c = top_rf["Country"].astype(str).tolist()
    dt_c = top_dt["Country"].astype(str).tolist()
    xg_c = top_xgb["Country"].astype(str).tolist()

    mapping = pd.DataFrame({
        "RF_rank": list(range(1, len(rf_c) + 1)) + ["" for _ in range(max_len - len(rf_c))],
        "RF_country": pad(rf_c, max_len),
        "DT_rank": list(range(1, len(dt_c) + 1)) + ["" for _ in range(max_len - len(dt_c))],
        "DT_country": pad(dt_c, max_len),
        "XGB_rank": list(range(1, len(xg_c) + 1)) + ["" for _ in range(max_len - len(xg_c))],
        "XGB_country": pad(xg_c, max_len),
    })
    mapping.to_csv(out_map, index=False)


# -------------------------
# MAIN
# -------------------------
def main():
    if not PERF_CSV.exists():
        raise FileNotFoundError(f"Missing PERF_CSV: {PERF_CSV.resolve()}")
    if not AGREE_CSV.exists():
        raise FileNotFoundError(f"Missing AGREE_CSV: {AGREE_CSV.resolve()}")

    perf = pd.read_csv(PERF_CSV)
    agree = pd.read_csv(AGREE_CSV)

    needed_perf = ["Country", "RF_test_R2", "DT_test_R2", "XGB_test_R2"]
    miss = [c for c in needed_perf if c not in perf.columns]
    if miss:
        raise ValueError(f"PERF_CSV missing columns: {miss}\nAvailable: {list(perf.columns)}")

    needed_ag = ["Country",
                 f"Spearman_RF_DT_top{TOPN_FEATURES}",
                 f"Spearman_RF_XGB_top{TOPN_FEATURES}",
                 f"Spearman_DT_XGB_top{TOPN_FEATURES}"]
    miss2 = [c for c in needed_ag if c not in agree.columns]
    if miss2:
        raise ValueError(f"AGREE_CSV missing columns: {miss2}\nAvailable: {list(agree.columns)}")

    top_rf = perf[["Country", "RF_test_R2"]].dropna().sort_values("RF_test_R2", ascending=False).head(TOPN)
    top_dt = perf[["Country", "DT_test_R2"]].dropna().sort_values("DT_test_R2", ascending=False).head(TOPN)
    top_xg = perf[["Country", "XGB_test_R2"]].dropna().sort_values("XGB_test_R2", ascending=False).head(TOPN)

    plot_circos(
        top_rf=top_rf, top_dt=top_dt, top_xgb=top_xg,
        agree_df=agree,
        out_png=OUT_PNG, out_pdf=OUT_PDF, out_map=OUT_MAP,
        title=f"Country-wise agreement in feature ranking (Spearman ρ, Top-{TOPN_FEATURES}; Top-{TOPN} countries per model)"
    )

    print("[OK] Saved:")
    print(" -", OUT_PNG.resolve())
    print(" -", OUT_PDF.resolve())
    print(" -", OUT_MAP.resolve())

if __name__ == "__main__":
    main()



# %% Cell 5
# ============================================================
# NEW FIGURE ONLY (Circos-style, CLEAN):
# - 3 sectors: RF / DT / XGB
# - TOPN country blocks per sector (Top-N by model performance)
# - Country NAMES shown on ring (TOPN=10 recommended)
# - Links connect same country across sectors (if present in both Top-N lists)
# - Link width/alpha ∝ |Spearman rho| for that country + pair
# - NO explanatory text under the figure (title only)
# Exports: PNG (600 dpi) + PDF into results_agreement/paper_plots
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path as MplPath
from pathlib import Path

# -------------------------
# USER PATHS
# -------------------------
PERF_CSV  = Path(r"results_paper_ready/per_country_train_val_test_summary.csv")
AGREE_CSV = Path(r"results_agreement/feature_ranking_agreement_top20.csv")

OUTDIR = Path("results_agreement") / "paper_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

TOPN = 10
TOPN_FEATURES = 20

DPI_SCREEN = 300
DPI_EXPORT = 600

OUT_PNG = OUTDIR / f"spearman_circos_top{TOPN}_names_top{TOPN_FEATURES}.png"
OUT_PDF = OUTDIR / f"spearman_circos_top{TOPN}_names_top{TOPN_FEATURES}.pdf"

# -------------------------
# Harmonious palette (soft, journal-friendly)
# (inspired by ColorBrewer Set2-like tones)
# -------------------------
SECTOR_COLORS = {
    "RF":  "#6C8EBF",   # soft blue
    "DT":  "#E5A36F",   # soft orange
    "XGB": "#7FBF9B",   # soft green
}

# Links: use a single hue (cleaner than red/blue mix)
# If you still want sign-coding, set SIGN_COLORING=True below.
SIGN_COLORING = False
POS_COLOR = "#5A91C2"
NEG_COLOR = "#C45A5A"
LINK_COLOR = "#6D9EC8"   # used when SIGN_COLORING=False

R_OUT  = 1.00
R_IN   = 0.84
R_LINK = 0.78

GAP_DEG = 18.0

SECTOR_LABEL_FONTSIZE = 14
COUNTRY_FONTSIZE = 8  # for TOPN=10; if too tight, drop to 7

# Show country names (as requested)
SHOW_COUNTRY_NAMES = True


# -------------------------
# Helpers
# -------------------------
def pol2cart(theta, r):
    return np.array([r*np.cos(theta), r*np.sin(theta)], float)

def bezier_link(theta_a, theta_b, r=R_LINK, bend=0.60):
    p0 = pol2cart(theta_a, r)
    p3 = pol2cart(theta_b, r)
    c0 = p0 * (1 - bend)
    c1 = p3 * (1 - bend)
    verts = [tuple(p0), tuple(c0), tuple(c1), tuple(p3)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    return MplPath(verts, codes)

def angle_for_block(sector_start, sector_end, i, n):
    frac = (i + 0.5) / n
    return sector_start + frac * (sector_end - sector_start)

def nice_text_rotation(theta):
    """Return rotation (deg) and alignment so labels remain readable."""
    rot = np.degrees(theta) - 90
    # Flip upside-down labels
    if rot < -90:
        rot += 180
        ha = "right"
    elif rot > 90:
        rot -= 180
        ha = "right"
    else:
        ha = "left"
    return rot, ha


# -------------------------
# Plot
# -------------------------
def plot_circos(top_rf, top_dt, top_xgb, agree_df, out_png, out_pdf, title):
    sectors = [("RF", top_rf), ("DT", top_dt), ("XGB", top_xgb)]

    total_gap = GAP_DEG * len(sectors)
    usable = 360.0 - total_gap
    span = usable / len(sectors)

    # Start angle to place RF on the right side-ish
    theta0 = np.deg2rad(40.0)
    sector_ranges = {}
    cursor = theta0

    for name, _ in sectors:
        start = cursor
        end = cursor - np.deg2rad(span)  # clockwise
        sector_ranges[name] = (start, end)
        cursor = end - np.deg2rad(GAP_DEG)

    fig = plt.figure(figsize=(10, 10), dpi=DPI_SCREEN, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)

    block_pos = {}  # (sector, country) -> theta

    # ----- draw sectors + blocks -----
    for sec_name, df_top in sectors:
        start, end = sector_ranges[sec_name]

        # Base sector ring
        base = Wedge((0, 0), R_OUT,
                     np.degrees(end), np.degrees(start),
                     width=(R_OUT - R_IN),
                     facecolor=SECTOR_COLORS[sec_name],
                     edgecolor="white", lw=2)
        ax.add_patch(base)

        # Sector label
        mid = 0.5 * (start + end)
        p = pol2cart(mid, 1.10)
        rot, ha = nice_text_rotation(mid)
        ax.text(p[0], p[1], sec_name, fontsize=SECTOR_LABEL_FONTSIZE,
                fontweight="bold", ha="center", va="center", rotation=rot)

        countries = df_top["Country"].astype(str).tolist()
        n = len(countries)

        for i, ctry in enumerate(countries):
            th_center = angle_for_block(start, end, i, n)
            block_pos[(sec_name, ctry)] = th_center

            th1 = start + (i / n) * (end - start)
            th2 = start + ((i + 1) / n) * (end - start)

            blk = Wedge((0, 0), R_OUT,
                        np.degrees(th2), np.degrees(th1),
                        width=(R_OUT - R_IN),
                        facecolor=SECTOR_COLORS[sec_name],
                        edgecolor="white", lw=1.6)
            ax.add_patch(blk)

            if SHOW_COUNTRY_NAMES:
                # Place country name slightly outside the ring
                ptxt = pol2cart(th_center, 1.05)
                rrot, ha2 = nice_text_rotation(th_center)
                ax.text(ptxt[0], ptxt[1], ctry,
                        fontsize=COUNTRY_FONTSIZE,
                        ha=ha2, va="center",
                        rotation=rrot, color="#111")

    # ----- agreement lookup -----
    c12 = f"Spearman_RF_DT_top{TOPN_FEATURES}"
    c13 = f"Spearman_RF_XGB_top{TOPN_FEATURES}"
    c23 = f"Spearman_DT_XGB_top{TOPN_FEATURES}"

    agree_df = agree_df.copy()
    agree_df["Country"] = agree_df["Country"].astype(str)
    agree_map = agree_df.set_index("Country")[[c12, c13, c23]].to_dict(orient="index")

    def get_rho(country, pair):
        if country not in agree_map:
            return np.nan
        d = agree_map[country]
        if pair == ("RF", "DT"):
            return float(d.get(c12, np.nan))
        if pair == ("RF", "XGB"):
            return float(d.get(c13, np.nan))
        if pair == ("DT", "XGB"):
            return float(d.get(c23, np.nan))
        return np.nan

    sector_sets = {
        "RF": set(top_rf["Country"].astype(str)),
        "DT": set(top_dt["Country"].astype(str)),
        "XGB": set(top_xgb["Country"].astype(str)),
    }

    # Normalize widths based on links that will be drawn
    abs_rhos = []
    for country in set.union(*sector_sets.values()):
        for a, b in [("RF", "DT"), ("RF", "XGB"), ("DT", "XGB")]:
            if (country in sector_sets[a]) and (country in sector_sets[b]):
                rho = get_rho(country, (a, b))
                if np.isfinite(rho):
                    abs_rhos.append(abs(rho))
    max_abs = max(abs_rhos) if abs_rhos else 1.0
    max_abs = max(max_abs, 1e-9)

    # ----- links -----
    for country in set.union(*sector_sets.values()):
        for a, b in [("RF", "DT"), ("RF", "XGB"), ("DT", "XGB")]:
            if (country in sector_sets[a]) and (country in sector_sets[b]):
                th_a = block_pos.get((a, country), None)
                th_b = block_pos.get((b, country), None)
                if th_a is None or th_b is None:
                    continue

                rho = get_rho(country, (a, b))
                if not np.isfinite(rho):
                    continue

                lw = 0.6 + 6.0 * (abs(rho) / max_abs)
                alpha = 0.10 + 0.45 * (abs(rho) / max_abs)

                if SIGN_COLORING:
                    col = POS_COLOR if rho >= 0 else NEG_COLOR
                else:
                    col = LINK_COLOR

                path = bezier_link(th_a, th_b, r=R_LINK, bend=0.60)
                ax.add_patch(PathPatch(path, facecolor="none", edgecolor=col,
                                       lw=lw, alpha=alpha, capstyle="round"))

    # Title only (no bottom explanatory text)
    ax.set_title(title, fontsize=16, pad=18)

    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# MAIN
# -------------------------
def main():
    if not PERF_CSV.exists():
        raise FileNotFoundError(f"Missing PERF_CSV: {PERF_CSV.resolve()}")
    if not AGREE_CSV.exists():
        raise FileNotFoundError(f"Missing AGREE_CSV: {AGREE_CSV.resolve()}")

    perf = pd.read_csv(PERF_CSV)
    agree = pd.read_csv(AGREE_CSV)

    needed_perf = ["Country", "RF_test_R2", "DT_test_R2", "XGB_test_R2"]
    miss = [c for c in needed_perf if c not in perf.columns]
    if miss:
        raise ValueError(f"PERF_CSV missing columns: {miss}\nAvailable: {list(perf.columns)}")

    needed_ag = ["Country",
                 f"Spearman_RF_DT_top{TOPN_FEATURES}",
                 f"Spearman_RF_XGB_top{TOPN_FEATURES}",
                 f"Spearman_DT_XGB_top{TOPN_FEATURES}"]
    miss2 = [c for c in needed_ag if c not in agree.columns]
    if miss2:
        raise ValueError(f"AGREE_CSV missing columns: {miss2}\nAvailable: {list(agree.columns)}")

    top_rf = perf[["Country", "RF_test_R2"]].dropna().sort_values("RF_test_R2", ascending=False).head(TOPN)
    top_dt = perf[["Country", "DT_test_R2"]].dropna().sort_values("DT_test_R2", ascending=False).head(TOPN)
    top_xg = perf[["Country", "XGB_test_R2"]].dropna().sort_values("XGB_test_R2", ascending=False).head(TOPN)

    plot_circos(
        top_rf=top_rf, top_dt=top_dt, top_xgb=top_xg,
        agree_df=agree,
        out_png=OUT_PNG, out_pdf=OUT_PDF,
        title=f"Country-wise agreement in feature ranking (Spearman ρ, Top-{TOPN_FEATURES}; Top-{TOPN} countries per model)"
    )

    print("[OK] Saved:")
    print(" -", OUT_PNG.resolve())
    print(" -", OUT_PDF.resolve())

if __name__ == "__main__":
    main()



# %% Cell 6



# %% Cell 7

