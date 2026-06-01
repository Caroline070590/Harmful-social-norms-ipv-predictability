# Auto-extracted from notebooks/11_country_clustering_macro_indicators.ipynb

# Review/reproducibility helper script. Original notebook is retained in notebooks/.



# %% Cell 0
# cluster_country_profiles_FIXED_FULL.py
# ------------------------------------------------------------
# Cluster countries by profile similarity (External indicators; Hofstede optional)
#
# INPUT:
#   - analysis_table_part2_with_external_indicators.csv
#     Must contain: iso3 + external indicator columns (see EXTERNAL_COLS)
#     Hofstede columns are optional (see HOFSTEDE_COLS)
#
# OUTPUT (in OUTDIR):
#   - Cluster_external_indicators.png/.pdf
#   - Cluster_hofstede_dimensions.png/.pdf   (only if Hofstede columns exist)
#
# Dependencies:
#   pip install numpy pandas matplotlib scipy
# Optional (nicer country labels):
#   pip install pycountry
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Set, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist

try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# ----------------------------
# USER SETTINGS
# ----------------------------
DATA = Path("results_part2_external_hierarchy/analysis_table_part2_with_external_indicators.csv")
OUTDIR = Path("figures_country_clustering")
DPI_EXPORT = 600

# Your country sets (names; we convert to ISO3 for tagging in labels)
COUNTRIES_PART1 = ["Afghanistan", "Madagascar", "Gambia", "Malawi", "Morocco"]
COUNTRIES_PART2 = ["Ethiopia", "Niger", "Morocco", "Senegal", "Benin"]

# External indicator columns (must exist in DATA)
EXTERNAL_COLS = [
    "gii_latest",
    "gdi_latest",
    "hdi_latest",
    "electoral-democracy-index_latest",
    "liberal-democracy-index_latest",
    "women-political-empowerment-index_latest",
]

# Hofstede columns (optional; only plotted if present in DATA)
HOFSTEDE_COLS = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]

PRETTY = {
    "gii_latest": "GII",
    "gdi_latest": "GDI",
    "hdi_latest": "HDI",
    "electoral-democracy-index_latest": "Electoral democracy",
    "liberal-democracy-index_latest": "Liberal democracy",
    "women-political-empowerment-index_latest": "Women political empowerment",
    "PDI": "PDI",
    "IDV": "IDV",
    "MAS": "MAS",
    "UAI": "UAI",
    "LTO": "LTO",
    "IVR": "IVR",
}

# Clustering choices
# - If METRIC="euclidean": LINKAGE_METHOD="ward" is recommended
# - If METRIC="correlation": use LINKAGE_METHOD="average" (ward is invalid)
METRIC = "euclidean"            # "euclidean" or "correlation"
LINKAGE_METHOD = "ward"         # "ward" or "average"
# NOTE: if you set METRIC="correlation", change LINKAGE_METHOD="average"


# ----------------------------
# Plot style
# ----------------------------
def set_pub_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(fig, out_png: Path, out_pdf: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# ISO3 helpers
# ----------------------------
def iso3_to_name(iso3: str) -> str:
    iso3 = str(iso3).upper()
    if not HAS_PYCOUNTRY:
        return iso3
    try:
        c = pycountry.countries.get(alpha_3=iso3)
        return c.name if c else iso3
    except Exception:
        return iso3


def name_to_iso3(name: str) -> Optional[str]:
    if not HAS_PYCOUNTRY:
        return None

    overrides = {
        "Morocco": "MAR",
        "Gambia": "GMB",
        "Malawi": "MWI",
        "Madagascar": "MDG",
        "Afghanistan": "AFG",
        "Ethiopia": "ETH",
        "Niger": "NER",
        "Senegal": "SEN",
        "Benin": "BEN",
    }
    if name in overrides:
        return overrides[name]

    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None


def build_tag_sets() -> Tuple[Set[str], Set[str]]:
    p1 = {name_to_iso3(n) for n in COUNTRIES_PART1}
    p2 = {name_to_iso3(n) for n in COUNTRIES_PART2}
    # remove None
    p1 = {x for x in p1 if x is not None}
    p2 = {x for x in p2 if x is not None}
    return p1, p2


def mark_groups(iso3: str, p1_iso3: Set[str], p2_iso3: Set[str]) -> str:
    iso3 = str(iso3).upper()
    tag = ""
    if iso3 in p1_iso3:
        tag += " [P1]"
    if iso3 in p2_iso3:
        tag += " [P2]" if tag == "" else "+P2"
    return tag


# ----------------------------
# Data transforms
# ----------------------------
def zscore_by_column(X: pd.DataFrame) -> pd.DataFrame:
    M = X.to_numpy(float)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (M - mu) / sd
    return pd.DataFrame(Z, index=X.index, columns=X.columns)


def cluster_order_rows(Z: pd.DataFrame, metric: str, method: str):
    # linkage on row distances
    D = pdist(Z.to_numpy(float), metric=metric)
    L = linkage(D, method=method)
    return leaves_list(L), L


# ----------------------------
# Plot: dendrogram + clustered heatmap
# ----------------------------
def plot_clustered_heatmap(
    Z: pd.DataFrame,
    title: str,
    outbase: Path,
    p1_iso3: Set[str],
    p2_iso3: Set[str],
):
    set_pub_style()

    row_order, L = cluster_order_rows(Z, METRIC, LINKAGE_METHOD)
    Zr = Z.iloc[row_order].copy()

    fig = plt.figure(figsize=(12.8, max(4.5, 0.52 * len(Zr) + 2.0)), dpi=300)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.1, 4.2], wspace=0.05)

    # dendrogram
    ax_d = fig.add_subplot(gs[0, 0])
    dendrogram(L, orientation="left", no_labels=True, ax=ax_d, color_threshold=None)
    ax_d.invert_yaxis()
    ax_d.axis("off")

    # heatmap
    ax_h = fig.add_subplot(gs[0, 1])
    im = ax_h.imshow(Zr.to_numpy(float), aspect="auto")

    # y labels: country name (or iso3) + tags
    ylabels: List[str] = []
    for iso3 in Zr.index.astype(str).tolist():
        base = iso3_to_name(iso3) if HAS_PYCOUNTRY else str(iso3).upper()
        ylabels.append(base + mark_groups(iso3, p1_iso3, p2_iso3))

    ax_h.set_yticks(np.arange(len(Zr)))
    ax_h.set_yticklabels(ylabels)

    ax_h.set_xticks(np.arange(Zr.shape[1]))
    ax_h.set_xticklabels([PRETTY.get(c, c) for c in Zr.columns], rotation=30, ha="right")

    ax_h.set_title(title)

    # annotate z-scores (comment out if too busy)
    for i in range(Zr.shape[0]):
        for j in range(Zr.shape[1]):
            v = Zr.iat[i, j]
            if np.isfinite(v):
                ax_h.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax_h, fraction=0.03, pad=0.02)
    cbar.set_label("z-score (column-wise, across all countries)")

    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


# ----------------------------
# Main
# ----------------------------
def main():
    if METRIC == "correlation" and LINKAGE_METHOD == "ward":
        raise ValueError("If METRIC='correlation', you must set LINKAGE_METHOD='average' (ward is invalid).")

    if not DATA.exists():
        raise FileNotFoundError(f"Cannot find: {DATA.resolve()}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    if "iso3" not in df.columns:
        raise ValueError(f"Your CSV must contain 'iso3'. Columns found: {list(df.columns)}")

    df["iso3"] = df["iso3"].astype(str).str.upper()
    df = df.set_index("iso3")

    p1_iso3, p2_iso3 = build_tag_sets()

    # ---------- External ----------
    missing_ext = [c for c in EXTERNAL_COLS if c not in df.columns]
    if missing_ext:
        raise ValueError(f"Missing required external columns: {missing_ext}")

    Xext = df[EXTERNAL_COLS].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if Xext.shape[0] < 3:
        raise ValueError("Too few countries with complete EXTERNAL indicators after dropna(how='any').")

    Zext = zscore_by_column(Xext)
    plot_clustered_heatmap(
        Zext,
        title="Country clustering by EXTERNAL hierarchy/gender indicators",
        outbase=OUTDIR / "Cluster_external_indicators",
        p1_iso3=p1_iso3,
        p2_iso3=p2_iso3,
    )

    # ---------- Hofstede (optional) ----------
    missing_hof = [c for c in HOFSTEDE_COLS if c not in df.columns]
    if not missing_hof:
        Xhof = df[HOFSTEDE_COLS].apply(pd.to_numeric, errors="coerce").dropna(how="any")
        if Xhof.shape[0] >= 3:
            Zhof = zscore_by_column(Xhof)
            plot_clustered_heatmap(
                Zhof,
                title="Country clustering by Hofstede dimensions",
                outbase=OUTDIR / "Cluster_hofstede_dimensions",
                p1_iso3=p1_iso3,
                p2_iso3=p2_iso3,
            )
        else:
            print("[INFO] Hofstede columns exist, but too few complete rows after dropna. Skipping Hofstede clustering.")
    else:
        print("[INFO] Hofstede columns not found in this table. Skipping Hofstede clustering.")
        print("       Missing:", missing_hof)
        print("       (If you want Hofstede clustering, merge Hofstede into this table or point DATA to the merged file.)")

    print("[OK] Done. Outputs in:", OUTDIR.resolve())
    print(" - Cluster_external_indicators.(png/pdf)")
    print(" - Cluster_hofstede_dimensions.(png/pdf)  (if available)")


if __name__ == "__main__":
    main()



# %% Cell 1
# cluster_selected_countries_one_dendrogram.py
# ------------------------------------------------------------
# Cluster ONLY your selected countries (union of Part1 + Part2),
# and draw ONE dendrogram + heatmap (external indicators).
#
# Countries requested:
#   Part1: Afghanistan, Madagascar, Malawi, Morocco
#   Part2: Ethiopia, Niger, Morocco, Senegal, Benin
# Union (unique): Afghanistan, Madagascar, Malawi, Morocco, Ethiopia, Niger, Senegal, Benin
#
# INPUT:
#   results_part2_external_hierarchy/analysis_table_part2_with_external_indicators.csv
#   must contain: iso3 + EXTERNAL_COLS
#
# OUTPUT (OUTDIR):
#   - Cluster_selected_external_indicators.(png/pdf)
#
# Dependencies:
#   pip install numpy pandas matplotlib scipy
# Optional:
#   pip install pycountry   (for country names in labels)
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist

try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# ----------------------------
# USER SETTINGS
# ----------------------------
DATA = Path("results_part2_external_hierarchy/analysis_table_part2_with_external_indicators.csv")
OUTDIR = Path("figures_country_clustering")
DPI_EXPORT = 600

COUNTRIES_PART1 = ["Afghanistan", "Madagascar", "Malawi", "Morocco"]
COUNTRIES_PART2 = ["Ethiopia", "Niger", "Morocco", "Senegal", "Benin"]

EXTERNAL_COLS = [
    "gii_latest",
    "gdi_latest",
    "hdi_latest",
    "electoral-democracy-index_latest",
    "liberal-democracy-index_latest",
    "women-political-empowerment-index_latest",
]

PRETTY = {
    "gii_latest": "GII",
    "gdi_latest": "GDI",
    "hdi_latest": "HDI",
    "electoral-democracy-index_latest": "Electoral democracy",
    "liberal-democracy-index_latest": "Liberal democracy",
    "women-political-empowerment-index_latest": "Women political empowerment",
}

# Clustering choices
METRIC = "euclidean"        # "euclidean" or "correlation"
LINKAGE_METHOD = "ward"     # for euclidean: "ward"; for correlation: "average"

FIGTITLE = "Selected countries: clustering by external hierarchy / gender indicators"


# ----------------------------
# Plot style
# ----------------------------
def set_pub_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(fig, out_png: Path, out_pdf: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# ISO3 helpers
# ----------------------------
def iso3_to_name(iso3: str) -> str:
    iso3 = str(iso3).upper()
    if not HAS_PYCOUNTRY:
        return iso3
    try:
        c = pycountry.countries.get(alpha_3=iso3)
        return c.name if c else iso3
    except Exception:
        return iso3


def name_to_iso3(name: str) -> Optional[str]:
    if not HAS_PYCOUNTRY:
        return None

    overrides = {
        "Morocco": "MAR",
        "Malawi": "MWI",
        "Madagascar": "MDG",
        "Afghanistan": "AFG",
        "Ethiopia": "ETH",
        "Niger": "NER",
        "Senegal": "SEN",
        "Benin": "BEN",
    }
    if name in overrides:
        return overrides[name]

    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None


def build_selected_iso3() -> Tuple[Set[str], Set[str], List[str]]:
    """
    Returns:
      p1_iso3, p2_iso3, selected_union_list (unique, deterministic order)
    """
    p1_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART1}
    p2_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART2}
    p1_iso3 = {x for x in p1_iso3 if x is not None}
    p2_iso3 = {x for x in p2_iso3 if x is not None}

    # deterministic union order: Part1 order, then Part2 order skipping duplicates
    union: List[str] = []
    for n in COUNTRIES_PART1:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    for n in COUNTRIES_PART2:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)

    return p1_iso3, p2_iso3, union


def mark_groups(iso3: str, p1_iso3: Set[str], p2_iso3: Set[str]) -> str:
    iso3 = str(iso3).upper()
    tag = ""
    if iso3 in p1_iso3:
        tag += " [P1]"
    if iso3 in p2_iso3:
        tag += " [P2]" if tag == "" else "+P2"
    return tag


# ----------------------------
# Data transforms
# ----------------------------
def zscore_by_column(X: pd.DataFrame) -> pd.DataFrame:
    M = X.to_numpy(float)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (M - mu) / sd
    return pd.DataFrame(Z, index=X.index, columns=X.columns)


def cluster_order_rows(Z: pd.DataFrame, metric: str, method: str):
    D = pdist(Z.to_numpy(float), metric=metric)
    L = linkage(D, method=method)
    return leaves_list(L), L


# ----------------------------
# Plot: ONE dendrogram + heatmap
# ----------------------------
def plot_clustered_heatmap_selected(
    Z: pd.DataFrame,
    title: str,
    outbase: Path,
    p1_iso3: Set[str],
    p2_iso3: Set[str],
):
    set_pub_style()

    row_order, L = cluster_order_rows(Z, METRIC, LINKAGE_METHOD)
    Zr = Z.iloc[row_order].copy()

    fig = plt.figure(figsize=(12.2, max(4.2, 0.65 * len(Zr) + 2.0)), dpi=300)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.15, 4.0], wspace=0.05)

    # dendrogram
    ax_d = fig.add_subplot(gs[0, 0])
    dendrogram(L, orientation="left", no_labels=True, ax=ax_d, color_threshold=None)
    ax_d.invert_yaxis()
    ax_d.axis("off")

    # heatmap
    ax_h = fig.add_subplot(gs[0, 1])
    im = ax_h.imshow(Zr.to_numpy(float), aspect="auto")

    ylabels: List[str] = []
    for iso3 in Zr.index.astype(str).tolist():
        base = iso3_to_name(iso3) if HAS_PYCOUNTRY else str(iso3).upper()
        ylabels.append(base + mark_groups(iso3, p1_iso3, p2_iso3))

    ax_h.set_yticks(np.arange(len(Zr)))
    ax_h.set_yticklabels(ylabels)

    ax_h.set_xticks(np.arange(Zr.shape[1]))
    ax_h.set_xticklabels([PRETTY.get(c, c) for c in Zr.columns], rotation=30, ha="right")

    ax_h.set_title(title)

    # annotate z-scores (few rows -> OK; comment out if you prefer cleaner)
    for i in range(Zr.shape[0]):
        for j in range(Zr.shape[1]):
            v = Zr.iat[i, j]
            if np.isfinite(v):
                ax_h.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax_h, fraction=0.035, pad=0.02)
    cbar.set_label("z-score (computed across the selected countries)")

    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


# ----------------------------
# Main
# ----------------------------
def main():
    if METRIC == "correlation" and LINKAGE_METHOD == "ward":
        raise ValueError("If METRIC='correlation', set LINKAGE_METHOD='average' (ward is invalid).")

    if not DATA.exists():
        raise FileNotFoundError(f"Cannot find: {DATA.resolve()}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # selected countries (iso3)
    p1_iso3, p2_iso3, selected_iso3_order = build_selected_iso3()
    if len(selected_iso3_order) < 3:
        raise ValueError(
            "Too few selected countries after name->ISO3 mapping. "
            "Install pycountry or check country name spelling."
        )

    df = pd.read_csv(DATA)
    if "iso3" not in df.columns:
        raise ValueError(f"Your CSV must contain 'iso3'. Columns found: {list(df.columns)}")

    df["iso3"] = df["iso3"].astype(str).str.upper()
    df = df.set_index("iso3")

    missing_ext = [c for c in EXTERNAL_COLS if c not in df.columns]
    if missing_ext:
        raise ValueError(f"Missing required external columns: {missing_ext}")

    # filter to selected only (keep your requested order; drop missing)
    keep = [c for c in selected_iso3_order if c in df.index]
    missing_sel = [c for c in selected_iso3_order if c not in df.index]
    if missing_sel:
        # not fatal, but tell you
        print("[WARN] These selected ISO3 codes were not found in the CSV and will be skipped:", missing_sel)

    X = df.loc[keep, EXTERNAL_COLS].apply(pd.to_numeric, errors="coerce")

    # if any indicator missing for a selected country, drop that country (strict)
    X = X.dropna(how="any")
    if X.shape[0] < 3:
        raise ValueError(
            "After filtering to selected countries, too few rows remain with complete data "
            "(need at least 3). Try allowing missing data or check indicator availability."
        )

    # z-score across selected countries ONLY (so clusters reflect relative profile within this subset)
    Z = zscore_by_column(X)

    plot_clustered_heatmap_selected(
        Z,
        title=FIGTITLE,
        outbase=OUTDIR / "Cluster_selected_external_indicators",
        p1_iso3=p1_iso3,
        p2_iso3=p2_iso3,
    )

    print("[OK] Done. Output in:", OUTDIR.resolve())
    print(" - Cluster_selected_external_indicators.(png/pdf)")


if __name__ == "__main__":
    main()



# %% Cell 2
# cluster_selected_countries_elbow_then_dendrogram.py
# ------------------------------------------------------------
# 1) Select only your countries (union of Part1 + Part2)
# 2) Z-score indicators across selected countries
# 3) Ward hierarchical clustering (Euclidean)
# 4) Choose k using an elbow (SSE) curve + knee heuristic
# 5) Plot elbow + dendrogram + heatmap (colored by chosen k)
#
# INPUT:
#   results_part2_external_hierarchy/analysis_table_part2_with_external_indicators.csv
#   must contain: iso3 + EXTERNAL_COLS
#
# OUTPUT (OUTDIR):
#   - Elbow_selected_external_indicators.(png/pdf)
#   - Cluster_selected_external_indicators.(png/pdf)
#
# Dependencies:
#   pip install numpy pandas matplotlib scipy
# Optional:
#   pip install pycountry
# ------------------------------------------------------------

from pathlib import Path
from typing import Optional, List, Set, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist

try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# ----------------------------
# USER SETTINGS
# ----------------------------
DATA = Path("results_part2_external_hierarchy/analysis_table_part2_with_external_indicators.csv")
OUTDIR = Path("figures_country_clustering")
DPI_EXPORT = 600

COUNTRIES_PART1 = ["Afghanistan", "Madagascar", "Malawi", "Morocco"]
COUNTRIES_PART2 = ["Ethiopia", "Niger", "Morocco", "Senegal", "Benin"]

EXTERNAL_COLS = [
    "gii_latest",
    "gdi_latest",
    "hdi_latest",
    "electoral-democracy-index_latest",
    "liberal-democracy-index_latest",
    "women-political-empowerment-index_latest",
]

PRETTY = {
    "gii_latest": "GII",
    "gdi_latest": "GDI",
    "hdi_latest": "HDI",
    "electoral-democracy-index_latest": "Electoral democracy",
    "liberal-democracy-index_latest": "Liberal democracy",
    "women-political-empowerment-index_latest": "Women political empowerment",
}

# Clustering choices
METRIC = "euclidean"        # elbow SSE assumes euclidean geometry
LINKAGE_METHOD = "ward"     # Ward requires Euclidean distances

FIGTITLE = "Selected countries: clustering by external hierarchy / gender indicators"

# Elbow search
K_MAX = 6  # with 8 countries, 2..6 is usually enough; will be capped automatically


# ----------------------------
# Plot style
# ----------------------------
def set_pub_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(fig, out_png: Path, out_pdf: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# ISO3 helpers
# ----------------------------
def iso3_to_name(iso3: str) -> str:
    iso3 = str(iso3).upper()
    if not HAS_PYCOUNTRY:
        return iso3
    try:
        c = pycountry.countries.get(alpha_3=iso3)
        return c.name if c else iso3
    except Exception:
        return iso3


def name_to_iso3(name: str) -> Optional[str]:
    if not HAS_PYCOUNTRY:
        return None

    overrides: Dict[str, str] = {
        "Morocco": "MAR",
        "Malawi": "MWI",
        "Madagascar": "MDG",
        "Afghanistan": "AFG",
        "Ethiopia": "ETH",
        "Niger": "NER",
        "Senegal": "SEN",
        "Benin": "BEN",
    }
    if name in overrides:
        return overrides[name]

    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None


def build_selected_iso3() -> Tuple[Set[str], Set[str], List[str]]:
    """
    Returns:
      p1_iso3, p2_iso3, union list in deterministic order
    """
    p1_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART1}
    p2_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART2}
    p1_iso3 = {x for x in p1_iso3 if x is not None}
    p2_iso3 = {x for x in p2_iso3 if x is not None}

    union: List[str] = []
    for n in COUNTRIES_PART1:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    for n in COUNTRIES_PART2:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)

    return p1_iso3, p2_iso3, union


def mark_groups(iso3: str, p1_iso3: Set[str], p2_iso3: Set[str]) -> str:
    iso3 = str(iso3).upper()
    tag = ""
    if iso3 in p1_iso3:
        tag += " [P1]"
    if iso3 in p2_iso3:
        tag += " [P2]" if tag == "" else "+P2"
    return tag


# ----------------------------
# Data transforms
# ----------------------------
def zscore_by_column(X: pd.DataFrame) -> pd.DataFrame:
    M = X.to_numpy(float)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (M - mu) / sd
    return pd.DataFrame(Z, index=X.index, columns=X.columns)


def compute_linkage(Z: pd.DataFrame) -> np.ndarray:
    D = pdist(Z.to_numpy(float), metric=METRIC)
    L = linkage(D, method=LINKAGE_METHOD)
    return L


# ----------------------------
# Elbow (SSE) for hierarchical clustering
# ----------------------------
def within_cluster_sse(X: np.ndarray, labels: np.ndarray) -> float:
    sse = 0.0
    for lab in np.unique(labels):
        pts = X[labels == lab]
        if pts.shape[0] <= 1:
            continue
        centroid = pts.mean(axis=0, keepdims=True)
        dif = pts - centroid
        sse += float(np.sum(dif * dif))
    return sse


def elbow_curve_from_linkage(Z: pd.DataFrame, L: np.ndarray, k_max: int) -> Tuple[np.ndarray, np.ndarray]:
    X = Z.to_numpy(float)
    n = X.shape[0]
    k_max = int(min(k_max, n - 1))  # cannot have k=n with maxclust in a meaningful elbow
    ks = np.arange(1, k_max + 1, dtype=int)
    sses = []
    for k in ks:
        if k == 1:
            labels = np.ones(n, dtype=int)
        else:
            labels = fcluster(L, t=k, criterion="maxclust")
        sses.append(within_cluster_sse(X, labels))
    return ks, np.array(sses, dtype=float)


def choose_k_knee(ks: np.ndarray, sses: np.ndarray) -> int:
    """
    Knee heuristic: choose point with maximum perpendicular distance
    to the straight line connecting (k1, sse1) and (k_last, sse_last).
    """
    if len(ks) < 3:
        return int(ks[-1])

    x = ks.astype(float)
    y = sses.astype(float)

    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    # If flat or degenerate
    if (x1 == x0) or (y1 == y0):
        return int(ks[1] if len(ks) > 1 else ks[0])

    # distance from each point to line
    # line in ax + by + c = 0 form
    a = (y1 - y0)
    b = -(x1 - x0)
    c = (x1 - x0) * y0 - (y1 - y0) * x0

    dist = np.abs(a * x + b * y + c) / np.sqrt(a * a + b * b)

    # avoid choosing k=1 as “elbow” (usually not informative)
    idx = int(np.argmax(dist[1:]) + 1)
    return int(ks[idx])


def plot_elbow(ks: np.ndarray, sses: np.ndarray, k_star: int, outbase: Path):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)
    ax.plot(ks, sses, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Within-cluster SSE (Ward / Euclidean)")
    ax.set_title("Elbow curve (selected countries)")
    ax.axvline(k_star, linestyle="--")
    ax.text(k_star + 0.05, np.min(sses), f"k* = {k_star}", va="bottom")
    ax.set_xticks(ks)
    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


# ----------------------------
# Dendrogram coloring: choose distance threshold that yields k clusters
# ----------------------------
def color_threshold_for_k(L: np.ndarray, k: int) -> float:
    """
    For hierarchical clustering, a color_threshold defines where branches
    are colored. This returns a threshold that produces ~k clusters.
    """
    n = L.shape[0] + 1
    if k <= 1:
        return np.inf
    # index between merges to get k clusters:
    idx = max(0, n - k - 1)
    return float(L[idx, 2] + 1e-12)


# ----------------------------
# Plot: dendrogram + heatmap
# ----------------------------
def plot_clustered_heatmap_selected(
    Z: pd.DataFrame,
    L: np.ndarray,
    k_star: int,
    title: str,
    outbase: Path,
    p1_iso3: Set[str],
    p2_iso3: Set[str],
):
    set_pub_style()

    # order leaves from dendrogram
    dd = dendrogram(L, no_plot=True)
    leaf_order = dd["leaves"]
    Zr = Z.iloc[leaf_order].copy()

    fig = plt.figure(figsize=(12.2, max(4.2, 0.65 * len(Zr) + 2.0)), dpi=300)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.15, 4.0], wspace=0.05)

    # dendrogram (colored by k*)
    ax_d = fig.add_subplot(gs[0, 0])
    thr = color_threshold_for_k(L, k_star)
    dendrogram(
        L,
        orientation="left",
        no_labels=True,
        ax=ax_d,
        color_threshold=thr,
        above_threshold_color="black",
    )
    ax_d.invert_yaxis()
    ax_d.axis("off")

    # heatmap
    ax_h = fig.add_subplot(gs[0, 1])
    im = ax_h.imshow(Zr.to_numpy(float), aspect="auto")

    ylabels: List[str] = []
    for iso3 in Zr.index.astype(str).tolist():
        base = iso3_to_name(iso3) if HAS_PYCOUNTRY else str(iso3).upper()
        ylabels.append(base + mark_groups(iso3, p1_iso3, p2_iso3))

    ax_h.set_yticks(np.arange(len(Zr)))
    ax_h.set_yticklabels(ylabels)

    ax_h.set_xticks(np.arange(Zr.shape[1]))
    ax_h.set_xticklabels([PRETTY.get(c, c) for c in Zr.columns], rotation=30, ha="right")

    ax_h.set_title(f"{title}\n(k* from elbow = {k_star})")

    # annotate z-scores
    for i in range(Zr.shape[0]):
        for j in range(Zr.shape[1]):
            v = Zr.iat[i, j]
            if np.isfinite(v):
                ax_h.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax_h, fraction=0.035, pad=0.02)
    cbar.set_label("z-score (computed across the selected countries)")

    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


# ----------------------------
# Main
# ----------------------------
def main():
    if METRIC != "euclidean" or LINKAGE_METHOD != "ward":
        raise ValueError("This elbow SSE implementation is intended for METRIC='euclidean' and LINKAGE_METHOD='ward'.")

    if not DATA.exists():
        raise FileNotFoundError(f"Cannot find: {DATA.resolve()}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # selected countries (iso3)
    p1_iso3, p2_iso3, selected_iso3_order = build_selected_iso3()
    if len(selected_iso3_order) < 3:
        raise ValueError(
            "Too few selected countries after name->ISO3 mapping. "
            "Install pycountry or check country name spelling."
        )

    df = pd.read_csv(DATA)
    if "iso3" not in df.columns:
        raise ValueError(f"Your CSV must contain 'iso3'. Columns found: {list(df.columns)}")

    df["iso3"] = df["iso3"].astype(str).str.upper()
    df = df.set_index("iso3")

    missing_ext = [c for c in EXTERNAL_COLS if c not in df.columns]
    if missing_ext:
        raise ValueError(f"Missing required external columns: {missing_ext}")

    # filter to selected only (keep requested order; drop missing)
    keep = [c for c in selected_iso3_order if c in df.index]
    missing_sel = [c for c in selected_iso3_order if c not in df.index]
    if missing_sel:
        print("[WARN] These selected ISO3 codes were not found in the CSV and will be skipped:", missing_sel)

    X = df.loc[keep, EXTERNAL_COLS].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(how="any")

    if X.shape[0] < 3:
        raise ValueError("Too few selected countries with complete data after dropna(how='any').")

    # z-score across selected countries
    Z = zscore_by_column(X)

    # linkage
    L = compute_linkage(Z)

    # elbow + knee selection for k*
    ks, sses = elbow_curve_from_linkage(Z, L, k_max=K_MAX)
    k_star = choose_k_knee(ks, sses)

    # save elbow plot
    plot_elbow(
        ks, sses, k_star,
        outbase=OUTDIR / "Elbow_selected_external_indicators"
    )

    # save dendrogram+heatmap colored by k*
    plot_clustered_heatmap_selected(
        Z=Z,
        L=L,
        k_star=k_star,
        title=FIGTITLE,
        outbase=OUTDIR / "Cluster_selected_external_indicators",
        p1_iso3=p1_iso3,
        p2_iso3=p2_iso3,
    )

    print("[OK] Done. Outputs in:", OUTDIR.resolve())
    print(" - Elbow_selected_external_indicators.(png/pdf)")
    print(" - Cluster_selected_external_indicators.(png/pdf)")
    print(f"[INFO] Selected k* = {k_star} from elbow knee heuristic")


if __name__ == "__main__":
    main()



# %% Cell 3
# cluster_selected_countries_elbow_then_dendrogram_WITH_CIRCULAR.py
# ------------------------------------------------------------
# Adds a circular (radial) dendrogram plot in addition to:
# - elbow plot
# - dendrogram + heatmap
# ------------------------------------------------------------

from pathlib import Path
from typing import Optional, List, Set, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# ----------------------------
# USER SETTINGS
# ----------------------------
DATA = Path("results_part2_external_hierarchy/analysis_table_part2_with_external_indicators.csv")
OUTDIR = Path("figures_country_clustering")
DPI_EXPORT = 600

COUNTRIES_PART1 = ["Afghanistan", "Madagascar", "Malawi", "Morocco"]
COUNTRIES_PART2 = ["Ethiopia", "Niger", "Morocco", "Senegal", "Benin"]

EXTERNAL_COLS = [
    "gii_latest",
    "gdi_latest",
    "hdi_latest",
    "electoral-democracy-index_latest",
    "liberal-democracy-index_latest",
    "women-political-empowerment-index_latest",
]

PRETTY = {
    "gii_latest": "GII",
    "gdi_latest": "GDI",
    "hdi_latest": "HDI",
    "electoral-democracy-index_latest": "Electoral democracy",
    "liberal-democracy-index_latest": "Liberal democracy",
    "women-political-empowerment-index_latest": "Women political empowerment",
}

# Clustering choices
METRIC = "euclidean"
LINKAGE_METHOD = "ward"

FIGTITLE = "Selected countries: clustering by external hierarchy / gender indicators"

# Elbow search
K_MAX = 6


# ----------------------------
# Plot style
# ----------------------------
def set_pub_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(fig, out_png: Path, out_pdf: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# ISO3 helpers
# ----------------------------
def iso3_to_name(iso3: str) -> str:
    iso3 = str(iso3).upper()
    if not HAS_PYCOUNTRY:
        return iso3
    try:
        c = pycountry.countries.get(alpha_3=iso3)
        return c.name if c else iso3
    except Exception:
        return iso3


def name_to_iso3(name: str) -> Optional[str]:
    if not HAS_PYCOUNTRY:
        return None

    overrides: Dict[str, str] = {
        "Morocco": "MAR",
        "Malawi": "MWI",
        "Madagascar": "MDG",
        "Afghanistan": "AFG",
        "Ethiopia": "ETH",
        "Niger": "NER",
        "Senegal": "SEN",
        "Benin": "BEN",
    }
    if name in overrides:
        return overrides[name]

    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None


def build_selected_iso3() -> Tuple[Set[str], Set[str], List[str]]:
    p1_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART1}
    p2_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART2}
    p1_iso3 = {x for x in p1_iso3 if x is not None}
    p2_iso3 = {x for x in p2_iso3 if x is not None}

    union: List[str] = []
    for n in COUNTRIES_PART1:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    for n in COUNTRIES_PART2:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)

    return p1_iso3, p2_iso3, union


def mark_groups(iso3: str, p1_iso3: Set[str], p2_iso3: Set[str]) -> str:
    iso3 = str(iso3).upper()
    tag = ""
    if iso3 in p1_iso3:
        tag += " [P1]"
    if iso3 in p2_iso3:
        tag += " [P2]" if tag == "" else "+P2"
    return tag


# ----------------------------
# Data transforms
# ----------------------------
def zscore_by_column(X: pd.DataFrame) -> pd.DataFrame:
    M = X.to_numpy(float)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (M - mu) / sd
    return pd.DataFrame(Z, index=X.index, columns=X.columns)


def compute_linkage(Z: pd.DataFrame) -> np.ndarray:
    D = pdist(Z.to_numpy(float), metric=METRIC)
    return linkage(D, method=LINKAGE_METHOD)


# ----------------------------
# Elbow (SSE)
# ----------------------------
def within_cluster_sse(X: np.ndarray, labels: np.ndarray) -> float:
    sse = 0.0
    for lab in np.unique(labels):
        pts = X[labels == lab]
        if pts.shape[0] <= 1:
            continue
        centroid = pts.mean(axis=0, keepdims=True)
        dif = pts - centroid
        sse += float(np.sum(dif * dif))
    return sse


def elbow_curve_from_linkage(Z: pd.DataFrame, L: np.ndarray, k_max: int) -> Tuple[np.ndarray, np.ndarray]:
    X = Z.to_numpy(float)
    n = X.shape[0]
    k_max = int(min(k_max, n - 1))
    ks = np.arange(1, k_max + 1, dtype=int)
    sses = []
    for k in ks:
        labels = np.ones(n, dtype=int) if k == 1 else fcluster(L, t=k, criterion="maxclust")
        sses.append(within_cluster_sse(X, labels))
    return ks, np.array(sses, dtype=float)


def choose_k_knee(ks: np.ndarray, sses: np.ndarray) -> int:
    if len(ks) < 3:
        return int(ks[-1])

    x = ks.astype(float)
    y = sses.astype(float)
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    if (x1 == x0) or (y1 == y0):
        return int(ks[1] if len(ks) > 1 else ks[0])

    a = (y1 - y0)
    b = -(x1 - x0)
    c = (x1 - x0) * y0 - (y1 - y0) * x0
    dist = np.abs(a * x + b * y + c) / np.sqrt(a * a + b * b)

    idx = int(np.argmax(dist[1:]) + 1)  # avoid k=1
    return int(ks[idx])


def plot_elbow(ks: np.ndarray, sses: np.ndarray, k_star: int, outbase: Path):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)
    ax.plot(ks, sses, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Within-cluster SSE (Ward / Euclidean)")
    ax.set_title("Elbow curve (selected countries)")
    ax.axvline(k_star, linestyle="--")
    ax.text(k_star + 0.05, np.min(sses), f"k* = {k_star}", va="bottom")
    ax.set_xticks(ks)
    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


# ----------------------------
# Dendrogram coloring threshold for k
# ----------------------------
def color_threshold_for_k(L: np.ndarray, k: int) -> float:
    n = L.shape[0] + 1
    if k <= 1:
        return np.inf
    idx = max(0, n - k - 1)
    return float(L[idx, 2] + 1e-12)


# ----------------------------
# Rectangular dendrogram + heatmap
# ----------------------------
def plot_clustered_heatmap_selected(
    Z: pd.DataFrame,
    L: np.ndarray,
    k_star: int,
    title: str,
    outbase: Path,
    p1_iso3: Set[str],
    p2_iso3: Set[str],
):
    set_pub_style()

    dd = dendrogram(L, no_plot=True)
    leaf_order = dd["leaves"]
    Zr = Z.iloc[leaf_order].copy()

    fig = plt.figure(figsize=(12.2, max(4.2, 0.65 * len(Zr) + 2.0)), dpi=300)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.15, 4.0], wspace=0.05)

    ax_d = fig.add_subplot(gs[0, 0])
    thr = color_threshold_for_k(L, k_star)
    dendrogram(
        L,
        orientation="left",
        no_labels=True,
        ax=ax_d,
        color_threshold=thr,
        above_threshold_color="black",
    )
    ax_d.invert_yaxis()
    ax_d.axis("off")

    ax_h = fig.add_subplot(gs[0, 1])
    im = ax_h.imshow(Zr.to_numpy(float), aspect="auto")

    ylabels: List[str] = []
    for iso3 in Zr.index.astype(str).tolist():
        base = iso3_to_name(iso3) if HAS_PYCOUNTRY else str(iso3).upper()
        ylabels.append(base + mark_groups(iso3, p1_iso3, p2_iso3))

    ax_h.set_yticks(np.arange(len(Zr)))
    ax_h.set_yticklabels(ylabels)

    ax_h.set_xticks(np.arange(Zr.shape[1]))
    ax_h.set_xticklabels([PRETTY.get(c, c) for c in Zr.columns], rotation=30, ha="right")

    ax_h.set_title(f"{title}\n(k* from elbow = {k_star})")

    for i in range(Zr.shape[0]):
        for j in range(Zr.shape[1]):
            v = Zr.iat[i, j]
            if np.isfinite(v):
                ax_h.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax_h, fraction=0.035, pad=0.02)
    cbar.set_label("z-score (computed across the selected countries)")

    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


# ----------------------------
# Circular dendrogram (radial)
# ----------------------------
def plot_circular_dendrogram(
    Z: pd.DataFrame,
    L: np.ndarray,
    k_star: int,
    title: str,
    outbase: Path,
    p1_iso3: Set[str],
    p2_iso3: Set[str],
):
    """
    Renders a radial dendrogram using SciPy dendrogram coordinates.

    Notes:
    - We map SciPy's x-coordinates (leaf positions) to angle (theta).
    - We map heights (cluster distance) to radius, with leaves at outer radius.
    - Colors are taken from SciPy's dendrogram color_list using color_threshold
      set to produce ~k_star clusters.
    """
    set_pub_style()

    thr = color_threshold_for_k(L, k_star)

    # Get dendrogram geometry without plotting
    dd = dendrogram(
        L,
        no_plot=True,
        color_threshold=thr,
        above_threshold_color="black",
    )

    # Leaf order and labels (in dendrogram leaf x-space)
    leaf_order = dd["leaves"]
    leaves = Z.index.astype(str).tolist()
    ordered_iso3 = [leaves[i] for i in leaf_order]

    # SciPy uses leaf x positions at 5, 15, 25, ... (step=10)
    # We'll build a mapping from those x positions to theta.
    n = len(ordered_iso3)
    x_leaf = np.arange(n) * 10.0 + 5.0  # expected leaf x positions
    x_min, x_max = x_leaf.min(), x_leaf.max()

    def x_to_theta(x: float) -> float:
        # map to [0, 2pi)
        if x_max == x_min:
            return 0.0
        return 2.0 * np.pi * (x - x_min) / (x_max - x_min)

    # Radius mapping: dendrogram "height" increases upward; for radial,
    # place leaves at r=R and root near r=inner.
    max_h = float(np.max(L[:, 2])) if L.size else 1.0
    R_outer = 1.0
    R_inner = 0.10

    def h_to_r(h: float) -> float:
        # leaves (h~0) at outer; root (h=max_h) at inner
        if max_h <= 0:
            return R_outer
        return R_outer - (R_outer - R_inner) * (h / max_h)

    # Create polar plot
    fig = plt.figure(figsize=(9.0, 9.0), dpi=300)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_axis_off()

    # Draw each U-link in polar coordinates
    for xs, ys, col in zip(dd["icoord"], dd["dcoord"], dd["color_list"]):
        # icoord has 4 x's, dcoord has 4 y's describing a "∩" polyline:
        # (x0,y0)->(x1,y1)->(x2,y2)->(x3,y3)
        thetas = [x_to_theta(x) for x in xs]
        rs = [h_to_r(y) for y in ys]
        ax.plot(thetas, rs, linewidth=1.6, color=col)

    # Add leaf labels around the circle
    for i, iso3 in enumerate(ordered_iso3):
        theta = x_to_theta(x_leaf[i])
        label = iso3_to_name(iso3) if HAS_PYCOUNTRY else iso3.upper()
        label += mark_groups(iso3, p1_iso3, p2_iso3)

        # Rotate text to be readable outward
        ang_deg = np.degrees(theta)
        rotation = ang_deg - 90.0
        ha = "left"
        if 90.0 < ang_deg < 270.0:
            rotation += 180.0
            ha = "right"

        ax.text(
            theta,
            R_outer + 0.05,
            label,
            rotation=rotation,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=10,
        )

    ax.set_title(f"{title}\nCircular dendrogram (k* from elbow = {k_star})", pad=30)

    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


# ----------------------------
# Main
# ----------------------------
def main():
    if METRIC != "euclidean" or LINKAGE_METHOD != "ward":
        raise ValueError("This elbow SSE implementation is intended for METRIC='euclidean' and LINKAGE_METHOD='ward'.")

    if not DATA.exists():
        raise FileNotFoundError(f"Cannot find: {DATA.resolve()}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    p1_iso3, p2_iso3, selected_iso3_order = build_selected_iso3()
    if len(selected_iso3_order) < 3:
        raise ValueError(
            "Too few selected countries after name->ISO3 mapping. "
            "Install pycountry or check country name spelling."
        )

    df = pd.read_csv(DATA)
    if "iso3" not in df.columns:
        raise ValueError(f"Your CSV must contain 'iso3'. Columns found: {list(df.columns)}")

    df["iso3"] = df["iso3"].astype(str).str.upper()
    df = df.set_index("iso3")

    missing_ext = [c for c in EXTERNAL_COLS if c not in df.columns]
    if missing_ext:
        raise ValueError(f"Missing required external columns: {missing_ext}")

    keep = [c for c in selected_iso3_order if c in df.index]
    missing_sel = [c for c in selected_iso3_order if c not in df.index]
    if missing_sel:
        print("[WARN] These selected ISO3 codes were not found in the CSV and will be skipped:", missing_sel)

    X = df.loc[keep, EXTERNAL_COLS].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if X.shape[0] < 3:
        raise ValueError("Too few selected countries with complete data after dropna(how='any').")

    Z = zscore_by_column(X)
    L = compute_linkage(Z)

    ks, sses = elbow_curve_from_linkage(Z, L, k_max=K_MAX)
    k_star = choose_k_knee(ks, sses)

    plot_elbow(ks, sses, k_star, outbase=OUTDIR / "Elbow_selected_external_indicators")

    plot_clustered_heatmap_selected(
        Z=Z,
        L=L,
        k_star=k_star,
        title=FIGTITLE,
        outbase=OUTDIR / "Cluster_selected_external_indicators",
        p1_iso3=p1_iso3,
        p2_iso3=p2_iso3,
    )

    # NEW: circular dendrogram
    plot_circular_dendrogram(
        Z=Z,
        L=L,
        k_star=k_star,
        title=FIGTITLE,
        outbase=OUTDIR / "Circular_dendrogram_selected_external_indicators",
        p1_iso3=p1_iso3,
        p2_iso3=p2_iso3,
    )

    print("[OK] Done. Outputs in:", OUTDIR.resolve())
    print(" - Elbow_selected_external_indicators.(png/pdf)")
    print(" - Cluster_selected_external_indicators.(png/pdf)")
    print(" - Circular_dendrogram_selected_external_indicators.(png/pdf)")
    print(f"[INFO] Selected k* = {k_star} from elbow knee heuristic")


if __name__ == "__main__":
    main()



# %% Cell 4
# cluster_selected_countries_elbow_then_dendrogram_WITH_CIRCULAR.py
# ------------------------------------------------------------
# Adds silhouette plots (curve + detailed silhouette plot) in addition to:
# - elbow plot
# - dendrogram + heatmap
# - circular dendrogram
#
# NEW OUTPUTS (OUTDIR):
# - Silhouette_curve_selected_external_indicators.(png/pdf)
# - Silhouette_plot_selected_external_indicators.(png/pdf)
#
# Extra dependency:
#   pip install scikit-learn
# ------------------------------------------------------------

from pathlib import Path
from typing import Optional, List, Set, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

# NEW: silhouette
from sklearn.metrics import silhouette_score, silhouette_samples  # pip install scikit-learn

try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# ----------------------------
# USER SETTINGS
# ----------------------------
DATA = Path("results_part2_external_hierarchy/analysis_table_part2_with_external_indicators.csv")
OUTDIR = Path("figures_country_clustering")
DPI_EXPORT = 600

COUNTRIES_PART1 = ["Afghanistan", "Madagascar", "Malawi", "Morocco"]
COUNTRIES_PART2 = ["Ethiopia", "Niger", "Morocco", "Senegal", "Benin"]

EXTERNAL_COLS = [
    "gii_latest",
    "gdi_latest",
    "hdi_latest",
    "electoral-democracy-index_latest",
    "liberal-democracy-index_latest",
    "women-political-empowerment-index_latest",
]

PRETTY = {
    "gii_latest": "GII",
    "gdi_latest": "GDI",
    "hdi_latest": "HDI",
    "electoral-democracy-index_latest": "Electoral democracy",
    "liberal-democracy-index_latest": "Liberal democracy",
    "women-political-empowerment-index_latest": "Women political empowerment",
}

# Clustering choices
METRIC = "euclidean"
LINKAGE_METHOD = "ward"

FIGTITLE = "Selected countries: clustering by external hierarchy / gender indicators"

# Elbow search
K_MAX = 6


# ----------------------------
# Plot style
# ----------------------------
def set_pub_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(fig, out_png: Path, out_pdf: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# ISO3 helpers
# ----------------------------
def iso3_to_name(iso3: str) -> str:
    iso3 = str(iso3).upper()
    if not HAS_PYCOUNTRY:
        return iso3
    try:
        c = pycountry.countries.get(alpha_3=iso3)
        return c.name if c else iso3
    except Exception:
        return iso3


def name_to_iso3(name: str) -> Optional[str]:
    if not HAS_PYCOUNTRY:
        return None

    overrides: Dict[str, str] = {
        "Morocco": "MAR",
        "Malawi": "MWI",
        "Madagascar": "MDG",
        "Afghanistan": "AFG",
        "Ethiopia": "ETH",
        "Niger": "NER",
        "Senegal": "SEN",
        "Benin": "BEN",
    }
    if name in overrides:
        return overrides[name]

    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None


def build_selected_iso3() -> Tuple[Set[str], Set[str], List[str]]:
    p1_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART1}
    p2_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART2}
    p1_iso3 = {x for x in p1_iso3 if x is not None}
    p2_iso3 = {x for x in p2_iso3 if x is not None}

    union: List[str] = []
    for n in COUNTRIES_PART1:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    for n in COUNTRIES_PART2:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)

    return p1_iso3, p2_iso3, union


def mark_groups(iso3: str, p1_iso3: Set[str], p2_iso3: Set[str]) -> str:
    iso3 = str(iso3).upper()
    tag = ""
    if iso3 in p1_iso3:
        tag += " [P1]"
    if iso3 in p2_iso3:
        tag += " [P2]" if tag == "" else "+P2"
    return tag


# ----------------------------
# Data transforms
# ----------------------------
def zscore_by_column(X: pd.DataFrame) -> pd.DataFrame:
    M = X.to_numpy(float)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (M - mu) / sd
    return pd.DataFrame(Z, index=X.index, columns=X.columns)


def compute_linkage(Z: pd.DataFrame) -> np.ndarray:
    D = pdist(Z.to_numpy(float), metric=METRIC)
    return linkage(D, method=LINKAGE_METHOD)


# ----------------------------
# Elbow (SSE)
# ----------------------------
def within_cluster_sse(X: np.ndarray, labels: np.ndarray) -> float:
    sse = 0.0
    for lab in np.unique(labels):
        pts = X[labels == lab]
        if pts.shape[0] <= 1:
            continue
        centroid = pts.mean(axis=0, keepdims=True)
        dif = pts - centroid
        sse += float(np.sum(dif * dif))
    return sse


def elbow_curve_from_linkage(Z: pd.DataFrame, L: np.ndarray, k_max: int) -> Tuple[np.ndarray, np.ndarray]:
    X = Z.to_numpy(float)
    n = X.shape[0]
    k_max = int(min(k_max, n - 1))
    ks = np.arange(1, k_max + 1, dtype=int)
    sses = []
    for k in ks:
        labels = np.ones(n, dtype=int) if k == 1 else fcluster(L, t=k, criterion="maxclust")
        sses.append(within_cluster_sse(X, labels))
    return ks, np.array(sses, dtype=float)


def choose_k_knee(ks: np.ndarray, sses: np.ndarray) -> int:
    if len(ks) < 3:
        return int(ks[-1])

    x = ks.astype(float)
    y = sses.astype(float)
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    if (x1 == x0) or (y1 == y0):
        return int(ks[1] if len(ks) > 1 else ks[0])

    a = (y1 - y0)
    b = -(x1 - x0)
    c = (x1 - x0) * y0 - (y1 - y0) * x0
    dist = np.abs(a * x + b * y + c) / np.sqrt(a * a + b * b)

    idx = int(np.argmax(dist[1:]) + 1)  # avoid k=1
    return int(ks[idx])


def plot_elbow(ks: np.ndarray, sses: np.ndarray, k_star: int, outbase: Path):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)
    ax.plot(ks, sses, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Within-cluster SSE (Ward / Euclidean)")
    ax.set_title("Elbow curve (selected countries)")
    ax.axvline(k_star, linestyle="--")
    ax.text(k_star + 0.05, float(np.min(sses)), f"k* = {k_star}", va="bottom")
    ax.set_xticks(ks)
    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


# ----------------------------
# Silhouette (NEW)
# ----------------------------
def silhouette_curve(Z: pd.DataFrame, L: np.ndarray, k_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Silhouette score is defined for k>=2.
    Uses Euclidean silhouette on the standardized feature space Z.
    """
    X = Z.to_numpy(float)
    n = X.shape[0]
    k_max = int(min(k_max, n - 1))
    if k_max < 2:
        return np.array([], dtype=int), np.array([], dtype=float)

    ks = np.arange(2, k_max + 1, dtype=int)
    scores = []
    for k in ks:
        labels = fcluster(L, t=k, criterion="maxclust")
        # If a cluster has 1 sample, silhouette is still defined overall, but can be unstable.
        # sklearn will error only if there is 1 cluster or k==n; we avoid those.
        scores.append(float(silhouette_score(X, labels, metric="euclidean")))
    return ks, np.array(scores, dtype=float)


def plot_silhouette_curve(ks: np.ndarray, scores: np.ndarray, k_star: int, outbase: Path):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)
    ax.plot(ks, scores, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean silhouette score (Euclidean)")
    ax.set_title("Silhouette curve (selected countries)")
    if k_star >= 2:
        ax.axvline(k_star, linestyle="--")
        ax.text(k_star + 0.05, float(np.min(scores)), f"k* = {k_star}", va="bottom")
    ax.set_xticks(ks)
    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


def plot_silhouette_detail(
    Z: pd.DataFrame,
    L: np.ndarray,
    k: int,
    title: str,
    outbase: Path,
):
    """
    Standard silhouette plot: per-sample silhouette values, grouped by cluster.
    """
    if k < 2:
        return

    set_pub_style()
    X = Z.to_numpy(float)
    labels = fcluster(L, t=k, criterion="maxclust")
    sil_vals = silhouette_samples(X, labels, metric="euclidean")
    sil_avg = float(np.mean(sil_vals))

    fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=300)

    y_lower = 10
    for lab in np.unique(labels):
        vals = np.sort(sil_vals[labels == lab])
        size = vals.size
        y_upper = y_lower + size

        # Fill between 0 and silhouette values (keep default matplotlib colors)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.85)

        ax.text(-0.05, y_lower + 0.5 * size, f"Cluster {int(lab)}", va="center")
        y_lower = y_upper + 10  # gap between clusters

    ax.axvline(sil_avg, linestyle="--")
    ax.set_title(f"{title}\nSilhouette plot (k={k}, mean={sil_avg:.3f})")
    ax.set_xlabel("Silhouette value")
    ax.set_ylabel("Samples (grouped by cluster)")
    ax.set_yticks([])
    ax.set_xlim(-0.2, 1.0)  # typical silhouette range
    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


# ----------------------------
# Dendrogram coloring threshold for k
# ----------------------------
def color_threshold_for_k(L: np.ndarray, k: int) -> float:
    n = L.shape[0] + 1
    if k <= 1:
        return np.inf
    idx = max(0, n - k - 1)
    return float(L[idx, 2] + 1e-12)


# ----------------------------
# Rectangular dendrogram + heatmap
# ----------------------------
def plot_clustered_heatmap_selected(
    Z: pd.DataFrame,
    L: np.ndarray,
    k_star: int,
    title: str,
    outbase: Path,
    p1_iso3: Set[str],
    p2_iso3: Set[str],
):
    set_pub_style()

    dd = dendrogram(L, no_plot=True)
    leaf_order = dd["leaves"]
    Zr = Z.iloc[leaf_order].copy()

    fig = plt.figure(figsize=(12.2, max(4.2, 0.65 * len(Zr) + 2.0)), dpi=300)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.15, 4.0], wspace=0.05)

    ax_d = fig.add_subplot(gs[0, 0])
    thr = color_threshold_for_k(L, k_star)
    dendrogram(
        L,
        orientation="left",
        no_labels=True,
        ax=ax_d,
        color_threshold=thr,
        above_threshold_color="black",
    )
    ax_d.invert_yaxis()
    ax_d.axis("off")

    ax_h = fig.add_subplot(gs[0, 1])
    im = ax_h.imshow(Zr.to_numpy(float), aspect="auto")

    ylabels: List[str] = []
    for iso3 in Zr.index.astype(str).tolist():
        base = iso3_to_name(iso3) if HAS_PYCOUNTRY else str(iso3).upper()
        ylabels.append(base + mark_groups(iso3, p1_iso3, p2_iso3))

    ax_h.set_yticks(np.arange(len(Zr)))
    ax_h.set_yticklabels(ylabels)

    ax_h.set_xticks(np.arange(Zr.shape[1]))
    ax_h.set_xticklabels([PRETTY.get(c, c) for c in Zr.columns], rotation=30, ha="right")

    ax_h.set_title(f"{title}\n(k* from elbow = {k_star})")

    for i in range(Zr.shape[0]):
        for j in range(Zr.shape[1]):
            v = Zr.iat[i, j]
            if np.isfinite(v):
                ax_h.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax_h, fraction=0.035, pad=0.02)
    cbar.set_label("z-score (computed across the selected countries)")

    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


# ----------------------------
# Circular dendrogram (radial)
# ----------------------------
def plot_circular_dendrogram(
    Z: pd.DataFrame,
    L: np.ndarray,
    k_star: int,
    title: str,
    outbase: Path,
    p1_iso3: Set[str],
    p2_iso3: Set[str],
):
    set_pub_style()

    thr = color_threshold_for_k(L, k_star)

    dd = dendrogram(
        L,
        no_plot=True,
        color_threshold=thr,
        above_threshold_color="black",
    )

    leaf_order = dd["leaves"]
    leaves = Z.index.astype(str).tolist()
    ordered_iso3 = [leaves[i] for i in leaf_order]

    n = len(ordered_iso3)
    x_leaf = np.arange(n) * 10.0 + 5.0
    x_min, x_max = x_leaf.min(), x_leaf.max()

    def x_to_theta(x: float) -> float:
        if x_max == x_min:
            return 0.0
        return 2.0 * np.pi * (x - x_min) / (x_max - x_min)

    max_h = float(np.max(L[:, 2])) if L.size else 1.0
    R_outer = 1.0
    R_inner = 0.10

    def h_to_r(h: float) -> float:
        if max_h <= 0:
            return R_outer
        return R_outer - (R_outer - R_inner) * (h / max_h)

    fig = plt.figure(figsize=(9.0, 9.0), dpi=300)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_axis_off()

    for xs, ys, col in zip(dd["icoord"], dd["dcoord"], dd["color_list"]):
        thetas = [x_to_theta(x) for x in xs]
        rs = [h_to_r(y) for y in ys]
        ax.plot(thetas, rs, linewidth=1.6, color=col)

    for i, iso3 in enumerate(ordered_iso3):
        theta = x_to_theta(x_leaf[i])
        label = iso3_to_name(iso3) if HAS_PYCOUNTRY else iso3.upper()
        label += mark_groups(iso3, p1_iso3, p2_iso3)

        ang_deg = np.degrees(theta)
        rotation = ang_deg - 90.0
        ha = "left"
        if 90.0 < ang_deg < 270.0:
            rotation += 180.0
            ha = "right"

        ax.text(
            theta,
            R_outer + 0.05,
            label,
            rotation=rotation,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=10,
        )

    ax.set_title(f"{title}\nCircular dendrogram (k* from elbow = {k_star})", pad=30)

    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


# ----------------------------
# Main
# ----------------------------
def main():
    if METRIC != "euclidean" or LINKAGE_METHOD != "ward":
        raise ValueError("This elbow SSE implementation is intended for METRIC='euclidean' and LINKAGE_METHOD='ward'.")

    if not DATA.exists():
        raise FileNotFoundError(f"Cannot find: {DATA.resolve()}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    p1_iso3, p2_iso3, selected_iso3_order = build_selected_iso3()
    if len(selected_iso3_order) < 3:
        raise ValueError(
            "Too few selected countries after name->ISO3 mapping. "
            "Install pycountry or check country name spelling."
        )

    df = pd.read_csv(DATA)
    if "iso3" not in df.columns:
        raise ValueError(f"Your CSV must contain 'iso3'. Columns found: {list(df.columns)}")

    df["iso3"] = df["iso3"].astype(str).str.upper()
    df = df.set_index("iso3")

    missing_ext = [c for c in EXTERNAL_COLS if c not in df.columns]
    if missing_ext:
        raise ValueError(f"Missing required external columns: {missing_ext}")

    keep = [c for c in selected_iso3_order if c in df.index]
    missing_sel = [c for c in selected_iso3_order if c not in df.index]
    if missing_sel:
        print("[WARN] These selected ISO3 codes were not found in the CSV and will be skipped:", missing_sel)

    X = df.loc[keep, EXTERNAL_COLS].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if X.shape[0] < 3:
        raise ValueError("Too few selected countries with complete data after dropna(how='any').")

    Z = zscore_by_column(X)
    L = compute_linkage(Z)

    # elbow + knee selection for k*
    ks, sses = elbow_curve_from_linkage(Z, L, k_max=K_MAX)
    k_star = choose_k_knee(ks, sses)

    plot_elbow(ks, sses, k_star, outbase=OUTDIR / "Elbow_selected_external_indicators")

    # NEW: silhouette curve + silhouette detailed plot (using k_star)
    ks_sil, sil_scores = silhouette_curve(Z, L, k_max=K_MAX)
    if ks_sil.size > 0:
        plot_silhouette_curve(
            ks_sil, sil_scores, k_star,
            outbase=OUTDIR / "Silhouette_curve_selected_external_indicators"
        )
        # If k_star is outside silhouette ks range (e.g., k_star=1), pick best k by silhouette for the detailed plot
        if k_star < 2:
            k_for_detail = int(ks_sil[int(np.argmax(sil_scores))])
        else:
            k_for_detail = k_star
        plot_silhouette_detail(
            Z=Z,
            L=L,
            k=k_for_detail,
            title=FIGTITLE,
            outbase=OUTDIR / "Silhouette_plot_selected_external_indicators",
        )
    else:
        print("[WARN] Silhouette not computed (need at least 3 samples and k_max>=2).")

    plot_clustered_heatmap_selected(
        Z=Z,
        L=L,
        k_star=k_star,
        title=FIGTITLE,
        outbase=OUTDIR / "Cluster_selected_external_indicators",
        p1_iso3=p1_iso3,
        p2_iso3=p2_iso3,
    )

    plot_circular_dendrogram(
        Z=Z,
        L=L,
        k_star=k_star,
        title=FIGTITLE,
        outbase=OUTDIR / "Circular_dendrogram_selected_external_indicators",
        p1_iso3=p1_iso3,
        p2_iso3=p2_iso3,
    )

    print("[OK] Done. Outputs in:", OUTDIR.resolve())
    print(" - Elbow_selected_external_indicators.(png/pdf)")
    print(" - Silhouette_curve_selected_external_indicators.(png/pdf)")
    print(" - Silhouette_plot_selected_external_indicators.(png/pdf)")
    print(" - Cluster_selected_external_indicators.(png/pdf)")
    print(" - Circular_dendrogram_selected_external_indicators.(png/pdf)")
    print(f"[INFO] Selected k* = {k_star} from elbow knee heuristic")


if __name__ == "__main__":
    main()



# %% Cell 5
# cluster_selected_countries_elbow_then_dendrogram_WITH_CIRCULAR_AND_SENSITIVITY.py
# --------------------------------------------------------------------------------
# Selected-countries clustering:
#  - Elbow curve (Ward/Euclidean SSE) + knee heuristic
#  - Silhouette curve + detailed silhouette plot (for the chosen k*)
#  - Rectangular dendrogram + heatmap
#  - Circular (radial) dendrogram
#  - NEW ADD-ON: Distance × Linkage sensitivity:
#       * best silhouette over k=2..K_MAX
#       * cophenetic correlation (hierarchical fit quality)
#       * summary CSV + heatmap + bar plot
#
# Outputs (OUTDIR):
#  - Elbow_selected_external_indicators.(png/pdf)
#  - Silhouette_curve_selected_external_indicators.(png/pdf)
#  - Silhouette_plot_selected_external_indicators.(png/pdf)
#  - Cluster_selected_external_indicators.(png/pdf)
#  - Circular_dendrogram_selected_external_indicators.(png/pdf)
#  - Sensitivity_distance_linkage_summary.csv
#  - Sensitivity_distance_linkage_heatmap.(png/pdf)
#  - Sensitivity_cophenetic_barplot.(png/pdf)
#
# Dependencies:
#   pip install numpy pandas matplotlib scipy scikit-learn pycountry
# --------------------------------------------------------------------------------

from pathlib import Path
from typing import Optional, List, Set, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform

from sklearn.metrics import silhouette_score, silhouette_samples

try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# ----------------------------
# USER SETTINGS
# ----------------------------
DATA = Path("results_part2_external_hierarchy/analysis_table_part2_with_external_indicators.csv")
OUTDIR = Path("figures_country_clustering")
DPI_EXPORT = 600

COUNTRIES_PART1 = ["Afghanistan", "Madagascar", "Malawi", "Morocco"]
COUNTRIES_PART2 = ["Ethiopia", "Niger", "Morocco", "Senegal", "Benin"]

EXTERNAL_COLS = [
    "gii_latest",
    "gdi_latest",
    "hdi_latest",
    "electoral-democracy-index_latest",
    "liberal-democracy-index_latest",
    "women-political-empowerment-index_latest",
]

PRETTY = {
    "gii_latest": "GII",
    "gdi_latest": "GDI",
    "hdi_latest": "HDI",
    "electoral-democracy-index_latest": "Electoral democracy",
    "liberal-democracy-index_latest": "Liberal democracy",
    "women-political-empowerment-index_latest": "Women political empowerment",
}

# Base clustering choice for the main figures
METRIC = "euclidean"
LINKAGE_METHOD = "ward"

FIGTITLE = "Selected countries: clustering by external hierarchy / gender indicators"

# Elbow/silhouette search max k
K_MAX = 6

# Sensitivity add-on: distance × linkage grid
DISTANCE_METRICS = ["euclidean", "cosine", "correlation"]   # SciPy pdist metrics
LINKAGE_METHODS  = ["ward", "average", "complete"]          # Ward requires euclidean


# ----------------------------
# Plot style
# ----------------------------
def set_pub_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(fig, out_png: Path, out_pdf: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# ISO3 helpers
# ----------------------------
def iso3_to_name(iso3: str) -> str:
    iso3 = str(iso3).upper()
    if not HAS_PYCOUNTRY:
        return iso3
    try:
        c = pycountry.countries.get(alpha_3=iso3)
        return c.name if c else iso3
    except Exception:
        return iso3


def name_to_iso3(name: str) -> Optional[str]:
    if not HAS_PYCOUNTRY:
        return None

    overrides: Dict[str, str] = {
        "Morocco": "MAR",
        "Malawi": "MWI",
        "Madagascar": "MDG",
        "Afghanistan": "AFG",
        "Ethiopia": "ETH",
        "Niger": "NER",
        "Senegal": "SEN",
        "Benin": "BEN",
    }
    if name in overrides:
        return overrides[name]

    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None


def build_selected_iso3() -> Tuple[Set[str], Set[str], List[str]]:
    p1_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART1}
    p2_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART2}
    p1_iso3 = {x for x in p1_iso3 if x is not None}
    p2_iso3 = {x for x in p2_iso3 if x is not None}

    union: List[str] = []
    for n in COUNTRIES_PART1:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    for n in COUNTRIES_PART2:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)

    return p1_iso3, p2_iso3, union


def mark_groups(iso3: str, p1_iso3: Set[str], p2_iso3: Set[str]) -> str:
    iso3 = str(iso3).upper()
    tag = ""
    if iso3 in p1_iso3:
        tag += " [P1]"
    if iso3 in p2_iso3:
        tag += " [P2]" if tag == "" else "+P2"
    return tag


# ----------------------------
# Data transforms
# ----------------------------
def zscore_by_column(X: pd.DataFrame) -> pd.DataFrame:
    M = X.to_numpy(float)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (M - mu) / sd
    return pd.DataFrame(Z, index=X.index, columns=X.columns)


def compute_linkage(Z: pd.DataFrame) -> np.ndarray:
    D = pdist(Z.to_numpy(float), metric=METRIC)
    return linkage(D, method=LINKAGE_METHOD)


# ----------------------------
# Elbow (SSE)
# ----------------------------
def within_cluster_sse(X: np.ndarray, labels: np.ndarray) -> float:
    sse = 0.0
    for lab in np.unique(labels):
        pts = X[labels == lab]
        if pts.shape[0] <= 1:
            continue
        centroid = pts.mean(axis=0, keepdims=True)
        dif = pts - centroid
        sse += float(np.sum(dif * dif))
    return sse


def elbow_curve_from_linkage(Z: pd.DataFrame, L: np.ndarray, k_max: int) -> Tuple[np.ndarray, np.ndarray]:
    X = Z.to_numpy(float)
    n = X.shape[0]
    k_max = int(min(k_max, n - 1))
    ks = np.arange(1, k_max + 1, dtype=int)
    sses = []
    for k in ks:
        labels = np.ones(n, dtype=int) if k == 1 else fcluster(L, t=k, criterion="maxclust")
        sses.append(within_cluster_sse(X, labels))
    return ks, np.array(sses, dtype=float)


def choose_k_knee(ks: np.ndarray, sses: np.ndarray) -> int:
    if len(ks) < 3:
        return int(ks[-1])

    x = ks.astype(float)
    y = sses.astype(float)
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    if (x1 == x0) or (y1 == y0):
        return int(ks[1] if len(ks) > 1 else ks[0])

    a = (y1 - y0)
    b = -(x1 - x0)
    c = (x1 - x0) * y0 - (y1 - y0) * x0
    dist = np.abs(a * x + b * y + c) / np.sqrt(a * a + b * b)

    idx = int(np.argmax(dist[1:]) + 1)  # avoid k=1
    return int(ks[idx])


def plot_elbow(ks: np.ndarray, sses: np.ndarray, k_star: int, outbase: Path):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)
    ax.plot(ks, sses, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Within-cluster SSE (Ward / Euclidean)")
    ax.set_title("Elbow curve (selected countries)")
    ax.axvline(k_star, linestyle="--")
    ax.text(k_star + 0.05, float(np.min(sses)), f"k* = {k_star}", va="bottom")
    ax.set_xticks(ks)
    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


# ----------------------------
# Silhouette
# ----------------------------
def silhouette_curve(Z: pd.DataFrame, L: np.ndarray, k_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Silhouette score is defined for k>=2.
    Uses Euclidean silhouette on the standardized feature space Z.
    """
    X = Z.to_numpy(float)
    n = X.shape[0]
    k_max = int(min(k_max, n - 1))
    if k_max < 2:
        return np.array([], dtype=int), np.array([], dtype=float)

    ks = np.arange(2, k_max + 1, dtype=int)
    scores = []
    for k in ks:
        labels = fcluster(L, t=k, criterion="maxclust")
        scores.append(float(silhouette_score(X, labels, metric="euclidean")))
    return ks, np.array(scores, dtype=float)


def plot_silhouette_curve(ks: np.ndarray, scores: np.ndarray, k_star: int, outbase: Path):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)
    ax.plot(ks, scores, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean silhouette score (Euclidean)")
    ax.set_title("Silhouette curve (selected countries)")
    if k_star >= 2:
        ax.axvline(k_star, linestyle="--")
        ax.text(k_star + 0.05, float(np.min(scores)), f"k* = {k_star}", va="bottom")
    ax.set_xticks(ks)
    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


def plot_silhouette_detail(Z: pd.DataFrame, L: np.ndarray, k: int, title: str, outbase: Path):
    if k < 2:
        return

    set_pub_style()
    X = Z.to_numpy(float)
    labels = fcluster(L, t=k, criterion="maxclust")
    sil_vals = silhouette_samples(X, labels, metric="euclidean")
    sil_avg = float(np.mean(sil_vals))

    fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=300)

    y_lower = 10
    for lab in np.unique(labels):
        vals = np.sort(sil_vals[labels == lab])
        size = vals.size
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.85)
        ax.text(-0.05, y_lower + 0.5 * size, f"Cluster {int(lab)}", va="center")
        y_lower = y_upper + 10

    ax.axvline(sil_avg, linestyle="--")
    ax.set_title(f"{title}\nSilhouette plot (k={k}, mean={sil_avg:.3f})")
    ax.set_xlabel("Silhouette value")
    ax.set_ylabel("Samples (grouped by cluster)")
    ax.set_yticks([])
    ax.set_xlim(-0.2, 1.0)
    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


# ----------------------------
# Dendrogram coloring threshold for k
# ----------------------------
def color_threshold_for_k(L: np.ndarray, k: int) -> float:
    n = L.shape[0] + 1
    if k <= 1:
        return np.inf
    idx = max(0, n - k - 1)
    return float(L[idx, 2] + 1e-12)


# ----------------------------
# Rectangular dendrogram + heatmap
# ----------------------------
def plot_clustered_heatmap_selected(
    Z: pd.DataFrame,
    L: np.ndarray,
    k_star: int,
    title: str,
    outbase: Path,
    p1_iso3: Set[str],
    p2_iso3: Set[str],
):
    set_pub_style()

    dd = dendrogram(L, no_plot=True)
    leaf_order = dd["leaves"]
    Zr = Z.iloc[leaf_order].copy()

    fig = plt.figure(figsize=(12.2, max(4.2, 0.65 * len(Zr) + 2.0)), dpi=300)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.15, 4.0], wspace=0.05)

    ax_d = fig.add_subplot(gs[0, 0])
    thr = color_threshold_for_k(L, k_star)
    dendrogram(
        L,
        orientation="left",
        no_labels=True,
        ax=ax_d,
        color_threshold=thr,
        above_threshold_color="black",
    )
    ax_d.invert_yaxis()
    ax_d.axis("off")

    ax_h = fig.add_subplot(gs[0, 1])
    im = ax_h.imshow(Zr.to_numpy(float), aspect="auto")

    ylabels: List[str] = []
    for iso3 in Zr.index.astype(str).tolist():
        base = iso3_to_name(iso3) if HAS_PYCOUNTRY else str(iso3).upper()
        ylabels.append(base + mark_groups(iso3, p1_iso3, p2_iso3))

    ax_h.set_yticks(np.arange(len(Zr)))
    ax_h.set_yticklabels(ylabels)

    ax_h.set_xticks(np.arange(Zr.shape[1]))
    ax_h.set_xticklabels([PRETTY.get(c, c) for c in Zr.columns], rotation=30, ha="right")

    ax_h.set_title(f"{title}\n(k* from elbow = {k_star})")

    for i in range(Zr.shape[0]):
        for j in range(Zr.shape[1]):
            v = Zr.iat[i, j]
            if np.isfinite(v):
                ax_h.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax_h, fraction=0.035, pad=0.02)
    cbar.set_label("z-score (computed across the selected countries)")

    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


# ----------------------------
# Circular dendrogram (radial)
# ----------------------------
def plot_circular_dendrogram(
    Z: pd.DataFrame,
    L: np.ndarray,
    k_star: int,
    title: str,
    outbase: Path,
    p1_iso3: Set[str],
    p2_iso3: Set[str],
):
    set_pub_style()

    thr = color_threshold_for_k(L, k_star)

    dd = dendrogram(
        L,
        no_plot=True,
        color_threshold=thr,
        above_threshold_color="black",
    )

    leaf_order = dd["leaves"]
    leaves = Z.index.astype(str).tolist()
    ordered_iso3 = [leaves[i] for i in leaf_order]

    n = len(ordered_iso3)
    x_leaf = np.arange(n) * 10.0 + 5.0
    x_min, x_max = x_leaf.min(), x_leaf.max()

    def x_to_theta(x: float) -> float:
        if x_max == x_min:
            return 0.0
        return 2.0 * np.pi * (x - x_min) / (x_max - x_min)

    max_h = float(np.max(L[:, 2])) if L.size else 1.0
    R_outer = 1.0
    R_inner = 0.10

    def h_to_r(h: float) -> float:
        if max_h <= 0:
            return R_outer
        return R_outer - (R_outer - R_inner) * (h / max_h)

    fig = plt.figure(figsize=(9.0, 9.0), dpi=300)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_axis_off()

    for xs, ys, col in zip(dd["icoord"], dd["dcoord"], dd["color_list"]):
        thetas = [x_to_theta(x) for x in xs]
        rs = [h_to_r(y) for y in ys]
        ax.plot(thetas, rs, linewidth=1.6, color=col)

    for i, iso3 in enumerate(ordered_iso3):
        theta = x_to_theta(x_leaf[i])
        label = iso3_to_name(iso3) if HAS_PYCOUNTRY else iso3.upper()
        label += mark_groups(iso3, p1_iso3, p2_iso3)

        ang_deg = np.degrees(theta)
        rotation = ang_deg - 90.0
        ha = "left"
        if 90.0 < ang_deg < 270.0:
            rotation += 180.0
            ha = "right"

        ax.text(
            theta,
            R_outer + 0.05,
            label,
            rotation=rotation,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=10,
        )

    ax.set_title(f"{title}\nCircular dendrogram (k* from elbow = {k_star})", pad=30)
    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"))


# ============================================================
# ADD-ON: Distance×Linkage sensitivity (silhouette + cophenetic)
# ============================================================
@dataclass
class SensRow:
    metric: str
    linkage: str
    k_best: int
    sil_best: float
    sil_at_2: float
    sil_at_3: float
    coph_corr: float
    n: int


def _safe_linkage(D_condensed: np.ndarray, linkage_method: str, metric: str) -> Optional[np.ndarray]:
    if linkage_method == "ward" and metric != "euclidean":
        return None
    try:
        return linkage(D_condensed, method=linkage_method)
    except Exception:
        return None


def _silhouette_over_k(
    X: np.ndarray,
    D_condensed: np.ndarray,
    labels_by_k: Dict[int, np.ndarray],
    metric: str,
) -> Dict[int, float]:
    """
    Returns {k: mean silhouette}.
    Uses precomputed distances when metric isn't supported by sklearn's metric string.
    """
    n = X.shape[0]
    scores: Dict[int, float] = {}

    sklearn_supported = {"euclidean", "manhattan", "cosine", "l1", "l2"}
    need_precomputed = metric not in sklearn_supported
    D_square = squareform(D_condensed) if need_precomputed else None

    for k, labels in labels_by_k.items():
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= n:
            continue
        try:
            if need_precomputed:
                scores[k] = float(silhouette_score(D_square, labels, metric="precomputed"))
            else:
                scores[k] = float(silhouette_score(X, labels, metric=metric))
        except Exception:
            continue
    return scores


def run_distance_linkage_sensitivity(
    Z: pd.DataFrame,
    outdir: Path,
    k_max: int,
    distance_metrics: List[str],
    linkage_methods: List[str],
    figtitle: str = "Sensitivity: distance × linkage",
):
    set_pub_style()
    outdir.mkdir(parents=True, exist_ok=True)

    X = Z.to_numpy(float)
    n = X.shape[0]
    k_max = int(min(k_max, n - 1))
    if n < 3 or k_max < 2:
        print("[WARN] Not enough samples to run sensitivity (need n>=3 and k_max>=2).")
        return

    rows: List[SensRow] = []
    grid = {(m, l): np.nan for m in distance_metrics for l in linkage_methods}

    for metric in distance_metrics:
        try:
            D = pdist(X, metric=metric)  # condensed
        except Exception as e:
            print(f"[WARN] pdist failed for metric={metric}: {e}")
            continue

        for link_m in linkage_methods:
            L = _safe_linkage(D, link_m, metric)
            if L is None:
                continue

            # Cophenetic correlation coefficient (uses original distances D)
            try:
                coph_corr, _ = cophenet(L, D)
                coph_corr = float(coph_corr)
            except Exception:
                coph_corr = float("nan")

            labels_by_k: Dict[int, np.ndarray] = {}
            for k in range(2, k_max + 1):
                labels_by_k[k] = fcluster(L, t=k, criterion="maxclust")

            sil_scores = _silhouette_over_k(X=X, D_condensed=D, labels_by_k=labels_by_k, metric=metric)

            sil_at_2 = float(sil_scores.get(2, np.nan))
            sil_at_3 = float(sil_scores.get(3, np.nan))

            if len(sil_scores) == 0:
                k_best = -1
                sil_best = float("nan")
            else:
                k_best = int(max(sil_scores, key=lambda kk: sil_scores[kk]))
                sil_best = float(sil_scores[k_best])

            rows.append(SensRow(
                metric=metric,
                linkage=link_m,
                k_best=k_best,
                sil_best=sil_best,
                sil_at_2=sil_at_2,
                sil_at_3=sil_at_3,
                coph_corr=coph_corr,
                n=n,
            ))
            grid[(metric, link_m)] = sil_best

    if len(rows) == 0:
        print("[WARN] No valid (distance, linkage) combinations were evaluated.")
        return

    summary = pd.DataFrame([r.__dict__ for r in rows]).sort_values(
        by=["sil_best", "coph_corr"], ascending=[False, False]
    )
    out_csv = outdir / "Sensitivity_distance_linkage_summary.csv"
    summary.to_csv(out_csv, index=False)
    print("[OK] Saved:", out_csv.resolve())

    # Heatmap of best silhouette
    M = np.full((len(distance_metrics), len(linkage_methods)), np.nan, dtype=float)
    for i, m in enumerate(distance_metrics):
        for j, l in enumerate(linkage_methods):
            M[i, j] = grid.get((m, l), np.nan)

    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=300)
    im = ax.imshow(M, aspect="auto")
    ax.set_xticks(np.arange(len(linkage_methods)))
    ax.set_xticklabels(linkage_methods)
    ax.set_yticks(np.arange(len(distance_metrics)))
    ax.set_yticklabels(distance_metrics)
    ax.set_title(f"{figtitle}\nCell value = best mean silhouette over k=2..{k_max}")
    ax.set_xlabel("Linkage method")
    ax.set_ylabel("Distance metric")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Best mean silhouette")

    plt.tight_layout()
    savefig(
        fig,
        outdir / "Sensitivity_distance_linkage_heatmap.png",
        outdir / "Sensitivity_distance_linkage_heatmap.pdf",
    )

    # Cophenetic correlation bar plot (sorted)
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    combo = summary.copy()
    combo["combo"] = combo["metric"] + " × " + combo["linkage"]

    ax.bar(np.arange(combo.shape[0]), combo["coph_corr"].to_numpy(float))
    ax.set_xticks(np.arange(combo.shape[0]))
    ax.set_xticklabels(combo["combo"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("Cophenetic correlation")
    ax.set_title("Cophenetic correlation by (distance × linkage)")

    plt.tight_layout()
    savefig(
        fig,
        outdir / "Sensitivity_cophenetic_barplot.png",
        outdir / "Sensitivity_cophenetic_barplot.pdf",
    )

    print("\nTop combinations by silhouette (then cophenetic):")
    print(summary.head(10).to_string(index=False))


# ----------------------------
# Main
# ----------------------------
def main():
    # The elbow SSE curve here is intended for Ward + Euclidean in this implementation.
    if METRIC != "euclidean" or LINKAGE_METHOD != "ward":
        raise ValueError("This elbow SSE implementation is intended for METRIC='euclidean' and LINKAGE_METHOD='ward'.")

    if not DATA.exists():
        raise FileNotFoundError(f"Cannot find: {DATA.resolve()}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    p1_iso3, p2_iso3, selected_iso3_order = build_selected_iso3()
    if len(selected_iso3_order) < 3:
        raise ValueError(
            "Too few selected countries after name->ISO3 mapping. "
            "Install pycountry or check country name spelling."
        )

    df = pd.read_csv(DATA)
    if "iso3" not in df.columns:
        raise ValueError(f"Your CSV must contain 'iso3'. Columns found: {list(df.columns)}")

    df["iso3"] = df["iso3"].astype(str).str.upper()
    df = df.set_index("iso3")

    missing_ext = [c for c in EXTERNAL_COLS if c not in df.columns]
    if missing_ext:
        raise ValueError(f"Missing required external columns: {missing_ext}")

    keep = [c for c in selected_iso3_order if c in df.index]
    missing_sel = [c for c in selected_iso3_order if c not in df.index]
    if missing_sel:
        print("[WARN] These selected ISO3 codes were not found in the CSV and will be skipped:", missing_sel)

    X = df.loc[keep, EXTERNAL_COLS].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if X.shape[0] < 3:
        raise ValueError("Too few selected countries with complete data after dropna(how='any').")

    # Standardize
    Z = zscore_by_column(X)

    # ----------------------------
    # NEW: Sensitivity add-on (THIS is the correct place to call it)
    # ----------------------------
    run_distance_linkage_sensitivity(
        Z=Z,
        outdir=OUTDIR,
        k_max=K_MAX,
        distance_metrics=DISTANCE_METRICS,
        linkage_methods=LINKAGE_METHODS,
        figtitle="Selected countries: sensitivity to distance × linkage",
    )

    # Base linkage for the main plots
    L = compute_linkage(Z)

    # Elbow + knee selection for k*
    ks, sses = elbow_curve_from_linkage(Z, L, k_max=K_MAX)
    k_star = choose_k_knee(ks, sses)
    plot_elbow(ks, sses, k_star, outbase=OUTDIR / "Elbow_selected_external_indicators")

    # Silhouette curve + detailed silhouette plot (using k_star)
    ks_sil, sil_scores = silhouette_curve(Z, L, k_max=K_MAX)
    if ks_sil.size > 0:
        plot_silhouette_curve(
            ks_sil, sil_scores, k_star,
            outbase=OUTDIR / "Silhouette_curve_selected_external_indicators"
        )
        k_for_detail = k_star if k_star >= 2 else int(ks_sil[int(np.argmax(sil_scores))])
        plot_silhouette_detail(
            Z=Z,
            L=L,
            k=k_for_detail,
            title=FIGTITLE,
            outbase=OUTDIR / "Silhouette_plot_selected_external_indicators",
        )
    else:
        print("[WARN] Silhouette not computed (need at least 3 samples and k_max>=2).")

    plot_clustered_heatmap_selected(
        Z=Z,
        L=L,
        k_star=k_star,
        title=FIGTITLE,
        outbase=OUTDIR / "Cluster_selected_external_indicators",
        p1_iso3=p1_iso3,
        p2_iso3=p2_iso3,
    )

    plot_circular_dendrogram(
        Z=Z,
        L=L,
        k_star=k_star,
        title=FIGTITLE,
        outbase=OUTDIR / "Circular_dendrogram_selected_external_indicators",
        p1_iso3=p1_iso3,
        p2_iso3=p2_iso3,
    )

    print("[OK] Done. Outputs in:", OUTDIR.resolve())
    print(" - Elbow_selected_external_indicators.(png/pdf)")
    print(" - Silhouette_curve_selected_external_indicators.(png/pdf)")
    print(" - Silhouette_plot_selected_external_indicators.(png/pdf)")
    print(" - Cluster_selected_external_indicators.(png/pdf)")
    print(" - Circular_dendrogram_selected_external_indicators.(png/pdf)")
    print(" - Sensitivity_distance_linkage_summary.csv")
    print(" - Sensitivity_distance_linkage_heatmap.(png/pdf)")
    print(" - Sensitivity_cophenetic_barplot.(png/pdf)")
    print(f"[INFO] Selected k* = {k_star} from elbow knee heuristic")


if __name__ == "__main__":
    main()



# %% Cell 6
# cluster_all_countries_global_clustering_with_p1p2_check_SAFE.py
# --------------------------------------------------------------------------------
# Global clustering on ALL countries (external indicators), then check whether
# the P1/P2 selected countries fall into the same clusters.
#
# What it does:
# 1) Loads ALL countries from DATA (expects column 'iso3' + EXTERNAL_COLS)
# 2) Drops rows with any missing EXTERNAL_COLS
# 3) Z-scores features across ALL remaining countries
# 4) Runs a distance × linkage sensitivity grid:
#       - best silhouette over k=2..K_MAX
#       - cophenetic correlation
#       - picks best combo by (sil_best desc, coph_corr desc)
# 5) Chooses best k for that best combo (k_best from silhouette)
# 6) Computes cluster labels for ALL countries
# 7) Exports a membership table for your P1/P2 countries
# 8) Saves ONE summary figure that is SAFE for many countries:
#       - left: dendrogram (all leaves, no labels)
#       - right: only P1/P2 positions + cluster id (no giant y-label list)
#
# Outputs (OUTDIR):
# - Global_Sensitivity_distance_linkage_summary.csv
# - Global_Sensitivity_distance_linkage_heatmap.(png/pdf)
# - Global_Sensitivity_cophenetic_barplot.(png/pdf)
# - Global_P1P2_cluster_membership.csv
# - Global_dendrogram_with_P1P2_positions.(png/pdf)
#
# Dependencies:
#   pip install numpy pandas matplotlib scipy scikit-learn pycountry
# --------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Set, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform

from sklearn.metrics import silhouette_score

try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# ----------------------------
# USER SETTINGS
# ----------------------------
DATA = Path("results_part2_external_hierarchy/analysis_table_part2_with_external_indicators.csv")
OUTDIR = Path("figures_country_clustering_global")
DPI_EXPORT = 600

COUNTRIES_PART1 = ["Afghanistan", "Madagascar", "Malawi", "Morocco"]
COUNTRIES_PART2 = ["Ethiopia", "Niger", "Morocco", "Senegal", "Benin"]

EXTERNAL_COLS = [
    "gii_latest",
    "gdi_latest",
    "hdi_latest",
    "electoral-democracy-index_latest",
    "liberal-democracy-index_latest",
    "women-political-empowerment-index_latest",
]

PRETTY = {
    "gii_latest": "GII",
    "gdi_latest": "GDI",
    "hdi_latest": "HDI",
    "electoral-democracy-index_latest": "Electoral democracy",
    "liberal-democracy-index_latest": "Liberal democracy",
    "women-political-empowerment-index_latest": "Women political empowerment",
}

FIGTITLE = "All countries: clustering by external hierarchy / gender indicators"

# silhouette search max k (k=2..K_MAX)
K_MAX = 12

# Sensitivity grid (SciPy pdist metrics + SciPy linkage methods)
DISTANCE_METRICS = ["euclidean", "cosine", "correlation"]
LINKAGE_METHODS  = ["ward", "average", "complete"]  # ward requires euclidean


# ----------------------------
# Plot style
# ----------------------------
def set_pub_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(fig, out_png: Path, out_pdf: Path, tight: bool = True):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
    else:
        fig.savefig(out_png, dpi=DPI_EXPORT)
        fig.savefig(out_pdf)
    plt.close(fig)


# ----------------------------
# ISO3 helpers
# ----------------------------
def iso3_to_name(iso3: str) -> str:
    iso3 = str(iso3).upper()
    if not HAS_PYCOUNTRY:
        return iso3
    try:
        c = pycountry.countries.get(alpha_3=iso3)
        return c.name if c else iso3
    except Exception:
        return iso3


def name_to_iso3(name: str) -> Optional[str]:
    if not HAS_PYCOUNTRY:
        return None

    overrides: Dict[str, str] = {
        "Morocco": "MAR",
        "Malawi": "MWI",
        "Madagascar": "MDG",
        "Afghanistan": "AFG",
        "Ethiopia": "ETH",
        "Niger": "NER",
        "Senegal": "SEN",
        "Benin": "BEN",
    }
    if name in overrides:
        return overrides[name]

    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None


def build_selected_iso3() -> Tuple[Set[str], Set[str], List[str]]:
    p1_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART1}
    p2_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART2}
    p1_iso3 = {x for x in p1_iso3 if x is not None}
    p2_iso3 = {x for x in p2_iso3 if x is not None}

    union: List[str] = []
    for n in COUNTRIES_PART1:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    for n in COUNTRIES_PART2:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    return p1_iso3, p2_iso3, union


def mark_groups(iso3: str, p1_iso3: Set[str], p2_iso3: Set[str]) -> str:
    iso3 = str(iso3).upper()
    tag = ""
    if iso3 in p1_iso3:
        tag += " [P1]"
    if iso3 in p2_iso3:
        tag += " [P2]" if tag == "" else "+P2"
    return tag


# ----------------------------
# Data transforms
# ----------------------------
def zscore_by_column(X: pd.DataFrame) -> pd.DataFrame:
    M = X.to_numpy(float)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (M - mu) / sd
    return pd.DataFrame(Z, index=X.index, columns=X.columns)


# ----------------------------
# Sensitivity grid
# ----------------------------
@dataclass
class SensRow:
    metric: str
    linkage: str
    k_best: int
    sil_best: float
    sil_at_2: float
    sil_at_3: float
    coph_corr: float
    n: int


def _safe_linkage(D_condensed: np.ndarray, linkage_method: str, metric: str) -> Optional[np.ndarray]:
    # Ward expects Euclidean in SciPy
    if linkage_method == "ward" and metric != "euclidean":
        return None
    try:
        return linkage(D_condensed, method=linkage_method)
    except Exception:
        return None


def _silhouette_over_k(
    X: np.ndarray,
    D_condensed: np.ndarray,
    labels_by_k: Dict[int, np.ndarray],
    metric: str,
) -> Dict[int, float]:
    """
    Returns {k: mean silhouette}.
    Uses precomputed distances when metric isn't supported by sklearn metric strings.
    """
    n = X.shape[0]
    scores: Dict[int, float] = {}

    sklearn_supported = {"euclidean", "manhattan", "cosine", "l1", "l2"}
    need_precomputed = metric not in sklearn_supported
    D_square = squareform(D_condensed) if need_precomputed else None

    for k, labels in labels_by_k.items():
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= n:
            continue
        try:
            if need_precomputed:
                scores[k] = float(silhouette_score(D_square, labels, metric="precomputed"))
            else:
                scores[k] = float(silhouette_score(X, labels, metric=metric))
        except Exception:
            continue
    return scores


def run_distance_linkage_sensitivity(
    Z: pd.DataFrame,
    outdir: Path,
    k_max: int,
    distance_metrics: List[str],
    linkage_methods: List[str],
    figtitle: str = "Sensitivity: distance × linkage",
) -> pd.DataFrame:
    """
    Returns summary dataframe sorted by sil_best desc, coph_corr desc.
    Also saves CSV + heatmap + coph barplot.
    """
    set_pub_style()
    outdir.mkdir(parents=True, exist_ok=True)

    X = Z.to_numpy(float)
    n = X.shape[0]
    k_max = int(min(k_max, n - 1))
    if n < 3 or k_max < 2:
        raise ValueError("Not enough samples for sensitivity (need n>=3 and k_max>=2).")

    rows: List[SensRow] = []
    grid = {(m, l): np.nan for m in distance_metrics for l in linkage_methods}

    for metric in distance_metrics:
        try:
            D = pdist(X, metric=metric)
        except Exception as e:
            print(f"[WARN] pdist failed for metric={metric}: {e}")
            continue

        for link_m in linkage_methods:
            L = _safe_linkage(D, link_m, metric)
            if L is None:
                continue

            try:
                coph_corr, _ = cophenet(L, D)
                coph_corr = float(coph_corr)
            except Exception:
                coph_corr = float("nan")

            labels_by_k = {k: fcluster(L, t=k, criterion="maxclust") for k in range(2, k_max + 1)}
            sil_scores = _silhouette_over_k(X=X, D_condensed=D, labels_by_k=labels_by_k, metric=metric)

            sil_at_2 = float(sil_scores.get(2, np.nan))
            sil_at_3 = float(sil_scores.get(3, np.nan))

            if len(sil_scores) == 0:
                k_best = -1
                sil_best = float("nan")
            else:
                k_best = int(max(sil_scores, key=lambda kk: sil_scores[kk]))
                sil_best = float(sil_scores[k_best])

            rows.append(SensRow(metric, link_m, k_best, sil_best, sil_at_2, sil_at_3, coph_corr, n))
            grid[(metric, link_m)] = sil_best

    if len(rows) == 0:
        raise RuntimeError("No valid (distance, linkage) combinations were evaluated.")

    summary = pd.DataFrame([r.__dict__ for r in rows]).sort_values(
        by=["sil_best", "coph_corr"], ascending=[False, False]
    )

    out_csv = outdir / "Global_Sensitivity_distance_linkage_summary.csv"
    summary.to_csv(out_csv, index=False)
    print("[OK] Saved:", out_csv.resolve())

    # Heatmap of best silhouette
    M = np.full((len(distance_metrics), len(linkage_methods)), np.nan, dtype=float)
    for i, m in enumerate(distance_metrics):
        for j, l in enumerate(linkage_methods):
            M[i, j] = grid.get((m, l), np.nan)

    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=300)
    im = ax.imshow(M, aspect="auto")
    ax.set_xticks(np.arange(len(linkage_methods)))
    ax.set_xticklabels(linkage_methods)
    ax.set_yticks(np.arange(len(distance_metrics)))
    ax.set_yticklabels(distance_metrics)
    ax.set_title(f"{figtitle}\nCell value = best mean silhouette over k=2..{k_max}")
    ax.set_xlabel("Linkage method")
    ax.set_ylabel("Distance metric")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Best mean silhouette")
    plt.tight_layout()
    savefig(
        fig,
        outdir / "Global_Sensitivity_distance_linkage_heatmap.png",
        outdir / "Global_Sensitivity_distance_linkage_heatmap.pdf",
        tight=True
    )

    # Cophenetic correlation bar plot
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    combo = summary.copy()
    combo["combo"] = combo["metric"] + " × " + combo["linkage"]
    ax.bar(np.arange(combo.shape[0]), combo["coph_corr"].to_numpy(float))
    ax.set_xticks(np.arange(combo.shape[0]))
    ax.set_xticklabels(combo["combo"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("Cophenetic correlation")
    ax.set_title("Cophenetic correlation by (distance × linkage)")
    plt.tight_layout()
    savefig(
        fig,
        outdir / "Global_Sensitivity_cophenetic_barplot.png",
        outdir / "Global_Sensitivity_cophenetic_barplot.pdf",
        tight=True
    )

    print("\nTop combinations by silhouette (then cophenetic):")
    print(summary.head(10).to_string(index=False))
    return summary


# ----------------------------
# Summary figure (SAFE): dendrogram + P1/P2 positions only
# ----------------------------
def color_threshold_for_k(L: np.ndarray, k: int) -> float:
    n = L.shape[0] + 1
    if k <= 1:
        return np.inf
    idx = max(0, n - k - 1)
    return float(L[idx, 2] + 1e-12)


def plot_global_dendrogram_with_p1p2_positions(
    Z: pd.DataFrame,
    L: np.ndarray,
    k_star: int,
    outbase: Path,
    p1_iso3: Set[str],
    p2_iso3: Set[str],
    title: str,
):
    """
    SAFE for many countries:
      - Left: dendrogram of all countries (no labels)
      - Right: only P1/P2 country names placed at their leaf positions, with cluster id
    """
    set_pub_style()

    # Leaf order from no_plot dendrogram
    dd0 = dendrogram(L, orientation="left", no_plot=True)
    leaf_order = dd0["leaves"]
    iso3_list = Z.index.astype(str).tolist()
    ordered_iso3 = [iso3_list[i] for i in leaf_order]

    # Cluster labels for ALL (Z order -> map)
    labels_all = fcluster(L, t=k_star, criterion="maxclust")
    label_map = {iso3_list[i]: int(labels_all[i]) for i in range(len(iso3_list))}

    # Indices of P1/P2 among ordered leaves
    idx_map = {iso3: i for i, iso3 in enumerate(ordered_iso3)}
    sel_iso3 = [iso3 for iso3 in ordered_iso3 if (iso3 in p1_iso3) or (iso3 in p2_iso3)]

    # Fixed-size figure prevents huge pixel heights
    fig = plt.figure(figsize=(12.8, 7.2), dpi=300)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4.7, 2.1], wspace=0.06)

    # --- Left: dendrogram ---
    ax_d = fig.add_subplot(gs[0, 0])
    thr = color_threshold_for_k(L, k_star)
    dendrogram(
        L,
        orientation="left",
        no_labels=True,
        ax=ax_d,
        color_threshold=thr,
        above_threshold_color="black",
    )
    ax_d.invert_yaxis()
    ax_d.set_title(f"{title}\nAll countries dendrogram (colored for k={k_star})")
    ax_d.set_xlabel("Distance")
    ax_d.set_yticks([])

    # --- Right: P1/P2 positions panel ---
    ax_r = fig.add_subplot(gs[0, 1])
    ax_r.set_title("P1/P2 positions in global clustering")

    # y axis = leaf index
    n_leaves = len(ordered_iso3)
    ax_r.set_xlim(0.0, 1.0)
    ax_r.set_ylim(-1, n_leaves)
    ax_r.invert_yaxis()
    ax_r.set_yticks([])
    ax_r.set_xticks([0.25, 0.50, 0.75])
    ax_r.set_xticklabels(["P1", "P1+P2", "P2"])

    # Light guide lines (not too many)
    step = max(1, n_leaves // 18)
    ax_r.hlines(
        y=np.arange(0, n_leaves, step),
        xmin=0.0, xmax=1.0,
        linewidth=0.6, alpha=0.25
    )

    for iso3 in sel_iso3:
        i = idx_map[iso3]
        v = 0
        if iso3 in p1_iso3:
            v += 1
        if iso3 in p2_iso3:
            v += 2

        if v == 1:
            x = 0.25
            tag = "P1"
        elif v == 2:
            x = 0.75
            tag = "P2"
        else:
            x = 0.50
            tag = "P1+P2"

        ax_r.plot([x], [i], marker="o", markersize=7, linestyle="None")

        name = iso3_to_name(iso3) if HAS_PYCOUNTRY else iso3
        cl = label_map.get(iso3, None)
        ax_r.text(
            x + 0.04, i,
            f"{name} ({iso3})  {tag} | C{int(cl)}",
            va="center", fontsize=10
        )

    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"), tight=True)


# ----------------------------
# Membership export
# ----------------------------
def export_p1p2_membership_table(
    Z: pd.DataFrame,
    labels_all: np.ndarray,
    p1_iso3: Set[str],
    p2_iso3: Set[str],
    out_csv: Path,
):
    """
    Exports only the union of P1/P2 countries (if present in Z.index), with their cluster label.
    """
    iso3_index = Z.index.astype(str).tolist()
    label_map = {iso3: int(labels_all[i]) for i, iso3 in enumerate(iso3_index)}

    p1_list = sorted([x for x in p1_iso3 if x in label_map])
    p2_list = sorted([x for x in p2_iso3 if x in label_map])
    union = []
    for x in p1_list + p2_list:
        if x not in union:
            union.append(x)

    rows = []
    for iso3 in union:
        rows.append({
            "iso3": iso3,
            "country": iso3_to_name(iso3) if HAS_PYCOUNTRY else iso3,
            "in_P1": iso3 in p1_iso3,
            "in_P2": iso3 in p2_iso3,
            "cluster": label_map.get(iso3, np.nan),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("[OK] Saved:", out_csv.resolve())


# ----------------------------
# Main
# ----------------------------
def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Cannot find: {DATA.resolve()}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    p1_iso3, p2_iso3, selected_iso3_union = build_selected_iso3()

    df = pd.read_csv(DATA)
    if "iso3" not in df.columns:
        raise ValueError(f"Your CSV must contain 'iso3'. Columns found: {list(df.columns)}")

    missing_ext = [c for c in EXTERNAL_COLS if c not in df.columns]
    if missing_ext:
        raise ValueError(f"Missing required external columns: {missing_ext}")

    # --- CLUSTER ALL COUNTRIES ---
    df["iso3"] = df["iso3"].astype(str).str.upper()
    df = df.set_index("iso3")

    X_all = df[EXTERNAL_COLS].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if X_all.shape[0] < 3:
        raise ValueError("Too few countries with complete EXTERNAL_COLS after dropna(how='any').")

    # Standardize across ALL countries
    Z_all = zscore_by_column(X_all)

    # --- SENSITIVITY GRID on ALL COUNTRIES ---
    summary = run_distance_linkage_sensitivity(
        Z=Z_all,
        outdir=OUTDIR,
        k_max=K_MAX,
        distance_metrics=DISTANCE_METRICS,
        linkage_methods=LINKAGE_METHODS,
        figtitle="All countries: sensitivity to distance × linkage",
    )

    # Choose best combo by silhouette then cophenetic (first row)
    best = summary.iloc[0]
    best_metric = str(best["metric"])
    best_linkage = str(best["linkage"])
    best_k = int(best["k_best"])

    if best_k < 2:
        raise RuntimeError("Best k from silhouette grid is <2. Try increasing K_MAX or check data quality.")

    print("\n[CHOSEN] Best configuration from grid:")
    print(f"  distance = {best_metric}")
    print(f"  linkage  = {best_linkage}")
    print(f"  k        = {best_k}")
    print(f"  sil_best = {float(best['sil_best']):.6f}")
    print(f"  coph     = {float(best['coph_corr']):.6f}")

    # --- Final clustering using chosen combo ---
    X = Z_all.to_numpy(float)
    D = pdist(X, metric=best_metric)
    L = _safe_linkage(D, best_linkage, best_metric)
    if L is None:
        raise RuntimeError("Chosen (distance, linkage) invalid (likely ward with non-euclidean).")

    labels_all = fcluster(L, t=best_k, criterion="maxclust")

    # --- Export membership for P1/P2 ---
    export_p1p2_membership_table(
        Z=Z_all,
        labels_all=labels_all,
        p1_iso3=p1_iso3,
        p2_iso3=p2_iso3,
        out_csv=OUTDIR / "Global_P1P2_cluster_membership.csv",
    )

    # Warn if some selected countries are missing from the global complete-data set
    missing_sel = [c for c in selected_iso3_union if c not in Z_all.index]
    if missing_sel:
        print("[WARN] Some P1/P2 countries NOT in the global complete-data set (missing indicators):", missing_sel)

    # --- Summary figure: dendrogram + P1/P2 positions (SAFE) ---
    plot_global_dendrogram_with_p1p2_positions(
        Z=Z_all,
        L=L,
        k_star=best_k,
        outbase=OUTDIR / "Global_dendrogram_with_P1P2_positions",
        p1_iso3=p1_iso3,
        p2_iso3=p2_iso3,
        title=FIGTITLE,
    )

    print("\n[OK] Done. Outputs in:", OUTDIR.resolve())
    print(" - Global_Sensitivity_distance_linkage_summary.csv")
    print(" - Global_Sensitivity_distance_linkage_heatmap.(png/pdf)")
    print(" - Global_Sensitivity_cophenetic_barplot.(png/pdf)")
    print(" - Global_P1P2_cluster_membership.csv")
    print(" - Global_dendrogram_with_P1P2_positions.(png/pdf)")


if __name__ == "__main__":
    main()



# %% Cell 7
# cluster_all_countries_global_clustering_with_p1p2_check_SAFE_PLUS_SUBSET.py
# --------------------------------------------------------------------------------
# Global clustering on ALL countries (external indicators), then check whether
# the P1/P2 selected countries fall into the same clusters + a readable dendrogram
# for ONLY P1/P2 countries.
#
# What it does:
# 1) Loads ALL countries from DATA (expects column 'iso3' + EXTERNAL_COLS)
# 2) Drops rows with any missing EXTERNAL_COLS
# 3) Z-scores features across ALL remaining countries
# 4) Runs a distance × linkage sensitivity grid:
#       - best silhouette over k=2..K_MAX
#       - cophenetic correlation
#       - picks best combo by (sil_best desc, coph_corr desc)
# 5) Chooses best k for that best combo (k_best from silhouette)
# 6) Computes cluster labels for ALL countries
# 7) Exports a membership table for your P1/P2 countries
# 8) Saves ONE summary figure (SAFE for many countries):
#       - left: dendrogram (all leaves, no labels)
#       - right: only P1/P2 positions + cluster id
# 9) Builds a SECOND dendrogram for ONLY P1/P2 countries (with labels)
#    using the SAME chosen (distance, linkage) from the global grid.
#
# Outputs (OUTDIR):
# - Global_Sensitivity_distance_linkage_summary.csv
# - Global_Sensitivity_distance_linkage_heatmap.(png/pdf)
# - Global_Sensitivity_cophenetic_barplot.(png/pdf)  <-- slategray bars
# - Global_P1P2_cluster_membership.csv
# - Global_dendrogram_with_P1P2_positions.(png/pdf)
# - P1P2_only_dendrogram.(png/pdf)
# - P1P2_only_clustered_heatmap.(png/pdf)
#
# Dependencies:
#   pip install numpy pandas matplotlib scipy scikit-learn pycountry
# --------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Set, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform

from sklearn.metrics import silhouette_score

try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# ----------------------------
# USER SETTINGS
# ----------------------------
DATA = Path("results_part2_external_hierarchy/analysis_table_part2_with_external_indicators.csv")
OUTDIR = Path("figures_country_clustering_global")
DPI_EXPORT = 600

COUNTRIES_PART1 = ["Afghanistan", "Madagascar", "Malawi", "Morocco"]
COUNTRIES_PART2 = ["Ethiopia", "Niger", "Morocco", "Senegal", "Benin"]

EXTERNAL_COLS = [
    "gii_latest",
    "gdi_latest",
    "hdi_latest",
    "electoral-democracy-index_latest",
    "liberal-democracy-index_latest",
    "women-political-empowerment-index_latest",
]

PRETTY = {
    "gii_latest": "GII",
    "gdi_latest": "GDI",
    "hdi_latest": "HDI",
    "electoral-democracy-index_latest": "Electoral democracy",
    "liberal-democracy-index_latest": "Liberal democracy",
    "women-political-empowerment-index_latest": "Women political empowerment",
}

FIGTITLE = "All countries: clustering by external hierarchy / gender indicators"

# silhouette search max k (k=2..K_MAX)
K_MAX = 12

# Sensitivity grid (SciPy pdist metrics + SciPy linkage methods)
DISTANCE_METRICS = ["euclidean", "cosine", "correlation"]
LINKAGE_METHODS  = ["ward", "average", "complete"]  # ward requires euclidean

# Requested change
COPHENETIC_BAR_COLOR = "slategray"


# ----------------------------
# Plot style
# ----------------------------
def set_pub_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(fig, out_png: Path, out_pdf: Path, tight: bool = True):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
    else:
        fig.savefig(out_png, dpi=DPI_EXPORT)
        fig.savefig(out_pdf)
    plt.close(fig)


# ----------------------------
# ISO3 helpers
# ----------------------------
def iso3_to_name(iso3: str) -> str:
    iso3 = str(iso3).upper()
    if not HAS_PYCOUNTRY:
        return iso3
    try:
        c = pycountry.countries.get(alpha_3=iso3)
        return c.name if c else iso3
    except Exception:
        return iso3


def name_to_iso3(name: str) -> Optional[str]:
    if not HAS_PYCOUNTRY:
        return None

    overrides: Dict[str, str] = {
        "Morocco": "MAR",
        "Malawi": "MWI",
        "Madagascar": "MDG",
        "Afghanistan": "AFG",
        "Ethiopia": "ETH",
        "Niger": "NER",
        "Senegal": "SEN",
        "Benin": "BEN",
    }
    if name in overrides:
        return overrides[name]

    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None


def build_selected_iso3() -> Tuple[Set[str], Set[str], List[str]]:
    p1_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART1}
    p2_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART2}
    p1_iso3 = {x for x in p1_iso3 if x is not None}
    p2_iso3 = {x for x in p2_iso3 if x is not None}

    union: List[str] = []
    for n in COUNTRIES_PART1:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    for n in COUNTRIES_PART2:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    return p1_iso3, p2_iso3, union


# ----------------------------
# Data transforms
# ----------------------------
def zscore_by_column(X: pd.DataFrame) -> pd.DataFrame:
    M = X.to_numpy(float)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (M - mu) / sd
    return pd.DataFrame(Z, index=X.index, columns=X.columns)


# ----------------------------
# Sensitivity grid
# ----------------------------
@dataclass
class SensRow:
    metric: str
    linkage: str
    k_best: int
    sil_best: float
    sil_at_2: float
    sil_at_3: float
    coph_corr: float
    n: int


def _safe_linkage(D_condensed: np.ndarray, linkage_method: str, metric: str) -> Optional[np.ndarray]:
    # Ward expects Euclidean in SciPy
    if linkage_method == "ward" and metric != "euclidean":
        return None
    try:
        return linkage(D_condensed, method=linkage_method)
    except Exception:
        return None


def _silhouette_over_k(
    X: np.ndarray,
    D_condensed: np.ndarray,
    labels_by_k: Dict[int, np.ndarray],
    metric: str,
) -> Dict[int, float]:
    """
    Returns {k: mean silhouette}.
    Uses precomputed distances when metric isn't supported by sklearn metric strings.
    """
    n = X.shape[0]
    scores: Dict[int, float] = {}

    sklearn_supported = {"euclidean", "manhattan", "cosine", "l1", "l2"}
    need_precomputed = metric not in sklearn_supported
    D_square = squareform(D_condensed) if need_precomputed else None

    for k, labels in labels_by_k.items():
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= n:
            continue
        try:
            if need_precomputed:
                scores[k] = float(silhouette_score(D_square, labels, metric="precomputed"))
            else:
                scores[k] = float(silhouette_score(X, labels, metric=metric))
        except Exception:
            continue
    return scores


def run_distance_linkage_sensitivity(
    Z: pd.DataFrame,
    outdir: Path,
    k_max: int,
    distance_metrics: List[str],
    linkage_methods: List[str],
    figtitle: str = "Sensitivity: distance × linkage",
) -> pd.DataFrame:
    """
    Returns summary dataframe sorted by sil_best desc, coph_corr desc.
    Also saves CSV + heatmap + coph barplot.
    """
    set_pub_style()
    outdir.mkdir(parents=True, exist_ok=True)

    X = Z.to_numpy(float)
    n = X.shape[0]
    k_max = int(min(k_max, n - 1))
    if n < 3 or k_max < 2:
        raise ValueError("Not enough samples for sensitivity (need n>=3 and k_max>=2).")

    rows: List[SensRow] = []
    grid = {(m, l): np.nan for m in distance_metrics for l in linkage_methods}

    for metric in distance_metrics:
        try:
            D = pdist(X, metric=metric)
        except Exception as e:
            print(f"[WARN] pdist failed for metric={metric}: {e}")
            continue

        for link_m in linkage_methods:
            L = _safe_linkage(D, link_m, metric)
            if L is None:
                continue

            try:
                coph_corr, _ = cophenet(L, D)
                coph_corr = float(coph_corr)
            except Exception:
                coph_corr = float("nan")

            labels_by_k = {k: fcluster(L, t=k, criterion="maxclust") for k in range(2, k_max + 1)}
            sil_scores = _silhouette_over_k(X=X, D_condensed=D, labels_by_k=labels_by_k, metric=metric)

            sil_at_2 = float(sil_scores.get(2, np.nan))
            sil_at_3 = float(sil_scores.get(3, np.nan))

            if len(sil_scores) == 0:
                k_best = -1
                sil_best = float("nan")
            else:
                k_best = int(max(sil_scores, key=lambda kk: sil_scores[kk]))
                sil_best = float(sil_scores[k_best])

            rows.append(SensRow(metric, link_m, k_best, sil_best, sil_at_2, sil_at_3, coph_corr, n))
            grid[(metric, link_m)] = sil_best

    if len(rows) == 0:
        raise RuntimeError("No valid (distance, linkage) combinations were evaluated.")

    summary = pd.DataFrame([r.__dict__ for r in rows]).sort_values(
        by=["sil_best", "coph_corr"], ascending=[False, False]
    )

    out_csv = outdir / "Global_Sensitivity_distance_linkage_summary.csv"
    summary.to_csv(out_csv, index=False)
    print("[OK] Saved:", out_csv.resolve())

    # Heatmap of best silhouette
    M = np.full((len(distance_metrics), len(linkage_methods)), np.nan, dtype=float)
    for i, m in enumerate(distance_metrics):
        for j, l in enumerate(linkage_methods):
            M[i, j] = grid.get((m, l), np.nan)

    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=300)
    im = ax.imshow(M, aspect="auto")
    ax.set_xticks(np.arange(len(linkage_methods)))
    ax.set_xticklabels(linkage_methods)
    ax.set_yticks(np.arange(len(distance_metrics)))
    ax.set_yticklabels(distance_metrics)
    ax.set_title(f"{figtitle}\nCell value = best mean silhouette over k=2..{k_max}")
    ax.set_xlabel("Linkage method")
    ax.set_ylabel("Distance metric")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Best mean silhouette")
    plt.tight_layout()
    savefig(
        fig,
        outdir / "Global_Sensitivity_distance_linkage_heatmap.png",
        outdir / "Global_Sensitivity_distance_linkage_heatmap.pdf",
        tight=True
    )

    # Cophenetic correlation bar plot (requested slategray)
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    combo = summary.copy()
    combo["combo"] = combo["metric"] + " × " + combo["linkage"]
    ax.bar(np.arange(combo.shape[0]), combo["coph_corr"].to_numpy(float), color=COPHENETIC_BAR_COLOR)
    ax.set_xticks(np.arange(combo.shape[0]))
    ax.set_xticklabels(combo["combo"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("Cophenetic correlation")
    ax.set_title("Cophenetic correlation by (distance × linkage)")
    plt.tight_layout()
    savefig(
        fig,
        outdir / "Global_Sensitivity_cophenetic_barplot.png",
        outdir / "Global_Sensitivity_cophenetic_barplot.pdf",
        tight=True
    )

    print("\nTop combinations by silhouette (then cophenetic):")
    print(summary.head(10).to_string(index=False))
    return summary


# ----------------------------
# Summary figure (SAFE): dendrogram + P1/P2 positions only
# ----------------------------
def color_threshold_for_k(L: np.ndarray, k: int) -> float:
    n = L.shape[0] + 1
    if k <= 1:
        return np.inf
    idx = max(0, n - k - 1)
    return float(L[idx, 2] + 1e-12)


def plot_global_dendrogram_with_p1p2_positions(
    Z: pd.DataFrame,
    L: np.ndarray,
    k_star: int,
    outbase: Path,
    p1_iso3: Set[str],
    p2_iso3: Set[str],
    title: str,
):
    """
    SAFE for many countries:
      - Left: dendrogram of all countries (no labels)
      - Right: only P1/P2 country names placed at their leaf positions, with cluster id
    """
    set_pub_style()

    # Leaf order from no_plot dendrogram
    dd0 = dendrogram(L, orientation="left", no_plot=True)
    leaf_order = dd0["leaves"]
    iso3_list = Z.index.astype(str).tolist()
    ordered_iso3 = [iso3_list[i] for i in leaf_order]

    # Cluster labels for ALL (Z order -> map)
    labels_all = fcluster(L, t=k_star, criterion="maxclust")
    label_map = {iso3_list[i]: int(labels_all[i]) for i in range(len(iso3_list))}

    # Indices of P1/P2 among ordered leaves
    idx_map = {iso3: i for i, iso3 in enumerate(ordered_iso3)}
    sel_iso3 = [iso3 for iso3 in ordered_iso3 if (iso3 in p1_iso3) or (iso3 in p2_iso3)]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=300)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4.7, 2.1], wspace=0.06)

    # --- Left: dendrogram ---
    ax_d = fig.add_subplot(gs[0, 0])
    thr = color_threshold_for_k(L, k_star)
    dendrogram(
        L,
        orientation="left",
        no_labels=True,
        ax=ax_d,
        color_threshold=thr,
        above_threshold_color="black",
    )
    ax_d.invert_yaxis()
    ax_d.set_title(f"{title}\nAll countries dendrogram (colored for k={k_star})")
    ax_d.set_xlabel("Distance")
    ax_d.set_yticks([])

    # --- Right: P1/P2 positions panel ---
    ax_r = fig.add_subplot(gs[0, 1])
    ax_r.set_title("P1/P2 positions in global clustering")

    n_leaves = len(ordered_iso3)
    ax_r.set_xlim(0.0, 1.0)
    ax_r.set_ylim(-1, n_leaves)
    ax_r.invert_yaxis()
    ax_r.set_yticks([])
    ax_r.set_xticks([0.25, 0.50, 0.75])
    ax_r.set_xticklabels(["P1", "P1+P2", "P2"])

    step = max(1, n_leaves // 18)
    ax_r.hlines(
        y=np.arange(0, n_leaves, step),
        xmin=0.0, xmax=1.0,
        linewidth=0.6, alpha=0.25
    )

    for iso3 in sel_iso3:
        i = idx_map[iso3]
        v = 0
        if iso3 in p1_iso3:
            v += 1
        if iso3 in p2_iso3:
            v += 2

        if v == 1:
            x, tag = 0.25, "P1"
        elif v == 2:
            x, tag = 0.75, "P2"
        else:
            x, tag = 0.50, "P1+P2"

        ax_r.plot([x], [i], marker="o", markersize=7, linestyle="None")

        name = iso3_to_name(iso3) if HAS_PYCOUNTRY else iso3
        cl = label_map.get(iso3, None)
        ax_r.text(
            x + 0.04, i,
            f"{name} ({iso3})  {tag} | C{int(cl)}",
            va="center", fontsize=10
        )

    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"), tight=True)


# ----------------------------
# Membership export
# ----------------------------
def export_p1p2_membership_table(
    Z: pd.DataFrame,
    labels_all: np.ndarray,
    p1_iso3: Set[str],
    p2_iso3: Set[str],
    out_csv: Path,
):
    """
    Exports only the union of P1/P2 countries (if present in Z.index), with their cluster label.
    """
    iso3_index = Z.index.astype(str).tolist()
    label_map = {iso3: int(labels_all[i]) for i, iso3 in enumerate(iso3_index)}

    union = []
    for x in sorted(p1_iso3) + sorted(p2_iso3):
        if x not in union:
            union.append(x)

    rows = []
    for iso3 in union:
        rows.append({
            "iso3": iso3,
            "country": iso3_to_name(iso3) if HAS_PYCOUNTRY else iso3,
            "in_P1": iso3 in p1_iso3,
            "in_P2": iso3 in p2_iso3,
            "cluster": label_map.get(iso3, np.nan),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("[OK] Saved:", out_csv.resolve())


# ----------------------------
# P1/P2-only dendrogram + heatmap (readable)
# ----------------------------
def plot_p1p2_only_dendrogram(
    Z_all: pd.DataFrame,
    union_iso3: List[str],
    best_metric: str,
    best_linkage: str,
    outbase: Path,
    title: str,
):
    """
    Builds a dendrogram ONLY for the P1/P2 countries, using the SAME chosen
    (distance, linkage) as global. This is the figure that is actually readable.
    """
    keep = [c for c in union_iso3 if c in Z_all.index]
    Z_sub = Z_all.loc[keep].copy()
    if Z_sub.shape[0] < 3:
        print("[WARN] Too few P1/P2 countries present with complete data to plot subset dendrogram.")
        return

    # If ward was chosen globally (should only happen with euclidean), keep it consistent.
    if best_linkage == "ward" and best_metric != "euclidean":
        print("[WARN] Subset dendrogram: ward requires euclidean; switching metric to euclidean for subset.")
        best_metric = "euclidean"

    X = Z_sub.to_numpy(float)
    D = pdist(X, metric=best_metric)
    L = linkage(D, method=best_linkage)

    # Pretty labels
    labels = []
    for iso3 in Z_sub.index.astype(str).tolist():
        labels.append(f"{iso3_to_name(iso3)} ({iso3})")

    set_pub_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)
    dendrogram(
        L,
        labels=labels,
        orientation="top",
        leaf_rotation=25,
        leaf_font_size=10,
        ax=ax,
        above_threshold_color="black",
    )
    ax.set_title(f"{title}\nP1/P2-only dendrogram  |  distance={best_metric}, linkage={best_linkage}")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    savefig(fig, outbase.with_suffix(".png"), outbase.with_suffix(".pdf"), tight=True)

    # Also save a clustered heatmap-like panel (Z-scores)
    set_pub_style()
    dd = dendrogram(L, no_plot=True)
    order = dd["leaves"]
    Zr = Z_sub.iloc[order].copy()

    fig, ax = plt.subplots(figsize=(10.2, 4.2), dpi=300)
    im = ax.imshow(Zr.to_numpy(float), aspect="auto")
    ax.set_yticks(np.arange(Zr.shape[0]))
    ax.set_yticklabels([f"{iso3_to_name(i)} ({i})" for i in Zr.index.astype(str)], fontsize=10)
    ax.set_xticks(np.arange(Zr.shape[1]))
    ax.set_xticklabels([PRETTY.get(c, c) for c in Zr.columns], rotation=25, ha="right")
    ax.set_title("P1/P2-only clustered heatmap (z-scored across ALL countries)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("z-score")
    plt.tight_layout()
    savefig(fig, outbase.with_name(outbase.name.replace("dendrogram", "clustered_heatmap")).with_suffix(".png"),
            outbase.with_name(outbase.name.replace("dendrogram", "clustered_heatmap")).with_suffix(".pdf"),
            tight=True)


# ----------------------------
# Main
# ----------------------------
def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Cannot find: {DATA.resolve()}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    p1_iso3, p2_iso3, union_iso3 = build_selected_iso3()

    df = pd.read_csv(DATA)
    if "iso3" not in df.columns:
        raise ValueError(f"Your CSV must contain 'iso3'. Columns found: {list(df.columns)}")

    missing_ext = [c for c in EXTERNAL_COLS if c not in df.columns]
    if missing_ext:
        raise ValueError(f"Missing required external columns: {missing_ext}")

    # --- CLUSTER ALL COUNTRIES ---
    df["iso3"] = df["iso3"].astype(str).str.upper()
    df = df.set_index("iso3")

    X_all = df[EXTERNAL_COLS].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if X_all.shape[0] < 3:
        raise ValueError("Too few countries with complete EXTERNAL_COLS after dropna(how='any').")

    # Standardize across ALL countries
    Z_all = zscore_by_column(X_all)

    # --- SENSITIVITY GRID on ALL COUNTRIES ---
    summary = run_distance_linkage_sensitivity(
        Z=Z_all,
        outdir=OUTDIR,
        k_max=K_MAX,
        distance_metrics=DISTANCE_METRICS,
        linkage_methods=LINKAGE_METHODS,
        figtitle="All countries: sensitivity to distance × linkage",
    )

    # Choose best combo by silhouette then cophenetic (first row)
    best = summary.iloc[0]
    best_metric = str(best["metric"])
    best_linkage = str(best["linkage"])
    best_k = int(best["k_best"])

    if best_k < 2:
        raise RuntimeError("Best k from silhouette grid is <2. Try increasing K_MAX or check data quality.")

    print("\n[CHOSEN] Best configuration from grid:")
    print(f"  distance = {best_metric}")
    print(f"  linkage  = {best_linkage}")
    print(f"  k        = {best_k}")
    print(f"  sil_best = {float(best['sil_best']):.6f}")
    print(f"  coph     = {float(best['coph_corr']):.6f}")

    # --- Final clustering using chosen combo ---
    X = Z_all.to_numpy(float)
    D = pdist(X, metric=best_metric)
    L = _safe_linkage(D, best_linkage, best_metric)
    if L is None:
        raise RuntimeError("Chosen (distance, linkage) invalid (likely ward with non-euclidean).")

    labels_all = fcluster(L, t=best_k, criterion="maxclust")

    # --- Export membership for P1/P2 ---
    export_p1p2_membership_table(
        Z=Z_all,
        labels_all=labels_all,
        p1_iso3=p1_iso3,
        p2_iso3=p2_iso3,
        out_csv=OUTDIR / "Global_P1P2_cluster_membership.csv",
    )

    # Warn if some selected countries are missing from the global complete-data set
    missing_sel = [c for c in union_iso3 if c not in Z_all.index]
    if missing_sel:
        print("[WARN] Some P1/P2 countries NOT in the global complete-data set (missing indicators):", missing_sel)

    # --- Summary figure: dendrogram + P1/P2 positions (SAFE) ---
    plot_global_dendrogram_with_p1p2_positions(
        Z=Z_all,
        L=L,
        k_star=best_k,
        outbase=OUTDIR / "Global_dendrogram_with_P1P2_positions",
        p1_iso3=p1_iso3,
        p2_iso3=p2_iso3,
        title=FIGTITLE,
    )

    # --- NEW: P1/P2-only dendrogram (readable) using same chosen metric/linkage ---
    plot_p1p2_only_dendrogram(
        Z_all=Z_all,
        union_iso3=union_iso3,
        best_metric=best_metric,
        best_linkage=best_linkage,
        outbase=OUTDIR / "P1P2_only_dendrogram",
        title="Similarity of P1/P2 countries using external indicators",
    )

    print("\n[OK] Done. Outputs in:", OUTDIR.resolve())
    print(" - Global_Sensitivity_distance_linkage_summary.csv")
    print(" - Global_Sensitivity_distance_linkage_heatmap.(png/pdf)")
    print(" - Global_Sensitivity_cophenetic_barplot.(png/pdf)  [slategray]")
    print(" - Global_P1P2_cluster_membership.csv")
    print(" - Global_dendrogram_with_P1P2_positions.(png/pdf)")
    print(" - P1P2_only_dendrogram.(png/pdf)")
    print(" - P1P2_only_clustered_heatmap.(png/pdf)")


if __name__ == "__main__":
    main()



# %% Cell 8
# cluster_all_countries_global_clustermap_PUB.py
# ------------------------------------------------------------
# Publication-quality clustermap-style figure:
#   row dendrogram + col dendrogram + heatmap + colorbar
#   + optional row/col cluster strips
#
# Keeps dendrogram/text as vector in PDF by rasterizing only the heatmap.
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Set, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform

from sklearn.metrics import silhouette_score

try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# ----------------------------
# USER SETTINGS
# ----------------------------
DATA = Path("results_part2_external_hierarchy/analysis_table_part2_with_external_indicators.csv")
OUTDIR = Path("figures_country_clustering_global")

# export quality
PNG_DPI = 900          # high-res raster for manuscripts
PDF_DPI = 300          # pdf is vector anyway; dpi mainly affects rasterized heatmap
HEATMAP_RASTER_DPI = 300  # raster resolution inside PDF

COUNTRIES_PART1 = ["Afghanistan", "Madagascar", "Malawi", "Morocco"]
COUNTRIES_PART2 = ["Ethiopia", "Niger", "Morocco", "Senegal", "Benin"]

EXTERNAL_COLS = [
    "gii_latest",
    "gdi_latest",
    "hdi_latest",
    "electoral-democracy-index_latest",
    "liberal-democracy-index_latest",
    "women-political-empowerment-index_latest",
]

PRETTY = {
    "gii_latest": "GII",
    "gdi_latest": "GDI",
    "hdi_latest": "HDI",
    "electoral-democracy-index_latest": "Electoral democracy",
    "liberal-democracy-index_latest": "Liberal democracy",
    "women-political-empowerment-index_latest": "Women political empowerment",
}

FIGTITLE = "All countries: clustering by external hierarchy / gender indicators"

K_MAX = 12
DISTANCE_METRICS = ["euclidean", "cosine", "correlation"]
LINKAGE_METHODS  = ["ward", "average", "complete"]  # ward requires euclidean

# clustermap look
MAKE_CLUSTER_STRIPS = True
CLUSTER_STRIPS_K = 6
VLIM = 3.0                           # symmetric clipping +/- VLIM
CMAP = "RdBu_r"                      # diverging centered on 0 (paper-standard)
SHOW_COUNTRY_NAMES_IF_N_LEQ = 60     # hide row labels when too many countries


# ----------------------------
# Style
# ----------------------------
def set_pub_style():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def save_png_pdf(fig: plt.Figure, outbase: Path):
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), dpi=PDF_DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Helpers
# ----------------------------
def iso3_to_name(iso3: str) -> str:
    iso3 = str(iso3).upper()
    if not HAS_PYCOUNTRY:
        return iso3
    try:
        c = pycountry.countries.get(alpha_3=iso3)
        return c.name if c else iso3
    except Exception:
        return iso3


def name_to_iso3(name: str) -> Optional[str]:
    if not HAS_PYCOUNTRY:
        return None
    overrides: Dict[str, str] = {
        "Morocco": "MAR",
        "Malawi": "MWI",
        "Madagascar": "MDG",
        "Afghanistan": "AFG",
        "Ethiopia": "ETH",
        "Niger": "NER",
        "Senegal": "SEN",
        "Benin": "BEN",
    }
    if name in overrides:
        return overrides[name]
    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None


def build_selected_iso3() -> Tuple[Set[str], Set[str], List[str]]:
    p1_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART1}
    p2_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART2}
    p1_iso3 = {x for x in p1_iso3 if x is not None}
    p2_iso3 = {x for x in p2_iso3 if x is not None}

    union: List[str] = []
    for n in COUNTRIES_PART1:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    for n in COUNTRIES_PART2:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    return p1_iso3, p2_iso3, union


def zscore_by_column(X: pd.DataFrame) -> pd.DataFrame:
    M = X.to_numpy(float)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (M - mu) / sd
    return pd.DataFrame(Z, index=X.index, columns=X.columns)


# ----------------------------
# Sensitivity grid (safe skip invalid combos)
# ----------------------------
@dataclass
class SensRow:
    metric: str
    linkage: str
    k_best: int
    sil_best: float
    coph_corr: float
    n: int


def safe_linkage_optional(D_condensed: np.ndarray, linkage_method: str, metric: str) -> Optional[np.ndarray]:
    if linkage_method == "ward" and metric != "euclidean":
        return None
    try:
        return linkage(D_condensed, method=linkage_method)
    except Exception:
        return None


def silhouette_over_k(
    X: np.ndarray,
    D_condensed: np.ndarray,
    labels_by_k: Dict[int, np.ndarray],
    metric: str,
) -> Dict[int, float]:
    sklearn_supported = {"euclidean", "manhattan", "cosine", "l1", "l2"}
    need_precomputed = metric not in sklearn_supported
    D_square = squareform(D_condensed) if need_precomputed else None

    scores: Dict[int, float] = {}
    n = X.shape[0]

    for k, labels in labels_by_k.items():
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= n:
            continue
        try:
            if need_precomputed:
                scores[k] = float(silhouette_score(D_square, labels, metric="precomputed"))
            else:
                scores[k] = float(silhouette_score(X, labels, metric=metric))
        except Exception:
            continue
    return scores


def run_distance_linkage_sensitivity(Z: pd.DataFrame) -> pd.DataFrame:
    X = Z.to_numpy(float)
    n = X.shape[0]
    k_max = int(min(K_MAX, n - 1))
    rows: List[SensRow] = []

    for metric in DISTANCE_METRICS:
        try:
            D = pdist(X, metric=metric)
        except Exception:
            continue
        for link_m in LINKAGE_METHODS:
            L = safe_linkage_optional(D, link_m, metric)
            if L is None:
                continue
            try:
                coph_corr, _ = cophenet(L, D)
                coph_corr = float(coph_corr)
            except Exception:
                coph_corr = float("nan")

            labels_by_k = {k: fcluster(L, t=k, criterion="maxclust") for k in range(2, k_max + 1)}
            sil_scores = silhouette_over_k(X, D, labels_by_k, metric)
            if len(sil_scores) == 0:
                k_best, sil_best = -1, float("nan")
            else:
                k_best = int(max(sil_scores, key=lambda kk: sil_scores[kk]))
                sil_best = float(sil_scores[k_best])

            rows.append(SensRow(metric, link_m, k_best, sil_best, coph_corr, n))

    if not rows:
        raise RuntimeError("No valid (distance, linkage) combinations evaluated.")

    summary = pd.DataFrame([r.__dict__ for r in rows]).sort_values(
        by=["sil_best", "coph_corr"], ascending=[False, False]
    )
    summary.to_csv(OUTDIR / "Global_Sensitivity_distance_linkage_summary.csv", index=False)
    return summary


# ----------------------------
# Publication-quality clustermap-like plot
# ----------------------------
def _labels_to_rgb_strip(labels: np.ndarray, cmap_name: str = "tab20") -> np.ndarray:
    uniq = list(dict.fromkeys(labels.tolist()))
    cmap = plt.get_cmap(cmap_name)
    color_map = {u: cmap(i % 20)[:3] for i, u in enumerate(uniq)}
    colors = np.array([color_map[int(x)] for x in labels], dtype=float)  # (n,3)
    return colors.reshape(1, -1, 3)


def plot_clustermap_like_pub(
    Z: pd.DataFrame,
    outbase: Path,
    row_metric: str,
    row_linkage: str,
    col_metric: str = "euclidean",
    col_linkage: str = "average",
    row_strip_k: Optional[int] = None,
    col_strip_k: Optional[int] = None,
    title: str = "",
    show_country_names_if_n_leq: int = 60,
    vlim: float = 3.0,
    cmap: str = "RdBu_r",
):
    """
    Z: rows=countries (ISO3), cols=features (recommended z-scored).
    Produces a clean, journal-ready clustermap layout.
    """
    set_pub_style()

    # Linkages
    Xr = Z.to_numpy(float)
    Dr = pdist(Xr, metric=row_metric)
    Lr = safe_linkage_optional(Dr, row_linkage, row_metric)
    if Lr is None:
        raise RuntimeError(f"Invalid row combo: linkage={row_linkage}, metric={row_metric}")

    Xc = Z.to_numpy(float).T
    Dc = pdist(Xc, metric=col_metric)
    Lc = linkage(Dc, method=col_linkage)

    # Orders
    dd_r = dendrogram(Lr, no_plot=True)
    dd_c = dendrogram(Lc, no_plot=True)
    r_order = dd_r["leaves"]
    c_order = dd_c["leaves"]
    Zr = Z.iloc[r_order, c_order]

    # Strips (visual only)
    row_strip = None
    if row_strip_k and row_strip_k >= 2:
        labs = fcluster(Lr, t=int(row_strip_k), criterion="maxclust")
        row_strip = _labels_to_rgb_strip(labs[r_order])
    col_strip = None
    if col_strip_k and col_strip_k >= 2:
        labs = fcluster(Lc, t=int(col_strip_k), criterion="maxclust")
        col_strip = _labels_to_rgb_strip(labs[c_order])

    # Heatmap values
    A = Zr.to_numpy(float)
    if np.isfinite(vlim) and vlim > 0:
        A = np.clip(A, -vlim, vlim)

    # ---- GridSpec layout (stable & clean) ----
    # rows: [col dendro, col strip, heatmap]
    # cols: [row dendro, row strip, heatmap, colorbar]
    fig = plt.figure(figsize=(12.0, 7.6), dpi=300)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=4,
        height_ratios=[1.1, 0.18 if col_strip is not None else 0.001, 6.0],
        width_ratios=[1.6, 0.18 if row_strip is not None else 0.001, 6.5, 0.22],
        wspace=0.02,
        hspace=0.02
    )

    ax_col = fig.add_subplot(gs[0, 2])
    ax_col_strip = fig.add_subplot(gs[1, 2]) if col_strip is not None else None
    ax_row = fig.add_subplot(gs[2, 0])
    ax_row_strip = fig.add_subplot(gs[2, 1]) if row_strip is not None else None
    ax_hm = fig.add_subplot(gs[2, 2])
    ax_cb = fig.add_subplot(gs[2, 3])

    # Dendrograms (slightly thicker lines)
    dendrogram(
        Lc, orientation="top", ax=ax_col, no_labels=True,
        above_threshold_color="black"
    )
    for ln in ax_col.get_lines():
        ln.set_linewidth(1.1)
    ax_col.set_xticks([]); ax_col.set_yticks([])
    ax_col.set_frame_on(False)

    dendrogram(
        Lr, orientation="left", ax=ax_row, no_labels=True,
        above_threshold_color="black"
    )
    for ln in ax_row.get_lines():
        ln.set_linewidth(1.1)
    ax_row.invert_yaxis()
    ax_row.set_xticks([]); ax_row.set_yticks([])
    ax_row.set_frame_on(False)

    # Strips
    if ax_col_strip is not None:
        ax_col_strip.imshow(col_strip, aspect="auto")
        ax_col_strip.set_xticks([]); ax_col_strip.set_yticks([])
        ax_col_strip.set_frame_on(False)

    if ax_row_strip is not None:
        ax_row_strip.imshow(row_strip.transpose(1, 0, 2), aspect="auto")  # (n,1,3)
        ax_row_strip.set_xticks([]); ax_row_strip.set_yticks([])
        ax_row_strip.set_frame_on(False)

    # Heatmap (rasterized -> crisp PDF)
    im = ax_hm.imshow(
        A,
        aspect="auto",
        cmap=cmap,
        vmin=-vlim,
        vmax=vlim,
        interpolation="nearest",
        rasterized=True
    )
    im.set_rasterized(True)
    im.set_resample(False)

    # Labels
    # Y (countries)
    if Zr.shape[0] <= show_country_names_if_n_leq:
        ylab = [f"{iso3_to_name(i)} ({i})" for i in Zr.index.astype(str)]
        ax_hm.set_yticks(np.arange(Zr.shape[0]))
        ax_hm.set_yticklabels(ylab, fontsize=8)
    else:
        ax_hm.set_yticks([])

    # X (features)
    xlab = [PRETTY.get(c, c) for c in Zr.columns]
    ax_hm.set_xticks(np.arange(Zr.shape[1]))
    ax_hm.set_xticklabels(xlab, rotation=90, ha="center", va="top", fontsize=9)

    ax_hm.tick_params(axis="both", length=0)
    ax_hm.set_xlabel("")
    ax_hm.set_ylabel("")

    # Title (use suptitle to avoid overlap with top dendrogram)
    if title:
        fig.suptitle(title, y=0.995)

    # Colorbar
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label(f"z-score (clipped to ±{vlim:g})")
    cb.ax.tick_params(labelsize=9, width=0.8, length=3)

    # If you want the heatmap raster inside PDF at a controlled DPI:
    im.set_rasterized(True)
    im.set_zorder(1)
    ax_hm.set_zorder(2)
    fig.canvas.draw_idle()

    save_png_pdf(fig, outbase)


# ----------------------------
# MAIN
# ----------------------------
def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Cannot find: {DATA.resolve()}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    p1_iso3, p2_iso3, union_iso3 = build_selected_iso3()

    df = pd.read_csv(DATA)
    if "iso3" not in df.columns:
        raise ValueError("CSV must contain an 'iso3' column.")
    df["iso3"] = df["iso3"].astype(str).str.upper()
    df = df.set_index("iso3")

    missing = [c for c in EXTERNAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required external columns: {missing}")

    X_all = df[EXTERNAL_COLS].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if X_all.shape[0] < 3:
        raise ValueError("Too few countries after dropna on external indicators.")

    Z_all = zscore_by_column(X_all)

    # Choose distance/linkage via sensitivity
    summary = run_distance_linkage_sensitivity(Z_all)
    best = summary.iloc[0]
    best_metric = str(best["metric"])
    best_linkage = str(best["linkage"])

    # --- ALL countries (labels likely hidden; still publication-ready) ---
    plot_clustermap_like_pub(
        Z=Z_all,
        outbase=OUTDIR / "AllCountries_clustermap_PUB",
        row_metric=best_metric,
        row_linkage=best_linkage,
        col_metric="euclidean",
        col_linkage="average",
        row_strip_k=(CLUSTER_STRIPS_K if MAKE_CLUSTER_STRIPS else None),
        col_strip_k=(CLUSTER_STRIPS_K if MAKE_CLUSTER_STRIPS else None),
        title=f"{FIGTITLE}\n(distance={best_metric}, linkage={best_linkage})",
        show_country_names_if_n_leq=SHOW_COUNTRY_NAMES_IF_N_LEQ,
        vlim=VLIM,
        cmap=CMAP,
    )

    # --- P1/P2 only (always readable labels) ---
    keep = [c for c in union_iso3 if c in Z_all.index]
    if len(keep) >= 3:
        plot_clustermap_like_pub(
            Z=Z_all.loc[keep].copy(),
            outbase=OUTDIR / "P1P2_only_clustermap_PUB",
            row_metric=best_metric,
            row_linkage=best_linkage,
            col_metric="euclidean",
            col_linkage="average",
            row_strip_k=min(4, len(keep)) if MAKE_CLUSTER_STRIPS else None,
            col_strip_k=min(4, Z_all.shape[1]) if MAKE_CLUSTER_STRIPS else None,
            title=f"P1/P2-only clustermap\n(distance={best_metric}, linkage={best_linkage})",
            show_country_names_if_n_leq=999,
            vlim=VLIM,
            cmap=CMAP,
        )

    print("[OK] Saved into:", OUTDIR.resolve())
    print(" - AllCountries_clustermap_PUB.(png/pdf)")
    print(" - P1P2_only_clustermap_PUB.(png/pdf)")


if __name__ == "__main__":
    main()



# %% Cell 9
# cluster_all_countries_global_dendrograms_bestMetric_PUB.py
# ------------------------------------------------------------
# Publication-quality dendrograms using the BEST (distance × linkage)
# found by the sensitivity grid.
#
# Outputs:
#  - AllCountries_dendrogram_bestMetric.(png/pdf)
#  - P1P2_only_dendrogram_bestMetric.(png/pdf)
#
# Notes:
#  - Dendrogram branch lengths are in the chosen distance units
#    (euclidean/cosine/correlation) computed on the z-scored feature matrix.
#  - Colormap (blue/red) is for heatmaps, not dendrograms.
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Set, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform

from sklearn.metrics import silhouette_score

try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# ----------------------------
# USER SETTINGS
# ----------------------------
DATA = Path("results_part2_external_hierarchy/analysis_table_part2_with_external_indicators.csv")
OUTDIR = Path("figures_country_clustering_global")

# export quality
PNG_DPI = 900
PDF_DPI = 300

COUNTRIES_PART1 = ["Afghanistan", "Madagascar", "Malawi", "Morocco"]
COUNTRIES_PART2 = ["Ethiopia", "Niger", "Morocco", "Senegal", "Benin"]

EXTERNAL_COLS = [
    "gii_latest",
    "gdi_latest",
    "hdi_latest",
    "electoral-democracy-index_latest",
    "liberal-democracy-index_latest",
    "women-political-empowerment-index_latest",
]

PRETTY = {
    "gii_latest": "GII",
    "gdi_latest": "GDI",
    "hdi_latest": "HDI",
    "electoral-democracy-index_latest": "Electoral democracy",
    "liberal-democracy-index_latest": "Liberal democracy",
    "women-political-empowerment-index_latest": "Women political empowerment",
}

FIGTITLE = "All countries: clustering by external hierarchy / gender indicators"

K_MAX = 12
DISTANCE_METRICS = ["euclidean", "cosine", "correlation"]
LINKAGE_METHODS  = ["ward", "average", "complete"]  # ward requires euclidean

# Dendrogram look
DENDRO_ORIENTATION_ALL = "left"   # best when there are many labels
DENDRO_ORIENTATION_SUB = "top"    # nice for small subset
MAX_LABELS_ALL = 40              # auto-hide labels if too many rows
LEAF_FONTSIZE_ALL = 8
LEAF_FONTSIZE_SUB = 10


# ----------------------------
# Style
# ----------------------------
def set_pub_style():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def save_png_pdf(fig: plt.Figure, outbase: Path):
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), dpi=PDF_DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Helpers
# ----------------------------
def iso3_to_name(iso3: str) -> str:
    iso3 = str(iso3).upper()
    if not HAS_PYCOUNTRY:
        return iso3
    try:
        c = pycountry.countries.get(alpha_3=iso3)
        return c.name if c else iso3
    except Exception:
        return iso3


def name_to_iso3(name: str) -> Optional[str]:
    if not HAS_PYCOUNTRY:
        return None
    overrides: Dict[str, str] = {
        "Morocco": "MAR",
        "Malawi": "MWI",
        "Madagascar": "MDG",
        "Afghanistan": "AFG",
        "Ethiopia": "ETH",
        "Niger": "NER",
        "Senegal": "SEN",
        "Benin": "BEN",
    }
    if name in overrides:
        return overrides[name]
    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None


def build_selected_iso3() -> Tuple[Set[str], Set[str], List[str]]:
    p1_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART1}
    p2_iso3 = {name_to_iso3(n) for n in COUNTRIES_PART2}
    p1_iso3 = {x for x in p1_iso3 if x is not None}
    p2_iso3 = {x for x in p2_iso3 if x is not None}

    union: List[str] = []
    for n in COUNTRIES_PART1:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    for n in COUNTRIES_PART2:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    return p1_iso3, p2_iso3, union


def zscore_by_column(X: pd.DataFrame) -> pd.DataFrame:
    M = X.to_numpy(float)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (M - mu) / sd
    return pd.DataFrame(Z, index=X.index, columns=X.columns)


# ----------------------------
# Sensitivity grid (safe skip invalid combos)
# ----------------------------
@dataclass
class SensRow:
    metric: str
    linkage: str
    k_best: int
    sil_best: float
    coph_corr: float
    n: int


def safe_linkage_optional(D_condensed: np.ndarray, linkage_method: str, metric: str) -> Optional[np.ndarray]:
    # during grid search: skip invalid ward combos
    if linkage_method == "ward" and metric != "euclidean":
        return None
    try:
        return linkage(D_condensed, method=linkage_method)
    except Exception:
        return None


def linkage_strict(D_condensed: np.ndarray, linkage_method: str, metric: str) -> np.ndarray:
    # for final plots: enforce SciPy rule
    if linkage_method == "ward" and metric != "euclidean":
        raise ValueError("ward linkage requires euclidean distance.")
    return linkage(D_condensed, method=linkage_method)


def silhouette_over_k(
    X: np.ndarray,
    D_condensed: np.ndarray,
    labels_by_k: Dict[int, np.ndarray],
    metric: str,
) -> Dict[int, float]:
    sklearn_supported = {"euclidean", "manhattan", "cosine", "l1", "l2"}
    need_precomputed = metric not in sklearn_supported
    D_square = squareform(D_condensed) if need_precomputed else None

    scores: Dict[int, float] = {}
    n = X.shape[0]

    for k, labels in labels_by_k.items():
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= n:
            continue
        try:
            if need_precomputed:
                scores[k] = float(silhouette_score(D_square, labels, metric="precomputed"))
            else:
                scores[k] = float(silhouette_score(X, labels, metric=metric))
        except Exception:
            continue
    return scores


def run_distance_linkage_sensitivity(Z: pd.DataFrame) -> pd.DataFrame:
    X = Z.to_numpy(float)
    n = X.shape[0]
    k_max = int(min(K_MAX, n - 1))
    rows: List[SensRow] = []

    for metric in DISTANCE_METRICS:
        try:
            D = pdist(X, metric=metric)
        except Exception:
            continue

        for link_m in LINKAGE_METHODS:
            L = safe_linkage_optional(D, link_m, metric)
            if L is None:
                continue

            try:
                coph_corr, _ = cophenet(L, D)
                coph_corr = float(coph_corr)
            except Exception:
                coph_corr = float("nan")

            labels_by_k = {k: fcluster(L, t=k, criterion="maxclust") for k in range(2, k_max + 1)}
            sil_scores = silhouette_over_k(X, D, labels_by_k, metric)

            if len(sil_scores) == 0:
                k_best, sil_best = -1, float("nan")
            else:
                k_best = int(max(sil_scores, key=lambda kk: sil_scores[kk]))
                sil_best = float(sil_scores[k_best])

            rows.append(SensRow(metric, link_m, k_best, sil_best, coph_corr, n))

    if not rows:
        raise RuntimeError("No valid (distance, linkage) combinations evaluated.")

    summary = pd.DataFrame([r.__dict__ for r in rows]).sort_values(
        by=["sil_best", "coph_corr"], ascending=[False, False]
    )
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTDIR / "Global_Sensitivity_distance_linkage_summary.csv", index=False)
    return summary


# ----------------------------
# Dendrogram plotting (best metric/linkage)
# ----------------------------
def plot_country_dendrogram_best(
    Z: pd.DataFrame,
    outbase: Path,
    metric: str,
    linkage_method: str,
    title: str,
    orientation: str,
    label_mode: str = "auto",  # auto|all|none
    max_labels: int = 40,
    leaf_font_size: int = 9,
):
    if Z.shape[0] < 3:
        raise ValueError("Need at least 3 countries for a dendrogram.")

    set_pub_style()

    X = Z.to_numpy(float)
    D = pdist(X, metric=metric)
    L = linkage_strict(D, linkage_method, metric)

    show_labels = (label_mode == "all") or (label_mode == "auto" and Z.shape[0] <= max_labels)
    labels = [f"{iso3_to_name(i)} ({i})" for i in Z.index.astype(str)] if show_labels else None

    fig = plt.figure(figsize=(11.0, 7.0), dpi=300)
    ax = fig.add_subplot(111)

    dendrogram(
        L,
        labels=labels,
        orientation=orientation,
        leaf_rotation=0 if orientation in {"left", "right"} else 25,
        leaf_font_size=leaf_font_size,
        above_threshold_color="black",
        ax=ax,
    )

    # cosmetic: slightly thicker lines
    for ln in ax.get_lines():
        ln.set_linewidth(1.1)

    ax.set_title(title)
    ax.set_xlabel(f"Distance ({metric})")
    if not show_labels:
        ax.set_ylabel("(labels hidden)")
    fig.tight_layout()
    save_png_pdf(fig, outbase)

    return L


# ----------------------------
# MAIN
# ----------------------------
def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Cannot find: {DATA.resolve()}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # load + prep
    df = pd.read_csv(DATA)
    if "iso3" not in df.columns:
        raise ValueError("CSV must contain an 'iso3' column.")
    df["iso3"] = df["iso3"].astype(str).str.upper()
    df = df.set_index("iso3")

    missing = [c for c in EXTERNAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required external columns: {missing}")

    X_all = df[EXTERNAL_COLS].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if X_all.shape[0] < 3:
        raise ValueError("Too few countries after dropna on external indicators.")

    # z-score for fair distance computation across different-feature scales
    Z_all = zscore_by_column(X_all)

    # best metric/linkage from sensitivity
    summary = run_distance_linkage_sensitivity(Z_all)
    best = summary.iloc[0]
    best_metric = str(best["metric"])
    best_linkage = str(best["linkage"])

    # dendrogram: ALL countries (best metric)
    plot_country_dendrogram_best(
        Z=Z_all,
        outbase=OUTDIR / "AllCountries_dendrogram_bestMetric",
        metric=best_metric,
        linkage_method=best_linkage,
        title=f"{FIGTITLE}\nCountry dendrogram (best: distance={best_metric}, linkage={best_linkage})",
        orientation=DENDRO_ORIENTATION_ALL,
        label_mode="auto",
        max_labels=MAX_LABELS_ALL,
        leaf_font_size=LEAF_FONTSIZE_ALL,
    )

    # dendrogram: P1/P2-only subset (Article 1+2 countries)
    p1_iso3, p2_iso3, union_iso3 = build_selected_iso3()
    keep = [c for c in union_iso3 if c in Z_all.index]
    if len(keep) >= 3:
        Z_sub = Z_all.loc[keep].copy()
        plot_country_dendrogram_best(
            Z=Z_sub,
            outbase=OUTDIR / "P1P2_only_dendrogram_bestMetric",
            metric=best_metric,
            linkage_method=best_linkage,
            title=f"P1/P2-only country dendrogram\n(best: distance={best_metric}, linkage={best_linkage})",
            orientation=DENDRO_ORIENTATION_SUB,
            label_mode="all",
            max_labels=999,
            leaf_font_size=LEAF_FONTSIZE_SUB,
        )

    print("[OK] Dendrograms saved in:", OUTDIR.resolve())
    print(" - AllCountries_dendrogram_bestMetric.(png/pdf)")
    print(" - P1P2_only_dendrogram_bestMetric.(png/pdf)")


if __name__ == "__main__":
    main()



# %% Cell 10
# cluster_all_countries_metric_similarity_heatmap_PUB.py
# ------------------------------------------------------------
# Publication-quality CLUSTERED HEATMAP of METRIC–METRIC similarity
# (correlation between indicators across countries)
#
# Outputs:
#  - AllCountries_metricSimilarity_clustermap.(png/pdf)
#  - P1P2_only_metricSimilarity_clustermap.(png/pdf)
#
# Heatmap values:
#  - reminder: these are correlations BETWEEN METRICS (columns),
#    computed across countries (rows).
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Set, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# ----------------------------
# USER SETTINGS
# ----------------------------
DATA = Path("results_part2_external_hierarchy/analysis_table_part2_with_external_indicators.csv")
OUTDIR = Path("figures_country_clustering_global")

# export quality
PNG_DPI = 900
PDF_DPI = 300

COUNTRIES_PART1 = ["Afghanistan", "Madagascar", "Malawi", "Morocco"]
COUNTRIES_PART2 = ["Ethiopia", "Niger", "Morocco", "Senegal", "Benin"]

EXTERNAL_COLS = [
    "gii_latest",
    "gdi_latest",
    "hdi_latest",
    "electoral-democracy-index_latest",
    "liberal-democracy-index_latest",
    "women-political-empowerment-index_latest",
]

PRETTY = {
    "gii_latest": "GII",
    "gdi_latest": "GDI",
    "hdi_latest": "HDI",
    "electoral-democracy-index_latest": "Electoral democracy",
    "liberal-democracy-index_latest": "Liberal democracy",
    "women-political-empowerment-index_latest": "Women political empowerment",
}

FIGTITLE_ALL = "Metric similarity (correlation across countries) — all countries"
FIGTITLE_SUB = "Metric similarity (correlation across countries) — P1/P2 only"

# similarity choice: "pearson" or "spearman"
SIMILARITY = "pearson"

# clustering on metrics
LINKAGE_METHOD = "average"  # "average", "complete", "single", "ward"(not valid with correlation distance)

# correlation-to-distance mapping
# distance = 1 - corr   (works for correlation in [-1,1], distance in [0,2])
DIST_FROM_CORR = "1-corr"  # keep as is unless you want e.g. "sqrt(0.5*(1-corr))"


# ----------------------------
# Style
# ----------------------------
def set_pub_style():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def save_png_pdf(fig: plt.Figure, outbase: Path):
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), dpi=PDF_DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Country name helpers
# ----------------------------
def name_to_iso3(name: str) -> Optional[str]:
    if not HAS_PYCOUNTRY:
        return None
    overrides: Dict[str, str] = {
        "Morocco": "MAR",
        "Malawi": "MWI",
        "Madagascar": "MDG",
        "Afghanistan": "AFG",
        "Ethiopia": "ETH",
        "Niger": "NER",
        "Senegal": "SEN",
        "Benin": "BEN",
    }
    if name in overrides:
        return overrides[name]
    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None


def build_selected_iso3() -> List[str]:
    # Preserve order, avoid duplicates
    union: List[str] = []
    for n in COUNTRIES_PART1 + COUNTRIES_PART2:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    return union


# ----------------------------
# Similarity + clustering
# ----------------------------
def corr_matrix_across_countries(X: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    X: rows=countries, cols=metrics
    returns: metric x metric correlation matrix
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError("SIMILARITY must be 'pearson' or 'spearman'.")
    C = X.corr(method=method, min_periods=max(3, int(0.6 * X.shape[0])))
    return C


def corr_to_distance(C: pd.DataFrame, mode: str) -> pd.DataFrame:
    A = C.to_numpy(float)
    if mode == "1-corr":
        D = 1.0 - A
    elif mode == "sqrt(0.5*(1-corr))":
        D = np.sqrt(0.5 * (1.0 - A))
    else:
        raise ValueError("Unknown DIST_FROM_CORR mode.")
    np.fill_diagonal(D, 0.0)
    # numeric safety
    D = np.clip(D, 0.0, np.nanmax(D[np.isfinite(D)]) if np.isfinite(D).any() else 0.0)
    return pd.DataFrame(D, index=C.index, columns=C.columns)


def clustered_order_from_distance(D: pd.DataFrame, linkage_method: str) -> List[str]:
    if linkage_method == "ward":
        raise ValueError("ward linkage is not appropriate for correlation-derived distances here.")
    # condensed distance for scipy
    D_cond = squareform(D.to_numpy(float), checks=False)
    L = linkage(D_cond, method=linkage_method)
    leaves = dendrogram(L, no_plot=True)["leaves"]
    labels = list(D.index)
    return [labels[i] for i in leaves]


# ----------------------------
# Plot
# ----------------------------
def plot_metric_similarity_clustermap(
    X_country_by_metric: pd.DataFrame,
    outbase: Path,
    title: str,
):
    """
    Heatmap is metric x metric correlation.
    Dendrogram clusters metrics using distance derived from correlation.
    """
    set_pub_style()

    # 1) metric similarity
    C = corr_matrix_across_countries(X_country_by_metric, method=SIMILARITY).dropna(axis=0, how="any").dropna(axis=1, how="any")
    if C.shape[0] < 3:
        raise ValueError("Need at least 3 metrics with valid correlations to cluster/plot.")

    # 2) metric clustering order
    D = corr_to_distance(C, mode=DIST_FROM_CORR)
    order = clustered_order_from_distance(D, linkage_method=LINKAGE_METHOD)

    # 3) reorder for square heatmap
    C_ord = C.loc[order, order]

    # pretty labels
    pretty = [PRETTY.get(c, c) for c in C_ord.columns]
    C_plot = C_ord.copy()
    C_plot.index = pretty
    C_plot.columns = pretty

    # 4) figure layout: top dendrogram + left dendrogram + heatmap + colorbar
    # Manual layout keeps it consistent (publication-style)
    fig = plt.figure(figsize=(10.5, 8.2), dpi=300)

    # axes positions (left, bottom, width, height)
    ax_dend_top  = fig.add_axes([0.24, 0.84, 0.62, 0.12])
    ax_dend_left = fig.add_axes([0.08, 0.22, 0.14, 0.62])
    ax_heat      = fig.add_axes([0.24, 0.22, 0.62, 0.62])
    ax_cbar      = fig.add_axes([0.88, 0.22, 0.03, 0.62])

    # dendrograms (same linkage used for both axes)
    D_cond = squareform(D.loc[order, order].to_numpy(float), checks=False)
    L = linkage(D_cond, method=LINKAGE_METHOD)

    dendrogram(
        L,
        ax=ax_dend_top,
        orientation="top",
        no_labels=True,
        color_threshold=None,
        above_threshold_color="black",
    )
    ax_dend_top.set_xticks([])
    ax_dend_top.set_yticks([])
    for s in ax_dend_top.spines.values():
        s.set_visible(False)

    dendrogram(
        L,
        ax=ax_dend_left,
        orientation="left",
        no_labels=True,
        color_threshold=None,
        above_threshold_color="black",
    )
    ax_dend_left.set_xticks([])
    ax_dend_left.set_yticks([])
    for s in ax_dend_left.spines.values():
        s.set_visible(False)

    # heatmap: correlation in [-1, 1]
    im = ax_heat.imshow(C_plot.to_numpy(float), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax_heat.set_xticks(np.arange(C_plot.shape[1]))
    ax_heat.set_yticks(np.arange(C_plot.shape[0]))
    ax_heat.set_xticklabels(C_plot.columns, rotation=90, ha="center")
    ax_heat.set_yticklabels(C_plot.index)

    # title
    fig.suptitle(f"{title}\n(similarity={SIMILARITY}, linkage={LINKAGE_METHOD})", y=0.98)

    # colorbar
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Correlation (metric–metric)")

    save_png_pdf(fig, outbase)


# ----------------------------
# MAIN
# ----------------------------
def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Cannot find: {DATA.resolve()}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    if "iso3" not in df.columns:
        raise ValueError("CSV must contain an 'iso3' column.")
    df["iso3"] = df["iso3"].astype(str).str.upper()
    df = df.set_index("iso3")

    missing = [c for c in EXTERNAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required external columns: {missing}")

    X_all = df[EXTERNAL_COLS].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if X_all.shape[0] < 5:
        raise ValueError("Too few countries after dropna; need more rows for stable correlations.")

    # ---- ALL countries metric similarity ----
    plot_metric_similarity_clustermap(
        X_country_by_metric=X_all,
        outbase=OUTDIR / "AllCountries_metricSimilarity_clustermap",
        title=FIGTITLE_ALL,
    )

    # ---- P1/P2 subset metric similarity ----
    union_iso3 = build_selected_iso3()
    keep = [c for c in union_iso3 if c in X_all.index]
    if len(keep) >= 3:
        X_sub = X_all.loc[keep].copy()
        # still need enough countries to compute correlation robustly
        if X_sub.shape[0] >= 5:
            plot_metric_similarity_clustermap(
                X_country_by_metric=X_sub,
                outbase=OUTDIR / "P1P2_only_metricSimilarity_clustermap",
                title=FIGTITLE_SUB,
            )
        else:
            print("[WARN] P1/P2 subset has <5 countries after cleaning; skipping subset plot.")
    else:
        print("[WARN] Fewer than 3 P1/P2 countries found in data; skipping subset plot.")

    print("[OK] Saved in:", OUTDIR.resolve())
    print(" - AllCountries_metricSimilarity_clustermap.(png/pdf)")
    print(" - P1P2_only_metricSimilarity_clustermap.(png/pdf)  (if generated)")


if __name__ == "__main__":
    main()



# %% Cell 11
# cluster_all_countries_country_similarity_2panel_PUB.py
# ------------------------------------------------------------
# EXACT 2-panel layout like your attached figure:
#   (A) P1/P2 subset (few countries) : top+left dendrogram
#   (B) All countries (many)         : bottom+right dendrogram
#
# Heatmap values are NOT z-scores per metric.
# They are COUNTRY–COUNTRY similarity (correlation) or dissimilarity (1-corr).
#
# Output:
#   - CountrySimilarity_2panel.(png/pdf)
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# ----------------------------
# USER SETTINGS
# ----------------------------
DATA = Path("results_part2_external_hierarchy/analysis_table_part2_with_external_indicators.csv")
OUTDIR = Path("figures_country_clustering_global")

PNG_DPI = 900
PDF_DPI = 300

COUNTRIES_PART1 = ["Afghanistan", "Madagascar", "Malawi", "Morocco"]
COUNTRIES_PART2 = ["Ethiopia", "Niger", "Morocco", "Senegal", "Benin"]

EXTERNAL_COLS = [
    "gii_latest",
    "gdi_latest",
    "hdi_latest",
    "electoral-democracy-index_latest",
    "liberal-democracy-index_latest",
    "women-political-empowerment-index_latest",
]

# Choose what the heatmap colors represent:
#   "correlation"  -> values in [-1, 1] (recommended with RdBu_r)
#   "dissimilarity"-> values in [0, 2]  where dissimilarity = 1 - corr
HEATMAP_MODE = "correlation"  # or "dissimilarity"

# Correlation type between countries (based on their 6-feature vectors)
CORR_METHOD = "pearson"  # "pearson" or "spearman"

# Linkage on distance derived from correlation
LINKAGE_METHOD = "average"  # "average", "complete", "single"

# If you want ISO3 labels vs full country names:
LABEL_STYLE = "name"  # "name" or "iso3"

# Make panel B labels lighter by hiding if too many
MAX_LABELS_PANEL_B = 60

FIGTITLE = "Country clustering by external hierarchy / gender indicators"


# ----------------------------
# Style
# ----------------------------
def set_pub_style():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def save_png_pdf(fig: plt.Figure, outbase: Path):
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), dpi=PDF_DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Country helpers
# ----------------------------
def iso3_to_name(iso3: str) -> str:
    iso3 = str(iso3).upper()
    if not HAS_PYCOUNTRY:
        return iso3
    try:
        c = pycountry.countries.get(alpha_3=iso3)
        return c.name if c else iso3
    except Exception:
        return iso3


def name_to_iso3(name: str) -> Optional[str]:
    if not HAS_PYCOUNTRY:
        return None
    overrides: Dict[str, str] = {
        "Morocco": "MAR",
        "Malawi": "MWI",
        "Madagascar": "MDG",
        "Afghanistan": "AFG",
        "Ethiopia": "ETH",
        "Niger": "NER",
        "Senegal": "SEN",
        "Benin": "BEN",
    }
    if name in overrides:
        return overrides[name]
    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None


def build_selected_iso3() -> List[str]:
    union: List[str] = []
    for n in COUNTRIES_PART1 + COUNTRIES_PART2:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    return union


def make_labels(index_iso3: List[str]) -> List[str]:
    if LABEL_STYLE == "iso3":
        return [str(x) for x in index_iso3]
    return [iso3_to_name(x) for x in index_iso3]


# ----------------------------
# Similarity matrix (country-country)
# ----------------------------
def country_country_corr(X: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    X: rows=countries, cols=metrics
    returns: corr between countries (rows) based on their feature vectors.
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError("CORR_METHOD must be 'pearson' or 'spearman'.")
    # corr across columns -> we need corr between rows
    C = X.T.corr(method=method)
    return C


def corr_to_distance(C: pd.DataFrame) -> np.ndarray:
    """
    distance = 1 - corr
    """
    A = C.to_numpy(float)
    D = 1.0 - A
    np.fill_diagonal(D, 0.0)
    # numeric safety
    D = np.clip(D, 0.0, 2.0)
    return D


def cluster_order_from_corr(C: pd.DataFrame) -> List[int]:
    D = corr_to_distance(C)
    D_cond = squareform(D, checks=False)
    L = linkage(D_cond, method=LINKAGE_METHOD)
    leaves = dendrogram(L, no_plot=True)["leaves"]
    return leaves


# ----------------------------
# Panel plotting (manual geometry like your attached)
# ----------------------------
def plot_panel(
    fig: plt.Figure,
    X: pd.DataFrame,
    panel_tag: str,
    mode: str,
    layout: str,
    title: str,
    show_labels: bool,
):
    """
    layout:
      - "A": top dendrogram + left dendrogram
      - "B": bottom dendrogram + right dendrogram
    """
    C = country_country_corr(X, method=CORR_METHOD)
    if C.shape[0] < 3:
        raise ValueError("Need at least 3 countries for a dendrogram.")

    leaves = cluster_order_from_corr(C)
    C_ord = C.iloc[leaves, leaves]
    labels = make_labels(list(C_ord.index.astype(str)))

    # choose heatmap values
    if mode == "correlation":
        H = C_ord.to_numpy(float)
        vmin, vmax = -1.0, 1.0
        cmap = "RdBu_r"
        cbar_label = "Correlation"
    elif mode == "dissimilarity":
        H = (1.0 - C_ord.to_numpy(float))
        vmin, vmax = 0.0, 2.0
        cmap = "RdBu_r"
        cbar_label = "Dissimilarity (1 - corr)"
    else:
        raise ValueError("HEATMAP_MODE must be 'correlation' or 'dissimilarity'.")

    # linkage again for dendrogram drawing (same order)
    D = corr_to_distance(C_ord)
    L = linkage(squareform(D, checks=False), method=LINKAGE_METHOD)

    if layout == "A":
        # compact panel like your (A)
        ax_top  = fig.add_axes([0.18, 0.82, 0.53, 0.10])
        ax_left = fig.add_axes([0.06, 0.52, 0.12, 0.30])
        ax_heat = fig.add_axes([0.18, 0.52, 0.53, 0.30])
        ax_cbar = fig.add_axes([0.73, 0.52, 0.02, 0.30])

        dendrogram(L, ax=ax_top, orientation="top", no_labels=True,
                   color_threshold=None, above_threshold_color="black")
        dendrogram(L, ax=ax_left, orientation="left", no_labels=True,
                   color_threshold=None, above_threshold_color="black")

        for ax in (ax_top, ax_left):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)

        im = ax_heat.imshow(H, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax_heat.set_xticks(np.arange(H.shape[1]))
        ax_heat.set_yticks(np.arange(H.shape[0]))

        if show_labels:
            ax_heat.set_xticklabels(labels, rotation=90, ha="center")
            ax_heat.set_yticklabels(labels)
        else:
            ax_heat.set_xticklabels([])
            ax_heat.set_yticklabels([])

        ax_heat.text(-0.14, 1.08, f"({panel_tag})", transform=ax_heat.transAxes,
                     fontsize=12, fontweight="bold", va="bottom")
        ax_heat.set_title(title, pad=8)

        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.set_label(cbar_label)

    elif layout == "B":
        # large panel like your (B): dendrogram on right + bottom
        ax_heat  = fig.add_axes([0.06, 0.08, 0.64, 0.36])
        ax_right = fig.add_axes([0.70, 0.08, 0.12, 0.36])
        ax_bot   = fig.add_axes([0.06, 0.44, 0.64, 0.10])
        ax_cbar  = fig.add_axes([0.84, 0.08, 0.02, 0.36])

        dendrogram(L, ax=ax_right, orientation="right", no_labels=True,
                   color_threshold=None, above_threshold_color="black")
        dendrogram(L, ax=ax_bot, orientation="bottom", no_labels=True,
                   color_threshold=None, above_threshold_color="black")

        for ax in (ax_right, ax_bot):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)

        im = ax_heat.imshow(H, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax_heat.set_xticks(np.arange(H.shape[1]))
        ax_heat.set_yticks(np.arange(H.shape[0]))

        if show_labels:
            ax_heat.set_xticklabels(labels, rotation=90, ha="center")
            ax_heat.set_yticklabels(labels)
        else:
            ax_heat.set_xticklabels([])
            ax_heat.set_yticklabels([])

        ax_heat.text(-0.06, 1.06, f"({panel_tag})", transform=ax_heat.transAxes,
                     fontsize=12, fontweight="bold", va="bottom")
        ax_heat.set_title(title, pad=8)

        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.set_label(cbar_label)

    else:
        raise ValueError("layout must be 'A' or 'B'.")


# ----------------------------
# MAIN
# ----------------------------
def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Cannot find: {DATA.resolve()}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_pub_style()

    df = pd.read_csv(DATA)
    if "iso3" not in df.columns:
        raise ValueError("CSV must contain an 'iso3' column.")
    df["iso3"] = df["iso3"].astype(str).str.upper()
    df = df.set_index("iso3")

    missing = [c for c in EXTERNAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required external columns: {missing}")

    X_all = df[EXTERNAL_COLS].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if X_all.shape[0] < 3:
        raise ValueError("Too few countries after cleaning.")

    # subset
    union_iso3 = build_selected_iso3()
    keep = [c for c in union_iso3 if c in X_all.index]
    if len(keep) < 3:
        raise ValueError("P1/P2 subset has fewer than 3 countries present in the data.")
    X_sub = X_all.loc[keep].copy()

    # decide whether to show labels in big panel
    show_labels_B = (X_all.shape[0] <= MAX_LABELS_PANEL_B)

    fig = plt.figure(figsize=(10.0, 6.0), dpi=300)

    plot_panel(
        fig=fig,
        X=X_sub,
        panel_tag="A",
        mode=HEATMAP_MODE,
        layout="A",
        title="P1/P2-only countries (country–country similarity)",
        show_labels=True,
    )

    plot_panel(
        fig=fig,
        X=X_all,
        panel_tag="B",
        mode=HEATMAP_MODE,
        layout="B",
        title="All countries (country–country similarity)",
        show_labels=show_labels_B,
    )

    fig.suptitle(f"{FIGTITLE}\n(mode={HEATMAP_MODE}, corr={CORR_METHOD}, linkage={LINKAGE_METHOD})", y=0.98)
    outbase = OUTDIR / "CountrySimilarity_2panel"
    save_png_pdf(fig, outbase)

    print("[OK] Saved:", outbase.with_suffix(".png"))
    print("[OK] Saved:", outbase.with_suffix(".pdf"))


if __name__ == "__main__":
    main()



# %% Cell 12
# cluster_P1P2_two_clustermaps_similarity_PUB.py
# ------------------------------------------------------------
# Creates TWO publication-quality clustermaps for ONLY P1+P2 countries:
#
# (1) Rectangular: rows=countries, cols=measures, but values are
#     "measure–measure similarity profile" (same across rows).
#     Layout matches your request: measures in X (with dendrogram),
#     countries in Y (with dendrogram), heatmap values derived from
#     measure–measure similarity (not z-scores).
#
# (2) Square: country–country similarity (correlation or dissimilarity),
#     with dendrograms on rows/cols.
#
# Outputs:
#   - P1P2_Rectangular_metricSimilarityProfile.(png/pdf)
#   - P1P2_CountrySimilarity.(png/pdf)
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# ----------------------------
# USER SETTINGS
# ----------------------------
DATA = Path("results_part2_external_hierarchy/analysis_table_part2_with_external_indicators.csv")
OUTDIR = Path("figures_country_clustering_global_final")

# export quality
PNG_DPI = 900
PDF_DPI = 300

COUNTRIES_PART1 = ["Afghanistan", "Madagascar", "Malawi", "Morocco"]
COUNTRIES_PART2 = ["Ethiopia", "Niger", "Morocco", "Senegal", "Benin"]

EXTERNAL_COLS = [
    "gii_latest",
    "gdi_latest",
    "hdi_latest",
    "electoral-democracy-index_latest",
    "liberal-democracy-index_latest",
    "women-political-empowerment-index_latest",
]

PRETTY = {
    "gii_latest": "GII",
    "gdi_latest": "GDI",
    "hdi_latest": "HDI",
    "electoral-democracy-index_latest": "Electoral democracy",
    "liberal-democracy-index_latest": "Liberal democracy",
    "women-political-empowerment-index_latest": "Women political empowerment",
}

# Similarity choice
CORR_METHOD = "pearson"  # "pearson" or "spearman"

# Heatmap mode for BOTH figures:
#   "correlation"  -> values in [-1, 1]
#   "dissimilarity"-> values in [0, 2]  where dissimilarity = 1 - corr
HEATMAP_MODE = "correlation"

# Heatmap colormap (examples: "RdBu_r", "twilight", "twilight_shifted", "coolwarm")
HEATMAP_CMAP = "twilight_shifted"  # <-- change this to whatever you want

# Linkage method (on distance derived from correlation)
LINKAGE_METHOD = "average"  # "average", "complete", "single"

# Labels:
LABEL_STYLE = "name"  # "name" or "iso3"


# ----------------------------
# Style
# ----------------------------
def set_pub_style():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def save_png_pdf(fig: plt.Figure, outbase: Path):
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), dpi=PDF_DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Country helpers
# ----------------------------
def iso3_to_name(iso3: str) -> str:
    iso3 = str(iso3).upper()
    if not HAS_PYCOUNTRY:
        return iso3
    try:
        c = pycountry.countries.get(alpha_3=iso3)
        return c.name if c else iso3
    except Exception:
        return iso3


def name_to_iso3(name: str) -> Optional[str]:
    if not HAS_PYCOUNTRY:
        return None
    overrides: Dict[str, str] = {
        "Morocco": "MAR",
        "Malawi": "MWI",
        "Madagascar": "MDG",
        "Afghanistan": "AFG",
        "Ethiopia": "ETH",
        "Niger": "NER",
        "Senegal": "SEN",
        "Benin": "BEN",
    }
    if name in overrides:
        return overrides[name]
    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None


def build_selected_iso3() -> List[str]:
    union: List[str] = []
    for n in COUNTRIES_PART1 + COUNTRIES_PART2:
        code = name_to_iso3(n)
        if code and code not in union:
            union.append(code)
    return union


def country_labels(index_iso3: List[str]) -> List[str]:
    if LABEL_STYLE == "iso3":
        return [str(x) for x in index_iso3]
    return [iso3_to_name(x) for x in index_iso3]


# ----------------------------
# Core math
# ----------------------------
def corr_country_country(X: pd.DataFrame) -> pd.DataFrame:
    # correlation between countries (rows) based on feature vectors (columns)
    return X.T.corr(method=CORR_METHOD)


def corr_metric_metric(X: pd.DataFrame) -> pd.DataFrame:
    # correlation between metrics (columns) across countries (rows)
    return X.corr(method=CORR_METHOD)


def to_distance_from_corr(C: pd.DataFrame) -> np.ndarray:
    # distance = 1 - corr, diagonal 0
    A = C.to_numpy(float)
    D = 1.0 - A
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, 2.0)


def clustered_order_from_corr(C: pd.DataFrame) -> List[int]:
    D = to_distance_from_corr(C)
    L = linkage(squareform(D, checks=False), method=LINKAGE_METHOD)
    return dendrogram(L, no_plot=True)["leaves"]


def heatmap_values_from_corr(C: pd.DataFrame) -> np.ndarray:
    A = C.to_numpy(float)
    if HEATMAP_MODE == "correlation":
        return A
    if HEATMAP_MODE == "dissimilarity":
        return 1.0 - A
    raise ValueError("HEATMAP_MODE must be 'correlation' or 'dissimilarity'.")


def heatmap_limits():
    if HEATMAP_MODE == "correlation":
        return -1.0, 1.0, "Correlation"
    return 0.0, 2.0, "Dissimilarity (1 - corr)"


# ----------------------------
# Plotting helpers
# ----------------------------
def plot_rectangular_metric_similarity_profile(X_sub: pd.DataFrame, outbase: Path):
    """
    Requested layout:
      rows = countries (with dendrogram)
      cols = measures (with dendrogram)
      BUT heatmap cells reflect measure–measure similarity (profile).

    Implementation:
      - Compute metric–metric correlation C (m x m)
      - Build a per-metric profile = average similarity to other metrics
      - Replicate that profile for each country
      - Cluster columns by metric–metric similarity
      - Cluster rows by country–country similarity
    """
    set_pub_style()

    # metric–metric similarity
    Cmm = corr_metric_metric(X_sub).dropna(axis=0, how="any").dropna(axis=1, how="any")
    if Cmm.shape[0] < 3:
        raise ValueError("Need at least 3 metrics for metric–metric similarity heatmap.")

    # profile per metric: average similarity to other metrics
    A = Cmm.to_numpy(float)
    m = A.shape[0]
    prof = (A.sum(axis=1) - np.diag(A)) / max(1, (m - 1))  # avg off-diagonal correlation

    # build rectangular matrix: countries x metrics (same across countries)
    metrics = list(Cmm.columns)
    H_rect = np.tile(prof.reshape(1, -1), (X_sub.shape[0], 1))

    # column clustering by metric–metric correlation
    col_leaves = clustered_order_from_corr(Cmm)
    metrics_ord = [metrics[i] for i in col_leaves]

    # row clustering by country–country correlation (so dendrogram is meaningful)
    Ccc = corr_country_country(X_sub)
    row_leaves = clustered_order_from_corr(Ccc)
    countries_ord = [str(X_sub.index[i]) for i in row_leaves]

    # reorder heatmap
    H_ord = H_rect[row_leaves, :][:, col_leaves]

    # labels
    xlabels = [PRETTY.get(c, c) for c in metrics_ord]
    ylabels = country_labels(countries_ord)

    # layout like classic clustermap (top + left)
    fig = plt.figure(figsize=(7.8, 4.3), dpi=300)
    ax_top  = fig.add_axes([0.24, 0.80, 0.60, 0.14])
    ax_left = fig.add_axes([0.06, 0.20, 0.14, 0.60])
    ax_heat = fig.add_axes([0.24, 0.20, 0.60, 0.60])
    ax_cbar = fig.add_axes([0.86, 0.20, 0.03, 0.60])

    # dendrograms
    Lc = linkage(squareform(to_distance_from_corr(Cmm.iloc[col_leaves, col_leaves]), checks=False),
                 method=LINKAGE_METHOD)
    dendrogram(Lc, ax=ax_top, orientation="top", no_labels=True,
               color_threshold=None, above_threshold_color="black")
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for s in ax_top.spines.values(): s.set_visible(False)

    Lr = linkage(squareform(to_distance_from_corr(Ccc.iloc[row_leaves, row_leaves]), checks=False),
                 method=LINKAGE_METHOD)
    dendrogram(Lr, ax=ax_left, orientation="left", no_labels=True,
               color_threshold=None, above_threshold_color="black")
    ax_left.set_xticks([]); ax_left.set_yticks([])
    for s in ax_left.spines.values(): s.set_visible(False)

    # heatmap values
    vmin, vmax, cbar_label = heatmap_limits()
    im = ax_heat.imshow(H_ord, vmin=vmin, vmax=vmax, cmap=HEATMAP_CMAP, aspect="auto")
    ax_heat.set_xticks(np.arange(len(xlabels)))
    ax_heat.set_yticks(np.arange(len(ylabels)))
    ax_heat.set_xticklabels(xlabels, rotation=90, ha="center")
    ax_heat.set_yticklabels(ylabels)

    ax_heat.set_title("P1/P2 clustermap (cells = metric–metric similarity profile)")

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label(cbar_label)

    save_png_pdf(fig, outbase)


def plot_square_country_similarity(X_sub: pd.DataFrame, outbase: Path):
    """
    Square country–country similarity (or dissimilarity) clustermap.
    """
    set_pub_style()

    Ccc = corr_country_country(X_sub)
    if Ccc.shape[0] < 3:
        raise ValueError("Need at least 3 countries.")

    leaves = clustered_order_from_corr(Ccc)
    C_ord = Ccc.iloc[leaves, leaves]
    labels = country_labels(list(C_ord.index.astype(str)))

    H = heatmap_values_from_corr(C_ord)
    vmin, vmax, cbar_label = heatmap_limits()

    fig = plt.figure(figsize=(6.4, 5.2), dpi=300)
    ax_top  = fig.add_axes([0.24, 0.82, 0.58, 0.14])
    ax_left = fig.add_axes([0.06, 0.20, 0.14, 0.58])
    ax_heat = fig.add_axes([0.24, 0.20, 0.58, 0.58])
    ax_cbar = fig.add_axes([0.84, 0.20, 0.03, 0.58])

    L = linkage(squareform(to_distance_from_corr(C_ord), checks=False), method=LINKAGE_METHOD)

    dendrogram(L, ax=ax_top, orientation="top", no_labels=True,
               color_threshold=None, above_threshold_color="black")
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for s in ax_top.spines.values(): s.set_visible(False)

    dendrogram(L, ax=ax_left, orientation="left", no_labels=True,
               color_threshold=None, above_threshold_color="black")
    ax_left.set_xticks([]); ax_left.set_yticks([])
    for s in ax_left.spines.values(): s.set_visible(False)

    im = ax_heat.imshow(H, vmin=vmin, vmax=vmax, cmap=HEATMAP_CMAP, aspect="auto")
    ax_heat.set_xticks(np.arange(len(labels)))
    ax_heat.set_yticks(np.arange(len(labels)))
    ax_heat.set_xticklabels(labels, rotation=90, ha="center")
    ax_heat.set_yticklabels(labels)

    ax_heat.set_title("P1/P2 clustermap (country–country similarity)")

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label(cbar_label)

    save_png_pdf(fig, outbase)


# ----------------------------
# MAIN
# ----------------------------
def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Cannot find: {DATA.resolve()}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    if "iso3" not in df.columns:
        raise ValueError("CSV must contain an 'iso3' column.")
    df["iso3"] = df["iso3"].astype(str).str.upper()
    df = df.set_index("iso3")

    missing = [c for c in EXTERNAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required external columns: {missing}")

    X_all = df[EXTERNAL_COLS].apply(pd.to_numeric, errors="coerce").dropna(how="any")

    # P1/P2 subset only
    union_iso3 = build_selected_iso3()
    keep = [c for c in union_iso3 if c in X_all.index]
    if len(keep) < 3:
        raise ValueError("Need at least 3 P1/P2 countries present after cleaning.")
    X_sub = X_all.loc[keep].copy()

    # 1) rectangular
    plot_rectangular_metric_similarity_profile(
        X_sub=X_sub,
        outbase=OUTDIR / "P1P2_Rectangular_metricSimilarityProfile",
    )

    # 2) square
    plot_square_country_similarity(
        X_sub=X_sub,
        outbase=OUTDIR / "P1P2_CountrySimilarity",
    )

    print("[OK] Saved in:", OUTDIR.resolve())
    print(" - P1P2_Rectangular_metricSimilarityProfile.(png/pdf)")
    print(" - P1P2_CountrySimilarity.(png/pdf)")


if __name__ == "__main__":
    main()




# %% Cell 13

