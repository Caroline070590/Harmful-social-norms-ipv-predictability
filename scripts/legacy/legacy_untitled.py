# Auto-extracted from notebooks/legacy/legacy_untitled.ipynb

# Review/reproducibility helper script. Original notebook is retained in notebooks/.



# %% Cell 0
# hofstede_part2_bayesian_nopymc.py
# ------------------------------------------------------------
# Part 2: Hofstede vs Violence (Part2 clean table)
# + Bayesian Spearman (rank correlation) WITHOUT PyMC
# + Bayes Factor via Savage–Dickey using:
#     - Likelihood approx: Fisher z ~ Normal(atanh(rho), 1/sqrt(n-3))
#     - Prior: rho ~ Uniform(-1, 1)  -> prior density at 0 is 0.5
#
# Dependencies:
#   pip install numpy pandas scipy matplotlib
# Optional:
#   pip install pycountry
# ------------------------------------------------------------

from __future__ import annotations

import re
from pathlib import Path
from difflib import get_close_matches
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import rankdata, norm
from scipy.integrate import quad


# ----------------------------
# USER SETTINGS
# ----------------------------
HOFSTEDE_CSV = Path("hofstede_country_scores.csv")
PART2_CLEAN  = Path("results_paper_ready/part2_demographics_only_xgb/clean_modeling_table.csv")

OUT_ROOT = Path("results_paper_ready/hofstede_vs_violence_part2_bayes_nopymc")

AGG_MODE = "median"     # "mean" | "median" | "trimmed_mean"
TRIM_Q = 0.10
MIN_N_PER_COUNTRY = 20

FUZZY_CUTOFF = 0.86

# Posterior sampling for summaries (importance resampling)
POST_SAMPLES = 200000
RESAMPLE_N = 20000
RANDOM_SEED = 42

SCATTER_ALPHA = 0.85
DPI_EXPORT = 600


# ----------------------------
# Optional dependency
# ----------------------------
try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# ============================================================
# Helpers: strings, iso3, matching
# ============================================================
def _clean_country(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_country_name(name: str) -> str:
    name = _clean_country(name)
    overrides = {
        "Russian Federation": "Russia",
        "Viet Nam": "Vietnam",
        "Iran, Islamic Rep.": "Iran",
        "Iran (Islamic Republic of)": "Iran",
        "Korea, Rep.": "South Korea",
        "Korea Rep.": "South Korea",
        "Korea, Dem. People's Rep.": "North Korea",
        "Czech Rep.": "Czech Republic",
        "Slovak Rep.": "Slovakia",
        "Egypt, Arab Rep.": "Egypt",
        "Yemen, Rep.": "Yemen",
        "Venezuela, RB": "Venezuela",
        "Bolivia (Plurinational State of)": "Bolivia",
        "Tanzania, United Rep.": "Tanzania",
        "United States of America": "United States",
        "USA": "United States",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "Côte d’Ivoire": "Ivory Coast",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "Lao PDR": "Laos",
        "Syrian Arab Republic": "Syria",
        "Türkiye": "Turkey",
        "Palestine": "Palestine, State of",
        "Cabo Verde": "Cape Verde",
        "Eswatini": "Swaziland",
    }
    return overrides.get(name, name)


def _iso3_from_name(name: str) -> Optional[str]:
    if not name or str(name).lower() == "nan":
        return None
    name = normalize_country_name(name)

    overrides = {
        "Russia": "RUS",
        "United States": "USA",
        "United Kingdom": "GBR",
        "Ivory Coast": "CIV",
        "Vietnam": "VNM",
        "Laos": "LAO",
        "Syria": "SYR",
        "Turkey": "TUR",
        "Bolivia": "BOL",
        "Venezuela": "VEN",
        "Tanzania": "TZA",
        "South Korea": "KOR",
        "North Korea": "PRK",
        "Czech Republic": "CZE",
        "Palestine, State of": "PSE",
        "Iran": "IRN",
        "Cape Verde": "CPV",
        "Swaziland": "SWZ",
    }
    if name in overrides:
        return overrides[name]

    if not HAS_PYCOUNTRY:
        return None

    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None


def fuzzy_match_series(left: pd.Series, right_values: list[str], cutoff: float) -> pd.Series:
    right_values = sorted(set([str(v) for v in right_values if pd.notna(v)]))
    out = []
    for x in left.astype(str):
        m = get_close_matches(x, right_values, n=1, cutoff=cutoff)
        out.append(m[0] if m else x)
    return pd.Series(out, index=left.index)


def assert_exists(p: Path, label: str):
    if not p.exists():
        raise FileNotFoundError(f"[{label}] Cannot find: {p.resolve()}")


# ============================================================
# Aggregation
# ============================================================
def trimmed_mean(x: np.ndarray, q: float = 0.10) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    lo = np.quantile(x, q)
    hi = np.quantile(x, 1 - q)
    x2 = x[(x >= lo) & (x <= hi)]
    return float(np.mean(x2)) if x2.size else float(np.mean(x))


def aggregate_country_values(
    df: pd.DataFrame,
    country_col: str,
    value_col: str,
    min_n: int,
    agg_mode: str,
    trim_q: float,
) -> pd.DataFrame:
    d = df.copy()
    d[country_col] = d[country_col].astype(str).map(normalize_country_name)
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    out_rows = []
    for c, g in d.groupby(country_col):
        vals = pd.to_numeric(g[value_col], errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size < min_n:
            continue

        if agg_mode == "mean":
            agg = float(np.mean(vals))
        elif agg_mode == "median":
            agg = float(np.median(vals))
        elif agg_mode == "trimmed_mean":
            agg = trimmed_mean(vals, q=trim_q)
        else:
            raise ValueError("AGG_MODE must be mean|median|trimmed_mean")

        out_rows.append({
            "Country_norm": c,
            "Violence_part2_all": agg,
            "n_points": int(vals.size),
        })

    cols = ["Country_norm", "Violence_part2_all", "n_points"]
    if not out_rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(out_rows)[cols]
    out["iso3"] = out["Country_norm"].map(_iso3_from_name)
    return out.sort_values("Country_norm").reset_index(drop=True)


# ============================================================
# Load Hofstede
# ============================================================
def load_hofstede(hofstede_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    hof = pd.read_csv(hofstede_csv)

    country_col = None
    for c in hof.columns:
        if str(c).strip().lower() == "country":
            country_col = c
            break
    if country_col is None:
        raise ValueError(f"Hofstede CSV must have a country column. Columns: {list(hof.columns)}")
    if country_col != "Country":
        hof = hof.rename(columns={country_col: "Country"})

    hof["Country"] = hof["Country"].astype(str).map(_clean_country)
    hof["Country_norm"] = hof["Country"].map(normalize_country_name)
    hof["iso3"] = hof["Country_norm"].map(_iso3_from_name).astype("string").str.upper()

    # critical: avoid duplicates by iso3
    hof = hof.dropna(subset=["iso3"]).drop_duplicates(subset=["iso3"], keep="first").copy()

    dims = []
    for col in hof.columns:
        if str(col).lower() in {"country", "country_norm", "iso3"}:
            continue
        s = pd.to_numeric(hof[col], errors="coerce")
        if np.isfinite(s).sum() >= 10:
            dims.append(col)

    if not dims:
        raise ValueError("No numeric Hofstede dimensions detected.")
    return hof[["Country", "Country_norm", "iso3"] + dims].copy(), dims


# ============================================================
# Merge iso3 then fuzzy
# ============================================================
def merge_iso3_then_fuzzy(hof: pd.DataFrame, v: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = hof.copy()
    vv = v.copy()

    vv["Country_norm"] = vv["Country_norm"].astype(str).map(normalize_country_name)
    vv["iso3"] = vv["iso3"].astype("string").str.upper()

    merged = vv.merge(h, on="iso3", how="left", suffixes=("", "_hof"))

    audit = pd.DataFrame({
        "Country_norm": merged["Country_norm"].astype(str),
        "iso3": merged["iso3"].astype(str),
        "matched_iso3": ~merged["Country"].isna(),
    })

    missing = merged["Country"].isna()

    if missing.any():
        candidates = h["Country_norm"].dropna().astype(str).tolist()
        fuzzy_target = fuzzy_match_series(merged.loc[missing, "Country_norm"], candidates, cutoff=FUZZY_CUTOFF)
        audit.loc[missing, "Country_norm_fuzzy"] = fuzzy_target.to_numpy()

        h2 = h.rename(columns={"Country_norm": "Country_norm_fuzzy"})
        tmp = merged.loc[missing].copy()
        tmp["Country_norm_fuzzy"] = fuzzy_target.to_numpy()
        filled = tmp.merge(h2, on="Country_norm_fuzzy", how="left", suffixes=("", "_fuzzyfill"))

        # Fill Hofstede columns back
        for col in [c for c in h.columns if c != "iso3"]:
            if col in filled.columns:
                merged.loc[missing, col] = filled[col].to_numpy()

        audit.loc[missing, "matched_fuzzy"] = ~filled["Country"].isna().to_numpy()
    else:
        audit["Country_norm_fuzzy"] = ""
        audit["matched_fuzzy"] = False

    return merged, audit


# ============================================================
# Bayesian Spearman (no PyMC): Fisher-z likelihood approx + Uniform prior on rho
# ============================================================
def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return np.nan
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def posterior_z_norm_const(z_hat: float, se: float) -> float:
    # posterior in z: p(z|data) ∝ Normal(z | z_hat, se) * sech^2(z)
    # sech^2(z) = 1 / cosh(z)^2
    def unnorm(z):
        return norm.pdf(z, loc=z_hat, scale=se) / (np.cosh(z) ** 2)

    # integrate on a wide finite interval; Fisher z rarely needs > ~10
    return float(quad(unnorm, -20, 20, limit=200)[0])


def bayes_spearman_uniform_prior(x: np.ndarray, y: np.ndarray, seed: int):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 8:
        return None

    r_s = spearman_rho(x, y)
    # protect atanh
    r_s = float(np.clip(r_s, -0.999999, 0.999999))

    # Fisher z likelihood approximation
    z_hat = float(np.arctanh(r_s))
    se = float(1.0 / np.sqrt(max(n - 3, 1)))

    Z = posterior_z_norm_const(z_hat, se)

    # posterior density at rho=0:
    # rho=0 <-> z=0; Jacobian at 0 is 1, so density in rho equals density in z at 0.
    post0 = float(norm.pdf(0.0, loc=z_hat, scale=se) / Z)  # sech^2(0)=1

    prior0 = 0.5  # Uniform(-1,1)

    BF10 = float(prior0 / post0) if post0 > 0 else np.inf

    # Posterior summaries via importance resampling
    rng = np.random.default_rng(seed)
    z_prop = rng.normal(loc=z_hat, scale=se, size=POST_SAMPLES)
    w = 1.0 / (np.cosh(z_prop) ** 2)  # weights ∝ sech^2(z)
    w = w / np.sum(w)
    idx = rng.choice(np.arange(POST_SAMPLES), size=RESAMPLE_N, replace=True, p=w)
    z_post = z_prop[idx]
    rho_post = np.tanh(z_post)

    out = {
        "n": n,
        "spearman_r": float(r_s),
        "rho_mean": float(np.mean(rho_post)),
        "rho_median": float(np.median(rho_post)),
        "rho_ci_low": float(np.quantile(rho_post, 0.025)),
        "rho_ci_high": float(np.quantile(rho_post, 0.975)),
        "P_rho_gt_0": float(np.mean(rho_post > 0)),
        "prior_density_rho0": float(prior0),
        "posterior_density_rho0": float(post0),
        "BF10": float(BF10),
    }
    return out


# ============================================================
# Plot helper
# ============================================================
def plot_joint_scatter(x, y, xlabel, ylabel, title, out_png: Path, out_pdf: Path):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

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
    })

    fig = plt.figure(figsize=(8.6, 8.2), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4], wspace=0.05, hspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax = fig.add_subplot(gs[1, 0])

    ax.scatter(x, y, s=45, alpha=SCATTER_ALPHA)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    if x.size >= 2:
        b1, b0 = np.polyfit(x, y, 1)
        xx = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(xx, b1 * xx + b0, linewidth=2)

    ax_top.hist(x, bins=12, alpha=0.6, edgecolor="black", linewidth=0.6)
    ax_right.hist(y, bins=12, orientation="horizontal", alpha=0.6, edgecolor="black", linewidth=0.6)

    for a in (ax_top, ax_right):
        a.grid(False)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_right.set_xticks([]); ax_right.set_yticks([])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    assert_exists(HOFSTEDE_CSV, "HOFSTEDE_CSV")
    assert_exists(PART2_CLEAN, "PART2_CLEAN")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    hof, dims = load_hofstede(HOFSTEDE_CSV)

    part2 = pd.read_csv(PART2_CLEAN)
    if not {"Country", "Value"}.issubset(part2.columns):
        raise ValueError(f"PART2_CLEAN must include Country and Value. Got columns: {list(part2.columns)}")

    v2 = aggregate_country_values(
        part2,
        country_col="Country",
        value_col="Value",
        min_n=MIN_N_PER_COUNTRY,
        agg_mode=AGG_MODE,
        trim_q=TRIM_Q,
    )
    if len(v2) < 10:
        raise ValueError(f"Too few countries after aggregation (len={len(v2)}). Lower MIN_N_PER_COUNTRY?")

    merged, audit = merge_iso3_then_fuzzy(hof, v2)

    merged.to_csv(OUT_ROOT / "analysis_table_part2_allcountries.csv", index=False)
    audit.to_csv(OUT_ROOT / "merge_audit_part2.csv", index=False)

    keep = merged.dropna(subset=["Violence_part2_all"]).copy()

    rows = []
    for d in dims:
        x = pd.to_numeric(keep[d], errors="coerce").to_numpy()
        y = pd.to_numeric(keep["Violence_part2_all"], errors="coerce").to_numpy()

        res = bayes_spearman_uniform_prior(x, y, seed=RANDOM_SEED)
        if res is None:
            continue
        res["Hofstede_dim"] = d
        rows.append(res)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("No Bayesian results computed (too few valid rows per dimension).")

    out = out.sort_values("BF10", ascending=False)
    out.to_csv(OUT_ROOT / "bayes_spearman_part2_all.csv", index=False)

    best_dim = str(out.iloc[0]["Hofstede_dim"])
    x = pd.to_numeric(keep[best_dim], errors="coerce").to_numpy()
    y = pd.to_numeric(keep["Violence_part2_all"], errors="coerce").to_numpy()
    plot_joint_scatter(
        x, y,
        xlabel=f"Hofstede: {best_dim}",
        ylabel="Violence_part2_all",
        title=f"Part2 all countries: best dim by BF10 = {best_dim}",
        out_png=OUT_ROOT / "joint_bestdim_part2_all.png",
        out_pdf=OUT_ROOT / "joint_bestdim_part2_all.pdf",
    )

    print("\n[OK] Finished Part2 Bayesian Spearman (no PyMC).")
    print("[OK] Output folder:", OUT_ROOT.resolve())
    print("[OK] Best dim by BF10:", best_dim)
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()



# %% Cell 1
# hofstede_part2_bayesian_nopymc_plus.py
# ------------------------------------------------------------
# Part 2: Hofstede vs Violence (Part2 clean table)
# + Bayesian Spearman WITHOUT PyMC (Fisher-z approx + Uniform prior on rho)
# + ADD-ONS implemented:
#   (1) Meta-analysis across all analyses (Fisher-z random effects via DerSimonian-Laird)
#   (2) Directional bootstrap: P(rho>0) and P(rho<0) from bootstrap distribution
#   (3) Cultural PCA: PCA on Hofstede dims -> analyze PC1/PC2/PC3 vs outcome
#   (4) Leave-one-country-out sensitivity per dim
#
# Dependencies:
#   pip install numpy pandas scipy matplotlib
# Optional:
#   pip install pycountry
# ------------------------------------------------------------

from __future__ import annotations

import re
import os
from pathlib import Path
from difflib import get_close_matches
from typing import Optional, Iterable, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import rankdata, norm
from scipy.integrate import quad


# ----------------------------
# USER SETTINGS
# ----------------------------
HOFSTEDE_CSV = Path("hofstede_country_scores.csv")
PART2_CLEAN  = Path("results_paper_ready/part2_demographics_only_xgb/clean_modeling_table.csv")

OUT_ROOT = Path("results_paper_ready/hofstede_vs_violence_part2_bayes_nopymc_plus")

# Aggregation
AGG_MODE = "median"     # "mean" | "median" | "trimmed_mean"
TRIM_Q = 0.10
MIN_N_PER_COUNTRY = 20

# Matching
FUZZY_CUTOFF = 0.86

# Bayesian posterior sampling (importance resampling)
POST_SAMPLES = 200000
RESAMPLE_N = 20000
RANDOM_SEED = 42

# Bootstrap settings (directional bootstrap + CI)
BOOTSTRAP_B = 10000

# PCA settings
PCA_N_COMPONENTS = 3  # analyze PC1..PCk
PCA_STANDARDIZE = True  # standardize Hofstede dims before PCA

# LOOCV settings
DO_LOOCV = True

# Meta-analysis: point to your global results folder that contains subfolders
# with files named spearman_bootstrap_*.csv (from your ABCD script outputs).
META_RESULTS_ROOT: Optional[Path] = Path("results_paper_ready")  # set None to skip

# Plot
SCATTER_ALPHA = 0.85
DPI_EXPORT = 600


# ----------------------------
# Optional dependency
# ----------------------------
try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False


# ============================================================
# Helpers: strings, iso3, matching
# ============================================================
def _clean_country(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_country_name(name: str) -> str:
    name = _clean_country(name)
    overrides = {
        "Russian Federation": "Russia",
        "Viet Nam": "Vietnam",
        "Iran, Islamic Rep.": "Iran",
        "Iran (Islamic Republic of)": "Iran",
        "Korea, Rep.": "South Korea",
        "Korea Rep.": "South Korea",
        "Korea, Dem. People's Rep.": "North Korea",
        "Czech Rep.": "Czech Republic",
        "Slovak Rep.": "Slovakia",
        "Egypt, Arab Rep.": "Egypt",
        "Yemen, Rep.": "Yemen",
        "Venezuela, RB": "Venezuela",
        "Bolivia (Plurinational State of)": "Bolivia",
        "Tanzania, United Rep.": "Tanzania",
        "United States of America": "United States",
        "USA": "United States",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "Côte d’Ivoire": "Ivory Coast",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "Lao PDR": "Laos",
        "Syrian Arab Republic": "Syria",
        "Türkiye": "Turkey",
        "Palestine": "Palestine, State of",
        "Cabo Verde": "Cape Verde",
        "Eswatini": "Swaziland",
    }
    return overrides.get(name, name)


def _iso3_from_name(name: str) -> Optional[str]:
    if not name or str(name).lower() == "nan":
        return None
    name = normalize_country_name(name)

    overrides = {
        "Russia": "RUS",
        "United States": "USA",
        "United Kingdom": "GBR",
        "Ivory Coast": "CIV",
        "Vietnam": "VNM",
        "Laos": "LAO",
        "Syria": "SYR",
        "Turkey": "TUR",
        "Bolivia": "BOL",
        "Venezuela": "VEN",
        "Tanzania": "TZA",
        "South Korea": "KOR",
        "North Korea": "PRK",
        "Czech Republic": "CZE",
        "Palestine, State of": "PSE",
        "Iran": "IRN",
        "Cape Verde": "CPV",
        "Swaziland": "SWZ",
    }
    if name in overrides:
        return overrides[name]

    if not HAS_PYCOUNTRY:
        return None

    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None


def fuzzy_match_series(left: pd.Series, right_values: list[str], cutoff: float) -> pd.Series:
    right_values = sorted(set([str(v) for v in right_values if pd.notna(v)]))
    out = []
    for x in left.astype(str):
        m = get_close_matches(x, right_values, n=1, cutoff=cutoff)
        out.append(m[0] if m else x)
    return pd.Series(out, index=left.index)


def assert_exists(p: Path, label: str):
    if not p.exists():
        raise FileNotFoundError(f"[{label}] Cannot find: {p.resolve()}")


# ============================================================
# Aggregation
# ============================================================
def trimmed_mean(x: np.ndarray, q: float = 0.10) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    lo = np.quantile(x, q)
    hi = np.quantile(x, 1 - q)
    x2 = x[(x >= lo) & (x <= hi)]
    return float(np.mean(x2)) if x2.size else float(np.mean(x))


def aggregate_country_values(
    df: pd.DataFrame,
    country_col: str,
    value_col: str,
    min_n: int,
    agg_mode: str,
    trim_q: float,
) -> pd.DataFrame:
    d = df.copy()
    d[country_col] = d[country_col].astype(str).map(normalize_country_name)
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    out_rows = []
    for c, g in d.groupby(country_col):
        vals = pd.to_numeric(g[value_col], errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size < min_n:
            continue

        if agg_mode == "mean":
            agg = float(np.mean(vals))
        elif agg_mode == "median":
            agg = float(np.median(vals))
        elif agg_mode == "trimmed_mean":
            agg = trimmed_mean(vals, q=trim_q)
        else:
            raise ValueError("AGG_MODE must be mean|median|trimmed_mean")

        out_rows.append({
            "Country_norm": c,
            "Violence_part2_all": agg,
            "n_points": int(vals.size),
        })

    cols = ["Country_norm", "Violence_part2_all", "n_points"]
    if not out_rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(out_rows)[cols]
    out["iso3"] = out["Country_norm"].map(_iso3_from_name)
    return out.sort_values("Country_norm").reset_index(drop=True)


# ============================================================
# Load Hofstede
# ============================================================
def load_hofstede(hofstede_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    hof = pd.read_csv(hofstede_csv)

    country_col = None
    for c in hof.columns:
        if str(c).strip().lower() == "country":
            country_col = c
            break
    if country_col is None:
        raise ValueError(f"Hofstede CSV must have a country column. Columns: {list(hof.columns)}")
    if country_col != "Country":
        hof = hof.rename(columns={country_col: "Country"})

    hof["Country"] = hof["Country"].astype(str).map(_clean_country)
    hof["Country_norm"] = hof["Country"].map(normalize_country_name)
    hof["iso3"] = hof["Country_norm"].map(_iso3_from_name).astype("string").str.upper()

    # avoid duplicates by iso3
    hof = hof.dropna(subset=["iso3"]).drop_duplicates(subset=["iso3"], keep="first").copy()

    dims = []
    for col in hof.columns:
        if str(col).lower() in {"country", "country_norm", "iso3"}:
            continue
        s = pd.to_numeric(hof[col], errors="coerce")
        if np.isfinite(s).sum() >= 10:
            dims.append(col)

    if not dims:
        raise ValueError("No numeric Hofstede dimensions detected.")
    return hof[["Country", "Country_norm", "iso3"] + dims].copy(), dims


# ============================================================
# Merge iso3 then fuzzy
# ============================================================
def merge_iso3_then_fuzzy(hof: pd.DataFrame, v: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = hof.copy()
    vv = v.copy()

    vv["Country_norm"] = vv["Country_norm"].astype(str).map(normalize_country_name)
    vv["iso3"] = vv["iso3"].astype("string").str.upper()

    merged = vv.merge(h, on="iso3", how="left", suffixes=("", "_hof"))

    audit = pd.DataFrame({
        "Country_norm": merged["Country_norm"].astype(str),
        "iso3": merged["iso3"].astype(str),
        "matched_iso3": ~merged["Country"].isna(),
    })

    missing = merged["Country"].isna()

    if missing.any():
        candidates = h["Country_norm"].dropna().astype(str).tolist()
        fuzzy_target = fuzzy_match_series(merged.loc[missing, "Country_norm"], candidates, cutoff=FUZZY_CUTOFF)
        audit.loc[missing, "Country_norm_fuzzy"] = fuzzy_target.to_numpy()

        h2 = h.rename(columns={"Country_norm": "Country_norm_fuzzy"})
        tmp = merged.loc[missing].copy()
        tmp["Country_norm_fuzzy"] = fuzzy_target.to_numpy()
        filled = tmp.merge(h2, on="Country_norm_fuzzy", how="left", suffixes=("", "_fuzzyfill"))

        for col in [c for c in h.columns if c != "iso3"]:
            if col in filled.columns:
                merged.loc[missing, col] = filled[col].to_numpy()

        audit.loc[missing, "matched_fuzzy"] = ~filled["Country"].isna().to_numpy()
    else:
        audit["Country_norm_fuzzy"] = ""
        audit["matched_fuzzy"] = False

    return merged, audit


# ============================================================
# Core stats: Spearman, bootstrap, Fisher-z helpers
# ============================================================
def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return np.nan
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def fisher_z(r: float) -> float:
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def inv_fisher_z(z: float) -> float:
    return float(np.tanh(z))


def se_fisher_z(n: int) -> float:
    return float(1.0 / np.sqrt(max(n - 3, 1)))


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, B: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 8:
        return {
            "n": n,
            "rho": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "P_rho_gt_0_boot": np.nan,
            "P_rho_lt_0_boot": np.nan,
        }

    r0 = spearman_rho(x, y)

    boots = np.empty(B, dtype=float)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        boots[i] = spearman_rho(x[idx], y[idx])

    ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
    return {
        "n": float(n),
        "rho": float(r0),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "P_rho_gt_0_boot": float(np.mean(boots > 0)),
        "P_rho_lt_0_boot": float(np.mean(boots < 0)),
    }


# ============================================================
# Bayesian Spearman (no PyMC): Fisher-z likelihood approx + Uniform prior on rho
# ============================================================
def posterior_z_norm_const(z_hat: float, se: float) -> float:
    # posterior in z: p(z|data) ∝ Normal(z | z_hat, se) * sech^2(z)
    def unnorm(z):
        return norm.pdf(z, loc=z_hat, scale=se) / (np.cosh(z) ** 2)
    return float(quad(unnorm, -20, 20, limit=200)[0])


def bayes_spearman_uniform_prior(x: np.ndarray, y: np.ndarray, seed: int):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 8:
        return None

    r_s = spearman_rho(x, y)
    r_s = float(np.clip(r_s, -0.999999, 0.999999))

    z_hat = fisher_z(r_s)
    se = se_fisher_z(n)

    Z = posterior_z_norm_const(z_hat, se)

    post0 = float(norm.pdf(0.0, loc=z_hat, scale=se) / Z)  # sech^2(0)=1
    prior0 = 0.5  # Uniform(-1,1)
    BF10 = float(prior0 / post0) if post0 > 0 else np.inf

    # Posterior summaries via importance resampling
    rng = np.random.default_rng(seed)
    z_prop = rng.normal(loc=z_hat, scale=se, size=POST_SAMPLES)
    w = 1.0 / (np.cosh(z_prop) ** 2)
    w = w / np.sum(w)
    idx = rng.choice(np.arange(POST_SAMPLES), size=RESAMPLE_N, replace=True, p=w)
    z_post = z_prop[idx]
    rho_post = np.tanh(z_post)

    out = {
        "n": n,
        "spearman_r": float(r_s),
        "rho_mean": float(np.mean(rho_post)),
        "rho_median": float(np.median(rho_post)),
        "rho_ci_low": float(np.quantile(rho_post, 0.025)),
        "rho_ci_high": float(np.quantile(rho_post, 0.975)),
        "P_rho_gt_0_post": float(np.mean(rho_post > 0)),
        "P_rho_lt_0_post": float(np.mean(rho_post < 0)),
        "prior_density_rho0": float(prior0),
        "posterior_density_rho0": float(post0),
        "BF10": float(BF10),
    }
    return out


# ============================================================
# PCA on Hofstede dimensions (cultural PCA)
# ============================================================
def pca_fit_transform(X: np.ndarray, n_components: int, standardize: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA via SVD on (optionally standardized) X.
    Returns:
      scores: (n, k)
      loadings: (p, k)  (each column corresponds to a PC in original feature space)
      explained_var_ratio: (k,)
    """
    X = np.asarray(X, dtype=float)

    # center + (optional) scale
    mu = np.nanmean(X, axis=0)
    Xc = X - mu

    if standardize:
        sd = np.nanstd(Xc, axis=0, ddof=0)
        sd = np.where(sd == 0, 1.0, sd)
        Xc = Xc / sd

    # remove rows with any nan (PCA needs complete cases)
    mask = np.all(np.isfinite(Xc), axis=1)
    Xc2 = Xc[mask]
    if Xc2.shape[0] < max(8, n_components + 2):
        raise ValueError("Too few complete rows for PCA. Reduce PCA_N_COMPONENTS or improve matching.")

    # SVD
    U, S, Vt = np.linalg.svd(Xc2, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    scores = U[:, :k] * S[:k]
    loadings = Vt[:k, :].T  # (p,k)

    # explained variance ratio
    # variance explained by each PC = S^2/(n-1)
    n = Xc2.shape[0]
    var = (S ** 2) / max(n - 1, 1)
    evr = var[:k] / np.sum(var)

    return scores, loadings, evr


# ============================================================
# Leave-one-country-out sensitivity
# ============================================================
def loocv_spearman(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 10:
        return {"n": n, "rho_full": np.nan, "rho_loocv_mean": np.nan, "rho_loocv_sd": np.nan,
                "rho_loocv_min": np.nan, "rho_loocv_max": np.nan}

    rho_full = spearman_rho(x, y)
    rhos = []
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        rhos.append(spearman_rho(x[keep], y[keep]))
    rhos = np.asarray(rhos, dtype=float)
    return {
        "n": float(n),
        "rho_full": float(rho_full),
        "rho_loocv_mean": float(np.nanmean(rhos)),
        "rho_loocv_sd": float(np.nanstd(rhos, ddof=1)),
        "rho_loocv_min": float(np.nanmin(rhos)),
        "rho_loocv_max": float(np.nanmax(rhos)),
    }


# ============================================================
# Meta-analysis across all analyses
# ============================================================
def find_spearman_bootstrap_csvs(root: Path) -> List[Path]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith("spearman_bootstrap_") and fn.endswith(".csv"):
                out.append(Path(dirpath) / fn)
    return sorted(out)


def dersimonian_laird_random_effects(z: np.ndarray, se: np.ndarray) -> Dict[str, float]:
    """
    Random effects meta-analysis on Fisher-z with DerSimonian-Laird.
    Returns pooled z, se, tau2, and pooled rho (tanh(z)).
    """
    z = np.asarray(z, dtype=float)
    se = np.asarray(se, dtype=float)
    mask = np.isfinite(z) & np.isfinite(se) & (se > 0)
    z, se = z[mask], se[mask]
    k = int(z.size)
    if k < 2:
        return {"k": k, "z_pooled": np.nan, "se_pooled": np.nan, "tau2": np.nan, "rho_pooled": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan}

    w = 1.0 / (se ** 2)
    z_fixed = np.sum(w * z) / np.sum(w)
    Q = np.sum(w * (z - z_fixed) ** 2)
    df = k - 1
    c = np.sum(w) - (np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0

    w_star = 1.0 / (se ** 2 + tau2)
    z_re = np.sum(w_star * z) / np.sum(w_star)
    se_re = np.sqrt(1.0 / np.sum(w_star))

    ci_lo = z_re - 1.96 * se_re
    ci_hi = z_re + 1.96 * se_re

    return {
        "k": float(k),
        "z_pooled": float(z_re),
        "se_pooled": float(se_re),
        "tau2": float(tau2),
        "rho_pooled": float(inv_fisher_z(z_re)),
        "ci_lo": float(inv_fisher_z(ci_lo)),
        "ci_hi": float(inv_fisher_z(ci_hi)),
    }


def meta_analyze_all_dims(root: Path, outdir: Path) -> pd.DataFrame:
    """
    Reads all spearman_bootstrap_*.csv and meta-analyzes per Hofstede_dim using Fisher-z.
    Requires those CSVs to have at least: Hofstede_dim, Spearman_rho, n
    """
    csvs = find_spearman_bootstrap_csvs(root)
    if not csvs:
        raise FileNotFoundError(f"No spearman_bootstrap_*.csv found under: {root}")

    rows = []
    for p in csvs:
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            if "Hofstede_dim" not in df.columns or "Spearman_rho" not in df.columns:
                continue
            dim = str(r["Hofstede_dim"])
            rho = pd.to_numeric(r["Spearman_rho"], errors="coerce")
            n = pd.to_numeric(r.get("n", np.nan), errors="coerce")
            if not np.isfinite(rho) or not np.isfinite(n) or n < 6:
                continue
            z = fisher_z(float(rho))
            se = se_fisher_z(int(n))
            rows.append({
                "source_csv": str(p),
                "Hofstede_dim": dim,
                "rho": float(rho),
                "n": int(n),
                "z": float(z),
                "se_z": float(se),
            })

    long = pd.DataFrame(rows)
    if long.empty:
        raise ValueError("Meta-analysis: no usable rows across CSVs (check columns and n).")

    out_rows = []
    for dim, g in long.groupby("Hofstede_dim"):
        res = dersimonian_laird_random_effects(g["z"].to_numpy(), g["se_z"].to_numpy())
        out_rows.append({
            "Hofstede_dim": dim,
            "k_sources": int(res["k"]),
            "rho_pooled_RE": res["rho_pooled"],
            "CI95_lo": res["ci_lo"],
            "CI95_hi": res["ci_hi"],
            "tau2": res["tau2"],
        })

    out = pd.DataFrame(out_rows).sort_values("rho_pooled_RE", ascending=False)
    outdir.mkdir(parents=True, exist_ok=True)
    long.to_csv(outdir / "meta_inputs_long.csv", index=False)
    out.to_csv(outdir / "meta_random_effects_by_dim.csv", index=False)
    return out


# ============================================================
# Plot helper
# ============================================================
def plot_joint_scatter(x, y, xlabel, ylabel, title, out_png: Path, out_pdf: Path):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

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
    })

    fig = plt.figure(figsize=(8.6, 8.2), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4], wspace=0.05, hspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax = fig.add_subplot(gs[1, 0])

    ax.scatter(x, y, s=45, alpha=SCATTER_ALPHA)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    if x.size >= 2:
        b1, b0 = np.polyfit(x, y, 1)
        xx = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(xx, b1 * xx + b0, linewidth=2)

    ax_top.hist(x, bins=12, alpha=0.6, edgecolor="black", linewidth=0.6)
    ax_right.hist(y, bins=12, orientation="horizontal", alpha=0.6, edgecolor="black", linewidth=0.6)

    for a in (ax_top, ax_right):
        a.grid(False)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_right.set_xticks([]); ax_right.set_yticks([])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    assert_exists(HOFSTEDE_CSV, "HOFSTEDE_CSV")
    assert_exists(PART2_CLEAN, "PART2_CLEAN")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Load Hofstede
    hof, dims = load_hofstede(HOFSTEDE_CSV)

    # Load Part2 and aggregate per country
    part2 = pd.read_csv(PART2_CLEAN)
    if not {"Country", "Value"}.issubset(part2.columns):
        raise ValueError(f"PART2_CLEAN must include Country and Value. Got columns: {list(part2.columns)}")

    v2 = aggregate_country_values(
        part2,
        country_col="Country",
        value_col="Value",
        min_n=MIN_N_PER_COUNTRY,
        agg_mode=AGG_MODE,
        trim_q=TRIM_Q,
    )
    if len(v2) < 10:
        raise ValueError(f"Too few countries after aggregation (len={len(v2)}). Lower MIN_N_PER_COUNTRY?")

    # Merge
    merged, audit = merge_iso3_then_fuzzy(hof, v2)

    merged.to_csv(OUT_ROOT / "analysis_table_part2_allcountries.csv", index=False)
    audit.to_csv(OUT_ROOT / "merge_audit_part2.csv", index=False)

    keep = merged.dropna(subset=["Violence_part2_all"]).copy()
    keep.to_csv(OUT_ROOT / "analysis_table_part2_matched.csv", index=False)

    # --------------------------------------------------------
    # 1) Bayesian + Directional bootstrap + LOOCV per dimension
    # --------------------------------------------------------
    rows = []
    for d in dims:
        x = pd.to_numeric(keep[d], errors="coerce").to_numpy()
        y = pd.to_numeric(keep["Violence_part2_all"], errors="coerce").to_numpy()

        bayes = bayes_spearman_uniform_prior(x, y, seed=RANDOM_SEED)
        if bayes is None:
            continue

        boot = bootstrap_spearman(x, y, B=BOOTSTRAP_B, seed=RANDOM_SEED + 7)
        loo = loocv_spearman(x, y) if DO_LOOCV else {}

        out = {}
        out.update(bayes)
        out.update({f"boot_{k}": v for k, v in boot.items()})
        out.update({f"loocv_{k}": v for k, v in loo.items()})
        out["Hofstede_dim"] = d
        rows.append(out)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No results computed (too few valid rows per dimension).")

    out = out.sort_values("BF10", ascending=False)
    out.to_csv(OUT_ROOT / "bayes_bootstrap_loocv_by_dim.csv", index=False)

    # Plot best dim (by BF10)
    best_dim = str(out.iloc[0]["Hofstede_dim"])
    x = pd.to_numeric(keep[best_dim], errors="coerce").to_numpy()
    y = pd.to_numeric(keep["Violence_part2_all"], errors="coerce").to_numpy()
    plot_joint_scatter(
        x, y,
        xlabel=f"Hofstede: {best_dim}",
        ylabel="Violence_part2_all",
        title=f"Part2 matched countries: best dim by BF10 = {best_dim}",
        out_png=OUT_ROOT / "joint_bestdim_part2_all.png",
        out_pdf=OUT_ROOT / "joint_bestdim_part2_all.pdf",
    )

    # --------------------------------------------------------
    # 2) Cultural PCA
    # --------------------------------------------------------
    # Build complete-case matrix for PCA: outcome + all dims finite
    dim_mat = np.column_stack([pd.to_numeric(keep[d], errors="coerce").to_numpy() for d in dims])
    y_all = pd.to_numeric(keep["Violence_part2_all"], errors="coerce").to_numpy()

    mask_pca = np.isfinite(y_all) & np.all(np.isfinite(dim_mat), axis=1)
    keep_pca = keep.loc[mask_pca].copy()
    dim_mat2 = dim_mat[mask_pca]
    y2 = y_all[mask_pca]

    pca_dir = OUT_ROOT / "cultural_pca"
    pca_dir.mkdir(parents=True, exist_ok=True)

    if dim_mat2.shape[0] >= max(10, PCA_N_COMPONENTS + 3):
        scores, loadings, evr = pca_fit_transform(dim_mat2, n_components=PCA_N_COMPONENTS, standardize=PCA_STANDARDIZE)

        # save loadings
        load_df = pd.DataFrame(loadings, index=dims, columns=[f"PC{i+1}" for i in range(loadings.shape[1])])
        load_df.to_csv(pca_dir / "pca_loadings.csv")

        evr_df = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(evr))], "explained_var_ratio": evr})
        evr_df.to_csv(pca_dir / "pca_explained_variance.csv", index=False)

        # analyze PCs
        pc_rows = []
        for j in range(scores.shape[1]):
            pc = scores[:, j]
            boot = bootstrap_spearman(pc, y2, B=BOOTSTRAP_B, seed=RANDOM_SEED + 100 + j)
            bay = bayes_spearman_uniform_prior(pc, y2, seed=RANDOM_SEED + 200 + j)
            loo = loocv_spearman(pc, y2) if DO_LOOCV else {}
            row = {"PC": f"PC{j+1}", "n": int(len(pc)), "explained_var_ratio": float(evr[j])}
            if bay is not None:
                row.update({f"bayes_{k}": v for k, v in bay.items()})
            row.update({f"boot_{k}": v for k, v in boot.items()})
            row.update({f"loocv_{k}": v for k, v in loo.items()})
            pc_rows.append(row)

        pc_out = pd.DataFrame(pc_rows).sort_values("bayes_BF10", ascending=False, na_position="last")
        pc_out.to_csv(pca_dir / "pca_pc_results.csv", index=False)

        # plot best PC by BF10 (or by |boot_rho|)
        if "bayes_BF10" in pc_out.columns and np.isfinite(pc_out["bayes_BF10"]).any():
            best_pc = pc_out.iloc[0]["PC"]
            jbest = int(str(best_pc).replace("PC", "")) - 1
        else:
            jbest = int(np.nanargmax(np.abs(pc_out["boot_rho"].to_numpy())))
            best_pc = f"PC{jbest+1}"

        plot_joint_scatter(
            scores[:, jbest], y2,
            xlabel=f"Cultural PCA {best_pc}",
            ylabel="Violence_part2_all",
            title=f"Part2: {best_pc} vs Violence (EVR={evr[jbest]:.3f})",
            out_png=pca_dir / f"joint_{best_pc}.png",
            out_pdf=pca_dir / f"joint_{best_pc}.pdf",
        )
    else:
        (pca_dir / "README_pca_skipped.txt").write_text(
            "PCA skipped: too few complete cases with all Hofstede dimensions present.\n",
            encoding="utf-8"
        )

    # --------------------------------------------------------
    # 3) Meta-analysis across all analyses (optional)
    # --------------------------------------------------------
    meta_dir = OUT_ROOT / "meta_analysis"
    meta_dir.mkdir(parents=True, exist_ok=True)

    if META_RESULTS_ROOT is None:
        (meta_dir / "README_meta_skipped.txt").write_text("META_RESULTS_ROOT is None (skipped).\n", encoding="utf-8")
    else:
        if META_RESULTS_ROOT.exists():
            try:
                meta = meta_analyze_all_dims(META_RESULTS_ROOT, meta_dir)
                # also save a short “top dims” file
                meta.head(20).to_csv(meta_dir / "meta_top20_dims.csv", index=False)
            except Exception as e:
                (meta_dir / "README_meta_failed.txt").write_text(f"Meta-analysis failed:\n{e}\n", encoding="utf-8")
        else:
            (meta_dir / "README_meta_skipped.txt").write_text(
                f"META_RESULTS_ROOT does not exist: {META_RESULTS_ROOT}\n", encoding="utf-8"
            )

    print("\n[OK] Finished Part2 + bootstrap + LOOCV + PCA + (optional) meta-analysis.")
    print("[OK] Output folder:", OUT_ROOT.resolve())
    print("[OK] Best dim by BF10:", best_dim)
    print(out[["Hofstede_dim", "spearman_r", "BF10", "P_rho_gt_0_post", "boot_P_rho_gt_0_boot"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()



# %% Cell 2
# hofstede_part2_bayesian_nopymc_plus_FULL.py
# ------------------------------------------------------------
# Part 2: Hofstede vs Violence (Part2 clean table)
# + Bayesian Spearman WITHOUT PyMC (Fisher-z likelihood approx + Uniform prior on rho)
# + Directional bootstrap P(rho>0)
# + Cultural PCA (PCA on Hofstede dims) + Spearman vs outcome
# + Leave-one-country-out (LOOCV) sensitivity
# + Meta-analysis across ALL your analyses (scan spearman_bootstrap_*.csv, random-effects Fisher-z)
# + High-impact (paper-ready) plots:
#     Fig1 forest rho ± bootstrap CI (dims)
#     Fig2 log10(BF10) + directional posterior probability
#     Fig3 LOOCV min–max range + full-sample dot (dims)
#     Fig4 PCA components (rho ± bootstrap CI, explained variance)
#     Fig5 Meta-analysis (random-effects) forest (dims)
#
# Dependencies:
#   pip install numpy pandas scipy matplotlib
# Optional (recommended):
#   pip install pycountry
#   pip install scikit-learn
#
# Notes:
# - Uses only country-level aggregated outcome from Part2 (as your script).
# - Bayesian BF is Savage–Dickey with Uniform(-1,1) prior on rho.
# - Meta-analysis expects your previous scripts wrote "spearman_bootstrap_*.csv"
#   with columns: Hofstede_dim, Spearman_rho, n (CI cols optional).
# ------------------------------------------------------------

from __future__ import annotations

import re
from pathlib import Path
from difflib import get_close_matches
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import rankdata, norm
from scipy.integrate import quad

# ----------------------------
# USER SETTINGS
# ----------------------------
HOFSTEDE_CSV = Path("hofstede_country_scores.csv")
PART2_CLEAN  = Path("results_paper_ready/part2_demographics_only_xgb/clean_modeling_table.csv")

# Where to write this script's outputs
OUT_ROOT = Path("results_paper_ready/hofstede_vs_violence_part2_bayes_nopymc_plus_FULL")

# Aggregation (country-level violence index from Part2)
AGG_MODE = "median"     # "mean" | "median" | "trimmed_mean"
TRIM_Q = 0.10
MIN_N_PER_COUNTRY = 20

# Matching
FUZZY_CUTOFF = 0.86

# Bayesian posterior sampling for summaries (importance resampling)
POST_SAMPLES = 200000
RESAMPLE_N   = 20000
RANDOM_SEED  = 42

# Directional bootstrap
BOOT_B = 20000

# Cultural PCA
PCA_N_COMPONENTS = 3

# Meta-analysis: point to your global results folder that contains many spearman_bootstrap_*.csv
# Example:
#   Path("results_paper_ready/hofstede_vs_violence_ABCD")
# Or set to None to skip.
GLOBAL_RESULTS_ROOT: Optional[Path] = Path("results_paper_ready/hofstede_vs_violence_ABCD")
META_PATTERN = "spearman_bootstrap_*.csv"

# Plot aesthetics
SCATTER_ALPHA = 0.85
DPI_EXPORT = 600

# ----------------------------
# Optional dependencies
# ----------------------------
try:
    import pycountry  # type: ignore
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False

try:
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# ============================================================
# Helpers: strings, iso3, matching
# ============================================================
def _clean_country(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_country_name(name: str) -> str:
    name = _clean_country(name)
    overrides = {
        "Russian Federation": "Russia",
        "Viet Nam": "Vietnam",
        "Iran, Islamic Rep.": "Iran",
        "Iran (Islamic Republic of)": "Iran",
        "Korea, Rep.": "South Korea",
        "Korea Rep.": "South Korea",
        "Korea, Dem. People's Rep.": "North Korea",
        "Czech Rep.": "Czech Republic",
        "Slovak Rep.": "Slovakia",
        "Egypt, Arab Rep.": "Egypt",
        "Yemen, Rep.": "Yemen",
        "Venezuela, RB": "Venezuela",
        "Bolivia (Plurinational State of)": "Bolivia",
        "Tanzania, United Rep.": "Tanzania",
        "United States of America": "United States",
        "USA": "United States",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "Côte d’Ivoire": "Ivory Coast",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "Lao PDR": "Laos",
        "Syrian Arab Republic": "Syria",
        "Türkiye": "Turkey",
        "Palestine": "Palestine, State of",
        "Cabo Verde": "Cape Verde",
        "Eswatini": "Swaziland",
    }
    return overrides.get(name, name)

def _iso3_from_name(name: str) -> Optional[str]:
    if not name or str(name).lower() == "nan":
        return None
    name = normalize_country_name(name)

    overrides = {
        "Russia": "RUS",
        "United States": "USA",
        "United Kingdom": "GBR",
        "Ivory Coast": "CIV",
        "Vietnam": "VNM",
        "Laos": "LAO",
        "Syria": "SYR",
        "Turkey": "TUR",
        "Bolivia": "BOL",
        "Venezuela": "VEN",
        "Tanzania": "TZA",
        "South Korea": "KOR",
        "North Korea": "PRK",
        "Czech Republic": "CZE",
        "Palestine, State of": "PSE",
        "Iran": "IRN",
        "Cape Verde": "CPV",
        "Swaziland": "SWZ",
    }
    if name in overrides:
        return overrides[name]

    if not HAS_PYCOUNTRY:
        return None

    try:
        c = pycountry.countries.lookup(name)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None

def fuzzy_match_series(left: pd.Series, right_values: list[str], cutoff: float) -> pd.Series:
    right_values = sorted(set([str(v) for v in right_values if pd.notna(v)]))
    out = []
    for x in left.astype(str):
        m = get_close_matches(x, right_values, n=1, cutoff=cutoff)
        out.append(m[0] if m else x)
    return pd.Series(out, index=left.index)

def assert_exists(p: Path, label: str):
    if not p.exists():
        raise FileNotFoundError(f"[{label}] Cannot find: {p.resolve()}")


# ============================================================
# Aggregation
# ============================================================
def trimmed_mean(x: np.ndarray, q: float = 0.10) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    lo = np.quantile(x, q)
    hi = np.quantile(x, 1 - q)
    x2 = x[(x >= lo) & (x <= hi)]
    return float(np.mean(x2)) if x2.size else float(np.mean(x))

def aggregate_country_values(
    df: pd.DataFrame,
    country_col: str,
    value_col: str,
    min_n: int,
    agg_mode: str,
    trim_q: float,
) -> pd.DataFrame:
    d = df.copy()
    d[country_col] = d[country_col].astype(str).map(normalize_country_name)
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    out_rows = []
    for c, g in d.groupby(country_col):
        vals = pd.to_numeric(g[value_col], errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size < min_n:
            continue

        if agg_mode == "mean":
            agg = float(np.mean(vals))
        elif agg_mode == "median":
            agg = float(np.median(vals))
        elif agg_mode == "trimmed_mean":
            agg = trimmed_mean(vals, q=trim_q)
        else:
            raise ValueError("AGG_MODE must be mean|median|trimmed_mean")

        out_rows.append({
            "Country_norm": c,
            "Violence_part2_all": agg,
            "n_points": int(vals.size),
        })

    cols = ["Country_norm", "Violence_part2_all", "n_points"]
    if not out_rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(out_rows)[cols]
    out["iso3"] = out["Country_norm"].map(_iso3_from_name)
    return out.sort_values("Country_norm").reset_index(drop=True)


# ============================================================
# Load Hofstede
# ============================================================
def load_hofstede(hofstede_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    hof = pd.read_csv(hofstede_csv)

    country_col = None
    for c in hof.columns:
        if str(c).strip().lower() == "country":
            country_col = c
            break
    if country_col is None:
        raise ValueError(f"Hofstede CSV must have a country column. Columns: {list(hof.columns)}")
    if country_col != "Country":
        hof = hof.rename(columns={country_col: "Country"})

    hof["Country"] = hof["Country"].astype(str).map(_clean_country)
    hof["Country_norm"] = hof["Country"].map(normalize_country_name)
    hof["iso3"] = hof["Country_norm"].map(_iso3_from_name).astype("string").str.upper()

    # critical: avoid duplicates by iso3
    hof = hof.dropna(subset=["iso3"]).drop_duplicates(subset=["iso3"], keep="first").copy()

    dims = []
    for col in hof.columns:
        if str(col).lower() in {"country", "country_norm", "iso3"}:
            continue
        s = pd.to_numeric(hof[col], errors="coerce")
        if np.isfinite(s).sum() >= 10:
            dims.append(col)

    if not dims:
        raise ValueError("No numeric Hofstede dimensions detected.")
    return hof[["Country", "Country_norm", "iso3"] + dims].copy(), dims


# ============================================================
# Merge iso3 then fuzzy
# ============================================================
def merge_iso3_then_fuzzy(hof: pd.DataFrame, v: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = hof.copy()
    vv = v.copy()

    vv["Country_norm"] = vv["Country_norm"].astype(str).map(normalize_country_name)
    vv["iso3"] = vv["iso3"].astype("string").str.upper()

    merged = vv.merge(h, on="iso3", how="left", suffixes=("", "_hof"))

    # robust audit aligned with merged length
    audit = pd.DataFrame({
        "Country_norm": merged["Country_norm"].astype(str),
        "iso3": merged["iso3"].astype(str),
        "matched_iso3": ~merged["Country"].isna(),
    })

    missing = merged["Country"].isna()

    if missing.any():
        candidates = h["Country_norm"].dropna().astype(str).tolist()
        fuzzy_target = fuzzy_match_series(merged.loc[missing, "Country_norm"], candidates, cutoff=FUZZY_CUTOFF)
        audit.loc[missing, "Country_norm_fuzzy"] = fuzzy_target.to_numpy()

        h2 = h.rename(columns={"Country_norm": "Country_norm_fuzzy"})
        tmp = merged.loc[missing].copy()
        tmp["Country_norm_fuzzy"] = fuzzy_target.to_numpy()
        filled = tmp.merge(h2, on="Country_norm_fuzzy", how="left", suffixes=("", "_fuzzyfill"))

        # Fill Hofstede columns back
        for col in [c for c in h.columns if c != "iso3"]:
            if col in filled.columns:
                merged.loc[missing, col] = filled[col].to_numpy()

        audit.loc[missing, "matched_fuzzy"] = ~filled["Country"].isna().to_numpy()
    else:
        audit["Country_norm_fuzzy"] = ""
        audit["matched_fuzzy"] = False

    return merged, audit


# ============================================================
# Spearman + Bayesian BF (no PyMC)
# ============================================================
def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return np.nan
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])

def posterior_z_norm_const(z_hat: float, se: float) -> float:
    # posterior in z: p(z|data) ∝ Normal(z | z_hat, se) * sech^2(z)
    # sech^2(z) = 1 / cosh(z)^2
    def unnorm(z):
        return norm.pdf(z, loc=z_hat, scale=se) / (np.cosh(z) ** 2)
    return float(quad(unnorm, -20, 20, limit=200)[0])

def bayes_spearman_uniform_prior(x: np.ndarray, y: np.ndarray, seed: int):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 8:
        return None

    r_s = spearman_rho(x, y)
    r_s = float(np.clip(r_s, -0.999999, 0.999999))

    # Fisher z likelihood approximation
    z_hat = float(np.arctanh(r_s))
    se = float(1.0 / np.sqrt(max(n - 3, 1)))

    Z = posterior_z_norm_const(z_hat, se)

    # posterior density at rho=0 (rho=0 <-> z=0; Jacobian at 0 is 1)
    post0 = float(norm.pdf(0.0, loc=z_hat, scale=se) / Z)  # sech^2(0)=1
    prior0 = 0.5  # Uniform(-1,1) density at 0

    BF10 = float(prior0 / post0) if post0 > 0 else np.inf

    # Posterior summaries via importance resampling
    rng = np.random.default_rng(seed)
    z_prop = rng.normal(loc=z_hat, scale=se, size=POST_SAMPLES)
    w = 1.0 / (np.cosh(z_prop) ** 2)  # weights ∝ sech^2(z)
    w = w / np.sum(w)
    idx = rng.choice(np.arange(POST_SAMPLES), size=RESAMPLE_N, replace=True, p=w)
    z_post = z_prop[idx]
    rho_post = np.tanh(z_post)

    out = {
        "n": n,
        "spearman_r": float(r_s),
        "rho_mean": float(np.mean(rho_post)),
        "rho_median": float(np.median(rho_post)),
        "rho_ci_low": float(np.quantile(rho_post, 0.025)),
        "rho_ci_high": float(np.quantile(rho_post, 0.975)),
        "P_rho_gt_0_post": float(np.mean(rho_post > 0)),
        "prior_density_rho0": float(prior0),
        "posterior_density_rho0": float(post0),
        "BF10": float(BF10),
    }
    return out


# ============================================================
# Directional bootstrap P(rho>0)
# ============================================================
def directional_bootstrap_spearman(x, y, B=20000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(x.size)
    if n < 8:
        return np.nan, np.nan, np.nan, np.nan, n

    def rho(a, b):
        ra = rankdata(a)
        rb = rankdata(b)
        return float(np.corrcoef(ra, rb)[0, 1])

    r0 = rho(x, y)
    boots = np.empty(B, float)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        boots[i] = rho(x[idx], y[idx])

    lo, hi = np.quantile(boots, [0.025, 0.975])
    p_gt0 = float(np.mean(boots > 0))
    return float(r0), float(lo), float(hi), p_gt0, n


# ============================================================
# LOOCV sensitivity
# ============================================================
def leave_one_country_out(merged_df, dim_col, y_col, country_col="Country_norm"):
    d = merged_df.copy()
    x = pd.to_numeric(d[dim_col], errors="coerce")
    y = pd.to_numeric(d[y_col], errors="coerce")
    keep = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    d = d.loc[keep].copy()
    if len(d) < 10:
        return None

    full = spearman_rho(d[dim_col].to_numpy(), d[y_col].to_numpy())
    rhos = []
    for c in d[country_col].astype(str).unique():
        dd = d[d[country_col].astype(str) != c]
        rhos.append(spearman_rho(dd[dim_col].to_numpy(), dd[y_col].to_numpy()))
    rhos = np.asarray(rhos, float)

    return {
        "loocv_rho_full": float(full),
        "loocv_min": float(np.nanmin(rhos)),
        "loocv_max": float(np.nanmax(rhos)),
        "loocv_mean": float(np.nanmean(rhos)),
        "loocv_sd": float(np.nanstd(rhos, ddof=1)),
        "n_countries": int(d[country_col].nunique()),
        "n_points": int(len(d)),
    }


# ============================================================
# Cultural PCA
# ============================================================
def cultural_pca_and_correlate(merged_df, hofstede_dims, y_col, n_components=3, seed=0):
    if not HAS_SKLEARN:
        return pd.DataFrame(), pd.DataFrame()

    d = merged_df.copy()
    X = d[hofstede_dims].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(d[y_col], errors="coerce")

    mask = np.isfinite(y.to_numpy()) & np.all(np.isfinite(X.to_numpy()), axis=1)
    X = X.loc[mask].to_numpy()
    y = y.loc[mask].to_numpy()
    if X.shape[0] < 10:
        return pd.DataFrame(), pd.DataFrame()

    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(n_components, Xs.shape[1]), random_state=seed)
    PCs = pca.fit_transform(Xs)

    rows = []
    for j in range(PCs.shape[1]):
        r = spearman_rho(PCs[:, j], y)
        rows.append({
            "component": f"PC{j+1}",
            "explained_var": float(pca.explained_variance_ratio_[j]),
            "spearman_rho": float(r),
            "n": int(y.size),
        })

    loadings = pd.DataFrame(
        pca.components_.T,
        index=hofstede_dims,
        columns=[f"PC{j+1}" for j in range(PCs.shape[1])]
    )
    results = pd.DataFrame(rows).sort_values("explained_var", ascending=False)
    return results, loadings


# ============================================================
# Meta-analysis across outputs (random-effects Fisher-z)
# ============================================================
def fisher_z(r: float) -> float:
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))

def inv_fisher_z(z: float) -> float:
    return float(np.tanh(z))

def meta_analysis_fisherz(rhos: np.ndarray, ns: np.ndarray):
    rhos = np.asarray(rhos, float)
    ns = np.asarray(ns, float)
    mask = np.isfinite(rhos) & np.isfinite(ns) & (ns > 6)
    rhos, ns = rhos[mask], ns[mask]
    k = int(rhos.size)
    if k < 2:
        return {"k": k, "rho_fixed": np.nan, "rho_random": np.nan, "tau2": np.nan, "Q": np.nan}

    zi = np.array([fisher_z(r) for r in rhos])
    vi = 1.0 / (ns - 3.0)
    wi = 1.0 / vi

    z_fixed = np.sum(wi * zi) / np.sum(wi)
    Q = np.sum(wi * (zi - z_fixed) ** 2)
    df = k - 1

    C = np.sum(wi) - (np.sum(wi**2) / np.sum(wi))
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0

    wi_star = 1.0 / (vi + tau2)
    z_random = np.sum(wi_star * zi) / np.sum(wi_star)

    return {
        "k": k,
        "rho_fixed": inv_fisher_z(z_fixed),
        "rho_random": inv_fisher_z(z_random),
        "tau2": float(tau2),
        "Q": float(Q),
    }

def meta_analysis_across_outputs(results_root: Path, pattern="spearman_bootstrap_*.csv"):
    files = list(results_root.rglob(pattern))
    if not files:
        return pd.DataFrame()

    all_rows = []
    for f in files:
        try:
            tmp = pd.read_csv(f)
        except Exception:
            continue

        # locate expected columns (case-insensitive)
        cols = {c.lower(): c for c in tmp.columns}
        dimc = cols.get("hofstede_dim")
        rc   = cols.get("spearman_rho")
        nc   = cols.get("n")

        if not (dimc and rc and nc):
            continue

        sub = tmp[[dimc, rc, nc]].copy()
        sub = sub.dropna()
        for _, row in sub.iterrows():
            all_rows.append({
                "file": str(f),
                "Hofstede_dim": str(row[dimc]),
                "rho": float(row[rc]),
                "n": float(row[nc]),
            })

    df_all = pd.DataFrame(all_rows)
    if df_all.empty:
        return pd.DataFrame()

    out = []
    for dim, g in df_all.groupby("Hofstede_dim"):
        mr = meta_analysis_fisherz(g["rho"].to_numpy(), g["n"].to_numpy())
        out.append({
            "Hofstede_dim": dim,
            "k_files": mr["k"],
            "rho_fixed": mr["rho_fixed"],
            "rho_random": mr["rho_random"],
            "tau2": mr["tau2"],
            "Q": mr["Q"],
        })
    out = pd.DataFrame(out)
    if out.empty:
        return out
    return out.sort_values(["k_files", "rho_random"], ascending=[False, True]).reset_index(drop=True)


# ============================================================
# Plot styling
# ============================================================
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


# ============================================================
# High-impact plots
# ============================================================
def plot_forest_dims(df: pd.DataFrame, out_png: Path, out_pdf: Path, title: str):
    """
    Forest plot with rho + bootstrap CI for Hofstede dims.
    Expects columns: Hofstede_dim, spearman_r, boot_ci_low, boot_ci_high, BF10, P_rho_gt_0_post, n
    """
    set_pub_style()
    d = df.copy().sort_values("spearman_r", ascending=True).reset_index(drop=True)

    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    ax.axvline(0.0, linewidth=1.2)

    # CI lines
    for i, row in d.iterrows():
        ax.plot([row["boot_ci_low"], row["boot_ci_high"]], [i, i], linewidth=2.0)

    ax.scatter(d["spearman_r"], y, s=55, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([f'{r["Hofstede_dim"]}  (n={int(r["n"])})' for _, r in d.iterrows()])
    ax.set_xlabel("Spearman ρ (country-level violence vs Hofstede dimension)")
    ax.set_title(title)

    # Right-side annotations (BF + direction)
    xmin, xmax = ax.get_xlim()
    xtext = xmax + 0.03 * (xmax - xmin)
    ax.set_xlim(xmin, xmax + 0.28 * (xmax - xmin))
    for i, row in d.iterrows():
        bf = row.get("BF10", np.nan)
        pgt = row.get("P_rho_gt_0_post", np.nan)
        ax.text(
            xtext, i,
            f"BF10={bf:.2f} | P(ρ>0)={pgt:.3f}",
            va="center", fontsize=10
        )

    savefig(fig, out_png, out_pdf)

def plot_bayesfactor_direction(df: pd.DataFrame, out_png: Path, out_pdf: Path, title: str):
    """
    Two-panel: log10(BF10) and P(ρ>0) posterior.
    """
    set_pub_style()
    d = df.copy().sort_values("BF10", ascending=True).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5), dpi=300, sharey=True)
    y = np.arange(len(d))
    labels = [f'{r["Hofstede_dim"]} (n={int(r["n"])})' for _, r in d.iterrows()]

    bf = pd.to_numeric(d["BF10"], errors="coerce").to_numpy()
    logbf = np.log10(np.clip(bf, 1e-6, 1e6))
    ax1.axvline(0.0, linewidth=1.2)
    ax1.scatter(logbf, y, s=55)
    ax1.set_xlabel("log10(BF10)")
    ax1.set_title("Bayesian evidence")

    pgt = pd.to_numeric(d["P_rho_gt_0_post"], errors="coerce").to_numpy()
    ax2.axvline(0.5, linewidth=1.2)
    ax2.scatter(pgt, y, s=55)
    ax2.set_xlabel("Posterior P(ρ > 0)")
    ax2.set_title("Directional probability")

    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    fig.suptitle(title, y=1.02)

    savefig(fig, out_png, out_pdf)

def plot_loocv_ranges(df_loocv: pd.DataFrame, out_png: Path, out_pdf: Path, title: str):
    """
    LOOCV range plot: min–max horizontal line, dot at full-sample rho.
    Expects columns: Hofstede_dim, loocv_min, loocv_max, loocv_rho_full
    """
    set_pub_style()
    d = df_loocv.copy().sort_values("loocv_rho_full", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    ax.axvline(0.0, linewidth=1.2)

    for i, row in d.iterrows():
        ax.plot([row["loocv_min"], row["loocv_max"]], [i, i], linewidth=2.0)
    ax.scatter(d["loocv_rho_full"], y, s=55, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(d["Hofstede_dim"].astype(str).tolist())
    ax.set_xlabel("Spearman ρ (leave-one-country-out range)")
    ax.set_title(title)

    savefig(fig, out_png, out_pdf)

def plot_pca_components(pca_stats: pd.DataFrame, pca_boot: pd.DataFrame, out_png: Path, out_pdf: Path, title: str):
    """
    PCA plot: rho ± bootstrap CI for PC1..PCk, with explained variance in labels.
    """
    if pca_stats.empty or pca_boot.empty:
        return
    set_pub_style()
    d = pca_stats.merge(pca_boot, on="component", how="left")
    d = d.sort_values("component").reset_index(drop=True)

    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9.0, 3.8), dpi=300)
    ax.axvline(0.0, linewidth=1.2)

    for i, row in d.iterrows():
        ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], linewidth=2.0)
    ax.scatter(d["rho"], y, s=55, zorder=3)

    labels = [f'{row["component"]}  (var={row["explained_var"]:.2f})' for _, row in d.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Spearman ρ (violence vs cultural PCA component)")
    ax.set_title(title)
    savefig(fig, out_png, out_pdf)

def plot_meta_forest(meta_df: pd.DataFrame, out_png: Path, out_pdf: Path, title: str):
    """
    Meta-analysis forest plot using random-effects rho_random.
    """
    if meta_df.empty:
        return
    set_pub_style()
    d = meta_df.copy()
    d = d.sort_values(["k_files", "rho_random"], ascending=[False, True]).reset_index(drop=True)
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=300)
    ax.axvline(0.0, linewidth=1.2)
    ax.scatter(d["rho_random"], y, s=55)

    ax.set_yticks(y)
    ax.set_yticklabels([f'{r["Hofstede_dim"]} (k={int(r["k_files"])})' for _, r in d.iterrows()])
    ax.set_xlabel("Random-effects meta-analytic Spearman ρ (Fisher-z)")
    ax.set_title(title)

    savefig(fig, out_png, out_pdf)


# ============================================================
# MAIN
# ============================================================
def main():
    assert_exists(HOFSTEDE_CSV, "HOFSTEDE_CSV")
    assert_exists(PART2_CLEAN, "PART2_CLEAN")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load Hofstede and Part2
    hof, dims = load_hofstede(HOFSTEDE_CSV)
    part2 = pd.read_csv(PART2_CLEAN)
    if not {"Country", "Value"}.issubset(part2.columns):
        raise ValueError(f"PART2_CLEAN must include Country and Value. Got columns: {list(part2.columns)}")

    # Aggregate outcome per country
    v2 = aggregate_country_values(
        part2,
        country_col="Country",
        value_col="Value",
        min_n=MIN_N_PER_COUNTRY,
        agg_mode=AGG_MODE,
        trim_q=TRIM_Q,
    )
    if len(v2) < 10:
        raise ValueError(f"Too few countries after aggregation (len={len(v2)}). Lower MIN_N_PER_COUNTRY?")

    # Merge Hofstede
    merged, audit = merge_iso3_then_fuzzy(hof, v2)
    merged.to_csv(OUT_ROOT / "analysis_table_part2_allcountries.csv", index=False)
    audit.to_csv(OUT_ROOT / "merge_audit_part2.csv", index=False)

    # Keep rows with outcome and at least one dim
    keep = merged.dropna(subset=["Violence_part2_all"]).copy()

    # ----------------------------
    # Compute per-dimension stats
    # ----------------------------
    rows = []
    for d in dims:
        x = pd.to_numeric(keep[d], errors="coerce").to_numpy()
        y = pd.to_numeric(keep["Violence_part2_all"], errors="coerce").to_numpy()

        b = bayes_spearman_uniform_prior(x, y, seed=RANDOM_SEED)
        if b is None:
            continue

        r0, lo, hi, pgt0_boot, n_boot = directional_bootstrap_spearman(x, y, B=BOOT_B, seed=RANDOM_SEED)

        out = {
            "Hofstede_dim": d,
            "n": int(b["n"]),
            "spearman_r": float(b["spearman_r"]),
            "BF10": float(b["BF10"]),
            "P_rho_gt_0_post": float(b["P_rho_gt_0_post"]),
            "boot_ci_low": float(lo),
            "boot_ci_high": float(hi),
            "boot_P_rho_gt_0_boot": float(pgt0_boot),
        }
        rows.append(out)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No results computed (too few valid rows per dimension).")

    df = df.sort_values("BF10", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_ROOT / "bayes_spearman_part2_all_plus.csv", index=False)

    # ----------------------------
    # LOOCV sensitivity
    # ----------------------------
    loocv_rows = []
    for d in dims:
        res = leave_one_country_out(keep, dim_col=d, y_col="Violence_part2_all")
        if res is None:
            continue
        res["Hofstede_dim"] = d
        loocv_rows.append(res)
    df_loocv = pd.DataFrame(loocv_rows)
    df_loocv.to_csv(OUT_ROOT / "loocv_sensitivity_by_dim.csv", index=False)

    # ----------------------------
    # Cultural PCA
    # ----------------------------
    pca_stats, pca_loadings = cultural_pca_and_correlate(
        keep, hofstede_dims=dims, y_col="Violence_part2_all", n_components=PCA_N_COMPONENTS, seed=0
    )
    if not pca_stats.empty:
        pca_stats.to_csv(OUT_ROOT / "cultural_pca_spearman.csv", index=False)
        pca_loadings.to_csv(OUT_ROOT / "cultural_pca_loadings.csv", index=True)

        # bootstrap PCs (compute bootstrap CI for each PC score vs y)
        # We recompute PCA once and use its scores for bootstrap correlation (good enough for a robustness check).
        # If you want full "PCA-in-the-loop bootstrap", say so and I’ll adapt.
        if HAS_SKLEARN:
            dtmp = keep.copy()
            X = dtmp[dims].apply(pd.to_numeric, errors="coerce")
            y = pd.to_numeric(dtmp["Violence_part2_all"], errors="coerce")
            mask = np.isfinite(y.to_numpy()) & np.all(np.isfinite(X.to_numpy()), axis=1)
            X = X.loc[mask].to_numpy()
            yv = y.loc[mask].to_numpy()

            Xs = StandardScaler().fit_transform(X)
            pca = PCA(n_components=min(PCA_N_COMPONENTS, Xs.shape[1]), random_state=0)
            PCs = pca.fit_transform(Xs)

            pca_boot_rows = []
            for j in range(PCs.shape[1]):
                r0, lo, hi, pgt0, n = directional_bootstrap_spearman(
                    PCs[:, j], yv, B=BOOT_B, seed=RANDOM_SEED
                )
                pca_boot_rows.append({
                    "component": f"PC{j+1}",
                    "rho": r0,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "P_rho_gt_0": pgt0,
                    "n": n,
                })
            pca_boot = pd.DataFrame(pca_boot_rows)
            pca_boot.to_csv(OUT_ROOT / "cultural_pca_bootstrap.csv", index=False)
        else:
            pca_boot = pd.DataFrame()
    else:
        pca_boot = pd.DataFrame()

    # ----------------------------
    # Meta-analysis across all your prior outputs (optional)
    # ----------------------------
    meta_df = pd.DataFrame()
    if GLOBAL_RESULTS_ROOT is not None:
        if GLOBAL_RESULTS_ROOT.exists():
            meta_df = meta_analysis_across_outputs(GLOBAL_RESULTS_ROOT, pattern=META_PATTERN)
            meta_df.to_csv(OUT_ROOT / "meta_analysis_across_outputs.csv", index=False)
        else:
            (OUT_ROOT / "meta_analysis_SKIPPED.txt").write_text(
                f"GLOBAL_RESULTS_ROOT does not exist: {GLOBAL_RESULTS_ROOT}\n",
                encoding="utf-8"
            )

    # ----------------------------
    # Plots
    # ----------------------------
    plot_forest_dims(
        df,
        out_png=fig_dir / "Fig1_forest_spearman_bootstrap.png",
        out_pdf=fig_dir / "Fig1_forest_spearman_bootstrap.pdf",
        title="Part 2 (all countries): Spearman ρ with bootstrap 95% CI (dims)",
    )

    plot_bayesfactor_direction(
        df,
        out_png=fig_dir / "Fig2_bayesfactor_directional.png",
        out_pdf=fig_dir / "Fig2_bayesfactor_directional.pdf",
        title="Part 2 (all countries): Bayesian evidence and directional probability (dims)",
    )

    if not df_loocv.empty:
        plot_loocv_ranges(
            df_loocv,
            out_png=fig_dir / "Fig3_loocv_sensitivity.png",
            out_pdf=fig_dir / "Fig3_loocv_sensitivity.pdf",
            title="Part 2 (all countries): Leave-one-country-out sensitivity (dims)",
        )

    if not pca_stats.empty and not pca_boot.empty:
        plot_pca_components(
            pca_stats,
            pca_boot,
            out_png=fig_dir / "Fig4_cultural_pca_components.png",
            out_pdf=fig_dir / "Fig4_cultural_pca_components.pdf",
            title="Cultural PCA: Spearman ρ with bootstrap 95% CI",
        )

    if not meta_df.empty:
        plot_meta_forest(
            meta_df,
            out_png=fig_dir / "Fig5_meta_analysis_forest.png",
            out_pdf=fig_dir / "Fig5_meta_analysis_forest.pdf",
            title="Meta-analysis across outputs: random-effects ρ (dims)",
        )

    # Also print a compact table to console
    top = df.sort_values("BF10", ascending=False).head(10)
    print("\n[OK] Finished Part 2 + extras.")
    print("[OK] Output folder:", OUT_ROOT.resolve())
    print("\nTop dimensions by BF10:")
    print(top[["Hofstede_dim", "spearman_r", "BF10", "P_rho_gt_0_post", "boot_ci_low", "boot_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()



# %% Cell 3

