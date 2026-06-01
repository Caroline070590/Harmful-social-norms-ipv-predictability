# Auto-extracted from notebooks/13_web_scraping_unwomen_measures.ipynb

# Review/reproducibility helper script. Original notebook is retained in notebooks/.



# %% Cell 0
# download_hierarchy_datasets_fixed.py
# ------------------------------------------------------------
# Downloads / prepares cross-national datasets for "hierarchy / patriarchy"
# to complement Hofstede in violence-against-women analysis.
#
# What this script downloads automatically (public, stable-ish):
#   1) UNDP HDR: "All composite indices and components time series" (CSV)
#   2) V-Dem proxies via Our World in Data grapher CSV endpoints:
#        - Electoral democracy index
#        - Liberal democracy index
#        - Women's political empowerment index
#
# What it does NOT scrape automatically (often requires terms/login/JS):
#   - World Values Survey (WVS) microdata
#   - ISSP microdata (GESIS)
#   - Social Dominance Orientation (SDO) cross-national (study-specific)
#   - Full historical WEF GGGI bulk data (links change / PDFs / ToS)
#
# Dependencies:
#   pip install pandas numpy requests
# ------------------------------------------------------------

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Optional

import requests
import pandas as pd


# ----------------------------
# USER SETTINGS
# ----------------------------
OUT_DIR = Path("external_country_datasets_v2")  # NEW folder (as you asked)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEOUT = 60
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}

# UNDP downloads page (we'll auto-discover the CSV link from HTML)
UNDP_DOWNLOADS_PAGE = "https://hdr.undp.org/data-center/documentation-and-downloads"

# Fallback (in case HTML changes / site blocks listing)
UNDP_FALLBACK_CSV = "https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Composite_indices_complete_time_series.csv"

# Our World In Data Grapher slugs (public CSV endpoints)
OWID_SPECS = {
    "vdem_electoral_democracy_index": "electoral-democracy-index",
    "vdem_liberal_democracy_index": "liberal-democracy-index",
    "vdem_womens_political_empowerment_index": "women-political-empowerment-index",
}


# ----------------------------
# Helpers
# ----------------------------
def download_file(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    print(f"[OK] Downloaded: {out_path} ({len(r.content)/1024:.1f} KB)")
    return out_path


def fetch_text(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text


def try_read_csv(path: Path) -> pd.DataFrame:
    # Try common encodings robustly
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


# ============================================================
# 1) UNDP HDR: auto-discover current "Composite indices...time series" CSV link
# ============================================================
def find_undp_composite_timeseries_csv() -> str:
    """
    Parse the UNDP documentation/downloads page HTML and find a CSV URL that looks like:
      *Composite_indices*complete*time_series*.csv

    Returns URL string. Falls back to UNDP_FALLBACK_CSV if not found.
    """
    try:
        html = fetch_text(UNDP_DOWNLOADS_PAGE)
    except Exception as e:
        print(f"[WARN] Could not fetch UNDP downloads page: {e}")
        print(f"       -> Using fallback: {UNDP_FALLBACK_CSV}")
        return UNDP_FALLBACK_CSV

    # Find all URLs in the HTML
    urls = set(re.findall(r"https?://[^\s\"'>]+", html))

    # Prefer explicit Composite_indices* csv
    patt = re.compile(r"Composite_indices.*time_series.*\.csv", re.IGNORECASE)
    hits = [u for u in urls if patt.search(u)]

    # Also allow relative links (sometimes href="/sites/default/files/...csv")
    rels = set(re.findall(r'href="([^"]+\.csv)"', html, flags=re.IGNORECASE))
    for rel in rels:
        if "Composite" in rel or "composite" in rel:
            if rel.startswith("/"):
                hits.append("https://hdr.undp.org" + rel)
            elif rel.startswith("http"):
                hits.append(rel)

    # Pick the best candidate
    if hits:
        # Prefer hdr.undp.org/sites/default/files/...csv
        hits = sorted(set(hits))
        best = next((u for u in hits if "hdr.undp.org/sites/default/files" in u), hits[0])
        return best

    print("[WARN] Could not locate Composite indices CSV link in HTML.")
    print(f"       -> Using fallback: {UNDP_FALLBACK_CSV}")
    return UNDP_FALLBACK_CSV


def download_undp_hdr_composite_timeseries() -> Path:
    url = find_undp_composite_timeseries_csv()
    out_path = OUT_DIR / "undp_hdr" / "HDR_composite_indices_complete_time_series.csv"
    return download_file(url, out_path)


def extract_country_year_indicator(df: pd.DataFrame, indicator_candidates: list[str]) -> Optional[pd.DataFrame]:
    """
    Attempt to extract (Country, Year, indicator) from UNDP composite time series file.

    Because column names vary slightly by release, this tries:
      - Country column: 'country' | 'Country' | 'Country or territory' | 'country_name'
      - Year column: 'year' | 'Year' | 'time'
      - Indicator column: any exact match to candidates OR close match (case-insensitive)
    """
    cols = [str(c).strip() for c in df.columns]
    df = df.copy()
    df.columns = cols

    # country
    country_col = next((c for c in cols if c.lower() in {"country", "country_name", "country or territory"}), None)
    year_col = next((c for c in cols if c.lower() in {"year", "time"}), None)

    # indicator: try exact, then case-insensitive contains
    ind_col = None
    cand_lower = [c.lower() for c in indicator_candidates]
    for c in cols:
        if c.lower() in cand_lower:
            ind_col = c
            break
    if ind_col is None:
        for c in cols:
            if any(cl in c.lower() for cl in cand_lower):
                ind_col = c
                break

    if country_col is None or year_col is None or ind_col is None:
        return None

    out = df[[country_col, year_col, ind_col]].copy()
    out = out.rename(columns={country_col: "Country", year_col: "Year", ind_col: "Value"})
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out = out.dropna(subset=["Country", "Year"])
    out["Indicator"] = ind_col
    return out


# ============================================================
# 2) V-Dem proxies via OWID Grapher CSV endpoints
# ============================================================
def download_owid_grapher_csv(slug: str, out_path: Path) -> Path:
    url = f"https://ourworldindata.org/grapher/{slug}.csv"
    return download_file(url, out_path)


def download_vdem_proxies() -> dict[str, Path]:
    out = {}
    vdir = OUT_DIR / "vdem_owid"
    vdir.mkdir(parents=True, exist_ok=True)

    for name, slug in OWID_SPECS.items():
        try:
            out[name] = download_owid_grapher_csv(slug, vdir / f"{safe_filename(slug)}.csv")
        except Exception as e:
            print(f"[WARN] Could not download {name} (slug='{slug}'): {e}")
    return out


# ============================================================
# 3) Guidance for datasets that often require manual download
# ============================================================
def print_manual_download_instructions():
    msg = """
    === Manual-download datasets (recommended, reproducible) ===

    1) WEF Global Gender Gap Index (GGGI):
       - Often provided as report PDFs and/or year-specific tables.
       - Put any CSV/XLSX you download here:
           external_country_datasets_v2/gggi_wef/raw/

    2) World Values Survey (WVS):
       - Registration/terms; microdata formats (.sav/.dta).
       - Put downloads here:
           external_country_datasets_v2/wvs/raw/

    3) ISSP (GESIS):
       - Similar constraints; put downloads here:
           external_country_datasets_v2/issp/raw/

    4) SDO cross-national:
       - No single canonical public dataset; best derived from WVS/ISSP hierarchy/authority items
         or from specific published studies.
    """
    print(textwrap.dedent(msg))


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n=== 1) UNDP HDR composite indices time series ===")
    undp_path = download_undp_hdr_composite_timeseries()
    undp = try_read_csv(undp_path)
    print(f"[OK] UNDP rows={len(undp):,} cols={len(undp.columns):,}")

    # Try extracting a GII-like series if available
    gii = extract_country_year_indicator(
        undp,
        indicator_candidates=["gii", "Gender Inequality Index", "GII"]
    )
    if gii is None:
        print("[WARN] Could not auto-extract GII from the UNDP file.")
        print("       -> Inspect columns in the CSV and update indicator_candidates if needed.")
    else:
        gii_out = OUT_DIR / "undp_hdr" / "GII_country_year.csv"
        gii.to_csv(gii_out, index=False)
        print(f"[OK] Extracted GII-like indicator to: {gii_out}")

    print("\n=== 2) V-Dem proxies via Our World in Data (Grapher) ===")
    download_vdem_proxies()

    print("\n=== 3) Manual datasets (GGGI / WVS / ISSP / SDO) ===")
    print_manual_download_instructions()

    print("\n[DONE] Files are in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()



# %% Cell 1

