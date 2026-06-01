# Auto-extracted from notebooks/legacy/legacy_untitled2.ipynb

# Review/reproducibility helper script. Original notebook is retained in notebooks/.



# %% Cell 0
import pandas as pd

path = "Part2/Data Map-export.csv"

with open(path, "r", encoding="utf-8", errors="replace") as f:
    for i in range(10):
        print(f"{i+1:02d}:", f.readline().rstrip("\n"))



# %% Cell 1
import pandas as pd

path = "Part2/Data Map-export.csv"

df_raw = pd.read_csv(
    path,
    sep=",",
    engine="python",
    encoding="utf-8-sig",
    quotechar='"'
)

print(df_raw.shape)
print(df_raw.columns)
df_raw.head()



# %% Cell 2
df_raw = pd.read_csv(
    path,
    sep=",",
    engine="python",
    encoding="utf-8-sig",
    quotechar='"',
    on_bad_lines="skip"
)
print(df_raw.shape)
print(df_raw.columns)
df_raw.head()


# %% Cell 3
df = df_raw.copy()

df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)

df.head()



# %% Cell 4
df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce").astype("Int64")
df = df.dropna(subset=["start_year"])



# %% Cell 5
df["form_of_violence"] = (
    df["form_of_violence"]
      .astype(str)
      .str.replace("\n", " ", regex=False)
)

df["form_of_violence_list"] = df["form_of_violence"].str.split(";")

df_long = df.explode("form_of_violence_list")

df_long["form_of_violence_list"] = (
    df_long["form_of_violence_list"]
      .astype(str)
      .str.strip()
)

df_long = df_long[df_long["form_of_violence_list"] != ""]



# %% Cell 6
panel_total = (
    df.groupby(["country", "start_year"])
      .size()
      .reset_index(name="n_measures_total")
)



# %% Cell 7
panel_by_violence = (
    df_long.groupby(["country", "start_year", "form_of_violence_list"])
           .size()
           .reset_index(name="n_measures")
)

panel_violence_wide = panel_by_violence.pivot_table(
    index=["country", "start_year"],
    columns="form_of_violence_list",
    values="n_measures",
    fill_value=0
).reset_index()



# %% Cell 8
df["measure_family"] = (
    df["type_of_measure"]
      .astype(str)
      .str.split(">")
      .str[0]
      .str.strip()
)

panel_by_family = (
    df.groupby(["country", "start_year", "measure_family"])
      .size()
      .reset_index(name="n_measures")
)

panel_family_wide = panel_by_family.pivot_table(
    index=["country", "start_year"],
    columns="measure_family",
    values="n_measures",
    fill_value=0
).reset_index()



# %% Cell 9
panel = panel_total.merge(
    panel_violence_wide,
    on=["country", "start_year"],
    how="left"
)

panel = panel.merge(
    panel_family_wide,
    on=["country", "start_year"],
    how="left"
)

panel = panel.fillna(0)
panel.head()



# %% Cell 10
panel.columns = (
    panel.columns
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("/", "_")
    .str.replace("-", "_")
)

panel.to_csv("unwomen_measures_country_year_panel_clean.csv", index=False)



# %% Cell 11
import pandas as pd

path = "unwomen_measures_country_year_panel_clean.csv"
panel = pd.read_csv(path)

print("Shape:", panel.shape)
print("\nColumn count:", len(panel.columns))
print("\nDuplicate column names?:", panel.columns.duplicated().any())

print("\nFirst 5 rows:")
display(panel.head())

print("\nLast 5 rows:")
display(panel.tail())



# %% Cell 12
import pandas as pd

panel = pd.read_csv("unwomen_measures_country_year_panel_clean.csv")

# Merge monitoring_and_evaluation
if "monitoring_and_evaluation.1" in panel.columns:
    panel["monitoring_and_evaluation"] = (
        panel["monitoring_and_evaluation"] +
        panel["monitoring_and_evaluation.1"]
    )
    panel = panel.drop(columns=["monitoring_and_evaluation.1"])

# Merge laws
if "laws.1" in panel.columns:
    panel["laws"] = (
        panel["laws"] +
        panel["laws.1"]
    )
    panel = panel.drop(columns=["laws.1"])



# %% Cell 13
if "femicide_feminicide" in panel.columns:
    panel["femicide"] = (
        panel["femicide"] +
        panel["femicide_feminicide"]
    )
    panel = panel.drop(columns=["femicide_feminicide"])



# %% Cell 14
print("New shape:", panel.shape)
print("Duplicate columns?:", panel.columns.duplicated().any())



# %% Cell 15
panel = panel.sort_values(["country","start_year"])

panel["institutional_depth"] = (
    panel.groupby("country")["n_measures_total"]
         .cumsum()
)



# %% Cell 16
print("New shape:", panel.shape)
print("Duplicate columns?:", panel.columns.duplicated().any())
panel.head()


# %% Cell 17
# Export (clean + cumulative)
out_path = "unwomen_measures_country_year_panel_FINAL_with_depth.csv"

panel.to_csv(out_path, index=False, encoding="utf-8")
print("Saved:", out_path)



# %% Cell 18
import pandas as pd

def read_csv_robust(path, sep=","):
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err

# HDR
hdr_path = "external_country_datasets_v2/undp_hdr/HDR_composite_indices_complete_time_series.csv"
hdr = read_csv_robust(hdr_path)
print(hdr.shape)
print(hdr.columns)
print(hdr.head())

# V-Dem / OWID
elec_path = "external_country_datasets_v2/vdem_owid/electoral-democracy-index.csv"
vdem_owid_elec = read_csv_robust(elec_path)
print(vdem_owid_elec.head())

demo_path = "external_country_datasets_v2/vdem_owid/liberal-democracy-index.csv"
vdem_owid_demo = read_csv_robust(demo_path)
print(vdem_owid_demo.head())

wpe_path = "external_country_datasets_v2/vdem_owid/women-political-empowerment-index.csv"
vdem_owid_wpe = read_csv_robust(wpe_path)
print(vdem_owid_wpe.head())



# %% Cell 19
import pandas as pd

# UN Women panel (your cleaned one with institutional_depth)
panel = pd.read_csv("unwomen_measures_country_year_panel_FINAL_with_depth.csv")

# Standardize key column name
panel = panel.rename(columns={"start_year": "year"})

# HDR (already loaded in your notebook as `hdr`)
# hdr has: iso3, country, ...
country_to_iso3 = hdr[["country", "iso3"]].dropna().drop_duplicates()

panel = panel.merge(country_to_iso3, on="country", how="left")

print("Missing ISO3:", panel["iso3"].isna().sum())
print(panel[panel["iso3"].isna()][["country"]].drop_duplicates().head(20))



# %% Cell 20
[c for c in hdr.columns if c.startswith("hdi_")][:5], [c for c in hdr.columns if c.startswith("gdi_")][:5], [c for c in hdr.columns if c.startswith("gii_")][:5]



# %% Cell 21
import re

def hdr_wide_to_long(hdr, prefix):
    cols = ["iso3", "country"] + [c for c in hdr.columns if c.startswith(prefix + "_")]
    tmp = hdr[cols].copy()

    long = tmp.melt(
        id_vars=["iso3", "country"],
        var_name="var",
        value_name=prefix
    )

    long["year"] = long["var"].str.extract(r"_(\d{4})").astype("Int64")
    long = long.drop(columns=["var"])

    return long

hdr_hdi = hdr_wide_to_long(hdr, "hdi")
hdr_gdi = hdr_wide_to_long(hdr, "gdi")
hdr_gii = hdr_wide_to_long(hdr, "gii")

hdr_long = hdr_hdi.merge(hdr_gdi, on=["iso3","country","year"], how="outer")
hdr_long = hdr_long.merge(hdr_gii, on=["iso3","country","year"], how="outer")

print(hdr_long.shape)
print(hdr_long.head())



# %% Cell 22
def owid_clean(df, value_col_name):
    df = df.copy()
    df = df.rename(columns={
        "Entity": "country_owid",
        "Code": "iso3",
        "Year": "year"
    })
    # Find the indicator column (the numeric one)
    indicator_cols = [c for c in df.columns if c not in ["country_owid","iso3","year","World region according to OWID"]]
    assert len(indicator_cols) == 1, indicator_cols
    df = df[["iso3","year", indicator_cols[0]]].rename(columns={indicator_cols[0]: value_col_name})
    return df

elec = owid_clean(vdem_owid_elec, "electoral_democracy")
ldem = owid_clean(vdem_owid_demo, "liberal_democracy")
wpe  = owid_clean(vdem_owid_wpe,  "women_political_empowerment")

vdem_all = elec.merge(ldem, on=["iso3","year"], how="outer")
vdem_all = vdem_all.merge(wpe,  on=["iso3","year"], how="outer")

print(vdem_all.head())



# %% Cell 23
# Check uniqueness in your panel (must be 1 row per iso3-year)
print("panel duplicates iso3-year:", panel.duplicated(["iso3","year"]).sum())

# Check HDR-long uniqueness
print("hdr_long duplicates iso3-year:", hdr_long.duplicated(["iso3","year"]).sum())

# Check V-Dem uniqueness
print("vdem_all duplicates iso3-year:", vdem_all.duplicated(["iso3","year"]).sum())



# %% Cell 24
hdr_long = hdr_long.sort_values(["iso3","year"])
hdr_long = hdr_long.groupby(["iso3","year"], as_index=False).agg({
    "country": "first",
    "hdi": "mean",
    "gdi": "mean",
    "gii": "mean"
})
print("hdr_long now duplicates:", hdr_long.duplicated(["iso3","year"]).sum())



# %% Cell 25
# If your panel still has country column, keep it
panel = panel.copy()

master = panel.merge(
    hdr_long[["iso3","year","hdi","gdi","gii"]],
    on=["iso3","year"],
    how="left"
)

master = master.merge(
    vdem_all[["iso3","year","electoral_democracy","liberal_democracy","women_political_empowerment"]],
    on=["iso3","year"],
    how="left"
)

print("MASTER SHAPE:", master.shape)
print(master[["country","iso3","year","institutional_depth","hdi","gdi","gii",
              "electoral_democracy","liberal_democracy","women_political_empowerment"]].head())



# %% Cell 26
master.to_csv("MASTER_unwomen_HDR_VDEM_panel.csv", index=False, encoding="utf-8")
print("Saved: MASTER_unwomen_HDR_VDEM_panel.csv")



# %% Cell 27
nan_counts = master[[
    "hdi","gdi","gii",
    "electoral_democracy",
    "liberal_democracy",
    "women_political_empowerment"
]].isna().sum()

print(nan_counts)



# %% Cell 28
master.groupby("year")[["hdi","gdi","gii"]].apply(lambda x: x.isna().sum())



# %% Cell 29
(master.shape[0] / panel.shape[0])



# %% Cell 30
print("panel dup iso3-year:", panel.duplicated(["iso3","year"]).sum())
print("hdr_long dup iso3-year:", hdr_long.duplicated(["iso3","year"]).sum())
print("vdem_all dup iso3-year:", vdem_all.duplicated(["iso3","year"]).sum())



# %% Cell 31
# Keep only needed columns first (safety)
vdem_all = vdem_all[["iso3","year","electoral_democracy","liberal_democracy","women_political_empowerment"]].copy()

# Collapse duplicates
vdem_all_u = (
    vdem_all.groupby(["iso3","year"], as_index=False)
            .agg({
                "electoral_democracy":"mean",
                "liberal_democracy":"mean",
                "women_political_empowerment":"mean"
            })
)

print("vdem_all_u rows:", vdem_all_u.shape[0])
print("vdem_all_u dup iso3-year:", vdem_all_u.duplicated(["iso3","year"]).sum())
print(vdem_all_u.head())



# %% Cell 32
dups = panel[panel.duplicated(["iso3","year"], keep=False)].sort_values(["iso3","year"])
print("Duplicated iso3-year rows:", len(dups))
display(dups[["country","iso3","year","n_measures_total","institutional_depth"]].head(30))



# %% Cell 33
# Identify which columns are measure counts (everything numeric except year)
count_cols = [c for c in panel.columns if c not in ["country","iso3","year"]]

panel_u = (
    panel.groupby(["iso3","year"], as_index=False)[count_cols].sum()
)

# Add country name back (first non-null)
country_map = panel[["iso3","country"]].dropna().drop_duplicates("iso3")
panel_u = panel_u.merge(country_map, on="iso3", how="left")

# Reorder columns nicely
panel_u = panel_u[["country","iso3","year"] + count_cols]

# Recompute institutional_depth from n_measures_total (after aggregation)
panel_u = panel_u.sort_values(["iso3","year"])
panel_u["institutional_depth"] = panel_u.groupby("iso3")["n_measures_total"].cumsum()

print("panel_u rows:", panel_u.shape[0])
print("panel_u dup iso3-year:", panel_u.duplicated(["iso3","year"]).sum())
panel_u.head()



# %% Cell 34
master_u = (
    panel_u.merge(hdr_long, on=["iso3","year"], how="left")
           .merge(vdem_all_u, on=["iso3","year"], how="left")
)

print("MASTER_U SHAPE:", master_u.shape)
print("MASTER_U dup iso3-year:", master_u.duplicated(["iso3","year"]).sum())

master_u[["country","iso3","year","institutional_depth","hdi","gdi","gii",
          "electoral_democracy","liberal_democracy","women_political_empowerment"]].head()



# %% Cell 35
 # If country columns exist under different names, pick one
country_cols = [c for c in master_u.columns if "country" in c.lower()]

print("Country-like columns:", country_cols)

if "country" not in master_u.columns:
    if "country_x" in master_u.columns:
        master_u = master_u.rename(columns={"country_x": "country"})
    elif "country_y" in master_u.columns:
        master_u = master_u.rename(columns={"country_y": "country"})
    elif len(country_cols) > 0:
        master_u = master_u.rename(columns={country_cols[0]: "country"})
    else:
        # fallback: rebuild country from panel_u mapping
        country_map = panel_u[["iso3","country"]].drop_duplicates("iso3")
        master_u = master_u.merge(country_map, on="iso3", how="left")

# If there are still leftover country_* columns, drop them
drop_cols = [c for c in master_u.columns if c.lower().startswith("country_")]
master_u = master_u.drop(columns=drop_cols, errors="ignore")

print("Now has country?:", "country" in master_u.columns)
print(master_u[["country","iso3","year"]].head())



# %% Cell 36
master_u[["country","iso3","year","institutional_depth","hdi","gdi","gii",
          "electoral_democracy","liberal_democracy","women_political_empowerment"]].head()



# %% Cell 37
master_u.to_csv("MASTER_unwomen_HDR_VDEM_panel_CLEAN.csv", index=False, encoding="utf-8")
print("Saved: MASTER_unwomen_HDR_VDEM_panel_CLEAN.csv")



# %% Cell 38
master_u


# %% Cell 39

