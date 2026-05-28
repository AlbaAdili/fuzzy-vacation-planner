import pandas as pd
import numpy as np
import os

# ============================================================
# 1. HDI  (human-development-index.csv from Our World In Data)
# ============================================================

print("=== HDI COLUMNS ===")
hdi_raw = pd.read_csv("human-development-index.csv")
print(hdi_raw.columns)

# Keep latest year (2023), rename to 'country', 'hdi'
hdi = (
    hdi_raw[hdi_raw["Year"] == 2023]
    .loc[:, ["Entity", "Human Development Index"]]
    .rename(columns={
        "Entity": "country",
        "Human Development Index": "hdi",
    })
)

print("\nHDI sample:")
print(hdi.head())

# ============================================================
# 2. COST OF LIVING (Cost_of_Living_Index_by_Country_2024.csv)
# ============================================================

print("\n=== COST OF LIVING COLUMNS ===")
cost_raw = pd.read_csv("Cost_of_Living_Index_by_Country_2024.csv")
print(cost_raw.columns)

cost = (
    cost_raw.loc[:, ["Country", "Cost of Living Index"]]
    .rename(columns={
        "Country": "country",
        "Cost of Living Index": "cost_index",
    })
)

print("\nCost sample:")
print(cost.head())

# ============================================================
# 3. TEMPERATURE
#    (worlds all cities with their avg temp - Sheet1.csv)
# ============================================================

print("\n=== TEMPERATURE COLUMNS ===")
temp_raw = pd.read_csv("worlds all cities with their avg temp - Sheet1.csv")
print(temp_raw.columns)

# 'Year' column looks like "12.1\n(53.8)" -> we want the first number (12.1)
temp_raw["Year_clean"] = (
    temp_raw["Year"]
    .astype(str)
    .str.extract(r"([-+]?\d*\.?\d+)")[0]   # first numeric group
)

temp_raw["Year_clean"] = pd.to_numeric(temp_raw["Year_clean"], errors="coerce")

# Group by country and take mean yearly temperature in °C
temp = (
    temp_raw
    .groupby("Country")["Year_clean"]
    .mean()
    .reset_index()
    .rename(columns={
        "Country": "country",
        "Year_clean": "avg_temp_c",
    })
)

print("\nTemperature sample:")
print(temp.head())

# ============================================================
# 4. TOURISM ARRIVALS -> CROWD INDEX
#    (API_ST.INT.ARVL_DS2_en_csv_v2_2695.csv from World Bank)
# ============================================================

print("\n=== TOURISM COLUMNS ===")
try:
    tourism_raw = pd.read_csv("API_ST.INT.ARVL_DS2_en_csv_v2_2695.csv", skiprows=4)
    print(tourism_raw.columns)

    # Year columns like '1960','1961',...,'2019'
    year_cols = [c for c in tourism_raw.columns if c.isdigit()]

    # latest non-NaN arrivals per country
    def latest_non_nan(row):
        vals = row[year_cols].dropna()
        if len(vals) == 0:
            return np.nan
        return vals.iloc[-1]

    tourism_raw["latest_arrivals"] = tourism_raw.apply(latest_non_nan, axis=1)

    tourism = (
        tourism_raw.loc[:, ["Country Name", "latest_arrivals"]]
        .rename(columns={
            "Country Name": "country",
            "latest_arrivals": "tourist_arrivals",
        })
        .dropna(subset=["tourist_arrivals"])
    )

    # Scale tourist_arrivals to [0,10]
    min_arr = tourism["tourist_arrivals"].min()
    max_arr = tourism["tourist_arrivals"].max()
    tourism["crowd_index"] = 10 * (
        (tourism["tourist_arrivals"] - min_arr) / (max_arr - min_arr)
    )

    tourism = tourism[["country", "crowd_index"]]

    print("\nTourism sample:")
    print(tourism.head())

except FileNotFoundError:
    print("!! Tourism data file API_ST.INT.ARVL_DS2_en_csv_v2_2695.csv not found.")
    print("   Download it from the World Bank (indicator ST.INT.ARVL) and place it in this folder.")
    tourism = pd.DataFrame(columns=["country", "crowd_index"])

# ============================================================
# 5. MERGE ALL DATASETS
# ============================================================

df = hdi.merge(cost, on="country", how="inner")
df = df.merge(temp, on="country", how="inner")
df = df.merge(tourism, on="country", how="left")  # crowd_index may be NaN

# Neutral crowd index for missing tourism data
df["crowd_index"] = df["crowd_index"].fillna(5.0)

# ============================================================
# 6. ADD is_eu AND distance_km
# ============================================================

eu_countries = {
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia",
    "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Germany",
    "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania",
    "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania",
    "Slovakia", "Slovenia", "Spain", "Sweden",
    "Switzerland", "Norway", "Iceland", "Liechtenstein",
}

df["is_eu"] = df["country"].apply(lambda c: 1 if c in eu_countries else 0)

# Placeholder distance (km) – can manually adjust later
df["distance_km"] = 2000.0

# Reorder columns as expected by fuzzy_travel_system.py
df_final = df[[
    "country",
    "is_eu",
    "hdi",
    "avg_temp_c",
    "cost_index",
    "crowd_index",
    "distance_km",
]]

print("\nFinal merged sample:")
print(df_final.head())

# ============================================================
# 7. SAVE TO countries_data.csv
# ============================================================

out_path = os.path.join(os.path.dirname(__file__), "countries_data.csv")
df_final.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
