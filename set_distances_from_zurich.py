import pandas as pd

# Load your combined dataset
df = pd.read_csv("data/countries_data.csv")

# Approximate great-circle-ish distances (km) from Zürich
distance_from_zurich = {
    "Albania": 1150,
    "Algeria": 1350,
    "Argentina": 11200,
    "Armenia": 2800,
    "Australia": 16500,
    "Austria": 600,
    "Azerbaijan": 3000,
    "Bahrain": 4300,
    "Bangladesh": 7300,
    "Barbados": 7800,
    "Belarus": 1500,
    "Belgium": 500,
    "Bolivia": 10700,
    "Botswana": 8300,
    "Brazil": 9500,
    "Bulgaria": 1450,
    "Cambodia": 9300,
    "Cameroon": 4500,
    "Canada": 6500,
    "Chile": 11800,
    "China": 7800,
    "Colombia": 8900,
    "Costa Rica": 9300,
    "Croatia": 800,
    "Cuba": 8200,
    "Cyprus": 2400,
    "Denmark": 950,
    "Dominican Republic": 7800,
    "Ecuador": 9800,
    "Egypt": 2600,
    "El Salvador": 9700,
    "Estonia": 1800,
    "Fiji": 16600,
    "Finland": 1800,
    "France": 500,
    "Georgia": 2700,
    "Germany": 500,
    "Ghana": 4800,
    "Greece": 1600,
    "Hungary": 850,
    "Iceland": 2600,
    "India": 6100,
    "Indonesia": 11200,
    "Iran": 3500,
    "Iraq": 3200,
    "Ireland": 1250,
    "Israel": 2800,
    "Italy": 700,
    "Jamaica": 8400,
    "Japan": 9700,
    "Kazakhstan": 4300,
    "Kenya": 6100,
    "Kuwait": 3900,
    "Kyrgyzstan": 5100,
    "Latvia": 1600,
    "Lebanon": 2700,
    "Libya": 1900,
    "Lithuania": 1500,
    "Luxembourg": 400,
    "Madagascar": 8400,
    "Malaysia": 9800,
    "Malta": 1350,
    "Mexico": 9700,
    "Moldova": 1600,
    "Montenegro": 1100,
    "Morocco": 1900,
    "Nepal": 6500,
    "Netherlands": 615,
    "New Zealand": 18000,
    "Nigeria": 4600,
    "Norway": 1650,
    "Oman": 4900,
    "Pakistan": 5700,
    "Panama": 9400,
    "Paraguay": 10300,
    "Peru": 10400,
    "Philippines": 10500,
    "Poland": 1100,
    "Portugal": 1700,
    "Romania": 1400,
    "Russia": 2200,
    "Saudi Arabia": 4000,
    "Serbia": 1050,
    "Singapore": 10300,
    "Slovakia": 800,
    "Slovenia": 480,
    "South Africa": 8500,
    "South Korea": 8600,
    "Spain": 1250,
    "Sri Lanka": 7800,
    "Sweden": 1500,
    "Switzerland": 0,
    "Syria": 2600,
    "Tanzania": 7000,
    "Thailand": 9200,
    "Tunisia": 1100,
    "Turkey": 1650,
    "Uganda": 5900,
    "Ukraine": 1700,
    "United Kingdom": 780,
    "United States": 7000,
    "Uruguay": 11400,
    "Uzbekistan": 4700,
    "Venezuela": 8400,
    "Vietnam": 8900,
    "Zimbabwe": 7700,
}

# Create an override column
df["distance_override"] = df["country"].map(distance_from_zurich)

# Show any countries that don't have an explicit mapping
missing = df[df["distance_override"].isna()]["country"].unique()
if len(missing) > 0:
    print("Countries WITHOUT explicit distance mapping (using old value 2000):")
    print(missing)
else:
    print("All countries in the CSV have an explicit distance mapping.")

# If mapping exists, use it; otherwise keep existing distance_km (2000)
df["distance_km"] = df["distance_override"].fillna(df["distance_km"])

# Drop helper column
df = df.drop(columns=["distance_override"])

# Save back to CSV
df.to_csv("data/countries_data.csv", index=False)
print("Updated data/countries_data.csv with distance_km values.")
