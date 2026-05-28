# Planning a Vacation
# Fuzzy Travel Recommendation System

This project is a **Mamdani-style fuzzy inference system** that recommends travel destinations based on:

- **Weather preference** (snow → sweltering)  
- **Budget level** (cheap / medium / expensive)  
- **Crowd level** (quiet / normal / busy)  
- **Distance from home (Zurich)** (near / moderate / far)  
- **Safety / development** via **HDI** (Human Development Index)

The fuzzy system combines these factors and outputs a **suitability score in [0, 100]** for each country, then shows the top-ranked destinations in a **Streamlit web app**.

---
[Survey.pdf](https://github.com/user-attachments/files/24151852/Survey.pdf)

## Project structure

```text
fuzzy-travel-system/
├── app.py                      # Streamlit UI
├── fuzzy_travel_system.py      # Core Mamdani fuzzy logic + recommender
├── fuzzy_explainer.py          # Plots + step-by-step fuzzy pipeline example
├── data/
│   ├── build_countries_data.py # Script to build countries_data.csv
│   ├── human-development-index.csv
│   ├── Cost_of_Living_Index_by_Country_2024.csv
│   ├── worlds all cities with their avg temp - Sheet1.csv
│   ├── API_ST.INT.ARVL_....csv # (tourist arrivals / crowd proxy)
│   └── countries_data.csv      # Final merged dataset used by the app
└── README.md
```

## Requirements

- Python **3.10+**

Recommended packages:

```bash
pip install streamlit numpy pandas matplotlib
```
## Running the Streamlit web app

From the project root (fuzzy-travel-system/):
```bash
streamlit run app.py
```



