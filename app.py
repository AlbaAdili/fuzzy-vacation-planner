import streamlit as st
import pandas as pd

from fuzzy_travel_system import recommend_countries

st.set_page_config(
    page_title="Fuzzy Travel Recommender",
    page_icon="✈️",
    layout="centered",
)

st.title("🌍 Fuzzy-Based Travel Recommendation System")

st.markdown(
    "This app uses **fuzzy logic** (weather, budget, crowds, distance, safety/HDI) "
    "to rank destinations based on your preferences."
)

st.divider()

# =============== USER INPUTS (UI) =================

st.subheader("Your travel preferences")

col1, col2 = st.columns(2)

with col1:
    preferred_weather = st.selectbox(
        "Preferred weather",
        ["snow", "cold", "chilly", "warm", "hot", "sweltering"],
        index=3,  # default: warm
    )

    preferred_crowd = st.selectbox(
        "Crowd level",
        ["quiet", "normal", "busy"],
        index=1,
    )

with col2:
    preferred_budget = st.selectbox(
        "Budget level",
        ["cheap", "medium", "expensive"],
        index=1,
        help="This controls how strict the system is about destination cost.",
    )

    preferred_distance = st.selectbox(
        "Distance from home (Zurich)",
        ["near", "moderate", "far"],
        index=0,
    )

eu_only = st.checkbox(
    "Show only European/Schengen destinations",
    value=True,
)

top_k = st.slider(
    "How many destinations to show?",
    min_value=3,
    max_value=10,
    value=5,
)

st.divider()

# =============== RUN RECOMMENDATION =================

if st.button("🔎 Find my ideal destinations"):
    try:
        results = recommend_countries(
            csv_path="data/countries_data.csv",
            preferred_weather=preferred_weather,
            preferred_budget=preferred_budget,
            preferred_crowd=preferred_crowd,
            preferred_distance=preferred_distance,
            eu_only=eu_only,
            top_k=top_k,
        )
        st.write("Debug – result columns:", list(results.columns))


        st.subheader("Top recommendations")

        display_cols = [
            "country",
            "suitability",
            "hdi",
            "avg_temp_c",
            "cost_index",
            "crowd_index",
            "distance_km",
        ]
        df_display = results[display_cols].copy()
        df_display["suitability"] = df_display["suitability"].round(1)

        st.dataframe(
            df_display.rename(
                columns={
                    "country": "Country",
                    "suitability": "Suitability (0-100)",
                    "hdi": "HDI",
                    "avg_temp_c": "Avg Temp (°C)",
                    "cost_index": "Cost Index",
                    "crowd_index": "Crowd Index (0-10)",
                    "distance_km": "Distance (km)",
                }
            ),
            use_container_width=True,
        )

        # Small explanation for the top choice
        best = results.iloc[0]
        st.markdown(
            f"**Best match: {best['country']}**  \n"
            f"- Suitability: **{best['suitability']:.1f}/100**  \n"
            f"- HDI: **{best['hdi']:.3f}**  \n"
            f"- Avg temperature: **{best['avg_temp_c']}°C**  \n"
            f"- Cost index: **{best['cost_index']}**  \n"
            f"- Distance: **{int(best['distance_km'])} km**"
        )

        # Optional: quick bar chart of suitability
        st.bar_chart(
            data=results.set_index("country")["suitability"],
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Check that `data/countries_data.csv` exists and has the correct columns.")
else:
    st.info("Set your preferences above and click **'Find my ideal destinations'**.")
