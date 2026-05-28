import numpy as np
import pandas as pd


# =========================
# 1. DATA LOADING
# =========================

def load_country_data(csv_path: str = "data/countries_data.csv") -> pd.DataFrame:
    """
    Load country indicators from a CSV file.
    Required columns:
      country,is_eu,hdi,avg_temp_c,cost_index,crowd_index,distance_km
    """
    df = pd.read_csv(csv_path)
    required_cols = [
        "country", "is_eu", "hdi",
        "avg_temp_c", "cost_index", "crowd_index", "distance_km",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df


# =========================
# 2. USER PREFERENCE MAPPINGS
# =========================

WEATHER_PREFERRED_TEMP = {
    "snow": 0.0,
    "cold": 5.0,
    "chilly": 12.0,
    "warm": 22.0,
    "hot": 28.0,
    "sweltering": 34.0,
}

# These are *ideal* distances for the fuzzy match function.
DISTANCE_PREFERRED_KM = {
    "near": 500.0,
    "moderate": 3000.0,
    "far": 8000.0,
}

CROWD_PREFERRED_IDX = {
    "quiet": 2.0,
    "normal": 5.0,
    "busy": 8.0,
}

BUDGET_MAX_COST = {
    "cheap": 80.0,
    "medium": 140.0,
    "expensive": 220.0,
}


def compatibility_score(actual: float, ideal: float, max_diff: float) -> float:
    """
    Turn difference between actual and ideal into a [0, 10] compatibility score.
    10 = perfect match, 0 = very bad.
    """
    diff = abs(actual - ideal)
    return max(0.0, 10.0 * (1.0 - diff / max_diff))


# =========================
# 3. FUZZY MEMBERSHIP FUNCTIONS (triangular, classical)
# =========================

def tri(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function μ(x; a,b,c)."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def membership_scores(score: float) -> dict:
    """
    Fuzzify a score in [0,10] into {low, medium, high}.
    """
    return {
        "low": tri(score, 0.0, 0.0, 4.0),
        "medium": tri(score, 2.0, 5.0, 8.0),
        "high": tri(score, 6.0, 10.0, 10.0),
    }


# Output universe and membership functions for Mamdani (0–100)
SU_UNIVERSE = np.linspace(0.0, 100.0, 1001)  # 0, 0.1, ..., 100


def mu_low_suitability(x: np.ndarray) -> np.ndarray:
    # low: 0–0–40
    return np.vectorize(tri)(x, 0.0, 0.0, 40.0)


def mu_medium_suitability(x: np.ndarray) -> np.ndarray:
    # medium: 25–50–75
    return np.vectorize(tri)(x, 25.0, 50.0, 75.0)


def mu_high_suitability(x: np.ndarray) -> np.ndarray:
    # high: 60–100–100
    return np.vectorize(tri)(x, 60.0, 100.0, 100.0)


def suitability_membership(label: str, x: np.ndarray) -> np.ndarray:
    if label == "low":
        return mu_low_suitability(x)
    if label == "medium":
        return mu_medium_suitability(x)
    if label == "high":
        return mu_high_suitability(x)
    raise ValueError(f"Unknown suitability label: {label}")


# =========================
# 4. MAMDANI DEFUZZIFICATION (min-max, centroid)
# =========================

def mamdani_defuzzify(rule_activations: list[tuple[float, str]]) -> float:
    """
    Classical Mamdani:
      - each rule i has firing strength w_i and output label L_i in {low,medium,high}
      - for each rule: clip μ_L_i(x) at height w_i (min operator)
      - aggregate by max across rules
      - defuzzify aggregated μ(x) by centroid
    """
    aggregated = np.zeros_like(SU_UNIVERSE)

    for w, label in rule_activations:
        if w <= 0.0:
            continue
        mu_label = suitability_membership(label, SU_UNIVERSE)
        clipped = np.minimum(w, mu_label)  # min(w, μ_label(x))
        aggregated = np.maximum(aggregated, clipped)  # max across rules

    denom = aggregated.sum()
    if denom == 0.0:
        # no rule fired → neutral suitability
        return 50.0

    num = (aggregated * SU_UNIVERSE).sum()
    return float(num / denom)


# =========================
# 5. EVALUATE ONE COUNTRY (full Mamdani pipeline)
# =========================

def evaluate_country(row: pd.Series, user_prefs: dict) -> float:
    """
    Evaluate a single country row and return suitability in [0, 100]
    using classical Mamdani fuzzy inference:
      - fuzzification of four inputs (weather, budget, crowd, distance)
      - rule base with min (AND) and max (aggregation)
      - centroid defuzzification
    """

    # ---------- CRISP MATCH SCORES [0,10] ----------

    # Weather
    ideal_temp = WEATHER_PREFERRED_TEMP[user_prefs["preferred_weather"]]
    weather_score = compatibility_score(row["avg_temp_c"], ideal_temp, max_diff=20.0)

    # Budget
    max_cost = BUDGET_MAX_COST[user_prefs["preferred_budget"]]
    if row["cost_index"] <= max_cost:
        budget_score = 10.0
    else:
        budget_score = compatibility_score(row["cost_index"], max_cost, max_diff=120.0)

    # Crowd
    ideal_crowd = CROWD_PREFERRED_IDX[user_prefs["preferred_crowd"]]
    crowd_score = compatibility_score(row["crowd_index"], ideal_crowd, max_diff=8.0)

    # Distance (match to ideal)
    ideal_dist = DISTANCE_PREFERRED_KM[user_prefs["preferred_distance"]]
    distance_score = compatibility_score(row["distance_km"], ideal_dist, max_diff=8000.0)

    # ---------- FUZZIFICATION ----------
    W = membership_scores(weather_score)
    B = membership_scores(budget_score)
    C = membership_scores(crowd_score)
    D = membership_scores(distance_score)

    # ---------- RULE BASE ----------
    rules: list[tuple[float, str]] = []

    # 1) Everything matches very well -> high
    w1 = min(W["high"], B["high"], C["high"], D["high"])
    rules.append((w1, "high"))

    # 2) Good weather + good budget + good distance -> high
    w2 = min(W["high"], B["high"], D["high"])
    rules.append((w2, "high"))

    # 3) Good weather + medium budget + medium distance -> medium
    w3 = min(W["high"], B["medium"], D["medium"])
    rules.append((w3, "medium"))

    # 4) All medium -> medium
    w4 = min(W["medium"], B["medium"], C["medium"], D["medium"])
    rules.append((w4, "medium"))

    # 5) Very bad budget (too expensive) -> low
    w5 = B["low"]
    rules.append((w5, "low"))

    # 6) Far (bad distance) AND busy crowds -> low
    w6 = min(D["low"], C["low"])
    rules.append((w6, "low"))

    # 7) Bad weather AND bad budget -> low
    w7 = min(W["low"], B["low"])
    rules.append((w7, "low"))

    # 8) Cheap + quiet + okay weather -> medium
    w8 = min(B["high"], C["high"], W["medium"])
    rules.append((w8, "medium"))

    # 9) Very good weather + medium budget + quiet + near/moderate distance -> high
    w9 = min(W["high"], B["medium"], C["high"], max(D["high"], D["medium"]))
    rules.append((w9, "high"))

    # 10) Medium everything but very near distance -> medium
    w10 = min(W["medium"], B["medium"], C["medium"], D["high"])
    rules.append((w10, "medium"))

    # 11) Good budget + near distance even if crowd medium/high -> high
    w11 = min(B["high"], D["high"], max(C["medium"], C["high"]))
    rules.append((w11, "high"))

    # 12) Good weather but far and busy and only medium budget -> low
    w12 = min(W["high"], D["low"], C["low"], B["medium"])
    rules.append((w12, "low"))

     # 13) Medium weather + very cheap + medium distance + quiet -> high
    w13 = min(W["medium"], B["high"], D["medium"], C["high"])
    rules.append((w13, "high"))

    # 14) Very bad weather and far away -> low
    w14 = min(W["low"], D["low"])
    rules.append((w14, "low"))

    # 15) Cheap + medium crowds + medium/near distance -> high
    w15 = min(B["high"], C["medium"], max(D["high"], D["medium"]))
    rules.append((w15, "high"))

    # 16) Medium budget + busy crowds + far -> low
    w16 = min(B["medium"], C["low"], D["low"])
    rules.append((w16, "low"))

    # ---------- AGGREGATION + DEFUZZIFICATION ----------
    suitability = mamdani_defuzzify(rules)
    return suitability


# =========================
# 6. CRISP DISTANCE FILTER
# =========================

def filter_by_distance_preference(
    df: pd.DataFrame,
    preferred_distance: str,
    eu_only: bool,
) -> pd.DataFrame:
    """
    Apply a crisp filter on distance_km so that the
    'near / moderate / far' preference makes sense.

    - If eu_only is True: interpret distances *within Europe*
    - If eu_only is False: interpret distances globally.
    """

    if eu_only:
        # relative distances INSIDE Europe / Schengen
        if preferred_distance == "near":
            # neighbours / very close
            return df[df["distance_km"] <= 800]

        if preferred_distance == "moderate":
            # mid-range inside Europe
            return df[(df["distance_km"] > 800) & (df["distance_km"] <= 1800)]

        if preferred_distance == "far":
            # farthest EU destinations ( Greece, Portugal, etc.)
            return df[df["distance_km"] > 1800]

        return df  # fallback if weird value

    # -------- global mode (non-EU or world) --------
    if preferred_distance == "near":
        # Europe-ish / short-haul
        return df[df["distance_km"] <= 2500]

    if preferred_distance == "moderate":
        # medium-haul
        return df[(df["distance_km"] > 1500) & (df["distance_km"] <= 6000)]

    if preferred_distance == "far":
        # long-haul / intercontinental
        return df[df["distance_km"] >= 5000]

    # fallback: no filter
    return df


# =========================
# 7. MAIN PIPELINE (used by Streamlit)
# =========================

def recommend_countries(
    csv_path: str = "data/countries_data.csv",
    preferred_weather: str = "warm",
    preferred_budget: str = "medium",
    preferred_crowd: str = "normal",
    preferred_distance: str = "near",
    eu_only: bool = False,
    top_k: int = 5,
    min_hdi: float = 0.70,
) -> pd.DataFrame:
    """
    Main function used by your Streamlit UI.

    Steps:
      1. Load real-country indicators from CSV.
      2. Filter by HDI (safety) and EU-only toggle.
      3. Apply crisp distance preference filter (near/moderate/far).
      4. Run Mamdani fuzzy inference for each remaining country.
      5. Return top_k with 'suitability' column.
    """
    df = load_country_data(csv_path)

    # Safety filter by HDI
    df = df[df["hdi"] >= min_hdi].copy()

    # Region filter
    if eu_only:
        df = df[df["is_eu"] == 1].copy()

    # Distance preference filter (aware of eu_only)
    df = filter_by_distance_preference(df, preferred_distance, eu_only)

    if df.empty:
        raise ValueError(
            "No countries left after applying HDI/EU and distance preference filters.\n"
            "Try relaxing HDI, turning off 'EU only', or choosing another distance preference."
        )

    user_prefs = {
        "preferred_weather": preferred_weather.lower(),
        "preferred_budget": preferred_budget.lower(),
        "preferred_crowd": preferred_crowd.lower(),
        "preferred_distance": preferred_distance.lower(),
    }

    scores = []
    for _, row in df.iterrows():
        s = evaluate_country(row, user_prefs)
        scores.append(s)

    df = df.copy()
    df["suitability"] = scores

    df_sorted = df.sort_values("suitability", ascending=False)
    return df_sorted.head(top_k)


# =========================
# 8. CLI TEST
# =========================

if __name__ == "__main__":
    res = recommend_countries(
        csv_path="data/countries_data.csv",
        preferred_weather="snow",
        preferred_budget="medium",
        preferred_crowd="normal",
        preferred_distance="far",
        eu_only=True,
        top_k=5,
    )
    print(res.head())
    print("Columns:", list(res.columns))
