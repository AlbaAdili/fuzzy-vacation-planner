import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fuzzy_travel_system import (
    load_country_data,
    WEATHER_PREFERRED_TEMP,
    DISTANCE_PREFERRED_KM,
    CROWD_PREFERRED_IDX,
    BUDGET_MAX_COST,
    compatibility_score,
    tri,
    membership_scores,
    SU_UNIVERSE,
    suitability_membership,
)

# =========================
# 1. Mamdani helpers
# =========================

def mamdani_aggregate(rule_activations: list[tuple[float, str]]) -> np.ndarray:
    """
    Build aggregated membership μ_agg(x) over the suitability universe,
    using min for implication and max for aggregation.
    """
    aggregated = np.zeros_like(SU_UNIVERSE)

    for w, label in rule_activations:
        if w <= 0.0:
            continue
        mu_label = suitability_membership(label, SU_UNIVERSE)
        clipped = np.minimum(w, mu_label)
        aggregated = np.maximum(aggregated, clipped)

    return aggregated


def mamdani_centroid(aggregated: np.ndarray) -> float:
    """
    Centroid defuzzification on an aggregated membership function.
    """
    denom = aggregated.sum()
    if denom == 0.0:
        return 50.0
    num = (aggregated * SU_UNIVERSE).sum()
    return float(num / denom)


# =========================
# 2. Plots: membership functions
# =========================

def plot_weather_membership():
    """
    Plot membership functions for weather_match (low, medium, high)
    and save as PNG.
    """
    x = np.linspace(0, 10, 101)
    low_vals = [membership_scores(s)["low"] for s in x]
    med_vals = [membership_scores(s)["medium"] for s in x]
    high_vals = [membership_scores(s)["high"] for s in x]

    plt.figure()
    plt.plot(x, low_vals, label="low")
    plt.plot(x, med_vals, label="medium")
    plt.plot(x, high_vals, label="high")
    plt.xlabel("weather_match score (0–10)")
    plt.ylabel("membership degree")
    plt.title("Weather match fuzzy sets (low, medium, high)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig_weather_membership.png", dpi=300)
    plt.close()


def plot_suitability_membership():
    """
    Plot membership functions for suitability (low, medium, high)
    and save as PNG.
    """
    x = SU_UNIVERSE
    low_vals = suitability_membership("low", x)
    med_vals = suitability_membership("medium", x)
    high_vals = suitability_membership("high", x)

    plt.figure()
    plt.plot(x, low_vals, label="low")
    plt.plot(x, med_vals, label="medium")
    plt.plot(x, high_vals, label="high")
    plt.xlabel("suitability (0–100)")
    plt.ylabel("membership degree")
    plt.title("Suitability fuzzy sets (low, medium, high)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig_suitability_membership.png", dpi=300)
    plt.close()


# =========================
# 3. Walkthrough example
# =========================

def walkthrough_example(
    country_name: str = "Italy",
    preferred_weather: str = "warm",
    preferred_budget: str = "medium",
    preferred_crowd: str = "normal",
    preferred_distance: str = "near",
):
    """
    Run one country through the fuzzy pipeline and:
      - print crisp scores
      - print fuzzified degrees
      - print rule activations
      - save aggregated μ(suitability) plot with centroid
    """

    df = load_country_data("data/countries_data.csv")
    row = df[df["country"] == country_name].iloc[0]

    # ----- 1) Crisp match scores -----
    ideal_temp = WEATHER_PREFERRED_TEMP[preferred_weather]
    weather_score = compatibility_score(row["avg_temp_c"], ideal_temp, max_diff=20.0)

    max_cost = BUDGET_MAX_COST[preferred_budget]
    if row["cost_index"] <= max_cost:
        budget_score = 10.0
    else:
        budget_score = compatibility_score(row["cost_index"], max_cost, max_diff=120.0)

    ideal_crowd = CROWD_PREFERRED_IDX[preferred_crowd]
    crowd_score = compatibility_score(row["crowd_index"], ideal_crowd, max_diff=8.0)

    ideal_dist = DISTANCE_PREFERRED_KM[preferred_distance]
    distance_score = compatibility_score(row["distance_km"], ideal_dist, max_diff=8000.0)

    print("=== CRISP SCORES for", country_name, "===")
    print(f"  weather_score  = {weather_score:.2f}")
    print(f"  budget_score   = {budget_score:.2f}")
    print(f"  crowd_score    = {crowd_score:.2f}")
    print(f"  distance_score = {distance_score:.2f}")
    print()

    # ----- 2) Fuzzification -----
    W = membership_scores(weather_score)
    B = membership_scores(budget_score)
    C = membership_scores(crowd_score)
    D = membership_scores(distance_score)

    print("=== FUZZIFIED INPUTS (membership degrees) ===")
    print("Weather match :", W)
    print("Budget match  :", B)
    print("Crowd match   :", C)
    print("Distance match:", D)
    print()

    # ----- 3) Rule base (same as in fuzzy_travel_system.evaluate_country) -----
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

    print("=== RULE ACTIVATIONS (w_i, output label) ===")
    for i, (w, label) in enumerate(rules, start=1):
        print(f"  R{i:02d}: w = {w:.3f}, output = {label}")
    print()

    # ----- 4) Aggregation + centroid -----
    aggregated = mamdani_aggregate(rules)
    suitability = mamdani_centroid(aggregated)

    print(f"=== FINAL SUITABILITY for {country_name}: {suitability:.2f} / 100 ===")

    # Plot aggregated μ(suitability) and centroid, save PNG
    plt.figure()
    plt.plot(SU_UNIVERSE, aggregated, label="aggregated μ(suitability)")
    plt.axvline(suitability, linestyle="--", label=f"centroid = {suitability:.1f}")
    plt.xlabel("suitability")
    plt.ylabel("membership degree")
    plt.title(f"Aggregated output and centroid for {country_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"fig_aggregated_{country_name}.png", dpi=300)
    plt.close()


# =========================
# 4. Run everything
# =========================

if __name__ == "__main__":
    # 1) Membership function plots
    plot_weather_membership()
    plot_suitability_membership()

    # 2) One full fuzzy walkthrough
    walkthrough_example(
        country_name="Italy",
        preferred_weather="warm",
        preferred_budget="medium",
        preferred_crowd="normal",
        preferred_distance="near",
    )
