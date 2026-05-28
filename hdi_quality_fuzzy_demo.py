import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reuse the same tri() from your main file if you want
def tri(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)

# --- universes ---
HDI_U = np.linspace(0, 1, 501)
LE_U = np.linspace(50, 85, 351)         # life expectancy
GNI_U = np.linspace(0, 80000, 401)      # GNI per capita (USD)
Q_U = np.linspace(0, 100, 501)          # quality score


def mu_hdi_low(x):    return np.vectorize(tri)(x, 0.0, 0.3, 0.6)
def mu_hdi_med(x):    return np.vectorize(tri)(x, 0.4, 0.6, 0.8)
def mu_hdi_high(x):   return np.vectorize(tri)(x, 0.7, 0.9, 1.0)

def mu_le_short(x):   return np.vectorize(tri)(x, 50, 55, 65)
def mu_le_normal(x):  return np.vectorize(tri)(x, 60, 70, 80)
def mu_le_long(x):    return np.vectorize(tri)(x, 75, 82, 85)

def mu_gni_low(x):    return np.vectorize(tri)(x, 0, 10000, 25000)
def mu_gni_mid(x):    return np.vectorize(tri)(x, 15000, 30000, 45000)
def mu_gni_high(x):   return np.vectorize(tri)(x, 35000, 60000, 80000)

def mu_q_poor(x):     return np.vectorize(tri)(x, 0, 0, 40)
def mu_q_avg(x):      return np.vectorize(tri)(x, 25, 50, 75)
def mu_q_excellent(x):return np.vectorize(tri)(x, 60, 100, 100)


def eval_country(hdi, le, gni):
    # fuzzify inputs
    H = {
        "low": tri(hdi, 0.0, 0.3, 0.6),
        "med": tri(hdi, 0.4, 0.6, 0.8),
        "high":tri(hdi, 0.7, 0.9, 1.0),
    }
    L = {
        "short": tri(le, 50, 55, 65),
        "normal":tri(le, 60, 70, 80),
        "long":  tri(le, 75, 82, 85),
    }
    G = {
        "low":  tri(gni, 0, 10000, 25000),
        "mid":  tri(gni, 15000, 30000, 45000),
        "high": tri(gni, 35000, 60000, 80000),
    }

    rules = []

    # R1: if HDI high AND LE long AND GNI high -> quality excellent
    w1 = min(H["high"], L["long"], G["high"])
    rules.append((w1, "excellent"))

    # R2: if HDI medium AND LE normal AND GNI mid -> quality average
    w2 = min(H["med"], L["normal"], G["mid"])
    rules.append((w2, "average"))

    # R3: if HDI low OR LE short OR GNI low -> quality poor
    w3 = max(H["low"], L["short"], G["low"])
    rules.append((w3, "poor"))

    # R4: if HDI high AND LE normal AND GNI mid -> quality excellent
    w4 = min(H["high"], L["normal"], G["mid"])
    rules.append((w4, "excellent"))

    # aggregate Mamdani-style
    agg = np.zeros_like(Q_U)
    for w, label in rules:
        if w <= 0:
            continue
        if label == "poor":
            mu = mu_q_poor(Q_U)
        elif label == "average":
            mu = mu_q_avg(Q_U)
        else:
            mu = mu_q_excellent(Q_U)

        agg = np.maximum(agg, np.minimum(w, mu))

    num = (agg * Q_U).sum()
    den = agg.sum() or 1.0
    q = num / den

    return q, agg


if __name__ == "__main__":
    # Example: Switzerland-ish
    q_ch, agg_ch = eval_country(hdi=0.97, le=83, gni=65000)
    print("Quality score (CH-like):", q_ch)

    # Example: lower-HDI country
    q_x, agg_x = eval_country(hdi=0.65, le=69, gni=8000)
    print("Quality score (example):", q_x)

    # Plot aggregated output for the first example
    import matplotlib.pyplot as plt
    plt.plot(Q_U, agg_ch)
    plt.xlabel("Quality")
    plt.ylabel("μ")
    plt.title("Aggregated output membership (CH-like country)")
    plt.show()
