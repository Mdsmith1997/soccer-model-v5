from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]

ODDS_FILE = ROOT / "data/processed/laliga_eliteserien_footystats_raw.csv"
V5_FILE = ROOT / "data/processed/footystats_multileague_v5_predictions.csv"


def btts_prob(h, a):
    h = float(h)
    a = float(a)
    return 1 - math.exp(-h) - math.exp(-a) + math.exp(-(h + a))


def season_start(s):
    try:
        x = str(s)

        # V5 format can be 2021, 2122, etc.
        if "/" in x:
            return int(x.split("/")[0])

        n = int(float(x))

        if n >= 2000:
            return n

        # e.g. 2122 -> 2021
        if 1000 <= n <= 9999:
            return 2000 + int(str(n)[:2])

        return n

    except Exception:
        return np.nan


print("=" * 120)
print("LA LIGA BTTS — EXACT-ID WALK-FORWARD MARKET BACKTEST")
print("=" * 120)

# ============================================================
# LOAD
# ============================================================

odds = pd.read_csv(ODDS_FILE, low_memory=False)
v5 = pd.read_csv(V5_FILE, low_memory=False)

print("\nRaw odds rows:", len(odds))
print("V5 rows:", len(v5))

# ============================================================
# FILTER ELITESERIEN EXPLICITLY
# ============================================================

odds = odds[
    odds["model_league"]
    .astype(str)
    .str.lower()
    .eq("la_liga")
].copy()

v5 = v5[
    v5["league"]
    .astype(str)
    .str.contains(
        "La Liga",
        case=False,
        na=False,
    )
].copy()

print("\nRaw La Liga rows:", len(odds))
print("V5 La Liga rows:", len(v5))

# ============================================================
# CLEAN IDs / PRICES / MODEL INPUTS
# ============================================================

odds["id"] = pd.to_numeric(
    odds["id"],
    errors="coerce",
).astype("Int64")

v5["footystats_match_id"] = pd.to_numeric(
    v5["footystats_match_id"],
    errors="coerce",
).astype("Int64")

for c in [
    "odds_btts_yes",
    "odds_btts_no",
]:
    odds[c] = pd.to_numeric(
        odds[c],
        errors="coerce",
    )

for c in [
    "home_goals",
    "away_goals",
    "home_lambda",
    "away_lambda",
]:
    v5[c] = pd.to_numeric(
        v5[c],
        errors="coerce",
    )

odds = odds[
    odds["id"].notna()
    & odds["odds_btts_yes"].gt(1)
    & odds["odds_btts_no"].gt(1)
].copy()

v5 = v5[
    v5["footystats_match_id"].notna()
    & v5["home_goals"].notna()
    & v5["away_goals"].notna()
    & v5["home_lambda"].gt(0)
    & v5["away_lambda"].gt(0)
].copy()

print("\nPriced La Liga rows:", len(odds))

# ============================================================
# EXACT FOOTYSTATS MATCH-ID JOIN
# ============================================================

keep = [
    "id",
    "model_season",
    "footystats_season_id",
    "home_name",
    "away_name",
    "odds_btts_yes",
    "odds_btts_no",
]

odds_small = (
    odds[keep]
    .drop_duplicates("id")
    .copy()
)

m = v5.merge(
    odds_small,
    left_on="footystats_match_id",
    right_on="id",
    how="inner",
    validate="one_to_one",
)

print("\n" + "=" * 120)
print("MATCH AUDIT")
print("=" * 120)

print("V5 usable La Liga games:", len(v5))
print("Priced raw La Liga games:", len(odds_small))
print("Exact-ID matched games:", len(m))

print(
    "V5 match coverage:",
    f"{len(m) / len(v5):.2%}"
    if len(v5)
    else "N/A",
)

print(
    "Raw-price match coverage:",
    f"{len(m) / len(odds_small):.2%}"
    if len(odds_small)
    else "N/A",
)

if len(m) < 750:
    raise RuntimeError(
        "Exact-ID match count is unexpectedly low. "
        "Stop before evaluating ROI."
    )

# ============================================================
# TEAM-NAME SANITY CHECK
# ============================================================

print("\nSample exact-ID matches:")

sample_cols = [
    "footystats_match_id",
    "date",
    "season",
    "home_team",
    "away_team",
    "home_name",
    "away_name",
    "odds_btts_yes",
    "odds_btts_no",
]

print(
    m[sample_cols]
    .head(12)
    .to_string(index=False)
)

# ============================================================
# OUTCOME / RAW BTTS PROBABILITY
# ============================================================

m["actual_yes"] = (
    (m["home_goals"] > 0)
    & (m["away_goals"] > 0)
).astype(int)

m["p_raw"] = [
    btts_prob(h, a)
    for h, a in zip(
        m["home_lambda"],
        m["away_lambda"],
    )
]

m["season_num"] = m["model_season"].map(
    season_start
)

print("\nMatched seasons:")
print(
    m["season_num"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ============================================================
# WALK-FORWARD PLATT CALIBRATION
# ============================================================

parts = []

seasons = sorted(
    m["season_num"]
    .dropna()
    .unique()
)

print("\nWalk-forward seasons:", seasons)

for season in seasons:

    train = m[
        m["season_num"] < season
    ].copy()

    test = m[
        m["season_num"] == season
    ].copy()

    if len(train) < 150:
        print(
            f"Skipping {season}: "
            f"only {len(train)} prior training games."
        )
        continue

    if len(test) < 50:
        print(
            f"Skipping {season}: "
            f"only {len(test)} test games."
        )
        continue

    if train["actual_yes"].nunique() < 2:
        continue

    model = LogisticRegression(
        solver="lbfgs"
    )

    model.fit(
        train[["p_raw"]],
        train["actual_yes"],
    )

    test["p_yes_cal"] = (
        model.predict_proba(
            test[["p_raw"]]
        )[:, 1]
    )

    print(
        f"{season}: "
        f"train={len(train)} "
        f"test={len(test)}"
    )

    parts.append(test)

if not parts:
    raise RuntimeError(
        "No valid walk-forward seasons."
    )

oos = pd.concat(
    parts,
    ignore_index=True,
)

oos["p_no_cal"] = (
    1 - oos["p_yes_cal"]
)

# ============================================================
# NO-VIG MARKET
# ============================================================

oos["imp_yes"] = (
    1 / oos["odds_btts_yes"]
)

oos["imp_no"] = (
    1 / oos["odds_btts_no"]
)

oos["overround"] = (
    oos["imp_yes"]
    + oos["imp_no"]
)

oos["market_yes_nv"] = (
    oos["imp_yes"]
    / oos["overround"]
)

oos["market_no_nv"] = (
    oos["imp_no"]
    / oos["overround"]
)

oos["edge_yes"] = (
    oos["p_yes_cal"]
    - oos["market_yes_nv"]
)

oos["edge_no"] = (
    oos["p_no_cal"]
    - oos["market_no_nv"]
)

print("\nOOS market games:", len(oos))

print(
    "Mean calibrated BTTS Yes:",
    f"{oos['p_yes_cal'].mean():.2%}",
)

print(
    "Actual BTTS Yes:",
    f"{oos['actual_yes'].mean():.2%}",
)

print(
    "Mean no-vig market BTTS Yes:",
    f"{oos['market_yes_nv'].mean():.2%}",
)

print(
    "Median market overround:",
    f"{oos['overround'].median():.3f}",
)

# ============================================================
# THRESHOLD AUDIT
# ============================================================

thresholds = [
    0.00,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.10,
    0.12,
    0.15,
]

rows = []

for side in ["YES", "NO"]:

    edge_col = (
        "edge_yes"
        if side == "YES"
        else "edge_no"
    )

    odds_col = (
        "odds_btts_yes"
        if side == "YES"
        else "odds_btts_no"
    )

    for threshold in thresholds:

        b = oos[
            oos[edge_col] >= threshold
        ].copy()

        if b.empty:
            continue

        wins = (
            b["actual_yes"].eq(1)
            if side == "YES"
            else b["actual_yes"].eq(0)
        )

        profit = np.where(
            wins,
            b[odds_col] - 1,
            -1.0,
        )

        rows.append({
            "side": side,
            "edge": threshold,
            "bets": len(b),
            "wins": int(wins.sum()),
            "win_rate": wins.mean(),
            "avg_odds": b[odds_col].mean(),
            "avg_edge": b[edge_col].mean(),
            "profit_u": profit.sum(),
            "roi": profit.mean(),
        })

result = pd.DataFrame(rows)

print("\n" + "=" * 120)
print("EDGE THRESHOLD RESULTS")
print("=" * 120)

show = result.copy()

for c in [
    "win_rate",
    "avg_edge",
    "roi",
]:
    show[c] = show[c].map(
        lambda z: f"{z:+.2%}"
    )

show["edge"] = show["edge"].map(
    lambda z: f">={z:.0%}"
)

show["avg_odds"] = (
    show["avg_odds"]
    .map(lambda z: f"{z:.2f}")
)

show["profit_u"] = (
    show["profit_u"]
    .map(lambda z: f"{z:+.2f}")
)

print(
    show.to_string(
        index=False
    )
)

# ============================================================
# SEASON STABILITY
# ============================================================

print("\n" + "=" * 120)
print("SEASON STABILITY")
print("=" * 120)

for side in ["YES", "NO"]:

    edge_col = (
        "edge_yes"
        if side == "YES"
        else "edge_no"
    )

    odds_col = (
        "odds_btts_yes"
        if side == "YES"
        else "odds_btts_no"
    )

    for threshold in [
        0.03,
        0.05,
        0.07,
        0.10,
    ]:

        print(
            f"\n{side} | "
            f"edge >= {threshold:.0%}"
        )

        rows = []

        for season, g in oos.groupby(
            "season_num"
        ):

            b = g[
                g[edge_col] >= threshold
            ].copy()

            if b.empty:
                continue

            wins = (
                b["actual_yes"].eq(1)
                if side == "YES"
                else b["actual_yes"].eq(0)
            )

            profit = np.where(
                wins,
                b[odds_col] - 1,
                -1.0,
            )

            rows.append({
                "season": int(season),
                "bets": len(b),
                "wins": int(wins.sum()),
                "profit": profit.sum(),
                "roi": profit.mean(),
            })

        s = pd.DataFrame(rows)

        if s.empty:
            print("NO BETS")
            continue

        s["profit"] = s["profit"].map(
            lambda z: f"{z:+.2f}"
        )

        s["roi"] = s["roi"].map(
            lambda z: f"{z:+.2%}"
        )

        print(
            s.to_string(
                index=False
            )
        )

# ============================================================
# SAVE RESEARCH OUTPUT
# ============================================================

OUT = (
    ROOT
    / "data/processed"
    / "laliga_btts_market_oos.csv"
)

oos.to_csv(
    OUT,
    index=False,
)

print("\nSaved:")
print(OUT)

print("\n" + "=" * 120)
print("LIVE-PROMOTION STANDARD")
print("=" * 120)

print("""
We are looking for robustness, not the single highest ROI.

A live candidate should show:
1. Positive OOS flat-bet ROI.
2. Meaningful number of bets.
3. Neighboring edge thresholds behaving sensibly.
4. Positive performance across multiple seasons.
5. No dependence on one tiny/extreme subgroup.
6. Walk-forward calibrated probability versus no-vig market.
7. A threshold chosen conservatively rather than retrospectively
   selecting whichever number happened to maximize ROI.

No live strategy is modified by this script.
""")
