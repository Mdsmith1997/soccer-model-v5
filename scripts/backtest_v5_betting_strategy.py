from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "v5_market_comparison.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "v5_betting_strategy_backtest.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["season"] = (
    df["season"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.zfill(4)
)

numeric_cols = [
    "home_goals",
    "away_goals",
    "p_home_v5",
    "p_draw_v5",
    "p_away_v5",
    "market_home_odds",
    "market_draw_odds",
    "market_away_odds",
    "market_nv_home",
    "market_nv_draw",
    "market_nv_away",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )


# ============================================================
# ACTUAL RESULT
# ============================================================

df["actual"] = np.select(
    [
        df["home_goals"] > df["away_goals"],
        df["home_goals"] == df["away_goals"],
    ],
    [
        "HOME",
        "DRAW",
    ],
    default="AWAY",
)


# ============================================================
# CONVERT EACH MATCH TO THREE BET OPPORTUNITIES
# ============================================================

records = []

for side, model_col, market_col, odds_col in [
    (
        "HOME",
        "p_home_v5",
        "market_nv_home",
        "market_home_odds",
    ),
    (
        "DRAW",
        "p_draw_v5",
        "market_nv_draw",
        "market_draw_odds",
    ),
    (
        "AWAY",
        "p_away_v5",
        "market_nv_away",
        "market_away_odds",
    ),
]:

    x = df[
        [
            "match_id",
            "date",
            "season",
            "league",
            "home_team",
            "away_team",
            "actual",
            model_col,
            market_col,
            odds_col,
        ]
    ].copy()

    x.columns = [
        "match_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
        "actual",
        "model_probability",
        "market_probability",
        "decimal_odds",
    ]

    x["side"] = side

    records.append(x)


bets = pd.concat(
    records,
    ignore_index=True,
)

bets = bets.dropna(
    subset=[
        "model_probability",
        "market_probability",
        "decimal_odds",
    ]
).copy()


# ============================================================
# BETTING METRICS
# ============================================================

bets["edge"] = (
    bets["model_probability"]
    - bets["market_probability"]
)

bets["ev"] = (
    bets["model_probability"]
    * bets["decimal_odds"]
    - 1.0
)

bets["won"] = (
    bets["side"] == bets["actual"]
).astype(int)

bets["profit"] = np.where(
    bets["won"] == 1,
    bets["decimal_odds"] - 1.0,
    -1.0,
)


# ============================================================
# DEVELOPMENT / HOLDOUT
#
# Never use 2025/26 to choose thresholds.
# ============================================================

development = bets[
    bets["season"].isin(
        [
            "2122",
            "2223",
            "2324",
            "2425",
        ]
    )
].copy()

holdout = bets[
    bets["season"] == "2526"
].copy()


print("=" * 120)
print("V5 BETTING STRATEGY BACKTEST")
print("=" * 120)

print()
print("All opportunities:", len(bets))
print("Development:", len(development))
print("2025/26 holdout:", len(holdout))


# ============================================================
# GRID
# ============================================================

results = []

edge_min_values = [
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
]

edge_max_values = [
    0.10,
    0.15,
    0.20,
    0.25,
    1.00,
]

ev_min_values = [
    0.00,
    0.03,
    0.05,
    0.10,
    0.15,
    0.20,
]

max_odds_values = [
    1.75,
    2.00,
    2.50,
    3.00,
    4.00,
    5.00,
    10.00,
    100.00,
]


for edge_min in edge_min_values:

    for edge_max in edge_max_values:

        if edge_max <= edge_min:
            continue

        for ev_min in ev_min_values:

            for max_odds in max_odds_values:

                x = development.loc[
                    (
                        development["edge"]
                        >= edge_min
                    )
                    &
                    (
                        development["edge"]
                        <= edge_max
                    )
                    &
                    (
                        development["ev"]
                        >= ev_min
                    )
                    &
                    (
                        development["decimal_odds"]
                        <= max_odds
                    )
                ].copy()

                n = len(x)

                if n < 75:
                    continue

                profit = x["profit"].sum()

                roi = profit / n

                wins = x["won"].sum()

                win_rate = x["won"].mean()

                avg_odds = x[
                    "decimal_odds"
                ].mean()

                # --------------------------------------------
                # Season robustness
                # --------------------------------------------

                season_stats = (
                    x.groupby("season")
                    .agg(
                        bets=("won", "size"),
                        profit=("profit", "sum"),
                    )
                )

                season_stats["roi"] = (
                    season_stats["profit"]
                    /
                    season_stats["bets"]
                )

                positive_seasons = int(
                    (
                        season_stats["roi"] > 0
                    ).sum()
                )

                worst_season_roi = (
                    season_stats["roi"].min()
                )

                results.append(
                    {
                        "edge_min": edge_min,
                        "edge_max": edge_max,
                        "ev_min": ev_min,
                        "max_odds": max_odds,
                        "bets": n,
                        "wins": wins,
                        "win_rate": win_rate,
                        "avg_odds": avg_odds,
                        "profit": profit,
                        "roi": roi,
                        "positive_seasons":
                            positive_seasons,
                        "worst_season_roi":
                            worst_season_roi,
                    }
                )


results = pd.DataFrame(results)


# ============================================================
# ROBUSTNESS FILTER
#
# We don't simply select highest ROI.
# ============================================================

robust = results.loc[
    (results["bets"] >= 150)
    &
    (results["positive_seasons"] >= 3)
    &
    (results["worst_season_roi"] > -0.15)
].copy()


if robust.empty:

    print()
    print(
        "No strategy passed the robustness filter."
    )

    robust = results.copy()


robust = robust.sort_values(
    [
        "positive_seasons",
        "roi",
        "bets",
    ],
    ascending=[
        False,
        False,
        False,
    ],
)


print()
print("=" * 120)
print("TOP DEVELOPMENT STRATEGIES")
print("=" * 120)

print(
    robust.head(25)
    .round(4)
    .to_string(index=False)
)


# ============================================================
# TEST TOP 10 ON UNTOUCHED 2025/26
# ============================================================

holdout_results = []

for _, row in robust.head(10).iterrows():

    x = holdout.loc[
        (
            holdout["edge"]
            >= row["edge_min"]
        )
        &
        (
            holdout["edge"]
            <= row["edge_max"]
        )
        &
        (
            holdout["ev"]
            >= row["ev_min"]
        )
        &
        (
            holdout["decimal_odds"]
            <= row["max_odds"]
        )
    ].copy()

    n = len(x)

    if n == 0:
        continue

    holdout_results.append(
        {
            "edge_min":
                row["edge_min"],

            "edge_max":
                row["edge_max"],

            "ev_min":
                row["ev_min"],

            "max_odds":
                row["max_odds"],

            "development_bets":
                int(row["bets"]),

            "development_roi":
                row["roi"],

            "holdout_bets":
                n,

            "holdout_wins":
                int(x["won"].sum()),

            "holdout_win_rate":
                x["won"].mean(),

            "holdout_avg_odds":
                x["decimal_odds"].mean(),

            "holdout_profit":
                x["profit"].sum(),

            "holdout_roi":
                x["profit"].sum() / n,
        }
    )


holdout_results = pd.DataFrame(
    holdout_results
)


print()
print("=" * 120)
print("2025/26 UNTOUCHED HOLDOUT")
print("=" * 120)

if len(holdout_results):

    print(
        holdout_results
        .round(4)
        .to_string(index=False)
    )

else:

    print(
        "No qualifying holdout bets."
    )


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("Saved:", OUTPUT_FILE)
