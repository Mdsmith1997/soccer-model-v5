from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "mls_v5_btts_market_matched.csv"
)

OUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "mls_btts_market_validation_v1.csv"
)

THRESHOLDS = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.14,
    0.16,
    0.18,
    0.20,
]


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 110)
print("MLS BTTS MARKET VALIDATION V1")
print("=" * 110)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
)

print()
print("Loaded rows:", len(df))


# ============================================================
# CLEAN
# ============================================================

df["model_btts_yes"] = pd.to_numeric(
    df["model_btts_yes"],
    errors="coerce",
)

df["market_btts_yes_novig"] = pd.to_numeric(
    df["market_btts_yes_novig"],
    errors="coerce",
)

df["market_btts_no_novig"] = pd.to_numeric(
    df["market_btts_no_novig"],
    errors="coerce",
)

df["odds_btts_yes"] = pd.to_numeric(
    df["odds_btts_yes"],
    errors="coerce",
)

df["odds_btts_no"] = pd.to_numeric(
    df["odds_btts_no"],
    errors="coerce",
)

df["homeGoalCount"] = pd.to_numeric(
    df["homeGoalCount"],
    errors="coerce",
)

df["awayGoalCount"] = pd.to_numeric(
    df["awayGoalCount"],
    errors="coerce",
)


valid = df[
    (df["odds_btts_yes"] > 1)
    &
    (df["odds_btts_no"] > 1)
    &
    df["model_btts_yes"].notna()
    &
    df["market_btts_yes_novig"].notna()
    &
    df["homeGoalCount"].notna()
    &
    df["awayGoalCount"].notna()
].copy()


valid["actual_btts"] = (
    (valid["homeGoalCount"] > 0)
    &
    (valid["awayGoalCount"] > 0)
).astype(int)


valid["model_btts_no"] = (
    1.0
    -
    valid["model_btts_yes"]
)


valid["yes_edge"] = (
    valid["model_btts_yes"]
    -
    valid["market_btts_yes_novig"]
)


valid["no_edge"] = (
    valid["model_btts_no"]
    -
    valid["market_btts_no_novig"]
)


# ============================================================
# SEASON
# ============================================================

if "source_year" in valid.columns:

    valid["season"] = pd.to_numeric(
        valid["source_year"],
        errors="coerce",
    )

elif "download_season" in valid.columns:

    valid["season"] = pd.to_numeric(
        valid["download_season"],
        errors="coerce",
    )

else:

    raise ValueError(
        "Could not identify MLS season column."
    )


# ============================================================
# PROBABILITY QUALITY
# ============================================================

actual = valid["actual_btts"].to_numpy()

model_p = valid["model_btts_yes"].to_numpy()

market_p = (
    valid["market_btts_yes_novig"]
    .to_numpy()
)


model_brier = np.mean(
    (model_p - actual) ** 2
)

market_brier = np.mean(
    (market_p - actual) ** 2
)


eps = 1e-12

model_clip = np.clip(
    model_p,
    eps,
    1 - eps,
)

market_clip = np.clip(
    market_p,
    eps,
    1 - eps,
)


model_logloss = -np.mean(
    actual * np.log(model_clip)
    +
    (1 - actual)
    * np.log(1 - model_clip)
)

market_logloss = -np.mean(
    actual * np.log(market_clip)
    +
    (1 - actual)
    * np.log(1 - market_clip)
)


print()
print("=" * 110)
print("PROBABILITY QUALITY: MODEL VS MARKET")
print("=" * 110)

print()
print("Games:", len(valid))

print()
print(
    f"Actual BTTS rate:       {valid['actual_btts'].mean():.2%}"
)

print(
    f"Average model P(BTTS):  {valid['model_btts_yes'].mean():.2%}"
)

print(
    f"Average market P(BTTS): {valid['market_btts_yes_novig'].mean():.2%}"
)

print()
print(
    f"Model Brier:  {model_brier:.5f}"
)

print(
    f"Market Brier: {market_brier:.5f}"
)

print()
print(
    f"Model Log Loss:  {model_logloss:.5f}"
)

print(
    f"Market Log Loss: {market_logloss:.5f}"
)


# ============================================================
# FLAT STAKE TEST
# ============================================================

def evaluate_bets(
    data,
    side,
    threshold,
):

    if side == "YES":

        bets = data[
            data["yes_edge"]
            >= threshold
        ].copy()

        if len(bets) == 0:

            return None

        bets["won"] = (
            bets["actual_btts"]
            == 1
        )

        bets["odds"] = (
            bets["odds_btts_yes"]
        )

        bets["edge"] = (
            bets["yes_edge"]
        )

    else:

        bets = data[
            data["no_edge"]
            >= threshold
        ].copy()

        if len(bets) == 0:

            return None

        bets["won"] = (
            bets["actual_btts"]
            == 0
        )

        bets["odds"] = (
            bets["odds_btts_no"]
        )

        bets["edge"] = (
            bets["no_edge"]
        )


    bets["profit"] = np.where(
        bets["won"],
        bets["odds"] - 1.0,
        -1.0,
    )


    n = len(bets)

    wins = int(
        bets["won"].sum()
    )

    profit = float(
        bets["profit"].sum()
    )

    roi = profit / n


    return {
        "side": side,
        "threshold": threshold,
        "bets": n,
        "wins": wins,
        "win_rate": wins / n,
        "avg_odds": bets["odds"].mean(),
        "avg_edge": bets["edge"].mean(),
        "profit_units": profit,
        "roi": roi,
    }


# ============================================================
# THRESHOLD SWEEP
# ============================================================

results = []

for threshold in THRESHOLDS:

    for side in [
        "YES",
        "NO",
    ]:

        result = evaluate_bets(
            valid,
            side,
            threshold,
        )

        if result is not None:

            results.append(result)


results_df = pd.DataFrame(
    results
)


print()
print("=" * 110)
print("FLAT-STAKE ROI BY MODEL EDGE")
print("=" * 110)

display = results_df.copy()

display["threshold"] = (
    display["threshold"]
    .map(lambda x: f"{x:.0%}")
)

display["win_rate"] = (
    display["win_rate"]
    .map(lambda x: f"{x:.2%}")
)

display["avg_odds"] = (
    display["avg_odds"]
    .map(lambda x: f"{x:.3f}")
)

display["avg_edge"] = (
    display["avg_edge"]
    .map(lambda x: f"{x:.2%}")
)

display["profit_units"] = (
    display["profit_units"]
    .map(lambda x: f"{x:+.2f}")
)

display["roi"] = (
    display["roi"]
    .map(lambda x: f"{x:+.2%}")
)

print()
print(
    display.to_string(
        index=False
    )
)


# ============================================================
# SEASON-BY-SEASON
# ============================================================

print()
print("=" * 110)
print("SEASON-BY-SEASON THRESHOLD TEST")
print("=" * 110)


season_results = []

for season in sorted(
    valid["season"]
    .dropna()
    .unique()
):

    season_data = valid[
        valid["season"] == season
    ]

    for threshold in THRESHOLDS:

        for side in [
            "YES",
            "NO",
        ]:

            result = evaluate_bets(
                season_data,
                side,
                threshold,
            )

            if result is None:

                continue

            result["season"] = int(
                season
            )

            season_results.append(
                result
            )


season_df = pd.DataFrame(
    season_results
)


# ============================================================
# ROBUSTNESS SUMMARY
# ============================================================

robustness = []

for side in [
    "YES",
    "NO",
]:

    for threshold in THRESHOLDS:

        x = season_df[
            (season_df["side"] == side)
            &
            (
                np.isclose(
                    season_df["threshold"],
                    threshold,
                )
            )
        ].copy()

        if len(x) == 0:

            continue

        robustness.append(
            {
                "side":
                    side,

                "threshold":
                    threshold,

                "seasons":
                    len(x),

                "positive_seasons":
                    int(
                        (x["roi"] > 0)
                        .sum()
                    ),

                "negative_seasons":
                    int(
                        (x["roi"] < 0)
                        .sum()
                    ),

                "total_bets":
                    int(
                        x["bets"].sum()
                    ),

                "mean_season_roi":
                    x["roi"].mean(),

                "median_season_roi":
                    x["roi"].median(),

                "worst_season_roi":
                    x["roi"].min(),

                "best_season_roi":
                    x["roi"].max(),
            }
        )


robust_df = pd.DataFrame(
    robustness
)


print()
print("ROBUSTNESS ACROSS MLS SEASONS")
print()


robust_display = robust_df.copy()

robust_display["threshold"] = (
    robust_display["threshold"]
    .map(lambda x: f"{x:.0%}")
)

for col in [
    "mean_season_roi",
    "median_season_roi",
    "worst_season_roi",
    "best_season_roi",
]:

    robust_display[col] = (
        robust_display[col]
        .map(lambda x: f"{x:+.2%}")
    )


print(
    robust_display.to_string(
        index=False
    )
)


# ============================================================
# INDIVIDUAL SEASON TABLES
# ============================================================

print()
print("=" * 110)
print("INDIVIDUAL SEASON RESULTS")
print("=" * 110)


for season in sorted(
    season_df["season"].unique()
):

    print()
    print("-" * 110)

    print(
        f"MLS {season}"
    )

    x = season_df[
        season_df["season"]
        == season
    ].copy()

    x["threshold"] = (
        x["threshold"]
        .map(lambda z: f"{z:.0%}")
    )

    x["win_rate"] = (
        x["win_rate"]
        .map(lambda z: f"{z:.2%}")
    )

    x["avg_odds"] = (
        x["avg_odds"]
        .map(lambda z: f"{z:.3f}")
    )

    x["avg_edge"] = (
        x["avg_edge"]
        .map(lambda z: f"{z:.2%}")
    )

    x["profit_units"] = (
        x["profit_units"]
        .map(lambda z: f"{z:+.2f}")
    )

    x["roi"] = (
        x["roi"]
        .map(lambda z: f"{z:+.2%}")
    )

    print(
        x[
            [
                "side",
                "threshold",
                "bets",
                "wins",
                "win_rate",
                "avg_odds",
                "avg_edge",
                "profit_units",
                "roi",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUT_FILE,
    index=False,
)


print()
print("=" * 110)
print("OUTPUT")
print("=" * 110)

print()
print(OUT_FILE)

print()
print("DONE")
