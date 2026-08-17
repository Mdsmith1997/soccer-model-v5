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
    / "eliteserien_btts_market_oos.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
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
print("ELITESERIEN BTTS MARKET VALIDATION V1")
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

numeric_cols = [
    "season",
    "home_goals",
    "away_goals",
    "home_lambda",
    "away_lambda",
    "actual_yes",
    "p_raw",
    "p_yes_cal",
    "p_no_cal",
    "odds_btts_yes",
    "odds_btts_no",
    "market_yes_nv",
    "market_no_nv",
]

for c in numeric_cols:

    if c in df.columns:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )


required = [
    "season",
    "actual_yes",
    "p_raw",
    "p_yes_cal",
    "p_no_cal",
    "odds_btts_yes",
    "odds_btts_no",
    "market_yes_nv",
    "market_no_nv",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


valid = df[
    df["actual_yes"].notna()
    &
    df["p_raw"].notna()
    &
    df["p_yes_cal"].notna()
    &
    df["market_yes_nv"].notna()
    &
    (df["odds_btts_yes"] > 1)
    &
    (df["odds_btts_no"] > 1)
].copy()


valid["actual_yes"] = (
    valid["actual_yes"]
    .astype(int)
)


# ============================================================
# RECOMPUTE EVERYTHING
#
# Do not trust stored edge columns blindly.
# ============================================================

valid["p_raw_no"] = (
    1.0
    -
    valid["p_raw"]
)

valid["p_cal_no"] = (
    1.0
    -
    valid["p_yes_cal"]
)


valid["raw_edge_yes"] = (
    valid["p_raw"]
    -
    valid["market_yes_nv"]
)

valid["raw_edge_no"] = (
    valid["p_raw_no"]
    -
    valid["market_no_nv"]
)


valid["cal_edge_yes"] = (
    valid["p_yes_cal"]
    -
    valid["market_yes_nv"]
)

valid["cal_edge_no"] = (
    valid["p_cal_no"]
    -
    valid["market_no_nv"]
)


# ============================================================
# SCORING FUNCTIONS
# ============================================================

def brier(y, p):

    return float(
        np.mean(
            (p - y) ** 2
        )
    )


def log_loss(y, p):

    eps = 1e-12

    p = np.clip(
        p,
        eps,
        1 - eps,
    )

    return float(
        -np.mean(
            y * np.log(p)
            +
            (1 - y)
            * np.log(1 - p)
        )
    )


# ============================================================
# PROBABILITY QUALITY
# ============================================================

y = valid["actual_yes"].to_numpy()

raw_p = valid["p_raw"].to_numpy()

cal_p = valid["p_yes_cal"].to_numpy()

market_p = valid["market_yes_nv"].to_numpy()


quality = pd.DataFrame(
    [
        {
            "source": "RAW MODEL",
            "avg_probability": raw_p.mean(),
            "brier": brier(y, raw_p),
            "log_loss": log_loss(y, raw_p),
        },
        {
            "source": "CALIBRATED MODEL",
            "avg_probability": cal_p.mean(),
            "brier": brier(y, cal_p),
            "log_loss": log_loss(y, cal_p),
        },
        {
            "source": "MARKET NO-VIG",
            "avg_probability": market_p.mean(),
            "brier": brier(y, market_p),
            "log_loss": log_loss(y, market_p),
        },
    ]
)


print()
print("=" * 110)
print("PROBABILITY QUALITY")
print("=" * 110)

print()
print("Games:", len(valid))
print(
    f"Actual BTTS rate: {valid['actual_yes'].mean():.2%}"
)

print()

q = quality.copy()

q["avg_probability"] = (
    q["avg_probability"]
    .map(lambda x: f"{x:.2%}")
)

q["brier"] = (
    q["brier"]
    .map(lambda x: f"{x:.5f}")
)

q["log_loss"] = (
    q["log_loss"]
    .map(lambda x: f"{x:.5f}")
)

print(
    q.to_string(
        index=False
    )
)


# ============================================================
# BET EVALUATION
# ============================================================

def evaluate_bets(
    data,
    model,
    side,
    threshold,
):

    if model == "RAW":

        edge_col = (
            "raw_edge_yes"
            if side == "YES"
            else "raw_edge_no"
        )

    else:

        edge_col = (
            "cal_edge_yes"
            if side == "YES"
            else "cal_edge_no"
        )


    bets = data[
        data[edge_col]
        >= threshold
    ].copy()


    if len(bets) == 0:

        return None


    if side == "YES":

        bets["won"] = (
            bets["actual_yes"]
            == 1
        )

        bets["odds"] = (
            bets["odds_btts_yes"]
        )

    else:

        bets["won"] = (
            bets["actual_yes"]
            == 0
        )

        bets["odds"] = (
            bets["odds_btts_no"]
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


    return {
        "model": model,
        "side": side,
        "threshold": threshold,
        "bets": n,
        "wins": wins,
        "win_rate": wins / n,
        "avg_odds": bets["odds"].mean(),
        "avg_edge": bets[edge_col].mean(),
        "profit_units": profit,
        "roi": profit / n,
    }


# ============================================================
# OVERALL THRESHOLD SWEEP
# ============================================================

results = []

for model in [
    "RAW",
    "CAL",
]:

    for threshold in THRESHOLDS:

        for side in [
            "YES",
            "NO",
        ]:

            result = evaluate_bets(
                valid,
                model,
                side,
                threshold,
            )

            if result is not None:

                results.append(
                    result
                )


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

season_results = []

for season in sorted(
    valid["season"]
    .dropna()
    .unique()
):

    season_data = valid[
        valid["season"] == season
    ]

    for model in [
        "RAW",
        "CAL",
    ]:

        for threshold in THRESHOLDS:

            for side in [
                "YES",
                "NO",
            ]:

                result = evaluate_bets(
                    season_data,
                    model,
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
# ROBUSTNESS
# ============================================================

robustness = []

for model in [
    "RAW",
    "CAL",
]:

    for side in [
        "YES",
        "NO",
    ]:

        for threshold in THRESHOLDS:

            x = season_df[
                (season_df["model"] == model)
                &
                (season_df["side"] == side)
                &
                np.isclose(
                    season_df["threshold"],
                    threshold,
                )
            ]

            if len(x) == 0:

                continue


            robustness.append(
                {
                    "model":
                        model,

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
print("=" * 110)
print("ROBUSTNESS ACROSS SEASONS")
print("=" * 110)
print()

rd = robust_df.copy()

rd["threshold"] = (
    rd["threshold"]
    .map(lambda x: f"{x:.0%}")
)

for c in [
    "mean_season_roi",
    "median_season_roi",
    "worst_season_roi",
    "best_season_roi",
]:

    rd[c] = (
        rd[c]
        .map(lambda x: f"{x:+.2%}")
    )


print(
    rd.to_string(
        index=False
    )
)


# ============================================================
# INDIVIDUAL SEASONS
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
    print(f"ELITESERIEN {season}")

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
                "model",
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
# ODDS BAND DIAGNOSTIC
# ============================================================

print()
print("=" * 110)
print("CALIBRATED MODEL — ODDS BAND DIAGNOSTIC")
print("=" * 110)


odds_bins = [
    1.00,
    1.50,
    1.60,
    1.70,
    1.80,
    2.00,
    2.25,
    2.50,
    3.00,
    100.0,
]


for side in [
    "YES",
    "NO",
]:

    print()
    print("-" * 110)
    print(side)

    odds_col = (
        "odds_btts_yes"
        if side == "YES"
        else "odds_btts_no"
    )

    edge_col = (
        "cal_edge_yes"
        if side == "YES"
        else "cal_edge_no"
    )


    rows = []

    for low, high in zip(
        odds_bins[:-1],
        odds_bins[1:],
    ):

        x = valid[
            (valid[odds_col] >= low)
            &
            (valid[odds_col] < high)
            &
            (valid[edge_col] > 0)
        ].copy()

        if len(x) == 0:

            continue


        if side == "YES":

            won = (
                x["actual_yes"] == 1
            )

        else:

            won = (
                x["actual_yes"] == 0
            )


        profit = np.where(
            won,
            x[odds_col] - 1.0,
            -1.0,
        )


        rows.append(
            {
                "odds_band":
                    f"{low:.2f}-{high:.2f}",

                "bets":
                    len(x),

                "wins":
                    int(won.sum()),

                "win_rate":
                    won.mean(),

                "avg_odds":
                    x[odds_col].mean(),

                "avg_edge":
                    x[edge_col].mean(),

                "profit_units":
                    profit.sum(),

                "roi":
                    profit.mean(),
            }
        )


    band_df = pd.DataFrame(
        rows
    )

    if len(band_df) == 0:

        print("No bets.")
        continue


    bd = band_df.copy()

    bd["win_rate"] = (
        bd["win_rate"]
        .map(lambda z: f"{z:.2%}")
    )

    bd["avg_odds"] = (
        bd["avg_odds"]
        .map(lambda z: f"{z:.3f}")
    )

    bd["avg_edge"] = (
        bd["avg_edge"]
        .map(lambda z: f"{z:.2%}")
    )

    bd["profit_units"] = (
        bd["profit_units"]
        .map(lambda z: f"{z:+.2f}")
    )

    bd["roi"] = (
        bd["roi"]
        .map(lambda z: f"{z:+.2%}")
    )

    print(
        bd.to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUT_DIR
    / "eliteserien_btts_market_validation_v1.csv",
    index=False,
)

season_df.to_csv(
    OUT_DIR
    / "eliteserien_btts_market_validation_v1_by_season.csv",
    index=False,
)

robust_df.to_csv(
    OUT_DIR
    / "eliteserien_btts_market_validation_v1_robustness.csv",
    index=False,
)

quality.to_csv(
    OUT_DIR
    / "eliteserien_btts_market_validation_v1_quality.csv",
    index=False,
)


print()
print("=" * 110)
print("OUTPUTS")
print("=" * 110)

print()
print(
    OUT_DIR
    / "eliteserien_btts_market_validation_v1.csv"
)

print(
    OUT_DIR
    / "eliteserien_btts_market_validation_v1_by_season.csv"
)

print(
    OUT_DIR
    / "eliteserien_btts_market_validation_v1_robustness.csv"
)

print(
    OUT_DIR
    / "eliteserien_btts_market_validation_v1_quality.csv"
)

print()
print("DONE")
