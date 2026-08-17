from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.test_three_leagues_v5_totals_oos import (
    load_predictions,
    load_market,
    p_under_2_5,
)

LEAGUE = "League One"

# Frozen before League One results were inspected.
THRESHOLD = 0.11

SEASONS = [
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
    "2526",
]

EARLY = [
    "1920",
    "2021",
    "2122",
    "2223",
]

RECENT = [
    "2324",
    "2425",
    "2526",
]


def build_dataset():

    pred = load_predictions()

    pred = pred[
        pred["league"].eq(LEAGUE)
    ].copy()

    market = load_market()

    market = market[
        market["league"].eq(LEAGUE)
    ].copy()

    df = pred.merge(
        market,
        on=[
            "season",
            "league",
            "date",
            "home_key",
            "away_key",
        ],
        how="inner",
        validate="one_to_one",
    )

    df = df[
        df["over_odds"].gt(1)
        &
        df["under_odds"].gt(1)
    ].copy()

    df["model_p_under"] = [
        p_under_2_5(h, a)
        for h, a in zip(
            df["home_lambda"],
            df["away_lambda"],
        )
    ]

    raw_over = (
        1.0
        /
        df["over_odds"]
    )

    raw_under = (
        1.0
        /
        df["under_odds"]
    )

    vig_sum = (
        raw_over
        +
        raw_under
    )

    df["market_p_under"] = (
        raw_under
        /
        vig_sum
    )

    df["under_edge"] = (
        df["model_p_under"]
        -
        df["market_p_under"]
    )

    df["under_ev"] = (
        df["model_p_under"]
        *
        df["under_odds"]
        -
        1.0
    )

    df["actual_total"] = (
        df["home_goals"]
        +
        df["away_goals"]
    )

    df["won"] = (
        df["actual_total"]
        <
        2.5
    ).astype(int)

    df["qualifies"] = (
        df["under_edge"]
        >=
        THRESHOLD
    )

    df["profit"] = np.where(
        df["won"].eq(1),
        df["under_odds"] - 1.0,
        -1.0,
    )

    return df


def performance(df):

    x = df[
        df["qualifies"]
    ].copy()

    if x.empty:

        return {
            "bets": 0,
            "wins": 0,
            "win_rate": np.nan,
            "avg_odds": np.nan,
            "avg_edge": np.nan,
            "avg_ev": np.nan,
            "profit": 0.0,
            "roi": np.nan,
        }

    bets = len(x)
    wins = int(
        x["won"].sum()
    )

    profit = float(
        x["profit"].sum()
    )

    return {
        "bets":
            bets,

        "wins":
            wins,

        "win_rate":
            wins / bets,

        "avg_odds":
            x["under_odds"].mean(),

        "avg_edge":
            x["under_edge"].mean(),

        "avg_ev":
            x["under_ev"].mean(),

        "profit":
            profit,

        "roi":
            profit / bets,
    }


def print_result(label, p):

    if not p["bets"]:

        print(
            f"{label:<28} "
            "no qualifying bets"
        )

        return

    print(
        f"{label:<28} "
        f"bets={p['bets']:4d}  "
        f"wins={p['wins']:4d}  "
        f"WR={p['win_rate']:7.2%}  "
        f"odds={p['avg_odds']:.3f}  "
        f"edge={p['avg_edge']:+7.2%}  "
        f"EV={p['avg_ev']:+7.2%}  "
        f"profit={p['profit']:+8.2f}u  "
        f"ROI={p['roi']:+7.2%}"
    )


def bootstrap_roi(
    df,
    iterations=20000,
    seed=42,
):

    x = df[
        df["qualifies"]
    ].copy()

    if len(x) < 2:
        return np.nan, np.nan, np.nan

    profits = (
        x["profit"]
        .to_numpy(
            dtype=float
        )
    )

    rng = np.random.default_rng(
        seed
    )

    sampled = rng.choice(
        profits,
        size=(
            iterations,
            len(profits),
        ),
        replace=True,
    )

    rois = sampled.mean(
        axis=1
    )

    low = np.quantile(
        rois,
        0.025,
    )

    median = np.quantile(
        rois,
        0.50,
    )

    high = np.quantile(
        rois,
        0.975,
    )

    return (
        low,
        median,
        high,
    )


def main():

    print()
    print("=" * 130)
    print(
        "LEAGUE ONE V5 TOTALS — "
        "FROZEN 11% STABILITY VALIDATION"
    )
    print("=" * 130)

    print()
    print(
        "Rule: RAW V5 UNDER 2.5 "
        "EDGE >= 11%"
    )

    print(
        "Threshold is frozen. "
        "No League One tuning."
    )

    df = build_dataset()

    print()
    print(
        f"Matched League One games: "
        f"{len(df):,}"
    )

    print()

    # ========================================================
    # FULL SAMPLE
    # ========================================================

    print("=" * 130)
    print("FULL OUT-OF-SAMPLE LEAGUE ONE RESULT")
    print("=" * 130)
    print()

    print_result(
        "ALL SEASONS",
        performance(df),
    )

    # ========================================================
    # CHRONOLOGICAL BLOCKS
    # ========================================================

    print()
    print("=" * 130)
    print("CHRONOLOGICAL STABILITY")
    print("=" * 130)
    print()

    early = df[
        df["season"].isin(
            EARLY
        )
    ]

    recent = df[
        df["season"].isin(
            RECENT
        )
    ]

    print_result(
        "EARLY 1920-2223",
        performance(early),
    )

    print_result(
        "RECENT 2324-2526",
        performance(recent),
    )

    # ========================================================
    # SEASON BY SEASON
    # ========================================================

    print()
    print("=" * 130)
    print("SEASON-BY-SEASON")
    print("=" * 130)
    print()

    for season in SEASONS:

        x = df[
            df["season"].eq(
                season
            )
        ]

        print_result(
            season,
            performance(x),
        )

    # ========================================================
    # LEAVE-ONE-SEASON-OUT
    # ========================================================

    print()
    print("=" * 130)
    print("LEAVE-ONE-SEASON-OUT ROBUSTNESS")
    print("=" * 130)

    print()
    print(
        "If profitability disappears whenever "
        "one strong season is removed, "
        "the edge is fragile."
    )
    print()

    for season in SEASONS:

        x = df[
            ~df["season"].eq(
                season
            )
        ]

        print_result(
            f"EXCLUDE {season}",
            performance(x),
        )

    # ========================================================
    # BOOTSTRAP
    # ========================================================

    print()
    print("=" * 130)
    print("BOOTSTRAP ROI UNCERTAINTY")
    print("=" * 130)
    print()

    low, median, high = (
        bootstrap_roi(
            df
        )
    )

    print(
        f"Bootstrap median ROI: "
        f"{median:+.2%}"
    )

    print(
        f"95% bootstrap interval: "
        f"{low:+.2%} to {high:+.2%}"
    )

    print()

    if low > 0:

        print(
            "STRONG PASS: even the lower "
            "bootstrap bound is positive."
        )

    elif median > 0:

        print(
            "PROVISIONAL PASS: aggregate edge "
            "is positive, but uncertainty still "
            "includes zero."
        )

    else:

        print(
            "FAIL: bootstrap evidence does not "
            "support a positive underlying ROI."
        )

    # ========================================================
    # MODEL EV VS REALIZED ROI
    # ========================================================

    p = performance(
        df
    )

    print()
    print("=" * 130)
    print("EV CALIBRATION CHECK")
    print("=" * 130)
    print()

    if p["bets"]:

        print(
            f"Average model EV: "
            f"{p['avg_ev']:+.2%}"
        )

        print(
            f"Realized ROI:     "
            f"{p['roi']:+.2%}"
        )

        print(
            f"EV - ROI gap:     "
            f"{p['avg_ev'] - p['roi']:+.2%}"
        )

        print()
        print(
            "A positive betting strategy does "
            "not require realized ROI to equal "
            "the model's estimated EV."
        )

        print(
            "But a very large persistent gap "
            "indicates probability overconfidence."
        )


if __name__ == "__main__":
    main()
