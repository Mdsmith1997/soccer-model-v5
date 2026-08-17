from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data/processed"
    / "over25_remaining3_t24_games.csv"
)

LEAGUE = "Swiss Super League"

# ------------------------------------------------------------
# IMPORTANT:
# 8% WAS SELECTED ON THE ORIGINAL 2024/25 + 2025/26 SCREEN
# BEFORE ACQUIRING / EVALUATING 2020/21-2023/24.
# DO NOT OPTIMIZE THIS THRESHOLD ON THE VALIDATION SAMPLE.
# ------------------------------------------------------------

FROZEN_THRESHOLD = 0.08

DISCOVERY_SEASONS = {"2425", "2526"}
VALIDATION_SEASONS = {"2021", "2122", "2223", "2324"}


def p_over_25(home_lambda, away_lambda):
    lam = float(home_lambda) + float(away_lambda)

    p_under = math.exp(-lam) * (
        1.0
        + lam
        + lam**2 / 2.0
    )

    return 1.0 - p_under


def prepare(df):
    df = df[
        df["league"].eq(LEAGUE)
    ].copy()

    # Preserve season codes as strings.
    df["season"] = df["season"].astype(str)

    numeric = [
        "home_lambda",
        "away_lambda",
        "home_goals",
        "away_goals",
        "avg_over_odds",
        "avg_under_odds",
    ]

    for c in numeric:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = df.dropna(
        subset=numeric
    ).copy()

    df = df[
        df["avg_over_odds"].gt(1)
        & df["avg_under_odds"].gt(1)
    ].copy()

    # Same raw V5 O2.5 probability used in discovery.
    df["raw_over_prob"] = [
        p_over_25(h, a)
        for h, a in zip(
            df["home_lambda"],
            df["away_lambda"],
        )
    ]

    # Same two-way T24 de-vig used in discovery.
    over_imp = 1.0 / df["avg_over_odds"]
    under_imp = 1.0 / df["avg_under_odds"]

    df["market_over_prob"] = (
        over_imp
        / (over_imp + under_imp)
    )

    df["raw_over_edge"] = (
        df["raw_over_prob"]
        - df["market_over_prob"]
    )

    df["raw_over_ev"] = (
        df["raw_over_prob"]
        * df["avg_over_odds"]
        - 1.0
    )

    df["actual_total"] = (
        df["home_goals"]
        + df["away_goals"]
    )

    df["over_win"] = (
        df["actual_total"] > 2.5
    ).astype(int)

    return df


def performance(df, threshold=FROZEN_THRESHOLD):
    x = df[
        df["raw_over_edge"] >= threshold
    ].copy()

    if x.empty:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "avg_odds": np.nan,
            "avg_edge": np.nan,
            "avg_ev": np.nan,
            "profit": 0.0,
            "roi": np.nan,
        }

    x["profit"] = np.where(
        x["over_win"].eq(1),
        x["avg_over_odds"] - 1.0,
        -1.0,
    )

    bets = len(x)
    wins = int(x["over_win"].sum())
    profit = float(x["profit"].sum())

    return {
        "bets": bets,
        "wins": wins,
        "losses": bets - wins,
        "win_rate": wins / bets,
        "avg_odds": x["avg_over_odds"].mean(),
        "avg_edge": x["raw_over_edge"].mean(),
        "avg_ev": x["raw_over_ev"].mean(),
        "profit": profit,
        "roi": profit / bets,
    }


def print_result(label, p):
    if not p["bets"]:
        print(f"{label:<24} no bets")
        return

    print(
        f"{label:<24} "
        f"bets={p['bets']:4d} | "
        f"W-L={p['wins']:3d}-{p['losses']:<3d} | "
        f"WR={p['win_rate']:7.2%} | "
        f"odds={p['avg_odds']:.3f} | "
        f"edge={p['avg_edge']:+7.2%} | "
        f"EV={p['avg_ev']:+7.2%} | "
        f"profit={p['profit']:+8.2f}u | "
        f"ROI={p['roi']:+7.2%}"
    )


def main():
    raw = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    df = prepare(raw)

    seasons = sorted(
        df["season"].dropna().unique()
    )

    print()
    print("=" * 130)
    print("SWISS SUPER LEAGUE — FROZEN O2.5 >= 8% VALIDATION")
    print("=" * 130)

    print()
    print("RULE:")
    print("  Market: OVER 2.5")
    print("  Edge:   raw V5 probability - de-vigged T24 market probability")
    print("  Frozen threshold: >= 8.00%")
    print()

    print(f"Usable exact-2.5 games: {len(df):,}")
    print("Seasons:", ", ".join(seasons))

    print()
    print("=" * 130)
    print("FROZEN 8% — ALL SIX SEASONS")
    print("=" * 130)

    print_result(
        "ALL",
        performance(df),
    )

    print()
    print("=" * 130)
    print("FROZEN 8% — SEASON BY SEASON")
    print("=" * 130)

    for season in seasons:
        x = df[
            df["season"].eq(season)
        ]

        print_result(
            season,
            performance(x),
        )

    print()
    print("=" * 130)
    print("TRUE HOLDOUT TEST")
    print("=" * 130)

    discovery = df[
        df["season"].isin(
            DISCOVERY_SEASONS
        )
    ]

    validation = df[
        df["season"].isin(
            VALIDATION_SEASONS
        )
    ]

    print_result(
        "DISCOVERY 24/25-25/26",
        performance(discovery),
    )

    print_result(
        "HOLDOUT 20/21-23/24",
        performance(validation),
    )

    print()
    print("=" * 130)
    print("LEAVE-ONE-SEASON-OUT — FROZEN 8%")
    print("=" * 130)

    for season in seasons:
        x = df[
            ~df["season"].eq(season)
        ]

        print_result(
            f"exclude {season}",
            performance(x),
        )

    print()
    print("=" * 130)
    print("INDEPENDENT EDGE BANDS — DIAGNOSTIC ONLY")
    print("=" * 130)
    print(
        "NOTE: These bands are NOT being used to change "
        "the frozen 8% threshold."
    )
    print()

    bands = [
        (.00, .02, "0%-2%"),
        (.02, .04, "2%-4%"),
        (.04, .06, "4%-6%"),
        (.06, .08, "6%-8%"),
        (.08, .10, "8%-10%"),
        (.10, .12, "10%-12%"),
        (.12, .14, "12%-14%"),
        (.14, .16, "14%-16%"),
        (.16, np.inf, "16%+"),
    ]

    for lo, hi, label in bands:
        x = df[
            (df["raw_over_edge"] >= lo)
            & (df["raw_over_edge"] < hi)
        ].copy()

        if x.empty:
            continue

        x["profit"] = np.where(
            x["over_win"].eq(1),
            x["avg_over_odds"] - 1.0,
            -1.0,
        )

        bets = len(x)
        wins = int(x["over_win"].sum())
        profit = float(x["profit"].sum())

        print(
            f"{label:<10} "
            f"bets={bets:4d} | "
            f"W-L={wins:3d}-{bets-wins:<3d} | "
            f"WR={wins/bets:7.2%} | "
            f"odds={x['avg_over_odds'].mean():.3f} | "
            f"edge={x['raw_over_edge'].mean():+7.2%} | "
            f"profit={profit:+8.2f}u | "
            f"ROI={profit/bets:+7.2%}"
        )

    print()
    print("=" * 130)
    print("FROZEN 8% — BET COUNTS BY SEASON")
    print("=" * 130)

    bets = df[
        df["raw_over_edge"] >= FROZEN_THRESHOLD
    ].copy()

    print(
        bets.groupby("season")
        .size()
        .rename("bets")
        .to_string()
    )


if __name__ == "__main__":
    main()
