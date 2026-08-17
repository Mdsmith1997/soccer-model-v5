from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PATH = (
    ROOT
    / "data/processed"
    / "superlig_v5_t24_totals_games.csv"
)

THRESHOLDS = [
    0.00, 0.02, 0.04, 0.06, 0.08,
    0.10, 0.11, 0.12, 0.13, 0.14,
    0.15, 0.16, 0.18, 0.20,
]


def p_over_25(home_lambda, away_lambda):
    mu = float(home_lambda) + float(away_lambda)

    p_under = math.exp(-mu) * (
        1.0
        + mu
        + mu**2 / 2.0
    )

    return 1.0 - p_under


def prepare(df):
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

    df = df.dropna(subset=numeric).copy()

    df = df[
        df["avg_over_odds"].gt(1)
        & df["avg_under_odds"].gt(1)
    ].copy()

    df["season"] = df["season"].astype(str)

    df["raw_over_prob"] = [
        p_over_25(h, a)
        for h, a in zip(
            df["home_lambda"],
            df["away_lambda"],
        )
    ]

    over_imp = 1.0 / df["avg_over_odds"]
    under_imp = 1.0 / df["avg_under_odds"]

    df["market_over_prob"] = (
        over_imp / (over_imp + under_imp)
    )

    df["raw_over_edge"] = (
        df["raw_over_prob"]
        - df["market_over_prob"]
    )

    df["actual_total"] = (
        df["home_goals"]
        + df["away_goals"]
    )

    df["over_win"] = (
        df["actual_total"] > 2.5
    ).astype(int)

    return df


def performance(df, threshold):
    x = df[
        df["raw_over_edge"] >= threshold
    ].copy()

    if x.empty:
        return None

    x["profit"] = np.where(
        x["over_win"].eq(1),
        x["avg_over_odds"] - 1.0,
        -1.0,
    )

    bets = len(x)
    wins = int(x["over_win"].sum())
    profit = float(x["profit"].sum())

    return {
        "threshold": threshold,
        "bets": bets,
        "wins": wins,
        "losses": bets - wins,
        "wr": wins / bets,
        "odds": x["avg_over_odds"].mean(),
        "edge": x["raw_over_edge"].mean(),
        "profit": profit,
        "roi": profit / bets,
    }


def show(label, p):
    if p is None:
        print(f"{label:<10} no bets")
        return

    print(
        f"{label:<10} "
        f"bets={p['bets']:4d} | "
        f"W-L={p['wins']:3d}-{p['losses']:<3d} | "
        f"WR={p['wr']:7.2%} | "
        f"odds={p['odds']:.3f} | "
        f"edge={p['edge']:+7.2%} | "
        f"profit={p['profit']:+8.2f}u | "
        f"ROI={p['roi']:+7.2%}"
    )


def main():
    df = prepare(
        pd.read_csv(PATH, low_memory=False)
    )

    print()
    print("=" * 125)
    print("TURKEY SUPER LIG — O2.5 SPECIALIST DISCOVERY")
    print("=" * 125)

    print(f"\nUsable matches: {len(df):,}")

    print("\nMatches by season:")
    print(
        df.groupby("season")
        .size()
        .to_string()
    )

    print()
    print("=" * 125)
    print("CUMULATIVE EDGE THRESHOLDS — BOTH SEASONS")
    print("=" * 125)

    for t in THRESHOLDS:
        show(
            f">={t:.0%}",
            performance(df, t),
        )

    print()
    print("=" * 125)
    print("SEASON-BY-SEASON")
    print("=" * 125)

    for season in sorted(df["season"].unique()):
        print(f"\n--- {season} ---")

        x = df[df["season"].eq(season)]

        for t in THRESHOLDS:
            show(
                f">={t:.0%}",
                performance(x, t),
            )

    print()
    print("=" * 125)
    print("INDEPENDENT EDGE BANDS")
    print("=" * 125)

    bands = [
        (0.00, 0.02),
        (0.02, 0.04),
        (0.04, 0.06),
        (0.06, 0.08),
        (0.08, 0.10),
        (0.10, 0.12),
        (0.12, 0.14),
        (0.14, 0.16),
        (0.16, 0.18),
        (0.18, 0.20),
        (0.20, np.inf),
    ]

    for lo, hi in bands:
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

        label = (
            f"{lo:.0%}+"
            if np.isinf(hi)
            else f"{lo:.0%}-{hi:.0%}"
        )

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


if __name__ == "__main__":
    main()
