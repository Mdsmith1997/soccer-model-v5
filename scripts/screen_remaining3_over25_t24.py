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

LEAGUES = [
    "Segunda División",
    "Super Lig",
    "Swiss Super League",
]

THRESHOLDS = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.11,
    0.12,
    0.14,
    0.16,
]


def p_over_25(home_lambda, away_lambda):
    lam = float(home_lambda) + float(away_lambda)

    p_under = math.exp(-lam) * (
        1.0
        + lam
        + lam**2 / 2.0
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

    df = df.dropna(
        subset=numeric
    ).copy()

    # -----------------------------------------
    # RAW V5 OVER 2.5 PROBABILITY
    # -----------------------------------------

    df["raw_over_prob"] = [
        p_over_25(h, a)
        for h, a in zip(
            df["home_lambda"],
            df["away_lambda"],
        )
    ]

    # -----------------------------------------
    # DEVIG T24 TWO-WAY TOTALS MARKET
    # -----------------------------------------

    over_imp = 1.0 / df["avg_over_odds"]
    under_imp = 1.0 / df["avg_under_odds"]

    df["market_over_prob"] = (
        over_imp
        /
        (over_imp + under_imp)
    )

    # -----------------------------------------
    # RAW MODEL EDGE / EV
    # -----------------------------------------

    df["raw_over_edge"] = (
        df["raw_over_prob"]
        -
        df["market_over_prob"]
    )

    df["raw_over_ev"] = (
        df["raw_over_prob"]
        * df["avg_over_odds"]
        - 1.0
    )

    # -----------------------------------------
    # SETTLEMENT
    # -----------------------------------------

    df["over_win"] = (
        (
            df["home_goals"]
            + df["away_goals"]
        )
        > 2
    ).astype(int)

    df["profit"] = np.where(
        df["over_win"].eq(1),
        df["avg_over_odds"] - 1.0,
        -1.0,
    )

    return df


def performance(df, threshold):

    x = df[
        df["raw_over_edge"] >= threshold
    ].copy()

    if x.empty:
        return None

    bets = len(x)
    wins = int(x["over_win"].sum())
    profit = float(x["profit"].sum())

    return {
        "bets": bets,
        "wins": wins,
        "hit": wins / bets,
        "avg_odds": x["avg_over_odds"].mean(),
        "avg_edge": x["raw_over_edge"].mean(),
        "avg_ev": x["raw_over_ev"].mean(),
        "profit": profit,
        "roi": profit / bets,
    }


def threshold_sweep(league, df):

    print()
    print("=" * 110)
    print(league.upper())
    print("=" * 110)

    print("Usable matches:", len(df))
    print(
        "Seasons:",
        ", ".join(
            sorted(
                df["season"]
                .astype(str)
                .unique()
            )
        ),
    )

    print()
    print(
        f"{'THRESH':>8}"
        f"{'BETS':>8}"
        f"{'WINS':>8}"
        f"{'HIT%':>10}"
        f"{'ODDS':>9}"
        f"{'EDGE':>10}"
        f"{'EV':>10}"
        f"{'PROFIT':>11}"
        f"{'ROI':>10}"
    )

    for t in THRESHOLDS:

        r = performance(df, t)

        if r is None:
            continue

        print(
            f"{t:>8.0%}"
            f"{r['bets']:>8}"
            f"{r['wins']:>8}"
            f"{r['hit']:>10.2%}"
            f"{r['avg_odds']:>9.3f}"
            f"{r['avg_edge']:>10.2%}"
            f"{r['avg_ev']:>10.2%}"
            f"{r['profit']:>11.2f}"
            f"{r['roi']:>10.2%}"
        )


def season_stability(league, df):

    print()
    print("-" * 110)
    print(f"{league.upper()} — SEASON STABILITY")
    print("-" * 110)

    for t in THRESHOLDS:

        x = df[
            df["raw_over_edge"] >= t
        ].copy()

        if len(x) < 10:
            continue

        print()
        print(f"EDGE >= {t:.0%}")

        for season, s in x.groupby("season"):

            bets = len(s)

            if bets == 0:
                continue

            wins = int(s["over_win"].sum())
            profit = float(s["profit"].sum())

            print(
                f"  {str(season):<6}"
                f" bets={bets:<4}"
                f" wins={wins:<4}"
                f" hit={wins/bets:>7.2%}"
                f" profit={profit:>8.2f}"
                f" ROI={profit/bets:>8.2%}"
            )


def edge_buckets(league, df):

    print()
    print("-" * 110)
    print(f"{league.upper()} — NON-OVERLAPPING EDGE BUCKETS")
    print("-" * 110)

    buckets = [
        (-1.00, 0.00),
        (0.00, 0.02),
        (0.02, 0.04),
        (0.04, 0.06),
        (0.06, 0.08),
        (0.08, 0.10),
        (0.10, 0.12),
        (0.12, 0.14),
        (0.14, 0.16),
        (0.16, 1.00),
    ]

    for lo, hi in buckets:

        x = df[
            (df["raw_over_edge"] >= lo)
            &
            (df["raw_over_edge"] < hi)
        ].copy()

        if x.empty:
            continue

        bets = len(x)
        wins = int(x["over_win"].sum())
        profit = float(x["profit"].sum())

        print(
            f"{lo:>7.0%} to {hi:<7.0%}"
            f" bets={bets:<4}"
            f" hit={wins/bets:>7.2%}"
            f" edge={x['raw_over_edge'].mean():>7.2%}"
            f" odds={x['avg_over_odds'].mean():>6.3f}"
            f" profit={profit:>8.2f}"
            f" ROI={profit/bets:>8.2%}"
        )


def main():

    if not INPUT.exists():
        raise SystemExit(f"Missing input: {INPUT}")

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    df = prepare(df)

    print()
    print("=" * 110)
    print("SEGUNDA + SUPER LIG + SWISS — RAW V5 OVER 2.5 DISCOVERY")
    print("=" * 110)
    print("Rows:", len(df))

    for league in LEAGUES:

        x = df[
            df["league"].astype(str).eq(league)
        ].copy()

        threshold_sweep(
            league,
            x,
        )

        season_stability(
            league,
            x,
        )

        edge_buckets(
            league,
            x,
        )

    out = (
        ROOT
        / "data/processed"
        / "over25_remaining3_screened.csv"
    )

    df.to_csv(
        out,
        index=False,
    )

    print()
    print("=" * 110)
    print("SAVED")
    print("=" * 110)
    print(out)


if __name__ == "__main__":
    main()
