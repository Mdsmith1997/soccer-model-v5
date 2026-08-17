from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    (
        "Primeira Liga",
        ROOT / "data/processed/under11_wave1_t24_games.csv",
    ),
    (
        "Serie A",
        ROOT / "data/processed/serie_a_under11_t24_games.csv",
    ),
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


def p_under_25(home_lambda, away_lambda):
    lam = float(home_lambda) + float(away_lambda)

    return math.exp(-lam) * (
        1.0
        + lam
        + lam ** 2 / 2.0
    )


def p_over_25(home_lambda, away_lambda):
    return 1.0 - p_under_25(
        home_lambda,
        away_lambda,
    )


def prepare(league, path):

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    df = df[
        df["league"].astype(str).eq(league)
    ].copy()

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
    # EXACT SAME RAW V5 OVER 2.5 PROBABILITY
    # -----------------------------------------

    df["raw_over_prob"] = [
        p_over_25(h, a)
        for h, a in zip(
            df["home_lambda"],
            df["away_lambda"],
        )
    ]

    # -----------------------------------------
    # DEVIG TWO-WAY T24 TOTALS MARKET
    # -----------------------------------------

    over_imp = 1.0 / df["avg_over_odds"]
    under_imp = 1.0 / df["avg_under_odds"]

    df["market_over_prob"] = (
        over_imp
        /
        (over_imp + under_imp)
    )

    # -----------------------------------------
    # RAW MODEL EDGE
    # -----------------------------------------

    df["raw_over_edge"] = (
        df["raw_over_prob"]
        -
        df["market_over_prob"]
    )

    # EV at actual T24 average Over price
    df["raw_over_ev"] = (
        df["raw_over_prob"]
        *
        df["avg_over_odds"]
        - 1.0
    )

    df["over_win"] = (
        (
            df["home_goals"]
            +
            df["away_goals"]
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
    wins = int(x["over_win"].sum())
    profit = float(x["profit"].sum())

    return {
        "bets": bets,
        "wins": wins,
        "win_rate": wins / bets,
        "avg_odds": x["avg_over_odds"].mean(),
        "avg_edge": x["raw_over_edge"].mean(),
        "avg_ev": x["raw_over_ev"].mean(),
        "profit": profit,
        "roi": profit / bets,
    }


def print_threshold_sweep(league, df):

    print()
    print("=" * 105)
    print(league.upper())
    print("=" * 105)

    print("Usable T24 matches:", len(df))
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
        f"{'WIN%':>10}"
        f"{'AVG ODDS':>11}"
        f"{'AVG EDGE':>11}"
        f"{'AVG EV':>11}"
        f"{'PROFIT':>11}"
        f"{'ROI':>10}"
    )

    for threshold in THRESHOLDS:

        r = performance(
            df,
            threshold,
        )

        print(
            f"{threshold:>8.0%}"
            f"{r['bets']:>8}"
            f"{r['wins']:>8}"
            f"{r['win_rate']:>10.2%}"
            f"{r['avg_odds']:>11.3f}"
            f"{r['avg_edge']:>11.2%}"
            f"{r['avg_ev']:>11.2%}"
            f"{r['profit']:>11.2f}"
            f"{r['roi']:>10.2%}"
        )


def print_season_sweep(league, df):

    print()
    print("-" * 105)
    print(f"{league.upper()} — SEASON STABILITY")
    print("-" * 105)

    for threshold in THRESHOLDS:

        x = df[
            df["raw_over_edge"] >= threshold
        ].copy()

        if len(x) < 10:
            continue

        print()
        print(f"EDGE >= {threshold:.0%}")

        for season, s in x.groupby("season"):

            bets = len(s)

            if bets == 0:
                continue

            wins = int(s["over_win"].sum())
            profit = float(s["profit"].sum())
            roi = profit / bets

            print(
                f"  {str(season):<6}"
                f" bets={bets:<4}"
                f" wins={wins:<4}"
                f" hit={wins/bets:>7.2%}"
                f" profit={profit:>8.2f}"
                f" ROI={roi:>8.2%}"
            )


def print_edge_buckets(league, df):

    print()
    print("-" * 105)
    print(f"{league.upper()} — NON-OVERLAPPING EDGE BUCKETS")
    print("-" * 105)

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
            f" avg_edge={x['raw_over_edge'].mean():>7.2%}"
            f" avg_odds={x['avg_over_odds'].mean():>6.3f}"
            f" profit={profit:>8.2f}"
            f" ROI={profit/bets:>8.2%}"
        )


def main():

    print()
    print("=" * 105)
    print("PRIMEIRA LIGA + SERIE A — OVER 2.5 T24 DISCOVERY")
    print("=" * 105)

    all_frames = []

    for league, path in FILES:

        if not path.exists():
            print("MISSING:", path)
            continue

        df = prepare(
            league,
            path,
        )

        all_frames.append(df)

        print_threshold_sweep(
            league,
            df,
        )

        print_season_sweep(
            league,
            df,
        )

        print_edge_buckets(
            league,
            df,
        )

    if all_frames:

        combined = pd.concat(
            all_frames,
            ignore_index=True,
        )

        out = (
            ROOT
            / "data/processed"
            / "over25_primeira_seriea_t24.csv"
        )

        combined.to_csv(
            out,
            index=False,
        )

        print()
        print("=" * 105)
        print("SAVED")
        print("=" * 105)
        print(out)
        print("Rows:", len(combined))


if __name__ == "__main__":
    main()
