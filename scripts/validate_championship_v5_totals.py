from pathlib import Path
import math
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.test_v5_totals_oos_expansion import (
    load_predictions,
    load_market,
    p_under_2_5,
)

LEAGUE = "Championship"

# Frozen before this dedicated stability analysis.
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
    pred = pred[pred["league"].eq(LEAGUE)].copy()

    market = load_market()
    market = market[market["league"].eq(LEAGUE)].copy()

    print("=" * 130)
    print("INPUT / MATCHING AUDIT")
    print("=" * 130)

    print("V5 Championship prediction rows:", len(pred))
    print("Historical Championship market rows:", len(market))

    pred_keys = pred[
        ["season", "date", "home_key", "away_key"]
    ].drop_duplicates()

    market_keys = market[
        ["season", "date", "home_key", "away_key"]
    ].drop_duplicates()

    merged = pred.merge(
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
        suffixes=("", "_market"),
    )

    print("Matched rows:", len(merged))
    print(
        "Prediction match rate:",
        f"{len(merged) / len(pred):.2%}" if len(pred) else "N/A",
    )
    print(
        "Market match rate:",
        f"{len(merged) / len(market):.2%}" if len(market) else "N/A",
    )

    # Season-level matching audit
    pred_by = pred.groupby("season").size()
    market_by = market.groupby("season").size()
    match_by = merged.groupby("season").size()

    audit = pd.concat(
        [
            pred_by.rename("pred_rows"),
            market_by.rename("market_rows"),
            match_by.rename("matched"),
        ],
        axis=1,
    ).fillna(0)

    audit["pred_match_pct"] = np.where(
        audit["pred_rows"] > 0,
        audit["matched"] / audit["pred_rows"] * 100,
        np.nan,
    )

    audit["market_match_pct"] = np.where(
        audit["market_rows"] > 0,
        audit["matched"] / audit["market_rows"] * 100,
        np.nan,
    )

    print("\nBY SEASON MATCHING")
    print(audit.to_string())

    df = merged[
        merged["over_odds"].gt(1)
        & merged["under_odds"].gt(1)
    ].copy()

    df["model_p_under"] = [
        p_under_2_5(h, a)
        for h, a in zip(
            df["home_lambda"],
            df["away_lambda"],
        )
    ]

    raw_over = 1.0 / df["over_odds"]
    raw_under = 1.0 / df["under_odds"]

    vig_sum = raw_over + raw_under

    df["market_p_under"] = raw_under / vig_sum

    df["under_edge"] = (
        df["model_p_under"]
        - df["market_p_under"]
    )

    df["under_ev"] = (
        df["model_p_under"]
        * df["under_odds"]
        - 1.0
    )

    df["actual_total"] = (
        df["home_goals"]
        + df["away_goals"]
    )

    df["won"] = (
        df["actual_total"] < 2.5
    ).astype(int)

    df["qualifies"] = (
        df["under_edge"] >= THRESHOLD
    )

    df["profit"] = np.where(
        df["won"].eq(1),
        df["under_odds"] - 1.0,
        -1.0,
    )

    return df


def performance(df):
    x = df[df["qualifies"]].copy()

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
    wins = int(x["won"].sum())
    profit = float(x["profit"].sum())

    return {
        "bets": bets,
        "wins": wins,
        "win_rate": wins / bets,
        "avg_odds": x["under_odds"].mean(),
        "avg_edge": x["under_edge"].mean(),
        "avg_ev": x["under_ev"].mean(),
        "profit": profit,
        "roi": profit / bets,
    }


def print_result(label, p):
    if not p["bets"]:
        print(f"{label:<30} no qualifying bets")
        return

    print(
        f"{label:<30} "
        f"bets={p['bets']:4d}  "
        f"wins={p['wins']:4d}  "
        f"WR={p['win_rate']:7.2%}  "
        f"odds={p['avg_odds']:.3f}  "
        f"edge={p['avg_edge']:+7.2%}  "
        f"EV={p['avg_ev']:+7.2%}  "
        f"profit={p['profit']:+8.2f}u  "
        f"ROI={p['roi']:+7.2%}"
    )


def bootstrap_roi(df, iterations=50000, seed=42):
    x = df[df["qualifies"]].copy()

    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.nan

    profits = x["profit"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)

    sampled = rng.choice(
        profits,
        size=(iterations, len(profits)),
        replace=True,
    )

    rois = sampled.mean(axis=1)

    return (
        float(np.median(rois)),
        float(np.quantile(rois, 0.025)),
        float(np.quantile(rois, 0.975)),
        float(np.mean(rois > 0)),
    )


def drawdown_stats(df):
    x = (
        df[df["qualifies"]]
        .sort_values("date")
        .copy()
    )

    if x.empty:
        return np.nan, 0, 0

    equity = x["profit"].cumsum()
    running_peak = equity.cummax()

    dd = equity - running_peak
    max_dd = float(dd.min())

    # longest losing streak
    longest = 0
    current = 0

    for won in x["won"]:
        if won == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return max_dd, longest, len(x)


def team_concentration(df):
    x = df[df["qualifies"]].copy()

    if x.empty:
        return

    rows = []

    for _, r in x.iterrows():
        rows.append({
            "team": r["home_team"],
            "profit": r["profit"],
        })
        rows.append({
            "team": r["away_team"],
            "profit": r["profit"],
        })

    t = pd.DataFrame(rows)

    out = (
        t.groupby("team")
        .agg(
            appearances=("team", "size"),
            attributed_profit=("profit", "sum"),
        )
        .sort_values(
            ["appearances", "attributed_profit"],
            ascending=[False, False],
        )
    )

    print(out.head(20).to_string())


def odds_buckets(df):
    x = df[df["qualifies"]].copy()

    if x.empty:
        return

    bins = [1.0, 1.75, 2.0, 2.25, 2.5, 3.0, np.inf]
    labels = [
        "1.00-1.74",
        "1.75-1.99",
        "2.00-2.24",
        "2.25-2.49",
        "2.50-2.99",
        "3.00+",
    ]

    x["odds_bucket"] = pd.cut(
        x["under_odds"],
        bins=bins,
        labels=labels,
        right=False,
    )

    rows = []

    for b, g in x.groupby(
        "odds_bucket",
        observed=True,
    ):
        rows.append({
            "bucket": str(b),
            "bets": len(g),
            "wins": int(g["won"].sum()),
            "profit": g["profit"].sum(),
            "roi": g["profit"].mean(),
        })

    out = pd.DataFrame(rows)

    if len(out):
        out["roi"] = out["roi"] * 100
        print(out.to_string(index=False))


def threshold_diagnostic(df):
    print(
        "\nDIAGNOSTIC ONLY — "
        "11% REMAINS THE FROZEN DECISION RULE"
    )

    for t in [
        .08,
        .09,
        .10,
        .11,
        .12,
        .13,
        .14,
        .15,
    ]:
        x = df[df["under_edge"] >= t].copy()

        if x.empty:
            print(f">= {t:.0%}: no bets")
            continue

        profit = x["profit"].sum()

        print(
            f">= {t:.0%}  "
            f"bets={len(x):3d}  "
            f"W-L={int(x['won'].sum())}-"
            f"{len(x)-int(x['won'].sum())}  "
            f"profit={profit:+7.2f}u  "
            f"ROI={profit/len(x):+7.2%}"
        )


def main():
    df = build_dataset()

    print("\n" + "=" * 130)
    print("CHAMPIONSHIP — FROZEN 11% STABILITY VALIDATION")
    print("=" * 130)
    print("Rule: RAW V5 UNDER 2.5 EDGE >= 11%")
    print("Threshold is frozen. No Championship tuning.")

    print("\n" + "=" * 130)
    print("FULL RESULT")
    print("=" * 130)

    print_result(
        "ALL SEASONS",
        performance(df),
    )

    print("\n" + "=" * 130)
    print("CHRONOLOGICAL STABILITY")
    print("=" * 130)

    print_result(
        "EARLY 1920-2223",
        performance(
            df[df["season"].isin(EARLY)]
        ),
    )

    print_result(
        "RECENT 2324-2526",
        performance(
            df[df["season"].isin(RECENT)]
        ),
    )

    print("\n" + "=" * 130)
    print("SEASON-BY-SEASON")
    print("=" * 130)

    for season in SEASONS:
        print_result(
            season,
            performance(
                df[df["season"].eq(season)]
            ),
        )

    print("\n" + "=" * 130)
    print("LEAVE-ONE-SEASON-OUT")
    print("=" * 130)

    for season in SEASONS:
        print_result(
            f"EXCLUDE {season}",
            performance(
                df[~df["season"].eq(season)]
            ),
        )

    print("\n" + "=" * 130)
    print("BOOTSTRAP ROI UNCERTAINTY")
    print("=" * 130)

    med, lo, hi, ppos = bootstrap_roi(df)

    print(f"Median bootstrap ROI: {med:+.2%}")
    print(
        f"95% bootstrap interval: "
        f"{lo:+.2%} to {hi:+.2%}"
    )
    print(
        f"P(ROI > 0): {ppos:.2%}"
    )

    print("\n" + "=" * 130)
    print("DRAWDOWN / LOSING STREAK")
    print("=" * 130)

    max_dd, longest, n = drawdown_stats(df)

    print("Bets:", n)
    print(f"Maximum drawdown: {max_dd:+.2f}u")
    print("Longest losing streak:", longest)

    print("\n" + "=" * 130)
    print("ODDS CONCENTRATION")
    print("=" * 130)

    odds_buckets(df)

    print("\n" + "=" * 130)
    print("TEAM CONCENTRATION")
    print("=" * 130)

    team_concentration(df)

    print("\n" + "=" * 130)
    print("EDGE THRESHOLD ROBUSTNESS")
    print("=" * 130)

    threshold_diagnostic(df)

    print("\n" + "=" * 130)
    print("QUALIFYING BETS")
    print("=" * 130)

    cols = [
        "season",
        "date",
        "home_team",
        "away_team",
        "model_p_under",
        "market_p_under",
        "under_edge",
        "under_odds",
        "actual_total",
        "won",
        "profit",
    ]

    q = (
        df[df["qualifies"]]
        .sort_values("date")
    )

    if len(q):
        print(q[cols].to_string(index=False))
    else:
        print("NONE")


if __name__ == "__main__":
    main()
