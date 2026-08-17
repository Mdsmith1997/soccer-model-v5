from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_all_leagues_market_oos.csv"
)

THRESHOLD = 0.06
N_BOOT = 50000
SEED = 42


def brier(y, p):
    return np.mean((p - y) ** 2)


def logloss(y, p):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)

    return -np.mean(
        y * np.log(p)
        + (1 - y) * np.log(1 - p)
    )


def wilson_interval(wins, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)

    phat = wins / n

    denom = 1 + z**2 / n

    center = (
        phat + z**2 / (2 * n)
    ) / denom

    margin = (
        z
        * math.sqrt(
            phat * (1 - phat) / n
            + z**2 / (4 * n**2)
        )
        / denom
    )

    return (
        center - margin,
        center + margin,
    )


df = pd.read_csv(
    FILE,
    low_memory=False,
)

swiss = df[
    df["audit_league"]
    .astype(str)
    .eq("Swiss Super League")
].copy()

for c in [
    "actual_yes",
    "p_raw",
    "p_yes_cal",
    "market_yes_nv",
    "odds_btts_yes",
    "edge_yes",
    "season_num",
]:
    swiss[c] = pd.to_numeric(
        swiss[c],
        errors="coerce",
    )

swiss = swiss[
    swiss[
        [
            "actual_yes",
            "p_raw",
            "p_yes_cal",
            "market_yes_nv",
            "odds_btts_yes",
            "edge_yes",
        ]
    ]
    .notna()
    .all(axis=1)
].copy()

swiss["actual_yes"] = (
    swiss["actual_yes"]
    .astype(int)
)

swiss["profit_yes"] = np.where(
    swiss["actual_yes"].eq(1),
    swiss["odds_btts_yes"] - 1,
    -1.0,
)

bets = swiss[
    swiss["edge_yes"] >= THRESHOLD
].copy()


print("=" * 125)
print("SWISS SUPER LEAGUE BTTS YES >=6% — FINAL VALIDATION")
print("=" * 125)

print()
print("All OOS Swiss games:", len(swiss))
print("Qualifying bets:", len(bets))
print(
    "Qualifying rate:",
    f"{len(bets) / len(swiss):.2%}"
)

print()
print("=" * 125)
print("1. MODEL / MARKET CALIBRATION")
print("=" * 125)

print(
    "Actual BTTS YES rate:",
    f"{swiss['actual_yes'].mean():.2%}",
)

print(
    "Raw model mean:",
    f"{swiss['p_raw'].mean():.2%}",
)

print(
    "Calibrated model mean:",
    f"{swiss['p_yes_cal'].mean():.2%}",
)

print(
    "Market no-vig mean:",
    f"{swiss['market_yes_nv'].mean():.2%}",
)

print()
print(
    "Raw Brier:",
    f"{brier(swiss['actual_yes'], swiss['p_raw']):.5f}",
)

print(
    "Calibrated Brier:",
    f"{brier(swiss['actual_yes'], swiss['p_yes_cal']):.5f}",
)

print(
    "Market Brier:",
    f"{brier(swiss['actual_yes'], swiss['market_yes_nv']):.5f}",
)

print()
print(
    "Raw Log Loss:",
    f"{logloss(swiss['actual_yes'], swiss['p_raw']):.5f}",
)

print(
    "Calibrated Log Loss:",
    f"{logloss(swiss['actual_yes'], swiss['p_yes_cal']):.5f}",
)

print(
    "Market Log Loss:",
    f"{logloss(swiss['actual_yes'], swiss['market_yes_nv']):.5f}",
)


print()
print("=" * 125)
print("2. CALIBRATION BINS")
print("=" * 125)

swiss["cal_bin"] = pd.cut(
    swiss["p_yes_cal"],
    bins=np.linspace(0, 1, 11),
    include_lowest=True,
)

cal = (
    swiss
    .groupby(
        "cal_bin",
        observed=True,
    )
    .agg(
        games=("actual_yes", "size"),
        model_p=("p_yes_cal", "mean"),
        market_p=("market_yes_nv", "mean"),
        actual=("actual_yes", "mean"),
    )
    .reset_index()
)

print(
    cal.to_string(
        index=False,
        formatters={
            "model_p": lambda x: f"{x:.2%}",
            "market_p": lambda x: f"{x:.2%}",
            "actual": lambda x: f"{x:.2%}",
        },
    )
)


print()
print("=" * 125)
print("3. EXACT EDGE BUCKET REALIZATION")
print("=" * 125)

edge_bins = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.14,
    np.inf,
]

rows = []

for lo, hi in zip(
    edge_bins[:-1],
    edge_bins[1:],
):
    if np.isinf(hi):
        g = swiss[
            swiss["edge_yes"] >= lo
        ]
        label = f"{lo:.0%}+"
    else:
        g = swiss[
            (swiss["edge_yes"] >= lo)
            & (swiss["edge_yes"] < hi)
        ]
        label = f"{lo:.0%}-{hi:.0%}"

    if g.empty:
        continue

    rows.append({
        "bucket": label,
        "bets": len(g),
        "avg_model": g["p_yes_cal"].mean(),
        "avg_market": g["market_yes_nv"].mean(),
        "avg_edge": g["edge_yes"].mean(),
        "actual": g["actual_yes"].mean(),
        "avg_odds": g["odds_btts_yes"].mean(),
        "profit": g["profit_yes"].sum(),
        "roi": g["profit_yes"].mean(),
    })

edge_df = pd.DataFrame(rows)

print(
    edge_df.to_string(
        index=False,
        formatters={
            "avg_model": lambda x: f"{x:.2%}",
            "avg_market": lambda x: f"{x:.2%}",
            "avg_edge": lambda x: f"{x:.2%}",
            "actual": lambda x: f"{x:.2%}",
            "avg_odds": lambda x: f"{x:.3f}",
            "profit": lambda x: f"{x:+.2f}u",
            "roi": lambda x: f"{x:+.2%}",
        },
    )
)


print()
print("=" * 125)
print("4. QUALIFYING BET STATISTICS")
print("=" * 125)

wins = int(
    bets["actual_yes"].sum()
)

n = len(bets)

win_rate = wins / n

avg_odds = (
    bets["odds_btts_yes"].mean()
)

break_even = (
    (1 / bets["odds_btts_yes"])
    .mean()
)

profit = (
    bets["profit_yes"].sum()
)

roi = (
    bets["profit_yes"].mean()
)

lo, hi = wilson_interval(
    wins,
    n,
)

print("Bets:", n)
print("Wins:", wins)
print(
    "Observed win rate:",
    f"{win_rate:.2%}",
)

print(
    "Wilson 95% win-rate CI:",
    f"{lo:.2%} to {hi:.2%}",
)

print(
    "Average odds:",
    f"{avg_odds:.3f}",
)

print(
    "Average raw break-even probability:",
    f"{break_even:.2%}",
)

print(
    "Average calibrated model probability:",
    f"{bets['p_yes_cal'].mean():.2%}",
)

print(
    "Average market no-vig probability:",
    f"{bets['market_yes_nv'].mean():.2%}",
)

print(
    "Average model edge:",
    f"{bets['edge_yes'].mean():.2%}",
)

print(
    "Profit:",
    f"{profit:+.2f}u",
)

print(
    "ROI:",
    f"{roi:+.2%}",
)


print()
print("=" * 125)
print("5. BOOTSTRAP ROI")
print("=" * 125)

rng = np.random.default_rng(SEED)

profits = (
    bets["profit_yes"]
    .to_numpy(dtype=float)
)

boot = np.empty(
    N_BOOT,
    dtype=float,
)

for start in range(
    0,
    N_BOOT,
    1000,
):
    k = min(
        1000,
        N_BOOT - start,
    )

    idx = rng.integers(
        0,
        len(profits),
        size=(k, len(profits)),
    )

    boot[
        start:start + k
    ] = profits[idx].mean(axis=1)

p_positive = (
    boot > 0
).mean()

p2_5, p5, p50, p95, p97_5 = (
    np.percentile(
        boot,
        [
            2.5,
            5,
            50,
            95,
            97.5,
        ],
    )
)

print(
    "P(ROI > 0):",
    f"{p_positive:.2%}",
)

print(
    "95% bootstrap ROI interval:",
    f"{p2_5:+.2%} to {p97_5:+.2%}",
)

print(
    "5th percentile:",
    f"{p5:+.2%}",
)

print(
    "Median:",
    f"{p50:+.2%}",
)

print(
    "95th percentile:",
    f"{p95:+.2%}",
)


print()
print("=" * 125)
print("6. SEASON-BY-SEASON EDGE REALIZATION")
print("=" * 125)

season_rows = []

for season, g in bets.groupby(
    "season_num"
):
    season_rows.append({
        "season": int(season),
        "bets": len(g),
        "avg_model": g["p_yes_cal"].mean(),
        "avg_market": g["market_yes_nv"].mean(),
        "avg_edge": g["edge_yes"].mean(),
        "actual": g["actual_yes"].mean(),
        "profit": g["profit_yes"].sum(),
        "roi": g["profit_yes"].mean(),
    })

season_df = pd.DataFrame(
    season_rows
)

print(
    season_df.to_string(
        index=False,
        formatters={
            "avg_model": lambda x: f"{x:.2%}",
            "avg_market": lambda x: f"{x:.2%}",
            "avg_edge": lambda x: f"{x:.2%}",
            "actual": lambda x: f"{x:.2%}",
            "profit": lambda x: f"{x:+.2f}u",
            "roi": lambda x: f"{x:+.2%}",
        },
    )
)


print()
print("=" * 125)
print("7. LEAVE-ONE-SEASON-OUT")
print("=" * 125)

for season in sorted(
    bets["season_num"]
    .dropna()
    .unique()
):
    g = bets[
        ~bets["season_num"].eq(
            season
        )
    ]

    print(
        f"Remove {int(season)} | "
        f"bets={len(g):3d} | "
        f"profit={g['profit_yes'].sum():+7.2f}u | "
        f"ROI={g['profit_yes'].mean():+7.2%}"
    )


print()
print("=" * 125)
print("8. MARKET PRICE / ODDS QUALITY")
print("=" * 125)

for lo_odds, hi_odds in [
    (1.50, 1.65),
    (1.65, 1.75),
    (1.75, 1.85),
    (1.85, 2.00),
    (2.00, np.inf),
]:
    if np.isinf(hi_odds):
        g = bets[
            bets[
                "odds_btts_yes"
            ] >= lo_odds
        ]
        label = f"{lo_odds:.2f}+"
    else:
        g = bets[
            (
                bets["odds_btts_yes"]
                >= lo_odds
            )
            &
            (
                bets["odds_btts_yes"]
                < hi_odds
            )
        ]
        label = (
            f"{lo_odds:.2f}-"
            f"{hi_odds:.2f}"
        )

    if g.empty:
        continue

    print(
        f"{label:<12} | "
        f"bets={len(g):3d} | "
        f"avg edge={g['edge_yes'].mean():6.2%} | "
        f"actual={g['actual_yes'].mean():6.2%} | "
        f"profit={g['profit_yes'].sum():+7.2f}u | "
        f"ROI={g['profit_yes'].mean():+7.2%}"
    )


print()
print("=" * 125)
print("FINAL VALIDATION COMPLETE")
print("=" * 125)
