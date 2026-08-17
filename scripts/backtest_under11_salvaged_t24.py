from pathlib import Path
import pandas as pd
import numpy as np
import math

IN = Path(
    "data/processed/"
    "under11_salvaged_t24_odds.csv"
)

OUT = Path(
    "data/processed/"
    "under11_salvaged_t24_backtest.csv"
)

EDGE_THRESHOLD = 0.11

df = pd.read_csv(IN, low_memory=False)

# ============================================================
# CLEAN
# ============================================================

numeric = [
    "home_lambda",
    "away_lambda",
    "home_goals",
    "away_goals",
    "under_odds",
    "over_odds",
    "distance_from_t24",
]

for c in numeric:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce",
    )

df = df.dropna(
    subset=numeric
).copy()

# ============================================================
# MODEL UNDER 2.5 PROBABILITY
#
# If independent home/away goals are Poisson:
#
# total goals ~ Poisson(home_lambda + away_lambda)
#
# Under 2.5 wins on 0, 1 or 2 total goals.
# ============================================================

df["total_lambda"] = (
    df["home_lambda"]
    + df["away_lambda"]
)

def poisson_under25(lam):
    return (
        math.exp(-lam)
        * (
            1
            + lam
            + (lam ** 2) / 2
        )
    )

df["model_under25"] = (
    df["total_lambda"]
    .apply(poisson_under25)
)

# ============================================================
# MARKET IMPLIED PROBABILITIES
# ============================================================

df["under_implied"] = (
    1 / df["under_odds"]
)

df["over_implied"] = (
    1 / df["over_odds"]
)

df["market_hold"] = (
    df["under_implied"]
    + df["over_implied"]
)

# Remove vig by normalizing the two sides.
df["market_under_fair"] = (
    df["under_implied"]
    / df["market_hold"]
)

df["market_over_fair"] = (
    df["over_implied"]
    / df["market_hold"]
)

# ============================================================
# RAW MODEL EDGE
# ============================================================

df["raw_under_edge"] = (
    df["model_under25"]
    - df["market_under_fair"]
)

# ============================================================
# RESULT / P&L
# ============================================================

df["total_goals"] = (
    df["home_goals"]
    + df["away_goals"]
)

df["under25_win"] = (
    df["total_goals"] <= 2
)

# Flat 1-unit stake:
# win = odds - 1
# loss = -1

df["profit"] = np.where(
    df["under25_win"],
    df["under_odds"] - 1,
    -1.0,
)

df["qualifies_under11"] = (
    df["raw_under_edge"]
    >= EDGE_THRESHOLD
)

df.to_csv(
    OUT,
    index=False,
)

# ============================================================
# REPORTING
# ============================================================

def stats(x):
    bets = len(x)

    if bets == 0:
        return pd.Series({
            "bets": 0,
            "wins": 0,
            "win_rate": np.nan,
            "avg_odds": np.nan,
            "avg_edge": np.nan,
            "profit": 0.0,
            "roi": np.nan,
        })

    wins = int(
        x["under25_win"].sum()
    )

    return pd.Series({
        "bets": bets,
        "wins": wins,
        "win_rate": wins / bets,
        "avg_odds": x["under_odds"].mean(),
        "avg_edge": x["raw_under_edge"].mean(),
        "profit": x["profit"].sum(),
        "roi": x["profit"].sum() / bets,
    })


print("=" * 120)
print("RAW UNDER 2.5 >= 11% EDGE — SALVAGED T-24 BACKTEST")
print("=" * 120)

print()
print("Usable exact-2.5 games:", len(df))
print("Fixed edge threshold:", EDGE_THRESHOLD)

# ============================================================
# TIMING ROBUSTNESS
# ============================================================

print()
print("=" * 120)
print("TIMING ROBUSTNESS")
print("=" * 120)

timing_rows = []

for tol in [4, 6, 8, 12]:

    sample = df[
        df["distance_from_t24"] <= tol
    ].copy()

    bets = sample[
        sample["qualifies_under11"]
    ].copy()

    s = stats(bets)

    timing_rows.append({
        "window": f"T24 ±{tol}h",
        "sample_games": len(sample),
        **s.to_dict(),
    })

timing = pd.DataFrame(timing_rows)

print(
    timing.to_string(
        index=False,
        formatters={
            "win_rate": lambda x: (
                f"{x:.2%}"
                if pd.notna(x)
                else "-"
            ),
            "avg_odds": lambda x: (
                f"{x:.3f}"
                if pd.notna(x)
                else "-"
            ),
            "avg_edge": lambda x: (
                f"{x:.2%}"
                if pd.notna(x)
                else "-"
            ),
            "profit": lambda x: f"{x:+.2f}",
            "roi": lambda x: (
                f"{x:+.2%}"
                if pd.notna(x)
                else "-"
            ),
        }
    )
)

# ============================================================
# PRIMARY TEST = ±6 HOURS
# ============================================================

primary = df[
    df["distance_from_t24"] <= 6
].copy()

bets = primary[
    primary["qualifies_under11"]
].copy()

print()
print("=" * 120)
print("PRIMARY TEST — T-24 ±6 HOURS")
print("=" * 120)

print()
print("Sample games:", len(primary))
print("Qualifying bets:", len(bets))
print(
    "Bet frequency:",
    f"{len(bets) / len(primary):.2%}"
    if len(primary)
    else "-"
)

print()
print("OVERALL")

print(
    stats(bets)
    .to_frame("value")
    .to_string()
)

# ============================================================
# LEAGUE
# ============================================================

print()
print("=" * 120)
print("BY LEAGUE — PRIMARY ±6H")
print("=" * 120)

league = (
    bets
    .groupby("league")
    .apply(
        stats,
        include_groups=False,
    )
)

print(league.to_string())

# ============================================================
# LEAGUE / SEASON
# ============================================================

print()
print("=" * 120)
print("BY LEAGUE / SEASON — PRIMARY ±6H")
print("=" * 120)

league_season = (
    bets
    .groupby(
        ["league", "season"]
    )
    .apply(
        stats,
        include_groups=False,
    )
)

print(league_season.to_string())

# ============================================================
# ALL SEASONS
#
# Include seasons with zero qualifying bets so we don't
# accidentally hide a season.
# ============================================================

print()
print("=" * 120)
print("SEASON COVERAGE — PRIMARY ±6H")
print("=" * 120)

coverage = (
    primary
    .groupby(
        ["league", "season"]
    )
    .agg(
        sample_games=(
            "event_id",
            "size",
        ),
        qualifying_bets=(
            "qualifies_under11",
            "sum",
        ),
    )
)

print(coverage.to_string())

# ============================================================
# EDGE BUCKETS
#
# Diagnostic only. DO NOT use this to change the threshold yet.
# ============================================================

print()
print("=" * 120)
print("EDGE BUCKET DIAGNOSTIC — PRIMARY ±6H")
print("=" * 120)

bins = [
    -999,
    0,
    .05,
    .08,
    .11,
    .14,
    .17,
    .20,
    999,
]

labels = [
    "<0%",
    "0-5%",
    "5-8%",
    "8-11%",
    "11-14%",
    "14-17%",
    "17-20%",
    "20%+",
]

primary["edge_bucket"] = pd.cut(
    primary["raw_under_edge"],
    bins=bins,
    labels=labels,
    right=False,
)

bucket = (
    primary
    .groupby(
        "edge_bucket",
        observed=True,
    )
    .apply(
        stats,
        include_groups=False,
    )
)

print(bucket.to_string())

print()
print("Saved:")
print(OUT)
