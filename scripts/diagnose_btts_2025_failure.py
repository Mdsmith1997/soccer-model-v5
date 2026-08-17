from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "btts_cfg0755_market_matched.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "btts_2025_failure_diag"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# BET_036028 RULE
# ============================================================

EDGE_MIN = 0.06
EV_MIN = 0.00

ODDS_MIN = 1.00
ODDS_MAX = 2.50

LAMBDA_MIN_FLOOR = 0.90
LAMBDA_MIN_CEILING = 1.20

LEAGUES = [
    "MLS",
    "Eliteserien",
]


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 120)
print("BTTS 2025 FAILURE DIAGNOSTIC")
print("RULE: BET_036028")
print("=" * 120)

df = pd.read_csv(
    INPUT,
    low_memory=False,
)


numeric_cols = [
    "test_year",
    "btts_yes",
    "champion_yes",
    "market_yes",
    "odds_yes",
    "champion_edge_yes",
    "champion_ev_yes",
    "home_lambda",
    "away_lambda",
    "lambda_min",
    "lambda_total",
    "lambda_gap",
]


for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )


if "lambda_min" not in df.columns:
    df["lambda_min"] = np.minimum(
        df["home_lambda"],
        df["away_lambda"],
    )

if "lambda_total" not in df.columns:
    df["lambda_total"] = (
        df["home_lambda"]
        +
        df["away_lambda"]
    )

if "lambda_gap" not in df.columns:
    df["lambda_gap"] = np.abs(
        df["home_lambda"]
        -
        df["away_lambda"]
    )


# ============================================================
# APPLY BET_036028
# ============================================================

bets = df[
    df["league"].isin(
        LEAGUES
    )
].copy()

bets = bets[
    bets["champion_edge_yes"]
    >= EDGE_MIN
]

bets = bets[
    bets["champion_ev_yes"]
    >= EV_MIN
]

bets = bets[
    bets["odds_yes"]
    >= ODDS_MIN
]

bets = bets[
    bets["odds_yes"]
    <
    ODDS_MAX
]

bets = bets[
    bets["lambda_min"]
    >= LAMBDA_MIN_FLOOR
]

bets = bets[
    bets["lambda_min"]
    <
    LAMBDA_MIN_CEILING
]


bets["won"] = (
    bets["btts_yes"]
    ==
    1
)


bets["profit"] = np.where(
    bets["won"],
    bets["odds_yes"]
    -
    1.0,
    -1.0,
)


bets["breakeven_prob"] = (
    1.0
    /
    bets["odds_yes"]
)


bets["model_minus_break_even"] = (
    bets["champion_yes"]
    -
    bets["breakeven_prob"]
)


print()
print("Total qualifying bets:", len(bets))


# ============================================================
# YEAR-BY-YEAR SUMMARY
# ============================================================

year_summary = (
    bets.groupby(
        "test_year"
    )
    .agg(
        bets=("won", "size"),
        wins=("won", "sum"),
        hit_rate=("won", "mean"),
        avg_model_prob=("champion_yes", "mean"),
        avg_market_prob=("market_yes", "mean"),
        avg_edge=("champion_edge_yes", "mean"),
        avg_ev=("champion_ev_yes", "mean"),
        avg_odds=("odds_yes", "mean"),
        avg_break_even=("breakeven_prob", "mean"),
        avg_lambda_min=("lambda_min", "mean"),
        avg_lambda_total=("lambda_total", "mean"),
        avg_lambda_gap=("lambda_gap", "mean"),
        profit_units=("profit", "sum"),
    )
    .reset_index()
)


year_summary["roi"] = (
    year_summary["profit_units"]
    /
    year_summary["bets"]
)


year_summary["calibration_error"] = (
    year_summary["hit_rate"]
    -
    year_summary["avg_model_prob"]
)


year_summary["market_error"] = (
    year_summary["hit_rate"]
    -
    year_summary["avg_market_prob"]
)


print()
print("=" * 120)
print("YEAR-BY-YEAR")
print("=" * 120)

yd = year_summary.copy()

for c in [
    "hit_rate",
    "avg_model_prob",
    "avg_market_prob",
    "avg_edge",
    "avg_ev",
    "avg_break_even",
    "roi",
    "calibration_error",
    "market_error",
]:
    yd[c] = yd[c].map(
        lambda x: f"{x:+.2%}"
    )

for c in [
    "avg_odds",
    "avg_lambda_min",
    "avg_lambda_total",
    "avg_lambda_gap",
]:
    yd[c] = yd[c].map(
        lambda x: f"{x:.3f}"
    )

yd["profit_units"] = yd[
    "profit_units"
].map(
    lambda x: f"{x:+.2f}"
)

print()
print(
    yd.to_string(
        index=False
    )
)


# ============================================================
# 2025 LEAGUE SPLIT
# ============================================================

y2025 = bets[
    bets["test_year"]
    ==
    2025
].copy()


league_2025 = (
    y2025.groupby(
        "league"
    )
    .agg(
        bets=("won", "size"),
        wins=("won", "sum"),
        hit_rate=("won", "mean"),
        avg_model_prob=("champion_yes", "mean"),
        avg_market_prob=("market_yes", "mean"),
        avg_edge=("champion_edge_yes", "mean"),
        avg_ev=("champion_ev_yes", "mean"),
        avg_odds=("odds_yes", "mean"),
        avg_lambda_min=("lambda_min", "mean"),
        avg_lambda_total=("lambda_total", "mean"),
        profit_units=("profit", "sum"),
    )
    .reset_index()
)


league_2025["roi"] = (
    league_2025["profit_units"]
    /
    league_2025["bets"]
)


league_2025["calibration_error"] = (
    league_2025["hit_rate"]
    -
    league_2025["avg_model_prob"]
)


print()
print("=" * 120)
print("2025 BY LEAGUE")
print("=" * 120)

ld = league_2025.copy()

for c in [
    "hit_rate",
    "avg_model_prob",
    "avg_market_prob",
    "avg_edge",
    "avg_ev",
    "roi",
    "calibration_error",
]:
    ld[c] = ld[c].map(
        lambda x: f"{x:+.2%}"
    )

for c in [
    "avg_odds",
    "avg_lambda_min",
    "avg_lambda_total",
]:
    ld[c] = ld[c].map(
        lambda x: f"{x:.3f}"
    )

ld["profit_units"] = ld[
    "profit_units"
].map(
    lambda x: f"{x:+.2f}"
)

print()
print(
    ld.to_string(
        index=False
    )
)


# ============================================================
# ODDS BAND COMPARISON
# ============================================================

bets["odds_band"] = pd.cut(
    bets["odds_yes"],
    bins=[
        1.00,
        1.70,
        1.80,
        1.90,
        2.00,
        2.20,
        2.50,
    ],
    include_lowest=True,
)


odds_summary = (
    bets.groupby(
        [
            "test_year",
            "odds_band",
        ],
        observed=True,
    )
    .agg(
        bets=("won", "size"),
        wins=("won", "sum"),
        hit_rate=("won", "mean"),
        avg_model=("champion_yes", "mean"),
        avg_market=("market_yes", "mean"),
        avg_odds=("odds_yes", "mean"),
        profit=("profit", "sum"),
    )
    .reset_index()
)


odds_summary["roi"] = (
    odds_summary["profit"]
    /
    odds_summary["bets"]
)


print()
print("=" * 120)
print("ODDS BAND BY YEAR")
print("=" * 120)

print()
print(
    odds_summary.to_string(
        index=False
    )
)


# ============================================================
# LAMBDA SUB-BANDS
# ============================================================

bets["lambda_band"] = pd.cut(
    bets["lambda_min"],
    bins=[
        0.90,
        1.00,
        1.05,
        1.10,
        1.15,
        1.20,
    ],
    include_lowest=True,
)


lambda_summary = (
    bets.groupby(
        [
            "test_year",
            "lambda_band",
        ],
        observed=True,
    )
    .agg(
        bets=("won", "size"),
        wins=("won", "sum"),
        hit_rate=("won", "mean"),
        avg_model=("champion_yes", "mean"),
        avg_market=("market_yes", "mean"),
        avg_odds=("odds_yes", "mean"),
        profit=("profit", "sum"),
    )
    .reset_index()
)


lambda_summary["roi"] = (
    lambda_summary["profit"]
    /
    lambda_summary["bets"]
)


print()
print("=" * 120)
print("LAMBDA_MIN BAND BY YEAR")
print("=" * 120)

print()
print(
    lambda_summary.to_string(
        index=False
    )
)


# ============================================================
# TEAM CONCENTRATION — 2025
# ============================================================

def top_team_table(
    data,
    column,
    label,
):

    if column not in data.columns:
        return pd.DataFrame()

    x = (
        data.groupby(
            column
        )
        .agg(
            bets=("won", "size"),
            wins=("won", "sum"),
            hit_rate=("won", "mean"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )

    x["roi"] = (
        x["profit"]
        /
        x["bets"]
    )

    x["role"] = label

    return x.sort_values(
        "bets",
        ascending=False,
    )


home_teams = top_team_table(
    y2025,
    "home_team",
    "HOME",
)

away_teams = top_team_table(
    y2025,
    "away_team",
    "AWAY",
)


team_summary = pd.concat(
    [
        home_teams,
        away_teams,
    ],
    ignore_index=True,
)


print()
print("=" * 120)
print("2025 TEAM CONCENTRATION")
print("=" * 120)

print()
print(
    team_summary
    .sort_values(
        "bets",
        ascending=False,
    )
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# EXACT 2025 BET LIST
# ============================================================

print()
print("=" * 120)
print("2025 BET LIST")
print("=" * 120)

bet_cols = [
    c
    for c in [
        "date",
        "league",
        "home_team",
        "away_team",
        "btts_yes",
        "champion_yes",
        "market_yes",
        "champion_edge_yes",
        "champion_ev_yes",
        "odds_yes",
        "home_lambda",
        "away_lambda",
        "lambda_min",
        "lambda_total",
        "lambda_gap",
        "won",
        "profit",
    ]
    if c in y2025.columns
]


print()
print(
    y2025[
        bet_cols
    ]
    .sort_values(
        [
            "league",
            "date",
        ]
    )
    .to_string(
        index=False
    )
)


# ============================================================
# VARIANCE CHECK
#
# Given each 2025 bet's model probability, simulate expected
# number of wins under the model.
# ============================================================

rng = np.random.default_rng(
    42
)

probs = (
    y2025[
        "champion_yes"
    ]
    .dropna()
    .to_numpy()
)

actual_wins = int(
    y2025["won"].sum()
)

actual_bets = len(
    y2025
)


if len(probs):

    sims = 200000

    sim_wins = (
        rng.random(
            (
                sims,
                len(probs),
            )
        )
        <
        probs
    ).sum(
        axis=1
    )


    expected_wins = (
        probs.sum()
    )

    p_at_or_below_actual = (
        sim_wins
        <=
        actual_wins
    ).mean()


    print()
    print("=" * 120)
    print("2025 VARIANCE CHECK")
    print("=" * 120)

    print()
    print(
        "Bets:",
        actual_bets,
    )

    print(
        "Actual wins:",
        actual_wins,
    )

    print(
        "Model expected wins:",
        f"{expected_wins:.2f}",
    )

    print(
        "Expected hit rate:",
        f"{probs.mean():.2%}",
    )

    print(
        "Actual hit rate:",
        f"{actual_wins / actual_bets:.2%}",
    )

    print(
        "P(simulated wins <= actual wins):",
        f"{p_at_or_below_actual:.4%}",
    )


# ============================================================
# MARKET VARIANCE CHECK
# ============================================================

market_probs = (
    y2025[
        "market_yes"
    ]
    .dropna()
    .to_numpy()
)


if len(market_probs):

    sim_market_wins = (
        rng.random(
            (
                200000,
                len(market_probs),
            )
        )
        <
        market_probs
    ).sum(
        axis=1
    )


    market_expected_wins = (
        market_probs.sum()
    )


    market_p_at_or_below = (
        sim_market_wins
        <=
        actual_wins
    ).mean()


    print()
    print(
        "Market expected wins:",
        f"{market_expected_wins:.2f}",
    )

    print(
        "Market expected hit rate:",
        f"{market_probs.mean():.2%}",
    )

    print(
        "P(market-sim wins <= actual):",
        f"{market_p_at_or_below:.4%}",
    )


# ============================================================
# SAVE
# ============================================================

bets.to_csv(
    OUT_DIR
    / "01_bet036028_all_years.csv",
    index=False,
)

year_summary.to_csv(
    OUT_DIR
    / "02_year_summary.csv",
    index=False,
)

league_2025.to_csv(
    OUT_DIR
    / "03_2025_by_league.csv",
    index=False,
)

odds_summary.to_csv(
    OUT_DIR
    / "04_odds_band_by_year.csv",
    index=False,
)

lambda_summary.to_csv(
    OUT_DIR
    / "05_lambda_band_by_year.csv",
    index=False,
)

team_summary.to_csv(
    OUT_DIR
    / "06_2025_team_concentration.csv",
    index=False,
)

y2025.to_csv(
    OUT_DIR
    / "07_2025_exact_bets.csv",
    index=False,
)


print()
print("=" * 120)
print("OUTPUTS")
print("=" * 120)

for p in sorted(
    OUT_DIR.glob("*")
):

    print(p)


print()
print("DONE")
