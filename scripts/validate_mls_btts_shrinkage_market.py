from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

STABILITY_FILE = (
    ROOT / "data" / "processed"
    / "mls_btts_stability_diag"
    / "01_mls_games_enriched.csv"
)

OUT = (
    ROOT / "data" / "processed"
    / "mls_btts_shrinkage_market_validation.csv"
)

BY_YEAR_OUT = (
    ROOT / "data" / "processed"
    / "mls_btts_shrinkage_market_validation_by_year.csv"
)

MATCH_OUT = (
    ROOT / "data" / "processed"
    / "mls_btts_shrinkage_market_matches.csv"
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
]

S1 = 0.60
S2 = 0.00
S3 = 0.30

df = pd.read_csv(
    STABILITY_FILE,
    low_memory=False,
)

print("=" * 120)
print("MLS BTTS — 0148_ALL_HISTORY MARKET VALIDATION")
print("=" * 120)

print()
print("Input:", STABILITY_FILE)
print("Rows:", len(df))


# ============================================================
# STANDARDIZE
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)

numeric_cols = [
    "btts_yes",
    "champion_yes",
    "market_yes",
    "market_no",
    "odds_yes",
    "odds_no",
    "minimum_team_game_number",
    "test_year",
]

for c in numeric_cols:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce",
    )

required = [
    "date",
    "btts_yes",
    "champion_yes",
    "market_yes",
    "market_no",
    "odds_yes",
    "odds_no",
    "minimum_team_game_number",
    "test_year",
]

before = len(df)

df = df.dropna(
    subset=required
).copy()

print(
    "Usable rows:",
    len(df),
    f"(dropped {before - len(df)})",
)


# ============================================================
# RECONSTRUCT EXACT ALL-HISTORY PRIOR
# ============================================================
#
# Exact robustness-lab definition:
#
# history = every row with date < Jan 1 of test year
# prior   = historical actual BTTS YES mean
#
# If no prior exists:
# prior = champion_yes
#
# ============================================================

years = sorted(
    df["test_year"]
    .dropna()
    .astype(int)
    .unique()
)

all_history_prior = {}

for year in years:

    history = df[
        df["date"]
        <
        pd.Timestamp(
            year=year,
            month=1,
            day=1,
        )
    ]

    all_history_prior[year] = (
        history["btts_yes"].mean()
        if len(history)
        else np.nan
    )


print()
print("ALL-HISTORY PRIORS")
print("-" * 120)

for year in years:
    p = all_history_prior[year]

    if np.isfinite(p):
        print(
            year,
            f"{p:.4%}",
        )
    else:
        print(
            year,
            "NO PRIOR — FALLBACK TO BASE",
        )


df["all_history_prior"] = [
    all_history_prior[
        int(year)
    ]
    for year in df["test_year"]
]

df["all_history_prior"] = np.where(
    np.isfinite(
        df["all_history_prior"]
    ),
    df["all_history_prior"],
    df["champion_yes"],
)


# ============================================================
# APPLY EXACT 0148_ALL_HISTORY CONFIG
# ============================================================

games = (
    df["minimum_team_game_number"]
    .to_numpy(dtype=float)
)

base = (
    df["champion_yes"]
    .to_numpy(dtype=float)
)

prior = (
    df["all_history_prior"]
    .to_numpy(dtype=float)
)

shrink = np.zeros(
    len(df),
    dtype=float,
)

shrink[
    games <= 5
] = S1

shrink[
    (games >= 6)
    &
    (games <= 10)
] = S2

shrink[
    (games >= 11)
    &
    (games <= 15)
] = S3

df["shrink_weight"] = shrink

df["p_yes_shrunk"] = (
    (1.0 - shrink)
    * base
    +
    shrink
    * prior
)

df["p_no_shrunk"] = (
    1.0
    -
    df["p_yes_shrunk"]
)


# ============================================================
# MARKET EDGE
# ============================================================

df["edge_yes_shrunk"] = (
    df["p_yes_shrunk"]
    -
    df["market_yes"]
)

df["edge_no_shrunk"] = (
    df["p_no_shrunk"]
    -
    df["market_no"]
)


# ============================================================
# PROFIT
# ============================================================

df["profit_yes"] = np.where(
    df["btts_yes"].eq(1),
    df["odds_yes"] - 1.0,
    -1.0,
)

df["profit_no"] = np.where(
    df["btts_yes"].eq(0),
    df["odds_no"] - 1.0,
    -1.0,
)


# ============================================================
# SANITY CHECK SHRINKAGE
# ============================================================

print()
print("SHRINKAGE SANITY CHECK")
print("-" * 120)

for label, lo, hi in [
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-15", 11, 15),
    ("16+", 16, np.inf),
]:

    if np.isinf(hi):
        g = df[
            df["minimum_team_game_number"]
            >= lo
        ]
    else:
        g = df[
            (
                df["minimum_team_game_number"]
                >= lo
            )
            &
            (
                df["minimum_team_game_number"]
                <= hi
            )
        ]

    if len(g):
        print(
            f"{label:<8}"
            f"N={len(g):>5} | "
            f"Weight={g['shrink_weight'].mean():.2f} | "
            f"Base={g['champion_yes'].mean():.2%} | "
            f"Shrunk={g['p_yes_shrunk'].mean():.2%}"
        )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    data,
    side,
    threshold,
):

    if side == "YES":

        edge_col = "edge_yes_shrunk"
        odds_col = "odds_yes"
        profit_col = "profit_yes"

        win = (
            data["btts_yes"]
            .eq(1)
        )

    else:

        edge_col = "edge_no_shrunk"
        odds_col = "odds_no"
        profit_col = "profit_no"

        win = (
            data["btts_yes"]
            .eq(0)
        )

    g = data[
        data[edge_col]
        >= threshold
    ].copy()

    if g.empty:
        return None

    return {
        "side": side,
        "threshold": threshold,
        "bets": len(g),
        "wins": int(
            win.loc[
                g.index
            ].sum()
        ),
        "win_rate": float(
            win.loc[
                g.index
            ].mean()
        ),
        "avg_odds": float(
            g[odds_col].mean()
        ),
        "avg_edge": float(
            g[edge_col].mean()
        ),
        "profit_units": float(
            g[profit_col].sum()
        ),
        "roi": float(
            g[profit_col].mean()
        ),
    }


# ============================================================
# AGGREGATE THRESHOLD SCREEN
# ============================================================

rows = []

for threshold in THRESHOLDS:

    for side in [
        "YES",
        "NO",
    ]:

        r = evaluate(
            df,
            side,
            threshold,
        )

        if r is not None:
            rows.append(r)


summary = pd.DataFrame(rows)

summary.to_csv(
    OUT,
    index=False,
)

print()
print("=" * 120)
print("AGGREGATE RESULTS")
print("=" * 120)

print(
    summary.to_string(
        index=False,
        formatters={
            "threshold":
                lambda x: f"{x:.0%}",

            "win_rate":
                lambda x: f"{x:.2%}",

            "avg_odds":
                lambda x: f"{x:.3f}",

            "avg_edge":
                lambda x: f"{x:.2%}",

            "profit_units":
                lambda x: f"{x:+.2f}u",

            "roi":
                lambda x: f"{x:+.2%}",
        },
    )
)


# ============================================================
# YEAR-BY-YEAR
# ============================================================

year_rows = []

for year, yg in df.groupby(
    "test_year",
    sort=True,
):

    for threshold in THRESHOLDS:

        for side in [
            "YES",
            "NO",
        ]:

            r = evaluate(
                yg,
                side,
                threshold,
            )

            if r is None:
                continue

            r["year"] = int(year)

            year_rows.append(r)


by_year = pd.DataFrame(
    year_rows
)

by_year.to_csv(
    BY_YEAR_OUT,
    index=False,
)

print()
print("=" * 120)
print("YEAR-BY-YEAR RESULTS")
print("=" * 120)

print(
    by_year.to_string(
        index=False,
        formatters={
            "threshold":
                lambda x: f"{x:.0%}",

            "win_rate":
                lambda x: f"{x:.2%}",

            "avg_odds":
                lambda x: f"{x:.3f}",

            "avg_edge":
                lambda x: f"{x:.2%}",

            "profit_units":
                lambda x: f"{x:+.2f}u",

            "roi":
                lambda x: f"{x:+.2%}",
        },
    )
)


# ============================================================
# SAVE MATCH LEVEL
# ============================================================

save_cols = [
    "date",
    "test_year",
    "home_team",
    "away_team",
    "btts_yes",
    "minimum_team_game_number",
    "champion_yes",
    "all_history_prior",
    "shrink_weight",
    "p_yes_shrunk",
    "p_no_shrunk",
    "market_yes",
    "market_no",
    "edge_yes_shrunk",
    "edge_no_shrunk",
    "odds_yes",
    "odds_no",
    "profit_yes",
    "profit_no",
]

df[
    save_cols
].to_csv(
    MATCH_OUT,
    index=False,
)


print()
print("Saved:")
print(OUT)
print(BY_YEAR_OUT)
print(MATCH_OUT)

print()
print("=" * 120)
print("MLS 0148_ALL_HISTORY VALIDATION COMPLETE")
print("=" * 120)
