from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    ROOT
    / "data"
    / "live"
    / "v5_live_predictions_master.csv"
)

ODDS_FILE = (
    ROOT
    / "data"
    / "live"
    / "odds_markets_snapshot.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "live"
    / "v5_live_totals_ev_board.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TOTAL_LINE = 2.5

ACTIVE_LEAGUES = {
    "Premier League",
    "Bundesliga",
    "2. Bundesliga",
    "Belgian Pro League",
    "Championship",
    "La Liga",
    "League One",
    "League Two",
    "MLS",
    "Eliteserien",
}

EPS = 1e-12


# ============================================================
# LOAD DATA
# ============================================================

pred = pd.read_csv(
    PREDICTIONS_FILE,
    low_memory=False,
)

odds = pd.read_csv(
    ODDS_FILE,
    low_memory=False,
)


# ============================================================
# BASIC CLEANUP
# ============================================================

pred["date"] = pd.to_datetime(
    pred["date"],
    errors="coerce",
).dt.date

odds["point"] = pd.to_numeric(
    odds["point"],
    errors="coerce",
)

odds["decimal_odds"] = pd.to_numeric(
    odds["decimal_odds"],
    errors="coerce",
)


# ============================================================
# ACTIVE LEAGUES
# ============================================================

pred = pred[
    pred["league"].isin(
        ACTIVE_LEAGUES
    )
].copy()

odds = odds[
    odds["league"].isin(
        ACTIVE_LEAGUES
    )
].copy()


# ============================================================
# FILTER EXACTLY O/U 2.5
# ============================================================

totals = odds[
    (odds["market"] == "TOTALS")
    &
    (
        np.isclose(
            odds["point"],
            TOTAL_LINE,
        )
    )
    &
    (
        odds["selection"].isin(
            [
                "Over",
                "Under",
            ]
        )
    )
].copy()


# ============================================================
# BEST PRICE BY MATCH / SIDE
# ============================================================

totals = (
    totals
    .sort_values(
        "decimal_odds",
        ascending=False,
    )
    .drop_duplicates(
        subset=[
            "match_id",
            "selection",
            "point",
        ],
        keep="first",
    )
    .copy()
)


# ============================================================
# MODEL PROBABILITY
# ============================================================

def get_model_probability(row):

    if row["selection"] == "Over":
        return row["p_over_2_5_v5"]

    return row["p_under_2_5_v5"]


# ============================================================
# MERGE MODEL + ODDS
# ============================================================

needed_pred_cols = [
    "match_id",
    "date",
    "league",
    "home_team",
    "away_team",
    "home_lambda_v5",
    "away_lambda_v5",
    "p_over_2_5_v5",
    "p_under_2_5_v5",
]

optional_cols = [
    "deployment_tier",
    "home_history_source",
    "away_history_source",
]

for c in optional_cols:
    if c in pred.columns:
        needed_pred_cols.append(c)

# ============================================================
# NORMALIZE MATCH KEYS
#
# Totals odds use LIVE_* match IDs while prediction files may
# use a different internal ID system. Match on fixture identity
# instead.
# ============================================================

totals["date"] = pd.to_datetime(
    totals["commence_time"],
    utc=True,
    errors="coerce",
).dt.date

for df in (
    totals,
    pred,
):
    df["league"] = (
        df["league"]
        .astype(str)
        .str.strip()
    )

    df["home_team"] = (
        df["home_team"]
        .astype(str)
        .str.strip()
    )

    df["away_team"] = (
        df["away_team"]
        .astype(str)
        .str.strip()
    )


# ============================================================
# MERGE MODEL + ODDS
# ============================================================

board = totals.merge(
    pred[
        needed_pred_cols
    ],
    on=[
        "date",
        "league",
        "home_team",
        "away_team",
    ],
    how="inner",
    suffixes=(
        "_odds",
        "",
    ),
)


# ============================================================
# MODEL PROBABILITY
# ============================================================

board["model_probability"] = (
    board.apply(
        get_model_probability,
        axis=1,
    )
)


# ============================================================
# DEVIG MARKET PROBABILITY
# ============================================================

# We calculate the market probability from matched
# Over/Under prices at the SAME sportsbook rather than
# simply using 1 / best_odds.
#
# This keeps the market comparison internally consistent.

market_pairs = odds[
    (odds["market"] == "TOTALS")
    &
    (
        np.isclose(
            odds["point"],
            TOTAL_LINE,
        )
    )
    &
    (
        odds["selection"].isin(
            [
                "Over",
                "Under",
            ]
        )
    )
].copy()

pair_pivot = (
    market_pairs
    .pivot_table(
        index=[
            "match_id",
            "bookmaker",
        ],
        columns="selection",
        values="decimal_odds",
        aggfunc="max",
    )
    .reset_index()
)

# ------------------------------------------------------------
# SAFE EMPTY / ONE-SIDED TOTALS MARKET HANDLING
#
# A valid live window can contain zero complete O/U pairs.
# pivot_table() will then omit "Over" and/or "Under".
# Create the missing columns so zero available markets becomes
# a normal no-signal state rather than a pipeline failure.
# ------------------------------------------------------------

for required_side in [
    "Over",
    "Under",
]:

    if required_side not in pair_pivot.columns:

        pair_pivot[
            required_side
        ] = np.nan


pair_pivot = pair_pivot.dropna(
    subset=[
        "Over",
        "Under",
    ]
).copy()


print(
    "Complete Over/Under bookmaker pairs:",
    len(pair_pivot),
)

pair_pivot["raw_over"] = (
    1.0
    /
    pair_pivot["Over"]
)

pair_pivot["raw_under"] = (
    1.0
    /
    pair_pivot["Under"]
)

pair_pivot["raw_sum"] = (
    pair_pivot["raw_over"]
    +
    pair_pivot["raw_under"]
)

pair_pivot["market_p_over"] = (
    pair_pivot["raw_over"]
    /
    pair_pivot["raw_sum"]
)

pair_pivot["market_p_under"] = (
    pair_pivot["raw_under"]
    /
    pair_pivot["raw_sum"]
)

pair_lookup = pair_pivot[
    [
        "match_id",
        "bookmaker",
        "market_p_over",
        "market_p_under",
    ]
]

board = board.merge(
    pair_lookup,
    on=[
        "match_id",
        "bookmaker",
    ],
    how="left",
)


def get_market_probability(row):

    if row["selection"] == "Over":
        return row["market_p_over"]

    return row["market_p_under"]


board[
    "market_probability"
] = board.apply(
    get_market_probability,
    axis=1,
)


# ============================================================
# FALLBACK IF MATCHED DEVIG PAIR IS UNAVAILABLE
# ============================================================

fallback = (
    1.0
    /
    board["decimal_odds"]
)

board[
    "market_probability"
] = board[
    "market_probability"
].fillna(
    fallback
)


# ============================================================
# EDGE + EV
# ============================================================

board["edge"] = (
    board["model_probability"]
    -
    board["market_probability"]
)

board["ev"] = (
    board["model_probability"]
    *
    board["decimal_odds"]
    -
    1.0
)


# ============================================================
# EXPECTED TOTAL GOALS
# ============================================================

board[
    "expected_goals"
] = (
    board["home_lambda_v5"]
    +
    board["away_lambda_v5"]
)


# ============================================================
# BET LABEL
# ============================================================

board["bet"] = (
    board["selection"]
    +
    " "
    +
    board["point"].map(
        lambda x:
            f"{x:.1f}"
    )
)


# ============================================================
# CLEAN OUTPUT
# ============================================================

output_cols = [
    "date",
    "league",
    "home_team",
    "away_team",
    "bet",
    "expected_goals",
    "model_probability",
    "market_probability",
    "edge",
    "decimal_odds",
    "bookmaker",
    "ev",
]

for c in optional_cols:
    if c in board.columns:
        output_cols.append(c)

output_cols += [
    "match_id",
]

board = board[
    output_cols
].copy()


# ============================================================
# SORT
# ============================================================

board = board.sort_values(
    [
        "edge",
        "ev",
    ],
    ascending=[
        False,
        False,
    ],
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

board.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# DISPLAY
# ============================================================

print()
print(
    "=" * 130
)
print(
    "LIVE V5 TOTALS EV BOARD — OVER/UNDER 2.5"
)
print(
    "=" * 130
)
print()

print(
    f"Prediction fixtures: "
    f"{pred['match_id'].nunique():,}"
)

print(
    f"2.5 market rows: "
    f"{len(totals):,}"
)

print(
    f"Matched model/market selections: "
    f"{len(board):,}"
)

print()

if len(board):

    display = board.copy()

    for c in [
        "model_probability",
        "market_probability",
        "edge",
        "ev",
    ]:
        display[c] *= 100

    cols = [
        "date",
        "league",
        "home_team",
        "away_team",
        "bet",
        "expected_goals",
        "model_probability",
        "market_probability",
        "edge",
        "decimal_odds",
        "bookmaker",
        "ev",
    ]

    print(
        display[
            cols
        ]
        .head(
            100
        )
        .round(
            2
        )
        .to_string(
            index=False
        )
    )

else:

    print(
        "No matched O/U 2.5 markets."
    )


# ============================================================
# POSITIVE EDGE
# ============================================================

positive = board[
    board["edge"] > 0
].copy()

print()
print(
    "=" * 130
)
print(
    "POSITIVE V5 TOTALS EDGES"
)
print(
    "=" * 130
)
print()

if len(positive):

    display = positive.copy()

    for c in [
        "model_probability",
        "market_probability",
        "edge",
        "ev",
    ]:
        display[c] *= 100

    print(
        display[
            [
                "date",
                "league",
                "home_team",
                "away_team",
                "bet",
                "expected_goals",
                "model_probability",
                "market_probability",
                "edge",
                "decimal_odds",
                "bookmaker",
                "ev",
            ]
        ]
        .round(
            2
        )
        .to_string(
            index=False
        )
    )

else:

    print(
        "No positive totals edges."
    )


print()
print(
    "=" * 130
)
print(
    "TOTALS BOARD COMPLETE"
)
print(
    "=" * 130
)

print(
    "Market: O/U 2.5"
)

print(
    "Model: frozen V5 goal probabilities"
)

print(
    "Best sportsbook price retained"
)

print(
    "Sportsbook Over/Under pair de-vigged"
)

print(
    "No betting threshold applied yet"
)

print(
    "No totals bets written to live ledger"
)

print()

print(
    "Saved:"
)

print(
    OUTPUT_FILE
)