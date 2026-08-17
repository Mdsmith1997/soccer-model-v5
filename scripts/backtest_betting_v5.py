from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"

RAW_META_FILE = (
    PROCESSED_DIR
    / "market_residual_meta_v5_predictions.csv"
)

RESIDUAL_FILE = (
    PROCESSED_DIR
    / "market_residual_v5_predictions.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "betting_backtest_v5.csv"
)

BET_OUTPUT_FILE = (
    PROCESSED_DIR
    / "betting_backtest_v5_bets.csv"
)


# ============================================================
# SETTINGS
# ============================================================

LOCKED_SEASON = "2526"

THRESHOLDS = [
    0.00,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.08,
    0.10,
    0.12,
    0.15,
]

OUTCOMES = [
    "HOME",
    "DRAW",
    "AWAY",
]

STRATEGIES = [
    "RAW_V5",
    "RESIDUAL_V5",
    "META_V5",
]


# ============================================================
# HELPERS
# ============================================================

def season_string(series):
    """
    Normalize season values such as:
        2526
        2526.0
        '2526'
    to:
        '2526'
    """
    return (
        series
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
        .str.zfill(4)
    )


def safe_float(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def actual_outcome(row):
    home_goals = row["home_goals"]
    away_goals = row["away_goals"]

    if pd.isna(home_goals) or pd.isna(away_goals):
        return np.nan

    if home_goals > away_goals:
        return "HOME"

    if home_goals == away_goals:
        return "DRAW"

    return "AWAY"


def profit_from_bet(
    actual,
    selection,
    odds,
):
    """
    Flat 1-unit stake.

    Win:
        decimal odds - 1

    Loss:
        -1
    """
    if (
        pd.isna(actual)
        or pd.isna(selection)
        or pd.isna(odds)
        or odds <= 1.0
    ):
        return np.nan

    if actual == selection:
        return float(odds) - 1.0

    return -1.0


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not RAW_META_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {RAW_META_FILE}"
        )

    if not RESIDUAL_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {RESIDUAL_FILE}"
        )

    meta = pd.read_csv(
        RAW_META_FILE
    )

    residual = pd.read_csv(
        RESIDUAL_FILE
    )

    meta["season"] = season_string(
        meta["season"]
    )

    residual["season"] = season_string(
        residual["season"]
    )

    # --------------------------------------------------------
    # Columns required from meta file
    # --------------------------------------------------------

    required_meta = [
        "match_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "p_home_v5",
        "p_draw_v5",
        "p_away_v5",
        "market_home_odds",
        "market_draw_odds",
        "market_away_odds",
        "market_nv_home",
        "market_nv_draw",
        "market_nv_away",
        "meta_p_home",
        "meta_p_draw",
        "meta_p_away",
    ]

    missing = [
        c
        for c in required_meta
        if c not in meta.columns
    ]

    if missing:
        raise ValueError(
            "Meta file missing columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Columns required from residual file
    # --------------------------------------------------------

    required_residual = [
        "match_id",
        "resid_p_home",
        "resid_p_draw",
        "resid_p_away",
    ]

    missing = [
        c
        for c in required_residual
        if c not in residual.columns
    ]

    if missing:
        raise ValueError(
            "Residual file missing columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Keep only residual probabilities for merge
    # --------------------------------------------------------

    residual_small = (
        residual[
            [
                "match_id",
                "resid_p_home",
                "resid_p_draw",
                "resid_p_away",
            ]
        ]
        .drop_duplicates(
            subset=["match_id"],
            keep="last",
        )
    )

    df = meta.merge(
        residual_small,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_cols = [
        "home_goals",
        "away_goals",
        "p_home_v5",
        "p_draw_v5",
        "p_away_v5",
        "resid_p_home",
        "resid_p_draw",
        "resid_p_away",
        "meta_p_home",
        "meta_p_draw",
        "meta_p_away",
        "market_home_odds",
        "market_draw_odds",
        "market_away_odds",
        "market_nv_home",
        "market_nv_draw",
        "market_nv_away",
    ]

    for col in numeric_cols:
        df[col] = safe_float(
            df[col]
        )

    df["actual_outcome"] = df.apply(
        actual_outcome,
        axis=1,
    )

    return df


# ============================================================
# CONVERT MATCHES INTO BET CANDIDATES
# ============================================================

def build_bet_candidates(df):

    strategy_prob_cols = {

        "RAW_V5": {
            "HOME": "p_home_v5",
            "DRAW": "p_draw_v5",
            "AWAY": "p_away_v5",
        },

        "RESIDUAL_V5": {
            "HOME": "resid_p_home",
            "DRAW": "resid_p_draw",
            "AWAY": "resid_p_away",
        },

        "META_V5": {
            "HOME": "meta_p_home",
            "DRAW": "meta_p_draw",
            "AWAY": "meta_p_away",
        },
    }

    market_prob_cols = {
        "HOME": "market_nv_home",
        "DRAW": "market_nv_draw",
        "AWAY": "market_nv_away",
    }

    odds_cols = {
        "HOME": "market_home_odds",
        "DRAW": "market_draw_odds",
        "AWAY": "market_away_odds",
    }

    rows = []

    for strategy in STRATEGIES:

        for outcome in OUTCOMES:

            prob_col = (
                strategy_prob_cols[
                    strategy
                ][
                    outcome
                ]
            )

            market_col = (
                market_prob_cols[
                    outcome
                ]
            )

            odds_col = (
                odds_cols[
                    outcome
                ]
            )

            x = df[
                [
                    "match_id",
                    "date",
                    "season",
                    "league",
                    "home_team",
                    "away_team",
                    "actual_outcome",
                    prob_col,
                    market_col,
                    odds_col,
                ]
            ].copy()

            x = x.rename(
                columns={
                    prob_col:
                        "model_probability",
                    market_col:
                        "market_probability",
                    odds_col:
                        "odds",
                }
            )

            x["strategy"] = strategy

            x["selection"] = outcome

            x["edge"] = (
                x["model_probability"]
                - x["market_probability"]
            )

            x["model_ev"] = (
                x["model_probability"]
                * x["odds"]
                - 1.0
            )

            x["won"] = (
                x["actual_outcome"]
                == x["selection"]
            ).astype(int)

            x["profit"] = x.apply(
                lambda row:
                    profit_from_bet(
                        row["actual_outcome"],
                        row["selection"],
                        row["odds"],
                    ),
                axis=1,
            )

            rows.append(x)

    bets = pd.concat(
        rows,
        ignore_index=True,
    )

    # Only valid historical betting observations
    bets = bets[
        bets[
            [
                "model_probability",
                "market_probability",
                "odds",
                "profit",
            ]
        ]
        .notna()
        .all(axis=1)
    ].copy()

    bets = bets[
        bets["odds"] > 1.0
    ].copy()

    return bets


# ============================================================
# EVALUATION
# ============================================================

def summarize_subset(
    subset,
    strategy,
    segment,
    selection,
    threshold,
):

    bets = len(subset)

    if bets == 0:
        return {
            "strategy": strategy,
            "segment": segment,
            "selection": selection,
            "threshold": threshold,
            "bets": 0,
            "wins": 0,
            "win_rate": np.nan,
            "avg_odds": np.nan,
            "avg_model_probability": np.nan,
            "avg_market_probability": np.nan,
            "avg_edge": np.nan,
            "avg_model_ev": np.nan,
            "profit_units": 0.0,
            "roi": np.nan,
        }

    wins = int(
        subset["won"].sum()
    )

    profit = float(
        subset["profit"].sum()
    )

    return {
        "strategy":
            strategy,

        "segment":
            segment,

        "selection":
            selection,

        "threshold":
            threshold,

        "bets":
            bets,

        "wins":
            wins,

        "win_rate":
            wins / bets,

        "avg_odds":
            subset[
                "odds"
            ].mean(),

        "avg_model_probability":
            subset[
                "model_probability"
            ].mean(),

        "avg_market_probability":
            subset[
                "market_probability"
            ].mean(),

        "avg_edge":
            subset[
                "edge"
            ].mean(),

        "avg_model_ev":
            subset[
                "model_ev"
            ].mean(),

        "profit_units":
            profit,

        "roi":
            profit / bets,
    }


def run_backtest(bets):

    results = []

    segments = {

        "DEVELOPMENT":
            bets[
                bets["season"]
                != LOCKED_SEASON
            ],

        "LOCKED_2526":
            bets[
                bets["season"]
                == LOCKED_SEASON
            ],

        "ALL":
            bets,
    }

    for (
        segment_name,
        segment_df,
    ) in segments.items():

        for strategy in STRATEGIES:

            strategy_df = segment_df[
                segment_df["strategy"]
                == strategy
            ]

            for threshold in THRESHOLDS:

                eligible = strategy_df[
                    strategy_df["edge"]
                    >= threshold
                ]

                # --------------------------------------------
                # ALL selections
                # --------------------------------------------

                results.append(
                    summarize_subset(
                        eligible,
                        strategy,
                        segment_name,
                        "ALL",
                        threshold,
                    )
                )

                # --------------------------------------------
                # Individual outcomes
                # --------------------------------------------

                for selection in OUTCOMES:

                    subset = eligible[
                        eligible["selection"]
                        == selection
                    ]

                    results.append(
                        summarize_subset(
                            subset,
                            strategy,
                            segment_name,
                            selection,
                            threshold,
                        )
                    )

    return pd.DataFrame(
        results
    )


# ============================================================
# PRINTING
# ============================================================

def print_table(
    results,
    strategy,
    segment,
    selection="ALL",
):

    x = results[
        (
            results["strategy"]
            == strategy
        )
        & (
            results["segment"]
            == segment
        )
        & (
            results["selection"]
            == selection
        )
    ].copy()

    if x.empty:
        return

    display = x[
        [
            "threshold",
            "bets",
            "wins",
            "win_rate",
            "avg_odds",
            "avg_edge",
            "profit_units",
            "roi",
        ]
    ].copy()

    display[
        "threshold"
    ] = (
        display[
            "threshold"
        ] * 100
    )

    display[
        "win_rate"
    ] = (
        display[
            "win_rate"
        ] * 100
    )

    display[
        "avg_edge"
    ] = (
        display[
            "avg_edge"
        ] * 100
    )

    display[
        "roi"
    ] = (
        display[
            "roi"
        ] * 100
    )

    print()
    print("=" * 110)

    print(
        f"{strategy} | "
        f"{segment} | "
        f"{selection}"
    )

    print("=" * 110)

    print(
        display.to_string(
            index=False,
            formatters={
                "threshold":
                    lambda x:
                        f"{x:.0f}%",

                "win_rate":
                    lambda x:
                        f"{x:.2f}%"
                        if pd.notna(x)
                        else "NaN",

                "avg_odds":
                    lambda x:
                        f"{x:.2f}"
                        if pd.notna(x)
                        else "NaN",

                "avg_edge":
                    lambda x:
                        f"{x:.2f}%"
                        if pd.notna(x)
                        else "NaN",

                "profit_units":
                    lambda x:
                        f"{x:+.2f}",

                "roi":
                    lambda x:
                        f"{x:+.2f}%"
                        if pd.notna(x)
                        else "NaN",
            }
        )
    )


# ============================================================
# DEVELOPMENT LEADERBOARD
# ============================================================

def development_leaderboard(results):

    x = results[
        (
            results["segment"]
            == "DEVELOPMENT"
        )
        & (
            results["selection"]
            == "ALL"
        )
    ].copy()

    # Avoid tiny samples winning the leaderboard.
    x = x[
        x["bets"] >= 100
    ].copy()

    if x.empty:
        return x

    x = x.sort_values(
        [
            "roi",
            "profit_units",
            "bets",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    return x


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print("BETTING BACKTEST V5")
    print("=" * 110)

    print()
    print(
        "Locked season:",
        LOCKED_SEASON,
    )

    print(
        "The locked season is reported "
        "separately and should NOT be used "
        "to choose a betting rule."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    print()
    print(
        "Historical matches:",
        f"{len(df):,}",
    )

    print(
        "Development matches:",
        f"{(df['season'] != LOCKED_SEASON).sum():,}",
    )

    print(
        "Locked 2025/26 matches:",
        f"{(df['season'] == LOCKED_SEASON).sum():,}",
    )

    # --------------------------------------------------------
    # Candidate bets
    # --------------------------------------------------------

    bets = build_bet_candidates(
        df
    )

    print()
    print(
        "Valid strategy/outcome "
        "bet candidates:",
        f"{len(bets):,}",
    )

    # --------------------------------------------------------
    # Backtest
    # --------------------------------------------------------

    results = run_backtest(
        bets
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    bets.to_csv(
        BET_OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Print ALL-outcome threshold tables
    # --------------------------------------------------------

    for strategy in STRATEGIES:

        print_table(
            results,
            strategy,
            "DEVELOPMENT",
            "ALL",
        )

        print_table(
            results,
            strategy,
            "LOCKED_2526",
            "ALL",
        )

    # --------------------------------------------------------
    # Leaderboard
    # --------------------------------------------------------

    leaderboard = (
        development_leaderboard(
            results
        )
    )

    print()
    print("=" * 110)
    print(
        "TOP DEVELOPMENT RULES "
        "(MINIMUM 100 BETS)"
    )
    print("=" * 110)

    if leaderboard.empty:

        print(
            "No rules with at least "
            "100 bets."
        )

    else:

        display = leaderboard[
            [
                "strategy",
                "threshold",
                "bets",
                "win_rate",
                "avg_odds",
                "avg_edge",
                "profit_units",
                "roi",
            ]
        ].head(20).copy()

        display[
            "threshold"
        ] *= 100

        display[
            "win_rate"
        ] *= 100

        display[
            "avg_edge"
        ] *= 100

        display[
            "roi"
        ] *= 100

        print(
            display.to_string(
                index=False,
                formatters={
                    "threshold":
                        lambda x:
                            f"{x:.0f}%",

                    "win_rate":
                        lambda x:
                            f"{x:.2f}%",

                    "avg_odds":
                        lambda x:
                            f"{x:.2f}",

                    "avg_edge":
                        lambda x:
                            f"{x:.2f}%",

                    "profit_units":
                        lambda x:
                            f"{x:+.2f}",

                    "roi":
                        lambda x:
                            f"{x:+.2f}%",
                }
            )
        )

    print()
    print("=" * 110)
    print("SAVED")
    print("=" * 110)

    print(
        OUTPUT_FILE
    )

    print(
        BET_OUTPUT_FILE
    )

    print()


if __name__ == "__main__":
    main()