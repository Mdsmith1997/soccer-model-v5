from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "betting_backtest_v5_bets.csv"
)

OUTPUT_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "v5_residual_4pct_robustness_summary.csv"
)

OUTPUT_BETS = (
    ROOT
    / "data"
    / "processed"
    / "v5_residual_4pct_robustness_bets.csv"
)


# ============================================================
# LOCKED RULE
#
# IMPORTANT:
# This threshold is already selected from the prior
# development analysis. Do NOT retune it here.
# ============================================================

STRATEGY = "RESIDUAL_V5"
EDGE_THRESHOLD = 0.04

LOCKED_SEASON = "2526"


# ============================================================
# HELPERS
# ============================================================

def season_string(series):

    return (
        series
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.strip()
        .str.zfill(4)
    )


def summarize(
    df,
    group_name,
):

    bets = len(df)

    if bets == 0:

        return {
            "group":
                group_name,

            "bets":
                0,

            "wins":
                0,

            "win_rate":
                np.nan,

            "avg_odds":
                np.nan,

            "avg_model_probability":
                np.nan,

            "avg_market_probability":
                np.nan,

            "avg_edge":
                np.nan,

            "avg_model_ev":
                np.nan,

            "profit_units":
                0.0,

            "roi":
                np.nan,
        }

    wins = int(
        df[
            "won"
        ].sum()
    )

    profit = float(
        df[
            "profit"
        ].sum()
    )

    return {
        "group":
            group_name,

        "bets":
            bets,

        "wins":
            wins,

        "win_rate":
            wins / bets,

        "avg_odds":
            df[
                "odds"
            ].mean(),

        "avg_model_probability":
            df[
                "model_probability"
            ].mean(),

        "avg_market_probability":
            df[
                "market_probability"
            ].mean(),

        "avg_edge":
            df[
                "edge"
            ].mean(),

        "avg_model_ev":
            df[
                "model_ev"
            ].mean(),

        "profit_units":
            profit,

        "roi":
            profit / bets,
    }


def print_section(
    title,
    df,
):

    print()
    print("=" * 130)
    print(title)
    print("=" * 130)

    if df.empty:

        print(
            "No rows."
        )

        return

    display = df.copy()

    percentage_cols = [
        "win_rate",
        "avg_model_probability",
        "avg_market_probability",
        "avg_edge",
        "avg_model_ev",
        "roi",
    ]

    for col in percentage_cols:

        if col in display.columns:

            display[
                col
            ] *= 100.0

    print(
        display
        .round(4)
        .to_string(
            index=False
        )
    )


# ============================================================
# LOAD
# ============================================================

def load_bets():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing input file:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    required = [
        "match_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
        "actual_outcome",
        "strategy",
        "selection",
        "model_probability",
        "market_probability",
        "odds",
        "edge",
        "model_ev",
        "won",
        "profit",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Input file missing columns:\n"
            +
            "\n".join(
                missing
            )
        )

    df[
        "season"
    ] = season_string(
        df[
            "season"
        ]
    )

    numeric_cols = [
        "model_probability",
        "market_probability",
        "odds",
        "edge",
        "model_ev",
        "won",
        "profit",
    ]

    for col in numeric_cols:

        df[
            col
        ] = pd.to_numeric(
            df[
                col
            ],
            errors="coerce",
        )

    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        errors="coerce",
    )

    return df


# ============================================================
# APPLY LOCKED RULE
# ============================================================

def build_locked_rule_bets(
    df,
):

    x = df.loc[
        (
            df[
                "strategy"
            ]
            ==
            STRATEGY
        )
        &
        (
            df[
                "edge"
            ]
            >=
            EDGE_THRESHOLD
        )
    ].copy()

    if x.empty:

        return x

    # ========================================================
    # ONE BET PER MATCH
    #
    # If multiple outcomes qualify for the same fixture,
    # select the one with the highest model EV.
    #
    # Tie-breakers:
    # 1. higher model EV
    # 2. higher probability edge
    # 3. lower decimal odds
    # ========================================================

    x = (
        x
        .sort_values(
            [
                "match_id",
                "model_ev",
                "edge",
                "odds",
            ],
            ascending=[
                True,
                False,
                False,
                True,
            ],
        )
        .drop_duplicates(
            subset=[
                "match_id",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # PERIOD
    # ========================================================

    x[
        "period"
    ] = np.where(
        x[
            "season"
        ]
        ==
        LOCKED_SEASON,
        "LOCKED_2526",
        "DEVELOPMENT",
    )

    # ========================================================
    # ODDS BUCKET
    # ========================================================

    x[
        "odds_bucket"
    ] = pd.cut(
        x[
            "odds"
        ],
        bins=[
            1.0,
            1.50,
            2.00,
            2.50,
            3.00,
            4.00,
            5.00,
            10.00,
            np.inf,
        ],
        labels=[
            "1.01-1.49",
            "1.50-1.99",
            "2.00-2.49",
            "2.50-2.99",
            "3.00-3.99",
            "4.00-4.99",
            "5.00-9.99",
            "10.00+",
        ],
        right=False,
    )

    # ========================================================
    # EDGE BUCKET
    # ========================================================

    x[
        "edge_bucket"
    ] = pd.cut(
        x[
            "edge"
        ],
        bins=[
            0.04,
            0.05,
            0.06,
            0.08,
            0.10,
            0.12,
            0.15,
            np.inf,
        ],
        labels=[
            "4-5%",
            "5-6%",
            "6-8%",
            "8-10%",
            "10-12%",
            "12-15%",
            "15%+",
        ],
        right=False,
    )

    return x


# ============================================================
# GROUP ANALYSIS
# ============================================================

def grouped_summary(
    df,
    group_cols,
):

    rows = []

    grouped = df.groupby(
        group_cols,
        dropna=False,
        observed=True,
    )

    for key, group in grouped:

        if not isinstance(
            key,
            tuple,
        ):

            key = (
                key,
            )

        label = " | ".join(
            [
                f"{col}={value}"
                for col, value
                in zip(
                    group_cols,
                    key,
                )
            ]
        )

        row = summarize(
            group,
            label,
        )

        for col, value in zip(
            group_cols,
            key,
        ):

            row[
                col
            ] = value

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 130)
    print(
        "V5 RESIDUAL 4% ROBUSTNESS ANALYSIS"
    )
    print("=" * 130)

    print()
    print(
        "Locked strategy:",
        STRATEGY,
    )

    print(
        "Locked edge threshold:",
        f"{EDGE_THRESHOLD:.1%}",
    )

    print(
        "One bet per match:",
        "YES",
    )

    print(
        "Locked holdout season:",
        LOCKED_SEASON,
    )

    # ========================================================
    # LOAD
    # ========================================================

    df = load_bets()

    print()
    print(
        "All candidate rows:",
        f"{len(df):,}",
    )

    # ========================================================
    # LOCKED RULE
    # ========================================================

    bets = build_locked_rule_bets(
        df
    )

    print(
        "Qualifying one-bet-per-match rows:",
        f"{len(bets):,}",
    )

    development = bets.loc[
        bets[
            "period"
        ]
        ==
        "DEVELOPMENT"
    ].copy()

    locked = bets.loc[
        bets[
            "period"
        ]
        ==
        "LOCKED_2526"
    ].copy()

    # ========================================================
    # OVERALL
    # ========================================================

    overall_rows = [
        summarize(
            development,
            "DEVELOPMENT",
        ),
        summarize(
            locked,
            "LOCKED_2526",
        ),
        summarize(
            bets,
            "ALL",
        ),
    ]

    overall = pd.DataFrame(
        overall_rows
    )

    print_section(
        "OVERALL FIXED RULE",
        overall,
    )

    # ========================================================
    # BY SEASON
    # ========================================================

    by_season = grouped_summary(
        bets,
        [
            "season",
        ],
    ).sort_values(
        "season"
    )

    print_section(
        "BY SEASON",
        by_season,
    )

    # ========================================================
    # BY PERIOD + LEAGUE
    # ========================================================

    by_league = grouped_summary(
        bets,
        [
            "period",
            "league",
        ],
    ).sort_values(
        [
            "period",
            "league",
        ]
    )

    print_section(
        "BY PERIOD + LEAGUE",
        by_league,
    )

    # ========================================================
    # BY PERIOD + SIDE
    # ========================================================

    by_side = grouped_summary(
        bets,
        [
            "period",
            "selection",
        ],
    ).sort_values(
        [
            "period",
            "selection",
        ]
    )

    print_section(
        "BY PERIOD + BET SIDE",
        by_side,
    )

    # ========================================================
    # BY PERIOD + ODDS BUCKET
    # ========================================================

    by_odds = grouped_summary(
        bets,
        [
            "period",
            "odds_bucket",
        ],
    ).sort_values(
        [
            "period",
            "odds_bucket",
        ]
    )

    print_section(
        "BY PERIOD + ODDS BUCKET",
        by_odds,
    )

    # ========================================================
    # BY PERIOD + EDGE BUCKET
    # ========================================================

    by_edge = grouped_summary(
        bets,
        [
            "period",
            "edge_bucket",
        ],
    ).sort_values(
        [
            "period",
            "edge_bucket",
        ]
    )

    print_section(
        "BY PERIOD + EDGE BUCKET",
        by_edge,
    )

    # ========================================================
    # SEASON x LEAGUE
    # ========================================================

    season_league = grouped_summary(
        bets,
        [
            "season",
            "league",
        ],
    ).sort_values(
        [
            "season",
            "league",
        ]
    )

    print_section(
        "SEASON x LEAGUE",
        season_league,
    )

    # ========================================================
    # SEASON x SIDE
    # ========================================================

    season_side = grouped_summary(
        bets,
        [
            "season",
            "selection",
        ],
    ).sort_values(
        [
            "season",
            "selection",
        ]
    )

    print_section(
        "SEASON x BET SIDE",
        season_side,
    )

    # ========================================================
    # BUILD SUMMARY OUTPUT
    # ========================================================

    frames = []

    for name, frame in [
        (
            "OVERALL",
            overall,
        ),
        (
            "SEASON",
            by_season,
        ),
        (
            "PERIOD_LEAGUE",
            by_league,
        ),
        (
            "PERIOD_SIDE",
            by_side,
        ),
        (
            "PERIOD_ODDS",
            by_odds,
        ),
        (
            "PERIOD_EDGE",
            by_edge,
        ),
        (
            "SEASON_LEAGUE",
            season_league,
        ),
        (
            "SEASON_SIDE",
            season_side,
        ),
    ]:

        temp = frame.copy()

        temp.insert(
            0,
            "analysis",
            name,
        )

        frames.append(
            temp
        )

    summary_output = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_output.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    bets.to_csv(
        OUTPUT_BETS,
        index=False,
    )

    print()
    print("=" * 130)
    print("ROBUSTNESS CHECKS")
    print("=" * 130)

    # ========================================================
    # SIMPLE STABILITY CHECKS
    # ========================================================

    season_stats = (
        development
        .groupby(
            "season"
        )
        .agg(
            bets=(
                "won",
                "size",
            ),
            profit=(
                "profit",
                "sum",
            ),
        )
    )

    season_stats[
        "roi"
    ] = (
        season_stats[
            "profit"
        ]
        /
        season_stats[
            "bets"
        ]
    )

    positive_development_seasons = int(
        (
            season_stats[
                "roi"
            ]
            >
            0
        ).sum()
    )

    total_development_seasons = len(
        season_stats
    )

    print(
        "Positive development seasons:",
        positive_development_seasons,
        "/",
        total_development_seasons,
    )

    if total_development_seasons:

        print(
            "Worst development season ROI:",
            f"{season_stats['roi'].min():.2%}",
        )

        print(
            "Best development season ROI:",
            f"{season_stats['roi'].max():.2%}",
        )

    if len(
        locked
    ):

        locked_roi = (
            locked[
                "profit"
            ].sum()
            /
            len(
                locked
            )
        )

        print(
            "Locked 2025/26 ROI:",
            f"{locked_roi:.2%}",
        )

    print()
    print(
        "Saved summary:",
        OUTPUT_SUMMARY,
    )

    print(
        "Saved bets:",
        OUTPUT_BETS,
    )

    print()


if __name__ == "__main__":
    main()