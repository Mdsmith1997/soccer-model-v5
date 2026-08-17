from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

BET_FILE = (
    ROOT
    / "data"
    / "processed"
    / "ev_v2_backtest_bets.csv"
)

OUTPUT_BUCKETS = (
    ROOT
    / "data"
    / "processed"
    / "ev_v2_diagnostics.csv"
)


# =========================================================
# BUCKETS
# =========================================================

EDGE_BINS = [
    -np.inf,
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.15,
    np.inf,
]

EDGE_LABELS = [
    "<0%",
    "0-2%",
    "2-4%",
    "4-6%",
    "6-8%",
    "8-10%",
    "10-15%",
    "15%+",
]


EV_BINS = [
    -np.inf,
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    np.inf,
]

EV_LABELS = [
    "<0%",
    "0-5%",
    "5-10%",
    "10-15%",
    "15-20%",
    "20-30%",
    "30%+",
]


ODDS_BINS = [
    1.20,
    1.50,
    2.00,
    3.00,
    4.00,
    6.00,
    10.01,
]

ODDS_LABELS = [
    "1.20-1.49",
    "1.50-1.99",
    "2.00-2.99",
    "3.00-3.99",
    "4.00-5.99",
    "6.00-10.00",
]


# =========================================================
# SUMMARY
# =========================================================

def summarize(
    df,
    group_name,
    group_value,
):

    if len(df) == 0:
        return None

    bets = len(df)
    wins = int(
        df["won"].sum()
    )

    avg_model_prob = (
        df["model_prob"].mean()
    )

    avg_market_prob = (
        df["market_prob"].mean()
    )

    actual_hit_rate = (
        df["won"].mean()
    )

    avg_edge = (
        df["probability_edge"].mean()
    )

    avg_ev = (
        df["expected_value"].mean()
    )

    avg_odds = (
        df["odds"].mean()
    )

    profit = (
        df["profit"].sum()
    )

    roi = (
        profit / bets
    )

    model_cal_error = (
        avg_model_prob
        - actual_hit_rate
    )

    market_cal_error = (
        avg_market_prob
        - actual_hit_rate
    )

    return {
        "group":
            group_name,

        "bucket":
            group_value,

        "bets":
            bets,

        "wins":
            wins,

        "actual_hit_rate":
            actual_hit_rate,

        "avg_model_prob":
            avg_model_prob,

        "avg_market_prob":
            avg_market_prob,

        "model_minus_actual":
            model_cal_error,

        "market_minus_actual":
            market_cal_error,

        "avg_probability_edge":
            avg_edge,

        "avg_expected_value":
            avg_ev,

        "avg_odds":
            avg_odds,

        "profit_units":
            profit,

        "roi":
            roi,
    }


# =========================================================
# PRINT TABLE
# =========================================================

def print_table(
    title,
    table,
):

    print()
    print("=" * 145)
    print(title)
    print("=" * 145)

    display = table.copy()

    percent_cols = [
        "actual_hit_rate",
        "avg_model_prob",
        "avg_market_prob",
        "model_minus_actual",
        "market_minus_actual",
        "avg_probability_edge",
        "avg_expected_value",
        "roi",
    ]

    for col in percent_cols:
        display[col] *= 100.0

    print(
        display[
            [
                "bucket",
                "bets",
                "wins",
                "actual_hit_rate",
                "avg_model_prob",
                "avg_market_prob",
                "model_minus_actual",
                "market_minus_actual",
                "avg_probability_edge",
                "avg_expected_value",
                "avg_odds",
                "profit_units",
                "roi",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


# =========================================================
# GENERIC GROUP ANALYSIS
# =========================================================

def analyze_groups(
    df,
    group_column,
    group_name,
):

    rows = []

    for value, group in df.groupby(
        group_column,
        observed=False,
    ):

        result = summarize(
            group,
            group_name,
            str(value),
        )

        if result is not None:
            rows.append(
                result
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("==============================")
    print("DIAGNOSING V2 EXPECTED VALUE")
    print("==============================")
    print()

    if not BET_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{BET_FILE}"
        )

    bets = pd.read_csv(
        BET_FILE,
        parse_dates=[
            "date",
        ],
    )

    bets[
        "season"
    ] = (
        bets[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    print(
        f"Bet candidate rows loaded: "
        f"{len(bets):,}"
    )

    print(
        f"Unique matches: "
        f"{bets['match_id'].nunique():,}"
    )

    # -----------------------------------------------------
    # POSITIVE-EV ONLY
    # -----------------------------------------------------

    positive = bets[
        bets[
            "expected_value"
        ]
        >= 0.0
    ].copy()

    print(
        f"Positive-EV rows: "
        f"{len(positive):,}"
    )

    # -----------------------------------------------------
    # BUCKETS
    # -----------------------------------------------------

    positive[
        "edge_bucket"
    ] = pd.cut(
        positive[
            "probability_edge"
        ],
        bins=EDGE_BINS,
        labels=EDGE_LABELS,
        right=False,
    )

    positive[
        "ev_bucket"
    ] = pd.cut(
        positive[
            "expected_value"
        ],
        bins=EV_BINS,
        labels=EV_LABELS,
        right=False,
    )

    positive[
        "odds_bucket"
    ] = pd.cut(
        positive[
            "odds"
        ],
        bins=ODDS_BINS,
        labels=ODDS_LABELS,
        right=False,
        include_lowest=True,
    )

    # =====================================================
    # EDGE BUCKET
    # =====================================================

    edge_table = analyze_groups(
        positive,
        "edge_bucket",
        "Probability Edge",
    )

    print_table(
        "PROBABILITY EDGE BUCKETS",
        edge_table,
    )

    # =====================================================
    # EV BUCKET
    # =====================================================

    ev_table = analyze_groups(
        positive,
        "ev_bucket",
        "Expected Value",
    )

    print_table(
        "PREDICTED EV BUCKETS",
        ev_table,
    )

    # =====================================================
    # ODDS BUCKET
    # =====================================================

    odds_table = analyze_groups(
        positive,
        "odds_bucket",
        "Odds",
    )

    print_table(
        "ODDS BUCKETS",
        odds_table,
    )

    # =====================================================
    # SIDE
    # =====================================================

    side_table = analyze_groups(
        positive,
        "side",
        "Side",
    )

    print_table(
        "HOME / DRAW / AWAY",
        side_table,
    )

    # =====================================================
    # LEAGUE
    # =====================================================

    league_table = analyze_groups(
        positive,
        "league",
        "League",
    )

    print_table(
        "LEAGUES",
        league_table,
    )

    # =====================================================
    # SEASON
    # =====================================================

    season_table = analyze_groups(
        positive,
        "season",
        "Season",
    )

    print_table(
        "SEASONS",
        season_table,
    )

    # =====================================================
    # ODDS x EDGE
    # =====================================================

    print()
    print("=" * 145)
    print("ODDS x PROBABILITY EDGE")
    print("=" * 145)

    cross_rows = []

    for (
        odds_bucket,
        edge_bucket,
    ), group in positive.groupby(
        [
            "odds_bucket",
            "edge_bucket",
        ],
        observed=False,
    ):

        if len(group) < 30:
            continue

        result = summarize(
            group,
            "Odds x Edge",
            (
                f"{odds_bucket}"
                " | "
                f"{edge_bucket}"
            ),
        )

        cross_rows.append(
            result
        )

    cross_table = pd.DataFrame(
        cross_rows
    )

    if len(cross_table) > 0:

        cross_display = (
            cross_table
            .copy()
        )

        percent_cols = [
            "actual_hit_rate",
            "avg_model_prob",
            "avg_market_prob",
            "model_minus_actual",
            "market_minus_actual",
            "avg_probability_edge",
            "avg_expected_value",
            "roi",
        ]

        for col in percent_cols:
            cross_display[
                col
            ] *= 100.0

        cross_display = (
            cross_display
            .sort_values(
                [
                    "roi",
                    "bets",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
        )

        print(
            cross_display[
                [
                    "bucket",
                    "bets",
                    "actual_hit_rate",
                    "avg_model_prob",
                    "avg_market_prob",
                    "model_minus_actual",
                    "market_minus_actual",
                    "avg_expected_value",
                    "avg_odds",
                    "roi",
                ]
            ]
            .round(4)
            .to_string(
                index=False
            )
        )

    # =====================================================
    # MODEL VS MARKET CALIBRATION DISTANCE
    # =====================================================

    print()
    print("=" * 145)
    print("WHO WAS CLOSER TO REALITY?")
    print("=" * 145)

    positive[
        "model_abs_error"
    ] = (
        positive[
            "model_prob"
        ]
        -
        positive[
            "won"
        ]
    ).abs()

    positive[
        "market_abs_error"
    ] = (
        positive[
            "market_prob"
        ]
        -
        positive[
            "won"
        ]
    ).abs()

    model_better = (
        positive[
            "model_abs_error"
        ]
        <
        positive[
            "market_abs_error"
        ]
    ).mean()

    market_better = (
        positive[
            "market_abs_error"
        ]
        <
        positive[
            "model_abs_error"
        ]
    ).mean()

    ties = (
        positive[
            "market_abs_error"
        ]
        ==
        positive[
            "model_abs_error"
        ]
    ).mean()

    print(
        f"Model closer:  "
        f"{model_better:.2%}"
    )

    print(
        f"Market closer: "
        f"{market_better:.2%}"
    )

    print(
        f"Ties:          "
        f"{ties:.2%}"
    )

    # =====================================================
    # SAVE EVERYTHING
    # =====================================================

    edge_table[
        "table_type"
    ] = "edge"

    ev_table[
        "table_type"
    ] = "ev"

    odds_table[
        "table_type"
    ] = "odds"

    side_table[
        "table_type"
    ] = "side"

    league_table[
        "table_type"
    ] = "league"

    season_table[
        "table_type"
    ] = "season"

    tables = [
        edge_table,
        ev_table,
        odds_table,
        side_table,
        league_table,
        season_table,
    ]

    if len(cross_table) > 0:

        cross_table[
            "table_type"
        ] = "odds_x_edge"

        tables.append(
            cross_table
        )

    output = pd.concat(
        tables,
        ignore_index=True,
        sort=False,
    )

    output.to_csv(
        OUTPUT_BUCKETS,
        index=False,
    )

    print()
    print("==============================")
    print("DIAGNOSIS COMPLETE")
    print("==============================")

    print(
        f"Saved:"
        f"\n{OUTPUT_BUCKETS}"
    )


if __name__ == "__main__":
    main()