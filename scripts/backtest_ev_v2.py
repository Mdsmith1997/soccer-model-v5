from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

CALIBRATED_FILE = (
    ROOT
    / "data"
    / "processed"
    / "v2_calibrated_predictions.csv"
)

MARKET_FILE = (
    ROOT
    / "data"
    / "processed"
    / "v2_market_comparison.csv"
)

OUTPUT_BETS = (
    ROOT
    / "data"
    / "processed"
    / "ev_v2_backtest_bets.csv"
)

OUTPUT_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "ev_v2_backtest_summary.csv"
)


# =========================================================
# SPLITS
# =========================================================

DEVELOPMENT_SEASONS = {
    "2324",
    "2425",
}

FINAL_SEASON = {
    "2526",
}


# =========================================================
# EV THRESHOLDS
# =========================================================

EV_THRESHOLDS = [
    0.00,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.15,
]


# =========================================================
# SETTINGS
# =========================================================

MIN_ODDS = 1.20
MAX_ODDS = 10.00

MIN_BETS_FOR_SEGMENT = 50


# =========================================================
# HELPERS
# =========================================================

def american_result(
    actual_result,
    selection,
):
    return int(
        actual_result == selection
    )


def result_from_score(
    home_goals,
    away_goals,
):
    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


def max_drawdown(
    profits,
):
    """
    Unit-based max drawdown from cumulative P/L.
    """

    cumulative = np.cumsum(
        np.asarray(
            profits,
            dtype=float,
        )
    )

    if len(cumulative) == 0:
        return 0.0

    equity = np.concatenate(
        [
            np.array([0.0]),
            cumulative,
        ]
    )

    running_peak = np.maximum.accumulate(
        equity
    )

    drawdowns = (
        equity
        - running_peak
    )

    return abs(
        drawdowns.min()
    )


# =========================================================
# BUILD BET CANDIDATES
# =========================================================

def build_candidates(
    calibrated,
    market,
):

    market_keep = market[
        [
            "match_id",
            "market_source",
            "market_home_odds",
            "market_draw_odds",
            "market_away_odds",
            "market_p_home",
            "market_p_draw",
            "market_p_away",
        ]
    ].copy()

    df = calibrated.merge(
        market_keep,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    # Closing prices only.
    df = df[
        df[
            "market_source"
        ]
        == "avg_close"
    ].copy()

    df[
        "season"
    ] = (
        df[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    df[
        "actual_result"
    ] = [
        result_from_score(
            h,
            a,
        )
        for h, a in zip(
            df[
                "home_goals"
            ],
            df[
                "away_goals"
            ],
        )
    ]

    rows = []

    for _, row in df.iterrows():

        outcomes = [
            (
                "H",
                "HOME",
                row[
                    "cal_p_home"
                ],
                row[
                    "market_p_home"
                ],
                row[
                    "market_home_odds"
                ],
            ),
            (
                "D",
                "DRAW",
                row[
                    "cal_p_draw"
                ],
                row[
                    "market_p_draw"
                ],
                row[
                    "market_draw_odds"
                ],
            ),
            (
                "A",
                "AWAY",
                row[
                    "cal_p_away"
                ],
                row[
                    "market_p_away"
                ],
                row[
                    "market_away_odds"
                ],
            ),
        ]

        for (
            selection,
            side,
            model_prob,
            market_prob,
            odds,
        ) in outcomes:

            if (
                pd.isna(model_prob)
                or pd.isna(market_prob)
                or pd.isna(odds)
            ):
                continue

            if (
                odds < MIN_ODDS
                or odds > MAX_ODDS
            ):
                continue

            ev = (
                model_prob
                * odds
                - 1.0
            )

            probability_edge = (
                model_prob
                - market_prob
            )

            won = american_result(
                row[
                    "actual_result"
                ],
                selection,
            )

            profit = (
                odds - 1.0
                if won
                else -1.0
            )

            rows.append({
                "match_id":
                    row[
                        "match_id"
                    ],

                "date":
                    row[
                        "date"
                    ],

                "season":
                    row[
                        "season"
                    ],

                "league":
                    row[
                        "league"
                    ],

                "home_team":
                    row[
                        "home_team"
                    ],

                "away_team":
                    row[
                        "away_team"
                    ],

                "selection":
                    selection,

                "side":
                    side,

                "actual_result":
                    row[
                        "actual_result"
                    ],

                "model_prob":
                    model_prob,

                "market_prob":
                    market_prob,

                "probability_edge":
                    probability_edge,

                "odds":
                    odds,

                "expected_value":
                    ev,

                "won":
                    won,

                "profit":
                    profit,
            })

    candidates = pd.DataFrame(
        rows
    )

    candidates = (
        candidates
        .sort_values(
            [
                "date",
                "match_id",
                "selection",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return candidates


# =========================================================
# SUMMARY
# =========================================================

def summarize_bets(
    bets,
    label,
    threshold,
    league="ALL",
    side="ALL",
):

    if len(bets) == 0:
        return {
            "sample":
                label,

            "threshold":
                threshold,

            "league":
                league,

            "side":
                side,

            "bets":
                0,

            "wins":
                0,

            "win_rate":
                np.nan,

            "avg_odds":
                np.nan,

            "avg_model_prob":
                np.nan,

            "avg_market_prob":
                np.nan,

            "avg_edge":
                np.nan,

            "avg_ev":
                np.nan,

            "profit_units":
                0.0,

            "roi":
                np.nan,

            "max_drawdown":
                0.0,
        }

    wins = bets[
        "won"
    ].sum()

    profit = bets[
        "profit"
    ].sum()

    roi = (
        profit
        / len(bets)
    )

    return {
        "sample":
            label,

        "threshold":
            threshold,

        "league":
            league,

        "side":
            side,

        "bets":
            len(bets),

        "wins":
            int(
                wins
            ),

        "win_rate":
            wins
            / len(bets),

        "avg_odds":
            bets[
                "odds"
            ].mean(),

        "avg_model_prob":
            bets[
                "model_prob"
            ].mean(),

        "avg_market_prob":
            bets[
                "market_prob"
            ].mean(),

        "avg_edge":
            bets[
                "probability_edge"
            ].mean(),

        "avg_ev":
            bets[
                "expected_value"
            ].mean(),

        "profit_units":
            profit,

        "roi":
            roi,

        "max_drawdown":
            max_drawdown(
                bets[
                    "profit"
                ].to_numpy()
            ),
    }


# =========================================================
# THRESHOLD BACKTEST
# =========================================================

def backtest_thresholds(
    candidates,
    seasons,
    label,
):

    season_mask = (
        candidates[
            "season"
        ].isin(
            seasons
        )
    )

    base = candidates[
        season_mask
    ].copy()

    rows = []

    for threshold in EV_THRESHOLDS:

        bets = base[
            base[
                "expected_value"
            ]
            >= threshold
        ].copy()

        rows.append(
            summarize_bets(
                bets,
                label,
                threshold,
            )
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# SEGMENT TABLE
# =========================================================

def segment_backtest(
    candidates,
    seasons,
    threshold,
    label,
):

    base = candidates[
        candidates[
            "season"
        ].isin(
            seasons
        )
        &
        (
            candidates[
                "expected_value"
            ]
            >= threshold
        )
    ].copy()

    rows = []

    # League only.
    for league, group in base.groupby(
        "league"
    ):

        rows.append(
            summarize_bets(
                group,
                label,
                threshold,
                league=league,
                side="ALL",
            )
        )

    # Side only.
    for side, group in base.groupby(
        "side"
    ):

        rows.append(
            summarize_bets(
                group,
                label,
                threshold,
                league="ALL",
                side=side,
            )
        )

    # League x side.
    for (
        league,
        side,
    ), group in base.groupby(
        [
            "league",
            "side",
        ]
    ):

        rows.append(
            summarize_bets(
                group,
                label,
                threshold,
                league=league,
                side=side,
            )
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# DISPLAY
# =========================================================

def display_threshold_table(
    title,
    table,
):

    print()
    print("=" * 115)
    print(title)
    print("=" * 115)

    display = table.copy()

    for col in [
        "win_rate",
        "avg_edge",
        "avg_ev",
        "roi",
    ]:

        display[
            col
        ] *= 100.0

    print(
        display[
            [
                "threshold",
                "bets",
                "wins",
                "win_rate",
                "avg_odds",
                "avg_edge",
                "avg_ev",
                "profit_units",
                "roi",
                "max_drawdown",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


def display_segments(
    title,
    table,
):

    print()
    print("=" * 135)
    print(title)
    print("=" * 135)

    display = table[
        table[
            "bets"
        ]
        >= MIN_BETS_FOR_SEGMENT
    ].copy()

    if len(display) == 0:

        print(
            "No segments meet minimum "
            f"{MIN_BETS_FOR_SEGMENT}-bet sample."
        )

        return

    for col in [
        "win_rate",
        "avg_edge",
        "avg_ev",
        "roi",
    ]:

        display[
            col
        ] *= 100.0

    display = display.sort_values(
        [
            "roi",
            "bets",
        ],
        ascending=[
            False,
            False,
        ],
    )

    print(
        display[
            [
                "league",
                "side",
                "bets",
                "win_rate",
                "avg_odds",
                "avg_edge",
                "avg_ev",
                "profit_units",
                "roi",
                "max_drawdown",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("==============================")
    print("BACKTESTING V2 EXPECTED VALUE")
    print("==============================")
    print()

    if not CALIBRATED_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{CALIBRATED_FILE}"
        )

    if not MARKET_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{MARKET_FILE}"
        )

    calibrated = pd.read_csv(
        CALIBRATED_FILE,
        parse_dates=[
            "date",
        ],
    )

    market = pd.read_csv(
        MARKET_FILE,
        parse_dates=[
            "date",
        ],
    )

    print(
        f"Calibrated matches: "
        f"{len(calibrated):,}"
    )

    print(
        f"Market comparison rows: "
        f"{len(market):,}"
    )

    print()
    print(
        "Building closing-line bet candidates..."
    )

    candidates = build_candidates(
        calibrated,
        market,
    )

    print(
        f"Candidate outcome rows: "
        f"{len(candidates):,}"
    )

    print(
        f"Unique matches: "
        f"{candidates['match_id'].nunique():,}"
    )

    print()
    print(
        "Odds range:"
    )

    print(
        f"{candidates['odds'].min():.2f}"
        " to "
        f"{candidates['odds'].max():.2f}"
    )

    # =====================================================
    # DEVELOPMENT THRESHOLDS
    # =====================================================

    development = backtest_thresholds(
        candidates,
        DEVELOPMENT_SEASONS,
        "2023/24-2024/25",
    )

    display_threshold_table(
        "DEVELOPMENT — 2023/24 TO 2024/25",
        development,
    )

    # =====================================================
    # PICK DEVELOPMENT THRESHOLD
    #
    # Rule:
    # Choose best ROI among thresholds with at least
    # 200 bets. Ties/near-ties prefer the LOWER threshold
    # for larger sample size.
    #
    # 2025/26 is NOT consulted.
    # =====================================================

    eligible = development[
        development[
            "bets"
        ]
        >= 200
    ].copy()

    if len(eligible) == 0:

        raise RuntimeError(
            "No development threshold has "
            "at least 200 bets."
        )

    eligible = eligible.sort_values(
        [
            "roi",
            "bets",
        ],
        ascending=[
            False,
            False,
        ],
    )

    selected_threshold = float(
        eligible.iloc[
            0
        ][
            "threshold"
        ]
    )

    print()
    print("==============================")
    print("SELECTED EV THRESHOLD")
    print("==============================")

    print(
        f"Selected from development only: "
        f"{selected_threshold:.1%}"
    )

    print(
        "2025/26 was NOT used "
        "to select this threshold ✅"
    )

    # =====================================================
    # DEVELOPMENT SEGMENTS
    # =====================================================

    dev_segments = segment_backtest(
        candidates,
        DEVELOPMENT_SEASONS,
        selected_threshold,
        "2023/24-2024/25",
    )

    display_segments(
        f"DEVELOPMENT SEGMENTS — EV >= "
        f"{selected_threshold:.1%}",
        dev_segments,
    )

    # =====================================================
    # FINAL 2025/26 — ALL THRESHOLDS
    #
    # We print all thresholds diagnostically,
    # but do NOT use them to change the selected threshold.
    # =====================================================

    final_thresholds = backtest_thresholds(
        candidates,
        FINAL_SEASON,
        "2025/26",
    )

    display_threshold_table(
        "FINAL — 2025/26 — DIAGNOSTIC THRESHOLDS",
        final_thresholds,
    )

    # =====================================================
    # FINAL SELECTED THRESHOLD
    # =====================================================

    final_selected = final_thresholds[
        final_thresholds[
            "threshold"
        ]
        == selected_threshold
    ]

    print()
    print("=" * 115)
    print(
        "FINAL 2025/26 — "
        f"PRESELECTED EV >= "
        f"{selected_threshold:.1%}"
    )
    print("=" * 115)

    final_display = final_selected.copy()

    for col in [
        "win_rate",
        "avg_edge",
        "avg_ev",
        "roi",
    ]:

        final_display[
            col
        ] *= 100.0

    print(
        final_display[
            [
                "threshold",
                "bets",
                "wins",
                "win_rate",
                "avg_odds",
                "avg_edge",
                "avg_ev",
                "profit_units",
                "roi",
                "max_drawdown",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # FINAL SEGMENTS
    # =====================================================

    final_segments = segment_backtest(
        candidates,
        FINAL_SEASON,
        selected_threshold,
        "2025/26",
    )

    display_segments(
        f"FINAL 2025/26 SEGMENTS — EV >= "
        f"{selected_threshold:.1%}",
        final_segments,
    )

    # =====================================================
    # YEAR-BY-YEAR AT SELECTED THRESHOLD
    # =====================================================

    print()
    print("=" * 115)
    print(
        "SEASON-BY-SEASON — "
        f"EV >= {selected_threshold:.1%}"
    )
    print("=" * 115)

    season_rows = []

    for season, group in (
        candidates[
            candidates[
                "expected_value"
            ]
            >= selected_threshold
        ]
        .groupby(
            "season"
        )
    ):

        season_rows.append(
            summarize_bets(
                group,
                season,
                selected_threshold,
            )
        )

    season_table = pd.DataFrame(
        season_rows
    )

    for col in [
        "win_rate",
        "avg_edge",
        "avg_ev",
        "roi",
    ]:

        season_table[
            col
        ] *= 100.0

    print(
        season_table[
            [
                "sample",
                "bets",
                "wins",
                "win_rate",
                "avg_odds",
                "avg_edge",
                "avg_ev",
                "profit_units",
                "roi",
                "max_drawdown",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # SAVE BET-LEVEL FILE
    # =====================================================

    candidates[
        "passes_selected_threshold"
    ] = (
        candidates[
            "expected_value"
        ]
        >= selected_threshold
    ).astype(int)

    candidates.to_csv(
        OUTPUT_BETS,
        index=False,
    )

    # =====================================================
    # SAVE SUMMARY
    # =====================================================

    development[
        "table_type"
    ] = (
        "development_threshold"
    )

    final_thresholds[
        "table_type"
    ] = (
        "final_threshold"
    )

    summary = pd.concat(
        [
            development,
            final_thresholds,
            dev_segments,
            final_segments,
            season_table,
        ],
        ignore_index=True,
        sort=False,
    )

    summary[
        "selected_threshold"
    ] = selected_threshold

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    print()
    print("==============================")
    print("VALIDATION")
    print("==============================")

    print(
        "Only avg_close prices used ✅"
    )

    print(
        "One-unit flat staking used ✅"
    )

    print(
        "2025/26 not used to choose "
        "EV threshold ✅"
    )

    print(
        "Market probabilities not used "
        "inside model probability generation ✅"
    )

    print()
    print(
        f"Bet-level file:"
        f"\n{OUTPUT_BETS}"
    )

    print()
    print(
        f"Summary file:"
        f"\n{OUTPUT_SUMMARY}"
    )


if __name__ == "__main__":
    main()