from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

V5_FILE = (
    ROOT
    / "data"
    / "processed"
    / "frozen_v5_predictions.csv"
)

MATCHES_FILE = (
    ROOT
    / "data"
    / "processed"
    / "matches.csv"
)

OUTPUT_MATCHES = (
    ROOT
    / "data"
    / "processed"
    / "v5_opening_clv_matches.csv"
)

OUTPUT_THRESHOLDS = (
    ROOT
    / "data"
    / "processed"
    / "v5_opening_clv_threshold_results.csv"
)

OUTPUT_BETS = (
    ROOT
    / "data"
    / "processed"
    / "v5_opening_clv_bets.csv"
)

OUTPUT_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "v5_opening_clv_summary.csv"
)


# ============================================================
# SPLITS
# ============================================================

TUNING_SEASONS = {
    "2122",
    "2223",
}

VALIDATION_SEASONS = {
    "2324",
}

FINAL_SEASONS = {
    "2425",
}

LOCKED_SEASONS = {
    "2526",
}


# ============================================================
# THRESHOLDS
# ============================================================

EDGE_THRESHOLDS = [
    0.00,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.075,
    0.10,
    0.125,
    0.15,
]

EV_THRESHOLDS = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
]

MIN_BETS_FOR_SELECTION = 50

FLAT_STAKE = 1.0

EPS = 1e-12


# ============================================================
# HELPERS
# ============================================================

def season_string(series):

    return (
        series
        .astype(str)
        .str.zfill(4)
    )


def normalize_three_way(
    home,
    draw,
    away,
):

    probs = np.column_stack(
        [
            home,
            draw,
            away,
        ]
    ).astype(float)

    probs = np.clip(
        probs,
        EPS,
        None,
    )

    probs /= probs.sum(
        axis=1,
        keepdims=True,
    )

    return probs


def actual_outcome(df):

    return np.where(
        df["home_goals"].to_numpy()
        >
        df["away_goals"].to_numpy(),
        "HOME",
        np.where(
            df["home_goals"].to_numpy()
            ==
            df["away_goals"].to_numpy(),
            "DRAW",
            "AWAY",
        ),
    )


# ============================================================
# LOAD V5
# ============================================================

def load_v5():

    if not V5_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{V5_FILE}"
        )

    df = pd.read_csv(
        V5_FILE,
        parse_dates=[
            "date",
        ],
    )

    df["season"] = season_string(
        df["season"]
    )

    required = [
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
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Frozen V5 missing columns: "
            + str(missing)
        )

    return df


# ============================================================
# LOAD PAIRED OPEN/CLOSE ODDS
# ============================================================

def load_matches():

    if not MATCHES_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{MATCHES_FILE}"
        )

    df = pd.read_csv(
        MATCHES_FILE,
        parse_dates=[
            "date",
        ],
    )

    df["season"] = season_string(
        df["season"]
    )

    required = [
        "match_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",

        "b365_home_open",
        "b365_draw_open",
        "b365_away_open",

        "avg_home_close",
        "avg_draw_close",
        "avg_away_close",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "matches.csv missing required "
            "open/close odds columns: "
            + str(missing)
        )

    return df


# ============================================================
# BUILD MATCH TABLE
# ============================================================

def build_match_table():

    v5 = load_v5()

    matches = load_matches()

    market_cols = [
        "match_id",

        "b365_home_open",
        "b365_draw_open",
        "b365_away_open",

        "avg_home_close",
        "avg_draw_close",
        "avg_away_close",
    ]

    market = (
        matches[
            market_cols
        ]
        .copy()
    )

    duplicate_matches = (
        market[
            "match_id"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_matches > 0:

        raise ValueError(
            f"matches.csv contains "
            f"{duplicate_matches:,} duplicate "
            f"match_id rows."
        )

    df = v5.merge(
        market,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Frozen V5 matches: "
        f"{len(v5):,}"
    )

    print(
        f"Paired odds matches: "
        f"{len(matches):,}"
    )

    print(
        f"Matched V5 + paired odds: "
        f"{len(df):,}"
    )

    # ========================================================
    # OPENING NO-VIG PROBS
    # ========================================================

    open_home_raw = (
        1.0
        /
        df[
            "b365_home_open"
        ]
    )

    open_draw_raw = (
        1.0
        /
        df[
            "b365_draw_open"
        ]
    )

    open_away_raw = (
        1.0
        /
        df[
            "b365_away_open"
        ]
    )

    open_probs = normalize_three_way(
        open_home_raw,
        open_draw_raw,
        open_away_raw,
    )

    df[
        "open_nv_home"
    ] = open_probs[
        :,
        0
    ]

    df[
        "open_nv_draw"
    ] = open_probs[
        :,
        1
    ]

    df[
        "open_nv_away"
    ] = open_probs[
        :,
        2
    ]

    df[
        "open_margin"
    ] = (
        open_home_raw
        +
        open_draw_raw
        +
        open_away_raw
        -
        1.0
    )

    # ========================================================
    # CLOSING NO-VIG PROBS
    # ========================================================

    close_home_raw = (
        1.0
        /
        df[
            "avg_home_close"
        ]
    )

    close_draw_raw = (
        1.0
        /
        df[
            "avg_draw_close"
        ]
    )

    close_away_raw = (
        1.0
        /
        df[
            "avg_away_close"
        ]
    )

    close_probs = normalize_three_way(
        close_home_raw,
        close_draw_raw,
        close_away_raw,
    )

    df[
        "close_nv_home"
    ] = close_probs[
        :,
        0
    ]

    df[
        "close_nv_draw"
    ] = close_probs[
        :,
        1
    ]

    df[
        "close_nv_away"
    ] = close_probs[
        :,
        2
    ]

    df[
        "close_margin"
    ] = (
        close_home_raw
        +
        close_draw_raw
        +
        close_away_raw
        -
        1.0
    )

    df[
        "actual_outcome"
    ] = actual_outcome(
        df
    )

    return df


# ============================================================
# LONG BET TABLE
# ============================================================

def build_bet_candidates(df):

    rows = []

    definitions = [
        (
            "HOME",
            "p_home_v5",
            "open_nv_home",
            "close_nv_home",
            "b365_home_open",
            "avg_home_close",
        ),

        (
            "DRAW",
            "p_draw_v5",
            "open_nv_draw",
            "close_nv_draw",
            "b365_draw_open",
            "avg_draw_close",
        ),

        (
            "AWAY",
            "p_away_v5",
            "open_nv_away",
            "close_nv_away",
            "b365_away_open",
            "avg_away_close",
        ),
    ]

    base_cols = [
        "match_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
        "actual_outcome",
    ]

    for (
        side,
        model_col,
        open_prob_col,
        close_prob_col,
        open_odds_col,
        close_odds_col,
    ) in definitions:

        sub = df[
            base_cols
            +
            [
                model_col,
                open_prob_col,
                close_prob_col,
                open_odds_col,
                close_odds_col,
            ]
        ].copy()

        sub = sub.rename(
            columns={
                model_col:
                    "model_probability",

                open_prob_col:
                    "open_market_probability",

                close_prob_col:
                    "close_market_probability",

                open_odds_col:
                    "open_odds",

                close_odds_col:
                    "close_odds",
            }
        )

        sub[
            "bet_side"
        ] = side

        # ====================================================
        # MODEL EDGE AT OPEN
        # ====================================================

        sub[
            "opening_probability_edge"
        ] = (
            sub[
                "model_probability"
            ]
            -
            sub[
                "open_market_probability"
            ]
        )

        # ====================================================
        # MODEL EV AT OPEN
        # ====================================================

        sub[
            "opening_expected_value"
        ] = (
            sub[
                "model_probability"
            ]
            *
            sub[
                "open_odds"
            ]
            -
            1.0
        )

        # ====================================================
        # PRICE CLV
        #
        # Positive = we got a better price than close.
        #
        # Example:
        # 3.00 open / 2.70 close - 1
        # = +11.11%
        # ====================================================

        sub[
            "price_clv"
        ] = (
            sub[
                "open_odds"
            ]
            /
            sub[
                "close_odds"
            ]
            -
            1.0
        )

        # ====================================================
        # PROBABILITY CLV
        #
        # Positive = closing no-vig probability
        # moved toward our selected side.
        # ====================================================

        sub[
            "probability_clv"
        ] = (
            sub[
                "close_market_probability"
            ]
            -
            sub[
                "open_market_probability"
            ]
        )

        # ====================================================
        # BEAT CLOSE
        # ====================================================

        sub[
            "beat_close"
        ] = (
            sub[
                "open_odds"
            ]
            >
            sub[
                "close_odds"
            ]
        ).astype(
            int
        )

        # ====================================================
        # RESULT + PROFIT
        # ====================================================

        sub[
            "won"
        ] = (
            sub[
                "actual_outcome"
            ]
            ==
            side
        ).astype(
            int
        )

        sub[
            "profit"
        ] = np.where(
            sub[
                "won"
            ]
            ==
            1,
            (
                sub[
                    "open_odds"
                ]
                -
                1.0
            )
            *
            FLAT_STAKE,
            -FLAT_STAKE,
        )

        rows.append(
            sub
        )

    bets = pd.concat(
        rows,
        ignore_index=True,
    )

    valid = (
        bets[
            "model_probability"
        ].notna()
        &
        bets[
            "open_market_probability"
        ].notna()
        &
        bets[
            "close_market_probability"
        ].notna()
        &
        bets[
            "open_odds"
        ].notna()
        &
        bets[
            "close_odds"
        ].notna()
        &
        np.isfinite(
            bets[
                "open_odds"
            ]
        )
        &
        np.isfinite(
            bets[
                "close_odds"
            ]
        )
        &
        (
            bets[
                "open_odds"
            ]
            >
            1.0
        )
        &
        (
            bets[
                "close_odds"
            ]
            >
            1.0
        )
    )

    return bets.loc[
        valid
    ].copy()


# ============================================================
# BET METRICS
# ============================================================

def evaluate_bets(bets):

    if len(bets) == 0:

        return {
            "bets":
                0,

            "wins":
                0,

            "win_rate":
                np.nan,

            "avg_open_odds":
                np.nan,

            "avg_close_odds":
                np.nan,

            "avg_open_edge":
                np.nan,

            "avg_open_ev":
                np.nan,

            "avg_price_clv":
                np.nan,

            "median_price_clv":
                np.nan,

            "beat_close_rate":
                np.nan,

            "avg_probability_clv":
                np.nan,

            "positive_probability_clv_rate":
                np.nan,

            "profit":
                0.0,

            "roi":
                np.nan,

            "max_drawdown":
                np.nan,
        }

    ordered = (
        bets
        .sort_values(
            [
                "date",
                "match_id",
            ]
        )
        .copy()
    )

    ordered[
        "cumulative_profit"
    ] = (
        ordered[
            "profit"
        ]
        .cumsum()
    )

    running_peak = (
        ordered[
            "cumulative_profit"
        ]
        .cummax()
    )

    drawdown = (
        ordered[
            "cumulative_profit"
        ]
        -
        running_peak
    )

    total_profit = float(
        bets[
            "profit"
        ].sum()
    )

    total_staked = (
        len(bets)
        *
        FLAT_STAKE
    )

    return {
        "bets":
            len(bets),

        "wins":
            int(
                bets[
                    "won"
                ].sum()
            ),

        "win_rate":
            float(
                bets[
                    "won"
                ].mean()
            ),

        "avg_open_odds":
            float(
                bets[
                    "open_odds"
                ].mean()
            ),

        "avg_close_odds":
            float(
                bets[
                    "close_odds"
                ].mean()
            ),

        "avg_open_edge":
            float(
                bets[
                    "opening_probability_edge"
                ].mean()
            ),

        "avg_open_ev":
            float(
                bets[
                    "opening_expected_value"
                ].mean()
            ),

        "avg_price_clv":
            float(
                bets[
                    "price_clv"
                ].mean()
            ),

        "median_price_clv":
            float(
                bets[
                    "price_clv"
                ].median()
            ),

        "beat_close_rate":
            float(
                bets[
                    "beat_close"
                ].mean()
            ),

        "avg_probability_clv":
            float(
                bets[
                    "probability_clv"
                ].mean()
            ),

        "positive_probability_clv_rate":
            float(
                (
                    bets[
                        "probability_clv"
                    ]
                    >
                    0
                ).mean()
            ),

        "profit":
            total_profit,

        "roi":
            float(
                total_profit
                /
                total_staked
            ),

        "max_drawdown":
            float(
                drawdown.min()
            ),
    }


# ============================================================
# FILTER
# ============================================================

def filter_bets(
    bets,
    edge_threshold,
    ev_threshold,
):

    return bets[
        (
            bets[
                "opening_probability_edge"
            ]
            >=
            edge_threshold
        )
        &
        (
            bets[
                "opening_expected_value"
            ]
            >=
            ev_threshold
        )
    ].copy()


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def tune_thresholds(bets):

    tune = bets[
        bets[
            "season"
        ].isin(
            TUNING_SEASONS
        )
    ].copy()

    rows = []

    for edge_threshold in EDGE_THRESHOLDS:

        for ev_threshold in EV_THRESHOLDS:

            selected = filter_bets(
                tune,
                edge_threshold,
                ev_threshold,
            )

            metrics = evaluate_bets(
                selected
            )

            rows.append(
                {
                    "edge_threshold":
                        float(
                            edge_threshold
                        ),

                    "ev_threshold":
                        float(
                            ev_threshold
                        ),

                    **metrics,
                }
            )

    results = pd.DataFrame(
        rows
    )

    eligible = results[
        results[
            "bets"
        ]
        >=
        MIN_BETS_FOR_SELECTION
    ].copy()

    if len(eligible) == 0:

        raise ValueError(
            "No threshold combination "
            "had enough bets."
        )

    # ========================================================
    # PRIMARY SELECTION:
    #
    # Require positive average CLV.
    # Then rank by ROI.
    #
    # ========================================================

    positive_clv = eligible[
        eligible[
            "avg_price_clv"
        ]
        >
        0
    ].copy()

    if len(positive_clv) > 0:

        pool = positive_clv

    else:

        print()
        print(
            "WARNING: no threshold produced "
            "positive average tuning CLV."
        )

        pool = eligible

    pool = (
        pool
        .sort_values(
            [
                "roi",
                "avg_price_clv",
                "beat_close_rate",
                "bets",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    pool[
        "rank"
    ] = (
        np.arange(
            len(pool)
        )
        + 1
    )

    return (
        pool,
        results,
    )


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    title,
    metrics,
):

    print()
    print("=" * 82)
    print(title)
    print("=" * 82)

    print(
        f"Bets:                "
        f"{metrics['bets']:,}"
    )

    print(
        f"Wins:                "
        f"{metrics['wins']:,}"
    )

    print(
        f"Win rate:            "
        f"{metrics['win_rate']:.2%}"
    )

    print(
        f"Avg opening odds:    "
        f"{metrics['avg_open_odds']:.3f}"
    )

    print(
        f"Avg closing odds:    "
        f"{metrics['avg_close_odds']:.3f}"
    )

    print(
        f"Avg V5 open edge:    "
        f"{metrics['avg_open_edge']:.2%}"
    )

    print(
        f"Avg V5 open EV:      "
        f"{metrics['avg_open_ev']:.2%}"
    )

    print(
        f"Avg price CLV:       "
        f"{metrics['avg_price_clv']:+.2%}"
    )

    print(
        f"Median price CLV:    "
        f"{metrics['median_price_clv']:+.2%}"
    )

    print(
        f"Beat close rate:     "
        f"{metrics['beat_close_rate']:.2%}"
    )

    print(
        f"Avg probability CLV: "
        f"{metrics['avg_probability_clv']:+.2%}"
    )

    print(
        f"Positive prob CLV:   "
        f"{metrics['positive_probability_clv_rate']:.2%}"
    )

    print(
        f"Profit:              "
        f"{metrics['profit']:+.2f} units"
    )

    print(
        f"ROI:                 "
        f"{metrics['roi']:+.2%}"
    )

    print(
        f"Max drawdown:        "
        f"{metrics['max_drawdown']:.2f} units"
    )


# ============================================================
# BREAKDOWN
# ============================================================

def build_breakdown(
    bets,
    group_col,
):

    rows = []

    for name, sub in (
        bets.groupby(
            group_col,
            observed=False,
        )
    ):

        if len(sub) == 0:
            continue

        metrics = evaluate_bets(
            sub
        )

        rows.append(
            {
                group_col:
                    str(name),

                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def print_breakdown(
    title,
    table,
    group_col,
):

    print()
    print("=" * 135)
    print(title)
    print("=" * 135)

    if len(table) == 0:

        print(
            "No bets."
        )

        return

    display = table.copy()

    percent_cols = [
        "win_rate",
        "avg_open_edge",
        "avg_open_ev",
        "avg_price_clv",
        "median_price_clv",
        "beat_close_rate",
        "avg_probability_clv",
        "positive_probability_clv_rate",
        "roi",
    ]

    for col in percent_cols:

        display[
            col
        ] *= 100.0

    print(
        display[
            [
                group_col,
                "bets",
                "wins",
                "win_rate",
                "avg_open_odds",
                "avg_close_odds",
                "avg_open_edge",
                "avg_open_ev",
                "avg_price_clv",
                "beat_close_rate",
                "avg_probability_clv",
                "profit",
                "roi",
                "max_drawdown",
            ]
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# OPENING EDGE → CLV DIAGNOSTIC
# ============================================================

def print_edge_clv_diagnostic(
    bets,
    title,
):

    print()
    print("=" * 135)
    print(title)
    print("=" * 135)

    if len(bets) == 0:

        print(
            "No bets."
        )

        return

    df = bets.copy()

    df[
        "edge_band"
    ] = pd.cut(
        df[
            "opening_probability_edge"
        ],
        bins=[
            -1.00,
            0.00,
            0.025,
            0.05,
            0.075,
            0.10,
            0.15,
            1.00,
        ],
        labels=[
            "<0%",
            "0-2.5%",
            "2.5-5%",
            "5-7.5%",
            "7.5-10%",
            "10-15%",
            "15%+",
        ],
        right=False,
    )

    table = build_breakdown(
        df,
        "edge_band",
    )

    print_breakdown(
        title,
        table,
        "edge_band",
    )


# ============================================================
# ODDS BANDS
# ============================================================

def print_odds_bands(
    bets,
):

    if len(bets) == 0:
        return

    df = bets.copy()

    df[
        "odds_band"
    ] = pd.cut(
        df[
            "open_odds"
        ],
        bins=[
            1.00,
            1.50,
            2.00,
            2.50,
            3.00,
            4.00,
            6.00,
            100.00,
        ],
        labels=[
            "1.01-1.49",
            "1.50-1.99",
            "2.00-2.49",
            "2.50-2.99",
            "3.00-3.99",
            "4.00-5.99",
            "6.00+",
        ],
        right=False,
    )

    table = build_breakdown(
        df,
        "odds_band",
    )

    print_breakdown(
        "2025/26 LOCKED TEST — OPENING ODDS BANDS",
        table,
        "odds_band",
    )


# ============================================================
# CLV BANDS
# ============================================================

def print_clv_bands(
    bets,
):

    if len(bets) == 0:
        return

    df = bets.copy()

    df[
        "clv_band"
    ] = pd.cut(
        df[
            "price_clv"
        ],
        bins=[
            -1.00,
            -0.10,
            -0.05,
            -0.025,
            0.00,
            0.025,
            0.05,
            0.10,
            1.00,
        ],
        labels=[
            "<-10%",
            "-10 to -5%",
            "-5 to -2.5%",
            "-2.5 to 0%",
            "0 to +2.5%",
            "+2.5 to +5%",
            "+5 to +10%",
            "+10%+",
        ],
        right=False,
    )

    table = build_breakdown(
        df,
        "clv_band",
    )

    print_breakdown(
        "2025/26 LOCKED TEST — CLV BANDS",
        table,
        "clv_band",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("V5 OPENING / CLV BACKTEST")
    print("==============================")
    print()

    print(
        "Frozen signal: V5"
    )

    print(
        "Entry: Bet365 opening odds"
    )

    print(
        "CLV benchmark: average closing odds"
    )

    print(
        "Stake: flat 1 unit"
    )

    # ========================================================
    # MATCH TABLE
    # ========================================================

    matches = build_match_table()

    if len(matches) == 0:

        raise ValueError(
            "No matches joined between "
            "frozen V5 and matches.csv."
        )

    print()
    print(
        f"Matched date range: "
        f"{matches['date'].min().date()} "
        f"-> "
        f"{matches['date'].max().date()}"
    )

    print()
    print(
        "Matched games by season:"
    )

    print(
        matches[
            "season"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    matches.to_csv(
        OUTPUT_MATCHES,
        index=False,
    )

    # ========================================================
    # BET TABLE
    # ========================================================

    bets = build_bet_candidates(
        matches
    )

    print()
    print(
        f"Usable outcome observations: "
        f"{len(bets):,}"
    )

    # ========================================================
    # FULL EDGE → CLV DIAGNOSTIC BEFORE SELECTION
    # ========================================================

    print_edge_clv_diagnostic(
        bets,
        "ALL PERIODS — OPENING EDGE VS CLV",
    )

    # ========================================================
    # TUNE
    # ========================================================

    ranked, all_results = tune_thresholds(
        bets
    )

    all_results.to_csv(
        OUTPUT_THRESHOLDS,
        index=False,
    )

    print()
    print("==============================")
    print("TOP 20 OPENING BET RULES")
    print("==============================")
    print()

    display = (
        ranked
        .head(20)
        .copy()
    )

    percent_cols = [
        "win_rate",
        "avg_open_edge",
        "avg_open_ev",
        "avg_price_clv",
        "median_price_clv",
        "beat_close_rate",
        "avg_probability_clv",
        "positive_probability_clv_rate",
        "roi",
    ]

    for col in percent_cols:

        display[
            col
        ] *= 100.0

    print(
        display[
            [
                "rank",
                "edge_threshold",
                "ev_threshold",
                "bets",
                "wins",
                "win_rate",
                "avg_open_odds",
                "avg_close_odds",
                "avg_open_edge",
                "avg_open_ev",
                "avg_price_clv",
                "beat_close_rate",
                "avg_probability_clv",
                "profit",
                "roi",
                "max_drawdown",
            ]
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # WINNER
    # ========================================================

    best = ranked.iloc[
        0
    ]

    best_edge = float(
        best[
            "edge_threshold"
        ]
    )

    best_ev = float(
        best[
            "ev_threshold"
        ]
    )

    print()
    print("==============================")
    print("SELECTED OPENING RULE")
    print("==============================")

    print(
        f"Min probability edge: "
        f"{best_edge:.1%}"
    )

    print(
        f"Min expected value:   "
        f"{best_ev:.1%}"
    )

    print(
        f"Tuning avg CLV:       "
        f"{best['avg_price_clv']:+.2%}"
    )

    print(
        f"Tuning beat-close:    "
        f"{best['beat_close_rate']:.2%}"
    )

    print(
        f"Tuning ROI:           "
        f"{best['roi']:+.2%}"
    )

    # ========================================================
    # TEST PERIODS
    # ========================================================

    samples = [
        (
            "TUNING — 2021/22 TO 2022/23",
            TUNING_SEASONS,
        ),

        (
            "VALIDATION — 2023/24",
            VALIDATION_SEASONS,
        ),

        (
            "FINAL CHECK — 2024/25",
            FINAL_SEASONS,
        ),

        (
            "LOCKED TEST — 2025/26",
            LOCKED_SEASONS,
        ),
    ]

    selected_rows = []

    summary_rows = []

    for title, seasons in samples:

        period = bets[
            bets[
                "season"
            ].isin(
                seasons
            )
        ].copy()

        selected = filter_bets(
            period,
            best_edge,
            best_ev,
        )

        selected[
            "sample"
        ] = title

        selected_rows.append(
            selected
        )

        metrics = evaluate_bets(
            selected
        )

        print_metrics(
            title,
            metrics,
        )

        summary_rows.append(
            {
                "sample":
                    title,

                "edge_threshold":
                    best_edge,

                "ev_threshold":
                    best_ev,

                **metrics,
            }
        )

    # ========================================================
    # SAVE SELECTED BETS
    # ========================================================

    selected_bets = pd.concat(
        selected_rows,
        ignore_index=True,
    )

    selected_bets.to_csv(
        OUTPUT_BETS,
        index=False,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # ========================================================
    # LOCKED BREAKDOWN
    # ========================================================

    locked = selected_bets[
        selected_bets[
            "sample"
        ]
        ==
        "LOCKED TEST — 2025/26"
    ].copy()

    side_table = build_breakdown(
        locked,
        "bet_side",
    )

    print_breakdown(
        "2025/26 LOCKED TEST — BY BET TYPE",
        side_table,
        "bet_side",
    )

    league_table = build_breakdown(
        locked,
        "league",
    )

    print_breakdown(
        "2025/26 LOCKED TEST — BY LEAGUE",
        league_table,
        "league",
    )

    print_edge_clv_diagnostic(
        locked,
        "2025/26 LOCKED TEST — EDGE VS CLV",
    )

    print_odds_bands(
        locked
    )

    print_clv_bands(
        locked
    )

    # ========================================================
    # PERIOD-BY-PERIOD UNFILTERED SIGNAL CLV
    #
    # No threshold optimization here.
    # Just asks whether V5 disagreement predicts
    # market movement.
    # ========================================================

    print()
    print("=" * 135)
    print("UNFILTERED POSITIVE V5 EDGE — CLV BY PERIOD")
    print("=" * 135)

    rows = []

    for title, seasons in samples:

        period = bets[
            bets[
                "season"
            ].isin(
                seasons
            )
        ].copy()

        for min_edge in [
            0.025,
            0.05,
            0.075,
            0.10,
            0.15,
        ]:

            selected = period[
                period[
                    "opening_probability_edge"
                ]
                >=
                min_edge
            ].copy()

            metrics = evaluate_bets(
                selected
            )

            rows.append(
                {
                    "sample":
                        title,

                    "min_edge":
                        min_edge,

                    **metrics,
                }
            )

    diagnostic = pd.DataFrame(
        rows
    )

    for col in [
        "win_rate",
        "avg_open_edge",
        "avg_open_ev",
        "avg_price_clv",
        "median_price_clv",
        "beat_close_rate",
        "avg_probability_clv",
        "positive_probability_clv_rate",
        "roi",
    ]:

        diagnostic[
            col
        ] *= 100.0

    print(
        diagnostic[
            [
                "sample",
                "min_edge",
                "bets",
                "avg_open_edge",
                "avg_price_clv",
                "beat_close_rate",
                "avg_probability_clv",
                "profit",
                "roi",
                "max_drawdown",
            ]
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("==============================")
    print("V5 OPENING / CLV COMPLETE")
    print("==============================")

    print(
        "Frozen V5 unchanged ✅"
    )

    print(
        "Opening and closing prices "
        "paired on the same matches.csv rows ✅"
    )

    print(
        "Bet365 opening odds used "
        "as historical entry price ✅"
    )

    print(
        "Average closing odds used "
        "as CLV benchmark ✅"
    )

    print(
        "Rule selected using "
        "2021/22–2022/23 only ✅"
    )

    print(
        "2023/24, 2024/25 and 2025/26 "
        "not used for rule selection ✅"
    )

    print(
        "Flat 1-unit staking ✅"
    )

    print()
    print(
        "Matched data:"
    )

    print(
        OUTPUT_MATCHES
    )

    print()

    print(
        "Threshold results:"
    )

    print(
        OUTPUT_THRESHOLDS
    )

    print()

    print(
        "Selected bets:"
    )

    print(
        OUTPUT_BETS
    )

    print()

    print(
        "Summary:"
    )

    print(
        OUTPUT_SUMMARY
    )


if __name__ == "__main__":
    main()