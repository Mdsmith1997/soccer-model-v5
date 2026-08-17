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
    / "v5_market_comparison.csv"
)

OUTPUT_FOLDS = (
    ROOT
    / "data"
    / "processed"
    / "walkforward_residual_v5_folds.csv"
)

OUTPUT_BETS = (
    ROOT
    / "data"
    / "processed"
    / "walkforward_residual_v5_bets.csv"
)

OUTPUT_ALPHA = (
    ROOT
    / "data"
    / "processed"
    / "walkforward_residual_v5_alpha_history.csv"
)


# ============================================================
# LOCKED BETTING RULE
#
# IMPORTANT:
# This was selected BEFORE this walk-forward analysis.
# Do not retune it here.
# ============================================================
# ============================================================
# LOCKED BETTING RULE
#
# The earlier Residual V5 >= 4% rule was selected with
# alpha = 0.25.
#
# residual_edge = alpha * raw_v5_edge
#
# 0.04 / 0.25 = 0.16
#
# Therefore the underlying frozen selection criterion is:
#
# RAW V5 EDGE >= 16%
#
# Walk-forward alpha is still tuned independently on prior
# seasons only and is used to create the residual probability
# and EV estimate.
# ============================================================

RAW_EDGE_THRESHOLD = 0.16

ALPHAS = np.round(
    np.arange(
        0.00,
        1.01,
        0.025,
    ),
    3,
)

EPS = 1e-12


# ============================================================
# WALK-FORWARD TEST SEASONS
#
# For each test season, alpha is tuned ONLY on seasons that
# occurred before it.
#
# Example:
#
# train <= 2019/20 -> test 2020/21
# train <= 2020/21 -> test 2021/22
# ...
# train <= 2024/25 -> test 2025/26
# ============================================================

TEST_SEASONS = [
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
    "2526",
]


# ============================================================
# HELPERS
# ============================================================

def season_string(
    series,
):

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


def normalize_probs(
    probs,
):

    probs = np.clip(
        probs,
        EPS,
        None,
    )

    return (
        probs
        /
        probs.sum(
            axis=1,
            keepdims=True,
        )
    )


def actual_classes(
    df,
):

    return np.where(
        df[
            "home_goals"
        ].to_numpy()
        >
        df[
            "away_goals"
        ].to_numpy(),
        0,
        np.where(
            df[
                "home_goals"
            ].to_numpy()
            ==
            df[
                "away_goals"
            ].to_numpy(),
            1,
            2,
        ),
    )


def log_loss(
    y_true,
    probs,
):

    chosen = probs[
        np.arange(
            len(y_true)
        ),
        y_true,
    ]

    return float(
        -np.log(
            np.clip(
                chosen,
                EPS,
                1.0,
            )
        ).mean()
    )


def brier(
    y_true,
    probs,
):

    truth = np.zeros_like(
        probs,
        dtype=float,
    )

    truth[
        np.arange(
            len(y_true)
        ),
        y_true,
    ] = 1.0

    return float(
        np.mean(
            np.sum(
                (
                    probs
                    -
                    truth
                )
                ** 2,
                axis=1,
            )
        )
    )


# ============================================================
# EXACT RESIDUAL SHRINKAGE
#
# Reproduces tune_market_residual_v5.py
# ============================================================

def apply_alpha(
    df,
    alpha,
):

    out = df.copy()

    out[
        "resid_p_home"
    ] = (
        out[
            "market_nv_home"
        ]
        +
        alpha
        *
        (
            out[
                "p_home_v5"
            ]
            -
            out[
                "market_nv_home"
            ]
        )
    )

    out[
        "resid_p_draw"
    ] = (
        out[
            "market_nv_draw"
        ]
        +
        alpha
        *
        (
            out[
                "p_draw_v5"
            ]
            -
            out[
                "market_nv_draw"
            ]
        )
    )

    out[
        "resid_p_away"
    ] = (
        out[
            "market_nv_away"
        ]
        +
        alpha
        *
        (
            out[
                "p_away_v5"
            ]
            -
            out[
                "market_nv_away"
            ]
        )
    )

    probs = normalize_probs(
        out[
            [
                "resid_p_home",
                "resid_p_draw",
                "resid_p_away",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    out[
        "resid_p_home"
    ] = probs[
        :,
        0
    ]

    out[
        "resid_p_draw"
    ] = probs[
        :,
        1
    ]

    out[
        "resid_p_away"
    ] = probs[
        :,
        2
    ]

    return out


# ============================================================
# LOAD
# ============================================================

def load_data():

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
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Historical market file missing columns:\n"
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

    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        errors="coerce",
    )

    numeric_cols = [
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

    valid = (
        df[
            numeric_cols
        ]
        .notna()
        .all(
            axis=1
        )
    )

    df = df.loc[
        valid
    ].copy()

    return df


# ============================================================
# TUNE ALPHA ON PRIOR SEASONS ONLY
# ============================================================

def tune_alpha(
    train,
):

    if train.empty:

        raise ValueError(
            "Walk-forward training set is empty."
        )

    y = actual_classes(
        train
    )

    rows = []

    for alpha in ALPHAS:

        adjusted = apply_alpha(
            train,
            alpha,
        )

        probs = adjusted[
            [
                "resid_p_home",
                "resid_p_draw",
                "resid_p_away",
            ]
        ].to_numpy(
            dtype=float
        )

        rows.append(
            {
                "alpha":
                    float(
                        alpha
                    ),

                "games":
                    len(
                        adjusted
                    ),

                "log_loss":
                    log_loss(
                        y,
                        probs,
                    ),

                "brier":
                    brier(
                        y,
                        probs,
                    ),
            }
        )

    results = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # EXACT WINNER RULE:
    #
    # 1. lowest log loss
    # 2. lowest Brier
    # --------------------------------------------------------

    results = (
        results
        .sort_values(
            [
                "log_loss",
                "brier",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    best = results.iloc[
        0
    ]

    return (
        float(
            best[
                "alpha"
            ]
        ),
        results,
    )


# ============================================================
# BUILD TEST-SEASON BETS
# ============================================================

def build_test_bets(
    test,
    alpha,
):

    adjusted = apply_alpha(
        test,
        alpha,
    )

    definitions = [
        (
            "HOME",
            "p_home_v5",
            "resid_p_home",
            "market_nv_home",
            "market_home_odds",
        ),
        (
            "DRAW",
            "p_draw_v5",
            "resid_p_draw",
            "market_nv_draw",
            "market_draw_odds",
        ),
        (
            "AWAY",
            "p_away_v5",
            "resid_p_away",
            "market_nv_away",
            "market_away_odds",
        ),
    ]

    frames = []

    for (
        selection,
        raw_prob_col,
        residual_prob_col,
        market_col,
        odds_col,
    ) in definitions:

        x = adjusted[
            [
                "match_id",
                "date",
                "season",
                "league",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                raw_prob_col,
                residual_prob_col,
                market_col,
                odds_col,
            ]
        ].copy()

        x = x.rename(
            columns={
                raw_prob_col:
                    "raw_v5_probability",

                residual_prob_col:
                    "residual_probability",

                market_col:
                    "market_probability",

                odds_col:
                    "odds",
            }
        )

        x[
            "selection"
        ] = selection

        x[
            "alpha"
        ] = alpha

        # ====================================================
        # RAW V5 EDGE
        #
        # This is the LOCKED selection criterion.
        # ====================================================

        x[
            "raw_v5_edge"
        ] = (
            x[
                "raw_v5_probability"
            ]
            -
            x[
                "market_probability"
            ]
        )

        # ====================================================
        # RESIDUAL EDGE
        #
        # Informational / audit only.
        # This changes with the fold-specific alpha.
        # ====================================================

        x[
            "residual_edge"
        ] = (
            x[
                "residual_probability"
            ]
            -
            x[
                "market_probability"
            ]
        )

        # ====================================================
        # RESIDUAL MODEL EV
        #
        # Selection comes from raw V5 edge.
        # Valuation comes from the fold-specific residual
        # probability.
        # ====================================================

        x[
            "model_ev"
        ] = (
            x[
                "residual_probability"
            ]
            *
            x[
                "odds"
            ]
            -
            1.0
        )

        frames.append(
            x
        )

    candidates = pd.concat(
        frames,
        ignore_index=True,
    )

    # ========================================================
    # LOCKED RAW-V5 EDGE RULE
    # ========================================================

    candidates = candidates.loc[
        candidates[
            "raw_v5_edge"
        ]
        >=
        RAW_EDGE_THRESHOLD
    ].copy()

    if candidates.empty:

        return candidates

    # ========================================================
    # ONE BET PER MATCH
    #
    # If multiple outcomes clear the same locked raw-edge
    # threshold, choose:
    #
    # 1. highest residual-model EV
    # 2. highest raw V5 edge
    # 3. highest residual edge
    # 4. lower odds
    # ========================================================

    candidates = (
        candidates
        .sort_values(
            [
                "match_id",
                "model_ev",
                "raw_v5_edge",
                "residual_edge",
                "odds",
            ],
            ascending=[
                True,
                False,
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
    # ACTUAL OUTCOME
    # ========================================================

    candidates[
        "actual_outcome"
    ] = np.where(
        candidates[
            "home_goals"
        ]
        >
        candidates[
            "away_goals"
        ],
        "HOME",
        np.where(
            candidates[
                "home_goals"
            ]
            ==
            candidates[
                "away_goals"
            ],
            "DRAW",
            "AWAY",
        ),
    )

    candidates[
        "won"
    ] = (
        candidates[
            "selection"
        ]
        ==
        candidates[
            "actual_outcome"
        ]
    ).astype(
        int
    )

    candidates[
        "profit"
    ] = np.where(
        candidates[
            "won"
        ]
        ==
        1,
        candidates[
            "odds"
        ]
        -
        1.0,
        -1.0,
    )

    return candidates


# ============================================================
# FOLD SUMMARY
# ============================================================

def summarize_fold(
    test_season,
    train,
    bets,
    alpha,
):

    if bets.empty:

        return {
            "test_season":
                test_season,

            "train_start":
                train[
                    "season"
                ].min(),

            "train_end":
                train[
                    "season"
                ].max(),

            "train_games":
                len(
                    train
                ),

            "alpha":
                alpha,

            "market_share":
                1.0
                -
                alpha,

            "bets":
                0,

            "wins":
                0,

            "win_rate":
                np.nan,

            "avg_odds":
                np.nan,

            "avg_raw_v5_edge":
                np.nan,

            "avg_residual_edge":
                np.nan,

            "avg_model_ev":
                np.nan,

            "profit_units":
                0.0,

            "roi":
                np.nan,
        }

    n = len(
        bets
    )

    wins = int(
        bets[
            "won"
        ].sum()
    )

    profit = float(
        bets[
            "profit"
        ].sum()
    )

    return {
        "test_season":
            test_season,

        "train_start":
            train[
                "season"
            ].min(),

        "train_end":
            train[
                "season"
            ].max(),

        "train_games":
            len(
                train
            ),

        "alpha":
            alpha,

        "market_share":
            1.0
            -
            alpha,

        "bets":
            n,

        "wins":
            wins,

        "win_rate":
            wins
            /
            n,

        "avg_odds":
            bets[
                "odds"
            ].mean(),

        "avg_raw_v5_edge":
            bets[
                "raw_v5_edge"
            ].mean(),

        "avg_residual_edge":
            bets[
                "residual_edge"
            ].mean(),

        "avg_model_ev":
            bets[
                "model_ev"
            ].mean(),

        "profit_units":
            profit,

        "roi":
            profit
            /
            n,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 135)
    print(
        "TRUE WALK-FORWARD RESIDUAL V5 BETTING BACKTEST"
    )
    print("=" * 135)

    print()
    print(
    "Locked raw V5 edge threshold:",
    f"{RAW_EDGE_THRESHOLD:.1%}",
)

    print(
    "Residual edge threshold:",
    "NONE — audit only",
)

    print(
        "Alpha selection metric:",
        "LOG LOSS (Brier tie-break)",
    )

    print(
        "One bet per match:",
        "YES",
    )

    print(
        "Future seasons used in prior folds:",
        "NO ✅",
    )

    # ========================================================
    # LOAD
    # ========================================================

    df = load_data()

    available_seasons = (
        sorted(
            df[
                "season"
            ]
            .dropna()
            .unique()
        )
    )

    print()
    print(
        "Historical matches:",
        f"{len(df):,}",
    )

    print(
        "Available seasons:",
        available_seasons,
    )

    # ========================================================
    # WALK FORWARD
    # ========================================================

    fold_rows = []

    all_bets = []

    alpha_history = []

    for test_season in TEST_SEASONS:

        if (
            test_season
            not in available_seasons
        ):

            print()
            print(
                "Skipping missing season:",
                test_season,
            )

            continue

        # ----------------------------------------------------
        # Use dates/seasons strictly before test season.
        #
        # Since season codes are chronological in this
        # historical set, compare using available-season
        # position rather than numeric season arithmetic.
        # ----------------------------------------------------

        test_position = (
            available_seasons.index(
                test_season
            )
        )

        prior_seasons = (
            available_seasons[
                :test_position
            ]
        )

        train = df.loc[
            df[
                "season"
            ].isin(
                prior_seasons
            )
        ].copy()

        test = df.loc[
            df[
                "season"
            ]
            ==
            test_season
        ].copy()

        if train.empty:

            print()
            print(
                "Skipping",
                test_season,
                "because there is no prior training history."
            )

            continue

        print()
        print("=" * 135)

        print(
            "TEST SEASON:",
            test_season,
        )

        print("=" * 135)

        print(
            "Training seasons:",
            prior_seasons,
        )

        print(
            "Training matches:",
            f"{len(train):,}",
        )

        print(
            "Test matches:",
            f"{len(test):,}",
        )

        # ====================================================
        # TUNE ALPHA
        # ====================================================

        (
            best_alpha,
            alpha_results,
        ) = tune_alpha(
            train
        )

        best_metrics = (
            alpha_results.iloc[
                0
            ]
        )

        print(
            "Selected alpha:",
            f"{best_alpha:.3f}",
        )

        print(
            "Market share:",
            f"{1.0 - best_alpha:.1%}",
        )

        print(
            "Training log loss:",
            f"{best_metrics['log_loss']:.6f}",
        )

        print(
            "Training Brier:",
            f"{best_metrics['brier']:.6f}",
        )

        # ----------------------------------------------------
        # Save complete alpha search history for this fold.
        # ----------------------------------------------------

        alpha_temp = (
            alpha_results.copy()
        )

        alpha_temp.insert(
            0,
            "test_season",
            test_season,
        )

        alpha_history.append(
            alpha_temp
        )

        # ====================================================
        # APPLY TO UNSEEN TEST SEASON
        # ====================================================

        bets = build_test_bets(
            test,
            best_alpha,
        )

        if not bets.empty:

            bets[
                "walkforward_test_season"
            ] = test_season

            bets[
                "walkforward_alpha"
            ] = best_alpha

            all_bets.append(
                bets
            )

        fold_summary = summarize_fold(
            test_season,
            train,
            bets,
            best_alpha,
        )

        fold_rows.append(
            fold_summary
        )

        print()
        print(
            "Qualifying bets:",
            fold_summary[
                "bets"
            ],
        )

        print(
            "Wins:",
            fold_summary[
                "wins"
            ],
        )

        if pd.notna(
            fold_summary[
                "win_rate"
            ]
        ):

            print(
                "Win rate:",
                f"{fold_summary['win_rate']:.2%}",
            )

        print(
            "Profit:",
            f"{fold_summary['profit_units']:+.2f}",
            "units",
        )

        if pd.notna(
            fold_summary[
                "roi"
            ]
        ):

            print(
                "ROI:",
                f"{fold_summary['roi']:+.2%}",
            )

    # ========================================================
    # RESULTS
    # ========================================================

    folds = pd.DataFrame(
        fold_rows
    )

    if all_bets:

        bets_output = pd.concat(
            all_bets,
            ignore_index=True,
        )

    else:

        bets_output = pd.DataFrame()

    if alpha_history:

        alpha_output = pd.concat(
            alpha_history,
            ignore_index=True,
        )

    else:

        alpha_output = pd.DataFrame()

    # ========================================================
    # DISPLAY FOLDS
    # ========================================================

    print()
    print("=" * 135)
    print(
        "WALK-FORWARD SEASON RESULTS"
    )
    print("=" * 135)

    if folds.empty:

        print(
            "No completed folds."
        )

    else:

        display = folds.copy()

        for col in [
    "market_share",
    "win_rate",
    "avg_raw_v5_edge",
    "avg_residual_edge",
    "avg_model_ev",
    "roi",
]:

            display[
                col
            ] *= 100.0

        print(
            display[
                [
    "test_season",
    "train_games",
    "alpha",
    "market_share",
    "bets",
    "wins",
    "win_rate",
    "avg_odds",
    "avg_raw_v5_edge",
    "avg_residual_edge",
    "avg_model_ev",
    "profit_units",
    "roi",
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
    # AGGREGATE WALK-FORWARD PERFORMANCE
    # ========================================================

    print()
    print("=" * 135)
    print(
        "AGGREGATE TRUE OUT-OF-SAMPLE PERFORMANCE"
    )
    print("=" * 135)

    if bets_output.empty:

        print(
            "No qualifying walk-forward bets."
        )

    else:

        total_bets = len(
            bets_output
        )

        total_wins = int(
            bets_output[
                "won"
            ].sum()
        )

        total_profit = float(
            bets_output[
                "profit"
            ].sum()
        )

        total_roi = (
            total_profit
            /
            total_bets
        )

        positive_folds = int(
            (
                folds[
                    "roi"
                ]
                >
                0
            ).sum()
        )

        valid_folds = int(
            folds[
                "roi"
            ]
            .notna()
            .sum()
        )

        print(
            "Bets:",
            total_bets,
        )

        print(
            "Wins:",
            total_wins,
        )

        print(
            "Win rate:",
            f"{total_wins / total_bets:.2%}",
        )

        print(
            "Average odds:",
            f"{bets_output['odds'].mean():.3f}",
        )

        print(
    "Average raw V5 edge:",
    f"{bets_output['raw_v5_edge'].mean():.2%}",
)

        print(
    "Average residual edge:",
    f"{bets_output['residual_edge'].mean():.2%}",
)

        print(
            "Average model EV:",
            f"{bets_output['model_ev'].mean():.2%}",
        )

        print(
            "Profit:",
            f"{total_profit:+.2f}",
            "units",
        )

        print(
            "ROI:",
            f"{total_roi:+.2%}",
        )

        print(
            "Positive test seasons:",
            positive_folds,
            "/",
            valid_folds,
        )

        if valid_folds:

            print(
                "Worst season ROI:",
                f"{folds['roi'].min():+.2%}",
            )

            print(
                "Best season ROI:",
                f"{folds['roi'].max():+.2%}",
            )

    # ========================================================
    # SIDE BREAKDOWN
    # ========================================================

    if not bets_output.empty:

        print()
        print("=" * 135)
        print(
            "TRUE OUT-OF-SAMPLE BY BET SIDE"
        )
        print("=" * 135)

        side_rows = []

        for side, group in (
            bets_output
            .groupby(
                "selection"
            )
        ):

            n = len(
                group
            )

            profit = float(
                group[
                    "profit"
                ].sum()
            )

            side_rows.append(
                {
                    "side":
                        side,

                    "bets":
                        n,

                    "wins":
                        int(
                            group[
                                "won"
                            ].sum()
                        ),

                    "win_rate":
                        group[
                            "won"
                        ].mean(),

                    "avg_odds":
                        group[
                            "odds"
                        ].mean(),

                    "avg_raw_v5_edge":
                        group[
                            "raw_v5_edge"
                        ].mean(),

                    "avg_residual_edge":
                        group[
                            "residual_edge"
                        ].mean(),

                    "profit":
                        profit,

                    "roi":
                        profit
                        /
                        n,
                }
            )

        side_df = pd.DataFrame(
            side_rows
        )

        print(
            side_df
            .round(
                4
            )
            .to_string(
                index=False
            )
        )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FOLDS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds.to_csv(
        OUTPUT_FOLDS,
        index=False,
    )

    bets_output.to_csv(
        OUTPUT_BETS,
        index=False,
    )

    alpha_output.to_csv(
        OUTPUT_ALPHA,
        index=False,
    )

    print()
    print("=" * 135)
    print("SAVED")
    print("=" * 135)

    print(
        OUTPUT_FOLDS
    )

    print(
        OUTPUT_BETS
    )

    print(
        OUTPUT_ALPHA
    )

    print()


if __name__ == "__main__":
    main()