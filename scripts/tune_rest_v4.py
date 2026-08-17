from pathlib import Path
import math

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

V3_FILE = (
    ROOT
    / "data"
    / "processed"
    / "opponent_strength_v3_predictions.csv"
)

REST_FILE = (
    ROOT
    / "data"
    / "processed"
    / "rest_features_v4.csv"
)

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "rest_v4_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "rest_v4_predictions.csv"
)


# =========================================================
# SPLITS
# =========================================================

TUNING_SEASONS = {
    "2122",
    "2223",
}

VALIDATION_SEASONS = {
    "2324",
    "2425",
}

CONFIRMATION_SEASON = {
    "2526",
}


# =========================================================
# SETTINGS
# =========================================================

MAX_GOALS = 10
EPS = 1e-12


FATIGUE_VALUES = [
    0.000,
    0.010,
    0.020,
    0.030,
    0.040,
    0.050,
    0.075,
    0.100,
]


# =========================================================
# POISSON
# =========================================================

FACTORIALS = np.array(
    [
        math.factorial(k)
        for k in range(
            MAX_GOALS + 1
        )
    ],
    dtype=float,
)


def poisson_probabilities(
    lambdas,
):

    lambdas = np.asarray(
        lambdas,
        dtype=float,
    )

    goals = np.arange(
        MAX_GOALS + 1
    )

    probs = (
        np.exp(
            -lambdas[:, None]
        )
        *
        (
            lambdas[:, None]
            ** goals[None, :]
        )
        /
        FACTORIALS[
            None,
            :
        ]
    )

    probs /= probs.sum(
        axis=1,
        keepdims=True,
    )

    return probs


def calculate_1x2_probs(
    home_lambda,
    away_lambda,
):

    hp = poisson_probabilities(
        home_lambda
    )

    ap = poisson_probabilities(
        away_lambda
    )

    away_cdf = np.cumsum(
        ap,
        axis=1,
    )

    home_win = np.zeros(
        len(home_lambda)
    )

    for h in range(
        1,
        MAX_GOALS + 1,
    ):

        home_win += (
            hp[:, h]
            *
            away_cdf[
                :,
                h - 1
            ]
        )

    home_cdf = np.cumsum(
        hp,
        axis=1,
    )

    away_win = np.zeros(
        len(home_lambda)
    )

    for a in range(
        1,
        MAX_GOALS + 1,
    ):

        away_win += (
            ap[:, a]
            *
            home_cdf[
                :,
                a - 1
            ]
        )

    draw = (
        hp * ap
    ).sum(
        axis=1
    )

    total = (
        home_win
        + draw
        + away_win
    )

    return np.column_stack(
        [
            home_win / total,
            draw / total,
            away_win / total,
        ]
    )


# =========================================================
# METRICS
# =========================================================

def result_classes(
    home_goals,
    away_goals,
):

    return np.where(
        home_goals > away_goals,
        0,
        np.where(
            home_goals == away_goals,
            1,
            2,
        ),
    )


def log_loss(
    y_true,
    probs,
):

    selected = probs[
        np.arange(
            len(y_true)
        ),
        y_true,
    ]

    selected = np.clip(
        selected,
        EPS,
        1.0,
    )

    return (
        -np.log(
            selected
        )
    ).mean()


def brier(
    y_true,
    probs,
):

    truth = np.zeros_like(
        probs
    )

    truth[
        np.arange(
            len(y_true)
        ),
        y_true,
    ] = 1.0

    return np.mean(
        np.sum(
            (
                probs
                - truth
            ) ** 2,
            axis=1,
        )
    )


def accuracy(
    y_true,
    probs,
):

    return (
        probs.argmax(
            axis=1
        )
        == y_true
    ).mean()


# =========================================================
# APPLY REST ADJUSTMENT
# =========================================================

def apply_rest_adjustment(
    df,
    fatigue,
):
    """
    Symmetric short-rest adjustment.

    If only the home team has <=3 days rest:
        home lambda is reduced
        away lambda is increased

    If only the away team has <=3 days rest:
        away lambda is reduced
        home lambda is increased

    If both or neither are on short rest:
        no adjustment.
    """

    home_lambda = (
        df[
            "home_lambda_v3"
        ]
        .astype(float)
        .copy()
    )

    away_lambda = (
        df[
            "away_lambda_v3"
        ]
        .astype(float)
        .copy()
    )

    home_tired = (
        df[
            "home_rest_3_or_less"
        ]
        == 1
    )

    away_tired = (
        df[
            "away_rest_3_or_less"
        ]
        == 1
    )

    home_only = (
        home_tired
        &
        ~away_tired
    )

    away_only = (
        away_tired
        &
        ~home_tired
    )

    # -----------------------------------------------------
    # HOME SHORT REST ONLY
    # -----------------------------------------------------

    home_lambda.loc[
        home_only
    ] *= (
        1.0
        - fatigue
    )

    away_lambda.loc[
        home_only
    ] *= (
        1.0
        + fatigue
    )

    # -----------------------------------------------------
    # AWAY SHORT REST ONLY
    # -----------------------------------------------------

    away_lambda.loc[
        away_only
    ] *= (
        1.0
        - fatigue
    )

    home_lambda.loc[
        away_only
    ] *= (
        1.0
        + fatigue
    )

    home_lambda = home_lambda.clip(
        lower=0.15,
        upper=4.50,
    )

    away_lambda = away_lambda.clip(
        lower=0.15,
        upper=4.50,
    )

    return (
        home_lambda,
        away_lambda,
    )


# =========================================================
# EVALUATE
# =========================================================

def evaluate(
    df,
    fatigue,
    seasons,
):

    season = (
        df[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    mask = (
        season.isin(
            seasons
        )
        &
        df[
            "home_lambda_v3"
        ].notna()
        &
        df[
            "away_lambda_v3"
        ].notna()
    )

    sub = df.loc[
        mask
    ].copy()

    home_lambda, away_lambda = (
        apply_rest_adjustment(
            sub,
            fatigue,
        )
    )

    probs = calculate_1x2_probs(
        home_lambda.to_numpy(),
        away_lambda.to_numpy(),
    )

    y = result_classes(
        sub[
            "home_goals"
        ].to_numpy(),
        sub[
            "away_goals"
        ].to_numpy(),
    )

    return {
        "games":
            len(sub),

        "accuracy":
            accuracy(
                y,
                probs,
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


# =========================================================
# PRINT COMPARISON
# =========================================================

def print_comparison(
    title,
    baseline,
    candidate,
):

    print()
    print("=" * 62)
    print(title)
    print("=" * 62)

    print(
        f"Games: "
        f"{candidate['games']:,}"
    )

    print()

    print(
        f"{'Metric':<15}"
        f"{'V3':>14}"
        f"{'V4':>14}"
        f"{'Change':>14}"
    )

    print("-" * 57)

    print(
        f"{'Accuracy':<15}"
        f"{baseline['accuracy']:>13.2%}"
        f"{candidate['accuracy']:>13.2%}"
        f"{candidate['accuracy'] - baseline['accuracy']:>+13.2%}"
    )

    print(
        f"{'Log Loss':<15}"
        f"{baseline['log_loss']:>14.5f}"
        f"{candidate['log_loss']:>14.5f}"
        f"{candidate['log_loss'] - baseline['log_loss']:>+14.5f}"
    )

    print(
        f"{'Brier':<15}"
        f"{baseline['brier']:>14.5f}"
        f"{candidate['brier']:>14.5f}"
        f"{candidate['brier'] - baseline['brier']:>+14.5f}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("==============================")
    print("TUNING REST MODEL V4")
    print("==============================")
    print()

    if not V3_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{V3_FILE}"
        )

    if not REST_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{REST_FILE}"
        )

    v3 = pd.read_csv(
        V3_FILE,
        parse_dates=[
            "date",
        ],
    )

    rest = pd.read_csv(
        REST_FILE,
        parse_dates=[
            "date",
        ],
    )

    v3[
        "season"
    ] = (
        v3[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    rest_keep = [
        "match_id",

        "home_days_since_last_match",
        "away_days_since_last_match",

        "home_rest_days_capped",
        "away_rest_days_capped",

        "home_rest_3_or_less",
        "away_rest_3_or_less",

        "home_matches_last_7d",
        "away_matches_last_7d",

        "home_matches_last_14d",
        "away_matches_last_14d",

        "home_rest_advantage",
        "home_matches_7d_advantage",
        "home_matches_14d_advantage",
    ]

    df = v3.merge(
        rest[
            rest_keep
        ],
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    print(
        f"V3 matches loaded: "
        f"{len(v3):,}"
    )

    print(
        f"Matches after rest merge: "
        f"{len(df):,}"
    )

    print()
    print(
        "V3 opponent strength: "
        "0.875"
    )

    # =====================================================
    # SHORT REST COVERAGE
    # =====================================================

    home_short = (
        df[
            "home_rest_3_or_less"
        ].mean()
    )

    away_short = (
        df[
            "away_rest_3_or_less"
        ].mean()
    )

    print(
        f"Home short-rest rate: "
        f"{home_short:.2%}"
    )

    print(
        f"Away short-rest rate: "
        f"{away_short:.2%}"
    )

    # =====================================================
    # TUNE FATIGUE
    # =====================================================

    rows = []

    for fatigue in FATIGUE_VALUES:

        metrics = evaluate(
            df,
            fatigue,
            TUNING_SEASONS,
        )

        rows.append({
            "fatigue":
                fatigue,

            "games":
                metrics[
                    "games"
                ],

            "accuracy":
                metrics[
                    "accuracy"
                ],

            "log_loss":
                metrics[
                    "log_loss"
                ],

            "brier":
                metrics[
                    "brier"
                ],
        })

    results = pd.DataFrame(
        rows
    )

    results = (
        results
        .sort_values(
            [
                "log_loss",
                "brier",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    results[
        "rank"
    ] = (
        np.arange(
            len(results)
        )
        + 1
    )

    results.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    print()
    print("==============================")
    print("REST ADJUSTMENT RESULTS")
    print("==============================")

    display = results.copy()

    display[
        "accuracy"
    ] *= 100.0

    print(
        display[
            [
                "rank",
                "fatigue",
                "games",
                "log_loss",
                "brier",
                "accuracy",
            ]
        ]
        .round(6)
        .to_string(
            index=False
        )
    )

    best = results.iloc[
        0
    ]

    best_fatigue = float(
        best[
            "fatigue"
        ]
    )

    print()
    print("==============================")
    print("WINNING REST ADJUSTMENT")
    print("==============================")

    print(
        f"Fatigue adjustment: "
        f"{best_fatigue:.1%}"
    )

    print(
        f"Tuning LL:         "
        f"{best['log_loss']:.5f}"
    )

    # =====================================================
    # TUNING
    # =====================================================

    baseline_tuning = evaluate(
        df,
        0.0,
        TUNING_SEASONS,
    )

    v4_tuning = evaluate(
        df,
        best_fatigue,
        TUNING_SEASONS,
    )

    print_comparison(
        "TUNING — 2021/22 TO 2022/23",
        baseline_tuning,
        v4_tuning,
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    baseline_validation = evaluate(
        df,
        0.0,
        VALIDATION_SEASONS,
    )

    v4_validation = evaluate(
        df,
        best_fatigue,
        VALIDATION_SEASONS,
    )

    print_comparison(
        "VALIDATION — 2023/24 TO 2024/25",
        baseline_validation,
        v4_validation,
    )

    # =====================================================
    # CONFIRMATION
    # =====================================================

    baseline_final = evaluate(
        df,
        0.0,
        CONFIRMATION_SEASON,
    )

    v4_final = evaluate(
        df,
        best_fatigue,
        CONFIRMATION_SEASON,
    )

    print_comparison(
        "2025/26 CONFIRMATION",
        baseline_final,
        v4_final,
    )

    # =====================================================
    # FINAL BY LEAGUE
    # =====================================================

    print()
    print("=" * 90)
    print("2025/26 — REST EFFECT BY LEAGUE")
    print("=" * 90)

    league_rows = []

    for league, group in (
        df.groupby(
            "league"
        )
    ):

        baseline = evaluate(
            group,
            0.0,
            CONFIRMATION_SEASON,
        )

        candidate = evaluate(
            group,
            best_fatigue,
            CONFIRMATION_SEASON,
        )

        league_rows.append({
            "league":
                league,

            "games":
                candidate[
                    "games"
                ],

            "v3_ll":
                baseline[
                    "log_loss"
                ],

            "v4_ll":
                candidate[
                    "log_loss"
                ],

            "ll_change":
                (
                    candidate[
                        "log_loss"
                    ]
                    -
                    baseline[
                        "log_loss"
                    ]
                ),

            "v3_brier":
                baseline[
                    "brier"
                ],

            "v4_brier":
                candidate[
                    "brier"
                ],

            "v3_acc":
                baseline[
                    "accuracy"
                ],

            "v4_acc":
                candidate[
                    "accuracy"
                ],
        })

    league_table = pd.DataFrame(
        league_rows
    )

    for col in [
        "v3_acc",
        "v4_acc",
    ]:

        league_table[
            col
        ] *= 100.0

    print(
        league_table
        .round(5)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # SAVE V4
    # =====================================================

    home_lambda, away_lambda = (
        apply_rest_adjustment(
            df,
            best_fatigue,
        )
    )

    output = df.copy()

    output[
        "home_lambda_v4"
    ] = home_lambda

    output[
        "away_lambda_v4"
    ] = away_lambda

    probs = calculate_1x2_probs(
        output[
            "home_lambda_v4"
        ].to_numpy(),
        output[
            "away_lambda_v4"
        ].to_numpy(),
    )

    output[
        "p_home_v4"
    ] = probs[:, 0]

    output[
        "p_draw_v4"
    ] = probs[:, 1]

    output[
        "p_away_v4"
    ] = probs[:, 2]

    output[
        "rest_fatigue_v4"
    ] = best_fatigue

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print()
    print("==============================")
    print("V4 REST TUNING COMPLETE")
    print("==============================")

    if best_fatigue == 0.0:

        print(
            "Short-rest adjustment rejected."
        )

    else:

        print(
            "Short-rest adjustment improved "
            "the tuning objective."
        )

    print()
    print(
        "Fatigue selected using "
        "2021/22–2022/23 only ✅"
    )

    print(
        "Later seasons used only "
        "for confirmation ✅"
    )

    print()
    print(
        f"Tuning results:"
        f"\n{OUTPUT_RESULTS}"
    )

    print()
    print(
        f"V4 predictions:"
        f"\n{OUTPUT_PREDICTIONS}"
    )


if __name__ == "__main__":
    main()