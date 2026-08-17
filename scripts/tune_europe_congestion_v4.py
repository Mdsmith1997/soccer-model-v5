from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

V3_FILE = (
    ROOT
    / "data"
    / "processed"
    / "opponent_strength_v3_predictions.csv"
)

WORKLOAD_FILE = (
    ROOT
    / "data"
    / "processed"
    / "europe_workload_v4.csv"
)

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "europe_congestion_v4_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "europe_congestion_v4_predictions.csv"
)


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


MAX_GOALS = 10
EPS = 1e-12


# Generic Europe penalty.
EUROPE_PENALTIES = [
    0.000,
    0.010,
    0.020,
    0.030,
    0.040,
    0.050,
    0.075,
]

# Extra penalty if previous European match was away.
AWAY_TRIP_PENALTIES = [
    0.000,
    0.010,
    0.020,
    0.030,
    0.040,
    0.050,
]


FACTORIALS = np.array(
    [
        math.factorial(k)
        for k in range(MAX_GOALS + 1)
    ],
    dtype=float,
)


def poisson_probabilities(lambdas):

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
        hp
        * ap
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

    chosen = probs[
        np.arange(
            len(y_true)
        ),
        y_true,
    ]

    chosen = np.clip(
        chosen,
        EPS,
        1.0,
    )

    return (
        -np.log(
            chosen
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


def apply_adjustment(
    df,
    europe_penalty,
    away_trip_penalty,
):
    """
    Apply congestion effect only when one side
    played Europe within 4 days.

    A team coming off Europe:
        own lambda decreases
        opponent lambda increases

    If that European match was away,
    apply an additional penalty.

    If both sides played Europe recently,
    each side receives its own adjustment.
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

    # -----------------------------------------------------
    # HOME TEAM CAME OFF EUROPE
    # -----------------------------------------------------

    home_europe = (
        df[
            "home_played_europe_last_4d"
        ]
        == 1
    )

    home_away_trip = (
        df[
            "home_europe_away_last_4d"
        ]
        == 1
    )

    home_total_penalty = (
        europe_penalty
        +
        away_trip_penalty
        * home_away_trip.astype(float)
    ).clip(
        upper=0.20
    )

    home_lambda.loc[
        home_europe
    ] *= (
        1.0
        -
        home_total_penalty.loc[
            home_europe
        ]
    )

    away_lambda.loc[
        home_europe
    ] *= (
        1.0
        +
        home_total_penalty.loc[
            home_europe
        ]
    )

    # -----------------------------------------------------
    # AWAY TEAM CAME OFF EUROPE
    # -----------------------------------------------------

    away_europe = (
        df[
            "away_played_europe_last_4d"
        ]
        == 1
    )

    away_away_trip = (
        df[
            "away_europe_away_last_4d"
        ]
        == 1
    )

    away_total_penalty = (
        europe_penalty
        +
        away_trip_penalty
        * away_away_trip.astype(float)
    ).clip(
        upper=0.20
    )

    away_lambda.loc[
        away_europe
    ] *= (
        1.0
        -
        away_total_penalty.loc[
            away_europe
        ]
    )

    home_lambda.loc[
        away_europe
    ] *= (
        1.0
        +
        away_total_penalty.loc[
            away_europe
        ]
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


def evaluate(
    df,
    seasons,
    europe_penalty,
    away_trip_penalty,
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
        apply_adjustment(
            sub,
            europe_penalty,
            away_trip_penalty,
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

    affected = (
        (
            sub[
                "home_played_europe_last_4d"
            ]
            == 1
        )
        |
        (
            sub[
                "away_played_europe_last_4d"
            ]
            == 1
        )
    ).sum()

    away_trip_affected = (
        (
            sub[
                "home_europe_away_last_4d"
            ]
            == 1
        )
        |
        (
            sub[
                "away_europe_away_last_4d"
            ]
            == 1
        )
    ).sum()

    return {
        "games":
            len(sub),

        "affected_games":
            int(affected),

        "away_trip_games":
            int(away_trip_affected),

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


def print_comparison(
    title,
    baseline,
    candidate,
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"Games: "
        f"{candidate['games']:,}"
    )

    print(
        f"Europe-affected games: "
        f"{candidate['affected_games']:,}"
    )

    print(
        f"Away-Europe-trip games: "
        f"{candidate['away_trip_games']:,}"
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


def main():

    print()
    print("==============================")
    print("TUNING EUROPE CONGESTION V4")
    print("==============================")
    print()

    v3 = pd.read_csv(
        V3_FILE,
        parse_dates=[
            "date",
        ],
    )

    workload = pd.read_csv(
        WORKLOAD_FILE,
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

    keep = [
        "match_id",

        "home_played_europe_last_3d",
        "away_played_europe_last_3d",

        "home_played_europe_last_4d",
        "away_played_europe_last_4d",

        "home_europe_away_last_4d",
        "away_europe_away_last_4d",

        "home_europe_matches_last_7d",
        "away_europe_matches_last_7d",

        "home_tracked_matches_last_7d",
        "away_tracked_matches_last_7d",
    ]

    df = v3.merge(
        workload[
            keep
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
        f"Matches after workload merge: "
        f"{len(df):,}"
    )

    print()
    print(
        "Clean full-Europe coverage:"
    )

    print(
        "Tuning:     2021/22–2022/23"
    )

    print(
        "Validation: 2023/24"
    )

    print(
        "Final:      2024/25"
    )

    # =====================================================
    # GRID SEARCH
    # =====================================================

    rows = []

    for europe_penalty in (
        EUROPE_PENALTIES
    ):

        for away_trip_penalty in (
            AWAY_TRIP_PENALTIES
        ):

            metrics = evaluate(
                df,
                TUNING_SEASONS,
                europe_penalty,
                away_trip_penalty,
            )

            rows.append(
                {
                    "europe_penalty":
                        europe_penalty,

                    "away_trip_penalty":
                        away_trip_penalty,

                    "games":
                        metrics[
                            "games"
                        ],

                    "affected_games":
                        metrics[
                            "affected_games"
                        ],

                    "away_trip_games":
                        metrics[
                            "away_trip_games"
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
                }
            )

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

    # =====================================================
    # TOP RESULTS
    # =====================================================

    print()
    print("==============================")
    print("TOP 20 EUROPE ADJUSTMENTS")
    print("==============================")

    display = (
        results
        .head(20)
        .copy()
    )

    display[
        "accuracy"
    ] *= 100.0

    print(
        display[
            [
                "rank",
                "europe_penalty",
                "away_trip_penalty",
                "affected_games",
                "away_trip_games",
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

    # =====================================================
    # WINNER
    # =====================================================

    best = results.iloc[
        0
    ]

    best_europe = float(
        best[
            "europe_penalty"
        ]
    )

    best_away = float(
        best[
            "away_trip_penalty"
        ]
    )

    print()
    print("==============================")
    print("WINNING EUROPE ADJUSTMENT")
    print("==============================")

    print(
        f"Base Europe penalty: "
        f"{best_europe:.1%}"
    )

    print(
        f"Extra away-trip penalty: "
        f"{best_away:.1%}"
    )

    print(
        f"Tuning LL: "
        f"{best['log_loss']:.5f}"
    )

    # =====================================================
    # TIME SPLITS
    # =====================================================

    baseline_tune = evaluate(
        df,
        TUNING_SEASONS,
        0.0,
        0.0,
    )

    winner_tune = evaluate(
        df,
        TUNING_SEASONS,
        best_europe,
        best_away,
    )

    print_comparison(
        "TUNING — 2021/22 TO 2022/23",
        baseline_tune,
        winner_tune,
    )

    baseline_validation = evaluate(
        df,
        VALIDATION_SEASONS,
        0.0,
        0.0,
    )

    winner_validation = evaluate(
        df,
        VALIDATION_SEASONS,
        best_europe,
        best_away,
    )

    print_comparison(
        "VALIDATION — 2023/24",
        baseline_validation,
        winner_validation,
    )

    baseline_final = evaluate(
        df,
        FINAL_SEASONS,
        0.0,
        0.0,
    )

    winner_final = evaluate(
        df,
        FINAL_SEASONS,
        best_europe,
        best_away,
    )

    print_comparison(
        "FINAL CHECK — 2024/25",
        baseline_final,
        winner_final,
    )

    # =====================================================
    # FINAL BY LEAGUE
    # =====================================================

    print()
    print("=" * 98)
    print("2024/25 — BY LEAGUE")
    print("=" * 98)

    league_rows = []

    for league, group in (
        df.groupby(
            "league"
        )
    ):

        baseline = evaluate(
            group,
            FINAL_SEASONS,
            0.0,
            0.0,
        )

        winner = evaluate(
            group,
            FINAL_SEASONS,
            best_europe,
            best_away,
        )

        league_rows.append(
            {
                "league":
                    league,

                "games":
                    winner[
                        "games"
                    ],

                "europe_games":
                    winner[
                        "affected_games"
                    ],

                "away_trip_games":
                    winner[
                        "away_trip_games"
                    ],

                "v3_ll":
                    baseline[
                        "log_loss"
                    ],

                "v4_ll":
                    winner[
                        "log_loss"
                    ],

                "ll_change":
                    (
                        winner[
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
                    winner[
                        "brier"
                    ],

                "v3_acc":
                    baseline[
                        "accuracy"
                    ],

                "v4_acc":
                    winner[
                        "accuracy"
                    ],
            }
        )

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
    # SAVE WINNER
    # =====================================================

    home_lambda, away_lambda = (
        apply_adjustment(
            df,
            best_europe,
            best_away,
        )
    )

    output = df.copy()

    output[
        "home_lambda_v4_europe"
    ] = home_lambda

    output[
        "away_lambda_v4_europe"
    ] = away_lambda

    probs = calculate_1x2_probs(
        home_lambda.to_numpy(),
        away_lambda.to_numpy(),
    )

    output[
        "p_home_v4_europe"
    ] = probs[:, 0]

    output[
        "p_draw_v4_europe"
    ] = probs[:, 1]

    output[
        "p_away_v4_europe"
    ] = probs[:, 2]

    output[
        "europe_penalty_v4"
    ] = best_europe

    output[
        "away_trip_penalty_v4"
    ] = best_away

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print()
    print("==============================")
    print("EUROPE CONGESTION TEST COMPLETE")
    print("==============================")

    if (
        best_europe == 0.0
        and
        best_away == 0.0
    ):

        print(
            "European congestion adjustment rejected."
        )

    else:

        print(
            "European congestion improved "
            "the tuning objective."
        )

    print()
    print(
        "Parameters selected using "
        "2021/22–2022/23 only ✅"
    )

    print(
        "2023/24 and 2024/25 were "
        "not used for parameter selection ✅"
    )

    print()
    print(
        f"Tuning results:"
        f"\n{OUTPUT_RESULTS}"
    )

    print()
    print(
        f"Predictions:"
        f"\n{OUTPUT_PREDICTIONS}"
    )


if __name__ == "__main__":
    main()