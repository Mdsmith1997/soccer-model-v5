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
    / "congestion_interactions_v4_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "congestion_interactions_v4_predictions.csv"
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


ADJUSTMENTS = [
    -0.050,
    -0.040,
    -0.030,
    -0.020,
    -0.010,
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


def apply_team_adjustment(
    own_lambda,
    opp_lambda,
    mask,
    adjustment,
):
    """
    Positive adjustment:
        boosts affected team's own lambda
        reduces opponent lambda.

    Negative adjustment:
        reduces affected team's own lambda
        boosts opponent lambda.
    """

    own_lambda.loc[
        mask
    ] *= (
        1.0
        +
        adjustment
    )

    opp_lambda.loc[
        mask
    ] *= (
        1.0
        -
        adjustment
    )


def build_flags(
    df,
):

    out = df.copy()

    # =====================================================
    # A) EUROPE <=3 DAYS
    # =====================================================

    out[
        "home_flag_a"
    ] = (
        out[
            "home_played_europe_last_3d"
        ]
        == 1
    )

    out[
        "away_flag_a"
    ] = (
        out[
            "away_played_europe_last_3d"
        ]
        == 1
    )

    # =====================================================
    # B) EUROPE AWAY <=4D + DOMESTIC AWAY
    #
    # Only possible for the away team in the domestic game.
    # =====================================================

    out[
        "home_flag_b"
    ] = False

    out[
        "away_flag_b"
    ] = (
        out[
            "away_europe_away_last_4d"
        ]
        == 1
    )

    # =====================================================
    # C) EUROPE <=4D + >=2 TRACKED MATCHES LAST 7D
    # =====================================================

    out[
        "home_flag_c"
    ] = (
        (
            out[
                "home_played_europe_last_4d"
            ]
            == 1
        )
        &
        (
            out[
                "home_tracked_matches_last_7d"
            ]
            >= 2
        )
    )

    out[
        "away_flag_c"
    ] = (
        (
            out[
                "away_played_europe_last_4d"
            ]
            == 1
        )
        &
        (
            out[
                "away_tracked_matches_last_7d"
            ]
            >= 2
        )
    )

    return out


def apply_interaction_adjustments(
    df,
    adj_a,
    adj_b,
    adj_c,
):

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
    # A
    # -----------------------------------------------------

    apply_team_adjustment(
        home_lambda,
        away_lambda,
        df[
            "home_flag_a"
        ],
        adj_a,
    )

    apply_team_adjustment(
        away_lambda,
        home_lambda,
        df[
            "away_flag_a"
        ],
        adj_a,
    )

    # -----------------------------------------------------
    # B
    # -----------------------------------------------------

    apply_team_adjustment(
        home_lambda,
        away_lambda,
        df[
            "home_flag_b"
        ],
        adj_b,
    )

    apply_team_adjustment(
        away_lambda,
        home_lambda,
        df[
            "away_flag_b"
        ],
        adj_b,
    )

    # -----------------------------------------------------
    # C
    # -----------------------------------------------------

    apply_team_adjustment(
        home_lambda,
        away_lambda,
        df[
            "home_flag_c"
        ],
        adj_c,
    )

    apply_team_adjustment(
        away_lambda,
        home_lambda,
        df[
            "away_flag_c"
        ],
        adj_c,
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
    adj_a,
    adj_b,
    adj_c,
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
        apply_interaction_adjustments(
            sub,
            adj_a,
            adj_b,
            adj_c,
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

    flag_a_games = (
        sub[
            [
                "home_flag_a",
                "away_flag_a",
            ]
        ]
        .any(
            axis=1
        )
        .sum()
    )

    flag_b_games = (
        sub[
            [
                "home_flag_b",
                "away_flag_b",
            ]
        ]
        .any(
            axis=1
        )
        .sum()
    )

    flag_c_games = (
        sub[
            [
                "home_flag_c",
                "away_flag_c",
            ]
        ]
        .any(
            axis=1
        )
        .sum()
    )

    return {
        "games":
            len(sub),

        "flag_a_games":
            int(
                flag_a_games
            ),

        "flag_b_games":
            int(
                flag_b_games
            ),

        "flag_c_games":
            int(
                flag_c_games
            ),

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
    print("=" * 74)
    print(title)
    print("=" * 74)

    print(
        f"Games: "
        f"{candidate['games']:,}"
    )

    print(
        f"Flag A games: "
        f"{candidate['flag_a_games']:,}"
    )

    print(
        f"Flag B games: "
        f"{candidate['flag_b_games']:,}"
    )

    print(
        f"Flag C games: "
        f"{candidate['flag_c_games']:,}"
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
    print("TUNING CONGESTION INTERACTIONS")
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

    df = build_flags(
        df
    )

    print(
        f"Matches loaded: "
        f"{len(df):,}"
    )

    print()
    print(
        "Flag definitions:"
    )

    print(
        "A = Europe <=3 days"
    )

    print(
        "B = Europe away <=4d "
        "+ domestic away"
    )

    print(
        "C = Europe <=4d "
        "+ >=2 tracked matches last 7d"
    )

    # =====================================================
    # GRID SEARCH
    #
    # 11^3 = 1,331 combinations.
    # =====================================================

    rows = []

    total = (
        len(ADJUSTMENTS)
        ** 3
    )

    tested = 0

    print()
    print(
        f"Testing "
        f"{total:,} combinations..."
    )

    for adj_a in ADJUSTMENTS:

        for adj_b in ADJUSTMENTS:

            for adj_c in ADJUSTMENTS:

                metrics = evaluate(
                    df,
                    TUNING_SEASONS,
                    adj_a,
                    adj_b,
                    adj_c,
                )

                rows.append(
                    {
                        "adj_a":
                            adj_a,

                        "adj_b":
                            adj_b,

                        "adj_c":
                            adj_c,

                        "games":
                            metrics[
                                "games"
                            ],

                        "flag_a_games":
                            metrics[
                                "flag_a_games"
                            ],

                        "flag_b_games":
                            metrics[
                                "flag_b_games"
                            ],

                        "flag_c_games":
                            metrics[
                                "flag_c_games"
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

                tested += 1

                if (
                    tested
                    % 200
                    == 0
                ):

                    print(
                        f"Tested "
                        f"{tested:,}/"
                        f"{total:,}"
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
    print("TOP 20 INTERACTION MODELS")
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
                "adj_a",
                "adj_b",
                "adj_c",
                "flag_a_games",
                "flag_b_games",
                "flag_c_games",
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

    best_a = float(
        best[
            "adj_a"
        ]
    )

    best_b = float(
        best[
            "adj_b"
        ]
    )

    best_c = float(
        best[
            "adj_c"
        ]
    )

    print()
    print("==============================")
    print("WINNING INTERACTION MODEL")
    print("==============================")

    print(
        f"A adjustment: "
        f"{best_a:+.1%}"
    )

    print(
        f"B adjustment: "
        f"{best_b:+.1%}"
    )

    print(
        f"C adjustment: "
        f"{best_c:+.1%}"
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
        0.0,
    )

    winner_tune = evaluate(
        df,
        TUNING_SEASONS,
        best_a,
        best_b,
        best_c,
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
        0.0,
    )

    winner_validation = evaluate(
        df,
        VALIDATION_SEASONS,
        best_a,
        best_b,
        best_c,
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
        0.0,
    )

    winner_final = evaluate(
        df,
        FINAL_SEASONS,
        best_a,
        best_b,
        best_c,
    )

    print_comparison(
        "FINAL CHECK — 2024/25",
        baseline_final,
        winner_final,
    )

    # =====================================================
    # BY LEAGUE
    # =====================================================

    print()
    print("=" * 105)
    print("2024/25 — BY LEAGUE")
    print("=" * 105)

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
            0.0,
        )

        winner = evaluate(
            group,
            FINAL_SEASONS,
            best_a,
            best_b,
            best_c,
        )

        league_rows.append(
            {
                "league":
                    league,

                "games":
                    winner[
                        "games"
                    ],

                "flag_a":
                    winner[
                        "flag_a_games"
                    ],

                "flag_b":
                    winner[
                        "flag_b_games"
                    ],

                "flag_c":
                    winner[
                        "flag_c_games"
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
        apply_interaction_adjustments(
            df,
            best_a,
            best_b,
            best_c,
        )
    )

    output = df.copy()

    output[
        "home_lambda_v4_interactions"
    ] = home_lambda

    output[
        "away_lambda_v4_interactions"
    ] = away_lambda

    probs = calculate_1x2_probs(
        home_lambda.to_numpy(),
        away_lambda.to_numpy(),
    )

    output[
        "p_home_v4_interactions"
    ] = probs[:, 0]

    output[
        "p_draw_v4_interactions"
    ] = probs[:, 1]

    output[
        "p_away_v4_interactions"
    ] = probs[:, 2]

    output[
        "interaction_a_v4"
    ] = best_a

    output[
        "interaction_b_v4"
    ] = best_b

    output[
        "interaction_c_v4"
    ] = best_c

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print()
    print("==============================")
    print("INTERACTION TEST COMPLETE")
    print("==============================")

    if (
        best_a == 0.0
        and
        best_b == 0.0
        and
        best_c == 0.0
    ):

        print(
            "Congestion interactions rejected."
        )

    else:

        print(
            "At least one interaction improved "
            "the tuning objective."
        )

    print()
    print(
        "Adjustments selected using "
        "2021/22–2022/23 only ✅"
    )

    print(
        "2023/24 and 2024/25 held out "
        "from parameter selection ✅"
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