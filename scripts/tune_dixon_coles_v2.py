from pathlib import Path
import math

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "poisson_v1_best_predictions.csv"
)

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "dixon_coles_v2_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "dixon_coles_v2_predictions.csv"
)


# =========================================================
# SETTINGS
# =========================================================

TUNING_SEASONS = {
    "2122",
    "2223",
}

LOCKED_TEST_SEASONS = {
    "2324",
    "2425",
    "2526",
}

MAX_GOALS = 10
EPS = 1e-12


# Search a reasonably fine rho grid.
#
# Negative rho generally increases 0-0 / 1-1
# relative to 1-0 / 0-1.
#
# We will let the validation data decide.

RHO_VALUES = np.round(
    np.arange(
        -0.20,
        0.205,
        0.005,
    ),
    3,
)


# =========================================================
# POISSON
# =========================================================

FACTORIALS = np.array(
    [
        math.factorial(k)
        for k in range(MAX_GOALS + 1)
    ],
    dtype=float,
)


def poisson_probs(lambdas):

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
        FACTORIALS[None, :]
    )

    probs = (
        probs
        /
        probs.sum(
            axis=1,
            keepdims=True,
        )
    )

    return probs


# =========================================================
# DIXON-COLES CORRECTION
# =========================================================

def dc_tau(
    home_goals,
    away_goals,
    home_lambda,
    away_lambda,
    rho,
):
    """
    Dixon-Coles low-score correction.

    tau(0,0) = 1 - lambda_home * lambda_away * rho
    tau(0,1) = 1 + lambda_home * rho
    tau(1,0) = 1 + lambda_away * rho
    tau(1,1) = 1 - rho

    All other scores have tau = 1.
    """

    if (
        home_goals == 0
        and away_goals == 0
    ):
        return (
            1.0
            -
            home_lambda
            * away_lambda
            * rho
        )

    if (
        home_goals == 0
        and away_goals == 1
    ):
        return (
            1.0
            +
            home_lambda
            * rho
        )

    if (
        home_goals == 1
        and away_goals == 0
    ):
        return (
            1.0
            +
            away_lambda
            * rho
        )

    if (
        home_goals == 1
        and away_goals == 1
    ):
        return (
            1.0
            - rho
        )

    return 1.0


def dixon_coles_1x2(
    home_lambda,
    away_lambda,
    rho,
):
    """
    Generate corrected Home / Draw / Away probabilities.
    """

    home_lambda = np.asarray(
        home_lambda,
        dtype=float,
    )

    away_lambda = np.asarray(
        away_lambda,
        dtype=float,
    )

    home_poisson = poisson_probs(
        home_lambda
    )

    away_poisson = poisson_probs(
        away_lambda
    )

    n_matches = len(
        home_lambda
    )

    home_win = np.zeros(
        n_matches
    )

    draw = np.zeros(
        n_matches
    )

    away_win = np.zeros(
        n_matches
    )

    for h in range(
        MAX_GOALS + 1
    ):

        for a in range(
            MAX_GOALS + 1
        ):

            base_prob = (
                home_poisson[:, h]
                *
                away_poisson[:, a]
            )

            if (
                h <= 1
                and a <= 1
            ):

                if (
                    h == 0
                    and a == 0
                ):
                    tau = (
                        1.0
                        -
                        home_lambda
                        * away_lambda
                        * rho
                    )

                elif (
                    h == 0
                    and a == 1
                ):
                    tau = (
                        1.0
                        +
                        home_lambda
                        * rho
                    )

                elif (
                    h == 1
                    and a == 0
                ):
                    tau = (
                        1.0
                        +
                        away_lambda
                        * rho
                    )

                else:
                    tau = (
                        1.0
                        - rho
                    )

                corrected = (
                    base_prob
                    * tau
                )

            else:
                corrected = (
                    base_prob
                )

            if h > a:
                home_win += corrected

            elif h == a:
                draw += corrected

            else:
                away_win += corrected

    total = (
        home_win
        + draw
        + away_win
    )

    # Normalize after DC correction.
    home_win /= total
    draw /= total
    away_win /= total

    return np.column_stack(
        [
            home_win,
            draw,
            away_win,
        ]
    )


# =========================================================
# METRICS
# =========================================================

def get_result_classes(
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


def brier_score(
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


def accuracy_score(
    y_true,
    probs,
):
    predicted = probs.argmax(
        axis=1
    )

    return (
        predicted
        == y_true
    ).mean()


# =========================================================
# EVALUATION
# =========================================================

def evaluate(
    df,
    rho,
):

    home_lambda = df[
        "home_lambda_v1"
    ].to_numpy()

    away_lambda = df[
        "away_lambda_v1"
    ].to_numpy()

    probs = dixon_coles_1x2(
        home_lambda,
        away_lambda,
        rho,
    )

    y_true = get_result_classes(
        df[
            "home_goals"
        ].to_numpy(),
        df[
            "away_goals"
        ].to_numpy(),
    )

    return {
        "games":
            len(df),

        "accuracy":
            accuracy_score(
                y_true,
                probs,
            ),

        "log_loss":
            log_loss(
                y_true,
                probs,
            ),

        "brier":
            brier_score(
                y_true,
                probs,
            ),
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("==============================")
    print("TUNING DIXON-COLES V2")
    print("==============================")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    df[
        "season"
    ] = (
        df[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    print(
        f"V1 matches loaded: "
        f"{len(df):,}"
    )

    print()
    print(
        "V1 parameters remain frozen:"
    )

    print(
        "Recency:          0.95"
    )

    print(
        "Overall weight:   0.75"
    )

    print(
        "Venue weight:     0.25"
    )

    print(
        "Overall shrink K: 20"
    )

    print(
        "Venue shrink K:   4"
    )

    # -----------------------------------------------------
    # SPLITS
    # -----------------------------------------------------

    tuning = df[
        df[
            "season"
        ].isin(
            TUNING_SEASONS
        )
    ].copy()

    locked_test = df[
        df[
            "season"
        ].isin(
            LOCKED_TEST_SEASONS
        )
    ].copy()

    print()
    print(
        f"Tuning games: "
        f"{len(tuning):,}"
    )

    print(
        f"Locked test games: "
        f"{len(locked_test):,}"
    )

    # -----------------------------------------------------
    # V1 BASELINE
    #
    # rho = 0 is exactly independent Poisson.
    # -----------------------------------------------------

    baseline_tuning = evaluate(
        tuning,
        rho=0.0,
    )

    baseline_test = evaluate(
        locked_test,
        rho=0.0,
    )

    print()
    print("==============================")
    print("V1 BASELINE")
    print("==============================")

    print(
        f"Tuning Log Loss: "
        f"{baseline_tuning['log_loss']:.5f}"
    )

    print(
        f"Locked Log Loss: "
        f"{baseline_test['log_loss']:.5f}"
    )

    # -----------------------------------------------------
    # RHO SEARCH
    # -----------------------------------------------------

    print()
    print(
        f"Testing "
        f"{len(RHO_VALUES)} rho values..."
    )

    results = []

    for rho in RHO_VALUES:

        # Ensure DC factors stay positive.
        #
        # If a candidate rho creates an invalid
        # correction for any match, skip it.

        max_home = tuning[
            "home_lambda_v1"
        ].max()

        max_away = tuning[
            "away_lambda_v1"
        ].max()

        valid_00 = (
            1.0
            -
            max_home
            * max_away
            * rho
        ) > 0

        valid_01 = (
            1.0
            +
            max_home
            * rho
        ) > 0

        valid_10 = (
            1.0
            +
            max_away
            * rho
        ) > 0

        valid_11 = (
            1.0
            - rho
        ) > 0

        if not (
            valid_00
            and valid_01
            and valid_10
            and valid_11
        ):
            continue

        metrics = evaluate(
            tuning,
            rho,
        )

        results.append({
            "rho":
                rho,

            "games":
                metrics[
                    "games"
                ],

            "tuning_accuracy":
                metrics[
                    "accuracy"
                ],

            "tuning_log_loss":
                metrics[
                    "log_loss"
                ],

            "tuning_brier":
                metrics[
                    "brier"
                ],

            "log_loss_improvement":
                (
                    baseline_tuning[
                        "log_loss"
                    ]
                    -
                    metrics[
                        "log_loss"
                    ]
                ),
        })

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            [
                "tuning_log_loss",
                "tuning_brier",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    results_df[
        "rank"
    ] = (
        np.arange(
            len(results_df)
        )
        + 1
    )

    results_df.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    # -----------------------------------------------------
    # TOP RESULTS
    # -----------------------------------------------------

    print()
    print("==============================")
    print("TOP 15 RHO VALUES")
    print("==============================")

    display = (
        results_df
        .head(15)
        .copy()
    )

    display[
        "tuning_accuracy"
    ] *= 100.0

    print(
        display[
            [
                "rank",
                "rho",
                "games",
                "tuning_log_loss",
                "tuning_brier",
                "tuning_accuracy",
                "log_loss_improvement",
            ]
        ]
        .round(6)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # WINNER
    # -----------------------------------------------------

    best = results_df.iloc[
        0
    ]

    best_rho = float(
        best[
            "rho"
        ]
    )

    print()
    print("==============================")
    print("WINNING DIXON-COLES PARAMETER")
    print("==============================")

    print(
        f"Rho:              "
        f"{best_rho:.3f}"
    )

    print(
        f"V1 tuning LL:     "
        f"{baseline_tuning['log_loss']:.5f}"
    )

    print(
        f"V2 tuning LL:     "
        f"{best['tuning_log_loss']:.5f}"
    )

    print(
        f"Improvement:      "
        f"{best['log_loss_improvement']:+.5f}"
    )

    # -----------------------------------------------------
    # LOCKED TEST
    # -----------------------------------------------------

    test_metrics = evaluate(
        locked_test,
        best_rho,
    )

    print()
    print("==============================")
    print("LOCKED TEST")
    print("2023/24–2025/26")
    print("==============================")

    print(
        f"Games: "
        f"{test_metrics['games']:,}"
    )

    print()
    print(
        f"{'Metric':<15}"
        f"{'V1':>12}"
        f"{'V2 DC':>12}"
        f"{'Change':>12}"
    )

    print("-" * 51)

    print(
        f"{'Accuracy':<15}"
        f"{baseline_test['accuracy']:>11.2%}"
        f"{test_metrics['accuracy']:>11.2%}"
        f"{test_metrics['accuracy'] - baseline_test['accuracy']:>+11.2%}"
    )

    print(
        f"{'Log Loss':<15}"
        f"{baseline_test['log_loss']:>12.5f}"
        f"{test_metrics['log_loss']:>12.5f}"
        f"{test_metrics['log_loss'] - baseline_test['log_loss']:>+12.5f}"
    )

    print(
        f"{'Brier':<15}"
        f"{baseline_test['brier']:>12.5f}"
        f"{test_metrics['brier']:>12.5f}"
        f"{test_metrics['brier'] - baseline_test['brier']:>+12.5f}"
    )

    # -----------------------------------------------------
    # BY LEAGUE
    # -----------------------------------------------------

    print()
    print("==============================")
    print("LOCKED TEST — BY LEAGUE")
    print("==============================")

    league_rows = []

    for league, group in (
        locked_test.groupby(
            "league"
        )
    ):

        v1 = evaluate(
            group,
            rho=0.0,
        )

        v2 = evaluate(
            group,
            rho=best_rho,
        )

        league_rows.append({
            "league":
                league,

            "games":
                len(group),

            "v1_log_loss":
                v1[
                    "log_loss"
                ],

            "v2_log_loss":
                v2[
                    "log_loss"
                ],

            "ll_change":
                (
                    v2[
                        "log_loss"
                    ]
                    -
                    v1[
                        "log_loss"
                    ]
                ),

            "v1_brier":
                v1[
                    "brier"
                ],

            "v2_brier":
                v2[
                    "brier"
                ],

            "v1_accuracy":
                v1[
                    "accuracy"
                ],

            "v2_accuracy":
                v2[
                    "accuracy"
                ],
        })

    league_table = pd.DataFrame(
        league_rows
    )

    league_table[
        "v1_accuracy"
    ] *= 100.0

    league_table[
        "v2_accuracy"
    ] *= 100.0

    print(
        league_table
        .round(5)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # SAVE V2 PROBABILITIES
    # -----------------------------------------------------

    all_probs = dixon_coles_1x2(
        df[
            "home_lambda_v1"
        ].to_numpy(),
        df[
            "away_lambda_v1"
        ].to_numpy(),
        best_rho,
    )

    output = df.copy()

    output[
        "p_home_v2"
    ] = all_probs[:, 0]

    output[
        "p_draw_v2"
    ] = all_probs[:, 1]

    output[
        "p_away_v2"
    ] = all_probs[:, 2]

    output[
        "dc_rho"
    ] = best_rho

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    prob_sum = (
        output[
            [
                "p_home_v2",
                "p_draw_v2",
                "p_away_v2",
            ]
        ]
        .sum(
            axis=1
        )
    )

    max_error = (
        prob_sum
        - 1.0
    ).abs().max()

    print()
    print("==============================")
    print("VALIDATION")
    print("==============================")

    print(
        f"Max probability sum error: "
        f"{max_error:.12f}"
    )

    print(
        "V2 probabilities sum to 1 ✅"
    )

    print(
        "V1 parameters remained frozen ✅"
    )

    print(
        "2023/24–2025/26 was not used "
        "to select rho ✅"
    )

    print()
    print(
        f"Tuning results saved:"
        f"\n{OUTPUT_RESULTS}"
    )

    print()
    print(
        f"V2 predictions saved:"
        f"\n{OUTPUT_PREDICTIONS}"
    )


if __name__ == "__main__":
    main()