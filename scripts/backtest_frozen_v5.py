from pathlib import Path
import math

import numpy as np
import pandas as pd

import tune_overall_venue_v5 as base


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "frozen_v5_backtest_summary.csv"
)

OUTPUT_SEASONS = (
    ROOT
    / "data"
    / "processed"
    / "frozen_v5_backtest_by_season.csv"
)

OUTPUT_LEAGUES = (
    ROOT
    / "data"
    / "processed"
    / "frozen_v5_backtest_by_league.csv"
)

OUTPUT_BUCKETS = (
    ROOT
    / "data"
    / "processed"
    / "frozen_v5_probability_buckets.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "frozen_v5_predictions.csv"
)


# ============================================================
# FROZEN V5
# ============================================================

GOAL_WEIGHT = 0.09
XG_WEIGHT = 0.75
SHOT_WEIGHT = 0.16
SOT_WEIGHT = 0.00

GOAL_RECENCY = 0.975
XG_RECENCY = 0.925
SHOT_RECENCY = 0.850

OPPONENT_STRENGTH = 0.875

OVERALL_WEIGHT = 0.80
VENUE_WEIGHT = 0.20


# ============================================================
# IMPORTANT DATA SPLITS
# ============================================================

TUNING_SEASONS = {
    "2122",
    "2223",
}

VALIDATION_SEASONS = {
    "2324",
}

FINAL_CHECK_SEASONS = {
    "2425",
}

LOCKED_TEST_SEASONS = {
    "2526",
}


# ============================================================
# POISSON
# ============================================================

MAX_GOALS = 10
EPS = 1e-12

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


# ============================================================
# METRICS
# ============================================================

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

    return float(
        (
            -np.log(
                chosen
            )
        ).mean()
    )


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

    return float(
        np.mean(
            np.sum(
                (
                    probs
                    - truth
                ) ** 2,
                axis=1,
            )
        )
    )


def accuracy(
    y_true,
    probs,
):

    return float(
        (
            probs.argmax(
                axis=1
            )
            ==
            y_true
        ).mean()
    )


# ============================================================
# MULTICLASS ECE
#
# Confidence calibration:
# predicted class probability vs actual correctness.
# ============================================================

def expected_calibration_error(
    y_true,
    probs,
    bins=10,
):

    predicted = probs.argmax(
        axis=1
    )

    confidence = probs.max(
        axis=1
    )

    correct = (
        predicted
        ==
        y_true
    ).astype(
        float
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0

    for i in range(
        bins
    ):

        lower = edges[i]
        upper = edges[i + 1]

        if i == bins - 1:

            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )

        count = mask.sum()

        if count == 0:

            continue

        avg_confidence = (
            confidence[
                mask
            ].mean()
        )

        avg_accuracy = (
            correct[
                mask
            ].mean()
        )

        ece += (
            count
            /
            len(y_true)
            *
            abs(
                avg_confidence
                -
                avg_accuracy
            )
        )

    return float(
        ece
    )


# ============================================================
# BLEND COMPONENT
# ============================================================

def blend_component(
    df,
    overall_col,
    venue_col,
):

    return (
        OVERALL_WEIGHT
        *
        df[
            overall_col
        ]
        +
        VENUE_WEIGHT
        *
        df[
            venue_col
        ]
    )


# ============================================================
# BUILD COMPONENT STORE
# ============================================================

def build_component_store():

    return (
        base
        .build_component_store()
    )


# ============================================================
# BUILD FROZEN V5 LAMBDAS
# ============================================================

def build_lambdas(
    df,
):

    # --------------------------------------------------------
    # GOALS
    # --------------------------------------------------------

    home_goal_attack = blend_component(
        df,
        "home_adj_goal_attack",
        "home_adj_venue_goal_attack",
    )

    home_goal_defense = blend_component(
        df,
        "home_adj_goal_defense",
        "home_adj_venue_goal_defense",
    )

    away_goal_attack = blend_component(
        df,
        "away_adj_goal_attack",
        "away_adj_venue_goal_attack",
    )

    away_goal_defense = blend_component(
        df,
        "away_adj_goal_defense",
        "away_adj_venue_goal_defense",
    )

    # --------------------------------------------------------
    # SHOTS
    # --------------------------------------------------------

    home_shot_attack = blend_component(
        df,
        "home_adj_shot_attack",
        "home_adj_venue_shot_attack",
    )

    home_shot_defense = blend_component(
        df,
        "home_adj_shot_defense",
        "home_adj_venue_shot_defense",
    )

    away_shot_attack = blend_component(
        df,
        "away_adj_shot_attack",
        "away_adj_venue_shot_attack",
    )

    away_shot_defense = blend_component(
        df,
        "away_adj_shot_defense",
        "away_adj_venue_shot_defense",
    )

    # --------------------------------------------------------
    # XG
    # --------------------------------------------------------

    home_xg_attack = blend_component(
        df,
        "home_xg_attack_overall",
        "home_xg_attack_venue",
    )

    home_xg_defense = blend_component(
        df,
        "home_xg_defense_overall",
        "home_xg_defense_venue",
    )

    away_xg_attack = blend_component(
        df,
        "away_xg_attack_overall",
        "away_xg_attack_venue",
    )

    away_xg_defense = blend_component(
        df,
        "away_xg_defense_overall",
        "away_xg_defense_venue",
    )

    # --------------------------------------------------------
    # FROZEN SIGNAL BLEND
    # --------------------------------------------------------

    home_attack = (
        GOAL_WEIGHT
        *
        home_goal_attack
        +
        XG_WEIGHT
        *
        home_xg_attack
        +
        SHOT_WEIGHT
        *
        home_shot_attack
    )

    home_defense = (
        GOAL_WEIGHT
        *
        home_goal_defense
        +
        XG_WEIGHT
        *
        home_xg_defense
        +
        SHOT_WEIGHT
        *
        home_shot_defense
    )

    away_attack = (
        GOAL_WEIGHT
        *
        away_goal_attack
        +
        XG_WEIGHT
        *
        away_xg_attack
        +
        SHOT_WEIGHT
        *
        away_shot_attack
    )

    away_defense = (
        GOAL_WEIGHT
        *
        away_goal_defense
        +
        XG_WEIGHT
        *
        away_xg_defense
        +
        SHOT_WEIGHT
        *
        away_shot_defense
    )

    home_lambda = (
        df[
            "lg_home_goals"
        ]
        *
        home_attack
        *
        away_defense
    ).clip(
        lower=0.15,
        upper=4.50,
    )

    away_lambda = (
        df[
            "lg_away_goals"
        ]
        *
        away_attack
        *
        home_defense
    ).clip(
        lower=0.15,
        upper=4.50,
    )

    return (
        home_lambda,
        away_lambda,
    )


# ============================================================
# GENERATE FROZEN PREDICTIONS
# ============================================================

def build_predictions(
    df,
):

    out = df.copy()

    (
        home_lambda,
        away_lambda,
    ) = build_lambdas(
        out
    )

    valid = (
        home_lambda.notna()
        &
        away_lambda.notna()
    )

    out = out.loc[
        valid
    ].copy()

    out[
        "home_lambda_v5"
    ] = (
        home_lambda.loc[
            valid
        ]
        .to_numpy()
    )

    out[
        "away_lambda_v5"
    ] = (
        away_lambda.loc[
            valid
        ]
        .to_numpy()
    )

    probs = calculate_1x2_probs(
        out[
            "home_lambda_v5"
        ].to_numpy(),

        out[
            "away_lambda_v5"
        ].to_numpy(),
    )

    out[
        "p_home_v5"
    ] = probs[
        :,
        0
    ]

    out[
        "p_draw_v5"
    ] = probs[
        :,
        1
    ]

    out[
        "p_away_v5"
    ] = probs[
        :,
        2
    ]

    out[
        "predicted_class"
    ] = probs.argmax(
        axis=1
    )

    out[
        "predicted_probability"
    ] = probs.max(
        axis=1
    )

    out[
        "actual_class"
    ] = result_classes(
        out[
            "home_goals"
        ].to_numpy(),

        out[
            "away_goals"
        ].to_numpy(),
    )

    out[
        "correct"
    ] = (
        out[
            "predicted_class"
        ]
        ==
        out[
            "actual_class"
        ]
    ).astype(
        int
    )

    # --------------------------------------------------------
    # MODEL PARAMETERS
    # --------------------------------------------------------

    out[
        "goal_weight_v5"
    ] = GOAL_WEIGHT

    out[
        "xg_weight_v5"
    ] = XG_WEIGHT

    out[
        "shot_weight_v5"
    ] = SHOT_WEIGHT

    out[
        "goal_recency_v5"
    ] = GOAL_RECENCY

    out[
        "xg_recency_v5"
    ] = XG_RECENCY

    out[
        "shot_recency_v5"
    ] = SHOT_RECENCY

    out[
        "opponent_strength_v5"
    ] = (
        OPPONENT_STRENGTH
    )

    out[
        "overall_weight_v5"
    ] = OVERALL_WEIGHT

    out[
        "venue_weight_v5"
    ] = VENUE_WEIGHT

    return out


# ============================================================
# EVALUATE SUBSET
# ============================================================

def evaluate_subset(
    df,
):

    if len(
        df
    ) == 0:

        return {
            "games":
                0,

            "accuracy":
                np.nan,

            "log_loss":
                np.nan,

            "brier":
                np.nan,

            "ece":
                np.nan,

            "avg_confidence":
                np.nan,
        }

    probs = df[
        [
            "p_home_v5",
            "p_draw_v5",
            "p_away_v5",
        ]
    ].to_numpy()

    y = df[
        "actual_class"
    ].to_numpy()

    return {
        "games":
            len(
                df
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

        "ece":
            expected_calibration_error(
                y,
                probs,
            ),

        "avg_confidence":
            float(
                probs.max(
                    axis=1
                ).mean()
            ),
    }


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    title,
    metrics,
):

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    print(
        f"Games:       "
        f"{metrics['games']:,}"
    )

    print(
        f"Accuracy:    "
        f"{metrics['accuracy']:.2%}"
    )

    print(
        f"Log Loss:    "
        f"{metrics['log_loss']:.5f}"
    )

    print(
        f"Brier:       "
        f"{metrics['brier']:.5f}"
    )

    print(
        f"ECE:         "
        f"{metrics['ece']:.2%}"
    )

    print(
        f"Avg Conf:    "
        f"{metrics['avg_confidence']:.2%}"
    )


# ============================================================
# BY SEASON
# ============================================================

def build_season_table(
    predictions,
):

    rows = []

    for season, sub in (
        predictions
        .groupby(
            "season"
        )
    ):

        metrics = evaluate_subset(
            sub
        )

        rows.append(
            {
                "season":
                    str(
                        season
                    ).zfill(
                        4
                    ),

                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# BY LEAGUE
# ============================================================

def build_league_table(
    predictions,
    seasons=None,
):

    df = predictions.copy()

    if seasons is not None:

        season = (
            df[
                "season"
            ]
            .astype(str)
            .str.zfill(4)
        )

        df = df[
            season.isin(
                seasons
            )
        ].copy()

    rows = []

    for league, sub in (
        df.groupby(
            "league"
        )
    ):

        metrics = evaluate_subset(
            sub
        )

        rows.append(
            {
                "league":
                    league,

                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PROBABILITY BUCKETS
# ============================================================

def build_probability_buckets(
    predictions,
    seasons,
):

    season = (
        predictions[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    df = predictions[
        season.isin(
            seasons
        )
    ].copy()

    bins = [
        0.00,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        1.01,
    ]

    labels = [
        "<35%",
        "35-40%",
        "40-45%",
        "45-50%",
        "50-55%",
        "55-60%",
        "60-65%",
        "65-70%",
        "70-75%",
        "75-80%",
        "80-85%",
        "85%+",
    ]

    df[
        "confidence_bucket"
    ] = pd.cut(
        df[
            "predicted_probability"
        ],
        bins=bins,
        labels=labels,
        right=False,
    )

    rows = []

    for bucket, sub in (
        df.groupby(
            "confidence_bucket",
            observed=False,
        )
    ):

        if len(sub) == 0:
            continue

        avg_predicted_probability = (
            sub[
                "predicted_probability"
            ].mean()
        )

        actual_accuracy = (
            sub[
                "correct"
            ].mean()
        )

        rows.append(
            {
                "bucket":
                    str(
                        bucket
                    ),

                "games":
                    len(
                        sub
                    ),

                "avg_predicted_probability":
                    avg_predicted_probability,

                "actual_accuracy":
                    actual_accuracy,

                "calibration_gap":
                    (
                        actual_accuracy
                        -
                        avg_predicted_probability
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PREDICTED CLASS BREAKDOWN
# ============================================================

def print_class_breakdown(
    predictions,
    seasons,
):

    season = (
        predictions[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    df = predictions[
        season.isin(
            seasons
        )
    ].copy()

    class_names = {
        0:
            "HOME",

        1:
            "DRAW",

        2:
            "AWAY",
    }

    print()
    print("=" * 90)
    print("LOCKED TEST — PREDICTED RESULT TYPE")
    print("=" * 90)

    rows = []

    for class_id, name in (
        class_names.items()
    ):

        sub = df[
            df[
                "predicted_class"
            ]
            ==
            class_id
        ]

        if len(
            sub
        ) == 0:

            continue

        rows.append(
            {
                "prediction":
                    name,

                "games":
                    len(
                        sub
                    ),

                "accuracy":
                    sub[
                        "correct"
                    ].mean()
                    * 100,

                "avg_confidence":
                    sub[
                        "predicted_probability"
                    ].mean()
                    * 100,
            }
        )

    table = pd.DataFrame(
        rows
    )

    print(
        table
        .round(
            3
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("FROZEN V5 BACKTEST")
    print("==============================")
    print()

    print(
        "FROZEN MODEL:"
    )

    print(
        f"Signals: "
        f"{GOAL_WEIGHT:.0%} goals / "
        f"{XG_WEIGHT:.0%} xG / "
        f"{SHOT_WEIGHT:.0%} shots"
    )

    print(
        f"Recency: "
        f"{GOAL_RECENCY:.3f} / "
        f"{XG_RECENCY:.3f} / "
        f"{SHOT_RECENCY:.3f}"
    )

    print(
        f"Opponent strength: "
        f"{OPPONENT_STRENGTH:.3f}"
    )

    print(
        f"Overall / venue: "
        f"{OVERALL_WEIGHT:.0%} / "
        f"{VENUE_WEIGHT:.0%}"
    )

    print(
        "Points form: 0% (rejected)"
    )

    print(
        "Attack / defense: symmetric"
    )

    # ========================================================
    # BUILD
    # ========================================================

    print()
    print(
        "Building frozen V5 components..."
    )

    df = build_component_store()

    predictions = build_predictions(
        df
    )

    predictions[
        "season"
    ] = (
        predictions[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    print(
        f"Usable predictions: "
        f"{len(predictions):,}"
    )

    # ========================================================
    # SAVE MASTER PREDICTIONS
    # ========================================================

    predictions.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # ========================================================
    # KEY SPLITS
    # ========================================================

    summary_rows = []

    evaluation_sets = [
        (
            "TUNING",
            TUNING_SEASONS,
        ),
        (
            "VALIDATION",
            VALIDATION_SEASONS,
        ),
        (
            "FINAL_CHECK",
            FINAL_CHECK_SEASONS,
        ),
        (
            "LOCKED_TEST",
            LOCKED_TEST_SEASONS,
        ),
    ]

    for name, seasons in (
        evaluation_sets
    ):

        sub = predictions[
            predictions[
                "season"
            ].isin(
                seasons
            )
        ].copy()

        metrics = evaluate_subset(
            sub
        )

        print_metrics(
            name,
            metrics,
        )

        summary_rows.append(
            {
                "sample":
                    name,

                **metrics,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # ========================================================
    # BY SEASON
    # ========================================================

    season_table = (
        build_season_table(
            predictions
        )
    )

    season_table.to_csv(
        OUTPUT_SEASONS,
        index=False,
    )

    print()
    print("=" * 90)
    print("PERFORMANCE BY SEASON")
    print("=" * 90)

    display_seasons = (
        season_table.copy()
    )

    display_seasons[
        "accuracy"
    ] *= 100

    display_seasons[
        "ece"
    ] *= 100

    display_seasons[
        "avg_confidence"
    ] *= 100

    print(
        display_seasons
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # LOCKED TEST BY LEAGUE
    # ========================================================

    league_table = (
        build_league_table(
            predictions,
            LOCKED_TEST_SEASONS,
        )
    )

    league_table.to_csv(
        OUTPUT_LEAGUES,
        index=False,
    )

    print()
    print("=" * 90)
    print("2025/26 LOCKED TEST — BY LEAGUE")
    print("=" * 90)

    display_league = (
        league_table.copy()
    )

    display_league[
        "accuracy"
    ] *= 100

    display_league[
        "ece"
    ] *= 100

    display_league[
        "avg_confidence"
    ] *= 100

    print(
        display_league
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # LOCKED TEST PROBABILITY BUCKETS
    # ========================================================

    buckets = (
        build_probability_buckets(
            predictions,
            LOCKED_TEST_SEASONS,
        )
    )

    buckets.to_csv(
        OUTPUT_BUCKETS,
        index=False,
    )

    print()
    print("=" * 90)
    print("2025/26 LOCKED TEST — CONFIDENCE BUCKETS")
    print("=" * 90)

    bucket_display = (
        buckets.copy()
    )

    for col in [
        "avg_predicted_probability",
        "actual_accuracy",
        "calibration_gap",
    ]:

        bucket_display[
            col
        ] *= 100

    print(
        bucket_display
        .round(
            3
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # RESULT CLASS BREAKDOWN
    # ========================================================

    print_class_breakdown(
        predictions,
        LOCKED_TEST_SEASONS,
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("==============================")
    print("FROZEN V5 BACKTEST COMPLETE")
    print("==============================")

    print(
        "V5 parameters frozen before "
        "2025/26 evaluation ✅"
    )

    print(
        "2025/26 treated as locked "
        "out-of-sample test ✅"
    )

    print(
        "No parameter selection performed "
        "using locked-test results ✅"
    )

    print()
    print(
        "Summary:"
    )

    print(
        OUTPUT_SUMMARY
    )

    print()

    print(
        "Predictions:"
    )

    print(
        OUTPUT_PREDICTIONS
    )


if __name__ == "__main__":
    main()