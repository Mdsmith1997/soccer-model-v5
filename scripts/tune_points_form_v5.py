from pathlib import Path
import math

import numpy as np
import pandas as pd

import tune_overall_venue_v5 as base


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREGAME_FILE = (
    ROOT
    / "data"
    / "processed"
    / "team_pregame_stats.csv"
)

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "points_form_v5_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "points_form_v5_predictions.csv"
)


# ============================================================
# FROZEN V5 CORE
# ============================================================

GOAL_WEIGHT = 0.09
XG_WEIGHT = 0.75
SHOT_WEIGHT = 0.16

GOAL_RECENCY = 0.975
XG_RECENCY = 0.925
SHOT_RECENCY = 0.850

OPPONENT_STRENGTH = 0.875

OVERALL_WEIGHT = 0.80
VENUE_WEIGHT = 0.20


# ============================================================
# FORM GRID
# ============================================================

FORM_WEIGHTS = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
]


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
        ==
        y_true
    ).mean()


# ============================================================
# BUILD FROZEN V5 COMPONENT STORE
# ============================================================

def build_component_store():

    df = base.build_component_store()

    return df


# ============================================================
# LOAD PREGAME POINTS
# ============================================================

def load_points():

    df = pd.read_csv(
        PREGAME_FILE,
        parse_dates=[
            "date",
        ],
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

    required = [
        "match_id",
        "team",
        "is_home",
        "pregame_ew_points",
        "pregame_venue_ew_points",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required points columns: "
            + str(missing)
        )

    return df


# ============================================================
# BUILD MATCH-LEVEL FORM
# ============================================================

def build_points_match_table(
    points,
):

    df = points.copy()

    df[
        "form_points"
    ] = (
        OVERALL_WEIGHT
        *
        df[
            "pregame_ew_points"
        ]
        +
        VENUE_WEIGHT
        *
        df[
            "pregame_venue_ew_points"
        ]
    )

    # --------------------------------------------------------
    # NEUTRAL FORM = 1.5 PPG
    #
    # Normalizing around the natural midpoint prevents form
    # from automatically inflating or shrinking every team.
    # --------------------------------------------------------

    df[
        "form_rating"
    ] = (
        df[
            "form_points"
        ]
        /
        1.5
    )

    df[
        "form_rating"
    ] = (
        df[
            "form_rating"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .clip(
            lower=0.40,
            upper=1.60,
        )
    )

    home = df[
        df[
            "is_home"
        ]
        == 1
    ][
        [
            "match_id",
            "form_points",
            "form_rating",
        ]
    ].rename(
        columns={
            "form_points":
                "home_form_points",

            "form_rating":
                "home_form_rating",
        }
    )

    away = df[
        df[
            "is_home"
        ]
        == 0
    ][
        [
            "match_id",
            "form_points",
            "form_rating",
        ]
    ].rename(
        columns={
            "form_points":
                "away_form_points",

            "form_rating":
                "away_form_rating",
        }
    )

    return home.merge(
        away,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )


# ============================================================
# ATTACH FORM TO V5
# ============================================================

def attach_form(
    df,
    points_match,
):

    out = df.merge(
        points_match,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    # Neutral fallback means no adjustment.
    out[
        "home_form_rating"
    ] = (
        out[
            "home_form_rating"
        ]
        .fillna(
            1.0
        )
    )

    out[
        "away_form_rating"
    ] = (
        out[
            "away_form_rating"
        ]
        .fillna(
            1.0
        )
    )

    return out


# ============================================================
# BUILD FROZEN V5 STRENGTHS
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


def build_core_strengths(
    df,
):

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

    home_attack = (
        GOAL_WEIGHT
        * home_goal_attack
        +
        XG_WEIGHT
        * home_xg_attack
        +
        SHOT_WEIGHT
        * home_shot_attack
    )

    home_defense = (
        GOAL_WEIGHT
        * home_goal_defense
        +
        XG_WEIGHT
        * home_xg_defense
        +
        SHOT_WEIGHT
        * home_shot_defense
    )

    away_attack = (
        GOAL_WEIGHT
        * away_goal_attack
        +
        XG_WEIGHT
        * away_xg_attack
        +
        SHOT_WEIGHT
        * away_shot_attack
    )

    away_defense = (
        GOAL_WEIGHT
        * away_goal_defense
        +
        XG_WEIGHT
        * away_xg_defense
        +
        SHOT_WEIGHT
        * away_shot_defense
    )

    return (
        home_attack,
        home_defense,
        away_attack,
        away_defense,
    )


# ============================================================
# FORM MODIFIER
# ============================================================

def form_modifier(
    rating,
    weight,
):

    # weight = 0 -> always 1
    # rating > 1 -> small upgrade
    # rating < 1 -> small downgrade

    return (
        1.0
        +
        weight
        *
        (
            rating
            - 1.0
        )
    )


# ============================================================
# BUILD LAMBDAS
# ============================================================

def build_lambdas(
    df,
    form_weight,
):

    (
        home_attack,
        home_defense,
        away_attack,
        away_defense,
    ) = build_core_strengths(
        df
    )

    home_form = form_modifier(
        df[
            "home_form_rating"
        ],
        form_weight,
    )

    away_form = form_modifier(
        df[
            "away_form_rating"
        ],
        form_weight,
    )

    # --------------------------------------------------------
    # Better recent form:
    #   increases attacking strength
    #   improves defensive strength
    #
    # Defense ratings are "higher = worse", so divide them
    # by the form modifier.
    # --------------------------------------------------------

    home_attack = (
        home_attack
        *
        home_form
    )

    away_attack = (
        away_attack
        *
        away_form
    )

    home_defense = (
        home_defense
        /
        home_form
    )

    away_defense = (
        away_defense
        /
        away_form
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
# EVALUATE
# ============================================================

def evaluate(
    df,
    seasons,
    form_weight,
):

    season = (
        df[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    sub = df[
        season.isin(
            seasons
        )
    ].copy()

    (
        home_lambda,
        away_lambda,
    ) = build_lambdas(
        sub,
        form_weight,
    )

    valid = (
        home_lambda.notna()
        &
        away_lambda.notna()
    )

    sub = sub.loc[
        valid
    ].copy()

    home_lambda = (
        home_lambda.loc[
            valid
        ]
        .to_numpy()
    )

    away_lambda = (
        away_lambda.loc[
            valid
        ]
        .to_numpy()
    )

    probs = calculate_1x2_probs(
        home_lambda,
        away_lambda,
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


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_comparison(
    title,
    control,
    winner,
):

    print()
    print("=" * 76)
    print(title)
    print("=" * 76)

    print(
        f"Games: "
        f"{winner['games']:,}"
    )

    print()

    print(
        f"{'Metric':<15}"
        f"{'No Form':>14}"
        f"{'Form':>14}"
        f"{'Change':>14}"
    )

    print("-" * 57)

    print(
        f"{'Accuracy':<15}"
        f"{control['accuracy']:>13.2%}"
        f"{winner['accuracy']:>13.2%}"
        f"{winner['accuracy'] - control['accuracy']:>+13.2%}"
    )

    print(
        f"{'Log Loss':<15}"
        f"{control['log_loss']:>14.5f}"
        f"{winner['log_loss']:>14.5f}"
        f"{winner['log_loss'] - control['log_loss']:>+14.5f}"
    )

    print(
        f"{'Brier':<15}"
        f"{control['brier']:>14.5f}"
        f"{winner['brier']:>14.5f}"
        f"{winner['brier'] - control['brier']:>+14.5f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("TUNING POINTS FORM V5")
    print("==============================")
    print()

    print(
        "Frozen V5 core:"
    )

    print(
        f"Signals: "
        f"{GOAL_WEIGHT:.0%} / "
        f"{XG_WEIGHT:.0%} / "
        f"{SHOT_WEIGHT:.0%}"
    )

    print(
        f"Recencies: "
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

    print()
    print(
        "Building frozen component store..."
    )

    df = build_component_store()

    points = load_points()

    points_match = (
        build_points_match_table(
            points
        )
    )

    df = attach_form(
        df,
        points_match,
    )

    print(
        f"Eligible matches: "
        f"{len(df):,}"
    )

    print()
    print(
        f"Form weights tested: "
        f"{len(FORM_WEIGHTS)}"
    )

    # ========================================================
    # TUNE
    # ========================================================

    rows = []

    for form_weight in (
        FORM_WEIGHTS
    ):

        metrics = evaluate(
            df,
            TUNING_SEASONS,
            form_weight,
        )

        rows.append(
            {
                "form_weight":
                    form_weight,

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

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("==============================")
    print("POINTS FORM RESULTS")
    print("==============================")
    print()

    display = results.copy()

    display[
        "accuracy"
    ] *= 100.0

    print(
        display[
            [
                "rank",
                "form_weight",
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

    # ========================================================
    # WINNER
    # ========================================================

    best = results.iloc[
        0
    ]

    best_weight = float(
        best[
            "form_weight"
        ]
    )

    print()
    print("==============================")
    print("WINNING FORM ADJUSTMENT")
    print("==============================")

    print(
        f"Form weight: "
        f"{best_weight:.1%}"
    )

    print(
        f"Tuning LL: "
        f"{best['log_loss']:.5f}"
    )

    # ========================================================
    # TUNING
    # ========================================================

    control_tune = evaluate(
        df,
        TUNING_SEASONS,
        0.0,
    )

    winner_tune = evaluate(
        df,
        TUNING_SEASONS,
        best_weight,
    )

    print_comparison(
        "TUNING — 2021/22 TO 2022/23",
        control_tune,
        winner_tune,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    control_validation = evaluate(
        df,
        VALIDATION_SEASONS,
        0.0,
    )

    winner_validation = evaluate(
        df,
        VALIDATION_SEASONS,
        best_weight,
    )

    print_comparison(
        "VALIDATION — 2023/24",
        control_validation,
        winner_validation,
    )

    # ========================================================
    # FINAL
    # ========================================================

    control_final = evaluate(
        df,
        FINAL_SEASONS,
        0.0,
    )

    winner_final = evaluate(
        df,
        FINAL_SEASONS,
        best_weight,
    )

    print_comparison(
        "FINAL CHECK — 2024/25",
        control_final,
        winner_final,
    )

    # ========================================================
    # SAVE WINNING PREDICTIONS
    # ========================================================

    (
        home_lambda,
        away_lambda,
    ) = build_lambdas(
        df,
        best_weight,
    )

    valid = (
        home_lambda.notna()
        &
        away_lambda.notna()
    )

    output = (
        df.loc[
            valid
        ]
        .copy()
    )

    output[
        "home_lambda_v5"
    ] = (
        home_lambda.loc[
            valid
        ]
        .to_numpy()
    )

    output[
        "away_lambda_v5"
    ] = (
        away_lambda.loc[
            valid
        ]
        .to_numpy()
    )

    probs = calculate_1x2_probs(
        output[
            "home_lambda_v5"
        ].to_numpy(),

        output[
            "away_lambda_v5"
        ].to_numpy(),
    )

    output[
        "p_home_v5"
    ] = probs[
        :,
        0
    ]

    output[
        "p_draw_v5"
    ] = probs[
        :,
        1
    ]

    output[
        "p_away_v5"
    ] = probs[
        :,
        2
    ]

    output[
        "points_form_weight_v5"
    ] = best_weight

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("==============================")
    print("POINTS FORM TUNING COMPLETE")
    print("==============================")

    if best_weight == 0.0:

        print(
            "Recent points form rejected."
        )

    else:

        print(
            "Recent points form improved "
            "the tuning objective."
        )

    print()

    print(
        "Form selected using "
        "2021/22–2022/23 only ✅"
    )

    print(
        "2023/24 and 2024/25 "
        "held out from selection ✅"
    )

    print()

    print(
        "Results:"
    )

    print(
        OUTPUT_RESULTS
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