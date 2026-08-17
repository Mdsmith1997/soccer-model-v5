from pathlib import Path
import math

import numpy as np
import pandas as pd

import tune_overall_venue_v5 as base


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "attack_defense_weights_v5_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "attack_defense_weights_v5_predictions.csv"
)


# ============================================================
# FROZEN V5 STRUCTURE
# ============================================================

OVERALL_WEIGHT = 0.80
VENUE_WEIGHT = 0.20

GOAL_RECENCY = 0.975
XG_RECENCY = 0.925
SHOT_RECENCY = 0.850

OPPONENT_STRENGTH = 0.875

SOT_WEIGHT = 0.00


# ============================================================
# CURRENT SYMMETRIC CONTROL
# ============================================================

CONTROL_GOAL_WEIGHT = 0.09
CONTROL_XG_WEIGHT = 0.75
CONTROL_SHOT_WEIGHT = 0.16


# ============================================================
# EXPANDED ASYMMETRIC SEARCH GRID
# ============================================================

ATTACK_GOAL_WEIGHTS = np.round(
    np.arange(
        0.15,
        0.301,
        0.025,
    ),
    3,
)

ATTACK_XG_WEIGHTS = np.round(
    np.arange(
        0.50,
        0.701,
        0.025,
    ),
    3,
)

DEFENSE_GOAL_WEIGHTS = np.round(
    np.arange(
        0.10,
        0.251,
        0.025,
    ),
    3,
)

DEFENSE_XG_WEIGHTS = np.round(
    np.arange(
        0.45,
        0.651,
        0.025,
    ),
    3,
)

MIN_SHOT_WEIGHT = 0.05
MAX_SHOT_WEIGHT = 0.40


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
            )
            ** 2,
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
# BUILD ATTACK GRID
# ============================================================

def build_attack_grid():

    rows = []

    for goal_weight in (
        ATTACK_GOAL_WEIGHTS
    ):

        for xg_weight in (
            ATTACK_XG_WEIGHTS
        ):

            shot_weight = (
                1.0
                - goal_weight
                - xg_weight
            )

            if (
                shot_weight
                <
                MIN_SHOT_WEIGHT
            ):

                continue

            if (
                shot_weight
                >
                MAX_SHOT_WEIGHT
            ):

                continue

            rows.append(
                {
                    "goal_weight":
                        float(
                            goal_weight
                        ),

                    "xg_weight":
                        float(
                            xg_weight
                        ),

                    "shot_weight":
                        float(
                            round(
                                shot_weight,
                                3,
                            )
                        ),
                }
            )

    return rows


# ============================================================
# BUILD DEFENSE GRID
# ============================================================

def build_defense_grid():

    rows = []

    for goal_weight in (
        DEFENSE_GOAL_WEIGHTS
    ):

        for xg_weight in (
            DEFENSE_XG_WEIGHTS
        ):

            shot_weight = (
                1.0
                - goal_weight
                - xg_weight
            )

            if (
                shot_weight
                <
                MIN_SHOT_WEIGHT
            ):

                continue

            if (
                shot_weight
                >
                MAX_SHOT_WEIGHT
            ):

                continue

            rows.append(
                {
                    "goal_weight":
                        float(
                            goal_weight
                        ),

                    "xg_weight":
                        float(
                            xg_weight
                        ),

                    "shot_weight":
                        float(
                            round(
                                shot_weight,
                                3,
                            )
                        ),
                }
            )

    return rows


# ============================================================
# BUILD FROZEN COMPONENT STORE
# ============================================================

def build_component_store():

    df = base.build_component_store()

    return df


# ============================================================
# BLEND OVERALL + VENUE
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
# PRECOMPUTE COMPONENTS
# ============================================================

def prepare_components(
    df,
):

    out = df.copy()

    # --------------------------------------------------------
    # HOME GOALS
    # --------------------------------------------------------

    out[
        "home_goal_attack_component"
    ] = blend_component(
        out,
        "home_adj_goal_attack",
        "home_adj_venue_goal_attack",
    )

    out[
        "home_goal_defense_component"
    ] = blend_component(
        out,
        "home_adj_goal_defense",
        "home_adj_venue_goal_defense",
    )

    # --------------------------------------------------------
    # AWAY GOALS
    # --------------------------------------------------------

    out[
        "away_goal_attack_component"
    ] = blend_component(
        out,
        "away_adj_goal_attack",
        "away_adj_venue_goal_attack",
    )

    out[
        "away_goal_defense_component"
    ] = blend_component(
        out,
        "away_adj_goal_defense",
        "away_adj_venue_goal_defense",
    )

    # --------------------------------------------------------
    # HOME SHOTS
    # --------------------------------------------------------

    out[
        "home_shot_attack_component"
    ] = blend_component(
        out,
        "home_adj_shot_attack",
        "home_adj_venue_shot_attack",
    )

    out[
        "home_shot_defense_component"
    ] = blend_component(
        out,
        "home_adj_shot_defense",
        "home_adj_venue_shot_defense",
    )

    # --------------------------------------------------------
    # AWAY SHOTS
    # --------------------------------------------------------

    out[
        "away_shot_attack_component"
    ] = blend_component(
        out,
        "away_adj_shot_attack",
        "away_adj_venue_shot_attack",
    )

    out[
        "away_shot_defense_component"
    ] = blend_component(
        out,
        "away_adj_shot_defense",
        "away_adj_venue_shot_defense",
    )

    # --------------------------------------------------------
    # HOME XG
    # --------------------------------------------------------

    out[
        "home_xg_attack_component"
    ] = blend_component(
        out,
        "home_xg_attack_overall",
        "home_xg_attack_venue",
    )

    out[
        "home_xg_defense_component"
    ] = blend_component(
        out,
        "home_xg_defense_overall",
        "home_xg_defense_venue",
    )

    # --------------------------------------------------------
    # AWAY XG
    # --------------------------------------------------------

    out[
        "away_xg_attack_component"
    ] = blend_component(
        out,
        "away_xg_attack_overall",
        "away_xg_attack_venue",
    )

    out[
        "away_xg_defense_component"
    ] = blend_component(
        out,
        "away_xg_defense_overall",
        "away_xg_defense_venue",
    )

    return out


# ============================================================
# BUILD LAMBDAS
# ============================================================

def build_lambdas(
    df,
    attack_goal_weight,
    attack_xg_weight,
    attack_shot_weight,
    defense_goal_weight,
    defense_xg_weight,
    defense_shot_weight,
):

    # ========================================================
    # ATTACK
    # ========================================================

    home_attack = (
        attack_goal_weight
        *
        df[
            "home_goal_attack_component"
        ]
        +
        attack_xg_weight
        *
        df[
            "home_xg_attack_component"
        ]
        +
        attack_shot_weight
        *
        df[
            "home_shot_attack_component"
        ]
    )

    away_attack = (
        attack_goal_weight
        *
        df[
            "away_goal_attack_component"
        ]
        +
        attack_xg_weight
        *
        df[
            "away_xg_attack_component"
        ]
        +
        attack_shot_weight
        *
        df[
            "away_shot_attack_component"
        ]
    )

    # ========================================================
    # DEFENSE
    # ========================================================

    home_defense = (
        defense_goal_weight
        *
        df[
            "home_goal_defense_component"
        ]
        +
        defense_xg_weight
        *
        df[
            "home_xg_defense_component"
        ]
        +
        defense_shot_weight
        *
        df[
            "home_shot_defense_component"
        ]
    )

    away_defense = (
        defense_goal_weight
        *
        df[
            "away_goal_defense_component"
        ]
        +
        defense_xg_weight
        *
        df[
            "away_xg_defense_component"
        ]
        +
        defense_shot_weight
        *
        df[
            "away_shot_defense_component"
        ]
    )

    # ========================================================
    # EXPECTED GOALS
    # ========================================================

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
    attack_weights,
    defense_weights,
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

        attack_weights[
            "goal_weight"
        ],

        attack_weights[
            "xg_weight"
        ],

        attack_weights[
            "shot_weight"
        ],

        defense_weights[
            "goal_weight"
        ],

        defense_weights[
            "xg_weight"
        ],

        defense_weights[
            "shot_weight"
        ],
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
        f"{'Symmetric':>14}"
        f"{'Asymmetric':>14}"
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
# FINAL BY LEAGUE
# ============================================================

def print_final_by_league(
    df,
    control_weights,
    attack_weights,
    defense_weights,
):

    print()
    print("=" * 108)
    print("2024/25 — BY LEAGUE")
    print("=" * 108)

    season = (
        df[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    final = df[
        season.isin(
            FINAL_SEASONS
        )
    ].copy()

    rows = []

    for league, sub in (
        final.groupby(
            "league"
        )
    ):

        control = evaluate(
            sub,
            FINAL_SEASONS,
            control_weights,
            control_weights,
        )

        winner = evaluate(
            sub,
            FINAL_SEASONS,
            attack_weights,
            defense_weights,
        )

        rows.append(
            {
                "league":
                    league,

                "games":
                    winner[
                        "games"
                    ],

                "control_ll":
                    control[
                        "log_loss"
                    ],

                "winner_ll":
                    winner[
                        "log_loss"
                    ],

                "ll_change":
                    (
                        winner[
                            "log_loss"
                        ]
                        -
                        control[
                            "log_loss"
                        ]
                    ),

                "control_brier":
                    control[
                        "brier"
                    ],

                "winner_brier":
                    winner[
                        "brier"
                    ],

                "control_acc":
                    (
                        control[
                            "accuracy"
                        ]
                        * 100
                    ),

                "winner_acc":
                    (
                        winner[
                            "accuracy"
                        ]
                        * 100
                    ),
            }
        )

    table = pd.DataFrame(
        rows
    )

    print(
        table
        .round(6)
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
    print("TUNING ATTACK / DEFENSE V5")
    print("==============================")
    print()

    print(
        "Frozen V5 structure:"
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
        "Current symmetric control:"
    )

    print(
        f"Goals: "
        f"{CONTROL_GOAL_WEIGHT:.0%}"
    )

    print(
        f"xG: "
        f"{CONTROL_XG_WEIGHT:.0%}"
    )

    print(
        f"Shots: "
        f"{CONTROL_SHOT_WEIGHT:.0%}"
    )

    # ========================================================
    # BUILD COMPONENT STORE
    # ========================================================

    print()
    print(
        "Building frozen component store..."
    )

    df = build_component_store()

    df = prepare_components(
        df
    )

    print(
        f"Eligible matches: "
        f"{len(df):,}"
    )

    # ========================================================
    # GRIDS
    # ========================================================

    attack_grid = (
        build_attack_grid()
    )

    defense_grid = (
        build_defense_grid()
    )

    combinations = (
        len(attack_grid)
        *
        len(defense_grid)
    )

    print()
    print(
        f"Valid attack blends: "
        f"{len(attack_grid):,}"
    )

    print(
        f"Valid defense blends: "
        f"{len(defense_grid):,}"
    )

    print(
        f"Total combinations: "
        f"{combinations:,}"
    )

    # ========================================================
    # TUNE
    # ========================================================

    rows = []

    tested = 0

    for attack in attack_grid:

        for defense in defense_grid:

            metrics = evaluate(
                df,
                TUNING_SEASONS,
                attack,
                defense,
            )

            rows.append(
                {
                    "attack_goal_weight":
                        attack[
                            "goal_weight"
                        ],

                    "attack_xg_weight":
                        attack[
                            "xg_weight"
                        ],

                    "attack_shot_weight":
                        attack[
                            "shot_weight"
                        ],

                    "defense_goal_weight":
                        defense[
                            "goal_weight"
                        ],

                    "defense_xg_weight":
                        defense[
                            "xg_weight"
                        ],

                    "defense_shot_weight":
                        defense[
                            "shot_weight"
                        ],

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

            tested += 1

            if (
                tested
                % 500
                == 0
            ):

                print(
                    f"Tested "
                    f"{tested:,}/"
                    f"{combinations:,}"
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
    # TOP 25
    # ========================================================

    print()
    print("==============================")
    print("TOP 25 ATTACK / DEFENSE BLENDS")
    print("==============================")
    print()

    display = (
        results
        .head(25)
        .copy()
    )

    display[
        "accuracy"
    ] *= 100.0

    print(
        display[
            [
                "rank",

                "attack_goal_weight",
                "attack_xg_weight",
                "attack_shot_weight",

                "defense_goal_weight",
                "defense_xg_weight",
                "defense_shot_weight",

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

    attack_winner = {
        "goal_weight":
            float(
                best[
                    "attack_goal_weight"
                ]
            ),

        "xg_weight":
            float(
                best[
                    "attack_xg_weight"
                ]
            ),

        "shot_weight":
            float(
                best[
                    "attack_shot_weight"
                ]
            ),
    }

    defense_winner = {
        "goal_weight":
            float(
                best[
                    "defense_goal_weight"
                ]
            ),

        "xg_weight":
            float(
                best[
                    "defense_xg_weight"
                ]
            ),

        "shot_weight":
            float(
                best[
                    "defense_shot_weight"
                ]
            ),
    }

    control_weights = {
        "goal_weight":
            CONTROL_GOAL_WEIGHT,

        "xg_weight":
            CONTROL_XG_WEIGHT,

        "shot_weight":
            CONTROL_SHOT_WEIGHT,
    }

    print()
    print("==============================")
    print("WINNING ASYMMETRIC SIGNAL BLEND")
    print("==============================")

    print()
    print("ATTACK")

    print(
        f"Goals: "
        f"{attack_winner['goal_weight']:.1%}"
    )

    print(
        f"xG: "
        f"{attack_winner['xg_weight']:.1%}"
    )

    print(
        f"Shots: "
        f"{attack_winner['shot_weight']:.1%}"
    )

    print()
    print("DEFENSE")

    print(
        f"Goals: "
        f"{defense_winner['goal_weight']:.1%}"
    )

    print(
        f"xG: "
        f"{defense_winner['xg_weight']:.1%}"
    )

    print(
        f"Shots: "
        f"{defense_winner['shot_weight']:.1%}"
    )

    print()

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
        control_weights,
        control_weights,
    )

    winner_tune = evaluate(
        df,
        TUNING_SEASONS,
        attack_winner,
        defense_winner,
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
        control_weights,
        control_weights,
    )

    winner_validation = evaluate(
        df,
        VALIDATION_SEASONS,
        attack_winner,
        defense_winner,
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
        control_weights,
        control_weights,
    )

    winner_final = evaluate(
        df,
        FINAL_SEASONS,
        attack_winner,
        defense_winner,
    )

    print_comparison(
        "FINAL CHECK — 2024/25",
        control_final,
        winner_final,
    )

    # ========================================================
    # BY LEAGUE
    # ========================================================

    print_final_by_league(
        df,
        control_weights,
        attack_winner,
        defense_winner,
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    (
        home_lambda,
        away_lambda,
    ) = build_lambdas(
        df,

        attack_winner[
            "goal_weight"
        ],

        attack_winner[
            "xg_weight"
        ],

        attack_winner[
            "shot_weight"
        ],

        defense_winner[
            "goal_weight"
        ],

        defense_winner[
            "xg_weight"
        ],

        defense_winner[
            "shot_weight"
        ],
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

    # ========================================================
    # STORE SETTINGS
    # ========================================================

    output[
        "attack_goal_weight_v5"
    ] = attack_winner[
        "goal_weight"
    ]

    output[
        "attack_xg_weight_v5"
    ] = attack_winner[
        "xg_weight"
    ]

    output[
        "attack_shot_weight_v5"
    ] = attack_winner[
        "shot_weight"
    ]

    output[
        "defense_goal_weight_v5"
    ] = defense_winner[
        "goal_weight"
    ]

    output[
        "defense_xg_weight_v5"
    ] = defense_winner[
        "xg_weight"
    ]

    output[
        "defense_shot_weight_v5"
    ] = defense_winner[
        "shot_weight"
    ]

    output[
        "goal_recency_v5"
    ] = GOAL_RECENCY

    output[
        "xg_recency_v5"
    ] = XG_RECENCY

    output[
        "shot_recency_v5"
    ] = SHOT_RECENCY

    output[
        "opponent_strength_v5"
    ] = OPPONENT_STRENGTH

    output[
        "overall_weight_v5"
    ] = OVERALL_WEIGHT

    output[
        "venue_weight_v5"
    ] = VENUE_WEIGHT

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("==============================")
    print("ATTACK / DEFENSE TUNING COMPLETE")
    print("==============================")

    print(
        "Recencies frozen ✅"
    )

    print(
        "Opponent strength frozen "
        "at 0.875 ✅"
    )

    print(
        "Overall / venue frozen "
        "at 80% / 20% ✅"
    )

    print(
        "Expanded asymmetric weights "
        "selected using "
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