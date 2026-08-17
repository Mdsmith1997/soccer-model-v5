from pathlib import Path
import math

import numpy as np
import pandas as pd

import confirm_opponent_adjusted_recency_v5 as v5


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "overall_venue_v5_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "overall_venue_v5_predictions.csv"
)


# ============================================================
# FROZEN V5 SETTINGS
# ============================================================

GOAL_WEIGHT = 0.09
XG_WEIGHT = 0.75
SHOT_WEIGHT = 0.16
SOT_WEIGHT = 0.00

GOAL_RECENCY = 0.975
XG_RECENCY = 0.925
SHOT_RECENCY = 0.850

OPPONENT_STRENGTH = 0.875

CONTROL_OVERALL_WEIGHT = 0.75


# ============================================================
# OVERALL / VENUE GRID
# ============================================================

OVERALL_WEIGHTS = np.round(
    np.arange(
        0.50,
        1.001,
        0.05,
    ),
    3,
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


# ============================================================
# POISSON SETTINGS
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


# ============================================================
# POISSON
# ============================================================

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
# BUILD XG COMPONENT TABLE
# ============================================================

def build_xg_component_table(
    xg_team,
):

    # --------------------------------------------------------
    # SAME V5 XG PIPELINE
    # --------------------------------------------------------

    df = v5.add_xg_league_baseline(
        xg_team
    )

    df = v5.add_raw_xg_strength(
        df,
        XG_RECENCY,
    )

    df = v5.attach_opponent_xg(
        df
    )

    df = v5.adjust_xg_performance(
        df
    )

    df = v5.add_adjusted_xg_history(
        df,
        XG_RECENCY,
    )

    df = v5.add_xg_venue_history(
        df,
        XG_RECENCY,
    )

    # --------------------------------------------------------
    # HOME COMPONENTS
    # --------------------------------------------------------

    home = df[
        df[
            "venue"
        ]
        == "HOME"
    ][
        [
            "match_id",

            "adj_xg_attack",
            "adj_xg_defense",

            "venue_adj_xg_attack",
            "venue_adj_xg_defense",
        ]
    ].rename(
        columns={
            "adj_xg_attack":
                "home_xg_attack_overall",

            "adj_xg_defense":
                "home_xg_defense_overall",

            "venue_adj_xg_attack":
                "home_xg_attack_venue",

            "venue_adj_xg_defense":
                "home_xg_defense_venue",
        }
    )

    # --------------------------------------------------------
    # AWAY COMPONENTS
    # --------------------------------------------------------

    away = df[
        df[
            "venue"
        ]
        == "AWAY"
    ][
        [
            "match_id",

            "adj_xg_attack",
            "adj_xg_defense",

            "venue_adj_xg_attack",
            "venue_adj_xg_defense",
        ]
    ].rename(
        columns={
            "adj_xg_attack":
                "away_xg_attack_overall",

            "adj_xg_defense":
                "away_xg_defense_overall",

            "venue_adj_xg_attack":
                "away_xg_attack_venue",

            "venue_adj_xg_defense":
                "away_xg_defense_venue",
        }
    )

    return home.merge(
        away,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )


# ============================================================
# BUILD FROZEN COMPONENT STORE
# ============================================================

def build_component_store():

    # --------------------------------------------------------
    # GOAL / SHOT DATA
    # --------------------------------------------------------

    team = v5.load_team_data()

    goal_shot_matches = (
        v5.build_goal_shot_base(
            team,
            GOAL_RECENCY,
            SHOT_RECENCY,
        )
    )

    # --------------------------------------------------------
    # XG DATA
    # --------------------------------------------------------

    xg = v5.load_xg()

    xg_team = (
        v5.build_xg_team_rows(
            xg
        )
    )

    xg_components = (
        build_xg_component_table(
            xg_team
        )
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    df = goal_shot_matches.merge(
        xg_components,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    return df


# ============================================================
# COMPONENT BLEND
# ============================================================

def blend_component(
    df,
    overall_col,
    venue_col,
    overall_weight,
):

    venue_weight = (
        1.0
        - overall_weight
    )

    return (
        overall_weight
        *
        df[
            overall_col
        ]
        +
        venue_weight
        *
        df[
            venue_col
        ]
    )


# ============================================================
# BUILD LAMBDAS
# ============================================================

def build_lambdas(
    df,
    overall_weight,
):

    # ========================================================
    # GOALS
    # ========================================================

    home_goal_attack = blend_component(
        df,
        "home_adj_goal_attack",
        "home_adj_venue_goal_attack",
        overall_weight,
    )

    home_goal_defense = blend_component(
        df,
        "home_adj_goal_defense",
        "home_adj_venue_goal_defense",
        overall_weight,
    )

    away_goal_attack = blend_component(
        df,
        "away_adj_goal_attack",
        "away_adj_venue_goal_attack",
        overall_weight,
    )

    away_goal_defense = blend_component(
        df,
        "away_adj_goal_defense",
        "away_adj_venue_goal_defense",
        overall_weight,
    )

    # ========================================================
    # SHOTS
    # ========================================================

    home_shot_attack = blend_component(
        df,
        "home_adj_shot_attack",
        "home_adj_venue_shot_attack",
        overall_weight,
    )

    home_shot_defense = blend_component(
        df,
        "home_adj_shot_defense",
        "home_adj_venue_shot_defense",
        overall_weight,
    )

    away_shot_attack = blend_component(
        df,
        "away_adj_shot_attack",
        "away_adj_venue_shot_attack",
        overall_weight,
    )

    away_shot_defense = blend_component(
        df,
        "away_adj_shot_defense",
        "away_adj_venue_shot_defense",
        overall_weight,
    )

    # ========================================================
    # XG
    # ========================================================

    home_xg_attack = blend_component(
        df,
        "home_xg_attack_overall",
        "home_xg_attack_venue",
        overall_weight,
    )

    home_xg_defense = blend_component(
        df,
        "home_xg_defense_overall",
        "home_xg_defense_venue",
        overall_weight,
    )

    away_xg_attack = blend_component(
        df,
        "away_xg_attack_overall",
        "away_xg_attack_venue",
        overall_weight,
    )

    away_xg_defense = blend_component(
        df,
        "away_xg_defense_overall",
        "away_xg_defense_venue",
        overall_weight,
    )

    # ========================================================
    # FROZEN V5 SIGNAL BLEND
    # ========================================================

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
    overall_weight,
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
        overall_weight,
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
    candidate,
):

    print()
    print("=" * 76)
    print(title)
    print("=" * 76)

    print(
        f"Games: "
        f"{candidate['games']:,}"
    )

    print()

    print(
        f"{'Metric':<15}"
        f"{'75/25 Control':>15}"
        f"{'Winner':>14}"
        f"{'Change':>14}"
    )

    print("-" * 58)

    print(
        f"{'Accuracy':<15}"
        f"{control['accuracy']:>14.2%}"
        f"{candidate['accuracy']:>13.2%}"
        f"{candidate['accuracy'] - control['accuracy']:>+13.2%}"
    )

    print(
        f"{'Log Loss':<15}"
        f"{control['log_loss']:>15.5f}"
        f"{candidate['log_loss']:>14.5f}"
        f"{candidate['log_loss'] - control['log_loss']:>+14.5f}"
    )

    print(
        f"{'Brier':<15}"
        f"{control['brier']:>15.5f}"
        f"{candidate['brier']:>14.5f}"
        f"{candidate['brier'] - control['brier']:>+14.5f}"
    )


# ============================================================
# BY LEAGUE
# ============================================================

def print_final_by_league(
    df,
    control_weight,
    winner_weight,
):

    print()
    print("=" * 100)
    print("2024/25 — BY LEAGUE")
    print("=" * 100)

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
            control_weight,
        )

        winner = evaluate(
            sub,
            FINAL_SEASONS,
            winner_weight,
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

    result = pd.DataFrame(
        rows
    )

    print(
        result
        .round(5)
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
    print("TUNING OVERALL / VENUE V5")
    print("==============================")
    print()

    print(
        "Frozen V5 model:"
    )

    print(
        f"Goals / xG / Shots: "
        f"{GOAL_WEIGHT:.0%} / "
        f"{XG_WEIGHT:.0%} / "
        f"{SHOT_WEIGHT:.0%}"
    )

    print(
        f"Goal recency: "
        f"{GOAL_RECENCY:.3f}"
    )

    print(
        f"xG recency: "
        f"{XG_RECENCY:.3f}"
    )

    print(
        f"Shot recency: "
        f"{SHOT_RECENCY:.3f}"
    )

    print(
        f"Opponent strength: "
        f"{OPPONENT_STRENGTH:.3f}"
    )

    print()
    print(
        "Building frozen component store..."
    )

    df = build_component_store()

    print(
        f"Eligible matches: "
        f"{len(df):,}"
    )

    print()
    print(
        f"Overall weights tested: "
        f"{len(OVERALL_WEIGHTS)}"
    )

    print(
        f"Range: "
        f"{OVERALL_WEIGHTS.min():.0%} "
        f"to "
        f"{OVERALL_WEIGHTS.max():.0%}"
    )

    # ========================================================
    # TUNE
    # ========================================================

    rows = []

    for overall_weight in (
        OVERALL_WEIGHTS
    ):

        venue_weight = (
            1.0
            - overall_weight
        )

        metrics = evaluate(
            df,
            TUNING_SEASONS,
            overall_weight,
        )

        rows.append(
            {
                "overall_weight":
                    overall_weight,

                "venue_weight":
                    venue_weight,

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

    # ========================================================
    # SAVE TUNING RESULTS
    # ========================================================

    results.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("==============================")
    print("OVERALL / VENUE RESULTS")
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
                "overall_weight",
                "venue_weight",
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

    best_overall = float(
        best[
            "overall_weight"
        ]
    )

    best_venue = (
        1.0
        - best_overall
    )

    print()
    print("==============================")
    print("WINNING OVERALL / VENUE SPLIT")
    print("==============================")

    print(
        f"Overall: "
        f"{best_overall:.1%}"
    )

    print(
        f"Venue:   "
        f"{best_venue:.1%}"
    )

    print(
        f"Tuning LL: "
        f"{best['log_loss']:.5f}"
    )

    # ========================================================
    # CONTROL
    # ========================================================

    control_tune = evaluate(
        df,
        TUNING_SEASONS,
        CONTROL_OVERALL_WEIGHT,
    )

    winner_tune = evaluate(
        df,
        TUNING_SEASONS,
        best_overall,
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
        CONTROL_OVERALL_WEIGHT,
    )

    winner_validation = evaluate(
        df,
        VALIDATION_SEASONS,
        best_overall,
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
        CONTROL_OVERALL_WEIGHT,
    )

    winner_final = evaluate(
        df,
        FINAL_SEASONS,
        best_overall,
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
        CONTROL_OVERALL_WEIGHT,
        best_overall,
    )

    # ========================================================
    # SAVE WINNING PREDICTIONS
    # ========================================================

    (
        home_lambda,
        away_lambda,
    ) = build_lambdas(
        df,
        best_overall,
    )

    valid = (
        home_lambda.notna()
        &
        away_lambda.notna()
    )

    output = df.loc[
        valid
    ].copy()

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

    # --------------------------------------------------------
    # FROZEN SETTINGS
    # --------------------------------------------------------

    output[
        "goal_weight_v5"
    ] = GOAL_WEIGHT

    output[
        "xg_weight_v5"
    ] = XG_WEIGHT

    output[
        "shot_weight_v5"
    ] = SHOT_WEIGHT

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
    ] = best_overall

    output[
        "venue_weight_v5"
    ] = best_venue

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("==============================")
    print("OVERALL / VENUE TUNING COMPLETE")
    print("==============================")

    print(
        "Signal weights frozen ✅"
    )

    print(
        "Signal recencies frozen ✅"
    )

    print(
        "Opponent strength frozen "
        "at 0.875 ✅"
    )

    print(
        "Overall / venue selected using "
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