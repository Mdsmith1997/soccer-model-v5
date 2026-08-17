from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

V3_FILE = (
    ROOT
    / "data"
    / "processed"
    / "opponent_strength_v3_predictions.csv"
)

TEAM_FILE = (
    ROOT
    / "data"
    / "processed"
    / "team_game_stats.csv"
)

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "transition_asymmetric_v4_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "transition_asymmetric_v4_predictions.csv"
)


# ============================================================
# SPLITS
# ============================================================

TUNING_SEASONS = {
    "1718",
    "1819",
    "1920",
    "2021",
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
# SETTINGS
# ============================================================

MAX_GOALS = 10
EPS = 1e-12

PROMOTION_ADJUSTMENTS = np.round(
    np.arange(
        0.175,
        0.226,
        0.005,
    ),
    3,
)

RELEGATION_ADJUSTMENTS = np.round(
    np.arange(
        0.100,
        0.151,
        0.005,
    ),
    3,
)


# ============================================================
# TRANSITION DETECTION
# ============================================================

def build_transition_rows(
    team_rows,
):

    df = team_rows.copy()

    df["season"] = (
        df["season"]
        .astype(str)
        .str.zfill(4)
    )

    df = (
        df
        .sort_values(
            [
                "team",
                "date",
                "match_id",
            ]
        )
        .reset_index(drop=True)
    )

    team_seasons = (
        df[
            [
                "team",
                "season",
                "league",
                "date",
            ]
        ]
        .sort_values(
            [
                "team",
                "date",
            ]
        )
        .drop_duplicates(
            subset=[
                "team",
                "season",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    team_seasons[
        "previous_league"
    ] = (
        team_seasons
        .groupby(
            "team"
        )[
            "league"
        ]
        .shift(1)
    )

    team_seasons[
        "promoted_flag"
    ] = (
        (
            team_seasons[
                "previous_league"
            ]
            == "Championship"
        )
        &
        (
            team_seasons[
                "league"
            ]
            == "Premier League"
        )
    ).astype(int)

    team_seasons[
        "relegated_flag"
    ] = (
        (
            team_seasons[
                "previous_league"
            ]
            == "Premier League"
        )
        &
        (
            team_seasons[
                "league"
            ]
            == "Championship"
        )
    ).astype(int)

    team_seasons[
        "transition_type"
    ] = np.where(
        team_seasons[
            "promoted_flag"
        ]
        == 1,
        "PROMOTED",
        np.where(
            team_seasons[
                "relegated_flag"
            ]
            == 1,
            "RELEGATED",
            "NONE",
        ),
    )

    return team_seasons[
        [
            "team",
            "season",
            "league",
            "previous_league",
            "promoted_flag",
            "relegated_flag",
            "transition_type",
        ]
    ]


# ============================================================
# ATTACH TO MATCHES
# ============================================================

def attach_match_transitions(
    v3,
    transitions,
):

    out = v3.copy()

    out["season"] = (
        out["season"]
        .astype(str)
        .str.zfill(4)
    )

    home = transitions.rename(
        columns={
            "team":
                "home_team",

            "previous_league":
                "home_previous_league",

            "promoted_flag":
                "home_promoted",

            "relegated_flag":
                "home_relegated",

            "transition_type":
                "home_transition_type",
        }
    )

    away = transitions.rename(
        columns={
            "team":
                "away_team",

            "previous_league":
                "away_previous_league",

            "promoted_flag":
                "away_promoted",

            "relegated_flag":
                "away_relegated",

            "transition_type":
                "away_transition_type",
        }
    )

    out = out.merge(
        home[
            [
                "home_team",
                "season",
                "home_previous_league",
                "home_promoted",
                "home_relegated",
                "home_transition_type",
            ]
        ],
        on=[
            "home_team",
            "season",
        ],
        how="left",
        validate="many_to_one",
    )

    out = out.merge(
        away[
            [
                "away_team",
                "season",
                "away_previous_league",
                "away_promoted",
                "away_relegated",
                "away_transition_type",
            ]
        ],
        on=[
            "away_team",
            "season",
        ],
        how="left",
        validate="many_to_one",
    )

    for col in [
        "home_promoted",
        "home_relegated",
        "away_promoted",
        "away_relegated",
    ]:

        out[col] = (
            out[col]
            .fillna(0)
            .astype(int)
        )

    return out


# ============================================================
# POISSON
# ============================================================

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
            home_goals
            == away_goals,
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


# ============================================================
# APPLY ASYMMETRIC TRANSITION ADJUSTMENT
# ============================================================

def apply_transition_adjustment(
    df,
    promotion_adjustment,
    relegation_adjustment,
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

    # ========================================================
    # PROMOTED TEAMS
    # ========================================================

    mask = (
        df[
            "home_promoted"
        ]
        == 1
    )

    home_lambda.loc[
        mask
    ] *= (
        1.0
        -
        promotion_adjustment
    )

    away_lambda.loc[
        mask
    ] *= (
        1.0
        +
        promotion_adjustment
    )

    mask = (
        df[
            "away_promoted"
        ]
        == 1
    )

    away_lambda.loc[
        mask
    ] *= (
        1.0
        -
        promotion_adjustment
    )

    home_lambda.loc[
        mask
    ] *= (
        1.0
        +
        promotion_adjustment
    )

    # ========================================================
    # RELEGATED TEAMS
    # ========================================================

    mask = (
        df[
            "home_relegated"
        ]
        == 1
    )

    home_lambda.loc[
        mask
    ] *= (
        1.0
        +
        relegation_adjustment
    )

    away_lambda.loc[
        mask
    ] *= (
        1.0
        -
        relegation_adjustment
    )

    mask = (
        df[
            "away_relegated"
        ]
        == 1
    )

    away_lambda.loc[
        mask
    ] *= (
        1.0
        +
        relegation_adjustment
    )

    home_lambda.loc[
        mask
    ] *= (
        1.0
        -
        relegation_adjustment
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


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    df,
    seasons,
    promotion_adjustment,
    relegation_adjustment,
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
        apply_transition_adjustment(
            sub,
            promotion_adjustment,
            relegation_adjustment,
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

    promoted_games = (
        (
            sub[
                "home_promoted"
            ]
            == 1
        )
        |
        (
            sub[
                "away_promoted"
            ]
            == 1
        )
    ).sum()

    relegated_games = (
        (
            sub[
                "home_relegated"
            ]
            == 1
        )
        |
        (
            sub[
                "away_relegated"
            ]
            == 1
        )
    ).sum()

    return {
        "games":
            len(sub),

        "promoted_games":
            int(
                promoted_games
            ),

        "relegated_games":
            int(
                relegated_games
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


# ============================================================
# PRINT COMPARISON
# ============================================================

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
        f"Promoted-team games: "
        f"{candidate['promoted_games']:,}"
    )

    print(
        f"Relegated-team games: "
        f"{candidate['relegated_games']:,}"
    )

    print()

    print(
        f"{'Metric':<15}"
        f"{'V3':>14}"
        f"{'V4 Asym':>14}"
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


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("TUNING ASYMMETRIC TRANSITIONS")
    print("==============================")
    print()

    v3 = pd.read_csv(
        V3_FILE,
        parse_dates=[
            "date",
        ],
    )

    team_rows = pd.read_csv(
        TEAM_FILE,
        parse_dates=[
            "date",
        ],
    )

    transitions = build_transition_rows(
        team_rows
    )

    df = attach_match_transitions(
        v3,
        transitions,
    )

    print(
        f"V3 matches loaded: "
        f"{len(v3):,}"
    )

    print()
    print(
        f"Promotion settings: "
        f"{len(PROMOTION_ADJUSTMENTS)}"
    )

    print(
        f"Relegation settings: "
        f"{len(RELEGATION_ADJUSTMENTS)}"
    )

    print(
        f"Combinations: "
        f"{len(PROMOTION_ADJUSTMENTS) * len(RELEGATION_ADJUSTMENTS):,}"
    )

    # ========================================================
    # GRID SEARCH
    # ========================================================

    rows = []

    for promotion in PROMOTION_ADJUSTMENTS:

        for relegation in RELEGATION_ADJUSTMENTS:

            metrics = evaluate(
                df,
                TUNING_SEASONS,
                promotion,
                relegation,
            )

            rows.append(
                {
                    "promotion_adjustment":
                        promotion,

                    "relegation_adjustment":
                        relegation,

                    "games":
                        metrics[
                            "games"
                        ],

                    "promoted_games":
                        metrics[
                            "promoted_games"
                        ],

                    "relegated_games":
                        metrics[
                            "relegated_games"
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
    # TOP 20
    # ========================================================

    print()
    print("==============================")
    print("TOP 20 ASYMMETRIC SETTINGS")
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
                "promotion_adjustment",
                "relegation_adjustment",
                "promoted_games",
                "relegated_games",
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

    best_promotion = float(
        best[
            "promotion_adjustment"
        ]
    )

    best_relegation = float(
        best[
            "relegation_adjustment"
        ]
    )

    print()
    print("==============================")
    print("WINNING ASYMMETRIC MODEL")
    print("==============================")

    print(
        f"Promotion downgrade: "
        f"{best_promotion:.1%}"
    )

    print(
        f"Relegation upgrade: "
        f"{best_relegation:.1%}"
    )

    print(
        f"Tuning LL: "
        f"{best['log_loss']:.5f}"
    )

    # ========================================================
    # TUNING
    # ========================================================

    baseline_tune = evaluate(
        df,
        TUNING_SEASONS,
        0.0,
        0.0,
    )

    winner_tune = evaluate(
        df,
        TUNING_SEASONS,
        best_promotion,
        best_relegation,
    )

    print_comparison(
        "TUNING — 2017/18 TO 2022/23",
        baseline_tune,
        winner_tune,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    baseline_validation = evaluate(
        df,
        VALIDATION_SEASONS,
        0.0,
        0.0,
    )

    winner_validation = evaluate(
        df,
        VALIDATION_SEASONS,
        best_promotion,
        best_relegation,
    )

    print_comparison(
        "VALIDATION — 2023/24",
        baseline_validation,
        winner_validation,
    )

    # ========================================================
    # FINAL
    # ========================================================

    baseline_final = evaluate(
        df,
        FINAL_SEASONS,
        0.0,
        0.0,
    )

    winner_final = evaluate(
        df,
        FINAL_SEASONS,
        best_promotion,
        best_relegation,
    )

    print_comparison(
        "FINAL CHECK — 2024/25",
        baseline_final,
        winner_final,
    )

    # ========================================================
    # TRANSITION-ONLY FINAL
    # ========================================================

    print()
    print("=" * 98)
    print("2024/25 — TRANSITION MATCHES ONLY")
    print("=" * 98)

    season = (
        df[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    transition_mask = (
        season.isin(
            FINAL_SEASONS
        )
        &
        (
            (
                df[
                    "home_promoted"
                ]
                == 1
            )
            |
            (
                df[
                    "away_promoted"
                ]
                == 1
            )
            |
            (
                df[
                    "home_relegated"
                ]
                == 1
            )
            |
            (
                df[
                    "away_relegated"
                ]
                == 1
            )
        )
    )

    transition_final = df.loc[
        transition_mask
    ].copy()

    baseline_transition = evaluate(
        transition_final,
        FINAL_SEASONS,
        0.0,
        0.0,
    )

    winner_transition = evaluate(
        transition_final,
        FINAL_SEASONS,
        best_promotion,
        best_relegation,
    )

    print(
        f"Matches: "
        f"{len(transition_final):,}"
    )

    print()

    print(
        f"V3 Log Loss: "
        f"{baseline_transition['log_loss']:.5f}"
    )

    print(
        f"Asym Log Loss: "
        f"{winner_transition['log_loss']:.5f}"
    )

    print(
        f"LL Change: "
        f"{winner_transition['log_loss'] - baseline_transition['log_loss']:+.5f}"
    )

    print()

    print(
        f"V3 Brier: "
        f"{baseline_transition['brier']:.5f}"
    )

    print(
        f"Asym Brier: "
        f"{winner_transition['brier']:.5f}"
    )

    print(
        f"Brier Change: "
        f"{winner_transition['brier'] - baseline_transition['brier']:+.5f}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    home_lambda, away_lambda = (
        apply_transition_adjustment(
            df,
            best_promotion,
            best_relegation,
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
        home_lambda.to_numpy(),
        away_lambda.to_numpy(),
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
        "promotion_adjustment_v4"
    ] = best_promotion

    output[
        "relegation_adjustment_v4"
    ] = best_relegation

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print()
    print("==============================")
    print("ASYMMETRIC TRANSITION TEST COMPLETE")
    print("==============================")

    print(
        "Parameters selected using "
        "2017/18–2022/23 only ✅"
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
        f"V4 predictions:"
        f"\n{OUTPUT_PREDICTIONS}"
    )


if __name__ == "__main__":
    main()
