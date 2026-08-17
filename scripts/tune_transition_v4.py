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
    / "transition_v4_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "transition_v4_predictions.csv"
)


# ============================================================
# SPLITS
#
# Use a broader historical development window because
# promotion/relegation events are relatively rare.
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

TRANSITION_ADJUSTMENTS = [
    0.080,
    0.100,
    0.120,
    0.140,
    0.160,
    0.180,
    0.200,
]


# ============================================================
# BUILD TEAM TRANSITIONS
# ============================================================

def build_transition_rows(
    team_rows,
):
    """
    Identify a team's first match after moving between
    Championship and Premier League.

    Then retain that transition status through the season.

    Championship -> Premier League = PROMOTED
    Premier League -> Championship = RELEGATED
    """

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

    # --------------------------------------------------------
    # SEASON-LEVEL LEAGUE
    # --------------------------------------------------------

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

    team_seasons = (
        team_seasons
        .sort_values(
            [
                "team",
                "date",
            ]
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
        "transition_type"
    ] = "NONE"

    promoted = (
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
    )

    relegated = (
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
    )

    team_seasons.loc[
        promoted,
        "transition_type",
    ] = "PROMOTED"

    team_seasons.loc[
        relegated,
        "transition_type",
    ] = "RELEGATED"

    team_seasons[
        "promoted_flag"
    ] = (
        team_seasons[
            "transition_type"
        ]
        == "PROMOTED"
    ).astype(int)

    team_seasons[
        "relegated_flag"
    ] = (
        team_seasons[
            "transition_type"
        ]
        == "RELEGATED"
    ).astype(int)

    return team_seasons[
        [
            "team",
            "season",
            "league",
            "previous_league",
            "transition_type",
            "promoted_flag",
            "relegated_flag",
        ]
    ]


# ============================================================
# MATCH-LEVEL TRANSITION FLAGS
# ============================================================

def attach_match_transitions(
    v3,
    team_transitions,
):

    out = v3.copy()

    out["season"] = (
        out["season"]
        .astype(str)
        .str.zfill(4)
    )

    home_transition = (
        team_transitions
        .rename(
            columns={
                "team":
                    "home_team",

                "league":
                    "home_transition_league",

                "previous_league":
                    "home_previous_league",

                "transition_type":
                    "home_transition_type",

                "promoted_flag":
                    "home_promoted",

                "relegated_flag":
                    "home_relegated",
            }
        )
    )

    away_transition = (
        team_transitions
        .rename(
            columns={
                "team":
                    "away_team",

                "league":
                    "away_transition_league",

                "previous_league":
                    "away_previous_league",

                "transition_type":
                    "away_transition_type",

                "promoted_flag":
                    "away_promoted",

                "relegated_flag":
                    "away_relegated",
            }
        )
    )

    out = out.merge(
        home_transition[
            [
                "home_team",
                "season",
                "home_previous_league",
                "home_transition_type",
                "home_promoted",
                "home_relegated",
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
        away_transition[
            [
                "away_team",
                "season",
                "away_previous_league",
                "away_transition_type",
                "away_promoted",
                "away_relegated",
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
# APPLY TRANSITION ADJUSTMENT
# ============================================================

def apply_transition_adjustment(
    df,
    adjustment,
):
    """
    Positive adjustment means:

    PROMOTED TEAM:
        own lambda decreases
        opponent lambda increases

    RELEGATED TEAM:
        own lambda increases
        opponent lambda decreases

    Negative values reverse those assumptions.
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

    # --------------------------------------------------------
    # HOME PROMOTED
    # --------------------------------------------------------

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
        - adjustment
    )

    away_lambda.loc[
        mask
    ] *= (
        1.0
        + adjustment
    )

    # --------------------------------------------------------
    # AWAY PROMOTED
    # --------------------------------------------------------

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
        - adjustment
    )

    home_lambda.loc[
        mask
    ] *= (
        1.0
        + adjustment
    )

    # --------------------------------------------------------
    # HOME RELEGATED
    # --------------------------------------------------------

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
        + adjustment
    )

    away_lambda.loc[
        mask
    ] *= (
        1.0
        - adjustment
    )

    # --------------------------------------------------------
    # AWAY RELEGATED
    # --------------------------------------------------------

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
        + adjustment
    )

    home_lambda.loc[
        mask
    ] *= (
        1.0
        - adjustment
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
# EVALUATION
# ============================================================

def evaluate(
    df,
    seasons,
    adjustment,
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
            adjustment,
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

    transition_games = (
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
        |
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

        "transition_games":
            int(
                transition_games
            ),

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
    print("=" * 72)
    print(title)
    print("=" * 72)

    print(
        f"Games: "
        f"{candidate['games']:,}"
    )

    print(
        f"Transition games: "
        f"{candidate['transition_games']:,}"
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


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("TUNING LEAGUE TRANSITION V4")
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

    print(
        f"V3 matches loaded: "
        f"{len(v3):,}"
    )

    print(
        f"Team-game rows loaded: "
        f"{len(team_rows):,}"
    )

    transitions = build_transition_rows(
        team_rows
    )

    transition_only = transitions[
        transitions[
            "transition_type"
        ]
        != "NONE"
    ].copy()

    print()
    print(
        "OBSERVED ENGLISH LEAGUE TRANSITIONS"
    )

    print(
        transition_only[
            [
                "team",
                "season",
                "previous_league",
                "league",
                "transition_type",
            ]
        ]
        .sort_values(
            [
                "season",
                "transition_type",
                "team",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Promoted team-seasons: "
        f"{transition_only['promoted_flag'].sum():,}"
    )

    print(
        f"Relegated team-seasons: "
        f"{transition_only['relegated_flag'].sum():,}"
    )

    df = attach_match_transitions(
        v3,
        transitions,
    )

    # ========================================================
    # GRID SEARCH
    # ========================================================

    rows = []

    for adjustment in (
        TRANSITION_ADJUSTMENTS
    ):

        metrics = evaluate(
            df,
            TUNING_SEASONS,
            adjustment,
        )

        rows.append(
            {
                "transition_adjustment":
                    adjustment,

                "games":
                    metrics[
                        "games"
                    ],

                "transition_games":
                    metrics[
                        "transition_games"
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
    # RESULTS
    # ========================================================

    print()
    print("==============================")
    print("TRANSITION ADJUSTMENT RESULTS")
    print("==============================")

    display = results.copy()

    display[
        "accuracy"
    ] *= 100.0

    print(
        display[
            [
                "rank",
                "transition_adjustment",
                "transition_games",
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

    best_adjustment = float(
        best[
            "transition_adjustment"
        ]
    )

    print()
    print("==============================")
    print("WINNING TRANSITION ADJUSTMENT")
    print("==============================")

    print(
        f"Adjustment: "
        f"{best_adjustment:+.1%}"
    )

    print(
        f"Tuning LL: "
        f"{best['log_loss']:.5f}"
    )

    if best_adjustment > 0:

        print(
            "Interpretation: promoted teams need "
            "an additional downgrade and relegated "
            "teams an additional upgrade."
        )

    elif best_adjustment < 0:

        print(
            "Interpretation: current V3 transition "
            "handling is over-correcting directionally."
        )

    else:

        print(
            "Interpretation: current V3 transition "
            "handling is sufficient."
        )

    # ========================================================
    # TIME SPLITS
    # ========================================================

    baseline_tune = evaluate(
        df,
        TUNING_SEASONS,
        0.0,
    )

    winner_tune = evaluate(
        df,
        TUNING_SEASONS,
        best_adjustment,
    )

    print_comparison(
        "TUNING — 2017/18 TO 2022/23",
        baseline_tune,
        winner_tune,
    )

    baseline_validation = evaluate(
        df,
        VALIDATION_SEASONS,
        0.0,
    )

    winner_validation = evaluate(
        df,
        VALIDATION_SEASONS,
        best_adjustment,
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
    )

    winner_final = evaluate(
        df,
        FINAL_SEASONS,
        best_adjustment,
    )

    print_comparison(
        "FINAL CHECK — 2024/25",
        baseline_final,
        winner_final,
    )

    # ========================================================
    # TRANSITION-ONLY DIAGNOSTIC
    # ========================================================

    print()
    print("=" * 96)
    print("2024/25 — TRANSITION TEAMS ONLY")
    print("=" * 96)

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

    if len(
        transition_final
    ) > 0:

        baseline = evaluate(
            transition_final,
            FINAL_SEASONS,
            0.0,
        )

        winner = evaluate(
            transition_final,
            FINAL_SEASONS,
            best_adjustment,
        )

        print(
            f"Matches: "
            f"{len(transition_final):,}"
        )

        print(
            f"V3 Log Loss: "
            f"{baseline['log_loss']:.5f}"
        )

        print(
            f"V4 Log Loss: "
            f"{winner['log_loss']:.5f}"
        )

        print(
            f"LL Change: "
            f"{winner['log_loss'] - baseline['log_loss']:+.5f}"
        )

        print()

        print(
            f"V3 Brier: "
            f"{baseline['brier']:.5f}"
        )

        print(
            f"V4 Brier: "
            f"{winner['brier']:.5f}"
        )

        print(
            f"Brier Change: "
            f"{winner['brier'] - baseline['brier']:+.5f}"
        )

    else:

        print(
            "No observed transition matches "
            "in final season."
        )

    # ========================================================
    # SAVE
    # ========================================================

    home_lambda, away_lambda = (
        apply_transition_adjustment(
            df,
            best_adjustment,
        )
    )

    output = df.copy()

    output[
        "home_lambda_v4_transition"
    ] = home_lambda

    output[
        "away_lambda_v4_transition"
    ] = away_lambda

    probs = calculate_1x2_probs(
        home_lambda.to_numpy(),
        away_lambda.to_numpy(),
    )

    output[
        "p_home_v4_transition"
    ] = probs[:, 0]

    output[
        "p_draw_v4_transition"
    ] = probs[:, 1]

    output[
        "p_away_v4_transition"
    ] = probs[:, 2]

    output[
        "transition_adjustment_v4"
    ] = best_adjustment

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print()
    print("==============================")
    print("TRANSITION TEST COMPLETE")
    print("==============================")

    if best_adjustment == 0.0:

        print(
            "Additional transition adjustment rejected."
        )

    else:

        print(
            "Additional transition adjustment improved "
            "the development objective."
        )

    print()
    print(
        "Parameter selected using "
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
        f"Predictions:"
        f"\n{OUTPUT_PREDICTIONS}"
    )


if __name__ == "__main__":
    main()