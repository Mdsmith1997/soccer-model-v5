from pathlib import Path
import math
import time

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
    / "team_game_stats.csv"
)

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "poisson_v1_tuning_results.csv"
)

OUTPUT_BEST = (
    ROOT
    / "data"
    / "processed"
    / "poisson_v1_best_predictions.csv"
)


# =========================================================
# DATA SPLIT
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


# =========================================================
# PARAMETER GRID
# =========================================================

RECENCY_VALUES = [
    0.80,
    0.85,
    0.90,
    0.93,
    0.95,
]

VENUE_WEIGHTS = [
    0.00,
    0.25,
    0.50,
    0.75,
]

OVERALL_SHRINK_VALUES = [
    4.0,
    8.0,
    12.0,
    20.0,
]

VENUE_SHRINK_VALUES = [
    4.0,
    8.0,
    12.0,
]

LEAGUE_CHANGE_CONFIDENCE = 0.70

MIN_PRIOR_GAMES = 5

MAX_GOALS = 10

EPS = 1e-12


# =========================================================
# ROLLING FEATURES
# =========================================================

ROLLING_COLUMNS = [
    "goals_for",
    "goals_against",
]


# =========================================================
# EXPONENTIALLY WEIGHTED HISTORY
# =========================================================

def weighted_prior_average(
    values,
    decay,
):
    """
    Exponentially weighted average using ONLY prior matches.

    Current match is never included.
    """

    results = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    numerator = 0.0
    denominator = 0.0

    arr = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy()

    for i, value in enumerate(arr):

        if denominator > 0:
            results[i] = (
                numerator
                / denominator
            )

        numerator *= decay
        denominator *= decay

        if not np.isnan(value):
            numerator += value
            denominator += 1.0

    return pd.Series(
        results,
        index=values.index,
    )


# =========================================================
# LEAGUE BASELINES
# =========================================================

def build_league_baselines(
    team_games,
):
    """
    Historical league home/away scoring rates.

    Same-day matches do NOT see one another's results.
    """

    home = (
        team_games[
            team_games["venue"] == "HOME"
        ][
            [
                "match_id",
                "date",
                "league_code",
                "league",
                "goals_for",
            ]
        ]
        .rename(
            columns={
                "goals_for":
                    "home_goals"
            }
        )
    )

    away = (
        team_games[
            team_games["venue"] == "AWAY"
        ][
            [
                "match_id",
                "goals_for",
            ]
        ]
        .rename(
            columns={
                "goals_for":
                    "away_goals"
            }
        )
    )

    matches = home.merge(
        away,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    daily = (
        matches
        .groupby(
            [
                "league_code",
                "league",
                "date",
            ],
            as_index=False,
        )
        .agg(
            daily_matches=(
                "match_id",
                "count",
            ),
            daily_home_goals=(
                "home_goals",
                "sum",
            ),
            daily_away_goals=(
                "away_goals",
                "sum",
            ),
        )
        .sort_values(
            [
                "league_code",
                "date",
            ]
        )
    )

    grouped = daily.groupby(
        "league_code"
    )

    daily[
        "league_prior_matches"
    ] = (
        grouped[
            "daily_matches"
        ].cumsum()
        -
        daily[
            "daily_matches"
        ]
    )

    daily[
        "prior_home_goals"
    ] = (
        grouped[
            "daily_home_goals"
        ].cumsum()
        -
        daily[
            "daily_home_goals"
        ]
    )

    daily[
        "prior_away_goals"
    ] = (
        grouped[
            "daily_away_goals"
        ].cumsum()
        -
        daily[
            "daily_away_goals"
        ]
    )

    daily[
        "league_avg_home_goals"
    ] = (
        daily[
            "prior_home_goals"
        ]
        /
        daily[
            "league_prior_matches"
        ]
    )

    daily[
        "league_avg_away_goals"
    ] = (
        daily[
            "prior_away_goals"
        ]
        /
        daily[
            "league_prior_matches"
        ]
    )

    daily[
        "league_avg_team_goals"
    ] = (
        daily[
            "league_avg_home_goals"
        ]
        +
        daily[
            "league_avg_away_goals"
        ]
    ) / 2.0

    return daily[
        [
            "league_code",
            "date",
            "league_prior_matches",
            "league_avg_home_goals",
            "league_avg_away_goals",
            "league_avg_team_goals",
        ]
    ]


# =========================================================
# BUILD HISTORY FOR ONE RECENCY VALUE
# =========================================================

def build_recency_features(
    base,
    decay,
):
    df = base.copy()

    # -----------------------------------------------------
    # OVERALL TEAM HISTORY
    # -----------------------------------------------------

    team_group = df.groupby(
        "team",
        sort=False,
        group_keys=False,
    )

    df[
        "pregame_games"
    ] = team_group.cumcount()

    for column in ROLLING_COLUMNS:

        df[
            f"overall_{column}"
        ] = (
            team_group[
                column
            ]
            .transform(
                lambda x:
                    weighted_prior_average(
                        x,
                        decay,
                    )
            )
        )

    # -----------------------------------------------------
    # VENUE-SPECIFIC HISTORY
    # -----------------------------------------------------

    venue_group = df.groupby(
        [
            "team",
            "venue",
        ],
        sort=False,
        group_keys=False,
    )

    df[
        "pregame_venue_games"
    ] = (
        venue_group.cumcount()
    )

    for column in ROLLING_COLUMNS:

        df[
            f"venue_{column}"
        ] = (
            venue_group[
                column
            ]
            .transform(
                lambda x:
                    weighted_prior_average(
                        x,
                        decay,
                    )
            )
        )

    # -----------------------------------------------------
    # LEAGUE CHANGES
    # -----------------------------------------------------

    df[
        "previous_league_code"
    ] = (
        team_group[
            "league_code"
        ]
        .shift(1)
    )

    df[
        "league_changed"
    ] = (
        df[
            "previous_league_code"
        ].notna()
        &
        (
            df[
                "previous_league_code"
            ]
            != df[
                "league_code"
            ]
        )
    ).astype(int)

    # -----------------------------------------------------
    # RAW RELATIVE STRENGTHS
    # -----------------------------------------------------

    df[
        "overall_attack_raw"
    ] = (
        df[
            "overall_goals_for"
        ]
        /
        df[
            "league_avg_team_goals"
        ]
    )

    df[
        "overall_defense_raw"
    ] = (
        df[
            "overall_goals_against"
        ]
        /
        df[
            "league_avg_team_goals"
        ]
    )

    home_mask = (
        df["venue"] == "HOME"
    )

    away_mask = (
        df["venue"] == "AWAY"
    )

    df[
        "venue_attack_raw"
    ] = np.nan

    df[
        "venue_defense_raw"
    ] = np.nan

    # Home attack relative to league home scoring.
    df.loc[
        home_mask,
        "venue_attack_raw",
    ] = (
        df.loc[
            home_mask,
            "venue_goals_for",
        ]
        /
        df.loc[
            home_mask,
            "league_avg_home_goals",
        ]
    )

    # Home defense concedes against normal away scoring.
    df.loc[
        home_mask,
        "venue_defense_raw",
    ] = (
        df.loc[
            home_mask,
            "venue_goals_against",
        ]
        /
        df.loc[
            home_mask,
            "league_avg_away_goals",
        ]
    )

    # Away attack relative to league away scoring.
    df.loc[
        away_mask,
        "venue_attack_raw",
    ] = (
        df.loc[
            away_mask,
            "venue_goals_for",
        ]
        /
        df.loc[
            away_mask,
            "league_avg_away_goals",
        ]
    )

    # Away defense concedes against normal home scoring.
    df.loc[
        away_mask,
        "venue_defense_raw",
    ] = (
        df.loc[
            away_mask,
            "venue_goals_against",
        ]
        /
        df.loc[
            away_mask,
            "league_avg_home_goals",
        ]
    )

    return df


# =========================================================
# SHRINKAGE
# =========================================================

def shrink(
    rating,
    games,
    k,
):
    weight = (
        games.astype(float)
        /
        (
            games.astype(float)
            + k
        )
    )

    return (
        weight
        * rating
        +
        (
            1.0 - weight
        )
        * 1.0
    )


def transition_adjust(
    rating,
    changed,
):
    adjusted = rating.copy()

    mask = (
        changed == 1
    ) & adjusted.notna()

    adjusted.loc[
        mask
    ] = (
        1.0
        +
        (
            adjusted.loc[
                mask
            ]
            - 1.0
        )
        * LEAGUE_CHANGE_CONFIDENCE
    )

    return adjusted


# =========================================================
# HOME / AWAY MATCH TABLE
# =========================================================

def build_match_table(
    df,
):
    home = (
        df[
            df["venue"] == "HOME"
        ]
        .copy()
    )

    away = (
        df[
            df["venue"] == "AWAY"
        ]
        .copy()
    )

    home = home[
        [
            "match_id",
            "date",
            "season",
            "league_code",
            "league",

            "team",
            "opponent",

            "goals_for",
            "goals_against",

            "pregame_games",
            "pregame_venue_games",

            "league_changed",

            "league_avg_home_goals",
            "league_avg_away_goals",

            "overall_attack_raw",
            "overall_defense_raw",

            "venue_attack_raw",
            "venue_defense_raw",
        ]
    ].rename(
        columns={
            "team":
                "home_team",

            "opponent":
                "away_team_check",

            "goals_for":
                "home_goals",

            "goals_against":
                "away_goals",

            "pregame_games":
                "home_games",

            "pregame_venue_games":
                "home_venue_games",

            "league_changed":
                "home_league_changed",

            "overall_attack_raw":
                "home_overall_attack_raw",

            "overall_defense_raw":
                "home_overall_defense_raw",

            "venue_attack_raw":
                "home_venue_attack_raw",

            "venue_defense_raw":
                "home_venue_defense_raw",
        }
    )

    away = away[
        [
            "match_id",
            "team",
            "opponent",

            "pregame_games",
            "pregame_venue_games",

            "league_changed",

            "overall_attack_raw",
            "overall_defense_raw",

            "venue_attack_raw",
            "venue_defense_raw",
        ]
    ].rename(
        columns={
            "team":
                "away_team",

            "opponent":
                "home_team_check",

            "pregame_games":
                "away_games",

            "pregame_venue_games":
                "away_venue_games",

            "league_changed":
                "away_league_changed",

            "overall_attack_raw":
                "away_overall_attack_raw",

            "overall_defense_raw":
                "away_overall_defense_raw",

            "venue_attack_raw":
                "away_venue_attack_raw",

            "venue_defense_raw":
                "away_venue_defense_raw",
        }
    )

    matches = home.merge(
        away,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    return matches


# =========================================================
# POISSON 1X2 — VECTORIZED
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
    """
    Returns N x (MAX_GOALS + 1) array.
    """

    lambdas = np.asarray(
        lambdas,
        dtype=float,
    )

    k = np.arange(
        MAX_GOALS + 1
    )

    probs = (
        np.exp(
            -lambdas[:, None]
        )
        *
        (
            lambdas[:, None]
            ** k[None, :]
        )
        /
        FACTORIALS[
            None,
            :
        ]
    )

    # Normalize truncated distribution.
    probs = (
        probs
        /
        probs.sum(
            axis=1,
            keepdims=True,
        )
    )

    return probs


def calculate_1x2_probs(
    home_lambda,
    away_lambda,
):
    home_probs = poisson_probabilities(
        home_lambda
    )

    away_probs = poisson_probabilities(
        away_lambda
    )

    # CDF for away goals.
    away_cdf = np.cumsum(
        away_probs,
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
            home_probs[:, h]
            *
            away_cdf[:, h - 1]
        )

    # CDF for home goals.
    home_cdf = np.cumsum(
        home_probs,
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
            away_probs[:, a]
            *
            home_cdf[:, a - 1]
        )

    draw = (
        home_probs
        * away_probs
    ).sum(
        axis=1
    )

    total = (
        home_win
        + draw
        + away_win
    )

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
# LOG LOSS / BRIER / ACCURACY
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
        predicted == y_true
    ).mean()


# =========================================================
# EVALUATE ONE PARAMETER SET
# =========================================================

def evaluate_params(
    matches,
    overall_k,
    venue_k,
    venue_weight,
    seasons,
):
    df = matches.copy()

    # -----------------------------------------------------
    # SHRINK RATINGS
    # -----------------------------------------------------

    home_overall_attack = shrink(
        df[
            "home_overall_attack_raw"
        ],
        df[
            "home_games"
        ],
        overall_k,
    )

    home_overall_defense = shrink(
        df[
            "home_overall_defense_raw"
        ],
        df[
            "home_games"
        ],
        overall_k,
    )

    away_overall_attack = shrink(
        df[
            "away_overall_attack_raw"
        ],
        df[
            "away_games"
        ],
        overall_k,
    )

    away_overall_defense = shrink(
        df[
            "away_overall_defense_raw"
        ],
        df[
            "away_games"
        ],
        overall_k,
    )

    home_venue_attack = shrink(
        df[
            "home_venue_attack_raw"
        ],
        df[
            "home_venue_games"
        ],
        venue_k,
    )

    home_venue_defense = shrink(
        df[
            "home_venue_defense_raw"
        ],
        df[
            "home_venue_games"
        ],
        venue_k,
    )

    away_venue_attack = shrink(
        df[
            "away_venue_attack_raw"
        ],
        df[
            "away_venue_games"
        ],
        venue_k,
    )

    away_venue_defense = shrink(
        df[
            "away_venue_defense_raw"
        ],
        df[
            "away_venue_games"
        ],
        venue_k,
    )

    # -----------------------------------------------------
    # LEAGUE CHANGE SHRINKAGE
    # -----------------------------------------------------

    home_overall_attack = transition_adjust(
        home_overall_attack,
        df[
            "home_league_changed"
        ],
    )

    home_overall_defense = transition_adjust(
        home_overall_defense,
        df[
            "home_league_changed"
        ],
    )

    away_overall_attack = transition_adjust(
        away_overall_attack,
        df[
            "away_league_changed"
        ],
    )

    away_overall_defense = transition_adjust(
        away_overall_defense,
        df[
            "away_league_changed"
        ],
    )

    home_venue_attack = transition_adjust(
        home_venue_attack,
        df[
            "home_league_changed"
        ],
    )

    home_venue_defense = transition_adjust(
        home_venue_defense,
        df[
            "home_league_changed"
        ],
    )

    away_venue_attack = transition_adjust(
        away_venue_attack,
        df[
            "away_league_changed"
        ],
    )

    away_venue_defense = transition_adjust(
        away_venue_defense,
        df[
            "away_league_changed"
        ],
    )

    # -----------------------------------------------------
    # BLEND OVERALL + VENUE
    # -----------------------------------------------------

    overall_weight = (
        1.0 - venue_weight
    )

    home_attack = (
        overall_weight
        * home_overall_attack
        +
        venue_weight
        * home_venue_attack
    )

    home_defense = (
        overall_weight
        * home_overall_defense
        +
        venue_weight
        * home_venue_defense
    )

    away_attack = (
        overall_weight
        * away_overall_attack
        +
        venue_weight
        * away_venue_attack
    )

    away_defense = (
        overall_weight
        * away_overall_defense
        +
        venue_weight
        * away_venue_defense
    )

    # -----------------------------------------------------
    # EXPECTED GOALS
    # -----------------------------------------------------

    home_lambda = (
        df[
            "league_avg_home_goals"
        ]
        *
        home_attack
        *
        away_defense
    )

    away_lambda = (
        df[
            "league_avg_away_goals"
        ]
        *
        away_attack
        *
        home_defense
    )

    home_lambda = home_lambda.clip(
        lower=0.15,
        upper=4.50,
    )

    away_lambda = away_lambda.clip(
        lower=0.15,
        upper=4.50,
    )

    # -----------------------------------------------------
    # VALID ROWS
    # -----------------------------------------------------

    season_strings = (
        df[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    mask = (
        season_strings.isin(
            seasons
        )
        &
        (
            df[
                "home_games"
            ]
            >= MIN_PRIOR_GAMES
        )
        &
        (
            df[
                "away_games"
            ]
            >= MIN_PRIOR_GAMES
        )
        &
        home_lambda.notna()
        &
        away_lambda.notna()
    )

    subset = df.loc[
        mask
    ].copy()

    h_lambda = (
        home_lambda.loc[
            mask
        ]
        .to_numpy()
    )

    a_lambda = (
        away_lambda.loc[
            mask
        ]
        .to_numpy()
    )

    probs = calculate_1x2_probs(
        h_lambda,
        a_lambda,
    )

    y_true = result_classes(
        subset[
            "home_goals"
        ].to_numpy(),
        subset[
            "away_goals"
        ].to_numpy(),
    )

    return {
        "games":
            len(subset),

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

        "accuracy":
            accuracy_score(
                y_true,
                probs,
            ),
    }


# =========================================================
# CREATE BEST-PARAMETER PREDICTIONS
# =========================================================

def build_best_predictions(
    matches,
    overall_k,
    venue_k,
    venue_weight,
):
    df = matches.copy()

    home_overall_attack = shrink(
        df[
            "home_overall_attack_raw"
        ],
        df[
            "home_games"
        ],
        overall_k,
    )

    home_overall_defense = shrink(
        df[
            "home_overall_defense_raw"
        ],
        df[
            "home_games"
        ],
        overall_k,
    )

    away_overall_attack = shrink(
        df[
            "away_overall_attack_raw"
        ],
        df[
            "away_games"
        ],
        overall_k,
    )

    away_overall_defense = shrink(
        df[
            "away_overall_defense_raw"
        ],
        df[
            "away_games"
        ],
        overall_k,
    )

    home_venue_attack = shrink(
        df[
            "home_venue_attack_raw"
        ],
        df[
            "home_venue_games"
        ],
        venue_k,
    )

    home_venue_defense = shrink(
        df[
            "home_venue_defense_raw"
        ],
        df[
            "home_venue_games"
        ],
        venue_k,
    )

    away_venue_attack = shrink(
        df[
            "away_venue_attack_raw"
        ],
        df[
            "away_venue_games"
        ],
        venue_k,
    )

    away_venue_defense = shrink(
        df[
            "away_venue_defense_raw"
        ],
        df[
            "away_venue_games"
        ],
        venue_k,
    )

    # League transition adjustment.
    home_overall_attack = transition_adjust(
        home_overall_attack,
        df[
            "home_league_changed"
        ],
    )

    home_overall_defense = transition_adjust(
        home_overall_defense,
        df[
            "home_league_changed"
        ],
    )

    away_overall_attack = transition_adjust(
        away_overall_attack,
        df[
            "away_league_changed"
        ],
    )

    away_overall_defense = transition_adjust(
        away_overall_defense,
        df[
            "away_league_changed"
        ],
    )

    home_venue_attack = transition_adjust(
        home_venue_attack,
        df[
            "home_league_changed"
        ],
    )

    home_venue_defense = transition_adjust(
        home_venue_defense,
        df[
            "home_league_changed"
        ],
    )

    away_venue_attack = transition_adjust(
        away_venue_attack,
        df[
            "away_league_changed"
        ],
    )

    away_venue_defense = transition_adjust(
        away_venue_defense,
        df[
            "away_league_changed"
        ],
    )

    overall_weight = (
        1.0 - venue_weight
    )

    df[
        "home_attack_v1"
    ] = (
        overall_weight
        * home_overall_attack
        +
        venue_weight
        * home_venue_attack
    )

    df[
        "home_defense_v1"
    ] = (
        overall_weight
        * home_overall_defense
        +
        venue_weight
        * home_venue_defense
    )

    df[
        "away_attack_v1"
    ] = (
        overall_weight
        * away_overall_attack
        +
        venue_weight
        * away_venue_attack
    )

    df[
        "away_defense_v1"
    ] = (
        overall_weight
        * away_overall_defense
        +
        venue_weight
        * away_venue_defense
    )

    df[
        "home_lambda_v1"
    ] = (
        df[
            "league_avg_home_goals"
        ]
        *
        df[
            "home_attack_v1"
        ]
        *
        df[
            "away_defense_v1"
        ]
    ).clip(
        lower=0.15,
        upper=4.50,
    )

    df[
        "away_lambda_v1"
    ] = (
        df[
            "league_avg_away_goals"
        ]
        *
        df[
            "away_attack_v1"
        ]
        *
        df[
            "home_defense_v1"
        ]
    ).clip(
        lower=0.15,
        upper=4.50,
    )

    usable = (
        (
            df[
                "home_games"
            ]
            >= MIN_PRIOR_GAMES
        )
        &
        (
            df[
                "away_games"
            ]
            >= MIN_PRIOR_GAMES
        )
        &
        df[
            "home_lambda_v1"
        ].notna()
        &
        df[
            "away_lambda_v1"
        ].notna()
    )

    df = df.loc[
        usable
    ].copy()

    probs = calculate_1x2_probs(
        df[
            "home_lambda_v1"
        ].to_numpy(),
        df[
            "away_lambda_v1"
        ].to_numpy(),
    )

    df[
        "p_home_v1"
    ] = probs[:, 0]

    df[
        "p_draw_v1"
    ] = probs[:, 1]

    df[
        "p_away_v1"
    ] = probs[:, 2]

    return df


# =========================================================
# MAIN
# =========================================================

def main():

    start_time = time.time()

    print()
    print("==============================")
    print("TUNING POISSON V1")
    print("==============================")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{INPUT_FILE}"
        )

    team_games = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "date",
        ],
    )

    team_games[
        "season"
    ] = (
        team_games[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    team_games = (
        team_games
        .sort_values(
            [
                "date",
                "match_id",
                "is_home",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Team-game rows loaded: "
        f"{len(team_games):,}"
    )

    print()
    print(
        "Building league baselines..."
    )

    baselines = build_league_baselines(
        team_games
    )

    team_games = team_games.merge(
        baselines,
        on=[
            "league_code",
            "date",
        ],
        how="left",
        validate="many_to_one",
    )

    total_combinations = (
        len(
            RECENCY_VALUES
        )
        *
        len(
            VENUE_WEIGHTS
        )
        *
        len(
            OVERALL_SHRINK_VALUES
        )
        *
        len(
            VENUE_SHRINK_VALUES
        )
    )

    print(
        f"Parameter combinations: "
        f"{total_combinations}"
    )

    print()
    print(
        "Tuning set: "
        "2021/22–2022/23"
    )

    print(
        "Locked test set: "
        "2023/24–2025/26"
    )

    print()

    results = []

    combination_number = 0

    # =====================================================
    # GRID SEARCH
    # =====================================================

    for decay in RECENCY_VALUES:

        print()
        print(
            f"Building features for "
            f"recency={decay:.2f}..."
        )

        recency_data = (
            build_recency_features(
                team_games,
                decay,
            )
        )

        match_table = build_match_table(
            recency_data
        )

        for venue_weight in VENUE_WEIGHTS:

            for overall_k in (
                OVERALL_SHRINK_VALUES
            ):

                for venue_k in (
                    VENUE_SHRINK_VALUES
                ):

                    combination_number += 1

                    metrics = evaluate_params(
                        match_table,
                        overall_k,
                        venue_k,
                        venue_weight,
                        TUNING_SEASONS,
                    )

                    results.append({
                        "recency":
                            decay,

                        "venue_weight":
                            venue_weight,

                        "overall_weight":
                            (
                                1.0
                                -
                                venue_weight
                            ),

                        "overall_shrink_k":
                            overall_k,

                        "venue_shrink_k":
                            venue_k,

                        "tuning_games":
                            metrics[
                                "games"
                            ],

                        "tuning_log_loss":
                            metrics[
                                "log_loss"
                            ],

                        "tuning_brier":
                            metrics[
                                "brier"
                            ],

                        "tuning_accuracy":
                            metrics[
                                "accuracy"
                            ],
                    })

                    if (
                        combination_number
                        % 20
                        == 0
                    ):
                        print(
                            f"Tested "
                            f"{combination_number}"
                            f"/"
                            f"{total_combinations}"
                        )

    # =====================================================
    # RANK RESULTS
    # =====================================================

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
            len(
                results_df
            )
        )
        + 1
    )

    results_df.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    print()
    print("==============================")
    print("TOP 15 PARAMETER SETS")
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
                "recency",
                "overall_weight",
                "venue_weight",
                "overall_shrink_k",
                "venue_shrink_k",
                "tuning_games",
                "tuning_log_loss",
                "tuning_brier",
                "tuning_accuracy",
            ]
        ]
        .round(5)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # WINNING PARAMS
    # =====================================================

    best = results_df.iloc[
        0
    ]

    best_recency = float(
        best[
            "recency"
        ]
    )

    best_venue_weight = float(
        best[
            "venue_weight"
        ]
    )

    best_overall_k = float(
        best[
            "overall_shrink_k"
        ]
    )

    best_venue_k = float(
        best[
            "venue_shrink_k"
        ]
    )

    print()
    print("==============================")
    print("WINNING V1 PARAMETERS")
    print("==============================")

    print(
        f"Recency:          "
        f"{best_recency:.2f}"
    )

    print(
        f"Overall weight:   "
        f"{1-best_venue_weight:.2f}"
    )

    print(
        f"Venue weight:     "
        f"{best_venue_weight:.2f}"
    )

    print(
        f"Overall shrink K: "
        f"{best_overall_k:.1f}"
    )

    print(
        f"Venue shrink K:   "
        f"{best_venue_k:.1f}"
    )

    print(
        f"Tuning Log Loss:  "
        f"{best['tuning_log_loss']:.4f}"
    )

    # =====================================================
    # REBUILD USING WINNING RECENCY
    # =====================================================

    print()
    print(
        "Rebuilding winning model..."
    )

    winning_recency_data = (
        build_recency_features(
            team_games,
            best_recency,
        )
    )

    winning_matches = (
        build_match_table(
            winning_recency_data
        )
    )

    # =====================================================
    # LOCKED TEST
    # =====================================================

    test_metrics = evaluate_params(
        winning_matches,
        best_overall_k,
        best_venue_k,
        best_venue_weight,
        LOCKED_TEST_SEASONS,
    )

    print()
    print("==============================")
    print("LOCKED TEST RESULTS")
    print("2023/24–2025/26")
    print("==============================")

    print(
        f"Games:     "
        f"{test_metrics['games']:,}"
    )

    print(
        f"Accuracy:  "
        f"{test_metrics['accuracy']:.2%}"
    )

    print(
        f"Log Loss:  "
        f"{test_metrics['log_loss']:.4f}"
    )

    print(
        f"Brier:     "
        f"{test_metrics['brier']:.4f}"
    )

    # =====================================================
    # BY LEAGUE ON LOCKED TEST
    # =====================================================

    print()
    print("==============================")
    print("LOCKED TEST — BY LEAGUE")
    print("==============================")

    league_results = []

    for league in sorted(
        winning_matches[
            "league"
        ].unique()
    ):

        league_df = winning_matches[
            winning_matches[
                "league"
            ]
            == league
        ]

        metrics = evaluate_params(
            league_df,
            best_overall_k,
            best_venue_k,
            best_venue_weight,
            LOCKED_TEST_SEASONS,
        )

        league_results.append({
            "league":
                league,

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

    league_table = pd.DataFrame(
        league_results
    )

    league_table[
        "accuracy"
    ] *= 100.0

    print(
        league_table
        .round(4)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # SAVE BEST PREDICTIONS
    # =====================================================

    best_predictions = (
        build_best_predictions(
            winning_matches,
            best_overall_k,
            best_venue_k,
            best_venue_weight,
        )
    )

    best_predictions[
        "v1_recency"
    ] = best_recency

    best_predictions[
        "v1_venue_weight"
    ] = best_venue_weight

    best_predictions[
        "v1_overall_shrink_k"
    ] = best_overall_k

    best_predictions[
        "v1_venue_shrink_k"
    ] = best_venue_k

    best_predictions.to_csv(
        OUTPUT_BEST,
        index=False,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("==============================")
    print("TUNING COMPLETE")
    print("==============================")

    print(
        f"Runtime: "
        f"{elapsed/60:.2f} minutes"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "2023/24–2025/26 was NOT "
        "used to choose parameters ✅"
    )

    print(
        "Parameter selection used only "
        "2021/22–2022/23 ✅"
    )

    print()
    print(
        f"Tuning results saved:"
        f"\n{OUTPUT_RESULTS}"
    )

    print()
    print(
        f"Best V1 predictions saved:"
        f"\n{OUTPUT_BEST}"
    )


if __name__ == "__main__":
    main()