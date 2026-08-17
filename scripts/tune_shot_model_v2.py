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
    / "team_game_stats.csv"
)

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "shot_model_v2_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "shot_model_v2_predictions.csv"
)


# =========================================================
# SPLITS
# =========================================================

TUNING_SEASONS = {
    "2122",
    "2223",
}

VALIDATION_SEASONS = {
    "2324",
    "2425",
}

FINAL_SEASON = {
    "2526",
}


# =========================================================
# FROZEN V1 SETTINGS
# =========================================================

RECENCY = 0.95

OVERALL_WEIGHT = 0.75
VENUE_WEIGHT = 0.25

OVERALL_SHRINK_K = 20.0
VENUE_SHRINK_K = 4.0

LEAGUE_CHANGE_CONFIDENCE = 0.70

MIN_PRIOR_GAMES = 5

MAX_GOALS = 10

EPS = 1e-12


# =========================================================
# WEIGHT GRID
#
# Each tuple:
# (goal weight, shot weight, SOT weight)
# =========================================================

SIGNAL_WEIGHTS = [
    (1.00, 0.00, 0.00),

    (0.80, 0.10, 0.10),
    (0.70, 0.10, 0.20),
    (0.70, 0.15, 0.15),

    (0.60, 0.10, 0.30),
    (0.60, 0.15, 0.25),
    (0.60, 0.20, 0.20),

    (0.50, 0.15, 0.35),
    (0.50, 0.20, 0.30),
    (0.50, 0.25, 0.25),

    (0.40, 0.20, 0.40),
    (0.40, 0.25, 0.35),
    (0.40, 0.30, 0.30),

    (0.30, 0.25, 0.45),
    (0.30, 0.30, 0.40),
]


# =========================================================
# HISTORY COLUMNS
# =========================================================

ROLLING_COLUMNS = [
    "goals_for",
    "goals_against",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
]


# =========================================================
# EXPONENTIALLY WEIGHTED HISTORY
# =========================================================

def weighted_prior_average(
    values,
    decay=RECENCY,
):

    arr = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy()

    results = np.full(
        len(arr),
        np.nan,
        dtype=float,
    )

    numerator = 0.0
    denominator = 0.0

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

    # We use team-game rows directly.
    # Same-date leakage is prevented by aggregating by date.

    daily = (
        team_games
        .groupby(
            [
                "league_code",
                "league",
                "date",
                "venue",
            ],
            as_index=False,
        )
        .agg(
            games=(
                "match_id",
                "count",
            ),

            goals=(
                "goals_for",
                "sum",
            ),

            shots=(
                "shots_for",
                "sum",
            ),

            sot=(
                "shots_on_target_for",
                "sum",
            ),
        )
    )

    home = (
        daily[
            daily["venue"] == "HOME"
        ]
        .copy()
    )

    away = (
        daily[
            daily["venue"] == "AWAY"
        ]
        .copy()
    )

    home = home.rename(
        columns={
            "games":
                "home_games",

            "goals":
                "home_goals",

            "shots":
                "home_shots",

            "sot":
                "home_sot",
        }
    )

    away = away.rename(
        columns={
            "games":
                "away_games",

            "goals":
                "away_goals",

            "shots":
                "away_shots",

            "sot":
                "away_sot",
        }
    )

    daily_match = home[
        [
            "league_code",
            "league",
            "date",
            "home_games",
            "home_goals",
            "home_shots",
            "home_sot",
        ]
    ].merge(
        away[
            [
                "league_code",
                "date",
                "away_games",
                "away_goals",
                "away_shots",
                "away_sot",
            ]
        ],
        on=[
            "league_code",
            "date",
        ],
        how="inner",
        validate="one_to_one",
    )

    daily_match = (
        daily_match
        .sort_values(
            [
                "league_code",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    grouped = (
        daily_match
        .groupby(
            "league_code"
        )
    )

    # -----------------------------------------------------
    # PRIOR COUNTS
    # -----------------------------------------------------

    daily_match[
        "prior_matches"
    ] = (
        grouped[
            "home_games"
        ]
        .cumsum()
        -
        daily_match[
            "home_games"
        ]
    )

    # -----------------------------------------------------
    # PRIOR HOME TOTALS
    # -----------------------------------------------------

    for stat in [
        "home_goals",
        "home_shots",
        "home_sot",
    ]:

        daily_match[
            f"prior_{stat}"
        ] = (
            grouped[
                stat
            ]
            .cumsum()
            -
            daily_match[
                stat
            ]
        )

    # -----------------------------------------------------
    # PRIOR AWAY TOTALS
    # -----------------------------------------------------

    for stat in [
        "away_goals",
        "away_shots",
        "away_sot",
    ]:

        daily_match[
            f"prior_{stat}"
        ] = (
            grouped[
                stat
            ]
            .cumsum()
            -
            daily_match[
                stat
            ]
        )

    # -----------------------------------------------------
    # BASELINES
    # -----------------------------------------------------

    daily_match[
        "lg_home_goals"
    ] = (
        daily_match[
            "prior_home_goals"
        ]
        /
        daily_match[
            "prior_matches"
        ]
    )

    daily_match[
        "lg_away_goals"
    ] = (
        daily_match[
            "prior_away_goals"
        ]
        /
        daily_match[
            "prior_matches"
        ]
    )

    daily_match[
        "lg_home_shots"
    ] = (
        daily_match[
            "prior_home_shots"
        ]
        /
        daily_match[
            "prior_matches"
        ]
    )

    daily_match[
        "lg_away_shots"
    ] = (
        daily_match[
            "prior_away_shots"
        ]
        /
        daily_match[
            "prior_matches"
        ]
    )

    daily_match[
        "lg_home_sot"
    ] = (
        daily_match[
            "prior_home_sot"
        ]
        /
        daily_match[
            "prior_matches"
        ]
    )

    daily_match[
        "lg_away_sot"
    ] = (
        daily_match[
            "prior_away_sot"
        ]
        /
        daily_match[
            "prior_matches"
        ]
    )

    daily_match[
        "lg_team_goals"
    ] = (
        daily_match[
            "lg_home_goals"
        ]
        +
        daily_match[
            "lg_away_goals"
        ]
    ) / 2.0

    daily_match[
        "lg_team_shots"
    ] = (
        daily_match[
            "lg_home_shots"
        ]
        +
        daily_match[
            "lg_away_shots"
        ]
    ) / 2.0

    daily_match[
        "lg_team_sot"
    ] = (
        daily_match[
            "lg_home_sot"
        ]
        +
        daily_match[
            "lg_away_sot"
        ]
    ) / 2.0

    return daily_match[
        [
            "league_code",
            "date",

            "lg_home_goals",
            "lg_away_goals",

            "lg_home_shots",
            "lg_away_shots",

            "lg_home_sot",
            "lg_away_sot",

            "lg_team_goals",
            "lg_team_shots",
            "lg_team_sot",
        ]
    ]


# =========================================================
# TEAM HISTORY
# =========================================================

def build_history(
    df,
):

    df = df.copy()

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
                weighted_prior_average
            )
        )

    # -----------------------------------------------------
    # VENUE HISTORY
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
                weighted_prior_average
            )
        )

    # -----------------------------------------------------
    # LEAGUE TRANSITION
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

    return df


# =========================================================
# SHRINKAGE
# =========================================================

def shrink(
    rating,
    games,
    k,
):

    games = pd.to_numeric(
        games,
        errors="coerce",
    ).fillna(0.0)

    weight = (
        games
        /
        (
            games + k
        )
    )

    return (
        weight
        * rating
        +
        (
            1.0
            - weight
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
# RAW STRENGTHS
# =========================================================

def add_strengths(
    df,
):

    df = df.copy()

    home_mask = (
        df["venue"] == "HOME"
    )

    away_mask = (
        df["venue"] == "AWAY"
    )

    # -----------------------------------------------------
    # OVERALL GOALS
    # -----------------------------------------------------

    df[
        "overall_goal_attack"
    ] = (
        df[
            "overall_goals_for"
        ]
        /
        df[
            "lg_team_goals"
        ]
    )

    df[
        "overall_goal_defense"
    ] = (
        df[
            "overall_goals_against"
        ]
        /
        df[
            "lg_team_goals"
        ]
    )

    # -----------------------------------------------------
    # OVERALL SHOTS
    # -----------------------------------------------------

    df[
        "overall_shot_attack"
    ] = (
        df[
            "overall_shots_for"
        ]
        /
        df[
            "lg_team_shots"
        ]
    )

    df[
        "overall_shot_defense"
    ] = (
        df[
            "overall_shots_against"
        ]
        /
        df[
            "lg_team_shots"
        ]
    )

    # -----------------------------------------------------
    # OVERALL SOT
    # -----------------------------------------------------

    df[
        "overall_sot_attack"
    ] = (
        df[
            "overall_shots_on_target_for"
        ]
        /
        df[
            "lg_team_sot"
        ]
    )

    df[
        "overall_sot_defense"
    ] = (
        df[
            "overall_shots_on_target_against"
        ]
        /
        df[
            "lg_team_sot"
        ]
    )

    # -----------------------------------------------------
    # VENUE GOAL RATINGS
    # -----------------------------------------------------

    df[
        "venue_goal_attack"
    ] = np.nan

    df[
        "venue_goal_defense"
    ] = np.nan

    df.loc[
        home_mask,
        "venue_goal_attack",
    ] = (
        df.loc[
            home_mask,
            "venue_goals_for",
        ]
        /
        df.loc[
            home_mask,
            "lg_home_goals",
        ]
    )

    df.loc[
        home_mask,
        "venue_goal_defense",
    ] = (
        df.loc[
            home_mask,
            "venue_goals_against",
        ]
        /
        df.loc[
            home_mask,
            "lg_away_goals",
        ]
    )

    df.loc[
        away_mask,
        "venue_goal_attack",
    ] = (
        df.loc[
            away_mask,
            "venue_goals_for",
        ]
        /
        df.loc[
            away_mask,
            "lg_away_goals",
        ]
    )

    df.loc[
        away_mask,
        "venue_goal_defense",
    ] = (
        df.loc[
            away_mask,
            "venue_goals_against",
        ]
        /
        df.loc[
            away_mask,
            "lg_home_goals",
        ]
    )

    # -----------------------------------------------------
    # VENUE SHOT RATINGS
    # -----------------------------------------------------

    df[
        "venue_shot_attack"
    ] = np.nan

    df[
        "venue_shot_defense"
    ] = np.nan

    df.loc[
        home_mask,
        "venue_shot_attack",
    ] = (
        df.loc[
            home_mask,
            "venue_shots_for",
        ]
        /
        df.loc[
            home_mask,
            "lg_home_shots",
        ]
    )

    df.loc[
        home_mask,
        "venue_shot_defense",
    ] = (
        df.loc[
            home_mask,
            "venue_shots_against",
        ]
        /
        df.loc[
            home_mask,
            "lg_away_shots",
        ]
    )

    df.loc[
        away_mask,
        "venue_shot_attack",
    ] = (
        df.loc[
            away_mask,
            "venue_shots_for",
        ]
        /
        df.loc[
            away_mask,
            "lg_away_shots",
        ]
    )

    df.loc[
        away_mask,
        "venue_shot_defense",
    ] = (
        df.loc[
            away_mask,
            "venue_shots_against",
        ]
        /
        df.loc[
            away_mask,
            "lg_home_shots",
        ]
    )

    # -----------------------------------------------------
    # VENUE SOT RATINGS
    # -----------------------------------------------------

    df[
        "venue_sot_attack"
    ] = np.nan

    df[
        "venue_sot_defense"
    ] = np.nan

    df.loc[
        home_mask,
        "venue_sot_attack",
    ] = (
        df.loc[
            home_mask,
            "venue_shots_on_target_for",
        ]
        /
        df.loc[
            home_mask,
            "lg_home_sot",
        ]
    )

    df.loc[
        home_mask,
        "venue_sot_defense",
    ] = (
        df.loc[
            home_mask,
            "venue_shots_on_target_against",
        ]
        /
        df.loc[
            home_mask,
            "lg_away_sot",
        ]
    )

    df.loc[
        away_mask,
        "venue_sot_attack",
    ] = (
        df.loc[
            away_mask,
            "venue_shots_on_target_for",
        ]
        /
        df.loc[
            away_mask,
            "lg_away_sot",
        ]
    )

    df.loc[
        away_mask,
        "venue_sot_defense",
    ] = (
        df.loc[
            away_mask,
            "venue_shots_on_target_against",
        ]
        /
        df.loc[
            away_mask,
            "lg_home_sot",
        ]
    )

    return df


# =========================================================
# MATCH TABLE
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

    strength_columns = [
        "overall_goal_attack",
        "overall_goal_defense",
        "overall_shot_attack",
        "overall_shot_defense",
        "overall_sot_attack",
        "overall_sot_defense",

        "venue_goal_attack",
        "venue_goal_defense",
        "venue_shot_attack",
        "venue_shot_defense",
        "venue_sot_attack",
        "venue_sot_defense",
    ]

    home_keep = [
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

        "lg_home_goals",
        "lg_away_goals",

    ] + strength_columns

    home = home[
        home_keep
    ].copy()

    home_rename = {
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
    }

    for col in strength_columns:
        home_rename[
            col
        ] = (
            "home_"
            + col
        )

    home = home.rename(
        columns=home_rename
    )

    away_keep = [
        "match_id",

        "team",
        "opponent",

        "pregame_games",
        "pregame_venue_games",

        "league_changed",

    ] + strength_columns

    away = away[
        away_keep
    ].copy()

    away_rename = {
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
    }

    for col in strength_columns:
        away_rename[
            col
        ] = (
            "away_"
            + col
        )

    away = away.rename(
        columns=away_rename
    )

    return home.merge(
        away,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )


# =========================================================
# PREPARE SHRUNK SIGNALS
# =========================================================

def prepare_signal(
    df,
    side,
    signal,
    attack_or_defense,
):

    overall_col = (
        f"{side}_overall_"
        f"{signal}_"
        f"{attack_or_defense}"
    )

    venue_col = (
        f"{side}_venue_"
        f"{signal}_"
        f"{attack_or_defense}"
    )

    games_col = (
        f"{side}_games"
    )

    venue_games_col = (
        f"{side}_venue_games"
    )

    changed_col = (
        f"{side}_league_changed"
    )

    overall = shrink(
        df[
            overall_col
        ],
        df[
            games_col
        ],
        OVERALL_SHRINK_K,
    )

    venue = shrink(
        df[
            venue_col
        ],
        df[
            venue_games_col
        ],
        VENUE_SHRINK_K,
    )

    overall = transition_adjust(
        overall,
        df[
            changed_col
        ],
    )

    venue = transition_adjust(
        venue,
        df[
            changed_col
        ],
    )

    return (
        OVERALL_WEIGHT
        * overall
        +
        VENUE_WEIGHT
        * venue
    )


# =========================================================
# POISSON
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
        MAX_GOALS + 1
    ):
        home_win += (
            hp[:, h]
            *
            away_cdf[:, h - 1]
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
        MAX_GOALS + 1
    ):
        away_win += (
            ap[:, a]
            *
            home_cdf[:, a - 1]
        )

    draw = (
        hp * ap
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


# =========================================================
# METRICS
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


# =========================================================
# BUILD MODEL FOR ONE SIGNAL WEIGHT SET
# =========================================================

def evaluate_weights(
    df,
    goal_weight,
    shot_weight,
    sot_weight,
    seasons,
):

    # -----------------------------------------------------
    # BASE SIGNALS
    # -----------------------------------------------------

    home_goal_attack = prepare_signal(
        df,
        "home",
        "goal",
        "attack",
    )

    home_goal_defense = prepare_signal(
        df,
        "home",
        "goal",
        "defense",
    )

    away_goal_attack = prepare_signal(
        df,
        "away",
        "goal",
        "attack",
    )

    away_goal_defense = prepare_signal(
        df,
        "away",
        "goal",
        "defense",
    )

    home_shot_attack = prepare_signal(
        df,
        "home",
        "shot",
        "attack",
    )

    home_shot_defense = prepare_signal(
        df,
        "home",
        "shot",
        "defense",
    )

    away_shot_attack = prepare_signal(
        df,
        "away",
        "shot",
        "attack",
    )

    away_shot_defense = prepare_signal(
        df,
        "away",
        "shot",
        "defense",
    )

    home_sot_attack = prepare_signal(
        df,
        "home",
        "sot",
        "attack",
    )

    home_sot_defense = prepare_signal(
        df,
        "home",
        "sot",
        "defense",
    )

    away_sot_attack = prepare_signal(
        df,
        "away",
        "sot",
        "attack",
    )

    away_sot_defense = prepare_signal(
        df,
        "away",
        "sot",
        "defense",
    )

    # -----------------------------------------------------
    # COMBINE SIGNALS
    # -----------------------------------------------------

    home_attack = (
        goal_weight
        * home_goal_attack

        +
        shot_weight
        * home_shot_attack

        +
        sot_weight
        * home_sot_attack
    )

    home_defense = (
        goal_weight
        * home_goal_defense

        +
        shot_weight
        * home_shot_defense

        +
        sot_weight
        * home_sot_defense
    )

    away_attack = (
        goal_weight
        * away_goal_attack

        +
        shot_weight
        * away_shot_attack

        +
        sot_weight
        * away_sot_attack
    )

    away_defense = (
        goal_weight
        * away_goal_defense

        +
        shot_weight
        * away_shot_defense

        +
        sot_weight
        * away_sot_defense
    )

    # -----------------------------------------------------
    # EXPECTED GOALS
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # SELECT SPLIT
    # -----------------------------------------------------

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

    sub = df.loc[
        mask
    ]

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


# =========================================================
# BUILD FINAL PREDICTION FILE
# =========================================================

def build_predictions(
    df,
    goal_weight,
    shot_weight,
    sot_weight,
):

    out = df.copy()

    # -----------------------------------------------------
    # SIGNALS
    # -----------------------------------------------------

    signals = {}

    for side in [
        "home",
        "away",
    ]:

        for signal in [
            "goal",
            "shot",
            "sot",
        ]:

            for kind in [
                "attack",
                "defense",
            ]:

                key = (
                    f"{side}_"
                    f"{signal}_"
                    f"{kind}"
                )

                signals[
                    key
                ] = prepare_signal(
                    out,
                    side,
                    signal,
                    kind,
                )

    # -----------------------------------------------------
    # BLENDED STRENGTHS
    # -----------------------------------------------------

    out[
        "home_attack_v2"
    ] = (
        goal_weight
        * signals[
            "home_goal_attack"
        ]
        +
        shot_weight
        * signals[
            "home_shot_attack"
        ]
        +
        sot_weight
        * signals[
            "home_sot_attack"
        ]
    )

    out[
        "home_defense_v2"
    ] = (
        goal_weight
        * signals[
            "home_goal_defense"
        ]
        +
        shot_weight
        * signals[
            "home_shot_defense"
        ]
        +
        sot_weight
        * signals[
            "home_sot_defense"
        ]
    )

    out[
        "away_attack_v2"
    ] = (
        goal_weight
        * signals[
            "away_goal_attack"
        ]
        +
        shot_weight
        * signals[
            "away_shot_attack"
        ]
        +
        sot_weight
        * signals[
            "away_sot_attack"
        ]
    )

    out[
        "away_defense_v2"
    ] = (
        goal_weight
        * signals[
            "away_goal_defense"
        ]
        +
        shot_weight
        * signals[
            "away_shot_defense"
        ]
        +
        sot_weight
        * signals[
            "away_sot_defense"
        ]
    )

    out[
        "home_lambda_v2"
    ] = (
        out[
            "lg_home_goals"
        ]
        *
        out[
            "home_attack_v2"
        ]
        *
        out[
            "away_defense_v2"
        ]
    ).clip(
        lower=0.15,
        upper=4.50,
    )

    out[
        "away_lambda_v2"
    ] = (
        out[
            "lg_away_goals"
        ]
        *
        out[
            "away_attack_v2"
        ]
        *
        out[
            "home_defense_v2"
        ]
    ).clip(
        lower=0.15,
        upper=4.50,
    )

    usable = (
        out[
            "home_lambda_v2"
        ].notna()
        &
        out[
            "away_lambda_v2"
        ].notna()
        &
        (
            out[
                "home_games"
            ]
            >= MIN_PRIOR_GAMES
        )
        &
        (
            out[
                "away_games"
            ]
            >= MIN_PRIOR_GAMES
        )
    )

    out = out.loc[
        usable
    ].copy()

    probs = calculate_1x2_probs(
        out[
            "home_lambda_v2"
        ].to_numpy(),
        out[
            "away_lambda_v2"
        ].to_numpy(),
    )

    out[
        "p_home_v2"
    ] = probs[:, 0]

    out[
        "p_draw_v2"
    ] = probs[:, 1]

    out[
        "p_away_v2"
    ] = probs[:, 2]

    out[
        "goal_weight_v2"
    ] = goal_weight

    out[
        "shot_weight_v2"
    ] = shot_weight

    out[
        "sot_weight_v2"
    ] = sot_weight

    return out


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("==============================")
    print("TUNING SHOT MODEL V2")
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

    df = (
        df
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
        f"Team rows loaded: "
        f"{len(df):,}"
    )

    print()
    print(
        "Frozen V1 settings:"
    )

    print(
        f"Recency:          "
        f"{RECENCY:.2f}"
    )

    print(
        f"Overall weight:   "
        f"{OVERALL_WEIGHT:.2f}"
    )

    print(
        f"Venue weight:     "
        f"{VENUE_WEIGHT:.2f}"
    )

    print(
        f"Overall shrink K: "
        f"{OVERALL_SHRINK_K:.0f}"
    )

    print(
        f"Venue shrink K:   "
        f"{VENUE_SHRINK_K:.0f}"
    )

    # -----------------------------------------------------
    # BASELINES
    # -----------------------------------------------------

    print()
    print(
        "Building leakage-safe "
        "league baselines..."
    )

    baselines = build_league_baselines(
        df
    )

    df = df.merge(
        baselines,
        on=[
            "league_code",
            "date",
        ],
        how="left",
        validate="many_to_one",
    )

    # -----------------------------------------------------
    # HISTORY
    # -----------------------------------------------------

    print(
        "Building weighted team history..."
    )

    df = build_history(
        df
    )

    print(
        "Building goal / shot / SOT strengths..."
    )

    df = add_strengths(
        df
    )

    print(
        "Combining home and away rows..."
    )

    matches = build_match_table(
        df
    )

    # -----------------------------------------------------
    # GRID SEARCH
    # -----------------------------------------------------

    print()
    print(
        f"Testing "
        f"{len(SIGNAL_WEIGHTS)} "
        f"signal combinations..."
    )

    results = []

    for (
        goal_weight,
        shot_weight,
        sot_weight,
    ) in SIGNAL_WEIGHTS:

        metrics = evaluate_weights(
            matches,
            goal_weight,
            shot_weight,
            sot_weight,
            TUNING_SEASONS,
        )

        results.append({
            "goal_weight":
                goal_weight,

            "shot_weight":
                shot_weight,

            "sot_weight":
                sot_weight,

            "tuning_games":
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
    # DISPLAY
    # -----------------------------------------------------

    print()
    print("==============================")
    print("SIGNAL WEIGHT RESULTS")
    print("==============================")

    display = results_df.copy()

    display[
        "tuning_accuracy"
    ] *= 100.0

    print(
        display[
            [
                "rank",
                "goal_weight",
                "shot_weight",
                "sot_weight",
                "tuning_games",
                "tuning_log_loss",
                "tuning_brier",
                "tuning_accuracy",
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

    goal_weight = float(
        best[
            "goal_weight"
        ]
    )

    shot_weight = float(
        best[
            "shot_weight"
        ]
    )

    sot_weight = float(
        best[
            "sot_weight"
        ]
    )

    print()
    print("==============================")
    print("WINNING V2 SIGNAL WEIGHTS")
    print("==============================")

    print(
        f"Goals:           "
        f"{goal_weight:.2f}"
    )

    print(
        f"Shots:           "
        f"{shot_weight:.2f}"
    )

    print(
        f"Shots on target: "
        f"{sot_weight:.2f}"
    )

    print(
        f"Tuning LL:       "
        f"{best['tuning_log_loss']:.5f}"
    )

    # -----------------------------------------------------
    # VALIDATION 2023/24–2024/25
    # -----------------------------------------------------

    validation = evaluate_weights(
        matches,
        goal_weight,
        shot_weight,
        sot_weight,
        VALIDATION_SEASONS,
    )

    # Goal-only comparator.
    v1_validation = evaluate_weights(
        matches,
        1.0,
        0.0,
        0.0,
        VALIDATION_SEASONS,
    )

    print()
    print("==============================")
    print("VALIDATION")
    print("2023/24–2024/25")
    print("==============================")

    print(
        f"{'Metric':<15}"
        f"{'Goal Only':>12}"
        f"{'V2':>12}"
        f"{'Change':>12}"
    )

    print("-" * 51)

    print(
        f"{'Accuracy':<15}"
        f"{v1_validation['accuracy']:>11.2%}"
        f"{validation['accuracy']:>11.2%}"
        f"{validation['accuracy'] - v1_validation['accuracy']:>+11.2%}"
    )

    print(
        f"{'Log Loss':<15}"
        f"{v1_validation['log_loss']:>12.5f}"
        f"{validation['log_loss']:>12.5f}"
        f"{validation['log_loss'] - v1_validation['log_loss']:>+12.5f}"
    )

    print(
        f"{'Brier':<15}"
        f"{v1_validation['brier']:>12.5f}"
        f"{validation['brier']:>12.5f}"
        f"{validation['brier'] - v1_validation['brier']:>+12.5f}"
    )

    # -----------------------------------------------------
    # FINAL 2025/26
    # -----------------------------------------------------

    final_metrics = evaluate_weights(
        matches,
        goal_weight,
        shot_weight,
        sot_weight,
        FINAL_SEASON,
    )

    v1_final = evaluate_weights(
        matches,
        1.0,
        0.0,
        0.0,
        FINAL_SEASON,
    )

    print()
    print("==============================")
    print("FINAL SEASON CHECK")
    print("2025/26")
    print("==============================")

    print(
        f"Games: "
        f"{final_metrics['games']:,}"
    )

    print()
    print(
        f"{'Metric':<15}"
        f"{'Goal Only':>12}"
        f"{'V2':>12}"
        f"{'Change':>12}"
    )

    print("-" * 51)

    print(
        f"{'Accuracy':<15}"
        f"{v1_final['accuracy']:>11.2%}"
        f"{final_metrics['accuracy']:>11.2%}"
        f"{final_metrics['accuracy'] - v1_final['accuracy']:>+11.2%}"
    )

    print(
        f"{'Log Loss':<15}"
        f"{v1_final['log_loss']:>12.5f}"
        f"{final_metrics['log_loss']:>12.5f}"
        f"{final_metrics['log_loss'] - v1_final['log_loss']:>+12.5f}"
    )

    print(
        f"{'Brier':<15}"
        f"{v1_final['brier']:>12.5f}"
        f"{final_metrics['brier']:>12.5f}"
        f"{final_metrics['brier'] - v1_final['brier']:>+12.5f}"
    )

    # -----------------------------------------------------
    # 2025/26 BY LEAGUE
    # -----------------------------------------------------

    print()
    print("==============================")
    print("2025/26 — BY LEAGUE")
    print("==============================")

    rows = []

    for league, group in (
        matches.groupby(
            "league"
        )
    ):

        goal_only = evaluate_weights(
            group,
            1.0,
            0.0,
            0.0,
            FINAL_SEASON,
        )

        v2 = evaluate_weights(
            group,
            goal_weight,
            shot_weight,
            sot_weight,
            FINAL_SEASON,
        )

        rows.append({
            "league":
                league,

            "games":
                v2[
                    "games"
                ],

            "goal_only_ll":
                goal_only[
                    "log_loss"
                ],

            "v2_ll":
                v2[
                    "log_loss"
                ],

            "ll_change":
                (
                    v2[
                        "log_loss"
                    ]
                    -
                    goal_only[
                        "log_loss"
                    ]
                ),

            "goal_only_brier":
                goal_only[
                    "brier"
                ],

            "v2_brier":
                v2[
                    "brier"
                ],

            "goal_only_acc":
                goal_only[
                    "accuracy"
                ],

            "v2_acc":
                v2[
                    "accuracy"
                ],
        })

    league_table = pd.DataFrame(
        rows
    )

    league_table[
        "goal_only_acc"
    ] *= 100.0

    league_table[
        "v2_acc"
    ] *= 100.0

    print(
        league_table
        .round(5)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # SAVE FULL PREDICTIONS
    # -----------------------------------------------------

    predictions = build_predictions(
        matches,
        goal_weight,
        shot_weight,
        sot_weight,
    )

    predictions.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print()
    print("==============================")
    print("V2 COMPLETE")
    print("==============================")

    print(
        "V1 structural parameters "
        "remained frozen ✅"
    )

    print(
        "Signal weights selected only "
        "on 2021/22–2022/23 ✅"
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