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
    / "opponent_adjusted_v3_features.csv"
)

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "opponent_adjustment_v3_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "opponent_adjustment_v3_predictions.csv"
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
# FROZEN V2 SETTINGS
# =========================================================

GOAL_WEIGHT = 0.70
SHOT_WEIGHT = 0.15
SOT_WEIGHT = 0.15

OVERALL_WEIGHT = 0.75
VENUE_WEIGHT = 0.25

OVERALL_SHRINK_K = 20.0
VENUE_SHRINK_K = 4.0

LEAGUE_CHANGE_CONFIDENCE = 0.70

MIN_PRIOR_GAMES = 5

MAX_GOALS = 10
EPS = 1e-12


# =========================================================
# OPPONENT-ADJUSTMENT BLEND GRID
# =========================================================

ADJUSTMENT_WEIGHTS = np.round(
    np.arange(
        0.0,
        1.01,
        0.10,
    ),
    2,
)


# =========================================================
# EXPONENTIALLY WEIGHTED PRIOR AVERAGE
# =========================================================

RECENCY = 0.95


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
# BUILD VENUE HISTORIES
#
# Existing opponent-adjusted file contains overall raw/V3
# strengths. We rebuild venue-specific raw and adjusted
# histories so V3 keeps the exact 75/25 V2 structure.
# =========================================================

def add_venue_histories(
    df,
):

    df = df.copy()

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
    ] = venue_group.cumcount()

    raw_cols = [
        "goals_for",
        "goals_against",
        "shots_for",
        "shots_against",
        "shots_on_target_for",
        "shots_on_target_against",
    ]

    adjusted_cols = [
        "adj_goals_for",
        "adj_goals_against",
        "adj_shots_for",
        "adj_shots_against",
        "adj_sot_for",
        "adj_sot_against",
    ]

    for col in (
        raw_cols
        + adjusted_cols
    ):

        df[
            f"venue_ew_{col}"
        ] = (
            venue_group[
                col
            ]
            .transform(
                weighted_prior_average
            )
        )

    return df


# =========================================================
# VENUE BASELINES
# =========================================================

def add_venue_strengths(
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
    # We need approximate home/away shot baselines.
    #
    # Existing feature file stores league-average team
    # shots/SOT, but not home/away versions.
    #
    # We calculate same-day-safe versions here.
    # -----------------------------------------------------

    return df


# =========================================================
# BUILD SAFE HOME/AWAY LEAGUE SHOT BASELINES
# =========================================================

def build_venue_league_baselines(
    df,
):

    work = df.copy()

    work[
        "shot_valid"
    ] = (
        work[
            "shots_for"
        ].notna()
    ).astype(int)

    work[
        "sot_valid"
    ] = (
        work[
            "shots_on_target_for"
        ].notna()
    ).astype(int)

    daily = (
        work
        .groupby(
            [
                "league_code",
                "date",
                "venue",
            ],
            as_index=False,
        )
        .agg(
            shot_obs=(
                "shot_valid",
                "sum",
            ),
            sot_obs=(
                "sot_valid",
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
            daily[
                "venue"
            ]
            == "HOME"
        ]
        .copy()
        .rename(
            columns={
                "shot_obs":
                    "home_shot_obs",

                "sot_obs":
                    "home_sot_obs",

                "shots":
                    "home_shots",

                "sot":
                    "home_sot",
            }
        )
    )

    away = (
        daily[
            daily[
                "venue"
            ]
            == "AWAY"
        ]
        .copy()
        .rename(
            columns={
                "shot_obs":
                    "away_shot_obs",

                "sot_obs":
                    "away_sot_obs",

                "shots":
                    "away_shots",

                "sot":
                    "away_sot",
            }
        )
    )

    daily_match = home[
        [
            "league_code",
            "date",
            "home_shot_obs",
            "home_sot_obs",
            "home_shots",
            "home_sot",
        ]
    ].merge(
        away[
            [
                "league_code",
                "date",
                "away_shot_obs",
                "away_sot_obs",
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

    grouped = daily_match.groupby(
        "league_code"
    )

    for col in [
        "home_shot_obs",
        "away_shot_obs",
        "home_sot_obs",
        "away_sot_obs",
        "home_shots",
        "away_shots",
        "home_sot",
        "away_sot",
    ]:

        daily_match[
            f"prior_{col}"
        ] = (
            grouped[
                col
            ].cumsum()
            -
            daily_match[
                col
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
            "prior_home_shot_obs"
        ].replace(
            0,
            np.nan,
        )
    )

    daily_match[
        "lg_away_shots"
    ] = (
        daily_match[
            "prior_away_shots"
        ]
        /
        daily_match[
            "prior_away_shot_obs"
        ].replace(
            0,
            np.nan,
        )
    )

    daily_match[
        "lg_home_sot"
    ] = (
        daily_match[
            "prior_home_sot"
        ]
        /
        daily_match[
            "prior_home_sot_obs"
        ].replace(
            0,
            np.nan,
        )
    )

    daily_match[
        "lg_away_sot"
    ] = (
        daily_match[
            "prior_away_sot"
        ]
        /
        daily_match[
            "prior_away_sot_obs"
        ].replace(
            0,
            np.nan,
        )
    )

    return daily_match[
        [
            "league_code",
            "date",
            "lg_home_shots",
            "lg_away_shots",
            "lg_home_sot",
            "lg_away_sot",
        ]
    ]


# =========================================================
# ADD RAW + ADJUSTED VENUE STRENGTH RATINGS
# =========================================================

def build_venue_strengths(
    df,
):

    df = df.copy()

    home_mask = (
        df[
            "venue"
        ]
        == "HOME"
    )

    away_mask = (
        df[
            "venue"
        ]
        == "AWAY"
    )

    columns = [
        "raw_venue_goal_attack",
        "raw_venue_goal_defense",

        "raw_venue_shot_attack",
        "raw_venue_shot_defense",

        "raw_venue_sot_attack",
        "raw_venue_sot_defense",

        "adj_venue_goal_attack",
        "adj_venue_goal_defense",

        "adj_venue_shot_attack",
        "adj_venue_shot_defense",

        "adj_venue_sot_attack",
        "adj_venue_sot_defense",
    ]

    for col in columns:
        df[col] = np.nan

    # =====================================================
    # HOME
    # =====================================================

    df.loc[
        home_mask,
        "raw_venue_goal_attack",
    ] = (
        df.loc[
            home_mask,
            "venue_ew_goals_for",
        ]
        /
        df.loc[
            home_mask,
            "lg_home_goals",
        ]
    )

    df.loc[
        home_mask,
        "raw_venue_goal_defense",
    ] = (
        df.loc[
            home_mask,
            "venue_ew_goals_against",
        ]
        /
        df.loc[
            home_mask,
            "lg_away_goals",
        ]
    )

    df.loc[
        home_mask,
        "raw_venue_shot_attack",
    ] = (
        df.loc[
            home_mask,
            "venue_ew_shots_for",
        ]
        /
        df.loc[
            home_mask,
            "lg_home_shots",
        ]
    )

    df.loc[
        home_mask,
        "raw_venue_shot_defense",
    ] = (
        df.loc[
            home_mask,
            "venue_ew_shots_against",
        ]
        /
        df.loc[
            home_mask,
            "lg_away_shots",
        ]
    )

    df.loc[
        home_mask,
        "raw_venue_sot_attack",
    ] = (
        df.loc[
            home_mask,
            "venue_ew_shots_on_target_for",
        ]
        /
        df.loc[
            home_mask,
            "lg_home_sot",
        ]
    )

    df.loc[
        home_mask,
        "raw_venue_sot_defense",
    ] = (
        df.loc[
            home_mask,
            "venue_ew_shots_on_target_against",
        ]
        /
        df.loc[
            home_mask,
            "lg_away_sot",
        ]
    )

    # Opponent-adjusted home.
    df.loc[
        home_mask,
        "adj_venue_goal_attack",
    ] = (
        df.loc[
            home_mask,
            "venue_ew_adj_goals_for",
        ]
        /
        df.loc[
            home_mask,
            "lg_home_goals",
        ]
    )

    df.loc[
        home_mask,
        "adj_venue_goal_defense",
    ] = (
        df.loc[
            home_mask,
            "venue_ew_adj_goals_against",
        ]
        /
        df.loc[
            home_mask,
            "lg_away_goals",
        ]
    )

    df.loc[
        home_mask,
        "adj_venue_shot_attack",
    ] = (
        df.loc[
            home_mask,
            "venue_ew_adj_shots_for",
        ]
        /
        df.loc[
            home_mask,
            "lg_home_shots",
        ]
    )

    df.loc[
        home_mask,
        "adj_venue_shot_defense",
    ] = (
        df.loc[
            home_mask,
            "venue_ew_adj_shots_against",
        ]
        /
        df.loc[
            home_mask,
            "lg_away_shots",
        ]
    )

    df.loc[
        home_mask,
        "adj_venue_sot_attack",
    ] = (
        df.loc[
            home_mask,
            "venue_ew_adj_sot_for",
        ]
        /
        df.loc[
            home_mask,
            "lg_home_sot",
        ]
    )

    df.loc[
        home_mask,
        "adj_venue_sot_defense",
    ] = (
        df.loc[
            home_mask,
            "venue_ew_adj_sot_against",
        ]
        /
        df.loc[
            home_mask,
            "lg_away_sot",
        ]
    )

    # =====================================================
    # AWAY
    # =====================================================

    df.loc[
        away_mask,
        "raw_venue_goal_attack",
    ] = (
        df.loc[
            away_mask,
            "venue_ew_goals_for",
        ]
        /
        df.loc[
            away_mask,
            "lg_away_goals",
        ]
    )

    df.loc[
        away_mask,
        "raw_venue_goal_defense",
    ] = (
        df.loc[
            away_mask,
            "venue_ew_goals_against",
        ]
        /
        df.loc[
            away_mask,
            "lg_home_goals",
        ]
    )

    df.loc[
        away_mask,
        "raw_venue_shot_attack",
    ] = (
        df.loc[
            away_mask,
            "venue_ew_shots_for",
        ]
        /
        df.loc[
            away_mask,
            "lg_away_shots",
        ]
    )

    df.loc[
        away_mask,
        "raw_venue_shot_defense",
    ] = (
        df.loc[
            away_mask,
            "venue_ew_shots_against",
        ]
        /
        df.loc[
            away_mask,
            "lg_home_shots",
        ]
    )

    df.loc[
        away_mask,
        "raw_venue_sot_attack",
    ] = (
        df.loc[
            away_mask,
            "venue_ew_shots_on_target_for",
        ]
        /
        df.loc[
            away_mask,
            "lg_away_sot",
        ]
    )

    df.loc[
        away_mask,
        "raw_venue_sot_defense",
    ] = (
        df.loc[
            away_mask,
            "venue_ew_shots_on_target_against",
        ]
        /
        df.loc[
            away_mask,
            "lg_home_sot",
        ]
    )

    # Opponent-adjusted away.
    df.loc[
        away_mask,
        "adj_venue_goal_attack",
    ] = (
        df.loc[
            away_mask,
            "venue_ew_adj_goals_for",
        ]
        /
        df.loc[
            away_mask,
            "lg_away_goals",
        ]
    )

    df.loc[
        away_mask,
        "adj_venue_goal_defense",
    ] = (
        df.loc[
            away_mask,
            "venue_ew_adj_goals_against",
        ]
        /
        df.loc[
            away_mask,
            "lg_home_goals",
        ]
    )

    df.loc[
        away_mask,
        "adj_venue_shot_attack",
    ] = (
        df.loc[
            away_mask,
            "venue_ew_adj_shots_for",
        ]
        /
        df.loc[
            away_mask,
            "lg_away_shots",
        ]
    )

    df.loc[
        away_mask,
        "adj_venue_shot_defense",
    ] = (
        df.loc[
            away_mask,
            "venue_ew_adj_shots_against",
        ]
        /
        df.loc[
            away_mask,
            "lg_home_shots",
        ]
    )

    df.loc[
        away_mask,
        "adj_venue_sot_attack",
    ] = (
        df.loc[
            away_mask,
            "venue_ew_adj_sot_for",
        ]
        /
        df.loc[
            away_mask,
            "lg_away_sot",
        ]
    )

    df.loc[
        away_mask,
        "adj_venue_sot_defense",
    ] = (
        df.loc[
            away_mask,
            "venue_ew_adj_sot_against",
        ]
        /
        df.loc[
            away_mask,
            "lg_home_sot",
        ]
    )

    return df


# =========================================================
# LEAGUE TRANSITION
# =========================================================

def add_league_transition(
    df,
):

    df = df.copy()

    group = df.groupby(
        "team",
        sort=False,
    )

    df[
        "previous_league_code"
    ] = (
        group[
            "league_code"
        ].shift(1)
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
# BLEND RAW VS OPPONENT-ADJUSTED SIGNAL
# =========================================================

def blend_strength(
    raw,
    adjusted,
    adjustment_weight,
):

    return (
        (
            1.0
            - adjustment_weight
        )
        * raw
        +
        adjustment_weight
        * adjusted
    )


# =========================================================
# PREPARE ONE SIGNAL
# =========================================================

def prepare_signal(
    df,
    side,
    signal,
    kind,
    adjustment_weight,
):

    raw_overall_col = (
        f"{side}_raw_"
        f"{signal}_"
        f"{kind}"
    )

    adj_overall_col = (
        f"{side}_adj_"
        f"{signal}_"
        f"{kind}"
    )

    raw_venue_col = (
        f"{side}_raw_venue_"
        f"{signal}_"
        f"{kind}"
    )

    adj_venue_col = (
        f"{side}_adj_venue_"
        f"{signal}_"
        f"{kind}"
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

    raw_overall = shrink(
        df[
            raw_overall_col
        ],
        df[
            games_col
        ],
        OVERALL_SHRINK_K,
    )

    adj_overall = shrink(
        df[
            adj_overall_col
        ],
        df[
            games_col
        ],
        OVERALL_SHRINK_K,
    )

    raw_venue = shrink(
        df[
            raw_venue_col
        ],
        df[
            venue_games_col
        ],
        VENUE_SHRINK_K,
    )

    adj_venue = shrink(
        df[
            adj_venue_col
        ],
        df[
            venue_games_col
        ],
        VENUE_SHRINK_K,
    )

    raw_overall = transition_adjust(
        raw_overall,
        df[
            changed_col
        ],
    )

    adj_overall = transition_adjust(
        adj_overall,
        df[
            changed_col
        ],
    )

    raw_venue = transition_adjust(
        raw_venue,
        df[
            changed_col
        ],
    )

    adj_venue = transition_adjust(
        adj_venue,
        df[
            changed_col
        ],
    )

    overall = blend_strength(
        raw_overall,
        adj_overall,
        adjustment_weight,
    )

    venue = blend_strength(
        raw_venue,
        adj_venue,
        adjustment_weight,
    )

    return (
        OVERALL_WEIGHT
        * overall
        +
        VENUE_WEIGHT
        * venue
    )


# =========================================================
# BUILD HOME/AWAY MATCH TABLE
# =========================================================

def build_match_table(
    df,
):

    home = df[
        df[
            "venue"
        ]
        == "HOME"
    ].copy()

    away = df[
        df[
            "venue"
        ]
        == "AWAY"
    ].copy()

    signal_map = {
        # Overall RAW
        "raw_attack_goals":
            "raw_goal_attack",

        "raw_defense_goals":
            "raw_goal_defense",

        "raw_attack_shots":
            "raw_shot_attack",

        "raw_defense_shots":
            "raw_shot_defense",

        "raw_attack_sot":
            "raw_sot_attack",

        "raw_defense_sot":
            "raw_sot_defense",

        # Overall ADJUSTED
        "v3_attack_goals":
            "adj_goal_attack",

        "v3_defense_goals":
            "adj_goal_defense",

        "v3_attack_shots":
            "adj_shot_attack",

        "v3_defense_shots":
            "adj_shot_defense",

        "v3_attack_sot":
            "adj_sot_attack",

        "v3_defense_sot":
            "adj_sot_defense",

        # Venue RAW
        "raw_venue_goal_attack":
            "raw_venue_goal_attack",

        "raw_venue_goal_defense":
            "raw_venue_goal_defense",

        "raw_venue_shot_attack":
            "raw_venue_shot_attack",

        "raw_venue_shot_defense":
            "raw_venue_shot_defense",

        "raw_venue_sot_attack":
            "raw_venue_sot_attack",

        "raw_venue_sot_defense":
            "raw_venue_sot_defense",

        # Venue ADJUSTED
        "adj_venue_goal_attack":
            "adj_venue_goal_attack",

        "adj_venue_goal_defense":
            "adj_venue_goal_defense",

        "adj_venue_shot_attack":
            "adj_venue_shot_attack",

        "adj_venue_shot_defense":
            "adj_venue_shot_defense",

        "adj_venue_sot_attack":
            "adj_venue_sot_attack",

        "adj_venue_sot_defense":
            "adj_venue_sot_defense",
    }

    # -----------------------------------------------------
    # HOME
    # -----------------------------------------------------

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

    ] + list(
        signal_map.keys()
    )

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

    for source, target in (
        signal_map.items()
    ):

        home_rename[
            source
        ] = (
            "home_"
            + target
        )

    home = home.rename(
        columns=home_rename
    )

    # -----------------------------------------------------
    # AWAY
    # -----------------------------------------------------

    away_keep = [
        "match_id",

        "team",
        "opponent",

        "pregame_games",
        "pregame_venue_games",

        "league_changed",

    ] + list(
        signal_map.keys()
    )

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

    for source, target in (
        signal_map.items()
    ):

        away_rename[
            source
        ] = (
            "away_"
            + target
        )

    away = away.rename(
        columns=away_rename
    )

    matches = home.merge(
        away,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    return matches


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


# =========================================================
# BUILD MODEL
# =========================================================

def calculate_model(
    df,
    adjustment_weight,
    seasons,
):

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
                    df,
                    side,
                    signal,
                    kind,
                    adjustment_weight,
                )

    # Exact V2 signal blend.
    home_attack = (
        GOAL_WEIGHT
        * signals[
            "home_goal_attack"
        ]
        +
        SHOT_WEIGHT
        * signals[
            "home_shot_attack"
        ]
        +
        SOT_WEIGHT
        * signals[
            "home_sot_attack"
        ]
    )

    home_defense = (
        GOAL_WEIGHT
        * signals[
            "home_goal_defense"
        ]
        +
        SHOT_WEIGHT
        * signals[
            "home_shot_defense"
        ]
        +
        SOT_WEIGHT
        * signals[
            "home_sot_defense"
        ]
    )

    away_attack = (
        GOAL_WEIGHT
        * signals[
            "away_goal_attack"
        ]
        +
        SHOT_WEIGHT
        * signals[
            "away_shot_attack"
        ]
        +
        SOT_WEIGHT
        * signals[
            "away_sot_attack"
        ]
    )

    away_defense = (
        GOAL_WEIGHT
        * signals[
            "away_goal_defense"
        ]
        +
        SHOT_WEIGHT
        * signals[
            "away_shot_defense"
        ]
        +
        SOT_WEIGHT
        * signals[
            "away_sot_defense"
        ]
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

    sub = df.loc[
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
# FULL PREDICTIONS
# =========================================================

def build_predictions(
    df,
    adjustment_weight,
):

    out = df.copy()

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
                    adjustment_weight,
                )

    out[
        "home_attack_v3"
    ] = (
        GOAL_WEIGHT
        * signals[
            "home_goal_attack"
        ]
        +
        SHOT_WEIGHT
        * signals[
            "home_shot_attack"
        ]
        +
        SOT_WEIGHT
        * signals[
            "home_sot_attack"
        ]
    )

    out[
        "home_defense_v3"
    ] = (
        GOAL_WEIGHT
        * signals[
            "home_goal_defense"
        ]
        +
        SHOT_WEIGHT
        * signals[
            "home_shot_defense"
        ]
        +
        SOT_WEIGHT
        * signals[
            "home_sot_defense"
        ]
    )

    out[
        "away_attack_v3"
    ] = (
        GOAL_WEIGHT
        * signals[
            "away_goal_attack"
        ]
        +
        SHOT_WEIGHT
        * signals[
            "away_shot_attack"
        ]
        +
        SOT_WEIGHT
        * signals[
            "away_sot_attack"
        ]
    )

    out[
        "away_defense_v3"
    ] = (
        GOAL_WEIGHT
        * signals[
            "away_goal_defense"
        ]
        +
        SHOT_WEIGHT
        * signals[
            "away_shot_defense"
        ]
        +
        SOT_WEIGHT
        * signals[
            "away_sot_defense"
        ]
    )

    out[
        "home_lambda_v3"
    ] = (
        out[
            "lg_home_goals"
        ]
        *
        out[
            "home_attack_v3"
        ]
        *
        out[
            "away_defense_v3"
        ]
    ).clip(
        lower=0.15,
        upper=4.50,
    )

    out[
        "away_lambda_v3"
    ] = (
        out[
            "lg_away_goals"
        ]
        *
        out[
            "away_attack_v3"
        ]
        *
        out[
            "home_defense_v3"
        ]
    ).clip(
        lower=0.15,
        upper=4.50,
    )

    usable = (
        out[
            "home_lambda_v3"
        ].notna()
        &
        out[
            "away_lambda_v3"
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
            "home_lambda_v3"
        ].to_numpy(),
        out[
            "away_lambda_v3"
        ].to_numpy(),
    )

    out[
        "p_home_v3"
    ] = probs[:, 0]

    out[
        "p_draw_v3"
    ] = probs[:, 1]

    out[
        "p_away_v3"
    ] = probs[:, 2]

    out[
        "opponent_adjustment_weight"
    ] = adjustment_weight

    return out


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("==============================")
    print("TUNING OPPONENT-ADJUSTED V3")
    print("==============================")
    print()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
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
        "Frozen V2 structure:"
    )

    print(
        "Goals:           70%"
    )

    print(
        "Shots:           15%"
    )

    print(
        "SOT:             15%"
    )

    print(
        "Overall history: 75%"
    )

    print(
        "Venue history:   25%"
    )

    print()

    # =====================================================
    # VENUE BASELINES
    # =====================================================

    print(
        "Building home/away shot baselines..."
    )

    venue_baselines = (
        build_venue_league_baselines(
            df
        )
    )

    df = df.merge(
        venue_baselines,
        on=[
            "league_code",
            "date",
        ],
        how="left",
        validate="many_to_one",
    )

    # =====================================================
    # VENUE HISTORIES
    # =====================================================

    print(
        "Building raw and adjusted "
        "venue histories..."
    )

    df = add_venue_histories(
        df
    )

    df = build_venue_strengths(
        df
    )

    df = add_league_transition(
        df
    )

    # =====================================================
    # MATCH TABLE
    # =====================================================

    print(
        "Building home/away match table..."
    )

    matches = build_match_table(
        df
    )

    print(
        f"Match rows: "
        f"{len(matches):,}"
    )

    # =====================================================
    # TUNE
    # =====================================================

    print()
    print(
        f"Testing "
        f"{len(ADJUSTMENT_WEIGHTS)} "
        f"opponent-adjustment weights..."
    )

    results = []

    for weight in (
        ADJUSTMENT_WEIGHTS
    ):

        metrics = calculate_model(
            matches,
            weight,
            TUNING_SEASONS,
        )

        results.append({
            "adjustment_weight":
                weight,

            "raw_weight":
                1.0 - weight,

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

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
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

    # =====================================================
    # RESULTS
    # =====================================================

    print()
    print("==============================")
    print("OPPONENT ADJUSTMENT RESULTS")
    print("==============================")

    display = results_df.copy()

    display[
        "accuracy"
    ] *= 100.0

    print(
        display[
            [
                "rank",
                "adjustment_weight",
                "raw_weight",
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

    best = (
        results_df
        .iloc[0]
    )

    best_weight = float(
        best[
            "adjustment_weight"
        ]
    )

    # =====================================================
    # WINNER
    # =====================================================

    print()
    print("==============================")
    print("WINNING V3 ADJUSTMENT")
    print("==============================")

    print(
        f"Opponent-adjusted: "
        f"{best_weight:.0%}"
    )

    print(
        f"Raw strength:      "
        f"{1-best_weight:.0%}"
    )

    print(
        f"Tuning LL:         "
        f"{best['log_loss']:.5f}"
    )

    # =====================================================
    # V2 RAW COMPARATOR
    # =====================================================

    v2_tuning = calculate_model(
        matches,
        0.0,
        TUNING_SEASONS,
    )

    print(
        f"0% adjustment LL:  "
        f"{v2_tuning['log_loss']:.5f}"
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    raw_validation = calculate_model(
        matches,
        0.0,
        VALIDATION_SEASONS,
    )

    v3_validation = calculate_model(
        matches,
        best_weight,
        VALIDATION_SEASONS,
    )

    print()
    print("==============================")
    print("VALIDATION 2023/24–2024/25")
    print("==============================")

    print(
        f"{'Metric':<15}"
        f"{'Raw':>12}"
        f"{'V3':>12}"
        f"{'Change':>12}"
    )

    print("-" * 51)

    print(
        f"{'Accuracy':<15}"
        f"{raw_validation['accuracy']:>11.2%}"
        f"{v3_validation['accuracy']:>11.2%}"
        f"{v3_validation['accuracy'] - raw_validation['accuracy']:>+11.2%}"
    )

    print(
        f"{'Log Loss':<15}"
        f"{raw_validation['log_loss']:>12.5f}"
        f"{v3_validation['log_loss']:>12.5f}"
        f"{v3_validation['log_loss'] - raw_validation['log_loss']:>+12.5f}"
    )

    print(
        f"{'Brier':<15}"
        f"{raw_validation['brier']:>12.5f}"
        f"{v3_validation['brier']:>12.5f}"
        f"{v3_validation['brier'] - raw_validation['brier']:>+12.5f}"
    )

    # =====================================================
    # FINAL 2025/26
    # =====================================================

    raw_final = calculate_model(
        matches,
        0.0,
        FINAL_SEASON,
    )

    v3_final = calculate_model(
        matches,
        best_weight,
        FINAL_SEASON,
    )

    print()
    print("==============================")
    print("FINAL CHECK — 2025/26")
    print("==============================")

    print(
        f"Games: "
        f"{v3_final['games']:,}"
    )

    print()
    print(
        f"{'Metric':<15}"
        f"{'Raw':>12}"
        f"{'V3':>12}"
        f"{'Change':>12}"
    )

    print("-" * 51)

    print(
        f"{'Accuracy':<15}"
        f"{raw_final['accuracy']:>11.2%}"
        f"{v3_final['accuracy']:>11.2%}"
        f"{v3_final['accuracy'] - raw_final['accuracy']:>+11.2%}"
    )

    print(
        f"{'Log Loss':<15}"
        f"{raw_final['log_loss']:>12.5f}"
        f"{v3_final['log_loss']:>12.5f}"
        f"{v3_final['log_loss'] - raw_final['log_loss']:>+12.5f}"
    )

    print(
        f"{'Brier':<15}"
        f"{raw_final['brier']:>12.5f}"
        f"{v3_final['brier']:>12.5f}"
        f"{v3_final['brier'] - raw_final['brier']:>+12.5f}"
    )

    # =====================================================
    # BY LEAGUE — FINAL
    # =====================================================

    print()
    print("==============================")
    print("2025/26 — BY LEAGUE")
    print("==============================")

    league_rows = []

    for league, group in (
        matches.groupby(
            "league"
        )
    ):

        raw_metrics = (
            calculate_model(
                group,
                0.0,
                FINAL_SEASON,
            )
        )

        v3_metrics = (
            calculate_model(
                group,
                best_weight,
                FINAL_SEASON,
            )
        )

        league_rows.append({
            "league":
                league,

            "games":
                v3_metrics[
                    "games"
                ],

            "raw_ll":
                raw_metrics[
                    "log_loss"
                ],

            "v3_ll":
                v3_metrics[
                    "log_loss"
                ],

            "ll_change":
                (
                    v3_metrics[
                        "log_loss"
                    ]
                    -
                    raw_metrics[
                        "log_loss"
                    ]
                ),

            "raw_brier":
                raw_metrics[
                    "brier"
                ],

            "v3_brier":
                v3_metrics[
                    "brier"
                ],

            "raw_acc":
                raw_metrics[
                    "accuracy"
                ],

            "v3_acc":
                v3_metrics[
                    "accuracy"
                ],
        })

    league_table = pd.DataFrame(
        league_rows
    )

    for col in [
        "raw_acc",
        "v3_acc",
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
    # SAVE BEST V3
    # =====================================================

    predictions = build_predictions(
        matches,
        best_weight,
    )

    predictions.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print()
    print("==============================")
    print("V3 TUNING COMPLETE")
    print("==============================")

    if best_weight == 0.0:

        print(
            "Opponent adjustment was rejected."
        )

    else:

        print(
            "Opponent adjustment improved "
            "the tuning objective."
        )

    print()
    print(
        "2025/26 was not used to "
        "choose adjustment weight ✅"
    )

    print()
    print(
        f"Tuning results:"
        f"\n{OUTPUT_RESULTS}"
    )

    print()
    print(
        f"V3 predictions:"
        f"\n{OUTPUT_PREDICTIONS}"
    )


if __name__ == "__main__":
    main()