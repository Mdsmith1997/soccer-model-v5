from pathlib import Path
import math

import numpy as np
import pandas as pd

import build_opponent_adjusted_v3 as builder
import tune_opponent_adjustment_v3 as model
import tune_opponent_strength_v3 as strength_model


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TEAM_FILE = (
    ROOT
    / "data"
    / "processed"
    / "team_game_stats.csv"
)

XG_FILE = (
    ROOT
    / "data"
    / "processed"
    / "understat_xg_matched.csv"
)

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "opponent_adjusted_recency_v5_confirmation.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "opponent_adjusted_recency_v5_predictions.csv"
)


# ============================================================
# FROZEN V5 SETTINGS
# ============================================================

GOAL_WEIGHT = 0.09
XG_WEIGHT = 0.75
SHOT_WEIGHT = 0.16
SOT_WEIGHT = 0.00

OVERALL_WEIGHT = 0.75
VENUE_WEIGHT = 0.25

OPPONENT_STRENGTH = 0.875

# SOT has zero final model weight, but keep its history
# construction frozen at the old recency for pipeline parity.
SOT_RECENCY = 0.95

MAX_GOALS = 10
EPS = 1e-12


# ============================================================
# TEST SETTINGS
# ============================================================

RECENCY_SETTINGS = [
    {
        "name": "CONTROL",
        "goal_recency": 0.950,
        "xg_recency": 0.950,
        "shot_recency": 0.950,
    },
    {
        "name": "WINNER",
        "goal_recency": 0.975,
        "xg_recency": 0.925,
        "shot_recency": 0.850,
    },
    {
        "name": "NEARBY_1",
        "goal_recency": 0.975,
        "xg_recency": 0.925,
        "shot_recency": 0.875,
    },
    {
        "name": "NEARBY_2",
        "goal_recency": 0.975,
        "xg_recency": 0.950,
        "shot_recency": 0.850,
    },
    {
        "name": "NEARBY_3",
        "goal_recency": 0.950,
        "xg_recency": 0.925,
        "shot_recency": 0.850,
    },
    {
        "name": "NEARBY_4",
        "goal_recency": 0.975,
        "xg_recency": 0.900,
        "shot_recency": 0.850,
    },
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

SUPPORTED_LEAGUES = {
    "Premier League",
    "Bundesliga",
}


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
        home_goals
        >
        away_goals,
        0,
        np.where(
            home_goals
            ==
            away_goals,
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

    return (
        -np.log(
            np.clip(
                chosen,
                EPS,
                1.0,
            )
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
# GENERIC PRIOR EW
# ============================================================

def weighted_prior_average(
    values,
    decay,
):

    arr = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    results = np.full(
        len(arr),
        np.nan,
        dtype=float,
    )

    numerator = 0.0
    denominator = 0.0

    for i, value in enumerate(
        arr
    ):

        if denominator > 0:

            results[i] = (
                numerator
                /
                denominator
            )

        numerator *= decay
        denominator *= decay

        if np.isfinite(
            value
        ):

            numerator += value
            denominator += 1.0

    return pd.Series(
        results,
        index=values.index,
    )


# ============================================================
# SIGNAL-SPECIFIC BASIC HISTORY
# ============================================================

def build_basic_history_signal_recency(
    df,
    goal_recency,
    shot_recency,
):

    out = df.copy()

    team_group = out.groupby(
        "team",
        sort=False,
        group_keys=False,
    )

    out[
        "pregame_games"
    ] = (
        team_group.cumcount()
    )

    recency_map = {
        "goals_for":
            goal_recency,

        "goals_against":
            goal_recency,

        "shots_for":
            shot_recency,

        "shots_against":
            shot_recency,

        "shots_on_target_for":
            SOT_RECENCY,

        "shots_on_target_against":
            SOT_RECENCY,
    }

    for col, decay in (
        recency_map.items()
    ):

        out[
            f"ew_{col}"
        ] = (
            team_group[
                col
            ]
            .transform(
                lambda s,
                d=decay:
                    weighted_prior_average(
                        s,
                        d,
                    )
            )
        )

    return out


# ============================================================
# SIGNAL-SPECIFIC ADJUSTED HISTORY
# ============================================================

def add_adjusted_history_signal_recency(
    df,
    goal_recency,
    shot_recency,
):

    out = df.copy()

    team_group = out.groupby(
        "team",
        sort=False,
        group_keys=False,
    )

    recency_map = {
        "adj_goals_for":
            goal_recency,

        "adj_goals_against":
            goal_recency,

        "adj_shots_for":
            shot_recency,

        "adj_shots_against":
            shot_recency,

        "adj_sot_for":
            SOT_RECENCY,

        "adj_sot_against":
            SOT_RECENCY,
    }

    for col, decay in (
        recency_map.items()
    ):

        out[
            f"ew_{col}"
        ] = (
            team_group[
                col
            ]
            .transform(
                lambda s,
                d=decay:
                    weighted_prior_average(
                        s,
                        d,
                    )
            )
        )

    return out


# ============================================================
# SIGNAL-SPECIFIC VENUE HISTORY
# ============================================================

def add_venue_histories_signal_recency(
    df,
    goal_recency,
    shot_recency,
):

    out = df.copy()

    venue_group = out.groupby(
        [
            "team",
            "venue",
        ],
        sort=False,
        group_keys=False,
    )

    out[
        "pregame_venue_games"
    ] = (
        venue_group.cumcount()
    )

    recency_map = {
        # RAW GOALS
        "goals_for":
            goal_recency,

        "goals_against":
            goal_recency,

        # RAW SHOTS
        "shots_for":
            shot_recency,

        "shots_against":
            shot_recency,

        # RAW SOT
        "shots_on_target_for":
            SOT_RECENCY,

        "shots_on_target_against":
            SOT_RECENCY,

        # ADJUSTED GOALS
        "adj_goals_for":
            goal_recency,

        "adj_goals_against":
            goal_recency,

        # ADJUSTED SHOTS
        "adj_shots_for":
            shot_recency,

        "adj_shots_against":
            shot_recency,

        # ADJUSTED SOT
        "adj_sot_for":
            SOT_RECENCY,

        "adj_sot_against":
            SOT_RECENCY,
    }

    for col, decay in (
        recency_map.items()
    ):

        out[
            f"venue_ew_{col}"
        ] = (
            venue_group[
                col
            ]
            .transform(
                lambda s,
                d=decay:
                    weighted_prior_average(
                        s,
                        d,
                    )
            )
        )

    return out


# ============================================================
# LOAD TEAM DATA
# ============================================================

def load_team_data():

    df = pd.read_csv(
        TEAM_FILE,
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

    df = df[
        df[
            "league"
        ].isin(
            SUPPORTED_LEAGUES
        )
    ].copy()

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

    return df


# ============================================================
# BUILD GOAL / SHOT OPPONENT-ADJUSTED PIPELINE
# ============================================================

def build_goal_shot_base(
    team,
    goal_recency,
    shot_recency,
):

    df = team.copy()

    # ========================================================
    # SAME LEAKAGE-SAFE LEAGUE BASELINES
    # ========================================================

    league = (
        builder
        .build_league_baselines(
            df
        )
    )

    df = df.merge(
        league,
        on=[
            "league_code",
            "date",
        ],
        how="left",
        validate="many_to_one",
    )

    # ========================================================
    # RAW PREGAME HISTORIES
    #
    # Explicit signal-specific recencies.
    # ========================================================

    df = (
        build_basic_history_signal_recency(
            df,
            goal_recency,
            shot_recency,
        )
    )

    # Existing normalization formulas.
    df = builder.add_raw_strengths(
        df
    )

    # ========================================================
    # OPPONENT PREGAME SNAPSHOTS
    # ========================================================

    df = (
        builder
        .attach_opponent_pregame_strength(
            df
        )
    )

    # ========================================================
    # HOME / AWAY LEAGUE BASELINES
    # ========================================================

    venue_baselines = (
        model
        .build_venue_league_baselines(
            df
        )
    )

    # ========================================================
    # GAME-LEVEL OPPONENT ADJUSTMENT
    #
    # Frozen production strength = 0.875.
    # ========================================================

    df = (
        strength_model
        .add_adjusted_game_performance(
            df,
            OPPONENT_STRENGTH,
        )
    )

    # ========================================================
    # OPPONENT-ADJUSTED OVERALL HISTORIES
    #
    # Explicit signal-specific recencies.
    # ========================================================

    df = (
        add_adjusted_history_signal_recency(
            df,
            goal_recency,
            shot_recency,
        )
    )

    # Existing V3 strength normalization.
    df = builder.add_v3_strengths(
        df
    )

    # ========================================================
    # ATTACH VENUE BASELINES
    # ========================================================

    df = df.merge(
        venue_baselines,
        on=[
            "league_code",
            "date",
        ],
        how="left",
        validate="many_to_one",
    )

    # ========================================================
    # VENUE HISTORIES
    #
    # Explicit signal-specific recencies.
    # ========================================================

    df = (
        add_venue_histories_signal_recency(
            df,
            goal_recency,
            shot_recency,
        )
    )

    # Existing venue strength formulas.
    df = model.build_venue_strengths(
        df
    )

    # Existing league-transition treatment.
    df = model.add_league_transition(
        df
    )

    # ========================================================
    # MATCH TABLE
    # ========================================================

    matches = model.build_match_table(
        df
    )

    return matches


# ============================================================
# LOAD XG
# ============================================================

def load_xg():

    xg = pd.read_csv(
        XG_FILE,
        parse_dates=[
            "date",
        ],
    )

    xg[
        "season"
    ] = (
        xg[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    return xg


# ============================================================
# XG TEAM ROWS
# ============================================================

def build_xg_team_rows(
    xg,
):

    home = pd.DataFrame(
        {
            "match_id":
                xg[
                    "match_id"
                ],

            "date":
                xg[
                    "date"
                ],

            "season":
                xg[
                    "season"
                ],

            "league":
                xg[
                    "league"
                ],

            "team":
                xg[
                    "home_team"
                ],

            "opponent":
                xg[
                    "away_team"
                ],

            "venue":
                "HOME",

            "xg_for":
                xg[
                    "home_xg"
                ],

            "xg_against":
                xg[
                    "away_xg"
                ],
        }
    )

    away = pd.DataFrame(
        {
            "match_id":
                xg[
                    "match_id"
                ],

            "date":
                xg[
                    "date"
                ],

            "season":
                xg[
                    "season"
                ],

            "league":
                xg[
                    "league"
                ],

            "team":
                xg[
                    "away_team"
                ],

            "opponent":
                xg[
                    "home_team"
                ],

            "venue":
                "AWAY",

            "xg_for":
                xg[
                    "away_xg"
                ],

            "xg_against":
                xg[
                    "home_xg"
                ],
        }
    )

    df = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    return (
        df
        .sort_values(
            [
                "team",
                "date",
                "match_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# XG LEAGUE BASELINE
# ============================================================

def add_xg_league_baseline(
    df,
):

    work = df.copy()

    work[
        "xg_valid"
    ] = (
        work[
            "xg_for"
        ].notna()
    ).astype(
        int
    )

    daily = (
        work
        .groupby(
            [
                "league",
                "date",
            ],
            as_index=False,
        )
        .agg(
            xg_obs=(
                "xg_valid",
                "sum",
            ),
            xg_total=(
                "xg_for",
                "sum",
            ),
        )
        .sort_values(
            [
                "league",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    grouped = (
        daily.groupby(
            "league",
            sort=False,
        )
    )

    daily[
        "prior_xg_obs"
    ] = (
        grouped[
            "xg_obs"
        ]
        .cumsum()
        -
        daily[
            "xg_obs"
        ]
    )

    daily[
        "prior_xg_total"
    ] = (
        grouped[
            "xg_total"
        ]
        .cumsum()
        -
        daily[
            "xg_total"
        ]
    )

    daily[
        "lg_team_xg"
    ] = (
        daily[
            "prior_xg_total"
        ]
        /
        daily[
            "prior_xg_obs"
        ]
        .replace(
            0,
            np.nan,
        )
    )

    return df.merge(
        daily[
            [
                "league",
                "date",
                "lg_team_xg",
            ]
        ],
        on=[
            "league",
            "date",
        ],
        how="left",
        validate="many_to_one",
    )


# ============================================================
# RAW XG PREGAME STRENGTH
# ============================================================

def add_raw_xg_strength(
    df,
    xg_recency,
):

    out = (
        df
        .sort_values(
            [
                "team",
                "date",
                "match_id",
            ]
        )
        .copy()
    )

    group = out.groupby(
        "team",
        sort=False,
        group_keys=False,
    )

    out[
        "ew_xg_for"
    ] = (
        group[
            "xg_for"
        ]
        .transform(
            lambda s:
                weighted_prior_average(
                    s,
                    xg_recency,
                )
        )
    )

    out[
        "ew_xg_against"
    ] = (
        group[
            "xg_against"
        ]
        .transform(
            lambda s:
                weighted_prior_average(
                    s,
                    xg_recency,
                )
        )
    )

    out[
        "raw_xg_attack"
    ] = (
        out[
            "ew_xg_for"
        ]
        /
        out[
            "lg_team_xg"
        ]
    )

    out[
        "raw_xg_defense"
    ] = (
        out[
            "ew_xg_against"
        ]
        /
        out[
            "lg_team_xg"
        ]
    )

    return out


# ============================================================
# ATTACH OPPONENT XG SNAPSHOT
# ============================================================

def attach_opponent_xg(
    df,
):

    opponent = df[
        [
            "match_id",
            "team",
            "raw_xg_attack",
            "raw_xg_defense",
        ]
    ].copy()

    opponent = opponent.rename(
        columns={
            "team":
                "opponent_check",

            "raw_xg_attack":
                "opp_xg_attack",

            "raw_xg_defense":
                "opp_xg_defense",
        }
    )

    return df.merge(
        opponent,
        left_on=[
            "match_id",
            "opponent",
        ],
        right_on=[
            "match_id",
            "opponent_check",
        ],
        how="left",
        validate="one_to_one",
    )


# ============================================================
# ADJUST XG FOR OPPONENT QUALITY
# ============================================================

def adjust_xg_performance(
    df,
):

    out = df.copy()

    out[
        "opp_xg_attack"
    ] = (
        out[
            "opp_xg_attack"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            1.0
        )
    )

    out[
        "opp_xg_defense"
    ] = (
        out[
            "opp_xg_defense"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            1.0
        )
    )

    defense_factor = (
        (
            1.0
            -
            OPPONENT_STRENGTH
        )
        +
        OPPONENT_STRENGTH
        *
        out[
            "opp_xg_defense"
        ]
    )

    attack_factor = (
        (
            1.0
            -
            OPPONENT_STRENGTH
        )
        +
        OPPONENT_STRENGTH
        *
        out[
            "opp_xg_attack"
        ]
    )

    out[
        "adj_xg_for"
    ] = (
        out[
            "xg_for"
        ]
        /
        defense_factor
    )

    out[
        "adj_xg_against"
    ] = (
        out[
            "xg_against"
        ]
        /
        attack_factor
    )

    return out


# ============================================================
# ADJUSTED XG HISTORY
# ============================================================

def add_adjusted_xg_history(
    df,
    xg_recency,
):

    out = (
        df
        .sort_values(
            [
                "team",
                "date",
                "match_id",
            ]
        )
        .copy()
    )

    group = out.groupby(
        "team",
        sort=False,
        group_keys=False,
    )

    for col in [
        "adj_xg_for",
        "adj_xg_against",
    ]:

        out[
            f"ew_{col}"
        ] = (
            group[
                col
            ]
            .transform(
                lambda s:
                    weighted_prior_average(
                        s,
                        xg_recency,
                    )
            )
        )

    out[
        "adj_xg_attack"
    ] = (
        out[
            "ew_adj_xg_for"
        ]
        /
        out[
            "lg_team_xg"
        ]
    )

    out[
        "adj_xg_defense"
    ] = (
        out[
            "ew_adj_xg_against"
        ]
        /
        out[
            "lg_team_xg"
        ]
    )

    return out


# ============================================================
# XG VENUE HISTORY
# ============================================================

def add_xg_venue_history(
    df,
    xg_recency,
):

    out = (
        df
        .sort_values(
            [
                "team",
                "venue",
                "date",
                "match_id",
            ]
        )
        .copy()
    )

    group = out.groupby(
        [
            "team",
            "venue",
        ],
        sort=False,
        group_keys=False,
    )

    for col in [
        "adj_xg_for",
        "adj_xg_against",
    ]:

        out[
            f"venue_ew_{col}"
        ] = (
            group[
                col
            ]
            .transform(
                lambda s:
                    weighted_prior_average(
                        s,
                        xg_recency,
                    )
            )
        )

    out[
        "venue_adj_xg_attack"
    ] = (
        out[
            "venue_ew_adj_xg_for"
        ]
        /
        out[
            "lg_team_xg"
        ]
    )

    out[
        "venue_adj_xg_defense"
    ] = (
        out[
            "venue_ew_adj_xg_against"
        ]
        /
        out[
            "lg_team_xg"
        ]
    )

    out[
        "xg_attack_final"
    ] = (
        OVERALL_WEIGHT
        *
        out[
            "adj_xg_attack"
        ]
        +
        VENUE_WEIGHT
        *
        out[
            "venue_adj_xg_attack"
        ]
    )

    out[
        "xg_defense_final"
    ] = (
        OVERALL_WEIGHT
        *
        out[
            "adj_xg_defense"
        ]
        +
        VENUE_WEIGHT
        *
        out[
            "venue_adj_xg_defense"
        ]
    )

    return out


# ============================================================
# BUILD XG MATCH TABLE
# ============================================================

def build_xg_match_table(
    xg_team,
    xg_recency,
):

    df = add_xg_league_baseline(
        xg_team
    )

    df = add_raw_xg_strength(
        df,
        xg_recency,
    )

    df = attach_opponent_xg(
        df
    )

    df = adjust_xg_performance(
        df
    )

    df = add_adjusted_xg_history(
        df,
        xg_recency,
    )

    df = add_xg_venue_history(
        df,
        xg_recency,
    )

    home = df[
        df[
            "venue"
        ]
        ==
        "HOME"
    ][
        [
            "match_id",
            "xg_attack_final",
            "xg_defense_final",
        ]
    ].rename(
        columns={
            "xg_attack_final":
                "home_xg_attack",

            "xg_defense_final":
                "home_xg_defense",
        }
    )

    away = df[
        df[
            "venue"
        ]
        ==
        "AWAY"
    ][
        [
            "match_id",
            "xg_attack_final",
            "xg_defense_final",
        ]
    ].rename(
        columns={
            "xg_attack_final":
                "away_xg_attack",

            "xg_defense_final":
                "away_xg_defense",
        }
    )

    return home.merge(
        away,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )


# ============================================================
# PREPARE COMPONENT SIGNAL
# ============================================================

def prepare_component(
    matches,
    side,
    signal,
    kind,
):

    overall_col = (
        f"{side}_adj_"
        f"{signal}_"
        f"{kind}"
    )

    venue_col = (
        f"{side}_adj_venue_"
        f"{signal}_"
        f"{kind}"
    )

    if (
        overall_col
        not in matches.columns
    ):

        raise ValueError(
            f"Missing column: "
            f"{overall_col}"
        )

    if (
        venue_col
        not in matches.columns
    ):

        raise ValueError(
            f"Missing column: "
            f"{venue_col}"
        )

    return (
        OVERALL_WEIGHT
        *
        matches[
            overall_col
        ]
        +
        VENUE_WEIGHT
        *
        matches[
            venue_col
        ]
    )


# ============================================================
# BUILD V5 MATCH TABLE
# ============================================================

def build_v5_match_table(
    goal_shot_matches,
    xg_matches,
):

    df = (
        goal_shot_matches
        .merge(
            xg_matches,
            on="match_id",
            how="inner",
            validate="one_to_one",
        )
    )

    # --------------------------------------------------------
    # GOALS
    # --------------------------------------------------------

    df[
        "home_goal_attack_v5"
    ] = prepare_component(
        df,
        "home",
        "goal",
        "attack",
    )

    df[
        "home_goal_defense_v5"
    ] = prepare_component(
        df,
        "home",
        "goal",
        "defense",
    )

    df[
        "away_goal_attack_v5"
    ] = prepare_component(
        df,
        "away",
        "goal",
        "attack",
    )

    df[
        "away_goal_defense_v5"
    ] = prepare_component(
        df,
        "away",
        "goal",
        "defense",
    )

    # --------------------------------------------------------
    # SHOTS
    # --------------------------------------------------------

    df[
        "home_shot_attack_v5"
    ] = prepare_component(
        df,
        "home",
        "shot",
        "attack",
    )

    df[
        "home_shot_defense_v5"
    ] = prepare_component(
        df,
        "home",
        "shot",
        "defense",
    )

    df[
        "away_shot_attack_v5"
    ] = prepare_component(
        df,
        "away",
        "shot",
        "attack",
    )

    df[
        "away_shot_defense_v5"
    ] = prepare_component(
        df,
        "away",
        "shot",
        "defense",
    )

    return df


# ============================================================
# LAMBDAS
# ============================================================

def build_lambdas(
    df,
):

    home_attack = (
        GOAL_WEIGHT
        *
        df[
            "home_goal_attack_v5"
        ]
        +
        XG_WEIGHT
        *
        df[
            "home_xg_attack"
        ]
        +
        SHOT_WEIGHT
        *
        df[
            "home_shot_attack_v5"
        ]
    )

    home_defense = (
        GOAL_WEIGHT
        *
        df[
            "home_goal_defense_v5"
        ]
        +
        XG_WEIGHT
        *
        df[
            "home_xg_defense"
        ]
        +
        SHOT_WEIGHT
        *
        df[
            "home_shot_defense_v5"
        ]
    )

    away_attack = (
        GOAL_WEIGHT
        *
        df[
            "away_goal_attack_v5"
        ]
        +
        XG_WEIGHT
        *
        df[
            "away_xg_attack"
        ]
        +
        SHOT_WEIGHT
        *
        df[
            "away_shot_attack_v5"
        ]
    )

    away_defense = (
        GOAL_WEIGHT
        *
        df[
            "away_goal_defense_v5"
        ]
        +
        XG_WEIGHT
        *
        df[
            "away_xg_defense"
        ]
        +
        SHOT_WEIGHT
        *
        df[
            "away_shot_defense_v5"
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
        sub
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
        f"{'Control':>14}"
        f"{'Candidate':>14}"
        f"{'Change':>14}"
    )

    print("-" * 57)

    print(
        f"{'Accuracy':<15}"
        f"{control['accuracy']:>13.2%}"
        f"{candidate['accuracy']:>13.2%}"
        f"{candidate['accuracy'] - control['accuracy']:>+13.2%}"
    )

    print(
        f"{'Log Loss':<15}"
        f"{control['log_loss']:>14.5f}"
        f"{candidate['log_loss']:>14.5f}"
        f"{candidate['log_loss'] - control['log_loss']:>+14.5f}"
    )

    print(
        f"{'Brier':<15}"
        f"{control['brier']:>14.5f}"
        f"{candidate['brier']:>14.5f}"
        f"{candidate['brier'] - control['brier']:>+14.5f}"
    )


# ============================================================
# BUILD ONE SETTING
# ============================================================

def build_setting(
    team,
    xg_team,
    setting,
):

    goal_recency = (
        setting[
            "goal_recency"
        ]
    )

    xg_recency = (
        setting[
            "xg_recency"
        ]
    )

    shot_recency = (
        setting[
            "shot_recency"
        ]
    )

    goal_shot_matches = (
        build_goal_shot_base(
            team,
            goal_recency,
            shot_recency,
        )
    )

    xg_matches = (
        build_xg_match_table(
            xg_team,
            xg_recency,
        )
    )

    matches = build_v5_match_table(
        goal_shot_matches,
        xg_matches,
    )

    return matches


# ============================================================
# DEBUG COMPONENT SAVE
# ============================================================

def save_debug_components(
    matches,
    setting_name,
):

    debug_names = {
        "CONTROL",
        "WINNER",
        "NEARBY_1",
        "NEARBY_2",
        "NEARBY_3",
        "NEARBY_4",
    }

    if (
        setting_name
        not in debug_names
    ):

        return

    cols = [
        "match_id",

        "home_goal_attack_v5",
        "home_goal_defense_v5",
        "away_goal_attack_v5",
        "away_goal_defense_v5",

        "home_shot_attack_v5",
        "home_shot_defense_v5",
        "away_shot_attack_v5",
        "away_shot_defense_v5",

        "home_xg_attack",
        "home_xg_defense",
        "away_xg_attack",
        "away_xg_defense",
    ]

    debug = (
        matches[
            cols
        ]
        .copy()
        .sort_values(
            "match_id"
        )
        .reset_index(
            drop=True
        )
    )

    path = (
        ROOT
        / "data"
        / "processed"
        / (
            "recency_debug_"
            + setting_name.lower()
            + ".csv"
        )
    )

    debug.to_csv(
        path,
        index=False,
    )


# ============================================================
# DEBUG DIFFERENCE REPORT
# ============================================================

def print_debug_differences():

    control_path = (
        ROOT
        / "data"
        / "processed"
        / "recency_debug_control.csv"
    )

    if not control_path.exists():

        return

    control = pd.read_csv(
        control_path
    )

    print()
    print("==============================")
    print("RECENCY COMPONENT DEBUG")
    print("==============================")

    comparisons = [
        "winner",
        "nearby_1",
        "nearby_2",
        "nearby_3",
        "nearby_4",
    ]

    cols = [
        "home_goal_attack_v5",
        "home_goal_defense_v5",
        "away_goal_attack_v5",
        "away_goal_defense_v5",

        "home_shot_attack_v5",
        "home_shot_defense_v5",
        "away_shot_attack_v5",
        "away_shot_defense_v5",

        "home_xg_attack",
        "home_xg_defense",
        "away_xg_attack",
        "away_xg_defense",
    ]

    for name in comparisons:

        path = (
            ROOT
            / "data"
            / "processed"
            / (
                f"recency_debug_"
                f"{name}.csv"
            )
        )

        if not path.exists():

            continue

        other = pd.read_csv(
            path
        )

        merged = control.merge(
            other,
            on="match_id",
            suffixes=(
                "_control",
                "_other",
            ),
            how="inner",
            validate="one_to_one",
        )

        print()
        print(
            name.upper()
        )

        print("-" * 90)

        for col in cols:

            a = pd.to_numeric(
                merged[
                    f"{col}_control"
                ],
                errors="coerce",
            )

            b = pd.to_numeric(
                merged[
                    f"{col}_other"
                ],
                errors="coerce",
            )

            diff = (
                a
                - b
            ).abs()

            finite = (
                diff.notna()
            )

            changed = (
                diff[
                    finite
                ]
                >
                1e-12
            ).sum()

            if finite.any():

                max_diff = (
                    diff[
                        finite
                    ].max()
                )

                mean_diff = (
                    diff[
                        finite
                    ].mean()
                )

            else:

                max_diff = np.nan
                mean_diff = np.nan

            print(
                f"{col:<30}"
                f" changed="
                f"{changed:>6,}"
                f"  max="
                f"{max_diff:.10f}"
                f"  mean="
                f"{mean_diff:.10f}"
            )


# ============================================================
# DEBUG EXPECTATION CHECK
# ============================================================

def validate_debug_behavior():

    paths = {
        name:
        (
            ROOT
            / "data"
            / "processed"
            / f"recency_debug_{name}.csv"
        )
        for name in [
            "control",
            "winner",
            "nearby_1",
            "nearby_2",
            "nearby_3",
            "nearby_4",
        ]
    }

    if not all(
        path.exists()
        for path in paths.values()
    ):

        return

    frames = {
        name:
            pd.read_csv(
                path
            )
        for name, path
        in paths.items()
    }

    control = frames[
        "control"
    ]

    def changed_count(
        other_name,
        col,
    ):

        merged = control[
            [
                "match_id",
                col,
            ]
        ].merge(
            frames[
                other_name
            ][
                [
                    "match_id",
                    col,
                ]
            ],
            on="match_id",
            suffixes=(
                "_control",
                "_other",
            ),
            how="inner",
            validate="one_to_one",
        )

        a = pd.to_numeric(
            merged[
                f"{col}_control"
            ],
            errors="coerce",
        )

        b = pd.to_numeric(
            merged[
                f"{col}_other"
            ],
            errors="coerce",
        )

        diff = (
            a
            - b
        ).abs()

        return int(
            (
                diff
                >
                1e-12
            )
            .fillna(
                False
            )
            .sum()
        )

    # --------------------------------------------------------
    # WINNER changes all three signals.
    # --------------------------------------------------------

    winner_goal = changed_count(
        "winner",
        "home_goal_attack_v5",
    )

    winner_shot = changed_count(
        "winner",
        "home_shot_attack_v5",
    )

    winner_xg = changed_count(
        "winner",
        "home_xg_attack",
    )

    if winner_goal == 0:

        raise ValueError(
            "Goal recency is still not "
            "reaching final components."
        )

    if winner_shot == 0:

        raise ValueError(
            "Shot recency is still not "
            "reaching final components."
        )

    if winner_xg == 0:

        raise ValueError(
            "xG recency is not reaching "
            "final components."
        )

    # --------------------------------------------------------
    # NEARBY_3 keeps goal recency at CONTROL 0.95,
    # so its goal component should remain identical.
    #
    # Shot and xG should change.
    # --------------------------------------------------------

    nearby3_goal = changed_count(
        "nearby_3",
        "home_goal_attack_v5",
    )

    nearby3_shot = changed_count(
        "nearby_3",
        "home_shot_attack_v5",
    )

    nearby3_xg = changed_count(
        "nearby_3",
        "home_xg_attack",
    )

    if nearby3_goal != 0:

        raise ValueError(
            "NEARBY_3 goal component changed "
            "even though goal recency stayed 0.95."
        )

    if nearby3_shot == 0:

        raise ValueError(
            "NEARBY_3 shot component failed "
            "to change."
        )

    if nearby3_xg == 0:

        raise ValueError(
            "NEARBY_3 xG component failed "
            "to change."
        )

    print()
    print(
        "Signal-specific recency wiring "
        "validated ✅"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("CONFIRMING OPPONENT-ADJUSTED")
    print("SIGNAL RECENCY V5")
    print("==============================")
    print()

    print(
        "Frozen signal weights:"
    )

    print(
        f"Goals: "
        f"{GOAL_WEIGHT:.0%}"
    )

    print(
        f"xG: "
        f"{XG_WEIGHT:.0%}"
    )

    print(
        f"Shots: "
        f"{SHOT_WEIGHT:.0%}"
    )

    print(
        f"SOT: "
        f"{SOT_WEIGHT:.0%}"
    )

    print()

    print(
        f"Opponent strength: "
        f"{OPPONENT_STRENGTH:.3f}"
    )

    print(
        f"Overall / venue: "
        f"{OVERALL_WEIGHT:.0%} / "
        f"{VENUE_WEIGHT:.0%}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    team = load_team_data()

    xg = load_xg()

    xg_team = build_xg_team_rows(
        xg
    )

    print()
    print(
        f"Team rows: "
        f"{len(team):,}"
    )

    print(
        f"xG matches: "
        f"{len(xg):,}"
    )

    print(
        f"Settings to test: "
        f"{len(RECENCY_SETTINGS)}"
    )

    # ========================================================
    # RUN SETTINGS
    # ========================================================

    rows = []

    match_cache = {}

    for i, setting in enumerate(
        RECENCY_SETTINGS,
        start=1,
    ):

        print()
        print(
            f"[{i}/"
            f"{len(RECENCY_SETTINGS)}] "
            f"{setting['name']}"
        )

        print(
            f"  Goal: "
            f"{setting['goal_recency']:.3f}"
        )

        print(
            f"  xG: "
            f"{setting['xg_recency']:.3f}"
        )

        print(
            f"  Shot: "
            f"{setting['shot_recency']:.3f}"
        )

        matches = build_setting(
            team,
            xg_team,
            setting,
        )

        match_cache[
            setting[
                "name"
            ]
        ] = matches

        save_debug_components(
            matches,
            setting[
                "name"
            ],
        )

        tune = evaluate(
            matches,
            TUNING_SEASONS,
        )

        validation = evaluate(
            matches,
            VALIDATION_SEASONS,
        )

        final = evaluate(
            matches,
            FINAL_SEASONS,
        )

        rows.append(
            {
                "name":
                    setting[
                        "name"
                    ],

                "goal_recency":
                    setting[
                        "goal_recency"
                    ],

                "xg_recency":
                    setting[
                        "xg_recency"
                    ],

                "shot_recency":
                    setting[
                        "shot_recency"
                    ],

                "tuning_games":
                    tune[
                        "games"
                    ],

                "tuning_accuracy":
                    tune[
                        "accuracy"
                    ],

                "tuning_log_loss":
                    tune[
                        "log_loss"
                    ],

                "tuning_brier":
                    tune[
                        "brier"
                    ],

                "validation_games":
                    validation[
                        "games"
                    ],

                "validation_accuracy":
                    validation[
                        "accuracy"
                    ],

                "validation_log_loss":
                    validation[
                        "log_loss"
                    ],

                "validation_brier":
                    validation[
                        "brier"
                    ],

                "final_games":
                    final[
                        "games"
                    ],

                "final_accuracy":
                    final[
                        "accuracy"
                    ],

                "final_log_loss":
                    final[
                        "log_loss"
                    ],

                "final_brier":
                    final[
                        "brier"
                    ],
            }
        )

    results = pd.DataFrame(
        rows
    )

    # ========================================================
    # RANK USING TUNING ONLY
    # ========================================================

    results = (
        results
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

    results[
        "rank"
    ] = (
        np.arange(
            len(results)
        )
        + 1
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("==============================")
    print("RECENCY CONFIRMATION RESULTS")
    print("==============================")
    print()

    display = (
        results.copy()
    )

    for col in [
        "tuning_accuracy",
        "validation_accuracy",
        "final_accuracy",
    ]:

        display[
            col
        ] *= 100.0

    print(
        display[
            [
                "rank",
                "name",

                "goal_recency",
                "xg_recency",
                "shot_recency",

                "tuning_log_loss",
                "validation_log_loss",
                "final_log_loss",

                "tuning_brier",
                "validation_brier",
                "final_brier",

                "tuning_accuracy",
                "validation_accuracy",
                "final_accuracy",
            ]
        ]
        .round(6)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # CONTROL
    # ========================================================

    control_matches = (
        match_cache[
            "CONTROL"
        ]
    )

    control_tune = evaluate(
        control_matches,
        TUNING_SEASONS,
    )

    control_validation = evaluate(
        control_matches,
        VALIDATION_SEASONS,
    )

    control_final = evaluate(
        control_matches,
        FINAL_SEASONS,
    )

    # ========================================================
    # WINNER SELECTED USING TUNING ONLY
    # ========================================================

    winner_row = (
        results.iloc[
            0
        ]
    )

    winner_name = (
        winner_row[
            "name"
        ]
    )

    winner_matches = (
        match_cache[
            winner_name
        ]
    )

    winner_tune = evaluate(
        winner_matches,
        TUNING_SEASONS,
    )

    winner_validation = evaluate(
        winner_matches,
        VALIDATION_SEASONS,
    )

    winner_final = evaluate(
        winner_matches,
        FINAL_SEASONS,
    )

    print()
    print("==============================")
    print("SELECTED RECENCY SETTING")
    print("==============================")

    print(
        f"Name: "
        f"{winner_name}"
    )

    print(
        f"Goals: "
        f"{winner_row['goal_recency']:.3f}"
    )

    print(
        f"xG: "
        f"{winner_row['xg_recency']:.3f}"
    )

    print(
        f"Shots: "
        f"{winner_row['shot_recency']:.3f}"
    )

    # ========================================================
    # COMPARISONS
    # ========================================================

    print_comparison(
        "TUNING — 2021/22 TO 2022/23",
        control_tune,
        winner_tune,
    )

    print_comparison(
        "VALIDATION — 2023/24",
        control_validation,
        winner_validation,
    )

    print_comparison(
        "FINAL CHECK — 2024/25",
        control_final,
        winner_final,
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    # ========================================================
    # SAVE WINNING PREDICTIONS
    # ========================================================

    (
        home_lambda,
        away_lambda,
    ) = build_lambdas(
        winner_matches
    )

    valid = (
        home_lambda.notna()
        &
        away_lambda.notna()
    )

    output = (
        winner_matches.loc[
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
        "goal_weight_v5"
    ] = GOAL_WEIGHT

    output[
        "xg_weight_v5"
    ] = XG_WEIGHT

    output[
        "shot_weight_v5"
    ] = SHOT_WEIGHT

    output[
        "sot_weight_v5"
    ] = SOT_WEIGHT

    output[
        "goal_recency_v5"
    ] = float(
        winner_row[
            "goal_recency"
        ]
    )

    output[
        "xg_recency_v5"
    ] = float(
        winner_row[
            "xg_recency"
        ]
    )

    output[
        "shot_recency_v5"
    ] = float(
        winner_row[
            "shot_recency"
        ]
    )

    output[
        "opponent_strength_v5"
    ] = OPPONENT_STRENGTH

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # ========================================================
    # DEBUG
    # ========================================================

    print_debug_differences()

    validate_debug_behavior()

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print()
    print("==============================")
    print("VALIDATION")
    print("==============================")

    print(
        "Opponent strength frozen "
        "at 0.875 ✅"
    )

    print(
        "Signal weights frozen "
        "at 9% / 75% / 16% / 0% ✅"
    )

    print(
        "Winner selected using "
        "2021/22–2022/23 only ✅"
    )

    print(
        "2023/24 and 2024/25 "
        "not used for selection ✅"
    )

    print(
        "xG opponent adjustment uses "
        "pregame xG strength only ✅"
    )

    print(
        "Goal / shot / xG recency "
        "wiring explicitly validated ✅"
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