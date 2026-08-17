from pathlib import Path

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

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "opponent_adjusted_v3_features.csv"
)


# =========================================================
# SETTINGS
# =========================================================

RECENCY = 0.95

MIN_PRIOR_GAMES = 5

EPS = 1e-9

# Controls how aggressively opponent adjustment feeds back.
# 0 = no adjustment
# 1 = full adjustment
OPPONENT_ADJUSTMENT_STRENGTH = 0.50


# =========================================================
# EXPONENTIALLY WEIGHTED PRIOR AVERAGE
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
# BASIC TEAM HISTORY
# =========================================================

def build_basic_history(
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
    ] = (
        team_group.cumcount()
    )

    for col in [
        "goals_for",
        "goals_against",
        "shots_for",
        "shots_against",
        "shots_on_target_for",
        "shots_on_target_against",
    ]:

        df[
            f"ew_{col}"
        ] = (
            team_group[
                col
            ]
            .transform(
                weighted_prior_average
            )
        )

    return df


# =========================================================
# LEAGUE BASELINES
# =========================================================

def build_league_baselines(
    df,
):
    """
    Same-day-safe league averages.

    IMPORTANT:
    Goals, shots, and shots on target each use their
    own valid-data denominator.

    Matches with missing shot/SOT data therefore do not
    dilute those league baselines toward zero.
    """

    work = df.copy()

    # -----------------------------------------------------
    # VALID OBSERVATION FLAGS
    # -----------------------------------------------------

    work[
        "goal_valid"
    ] = (
        work[
            "goals_for"
        ].notna()
    ).astype(int)

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

    # -----------------------------------------------------
    # DAILY LEAGUE TOTALS
    # -----------------------------------------------------

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
            goal_obs=(
                "goal_valid",
                "sum",
            ),

            shot_obs=(
                "shot_valid",
                "sum",
            ),

            sot_obs=(
                "sot_valid",
                "sum",
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

    # -----------------------------------------------------
    # HOME / AWAY
    # -----------------------------------------------------

    home = (
        daily[
            daily[
                "venue"
            ] == "HOME"
        ]
        .copy()
        .rename(
            columns={
                "goal_obs":
                    "home_goal_obs",

                "shot_obs":
                    "home_shot_obs",

                "sot_obs":
                    "home_sot_obs",

                "goals":
                    "home_goals",

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
            ] == "AWAY"
        ]
        .copy()
        .rename(
            columns={
                "goal_obs":
                    "away_goal_obs",

                "shot_obs":
                    "away_shot_obs",

                "sot_obs":
                    "away_sot_obs",

                "goals":
                    "away_goals",

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

            "home_goal_obs",
            "home_shot_obs",
            "home_sot_obs",

            "home_goals",
            "home_shots",
            "home_sot",
        ]
    ].merge(
        away[
            [
                "league_code",
                "date",

                "away_goal_obs",
                "away_shot_obs",
                "away_sot_obs",

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

    group = (
        daily_match
        .groupby(
            "league_code"
        )
    )

    # -----------------------------------------------------
    # PRIOR TOTALS — SAME DAY EXCLUDED
    # -----------------------------------------------------

    cumulative_columns = [
        "home_goal_obs",
        "away_goal_obs",

        "home_shot_obs",
        "away_shot_obs",

        "home_sot_obs",
        "away_sot_obs",

        "home_goals",
        "away_goals",

        "home_shots",
        "away_shots",

        "home_sot",
        "away_sot",
    ]

    for col in cumulative_columns:

        daily_match[
            f"prior_{col}"
        ] = (
            group[
                col
            ].cumsum()
            -
            daily_match[
                col
            ]
        )

    # -----------------------------------------------------
    # GOAL BASELINES
    # -----------------------------------------------------

    daily_match[
        "lg_home_goals"
    ] = (
        daily_match[
            "prior_home_goals"
        ]
        /
        daily_match[
            "prior_home_goal_obs"
        ].replace(
            0,
            np.nan,
        )
    )

    daily_match[
        "lg_away_goals"
    ] = (
        daily_match[
            "prior_away_goals"
        ]
        /
        daily_match[
            "prior_away_goal_obs"
        ].replace(
            0,
            np.nan,
        )
    )

    total_goal_obs = (
        daily_match[
            "prior_home_goal_obs"
        ]
        +
        daily_match[
            "prior_away_goal_obs"
        ]
    )

    daily_match[
        "lg_team_goals"
    ] = (
        (
            daily_match[
                "prior_home_goals"
            ]
            +
            daily_match[
                "prior_away_goals"
            ]
        )
        /
        total_goal_obs.replace(
            0,
            np.nan,
        )
    )

    # -----------------------------------------------------
    # SHOT BASELINE
    # -----------------------------------------------------

    total_shot_obs = (
        daily_match[
            "prior_home_shot_obs"
        ]
        +
        daily_match[
            "prior_away_shot_obs"
        ]
    )

    daily_match[
        "lg_team_shots"
    ] = (
        (
            daily_match[
                "prior_home_shots"
            ]
            +
            daily_match[
                "prior_away_shots"
            ]
        )
        /
        total_shot_obs.replace(
            0,
            np.nan,
        )
    )

    # -----------------------------------------------------
    # SHOTS-ON-TARGET BASELINE
    # -----------------------------------------------------

    total_sot_obs = (
        daily_match[
            "prior_home_sot_obs"
        ]
        +
        daily_match[
            "prior_away_sot_obs"
        ]
    )

    daily_match[
        "lg_team_sot"
    ] = (
        (
            daily_match[
                "prior_home_sot"
            ]
            +
            daily_match[
                "prior_away_sot"
            ]
        )
        /
        total_sot_obs.replace(
            0,
            np.nan,
        )
    )

    return daily_match[
        [
            "league_code",
            "date",

            "lg_home_goals",
            "lg_away_goals",

            "lg_team_goals",
            "lg_team_shots",
            "lg_team_sot",
        ]
    ]


# =========================================================
# RAW RELATIVE STRENGTHS
# =========================================================

def add_raw_strengths(
    df,
):
    df = df.copy()

    df[
        "raw_attack_goals"
    ] = (
        df[
            "ew_goals_for"
        ]
        /
        df[
            "lg_team_goals"
        ]
    )

    df[
        "raw_defense_goals"
    ] = (
        df[
            "ew_goals_against"
        ]
        /
        df[
            "lg_team_goals"
        ]
    )

    df[
        "raw_attack_shots"
    ] = (
        df[
            "ew_shots_for"
        ]
        /
        df[
            "lg_team_shots"
        ]
    )

    df[
        "raw_defense_shots"
    ] = (
        df[
            "ew_shots_against"
        ]
        /
        df[
            "lg_team_shots"
        ]
    )

    df[
        "raw_attack_sot"
    ] = (
        df[
            "ew_shots_on_target_for"
        ]
        /
        df[
            "lg_team_sot"
        ]
    )

    df[
        "raw_defense_sot"
    ] = (
        df[
            "ew_shots_on_target_against"
        ]
        /
        df[
            "lg_team_sot"
        ]
    )

    return df


# =========================================================
# OPPONENT SNAPSHOT
# =========================================================

def attach_opponent_pregame_strength(
    df,
):
    """
    Joins each team's row to the opponent's PREMATCH
    strength from the exact same match.

    This is safe because all raw strength columns were
    themselves built only from earlier matches.
    """

    opponent = df[
        [
            "match_id",
            "team",

            "raw_attack_goals",
            "raw_defense_goals",

            "raw_attack_shots",
            "raw_defense_shots",

            "raw_attack_sot",
            "raw_defense_sot",
        ]
    ].copy()

    opponent = opponent.rename(
        columns={
            "team":
                "opponent_check",

            "raw_attack_goals":
                "opp_attack_goals",

            "raw_defense_goals":
                "opp_defense_goals",

            "raw_attack_shots":
                "opp_attack_shots",

            "raw_defense_shots":
                "opp_defense_shots",

            "raw_attack_sot":
                "opp_attack_sot",

            "raw_defense_sot":
                "opp_defense_sot",
        }
    )

    merged = df.merge(
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

    return merged


# =========================================================
# GAME-LEVEL OPPONENT-ADJUSTED PERFORMANCE
# =========================================================

def add_adjusted_game_performance(
    df,
):
    df = df.copy()

    strength = (
        OPPONENT_ADJUSTMENT_STRENGTH
    )

    # -----------------------------------------------------
    # ATTACK
    #
    # If opponent defense is weak (>1.0 means allows more),
    # discount offensive production.
    #
    # If opponent defense is strong (<1.0), boost it.
    # -----------------------------------------------------

    df[
        "adj_goals_for"
    ] = (
        df[
            "goals_for"
        ]
        /
        (
            (
                1.0
                - strength
            )
            +
            strength
            * df[
                "opp_defense_goals"
            ]
        )
    )

    df[
        "adj_shots_for"
    ] = (
        df[
            "shots_for"
        ]
        /
        (
            (
                1.0
                - strength
            )
            +
            strength
            * df[
                "opp_defense_shots"
            ]
        )
    )

    df[
        "adj_sot_for"
    ] = (
        df[
            "shots_on_target_for"
        ]
        /
        (
            (
                1.0
                - strength
            )
            +
            strength
            * df[
                "opp_defense_sot"
            ]
        )
    )

    # -----------------------------------------------------
    # DEFENSE
    #
    # Conceding to an elite attack should be penalized less.
    # Conceding to a weak attack should be penalized more.
    # -----------------------------------------------------

    df[
        "adj_goals_against"
    ] = (
        df[
            "goals_against"
        ]
        /
        (
            (
                1.0
                - strength
            )
            +
            strength
            * df[
                "opp_attack_goals"
            ]
        )
    )

    df[
        "adj_shots_against"
    ] = (
        df[
            "shots_against"
        ]
        /
        (
            (
                1.0
                - strength
            )
            +
            strength
            * df[
                "opp_attack_shots"
            ]
        )
    )

    df[
        "adj_sot_against"
    ] = (
        df[
            "shots_on_target_against"
        ]
        /
        (
            (
                1.0
                - strength
            )
            +
            strength
            * df[
                "opp_attack_sot"
            ]
        )
    )

    return df


# =========================================================
# ROLL OPPONENT-ADJUSTED PERFORMANCE
# =========================================================

def add_adjusted_history(
    df,
):
    df = df.copy()

    team_group = df.groupby(
        "team",
        sort=False,
        group_keys=False,
    )

    adjusted_cols = [
        "adj_goals_for",
        "adj_goals_against",
        "adj_shots_for",
        "adj_shots_against",
        "adj_sot_for",
        "adj_sot_against",
    ]

    for col in adjusted_cols:

        df[
            f"ew_{col}"
        ] = (
            team_group[
                col
            ]
            .transform(
                weighted_prior_average
            )
        )

    return df


# =========================================================
# FINAL V3 STRENGTH FEATURES
# =========================================================

def add_v3_strengths(
    df,
):
    df = df.copy()

    df[
        "v3_attack_goals"
    ] = (
        df[
            "ew_adj_goals_for"
        ]
        /
        df[
            "lg_team_goals"
        ]
    )

    df[
        "v3_defense_goals"
    ] = (
        df[
            "ew_adj_goals_against"
        ]
        /
        df[
            "lg_team_goals"
        ]
    )

    df[
        "v3_attack_shots"
    ] = (
        df[
            "ew_adj_shots_for"
        ]
        /
        df[
            "lg_team_shots"
        ]
    )

    df[
        "v3_defense_shots"
    ] = (
        df[
            "ew_adj_shots_against"
        ]
        /
        df[
            "lg_team_shots"
        ]
    )

    df[
        "v3_attack_sot"
    ] = (
        df[
            "ew_adj_sot_for"
        ]
        /
        df[
            "lg_team_sot"
        ]
    )

    df[
        "v3_defense_sot"
    ] = (
        df[
            "ew_adj_sot_against"
        ]
        /
        df[
            "lg_team_sot"
        ]
    )

    return df


# =========================================================
# VALIDATION
# =========================================================

def validate(
    df,
):

    first_games = (
        df[
            "pregame_games"
        ]
        == 0
    )

    leakage_cols = [
        "ew_goals_for",
        "ew_goals_against",
        "ew_adj_goals_for",
        "ew_adj_goals_against",
    ]

    for col in leakage_cols:

        if (
            df.loc[
                first_games,
                col,
            ]
            .notna()
            .any()
        ):

            raise ValueError(
                f"Leakage detected in {col}"
            )

    match_counts = (
        df[
            "match_id"
        ]
        .value_counts()
    )

    if (
        match_counts
        != 2
    ).any():

        raise ValueError(
            "Not every match has exactly "
            "two team rows."
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("==============================")
    print("BUILDING OPPONENT-ADJUSTED V3")
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
        f"Team-game rows loaded: "
        f"{len(df):,}"
    )

    # =====================================================
    # LEAGUE BASELINES
    # =====================================================

    print(
        "Building leakage-safe league baselines..."
    )

    league = build_league_baselines(
        df
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

    # =====================================================
    # BASIC HISTORY
    # =====================================================

    print(
        "Building raw pregame team strengths..."
    )

    df = build_basic_history(
        df
    )

    df = add_raw_strengths(
        df
    )

    # =====================================================
    # OPPONENT SNAPSHOTS
    # =====================================================

    print(
        "Attaching opponent pregame strengths..."
    )

    df = attach_opponent_pregame_strength(
        df
    )

    # =====================================================
    # ADJUST EACH GAME
    # =====================================================

    print(
        "Adjusting historical performance "
        "for opponent quality..."
    )

    df = add_adjusted_game_performance(
        df
    )

    # =====================================================
    # ROLL ADJUSTED HISTORY
    # =====================================================

    print(
        "Building weighted opponent-adjusted history..."
    )

    df = add_adjusted_history(
        df
    )

    df = add_v3_strengths(
        df
    )

    # =====================================================
    # VALIDATE
    # =====================================================

    print(
        "Running leakage and integrity checks..."
    )

    validate(
        df
    )

    # =====================================================
    # SAVE
    # =====================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("==============================")
    print("OPPONENT-ADJUSTED V3 COMPLETE")
    print("==============================")

    print(
        f"Rows: "
        f"{len(df):,}"
    )

    print(
        f"Unique matches: "
        f"{df['match_id'].nunique():,}"
    )

    print(
        f"Unique teams: "
        f"{df['team'].nunique():,}"
    )

    usable = df[
        df[
            "pregame_games"
        ]
        >= MIN_PRIOR_GAMES
    ]

    print(
        f"Rows with >= "
        f"{MIN_PRIOR_GAMES} prior games: "
        f"{len(usable):,}"
    )

    print()
    print(
        "RAW VS OPPONENT-ADJUSTED "
        "STRENGTH DISPERSION"
    )

    summary = pd.DataFrame(
        {
            "feature": [
                "goal attack",
                "goal defense",
                "shot attack",
                "shot defense",
                "SOT attack",
                "SOT defense",
            ],

            "raw_std": [
                usable[
                    "raw_attack_goals"
                ].std(),

                usable[
                    "raw_defense_goals"
                ].std(),

                usable[
                    "raw_attack_shots"
                ].std(),

                usable[
                    "raw_defense_shots"
                ].std(),

                usable[
                    "raw_attack_sot"
                ].std(),

                usable[
                    "raw_defense_sot"
                ].std(),
            ],

            "adjusted_std": [
                usable[
                    "v3_attack_goals"
                ].std(),

                usable[
                    "v3_defense_goals"
                ].std(),

                usable[
                    "v3_attack_shots"
                ].std(),

                usable[
                    "v3_defense_shots"
                ].std(),

                usable[
                    "v3_attack_sot"
                ].std(),

                usable[
                    "v3_defense_sot"
                ].std(),
            ],
        }
    )

    print(
        summary
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "SAMPLE V3 RATINGS"
    )

    sample = (
        usable
        .dropna(
            subset=[
                "v3_attack_goals",
                "v3_defense_goals",
            ]
        )
        .tail(12)
        [
            [
                "date",
                "league",
                "team",
                "opponent",
                "raw_attack_goals",
                "v3_attack_goals",
                "raw_defense_goals",
                "v3_defense_goals",
                "raw_attack_sot",
                "v3_attack_sot",
            ]
        ]
        .copy()
    )

    numeric_cols = (
        sample
        .select_dtypes(
            include="number"
        )
        .columns
    )

    sample[
        numeric_cols
    ] = (
        sample[
            numeric_cols
        ]
        .round(3)
    )

    print(
        sample.to_string(
            index=False
        )
    )

    print()
    print(
        "First-game histories contain no "
        "current-match information ✅"
    )

    print(
        "Opponent strength used for a match "
        "comes only from pregame history ✅"
    )

    print()
    print(
        f"Saved:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()