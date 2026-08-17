from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "team_pregame_stats.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "match_features.csv"
)

# How strongly we shrink small-sample ratings toward 1.00.
# Higher = more conservative.
OVERALL_SHRINK_K = 8.0
VENUE_SHRINK_K = 6.0

# If a team changes league, reduce confidence in its
# historical rating before using it in the new competition.
LEAGUE_CHANGE_CONFIDENCE = 0.70


# =========================================================
# SHRINKAGE HELPERS
# =========================================================

def shrink_rating(
    raw_rating,
    games,
    k,
    neutral=1.0,
):
    """
    Bayesian-style shrinkage toward neutral.

    Example:
        raw rating = 1.50
        only 2 games available

    becomes much closer to 1.00.

    With a large sample, the shrunk value approaches
    the raw value.
    """

    if pd.isna(raw_rating):
        return np.nan

    if pd.isna(games):
        games = 0

    games = max(float(games), 0.0)

    weight = games / (games + k)

    return (
        weight * float(raw_rating)
        + (1.0 - weight) * neutral
    )


def apply_series_shrinkage(
    ratings,
    games,
    k,
):
    weights = (
        games.astype(float)
        / (
            games.astype(float)
            + k
        )
    )

    return (
        weights * ratings
        + (1.0 - weights) * 1.0
    )


# =========================================================
# LEAGUE TRANSITION FEATURES
# =========================================================

def add_previous_league(df):
    """
    Finds the competition from each team's previous match.

    This lets us identify promotions, relegations or other
    competition changes without using future information.
    """

    df = df.copy()

    chronological = (
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
        .copy()
    )

    chronological[
        "previous_league_code"
    ] = (
        chronological
        .groupby("team")[
            "league_code"
        ]
        .shift(1)
    )

    chronological[
        "league_changed"
    ] = (
        chronological[
            "previous_league_code"
        ].notna()
        &
        (
            chronological[
                "previous_league_code"
            ]
            != chronological[
                "league_code"
            ]
        )
    ).astype(int)

    return chronological


def transition_adjust_rating(
    rating,
    changed,
):
    """
    When a club changes leagues, shrink its rating further
    toward neutral.

    Example:
        1.30 rating
        league change

    adjusted:
        1 + (1.30 - 1) * 0.70
        = 1.21
    """

    rating = rating.copy()

    mask = (
        changed == 1
    ) & rating.notna()

    rating.loc[mask] = (
        1.0
        + (
            rating.loc[mask]
            - 1.0
        )
        * LEAGUE_CHANGE_CONFIDENCE
    )

    return rating


# =========================================================
# PREPARE TEAM ROWS
# =========================================================

def prepare_team_rows(df):

    df = df.copy()

    # -----------------------------------------------------
    # SHRINK OVERALL ATTACK / DEFENSE
    # -----------------------------------------------------

    df[
        "shrunk_attack_strength"
    ] = apply_series_shrinkage(
        df[
            "pregame_attack_strength"
        ],
        df[
            "pregame_games"
        ],
        OVERALL_SHRINK_K,
    )

    df[
        "shrunk_defense_strength"
    ] = apply_series_shrinkage(
        df[
            "pregame_defense_strength"
        ],
        df[
            "pregame_games"
        ],
        OVERALL_SHRINK_K,
    )

    # -----------------------------------------------------
    # SHRINK VENUE RATINGS
    # -----------------------------------------------------

    df[
        "shrunk_venue_attack_strength"
    ] = apply_series_shrinkage(
        df[
            "pregame_venue_attack_strength"
        ],
        df[
            "pregame_venue_games"
        ],
        VENUE_SHRINK_K,
    )

    df[
        "shrunk_venue_defense_strength"
    ] = apply_series_shrinkage(
        df[
            "pregame_venue_defense_strength"
        ],
        df[
            "pregame_venue_games"
        ],
        VENUE_SHRINK_K,
    )

    # -----------------------------------------------------
    # EXTRA SHRINKAGE AFTER LEAGUE CHANGE
    # -----------------------------------------------------

    df[
        "transition_attack_strength"
    ] = transition_adjust_rating(
        df[
            "shrunk_attack_strength"
        ],
        df[
            "league_changed"
        ],
    )

    df[
        "transition_defense_strength"
    ] = transition_adjust_rating(
        df[
            "shrunk_defense_strength"
        ],
        df[
            "league_changed"
        ],
    )

    df[
        "transition_venue_attack_strength"
    ] = transition_adjust_rating(
        df[
            "shrunk_venue_attack_strength"
        ],
        df[
            "league_changed"
        ],
    )

    df[
        "transition_venue_defense_strength"
    ] = transition_adjust_rating(
        df[
            "shrunk_venue_defense_strength"
        ],
        df[
            "league_changed"
        ],
    )

    return df


# =========================================================
# BUILD HOME / AWAY TABLE
# =========================================================

def build_match_table(df):

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

    # -----------------------------------------------------
    # HOME COLUMNS
    # -----------------------------------------------------

    home_columns = {
        "team":
            "home_team",

        "opponent":
            "away_team_check",

        "pregame_games":
            "home_pregame_games",

        "pregame_venue_games":
            "home_pregame_venue_games",

        "days_rest":
            "home_days_rest",

        "league_changed":
            "home_league_changed",

        "previous_league_code":
            "home_previous_league_code",

        "pregame_ew_goals_for":
            "home_ew_goals_for",

        "pregame_ew_goals_against":
            "home_ew_goals_against",

        "pregame_ew_points":
            "home_ew_points",

        "pregame_ew_shots_for":
            "home_ew_shots_for",

        "pregame_ew_shots_against":
            "home_ew_shots_against",

        "pregame_ew_shots_on_target_for":
            "home_ew_sot_for",

        "pregame_ew_shots_on_target_against":
            "home_ew_sot_against",

        "pregame_attack_strength":
            "home_attack_strength_raw",

        "pregame_defense_strength":
            "home_defense_strength_raw",

        "pregame_venue_attack_strength":
            "home_venue_attack_strength_raw",

        "pregame_venue_defense_strength":
            "home_venue_defense_strength_raw",

        "transition_attack_strength":
            "home_attack_strength",

        "transition_defense_strength":
            "home_defense_strength",

        "transition_venue_attack_strength":
            "home_venue_attack_strength",

        "transition_venue_defense_strength":
            "home_venue_defense_strength",
    }

    keep_home = [
        "match_id",
        "date",
        "season",
        "league_code",
        "league",

        "league_prior_matches",
        "league_avg_home_goals",
        "league_avg_away_goals",
        "league_avg_total_goals",
        "league_avg_team_goals",

        "goals_for",
        "goals_against",
    ] + list(home_columns.keys())

    home = home[
        keep_home
    ].rename(
        columns={
            **home_columns,

            "goals_for":
                "home_goals",

            "goals_against":
                "away_goals",
        }
    )

    # -----------------------------------------------------
    # AWAY COLUMNS
    # -----------------------------------------------------

    away_columns = {
        "team":
            "away_team",

        "opponent":
            "home_team_check",

        "pregame_games":
            "away_pregame_games",

        "pregame_venue_games":
            "away_pregame_venue_games",

        "days_rest":
            "away_days_rest",

        "league_changed":
            "away_league_changed",

        "previous_league_code":
            "away_previous_league_code",

        "pregame_ew_goals_for":
            "away_ew_goals_for",

        "pregame_ew_goals_against":
            "away_ew_goals_against",

        "pregame_ew_points":
            "away_ew_points",

        "pregame_ew_shots_for":
            "away_ew_shots_for",

        "pregame_ew_shots_against":
            "away_ew_shots_against",

        "pregame_ew_shots_on_target_for":
            "away_ew_sot_for",

        "pregame_ew_shots_on_target_against":
            "away_ew_sot_against",

        "pregame_attack_strength":
            "away_attack_strength_raw",

        "pregame_defense_strength":
            "away_defense_strength_raw",

        "pregame_venue_attack_strength":
            "away_venue_attack_strength_raw",

        "pregame_venue_defense_strength":
            "away_venue_defense_strength_raw",

        "transition_attack_strength":
            "away_attack_strength",

        "transition_defense_strength":
            "away_defense_strength",

        "transition_venue_attack_strength":
            "away_venue_attack_strength",

        "transition_venue_defense_strength":
            "away_venue_defense_strength",
    }

    keep_away = [
        "match_id",
    ] + list(
        away_columns.keys()
    )

    away = away[
        keep_away
    ].rename(
        columns=away_columns
    )

    # -----------------------------------------------------
    # MERGE
    # -----------------------------------------------------

    matches = home.merge(
        away,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    return matches


# =========================================================
# MODEL INPUT FEATURES
# =========================================================

def add_relative_features(df):

    df = df.copy()

    # -----------------------------------------------------
    # COMBINED ATTACK / DEFENSE
    #
    # Blend overall and venue-specific information.
    # -----------------------------------------------------

    df[
        "home_attack_blend"
    ] = (
        0.50
        * df[
            "home_attack_strength"
        ]
        +
        0.50
        * df[
            "home_venue_attack_strength"
        ]
    )

    df[
        "home_defense_blend"
    ] = (
        0.50
        * df[
            "home_defense_strength"
        ]
        +
        0.50
        * df[
            "home_venue_defense_strength"
        ]
    )

    df[
        "away_attack_blend"
    ] = (
        0.50
        * df[
            "away_attack_strength"
        ]
        +
        0.50
        * df[
            "away_venue_attack_strength"
        ]
    )

    df[
        "away_defense_blend"
    ] = (
        0.50
        * df[
            "away_defense_strength"
        ]
        +
        0.50
        * df[
            "away_venue_defense_strength"
        ]
    )

    # -----------------------------------------------------
    # SIMPLE EXPECTED-GOAL INPUTS
    #
    # These are NOT our final predictions yet.
    # They become Baseline Model 0.
    # -----------------------------------------------------

    df[
        "baseline_home_xg"
    ] = (
        df[
            "league_avg_home_goals"
        ]
        *
        df[
            "home_attack_blend"
        ]
        *
        df[
            "away_defense_blend"
        ]
    )

    df[
        "baseline_away_xg"
    ] = (
        df[
            "league_avg_away_goals"
        ]
        *
        df[
            "away_attack_blend"
        ]
        *
        df[
            "home_defense_blend"
        ]
    )

    # Avoid pathological values from tiny samples.
    df[
        "baseline_home_xg"
    ] = df[
        "baseline_home_xg"
    ].clip(
        lower=0.15,
        upper=4.50,
    )

    df[
        "baseline_away_xg"
    ] = df[
        "baseline_away_xg"
    ].clip(
        lower=0.15,
        upper=4.50,
    )

    df[
        "baseline_total_xg"
    ] = (
        df[
            "baseline_home_xg"
        ]
        +
        df[
            "baseline_away_xg"
        ]
    )

    # -----------------------------------------------------
    # RELATIVE DIFFERENCES
    # -----------------------------------------------------

    df[
        "attack_strength_diff"
    ] = (
        df[
            "home_attack_blend"
        ]
        -
        df[
            "away_attack_blend"
        ]
    )

    df[
        "defense_strength_diff"
    ] = (
        df[
            "away_defense_blend"
        ]
        -
        df[
            "home_defense_blend"
        ]
    )

    df[
        "ew_points_diff"
    ] = (
        df[
            "home_ew_points"
        ]
        -
        df[
            "away_ew_points"
        ]
    )

    df[
        "rest_diff"
    ] = (
        df[
            "home_days_rest"
        ]
        -
        df[
            "away_days_rest"
        ]
    )

    return df


# =========================================================
# VALIDATION
# =========================================================

def validate_matches(df):

    if df["match_id"].duplicated().any():
        raise ValueError(
            "Duplicate match IDs found."
        )

    mismatch_home = (
        df["home_team"]
        != df["home_team_check"]
    )

    mismatch_away = (
        df["away_team"]
        != df["away_team_check"]
    )

    if mismatch_home.any():
        raise ValueError(
            "Home team/opponent merge mismatch."
        )

    if mismatch_away.any():
        raise ValueError(
            "Away team/opponent merge mismatch."
        )

    impossible_goals = (
        df["home_goals"].isna()
        |
        df["away_goals"].isna()
    )

    if impossible_goals.any():
        raise ValueError(
            "Missing match results after merge."
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("==============================")
    print("BUILDING MATCH FEATURES")
    print("==============================")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{INPUT_FILE}"
        )

    team_data = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    print(
        f"Team rows loaded: "
        f"{len(team_data):,}"
    )

    print(
        "Detecting league transitions..."
    )

    team_data = add_previous_league(
        team_data
    )

    print(
        "Applying sample-size shrinkage..."
    )

    team_data = prepare_team_rows(
        team_data
    )

    print(
        "Combining home and away rows..."
    )

    match_features = build_match_table(
        team_data
    )

    print(
        "Building relative model features..."
    )

    match_features = add_relative_features(
        match_features
    )

    print(
        "Running validation checks..."
    )

    validate_matches(
        match_features
    )

    match_features = (
        match_features
        .sort_values(
            [
                "date",
                "league_code",
                "match_id",
            ]
        )
        .reset_index(drop=True)
    )

    match_features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("==============================")
    print("MATCH FEATURES COMPLETE")
    print("==============================")

    print(
        f"Matches: "
        f"{len(match_features):,}"
    )

    print(
        f"Unique match IDs: "
        f"{match_features['match_id'].nunique():,}"
    )

    print()
    print("LEAGUE TRANSITIONS")

    transition_summary = (
        match_features
        .groupby("league")
        .agg(
            home_transitions=(
                "home_league_changed",
                "sum",
            ),
            away_transitions=(
                "away_league_changed",
                "sum",
            ),
        )
    )

    print(
        transition_summary.to_string()
    )

    print()
    print("BASELINE EXPECTED GOALS")

    xg_summary = (
        match_features
        .groupby("league")
        .agg(
            actual_home_goals=(
                "home_goals",
                "mean",
            ),
            baseline_home_xg=(
                "baseline_home_xg",
                "mean",
            ),
            actual_away_goals=(
                "away_goals",
                "mean",
            ),
            baseline_away_xg=(
                "baseline_away_xg",
                "mean",
            ),
            baseline_total_xg=(
                "baseline_total_xg",
                "mean",
            ),
        )
        .round(3)
    )

    print(
        xg_summary.to_string()
    )

    print()
    print("SAMPLE MATCHES")

    sample_columns = [
        "date",
        "league",
        "home_team",
        "away_team",
        "home_attack_blend",
        "away_attack_blend",
        "home_defense_blend",
        "away_defense_blend",
        "baseline_home_xg",
        "baseline_away_xg",
        "home_goals",
        "away_goals",
    ]

    sample = (
        match_features
        .dropna(
            subset=[
                "baseline_home_xg",
                "baseline_away_xg",
            ]
        )
        .tail(10)[
            sample_columns
        ]
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
    ] = sample[
        numeric_cols
    ].round(3)

    print(
        sample.to_string(
            index=False
        )
    )

    print()
    print(
        "Validation: "
        "One row per match ✅"
    )

    print(
        "Validation: "
        "Home/away merge aligned ✅"
    )

    print()
    print(
        f"Saved:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()