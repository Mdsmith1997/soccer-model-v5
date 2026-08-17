from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================

RECENCY_WEIGHT = 0.90

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
    / "team_pregame_stats.csv"
)


# =========================================================
# FEATURE COLUMNS
# =========================================================

ROLLING_STATS = [
    "goals_for",
    "goals_against",
    "goal_difference",
    "points",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "corners_for",
    "corners_against",
    "win",
    "draw",
    "loss",
    "scored",
    "clean_sheet",
    "over_2_5",
    "btts",
]


# =========================================================
# EXPONENTIALLY WEIGHTED PRIOR AVERAGE
# =========================================================

def weighted_prior_average(values, decay=RECENCY_WEIGHT):
    """
    For every row, calculate an exponentially weighted
    average using ONLY rows before the current row.

    Current match is never included.
    """

    results = []

    numerator = 0.0
    denominator = 0.0

    for value in values:

        if denominator > 0:
            results.append(
                numerator / denominator
            )
        else:
            results.append(np.nan)

        # Age all previous information.
        numerator *= decay
        denominator *= decay

        if pd.notna(value):
            numerator += float(value)
            denominator += 1.0

    return pd.Series(
        results,
        index=values.index,
        dtype="float64",
    )


# =========================================================
# TEAM FEATURES
# =========================================================

def build_team_history_features(df):
    """
    Creates historical features using all previous matches
    played by each team.

    Team history carries across seasons.
    """

    df = df.copy()

    grouped = df.groupby(
        "team",
        sort=False,
        group_keys=False,
    )

    # Number of prior games available.
    df["pregame_games"] = grouped.cumcount()

    # -----------------------------------------------------
    # DAYS REST
    # -----------------------------------------------------

    previous_date = grouped["date"].shift(1)

    df["days_rest"] = (
        df["date"] - previous_date
    ).dt.days

    # -----------------------------------------------------
    # EXPONENTIALLY WEIGHTED OVERALL FORM
    # -----------------------------------------------------

    for column in ROLLING_STATS:

        if column not in df.columns:
            continue

        feature_name = f"pregame_ew_{column}"

        df[feature_name] = grouped[
            column
        ].transform(
            weighted_prior_average
        )

    return df


# =========================================================
# VENUE-SPECIFIC FEATURES
# =========================================================

def build_venue_features(df):
    """
    Separately tracks how teams perform at HOME and AWAY.

    Example:
    Arsenal's home attack rating uses only Arsenal's
    previous home matches.
    """

    df = df.copy()

    grouped = df.groupby(
        [
            "team",
            "venue",
        ],
        sort=False,
        group_keys=False,
    )

    df["pregame_venue_games"] = (
        grouped.cumcount()
    )

    venue_columns = [
        "goals_for",
        "goals_against",
        "goal_difference",
        "points",
        "shots_for",
        "shots_against",
        "shots_on_target_for",
        "shots_on_target_against",
        "win",
        "clean_sheet",
        "over_2_5",
        "btts",
    ]

    for column in venue_columns:

        if column not in df.columns:
            continue

        feature_name = (
            f"pregame_venue_ew_{column}"
        )

        df[feature_name] = grouped[
            column
        ].transform(
            weighted_prior_average
        )

    return df


# =========================================================
# LEAGUE HISTORICAL BASELINES
# =========================================================

def build_league_baselines(team_games):
    """
    Creates league scoring baselines based ONLY on matches
    before the current date.

    This is done at the MATCH level, not team-row level,
    so every match contributes exactly once.

    Same-day matches receive the same pregame league
    baseline.
    """

    match_level = (
        team_games[
            [
                "match_id",
                "date",
                "league_code",
                "league",
                "venue",
                "goals_for",
            ]
        ]
        .copy()
    )

    home = (
        match_level[
            match_level["venue"] == "HOME"
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
                "goals_for": "home_goals"
            }
        )
    )

    away = (
        match_level[
            match_level["venue"] == "AWAY"
        ][
            [
                "match_id",
                "goals_for",
            ]
        ]
        .rename(
            columns={
                "goals_for": "away_goals"
            }
        )
    )

    matches = home.merge(
        away,
        on="match_id",
        how="inner",
    )

    matches = matches.sort_values(
        [
            "date",
            "league_code",
            "match_id",
        ]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # DAILY TOTALS
    #
    # Using date-level aggregation ensures matches played
    # on the same day cannot see each other's results.
    # -----------------------------------------------------

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

    # Previous cumulative totals only.
    daily["league_prior_matches"] = (
        daily
        .groupby("league_code")[
            "daily_matches"
        ]
        .cumsum()
        - daily["daily_matches"]
    )

    daily["league_prior_home_goals"] = (
        daily
        .groupby("league_code")[
            "daily_home_goals"
        ]
        .cumsum()
        - daily["daily_home_goals"]
    )

    daily["league_prior_away_goals"] = (
        daily
        .groupby("league_code")[
            "daily_away_goals"
        ]
        .cumsum()
        - daily["daily_away_goals"]
    )

    daily["league_avg_home_goals"] = (
        daily["league_prior_home_goals"]
        / daily["league_prior_matches"]
    )

    daily["league_avg_away_goals"] = (
        daily["league_prior_away_goals"]
        / daily["league_prior_matches"]
    )

    daily["league_avg_total_goals"] = (
        daily["league_avg_home_goals"]
        + daily["league_avg_away_goals"]
    )

    daily["league_avg_team_goals"] = (
        daily["league_avg_total_goals"]
        / 2.0
    )

    baseline_columns = [
        "league_code",
        "date",
        "league_prior_matches",
        "league_avg_home_goals",
        "league_avg_away_goals",
        "league_avg_total_goals",
        "league_avg_team_goals",
    ]

    return daily[baseline_columns]


# =========================================================
# ATTACK / DEFENSE STRENGTH
# =========================================================

def build_strength_features(df):
    """
    Converts raw historical scoring averages into relative
    attack and defense strengths.

    1.00 = approximately league average
    >1.00 attack = stronger attack
    >1.00 defense number = CONCEDES more than average
                         (therefore worse defense)
    <1.00 defense number = better defense
    """

    df = df.copy()

    # -----------------------------------------------------
    # OVERALL ATTACK / DEFENSE
    # -----------------------------------------------------

    df["pregame_attack_strength"] = (
        df["pregame_ew_goals_for"]
        / df["league_avg_team_goals"]
    )

    df["pregame_defense_strength"] = (
        df["pregame_ew_goals_against"]
        / df["league_avg_team_goals"]
    )

    # -----------------------------------------------------
    # VENUE-SPECIFIC ATTACK / DEFENSE
    # -----------------------------------------------------

    home_mask = (
        df["venue"] == "HOME"
    )

    away_mask = (
        df["venue"] == "AWAY"
    )

    # Home teams score relative to league home scoring.
    df.loc[
        home_mask,
        "pregame_venue_attack_strength",
    ] = (
        df.loc[
            home_mask,
            "pregame_venue_ew_goals_for",
        ]
        / df.loc[
            home_mask,
            "league_avg_home_goals",
        ]
    )

    # Home defenses concede relative to normal away scoring.
    df.loc[
        home_mask,
        "pregame_venue_defense_strength",
    ] = (
        df.loc[
            home_mask,
            "pregame_venue_ew_goals_against",
        ]
        / df.loc[
            home_mask,
            "league_avg_away_goals",
        ]
    )

    # Away teams score relative to league away scoring.
    df.loc[
        away_mask,
        "pregame_venue_attack_strength",
    ] = (
        df.loc[
            away_mask,
            "pregame_venue_ew_goals_for",
        ]
        / df.loc[
            away_mask,
            "league_avg_away_goals",
        ]
    )

    # Away defenses concede relative to normal home scoring.
    df.loc[
        away_mask,
        "pregame_venue_defense_strength",
    ] = (
        df.loc[
            away_mask,
            "pregame_venue_ew_goals_against",
        ]
        / df.loc[
            away_mask,
            "league_avg_home_goals",
        ]
    )

    return df


# =========================================================
# VALIDATION
# =========================================================

def validate_no_current_match_leakage(df):
    """
    Basic leakage tests.

    First-ever game for a team must have no previous
    weighted statistics.
    """

    first_games = (
        df["pregame_games"] == 0
    )

    leakage_columns = [
        "pregame_ew_goals_for",
        "pregame_ew_goals_against",
        "pregame_ew_points",
    ]

    for column in leakage_columns:

        if column not in df.columns:
            continue

        leaked = (
            df.loc[
                first_games,
                column,
            ]
            .notna()
            .any()
        )

        if leaked:
            raise ValueError(
                f"Leakage detected in {column}"
            )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("==============================")
    print("BUILDING PREGAME FEATURES")
    print("==============================")
    print()

    print(
        f"Recency weight: "
        f"{RECENCY_WEIGHT}"
    )

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Could not find:\n{INPUT_FILE}"
        )

    team_games = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    print(
        f"Team-game rows loaded: "
        f"{len(team_games):,}"
    )

    # Important chronological ordering.
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
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # LEAGUE BASELINES
    # -----------------------------------------------------

    print(
        "Building historical "
        "league scoring baselines..."
    )

    league_baselines = (
        build_league_baselines(
            team_games
        )
    )

    team_games = team_games.merge(
        league_baselines,
        on=[
            "league_code",
            "date",
        ],
        how="left",
        validate="many_to_one",
    )

    # -----------------------------------------------------
    # TEAM HISTORY
    # -----------------------------------------------------

    print(
        "Building overall "
        "team form features..."
    )

    team_games = (
        build_team_history_features(
            team_games
        )
    )

    # -----------------------------------------------------
    # HOME / AWAY HISTORY
    # -----------------------------------------------------

    print(
        "Building venue-specific "
        "team features..."
    )

    team_games = (
        build_venue_features(
            team_games
        )
    )

    # -----------------------------------------------------
    # STRENGTH RATINGS
    # -----------------------------------------------------

    print(
        "Building attack and "
        "defense strengths..."
    )

    team_games = (
        build_strength_features(
            team_games
        )
    )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    print(
        "Checking for "
        "current-match leakage..."
    )

    validate_no_current_match_leakage(
        team_games
    )

    # Every original team-game row must remain.
    if len(team_games) != 39910:

        print(
            "WARNING: Expected approximately "
            "39,910 rows based on the current "
            "database."
        )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    team_games.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print()
    print("==============================")
    print("PREGAME DATABASE COMPLETE")
    print("==============================")

    print(
        f"Rows: "
        f"{len(team_games):,}"
    )

    print(
        f"Unique matches: "
        f"{team_games['match_id'].nunique():,}"
    )

    print(
        f"Unique teams: "
        f"{team_games['team'].nunique():,}"
    )

    print()
    print(
        "PREGAME HISTORY AVAILABILITY"
    )

    print(
        team_games[
            "pregame_games"
        ]
        .describe()
        .round(2)
        .to_string()
    )

    print()
    print(
        "LEAGUE SCORING BASELINES"
    )

    latest_baselines = (
        team_games
        .sort_values("date")
        .groupby("league")
        .tail(1)
        [
            [
                "league",
                "league_avg_home_goals",
                "league_avg_away_goals",
                "league_avg_total_goals",
            ]
        ]
        .set_index("league")
        .round(3)
        .sort_index()
    )

    print(
        latest_baselines.to_string()
    )

    print()
    print(
        "SAMPLE STRENGTH RATINGS"
    )

    sample = (
        team_games[
            team_games["pregame_games"] >= 20
        ]
        .tail(10)
        [
            [
                "date",
                "league",
                "team",
                "venue",
                "pregame_games",
                "pregame_ew_goals_for",
                "pregame_ew_goals_against",
                "pregame_attack_strength",
                "pregame_defense_strength",
                "pregame_venue_attack_strength",
                "pregame_venue_defense_strength",
            ]
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
        "Validation: "
        "First team games contain no "
        "current-match history ✅"
    )

    print(
        "Validation: "
        "Same-day matches do not enter "
        "league baselines ✅"
    )

    print()
    print(
        f"Saved:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()