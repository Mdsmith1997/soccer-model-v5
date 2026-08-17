from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "matches.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "team_game_stats.csv"
)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def get_value(row, column):
    """
    Safely return a value from a match row.

    Some older league/season files may not contain every
    statistical field, so missing columns return pd.NA.
    """
    if column in row.index:
        return row[column]

    return pd.NA


def determine_team_result(
    goals_for,
    goals_against,
):
    if goals_for > goals_against:
        return "W"

    if goals_for < goals_against:
        return "L"

    return "D"


def determine_points(result):
    if result == "W":
        return 3

    if result == "D":
        return 1

    return 0


# ---------------------------------------------------------
# BUILD HOME TEAM ROW
# ---------------------------------------------------------

def build_home_row(row):

    goals_for = row["home_goals"]
    goals_against = row["away_goals"]

    result = determine_team_result(
        goals_for,
        goals_against,
    )

    return {
        "match_id": row["match_id"],
        "date": row["date"],
        "season": row["season"],
        "league_code": row["league_code"],
        "league": row["league"],

        "team": row["home_team"],
        "opponent": row["away_team"],

        "venue": "HOME",
        "is_home": 1,

        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_difference": (
            goals_for - goals_against
        ),

        "result": result,
        "points": determine_points(result),

        "win": int(result == "W"),
        "draw": int(result == "D"),
        "loss": int(result == "L"),

        "scored": int(goals_for > 0),
        "clean_sheet": int(
            goals_against == 0
        ),

        "over_2_5": int(
            (
                goals_for
                + goals_against
            ) > 2.5
        ),

        "btts": int(
            (goals_for > 0)
            and
            (goals_against > 0)
        ),

        "shots_for": get_value(
            row,
            "home_shots",
        ),

        "shots_against": get_value(
            row,
            "away_shots",
        ),

        "shots_on_target_for": get_value(
            row,
            "home_shots_on_target",
        ),

        "shots_on_target_against": get_value(
            row,
            "away_shots_on_target",
        ),

        "corners_for": get_value(
            row,
            "home_corners",
        ),

        "corners_against": get_value(
            row,
            "away_corners",
        ),

        "fouls_for": get_value(
            row,
            "home_fouls",
        ),

        "fouls_against": get_value(
            row,
            "away_fouls",
        ),

        "yellow_cards": get_value(
            row,
            "home_yellow",
        ),

        "opponent_yellow_cards": get_value(
            row,
            "away_yellow",
        ),

        "red_cards": get_value(
            row,
            "home_red",
        ),

        "opponent_red_cards": get_value(
            row,
            "away_red",
        ),
    }


# ---------------------------------------------------------
# BUILD AWAY TEAM ROW
# ---------------------------------------------------------

def build_away_row(row):

    goals_for = row["away_goals"]
    goals_against = row["home_goals"]

    result = determine_team_result(
        goals_for,
        goals_against,
    )

    return {
        "match_id": row["match_id"],
        "date": row["date"],
        "season": row["season"],
        "league_code": row["league_code"],
        "league": row["league"],

        "team": row["away_team"],
        "opponent": row["home_team"],

        "venue": "AWAY",
        "is_home": 0,

        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_difference": (
            goals_for - goals_against
        ),

        "result": result,
        "points": determine_points(result),

        "win": int(result == "W"),
        "draw": int(result == "D"),
        "loss": int(result == "L"),

        "scored": int(goals_for > 0),
        "clean_sheet": int(
            goals_against == 0
        ),

        "over_2_5": int(
            (
                goals_for
                + goals_against
            ) > 2.5
        ),

        "btts": int(
            (goals_for > 0)
            and
            (goals_against > 0)
        ),

        "shots_for": get_value(
            row,
            "away_shots",
        ),

        "shots_against": get_value(
            row,
            "home_shots",
        ),

        "shots_on_target_for": get_value(
            row,
            "away_shots_on_target",
        ),

        "shots_on_target_against": get_value(
            row,
            "home_shots_on_target",
        ),

        "corners_for": get_value(
            row,
            "away_corners",
        ),

        "corners_against": get_value(
            row,
            "home_corners",
        ),

        "fouls_for": get_value(
            row,
            "away_fouls",
        ),

        "fouls_against": get_value(
            row,
            "home_fouls",
        ),

        "yellow_cards": get_value(
            row,
            "away_yellow",
        ),

        "opponent_yellow_cards": get_value(
            row,
            "home_yellow",
        ),

        "red_cards": get_value(
            row,
            "away_red",
        ),

        "opponent_red_cards": get_value(
            row,
            "home_red",
        ),
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print()
    print("==============================")
    print("BUILDING TEAM GAME DATABASE")
    print("==============================")
    print()

    # -----------------------------------------------------
    # LOAD MATCHES
    # -----------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{INPUT_FILE}"
        )

    matches = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    print(
        f"Matches loaded: "
        f"{len(matches):,}"
    )

    # -----------------------------------------------------
    # VALIDATE CORE COLUMNS
    # -----------------------------------------------------

    required_columns = [
        "match_id",
        "date",
        "season",
        "league_code",
        "league",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in matches.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # -----------------------------------------------------
    # CREATE TWO ROWS PER MATCH
    # -----------------------------------------------------

    team_rows = []

    for _, row in matches.iterrows():

        team_rows.append(
            build_home_row(row)
        )

        team_rows.append(
            build_away_row(row)
        )

    team_games = pd.DataFrame(
        team_rows
    )

    # -----------------------------------------------------
    # SORT CHRONOLOGICALLY
    # -----------------------------------------------------

    team_games = team_games.sort_values(
        [
            "date",
            "league_code",
            "match_id",
            "is_home",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # ADD TEAM GAME NUMBER
    #
    # This is useful later when determining how much
    # historical information was available before a match.
    # -----------------------------------------------------

    team_games["team_game_number"] = (
        team_games
        .groupby(
            [
                "league_code",
                "team",
            ]
        )
        .cumcount()
        + 1
    )

    # -----------------------------------------------------
    # DATA CHECKS
    # -----------------------------------------------------

    expected_rows = (
        len(matches) * 2
    )

    actual_rows = len(team_games)

    if actual_rows != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows "
            f"but created {actual_rows:,}."
        )

    # Every match should appear exactly twice.
    match_counts = (
        team_games["match_id"]
        .value_counts()
    )

    bad_matches = match_counts[
        match_counts != 2
    ]

    if len(bad_matches) > 0:
        raise ValueError(
            "Some matches do not have "
            "exactly two team records."
        )

    # Home + away goal differential should equal zero.
    goal_diff_check = (
        team_games
        .groupby("match_id")
        ["goal_difference"]
        .sum()
    )

    bad_goal_diff = goal_diff_check[
        goal_diff_check != 0
    ]

    if len(bad_goal_diff) > 0:
        raise ValueError(
            "Goal difference validation failed."
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
    print("TEAM GAME DATABASE COMPLETE")
    print("==============================")

    print(
        f"Team-game rows: "
        f"{len(team_games):,}"
    )

    print(
        f"Expected rows: "
        f"{expected_rows:,}"
    )

    print(
        f"Unique matches: "
        f"{team_games['match_id'].nunique():,}"
    )

    print(
        f"Unique teams: "
        f"{team_games['team'].nunique():,}"
    )

    print(
        f"Date range: "
        f"{team_games['date'].min().date()} "
        f"-> "
        f"{team_games['date'].max().date()}"
    )

    print()
    print("ROWS BY LEAGUE")

    print(
        team_games
        .groupby("league")
        .size()
        .sort_values(
            ascending=False
        )
        .to_string()
    )

    print()
    print("AVERAGE TEAM PERFORMANCE")

    summary = (
        team_games
        .groupby("league")
        .agg(
            goals_for=(
                "goals_for",
                "mean",
            ),
            shots_for=(
                "shots_for",
                "mean",
            ),
            shots_on_target=(
                "shots_on_target_for",
                "mean",
            ),
            points=(
                "points",
                "mean",
            ),
        )
        .round(3)
    )

    print(
        summary.to_string()
    )

    print()
    print(
        "Validation: "
        "Every match has exactly "
        "2 team records ✅"
    )

    print(
        "Validation: "
        "Goal differences balance "
        "to zero ✅"
    )

    print()
    print(
        f"Saved:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()