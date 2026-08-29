from pathlib import Path
import re

import numpy as np
import pandas as pd

import confirm_opponent_adjusted_recency_v5 as v5
import tune_overall_venue_v5 as ov


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FIXTURES_FILE = (
    ROOT
    / "data"
    / "live"
    / "upcoming_fixtures.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "live"
    / "v5_live_predictions_core.csv"
)


# ============================================================
# FROZEN V5 SETTINGS
# ============================================================

GOAL_WEIGHT = 0.09
XG_WEIGHT = 0.75
SHOT_WEIGHT = 0.16
SOT_WEIGHT = 0.00

GOAL_RECENCY = 0.975
XG_RECENCY = 0.925
SHOT_RECENCY = 0.850

OPPONENT_STRENGTH = 0.875

OVERALL_WEIGHT = 0.80
VENUE_WEIGHT = 0.20


# ============================================================
# SUPPORTED LEAGUES
# ============================================================

SUPPORTED_LEAGUES = {
    "Premier League",
    "Bundesliga",
}

FOOTYSTATS_TRANSFER_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "footystats_multileague_team_pregame_v2.csv"
)

TRANSFER_TEAM_SHADOWS = {
    # Real promoted team : temporary native top-flight team
    "Coventry City": "Bournemouth",
    "Elversberg": "Augsburg",
}

TRANSFER_TEAM_HISTORY = {
    "Coventry City": "Coventry City",
    "Elversberg": "Elversberg",
}

# ============================================================
# ENGLISH LEAGUE TRANSITION ADJUSTMENTS
#
# Frozen from tune_transition_asymmetric_v4.py.
#
# Parameters selected using 2017/18–2022/23 only.
# Validated on 2023/24 and final checked on 2024/25.
#
# IMPORTANT:
# These adjustments are validated ONLY for:
#
# Championship -> Premier League
# Premier League -> Championship
#
# Do not automatically apply these coefficients to other
# countries or league pyramids.
# ============================================================

PROMOTION_ADJUSTMENT = 0.205
RELEGATION_ADJUSTMENT = 0.135


# ============================================================
# HELPERS
# ============================================================

def compact_season_from_date(
    date,
):

    date = pd.Timestamp(
        date
    )

    if date.month >= 7:

        start_year = date.year

    else:

        start_year = (
            date.year
            -
            1
        )

    return (
        f"{str(start_year)[-2:]}"
        f"{str(start_year + 1)[-2:]}"
    )


def clean_id_text(
    text,
):

    text = str(
        text
    )

    text = re.sub(
        r"[^A-Za-z0-9]+",
        "",
        text,
    )

    return text


def make_live_match_id(
    date,
    home_team,
    away_team,
):

    date_text = (
        pd.Timestamp(
            date
        )
        .strftime(
            "%Y%m%d"
        )
    )

    return (
        "LIVE_"
        f"{date_text}_"
        f"{clean_id_text(home_team)}_"
        f"{clean_id_text(away_team)}"
    )


# ============================================================
# VALIDATE FROZEN SETTINGS
# ============================================================

def validate_frozen_settings():

    checks = {
        "GOAL_WEIGHT":
            (
                ov.GOAL_WEIGHT,
                GOAL_WEIGHT,
            ),

        "XG_WEIGHT":
            (
                ov.XG_WEIGHT,
                XG_WEIGHT,
            ),

        "SHOT_WEIGHT":
            (
                ov.SHOT_WEIGHT,
                SHOT_WEIGHT,
            ),

        "SOT_WEIGHT":
            (
                ov.SOT_WEIGHT,
                SOT_WEIGHT,
            ),

        "GOAL_RECENCY":
            (
                ov.GOAL_RECENCY,
                GOAL_RECENCY,
            ),

        "XG_RECENCY":
            (
                ov.XG_RECENCY,
                XG_RECENCY,
            ),

        "SHOT_RECENCY":
            (
                ov.SHOT_RECENCY,
                SHOT_RECENCY,
            ),

        "OPPONENT_STRENGTH":
            (
                ov.OPPONENT_STRENGTH,
                OPPONENT_STRENGTH,
            ),
    }

    for name, (
        actual,
        expected,
    ) in checks.items():

        if not np.isclose(
            float(actual),
            float(expected),
        ):

            raise ValueError(
                f"Frozen setting mismatch: "
                f"{name}\n"
                f"Expected: {expected}\n"
                f"Found: {actual}"
            )

    # --------------------------------------------------------
    # confirm_opponent_adjusted_recency_v5 may also expose
    # the opponent-strength constant directly.
    # --------------------------------------------------------

    if hasattr(
        v5,
        "OPPONENT_STRENGTH",
    ):

        actual = float(
            getattr(
                v5,
                "OPPONENT_STRENGTH",
            )
        )

        if not np.isclose(
            actual,
            OPPONENT_STRENGTH,
        ):

            raise ValueError(
                "confirm_opponent_adjusted_recency_v5 "
                "has a different opponent strength.\n"
                f"Expected: {OPPONENT_STRENGTH}\n"
                f"Found: {actual}"
            )


# ============================================================
# FIXTURE FILE
# ============================================================

def create_fixture_template():

    FIXTURES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    template = pd.DataFrame(
        columns=[
            "match_id",
            "date",
            "league",
            "home_team",
            "away_team",
        ]
    )

    template.to_csv(
        FIXTURES_FILE,
        index=False,
    )


def load_fixtures():

    if not FIXTURES_FILE.exists():

        create_fixture_template()

        raise FileNotFoundError(
            "\nCreated upcoming-fixture template:\n"
            f"{FIXTURES_FILE}\n\n"
            "Add upcoming fixtures and run this script again.\n\n"
            "Required columns:\n"
            "match_id,date,league,home_team,away_team\n\n"
            "match_id may be left blank."
        )

    df = pd.read_csv(
        FIXTURES_FILE
    )

    required = [
        "date",
        "league",
        "home_team",
        "away_team",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "upcoming_fixtures.csv missing columns: "
            + str(
                missing
            )
        )

    if len(df) == 0:

        raise ValueError(
            "\nupcoming_fixtures.csv is empty.\n\n"
            "Add upcoming fixtures first."
        )

    # ========================================================
    # DATE CLEANING
    # ========================================================

    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        errors="coerce",
    )

    if df[
        "date"
    ].isna().any():

        bad = df[
            df[
                "date"
            ].isna()
        ]

        raise ValueError(
            "Invalid fixture dates found:\n"
            +
            bad.to_string(
                index=False
            )
        )

    # ========================================================
    # TEXT CLEANING
    # ========================================================

    df[
        "league"
    ] = (
        df[
            "league"
        ]
        .astype(str)
        .str.strip()
    )

    df[
        "home_team"
    ] = (
        df[
            "home_team"
        ]
        .astype(str)
        .str.strip()
    )

    df[
        "away_team"
    ] = (
        df[
            "away_team"
        ]
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # ROUTE ONLY CORE / UNDERSTAT V5 LEAGUES
    # ========================================================
    #
    # upcoming_fixtures.csv is the master multi-league board.
    #
    # This scorer handles only:
    #
    #   Premier League
    #   Bundesliga
    #
    # All other leagues are intentionally ignored here.
    # ========================================================

    all_fixture_count = len(
        df
    )

    all_leagues = (
        df[
            "league"
        ]
        .value_counts()
        .sort_index()
    )

    df = df.loc[
        df[
            "league"
        ].isin(
            SUPPORTED_LEAGUES
        )
    ].copy()

    ignored_count = (
        all_fixture_count
        -
        len(df)
    )

    print()
    print(
        "=" * 70
    )
    print(
        "CORE V5 FIXTURE ROUTING"
    )
    print(
        "=" * 70
    )

    print(
        "Master fixture rows:",
        all_fixture_count,
    )

    print(
        "Core V5 fixtures:",
        len(df),
    )

    print(
        "Ignored non-core fixtures:",
        ignored_count,
    )

    print()
    print(
        "MASTER BOARD BY LEAGUE"
    )

    print(
        all_leagues.to_string()
    )

    if len(df):

        print()
        print(
            "CORE V5 BOARD BY LEAGUE"
        )

        print(
            df[
                "league"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    if len(df) == 0:

        raise ValueError(
            "\nNo supported core V5 fixtures found "
            "in upcoming_fixtures.csv.\n\n"
            "Current core V5 deployment supports:\n"
            +
            "\n".join(
                sorted(
                    SUPPORTED_LEAGUES
                )
            )
        )

    # ========================================================
    # SEASON
    # ========================================================
    #
    # The historical V5 pipeline expects each future fixture
    # row to include a season such as:
    #
    #   2526
    #   2627
    #
    # Derive it automatically from the fixture date.
    # ========================================================

    if "season" not in df.columns:

        def season_from_date(
            value,
        ):

            year = (
                value.year
            )

            month = (
                value.month
            )

            if month >= 7:

                start_year = (
                    year
                )

                end_year = (
                    year
                    +
                    1
                )

            else:

                start_year = (
                    year
                    -
                    1
                )

                end_year = (
                    year
                )

            return (
                f"{str(start_year)[-2:]}"
                f"{str(end_year)[-2:]}"
            )

        df[
            "season"
        ] = (
            df[
                "date"
            ]
            .apply(
                season_from_date
            )
        )

    else:

        df[
            "season"
        ] = (
            df[
                "season"
            ]
            .astype(str)
            .str.strip()
            .str.replace(
                ".0",
                "",
                regex=False,
            )
            .str.zfill(
                4
            )
        )

    # ========================================================
    # MATCH IDS
    # ========================================================

    if "match_id" not in df.columns:

        df[
            "match_id"
        ] = np.nan

    missing_id = (
        df[
            "match_id"
        ].isna()
        |
        (
            df[
                "match_id"
            ]
            .astype(str)
            .str.strip()
            ==
            ""
        )
    )

    if missing_id.any():

        generated = (
            df.loc[
                missing_id,
                "date",
            ]
            .dt.strftime(
                "%Y%m%d"
            )
            +
            "_"
            +
            df.loc[
                missing_id,
                "league",
            ]
            .astype(str)
            .str.lower()
            .str.replace(
                r"[^a-z0-9]+",
                "_",
                regex=True,
            )
            .str.strip(
                "_"
            )
            +
            "_"
            +
            df.loc[
                missing_id,
                "home_team",
            ]
            .astype(str)
            .str.lower()
            .str.replace(
                r"[^a-z0-9]+",
                "_",
                regex=True,
            )
            .str.strip(
                "_"
            )
            +
            "_"
            +
            df.loc[
                missing_id,
                "away_team",
            ]
            .astype(str)
            .str.lower()
            .str.replace(
                r"[^a-z0-9]+",
                "_",
                regex=True,
            )
            .str.strip(
                "_"
            )
        )

        df.loc[
            missing_id,
            "match_id",
        ] = generated

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if df[
        "home_team"
    ].eq("").any():

        raise ValueError(
            "Blank home_team values found "
            "in core V5 fixtures."
        )

    if df[
        "away_team"
    ].eq("").any():

        raise ValueError(
            "Blank away_team values found "
            "in core V5 fixtures."
        )

    if df[
        "season"
    ].isna().any():

        raise ValueError(
            "Missing season values found "
            "in core V5 fixtures."
        )

    duplicate_ids = (
        df[
            "match_id"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicate_ids.any():

        bad = df.loc[
            duplicate_ids,
            [
                "match_id",
                "date",
                "season",
                "league",
                "home_team",
                "away_team",
            ],
        ]

        raise ValueError(
            "Duplicate match_id values found "
            "in core V5 fixtures:\n"
            +
            bad.to_string(
                index=False
            )
        )

    # ========================================================
    # SORT / RETURN
    # ========================================================

    df = (
        df
        .sort_values(
            [
                "date",
                "league",
                "home_team",
                "away_team",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# LEAGUE CODE LOOKUP
# ============================================================

def league_code_lookup(
    team_data,
):

    required = [
        "league",
        "league_code",
    ]

    missing = [
        c
        for c in required
        if c not in team_data.columns
    ]

    if missing:

        raise ValueError(
            "Team data missing league lookup columns: "
            + str(
                missing
            )
        )

    lookup = (
        team_data[
            [
                "date",
                "league",
                "league_code",
            ]
        ]
        .dropna()
        .sort_values(
            "date"
        )
        .drop_duplicates(
            subset=[
                "league",
            ],
            keep="last",
        )
        .set_index(
            "league"
        )[
            "league_code"
        ]
        .to_dict()
    )

    return lookup


# ============================================================
# LIVE → HISTORICAL TEAM NAME MAP
#
# Converts provider / live fixture names into the exact
# historical model names used by the frozen V5 databases.
#
# This is name reconciliation only.
# It does NOT create history for genuinely new teams.
# ============================================================

LIVE_TEAM_ALIASES = {

    # --------------------------------------------------------
    # PREMIER LEAGUE
    # --------------------------------------------------------

    "Brighton and Hove Albion":
        "Brighton",

    "Hull City":
        "Hull",

    "Ipswich Town":
        "Ipswich",

    "Leeds United":
        "Leeds",

    "Manchester City":
        "Man City",

    "Manchester United":
        "Man United",

    "Newcastle United":
        "Newcastle",

    "Nottingham Forest":
        "Nott'm Forest",

    "Tottenham Hotspur":
        "Tottenham",

    # --------------------------------------------------------
    # BUNDESLIGA
    # --------------------------------------------------------

    "1. FC Köln":
        "FC Koln",

    "Bayer Leverkusen":
        "Leverkusen",

    "Borussia Dortmund":
        "Dortmund",

    "Borussia Monchengladbach":
        "M'gladbach",

    "Eintracht Frankfurt":
        "Ein Frankfurt",

    "FC Schalke 04":
        "Schalke 04",

    "FSV Mainz 05":
        "Mainz",

    "Hamburger SV":
        "Hamburg",

    "SC Freiburg":
        "Freiburg",

    "SC Paderborn":
        "Paderborn",

    "TSG Hoffenheim":
        "Hoffenheim",

    "VfB Stuttgart":
        "Stuttgart",
}


def reconcile_fixture_team_names(
    fixtures,
):
    """
    Convert live/provider team names into the exact names used
    by the frozen historical V5 databases.

    Original provider names are retained for auditing.
    """

    fixtures = fixtures.copy()

    fixtures[
        "provider_home_team"
    ] = fixtures[
        "home_team"
    ]

    fixtures[
        "provider_away_team"
    ] = fixtures[
        "away_team"
    ]

    fixtures[
        "home_team"
    ] = fixtures[
        "home_team"
    ].replace(
        LIVE_TEAM_ALIASES
    )

    fixtures[
        "away_team"
    ] = fixtures[
        "away_team"
    ].replace(
        LIVE_TEAM_ALIASES
    )

    return fixtures

# ============================================================
# VALIDATE LIVE TEAMS
# ============================================================

def validate_team_coverage(
    fixtures,
    team_data,
    xg_data,
):

    historical_teams = set(
        team_data[
            "team"
        ]
        .dropna()
        .astype(str)
    )

    fixture_teams = set(
        fixtures[
            "home_team"
        ]
    ) | set(
        fixtures[
            "away_team"
        ]
    )

    missing_team = sorted(
        fixture_teams
        -
        historical_teams
    )

    if missing_team:

        raise ValueError(
            "\nTeams missing from historical "
            "goal/shot database:\n"
            +
            "\n".join(
                missing_team
            )
            +
            "\n\nTeam names in upcoming_fixtures.csv "
            "must exactly match the historical model names."
        )

    # --------------------------------------------------------
    # xG input is usually match-level.
    # --------------------------------------------------------

    xg_teams = set()

    for col in [
        "home_team",
        "away_team",
    ]:

        if col in xg_data.columns:

            xg_teams.update(
                xg_data[
                    col
                ]
                .dropna()
                .astype(str)
            )

    missing_xg = sorted(
        fixture_teams
        -
        xg_teams
    )

    if missing_xg:

        raise ValueError(
            "\nTeams missing from historical xG database:\n"
            +
            "\n".join(
                missing_xg
            )
            +
            "\n\nV5 requires xG history for every "
            "live team."
        )


# ============================================================
# BUILD FUTURE TEAM ROWS
#
# These rows contain fixture identity only.
# Match outcomes/stats remain NaN.
#
# Because the V5 histories are prior-only, these
# rows allow the historical pipeline to calculate
# the NEXT pregame state without adding future data.
# ============================================================

def build_future_team_rows(
    fixtures,
    team_data,
):

    league_codes = league_code_lookup(
        team_data
    )

    rows = []

    for fixture in (
        fixtures
        .itertuples(
            index=False
        )
    ):

        league = fixture.league

        if league not in league_codes:

            raise ValueError(
                f"No league_code found for "
                f"{league}"
            )

        league_code = (
            league_codes[
                league
            ]
        )

        definitions = [
            (
                fixture.home_team,
                fixture.away_team,
                "HOME",
                True,
            ),
            (
                fixture.away_team,
                fixture.home_team,
                "AWAY",
                False,
            ),
        ]

        for (
            team,
            opponent,
            venue,
            is_home,
        ) in definitions:

            row = {
                col:
                    np.nan
                for col in team_data.columns
            }

            row[
                "match_id"
            ] = fixture.match_id

            row[
                "date"
            ] = fixture.date

            if "season" in row:

                row[
                    "season"
                ] = fixture.season

            if "league_code" in row:

                row[
                    "league_code"
                ] = league_code

            if "league" in row:

                row[
                    "league"
                ] = league

            row[
                "team"
            ] = team

            row[
                "opponent"
            ] = opponent

            row[
                "venue"
            ] = venue

            if "is_home" in row:

                row[
                    "is_home"
                ] = is_home

            # ------------------------------------------------
            # Optional sequence field.
            # ------------------------------------------------

            if (
                "team_game_number"
                in row
            ):

                prior_numbers = pd.to_numeric(
                    team_data.loc[
                        team_data[
                            "team"
                        ]
                        ==
                        team,
                        "team_game_number",
                    ],
                    errors="coerce",
                )

                if prior_numbers.notna().any():

                    row[
                        "team_game_number"
                    ] = (
                        prior_numbers.max()
                        +
                        1
                    )

            rows.append(
                row
            )

    future = pd.DataFrame(
        rows,
        columns=team_data.columns,
    )

    future[
        "date"
    ] = pd.to_datetime(
        future[
            "date"
        ]
    )

    return future


# ============================================================
# BUILD FUTURE XG MATCH ROWS
# ============================================================

def build_future_xg_rows(
    fixtures,
    xg_data,
):

    rows = []

    for index, fixture in enumerate(
        fixtures.itertuples(
            index=False
        )
    ):

        row = {
            col:
                np.nan
            for col in xg_data.columns
        }

        row[
            "match_id"
        ] = fixture.match_id

        row[
            "date"
        ] = fixture.date

        if "understat_date" in row:

            row[
                "understat_date"
            ] = fixture.date

        if "season" in row:

            row[
                "season"
            ] = fixture.season

        if "league" in row:

            row[
                "league"
            ] = fixture.league

        row[
            "home_team"
        ] = fixture.home_team

        row[
            "away_team"
        ] = fixture.away_team

        # ----------------------------------------------------
        # Some xG tables include a source game ID.
        # Use a negative integer so it cannot collide with
        # real Understat IDs.
        # ----------------------------------------------------

        if "game_id" in row:

            row[
                "game_id"
            ] = (
                -9_000_000
                -
                index
            )

        if "date_difference_days" in row:

            row[
                "date_difference_days"
            ] = 0

        rows.append(
            row
        )

    future = pd.DataFrame(
        rows,
        columns=xg_data.columns,
    )

    future[
        "date"
    ] = pd.to_datetime(
        future[
            "date"
        ]
    )

    return future


# ============================================================
# AUGMENT HISTORICAL DATABASES
# ============================================================

def build_augmented_inputs(
    fixtures,
):

    fixtures = (
        reconcile_fixture_team_names(
            fixtures
        )
    )

    print(
        "Loading historical goal / shot data..."
    )

    team_data = (
        v5.load_team_data()
        .copy()
    )

    team_data[
        "date"
    ] = pd.to_datetime(
        team_data[
            "date"
        ]
    )

    print(
        "Loading historical xG data..."
    )

    xg_data = (
        v5.load_xg()
        .copy()
    )

    xg_data[
        "date"
    ] = pd.to_datetime(
        xg_data[
            "date"
        ]
    )

    validate_team_coverage(
        fixtures,
        team_data,
        xg_data,
    )

    future_team = (
        build_future_team_rows(
            fixtures,
            team_data,
        )
    )

    future_xg = (
        build_future_xg_rows(
            fixtures,
            xg_data,
        )
    )

    augmented_team = pd.concat(
        [
            team_data,
            future_team,
        ],
        ignore_index=True,
    )

    augmented_xg = pd.concat(
        [
            xg_data,
            future_xg,
        ],
        ignore_index=True,
    )

    augmented_team = (
        augmented_team
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

    augmented_xg = (
        augmented_xg
        .sort_values(
            [
                "date",
                "match_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        augmented_team,
        augmented_xg,
    )

# ============================================================
# FOOTYSTATS TRANSFERRED-TEAM COMPONENT BRIDGE
# ============================================================

def final_ewma_transfer(
    values,
    decay,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    num = 0.0
    den = 0.0

    for value in values:
        num = (
            decay
            *
            num
            +
            value
        )

        den = (
            decay
            *
            den
            +
            1.0
        )

    if den <= 0:
        return np.nan

    return num / den


def load_footystats_transfer_state(
    team_name,
    venue,
):
    """
    Build the latest transferred state from the completed
    FootyStats opponent-adjusted game-performance history.

    No parameters are fitted here.
    Uses frozen V5 recencies:
        goals = 0.975
        xG    = 0.925
        shots = 0.850
    """

    if not FOOTYSTATS_TRANSFER_FILE.exists():

        raise FileNotFoundError(
            FOOTYSTATS_TRANSFER_FILE
        )

    required = [
        "date",
        "team",
        "venue",

        "adj_goal_attack_perf",
        "adj_goal_defense_perf",

        "adj_xg_attack_perf",
        "adj_xg_defense_perf",

        "adj_shot_attack_perf",
        "adj_shot_defense_perf",
    ]

    # Load only the columns required to reconstruct the
    # transferred team's frozen V5 state.
    df = pd.read_csv(
        FOOTYSTATS_TRANSFER_FILE,
        usecols=required,
        low_memory=False,
    )

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "FootyStats transfer file missing columns:\n"
            +
            "\n".join(
                missing
            )
        )

    club = df.loc[
        df["team"].astype(str)
        ==
        str(team_name)
    ].copy()

    if club.empty:

        raise ValueError(
            f"No FootyStats transfer history found "
            f"for {team_name}"
        )

    club["date"] = pd.to_datetime(
        club["date"],
        errors="coerce",
    )

    club = (
        club
        .dropna(
            subset=["date"]
        )
        .sort_values(
            "date"
        )
    )

    venue = str(
        venue
    ).upper()

    if venue not in {
        "HOME",
        "AWAY",
    }:

        raise ValueError(
            f"Invalid transfer venue: {venue}"
        )

    decays = {
        "goal": GOAL_RECENCY,
        "xg": XG_RECENCY,
        "shot": SHOT_RECENCY,
    }

    state = {
        "history_team":
            team_name,

        "history_games":
            len(club),

        "venue_history_games":
            int(
                (
                    club["venue"]
                    .astype(str)
                    .str.upper()
                    ==
                    venue
                ).sum()
            ),

        "last_history_date":
            club["date"].max(),
    }

    for signal, decay in decays.items():

        for role in (
            "attack",
            "defense",
        ):

            col = (
                f"adj_{signal}_"
                f"{role}_perf"
            )

            overall = (
                final_ewma_transfer(
                    club[col].to_numpy(
                        dtype=float
                    ),
                    decay,
                )
            )

            venue_values = (
                club.loc[
                    club["venue"]
                    ==
                    venue,
                    col,
                ]
                .to_numpy(
                    dtype=float
                )
            )

            venue_value = (
                final_ewma_transfer(
                    venue_values,
                    decay,
                )
            )

            if not np.isfinite(
                overall
            ):
                raise ValueError(
                    f"No usable {signal} "
                    f"{role} history for "
                    f"{team_name}"
                )

            # If venue history happened to be absent,
            # fall back to overall rather than neutral.
            if not np.isfinite(
                venue_value
            ):
                venue_value = overall

            state[
                f"{signal}_{role}_overall"
            ] = float(
                overall
            )

            state[
                f"{signal}_{role}_venue"
            ] = float(
                venue_value
            )

    return state


def overwrite_transfer_components(
    components,
    fixtures,
):
    """
    Replace only the shadowed promoted-team side of the
    frozen component row with its genuine FootyStats
    transferred state.

    Native opponent components and league goal baselines
    remain untouched.
    """

    components = components.copy()

    for fixture in (
        fixtures.itertuples(
            index=False
        )
    ):

        teams = {
            "home":
                fixture.home_team,

            "away":
                fixture.away_team,
        }

        for side, live_team in teams.items():

            if (
                live_team
                not in
                TRANSFER_TEAM_HISTORY
            ):
                continue

            venue = (
                "HOME"
                if side == "home"
                else
                "AWAY"
            )

            history_team = (
                TRANSFER_TEAM_HISTORY[
                    live_team
                ]
            )

            state = (
                load_footystats_transfer_state(
                    history_team,
                    venue,
                )
            )

            mask = (
                components[
                    "match_id"
                ]
                ==
                fixture.match_id
            )

            if mask.sum() != 1:

                raise ValueError(
                    "Expected exactly one component "
                    f"row for transfer fixture "
                    f"{fixture.match_id}; "
                    f"found {int(mask.sum())}"
                )

            # ----------------------------------------------
            # RESTORE REAL TRANSFERRED TEAM IDENTITY
            # ----------------------------------------------

            team_col = (
                "home_team"
                if side == "home"
                else "away_team"
            )

            components.loc[
                mask,
                team_col,
            ] = live_team

            # The frozen pipeline's *_team_check fields
            # mirror the canonical component identity.
            check_col = (
                "home_team_check"
                if side == "home"
                else "away_team_check"
            )

            if check_col in components.columns:
                components.loc[
                    mask,
                    check_col,
                ] = live_team

            games_col = (
                f"{side}_games"
            )

            if games_col in components.columns:
                components.loc[
                    mask,
                    games_col,
                ] = state[
                    "history_games"
                ]

            venue_games_col = (
                f"{side}_venue_games"
            )

            if venue_games_col in components.columns:
                components.loc[
                    mask,
                    venue_games_col,
                ] = state[
                    "venue_history_games"
                ]

            league_changed_col = (
                f"{side}_league_changed"
            )

            if league_changed_col in components.columns:
                components.loc[
                    mask,
                    league_changed_col,
                ] = 1

            # ----------------------------------------------
            # GOALS
            # ----------------------------------------------

            components.loc[
                mask,
                f"{side}_adj_goal_attack",
            ] = (
                state[
                    "goal_attack_overall"
                ]
            )

            components.loc[
                mask,
                f"{side}_adj_goal_defense",
            ] = (
                state[
                    "goal_defense_overall"
                ]
            )

            components.loc[
                mask,
                f"{side}_adj_venue_goal_attack",
            ] = (
                state[
                    "goal_attack_venue"
                ]
            )

            components.loc[
                mask,
                f"{side}_adj_venue_goal_defense",
            ] = (
                state[
                    "goal_defense_venue"
                ]
            )

            # ----------------------------------------------
            # SHOTS
            # ----------------------------------------------

            components.loc[
                mask,
                f"{side}_adj_shot_attack",
            ] = (
                state[
                    "shot_attack_overall"
                ]
            )

            components.loc[
                mask,
                f"{side}_adj_shot_defense",
            ] = (
                state[
                    "shot_defense_overall"
                ]
            )

            components.loc[
                mask,
                f"{side}_adj_venue_shot_attack",
            ] = (
                state[
                    "shot_attack_venue"
                ]
            )

            components.loc[
                mask,
                f"{side}_adj_venue_shot_defense",
            ] = (
                state[
                    "shot_defense_venue"
                ]
            )

            # ----------------------------------------------
            # XG
            # ----------------------------------------------

            components.loc[
                mask,
                f"{side}_xg_attack_overall",
            ] = (
                state[
                    "xg_attack_overall"
                ]
            )

            components.loc[
                mask,
                f"{side}_xg_defense_overall",
            ] = (
                state[
                    "xg_defense_overall"
                ]
            )

            components.loc[
                mask,
                f"{side}_xg_attack_venue",
            ] = (
                state[
                    "xg_attack_venue"
                ]
            )

            components.loc[
                mask,
                f"{side}_xg_defense_venue",
            ] = (
                state[
                    "xg_defense_venue"
                ]
            )

            print(
                "Transferred FootyStats state:",
                live_team,
                "|",
                state["history_games"],
                "games",
                "| last:",
                state[
                    "last_history_date"
                ].date(),
            )

    return components


# ============================================================
# BUILD FROZEN COMPONENT STORE
#
# We temporarily replace the two loader functions used by
# tune_overall_venue_v5.build_component_store().
#
# All feature construction below is therefore performed by
# the SAME historical V5 functions already used in backtests.
# ============================================================

def build_live_components(
    fixtures,
):

    # ========================================================
    # CREATE SHADOW FIXTURES FOR TEAMS MISSING FROM CORE V5
    # ========================================================

    pipeline_fixtures = (
        fixtures.copy()
    )

    for real_team, shadow_team in (
        TRANSFER_TEAM_SHADOWS.items()
    ):

        pipeline_fixtures[
            "home_team"
        ] = (
            pipeline_fixtures[
                "home_team"
            ]
            .replace(
                {
                    real_team:
                        shadow_team
                }
            )
        )

        pipeline_fixtures[
            "away_team"
        ] = (
            pipeline_fixtures[
                "away_team"
            ]
            .replace(
                {
                    real_team:
                        shadow_team
                }
            )
        )

    (
        augmented_team,
        augmented_xg,
    ) = build_augmented_inputs(
        pipeline_fixtures
    )

    original_team_loader = (
        v5.load_team_data
    )

    original_xg_loader = (
        v5.load_xg
    )

    try:

        v5.load_team_data = (
            lambda:
                augmented_team.copy()
        )

        v5.load_xg = (
            lambda:
                augmented_xg.copy()
        )

        print(
            "Running frozen V5 component pipeline..."
        )

        components = (
            ov.build_component_store()
        )

    finally:

        v5.load_team_data = (
            original_team_loader
        )

        v5.load_xg = (
            original_xg_loader
        )

    live_ids = set(
        fixtures[
            "match_id"
        ]
    )

    live = components[
        components[
            "match_id"
        ].isin(
            live_ids
        )
    ].copy()

    missing = (
        live_ids
        -
        set(
            live[
                "match_id"
            ]
        )
    )

    if missing:

        raise ValueError(
            "\nFrozen component pipeline did not "
            "produce live components for:\n"
            +
            "\n".join(
                sorted(
                    missing
                )
            )
            +
            "\n\nThis usually means one of the "
            "future placeholder rows was dropped "
            "inside the historical pipeline."
        )

    if live[
        "match_id"
    ].duplicated().any():

        raise ValueError(
            "Frozen component store returned "
            "duplicate live match IDs."
        )

    # ========================================================
    # REPLACE SHADOW COMPONENTS WITH REAL TRANSFER STATES
    # ========================================================

    live = (
        overwrite_transfer_components(
            live,
            fixtures,
        )
    )

    return live


# ============================================================
# LIVE ENGLISH TRANSITION DETECTION
# ============================================================

def build_live_transition_flags(
    fixtures,
):

    print()
    print(
        "Building live promotion / relegation "
        "transition flags..."
    )

    # ========================================================
    # LOAD FOOTYSTATS MULTI-LEAGUE HISTORY
    # ========================================================

    history_file = (
        ROOT
        / "data"
        / "processed"
        / "footystats_multileague_history.csv"
    )

    if not history_file.exists():

        raise FileNotFoundError(
            "FootyStats multi-league history "
            "not found:\n"
            f"{history_file}"
        )

    history = pd.read_csv(
        history_file
    )

    required = [
        "season",
        "league",
        "date",
        "home_team",
        "away_team",
    ]

    missing = [
        col
        for col in required
        if col not in history.columns
    ]

    if missing:

        raise ValueError(
            "FootyStats history missing "
            "transition columns: "
            + str(
                missing
            )
        )

    # ========================================================
    # CLEAN HISTORY
    # ========================================================

    history[
        "date"
    ] = pd.to_datetime(
        history[
            "date"
        ],
        errors="coerce",
    )

    history[
        "season"
    ] = (
        history[
            "season"
        ]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.zfill(4)
    )

    history[
        "league"
    ] = (
        history[
            "league"
        ]
        .astype(str)
        .str.strip()
    )

    history[
        "home_team"
    ] = (
        history[
            "home_team"
        ]
        .astype(str)
        .str.strip()
    )

    history[
        "away_team"
    ] = (
        history[
            "away_team"
        ]
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # CONVERT MATCH HISTORY TO TEAM-SEASON HISTORY
    #
    # FootyStats history is match-level:
    #
    # season | league | home_team | away_team
    #
    # Transition detection needs:
    #
    # team | season | league
    # ========================================================

    home = history[
        [
            "season",
            "league",
            "date",
            "home_team",
        ]
    ].rename(
        columns={
            "home_team":
                "team",
        }
    )

    away = history[
        [
            "season",
            "league",
            "date",
            "away_team",
        ]
    ].rename(
        columns={
            "away_team":
                "team",
        }
    )

    team_history = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    team_history[
        "team"
    ] = (
        team_history[
            "team"
        ]
        .astype(str)
        .str.strip()
    )

    team_history = team_history[
        team_history[
            "date"
        ].notna()
    ].copy()

    # --------------------------------------------------------
    # One row per team / season / league.
    #
    # Keep the latest date simply for auditing.
    # --------------------------------------------------------

    team_seasons = (
        team_history
        .groupby(
            [
                "team",
                "season",
                "league",
            ],
            as_index=False,
        )
        .agg(
            last_date=(
                "date",
                "max",
            )
        )
    )

    # ========================================================
    # LEAGUE PYRAMIDS
    #
    # Only transitions where we possess adjacent-league
    # historical data should be inferred.
    # ========================================================

    league_levels = {

        # England
        "Premier League": 1,
        "Championship": 2,
        "League One": 3,
        "League Two": 4,
        "National League": 5,

        # Germany
        "Bundesliga": 1,
        "2. Bundesliga": 2,

        # Spain
        "La Liga": 1,
        "Segunda División": 2,
    }

    league_countries = {

        "Premier League":
            "England",

        "Championship":
            "England",

        "League One":
            "England",

        "League Two":
            "England",

        "National League":
            "England",

        "Bundesliga":
            "Germany",

        "2. Bundesliga":
            "Germany",

        "La Liga":
            "Spain",

        "Segunda División":
            "Spain",
    }

    # ========================================================
    # CURRENT FIXTURE SEASON
    # ========================================================

    live = fixtures.copy()

    if "season" not in live.columns:

        raise ValueError(
            "Live fixtures missing season column."
        )

    live[
        "season"
    ] = (
        live[
            "season"
        ]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.zfill(4)
    )

    # ========================================================
    # TEAM NAME RESOLUTION
    #
    # Fixtures use canonical live model names while
    # FootyStats may use slightly different aliases.
    #
    # These are identity aliases only. They do NOT encode
    # promotion or relegation status.
    # ========================================================

    history_aliases = {

        # England
        "Brighton and Hove Albion":
            "Brighton",

        "Manchester City":
            "Man City",

        "Manchester United":
            "Man United",

        "Newcastle United":
            "Newcastle",

        "Nottingham Forest":
            "Nott'm Forest",

        "Tottenham Hotspur":
            "Tottenham",

        # Germany
        "1. FC Köln":
            "FC Koln",

        "Bayer Leverkusen":
            "Leverkusen",

        "Borussia Dortmund":
            "Dortmund",

        "Borussia Monchengladbach":
            "M'gladbach",

        "Eintracht Frankfurt":
            "Ein Frankfurt",

        "FSV Mainz 05":
            "Mainz",

        "Hamburger SV":
            "Hamburg",

        "SC Freiburg":
            "Freiburg",

        "SC Paderborn":
            "Paderborn",

        "TSG Hoffenheim":
            "Hoffenheim",

        "VfB Stuttgart":
            "Stuttgart",

        "FC Schalke 04":
            "Schalke 04",
    }

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # First try the fixture name exactly because FootyStats
    # now contains many canonical names directly.
    #
    # Only use the historical V5 alias as a fallback.
    # --------------------------------------------------------

    history_team_set = set(
        team_seasons[
            "team"
        ]
        .dropna()
        .astype(str)
    )

    def resolve_history_name(
        team,
    ):

        team = str(
            team
        ).strip()

        if team in history_team_set:

            return team

        alias = history_aliases.get(
            team
        )

        if (
            alias is not None
            and alias in history_team_set
        ):

            return alias

        return team

    # ========================================================
    # PREVIOUS SEASON
    # ========================================================

    def previous_season(
        season,
    ):

        season = str(
            season
        ).zfill(4)

        start = int(
            season[
                :2
            ]
        )

        end = int(
            season[
                2:
            ]
        )

        return (
            f"{start - 1:02d}"
            f"{end - 1:02d}"
        )

    # ========================================================
    # FIND TEAM'S PREVIOUS LEAGUE
    # ========================================================

    def get_previous_league(
        team,
        season,
        current_league,
    ):

        resolved_team = (
            resolve_history_name(
                team
            )
        )

        prev_season = (
            previous_season(
                season
            )
        )

        candidates = team_seasons[
            (
                team_seasons[
                    "team"
                ]
                ==
                resolved_team
            )
            &
            (
                team_seasons[
                    "season"
                ]
                ==
                prev_season
            )
        ].copy()

        if candidates.empty:

            return (
                resolved_team,
                None,
            )

        current_country = (
            league_countries.get(
                current_league
            )
        )

        if current_country is not None:

            candidates[
                "country"
            ] = (
                candidates[
                    "league"
                ]
                .map(
                    league_countries
                )
            )

            same_country = candidates[
                candidates[
                    "country"
                ]
                ==
                current_country
            ]

            if not same_country.empty:

                candidates = (
                    same_country.copy()
                )

        # ----------------------------------------------------
        # Normally only one league exists for a team-season.
        #
        # If duplicates somehow exist, use the league with
        # the latest recorded match date.
        # ----------------------------------------------------

        candidates = (
            candidates
            .sort_values(
                "last_date"
            )
        )

        previous = str(
            candidates.iloc[
                -1
            ][
                "league"
            ]
        )

        return (
            resolved_team,
            previous,
        )

    # ========================================================
    # BUILD LIVE FLAGS
    # ========================================================

    rows = []

    for fixture in live.itertuples(
        index=False
    ):

        current_league = str(
            fixture.league
        ).strip()

        season = str(
            fixture.season
        ).zfill(4)

        home_team = str(
            fixture.home_team
        ).strip()

        away_team = str(
            fixture.away_team
        ).strip()

        (
            home_history_team,
            home_previous_league,
        ) = get_previous_league(
            home_team,
            season,
            current_league,
        )

        (
            away_history_team,
            away_previous_league,
        ) = get_previous_league(
            away_team,
            season,
            current_league,
        )

        current_level = (
            league_levels.get(
                current_league
            )
        )

        # ====================================================
        # DEFAULT FLAGS
        # ====================================================

        home_promoted = 0
        away_promoted = 0

        home_relegated = 0
        away_relegated = 0

        # ====================================================
        # HOME TRANSITION
        # ====================================================

        if (
            current_level is not None
            and
            home_previous_league
            in league_levels
        ):

            previous_level = (
                league_levels[
                    home_previous_league
                ]
            )

            previous_country = (
                league_countries.get(
                    home_previous_league
                )
            )

            current_country = (
                league_countries.get(
                    current_league
                )
            )

            if (
                previous_country
                ==
                current_country
            ):

                # Previous level numerically larger:
                # Championship (2) -> Premier League (1)
                #
                # Team was promoted.
                if (
                    previous_level
                    >
                    current_level
                ):

                    home_promoted = 1

                # Previous level numerically smaller:
                # Premier League (1) -> Championship (2)
                #
                # Team was relegated.
                elif (
                    previous_level
                    <
                    current_level
                ):

                    home_relegated = 1

        # ====================================================
        # AWAY TRANSITION
        # ====================================================

        if (
            current_level is not None
            and
            away_previous_league
            in league_levels
        ):

            previous_level = (
                league_levels[
                    away_previous_league
                ]
            )

            previous_country = (
                league_countries.get(
                    away_previous_league
                )
            )

            current_country = (
                league_countries.get(
                    current_league
                )
            )

            if (
                previous_country
                ==
                current_country
            ):

                if (
                    previous_level
                    >
                    current_level
                ):

                    away_promoted = 1

                elif (
                    previous_level
                    <
                    current_level
                ):

                    away_relegated = 1

        transition_applied = int(
            home_promoted
            or
            away_promoted
            or
            home_relegated
            or
            away_relegated
        )

        rows.append(
            {
                "match_id":
                    fixture.match_id,

                "league":
                    current_league,

                "home_team":
                    home_team,

                "away_team":
                    away_team,

                "home_history_team":
                    home_history_team,

                "away_history_team":
                    away_history_team,

                "home_previous_league":
                    home_previous_league,

                "away_previous_league":
                    away_previous_league,

                "home_promoted":
                    home_promoted,

                "away_promoted":
                    away_promoted,

                "home_relegated":
                    home_relegated,

                "away_relegated":
                    away_relegated,

                "transition_applied":
                    transition_applied,
            }
        )

    flags = pd.DataFrame(
        rows
    )

    # ========================================================
    # AUDIT
    # ========================================================

    print()
    print("=" * 150)
    print("LIVE TRANSITION DETECTION")
    print("=" * 150)

    print(
        flags[
            [
                "league",
                "home_team",
                "away_team",

                "home_previous_league",
                "away_previous_league",

                "home_promoted",
                "away_promoted",

                "home_relegated",
                "away_relegated",

                "transition_applied",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        "Transition fixtures:",
        int(
            flags[
                "transition_applied"
            ].sum()
        ),
    )

    print(
        "Home promoted:",
        int(
            flags[
                "home_promoted"
            ].sum()
        ),
    )

    print(
        "Away promoted:",
        int(
            flags[
                "away_promoted"
            ].sum()
        ),
    )

    print(
        "Home relegated:",
        int(
            flags[
                "home_relegated"
            ].sum()
        ),
    )

    print(
        "Away relegated:",
        int(
            flags[
                "away_relegated"
            ].sum()
        ),
    )

    return flags


# ============================================================
# APPLY FROZEN ENGLISH TRANSITION ADJUSTMENT
# ============================================================

def apply_live_transition_adjustment(
    fixtures,
    home_lambda,
    away_lambda,
):
    """
    Apply the frozen asymmetric English transition model.

    Promotion:
        own lambda     *= 1 - 0.205
        opponent lambda *= 1 + 0.205

    Relegation:
        own lambda     *= 1 + 0.135
        opponent lambda *= 1 - 0.135

    Adjustment occurs AFTER normal V5 lambda construction
    and BEFORE Poisson probability generation.
    """

    transitions = (
        build_live_transition_flags(
            fixtures
        )
    )

    home = pd.Series(
        np.asarray(
            home_lambda,
            dtype=float,
        ),
        index=fixtures.index,
    )

    away = pd.Series(
        np.asarray(
            away_lambda,
            dtype=float,
        ),
        index=fixtures.index,
    )

    # Preserve original V5 values for auditing.

    raw_home = home.copy()
    raw_away = away.copy()

    # --------------------------------------------------------
    # HOME PROMOTED
    # --------------------------------------------------------

    mask = (
        transitions["home_promoted"]
        .to_numpy(dtype=bool)
    )

    home.loc[mask] *= (
        1.0
        - PROMOTION_ADJUSTMENT
    )

    away.loc[mask] *= (
        1.0
        + PROMOTION_ADJUSTMENT
    )

    # --------------------------------------------------------
    # AWAY PROMOTED
    # --------------------------------------------------------

    mask = (
        transitions["away_promoted"]
        .to_numpy(dtype=bool)
    )

    away.loc[mask] *= (
        1.0
        - PROMOTION_ADJUSTMENT
    )

    home.loc[mask] *= (
        1.0
        + PROMOTION_ADJUSTMENT
    )

    # --------------------------------------------------------
    # HOME RELEGATED
    # --------------------------------------------------------

    mask = (
        transitions["home_relegated"]
        .to_numpy(dtype=bool)
    )

    home.loc[mask] *= (
        1.0
        + RELEGATION_ADJUSTMENT
    )

    away.loc[mask] *= (
        1.0
        - RELEGATION_ADJUSTMENT
    )

    # --------------------------------------------------------
    # AWAY RELEGATED
    # --------------------------------------------------------

    mask = (
        transitions["away_relegated"]
        .to_numpy(dtype=bool)
    )

    away.loc[mask] *= (
        1.0
        + RELEGATION_ADJUSTMENT
    )

    home.loc[mask] *= (
        1.0
        - RELEGATION_ADJUSTMENT
    )

    # Same bounds used by historical transition model.

    home = home.clip(
        lower=0.15,
        upper=4.50,
    )

    away = away.clip(
        lower=0.15,
        upper=4.50,
    )

    audit = transitions.copy()

    audit["home_lambda_v5_raw"] = (
        raw_home.to_numpy()
    )

    audit["away_lambda_v5_raw"] = (
        raw_away.to_numpy()
    )

    audit["home_lambda_v5_transition"] = (
        home.to_numpy()
    )

    audit["away_lambda_v5_transition"] = (
        away.to_numpy()
    )

    audit["transition_applied"] = (
        (
            audit[
                [
                    "home_promoted",
                    "away_promoted",
                    "home_relegated",
                    "away_relegated",
                ]
            ]
            .sum(axis=1)
        )
        > 0
    ).astype(int)

    return (
        home.to_numpy(),
        away.to_numpy(),
        audit,
    )


# ============================================================
# PREDICT LIVE FIXTURES
# ============================================================

def build_predictions(
    fixtures,
    components,
):

    print(
        "Building frozen V5 lambdas..."
    )

    # ========================================================
    # LIVE V5 COMPONENT AUDIT
    # ========================================================

    audit_cols = [
        "match_id",
        "lg_home_goals",
        "lg_away_goals",

        "home_adj_goal_attack",
        "home_adj_goal_defense",
        "home_adj_venue_goal_attack",
        "home_adj_venue_goal_defense",

        "away_adj_goal_attack",
        "away_adj_goal_defense",
        "away_adj_venue_goal_attack",
        "away_adj_venue_goal_defense",

        "home_adj_shot_attack",
        "home_adj_shot_defense",
        "home_adj_venue_shot_attack",
        "home_adj_venue_shot_defense",

        "away_adj_shot_attack",
        "away_adj_shot_defense",
        "away_adj_venue_shot_attack",
        "away_adj_venue_shot_defense",

        "home_xg_attack_overall",
        "home_xg_defense_overall",
        "home_xg_attack_venue",
        "home_xg_defense_venue",

        "away_xg_attack_overall",
        "away_xg_defense_overall",
        "away_xg_attack_venue",
        "away_xg_defense_venue",
    ]

    missing_audit_cols = [
        col
        for col in audit_cols
        if col not in components.columns
    ]

    if missing_audit_cols:

        raise ValueError(
            "Missing component audit columns:\n"
            +
            "\n".join(
                missing_audit_cols
            )
        )

    component_audit = fixtures[
        [
            "match_id",
            "date",
            "league",
            "home_team",
            "away_team",
        ]
    ].merge(
        components[
            audit_cols
        ],
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    print()
    print("=" * 160)
    print("LIVE V5 COMPONENT AUDIT")
    print("=" * 160)

    print(
        component_audit
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # RAW FROZEN V5 LAMBDAS
    # ========================================================

    (
        home_lambda_raw,
        away_lambda_raw,
    ) = ov.build_lambdas(
        components,
        OVERALL_WEIGHT,
    )

    # --------------------------------------------------------
    # Preserve component order.
    #
    # build_lambdas() returns values in component-row order.
    # Align fixture identity to that same order before
    # building transition audit metadata.
    # --------------------------------------------------------

    ordered_fixtures = components[
        [
            "match_id",
        ]
    ].merge(
        fixtures,
        on="match_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(
        ordered_fixtures
    ) != len(
        components
    ):

        raise ValueError(
            "Could not align live fixtures "
            "to V5 component rows."
        )

    # ========================================================
    # TRANSITION DETECTION ONLY
    #
    # The old V4 promotion / relegation lambda modifier is
    # intentionally disabled for V5 deployment.
    #
    # Transition status is still retained for auditing.
    # ========================================================

    transition_audit = (
        build_live_transition_flags(
            ordered_fixtures
        )
    )

    home_lambda = np.asarray(
        home_lambda_raw,
        dtype=float,
    ).copy()

    away_lambda = np.asarray(
        away_lambda_raw,
        dtype=float,
    ).copy()

    # --------------------------------------------------------
    # Preserve raw/final values in transition audit.
    #
    # Since the transition modifier is disabled, raw and final
    # lambdas MUST be identical.
    # --------------------------------------------------------

    transition_audit[
        "home_lambda_v5_raw"
    ] = home_lambda.copy()

    transition_audit[
        "away_lambda_v5_raw"
    ] = away_lambda.copy()

    transition_audit[
        "home_lambda_v5_transition"
    ] = home_lambda.copy()

    transition_audit[
        "away_lambda_v5_transition"
    ] = away_lambda.copy()

    transition_count = int(
        transition_audit[
            "transition_applied"
        ].sum()
    )

    print()

    print(
        "Transition fixtures detected:",
        transition_count,
    )

    print(
        "V5 transition lambda adjustment: "
        "DISABLED ✅"
    )

    if transition_count:

        print()
        print("=" * 150)
        print(
            "TRANSITION DETECTION AUDIT "
            "— NO LAMBDA MODIFIER"
        )
        print("=" * 150)

        print(
            transition_audit.loc[
                transition_audit[
                    "transition_applied"
                ]
                == 1,
                [
                    "league",
                    "home_team",
                    "away_team",

                    "home_previous_league",
                    "away_previous_league",

                    "home_promoted",
                    "away_promoted",

                    "home_relegated",
                    "away_relegated",

                    "home_lambda_v5_raw",
                    "away_lambda_v5_raw",

                    "home_lambda_v5_transition",
                    "away_lambda_v5_transition",
                ],
            ]
            .round(4)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # POISSON 1X2 PROBABILITIES
    #
    # Probabilities are now generated directly from the raw
    # frozen V5 lambdas.
    # ========================================================

    probabilities = (
        ov.calculate_1x2_probs(
            np.asarray(
                home_lambda,
                dtype=float,
            ),
            np.asarray(
                away_lambda,
                dtype=float,
            ),
        )
    )

    # ========================================================
    # BASE OUTPUT
    # ========================================================

    output = components[
        [
            "match_id",
        ]
    ].copy()

    # --------------------------------------------------------
    # Final deployment lambdas
    # --------------------------------------------------------

    output[
        "home_lambda_v5"
    ] = np.asarray(
        home_lambda,
        dtype=float,
    )

    output[
        "away_lambda_v5"
    ] = np.asarray(
        away_lambda,
        dtype=float,
    )

    # --------------------------------------------------------
    # Raw frozen V5 lambdas
    # --------------------------------------------------------

    output[
        "home_lambda_v5_raw"
    ] = np.asarray(
        home_lambda_raw,
        dtype=float,
    )

    output[
        "away_lambda_v5_raw"
    ] = np.asarray(
        away_lambda_raw,
        dtype=float,
    )

    # --------------------------------------------------------
    # 1X2 probabilities
    # --------------------------------------------------------

    output[
        "p_home_v5"
    ] = probabilities[
        :,
        0
    ]

    output[
        "p_draw_v5"
    ] = probabilities[
        :,
        1
    ]

    output[
        "p_away_v5"
    ] = probabilities[
        :,
        2
    ]

    # --------------------------------------------------------
    # O/U 2.5 probabilities
    #
    # Independent Poisson home/away goals imply total goals
    # are Poisson with lambda = home_lambda + away_lambda.
    # These use the same final frozen V5 deployment lambdas
    # as the 1X2 probabilities above.
    # --------------------------------------------------------

    total_lambda = (
        output["home_lambda_v5"]
        +
        output["away_lambda_v5"]
    )

    output["p_under_2_5_v5"] = (
        np.exp(-total_lambda)
        *
        (
            1.0
            +
            total_lambda
            +
            (total_lambda ** 2) / 2.0
        )
    )

    output["p_over_2_5_v5"] = (
        1.0
        -
        output["p_under_2_5_v5"]
    )

    # ========================================================
    # TRANSITION AUDIT METADATA
    # ========================================================

    transition_cols = [
        "match_id",

        "home_previous_league",
        "away_previous_league",

        "home_promoted",
        "away_promoted",

        "home_relegated",
        "away_relegated",

        "transition_applied",
    ]

    output = output.merge(
        transition_audit[
            transition_cols
        ],
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Active deployment settings.
    #
    # These are zero because transition lambda modification is
    # disabled.
    # --------------------------------------------------------

    output[
        "promotion_adjustment"
    ] = 0.0

    output[
        "relegation_adjustment"
    ] = 0.0

    output[
        "transition_adjustment_enabled"
    ] = 0

    # ========================================================
    # FIXTURE IDENTITY
    # ========================================================

    fixture_cols = [
        "match_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
    ]

    output = fixtures[
        fixture_cols
    ].merge(
        output,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # MODEL METADATA
    # ========================================================

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
        "goal_recency_v5"
    ] = GOAL_RECENCY

    output[
        "xg_recency_v5"
    ] = XG_RECENCY

    output[
        "shot_recency_v5"
    ] = SHOT_RECENCY

    output[
        "opponent_strength_v5"
    ] = OPPONENT_STRENGTH

    output[
        "overall_weight_v5"
    ] = OVERALL_WEIGHT

    output[
        "venue_weight_v5"
    ] = VENUE_WEIGHT

    # ========================================================
    # INTEGRITY — PROBABILITIES
    # ========================================================

    probability_sum = (
        output[
            [
                "p_home_v5",
                "p_draw_v5",
                "p_away_v5",
            ]
        ]
        .sum(
            axis=1
        )
    )

    if not np.allclose(
        probability_sum,
        1.0,
        atol=1e-8,
    ):

        raise ValueError(
            "Live V5 probabilities do not "
            "sum to 1."
        )

    # ========================================================
    # INTEGRITY — REQUIRED NUMERIC VALUES
    # ========================================================

    required_numeric = [
        "home_lambda_v5",
        "away_lambda_v5",

        "home_lambda_v5_raw",
        "away_lambda_v5_raw",

        "p_home_v5",
        "p_draw_v5",
        "p_away_v5",

        "p_over_2_5_v5",
        "p_under_2_5_v5",
    ]

    if output[
        required_numeric
    ].isna().any().any():

        bad = output[
            output[
                required_numeric
            ]
            .isna()
            .any(
                axis=1
            )
        ]

        raise ValueError(
            "Missing live V5 values:\n"
            +
            bad.to_string(
                index=False
            )
        )

    # ========================================================
    # INTEGRITY — ALL LIVE LAMBDAS MUST EQUAL RAW V5
    #
    # Transition detection is informational only.
    # ========================================================

    if not np.allclose(
        output[
            "home_lambda_v5"
        ],
        output[
            "home_lambda_v5_raw"
        ],
        atol=1e-12,
    ):

        raise ValueError(
            "Home lambda differs from raw frozen V5 "
            "while transition adjustment is disabled."
        )

    if not np.allclose(
        output[
            "away_lambda_v5"
        ],
        output[
            "away_lambda_v5_raw"
        ],
        atol=1e-12,
    ):

        raise ValueError(
            "Away lambda differs from raw frozen V5 "
            "while transition adjustment is disabled."
        )

    # ========================================================
    # FINAL AUDIT
    # ========================================================

    transition_rows = output.loc[
        output[
            "transition_applied"
        ]
        .fillna(0)
        .astype(int)
        == 1
    ]

    print()
    print(
        "Transition integrity:"
    )

    print(
        "Transition fixtures detected:",
        len(
            transition_rows
        ),
    )

    print(
        "All live fixtures use raw frozen V5 lambdas:",
        len(
            output
        ),
        "✅",
    )

    print(
        "Transition lambda modifier enabled:",
        "NO ✅",
    )

    return output

# ============================================================
# DISPLAY
# ============================================================

def print_predictions(
    output,
):

    display = output[
        [
            "date",
            "league",
            "home_team",
            "away_team",
            "home_lambda_v5",
            "away_lambda_v5",
            "p_home_v5",
            "p_draw_v5",
            "p_away_v5",
        ]
    ].copy()

    for col in [
        "p_home_v5",
        "p_draw_v5",
        "p_away_v5",
    ]:

        display[
            col
        ] *= 100.0

    print()
    print("=" * 120)
    print("LIVE V5 PREDICTIONS")
    print("=" * 120)

    print(
        display
        .round(
            3
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("BUILDING LIVE V5 PREDICTIONS")
    print("==============================")
    print()

    print(
        "Frozen signal weights:"
    )

    print(
        "Goals: 9%"
    )

    print(
        "xG: 75%"
    )

    print(
        "Shots: 16%"
    )

    print()

    print(
        "Frozen recencies:"
    )

    print(
        "Goals: 0.975"
    )

    print(
        "xG: 0.925"
    )

    print(
        "Shots: 0.850"
    )

    print()

    print(
        "Opponent strength: 0.875"
    )

    print(
        "Overall / venue: 80% / 20%"
    )

    # ========================================================
    # SETTINGS
    # ========================================================

    validate_frozen_settings()

    print()
    print(
        "Frozen settings validated ✅"
    )

    # ========================================================
    # FIXTURES
    # ========================================================

    fixtures = load_fixtures()

    print()
    print(
        f"Upcoming fixtures: "
        f"{len(fixtures):,}"
    )

    print(
        f"Date range: "
        f"{fixtures['date'].min().date()} "
        f"-> "
        f"{fixtures['date'].max().date()}"
    )

    print()
    print(
        "Fixtures by league:"
    )

    print(
        fixtures[
            "league"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # COMPONENTS
    # ========================================================

    components = build_live_components(
        fixtures
    )

    print(
        f"Live component rows: "
        f"{len(components):,}"
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    output = build_predictions(
        fixtures,
        components,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print_predictions(
        output
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("==============================")
    print("LIVE V5 BUILD COMPLETE")
    print("==============================")

    print(
        "Historical V5 functions reused ✅"
    )

    print(
        "Future fixture rows contain "
        "no match outcomes ✅"
    )

    print(
        "Goals / xG / shots remain "
        "prior-only ✅"
    )

    print(
        "Frozen signal weights unchanged ✅"
    )

    print(
        "Frozen recencies unchanged ✅"
    )

    print(
        "Opponent strength unchanged ✅"
    )

    print(
        "80% / 20% overall-venue blend ✅"
    )

    print(
        "Poisson 1X2 engine unchanged ✅"
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
    