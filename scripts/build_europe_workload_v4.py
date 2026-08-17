from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DOMESTIC_FILE = (
    ROOT
    / "data"
    / "processed"
    / "team_game_stats.csv"
)

EUROPE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "european_fixtures.csv"
)

MAP_FILE = (
    ROOT
    / "data"
    / "processed"
    / "europe_team_map.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "europe_workload_v4.csv"
)


# ============================================================
# SETTINGS
# ============================================================

WINDOWS = [
    3,
    4,
    5,
    7,
    14,
]


# ============================================================
# BUILD EUROPE TEAM EVENTS
# ============================================================

def build_europe_events(
    europe,
    team_map,
):

    mapping = (
        team_map[
            [
                "europe_team",
                "domestic_team",
            ]
        ]
        .drop_duplicates()
        .set_index(
            "europe_team"
        )[
            "domestic_team"
        ]
        .to_dict()
    )

    rows = []

    for _, row in europe.iterrows():

        home_domestic = mapping.get(
            row[
                "home_team"
            ]
        )

        away_domestic = mapping.get(
            row[
                "away_team"
            ]
        )

        if home_domestic is not None:

            rows.append(
                {
                    "team":
                        home_domestic,

                    "date":
                        row[
                            "date"
                        ],

                    "event_type":
                        "EUROPE",

                    "competition":
                        row[
                            "competition"
                        ],

                    "event_venue":
                        "HOME",

                    "europe_opponent":
                        row[
                            "away_team"
                        ],
                }
            )

        if away_domestic is not None:

            rows.append(
                {
                    "team":
                        away_domestic,

                    "date":
                        row[
                            "date"
                        ],

                    "event_type":
                        "EUROPE",

                    "competition":
                        row[
                            "competition"
                        ],

                    "event_venue":
                        "AWAY",

                    "europe_opponent":
                        row[
                            "home_team"
                        ],
                }
            )

    events = pd.DataFrame(
        rows
    )

    if len(events) == 0:

        raise RuntimeError(
            "No European events matched "
            "to domestic teams."
        )

    events[
        "date"
    ] = pd.to_datetime(
        events[
            "date"
        ]
    )

    events = (
        events
        .drop_duplicates(
            subset=[
                "team",
                "date",
                "competition",
                "event_venue",
                "europe_opponent",
            ]
        )
        .sort_values(
            [
                "team",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return events


# ============================================================
# DOMESTIC TEAM EVENTS
# ============================================================

def build_domestic_events(
    domestic,
):

    events = domestic[
        [
            "match_id",
            "date",
            "season",
            "league",
            "team",
            "opponent",
            "venue",
        ]
    ].copy()

    events[
        "event_type"
    ] = "DOMESTIC"

    events[
        "competition"
    ] = events[
        "league"
    ]

    events[
        "event_venue"
    ] = events[
        "venue"
    ]

    events[
        "europe_opponent"
    ] = np.nan

    return events


# ============================================================
# TEAM WORKLOAD FEATURES
# ============================================================

def build_team_workload(
    domestic,
    europe_events,
):

    domestic = (
        domestic
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

    europe_by_team = {}

    for team, group in europe_events.groupby(
        "team",
        sort=False,
    ):

        europe_by_team[
            team
        ] = (
            group
            .sort_values(
                "date"
            )
            .reset_index(
                drop=True
            )
        )

    domestic_by_team = {}

    for team, group in domestic.groupby(
        "team",
        sort=False,
    ):

        domestic_by_team[
            team
        ] = (
            group
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

    rows = []

    for team, team_domestic in (
        domestic_by_team.items()
    ):

        team_europe = europe_by_team.get(
            team,
            pd.DataFrame(
                columns=[
                    "date",
                    "competition",
                    "event_venue",
                    "europe_opponent",
                ]
            ),
        )

        domestic_dates = (
            team_domestic[
                "date"
            ]
            .to_numpy(
                dtype="datetime64[D]"
            )
        )

        europe_dates = (
            team_europe[
                "date"
            ]
            .to_numpy(
                dtype="datetime64[D]"
            )
        )

        for i, row in (
            team_domestic.iterrows()
        ):

            current_date = np.datetime64(
                row[
                    "date"
                ].date()
            )

            # ------------------------------------------------
            # PREVIOUS DOMESTIC MATCH
            # ------------------------------------------------

            previous_domestic_date = (
                domestic_dates[
                    i - 1
                ]
                if i > 0
                else None
            )

            if (
                previous_domestic_date
                is not None
            ):

                days_since_domestic = int(
                    (
                        current_date
                        -
                        previous_domestic_date
                    ).astype(int)
                )

            else:

                days_since_domestic = (
                    np.nan
                )

            # ------------------------------------------------
            # PREVIOUS EUROPE MATCH
            # ------------------------------------------------

            prior_euro_mask = (
                europe_dates
                <
                current_date
            )

            prior_euro_indices = np.where(
                prior_euro_mask
            )[0]

            if len(
                prior_euro_indices
            ) > 0:

                last_euro_index = (
                    prior_euro_indices[
                        -1
                    ]
                )

                last_euro = (
                    team_europe.iloc[
                        last_euro_index
                    ]
                )

                last_euro_date = (
                    europe_dates[
                        last_euro_index
                    ]
                )

                days_since_europe = int(
                    (
                        current_date
                        -
                        last_euro_date
                    ).astype(int)
                )

                last_europe_competition = (
                    last_euro[
                        "competition"
                    ]
                )

                last_europe_venue = (
                    last_euro[
                        "event_venue"
                    ]
                )

                last_europe_opponent = (
                    last_euro[
                        "europe_opponent"
                    ]
                )

            else:

                days_since_europe = (
                    np.nan
                )

                last_europe_competition = (
                    np.nan
                )

                last_europe_venue = (
                    np.nan
                )

                last_europe_opponent = (
                    np.nan
                )

            # ------------------------------------------------
            # PREVIOUS ANY TRACKED MATCH
            #
            # League + Europe only.
            # ------------------------------------------------

            previous_dates = []

            if (
                previous_domestic_date
                is not None
            ):

                previous_dates.append(
                    previous_domestic_date
                )

            if len(
                prior_euro_indices
            ) > 0:

                previous_dates.append(
                    europe_dates[
                        prior_euro_indices[
                            -1
                        ]
                    ]
                )

            if previous_dates:

                previous_any = max(
                    previous_dates
                )

                days_since_any = int(
                    (
                        current_date
                        -
                        previous_any
                    ).astype(int)
                )

            else:

                days_since_any = (
                    np.nan
                )

            # ------------------------------------------------
            # WINDOW COUNTS
            # ------------------------------------------------

            features = {}

            for window in WINDOWS:

                lower = (
                    current_date
                    -
                    np.timedelta64(
                        window,
                        "D",
                    )
                )

                # Prior domestic events.
                domestic_count = int(
                    (
                        (
                            domestic_dates
                            <
                            current_date
                        )
                        &
                        (
                            domestic_dates
                            >=
                            lower
                        )
                    ).sum()
                )

                # Prior Europe events.
                europe_count = int(
                    (
                        (
                            europe_dates
                            <
                            current_date
                        )
                        &
                        (
                            europe_dates
                            >=
                            lower
                        )
                    ).sum()
                )

                features[
                    f"domestic_matches_last_{window}d"
                ] = (
                    domestic_count
                )

                features[
                    f"europe_matches_last_{window}d"
                ] = (
                    europe_count
                )

                features[
                    f"tracked_matches_last_{window}d"
                ] = (
                    domestic_count
                    +
                    europe_count
                )

            # ------------------------------------------------
            # EUROPE TURNAROUND FLAGS
            # ------------------------------------------------

            played_europe_last_3d = int(
                pd.notna(
                    days_since_europe
                )
                and
                days_since_europe
                <= 3
            )

            played_europe_last_4d = int(
                pd.notna(
                    days_since_europe
                )
                and
                days_since_europe
                <= 4
            )

            played_europe_last_5d = int(
                pd.notna(
                    days_since_europe
                )
                and
                days_since_europe
                <= 5
            )

            europe_away_last_4d = int(
                played_europe_last_4d
                == 1
                and
                last_europe_venue
                == "AWAY"
            )

            europe_home_last_4d = int(
                played_europe_last_4d
                == 1
                and
                last_europe_venue
                == "HOME"
            )

            rows.append(
                {
                    "match_id":
                        row[
                            "match_id"
                        ],

                    "date":
                        row[
                            "date"
                        ],

                    "season":
                        row[
                            "season"
                        ],

                    "league":
                        row[
                            "league"
                        ],

                    "team":
                        team,

                    "opponent":
                        row[
                            "opponent"
                        ],

                    "venue":
                        row[
                            "venue"
                        ],

                    "days_since_domestic":
                        days_since_domestic,

                    "days_since_europe":
                        days_since_europe,

                    "days_since_tracked_match":
                        days_since_any,

                    "last_europe_competition":
                        last_europe_competition,

                    "last_europe_venue":
                        last_europe_venue,

                    "last_europe_opponent":
                        last_europe_opponent,

                    "played_europe_last_3d":
                        played_europe_last_3d,

                    "played_europe_last_4d":
                        played_europe_last_4d,

                    "played_europe_last_5d":
                        played_europe_last_5d,

                    "europe_away_last_4d":
                        europe_away_last_4d,

                    "europe_home_last_4d":
                        europe_home_last_4d,

                    **features,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MATCH-LEVEL FEATURES
# ============================================================

def build_match_table(
    team_workload,
):

    feature_cols = [
        "days_since_domestic",
        "days_since_europe",
        "days_since_tracked_match",

        "played_europe_last_3d",
        "played_europe_last_4d",
        "played_europe_last_5d",

        "europe_away_last_4d",
        "europe_home_last_4d",

        "domestic_matches_last_3d",
        "europe_matches_last_3d",
        "tracked_matches_last_3d",

        "domestic_matches_last_4d",
        "europe_matches_last_4d",
        "tracked_matches_last_4d",

        "domestic_matches_last_5d",
        "europe_matches_last_5d",
        "tracked_matches_last_5d",

        "domestic_matches_last_7d",
        "europe_matches_last_7d",
        "tracked_matches_last_7d",

        "domestic_matches_last_14d",
        "europe_matches_last_14d",
        "tracked_matches_last_14d",
    ]

    home = team_workload[
        team_workload[
            "venue"
        ]
        == "HOME"
    ].copy()

    away = team_workload[
        team_workload[
            "venue"
        ]
        == "AWAY"
    ].copy()

    home_keep = [
        "match_id",
        "date",
        "season",
        "league",
        "team",
        "opponent",

        "last_europe_competition",
        "last_europe_venue",
        "last_europe_opponent",
    ] + feature_cols

    away_keep = [
        "match_id",
        "team",
        "opponent",

        "last_europe_competition",
        "last_europe_venue",
        "last_europe_opponent",
    ] + feature_cols

    home = home[
        home_keep
    ].copy()

    away = away[
        away_keep
    ].copy()

    home = home.rename(
        columns={
            "team":
                "home_team",

            "opponent":
                "away_team_check",

            "last_europe_competition":
                "home_last_europe_competition",

            "last_europe_venue":
                "home_last_europe_venue",

            "last_europe_opponent":
                "home_last_europe_opponent",
        }
    )

    away = away.rename(
        columns={
            "team":
                "away_team",

            "opponent":
                "home_team_check",

            "last_europe_competition":
                "away_last_europe_competition",

            "last_europe_venue":
                "away_last_europe_venue",

            "last_europe_opponent":
                "away_last_europe_opponent",
        }
    )

    for col in feature_cols:

        home = home.rename(
            columns={
                col:
                    f"home_{col}"
            }
        )

        away = away.rename(
            columns={
                col:
                    f"away_{col}"
            }
        )

    matches = home.merge(
        away,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # RELATIVE FEATURES
    # --------------------------------------------------------

    matches[
        "home_europe_turnaround_advantage_4d"
    ] = (
        matches[
            "away_played_europe_last_4d"
        ]
        -
        matches[
            "home_played_europe_last_4d"
        ]
    )

    matches[
        "home_europe_away_trip_advantage_4d"
    ] = (
        matches[
            "away_europe_away_last_4d"
        ]
        -
        matches[
            "home_europe_away_last_4d"
        ]
    )

    matches[
        "home_tracked_matches_7d_advantage"
    ] = (
        matches[
            "away_tracked_matches_last_7d"
        ]
        -
        matches[
            "home_tracked_matches_last_7d"
        ]
    )

    matches[
        "home_tracked_matches_14d_advantage"
    ] = (
        matches[
            "away_tracked_matches_last_14d"
        ]
        -
        matches[
            "home_tracked_matches_last_14d"
        ]
    )

    return matches


# ============================================================
# VALIDATION
# ============================================================

def validate(
    domestic,
    europe_events,
    match_table,
):

    expected_matches = (
        domestic[
            "match_id"
        ]
        .nunique()
    )

    if (
        len(
            match_table
        )
        != expected_matches
    ):

        raise RuntimeError(
            "Workload table does not have "
            "exactly one row per domestic match."
        )

    if (
        match_table[
            "match_id"
        ].duplicated().any()
    ):

        raise RuntimeError(
            "Duplicate domestic match IDs "
            "in workload table."
        )

    # European dates used as prior events
    # must never produce zero or negative
    # days-since-Europe.
    euro_days = pd.concat(
        [
            match_table[
                "home_days_since_europe"
            ],
            match_table[
                "away_days_since_europe"
            ],
        ]
    ).dropna()

    if (
        euro_days
        <= 0
    ).any():

        raise RuntimeError(
            "Current/future European match "
            "entered pregame workload."
        )

    print(
        "One workload row per "
        "domestic match ✅"
    )

    print(
        "European events use only "
        "strictly earlier dates ✅"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("BUILDING EUROPE WORKLOAD V4")
    print("==============================")
    print()

    domestic = pd.read_csv(
        DOMESTIC_FILE,
        parse_dates=[
            "date",
        ],
    )

    europe = pd.read_csv(
        EUROPE_FILE,
        parse_dates=[
            "date",
        ],
    )

    team_map = pd.read_csv(
        MAP_FILE,
    )

    domestic[
        "season"
    ] = (
        domestic[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    print(
        f"Domestic team rows: "
        f"{len(domestic):,}"
    )

    print(
        f"European fixtures: "
        f"{len(europe):,}"
    )

    print(
        f"Accepted European aliases: "
        f"{len(team_map):,}"
    )

    # ========================================================
    # EUROPE EVENTS
    # ========================================================

    print()
    print(
        "Building mapped European "
        "team events..."
    )

    europe_events = (
        build_europe_events(
            europe,
            team_map,
        )
    )

    print(
        f"Mapped European "
        f"team-events: "
        f"{len(europe_events):,}"
    )

    print(
        f"Domestic clubs with "
        f"European events: "
        f"{europe_events['team'].nunique():,}"
    )

    # ========================================================
    # TEAM WORKLOAD
    # ========================================================

    print(
        "Building leakage-safe "
        "league + Europe workloads..."
    )

    team_workload = (
        build_team_workload(
            domestic,
            europe_events,
        )
    )

    print(
        "Building domestic "
        "match-level features..."
    )

    matches = build_match_table(
        team_workload
    )

    print(
        "Running validation checks..."
    )

    validate(
        domestic,
        europe_events,
        matches,
    )

    # ========================================================
    # SAVE
    # ========================================================

    matches.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("==============================")
    print("EUROPE WORKLOAD COMPLETE")
    print("==============================")
    print()

    print(
        f"Domestic matches: "
        f"{len(matches):,}"
    )

    print()

    # --------------------------------------------------------
    # EUROPE TURNAROUND RATES
    # --------------------------------------------------------

    home_3 = (
        matches[
            "home_played_europe_last_3d"
        ].sum()
    )

    away_3 = (
        matches[
            "away_played_europe_last_3d"
        ].sum()
    )

    home_4 = (
        matches[
            "home_played_europe_last_4d"
        ].sum()
    )

    away_4 = (
        matches[
            "away_played_europe_last_4d"
        ].sum()
    )

    print(
        "EUROPE TURNAROUND COUNTS"
    )

    print(
        f"Home team Europe <=3d: "
        f"{home_3:,}"
    )

    print(
        f"Away team Europe <=3d: "
        f"{away_3:,}"
    )

    print(
        f"Home team Europe <=4d: "
        f"{home_4:,}"
    )

    print(
        f"Away team Europe <=4d: "
        f"{away_4:,}"
    )

    # --------------------------------------------------------
    # AWAY EUROPE
    # --------------------------------------------------------

    print()
    print(
        "EUROPE AWAY TRIP <=4 DAYS"
    )

    print(
        "Home domestic team: "
        f"{matches['home_europe_away_last_4d'].sum():,}"
    )

    print(
        "Away domestic team: "
        f"{matches['away_europe_away_last_4d'].sum():,}"
    )

    # --------------------------------------------------------
    # BY LEAGUE
    # --------------------------------------------------------

    print()
    print(
        "EUROPE TURNAROUND BY LEAGUE"
    )

    league_rows = []

    for league, group in (
        matches.groupby(
            "league"
        )
    ):

        league_rows.append(
            {
                "league":
                    league,

                "games":
                    len(
                        group
                    ),

                "home_europe_4d":
                    group[
                        "home_played_europe_last_4d"
                    ].sum(),

                "away_europe_4d":
                    group[
                        "away_played_europe_last_4d"
                    ].sum(),

                "home_europe_away_trip_4d":
                    group[
                        "home_europe_away_last_4d"
                    ].sum(),

                "away_europe_away_trip_4d":
                    group[
                        "away_europe_away_last_4d"
                    ].sum(),
            }
        )

    league_table = pd.DataFrame(
        league_rows
    )

    print(
        league_table.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    print()
    print(
        "SAMPLE DOMESTIC MATCHES "
        "AFTER EUROPE"
    )

    sample = matches[
        (
            matches[
                "home_played_europe_last_4d"
            ]
            == 1
        )
        |
        (
            matches[
                "away_played_europe_last_4d"
            ]
            == 1
        )
    ].copy()

    sample = (
        sample
        .sort_values(
            "date"
        )
        .tail(30)
    )

    print(
        sample[
            [
                "date",
                "season",
                "league",
                "home_team",
                "away_team",

                "home_days_since_europe",
                "home_last_europe_competition",
                "home_last_europe_venue",

                "away_days_since_europe",
                "away_last_europe_competition",
                "away_last_europe_venue",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "IMPORTANT COVERAGE NOTE"
    )

    print(
        "2021/22–2024/25 has "
        "Champions + Europa + Conference coverage."
    )

    print(
        "Earlier seasons have partial "
        "European competition coverage."
    )

    print(
        "2025/26 currently has "
        "Champions League only."
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