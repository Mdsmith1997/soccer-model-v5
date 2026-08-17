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
    / "rest_features_v4.csv"
)


# =========================================================
# SETTINGS
# =========================================================

MAX_REST_DAYS = 30

SHORT_REST_DAYS = 3

NORMAL_REST_LOW = 4
NORMAL_REST_HIGH = 8

LONG_REST_DAYS = 9


# =========================================================
# REST FEATURES
# =========================================================

def build_team_rest_features(
    df,
):
    """
    Build leakage-safe pregame rest/congestion features.

    All features use only match dates prior to the
    current match.
    """

    out = df.copy()

    out = (
        out
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

    team_group = out.groupby(
        "team",
        sort=False,
        group_keys=False,
    )

    # -----------------------------------------------------
    # PREVIOUS MATCH DATE
    # -----------------------------------------------------

    out[
        "previous_match_date"
    ] = (
        team_group[
            "date"
        ]
        .shift(1)
    )

    out[
        "days_since_last_match"
    ] = (
        out[
            "date"
        ]
        -
        out[
            "previous_match_date"
        ]
    ).dt.days

    # Cap very large offseason gaps.
    out[
        "rest_days_capped"
    ] = (
        out[
            "days_since_last_match"
        ]
        .clip(
            upper=MAX_REST_DAYS
        )
    )

    # -----------------------------------------------------
    # REST BUCKETS
    # -----------------------------------------------------

    out[
        "rest_3_or_less"
    ] = (
        out[
            "days_since_last_match"
        ]
        <= SHORT_REST_DAYS
    ).astype(int)

    out[
        "rest_4_to_5"
    ] = (
        out[
            "days_since_last_match"
        ]
        .between(
            4,
            5,
        )
    ).astype(int)

    out[
        "rest_6_to_8"
    ] = (
        out[
            "days_since_last_match"
        ]
        .between(
            6,
            8,
        )
    ).astype(int)

    out[
        "rest_9_plus"
    ] = (
        out[
            "days_since_last_match"
        ]
        >= LONG_REST_DAYS
    ).astype(int)

    # -----------------------------------------------------
    # MATCH COUNTS IN PRIOR WINDOWS
    #
    # Important:
    # Current match is excluded.
    # -----------------------------------------------------

    out[
        "matches_last_7d"
    ] = 0

    out[
        "matches_last_14d"
    ] = 0

    out[
        "matches_last_21d"
    ] = 0

    for team, group in out.groupby(
        "team",
        sort=False,
    ):

        indices = group.index.to_numpy()

        dates = (
            group[
                "date"
            ]
            .to_numpy(
                dtype="datetime64[D]"
            )
        )

        counts_7 = np.zeros(
            len(group),
            dtype=int,
        )

        counts_14 = np.zeros(
            len(group),
            dtype=int,
        )

        counts_21 = np.zeros(
            len(group),
            dtype=int,
        )

        left_7 = 0
        left_14 = 0
        left_21 = 0

        for i in range(
            len(group)
        ):

            current_date = dates[i]

            while (
                left_7 < i
                and (
                    current_date
                    - dates[
                        left_7
                    ]
                ).astype(int)
                > 7
            ):
                left_7 += 1

            while (
                left_14 < i
                and (
                    current_date
                    - dates[
                        left_14
                    ]
                ).astype(int)
                > 14
            ):
                left_14 += 1

            while (
                left_21 < i
                and (
                    current_date
                    - dates[
                        left_21
                    ]
                ).astype(int)
                > 21
            ):
                left_21 += 1

            counts_7[i] = (
                i - left_7
            )

            counts_14[i] = (
                i - left_14
            )

            counts_21[i] = (
                i - left_21
            )

        out.loc[
            indices,
            "matches_last_7d",
        ] = counts_7

        out.loc[
            indices,
            "matches_last_14d",
        ] = counts_14

        out.loc[
            indices,
            "matches_last_21d",
        ] = counts_21

    # -----------------------------------------------------
    # CONGESTION FLAGS
    # -----------------------------------------------------

    out[
        "two_plus_matches_last_7d"
    ] = (
        out[
            "matches_last_7d"
        ]
        >= 2
    ).astype(int)

    out[
        "three_plus_matches_last_14d"
    ] = (
        out[
            "matches_last_14d"
        ]
        >= 3
    ).astype(int)

    out[
        "four_plus_matches_last_14d"
    ] = (
        out[
            "matches_last_14d"
        ]
        >= 4
    ).astype(int)

    return out


# =========================================================
# MATCH-LEVEL REST FEATURES
# =========================================================

def build_match_rest_table(
    df,
):

    feature_cols = [
        "days_since_last_match",
        "rest_days_capped",

        "rest_3_or_less",
        "rest_4_to_5",
        "rest_6_to_8",
        "rest_9_plus",

        "matches_last_7d",
        "matches_last_14d",
        "matches_last_21d",

        "two_plus_matches_last_7d",
        "three_plus_matches_last_14d",
        "four_plus_matches_last_14d",
    ]

    home = (
        df[
            df[
                "venue"
            ]
            == "HOME"
        ]
        .copy()
    )

    away = (
        df[
            df[
                "venue"
            ]
            == "AWAY"
        ]
        .copy()
    )

    home = home[
        [
            "match_id",
            "date",
            "season",
            "league",
            "home_team",
            "away_team",
        ]
        if (
            "home_team" in home.columns
            and "away_team" in home.columns
        )
        else
        [
            "match_id",
            "date",
            "season",
            "league",
            "team",
            "opponent",
        ]
        +
        feature_cols
    ].copy()

    if "team" in home.columns:

        home = home.rename(
            columns={
                "team":
                    "home_team",

                "opponent":
                    "away_team",
            }
        )

    home_rename = {}

    for col in feature_cols:

        home_rename[
            col
        ] = (
            "home_"
            + col
        )

    home = home.rename(
        columns=home_rename
    )

    away = away[
        [
            "match_id",
            "team",
            "opponent",
        ]
        +
        feature_cols
    ].copy()

    away = away.rename(
        columns={
            "team":
                "away_team_check",

            "opponent":
                "home_team_check",
        }
    )

    away_rename = {}

    for col in feature_cols:

        away_rename[
            col
        ] = (
            "away_"
            + col
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

    # -----------------------------------------------------
    # RELATIVE REST FEATURES
    # -----------------------------------------------------

    matches[
        "home_rest_advantage"
    ] = (
        matches[
            "home_rest_days_capped"
        ]
        -
        matches[
            "away_rest_days_capped"
        ]
    )

    matches[
        "home_matches_7d_advantage"
    ] = (
        matches[
            "away_matches_last_7d"
        ]
        -
        matches[
            "home_matches_last_7d"
        ]
    )

    matches[
        "home_matches_14d_advantage"
    ] = (
        matches[
            "away_matches_last_14d"
        ]
        -
        matches[
            "home_matches_last_14d"
        ]
    )

    matches[
        "home_short_rest_advantage"
    ] = (
        matches[
            "away_rest_3_or_less"
        ]
        -
        matches[
            "home_rest_3_or_less"
        ]
    )

    return matches


# =========================================================
# VALIDATION
# =========================================================

def validate_rest_features(
    team_rows,
    matches,
):

    # -----------------------------------------------------
    # FIRST MATCH FOR EACH TEAM MUST HAVE NO PRIOR REST
    # -----------------------------------------------------

    first_team_matches = (
        team_rows
        .sort_values(
            [
                "team",
                "date",
                "match_id",
            ]
        )
        .groupby(
            "team",
            sort=False,
        )
        .head(1)
    )

    if (
        first_team_matches[
            "days_since_last_match"
        ]
        .notna()
        .any()
    ):

        raise ValueError(
            "First team match has prior rest data."
        )

    # -----------------------------------------------------
    # WINDOW COUNTS CANNOT BE NEGATIVE
    # -----------------------------------------------------

    count_cols = [
        "matches_last_7d",
        "matches_last_14d",
        "matches_last_21d",
    ]

    for col in count_cols:

        if (
            team_rows[
                col
            ]
            < 0
        ).any():

            raise ValueError(
                f"Negative congestion count in {col}"
            )

    # -----------------------------------------------------
    # LONGER WINDOWS SHOULD NEVER HAVE FEWER MATCHES
    # -----------------------------------------------------

    if (
        team_rows[
            "matches_last_14d"
        ]
        <
        team_rows[
            "matches_last_7d"
        ]
    ).any():

        raise ValueError(
            "14-day counts below 7-day counts."
        )

    if (
        team_rows[
            "matches_last_21d"
        ]
        <
        team_rows[
            "matches_last_14d"
        ]
    ).any():

        raise ValueError(
            "21-day counts below 14-day counts."
        )

    # -----------------------------------------------------
    # TWO ROWS PER MATCH
    # -----------------------------------------------------

    if (
        len(matches)
        != team_rows[
            "match_id"
        ].nunique()
    ):

        raise ValueError(
            "Match-level rest table does not have "
            "one row per match."
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("==============================")
    print("BUILDING REST FEATURES V4")
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

    print(
        f"Team-game rows loaded: "
        f"{len(df):,}"
    )

    print(
        "Building team rest histories..."
    )

    team_rest = build_team_rest_features(
        df
    )

    print(
        "Building match-level "
        "rest differences..."
    )

    matches = build_match_rest_table(
        team_rest
    )

    print(
        "Running validation checks..."
    )

    validate_rest_features(
        team_rest,
        matches,
    )

    # =====================================================
    # SAVE
    # =====================================================

    matches.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("==============================")
    print("REST FEATURES COMPLETE")
    print("==============================")

    print(
        f"Match rows: "
        f"{len(matches):,}"
    )

    print(
        f"Unique matches: "
        f"{matches['match_id'].nunique():,}"
    )

    # -----------------------------------------------------
    # REST DISTRIBUTION
    # -----------------------------------------------------

    all_rest = pd.concat(
        [
            matches[
                "home_days_since_last_match"
            ],

            matches[
                "away_days_since_last_match"
            ],
        ],
        ignore_index=True,
    ).dropna()

    print()
    print(
        "REST DAYS DISTRIBUTION"
    )

    print(
        all_rest.describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
        .round(2)
        .to_string()
    )

    # -----------------------------------------------------
    # SHORT REST
    # -----------------------------------------------------

    print()
    print(
        "SHORT REST RATE"
    )

    short_home = (
        matches[
            "home_rest_3_or_less"
        ].mean()
    )

    short_away = (
        matches[
            "away_rest_3_or_less"
        ].mean()
    )

    print(
        f"Home <=3 days: "
        f"{short_home:.2%}"
    )

    print(
        f"Away <=3 days: "
        f"{short_away:.2%}"
    )

    # -----------------------------------------------------
    # CONGESTION
    # -----------------------------------------------------

    print()
    print(
        "CONGESTION DISTRIBUTION"
    )

    congestion = pd.DataFrame(
        {
            "home_matches_7d":
                matches[
                    "home_matches_last_7d"
                ],

            "away_matches_7d":
                matches[
                    "away_matches_last_7d"
                ],

            "home_matches_14d":
                matches[
                    "home_matches_last_14d"
                ],

            "away_matches_14d":
                matches[
                    "away_matches_last_14d"
                ],
        }
    )

    print(
        congestion
        .describe()
        .round(3)
        .to_string()
    )

    # -----------------------------------------------------
    # REST ADVANTAGE
    # -----------------------------------------------------

    print()
    print(
        "HOME REST ADVANTAGE"
    )

    print(
        matches[
            "home_rest_advantage"
        ]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
        .round(2)
        .to_string()
    )

    # -----------------------------------------------------
    # LEAGUE SUMMARY
    # -----------------------------------------------------

    print()
    print(
        "REST / CONGESTION BY LEAGUE"
    )

    league_summary = (
        matches
        .groupby(
            "league"
        )
        .agg(
            games=(
                "match_id",
                "count",
            ),

            avg_home_rest=(
                "home_days_since_last_match",
                "mean",
            ),

            avg_away_rest=(
                "away_days_since_last_match",
                "mean",
            ),

            home_short_rest=(
                "home_rest_3_or_less",
                "mean",
            ),

            away_short_rest=(
                "away_rest_3_or_less",
                "mean",
            ),

            avg_home_matches_7d=(
                "home_matches_last_7d",
                "mean",
            ),

            avg_away_matches_7d=(
                "away_matches_last_7d",
                "mean",
            ),
        )
        .reset_index()
    )

    league_summary[
        "home_short_rest"
    ] *= 100.0

    league_summary[
        "away_short_rest"
    ] *= 100.0

    print(
        league_summary
        .round(3)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "First match for each team has "
        "no fabricated rest history ✅"
    )

    print(
        "Current match excluded from "
        "all congestion counts ✅"
    )

    print()
    print(
        f"Saved:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()