from pathlib import Path
import os

import pandas as pd
import requests


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = (
    ROOT
    / "data"
    / "processed"
)

OUTPUT_HISTORY = (
    PROCESSED
    / "footystats_multileague_history.csv"
)

OUTPUT_SUMMARY = (
    PROCESSED
    / "footystats_multileague_history_summary.csv"
)


# ============================================================
# API
# ============================================================

BASE = "https://api.football-data-api.com"

API_KEY = (
    os.getenv("FOOTYSTATS_API_KEY")
)

if not API_KEY:

    raise RuntimeError(
        "\nFOOTYSTATS_API_KEY is not set.\n"
        "\nRun:\n"
        "export FOOTYSTATS_API_KEY='YOUR_KEY'\n"
    )


# ============================================================
# VERIFIED LEAGUE / SEASON IDS
# ============================================================

LEAGUES = {

    "Championship": {
        "1819": 1624,
        "1920": 2187,
        "2021": 4912,
        "2122": 6089,
        "2223": 7593,
        "2324": 9663,
        "2425": 12451,
        "2526": 14930,
        "2627": 17184,
    },

    "League One": {
        "1819": 1564,
        "1920": 2191,
        "2021": 4845,
        "2122": 6017,
        "2223": 7570,
        "2324": 9582,
        "2425": 12446,
        "2526": 14934,
        "2627": 17180,
    },

    "League Two": {
        "1819": 1574,
        "1920": 2192,
        "2021": 4844,
        "2122": 6015,
        "2223": 7574,
        "2324": 9581,
        "2425": 12422,
        "2526": 14935,
        "2627": 17185,
    },

    "National League": {
    "2526": 15657,
    },

    "La Liga": {
        "1819": 1677,
        "1920": 2319,
        "2021": 4944,
        "2122": 6211,
        "2223": 7665,
        "2324": 9665,
        "2425": 12316,
        "2526": 14956,
        "2627": 17199,
    },

    "Ligue 1": {
        "1819": 1508,
        "1920": 2392,
        "2021": 4505,
        "2122": 6019,
        "2223": 7500,
        "2324": 9674,
        "2425": 12337,
        "2526": 14932,
        "2627": 17102,
    },

        "Segunda División": {
        "2324": 9675,
        "2425": 12467,
        "2526": 15066,
    },

    "2. Bundesliga": {
        "1819": 1578,
        "1920": 4388,
        "2021": 4676,
        "2122": 6020,
        "2223": 7499,
        "2324": 9656,
        "2425": 12528,
        "2526": 14931,
        "2627": 17212,
    },

    "Belgian Pro League": {
        "1819": 1537,
        "1920": 2262,
        "2021": 4567,
        "2122": 6079,
        "2223": 7544,
        "2324": 9577,
        "2425": 12137,
        "2526": 14937,
        "2627": 17171,
    },

    "Eredivisie": {
        "1819": 1585,
        "1920": 2272,
        "2021": 4746,
        "2122": 5951,
        "2223": 7482,
        "2324": 9653,
        "2425": 12322,
        "2526": 14936,
        "2627": 17097,
    },
    "Serie A": {
        "2021": 4889,
        "2122": 6198,
        "2223": 7608,
        "2324": 9697,
        "2425": 12530,
        "2526": 15068,
    },

    "Swiss Super League": {
        "2021": 4906,
        "2122": 6044,
        "2223": 7504,
        "2324": 9580,
        "2425": 12326,
        "2526": 15047,
    },

    "Super Lig": {
        "2021": 4840,
        "2122": 6125,
        "2223": 7768,
        "2324": 9913,
        "2425": 12641,
        "2526": 14972,
    },

    "Primeira Liga": {
        "2021": 4885,
        "2122": 6117,
        "2223": 7731,
        "2324": 9984,
        "2425": 12931,
        "2526": 15115,
    },

    "Eliteserien": {
        "2020": 3695,
        "2021": 5496,
        "2022": 7048,
        "2023": 8739,
        "2024": 17353,
        "2025": 16260,
    },

    "MLS": {
        "2019": 1846,
        "2020": 4473,
        "2021": 5674,
        "2022": 6969,
        "2023": 8777,
        "2024": 10977,
        "2025": 13973,
        "2026": 16504,
    },

}


# ============================================================
# HELPERS
# ============================================================

def request_json(
    endpoint,
    params,
):

    response = requests.get(
        f"{BASE}/{endpoint}",
        params=params,
        timeout=60,
    )

    print(
        f"{endpoint}: "
        f"status={response.status_code}"
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"\nFootyStats request failed.\n"
            f"Status: {response.status_code}\n"
            f"Response:\n"
            f"{response.text[:1000]}\n"
        )

    payload = response.json()

    if not payload.get(
        "success",
        False,
    ):

        raise RuntimeError(
            "\nFootyStats success=False\n"
            f"{payload}\n"
        )

    return payload


# ============================================================
# FETCH ONE SEASON
# ============================================================

def fetch_season(
    league,
    season,
    season_id,
):

    print()
    print(
        "=" * 90
    )

    print(
        f"{league} | "
        f"{season} | "
        f"season_id={season_id}"
    )

    print(
        "=" * 90
    )

    payload = request_json(
        "league-matches",
        {
            "key":
                API_KEY,

            "season_id":
                season_id,

            "max_per_page":
                1000,
        },
    )

    data = payload.get(
        "data",
        [],
    )

    if not data:

        print(
            "No rows returned."
        )

        return pd.DataFrame()

    raw = pd.DataFrame(
        data
    )

    required = [
        "id",
        "date_unix",
        "status",
        "home_name",
        "away_name",
        "homeGoalCount",
        "awayGoalCount",
        "team_a_xg",
        "team_b_xg",
        "team_a_shots",
        "team_b_shots",
    ]

    missing = [
        col
        for col in required
        if col not in raw.columns
    ]

    if missing:

        raise ValueError(
            f"\n{league} {season} "
            f"missing required fields:\n"
            f"{missing}\n"
            f"\nAvailable columns:\n"
            f"{raw.columns.tolist()}\n"
        )

    # --------------------------------------------------------
    # OPTIONAL SOT FIELDS
    # --------------------------------------------------------

    home_sot_col = None

    away_sot_col = None

    home_candidates = [
        "team_a_shotsOnTarget",
        "team_a_shots_on_target",
        "team_a_sot",
    ]

    away_candidates = [
        "team_b_shotsOnTarget",
        "team_b_shots_on_target",
        "team_b_sot",
    ]

    for col in home_candidates:

        if col in raw.columns:

            home_sot_col = col
            break

    for col in away_candidates:

        if col in raw.columns:

            away_sot_col = col
            break

    # --------------------------------------------------------
    # BUILD STANDARD TABLE
    # --------------------------------------------------------

    df = pd.DataFrame()

    df[
        "footystats_match_id"
    ] = raw[
        "id"
    ]

    df[
        "footystats_season_id"
    ] = season_id

    df[
        "season"
    ] = season

    df[
        "league"
    ] = league

    df[
        "date"
    ] = pd.to_datetime(
        raw[
            "date_unix"
        ],
        unit="s",
        utc=True,
        errors="coerce",
    ).dt.tz_convert(
        None
    ).dt.normalize()

    df[
        "status"
    ] = (
        raw[
            "status"
        ]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df[
        "home_team"
    ] = (
        raw[
            "home_name"
        ]
        .astype(str)
        .str.strip()
    )

    df[
        "away_team"
    ] = (
        raw[
            "away_name"
        ]
        .astype(str)
        .str.strip()
    )

    numeric_map = {

        "home_goals":
            "homeGoalCount",

        "away_goals":
            "awayGoalCount",

        "home_xg":
            "team_a_xg",

        "away_xg":
            "team_b_xg",

        "home_shots":
            "team_a_shots",

        "away_shots":
            "team_b_shots",
    }

    for output_col, source_col in (
        numeric_map.items()
    ):

        df[
            output_col
        ] = pd.to_numeric(
            raw[
                source_col
            ],
            errors="coerce",
        )

    if home_sot_col:

        df[
            "home_shots_on_target"
        ] = pd.to_numeric(
            raw[
                home_sot_col
            ],
            errors="coerce",
        )

    else:

        df[
            "home_shots_on_target"
        ] = pd.NA

    if away_sot_col:

        df[
            "away_shots_on_target"
        ] = pd.to_numeric(
            raw[
                away_sot_col
            ],
            errors="coerce",
        )

    else:

        df[
            "away_shots_on_target"
        ] = pd.NA

    # --------------------------------------------------------
    # ONLY COMPLETED MATCHES
    # --------------------------------------------------------

    df = df[
        df[
            "status"
        ]
        ==
        "complete"
    ].copy()

    # --------------------------------------------------------
    # DROP ROWS MISSING CORE V5 SIGNALS
    # --------------------------------------------------------

    core_cols = [
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "home_shots",
        "away_shots",
    ]

    before = len(
        df
    )

    df = df.dropna(
        subset=core_cols
    ).copy()

    dropped = (
        before
        -
        len(
            df
        )
    )

    print(
        "Completed:",
        f"{before:,}",
    )

    print(
        "Usable:",
        f"{len(df):,}",
    )

    print(
        "Dropped missing core signals:",
        f"{dropped:,}",
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 90
    )

    print(
        "BUILD FOOTYSTATS "
        "MULTI-LEAGUE HISTORY"
    )

    print(
        "=" * 90
    )

    frames = []

    summary_rows = []

    for league, seasons in (
        LEAGUES.items()
    ):

        for season, season_id in (
            seasons.items()
        ):

            df = fetch_season(
                league,
                season,
                season_id,
            )

            if len(df) == 0:

                summary_rows.append(
                    {
                        "league":
                            league,

                        "season":
                            season,

                        "season_id":
                            season_id,

                        "matches":
                            0,
                    }
                )

                continue

            frames.append(
                df
            )

            summary_rows.append(
                {
                    "league":
                        league,

                    "season":
                        season,

                    "season_id":
                        season_id,

                    "matches":
                        len(
                            df
                        ),

                    "home_xg_mean":
                        df[
                            "home_xg"
                        ].mean(),

                    "away_xg_mean":
                        df[
                            "away_xg"
                        ].mean(),

                    "home_shots_mean":
                        df[
                            "home_shots"
                        ].mean(),

                    "away_shots_mean":
                        df[
                            "away_shots"
                        ].mean(),

                    "home_goals_mean":
                        df[
                            "home_goals"
                        ].mean(),

                    "away_goals_mean":
                        df[
                            "away_goals"
                        ].mean(),
                }
            )

    if not frames:

        raise RuntimeError(
            "No historical FootyStats "
            "matches were built."
        )

    history = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # UNIQUE CHECK
    # --------------------------------------------------------

    duplicate_mask = (
        history.duplicated(
            subset=[
                "league",
                "footystats_match_id",
            ],
            keep=False,
        )
    )

    if duplicate_mask.any():

        print()
        print(
            "WARNING: duplicate "
            "FootyStats match IDs:"
        )

        print(
            history.loc[
                duplicate_mask,
                [
                    "league",
                    "season",
                    "date",
                    "home_team",
                    "away_team",
                    "footystats_match_id",
                ],
            ]
            .sort_values(
                [
                    "league",
                    "date",
                ]
            )
            .to_string(
                index=False
            )
        )

        raise ValueError(
            "Duplicate match IDs detected."
        )

    history = history.sort_values(
        [
            "league",
            "date",
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = pd.DataFrame(
        summary_rows
    )

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.to_csv(
        OUTPUT_HISTORY,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "HISTORY SUMMARY"
    )

    print(
        "=" * 100
    )

    print()

    print(
        summary[
            [
                "league",
                "season",
                "matches",
                "home_xg_mean",
                "away_xg_mean",
                "home_shots_mean",
                "away_shots_mean",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "TOTALS BY LEAGUE"
    )

    print(
        "=" * 100
    )

    print(
        history[
            "league"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Total historical matches:",
        f"{len(history):,}",
    )

    print()
    print(
        "Date range:"
    )

    print(
        history[
            "date"
        ].min(),
        "->",
        history[
            "date"
        ].max(),
    )

    print()
    print(
        "History:"
    )

    print(
        OUTPUT_HISTORY
    )

    print()
    print(
        "Summary:"
    )

    print(
        OUTPUT_SUMMARY
    )

    print()
    print(
        "=" * 90
    )

    print(
        "MULTI-LEAGUE HISTORY COMPLETE"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()