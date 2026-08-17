from pathlib import Path
import re

import numpy as np
import pandas as pd
import requests


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"

UNDERSTAT_FILE = (
    PROCESSED
    / "understat_xg_matched.csv"
)

OUTPUT_FOOTYSTATS = (
    PROCESSED
    / "footystats_epl_xg_history.csv"
)

OUTPUT_MATCHED = (
    PROCESSED
    / "footystats_understat_xg_history_matched.csv"
)

OUTPUT_UNMATCHED_FS = (
    PROCESSED
    / "footystats_understat_unmatched_footystats.csv"
)

OUTPUT_UNMATCHED_US = (
    PROCESSED
    / "footystats_understat_unmatched_understat.csv"
)


# ============================================================
# FOOTYSTATS
# ============================================================

BASE = "https://api.football-data-api.com"
KEY = "example"

# Confirmed accessible EPL seasons from your test.
SEASONS = {
    "1819": 1625,
    "1920": 2012,
    "2021": 4759,
    "2122": 6135,
    "2223": 7704,
    "2324": 9660,
    "2425": 12325,
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_team(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = value.replace(
        "&",
        "and",
    )

    value = value.replace(
        "’",
        "'",
    )

    value = value.replace(
        "'",
        "",
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = " ".join(
        value.split()
    )

    aliases = {
    "manchester united":
        "man united",

    "man utd":
        "man united",

    "manchester city":
        "man city",

    "tottenham hotspur":
        "tottenham",

    "wolverhampton wanderers":
        "wolves",

    "wolverhampton":
        "wolves",

    "nottingham forest":
        "nottm forest",

    "afc bournemouth":
        "bournemouth",

    "brighton and hove albion":
        "brighton",

    "newcastle united":
        "newcastle",

    "west ham united":
        "west ham",

    "leicester city":
        "leicester",

    "ipswich town":
        "ipswich",

    "sheffield united":
        "sheffield utd",

    "sheffield utd":
        "sheffield utd",

    "west bromwich albion":
        "west brom",

    "norwich city":
        "norwich",

    "huddersfield town":
        "huddersfield",

    "cardiff city":
        "cardiff",

    "swansea city":
        "swansea",

    "hull city":
        "hull",

    "stoke city":
        "stoke",

    "leeds united":
        "leeds",

    "luton town":
        "luton",
}

    return aliases.get(
        value,
        value,
    )


# ============================================================
# FETCH ONE FOOTYSTATS SEASON
# ============================================================

def fetch_season(
    season_code,
    season_id,
):

    print()
    print(
        "=" * 90
    )

    print(
        f"FETCHING FOOTYSTATS EPL "
        f"{season_code} "
        f"(season_id={season_id})"
    )

    print(
        "=" * 90
    )

    response = requests.get(
        f"{BASE}/league-matches",
        params={
            "key":
                KEY,

            "season_id":
                season_id,

            "max_per_page":
                1000,
        },
        timeout=60,
    )

    print(
        "Status:",
        response.status_code,
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"FootyStats request failed "
            f"for {season_code}: "
            f"{response.text[:500]}"
        )

    payload = response.json()

    if not payload.get(
        "success",
        False,
    ):

        raise RuntimeError(
            f"FootyStats success=False "
            f"for {season_code}"
        )

    data = payload.get(
        "data",
        [],
    )

    df = pd.DataFrame(
        data
    )

    required = [
        "id",
        "date_unix",
        "status",
        "home_name",
        "away_name",
        "team_a_xg",
        "team_b_xg",
        "team_a_shots",
        "team_b_shots",
        "team_a_shotsOnTarget",
        "team_b_shotsOnTarget",
        "homeGoalCount",
        "awayGoalCount",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{season_code} missing columns: "
            f"{missing}"
        )

    df = df[
        required
    ].copy()

    df = df.rename(
        columns={
            "id":
                "footystats_match_id",

            "home_name":
                "home_team_fs",

            "away_name":
                "away_team_fs",

            "team_a_xg":
                "home_xg_fs",

            "team_b_xg":
                "away_xg_fs",

            "team_a_shots":
                "home_shots_fs",

            "team_b_shots":
                "away_shots_fs",

            "team_a_shotsOnTarget":
                "home_sot_fs",

            "team_b_shotsOnTarget":
                "away_sot_fs",

            "homeGoalCount":
                "home_goals_fs",

            "awayGoalCount":
                "away_goals_fs",
        }
    )

    df["date"] = pd.to_datetime(
        df["date_unix"],
        unit="s",
        utc=True,
    ).dt.tz_convert(
        None
    ).dt.normalize()

    df["season"] = (
        season_code
    )

    df["league"] = (
        "Premier League"
    )

    numeric_cols = [
        "home_xg_fs",
        "away_xg_fs",
        "home_shots_fs",
        "away_shots_fs",
        "home_sot_fs",
        "away_sot_fs",
        "home_goals_fs",
        "away_goals_fs",
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df[
        df[
            "status"
        ]
        .astype(str)
        .str.lower()
        ==
        "complete"
    ].copy()

    df = df.dropna(
        subset=[
            "date",
            "home_team_fs",
            "away_team_fs",
            "home_xg_fs",
            "away_xg_fs",
        ]
    )

    df = df[
        (
            df["home_xg_fs"] > 0
        )
        |
        (
            df["away_xg_fs"] > 0
        )
    ].copy()

    df[
        "home_norm"
    ] = df[
        "home_team_fs"
    ].map(
        normalize_team
    )

    df[
        "away_norm"
    ] = df[
        "away_team_fs"
    ].map(
        normalize_team
    )

    print(
        "Completed matches:",
        f"{len(df):,}",
    )

    print(
        "xG rows:",
        f"{df[['home_xg_fs','away_xg_fs']].notna().all(axis=1).sum():,}",
    )

    return df


# ============================================================
# BUILD FOOTYSTATS HISTORY
# ============================================================

def build_footystats_history():

    frames = []

    for season_code, season_id in (
        SEASONS.items()
    ):

        frames.append(
            fetch_season(
                season_code,
                season_id,
            )
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df = df.sort_values(
        [
            "date",
            "home_norm",
            "away_norm",
        ]
    ).reset_index(
        drop=True
    )

    duplicates = (
        df.duplicated(
            subset=[
                "season",
                "date",
                "home_norm",
                "away_norm",
            ],
            keep=False,
        )
    )

    if duplicates.any():

        print()
        print(
            "WARNING: duplicate "
            "FootyStats match keys:"
        )

        print(
            df.loc[
                duplicates,
                [
                    "season",
                    "date",
                    "home_team_fs",
                    "away_team_fs",
                ],
            ]
            .to_string(
                index=False
            )
        )

    return df


# ============================================================
# LOAD UNDERSTAT
# ============================================================

def load_understat():

    print()
    print(
        "=" * 90
    )

    print(
        "LOADING UNDERSTAT REFERENCE"
    )

    print(
        "=" * 90
    )

    if not UNDERSTAT_FILE.exists():

        raise FileNotFoundError(
            UNDERSTAT_FILE
        )

    df = pd.read_csv(
        UNDERSTAT_FILE
    )

    required = [
        "match_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
        "home_xg",
        "away_xg",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Understat file missing: "
            f"{missing}"
        )

    df = df[
        required
    ].copy()

    df = df[
        df[
            "league"
        ]
        ==
        "Premier League"
    ].copy()

    df["season"] = (
        df["season"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
    )

    df = df[
        df[
            "season"
        ].isin(
            SEASONS.keys()
        )
    ].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.normalize()

    df[
        "home_xg_us"
    ] = pd.to_numeric(
        df[
            "home_xg"
        ],
        errors="coerce",
    )

    df[
        "away_xg_us"
    ] = pd.to_numeric(
        df[
            "away_xg"
        ],
        errors="coerce",
    )

    df[
        "home_norm"
    ] = df[
        "home_team"
    ].map(
        normalize_team
    )

    df[
        "away_norm"
    ] = df[
        "away_team"
    ].map(
        normalize_team
    )

    df = df.dropna(
        subset=[
            "date",
            "home_xg_us",
            "away_xg_us",
        ]
    )

    print(
        "Understat EPL rows:",
        f"{len(df):,}",
    )

    print()
    print(
        "Understat by season:"
    )

    print(
        df[
            "season"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return df


# ============================================================
# MATCH SOURCES
# ============================================================

def match_sources(
    fs,
    us,
):

    print()
    print(
        "=" * 90
    )

    print(
        "MATCHING FOOTYSTATS ↔ UNDERSTAT"
    )

    print(
        "=" * 90
    )

    merge_keys = [
        "season",
        "date",
        "home_norm",
        "away_norm",
    ]

    matched = fs.merge(
        us[
            [
                "match_id",
                "season",
                "date",
                "home_norm",
                "away_norm",
                "home_team",
                "away_team",
                "home_xg_us",
                "away_xg_us",
            ]
        ],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )

    print(
        "Matched:",
        f"{len(matched):,}",
    )

    print()
    print(
        "Matched by season:"
    )

    season_summary = []

    for season in SEASONS:

        fs_n = (
            fs[
                fs[
                    "season"
                ]
                ==
                season
            ]
            .shape[
                0
            ]
        )

        us_n = (
            us[
                us[
                    "season"
                ]
                ==
                season
            ]
            .shape[
                0
            ]
        )

        m_n = (
            matched[
                matched[
                    "season"
                ]
                ==
                season
            ]
            .shape[
                0
            ]
        )

        season_summary.append(
            {
                "season":
                    season,

                "footystats":
                    fs_n,

                "understat":
                    us_n,

                "matched":
                    m_n,

                "fs_coverage_pct":
                    (
                        100.0
                        * m_n
                        / fs_n
                        if fs_n
                        else np.nan
                    ),

                "us_coverage_pct":
                    (
                        100.0
                        * m_n
                        / us_n
                        if us_n
                        else np.nan
                    ),
            }
        )

    season_summary = pd.DataFrame(
        season_summary
    )

    print(
        season_summary
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # UNMATCHED FOOTYSTATS
    # --------------------------------------------------------

    fs_check = fs.merge(
        matched[
            [
                "footystats_match_id",
            ]
        ],
        on="footystats_match_id",
        how="left",
        indicator=True,
    )

    unmatched_fs = fs_check[
        fs_check[
            "_merge"
        ]
        ==
        "left_only"
    ].drop(
        columns="_merge"
    )

    # --------------------------------------------------------
    # UNMATCHED UNDERSTAT
    # --------------------------------------------------------

    us_check = us.merge(
        matched[
            [
                "match_id",
            ]
        ],
        on="match_id",
        how="left",
        indicator=True,
    )

    unmatched_us = us_check[
        us_check[
            "_merge"
        ]
        ==
        "left_only"
    ].drop(
        columns="_merge"
    )

    return (
        matched,
        unmatched_fs,
        unmatched_us,
        season_summary,
    )


# ============================================================
# ADD DIAGNOSTICS
# ============================================================

def add_diagnostics(
    matched,
):

    df = matched.copy()

    df[
        "home_xg_diff_raw"
    ] = (
        df[
            "home_xg_fs"
        ]
        -
        df[
            "home_xg_us"
        ]
    )

    df[
        "away_xg_diff_raw"
    ] = (
        df[
            "away_xg_fs"
        ]
        -
        df[
            "away_xg_us"
        ]
    )

    df[
        "total_xg_fs"
    ] = (
        df[
            "home_xg_fs"
        ]
        +
        df[
            "away_xg_fs"
        ]
    )

    df[
        "total_xg_us"
    ] = (
        df[
            "home_xg_us"
        ]
        +
        df[
            "away_xg_us"
        ]
    )

    df[
        "total_xg_diff_raw"
    ] = (
        df[
            "total_xg_fs"
        ]
        -
        df[
            "total_xg_us"
        ]
    )

    return df


# ============================================================
# PRINT RAW SEASON COMPARISON
# ============================================================

def print_raw_comparison(
    matched,
):

    print()
    print(
        "=" * 110
    )

    print(
        "RAW FOOTYSTATS VS UNDERSTAT "
        "BY SEASON"
    )

    print(
        "=" * 110
    )

    rows = []

    for season, sub in (
        matched.groupby(
            "season"
        )
    ):

        rows.append(
            {
                "season":
                    season,

                "games":
                    len(
                        sub
                    ),

                "fs_home_mean":
                    sub[
                        "home_xg_fs"
                    ].mean(),

                "us_home_mean":
                    sub[
                        "home_xg_us"
                    ].mean(),

                "fs_away_mean":
                    sub[
                        "away_xg_fs"
                    ].mean(),

                "us_away_mean":
                    sub[
                        "away_xg_us"
                    ].mean(),

                "home_corr":
                    sub[
                        [
                            "home_xg_fs",
                            "home_xg_us",
                        ]
                    ].corr().iloc[
                        0,
                        1
                    ],

                "away_corr":
                    sub[
                        [
                            "away_xg_fs",
                            "away_xg_us",
                        ]
                    ].corr().iloc[
                        0,
                        1
                    ],

                "home_mae":
                    (
                        sub[
                            "home_xg_fs"
                        ]
                        -
                        sub[
                            "home_xg_us"
                        ]
                    )
                    .abs()
                    .mean(),

                "away_mae":
                    (
                        sub[
                            "away_xg_fs"
                        ]
                        -
                        sub[
                            "away_xg_us"
                        ]
                    )
                    .abs()
                    .mean(),
            }
        )

    table = pd.DataFrame(
        rows
    ).sort_values(
        "season"
    )

    print(
        table.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 90
    )

    print(
        "BUILDING FOOTYSTATS EPL "
        "HISTORICAL XG STORE"
    )

    print(
        "=" * 90
    )

    print()
    print(
        "Seasons:"
    )

    for season, season_id in (
        SEASONS.items()
    ):

        print(
            f"  {season}: "
            f"{season_id}"
        )

    fs = build_footystats_history()

    print()
    print(
        "Total FootyStats rows:",
        f"{len(fs):,}",
    )

    fs.to_csv(
        OUTPUT_FOOTYSTATS,
        index=False,
    )

    us = load_understat()

    (
        matched,
        unmatched_fs,
        unmatched_us,
        season_summary,
    ) = match_sources(
        fs,
        us,
    )

    matched = add_diagnostics(
        matched
    )

    print_raw_comparison(
        matched
    )

    matched.to_csv(
        OUTPUT_MATCHED,
        index=False,
    )

    unmatched_fs.to_csv(
        OUTPUT_UNMATCHED_FS,
        index=False,
    )

    unmatched_us.to_csv(
        OUTPUT_UNMATCHED_US,
        index=False,
    )

    print()
    print(
        "=" * 90
    )

    print(
        "HISTORICAL XG STORE COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        "FootyStats history:"
    )

    print(
        OUTPUT_FOOTYSTATS
    )

    print()
    print(
        "Matched history:"
    )

    print(
        OUTPUT_MATCHED
    )

    print()
    print(
        "Unmatched FootyStats:"
    )

    print(
        OUTPUT_UNMATCHED_FS
    )

    print()
    print(
        "Unmatched Understat:"
    )

    print(
        OUTPUT_UNMATCHED_US
    )

    print()
    print(
        "NO V5 PARAMETERS CHANGED ✅"
    )


if __name__ == "__main__":
    main()