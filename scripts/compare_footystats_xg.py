from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"

OUTPUT_MATCHES = (
    PROCESSED
    / "footystats_xg_compatibility_matches.csv"
)

OUTPUT_SUMMARY = (
    PROCESSED
    / "footystats_xg_compatibility_summary.csv"
)


# ============================================================
# FOOTYSTATS TEST SETTINGS
# ============================================================

FOOTYSTATS_BASE = (
    "https://api.football-data-api.com"
)

FOOTYSTATS_KEY = "example"

# FootyStats documented example:
# Premier League 2024/25
FOOTYSTATS_SEASON_ID = 12325


# ============================================================
# POSSIBLE EXISTING XG FILES
# ============================================================

XG_CANDIDATES = [

    PROCESSED / "xg_matches.csv",

    PROCESSED / "understat_matches.csv",

    PROCESSED / "understat_xg.csv",

    PROCESSED / "xg_data.csv",

    PROCESSED / "match_xg.csv",

    ROOT / "data" / "raw" / "xg_matches.csv",

    ROOT / "data" / "raw" / "understat_matches.csv",

    ROOT / "data" / "raw" / "understat_xg.csv",
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_team(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    replacements = {
        "&": "and",
        "'": "",
        "’": "",
        ".": "",
        "-": " ",
    }

    for old, new in replacements.items():
        value = value.replace(
            old,
            new,
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

        "nott'm forest":
            "nottm forest",

        "afc bournemouth":
            "bournemouth",

        "brighton and hove albion":
            "brighton",

        "brighton hove albion":
            "brighton",

        "west ham united":
            "west ham",

        "newcastle united":
            "newcastle",

        "leicester city":
            "leicester",

        "ipswich town":
            "ipswich",
    }

    value = " ".join(
        value.split()
    )

    return aliases.get(
        value,
        value,
    )


# ============================================================
# FETCH FOOTYSTATS
# ============================================================

def fetch_footystats():

    print()
    print(
        "Fetching FootyStats "
        "Premier League 2024/25..."
    )

    url = (
        f"{FOOTYSTATS_BASE}/league-matches"
    )

    params = {
        "key":
            FOOTYSTATS_KEY,

        "season_id":
            FOOTYSTATS_SEASON_ID,
    }

    response = requests.get(
        url,
        params=params,
        timeout=60,
    )

    print(
        "FootyStats status:",
        response.status_code,
    )

    response.raise_for_status()

    payload = response.json()

    if not payload.get(
        "success",
        False,
    ):
        raise RuntimeError(
            "FootyStats returned success=False"
        )

    data = payload.get(
        "data",
        [],
    )

    if not data:
        raise RuntimeError(
            "FootyStats returned no matches."
        )

    df = pd.DataFrame(
        data
    )

    required = [
        "id",
        "date_unix",
        "home_name",
        "away_name",
        "team_a_xg",
        "team_b_xg",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "FootyStats missing columns: "
            + str(missing)
        )

    df = df[
        required
    ].copy()

    df["date"] = pd.to_datetime(
        df["date_unix"],
        unit="s",
        utc=True,
    ).dt.date

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.rename(
        columns={
            "id":
                "footystats_id",

            "home_name":
                "fs_home_team",

            "away_name":
                "fs_away_team",

            "team_a_xg":
                "fs_home_xg",

            "team_b_xg":
                "fs_away_xg",
        }
    )

    df["fs_home_xg"] = pd.to_numeric(
        df["fs_home_xg"],
        errors="coerce",
    )

    df["fs_away_xg"] = pd.to_numeric(
        df["fs_away_xg"],
        errors="coerce",
    )

    df["home_norm"] = (
        df["fs_home_team"]
        .map(normalize_team)
    )

    df["away_norm"] = (
        df["fs_away_team"]
        .map(normalize_team)
    )

    df = df.dropna(
        subset=[
            "fs_home_xg",
            "fs_away_xg",
        ]
    )

    print(
        "FootyStats matches:",
        f"{len(df):,}",
    )

    return df


# ============================================================
# FIND EXISTING XG DATA
# ============================================================

def find_existing_xg():

    print()
    print("=" * 80)
    print("SEARCHING FOR EXISTING XG DATA")
    print("=" * 80)

    files = []

    for path in XG_CANDIDATES:

        if path.exists():
            files.append(
                path
            )

    # Also inspect processed/raw CSVs for likely xG columns.
    for folder in [
        PROCESSED,
        ROOT / "data" / "raw",
    ]:

        if not folder.exists():
            continue

        for path in folder.glob(
            "*.csv"
        ):

            if path in files:
                continue

            try:

                sample = pd.read_csv(
                    path,
                    nrows=5,
                )

            except Exception:
                continue

            columns = [
                str(c).lower()
                for c in sample.columns
            ]

            has_xg = any(
                "xg" in c
                for c in columns
            )

            if has_xg:
                files.append(
                    path
                )

    if not files:

        print(
            "No candidate xG CSVs found."
        )

        return None

    print(
        "Candidate files:"
    )

    for i, path in enumerate(
        files,
        start=1,
    ):

        print(
            f"{i:>2}. "
            f"{path.relative_to(ROOT)}"
        )

    # --------------------------------------------------------
    # SCORE FILES BASED ON EXPECTED MATCH-LEVEL STRUCTURE
    # --------------------------------------------------------

    scored = []

    for path in files:

        try:

            sample = pd.read_csv(
                path,
                nrows=10,
            )

        except Exception:
            continue

        cols = {
            str(c).lower():
                c
            for c in sample.columns
        }

        score = 0

        if any(
            x in cols
            for x in [
                "date",
                "datetime",
            ]
        ):
            score += 2

        if any(
            x in cols
            for x in [
                "home_team",
                "home",
                "h_team",
            ]
        ):
            score += 2

        if any(
            x in cols
            for x in [
                "away_team",
                "away",
                "a_team",
            ]
        ):
            score += 2

        if any(
            x in cols
            for x in [
                "home_xg",
                "xg_home",
                "hxg",
            ]
        ):
            score += 4

        if any(
            x in cols
            for x in [
                "away_xg",
                "xg_away",
                "axg",
            ]
        ):
            score += 4

        # Generic xG presence
        if any(
            "xg" in c
            for c in cols
        ):
            score += 1

        scored.append(
            (
                score,
                path,
                sample.columns.tolist(),
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    print()
    print(
        "Top candidate:"
    )

    score, path, columns = (
        scored[0]
    )

    print(
        path.relative_to(ROOT)
    )

    print(
        "Score:",
        score,
    )

    print(
        "Columns:"
    )

    print(
        columns
    )

    return path


# ============================================================
# DETECT COLUMN
# ============================================================

def detect_column(
    columns,
    candidates,
):

    lookup = {
        str(c).lower():
            c
        for c in columns
    }

    for candidate in candidates:

        if candidate.lower() in lookup:
            return lookup[
                candidate.lower()
            ]

    return None


# ============================================================
# LOAD EXISTING XG
# ============================================================

def load_existing_xg(
    path,
):

    print()
    print(
        "Loading existing xG:"
    )

    print(
        path
    )

    df = pd.read_csv(
        path
    )

    date_col = detect_column(
        df.columns,
        [
            "date",
            "datetime",
        ],
    )

    home_col = detect_column(
        df.columns,
        [
            "home_team",
            "home",
            "h_team",
        ],
    )

    away_col = detect_column(
        df.columns,
        [
            "away_team",
            "away",
            "a_team",
        ],
    )

    home_xg_col = detect_column(
        df.columns,
        [
            "home_xg",
            "xg_home",
            "hxg",
        ],
    )

    away_xg_col = detect_column(
        df.columns,
        [
            "away_xg",
            "xg_away",
            "axg",
        ],
    )

    required = {
        "date":
            date_col,

        "home team":
            home_col,

        "away team":
            away_col,

        "home xG":
            home_xg_col,

        "away xG":
            away_xg_col,
    }

    missing = [
        name
        for name, value
        in required.items()
        if value is None
    ]

    if missing:

        print()
        print(
            "Automatic column detection "
            "could not identify:"
        )

        for item in missing:
            print(
                " -",
                item,
            )

        print()
        print(
            "Available columns:"
        )

        print(
            df.columns.tolist()
        )

        print()
        print(
            "STOPPING WITHOUT MODIFYING "
            "ANY MODEL FILES."
        )

        sys.exit(2)

    keep = df[
        [
            date_col,
            home_col,
            away_col,
            home_xg_col,
            away_xg_col,
        ]
    ].copy()

    keep = keep.rename(
        columns={
            date_col:
                "date",

            home_col:
                "existing_home_team",

            away_col:
                "existing_away_team",

            home_xg_col:
                "existing_home_xg",

            away_xg_col:
                "existing_away_xg",
        }
    )

    keep["date"] = pd.to_datetime(
        keep["date"],
        errors="coerce",
    ).dt.normalize()

    keep[
        "existing_home_xg"
    ] = pd.to_numeric(
        keep[
            "existing_home_xg"
        ],
        errors="coerce",
    )

    keep[
        "existing_away_xg"
    ] = pd.to_numeric(
        keep[
            "existing_away_xg"
        ],
        errors="coerce",
    )

    keep["home_norm"] = (
        keep[
            "existing_home_team"
        ].map(
            normalize_team
        )
    )

    keep["away_norm"] = (
        keep[
            "existing_away_team"
        ].map(
            normalize_team
        )
    )

    keep = keep.dropna(
        subset=[
            "date",
            "existing_home_xg",
            "existing_away_xg",
        ]
    )

    # Only EPL 2024/25 overlap window.
    keep = keep[
        (
            keep["date"]
            >= pd.Timestamp(
                "2024-08-01"
            )
        )
        &
        (
            keep["date"]
            <= pd.Timestamp(
                "2025-06-30"
            )
        )
    ].copy()

    print(
        "Existing xG rows in "
        "2024/25 window:",
        f"{len(keep):,}",
    )

    return keep


# ============================================================
# METRICS
# ============================================================

def mae(
    a,
    b,
):

    return np.mean(
        np.abs(
            np.asarray(a)
            -
            np.asarray(b)
        )
    )


def rmse(
    a,
    b,
):

    return np.sqrt(
        np.mean(
            (
                np.asarray(a)
                -
                np.asarray(b)
            ) ** 2
        )
    )


def correlation(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    if (
        len(a) < 2
        or
        np.std(a) == 0
        or
        np.std(b) == 0
    ):
        return np.nan

    return np.corrcoef(
        a,
        b,
    )[0, 1]


# ============================================================
# COMPARE
# ============================================================

def compare(
    fs,
    existing,
):

    print()
    print("=" * 80)
    print("MATCHING XG SOURCES")
    print("=" * 80)

    merged = fs.merge(
        existing,
        on=[
            "date",
            "home_norm",
            "away_norm",
        ],
        how="inner",
        validate="one_to_one",
    )

    print(
        "Matched games:",
        f"{len(merged):,}",
    )

    print(
        "FootyStats games:",
        f"{len(fs):,}",
    )

    coverage = (
        len(merged)
        /
        len(fs)
        *
        100
    )

    print(
        "Match coverage:",
        f"{coverage:.2f}%",
    )

    if len(merged) == 0:

        print()
        print(
            "No matches joined."
        )

        print(
            "This is probably a team-name "
            "or date-format issue."
        )

        return None, None

    # --------------------------------------------------------
    # DIFFERENCES
    # --------------------------------------------------------

    merged[
        "home_xg_diff"
    ] = (
        merged[
            "fs_home_xg"
        ]
        -
        merged[
            "existing_home_xg"
        ]
    )

    merged[
        "away_xg_diff"
    ] = (
        merged[
            "fs_away_xg"
        ]
        -
        merged[
            "existing_away_xg"
        ]
    )

    merged[
        "fs_total_xg"
    ] = (
        merged[
            "fs_home_xg"
        ]
        +
        merged[
            "fs_away_xg"
        ]
    )

    merged[
        "existing_total_xg"
    ] = (
        merged[
            "existing_home_xg"
        ]
        +
        merged[
            "existing_away_xg"
        ]
    )

    merged[
        "total_xg_diff"
    ] = (
        merged[
            "fs_total_xg"
        ]
        -
        merged[
            "existing_total_xg"
        ]
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    rows = []

    comparisons = [

        (
            "HOME_XG",
            "fs_home_xg",
            "existing_home_xg",
        ),

        (
            "AWAY_XG",
            "fs_away_xg",
            "existing_away_xg",
        ),

        (
            "TOTAL_XG",
            "fs_total_xg",
            "existing_total_xg",
        ),
    ]

    for (
        name,
        fs_col,
        ex_col,
    ) in comparisons:

        rows.append(
            {
                "metric":
                    name,

                "games":
                    len(merged),

                "correlation":
                    correlation(
                        merged[fs_col],
                        merged[ex_col],
                    ),

                "mae":
                    mae(
                        merged[fs_col],
                        merged[ex_col],
                    ),

                "rmse":
                    rmse(
                        merged[fs_col],
                        merged[ex_col],
                    ),

                "footystats_mean":
                    merged[
                        fs_col
                    ].mean(),

                "existing_mean":
                    merged[
                        ex_col
                    ].mean(),

                "mean_bias_fs_minus_existing":
                    (
                        merged[
                            fs_col
                        ]
                        -
                        merged[
                            ex_col
                        ]
                    ).mean(),

                "footystats_std":
                    merged[
                        fs_col
                    ].std(),

                "existing_std":
                    merged[
                        ex_col
                    ].std(),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    return merged, summary


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 80
    )

    print(
        "FOOTYSTATS XG "
        "COMPATIBILITY TEST"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "Purpose:"
    )

    print(
        "Compare FootyStats match-level xG "
        "against the existing V5 xG source."
    )

    print()
    print(
        "NO V5 PARAMETERS WILL BE CHANGED."
    )

    fs = fetch_footystats()

    path = find_existing_xg()

    if path is None:

        print()
        print(
            "Send me the candidate-file "
            "output and we'll point the "
            "script at the correct xG file."
        )

        return

    existing = load_existing_xg(
        path
    )

    merged, summary = compare(
        fs,
        existing,
    )

    if merged is None:
        return

    print()
    print(
        "=" * 80
    )

    print(
        "COMPATIBILITY RESULTS"
    )

    print(
        "=" * 80
    )

    print()

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "LARGEST HOME XG DIFFERENCES"
    )

    print(
        "=" * 80
    )

    cols = [
        "date",
        "fs_home_team",
        "fs_away_team",
        "fs_home_xg",
        "existing_home_xg",
        "home_xg_diff",
    ]

    largest_home = (
        merged
        .assign(
            abs_diff=lambda x:
                x[
                    "home_xg_diff"
                ].abs()
        )
        .sort_values(
            "abs_diff",
            ascending=False,
        )
        .head(15)
    )

    print(
        largest_home[
            cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "LARGEST AWAY XG DIFFERENCES"
    )

    print(
        "=" * 80
    )

    cols = [
        "date",
        "fs_home_team",
        "fs_away_team",
        "fs_away_xg",
        "existing_away_xg",
        "away_xg_diff",
    ]

    largest_away = (
        merged
        .assign(
            abs_diff=lambda x:
                x[
                    "away_xg_diff"
                ].abs()
        )
        .sort_values(
            "abs_diff",
            ascending=False,
        )
        .head(15)
    )

    print(
        largest_away[
            cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "INTERPRETATION GUIDE"
    )

    print(
        "=" * 80
    )

    print(
        """
This is NOT a pass/fail rule yet.

Things we want to see:

1. High match coverage.
2. Strong positive xG correlation.
3. Small systematic mean bias.
4. Similar xG scale / standard deviation.
5. Differences that look like provider-model
   disagreement rather than broken matching.

Do NOT change frozen V5 based on this test.
"""
    )

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_MATCHES,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print(
        "Matched data:"
    )

    print(
        OUTPUT_MATCHES
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
        "=" * 80
    )

    print(
        "TEST COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()