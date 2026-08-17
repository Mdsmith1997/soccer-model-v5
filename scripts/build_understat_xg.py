from pathlib import Path
import time
import re
import unicodedata

import numpy as np
import pandas as pd
import soccerdata as sd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_MATCHES_FILE = (
    ROOT
    / "data"
    / "processed"
    / "matches.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "understat_xg_matches.csv"
)

MATCHED_FILE = (
    ROOT
    / "data"
    / "processed"
    / "understat_xg_matched.csv"
)

UNMATCHED_MODEL_FILE = (
    ROOT
    / "data"
    / "processed"
    / "understat_xg_unmatched_model.csv"
)

UNMATCHED_UNDERSTAT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "understat_xg_unmatched_understat.csv"
)

BAD_DATE_MATCH_FILE = (
    ROOT
    / "data"
    / "processed"
    / "understat_xg_bad_date_matches.csv"
)


# ============================================================
# SETTINGS
# ============================================================

LEAGUES = {
    "ENG-Premier League": "Premier League",
    "GER-Bundesliga": "Bundesliga",
}

SEASON_STARTS = range(
    2015,
    2026,
)

REQUEST_PAUSE = 1.0

# We already downloaded the full database.
# True = reuse it and rerun matching instantly.
# False = redownload from Understat.
REUSE_EXISTING_DOWNLOAD = True

# Understat timestamps occasionally shift a fixture
# into the following UTC/calendar date.
MAX_DATE_DIFFERENCE_DAYS = 1


# ============================================================
# TEAM ALIASES
#
# Left side = normalized Understat name
# Right side = exact domestic model name
# ============================================================

TEAM_ALIASES = {

    # ========================================================
    # PREMIER LEAGUE
    # ========================================================

    "manchester united":
        "Man United",

    "manchester city":
        "Man City",

    "newcastle united":
        "Newcastle",

    "wolverhampton wanderers":
        "Wolves",

    "wolves":
        "Wolves",

    "tottenham hotspur":
        "Tottenham",

    "tottenham":
        "Tottenham",

    "west ham united":
        "West Ham",

    "west ham":
        "West Ham",

    "nottingham forest":
        "Nott'm Forest",

    "nottm forest":
        "Nott'm Forest",

    "sheffield united":
        "Sheffield United",

    "leicester city":
        "Leicester",

    "leicester":
        "Leicester",

    "norwich city":
        "Norwich",

    "norwich":
        "Norwich",

    "huddersfield town":
        "Huddersfield",

    "huddersfield":
        "Huddersfield",

    "cardiff city":
        "Cardiff",

    "cardiff":
        "Cardiff",

    "swansea city":
        "Swansea",

    "swansea":
        "Swansea",

    "stoke city":
        "Stoke",

    "stoke":
        "Stoke",

    "west bromwich albion":
        "West Brom",

    "west brom":
        "West Brom",

    "hull city":
        "Hull",

    "hull":
        "Hull",

    "brighton hove albion":
        "Brighton",

    "brighton and hove albion":
        "Brighton",

    "brighton":
        "Brighton",

    "afc bournemouth":
        "Bournemouth",

    "bournemouth":
        "Bournemouth",

    "crystal palace":
        "Crystal Palace",

    "aston villa":
        "Aston Villa",

    "burnley":
        "Burnley",

    "chelsea":
        "Chelsea",

    "arsenal":
        "Arsenal",

    "liverpool":
        "Liverpool",

    "everton":
        "Everton",

    "fulham":
        "Fulham",

    "southampton":
        "Southampton",

    "watford":
        "Watford",

    "brentford":
        "Brentford",

    "leeds":
        "Leeds",

    "leeds united":
        "Leeds",

    "ipswich":
        "Ipswich",

    "ipswich town":
        "Ipswich",

    "luton":
        "Luton",

    "luton town":
        "Luton",

    "sunderland":
        "Sunderland",

    "middlesbrough":
        "Middlesbrough",


    # ========================================================
    # BUNDESLIGA
    # ========================================================

    "bayern munich":
        "Bayern Munich",

    "bayern munchen":
        "Bayern Munich",

    "fc bayern munchen":
        "Bayern Munich",

    "borussia dortmund":
        "Dortmund",

    "dortmund":
        "Dortmund",

    "bayer leverkusen":
        "Leverkusen",

    "bayer 04 leverkusen":
        "Leverkusen",

    "rb leipzig":
        "RB Leipzig",

    "rasenballsport leipzig":
        "RB Leipzig",

    # Critical Understat spelling.
    "borussia m gladbach":
        "M'gladbach",

    "borussia monchengladbach":
        "M'gladbach",

    "borussia mgladbach":
        "M'gladbach",

    "monchengladbach":
        "M'gladbach",

    "eintracht frankfurt":
        "Ein Frankfurt",

    "frankfurt":
        "Ein Frankfurt",

    # Critical Understat spelling.
    "fc cologne":
        "FC Koln",

    "fc koln":
        "FC Koln",

    "1 fc koln":
        "FC Koln",

    "koln":
        "FC Koln",

    "cologne":
        "FC Koln",

    "schalke 04":
        "Schalke 04",

    "fc schalke 04":
        "Schalke 04",

    "hoffenheim":
        "Hoffenheim",

    "1899 hoffenheim":
        "Hoffenheim",

    "tsg hoffenheim":
        "Hoffenheim",

    "hertha berlin":
        "Hertha",

    "hertha bsc":
        "Hertha",

    "hertha":
        "Hertha",

    "werder bremen":
        "Werder Bremen",

    "wolfsburg":
        "Wolfsburg",

    "vfl wolfsburg":
        "Wolfsburg",

    "freiburg":
        "Freiburg",

    "sc freiburg":
        "Freiburg",

    "mainz 05":
        "Mainz",

    "mainz":
        "Mainz",

    "fsv mainz 05":
        "Mainz",

    "1 fsv mainz 05":
        "Mainz",

    "augsburg":
        "Augsburg",

    "fc augsburg":
        "Augsburg",

    "stuttgart":
        "Stuttgart",

    "vfb stuttgart":
        "Stuttgart",

    "union berlin":
        "Union Berlin",

    "1 fc union berlin":
        "Union Berlin",

    "bochum":
        "Bochum",

    "vfl bochum":
        "Bochum",

    "heidenheim":
        "Heidenheim",

    "fc heidenheim":
        "Heidenheim",

    "1 fc heidenheim":
        "Heidenheim",

    "1 fc heidenheim 1846":
        "Heidenheim",

    "darmstadt":
        "Darmstadt",

    "darmstadt 98":
        "Darmstadt",

    "sv darmstadt 98":
        "Darmstadt",

    "fortuna dusseldorf":
        "Fortuna Dusseldorf",

    "dusseldorf":
        "Fortuna Dusseldorf",

    "hamburger sv":
        "Hamburg",

    "hamburg":
        "Hamburg",

    "hannover 96":
        "Hannover",

    "hannover":
        "Hannover",

    "ingolstadt":
        "Ingolstadt",

    "fc ingolstadt 04":
        "Ingolstadt",

    "nurnberg":
        "Nurnberg",

    "1 fc nurnberg":
        "Nurnberg",

    "fc nurnberg":
        "Nurnberg",

    "paderborn":
        "Paderborn",

    "sc paderborn 07":
        "Paderborn",

    "arminia bielefeld":
        "Bielefeld",

    "bielefeld":
        "Bielefeld",

    "greuther furth":
        "Greuther Furth",

    "spvgg greuther furth":
        "Greuther Furth",

    "st pauli":
        "St Pauli",

    "fc st pauli":
        "St Pauli",

    "holstein kiel":
        "Holstein Kiel",

        "fortuna duesseldorf":
        "Fortuna Dusseldorf",

    "nuernberg":
        "Nurnberg",

    "greuther fuerth":
    "Greuther Furth",
}


# ============================================================
# HELPERS
# ============================================================

def season_code(
    start_year,
):
    """
    2015 -> 1516
    2024 -> 2425
    """

    return (
        f"{str(start_year)[-2:]}"
        f"{str(start_year + 1)[-2:]}"
    )


def soccerdata_season(
    start_year,
):
    """
    soccerdata format:
        2024-2025
    """

    return (
        f"{start_year}-"
        f"{start_year + 1}"
    )


def normalize_text(
    value,
):
    """
    Normalize a team name for matching only.
    """

    if pd.isna(
        value
    ):

        return ""

    value = str(
        value
    ).strip()

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = (
        value
        .encode(
            "ascii",
            "ignore",
        )
        .decode(
            "ascii"
        )
    )

    value = value.lower()

    value = value.replace(
        "&",
        " and ",
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def map_team_name(
    name,
):

    normalized = normalize_text(
        name
    )

    if normalized in TEAM_ALIASES:

        return TEAM_ALIASES[
            normalized
        ]

    return str(
        name
    ).strip()


def flatten_index(
    df,
):
    """
    soccerdata often returns a MultiIndex.
    Convert everything into ordinary columns.
    """

    if isinstance(
        df.index,
        pd.MultiIndex,
    ):

        df = df.reset_index()

    elif (
        df.index.name
        is not None
    ):

        df = df.reset_index()

    return df


# ============================================================
# DOWNLOAD ONE LEAGUE / SEASON
# ============================================================

def download_one(
    understat_league,
    model_league,
    start_year,
):

    sd_season = soccerdata_season(
        start_year
    )

    code = season_code(
        start_year
    )

    print(
        f"{code}  "
        f"{model_league:<15} ",
        end="",
        flush=True,
    )

    reader = sd.Understat(
        leagues=understat_league,
        seasons=sd_season,
    )

    stats = (
        reader
        .read_team_match_stats()
    )

    stats = flatten_index(
        stats
    )

    if stats.empty:

        print(
            "0 matches"
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required = [
        "game_id",
        "date",

        "home_team",
        "away_team",

        "home_goals",
        "away_goals",

        "home_xg",
        "away_xg",

        "home_np_xg",
        "away_np_xg",

        "home_expected_points",
        "away_expected_points",

        "home_ppda",
        "away_ppda",

        "home_deep_completions",
        "away_deep_completions",
    ]

    missing = [
        col
        for col in required
        if col not in stats.columns
    ]

    if missing:

        raise ValueError(
            f"Missing Understat columns "
            f"for {model_league} "
            f"{sd_season}: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    stats[
        "date"
    ] = pd.to_datetime(
        stats[
            "date"
        ]
    ).dt.normalize()

    stats[
        "season"
    ] = code

    stats[
        "league"
    ] = model_league

    stats[
        "understat_league"
    ] = understat_league

    stats[
        "understat_home_team"
    ] = stats[
        "home_team"
    ]

    stats[
        "understat_away_team"
    ] = stats[
        "away_team"
    ]

    stats[
        "home_team"
    ] = stats[
        "home_team"
    ].map(
        map_team_name
    )

    stats[
        "away_team"
    ] = stats[
        "away_team"
    ].map(
        map_team_name
    )

    # --------------------------------------------------------
    # NUMERIC TYPES
    # --------------------------------------------------------

    numeric_cols = [
        "home_goals",
        "away_goals",

        "home_xg",
        "away_xg",

        "home_np_xg",
        "away_np_xg",

        "home_expected_points",
        "away_expected_points",

        "home_ppda",
        "away_ppda",

        "home_deep_completions",
        "away_deep_completions",
    ]

    for col in numeric_cols:

        stats[
            col
        ] = pd.to_numeric(
            stats[
                col
            ],
            errors="coerce",
        )

    # --------------------------------------------------------
    # DERIVED MATCH SIGNALS
    # --------------------------------------------------------

    stats[
        "total_xg"
    ] = (
        stats[
            "home_xg"
        ]
        +
        stats[
            "away_xg"
        ]
    )

    stats[
        "xg_diff_home"
    ] = (
        stats[
            "home_xg"
        ]
        -
        stats[
            "away_xg"
        ]
    )

    stats[
        "total_np_xg"
    ] = (
        stats[
            "home_np_xg"
        ]
        +
        stats[
            "away_np_xg"
        ]
    )

    stats[
        "np_xg_diff_home"
    ] = (
        stats[
            "home_np_xg"
        ]
        -
        stats[
            "away_np_xg"
        ]
    )

    stats[
        "expected_points_diff_home"
    ] = (
        stats[
            "home_expected_points"
        ]
        -
        stats[
            "away_expected_points"
        ]
    )

    keep = [
        "season",
        "league",
        "understat_league",

        "game_id",
        "date",

        "home_team",
        "away_team",

        "understat_home_team",
        "understat_away_team",

        "home_goals",
        "away_goals",

        "home_xg",
        "away_xg",
        "total_xg",
        "xg_diff_home",

        "home_np_xg",
        "away_np_xg",
        "total_np_xg",
        "np_xg_diff_home",

        "home_expected_points",
        "away_expected_points",
        "expected_points_diff_home",

        "home_ppda",
        "away_ppda",

        "home_deep_completions",
        "away_deep_completions",
    ]

    stats = stats[
        keep
    ].copy()

    print(
        f"{len(stats):,} matches"
    )

    return stats


# ============================================================
# DOWNLOAD ALL
# ============================================================

def download_all():

    frames = []

    print()
    print("==============================")
    print("DOWNLOADING UNDERSTAT xG")
    print("==============================")
    print()

    for (
        understat_league,
        model_league,
    ) in LEAGUES.items():

        for start_year in (
            SEASON_STARTS
        ):

            try:

                df = download_one(
                    understat_league,
                    model_league,
                    start_year,
                )

                if not df.empty:

                    frames.append(
                        df
                    )

            except Exception as exc:

                print(
                    f"FAILED: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            time.sleep(
                REQUEST_PAUSE
            )

    if not frames:

        raise RuntimeError(
            "No Understat data downloaded."
        )

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    return clean_understat_dataset(
        out
    )


# ============================================================
# CLEAN / REMAP EXISTING UNDERSTAT DATABASE
# ============================================================

def clean_understat_dataset(
    understat,
):

    out = understat.copy()

    out[
        "date"
    ] = pd.to_datetime(
        out[
            "date"
        ]
    ).dt.normalize()

    out[
        "season"
    ] = (
        out[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Re-map from ORIGINAL Understat names.
    # This lets us reuse the existing downloaded CSV even
    # after adding new aliases.
    # --------------------------------------------------------

    if (
        "understat_home_team"
        in out.columns
    ):

        out[
            "home_team"
        ] = (
            out[
                "understat_home_team"
            ]
            .map(
                map_team_name
            )
        )

    else:

        out[
            "understat_home_team"
        ] = out[
            "home_team"
        ]

        out[
            "home_team"
        ] = (
            out[
                "home_team"
            ]
            .map(
                map_team_name
            )
        )

    if (
        "understat_away_team"
        in out.columns
    ):

        out[
            "away_team"
        ] = (
            out[
                "understat_away_team"
            ]
            .map(
                map_team_name
            )
        )

    else:

        out[
            "understat_away_team"
        ] = out[
            "away_team"
        ]

        out[
            "away_team"
        ] = (
            out[
                "away_team"
            ]
            .map(
                map_team_name
            )
        )

    out = (
        out
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

    return out


# ============================================================
# LOAD / DOWNLOAD UNDERSTAT
# ============================================================

def get_understat():

    if (
        REUSE_EXISTING_DOWNLOAD
        and
        OUTPUT_FILE.exists()
    ):

        print()
        print("==============================")
        print("LOADING EXISTING UNDERSTAT DATA")
        print("==============================")
        print()

        print(
            OUTPUT_FILE
        )

        understat = pd.read_csv(
            OUTPUT_FILE
        )

        understat = (
            clean_understat_dataset(
                understat
            )
        )

        print()
        print(
            f"Rows loaded: "
            f"{len(understat):,}"
        )

        print(
            "Existing data re-mapped using "
            "latest aliases ✅"
        )

        return understat

    return download_all()


# ============================================================
# LOAD MODEL MATCHES
# ============================================================

def load_model_matches():

    if not MODEL_MATCHES_FILE.exists():

        raise FileNotFoundError(
            f"Missing model match file:\n"
            f"{MODEL_MATCHES_FILE}"
        )

    model = pd.read_csv(
        MODEL_MATCHES_FILE
    )

    required = [
        "match_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
    ]

    missing = [
        col
        for col in required
        if col not in model.columns
    ]

    if missing:

        print()
        print(
            "Available columns "
            "in matches.csv:"
        )

        print(
            list(
                model.columns
            )
        )

        raise ValueError(
            f"matches.csv missing "
            f"required columns: "
            f"{missing}"
        )

    model[
        "date"
    ] = pd.to_datetime(
        model[
            "date"
        ]
    ).dt.normalize()

    model[
        "season"
    ] = (
        model[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    return model


# ============================================================
# PAIR KEY
# ============================================================

def add_pair_key(
    df,
):

    out = df.copy()

    out[
        "pair_key"
    ] = (
        out[
            "season"
        ].astype(str)
        + "|"
        +
        out[
            "league"
        ].astype(str)
        + "|"
        +
        out[
            "home_team"
        ].map(
            normalize_text
        )
        + "|"
        +
        out[
            "away_team"
        ].map(
            normalize_text
        )
    )

    return out


# ============================================================
# MATCH AUDIT
# ============================================================

def audit_matching(
    understat,
    model,
):

    model_sub = model[
        model[
            "league"
        ].isin(
            LEAGUES.values()
        )
    ].copy()

    model_sub = add_pair_key(
        model_sub
    )

    understat = add_pair_key(
        understat
    )

    # --------------------------------------------------------
    # DUPLICATE PAIR CHECK
    #
    # A home-away league pairing should occur exactly once
    # in a single league season.
    # --------------------------------------------------------

    model_duplicates = model_sub[
        model_sub[
            "pair_key"
        ].duplicated(
            keep=False
        )
    ]

    understat_duplicates = understat[
        understat[
            "pair_key"
        ].duplicated(
            keep=False
        )
    ]

    if not model_duplicates.empty:

        print()
        print(
            "ERROR: DUPLICATE MODEL PAIR KEYS"
        )

        print(
            model_duplicates[
                [
                    "date",
                    "season",
                    "league",
                    "home_team",
                    "away_team",
                ]
            ]
            .head(40)
            .to_string(
                index=False
            )
        )

        raise ValueError(
            "Duplicate model pair keys found."
        )

    if not understat_duplicates.empty:

        print()
        print(
            "ERROR: DUPLICATE UNDERSTAT PAIR KEYS"
        )

        print(
            understat_duplicates[
                [
                    "date",
                    "season",
                    "league",
                    "home_team",
                    "away_team",
                ]
            ]
            .head(40)
            .to_string(
                index=False
            )
        )

        raise ValueError(
            "Duplicate Understat pair keys found."
        )

    # --------------------------------------------------------
    # PAIR MATCH
    # --------------------------------------------------------

    candidates = model_sub.merge(
        understat,
        on="pair_key",
        how="inner",
        suffixes=(
            "_model",
            "_understat",
        ),
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # DATE DIFFERENCE
    # --------------------------------------------------------

    candidates[
        "date_difference_days"
    ] = (
        (
            candidates[
                "date_model"
            ]
            -
            candidates[
                "date_understat"
            ]
        )
        .dt.days
        .abs()
    )

    # --------------------------------------------------------
    # ACCEPT ONLY <= 1 DAY
    # --------------------------------------------------------

    matched = candidates[
        candidates[
            "date_difference_days"
        ]
        <= MAX_DATE_DIFFERENCE_DAYS
    ].copy()

    bad_dates = candidates[
        candidates[
            "date_difference_days"
        ]
        > MAX_DATE_DIFFERENCE_DAYS
    ].copy()

    matched_keys = set(
        matched[
            "pair_key"
        ]
    )

    unmatched_model = model_sub[
        ~model_sub[
            "pair_key"
        ].isin(
            matched_keys
        )
    ].copy()

    unmatched_understat = understat[
        ~understat[
            "pair_key"
        ].isin(
            matched_keys
        )
    ].copy()

    return (
        matched,
        unmatched_model,
        unmatched_understat,
        bad_dates,
        model_sub,
        understat,
    )


# ============================================================
# COVERAGE REPORT
# ============================================================

def print_coverage(
    understat,
    matched,
    unmatched_model,
    unmatched_understat,
    bad_dates,
    model_sub,
):

    print()
    print("==============================")
    print("UNDERSTAT MATCHING COMPLETE")
    print("==============================")
    print()

    print(
        f"Understat matches: "
        f"{len(understat):,}"
    )

    print(
        f"Model PL/Bundesliga matches: "
        f"{len(model_sub):,}"
    )

    print(
        f"Matched: "
        f"{len(matched):,}"
    )

    print(
        f"Bad-date candidates: "
        f"{len(bad_dates):,}"
    )

    if len(model_sub) > 0:

        coverage = (
            len(matched)
            /
            len(model_sub)
            * 100.0
        )

    else:

        coverage = np.nan

    print(
        f"Coverage: "
        f"{coverage:.2f}%"
    )

    # ========================================================
    # DATE DIFFERENCE DISTRIBUTION
    # ========================================================

    print()
    print("==============================")
    print("MATCHED DATE DIFFERENCES")
    print("==============================")
    print()

    if matched.empty:

        print(
            "No matched games."
        )

    else:

        print(
            matched[
                "date_difference_days"
            ]
            .value_counts()
            .sort_index()
            .rename_axis(
                "days"
            )
            .to_string()
        )

    # ========================================================
    # LEAGUE COVERAGE
    # ========================================================

    print()
    print("==============================")
    print("COVERAGE BY LEAGUE")
    print("==============================")
    print()

    rows = []

    for league in (
        LEAGUES.values()
    ):

        model_count = (
            model_sub[
                "league"
            ]
            .eq(
                league
            )
            .sum()
        )

        us_count = (
            understat[
                "league"
            ]
            .eq(
                league
            )
            .sum()
        )

        matched_count = (
            matched[
                "league_model"
            ]
            .eq(
                league
            )
            .sum()
        )

        if model_count:

            league_coverage = (
                matched_count
                /
                model_count
                * 100.0
            )

        else:

            league_coverage = np.nan

        rows.append(
            {
                "league":
                    league,

                "model_matches":
                    model_count,

                "understat_matches":
                    us_count,

                "matched":
                    matched_count,

                "coverage":
                    league_coverage,
            }
        )

    coverage_df = pd.DataFrame(
        rows
    )

    print(
        coverage_df
        .round(2)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SEASON COVERAGE
    # ========================================================

    print()
    print("==============================")
    print("COVERAGE BY SEASON / LEAGUE")
    print("==============================")
    print()

    rows = []

    for season in sorted(
        model_sub[
            "season"
        ].unique()
    ):

        for league in (
            LEAGUES.values()
        ):

            model_mask = (
                (
                    model_sub[
                        "season"
                    ]
                    == season
                )
                &
                (
                    model_sub[
                        "league"
                    ]
                    == league
                )
            )

            model_count = (
                model_mask.sum()
            )

            us_mask = (
                (
                    understat[
                        "season"
                    ]
                    == season
                )
                &
                (
                    understat[
                        "league"
                    ]
                    == league
                )
            )

            us_count = (
                us_mask.sum()
            )

            matched_mask = (
                (
                    matched[
                        "season_model"
                    ]
                    == season
                )
                &
                (
                    matched[
                        "league_model"
                    ]
                    == league
                )
            )

            matched_count = (
                matched_mask.sum()
            )

            if model_count > 0:

                cov = (
                    matched_count
                    /
                    model_count
                    * 100.0
                )

            else:

                cov = np.nan

            rows.append(
                {
                    "season":
                        season,

                    "league":
                        league,

                    "model":
                        model_count,

                    "understat":
                        us_count,

                    "matched":
                        matched_count,

                    "coverage":
                        cov,
                }
            )

    season_table = pd.DataFrame(
        rows
    )

    print(
        season_table
        .round(2)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # BAD DATE EXAMPLES
    # ========================================================

    print()
    print("==============================")
    print("BAD DATE MATCH CANDIDATES")
    print("==============================")

    if bad_dates.empty:

        print(
            "None ✅"
        )

    else:

        print(
            bad_dates[
                [
                    "season_model",
                    "league_model",

                    "home_team_model",
                    "away_team_model",

                    "date_model",
                    "date_understat",

                    "date_difference_days",
                ]
            ]
            .head(40)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # UNMATCHED MODEL
    # ========================================================

    print()
    print("==============================")
    print("UNMATCHED MODEL EXAMPLES")
    print("==============================")

    if unmatched_model.empty:

        print(
            "None ✅"
        )

    else:

        print(
            unmatched_model[
                [
                    "date",
                    "season",
                    "league",
                    "home_team",
                    "away_team",
                ]
            ]
            .head(50)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # UNMATCHED UNDERSTAT
    # ========================================================

    print()
    print("==============================")
    print("UNMATCHED UNDERSTAT EXAMPLES")
    print("==============================")

    if unmatched_understat.empty:

        print(
            "None ✅"
        )

    else:

        print(
            unmatched_understat[
                [
                    "date",
                    "season",
                    "league",

                    "understat_home_team",
                    "understat_away_team",

                    "home_team",
                    "away_team",
                ]
            ]
            .head(50)
            .to_string(
                index=False
            )
        )


# ============================================================
# BUILD CLEAN MATCHED DATABASE
# ============================================================

def build_clean_matched(
    matched,
):

    clean = pd.DataFrame()

    clean[
        "match_id"
    ] = matched[
        "match_id"
    ]

    clean[
        "date"
    ] = matched[
        "date_model"
    ]

    clean[
        "understat_date"
    ] = matched[
        "date_understat"
    ]

    clean[
        "date_difference_days"
    ] = matched[
        "date_difference_days"
    ]

    clean[
        "season"
    ] = matched[
        "season_model"
    ]

    clean[
        "league"
    ] = matched[
        "league_model"
    ]

    clean[
        "home_team"
    ] = matched[
        "home_team_model"
    ]

    clean[
        "away_team"
    ] = matched[
        "away_team_model"
    ]

    # --------------------------------------------------------
    # UNDERSTAT SIGNALS
    # --------------------------------------------------------

    cols = [
        "game_id",

        "home_xg",
        "away_xg",
        "total_xg",
        "xg_diff_home",

        "home_np_xg",
        "away_np_xg",
        "total_np_xg",
        "np_xg_diff_home",

        "home_expected_points",
        "away_expected_points",
        "expected_points_diff_home",

        "home_ppda",
        "away_ppda",

        "home_deep_completions",
        "away_deep_completions",
    ]

    for col in cols:

        clean[
            col
        ] = matched[
            col
        ]

    clean = (
        clean
        .sort_values(
            [
                "date",
                "league",
                "home_team",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return clean


# ============================================================
# VALIDATION
# ============================================================

def validate(
    clean_matched,
    model_sub,
    understat,
):

    print()
    print("==============================")
    print("VALIDATION")
    print("==============================")

    # --------------------------------------------------------
    # DUPLICATE MATCH IDS
    # --------------------------------------------------------

    if (
        clean_matched[
            "match_id"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate matched match_id values."
        )

    print(
        "No duplicate matched match IDs ✅"
    )

    # --------------------------------------------------------
    # xG COMPLETENESS
    # --------------------------------------------------------

    if (
        clean_matched[
            [
                "home_xg",
                "away_xg",
            ]
        ]
        .isna()
        .any()
        .any()
    ):

        raise ValueError(
            "Matched games contain missing xG."
        )

    print(
        "All matched games have xG ✅"
    )

    # --------------------------------------------------------
    # npxG COMPLETENESS
    # --------------------------------------------------------

    if (
        clean_matched[
            [
                "home_np_xg",
                "away_np_xg",
            ]
        ]
        .isna()
        .any()
        .any()
    ):

        print(
            "WARNING: Some matched games "
            "have missing npxG."
        )

    else:

        print(
            "All matched games have npxG ✅"
        )

    # --------------------------------------------------------
    # EXACT DATASET SIZE
    # --------------------------------------------------------

    print()
    print(
        f"Model eligible matches: "
        f"{len(model_sub):,}"
    )

    print(
        f"Understat matches: "
        f"{len(understat):,}"
    )

    print(
        f"Clean matches: "
        f"{len(clean_matched):,}"
    )

    # --------------------------------------------------------
    # DATE DIFFERENCE
    # --------------------------------------------------------

    if (
        clean_matched[
            "date_difference_days"
        ]
        .max()
        >
        MAX_DATE_DIFFERENCE_DAYS
    ):

        raise ValueError(
            "Accepted match exceeds "
            "date tolerance."
        )

    print(
        "All accepted date differences "
        f"<= {MAX_DATE_DIFFERENCE_DAYS} day ✅"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("BUILDING UNDERSTAT xG DATABASE")
    print("==============================")

    # ========================================================
    # GET UNDERSTAT DATA
    # ========================================================

    understat = get_understat()

    # Always rewrite using newest aliases.
    understat.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # MODEL MATCHES
    # ========================================================

    model = load_model_matches()

    # ========================================================
    # MATCH AUDIT
    # ========================================================

    (
        matched,
        unmatched_model,
        unmatched_understat,
        bad_dates,
        model_sub,
        understat,
    ) = audit_matching(
        understat,
        model,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print_coverage(
        understat,
        matched,
        unmatched_model,
        unmatched_understat,
        bad_dates,
        model_sub,
    )

    # ========================================================
    # CLEAN MATCHED DATABASE
    # ========================================================

    clean_matched = (
        build_clean_matched(
            matched
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    clean_matched.to_csv(
        MATCHED_FILE,
        index=False,
    )

    unmatched_model.to_csv(
        UNMATCHED_MODEL_FILE,
        index=False,
    )

    unmatched_understat.to_csv(
        UNMATCHED_UNDERSTAT_FILE,
        index=False,
    )

    bad_dates.to_csv(
        BAD_DATE_MATCH_FILE,
        index=False,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    validate(
        clean_matched,
        model_sub,
        understat,
    )

    # ========================================================
    # FILES
    # ========================================================

    print()
    print("==============================")
    print("FILES SAVED")
    print("==============================")

    print()
    print(
        OUTPUT_FILE
    )

    print()
    print(
        MATCHED_FILE
    )

    print()
    print(
        UNMATCHED_MODEL_FILE
    )

    print()
    print(
        UNMATCHED_UNDERSTAT_FILE
    )

    print()
    print(
        BAD_DATE_MATCH_FILE
    )


if __name__ == "__main__":
    main()