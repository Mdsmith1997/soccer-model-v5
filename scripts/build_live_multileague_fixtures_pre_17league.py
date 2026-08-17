from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

LIVE = ROOT / "data" / "live"

EVENT_MAP = LIVE / "odds_events_snapshot.csv"
H2H_SNAPSHOT = LIVE / "odds_snapshot.csv"

OUTPUT_FIXTURES = LIVE / "upcoming_fixtures.csv"
OUTPUT_AUDIT = LIVE / "upcoming_fixtures_provider_audit.csv"


# ============================================================
# TARGET LEAGUES
# ============================================================

SPORT_KEY_TO_LEAGUE = {
    "soccer_epl":
        "Premier League",

    "soccer_efl_champ":
        "Championship",

    "soccer_england_league1":
        "League One",

    "soccer_england_league2":
        "League Two",

    "soccer_spain_la_liga":
        "La Liga",

    "soccer_germany_bundesliga":
        "Bundesliga",

    "soccer_germany_bundesliga2":
        "2. Bundesliga",

    "soccer_belgium_first_div":
        "Belgian Pro League",
}


# ============================================================
# XG PROVIDER PLAN
# ============================================================

# UNDERSTAT_PRIMARY:
# Existing preferred V5 xG family where supported.
#
# UNDERSTAT_EXPAND:
# Understat supports this competition, but the local V5
# historical store still needs to be expanded before live use.
#
# FOOTYSTATS_EXPAND:
# FootyStats is the intended coverage source after its
# historical league data is added locally.

LEAGUE_PROVIDER_PLAN = {
    "Premier League":
        "UNDERSTAT_PRIMARY",

    "Bundesliga":
        "UNDERSTAT_PRIMARY",

    "La Liga":
        "UNDERSTAT_EXPAND",

    "Championship":
        "FOOTYSTATS_EXPAND",

    "League One":
        "FOOTYSTATS_EXPAND",

    "League Two":
        "FOOTYSTATS_EXPAND",

    "2. Bundesliga":
        "FOOTYSTATS_EXPAND",

    "Belgian Pro League":
        "FOOTYSTATS_EXPAND",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        ch
        for ch in value
        if not unicodedata.combining(ch)
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

    return " ".join(
        value.split()
    )


def safe_match_id(
    event_id,
    date,
    home_team,
    away_team,
):

    if (
        pd.notna(event_id)
        and
        str(event_id).strip()
    ):

        return str(
            event_id
        ).strip()

    date_part = (
        pd.Timestamp(date)
        .strftime("%Y%m%d")
    )

    home = re.sub(
        r"[^A-Za-z0-9]+",
        "",
        str(home_team),
    )

    away = re.sub(
        r"[^A-Za-z0-9]+",
        "",
        str(away_team),
    )

    return (
        f"LIVE_{date_part}_"
        f"{home}_{away}"
    )


def first_existing_column(
    df,
    candidates,
):

    lookup = {
        str(col).lower():
            col
        for col in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lookup:

            return lookup[
                candidate.lower()
            ]

    return None


# ============================================================
# LOAD EVENT MAP
# ============================================================

def load_event_map():

    if not EVENT_MAP.exists():

        raise FileNotFoundError(
            "\nMissing Odds API event map:\n"
            f"{EVENT_MAP}\n\n"
            "Run first:\n"
            "python scripts/fetch_us_soccer_odds.py\n"
        )

    df = pd.read_csv(
        EVENT_MAP,
        low_memory=False,
    )

    if len(df) == 0:

        raise ValueError(
            "odds_events_snapshot.csv is empty."
        )

    print(
        "Event-map rows:",
        f"{len(df):,}",
    )

    print(
        "Event-map columns:"
    )

    print(
        df.columns.tolist()
    )

    return df


# ============================================================
# STANDARDIZE EVENT MAP
# ============================================================

def standardize_events(
    raw,
):

    df = raw.copy()

    event_col = first_existing_column(
        df,
        [
            "event_id",
            "id",
            "odds_event_id",
        ],
    )

    sport_key_col = first_existing_column(
        df,
        [
            "sport_key",
            "sport",
        ],
    )

    league_col = first_existing_column(
        df,
        [
            "league",
            "league_name",
            "competition",
        ],
    )

    time_col = first_existing_column(
        df,
        [
            "commence_time",
            "start_time",
            "kickoff",
            "date",
        ],
    )

    home_col = first_existing_column(
        df,
        [
            "home_team",
            "home",
        ],
    )

    away_col = first_existing_column(
        df,
        [
            "away_team",
            "away",
        ],
    )

    missing = []

    if time_col is None:
        missing.append(
            "commence_time/date"
        )

    if home_col is None:
        missing.append(
            "home_team"
        )

    if away_col is None:
        missing.append(
            "away_team"
        )

    if (
        sport_key_col is None
        and
        league_col is None
    ):

        missing.append(
            "sport_key or league"
        )

    if missing:

        raise ValueError(
            "Could not identify required "
            "event-map columns:\n"
            +
            "\n".join(
                f" - {item}"
                for item in missing
            )
            +
            "\n\nAvailable columns:\n"
            +
            str(
                df.columns.tolist()
            )
        )

    out = pd.DataFrame()

    if event_col is not None:

        out[
            "odds_event_id"
        ] = df[
            event_col
        ]

    else:

        out[
            "odds_event_id"
        ] = np.nan

    if sport_key_col is not None:

        out[
            "sport_key"
        ] = df[
            sport_key_col
        ].astype(str)

    else:

        out[
            "sport_key"
        ] = ""

    if league_col is not None:

        out[
            "league_raw"
        ] = df[
            league_col
        ].astype(str)

    else:

        out[
            "league_raw"
        ] = ""

    out[
        "commence_time"
    ] = pd.to_datetime(
        df[
            time_col
        ],
        utc=True,
        errors="coerce",
    )

    out[
        "home_team"
    ] = (
        df[
            home_col
        ]
        .astype(str)
        .str.strip()
    )

    out[
        "away_team"
    ] = (
        df[
            away_col
        ]
        .astype(str)
        .str.strip()
    )

    out = out.dropna(
        subset=[
            "commence_time",
        ]
    ).copy()

    out = out[
        (
            out[
                "home_team"
            ]
            !=
            ""
        )
        &
        (
            out[
                "away_team"
            ]
            !=
            ""
        )
    ].copy()

    return out


# ============================================================
# RESOLVE LEAGUE
# ============================================================

def resolve_league(
    row,
):

    sport_key = str(
        row[
            "sport_key"
        ]
    ).strip()

    if (
        sport_key
        in
        SPORT_KEY_TO_LEAGUE
    ):

        return (
            SPORT_KEY_TO_LEAGUE[
                sport_key
            ]
        )

    raw = normalize_text(
        row[
            "league_raw"
        ]
    )

    aliases = {
        "premier league":
            "Premier League",

        "epl":
            "Premier League",

        "championship":
            "Championship",

        "efl championship":
            "Championship",

        "league 1":
            "League One",

        "league one":
            "League One",

        "efl league 1":
            "League One",

        "league 2":
            "League Two",

        "league two":
            "League Two",

        "efl league 2":
            "League Two",

        "la liga":
            "La Liga",

        "la liga spain":
            "La Liga",

        "bundesliga":
            "Bundesliga",

        "bundesliga germany":
            "Bundesliga",

        "bundesliga 2 germany":
            "2. Bundesliga",

        "2 bundesliga":
            "2. Bundesliga",

        "bundesliga 2":
            "2. Bundesliga",

        "belgian pro league":
            "Belgian Pro League",

        "belgium first div":
            "Belgian Pro League",

        "belgian first division a":
            "Belgian Pro League",
    }

    return aliases.get(
        raw
    )


# ============================================================
# ADD H2H COVERAGE CHECK
# ============================================================

def add_h2h_coverage(
    fixtures,
):

    out = fixtures.copy()

    out[
        "has_1x2_odds"
    ] = False

    if not H2H_SNAPSHOT.exists():

        return out

    try:

        odds = pd.read_csv(
            H2H_SNAPSHOT,
            low_memory=False,
        )

    except Exception:

        return out

    if len(odds) == 0:

        return out

    event_col = first_existing_column(
        odds,
        [
            "event_id",
            "odds_event_id",
            "match_id",
        ],
    )

    if event_col is not None:

        available_ids = set(
            odds[
                event_col
            ]
            .dropna()
            .astype(str)
        )

        out[
            "has_1x2_odds"
        ] = (
            out[
                "odds_event_id"
            ]
            .astype(str)
            .isin(
                available_ids
            )
        )

    # Some snapshot formats are book-by-book without
    # the event id. Fall back to team matching.
    if not out[
        "has_1x2_odds"
    ].any():

        home_col = first_existing_column(
            odds,
            [
                "home_team",
                "home",
            ],
        )

        away_col = first_existing_column(
            odds,
            [
                "away_team",
                "away",
            ],
        )

        if (
            home_col is not None
            and
            away_col is not None
        ):

            keys = set(
                zip(
                    odds[
                        home_col
                    ].map(
                        normalize_text
                    ),
                    odds[
                        away_col
                    ].map(
                        normalize_text
                    ),
                )
            )

            out[
                "has_1x2_odds"
            ] = [
                (
                    normalize_text(
                        home
                    ),
                    normalize_text(
                        away
                    ),
                )
                in
                keys
                for home, away
                in zip(
                    out[
                        "home_team"
                    ],
                    out[
                        "away_team"
                    ],
                )
            ]

    return out


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=============================="
    )

    print(
        "BUILD LIVE MULTI-LEAGUE "
        "FIXTURES"
    )

    print(
        "=============================="
    )

    LIVE.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = load_event_map()

    events = standardize_events(
        raw
    )

    events[
        "league"
    ] = events.apply(
        resolve_league,
        axis=1,
    )

    unsupported = events[
        events[
            "league"
        ].isna()
    ].copy()

    events = events[
        events[
            "league"
        ].notna()
    ].copy()

    events[
        "xg_provider_plan"
    ] = events[
        "league"
    ].map(
        LEAGUE_PROVIDER_PLAN
    )

    events[
        "date"
    ] = (
        events[
            "commence_time"
        ]
        .dt.tz_convert(
            None
        )
        .dt.normalize()
    )

    events[
        "match_id"
    ] = [
        safe_match_id(
            event_id,
            date,
            home,
            away,
        )
        for (
            event_id,
            date,
            home,
            away,
        )
        in zip(
            events[
                "odds_event_id"
            ],
            events[
                "date"
            ],
            events[
                "home_team"
            ],
            events[
                "away_team"
            ],
        )
    ]

    events = events.drop_duplicates(
        subset=[
            "match_id",
        ],
        keep="first",
    ).copy()

    events = events.sort_values(
        [
            "commence_time",
            "league",
            "home_team",
        ]
    ).reset_index(
        drop=True
    )

    events = add_h2h_coverage(
        events
    )

    fixtures = events[
        [
            "match_id",
            "date",
            "league",
            "home_team",
            "away_team",
        ]
    ].copy()

    fixtures.to_csv(
        OUTPUT_FIXTURES,
        index=False,
    )

    audit = events[
        [
            "match_id",
            "odds_event_id",
            "commence_time",
            "date",
            "league",
            "sport_key",
            "home_team",
            "away_team",
            "xg_provider_plan",
            "has_1x2_odds",
        ]
    ].copy()

    audit.to_csv(
        OUTPUT_AUDIT,
        index=False,
    )

    print()
    print(
        "Target fixtures:",
        f"{len(fixtures):,}",
    )

    print()
    print(
        "BY LEAGUE"
    )

    print(
        audit[
            "league"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "BY XG PROVIDER PLAN"
    )

    print(
        audit[
            "xg_provider_plan"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "1X2 odds coverage:"
    )

    print(
        f"{int(audit['has_1x2_odds'].sum()):,}"
        f" / {len(audit):,}"
    )

    if len(unsupported):

        print()
        print(
            "Ignored non-target / unresolved "
            "events:",
            f"{len(unsupported):,}",
        )

    print()
    print(
        "Upcoming fixtures:"
    )

    print(
        OUTPUT_FIXTURES
    )

    print()
    print(
        "Provider audit:"
    )

    print(
        OUTPUT_AUDIT
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Expand historical xG stores for "
        "UNDERSTAT_EXPAND and "
        "FOOTYSTATS_EXPAND leagues, then "
        "score this exact fixture file."
    )


if __name__ == "__main__":
    main()
