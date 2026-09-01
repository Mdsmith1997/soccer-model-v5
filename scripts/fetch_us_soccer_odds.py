from pathlib import Path
from datetime import datetime, timezone
import os
import re
import time

import pandas as pd
import requests


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

LIVE_DIR = (
    ROOT
    / "data"
    / "live"
)

LIVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ------------------------------------------------------------
# CURRENT SNAPSHOTS
# ------------------------------------------------------------

OUTPUT_H2H = (
    LIVE_DIR
    / "odds_snapshot.csv"
)

OUTPUT_ALL_MARKETS = (
    LIVE_DIR
    / "odds_markets_snapshot.csv"
)

OUTPUT_EVENTS = (
    LIVE_DIR
    / "odds_events_snapshot.csv"
)

OUTPUT_BOOKMAKERS = (
    LIVE_DIR
    / "odds_bookmaker_snapshot.csv"
)

OUTPUT_DISCOVERY = (
    LIVE_DIR
    / "odds_sport_discovery.csv"
)

# ------------------------------------------------------------
# HISTORICAL SNAPSHOT LEDGER
#
# These files are NEVER meant to represent only the latest
# price. Every successful run is appended so we can later
# reconstruct:
#
# T-24h
# T-6h
# T-1h
# T-30m
# close
#
# and calculate real CLV.
# ------------------------------------------------------------

HISTORY_H2H = (
    LIVE_DIR
    / "odds_h2h_history.csv"
)

HISTORY_MARKETS = (
    LIVE_DIR
    / "odds_markets_history.csv"
)

HISTORY_EVENTS = (
    LIVE_DIR
    / "odds_events_history.csv"
)

# ------------------------------------------------------------
# OPTIONAL V5 LIVE FILE
#
# If present, BTTS can be restricted to matches where we
# actually have valid model coverage.
# ------------------------------------------------------------

LIVE_PREDICTIONS_FILE = (
    LIVE_DIR
    / "v5_live_predictions.csv"
)


# ============================================================
# API
# ============================================================

BASE_URL = (
    "https://api.the-odds-api.com/v4"
)

API_KEY = os.getenv(
    "THE_ODDS_API_KEY"
)


# ============================================================
# QUOTA SETTINGS
# ============================================================

# Keep this many credits in reserve.
MIN_REMAINING_CREDITS = int(
    os.getenv(
        "ODDS_API_MIN_REMAINING",
        "20",
    )
)

# BTTS is much more expensive because it is queried
# event-by-event.
BTTS_MIN_REMAINING_CREDITS = int(
    os.getenv(
        "ODDS_API_BTTS_MIN_REMAINING",
        "50",
    )
)

# Hard safety cap even if BTTS is enabled.
MAX_BTTS_EVENTS = int(
    os.getenv(
        "MAX_BTTS_EVENTS",
        "25",
    )
)

REQUESTS_REMAINING = None
REQUESTS_USED = None
LAST_REQUEST_COST = None


# ============================================================
# FETCH SETTINGS
# ============================================================

# Both U.S. bookmaker regions.
REGIONS = [
    "us",
    "us2",
]

ODDS_FORMAT = "decimal"

# Core markets fetched league-by-league.
FEATURED_MARKETS = [
    "h2h",
    "totals",
]

# ============================================================
# CEMENTED LIVE MARKET COVERAGE
# ============================================================

TOTALS_FETCH_LEAGUES = {
    "Premier League",
    "Bundesliga",
    "Belgian Pro League",
    "Eliteserien",
}

BTTS_FETCH_LEAGUES = {
    "Swiss Super League",
    "Super Lig",
    "Segunda División",
}

# ------------------------------------------------------------
# BTTS DEFAULTS TO OFF.
#
# To enable:
#
# export FETCH_BTTS=1
#
# ------------------------------------------------------------

FETCH_BTTS = (
    os.getenv(
        "FETCH_BTTS",
        "0",
    )
    ==
    "1"
)

# When enabled, only fetch BTTS for matches with valid
# V5 probabilities when the live prediction file exists.
BTTS_ONLY_MODEL_COVERAGE = (
    os.getenv(
        "BTTS_ONLY_MODEL_COVERAGE",
        "1",
    )
    ==
    "1"
)


# ============================================================
# TARGET LEAGUES
#
# Explicit keys are safer than fuzzy matching.
# ============================================================

TARGET_LEAGUES = {
    "Premier League": {
        "sport_key": "soccer_epl",
    },

    "Championship": {
        "sport_key": "soccer_efl_champ",
    },

    "League One": {
        "sport_key": "soccer_england_league1",
    },

    "League Two": {
        "sport_key": "soccer_england_league2",
    },

    "La Liga": {
        "sport_key": "soccer_spain_la_liga",
    },

    "Segunda División": {
        "sport_key": "soccer_spain_segunda_division",
    },

    "Bundesliga": {
        "sport_key": "soccer_germany_bundesliga",
    },

    "2. Bundesliga": {
        "sport_key": "soccer_germany_bundesliga2",
    },

    "Belgian Pro League": {
        "sport_key": "soccer_belgium_first_div",
    },

    "Eredivisie": {
        "sport_key": "soccer_netherlands_eredivisie",
    },

    "Serie A": {
        "sport_key": "soccer_italy_serie_a",
    },

    "Swiss Super League": {
        "sport_key": "soccer_switzerland_superleague",
    },

    "Super Lig": {
        "sport_key": "soccer_turkey_super_league",
    },

    "Primeira Liga": {
        "sport_key": "soccer_portugal_primeira_liga",
    },

    "Eliteserien": {
        "sport_key": "soccer_norway_eliteserien",
    },

    "MLS": {
        "sport_key": "soccer_usa_mls",
    },
}



# ============================================================
# CUSTOM ERRORS
# ============================================================

class OddsAPIError(
    RuntimeError
):
    pass


class OddsAPIQuotaError(
    RuntimeError
):
    pass


# ============================================================
# TIME
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    )


# ============================================================
# ID HELPERS
# ============================================================

def clean_id_text(
    value,
):

    return re.sub(
        r"[^A-Za-z0-9]+",
        "",
        str(value),
    )


def make_live_match_id(
    commence_time,
    home_team,
    away_team,
):

    dt = pd.Timestamp(
        commence_time
    )

    if dt.tzinfo is None:

        dt = dt.tz_localize(
            "UTC"
        )

    date_text = (
        dt
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
# QUOTA HELPERS
# ============================================================

def parse_int_header(
    value,
):

    if value is None:

        return None

    try:

        return int(
            value
        )

    except (
        ValueError,
        TypeError,
    ):

        return None


def update_quota_state(
    response,
):

    global REQUESTS_REMAINING
    global REQUESTS_USED
    global LAST_REQUEST_COST

    REQUESTS_REMAINING = parse_int_header(
        response.headers.get(
            "x-requests-remaining"
        )
    )

    REQUESTS_USED = parse_int_header(
        response.headers.get(
            "x-requests-used"
        )
    )

    LAST_REQUEST_COST = parse_int_header(
        response.headers.get(
            "x-requests-last"
        )
    )


def quota_string():

    return (
        f"cost={LAST_REQUEST_COST} "
        f"used={REQUESTS_USED} "
        f"remaining={REQUESTS_REMAINING}"
    )


def ensure_quota(
    minimum,
    purpose,
):

    if REQUESTS_REMAINING is None:

        return

    if REQUESTS_REMAINING <= minimum:

        raise OddsAPIQuotaError(
            f"Stopping {purpose}. "
            f"Only {REQUESTS_REMAINING} "
            f"API credits remain. "
            f"Reserve threshold: {minimum}."
        )


# ============================================================
# API REQUEST
#
# Important:
# We intentionally do NOT call response.raise_for_status()
# because its exception contains the complete URL including
# the API key.
# ============================================================

def api_get(
    path,
    params=None,
    optional=False,
):

    if not API_KEY:

        raise OddsAPIError(
            "THE_ODDS_API_KEY is not set."
        )

    if params is None:

        params = {}

    params = dict(
        params
    )

    params[
        "apiKey"
    ] = API_KEY

    if optional:

        ensure_quota(
            BTTS_MIN_REMAINING_CREDITS,
            "optional API calls",
        )

    else:

        ensure_quota(
            MIN_REMAINING_CREDITS,
            "core odds collection",
        )

    url = (
        BASE_URL
        +
        path
    )

    # --------------------------------------------------------
    # OFFICIAL LIVE MODEL WINDOW
    #
    # Apply NOW -> T+72h ONLY to odds endpoints.
    # Scores/results requests must retain their normal horizon
    # so settlement history is not affected.
    # --------------------------------------------------------

    request_params = dict(params or {})

    if "/odds" in path:

        now_utc = datetime.now(timezone.utc)

        window_end_utc = (
            now_utc
            +
            pd.Timedelta(hours=72)
        )

        request_params["commenceTimeFrom"] = (
            now_utc
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

        request_params["commenceTimeTo"] = (
            window_end_utc
            
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

    try:

        response = requests.get(
            url,
            params=request_params,
            timeout=30,
        )

    except requests.RequestException as exc:

        raise OddsAPIError(
            f"Network error while requesting "
            f"{path}: "
            f"{type(exc).__name__}"
        ) from None

    update_quota_state(
        response
    )

    print(
        f"API: {path}"
    )

    print(
        f"  status="
        f"{response.status_code} "
        f"{quota_string()}"
    )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    if (
        200
        <=
        response.status_code
        <
        300
    ):

        try:

            return response.json()

        except ValueError:

            raise OddsAPIError(
                f"Invalid JSON returned "
                f"for {path}."
            ) from None

    # --------------------------------------------------------
    # AUTH / QUOTA
    # --------------------------------------------------------

    if response.status_code == 401:

        raise OddsAPIError(
            "The Odds API returned 401 Unauthorized. "
            "The API key may be invalid, expired, "
            "revoked, or the account may have "
            "insufficient access/quota."
        )

    if response.status_code == 429:

        raise OddsAPIQuotaError(
            "The Odds API returned 429 "
            "rate/quota limit reached."
        )

    # --------------------------------------------------------
    # OTHER ERROR
    # --------------------------------------------------------

    try:

        body = response.json()

        message = (
            body.get(
                "message"
            )
            or
            body.get(
                "error"
            )
            or
            str(body)
        )

    except ValueError:

        message = (
            response.text[
                :300
            ]
        )

    raise OddsAPIError(
        f"The Odds API returned "
        f"HTTP {response.status_code} "
        f"for {path}: "
        f"{message}"
    )


# ============================================================
# ACTIVE SPORTS
# ============================================================

def load_active_sports():

    data = api_get(
        "/sports"
    )

    rows = []

    for sport in data:

        group = str(
            sport.get(
                "group",
                "",
            )
        )

        if (
            "soccer"
            not in
            group.lower()
        ):

            continue

        rows.append(
            {
                "sport_key":
                    sport.get(
                        "key"
                    ),

                "group":
                    group,

                "title":
                    sport.get(
                        "title"
                    ),

                "description":
                    sport.get(
                        "description"
                    ),

                "active":
                    sport.get(
                        "active"
                    ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUTPUT_DISCOVERY,
        index=False,
    )

    return df


# ============================================================
# EXACT LEAGUE DISCOVERY
# ============================================================

def discover_target_sports(
    sports,
):

    rows = []

    for league, config in (
        TARGET_LEAGUES.items()
    ):

        sport_key = (
            config[
                "sport_key"
            ]
        )

        matched = sports[
            sports[
                "sport_key"
            ]
            ==
            sport_key
        ]

        if len(matched) == 1:

            source = (
                matched.iloc[
                    0
                ]
            )

            rows.append(
                {
                    "league":
                        league,

                    "sport_key":
                        sport_key,

                    "title":
                        source[
                            "title"
                        ],

                    "description":
                        source[
                            "description"
                        ],

                    "group":
                        source[
                            "group"
                        ],
                }
            )

        elif len(matched) == 0:

            print(
                f"NOT CURRENTLY ACTIVE: "
                f"{league} "
                f"({sport_key})"
            )

        else:

            raise OddsAPIError(
                "Duplicate active sport key "
                f"returned for {sport_key}."
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# BOOKMAKER POLICY
#
# Since the API request itself is restricted to:
#
# regions=us,us2
#
# we keep every bookmaker returned from those U.S. regions.
#
# This prevents us from accidentally excluding a new U.S.
# sportsbook and maximizes line-shopping opportunities.
# ============================================================

def accept_bookmaker(
    bookmaker,
):

    key = bookmaker.get(
        "key"
    )

    title = bookmaker.get(
        "title"
    )

    if not key:

        return False

    if not title:

        return False

    return True


# ============================================================
# PARSE FEATURED EVENT
# ============================================================

def parse_featured_event(
    event,
    league,
    snapshot_time,
):

    event_id = event.get(
        "id"
    )

    sport_key = event.get(
        "sport_key"
    )

    commence_time = pd.to_datetime(
        event.get(
            "commence_time"
        ),
        utc=True,
    )

    home_team = event.get(
        "home_team"
    )

    away_team = event.get(
        "away_team"
    )

    match_id = make_live_match_id(
        commence_time,
        home_team,
        away_team,
    )

    h2h_rows = []

    market_rows = []

    bookmaker_rows = []

    for bookmaker in event.get(
        "bookmakers",
        [],
    ):

        if not accept_bookmaker(
            bookmaker
        ):

            continue

        bookmaker_key = (
            bookmaker.get(
                "key"
            )
        )

        bookmaker_title = (
            bookmaker.get(
                "title"
            )
        )

        bookmaker_rows.append(
            {
                "snapshot_time":
                    snapshot_time,

                "match_id":
                    match_id,

                "event_id":
                    event_id,

                "sport_key":
                    sport_key,

                "league":
                    league,

                "home_team":
                    home_team,

                "away_team":
                    away_team,

                "commence_time":
                    commence_time,

                "bookmaker_key":
                    bookmaker_key,

                "bookmaker":
                    bookmaker_title,
            }
        )

        for market in bookmaker.get(
            "markets",
            [],
        ):

            market_key = (
                market.get(
                    "key"
                )
            )

            last_update = (
                market.get(
                    "last_update"
                )
            )

            outcomes = (
                market.get(
                    "outcomes",
                    [],
                )
            )

            # =================================================
            # 1X2
            # =================================================

            if market_key == "h2h":

                home_odds = None
                draw_odds = None
                away_odds = None

                for outcome in outcomes:

                    name = (
                        outcome.get(
                            "name"
                        )
                    )

                    price = (
                        outcome.get(
                            "price"
                        )
                    )

                    if name == home_team:

                        home_odds = price

                    elif name == away_team:

                        away_odds = price

                    elif (
                        str(
                            name
                        )
                        .strip()
                        .lower()
                        ==
                        "draw"
                    ):

                        draw_odds = price

                if (
                    home_odds is not None
                    and
                    draw_odds is not None
                    and
                    away_odds is not None
                ):

                    h2h_rows.append(
                        {
                            "match_id":
                                match_id,

                            "bookmaker":
                                bookmaker_title,

                            "bookmaker_key":
                                bookmaker_key,

                            "home_odds":
                                float(
                                    home_odds
                                ),

                            "draw_odds":
                                float(
                                    draw_odds
                                ),

                            "away_odds":
                                float(
                                    away_odds
                                ),

                            "snapshot_time":
                                snapshot_time,

                            "event_id":
                                event_id,

                            "sport_key":
                                sport_key,

                            "league":
                                league,

                            "home_team":
                                home_team,

                            "away_team":
                                away_team,

                            "commence_time":
                                commence_time,

                            "market_last_update":
                                last_update,
                        }
                    )

            # =================================================
            # TOTALS
            # =================================================

            elif market_key == "totals":

                for outcome in outcomes:

                    price = (
                        outcome.get(
                            "price"
                        )
                    )

                    point = (
                        outcome.get(
                            "point"
                        )
                    )

                    if price is None:

                        continue

                    market_rows.append(
                        {
                            "snapshot_time":
                                snapshot_time,

                            "match_id":
                                match_id,

                            "event_id":
                                event_id,

                            "sport_key":
                                sport_key,

                            "league":
                                league,

                            "home_team":
                                home_team,

                            "away_team":
                                away_team,

                            "commence_time":
                                commence_time,

                            "bookmaker":
                                bookmaker_title,

                            "bookmaker_key":
                                bookmaker_key,

                            "market":
                                "TOTALS",

                            "selection":
                                outcome.get(
                                    "name"
                                ),

                            "point":
                                point,

                            "decimal_odds":
                                float(
                                    price
                                ),

                            "market_last_update":
                                last_update,
                        }
                    )

    return (
        h2h_rows,
        market_rows,
        bookmaker_rows,
    )


# ============================================================
# FEATURED ODDS
# ============================================================

def fetch_featured_odds(
    league_map,
):

    snapshot_time = utc_now()

    all_h2h = []

    all_markets = []

    all_bookmakers = []

    all_events = []

    regions_text = (
        ",".join(
            REGIONS
        )
    )

    markets_text = (
        ",".join(
            FEATURED_MARKETS
        )
    )

    stopped_for_quota = False

    for row in (
        league_map
        .itertuples(
            index=False
        )
    ):

        try:

            ensure_quota(
                MIN_REMAINING_CREDITS,
                "core odds collection",
            )

        except OddsAPIQuotaError as exc:

            print()
            print(
                "QUOTA GUARD:"
            )

            print(
                exc
            )

            stopped_for_quota = True

            break

        print()
        print(
            "=" * 90
        )

        print(
            f"FETCHING: "
            f"{row.league}"
        )

        print(
            f"SPORT KEY: "
            f"{row.sport_key}"
        )

        print(
            "=" * 90
        )

        try:

            data = api_get(
                (
                    f"/sports/"
                    f"{row.sport_key}"
                    f"/odds"
                ),
                params={
                    "regions":
                        regions_text,

                    "markets":
                        (
                            "h2h,totals"
                            if row.league in TOTALS_FETCH_LEAGUES
                            else "h2h"
                        ),

                    "oddsFormat":
                        ODDS_FORMAT,

                    "dateFormat":
                        "iso",
                },
                optional=False,
            )

        except OddsAPIQuotaError as exc:

            print(
                exc
            )

            stopped_for_quota = True

            break

        except OddsAPIError as exc:

            print(
                f"ERROR fetching "
                f"{row.league}: "
                f"{exc}"
            )

            continue

        print(
            f"Events returned: "
            f"{len(data):,}"
        )

        for event in data:

            event_id = (
                event.get(
                    "id"
                )
            )

            commence_time = (
                pd.to_datetime(
                    event.get(
                        "commence_time"
                    ),
                    utc=True,
                )
            )

            home_team = (
                event.get(
                    "home_team"
                )
            )

            away_team = (
                event.get(
                    "away_team"
                )
            )

            match_id = make_live_match_id(
                commence_time,
                home_team,
                away_team,
            )

            all_events.append(
                {
                    "snapshot_time":
                        snapshot_time,

                    "match_id":
                        match_id,

                    "event_id":
                        event_id,

                    "sport_key":
                        row.sport_key,

                    "league":
                        row.league,

                    "commence_time":
                        commence_time,

                    "home_team":
                        home_team,

                    "away_team":
                        away_team,
                }
            )

            (
                h2h_rows,
                market_rows,
                bookmaker_rows,
            ) = parse_featured_event(
                event,
                row.league,
                snapshot_time,
            )

            all_h2h.extend(
                h2h_rows
            )

            all_markets.extend(
                market_rows
            )

            all_bookmakers.extend(
                bookmaker_rows
            )

        time.sleep(
            0.10
        )

    return (
        pd.DataFrame(
            all_events
        ),
        pd.DataFrame(
            all_h2h
        ),
        pd.DataFrame(
            all_markets
        ),
        pd.DataFrame(
            all_bookmakers
        ),
        stopped_for_quota,
    )


# ============================================================
# MODEL-COVERED IDS FOR OPTIONAL BTTS
# ============================================================

def load_model_covered_match_ids():

    # --------------------------------------------------------
    # BTTS MODEL COVERAGE
    #
    # BTTS specialists operate on:
    #   Swiss Super League
    #   Super Lig
    #   Segunda División
    #
    # Core and expansion predictions may live in separate
    # files, so discover ALL current V5 prediction CSVs rather
    # than restricting discovery to EPL/Bundesliga files.
    # --------------------------------------------------------

    model_files = []

    for candidate in LIVE_DIR.glob("*v5*prediction*.csv"):

        if candidate.is_file():
            model_files.append(candidate)

    combined_ids = set()
    source_counts = {}

    seen_paths = set()

    for path in model_files:

        path = Path(path)

        if path in seen_paths:
            continue

        seen_paths.add(path)

        if not path.exists():
            continue

        try:

            df = pd.read_csv(
                path,
                low_memory=False,
            )

        except Exception as exc:

            print(
                f"Could not read BTTS model coverage source "
                f"{path.name}: {exc}"
            )

            continue

        # A prediction file must contain a usable match/event ID.
        id_col = None

        if "event_id" in df.columns:
            id_col = "event_id"

        elif "match_id" in df.columns:
            id_col = "match_id"

        if id_col is None:
            continue

        # Prefer V5 1X2 completeness when available.
        prob_cols = [
            "p_home_v5",
            "p_draw_v5",
            "p_away_v5",
        ]

        if all(
            col in df.columns
            for col in prob_cols
        ):

            for col in prob_cols:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

            valid = (
                df[prob_cols]
                .notna()
                .all(axis=1)
            )

        else:

            # Expansion prediction files may not expose the
            # exact core 1X2 probability columns. Presence of
            # the current event ID is sufficient here; BTTS
            # feature validation happens downstream.
            valid = df[id_col].notna()

        ids = set(
            df.loc[
                valid,
                id_col,
            ]
            .dropna()
            .astype(str)
        )

        source_counts[
            path.name
        ] = len(ids)

        combined_ids.update(ids)

    print()
    print("=" * 80)
    print("BTTS MODEL COVERAGE")
    print("=" * 80)

    for name, count in source_counts.items():

        print(
            f"{name:45s} "
            f"{count:4d}"
        )

    print(
        f"{'Combined unique model IDs':45s} "
        f"{len(combined_ids):4d}"
    )

    if len(combined_ids) == 0:
        return None

    return combined_ids


# ============================================================
# CHOOSE BTTS EVENTS
# ============================================================

def select_btts_events(
    events,
):

    if len(events) == 0:

        return events.copy()

    selected = events.copy()

    # --------------------------------------------------------
    # CEMENTED BTTS LEAGUE FILTER
    # --------------------------------------------------------
    # Only these leagues can generate an official BTTS bet.
    # Filter BEFORE event-level BTTS API requests to conserve credits.

    if "league" in selected.columns:

        before = len(selected)

        selected = selected[
            selected["league"].isin(
                BTTS_FETCH_LEAGUES
            )
        ].copy()

        print(
            f"BTTS cemented-league filter: "
            f"{before:,} -> "
            f"{len(selected):,} events"
        )

    # --------------------------------------------------------
    # MODEL COVERAGE FILTER
    # --------------------------------------------------------

    if BTTS_ONLY_MODEL_COVERAGE:

        covered_ids = (
            load_model_covered_match_ids()
        )

        if covered_ids is not None:

            before = len(
                selected
            )

            # Model prediction match_id values are the
            # native Odds API event IDs. The live odds dataframe
            # also contains a separately generated match_id, so
            # coverage must be joined against event_id.
            coverage_id_col = (
                "event_id"
                if "event_id" in selected.columns
                else "match_id"
            )

            selected = selected[
                selected[
                    coverage_id_col
                ]
                .astype(str)
                .isin(
                    covered_ids
                )
            ].copy()

            print(
                f"BTTS coverage ID column: "
                f"{coverage_id_col}"
            )

            print(
                f"BTTS model-coverage filter: "
                f"{before:,} -> "
                f"{len(selected):,} events"
            )

    # --------------------------------------------------------
    # NEAREST KICKOFF FIRST
    # --------------------------------------------------------

    if (
        "commence_time"
        in
        selected.columns
    ):

        selected[
            "commence_time"
        ] = pd.to_datetime(
            selected[
                "commence_time"
            ],
            utc=True,
            errors="coerce",
        )

        selected = (
            selected
            .sort_values(
                "commence_time"
            )
        )

    # --------------------------------------------------------
    # HARD CAP
    # --------------------------------------------------------

    if (
        len(selected)
        >
        MAX_BTTS_EVENTS
    ):

        print(
            f"BTTS event cap applied: "
            f"{len(selected):,} -> "
            f"{MAX_BTTS_EVENTS:,}"
        )

        selected = (
            selected
            .head(
                MAX_BTTS_EVENTS
            )
            .copy()
        )

    return selected


# ============================================================
# BTTS
# ============================================================

def fetch_btts(
    events,
):

    if not FETCH_BTTS:

        print()
        print(
            "BTTS fetch disabled."
        )

        return pd.DataFrame()

    selected_events = (
        select_btts_events(
            events
        )
    )

    if len(selected_events) == 0:

        print()
        print(
            "No BTTS events selected."
        )

        return pd.DataFrame()

    rows = []

    snapshot_time = utc_now()

    regions_text = (
        ",".join(
            REGIONS
        )
    )

    print()
    print(
        f"BTTS events selected: "
        f"{len(selected_events):,}"
    )

    for index, event in enumerate(
        selected_events.itertuples(
            index=False
        ),
        start=1,
    ):

        try:

            ensure_quota(
                BTTS_MIN_REMAINING_CREDITS,
                "BTTS collection",
            )

        except OddsAPIQuotaError as exc:

            print()
            print(
                "BTTS QUOTA GUARD:"
            )

            print(
                exc
            )

            break

        print(
            f"BTTS "
            f"{index}/"
            f"{len(selected_events)}: "
            f"{event.home_team} "
            f"vs "
            f"{event.away_team}"
        )

        btts_regions_text = (
            "uk"
            if event.sport_key == "soccer_turkey_super_league"
            else regions_text
        )

        try:

            data = api_get(
                (
                    f"/sports/"
                    f"{event.sport_key}"
                    f"/events/"
                    f"{event.event_id}"
                    f"/odds"
                ),
                params={
                    "regions":
                        btts_regions_text,

                    "markets":
                        "btts",

                    "oddsFormat":
                        ODDS_FORMAT,

                    "dateFormat":
                        "iso",
                },
                optional=True,
            )

        except OddsAPIQuotaError as exc:

            print(
                exc
            )

            break

        except OddsAPIError as exc:

            print(
                f"  BTTS unavailable: "
                f"{exc}"
            )

            continue

        for bookmaker in data.get(
            "bookmakers",
            [],
        ):

            if not accept_bookmaker(
                bookmaker
            ):

                continue

            bookmaker_key = (
                bookmaker.get(
                    "key"
                )
            )

            bookmaker_title = (
                bookmaker.get(
                    "title"
                )
            )

            for market in bookmaker.get(
                "markets",
                [],
            ):

                if (
                    market.get(
                        "key"
                    )
                    !=
                    "btts"
                ):

                    continue

                for outcome in market.get(
                    "outcomes",
                    [],
                ):

                    price = (
                        outcome.get(
                            "price"
                        )
                    )

                    if price is None:

                        continue

                    rows.append(
                        {
                            "snapshot_time":
                                snapshot_time,

                            "match_id":
                                event.match_id,

                            "event_id":
                                event.event_id,

                            "sport_key":
                                event.sport_key,

                            "league":
                                event.league,

                            "home_team":
                                event.home_team,

                            "away_team":
                                event.away_team,

                            "commence_time":
                                event.commence_time,

                            "bookmaker":
                                bookmaker_title,

                            "bookmaker_key":
                                bookmaker_key,

                            "market":
                                "BTTS",

                            "selection":
                                outcome.get(
                                    "name"
                                ),

                            "point":
                                None,

                            "decimal_odds":
                                float(
                                    price
                                ),

                            "market_last_update":
                                market.get(
                                    "last_update"
                                ),
                        }
                    )

        time.sleep(
            0.10
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# HISTORY APPENDER
# ============================================================

def append_history(
    current,
    history_file,
    duplicate_subset,
):

    if len(current) == 0:

        return

    new = current.copy()

    if history_file.exists():

        try:

            old = pd.read_csv(
                history_file,
                low_memory=False,
            )

        except Exception:

            old = pd.DataFrame()

        history = pd.concat(
            [
                old,
                new,
            ],
            ignore_index=True,
        )

    else:

        history = (
            new.copy()
        )

    available_subset = [
        col
        for col in duplicate_subset
        if col in history.columns
    ]

    if available_subset:

        history = history.drop_duplicates(
            subset=available_subset,
            keep="last",
        )

    if (
        "snapshot_time"
        in
        history.columns
    ):

        history[
            "snapshot_time"
        ] = pd.to_datetime(
            history[
                "snapshot_time"
            ],
            utc=True,
            errors="coerce",
        )

        history = history.sort_values(
            [
                "snapshot_time",
            ]
        )

    history.to_csv(
        history_file,
        index=False,
    )


# ============================================================
# SAVE CURRENT + HISTORY
# ============================================================

def save_outputs(
    events,
    h2h,
    markets,
    bookmakers,
):

    # --------------------------------------------------------
    # CURRENT
    # --------------------------------------------------------

    events.to_csv(
        OUTPUT_EVENTS,
        index=False,
    )

    h2h.to_csv(
        OUTPUT_H2H,
        index=False,
    )

    markets.to_csv(
        OUTPUT_ALL_MARKETS,
        index=False,
    )

    bookmakers.to_csv(
        OUTPUT_BOOKMAKERS,
        index=False,
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    append_history(
        events,
        HISTORY_EVENTS,
        [
            "snapshot_time",
            "event_id",
        ],
    )

    append_history(
        h2h,
        HISTORY_H2H,
        [
            "snapshot_time",
            "event_id",
            "bookmaker_key",
        ],
    )

    append_history(
        markets,
        HISTORY_MARKETS,
        [
            "snapshot_time",
            "event_id",
            "bookmaker_key",
            "market",
            "selection",
            "point",
        ],
    )


# ============================================================
# BEST 1X2
# ============================================================

def build_best_h2h(
    h2h,
):

    if len(h2h) == 0:

        return pd.DataFrame()

    rows = []

    for match_id, sub in (
        h2h.groupby(
            "match_id"
        )
    ):

        first = (
            sub.iloc[
                0
            ]
        )

        home = sub.loc[
            sub[
                "home_odds"
            ].idxmax()
        ]

        draw = sub.loc[
            sub[
                "draw_odds"
            ].idxmax()
        ]

        away = sub.loc[
            sub[
                "away_odds"
            ].idxmax()
        ]

        rows.append(
            {
                "match_id":
                    match_id,

                "league":
                    first[
                        "league"
                    ],

                "commence_time":
                    first[
                        "commence_time"
                    ],

                "home_team":
                    first[
                        "home_team"
                    ],

                "away_team":
                    first[
                        "away_team"
                    ],

                "home_odds":
                    home[
                        "home_odds"
                    ],

                "home_book":
                    home[
                        "bookmaker"
                    ],

                "draw_odds":
                    draw[
                        "draw_odds"
                    ],

                "draw_book":
                    draw[
                        "bookmaker"
                    ],

                "away_odds":
                    away[
                        "away_odds"
                    ],

                "away_book":
                    away[
                        "bookmaker"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


def print_best_h2h(
    h2h,
):

    print()
    print(
        "=" * 125
    )

    print(
        "BEST AVAILABLE U.S. 1X2 PRICES"
    )

    print(
        "=" * 125
    )

    table = build_best_h2h(
        h2h
    )

    if len(table) == 0:

        print(
            "No U.S. 1X2 prices returned."
        )

        return

    print(
        table
        .sort_values(
            [
                "commence_time",
                "league",
                "home_team",
            ]
        )
        [
            [
                "league",
                "home_team",
                "away_team",
                "home_odds",
                "home_book",
                "draw_odds",
                "draw_book",
                "away_odds",
                "away_book",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# MARKET COVERAGE
# ============================================================

def print_market_coverage(
    market_df,
):

    print()
    print(
        "=" * 105
    )

    print(
        "U.S. MARKET COVERAGE"
    )

    print(
        "=" * 105
    )

    if len(market_df) == 0:

        print(
            "No totals/BTTS markets returned."
        )

        return

    summary = (
        market_df
        .groupby(
            [
                "league",
                "market",
            ]
        )[
            "match_id"
        ]
        .nunique()
        .unstack(
            fill_value=0
        )
    )

    print(
        summary.to_string()
    )


# ============================================================
# BOOKMAKER COVERAGE
# ============================================================

def print_bookmaker_coverage(
    h2h,
):

    print()
    print(
        "=" * 105
    )

    print(
        "U.S. BOOKMAKER COVERAGE"
    )

    print(
        "=" * 105
    )

    if len(h2h) == 0:

        print(
            "No bookmaker rows."
        )

        return

    table = (
        h2h
        .groupby(
            [
                "bookmaker",
            ]
        )[
            "match_id"
        ]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    print(
        table.to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=============================="
    )

    print(
        "FETCHING U.S. SOCCER ODDS"
    )

    print(
        "=============================="
    )

    print()

    if not API_KEY:

        raise RuntimeError(
            "\nTHE_ODDS_API_KEY is not set.\n\n"
            "Set it locally with:\n"
            "export THE_ODDS_API_KEY='YOUR_NEW_KEY'\n"
        )

    print(
        f"Regions: "
        f"{','.join(REGIONS)}"
    )

    print(
        f"Core markets: "
        f"{','.join(FEATURED_MARKETS)}"
    )

    print(
        f"BTTS enabled: "
        f"{FETCH_BTTS}"
    )

    print(
        f"Core quota reserve: "
        f"{MIN_REMAINING_CREDITS}"
    )

    print(
        f"BTTS quota reserve: "
        f"{BTTS_MIN_REMAINING_CREDITS}"
    )

    # ========================================================
    # DISCOVERY
    # ========================================================

    print()
    print(
        "Discovering active soccer leagues..."
    )

    try:

        sports = load_active_sports()

    except (
        OddsAPIError,
        OddsAPIQuotaError,
    ) as exc:

        raise RuntimeError(
            str(
                exc
            )
        ) from None

    print(
        f"Active soccer competitions: "
        f"{len(sports):,}"
    )

    league_map = discover_target_sports(
        sports
    )

    print()
    print(
        "=" * 100
    )

    print(
        "TARGET LEAGUE DISCOVERY"
    )

    print(
        "=" * 100
    )

    if len(league_map):

        print(
            league_map.to_string(
                index=False
            )
        )

    else:

        raise RuntimeError(
            "No target leagues are "
            "currently active."
        )

    # ========================================================
    # CORE ODDS
    # ========================================================

    (
        events,
        h2h,
        featured_markets,
        bookmakers,
        stopped_for_quota,
    ) = fetch_featured_odds(
        league_map
    )

    # ========================================================
    # BTTS
    #
    # Never run if the core fetch itself had to stop
    # because of quota.
    # ========================================================

    if stopped_for_quota:

        print()
        print(
            "Skipping BTTS because core "
            "collection stopped for quota."
        )

        btts = pd.DataFrame()

    else:

        btts = fetch_btts(
            events
        )

    # ========================================================
    # COMBINE OTHER MARKETS
    # ========================================================

    market_frames = []

    if len(featured_markets):

        market_frames.append(
            featured_markets
        )

    if len(btts):

        market_frames.append(
            btts
        )

    if market_frames:

        all_markets = pd.concat(
            market_frames,
            ignore_index=True,
        )

    else:

        all_markets = pd.DataFrame()

    # ========================================================
    # SAVE
    # ========================================================

    save_outputs(
        events,
        h2h,
        all_markets,
        bookmakers,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "=============================="
    )

    print(
        "ODDS FETCH COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        f"Events: "
        f"{events['match_id'].nunique() if len(events) else 0:,}"
    )

    print(
        f"1X2 bookmaker rows: "
        f"{len(h2h):,}"
    )

    print(
        f"Other market rows: "
        f"{len(all_markets):,}"
    )

    print(
        f"API used: "
        f"{REQUESTS_USED}"
    )

    print(
        f"API remaining: "
        f"{REQUESTS_REMAINING}"
    )

    print_best_h2h(
        h2h
    )

    print_market_coverage(
        all_markets
    )

    print_bookmaker_coverage(
        h2h
    )

    # ========================================================
    # OUTPUTS
    # ========================================================

    print()
    print(
        "Current 1X2:"
    )

    print(
        OUTPUT_H2H
    )

    print()

    print(
        "Current totals + BTTS:"
    )

    print(
        OUTPUT_ALL_MARKETS
    )

    print()

    print(
        "Current event map:"
    )

    print(
        OUTPUT_EVENTS
    )

    print()

    print(
        "1X2 history:"
    )

    print(
        HISTORY_H2H
    )

    print()

    print(
        "Market history:"
    )

    print(
        HISTORY_MARKETS
    )

    print()

    print(
        "Event history:"
    )

    print(
        HISTORY_EVENTS
    )

    print()
    print(
        "Snapshots preserved for "
        "future CLV analysis ✅"
    )

    print(
        "Best U.S. prices retained "
        "book-by-book ✅"
    )

    if not FETCH_BTTS:

        print(
            "BTTS disabled to conserve "
            "API credits ✅"
        )

    if stopped_for_quota:

        print(
            "Quota guard activated before "
            "credits were exhausted ✅"
        )


if __name__ == "__main__":
    main()