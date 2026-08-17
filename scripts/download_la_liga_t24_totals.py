from pathlib import Path
import argparse
import json
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "footystats_multileague_v5_predictions.csv"
)

RAW_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "odds_api"
    / "la_liga_under11_t24"
)

OUTPUT_BOOKS = (
    ROOT
    / "data"
    / "processed"
    / "la_liga_under11_t24_bookmakers.csv"
)

OUTPUT_GAMES = (
    ROOT
    / "data"
    / "processed"
    / "la_liga_under11_t24_games.csv"
)

LEAGUES = {
    "La Liga": {
        "sport_key": "soccer_spain_la_liga",
        "cache_slug": "la_liga",
    },
}

# EXACTLY four completed seasons.
SEASONS = {
    "2324",
    "2425",
    "2526",
}

REGION = "us"
MARKET = "totals"

HOURS_BEFORE = 24

EXPECTED_COST_PER_REQUEST = 10

# Hard experiment-level guard.
MAX_NEW_CREDITS = 4500

# Don't consume the account down to zero.
MIN_CREDITS_RESERVE = 500

REQUEST_SLEEP = 0.10


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value):
    value = "" if pd.isna(value) else str(value)

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        c
        for c in value
        if not unicodedata.combining(c)
    )

    value = value.lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return " ".join(
        value.split()
    )


def poisson_under_25(
    home_lambda,
    away_lambda,
):
    mu = (
        float(home_lambda)
        + float(away_lambda)
    )

    return (
        np.exp(-mu)
        * (
            1
            + mu
            + (mu ** 2) / 2
        )
    )


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def season_string(value):
    if pd.isna(value):
        return ""

    # Handles 2223.0 from CSV inference.
    try:
        f = float(value)

        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return str(value).strip()


# ============================================================
# CACHE
# ============================================================

def cache_path(
    league,
    target_timestamp,
):
    config = LEAGUES[league]

    stamp = target_timestamp.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    return (
        RAW_ROOT
        / config["cache_slug"]
        / f"{stamp}.json"
    )


def save_json(
    path,
    payload,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
    )


def load_json(path):
    return json.loads(
        path.read_text()
    )


# ============================================================
# API
# ============================================================

def fetch_snapshot(
    league,
    timestamp,
    api_key,
):
    path = cache_path(
        league,
        timestamp,
    )

    if path.exists():
        return (
            load_json(path),
            True,
            None,
        )

    sport = (
        LEAGUES[league]["sport_key"]
    )

    url = (
        "https://api.the-odds-api.com/v4/"
        f"historical/sports/{sport}/odds"
    )

    params = {
        "apiKey": api_key,
        "regions": REGION,
        "markets": MARKET,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "date": timestamp.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }

    r = requests.get(
        url,
        params=params,
        timeout=60,
    )

    remaining = r.headers.get(
        "x-requests-remaining"
    )

    used = r.headers.get(
        "x-requests-used"
    )

    cost = r.headers.get(
        "x-requests-last"
    )

    print(
        "HTTP:",
        r.status_code,
        "| remaining:",
        remaining,
        "| used:",
        used,
        "| cost:",
        cost,
    )

    if r.status_code != 200:
        print(
            "ERROR BODY:",
            r.text[:1000],
        )

        r.raise_for_status()

    payload = r.json()

    # Save immediately so interruption is resumable.
    save_json(
        path,
        payload,
    )

    quota = {
        "remaining": remaining,
        "used": used,
        "cost": cost,
    }

    return (
        payload,
        False,
        quota,
    )


# ============================================================
# MATCHING
# ============================================================

ALIASES = {
    # Netherlands
    "psv eindhoven": "psv",
    "psv": "psv",
    "ajax amsterdam": "ajax",
    "afc ajax": "ajax",
    "ajax": "ajax",
    "feyenoord rotterdam": "feyenoord",
    "feyenoord": "feyenoord",
    "az alkmaar": "az",
    "az": "az",
    "fc twente": "twente",
    "twente": "twente",

    # Portugal
    "sporting clube de portugal": "sporting cp",
    "sporting lisbon": "sporting cp",
    "sporting cp": "sporting cp",
    "sl benfica": "benfica",
    "benfica": "benfica",
    "fc porto": "porto",
    "porto": "porto",
    "sporting braga": "braga",
    "sc braga": "braga",
    "braga": "braga",

    # Italy
    "roma": "roma",
    "as roma": "roma",

    "atalanta": "atalanta",
    "atalanta bc": "atalanta",

    "hellas verona": "hellas verona",
    "hellas verona fc": "hellas verona",

    # Spain / La Liga
    "sevilla fc": "sevilla",
    "sevilla": "sevilla",

    "athletic club bilbao": "athletic bilbao",
    "athletic bilbao": "athletic bilbao",

    "celta de vigo": "celta vigo",
    "celta vigo": "celta vigo",

    "getafe cf": "getafe",
    "getafe": "getafe",

    "rcd mallorca": "mallorca",
    "mallorca": "mallorca",

    "valencia cf": "valencia",
    "valencia": "valencia",

    "fc barcelona": "barcelona",
    "barcelona": "barcelona",

    "girona fc": "girona",
    "girona": "girona",

    "deportivo alaves": "alaves",
    "alaves": "alaves",

    "ud las palmas": "las palmas",
    "las palmas": "las palmas",

    "rcd espanyol": "espanyol",
    "espanyol": "espanyol",

    "cadiz": "cadiz cf",
    "cadiz cf": "cadiz cf",

    "real valladolid": "valladolid",
    "valladolid": "valladolid",

    "levante ud": "levante",
    "levante": "levante",

    "real oviedo": "oviedo",
    "oviedo": "oviedo",
}


def team_key(value):
    x = normalize_text(value)

    return ALIASES.get(
        x,
        x,
    )


def event_lookup(payload):
    lookup = {}

    for game in payload.get(
        "data",
        [],
    ):
        home = team_key(
            game.get("home_team")
        )

        away = team_key(
            game.get("away_team")
        )

        lookup[
            (home, away)
        ] = game

    return lookup


def find_game(
    lookup,
    home,
    away,
):
    hk = team_key(home)
    ak = team_key(away)

    exact = lookup.get(
        (hk, ak)
    )

    if exact is not None:
        return exact

    # Conservative fallback:
    # avoid fuzzy guesses that could contaminate
    # a historical backtest.
    return None


# ============================================================
# EXTRACT EXACT O/U 2.5
# ============================================================

def extract_exact_25(
    league,
    source_row,
    game,
    requested_timestamp,
    snapshot_timestamp,
):
    rows = []

    for book in game.get(
        "bookmakers",
        [],
    ):
        bookmaker_key = book.get("key")
        bookmaker_title = book.get("title")

        for market in book.get(
            "markets",
            [],
        ):
            if market.get("key") != "totals":
                continue

            over = None
            under = None

            for outcome in market.get(
                "outcomes",
                [],
            ):
                point = safe_float(
                    outcome.get("point")
                )

                if (
                    pd.isna(point)
                    or abs(point - 2.5) > 1e-9
                ):
                    continue

                name = (
                    str(
                        outcome.get(
                            "name",
                            "",
                        )
                    )
                    .strip()
                    .lower()
                )

                price = safe_float(
                    outcome.get("price")
                )

                if name == "over":
                    over = price

                elif name == "under":
                    under = price

            if (
                over is None
                or under is None
                or pd.isna(over)
                or pd.isna(under)
            ):
                continue

            rows.append({
                "league": league,
                "season":
                    source_row["season"],

                "footystats_match_id":
                    source_row[
                        "footystats_match_id"
                    ],

                "date":
                    source_row["date"],

                "home_team":
                    source_row["home_team"],

                "away_team":
                    source_row["away_team"],

                "home_goals":
                    source_row["home_goals"],

                "away_goals":
                    source_row["away_goals"],

                "home_lambda":
                    source_row["home_lambda"],

                "away_lambda":
                    source_row["away_lambda"],

                "p_under_raw":
                    source_row[
                        "p_under_raw"
                    ],

                "requested_t24":
                    requested_timestamp,

                "snapshot_timestamp":
                    snapshot_timestamp,

                "odds_api_event_id":
                    game.get("id"),

                "odds_api_commence_time":
                    game.get(
                        "commence_time"
                    ),

                "odds_api_home":
                    game.get(
                        "home_team"
                    ),

                "odds_api_away":
                    game.get(
                        "away_team"
                    ),

                "bookmaker_key":
                    bookmaker_key,

                "bookmaker":
                    bookmaker_title,

                "market_last_update":
                    market.get(
                        "last_update"
                    ),

                "over_odds":
                    over,

                "under_odds":
                    under,
            })

    return rows


# ============================================================
# CONSENSUS MARKET
# ============================================================

def build_game_consensus(
    bookmaker_df,
):
    if bookmaker_df.empty:
        return pd.DataFrame()

    rows = []

    group_cols = [
        "league",
        "season",
        "footystats_match_id",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_lambda",
        "away_lambda",
        "p_under_raw",
        "requested_t24",
        "snapshot_timestamp",
        "odds_api_event_id",
        "odds_api_commence_time",
    ]

    for keys, g in bookmaker_df.groupby(
        group_cols,
        dropna=False,
    ):
        values = dict(
            zip(
                group_cols,
                keys,
            )
        )

        valid = g[
            (
                pd.to_numeric(
                    g["over_odds"],
                    errors="coerce",
                ) > 1
            )
            & (
                pd.to_numeric(
                    g["under_odds"],
                    errors="coerce",
                ) > 1
            )
        ].copy()

        if valid.empty:
            continue

        # This matches the historical approach:
        # average O/U market price across books.
        avg_over = (
            pd.to_numeric(
                valid["over_odds"],
                errors="coerce",
            )
            .mean()
        )

        avg_under = (
            pd.to_numeric(
                valid["under_odds"],
                errors="coerce",
            )
            .mean()
        )

        # RAW implied probability.
        market_p_under = (
            1.0 / avg_under
        )

        p_under_raw = float(
            values["p_under_raw"]
        )

        raw_edge = (
            p_under_raw
            - market_p_under
        )

        under_ev = (
            p_under_raw
            * avg_under
            - 1.0
        )

        actual_total = (
            float(values["home_goals"])
            + float(values["away_goals"])
        )

        actual_under = (
            actual_total < 2.5
        )

        rows.append({
            **values,

            "bookmakers_exact25":
                len(valid),

            "avg_over_odds":
                avg_over,

            "avg_under_odds":
                avg_under,

            "market_p_under":
                market_p_under,

            "under_edge_raw":
                raw_edge,

            "under_ev_raw":
                under_ev,

            "actual_total":
                actual_total,

            "actual_under":
                int(actual_under),

            "qualifies_under_11":
                int(
                    raw_edge >= 0.11
                ),

            # Flat one-unit result at average price.
            "under_profit_if_bet":
                (
                    avg_under - 1.0
                    if actual_under
                    else -1.0
                ),
        })

    return pd.DataFrame(rows)


# ============================================================
# PREPARE INPUT
# ============================================================

def load_input():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            INPUT_PATH
        )

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    required = [
        "footystats_match_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_lambda",
        "away_lambda",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Input missing columns: {missing}"
        )

    df["season"] = (
        df["season"]
        .map(season_string)
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True,
    )

    for c in [
        "home_lambda",
        "away_lambda",
        "home_goals",
        "away_goals",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = df[
        df["league"].isin(
            LEAGUES.keys()
        )
    ].copy()

    df = df[
        df["season"].isin(
            SEASONS
        )
    ].copy()

    df = df.dropna(
        subset=[
            "date",
            "home_lambda",
            "away_lambda",
            "home_goals",
            "away_goals",
        ]
    ).copy()

    # Exact reproduction already independently verified.
    df["p_under_raw"] = [
        poisson_under_25(h, a)
        for h, a in zip(
            df["home_lambda"],
            df["away_lambda"],
        )
    ]

    # IMPORTANT:
    # footystats V5 date timestamps have shared kickoff
    # structure in this project. T-24 is frozen here.
    df["target_t24"] = (
        df["date"]
        - pd.Timedelta(
            hours=HOURS_BEFORE
        )
    )

    return (
        df.sort_values(
            [
                "league",
                "target_t24",
                "home_team",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# PLAN
# ============================================================

def build_plan(df):
    rows = []

    for league, x in df.groupby(
        "league"
    ):
        targets = (
            x["target_t24"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        cached = sum(
            cache_path(
                league,
                ts,
            ).exists()
            for ts in targets
        )

        new_requests = (
            len(targets)
            - cached
        )

        rows.append({
            "league": league,
            "matches": len(x),
            "unique_t24":
                len(targets),
            "cached":
                cached,
            "new_requests":
                new_requests,
            "estimated_new_credits":
                new_requests
                * EXPECTED_COST_PER_REQUEST,
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually call The Odds API. "
            "Without this flag the script "
            "is DRY-RUN ONLY."
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 120)
    print(
        "LA LIGA — "
        "FOUR-SEASON EXACT T-24 TOTALS"
    )
    print("=" * 120)

    print()
    print("Mode:",
          "EXECUTE" if args.execute
          else "DRY RUN")

    print("Region:", REGION)
    print("Market:", MARKET)
    print("T-minus:", HOURS_BEFORE, "hours")
    print(
        "Seasons:",
        ", ".join(
            sorted(SEASONS)
        ),
    )

    df = load_input()

    print()
    print("Matches:", f"{len(df):,}")

    print()
    print("Matches by league / season:")
    print(
        df.groupby(
            ["league", "season"]
        )
        .size()
        .to_string()
    )

    plan = build_plan(df)

    print()
    print("=" * 120)
    print("CREDIT PLAN")
    print("=" * 120)

    print(
        plan.to_string(
            index=False
        )
    )

    total_new_requests = int(
        plan["new_requests"].sum()
    )

    expected_credits = int(
        plan[
            "estimated_new_credits"
        ].sum()
    )

    print()
    print(
        "New API requests:",
        f"{total_new_requests:,}",
    )

    print(
        "Estimated maximum new credits:",
        f"{expected_credits:,}",
    )

    print(
        "Hard credit ceiling:",
        f"{MAX_NEW_CREDITS:,}",
    )

    if expected_credits > MAX_NEW_CREDITS:
        raise RuntimeError(
            "Projected usage exceeds "
            f"{MAX_NEW_CREDITS:,} credits. "
            "ABORTING BEFORE API CALLS."
        )

    if not args.execute:
        print()
        print("=" * 120)
        print("DRY RUN COMPLETE ✅")
        print("=" * 120)
        print()
        print(
            "ZERO API requests were made."
        )
        print()
        print(
            "If this plan is correct, run:"
        )
        print()
        print(
            "python "
            "scripts/"
            "download_portugal_"
            "eredivisie_t24_totals.py "
            "--execute"
        )
        return

    api_key = os.environ.get(
        "ODDS_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "ODDS_API_KEY is not set."
        )

    print()
    print("=" * 120)
    print("DOWNLOADING")
    print("=" * 120)

    snapshots = {}

    requested_new_credits = 0

    for league, league_df in df.groupby(
        "league"
    ):
        targets = (
            league_df[
                "target_t24"
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        snapshots[league] = {}

        print()
        print("#" * 120)
        print(league.upper())
        print("#" * 120)

        for i, timestamp in enumerate(
            targets,
            start=1,
        ):
            print()
            print(
                f"[{i}/{len(targets)}]",
                timestamp,
            )

            payload, from_cache, quota = (
                fetch_snapshot(
                    league,
                    timestamp,
                    api_key,
                )
            )

            if from_cache:
                print("CACHE HIT")

            else:
                requested_new_credits += (
                    EXPECTED_COST_PER_REQUEST
                )

                if (
                    requested_new_credits
                    > MAX_NEW_CREDITS
                ):
                    raise RuntimeError(
                        "Runtime credit ceiling "
                        "exceeded. Stopping."
                    )

                if quota:
                    remaining = quota.get(
                        "remaining"
                    )

                    if remaining is not None:
                        try:
                            remaining_num = int(
                                remaining
                            )
                        except ValueError:
                            remaining_num = None

                        if (
                            remaining_num
                            is not None
                            and remaining_num
                            <= MIN_CREDITS_RESERVE
                        ):
                            raise RuntimeError(
                                "Credit reserve reached. "
                                "Stopping safely."
                            )

                time.sleep(
                    REQUEST_SLEEP
                )

            snapshots[
                league
            ][timestamp] = payload

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    print()
    print("=" * 120)
    print("EXTRACTING EXACT O/U 2.5")
    print("=" * 120)

    bookmaker_rows = []

    missing_events = []

    for _, row in df.iterrows():
        league = row["league"]
        target = row["target_t24"]

        payload = (
            snapshots
            .get(league, {})
            .get(target)
        )

        if payload is None:
            missing_events.append({
                "league": league,
                "season": row["season"],
                "date": row["date"],
                "home_team":
                    row["home_team"],
                "away_team":
                    row["away_team"],
                "reason":
                    "snapshot_missing",
            })
            continue

        lookup = event_lookup(
            payload
        )

        game = find_game(
            lookup,
            row["home_team"],
            row["away_team"],
        )

        if game is None:
            missing_events.append({
                "league": league,
                "season": row["season"],
                "date": row["date"],
                "home_team":
                    row["home_team"],
                "away_team":
                    row["away_team"],
                "reason":
                    "event_not_found",
            })
            continue

        snapshot_timestamp = (
            payload.get(
                "timestamp"
            )
        )

        extracted = extract_exact_25(
            league=league,
            source_row=row,
            game=game,
            requested_timestamp=target,
            snapshot_timestamp=
                snapshot_timestamp,
        )

        if not extracted:
            missing_events.append({
                "league": league,
                "season": row["season"],
                "date": row["date"],
                "home_team":
                    row["home_team"],
                "away_team":
                    row["away_team"],
                "reason":
                    "no_exact_25_market",
            })
            continue

        bookmaker_rows.extend(
            extracted
        )

    bookmaker_df = pd.DataFrame(
        bookmaker_rows
    )

    game_df = build_game_consensus(
        bookmaker_df
    )

    OUTPUT_BOOKS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bookmaker_df.to_csv(
        OUTPUT_BOOKS,
        index=False,
    )

    game_df.to_csv(
        OUTPUT_GAMES,
        index=False,
    )

    missing_path = (
        ROOT
        / "data"
        / "processed"
        / "la_liga_under11_t24_missing.csv"
    )

    pd.DataFrame(
        missing_events
    ).to_csv(
        missing_path,
        index=False,
    )

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    print()
    print("=" * 120)
    print("DOWNLOAD / EXTRACTION AUDIT")
    print("=" * 120)

    print(
        "Input matches:",
        f"{len(df):,}",
    )

    print(
        "Matches with exact 2.5:",
        f"{len(game_df):,}",
    )

    print(
        "Missing / unavailable:",
        f"{len(missing_events):,}",
    )

    if len(df):
        coverage = (
            100
            * len(game_df)
            / len(df)
        )
    else:
        coverage = 0

    print(
        "Exact-2.5 coverage:",
        f"{coverage:.2f}%",
    )

    if not game_df.empty:
        print()
        print("Coverage by league / season:")

        covered = (
            game_df.groupby(
                ["league", "season"]
            )
            .size()
            .rename("with_odds")
        )

        total = (
            df.groupby(
                ["league", "season"]
            )
            .size()
            .rename("matches")
        )

        coverage_table = (
            pd.concat(
                [total, covered],
                axis=1,
            )
            .fillna(0)
        )

        coverage_table[
            "coverage_pct"
        ] = (
            100
            * coverage_table[
                "with_odds"
            ]
            / coverage_table[
                "matches"
            ]
        )

        print(
            coverage_table.to_string()
        )

        print()
        print("RAW UNDER >= 11% QUALIFIERS:")

        qualifiers = game_df[
            game_df[
                "qualifies_under_11"
            ] == 1
        ]

        print(
            qualifiers.groupby(
                ["league", "season"]
            )
            .size()
            .to_string()
            if len(qualifiers)
            else "NONE"
        )

    print()
    print("Saved:")
    print(OUTPUT_BOOKS)
    print(OUTPUT_GAMES)
    print(missing_path)

    print()
    print(
        "Actual maximum credits requested "
        "during this run:",
        requested_new_credits,
    )


if __name__ == "__main__":
    main()
