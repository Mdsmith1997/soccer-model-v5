from pathlib import Path
from difflib import SequenceMatcher
import argparse
import json
import os
import re
import time
import unicodedata

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "footystats_multileague_v5_predictions.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "under11_wave1_event_map.csv"
)

CACHE_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "odds_api"
    / "under11_wave1_events"
)

LEAGUES = {
    "Eredivisie": {
        "sport": "soccer_netherlands_eredivisie",
        "slug": "eredivisie",
    },
    "Primeira Liga": {
        "sport": "soccer_portugal_primeira_liga",
        "slug": "primeira_liga",
    },
}

SEASONS = {
    "2223",
    "2324",
    "2425",
    "2526",
}

# Query sufficiently before the calendar match day
# so the event should still be upcoming.
QUERY_HOURS_BEFORE_DAY = 36

# Historical events endpoint expected cost.
EXPECTED_COST = 1

# Safety ceiling.
MAX_NEW_CREDITS = 1200

MIN_RESERVE = 500


def season_string(x):
    if pd.isna(x):
        return ""

    try:
        f = float(x)

        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return str(x).strip()


def normalize(x):
    x = "" if pd.isna(x) else str(x)

    x = unicodedata.normalize(
        "NFKD",
        x,
    )

    x = "".join(
        c for c in x
        if not unicodedata.combining(c)
    )

    x = x.lower()

    x = re.sub(
        r"[^a-z0-9]+",
        " ",
        x,
    )

    return " ".join(x.split())


ALIASES = {
    # ============================================================
    # Netherlands
    # ============================================================

    "ajax amsterdam": "ajax",
    "afc ajax": "ajax",
    "ajax": "ajax",

    "psv eindhoven": "psv",
    "psv": "psv",

    "feyenoord rotterdam": "feyenoord",
    "feyenoord": "feyenoord",

    "az alkmaar": "az",
    "az": "az",

    "fc twente enschede": "twente",
    "fc twente": "twente",
    "twente": "twente",

    "nec nijmegen": "nec",
    "nec": "nec",

    "vitesse arnhem": "vitesse",
    "vitesse": "vitesse",

    "willem ii": "willem ii",

    # ============================================================
    # Portugal
    # ============================================================

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

    "gd estoril praia": "estoril",
    "estoril praia": "estoril",
    "estoril": "estoril",

    "boavista porto": "boavista",
    "boavista fc": "boavista",
    "boavista": "boavista",

    "cf estrela": "estrela amadora",
    "estrela amadora": "estrela amadora",

    "avs futebol sad": "avs",
    "avs": "avs",

    "alverca": "alverca",
}



def team_key(x):
    x = normalize(x)

    return ALIASES.get(
        x,
        x,
    )


def similarity(a, b):
    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def cache_path(
    league,
    query_time,
):
    slug = LEAGUES[league]["slug"]

    stamp = query_time.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    return (
        CACHE_ROOT
        / slug
        / f"{stamp}.json"
    )


def load_cache(path):
    return json.loads(
        path.read_text()
    )


def save_cache(
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


def historical_events(
    league,
    query_time,
    api_key,
):
    path = cache_path(
        league,
        query_time,
    )

    if path.exists():
        return (
            load_cache(path),
            True,
            None,
        )

    sport = LEAGUES[league]["sport"]

    url = (
        "https://api.the-odds-api.com/v4/"
        f"historical/sports/{sport}/events"
    )

    r = requests.get(
        url,
        params={
            "apiKey": api_key,
            "date": query_time.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        },
        timeout=60,
    )

    quota = {
        "remaining":
            r.headers.get(
                "x-requests-remaining"
            ),
        "used":
            r.headers.get(
                "x-requests-used"
            ),
        "cost":
            r.headers.get(
                "x-requests-last"
            ),
    }

    print(
        "HTTP:",
        r.status_code,
        "| remaining:",
        quota["remaining"],
        "| used:",
        quota["used"],
        "| cost:",
        quota["cost"],
    )

    if r.status_code != 200:
        print(
            "BODY:",
            r.text[:1000],
        )

        r.raise_for_status()

    payload = r.json()

    # Save immediately.
    save_cache(
        path,
        payload,
    )

    return (
        payload,
        False,
        quota,
    )


def find_event(
    payload,
    home,
    away,
    approx_day,
):
    hk = team_key(home)
    ak = team_key(away)

    candidates = []

    for game in payload.get(
        "data",
        [],
    ):
        api_home = team_key(
            game.get("home_team")
        )

        api_away = team_key(
            game.get("away_team")
        )

        kickoff = pd.to_datetime(
            game.get("commence_time"),
            errors="coerce",
            utc=True,
        )

        if pd.isna(kickoff):
            continue

        # FootyStats input only gives us calendar date,
        # so allow adjacent UTC dates.
        day_diff = abs(
            (
                kickoff.floor("D")
                - approx_day.floor("D")
            ).total_seconds()
            / 86400
        )

        if day_diff > 1:
            continue

        direct_home = similarity(
            hk,
            api_home,
        )

        direct_away = similarity(
            ak,
            api_away,
        )

        direct_score = (
            direct_home
            + direct_away
        ) / 2

        reversed_score = (
            similarity(
                hk,
                api_away,
            )
            + similarity(
                ak,
                api_home,
            )
        ) / 2

        # Home/away should agree.
        if reversed_score > direct_score:
            continue

        candidates.append({
            "event": game,
            "api_kickoff": kickoff,
            "name_score": direct_score,
            "day_diff": day_diff,
        })

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            -x["name_score"],
            x["day_diff"],
        )
    )

    best = candidates[0]

    # Conservative threshold.
    if best["name_score"] < 0.72:
        return None

    # Ambiguity protection.
    if len(candidates) > 1:
        second = candidates[1]

        if (
            best["name_score"]
            - second["name_score"]
            < 0.05
            and best["day_diff"]
            == second["day_diff"]
        ):
            return None

    return best


def load_matches():
    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required = [
        "footystats_match_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
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

    df = df[
        df["league"].isin(
            LEAGUES
        )
        & df["season"].isin(
            SEASONS
        )
    ].copy()

    df = df.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
        ]
    )

    df["match_day"] = (
        df["date"]
        .dt.floor("D")
    )

    return (
        df.sort_values(
            [
                "league",
                "match_day",
                "home_team",
            ]
        )
        .reset_index(drop=True)
    )


def build_plan(df):
    rows = []

    for league, g in df.groupby(
        "league"
    ):
        days = (
            g["match_day"]
            .drop_duplicates()
            .sort_values()
        )

        cached = 0

        for day in days:
            query_time = (
                day
                - pd.Timedelta(
                    hours=QUERY_HOURS_BEFORE_DAY
                )
            )

            if cache_path(
                league,
                query_time,
            ).exists():
                cached += 1

        total = len(days)

        new = total - cached

        rows.append({
            "league": league,
            "matches": len(g),
            "match_days": total,
            "cached": cached,
            "new_requests": new,
            "estimated_credits":
                new * EXPECTED_COST,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    args = parser.parse_args()

    df = load_matches()

    print()
    print("=" * 120)
    print(
        "PORTUGAL + EREDIVISIE "
        "HISTORICAL EVENT MAP"
    )
    print("=" * 120)

    print(
        "MODE:",
        "EXECUTE"
        if args.execute
        else "DRY RUN",
    )

    print()
    print(
        "Matches:",
        f"{len(df):,}",
    )

    print()
    print("MATCHES BY LEAGUE / SEASON")

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
    print("EVENT SNAPSHOT PLAN")
    print("=" * 120)

    print(
        plan.to_string(
            index=False
        )
    )

    requests_needed = int(
        plan["new_requests"].sum()
    )

    expected = int(
        plan["estimated_credits"].sum()
    )

    print()
    print(
        "New event requests:",
        f"{requests_needed:,}",
    )

    print(
        "Estimated credits:",
        f"{expected:,}",
    )

    print(
        "Hard ceiling:",
        f"{MAX_NEW_CREDITS:,}",
    )

    if expected > MAX_NEW_CREDITS:
        raise RuntimeError(
            "Event-map cost exceeds "
            "hard safety ceiling."
        )

    if not args.execute:
        print()
        print("DRY RUN COMPLETE ✅")
        print("ZERO API CALLS MADE")
        return

    api_key = os.environ.get(
        "ODDS_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "ODDS_API_KEY is not set."
        )

    snapshots = {}

    credits_this_run = 0

    # --------------------------------------------------------
    # DOWNLOAD / CACHE EVENTS
    # --------------------------------------------------------

    for league, g in df.groupby(
        "league"
    ):
        snapshots[league] = {}

        days = (
            g["match_day"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        print()
        print("#" * 120)
        print(league.upper())
        print("#" * 120)

        for i, day in enumerate(
            days,
            start=1,
        ):
            day = pd.Timestamp(day)

            if day.tzinfo is None:
                day = day.tz_localize(
                    "UTC"
                )

            query_time = (
                day
                - pd.Timedelta(
                    hours=QUERY_HOURS_BEFORE_DAY
                )
            )

            print()
            print(
                f"[{i}/{len(days)}]",
                "match day:",
                day.date(),
                "| query:",
                query_time,
            )

            payload, cached, quota = (
                historical_events(
                    league,
                    query_time,
                    api_key,
                )
            )

            if cached:
                print("CACHE HIT")

            else:
                # Use actual API-reported cost
                # when possible.
                actual_cost = EXPECTED_COST

                if quota:
                    try:
                        actual_cost = int(
                            quota["cost"]
                        )
                    except Exception:
                        pass

                credits_this_run += (
                    actual_cost
                )

                if (
                    credits_this_run
                    > MAX_NEW_CREDITS
                ):
                    raise RuntimeError(
                        "Runtime event credit "
                        "ceiling exceeded."
                    )

                if quota:
                    try:
                        remaining = int(
                            quota["remaining"]
                        )
                    except Exception:
                        remaining = None

                    if (
                        remaining is not None
                        and remaining
                        <= MIN_RESERVE
                    ):
                        raise RuntimeError(
                            "Credit reserve reached."
                        )

                time.sleep(0.05)

            snapshots[
                league
            ][day.date()] = payload

    # --------------------------------------------------------
    # MAP EVENTS
    # --------------------------------------------------------

    results = []

    for _, row in df.iterrows():
        league = row["league"]
        match_day = row["match_day"]

        # Try the nominal date plus adjacent dates.
        # This protects against UTC/local-date differences.
        days_to_try = [
            match_day,
            match_day
            - pd.Timedelta(days=1),
            match_day
            + pd.Timedelta(days=1),
        ]

        match = None

        for d in days_to_try:
            payload = (
                snapshots
                .get(league, {})
                .get(d.date())
            )

            if not payload:
                continue

            candidate = find_event(
                payload,
                row["home_team"],
                row["away_team"],
                match_day,
            )

            if candidate is not None:
                match = candidate
                break

        result = row.to_dict()

        if match is None:
            result.update({
                "odds_api_event_found":
                    False,
                "odds_api_event_id":
                    None,
                "odds_api_home":
                    None,
                "odds_api_away":
                    None,
                "odds_api_kickoff":
                    None,
                "event_match_score":
                    None,
                "calendar_day_diff":
                    None,
            })

        else:
            game = match["event"]

            result.update({
                "odds_api_event_found":
                    True,

                "odds_api_event_id":
                    game.get("id"),

                "odds_api_home":
                    game.get(
                        "home_team"
                    ),

                "odds_api_away":
                    game.get(
                        "away_team"
                    ),

                "odds_api_kickoff":
                    match[
                        "api_kickoff"
                    ],

                "event_match_score":
                    match[
                        "name_score"
                    ],

                "calendar_day_diff":
                    match[
                        "day_diff"
                    ],
            })

        results.append(result)

    out = pd.DataFrame(results)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    found = (
        out[
            "odds_api_event_found"
        ]
        .eq(True)
    )

    print()
    print("=" * 120)
    print("EVENT MAP SUMMARY")
    print("=" * 120)

    print(
        "Rows:",
        f"{len(out):,}",
    )

    print(
        "Events found:",
        f"{found.sum():,}",
    )

    print(
        "Match rate:",
        f"{found.mean():.2%}",
    )

    print()
    print("BY LEAGUE / SEASON")

    audit = (
        out.assign(
            found=found.astype(int)
        )
        .groupby(
            ["league", "season"]
        )
        .agg(
            matches=("found", "size"),
            found=("found", "sum"),
        )
    )

    audit["match_rate"] = (
        audit["found"]
        / audit["matches"]
    )

    print(
        audit.to_string()
    )

    if found.any():
        matched = out[found].copy()

        matched[
            "odds_api_kickoff"
        ] = pd.to_datetime(
            matched[
                "odds_api_kickoff"
            ],
            errors="coerce",
            utc=True,
        )

        matched[
            "true_t24"
        ] = (
            matched[
                "odds_api_kickoff"
            ]
            - pd.Timedelta(
                hours=24
            )
        )

        print()
        print("=" * 120)
        print("TRUE T-24 COST PLAN")
        print("=" * 120)

        rows = []

        for league, g in matched.groupby(
            "league"
        ):
            unique = (
                g["true_t24"]
                .nunique()
            )

            rows.append({
                "league": league,
                "matched_games": len(g),
                "unique_true_t24":
                    unique,
                "future_totals_credits":
                    unique * 10,
            })

        t24 = pd.DataFrame(rows)

        print(
            t24.to_string(
                index=False
            )
        )

        print()
        print(
            "TOTAL FUTURE TRUE-T24 "
            "TOTALS CREDITS:",
            f"{t24['future_totals_credits'].sum():,.0f}",
        )

    print()
    print("Saved:")
    print(OUTPUT)

    print()
    print(
        "Event credits used this run:",
        credits_this_run,
    )


if __name__ == "__main__":
    main()
