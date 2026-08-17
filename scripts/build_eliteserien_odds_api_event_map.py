from pathlib import Path
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "data/processed"
    / "footystats_multileague_v5_predictions.csv"
)

OUTPUT_PATH = (
    ROOT / "data/processed"
    / "eliteserien_odds_api_event_map.csv"
)

CACHE_DIR = (
    ROOT / "data/raw/odds_api"
    / "eliteserien_event_map"
)

SPORT = "soccer_norway_eliteserien"
API_KEY = os.environ["ODDS_API_KEY"]


# ============================================================
# TEAM NORMALIZATION
# ============================================================

ALIASES = {
    "bodo glimt": "bodo glimt",
    "bodo glimt fk": "bodo glimt",

    "viking": "viking",
    "viking fk": "viking",

    "odd": "odd",
    "odds bk": "odd",
    "odd grenland": "odd",

    "valerenga": "valerenga",
    "valerenga if": "valerenga",

    "stromsgodset": "stromsgodset",
    "stromsgodset if": "stromsgodset",

    "sarpsborg 08": "sarpsborg 08",
    "sarpsborg 08 ff": "sarpsborg 08",

    "kristiansund": "kristiansund",
    "kristiansund bk": "kristiansund",

    "tromso": "tromso",
    "tromso il": "tromso",

    "hamkam": "hamkam",
    "ham kam": "hamkam",
    "hamarkameratene": "hamkam",

    "kfums": "kfum oslo",
    "kfum": "kfum oslo",
    "kfum oslo": "kfum oslo",
}


def normalize_team(value):
    s = str(value)

    s = unicodedata.normalize(
        "NFKD",
        s,
    )

    s = "".join(
        c for c in s
        if not unicodedata.combining(c)
    )

    s = s.lower()

    # Explicit characters that do not cleanly decompose.
    s = s.replace("ø", "o")
    s = s.replace("æ", "ae")
    s = s.replace("å", "a")

    s = re.sub(
        r"[^a-z0-9]+",
        " ",
        s,
    )

    s = " ".join(
        s.split()
    )

    return ALIASES.get(
        s,
        s,
    )


def similarity(a, b):
    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


# ============================================================
# HISTORICAL EVENTS
# ============================================================

def historical_events(query_time):

    url = (
        "https://api.the-odds-api.com/v4/"
        f"historical/sports/{SPORT}/events"
    )

    r = requests.get(
        url,
        params={
            "apiKey": API_KEY,
            "date": query_time.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        },
        timeout=60,
    )

    print(
        "HTTP:",
        r.status_code,
        "| remaining:",
        r.headers.get("x-requests-remaining"),
        "| used:",
        r.headers.get("x-requests-used"),
        "| cost:",
        r.headers.get("x-requests-last"),
    )

    r.raise_for_status()
    return r.json()


def find_event(
    payload,
    home,
    away,
    approx_kickoff,
):

    hk = normalize_team(home)
    ak = normalize_team(away)

    candidates = []

    for game in payload.get("data", []):

        gh = normalize_team(
            game.get("home_team", "")
        )

        ga = normalize_team(
            game.get("away_team", "")
        )

        home_sim = similarity(hk, gh)
        away_sim = similarity(ak, ga)

        if (
            home_sim < 0.72
            or away_sim < 0.72
        ):
            continue

        kickoff = pd.to_datetime(
            game.get("commence_time"),
            errors="coerce",
            utc=True,
        )

        if pd.isna(kickoff):
            continue

        delta_hours = abs(
            (
                kickoff
                - approx_kickoff
            ).total_seconds()
        ) / 3600

        # V5 dates are date-level rather than exact kickoff.
        # Give ourselves enough room to locate the correct fixture.
        if delta_hours > 36:
            continue

        name_score = (
            home_sim + away_sim
        ) / 2

        candidates.append(
            (
                delta_hours,
                -name_score,
                game,
                kickoff,
                name_score,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    (
        delta_hours,
        _,
        game,
        kickoff,
        name_score,
    ) = candidates[0]

    return {
        "event": game,
        "api_kickoff": kickoff,
        "kickoff_diff_hours": delta_hours,
        "name_score": name_score,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 115)
    print(
        "ELITESERIEN — ODDS API HISTORICAL EVENT MAP"
    )
    print("=" * 115)

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    df = df[
        df["league"]
        .astype(str)
        .str.contains(
            "Eliteserien",
            case=False,
            na=False,
        )
    ].copy()

    df["season"] = pd.to_numeric(
        df["season"],
        errors="coerce",
    )

    df = df[
        df["season"].between(
            2020,
            2025,
        )
    ].copy()

    # Drop only games that cannot be evaluated by V5.
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

    df = df.dropna(
        subset=[
            "footystats_match_id",
            "date",
            "home_team",
            "away_team",
            "home_lambda",
            "away_lambda",
            "home_goals",
            "away_goals",
        ]
    ).copy()

    df["match_date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=["match_date"]
    ).copy()

    df = df.sort_values(
        "match_date"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # One historical events query per match day.
    # --------------------------------------------------------

    df["query_day"] = (
        df["match_date"]
        .dt.floor("D")
    )

    unique_days = sorted(
        df["query_day"]
        .dropna()
        .unique()
    )

    print()
    print("V5 usable matches:", len(df))
    print("Unique match days:", len(unique_days))
    print(
        "Estimated event-map cost:",
        len(unique_days),
        "credits maximum before cache reuse",
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshots = {}

    for i, day in enumerate(
        unique_days,
        start=1,
    ):

        day = pd.Timestamp(day)

        if day.tzinfo is None:
            day = day.tz_localize("UTC")
        else:
            day = day.tz_convert("UTC")

        # Same general architecture as MLS:
        # query ahead of the match day so events are upcoming.
        query_time = (
            day
            - pd.Timedelta(hours=36)
        )

        cache = (
            CACHE_DIR
            / (
                query_time.strftime(
                    "%Y%m%dT%H%M%SZ"
                )
                + ".json"
            )
        )

        if cache.exists():
            payload = pd.read_json(
                cache,
                typ="series",
            ).to_dict()

            print(
                f"[{i}/{len(unique_days)}]",
                day.date(),
                "CACHE",
            )

        else:
            print(
                f"[{i}/{len(unique_days)}]",
                day.date(),
                "| query:",
                query_time,
            )

            payload = historical_events(
                query_time
            )

            import json

            cache.write_text(
                json.dumps(
                    payload,
                    indent=2,
                )
            )

            time.sleep(0.15)

        snapshots[day] = payload

    # --------------------------------------------------------
    # MATCH EVENTS
    # --------------------------------------------------------

    rows = []

    for _, row in df.iterrows():

        day = row["query_day"]
        payload = snapshots.get(day)

        # Approximate noon on the V5 match date.
        # Exact API kickoff replaces this once matched.
        approx = (
            row["match_date"].floor("D")
            + pd.Timedelta(hours=12)
        )

        result = find_event(
            payload,
            row["home_team"],
            row["away_team"],
            approx,
        )

        out = row.to_dict()

        if result is None:

            out.update({
                "odds_api_event_found": False,
                "odds_api_event_id": None,
                "odds_api_home_team": None,
                "odds_api_away_team": None,
                "odds_api_kickoff": None,
                "kickoff_diff_hours": None,
                "event_name_score": None,
            })

        else:

            event = result["event"]

            out.update({
                "odds_api_event_found": True,
                "odds_api_event_id":
                    event.get("id"),
                "odds_api_home_team":
                    event.get("home_team"),
                "odds_api_away_team":
                    event.get("away_team"),
                "odds_api_kickoff":
                    result["api_kickoff"],
                "kickoff_diff_hours":
                    result["kickoff_diff_hours"],
                "event_name_score":
                    result["name_score"],
            })

        rows.append(out)

    out = pd.DataFrame(rows)

    out.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    found = (
        out["odds_api_event_found"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    print()
    print("=" * 115)
    print("EVENT MAP SUMMARY")
    print("=" * 115)

    print("Games:", len(out))
    print("Mapped:", int(found.sum()))
    print("Missing:", int((~found).sum()))
    print("Coverage:", f"{found.mean():.2%}")

    print()
    print("Coverage by season:")

    for season, s in out.groupby("season"):

        sf = (
            s["odds_api_event_found"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

        print(
            season,
            "| games:",
            len(s),
            "| mapped:",
            int(sf.sum()),
            "| coverage:",
            f"{sf.mean():.2%}",
        )

    missing = out[~found]

    if len(missing):

        print()
        print("Sample missing:")
        print(
            missing[
                [
                    "season",
                    "date",
                    "home_team",
                    "away_team",
                ]
            ]
            .head(40)
            .to_string(index=False)
        )

    print()
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
