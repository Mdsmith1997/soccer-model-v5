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
    ROOT / "data" / "processed"
    / "mls_v5_with_kickoff.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed"
    / "mls_odds_api_event_map.csv"
)

SPORT = "soccer_usa_mls"
API_KEY = os.environ["ODDS_API_KEY"]


ALIASES = {
    "sj earthquakes": "san jose earthquakes",
    "san jose earthquakes": "san jose earthquakes",

    "la galaxy": "los angeles galaxy",
    "los angeles galaxy": "los angeles galaxy",

    "lafc": "los angeles fc",
    "los angeles fc": "los angeles fc",

    "new york rb": "new york red bulls",
    "newyork rb": "new york red bulls",
    "new york red bulls": "new york red bulls",

    "new york city": "new york city",
    "newyork city": "new york city",
    "new york city fc": "new york city",

    "dc united": "dc united",
    "d c united": "dc united",

    "inter miami": "inter miami",
    "inter miami cf": "inter miami",

    "columbus crew": "columbus crew",
    "columbus crew sc": "columbus crew",

    "seattle sounders": "seattle sounders",
    "seattle sounders fc": "seattle sounders",

    "vancouver whitecaps": "vancouver whitecaps",
    "vancouver whitecaps fc": "vancouver whitecaps",

    "sporting kc": "sporting kansas city",
    "sporting kansas city": "sporting kansas city",

    "st louis city": "st louis city",
    "st louis city sc": "st louis city",

    "austin": "austin",
    "austin fc": "austin",

    "toronto": "toronto",
    "toronto fc": "toronto",

    "cf montreal": "montreal",
    "montreal impact": "montreal",
    "montreal": "montreal",

    "atlanta utd": "atlanta united",
    "atlanta united": "atlanta united",
    "atlanta united fc": "atlanta united",

    "san diego": "san diego",
    "san diego fc": "san diego",
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

    replacements = {
        "newyork": "new york",
        "st.": "st ",
        "stlouis": "st louis",
        "chicagofire": "chicago fire",
        "columbuscrew": "columbus crew",
        "losangeles": "los angeles",
    }

    for a, b in replacements.items():
        s = s.replace(a, b)

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


def build_fd_kickoff(row):

    dt = pd.to_datetime(
        (
            str(row["fd_date"]).strip()
            + " "
            + str(row["fd_time"]).strip()
        ),
        format="%d/%m/%Y %H:%M",
        errors="coerce",
        utc=True,
    )

    return dt


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
        r.headers.get(
            "x-requests-remaining"
        ),
        "| used:",
        r.headers.get(
            "x-requests-used"
        ),
        "| cost:",
        r.headers.get(
            "x-requests-last"
        ),
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

    for game in payload.get(
        "data",
        [],
    ):

        gh = normalize_team(
            game.get(
                "home_team",
                "",
            )
        )

        ga = normalize_team(
            game.get(
                "away_team",
                "",
            )
        )

        home_sim = similarity(
            hk,
            gh,
        )

        away_sim = similarity(
            ak,
            ga,
        )

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

        # Football-Data is generally within ~1h,
        # but allow a generous safety margin.
        if delta_hours > 15:
            continue

        name_score = (
            home_sim
            + away_sim
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
        "event":
            game,

        "api_kickoff":
            kickoff,

        "kickoff_diff_hours":
            delta_hours,

        "name_score":
            name_score,
    }


def main():

    print()
    print("=" * 115)
    print(
        "MLS — ODDS API HISTORICAL EVENT MAP"
    )
    print("=" * 115)

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    df = df[
        pd.to_numeric(
            df["year"],
            errors="coerce",
        ).between(
            2020,
            2025,
        )
    ].copy()

    df["fd_kickoff"] = df.apply(
        build_fd_kickoff,
        axis=1,
    )

    df = df.dropna(
        subset=[
            "fd_kickoff",
            "home_team",
            "away_team",
        ]
    ).copy()

    df = df.sort_values(
        "fd_kickoff"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We do NOT need one events API call per match.
    #
    # A single historical events snapshot can contain
    # many upcoming MLS events.
    #
    # Query once per UTC calendar day, around noon UTC.
    # Then use those responses to map nearby games.
    # --------------------------------------------------------

    df["query_day"] = (
        df["fd_kickoff"]
        .dt.floor("D")
    )

    unique_days = sorted(
        df["query_day"]
        .dropna()
        .unique()
    )

    print()
    print("Matches:", len(df))
    print(
        "Unique match days:",
        len(unique_days),
    )

    print()
    print(
        "Historical-events endpoint cost:"
    )
    print(
        "~1 credit per successful daily snapshot"
    )

    snapshots = {}

    for i, day in enumerate(
        unique_days,
        start=1,
    ):

        day = pd.Timestamp(
            day
        )

        if day.tzinfo is None:
            day = day.tz_localize(
                "UTC"
            )
        else:
            day = day.tz_convert(
                "UTC"
            )

        # Query 36 hours before the UTC match day.
        # This gives bookmakers time to have listed
        # games while still capturing upcoming events.
        query_time = (
            day
            - pd.Timedelta(
                hours=36
            )
        )

        print()
        print(
            f"[{i}/{len(unique_days)}]",
            "match day:",
            day.date(),
            "| query:",
            query_time,
        )

        try:

            payload = historical_events(
                query_time.to_pydatetime()
            )

        except Exception as exc:

            print(
                "REQUEST ERROR:",
                exc,
            )

            snapshots[
                day.date()
            ] = None

            continue

        snapshots[
            day.date()
        ] = payload

        time.sleep(0.05)

    # --------------------------------------------------------
    # MATCH
    # --------------------------------------------------------

    results = []

    for _, row in df.iterrows():

        fd_kickoff = row[
            "fd_kickoff"
        ]

        # Search the snapshot associated with
        # this match day first, then adjacent days.
        days_to_try = [
            fd_kickoff.floor("D"),
            fd_kickoff.floor("D")
            - pd.Timedelta(days=1),
            fd_kickoff.floor("D")
            + pd.Timedelta(days=1),
        ]

        match = None

        for day in days_to_try:

            payload = snapshots.get(
                day.date()
            )

            if not payload:
                continue

            candidate = find_event(
                payload,
                row["home_team"],
                row["away_team"],
                fd_kickoff,
            )

            if candidate is not None:

                match = candidate
                break

        result = row.to_dict()

        if match is None:

            result.update(
                {
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

                    "kickoff_diff_hours":
                        None,
                }
            )

        else:

            game = match["event"]

            result.update(
                {
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

                    "kickoff_diff_hours":
                        match[
                            "kickoff_diff_hours"
                        ],
                }
            )

        results.append(result)

    out = pd.DataFrame(
        results
    )

    found = (
        out[
            "odds_api_event_found"
        ]
        .eq(True)
    )

    print()
    print("=" * 115)
    print("EVENT MAP SUMMARY")
    print("=" * 115)

    print(
        "Rows:",
        len(out),
    )

    print(
        "Events found:",
        int(found.sum()),
    )

    print(
        "Match rate:",
        f"{found.mean():.2%}",
    )

    print()
    print("BY YEAR")

    for year, g in out.groupby(
        "year"
    ):

        ok = (
            g[
                "odds_api_event_found"
            ]
            .eq(True)
        )

        print(
            year,
            "| games:",
            len(g),
            "| found:",
            int(ok.sum()),
            "| rate:",
            f"{ok.mean():.2%}",
        )

    if found.any():

        print()
        print(
            "KICKOFF DIFFERENCE"
        )

        print(
            out.loc[
                found,
                "kickoff_diff_hours",
            ]
            .describe()
            .to_string()
        )

        print()
        print(
            "DIFFERENCE FREQUENCIES"
        )

        print(
            out.loc[
                found,
                "kickoff_diff_hours",
            ]
            .round(2)
            .value_counts()
            .sort_index()
            .head(30)
            .to_string()
        )

    print()
    print("=" * 115)
    print("UNMATCHED SAMPLE")
    print("=" * 115)

    print(
        out.loc[
            ~found,
            [
                "fd_kickoff",
                "home_team",
                "away_team",
            ],
        ]
        .head(60)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 115)
    print("LARGEST KICKOFF DIFFERENCES")
    print("=" * 115)

    print(
        out.loc[
            found,
            [
                "fd_kickoff",
                "odds_api_kickoff",
                "home_team",
                "away_team",
                "odds_api_home",
                "odds_api_away",
                "kickoff_diff_hours",
                "event_match_score",
            ],
        ]
        .sort_values(
            "kickoff_diff_hours",
            ascending=False,
        )
        .head(40)
        .to_string(
            index=False
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("Saved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
