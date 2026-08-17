from pathlib import Path
from datetime import datetime, timezone
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "data" / "processed"
    / "mls_v5_with_kickoff.csv"
)

OUT_PATH = (
    ROOT / "data" / "processed"
    / "mls_odds_api_true_t24_pilot.csv"
)

SPORT = "soccer_usa_mls"
API_KEY = os.environ["ODDS_API_KEY"]

N_GAMES = 20
HOURS_BEFORE = 24


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

    "san diego": "san diego",
    "san diego fc": "san diego",

    "atlanta utd": "atlanta united",
    "atlanta united": "atlanta united",
    "atlanta united fc": "atlanta united",
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

    s = s.replace(
        "newyork",
        "new york",
    )

    s = s.replace(
        "st.",
        "st ",
    )

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


def build_kickoff(row):

    date_text = str(
        row["fd_date"]
    ).strip()

    time_text = str(
        row["fd_time"]
    ).strip()

    # fd_date is dd/mm/YYYY
    dt = datetime.strptime(
        f"{date_text} {time_text}",
        "%d/%m/%Y %H:%M",
    )

    # Football-Data MLS timestamps line up
    # with the UTC kickoff timestamps we've
    # already observed from Odds API.
    return pd.Timestamp(
        dt,
        tz="UTC",
    )


def get_snapshot(query_time):

    url = (
        "https://api.the-odds-api.com/v4/"
        f"historical/sports/{SPORT}/odds"
    )

    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "date": query_time.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }

    r = requests.get(
        url,
        params=params,
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
    kickoff,
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

        if gh != hk or ga != ak:
            continue

        api_kickoff = pd.to_datetime(
            game.get("commence_time"),
            errors="coerce",
            utc=True,
        )

        if pd.isna(api_kickoff):
            continue

        delta = abs(
            (
                api_kickoff
                - kickoff
            ).total_seconds()
        )

        candidates.append(
            (
                delta,
                game,
                api_kickoff,
            )
        )

    if not candidates:
        return None, None

    candidates.sort(
        key=lambda x: x[0]
    )

    delta, game, api_kickoff = (
        candidates[0]
    )

    # Safety check: reject anything
    # more than 12 hours away.
    if delta > 12 * 3600:
        return None, None

    return game, api_kickoff


def extract_25_prices(game):

    rows = []

    for book in game.get(
        "bookmakers",
        [],
    ):

        for market in book.get(
            "markets",
            [],
        ):

            if (
                market.get("key")
                != "totals"
            ):
                continue

            over = None
            under = None

            for outcome in market.get(
                "outcomes",
                [],
            ):

                try:
                    point = float(
                        outcome.get("point")
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if abs(
                    point - 2.5
                ) > 1e-9:
                    continue

                name = str(
                    outcome.get("name")
                ).lower()

                if name == "over":
                    over = outcome

                elif name == "under":
                    under = outcome

            if (
                over is None
                or under is None
            ):
                continue

            o = float(
                over["price"]
            )

            u = float(
                under["price"]
            )

            q_o = 1.0 / o
            q_u = 1.0 / u

            fair_under = (
                q_u
                / (q_o + q_u)
            )

            rows.append(
                {
                    "bookmaker":
                        book.get("title"),

                    "bookmaker_key":
                        book.get("key"),

                    "market_update":
                        market.get(
                            "last_update"
                        ),

                    "over_odds":
                        o,

                    "under_odds":
                        u,

                    "market_under_prob":
                        fair_under,
                }
            )

    return rows


def main():

    print()
    print("=" * 115)
    print(
        "MLS V5 — TRUE T-24H ODDS API PILOT"
    )
    print("=" * 115)

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    required = [
        "home_team",
        "away_team",
        "home_lambda",
        "away_lambda",
        "fd_date",
        "fd_time",
        "kickoff_matched",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    df = df[
        df["kickoff_matched"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
            ]
        )
    ].copy()

    df["kickoff"] = df.apply(
        build_kickoff,
        axis=1,
    )

    df = df.sort_values(
        "kickoff"
    ).reset_index(
        drop=True
    )

    # Use 2021-2025 because that's where
    # our previous Odds API pilot was focused.
    df = df[
        df["kickoff"]
        .dt.year
        .between(
            2021,
            2025,
        )
    ].copy()

    if len(df) > N_GAMES:

        indices = np.linspace(
            0,
            len(df) - 1,
            N_GAMES,
            dtype=int,
        )

        sample = (
            df.iloc[indices]
            .copy()
        )

    else:
        sample = df.copy()

    print()
    print(
        "Available matches:",
        len(df),
    )

    print(
        "Pilot games:",
        len(sample),
    )

    results = []

    for number, (_, row) in enumerate(
        sample.iterrows(),
        start=1,
    ):

        kickoff = row["kickoff"]

        query_time = (
            kickoff
            - pd.Timedelta(
                hours=HOURS_BEFORE
            )
        )

        model_under = (
            poisson_under_25(
                row["home_lambda"],
                row["away_lambda"],
            )
        )

        print()
        print("-" * 115)

        print(
            f"[{number}/{len(sample)}]",
            row["home_team"],
            "vs",
            row["away_team"],
        )

        print(
            "Football-Data kickoff:",
            kickoff,
        )

        print(
            "TRUE T-24 query:",
            query_time,
        )

        try:

            payload = get_snapshot(
                query_time.to_pydatetime()
            )

        except Exception as exc:

            print(
                "REQUEST ERROR:",
                exc,
            )

            continue

        game, api_kickoff = (
            find_event(
                payload,
                row["home_team"],
                row["away_team"],
                kickoff,
            )
        )

        base = {
            "kickoff":
                kickoff,

            "query_time":
                query_time,

            "home_team":
                row["home_team"],

            "away_team":
                row["away_team"],

            "home_goals":
                row.get("home_goals"),

            "away_goals":
                row.get("away_goals"),

            "home_lambda":
                row["home_lambda"],

            "away_lambda":
                row["away_lambda"],

            "model_under_prob":
                model_under,
        }

        if game is None:

            print(
                "EVENT NOT FOUND"
            )

            results.append(
                {
                    **base,
                    "event_found":
                        False,
                    "has_25":
                        False,
                }
            )

            time.sleep(0.1)
            continue

        print(
            "API event:",
            game.get("home_team"),
            "vs",
            game.get("away_team"),
        )

        print(
            "API kickoff:",
            api_kickoff,
        )

        kickoff_diff = abs(
            (
                api_kickoff
                - kickoff
            ).total_seconds()
            / 3600
        )

        print(
            "Kickoff difference:",
            f"{kickoff_diff:.2f}h",
        )

        prices = (
            extract_25_prices(
                game
            )
        )

        if not prices:

            print(
                "EVENT FOUND — NO EXACT 2.5 PAIR"
            )

            results.append(
                {
                    **base,

                    "event_found":
                        True,

                    "has_25":
                        False,

                    "api_event_id":
                        game.get("id"),

                    "api_kickoff":
                        api_kickoff,

                    "kickoff_diff_hours":
                        kickoff_diff,

                    "snapshot_time":
                        payload.get(
                            "timestamp"
                        ),
                }
            )

        else:

            print(
                "Exact 2.5 bookmakers:",
                len(prices),
            )

            for price in prices:

                edge = (
                    model_under
                    - price[
                        "market_under_prob"
                    ]
                )

                print(
                    f"  "
                    f"{price['bookmaker']:<22} "
                    f"O={price['over_odds']:.3f} "
                    f"U={price['under_odds']:.3f} "
                    f"MktU="
                    f"{price['market_under_prob']:.2%} "
                    f"V5U="
                    f"{model_under:.2%} "
                    f"Edge="
                    f"{edge:+.2%}"
                )

                results.append(
                    {
                        **base,

                        "event_found":
                            True,

                        "has_25":
                            True,

                        "api_event_id":
                            game.get("id"),

                        "api_kickoff":
                            api_kickoff,

                        "kickoff_diff_hours":
                            kickoff_diff,

                        "snapshot_time":
                            payload.get(
                                "timestamp"
                            ),

                        **price,

                        "under_edge":
                            edge,
                    }
                )

        time.sleep(0.1)

    out = pd.DataFrame(
        results
    )

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUT_PATH,
        index=False,
    )

    print()
    print("=" * 115)
    print("TRUE T-24 PILOT SUMMARY")
    print("=" * 115)

    games_total = len(sample)

    if out.empty:
        print("No results.")
        return

    game_cols = [
        "kickoff",
        "home_team",
        "away_team",
    ]

    found = (
        out[
            out["event_found"]
            .eq(True)
        ][game_cols]
        .drop_duplicates()
    )

    with_25 = (
        out[
            out["has_25"]
            .eq(True)
        ][game_cols]
        .drop_duplicates()
    )

    print(
        "Pilot games:",
        games_total,
    )

    print(
        "Events found:",
        len(found),
        f"({len(found)/games_total:.1%})",
    )

    print(
        "Games with exact O/U 2.5:",
        len(with_25),
        f"({len(with_25)/games_total:.1%})",
    )

    price_rows = out[
        out["has_25"]
        .eq(True)
    ].copy()

    if len(price_rows):

        print(
            "Bookmaker price rows:",
            len(price_rows),
        )

        print(
            "Average books / 2.5 game:",
            len(price_rows)
            / len(with_25),
        )

        print(
            "Average U2.5 odds:",
            price_rows[
                "under_odds"
            ].mean(),
        )

        print(
            "Average V5 U2.5 probability:",
            price_rows[
                "model_under_prob"
            ].mean(),
        )

        print(
            "Average market U2.5 probability:",
            price_rows[
                "market_under_prob"
            ].mean(),
        )

        print(
            "Average V5 edge:",
            price_rows[
                "under_edge"
            ].mean(),
        )

        signals = price_rows[
            price_rows[
                "under_edge"
            ] >= 0.11
        ].copy()

        print(
            "Raw >=11% price rows:",
            len(signals),
        )

        signal_games = (
            signals[
                game_cols
            ]
            .drop_duplicates()
        )

        print(
            "Games producing >=11%:",
            len(signal_games),
        )

        if len(signals):

            print()
            print(
                ">=11% SIGNALS"
            )

            cols = [
                "kickoff",
                "home_team",
                "away_team",
                "bookmaker",
                "under_odds",
                "model_under_prob",
                "market_under_prob",
                "under_edge",
            ]

            print(
                signals[
                    cols
                ]
                .sort_values(
                    "under_edge",
                    ascending=False,
                )
                .to_string(
                    index=False
                )
            )

    print()
    print("Saved:")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
