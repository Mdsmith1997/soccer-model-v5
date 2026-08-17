from pathlib import Path
from datetime import timedelta
import os
import time

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

PRED_PATH = (
    ROOT
    / "data"
    / "processed"
    / "footystats_mls_v5_predictions.csv"
)

OUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "mls_odds_api_pilot.csv"
)

SPORT = "soccer_usa_mls"
API_KEY = os.environ["ODDS_API_KEY"]

N_GAMES = 20
HOURS_BEFORE = 24


def poisson_under_25(home_lambda, away_lambda):
    """
    P(total goals <= 2) where independent Poisson totals
    collapse to Poisson(home_lambda + away_lambda).
    """
    mu = float(home_lambda) + float(away_lambda)

    return (
        np.exp(-mu)
        * (
            1
            + mu
            + (mu ** 2) / 2
        )
    )


def get_snapshot(target_time):
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
        "date": target_time.strftime(
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
        r.headers.get("x-requests-remaining"),
        "| used:",
        r.headers.get("x-requests-used"),
    )

    r.raise_for_status()

    return r.json()


def normalize_team(x):
    x = str(x).lower()

    replacements = {
        "fc": "",
        "sc": "",
        "cf": "",
        "club": "",
        ".": "",
        "-": " ",
    }

    for a, b in replacements.items():
        x = x.replace(a, b)

    return " ".join(x.split())


def find_event(payload, home, away):
    hk = normalize_team(home)
    ak = normalize_team(away)

    # First try exact normalized pairing.
    for game in payload.get("data", []):
        gh = normalize_team(
            game.get("home_team", "")
        )
        ga = normalize_team(
            game.get("away_team", "")
        )

        if gh == hk and ga == ak:
            return game

    # Then conservative substring fallback.
    for game in payload.get("data", []):
        gh = normalize_team(
            game.get("home_team", "")
        )
        ga = normalize_team(
            game.get("away_team", "")
        )

        home_match = hk in gh or gh in hk
        away_match = ak in ga or ga in ak

        if home_match and away_match:
            return game

    return None


def extract_25_prices(game):
    rows = []

    for book in game.get("bookmakers", []):

        for market in book.get("markets", []):

            if market.get("key") != "totals":
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
                except (TypeError, ValueError):
                    continue

                if abs(point - 2.5) > 1e-9:
                    continue

                if outcome.get("name") == "Over":
                    over = outcome

                elif outcome.get("name") == "Under":
                    under = outcome

            if over is None or under is None:
                continue

            over_odds = float(over["price"])
            under_odds = float(under["price"])

            # De-vig two-way 2.5 market.
            q_over = 1.0 / over_odds
            q_under = 1.0 / under_odds

            market_under = (
                q_under
                / (q_under + q_over)
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
                        over_odds,
                    "under_odds":
                        under_odds,
                    "market_under_prob":
                        market_under,
                }
            )

    return rows


def main():
    print()
    print("=" * 110)
    print("MLS ODDS API — 20 GAME V5 PILOT")
    print("=" * 110)

    pred = pd.read_csv(
        PRED_PATH,
        low_memory=False,
    )

    pred["date"] = pd.to_datetime(
        pred["date"],
        errors="coerce",
        utc=True,
    )

    pred = pred.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_lambda",
            "away_lambda",
        ]
    ).copy()

    # Historical API coverage test period.
    pred = pred[
        pred["date"].dt.year.between(
            2021,
            2025,
        )
    ].copy()

    # Use a spread of matches rather than
    # simply taking the first 20.
    pred = pred.sort_values("date")

    if len(pred) > N_GAMES:
        idx = np.linspace(
            0,
            len(pred) - 1,
            N_GAMES,
            dtype=int,
        )
        sample = pred.iloc[idx].copy()
    else:
        sample = pred.copy()

    results = []

    print()
    print("Prediction rows:", len(pred))
    print("Pilot games:", len(sample))
    print(
        "Snapshot target:",
        f"T-{HOURS_BEFORE}h",
    )

    for number, (_, row) in enumerate(
        sample.iterrows(),
        start=1,
    ):
        kickoff = row["date"]

        target = (
            kickoff
            - pd.Timedelta(
                hours=HOURS_BEFORE
            )
        )

        print()
        print("-" * 110)

        print(
            f"[{number}/{len(sample)}]",
            row["home_team"],
            "vs",
            row["away_team"],
        )

        print("Model date:", kickoff)
        print("Query time:", target)

        try:
            payload = get_snapshot(
                target.to_pydatetime()
            )
        except Exception as exc:
            print("REQUEST ERROR:", exc)
            continue

        game = find_event(
            payload,
            row["home_team"],
            row["away_team"],
        )

        if game is None:
            print("EVENT NOT FOUND")

            results.append(
                {
                    "date": kickoff,
                    "home_team":
                        row["home_team"],
                    "away_team":
                        row["away_team"],
                    "event_found": False,
                    "has_25": False,
                }
            )

            time.sleep(0.15)
            continue

        print(
            "API event:",
            game.get("home_team"),
            "vs",
            game.get("away_team"),
        )

        print(
            "API kickoff:",
            game.get("commence_time"),
        )

        prices = extract_25_prices(game)

        model_under = poisson_under_25(
            row["home_lambda"],
            row["away_lambda"],
        )

        if not prices:
            print(
                "EVENT FOUND — NO 2.5 PAIR"
            )

            results.append(
                {
                    "date": kickoff,
                    "home_team":
                        row["home_team"],
                    "away_team":
                        row["away_team"],
                    "event_found": True,
                    "has_25": False,
                    "model_under_prob":
                        model_under,
                }
            )

        else:
            print(
                "2.5 bookmakers:",
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
                    f"  {price['bookmaker']:<20} "
                    f"O={price['over_odds']:.3f} "
                    f"U={price['under_odds']:.3f} "
                    f"MktU="
                    f"{price['market_under_prob']:.2%} "
                    f"V5U={model_under:.2%} "
                    f"Edge={edge:+.2%}"
                )

                results.append(
                    {
                        "date": kickoff,
                        "home_team":
                            row["home_team"],
                        "away_team":
                            row["away_team"],
                        "event_found": True,
                        "has_25": True,
                        "api_event_id":
                            game.get("id"),
                        "api_commence_time":
                            game.get(
                                "commence_time"
                            ),
                        "snapshot_time":
                            payload.get(
                                "timestamp"
                            ),
                        "model_under_prob":
                            model_under,
                        **price,
                        "under_edge": edge,
                    }
                )

        time.sleep(0.15)

    out = pd.DataFrame(results)

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUT_PATH,
        index=False,
    )

    print()
    print("=" * 110)
    print("PILOT SUMMARY")
    print("=" * 110)

    if out.empty:
        print("No results.")
        return

    games_total = len(sample)

    found_games = (
        out.loc[
            out["event_found"].eq(True),
            [
                "date",
                "home_team",
                "away_team",
            ],
        ]
        .drop_duplicates()
        .shape[0]
    )

    games_25 = (
        out.loc[
            out["has_25"].eq(True),
            [
                "date",
                "home_team",
                "away_team",
            ],
        ]
        .drop_duplicates()
        .shape[0]
    )

    print("Pilot games:", games_total)

    print(
        "Events found:",
        found_games,
        f"({found_games/games_total:.1%})",
    )

    print(
        "Games with O/U 2.5:",
        games_25,
        f"({games_25/games_total:.1%})",
    )

    prices = out[
        out["has_25"].eq(True)
    ].copy()

    if not prices.empty:

        print(
            "Bookmaker price rows:",
            len(prices),
        )

        print(
            "Average books/game:",
            len(prices) / games_25
            if games_25
            else 0,
        )

        print(
            "Average U2.5 odds:",
            prices["under_odds"].mean(),
        )

        print(
            "Average V5 edge:",
            prices["under_edge"].mean(),
        )

        signals = prices[
            prices["under_edge"] >= 0.11
        ]

        print(
            "Raw >=11% price rows:",
            len(signals),
        )

    print()
    print("Saved:")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
