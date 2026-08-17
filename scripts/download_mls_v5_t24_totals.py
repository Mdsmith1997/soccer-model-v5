from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import requests


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "mls_odds_api_event_map.csv"
)

RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "odds_api"
    / "mls_t24_totals"
)

OUTPUT_BOOKS = (
    ROOT
    / "data"
    / "processed"
    / "mls_v5_t24_totals_bookmakers.csv"
)

OUTPUT_GAMES = (
    ROOT
    / "data"
    / "processed"
    / "mls_v5_t24_totals_games.csv"
)

SPORT = "soccer_usa_mls"

API_KEY = os.environ["ODDS_API_KEY"]

HOURS_BEFORE = 24

MARKET = "totals"
REGION = "us"

# Historical featured-market request cost observed in our tests.
EXPECTED_COST_PER_REQUEST = 10

# Safety reserve. Script stops before using the final reserve.
MIN_CREDITS_RESERVE = 1000


# ============================================================
# MODEL TOTALS
# ============================================================

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


# ============================================================
# CACHE
# ============================================================

def cache_path(timestamp):

    stamp = timestamp.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    return (
        RAW_DIR
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

def fetch_snapshot(timestamp):

    path = cache_path(
        timestamp
    )

    if path.exists():

        return (
            load_json(path),
            True,
            None,
        )

    url = (
        "https://api.the-odds-api.com/v4/"
        f"historical/sports/{SPORT}/odds"
    )

    params = {
        "apiKey":
            API_KEY,

        "regions":
            REGION,

        "markets":
            MARKET,

        "oddsFormat":
            "decimal",

        "dateFormat":
            "iso",

        "date":
            timestamp.strftime(
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

    save_json(
        path,
        payload,
    )

    quota = {
        "remaining":
            remaining,

        "used":
            used,

        "cost":
            cost,
    }

    return (
        payload,
        False,
        quota,
    )


# ============================================================
# EXACT EVENT LOOKUP
# ============================================================

def event_lookup(payload):

    lookup = {}

    for game in payload.get(
        "data",
        [],
    ):

        event_id = str(
            game.get("id", "")
        ).strip()

        if event_id:

            lookup[
                event_id
            ] = game

    return lookup


# ============================================================
# EXACT 2.5 EXTRACTION
# ============================================================

def extract_exact_25(
    game,
):

    rows = []

    for book in game.get(
        "bookmakers",
        [],
    ):

        bookmaker = (
            book.get("title")
        )

        bookmaker_key = (
            book.get("key")
        )

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
                        outcome.get(
                            "point"
                        )
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
                    outcome.get(
                        "name",
                        ""
                    )
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

            try:

                over_odds = float(
                    over["price"]
                )

                under_odds = float(
                    under["price"]
                )

            except (
                TypeError,
                ValueError,
                KeyError,
            ):
                continue

            if (
                over_odds <= 1
                or under_odds <= 1
            ):
                continue

            q_over = (
                1.0 / over_odds
            )

            q_under = (
                1.0 / under_odds
            )

            market_under_prob = (
                q_under
                / (
                    q_under
                    + q_over
                )
            )

            hold = (
                q_under
                + q_over
                - 1.0
            )

            rows.append(
                {
                    "bookmaker":
                        bookmaker,

                    "bookmaker_key":
                        bookmaker_key,

                    "market_last_update":
                        market.get(
                            "last_update"
                        ),

                    "over_odds":
                        over_odds,

                    "under_odds":
                        under_odds,

                    "market_under_prob":
                        market_under_prob,

                    "market_hold":
                        hold,
                }
            )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 120)
    print(
        "MLS V5 — FULL HISTORICAL T-24H TOTALS DOWNLOAD"
    )
    print("=" * 120)

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    found = (
        df[
            "odds_api_event_found"
        ]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
            ]
        )
    )

    df = df[
        found
    ].copy()

    df["odds_api_kickoff"] = (
        pd.to_datetime(
            df[
                "odds_api_kickoff"
            ],
            errors="coerce",
            utc=True,
        )
    )

    df = df.dropna(
        subset=[
            "odds_api_event_id",
            "odds_api_kickoff",
            "home_lambda",
            "away_lambda",
            "home_goals",
            "away_goals",
        ]
    ).copy()

    # --------------------------------------------------------
    # Freeze historical evaluation window.
    #
    # 2020 is retained in raw extraction, but we'll mainly
    # judge OOS stability on later seasons.
    # --------------------------------------------------------

    df["season_num"] = (
        pd.to_numeric(
            df["season"],
            errors="coerce",
        )
    )

    df = df[
        df[
            "season_num"
        ].between(
            2020,
            2025,
        )
    ].copy()

    # --------------------------------------------------------
    # AUTHORITATIVE T-24
    # --------------------------------------------------------

    df["target_timestamp"] = (
        df[
            "odds_api_kickoff"
        ]
        - pd.Timedelta(
            hours=HOURS_BEFORE
        )
    )

    # The historical endpoint selects the nearest archived
    # snapshot at or before this target.
    unique_targets = (
        df[
            "target_timestamp"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cached = sum(
        cache_path(
            ts
        ).exists()
        for ts in unique_targets
    )

    new_requests = (
        len(unique_targets)
        - cached
    )

    expected_cost = (
        new_requests
        * EXPECTED_COST_PER_REQUEST
    )

    print()
    print(
        "Mapped games:",
        len(df),
    )

    print(
        "Unique authoritative T-24 timestamps:",
        len(unique_targets),
    )

    print(
        "Already cached:",
        cached,
    )

    print(
        "New API requests required:",
        new_requests,
    )

    print(
        "Estimated maximum new credit cost:",
        expected_cost,
    )

    print()

    # --------------------------------------------------------
    # IMPORTANT CREDIT GUARD
    # --------------------------------------------------------

    if expected_cost > 17000:

        raise RuntimeError(
            "Estimated cost is too high. "
            "Aborting before API download."
        )

    # --------------------------------------------------------
    # Download / cache snapshots
    # --------------------------------------------------------

    snapshots = {}

    for i, timestamp in enumerate(
        unique_targets,
        start=1,
    ):

        print()
        print(
            f"[{i}/{len(unique_targets)}]",
            timestamp,
        )

        payload, from_cache, quota = (
            fetch_snapshot(
                timestamp
            )
        )

        if from_cache:

            print(
                "CACHE HIT"
            )

        else:

            if quota:

                remaining = (
                    quota.get(
                        "remaining"
                    )
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
                0.08
            )

        snapshots[
            timestamp
        ] = payload

    # --------------------------------------------------------
    # Extract every mapped game
    # --------------------------------------------------------

    bookmaker_rows = []

    game_rows = []

    print()
    print("=" * 120)
    print(
        "EXTRACTING EXACT O/U 2.5"
    )
    print("=" * 120)

    for _, row in df.iterrows():

        target = row[
            "target_timestamp"
        ]

        payload = snapshots.get(
            target
        )

        if payload is None:
            continue

        lookup = event_lookup(
            payload
        )

        event_id = str(
            row[
                "odds_api_event_id"
            ]
        )

        game = lookup.get(
            event_id
        )

        model_under = (
            poisson_under_25(
                row[
                    "home_lambda"
                ],
                row[
                    "away_lambda"
                ],
            )
        )

        actual_total = (
            float(
                row[
                    "home_goals"
                ]
            )
            + float(
                row[
                    "away_goals"
                ]
            )
        )

        under_win = (
            actual_total < 2.5
        )

        base = {
            "season":
                row["season"],

            "date":
                row["date"],

            "home_team":
                row["home_team"],

            "away_team":
                row["away_team"],

            "home_goals":
                row["home_goals"],

            "away_goals":
                row["away_goals"],

            "actual_total":
                actual_total,

            "under_25_win":
                under_win,

            "home_lambda":
                row["home_lambda"],

            "away_lambda":
                row["away_lambda"],

            "model_under_prob":
                model_under,

            "odds_api_event_id":
                event_id,

            "odds_api_kickoff":
                row[
                    "odds_api_kickoff"
                ],

            "target_timestamp":
                target,

            "snapshot_timestamp":
                payload.get(
                    "timestamp"
                ),

            "event_in_snapshot":
                game is not None,
        }

        if game is None:

            game_rows.append(
                {
                    **base,

                    "has_exact_25":
                        False,

                    "book_count":
                        0,
                }
            )

            continue

        prices = (
            extract_exact_25(
                game
            )
        )

        if not prices:

            game_rows.append(
                {
                    **base,

                    "has_exact_25":
                        False,

                    "book_count":
                        0,
                }
            )

            continue

        for price in prices:

            edge = (
                model_under
                - price[
                    "market_under_prob"
                ]
            )

            ev = (
                model_under
                * price[
                    "under_odds"
                ]
                - 1.0
            )

            profit = (
                price[
                    "under_odds"
                ]
                - 1.0
                if under_win
                else -1.0
            )

            bookmaker_rows.append(
                {
                    **base,
                    **price,

                    "under_edge":
                        edge,

                    "under_ev":
                        ev,

                    "flat_profit":
                        profit,

                    "signal_11":
                        edge >= 0.11,
                }
            )

        p = pd.DataFrame(
            prices
        )

        # ----------------------------------------------------
        # ONE GAME = ONE BETTING OPPORTUNITY
        #
        # Best available U2.5 execution price is recorded.
        # Consensus probability is median de-vig probability.
        # ----------------------------------------------------

        best_idx = (
            p["under_odds"]
            .idxmax()
        )

        best = p.loc[
            best_idx
        ]

        consensus_under = (
            p[
                "market_under_prob"
            ]
            .median()
        )

        consensus_edge = (
            model_under
            - consensus_under
        )

        best_execution_ev = (
            model_under
            * float(
                best[
                    "under_odds"
                ]
            )
            - 1
        )

        best_profit = (
            float(
                best[
                    "under_odds"
                ]
            )
            - 1
            if under_win
            else -1.0
        )

        game_rows.append(
            {
                **base,

                "has_exact_25":
                    True,

                "book_count":
                    len(p),

                "consensus_market_under_prob":
                    consensus_under,

                "consensus_under_edge":
                    consensus_edge,

                "best_bookmaker":
                    best[
                        "bookmaker"
                    ],

                "best_bookmaker_key":
                    best[
                        "bookmaker_key"
                    ],

                "best_under_odds":
                    float(
                        best[
                            "under_odds"
                        ]
                    ),

                "best_over_odds_same_book":
                    float(
                        best[
                            "over_odds"
                        ]
                    ),

                "best_book_market_under_prob":
                    float(
                        best[
                            "market_under_prob"
                        ]
                    ),

                "best_book_under_edge":
                    (
                        model_under
                        - float(
                            best[
                                "market_under_prob"
                            ]
                        )
                    ),

                "best_execution_ev":
                    best_execution_ev,

                "flat_profit_best_price":
                    best_profit,

                # Primary frozen signal:
                # RAW V5 vs consensus no-vig market.
                "signal_11_consensus":
                    consensus_edge
                    >= 0.11,
            }
        )

    books = pd.DataFrame(
        bookmaker_rows
    )

    games = pd.DataFrame(
        game_rows
    )

    OUTPUT_BOOKS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    books.to_csv(
        OUTPUT_BOOKS,
        index=False,
    )

    games.to_csv(
        OUTPUT_GAMES,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 120)
    print(
        "MLS T-24 TOTALS DOWNLOAD SUMMARY"
    )
    print("=" * 120)

    print(
        "Mapped games:",
        len(games),
    )

    event_present = (
        games[
            "event_in_snapshot"
        ]
        .eq(True)
    )

    print(
        "Event present at T-24 snapshot:",
        int(
            event_present.sum()
        ),
        f"({event_present.mean():.2%})",
    )

    exact = (
        games[
            "has_exact_25"
        ]
        .eq(True)
    )

    print(
        "Games with exact O/U 2.5:",
        int(
            exact.sum()
        ),
        f"({exact.mean():.2%})",
    )

    if exact.any():

        g25 = games[
            exact
        ].copy()

        print(
            "Bookmaker rows:",
            len(books),
        )

        print(
            "Average books / game:",
            g25[
                "book_count"
            ].mean(),
        )

        print(
            "Median books / game:",
            g25[
                "book_count"
            ].median(),
        )

        print(
            "Average model U2.5:",
            g25[
                "model_under_prob"
            ].mean(),
        )

        print(
            "Average consensus market U2.5:",
            g25[
                "consensus_market_under_prob"
            ].mean(),
        )

        print(
            "Average consensus edge:",
            g25[
                "consensus_under_edge"
            ].mean(),
        )

        print()
        print(
            "EXACT 2.5 COVERAGE BY SEASON"
        )

        for season, g in games.groupby(
            "season"
        ):

            has = (
                g[
                    "has_exact_25"
                ]
                .eq(True)
            )

            print(
                season,
                "| games:",
                len(g),
                "| exact 2.5:",
                int(
                    has.sum()
                ),
                "| rate:",
                f"{has.mean():.2%}",
            )

        # ----------------------------------------------------
        # PRIMARY FROZEN TEST
        # ----------------------------------------------------

        signals = g25[
            g25[
                "signal_11_consensus"
            ]
            .eq(True)
        ].copy()

        print()
        print("=" * 120)
        print(
            "FROZEN V5 UNDER 2.5 >=11% — PREVIEW"
        )
        print("=" * 120)

        print(
            "Signal games:",
            len(signals),
        )

        if len(signals):

            wins = int(
                signals[
                    "under_25_win"
                ]
                .sum()
            )

            profit = (
                signals[
                    "flat_profit_best_price"
                ]
                .sum()
            )

            roi = (
                profit
                / len(signals)
            )

            print(
                "Wins:",
                wins,
            )

            print(
                "Win rate:",
                f"{wins / len(signals):.2%}",
            )

            print(
                "Average best U2.5 odds:",
                signals[
                    "best_under_odds"
                ].mean(),
            )

            print(
                "Average consensus edge:",
                signals[
                    "consensus_under_edge"
                ].mean(),
            )

            print(
                "Profit:",
                f"{profit:+.2f}u",
            )

            print(
                "ROI:",
                f"{roi:+.2%}",
            )

            print()
            print(
                "BY SEASON"
            )

            for season, g in (
                signals.groupby(
                    "season"
                )
            ):

                season_profit = (
                    g[
                        "flat_profit_best_price"
                    ]
                    .sum()
                )

                print(
                    season,
                    "| bets:",
                    len(g),
                    "| wins:",
                    int(
                        g[
                            "under_25_win"
                        ]
                        .sum()
                    ),
                    "| profit:",
                    f"{season_profit:+.2f}u",
                    "| ROI:",
                    f"{season_profit / len(g):+.2%}",
                )

    print()
    print("Saved bookmaker-level data:")
    print(OUTPUT_BOOKS)

    print()
    print("Saved game-level data:")
    print(OUTPUT_GAMES)

    print()
    print(
        "Raw snapshots cached in:"
    )
    print(RAW_DIR)


if __name__ == "__main__":
    main()
