from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LEDGER_FILE = (
    ROOT
    / "data"
    / "live"
    / "v5_live_totals_bet_ledger.csv"
)

HISTORY_FILE = (
    ROOT
    / "data"
    / "live"
    / "odds_markets_history.csv"
)

TARGET_POINT = 2.5


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):

    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(x):
        return np.nan

    return x


def decimal_to_american(decimal_odds):

    d = safe_float(decimal_odds)

    if (
        not np.isfinite(d)
        or
        d <= 1.0
    ):
        return np.nan

    if d >= 2.0:

        return int(
            round(
                (d - 1.0)
                *
                100.0
            )
        )

    return int(
        round(
            -100.0
            /
            (d - 1.0)
        )
    )


def normalize_selection(value):

    value = (
        str(value)
        .strip()
        .upper()
    )

    if value in {
        "UNDER",
        "UNDER 2.5",
    }:
        return "UNDER"

    if value in {
        "OVER",
        "OVER 2.5",
    }:
        return "OVER"

    return None


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():

    if not HISTORY_FILE.exists():

        raise FileNotFoundError(
            f"Missing totals odds history: "
            f"{HISTORY_FILE}"
        )

    hist = pd.read_csv(
        HISTORY_FILE,
        low_memory=False,
    )

    required = {
        "snapshot_time",
        "commence_time",
        "event_id",
        "market",
        "selection",
        "point",
        "decimal_odds",
        "bookmaker",
    }

    missing = (
        required
        -
        set(hist.columns)
    )

    if missing:

        raise RuntimeError(
            "Totals history missing columns: "
            f"{sorted(missing)}"
        )

    hist[
        "snapshot_time"
    ] = pd.to_datetime(
        hist["snapshot_time"],
        utc=True,
        errors="coerce",
    )

    hist[
        "commence_time"
    ] = pd.to_datetime(
        hist["commence_time"],
        utc=True,
        errors="coerce",
    )

    hist[
        "point"
    ] = pd.to_numeric(
        hist["point"],
        errors="coerce",
    )

    hist[
        "decimal_odds"
    ] = pd.to_numeric(
        hist["decimal_odds"],
        errors="coerce",
    )

    hist[
        "market"
    ] = (
        hist["market"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    hist[
        "selection"
    ] = (
        hist["selection"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # We are ONLY evaluating standard O/U 2.5.
    # --------------------------------------------------------

    hist = hist[
        hist["market"].eq("TOTALS")
        &
        hist["point"].round(2).eq(
            TARGET_POINT
        )
        &
        hist["selection"].isin(
            [
                "OVER",
                "UNDER",
            ]
        )
        &
        hist["decimal_odds"].gt(1.0)
        &
        hist["snapshot_time"].notna()
        &
        hist["commence_time"].notna()
    ].copy()

    return hist


# ============================================================
# BUILD ONE BOOK'S COMPLETE O/U MARKET
# ============================================================

def build_book_market(close):

    """
    Convert row-level Over / Under observations into one
    row per bookmaker.

    We require BOTH sides from the SAME bookmaker at the
    SAME closing snapshot before that book contributes to
    the consensus.
    """

    over = close[
        close["selection"].eq("OVER")
    ][
        [
            "bookmaker",
            "decimal_odds",
        ]
    ].copy()

    under = close[
        close["selection"].eq("UNDER")
    ][
        [
            "bookmaker",
            "decimal_odds",
        ]
    ].copy()

    over = over.rename(
        columns={
            "decimal_odds":
                "over_odds",
        }
    )

    under = under.rename(
        columns={
            "decimal_odds":
                "under_odds",
        }
    )

    # Protect against duplicate bookmaker rows.

    over = (
        over
        .groupby(
            "bookmaker",
            as_index=False,
        )["over_odds"]
        .mean()
    )

    under = (
        under
        .groupby(
            "bookmaker",
            as_index=False,
        )["under_odds"]
        .mean()
    )

    books = over.merge(
        under,
        on="bookmaker",
        how="inner",
        validate="one_to_one",
    )

    books = books[
        books["over_odds"].gt(1.0)
        &
        books["under_odds"].gt(1.0)
    ].copy()

    if books.empty:
        return books

    # --------------------------------------------------------
    # DE-VIG EACH BOOK INDEPENDENTLY
    # --------------------------------------------------------

    books[
        "_raw_over"
    ] = (
        1.0
        /
        books["over_odds"]
    )

    books[
        "_raw_under"
    ] = (
        1.0
        /
        books["under_odds"]
    )

    books[
        "_vig_sum"
    ] = (
        books["_raw_over"]
        +
        books["_raw_under"]
    )

    books[
        "_nv_over"
    ] = (
        books["_raw_over"]
        /
        books["_vig_sum"]
    )

    books[
        "_nv_under"
    ] = (
        books["_raw_under"]
        /
        books["_vig_sum"]
    )

    return books


# ============================================================
# UPDATE CLV
# ============================================================

def update_clv(
    ledger,
    hist,
):

    if ledger.empty:
        return ledger, []

    if hist.empty:
        return ledger, []

    now = pd.Timestamp.now(
        tz="UTC"
    )

    updated = []

    for idx, bet in ledger.iterrows():

        status = (
            str(
                bet.get(
                    "status",
                    ""
                )
            )
            .strip()
            .upper()
        )

        if status not in {
            "OPEN",
            "CLOSED_LINE",
        }:
            continue

        # Already captured.
        existing_close = safe_float(
            bet.get(
                "closing_decimal_odds"
            )
        )

        if np.isfinite(
            existing_close
        ):
            continue

        side = normalize_selection(
            bet.get(
                "selection"
            )
        )

        if side is None:
            continue

        event_id = (
            str(
                bet.get(
                    "event_id",
                    ""
                )
            )
            .strip()
        )

        if (
            not event_id
            or
            event_id.lower() == "nan"
        ):
            continue

        # ----------------------------------------------------
        # EVENT MATCH
        # ----------------------------------------------------

        match_hist = hist[
            hist["event_id"]
            .astype(str)
            .str.strip()
            .eq(event_id)
        ].copy()

        if match_hist.empty:
            continue

        commence_values = (
            match_hist[
                "commence_time"
            ]
            .dropna()
        )

        if commence_values.empty:
            continue

        commence_time = (
            commence_values.iloc[0]
        )

        ledger.at[
            idx,
            "commence_time"
        ] = (
            commence_time.isoformat()
        )

        # ----------------------------------------------------
        # DO NOT DECLARE A CLOSE BEFORE KICKOFF
        # ----------------------------------------------------

        if now < commence_time:
            continue

        # ----------------------------------------------------
        # ABSOLUTE SAFEGUARD:
        # NO SNAPSHOT AT OR AFTER KICKOFF.
        # ----------------------------------------------------

        eligible = match_hist[
            match_hist[
                "snapshot_time"
            ]
            <
            commence_time
        ].copy()

        if eligible.empty:
            continue

        # Latest recorded legitimate pregame snapshot.

        final_snapshot = (
            eligible[
                "snapshot_time"
            ]
            .max()
        )

        close = eligible[
            eligible[
                "snapshot_time"
            ]
            .eq(final_snapshot)
        ].copy()

        if close.empty:
            continue

        # ----------------------------------------------------
        # REQUIRE COMPLETE SAME-BOOK O/U PAIRS
        # ----------------------------------------------------

        books = build_book_market(
            close
        )

        if books.empty:
            continue

        probability_column = {
            "OVER":
                "_nv_over",

            "UNDER":
                "_nv_under",
        }[
            side
        ]

        odds_column = {
            "OVER":
                "over_odds",

            "UNDER":
                "under_odds",
        }[
            side
        ]

        # ----------------------------------------------------
        # CONSENSUS CLOSE
        #
        # Keep this aligned with the existing V5 1X2 CLV:
        # mean across valid bookmaker markets.
        # ----------------------------------------------------

        closing_market_probability = (
            books[
                probability_column
            ]
            .mean()
        )

        closing_decimal_odds = (
            books[
                odds_column
            ]
            .mean()
        )

        bet_odds = safe_float(
            bet.get(
                "decimal_odds"
            )
        )

        signal_market_probability = safe_float(
            bet.get(
                "market_probability"
            )
        )

        if (
            not np.isfinite(
                closing_decimal_odds
            )
            or
            closing_decimal_odds <= 1.0
            or
            not np.isfinite(
                closing_market_probability
            )
            or
            not np.isfinite(
                bet_odds
            )
            or
            bet_odds <= 1.0
            or
            not np.isfinite(
                signal_market_probability
            )
        ):
            continue

        # ----------------------------------------------------
        # CLV
        # ----------------------------------------------------

        price_clv = (
            bet_odds
            /
            closing_decimal_odds
            -
            1.0
        )

        probability_clv = (
            closing_market_probability
            -
            signal_market_probability
        )

        beat_close = int(
            price_clv > 0
        )

        closing_american_odds = (
            decimal_to_american(
                closing_decimal_odds
            )
        )

        # ----------------------------------------------------
        # WRITE TO LEDGER
        # ----------------------------------------------------

        ledger.at[
            idx,
            "closing_decimal_odds"
        ] = closing_decimal_odds

        ledger.at[
            idx,
            "closing_american_odds"
        ] = closing_american_odds

        ledger.at[
            idx,
            "closing_market_probability"
        ] = closing_market_probability

        ledger.at[
            idx,
            "closing_snapshot_time"
        ] = (
            final_snapshot.isoformat()
        )

        ledger.at[
            idx,
            "price_clv"
        ] = price_clv

        ledger.at[
            idx,
            "probability_clv"
        ] = probability_clv

        ledger.at[
            idx,
            "beat_close"
        ] = beat_close

        if status == "OPEN":

            ledger.at[
                idx,
                "status"
            ] = "CLOSED_LINE"

        updated.append(
            {
                "league":
                    bet.get(
                        "league",
                        ""
                    ),

                "home_team":
                    bet.get(
                        "home_team",
                        ""
                    ),

                "away_team":
                    bet.get(
                        "away_team",
                        ""
                    ),

                "selection":
                    bet.get(
                        "selection",
                        ""
                    ),

                "bet_odds":
                    bet_odds,

                "closing_odds":
                    closing_decimal_odds,

                "closing_american":
                    closing_american_odds,

                "price_clv":
                    price_clv,

                "probability_clv":
                    probability_clv,

                "beat_close":
                    beat_close,

                "books":
                    len(books),

                "snapshot":
                    final_snapshot,
            }
        )

    return ledger, updated


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print("UPDATE LIVE V5 TOTALS CLV")
    print("=" * 110)

    if not LEDGER_FILE.exists():

        raise FileNotFoundError(
            f"Missing totals ledger: "
            f"{LEDGER_FILE}"
        )

    ledger = pd.read_csv(
        LEDGER_FILE,
        low_memory=False,
    )

    hist = load_history()

    print()
    print(
        f"Ledger rows: "
        f"{len(ledger)}"
    )

    print(
        f"Stored O/U 2.5 history rows: "
        f"{len(hist)}"
    )

    # --------------------------------------------------------
    # ENSURE CLV COLUMNS
    # --------------------------------------------------------

    columns = [
        "closing_decimal_odds",
        "closing_american_odds",
        "closing_market_probability",
        "closing_snapshot_time",
        "price_clv",
        "probability_clv",
        "beat_close",
    ]

    for col in columns:

        if col not in ledger.columns:
            ledger[col] = np.nan

    ledger, updated = update_clv(
        ledger,
        hist,
    )

    ledger.to_csv(
        LEDGER_FILE,
        index=False,
    )

    print()
    print(
        f"New closing lines captured: "
        f"{len(updated)}"
    )

    if updated:

        show = pd.DataFrame(
            updated
        )

        show[
            "bet_american"
        ] = show[
            "bet_odds"
        ].map(
            decimal_to_american
        )

        show[
            "price_clv_pct"
        ] = (
            show["price_clv"]
            *
            100.0
        )

        show[
            "prob_clv_pct"
        ] = (
            show[
                "probability_clv"
            ]
            *
            100.0
        )

        print()
        print("=" * 110)
        print("NEW TOTALS CLOSING LINES")
        print("=" * 110)
        print()

        print(
            show[
                [
                    "league",
                    "home_team",
                    "away_team",
                    "selection",
                    "bet_american",
                    "closing_american",
                    "price_clv_pct",
                    "prob_clv_pct",
                    "beat_close",
                    "books",
                    "snapshot",
                ]
            ]
            .to_string(
                index=False,
                formatters={
                    "price_clv_pct":
                        lambda x:
                            f"{x:+.2f}%",

                    "prob_clv_pct":
                        lambda x:
                            f"{x:+.2f}%",
                },
            )
        )

    # --------------------------------------------------------
    # ALL CAPTURED CLV SUMMARY
    # --------------------------------------------------------

    captured = ledger[
        pd.to_numeric(
            ledger[
                "closing_decimal_odds"
            ],
            errors="coerce",
        ).notna()
    ].copy()

    print()
    print("=" * 110)
    print("TOTALS CLV SUMMARY")
    print("=" * 110)

    if captured.empty:

        print()
        print(
            "No totals closing lines "
            "captured yet."
        )

    else:

        captured[
            "price_clv"
        ] = pd.to_numeric(
            captured[
                "price_clv"
            ],
            errors="coerce",
        )

        captured[
            "probability_clv"
        ] = pd.to_numeric(
            captured[
                "probability_clv"
            ],
            errors="coerce",
        )

        captured[
            "beat_close"
        ] = pd.to_numeric(
            captured[
                "beat_close"
            ],
            errors="coerce",
        )

        n = len(captured)

        avg_price_clv = (
            captured[
                "price_clv"
            ].mean()
        )

        avg_probability_clv = (
            captured[
                "probability_clv"
            ].mean()
        )

        beat_rate = (
            captured[
                "beat_close"
            ].mean()
        )

        print()
        print(
            f"Captured bets:       {n}"
        )

        print(
            f"Average price CLV:   "
            f"{avg_price_clv:+.2%}"
        )

        print(
            f"Average prob. CLV:   "
            f"{avg_probability_clv:+.2%}"
        )

        print(
            f"Beat close:          "
            f"{beat_rate:.2%}"
        )

    print()
    print(
        f"Saved: {LEDGER_FILE}"
    )


if __name__ == "__main__":
    main()
