from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BOARD_FILE = (
    ROOT
    / "data"
    / "live"
    / "v5_live_totals_ev_board.csv"
)

LEDGER_FILE = (
    ROOT
    / "data"
    / "live"
    / "v5_live_totals_bet_ledger.csv"
)


# ============================================================
# FROZEN FORWARD-TEST RULE
# ============================================================

MARKET = "Under 2.5"
RAW_EDGE_THRESHOLD = 0.11

VALIDATED_LEAGUES = {
    "Premier League",
    "Bundesliga",
}


# ============================================================
# LEDGER SCHEMA
# ============================================================

LEDGER_COLUMNS = [
    "signal_id",
    "first_seen_utc",
    "date",
    "league",
    "home_team",
    "away_team",
    "match_id",

    # Event linkage / timing.
    "event_id",
    "commence_time",

    # Market / selection.
    "market",
    "selection",

    # Entry signal.
    "model_probability",
    "market_probability",
    "edge",
    "decimal_odds",
    "american_odds",
    "bookmaker",
    "ev",

    # Model metadata.
    "deployment_tier",
    "home_history_source",
    "away_history_source",
    "validation_status",
    "rule_name",
    "rule_threshold",

    # Closing-line tracking.
    "closing_decimal_odds",
    "closing_american_odds",
    "closing_market_probability",
    "closing_snapshot_time",
    "price_clv",
    "probability_clv",
    "beat_close",

    # Settlement.
    "status",
    "result",
    "home_score",
    "away_score",
    "actual_total",
    "won",
    "profit_units",
]


# ============================================================
# HELPERS
# ============================================================

def decimal_to_american(decimal_odds):

    try:
        d = float(decimal_odds)
    except (TypeError, ValueError):
        return pd.NA

    if pd.isna(d) or d <= 1.0:
        return pd.NA

    if d >= 2.0:
        return round(
            (d - 1.0) * 100
        )

    return round(
        -100 / (d - 1.0)
    )


def load_existing():

    if not LEDGER_FILE.exists():
        return pd.DataFrame(
            columns=LEDGER_COLUMNS
        )

    ledger = pd.read_csv(
        LEDGER_FILE,
        low_memory=False,
    )

    for col in LEDGER_COLUMNS:
        if col not in ledger.columns:
            ledger[col] = pd.NA

    return ledger[
        LEDGER_COLUMNS
    ].copy()


def save_ledger(ledger):

    LEDGER_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for col in LEDGER_COLUMNS:
        if col not in ledger.columns:
            ledger[col] = pd.NA

    ledger[
        LEDGER_COLUMNS
    ].to_csv(
        LEDGER_FILE,
        index=False,
    )


def safe_value(row, column):

    if column not in row.index:
        return pd.NA

    value = row[column]

    if pd.isna(value):
        return pd.NA

    return value


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print("LIVE V5 TOTALS FORWARD-TEST LEDGER")
    print("=" * 100)

    # --------------------------------------------------------
    # Load/create ledger FIRST.
    #
    # This guarantees the ledger exists even when there are
    # zero qualifying bets today.
    # --------------------------------------------------------

    ledger = load_existing()

    save_ledger(
        ledger
    )

    if not BOARD_FILE.exists():
        raise FileNotFoundError(
            f"Missing totals board: {BOARD_FILE}"
        )

    board = pd.read_csv(
        BOARD_FILE,
        low_memory=False,
    )

    required = [
        "date",
        "league",
        "home_team",
        "away_team",
        "bet",
        "model_probability",
        "market_probability",
        "edge",
        "decimal_odds",
        "bookmaker",
        "ev",
        "match_id",
    ]

    missing = [
        c
        for c in required
        if c not in board.columns
    ]

    if missing:
        raise RuntimeError(
            f"Totals board missing columns: {missing}"
        )

    for col in [
        "model_probability",
        "market_probability",
        "edge",
        "decimal_odds",
        "ev",
    ]:
        board[col] = pd.to_numeric(
            board[col],
            errors="coerce",
        )

    # ========================================================
    # FROZEN RULE
    #
    # UNDER 2.5
    # RAW V5 EDGE >= 11%
    # ========================================================

    qualifiers = board[
        board["bet"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq(MARKET.casefold())
        &
        (
            board["edge"]
            >= RAW_EDGE_THRESHOLD
        )
    ].copy()

    if qualifiers.empty:

        print()
        print(
            "No RAW Under 2.5 signals >= 11% "
            "on the current board."
        )

        print()
        print(
            f"Existing ledger signals: "
            f"{len(ledger)}"
        )

        print(
            f"Ledger initialized: "
            f"{LEDGER_FILE}"
        )

        return

    # ========================================================
    # VALIDATION CLASSIFICATION
    # ========================================================

    qualifiers[
        "validation_status"
    ] = qualifiers[
        "league"
    ].apply(
        lambda x:
            "VALIDATED"
            if x in VALIDATED_LEAGUES
            else "RESEARCH_ONLY"
    )

    qualifiers[
        "american_odds"
    ] = qualifiers[
        "decimal_odds"
    ].apply(
        decimal_to_american
    )

    # ========================================================
    # ONE TOTALS SIGNAL PER MATCH
    # ========================================================

    qualifiers = (
        qualifiers
        .sort_values(
            [
                "match_id",
                "edge",
                "ev",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=["match_id"],
            keep="first",
        )
        .copy()
    )

    now = datetime.now(
        timezone.utc
    ).isoformat()

    new_rows = []

    # ========================================================
    # FREEZE FIRST QUALIFYING SIGNAL
    # ========================================================

    for _, row in qualifiers.iterrows():

        signal_id = (
            f"{row['match_id']}"
            f"|TOTALS"
            f"|UNDER_2_5"
            f"|RAW11"
        )

        if (
            not ledger.empty
            and
            ledger["signal_id"]
            .astype(str)
            .eq(signal_id)
            .any()
        ):
            continue

        new_rows.append(
            {
                "signal_id":
                    signal_id,

                "first_seen_utc":
                    now,

                "date":
                    row["date"],

                "league":
                    row["league"],

                "home_team":
                    row["home_team"],

                "away_team":
                    row["away_team"],

                "match_id":
                    row["match_id"],

                "event_id":
                    safe_value(
                        row,
                        "event_id",
                    ),

                "commence_time":
                    safe_value(
                        row,
                        "commence_time",
                    ),

                "market":
                    "TOTALS",

                "selection":
                    "Under 2.5",

                "model_probability":
                    row[
                        "model_probability"
                    ],

                "market_probability":
                    row[
                        "market_probability"
                    ],

                "edge":
                    row["edge"],

                "decimal_odds":
                    row[
                        "decimal_odds"
                    ],

                "american_odds":
                    row[
                        "american_odds"
                    ],

                "bookmaker":
                    row["bookmaker"],

                "ev":
                    row["ev"],

                "deployment_tier":
                    safe_value(
                        row,
                        "deployment_tier",
                    ),

                "home_history_source":
                    safe_value(
                        row,
                        "home_history_source",
                    ),

                "away_history_source":
                    safe_value(
                        row,
                        "away_history_source",
                    ),

                "validation_status":
                    row[
                        "validation_status"
                    ],

                "rule_name":
                    "RAW_UNDER_11",

                "rule_threshold":
                    RAW_EDGE_THRESHOLD,

                "closing_decimal_odds":
                    pd.NA,

                "closing_american_odds":
                    pd.NA,

                "closing_market_probability":
                    pd.NA,

                "closing_snapshot_time":
                    pd.NA,

                "price_clv":
                    pd.NA,

                "probability_clv":
                    pd.NA,

                "beat_close":
                    pd.NA,

                "status":
                    "OPEN",

                "result":
                    pd.NA,

                "home_score":
                    pd.NA,

                "away_score":
                    pd.NA,

                "actual_total":
                    pd.NA,

                "won":
                    pd.NA,

                "profit_units":
                    pd.NA,
            }
        )

    # ========================================================
    # APPEND + SAVE
    # ========================================================

    if new_rows:

        ledger = pd.concat(
            [
                ledger,
                pd.DataFrame(
                    new_rows
                ),
            ],
            ignore_index=True,
        )

    save_ledger(
        ledger
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print(
        f"Current qualifying signals: "
        f"{len(qualifiers)}"
    )

    print(
        f"New frozen signals: "
        f"{len(new_rows)}"
    )

    print(
        f"Total ledger signals: "
        f"{len(ledger)}"
    )

    validated = qualifiers[
        qualifiers[
            "validation_status"
        ]
        ==
        "VALIDATED"
    ]

    research = qualifiers[
        qualifiers[
            "validation_status"
        ]
        ==
        "RESEARCH_ONLY"
    ]

    print(
        f"Validated EPL/Bundesliga: "
        f"{len(validated)}"
    )

    print(
        f"Research-only other leagues: "
        f"{len(research)}"
    )

    if not qualifiers.empty:

        show = qualifiers[
            [
                "date",
                "league",
                "home_team",
                "away_team",
                "bet",
                "model_probability",
                "market_probability",
                "edge",
                "decimal_odds",
                "american_odds",
                "bookmaker",
                "ev",
                "validation_status",
            ]
        ].copy()

        for c in [
            "model_probability",
            "market_probability",
            "edge",
            "ev",
        ]:
            show[c] *= 100.0

        print()
        print("=" * 130)
        print("CURRENT QUALIFIERS")
        print("=" * 130)
        print()

        print(
            show.to_string(
                index=False,
                formatters={
                    "model_probability":
                        lambda x:
                            f"{x:.2f}%",

                    "market_probability":
                        lambda x:
                            f"{x:.2f}%",

                    "edge":
                        lambda x:
                            f"{x:+.2f}%",

                    "decimal_odds":
                        lambda x:
                            f"{x:.2f}",

                    "american_odds":
                        lambda x:
                            (
                                ""
                                if pd.isna(x)
                                else f"{int(x):+d}"
                            ),

                    "ev":
                        lambda x:
                            f"{x:+.2f}%",
                },
            )
        )

    print()
    print(
        f"Saved: {LEDGER_FILE}"
    )


if __name__ == "__main__":
    main()
