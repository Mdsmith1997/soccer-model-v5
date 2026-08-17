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

RESULTS_FILE = (
    ROOT
    / "data"
    / "live"
    / "soccer_results.csv"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value):

    return (
        str(value)
        .strip()
        .casefold()
    )


def safe_float(value):

    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(x):
        return np.nan

    return x


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print("SETTLE LIVE V5 TOTALS BETS")
    print("=" * 100)

    # --------------------------------------------------------
    # FILE CHECKS
    # --------------------------------------------------------

    if not LEDGER_FILE.exists():

        raise FileNotFoundError(
            f"Missing totals ledger: {LEDGER_FILE}"
        )

    if not RESULTS_FILE.exists():

        raise FileNotFoundError(
            f"Missing results file: {RESULTS_FILE}"
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    ledger = pd.read_csv(
        LEDGER_FILE,
        low_memory=False,
    )

    results = pd.read_csv(
        RESULTS_FILE,
        low_memory=False,
    )

    print()
    print(f"Ledger rows: {len(ledger)}")
    print(f"Results rows: {len(results)}")

    # --------------------------------------------------------
    # EMPTY LEDGER IS NORMAL
    # --------------------------------------------------------

    if ledger.empty:

        print()
        print(
            "No totals signals have been recorded yet."
        )

        print(
            "Nothing to settle."
        )

        return

    # --------------------------------------------------------
    # REQUIRED LEDGER COLUMNS
    # --------------------------------------------------------

    required_ledger = [
        "status",
        "selection",
        "decimal_odds",
        "home_team",
        "away_team",
    ]

    missing = [
        c
        for c in required_ledger
        if c not in ledger.columns
    ]

    if missing:

        raise RuntimeError(
            f"Totals ledger missing columns: {missing}"
        )

    # --------------------------------------------------------
    # REQUIRED RESULTS COLUMNS
    # --------------------------------------------------------

    required_results = [
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]

    missing = [
        c
        for c in required_results
        if c not in results.columns
    ]

    if missing:

        raise RuntimeError(
            f"Results file missing columns: {missing}"
        )

    # --------------------------------------------------------
    # MAKE SURE SETTLEMENT COLUMNS EXIST
    # --------------------------------------------------------

    settlement_columns = [
        "result",
        "home_score",
        "away_score",
        "actual_total",
        "won",
        "profit_units",
    ]

    for col in settlement_columns:

        if col not in ledger.columns:
            ledger[col] = np.nan

    # --------------------------------------------------------
    # NORMALIZE RESULTS
    # --------------------------------------------------------

    results["home_score"] = pd.to_numeric(
        results["home_score"],
        errors="coerce",
    )

    results["away_score"] = pd.to_numeric(
        results["away_score"],
        errors="coerce",
    )

    # Only completed games can settle.

    if "completed" in results.columns:

        completed = (
            results["completed"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .isin(
                {
                    "true",
                    "1",
                    "yes",
                }
            )
        )

        results = results[
            completed
        ].copy()

    results = results[
        results["home_score"].notna()
        &
        results["away_score"].notna()
    ].copy()

    # --------------------------------------------------------
    # MATCHING KEYS
    # --------------------------------------------------------

    ledger["_home_key"] = (
        ledger["home_team"]
        .map(normalize_text)
    )

    ledger["_away_key"] = (
        ledger["away_team"]
        .map(normalize_text)
    )

    results["_home_key"] = (
        results["home_team"]
        .map(normalize_text)
    )

    results["_away_key"] = (
        results["away_team"]
        .map(normalize_text)
    )

    # --------------------------------------------------------
    # SETTLE OPEN BETS
    # --------------------------------------------------------

    settled_rows = []

    open_count = 0
    unmatched_count = 0

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

        if status != "OPEN":
            continue

        open_count += 1

        # ----------------------------------------------------
        # Prefer event_id when both datasets contain it.
        # ----------------------------------------------------

        matches = pd.DataFrame()

        event_id = str(
            bet.get(
                "event_id",
                ""
            )
        ).strip()

        if (
            event_id
            and
            event_id.lower() != "nan"
            and
            "event_id" in results.columns
        ):

            matches = results[
                results["event_id"]
                .astype(str)
                .str.strip()
                .eq(event_id)
            ]

        # ----------------------------------------------------
        # Fallback: home/away team matching.
        # ----------------------------------------------------

        if matches.empty:

            matches = results[
                (
                    results["_home_key"]
                    ==
                    bet["_home_key"]
                )
                &
                (
                    results["_away_key"]
                    ==
                    bet["_away_key"]
                )
            ]

        if matches.empty:

            unmatched_count += 1
            continue

        # Most recent duplicate if necessary.

        result = matches.iloc[-1]

        home_score = safe_float(
            result["home_score"]
        )

        away_score = safe_float(
            result["away_score"]
        )

        if (
            not np.isfinite(home_score)
            or
            not np.isfinite(away_score)
        ):
            continue

        actual_total = (
            home_score
            +
            away_score
        )

        selection = (
            str(
                bet["selection"]
            )
            .strip()
            .upper()
        )

        # ----------------------------------------------------
        # TOTALS SETTLEMENT
        # ----------------------------------------------------

        if selection == "UNDER 2.5":

            won = (
                actual_total
                <
                2.5
            )

        elif selection == "OVER 2.5":

            won = (
                actual_total
                >
                2.5
            )

        else:

            print(
                f"Skipping unsupported selection: "
                f"{bet['selection']}"
            )

            continue

        decimal_odds = safe_float(
            bet["decimal_odds"]
        )

        if not np.isfinite(decimal_odds):
            continue

        if decimal_odds <= 1.0:
            continue

        # ----------------------------------------------------
        # UNIT PROFIT
        #
        # 1-unit flat stake:
        #
        # win  = decimal odds - 1
        # loss = -1
        # ----------------------------------------------------

        if won:

            profit_units = (
                decimal_odds
                -
                1.0
            )

            result_label = "WIN"

        else:

            profit_units = -1.0
            result_label = "LOSS"

        # ----------------------------------------------------
        # UPDATE LEDGER
        # ----------------------------------------------------

        ledger.at[
            idx,
            "home_score"
        ] = home_score

        ledger.at[
            idx,
            "away_score"
        ] = away_score

        ledger.at[
            idx,
            "actual_total"
        ] = actual_total

        ledger.at[
            idx,
            "won"
        ] = int(won)

        ledger.at[
            idx,
            "profit_units"
        ] = profit_units

        ledger.at[
            idx,
            "result"
        ] = result_label

        ledger.at[
            idx,
            "status"
        ] = "SETTLED"

        settled_rows.append(
            {
                "league":
                    bet.get(
                        "league",
                        ""
                    ),

                "home_team":
                    bet["home_team"],

                "away_team":
                    bet["away_team"],

                "selection":
                    bet["selection"],

                "home_score":
                    int(home_score),

                "away_score":
                    int(away_score),

                "actual_total":
                    int(actual_total),

                "decimal_odds":
                    decimal_odds,

                "result":
                    result_label,

                "profit_units":
                    profit_units,
            }
        )

    # --------------------------------------------------------
    # REMOVE TEMP KEYS
    # --------------------------------------------------------

    ledger = ledger.drop(
        columns=[
            "_home_key",
            "_away_key",
        ],
        errors="ignore",
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    ledger.to_csv(
        LEDGER_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print(f"Open bets checked: {open_count}")
    print(
        f"New ledger bets settled: "
        f"{len(settled_rows)}"
    )
    print(
        f"Open bets without result: "
        f"{unmatched_count}"
    )

    if settled_rows:

        settled = pd.DataFrame(
            settled_rows
        )

        print()
        print("=" * 100)
        print("NEWLY SETTLED TOTALS BETS")
        print("=" * 100)
        print()

        print(
            settled.to_string(
                index=False,
                formatters={
                    "decimal_odds":
                        lambda x:
                            f"{x:.2f}",

                    "profit_units":
                        lambda x:
                            f"{x:+.2f}",
                },
            )
        )

    # ========================================================
    # FORWARD-TEST PERFORMANCE
    # ========================================================

    all_settled = ledger[
        ledger["status"]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("SETTLED")
    ].copy()

    if not all_settled.empty:

        all_settled[
            "profit_units"
        ] = pd.to_numeric(
            all_settled[
                "profit_units"
            ],
            errors="coerce",
        )

        all_settled[
            "won"
        ] = pd.to_numeric(
            all_settled[
                "won"
            ],
            errors="coerce",
        )

        bets = len(
            all_settled
        )

        wins = int(
            all_settled[
                "won"
            ].fillna(0).sum()
        )

        profit = (
            all_settled[
                "profit_units"
            ]
            .fillna(0)
            .sum()
        )

        roi = (
            profit
            /
            bets
            if bets
            else 0.0
        )

        win_rate = (
            wins
            /
            bets
            if bets
            else 0.0
        )

        print()
        print("=" * 100)
        print("TOTALS FORWARD-TEST PERFORMANCE")
        print("=" * 100)

        print()
        print(
            f"Settled bets: {bets}"
        )

        print(
            f"Wins:         {wins}"
        )

        print(
            f"Win rate:     "
            f"{win_rate:.2%}"
        )

        print(
            f"Profit:       "
            f"{profit:+.2f}u"
        )

        print(
            f"ROI:          "
            f"{roi:+.2%}"
        )

    else:

        print()
        print(
            "No settled totals bets yet."
        )

    print()
    print(
        f"Saved: {LEDGER_FILE}"
    )


if __name__ == "__main__":
    main()
