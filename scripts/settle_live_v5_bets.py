from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LEDGER_FILE = (
    ROOT
    / "data"
    / "live"
    / "v5_live_bet_ledger.csv"
)

RESULTS_FILE = (
    ROOT
    / "data"
    / "live"
    / "soccer_results.csv"
)


def normalize_id(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():

    print()
    print("=" * 100)
    print("SETTLE LIVE V5 BETS")
    print("=" * 100)

    if not LEDGER_FILE.exists():
        raise FileNotFoundError(
            f"Missing ledger: {LEDGER_FILE}"
        )

    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing results: {RESULTS_FILE}"
        )

    ledger = pd.read_csv(
        LEDGER_FILE,
        low_memory=False,
    )

    # Force settlement columns to usable dtypes.
    if "actual_outcome" in ledger.columns:
        ledger["actual_outcome"] = (
            ledger["actual_outcome"]
            .astype("string")
        )
    else:
        ledger["actual_outcome"] = pd.Series(
            pd.NA,
            index=ledger.index,
            dtype="string",
        )

    if "status" in ledger.columns:
        ledger["status"] = (
            ledger["status"]
            .fillna("OPEN")
            .astype("string")
        )
    else:
        ledger["status"] = "OPEN"

    if "won" in ledger.columns:
        ledger["won"] = pd.to_numeric(
            ledger["won"],
            errors="coerce",
        )
    else:
        ledger["won"] = np.nan

    if "profit_units" in ledger.columns:
        ledger["profit_units"] = pd.to_numeric(
            ledger["profit_units"],
            errors="coerce",
        )
    else:
        ledger["profit_units"] = np.nan

    # Settlement columns need explicit dtypes because empty
    # CSV columns are otherwise often inferred as float64.
    ledger["actual_outcome"] = (
        ledger["actual_outcome"]
        .astype("string")
    )

    ledger["status"] = (
        ledger["status"]
        .fillna("OPEN")
        .astype("string")
    )

    ledger["won"] = pd.to_numeric(
        ledger["won"],
        errors="coerce",
    )

    ledger["profit_units"] = pd.to_numeric(
        ledger["profit_units"],
        errors="coerce",
    )

    results = pd.read_csv(
        RESULTS_FILE,
        low_memory=False,
    )

    # --------------------------------------------------------
    # CLEAN RESULTS
    # --------------------------------------------------------

    results["event_id"] = (
        results["event_id"]
        .map(normalize_id)
    )

    results["completed"] = (
        results["completed"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    results["home_score"] = pd.to_numeric(
        results["home_score"],
        errors="coerce",
    )

    results["away_score"] = pd.to_numeric(
        results["away_score"],
        errors="coerce",
    )

    results["result"] = (
        results["result"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    results = results[
        results["completed"]
        &
        results["home_score"].notna()
        &
        results["away_score"].notna()
        &
        results["result"].isin(
            ["HOME", "DRAW", "AWAY"]
        )
    ].copy()

    # One result per API event.
    results = (
        results
        .drop_duplicates(
            subset=["event_id"],
            keep="last",
        )
        .set_index("event_id")
    )

    # --------------------------------------------------------
    # CLEAN LEDGER
    # --------------------------------------------------------

    ledger["event_id"] = (
        ledger["event_id"]
        .map(normalize_id)
    )

    settled_rows = []

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

        # Only legitimate OPEN bets may be settled.
        # INVALID / SETTLED / other administrative rows are ignored.
        if status != "OPEN":
            continue

        event_id = normalize_id(
            bet.get("event_id")
        )

        if not event_id:
            continue

        if event_id not in results.index:
            continue

        result = results.loc[event_id]

        # Guard against accidental duplicate result rows.
        if isinstance(result, pd.DataFrame):
            result = result.iloc[-1]

        actual_outcome = (
            str(result["result"])
            .strip()
            .upper()
        )

        bet_side = (
            str(
                bet.get(
                    "bet_side",
                    ""
                )
            )
            .strip()
            .upper()
        )

        try:
            odds = float(
                bet["bet_odds"]
            )
        except (TypeError, ValueError):
            continue

        if (
            bet_side
            not in
            {"HOME", "DRAW", "AWAY"}
        ):
            continue

        if (
            not np.isfinite(odds)
            or odds <= 1.0
        ):
            continue

        won = int(
            bet_side
            ==
            actual_outcome
        )

        if won:
            profit_units = (
                odds
                -
                1.0
            )
        else:
            profit_units = -1.0

        ledger.at[
            idx,
            "actual_outcome"
        ] = actual_outcome

        ledger.at[
            idx,
            "won"
        ] = won

        ledger.at[
            idx,
            "profit_units"
        ] = profit_units

        ledger.at[
            idx,
            "status"
        ] = "SETTLED"

        settled_rows.append(
            {
                "league":
                    bet.get("league"),

                "home_team":
                    bet.get("home_team"),

                "away_team":
                    bet.get("away_team"),

                "bet_side":
                    bet_side,

                "bet_odds":
                    odds,

                "home_score":
                    int(
                        result["home_score"]
                    ),

                "away_score":
                    int(
                        result["away_score"]
                    ),

                "actual_outcome":
                    actual_outcome,

                "won":
                    won,

                "profit_units":
                    profit_units,
            }
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    ledger.to_csv(
        LEDGER_FILE,
        index=False,
    )

    print()
    print(
        f"Completed result events available: "
        f"{len(results)}"
    )

    print(
        f"New ledger bets settled: "
        f"{len(settled_rows)}"
    )

    if settled_rows:

        settled = pd.DataFrame(
            settled_rows
        )

        settled[
            "result"
        ] = (
            settled[
                "home_score"
            ]
            .astype(str)
            +
            "-"
            +
            settled[
                "away_score"
            ]
            .astype(str)
        )

        settled[
            "bet_result"
        ] = np.where(
            settled[
                "won"
            ]
            ==
            1,
            "WIN",
            "LOSS",
        )

        print()
        print("=" * 120)
        print("NEWLY SETTLED BETS")
        print("=" * 120)
        print()

        print(
            settled[
                [
                    "league",
                    "home_team",
                    "away_team",
                    "bet_side",
                    "bet_odds",
                    "result",
                    "bet_result",
                    "profit_units",
                ]
            ]
            .round(
                {
                    "bet_odds": 2,
                    "profit_units": 2,
                }
            )
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # FULL PERFORMANCE
    # --------------------------------------------------------

    all_settled = ledger[
        ledger[
            "status"
        ]
        .astype(str)
        .str.upper()
        .eq("SETTLED")
    ].copy()

    if not all_settled.empty:

        all_settled[
            "won"
        ] = pd.to_numeric(
            all_settled["won"],
            errors="coerce",
        )

        all_settled[
            "profit_units"
        ] = pd.to_numeric(
            all_settled[
                "profit_units"
            ],
            errors="coerce",
        )

        bets = len(
            all_settled
        )

        wins = int(
            all_settled[
                "won"
            ].sum()
        )

        losses = (
            bets
            -
            wins
        )

        profit = (
            all_settled[
                "profit_units"
            ].sum()
        )

        roi = (
            profit
            /
            bets
            if bets
            else np.nan
        )

        print()
        print("=" * 100)
        print("FORWARD TEST PERFORMANCE")
        print("=" * 100)
        print()

        print(
            f"Settled bets: {bets}"
        )

        print(
            f"Record: {wins}-{losses}"
        )

        print(
            f"Win rate: "
            f"{wins / bets:.2%}"
        )

        print(
            f"Profit: "
            f"{profit:+.2f} units"
        )

        print(
            f"ROI: "
            f"{roi:+.2%}"
        )

        # ----------------------------------------------------
        # HISTORY TYPE BREAKDOWN
        # ----------------------------------------------------

        if (
            "history_type"
            in
            all_settled.columns
        ):

            print()
            print("BY HISTORY TYPE")
            print()

            rows = []

            for (
                history_type,
                group,
            ) in all_settled.groupby(
                "history_type",
                dropna=False,
            ):

                n = len(group)

                w = int(
                    group[
                        "won"
                    ].sum()
                )

                p = (
                    group[
                        "profit_units"
                    ].sum()
                )

                rows.append(
                    {
                        "history_type":
                            history_type,

                        "bets":
                            n,

                        "wins":
                            w,

                        "losses":
                            n - w,

                        "profit_units":
                            p,

                        "roi":
                            p / n,
                    }
                )

            breakdown = pd.DataFrame(
                rows
            )

            breakdown[
                "roi"
            ] *= 100

            print(
                breakdown
                .round(2)
                .to_string(
                    index=False
                )
            )

    print()
    print(
        f"Saved: {LEDGER_FILE}"
    )


if __name__ == "__main__":
    main()
