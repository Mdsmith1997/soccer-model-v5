from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"

LEDGER_FILE = LIVE / "btts_specialist_bet_ledger.csv"
RESULTS_FILE = LIVE / "soccer_results.csv"


def banner(text):
    print()
    print("=" * 110)
    print(text)
    print("=" * 110)


def norm_text(value):
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("fc", "")
        .replace(".", "")
        .replace("-", " ")
    )


def settle_row(row, results):

    league = str(row["league"])
    home = norm_text(row["home_team"])
    away = norm_text(row["away_team"])

    r = results[
        results["league"]
        .astype(str)
        .eq(league)
    ].copy()

    if len(r) == 0:
        return None

    r["_home"] = r[
        "home_team"
    ].map(norm_text)

    r["_away"] = r[
        "away_team"
    ].map(norm_text)

    exact = r[
        r["_home"].eq(home)
        &
        r["_away"].eq(away)
    ].copy()

    if len(exact) == 0:
        return None

    # Results file is a completed-match snapshot.
    game = exact.iloc[-1]

    hg = pd.to_numeric(
        game["home_score"],
        errors="coerce",
    )

    ag = pd.to_numeric(
        game["away_score"],
        errors="coerce",
    )

    if pd.isna(hg) or pd.isna(ag):
        return None

    btts_yes = (
        float(hg) > 0
        and
        float(ag) > 0
    )

    # Frozen BTTS ledger stores the official entry price as bet_odds.
    odds = float(
        row["bet_odds"]
    )

    if btts_yes:
        bet_result = "WIN"
        profit = odds - 1.0
        result = "YES"
    else:
        bet_result = "LOSS"
        profit = -1.0
        result = "NO"

    return {
        "result": result,
        "bet_result": bet_result,
        "profit_units": profit,
        "home_score": float(hg),
        "away_score": float(ag),
    }


def main():

    banner("SETTLE LIVE BTTS SPECIALIST BETS")

    if not LEDGER_FILE.exists():
        print(
            "No BTTS specialist ledger exists yet."
        )
        return

    if not RESULTS_FILE.exists():
        print(
            "No soccer result file available."
        )
        return

    ledger = pd.read_csv(
        LEDGER_FILE,
        low_memory=False,
    )

    # Settlement text fields may be entirely blank in a new ledger.
    # Pandas then infers them as float64, which prevents writing
    # values such as "YES", "NO", "WIN", and "LOSS".
    for col in [
        "status",
        "result",
        "bet_result",
    ]:
        if col in ledger.columns:
            ledger[col] = ledger[col].astype("object")

    results = pd.read_csv(
        RESULTS_FILE,
        low_memory=False,
    )

    open_mask = (
        ledger["status"]
        .astype(str)
        .str.upper()
        .eq("OPEN")
    )

    newly_settled = []

    for idx in ledger[
        open_mask
    ].index:

        outcome = settle_row(
            ledger.loc[idx],
            results,
        )

        if outcome is None:
            continue

        ledger.loc[
            idx,
            "status",
        ] = "SETTLED"

        for key, value in outcome.items():
            ledger.loc[
                idx,
                key,
            ] = value

        newly_settled.append(idx)

    ledger.to_csv(
        LEDGER_FILE,
        index=False,
    )

    print(
        f"Ledger rows: {len(ledger)}"
    )
    print(
        f"New bets settled: "
        f"{len(newly_settled)}"
    )

    if newly_settled:

        banner("NEWLY SETTLED BTTS BETS")

        cols = [
            "league",
            "home_team",
            "away_team",
            "strategy",
            "bet_odds",
            "result",
            "bet_result",
            "profit_units",
        ]

        print(
            ledger.loc[
                newly_settled,
                cols,
            ]
            .to_string(index=False)
        )

    settled = ledger[
        ledger["status"]
        .astype(str)
        .str.upper()
        .eq("SETTLED")
    ].copy()

    banner("BTTS SPECIALIST FORWARD TEST")

    if len(settled) == 0:

        print("Settled bets: 0")
        print("Record: 0-0")
        print("Profit: +0.00 units")
        print("ROI: -")
        return

    wins = (
        settled["bet_result"]
        .astype(str)
        .eq("WIN")
        .sum()
    )

    losses = (
        settled["bet_result"]
        .astype(str)
        .eq("LOSS")
        .sum()
    )

    profit = pd.to_numeric(
        settled["profit_units"],
        errors="coerce",
    ).sum()

    roi = (
        profit
        /
        len(settled)
        *
        100
    )

    print(
        f"Settled bets: {len(settled)}"
    )
    print(
        f"Record: {wins}-{losses}"
    )
    print(
        f"Win rate: "
        f"{wins / len(settled) * 100:.2f}%"
    )
    print(
        f"Profit: {profit:+.2f} units"
    )
    print(
        f"Flat-stake ROI: {roi:+.2f}%"
    )

    banner("BY BTTS SPECIALIST")

    grouped = []

    for strategy, z in settled.groupby(
        "strategy"
    ):

        w = (
            z["bet_result"]
            .astype(str)
            .eq("WIN")
            .sum()
        )

        l = (
            z["bet_result"]
            .astype(str)
            .eq("LOSS")
            .sum()
        )

        p = pd.to_numeric(
            z["profit_units"],
            errors="coerce",
        ).sum()

        grouped.append(
            {
                "strategy": strategy,
                "bets": len(z),
                "record": f"{w}-{l}",
                "profit_units": p,
                "roi":
                    p / len(z) * 100,
            }
        )

    print(
        pd.DataFrame(
            grouped
        ).to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(LEDGER_FILE)


if __name__ == "__main__":
    main()
