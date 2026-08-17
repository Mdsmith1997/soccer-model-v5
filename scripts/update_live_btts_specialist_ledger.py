from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"

PRED_FILE = LIVE / "btts_live_predictions.csv"
LEDGER_FILE = LIVE / "btts_specialist_bet_ledger.csv"


SPECIALISTS = {
    "Swiss Super League": {
        "strategy": "SWISS_BTTS_YES",
        "threshold": 0.06,
    },
    "Super Lig": {
        "strategy": "SUPER_LIG_BTTS_YES",
        "threshold": 0.10,
    },
    "Segunda División": {
        "strategy": "SEGUNDA_BTTS_YES",
        "threshold": 0.04,
    },
}


LEDGER_COLUMNS = [
    "frozen_time",
    "match_id",
    "commence_time",
    "date",
    "league",
    "home_team",
    "away_team",
    "strategy",
    "bet_side",
    "threshold",
    "bookmaker",
    "bookmaker_key",
    "decimal_odds",
    "american_odds",
    "model_probability",
    "market_probability",
    "edge",
    "ev",
    "status",
    "result",
    "bet_result",
    "profit_units",
    "home_score",
    "away_score",
]


def banner(text):
    print()
    print("=" * 110)
    print(text)
    print("=" * 110)


def decimal_to_american(decimal_odds):
    try:
        d = float(decimal_odds)
    except Exception:
        return None

    if d <= 1:
        return None

    if d >= 2.0:
        return int(round((d - 1.0) * 100))

    return int(round(-100 / (d - 1.0)))


def empty_ledger():
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def load_ledger():
    if not LEDGER_FILE.exists():
        return empty_ledger()

    x = pd.read_csv(
        LEDGER_FILE,
        low_memory=False,
    )

    for col in LEDGER_COLUMNS:
        if col not in x.columns:
            x[col] = np.nan

    return x[LEDGER_COLUMNS].copy()


def main():

    banner("UPDATE LIVE BTTS SPECIALIST BET LEDGER")

    if not PRED_FILE.exists():
        print("No BTTS live prediction file found.")
        return

    pred = pd.read_csv(
        PRED_FILE,
        low_memory=False,
    )

    ledger = load_ledger()

    required = [
        "match_id",
        "commence_time",
        "league_market",
        "home_team_market",
        "away_team_market",
        "bookmaker",
        "yes_odds",
        "final_yes_probability",
        "market_yes",
        "yes_edge",
        "yes_ev",
    ]

    missing = [
        c for c in required
        if c not in pred.columns
    ]

    if missing:
        raise RuntimeError(
            f"BTTS prediction file missing columns: {missing}"
        )

    for col in [
        "yes_odds",
        "final_yes_probability",
        "market_yes",
        "yes_edge",
        "yes_ev",
    ]:
        pred[col] = pd.to_numeric(
            pred[col],
            errors="coerce",
        )

    pred["commence_time"] = pd.to_datetime(
        pred["commence_time"],
        errors="coerce",
        utc=True,
    )

    # ---------------------------------------------------------
    # EXACT CEMENTED SPECIALISTS ONLY
    # ---------------------------------------------------------

    qualifying = []

    for league, cfg in SPECIALISTS.items():

        q = pred[
            pred["league_market"]
            .astype(str)
            .eq(league)
            &
            pred["yes_edge"]
            .ge(cfg["threshold"])
            &
            pred["yes_odds"]
            .notna()
        ].copy()

        if len(q) == 0:
            continue

        q["strategy"] = cfg["strategy"]
        q["threshold"] = cfg["threshold"]

        qualifying.append(q)

    if not qualifying:
        print("No qualifying cemented BTTS signals.")
        print(f"Existing ledger bets: {len(ledger)}")
        return

    q = pd.concat(
        qualifying,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # ONE OFFICIAL PRICE PER FIXTURE PER SNAPSHOT
    #
    # Use the best available YES price at the moment the signal
    # first qualifies.
    # ---------------------------------------------------------

    q = (
        q.sort_values(
            [
                "match_id",
                "yes_odds",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=["match_id"],
            keep="first",
        )
        .copy()
    )

    existing_ids = set(
        ledger["match_id"]
        .dropna()
        .astype(str)
    )

    new_rows = []

    now = datetime.now(
        timezone.utc
    ).isoformat()

    for row in q.itertuples(index=False):

        match_id = str(row.match_id)

        # FIRST QUALIFYING SIGNAL FROZEN.
        if match_id in existing_ids:
            continue

        commence = pd.Timestamp(
            row.commence_time
        )

        new_rows.append(
            {
                "frozen_time": now,
                "match_id": match_id,
                "commence_time":
                    commence.isoformat(),
                "date":
                    commence.date().isoformat(),
                "league":
                    row.league_market,
                "home_team":
                    row.home_team_market,
                "away_team":
                    row.away_team_market,
                "strategy":
                    row.strategy,
                "bet_side":
                    "YES",
                "threshold":
                    float(row.threshold),
                "bookmaker":
                    row.bookmaker,
                "bookmaker_key":
                    getattr(
                        row,
                        "bookmaker_key",
                        np.nan,
                    ),
                "decimal_odds":
                    float(row.yes_odds),
                "american_odds":
                    decimal_to_american(
                        row.yes_odds
                    ),
                "model_probability":
                    float(
                        row.final_yes_probability
                    ),
                "market_probability":
                    float(row.market_yes),
                "edge":
                    float(row.yes_edge),
                "ev":
                    float(row.yes_ev),
                "status":
                    "OPEN",
                "result":
                    np.nan,
                "bet_result":
                    np.nan,
                "profit_units":
                    np.nan,
                "home_score":
                    np.nan,
                "away_score":
                    np.nan,
            }
        )

        existing_ids.add(match_id)

    if new_rows:

        new = pd.DataFrame(
            new_rows,
            columns=LEDGER_COLUMNS,
        )

        ledger = pd.concat(
            [
                ledger,
                new,
            ],
            ignore_index=True,
        )

        ledger.to_csv(
            LEDGER_FILE,
            index=False,
        )

        banner("NEWLY FROZEN BTTS BETS")

        show = new[
            [
                "date",
                "league",
                "home_team",
                "away_team",
                "strategy",
                "american_odds",
                "model_probability",
                "market_probability",
                "edge",
                "ev",
                "bookmaker",
            ]
        ].copy()

        for col in [
            "model_probability",
            "market_probability",
            "edge",
            "ev",
        ]:
            show[col] = (
                show[col] * 100
            ).map(
                lambda x: f"{x:.2f}%"
            )

        print(
            show.to_string(index=False)
        )

    else:
        print(
            "No new BTTS bets. "
            "Existing qualifying fixtures remain frozen."
        )

    print()
    print(
        f"New bets recorded: {len(new_rows)}"
    )
    print(
        f"Total BTTS specialist ledger bets: "
        f"{len(ledger)}"
    )
    print(
        "First qualifying signal frozen: YES"
    )
    print(
        "One official BTTS bet per match: YES"
    )

    print()
    print("Saved:")
    print(LEDGER_FILE)


if __name__ == "__main__":
    main()
