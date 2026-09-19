#!/usr/bin/env python3

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
LEAN_FILE = LIVE / "v5_live_lean_ledger.csv"
RESULTS_FILE = LIVE / "soccer_results.csv"


def clean(v):
    return "" if v is None else str(v).strip()


def number(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def truthy(v):
    return clean(v).lower() in {"true", "1", "yes"}


def load_csv(path):
    if not path.exists():
        return [], []

    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return list(r), list(r.fieldnames or [])


def main():
    print("=" * 90)
    print("SETTLE V5 1X2 RESEARCH LEANS")
    print("=" * 90)
    print("Research only: YES")
    print("Official ledger touched: NO")
    print()

    if not LEAN_FILE.exists():
        raise FileNotFoundError(f"Missing lean ledger: {LEAN_FILE}")

    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Missing results: {RESULTS_FILE}")

    leans, fields = load_csv(LEAN_FILE)
    results, _ = load_csv(RESULTS_FILE)

    # Match official settlement eligibility.
    by_event = {}

    for r in results:
        event_id = clean(r.get("event_id"))

        if not event_id or not truthy(r.get("completed")):
            continue

        home_score = number(r.get("home_score"))
        away_score = number(r.get("away_score"))
        outcome = clean(r.get("result")).upper()

        if (
            home_score is None
            or away_score is None
            or outcome not in {"HOME", "DRAW", "AWAY"}
        ):
            continue

        # Equivalent to drop_duplicates(..., keep="last").
        by_event[event_id] = r

    for col in [
        "actual_outcome",
        "won",
        "profit_units",
        "home_score",
        "away_score",
    ]:
        if col not in fields:
            fields.append(col)

    settled = []

    for lean in leans:
        if clean(lean.get("status")).upper() != "OPEN":
            continue

        event_id = clean(lean.get("event_id"))

        if not event_id or event_id not in by_event:
            continue

        result = by_event[event_id]

        actual = clean(result.get("result")).upper()
        side = clean(lean.get("bet_side")).upper()

        # Tracker freezes decimal_odds; accept bet_odds too if
        # the schema is later aligned with the official ledger.
        odds = number(
            lean.get("decimal_odds")
            or lean.get("bet_odds")
        )

        if side not in {"HOME", "DRAW", "AWAY"}:
            continue

        if odds is None or odds <= 1.0:
            continue

        won = int(side == actual)
        profit = odds - 1.0 if won else -1.0

        lean["actual_outcome"] = actual
        lean["won"] = str(won)
        lean["profit_units"] = str(profit)
        lean["home_score"] = str(int(float(result["home_score"])))
        lean["away_score"] = str(int(float(result["away_score"])))
        lean["status"] = "SETTLED"

        settled.append(lean)

    tmp = LEAN_FILE.with_suffix(".csv.tmp")

    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(leans)

    tmp.replace(LEAN_FILE)

    all_settled = [
        r for r in leans
        if clean(r.get("status")).upper() == "SETTLED"
    ]

    wins = sum(clean(r.get("won")) == "1" for r in all_settled)
    losses = len(all_settled) - wins
    units = sum(number(r.get("profit_units")) or 0.0 for r in all_settled)
    roi = units / len(all_settled) if all_settled else 0.0

    print("Completed result events:", len(by_event))
    print("Newly settled leans:", len(settled))
    print("Total settled leans:", len(all_settled))
    print("Open leans:", len(leans) - len(all_settled))
    print(f"Record: {wins}-{losses}")
    print(f"Net units: {units:+.2f}u")
    print(f"ROI: {roi:+.2%}")
    print("Ledger:", LEAN_FILE)


if __name__ == "__main__":
    main()
