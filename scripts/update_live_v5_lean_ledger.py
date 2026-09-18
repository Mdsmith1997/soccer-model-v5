#!/usr/bin/env python3

import csv
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "soccer/scripts" in str(Path(__file__).resolve()) else Path.cwd() / "soccer"
LIVE = Path("/var/data/soccer")

EV_FILE = LIVE / "v5_live_ev_board.csv"
OFFICIAL_FILE = LIVE / "v5_live_bet_ledger.csv"
LEAN_FILE = LIVE / "v5_live_lean_ledger.csv"

LEAN_MIN = 0.02
OFFICIAL_MIN = 0.16

FIELDS = [
    "lean_id",
    "match_id",
    "event_id",
    "date",
    "league",
    "home_team",
    "away_team",
    "bet_side",
    "model_probability",
    "market_probability",
    "edge",
    "decimal_odds",
    "bookmaker",
    "odds_snapshot_time",
    "commence_time",
    "home_history_league",
    "away_history_league",
    "history_type",
    "first_seen_at",
    "promoted_to_official",
    "promoted_at",
    "status",
    "result",
    "won",
    "profit_units",
]


def clean(v):
    return str(v or "").strip()


def number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_dt(v):
    s = clean(v)
    if not s:
        return None

    try:
        dt = datetime.fromisoformat(
            s.replace("Z", "+00:00")
        )
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt


def history_type(row):
    home_source = clean(
        row.get("home_history_source")
    ).upper()

    away_source = clean(
        row.get("away_history_source")
    ).upper()

    transition_applied = number(
        row.get("transition_applied")
    )

    prediction_provider = clean(
        row.get("prediction_provider")
    ).upper()

    sources = (
        home_source
        + " "
        + away_source
    )

    # Match official ledger classification exactly.
    if (
        transition_applied is not None
        and transition_applied == 1
    ):
        return "TRANSITION"

    if "TRANSFER" in sources:
        return "TRANSFERRED"

    if (
        home_source == "SAME_LEAGUE"
        and away_source == "SAME_LEAGUE"
    ):
        return "SAME_LEAGUE"

    if prediction_provider == "CORE_V5":
        return "CORE"

    return "OTHER"

def load_csv(path):
    if not path.exists():
        return []

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    tmp = path.with_suffix(".tmp")

    with tmp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(tmp, path)


def main():
    if not EV_FILE.exists():
        raise SystemExit(
            f"Missing EV board: {EV_FILE}"
        )

    ev_rows = load_csv(EV_FILE)
    lean_rows = load_csv(LEAN_FILE)
    official_rows = load_csv(OFFICIAL_FILE)

    existing = {
        clean(r.get("lean_id"))
        for r in lean_rows
        if clean(r.get("lean_id"))
    }

    official = {}

    for r in official_rows:
        mid = clean(r.get("match_id"))
        side = clean(r.get("bet_side")).upper()

        if mid and side:
            official[(mid, side)] = r

    now = datetime.now(timezone.utc).isoformat()

    added = 0

    for r in ev_rows:
        match_id = clean(r.get("match_id"))

        if not match_id:
            continue

        snapshot = parse_dt(
            r.get("odds_snapshot_time")
        )
        commence = parse_dt(
            r.get("odds_commence_time")
            or r.get("commence_time")
        )

        # Match official pre-kickoff safety:
        # require a valid snapshot and kickoff and
        # refuse signals inside the final minute.
        if snapshot is None or commence is None:
            continue

        if snapshot.timestamp() >= (
            commence.timestamp() - 60
        ):
            continue

        # Mirror dashboard live window exactly:
        # only track matches kicking off in the next 72 hours.
        current_time = datetime.now(timezone.utc)
        if not (
            current_time <= commence
            <= current_time + timedelta(hours=72)
        ):
            continue

        for side in ("HOME", "AWAY"):
            if side == "HOME":
                model = number(r.get("p_home_v5"))
                market = number(r.get("market_p_home"))
                odds = number(r.get("best_home_odds"))
                book = clean(r.get("best_home_book"))
            else:
                model = number(r.get("p_away_v5"))
                market = number(r.get("market_p_away"))
                odds = number(r.get("best_away_odds"))
                book = clean(r.get("best_away_book"))

            if None in (model, market, odds):
                continue

            edge = model - market

            # Exact dashboard 1X2 lean range.
            if not (
                LEAN_MIN <= edge < OFFICIAL_MIN
            ):
                continue

            lean_id = f"{match_id}_{side}"

            # Freeze first qualifying appearance.
            if lean_id in existing:
                continue

            off = official.get(
                (match_id, side)
            )

            lean_rows.append({
                "lean_id": lean_id,
                "match_id": match_id,
                "event_id": clean(
                    r.get("event_id")
                ),
                "date": clean(r.get("date")),
                "league": clean(r.get("league")),
                "home_team": clean(
                    r.get("home_team")
                ),
                "away_team": clean(
                    r.get("away_team")
                ),
                "bet_side": side,
                "model_probability": model,
                "market_probability": market,
                "edge": edge,
                "decimal_odds": odds,
                "bookmaker": book,
                "odds_snapshot_time": clean(
                    r.get("odds_snapshot_time")
                ),
                "commence_time": clean(
                    r.get("odds_commence_time")
                    or r.get("commence_time")
                ),
                "home_history_league": clean(
                    r.get("home_history_league")
                ),
                "away_history_league": clean(
                    r.get("away_history_league")
                ),
                "history_type": history_type(r),
                "first_seen_at": now,
                "promoted_to_official":
                    "1" if off else "0",
                "promoted_at":
                    clean(off.get("first_seen_at"))
                    if off else "",
                "status": "OPEN",
                "result": "",
                "won": "",
                "profit_units": "",
            })

            existing.add(lean_id)
            added += 1

    # Update promotion status on previously frozen leans.
    promoted_now = 0

    for r in lean_rows:
        if clean(
            r.get("promoted_to_official")
        ) == "1":
            continue

        key = (
            clean(r.get("match_id")),
            clean(r.get("bet_side")).upper(),
        )

        off = official.get(key)

        if off:
            r["promoted_to_official"] = "1"
            r["promoted_at"] = (
                clean(off.get("first_seen_at"))
                or now
            )
            promoted_now += 1

    save_csv(
        LEAN_FILE,
        lean_rows,
    )

    print("=" * 90)
    print("V5 1X2 RESEARCH LEAN LEDGER")
    print("=" * 90)
    print(
        "Rule: HOME/AWAY, "
        "2% <= raw edge < 16%"
    )
    print("Research only: YES")
    print("Official betting rules changed: NO")
    print()
    print("Existing rows:", len(lean_rows) - added)
    print("New frozen leans:", added)
    print("New promotions:", promoted_now)
    print("Total lean rows:", len(lean_rows))
    print("Ledger:", LEAN_FILE)


if __name__ == "__main__":
    main()
