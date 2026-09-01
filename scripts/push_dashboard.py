from pathlib import Path
from datetime import datetime, timezone
import os
import re
import subprocess
import sys

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"

URL = os.environ.get("V5_DASHBOARD_URL")
TOKEN = os.environ.get("V5_DASHBOARD_TOKEN")

LEDGER_FILE = LIVE / "v5_live_bet_ledger.csv"
BTTS_FILE = LIVE / "btts_live_predictions.csv"
TOTALS_FILE = LIVE / "v5_live_totals_ev_board.csv"

BTTS_SPECIALISTS = {
    "Swiss Super League": 0.06,
    "Super Lig": 0.10,
    "Segunda División": 0.04,
}

TOTALS_SPECIALISTS = {
    ("Premier League", "UNDER"): 0.11,
    ("Bundesliga", "UNDER"): 0.11,
    ("Belgian Pro League", "OVER"): 0.15,
    ("Eliteserien", "OVER"): 0.11,
}


def decimal_to_american(value):
    try:
        d = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(d) or d <= 1:
        return None

    if d >= 2:
        return int(round((d - 1) * 100))

    return int(round(-100 / (d - 1)))


def number(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(x):
        return None

    return x


def pct(value):
    x = number(value)

    if x is None:
        return None

    if abs(x) <= 1:
        x *= 100

    return round(x, 2)


def iso_time(value):
    ts = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(ts):
        return None

    return ts.isoformat()


def live_window():
    now = pd.Timestamp.now(tz="America/Detroit")
    end = now + pd.Timedelta(hours=72)
    return now, end


def build_official_bets():
    official = []
    window_start, window_end = live_window()

    # Global 1X2
    if LEDGER_FILE.exists():
        x = pd.read_csv(
            LEDGER_FILE,
            low_memory=False,
        )

        if "commence_time" in x.columns:
            x["_kickoff"] = pd.to_datetime(
                x["commence_time"],
                errors="coerce",
                utc=True,
            ).dt.tz_convert(
                "America/Detroit"
            )

            mask = (
                x["_kickoff"].ge(window_start)
                &
                x["_kickoff"].le(window_end)
            )

            if "status" in x.columns:
                mask &= (
                    ~x["status"]
                    .astype(str)
                    .str.upper()
                    .eq("INVALID")
                )

            x = x[mask].copy()

            for row in x.itertuples(index=False):
                edge = number(
                    getattr(row, "signal_edge", None)
                )

                threshold = 0.16

                official.append({
                    "id": str(
                        getattr(
                            row,
                            "ledger_id",
                            getattr(row, "match_id", ""),
                        )
                    ),
                    "market": "1X2",
                    "league": str(
                        getattr(row, "league", "")
                    ),
                    "home_team": str(
                        getattr(row, "home_team", "")
                    ),
                    "away_team": str(
                        getattr(row, "away_team", "")
                    ),
                    "selection": str(
                        getattr(row, "bet_side", "")
                    ).upper(),
                    "american_odds": decimal_to_american(
                        getattr(row, "bet_odds", None)
                    ),
                    "bookmaker": str(
                        getattr(row, "bet_book", "")
                    ),
                    "commence_time": iso_time(
                        getattr(row, "commence_time", None)
                    ),
                    "edge_pct": pct(edge),
                    "threshold_pct": pct(threshold),
                    "difference_pp": (
                        round(
                            (edge - threshold) * 100,
                            2,
                        )
                        if edge is not None
                        else None
                    ),
                })

    # BTTS YES specialists
    if BTTS_FILE.exists():
        b = pd.read_csv(
            BTTS_FILE,
            low_memory=False,
        )

        if (
            len(b)
            and
            "commence_time" in b.columns
        ):
            b["_kickoff"] = pd.to_datetime(
                b["commence_time"],
                errors="coerce",
                utc=True,
            ).dt.tz_convert(
                "America/Detroit"
            )

            b = b[
                b["_kickoff"].ge(window_start)
                &
                b["_kickoff"].le(window_end)
            ].copy()

            if len(b):
                b["yes_edge"] = pd.to_numeric(
                    b.get("yes_edge"),
                    errors="coerce",
                )

                b["yes_odds"] = pd.to_numeric(
                    b.get("yes_odds"),
                    errors="coerce",
                )

                for league, threshold in BTTS_SPECIALISTS.items():
                    q = b[
                        b["league_market"]
                        .astype(str)
                        .eq(league)
                        &
                        b["yes_edge"]
                        .ge(threshold)
                    ].copy()

                    if not len(q):
                        continue

                    q = (
                        q.sort_values(
                            ["match_id", "yes_odds"],
                            ascending=[True, False],
                        )
                        .drop_duplicates(
                            subset=["match_id"],
                            keep="first",
                        )
                    )

                    for row in q.itertuples(index=False):
                        edge = number(row.yes_edge)

                        official.append({
                            "id": (
                                str(row.match_id)
                                + "_BTTS_YES"
                            ),
                            "market": "BTTS",
                            "league": str(
                                row.league_market
                            ),
                            "home_team": str(
                                row.home_team_market
                            ),
                            "away_team": str(
                                row.away_team_market
                            ),
                            "selection": "YES",
                            "american_odds":
                                decimal_to_american(
                                    row.yes_odds
                                ),
                            "bookmaker": str(
                                row.bookmaker
                            ),
                            "commence_time":
                                iso_time(
                                    row.commence_time
                                ),
                            "edge_pct": pct(edge),
                            "threshold_pct":
                                pct(threshold),
                            "difference_pp": round(
                                (edge - threshold) * 100,
                                2,
                            ),
                        })

    # O/U 2.5 specialists
    if TOTALS_FILE.exists():
        t = pd.read_csv(
            TOTALS_FILE,
            low_memory=False,
        )

        if len(t):
            if "commence_time" in t.columns:
                t["_kickoff"] = pd.to_datetime(
                    t["commence_time"],
                    errors="coerce",
                    utc=True,
                ).dt.tz_convert(
                    "America/Detroit"
                )

                t = t[
                    t["_kickoff"].ge(window_start)
                    &
                    t["_kickoff"].le(window_end)
                ].copy()

            elif "date" in t.columns:
                t["_date"] = pd.to_datetime(
                    t["date"],
                    errors="coerce",
                ).dt.date

                allowed_dates = {
                    window_start.date(),
                    (
                        window_start
                        + pd.Timedelta(days=1)
                    ).date(),
                    (
                        window_start
                        + pd.Timedelta(days=2)
                    ).date(),
                    (
                        window_start
                        + pd.Timedelta(days=3)
                    ).date(),
                }

                t = t[
                    t["_date"].isin(allowed_dates)
                ].copy()

        if len(t):
            t["edge"] = pd.to_numeric(
                t.get("edge"),
                errors="coerce",
            )

            t["decimal_odds"] = pd.to_numeric(
                t.get("decimal_odds"),
                errors="coerce",
            )

            bet_text = (
                t["bet"]
                .astype(str)
                .str.upper()
            )

            for (
                league,
                side,
            ), threshold in TOTALS_SPECIALISTS.items():

                q = t[
                    t["league"]
                    .astype(str)
                    .eq(league)
                    &
                    bet_text.str.contains(
                        side,
                        regex=False,
                    )
                    &
                    t["edge"].ge(threshold)
                ].copy()

                if not len(q):
                    continue

                q = (
                    q.sort_values(
                        ["match_id", "decimal_odds"],
                        ascending=[True, False],
                    )
                    .drop_duplicates(
                        subset=["match_id"],
                        keep="first",
                    )
                )

                for row in q.itertuples(index=False):
                    edge = number(row.edge)

                    bet = str(row.bet)

                    selection = (
                        "OVER"
                        if "OVER" in bet.upper()
                        else "UNDER"
                    )

                    commence_time = None

                    if hasattr(row, "commence_time"):
                        commence_time = iso_time(
                            row.commence_time
                        )

                    official.append({
                        "id": (
                            str(row.match_id)
                            + "_TOTALS_"
                            + selection
                        ),
                        "market": "O/U 2.5",
                        "league": str(row.league),
                        "home_team": str(row.home_team),
                        "away_team": str(row.away_team),
                        "selection": selection,
                        "american_odds":
                            decimal_to_american(
                                row.decimal_odds
                            ),
                        "bookmaker": str(
                            row.bookmaker
                        ),
                        "commence_time":
                            commence_time,
                        "date": str(
                            getattr(row, "date", "")
                        ),
                        "edge_pct": pct(edge),
                        "threshold_pct":
                            pct(threshold),
                        "difference_pp": round(
                            (edge - threshold) * 100,
                            2,
                        ),
                    })

    return official


def main():
    if not URL:
        raise RuntimeError(
            "V5_DASHBOARD_URL is not configured."
        )

    if not TOKEN:
        raise RuntimeError(
            "V5_DASHBOARD_TOKEN is not configured."
        )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "show_board.py"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    board = (
        (result.stdout or "")
        +
        (result.stderr or "")
    )

    print(board)

    if result.returncode != 0:
        raise RuntimeError(
            "show_board.py failed with "
            f"exit code {result.returncode}"
        )

    official_bets = build_official_bets()

    # No authoritative production lean definition yet.
    leans = []

    manual_count = len(
        re.findall(
            r"MANUAL .*?PRICE CHECK",
            board,
            flags=re.IGNORECASE,
        )
    )

    captured_count = len(
        re.findall(
            r"PRICE CAPTURED",
            board,
            flags=re.IGNORECASE,
        )
    )

    unavailable_count = 0
    in_1x2 = False

    for line in board.splitlines():
        if "1X2 PRICE COVERAGE" in line:
            in_1x2 = True
            continue

        if "BTTS SPECIALIST PRICE COVERAGE" in line:
            in_1x2 = False

        if not in_1x2:
            continue

        if "MODEL UNAVAILABLE" not in line:
            continue

        if (
            "Denmark Superliga" in line
            or
            "Liga MX" in line
        ):
            continue

        unavailable_count += 1

    payload = {
        "token": TOKEN,
        "dashboard": {
            "pushed_at":
                datetime.now(timezone.utc).isoformat(),
            "window_hours": 72,
            "official_bets": official_bets,
            "leans": leans,
            "official_count": len(official_bets),
            "manual_count": manual_count,
            "captured_count": captured_count,
            "unavailable_count": unavailable_count,
            "raw_board": board,
        },
    }

    print()
    print("=" * 70)
    print("PUSHING PHONE DASHBOARD")
    print("=" * 70)
    print("Official plays:", len(official_bets))
    print("Leans:", len(leans))
    print("Manual checks:", manual_count)
    print("Prices captured:", captured_count)
    print("Model unavailable:", unavailable_count)

    for bet in official_bets:
        print(
            "OFFICIAL:",
            bet["league"],
            "|",
            bet["home_team"],
            "vs",
            bet["away_team"],
            "|",
            bet["market"],
            bet["selection"],
            "|",
            bet["american_odds"],
            "| edge",
            bet["edge_pct"],
        )

    response = requests.post(
        URL,
        json=payload,
        timeout=30,
    )

    print("HTTP:", response.status_code)
    print("Response:", response.text)

    response.raise_for_status()

    print("✅ PHONE DASHBOARD UPDATED")


if __name__ == "__main__":
    main()
