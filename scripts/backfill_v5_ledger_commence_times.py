from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LEDGER_FILE = (
    ROOT
    / "data"
    / "live"
    / "v5_live_bet_ledger.csv"
)

EVENT_HISTORY_FILE = (
    ROOT
    / "data"
    / "live"
    / "odds_events_history.csv"
)

H2H_HISTORY_FILE = (
    ROOT
    / "data"
    / "live"
    / "odds_h2h_history.csv"
)


def load_event_map():

    frames = []

    for path in [
        EVENT_HISTORY_FILE,
        H2H_HISTORY_FILE,
    ]:

        if not path.exists():
            continue

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        if (
            "event_id" not in df.columns
            or
            "commence_time" not in df.columns
        ):
            continue

        x = df[
            [
                "event_id",
                "commence_time",
            ]
        ].copy()

        x["event_id"] = (
            x["event_id"]
            .astype(str)
            .str.strip()
        )

        x["commence_time"] = pd.to_datetime(
            x["commence_time"],
            utc=True,
            errors="coerce",
        )

        x = x.dropna(
            subset=[
                "event_id",
                "commence_time",
            ]
        )

        frames.append(x)

    if not frames:

        raise RuntimeError(
            "Could not find event/odds history "
            "containing event_id + commence_time."
        )

    events = pd.concat(
        frames,
        ignore_index=True,
    )

    # Same event may appear in many snapshots.
    # Kickoff should be identical; keep the latest
    # valid observation for each event.
    events = (
        events
        .drop_duplicates(
            subset=["event_id"],
            keep="last",
        )
        .set_index("event_id")[
            "commence_time"
        ]
    )

    return events


def main():

    if not LEDGER_FILE.exists():

        raise RuntimeError(
            f"Ledger not found: {LEDGER_FILE}"
        )

    ledger = pd.read_csv(
        LEDGER_FILE,
        low_memory=False,
    )

    if "event_id" not in ledger.columns:

        raise RuntimeError(
            "Ledger has no event_id column."
        )

    ledger["event_id"] = (
        ledger["event_id"]
        .astype(str)
        .str.strip()
    )

    if "commence_time" not in ledger.columns:

        ledger["commence_time"] = pd.NaT

    current = pd.to_datetime(
        ledger["commence_time"],
        utc=True,
        errors="coerce",
    )

    event_map = load_event_map()

    mapped = (
        ledger["event_id"]
        .map(event_map)
    )

    missing_before = int(
        current.isna().sum()
    )

    ledger["commence_time"] = (
        current.fillna(mapped)
    )

    missing_after = int(
        pd.to_datetime(
            ledger["commence_time"],
            utc=True,
            errors="coerce",
        )
        .isna()
        .sum()
    )

    filled = (
        missing_before
        -
        missing_after
    )

    ledger.to_csv(
        LEDGER_FILE,
        index=False,
    )

    print()
    print("=" * 100)
    print("BACKFILL V5 LEDGER KICKOFF TIMES")
    print("=" * 100)
    print()

    print(
        f"Ledger bets:       {len(ledger)}"
    )

    print(
        f"Missing before:    {missing_before}"
    )

    print(
        f"Kickoffs filled:   {filled}"
    )

    print(
        f"Missing after:     {missing_after}"
    )

    print()

    status = (
        ledger["status"]
        .astype(str)
        .str.upper()
    )

    open_bets = ledger[
        status.eq("OPEN")
    ].copy()

    cols = [
        c
        for c in [
            "date",
            "league",
            "home_team",
            "away_team",
            "bet_side",
            "commence_time",
            "event_id",
        ]
        if c in open_bets.columns
    ]

    print("=" * 100)
    print("OPEN BETS")
    print("=" * 100)
    print()

    print(
        open_bets[
            cols
        ].to_string(
            index=False
        )
    )

    print()
    print(f"Saved: {LEDGER_FILE}")


if __name__ == "__main__":
    main()
