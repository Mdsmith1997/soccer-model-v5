from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

OUT = (
    ROOT
    / "data"
    / "live"
    / "league_two_home_live_forward.csv"
)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

COLUMNS = [
    # identification
    "bet_id",
    "date",
    "kickoff",
    "league",
    "home_team",
    "away_team",

    # frozen strategy
    "strategy",
    "selection",
    "history_class",
    "threshold",

    # model at decision time
    "model_probability",
    "market_probability",
    "raw_edge",

    # wager
    "bet_odds",
    "stake_units",
    "sportsbook",
    "bet_timestamp",

    # closing-line tracking
    "closing_odds",
    "closing_timestamp",
    "closing_probability",
    "clv_probability",
    "clv_percent",

    # result
    "home_goals",
    "away_goals",
    "result",
    "win",
    "profit_units",

    # running performance
    "cumulative_profit",
    "cumulative_staked",
    "cumulative_roi",

    # audit
    "status",
    "notes",
]

if OUT.exists():

    df = pd.read_csv(OUT)

    missing = [
        c for c in COLUMNS
        if c not in df.columns
    ]

    for c in missing:
        df[c] = pd.NA

    df = df[COLUMNS]

    print("Existing ledger found.")
    print("Rows:", len(df))

else:

    df = pd.DataFrame(
        columns=COLUMNS
    )

    print("Creating new prospective ledger.")

df.to_csv(
    OUT,
    index=False,
)

print("\n" + "=" * 100)
print("LEAGUE TWO HOME — LIVE-FORWARD LEDGER")
print("=" * 100)

print("\nFROZEN RULE")
print("League: League Two")
print("Selection: HOME ML")
print("Minimum raw V5 edge: 16%")
print("Required history: BOTH_SAME_LEAGUE")
print("Neutral history: NOT ELIGIBLE")
print("Transferred history: NOT ELIGIBLE")
print("Odds filter: NONE")
print("Validation stake: 1.00 unit")

print("\nIMPORTANT")
print("Do not change the threshold based on forward results.")
print("Do not add an odds filter based on forward results.")
print("Do not selectively omit qualifying losses.")
print("Record every qualifying signal.")
print("Record the market price available when the signal appears.")
print("Record closing odds whenever possible.")

print("\nSaved:")
print(OUT)
