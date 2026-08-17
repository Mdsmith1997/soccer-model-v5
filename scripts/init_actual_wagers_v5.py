from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OUT = (
    ROOT
    / "data"
    / "live"
    / "v5_actual_wagers.csv"
)

STARTING_BANKROLL = 60.00


wagers = [
    {
        "date": "2026-08-15",
        "wager_type": "STRAIGHT",
        "selection": "York City ML",
        "american_odds": 130,
        "stake": 20.00,
        "result": "WIN",
        "return": 46.00,
        "profit": 26.00,
        "model_generated": True,
        "notes": "",
    },
    {
        "date": "2026-08-15",
        "wager_type": "STRAIGHT",
        "selection": "Sheffield Wednesday ML",
        "american_odds": 155,
        "stake": 20.00,
        "result": "WIN",
        "return": 51.00,
        "profit": 31.00,
        "model_generated": True,
        "notes": "",
    },
    {
        "date": "2026-08-15",
        "wager_type": "STRAIGHT",
        "selection": "Newport County ML",
        "american_odds": 215,
        "stake": 10.00,
        "result": "WIN",
        "return": 31.50,
        "profit": 21.50,
        "model_generated": True,
        "notes": "",
    },
    {
        "date": "2026-08-15",
        "wager_type": "PARLAY",
        "selection": (
            "York City ML + "
            "Sheffield Wednesday ML + "
            "Newport County ML"
        ),
        "american_odds": 1602,
        "stake": 10.00,
        "result": "WIN",
        "return": 170.25,
        "profit": 160.25,
        "model_generated": True,
        "notes": "Actual sportsbook payout recorded",
    },
]


def main():

    if OUT.exists():
        raise RuntimeError(
            f"{OUT} already exists. "
            "Refusing to overwrite actual wager history."
        )

    df = pd.DataFrame(wagers)

    df["starting_bankroll"] = STARTING_BANKROLL

    cumulative_profit = (
        df["profit"]
        .cumsum()
    )

    df["bankroll_after"] = (
        STARTING_BANKROLL
        +
        cumulative_profit
    )

    df.to_csv(
        OUT,
        index=False,
    )

    total_staked = df["stake"].sum()
    total_return = df["return"].sum()
    total_profit = df["profit"].sum()

    ending_bankroll = (
        STARTING_BANKROLL
        +
        total_profit
    )

    roi = (
        total_profit
        /
        total_staked
    )

    print()
    print("=" * 90)
    print("V5 ACTUAL WAGERING LEDGER")
    print("=" * 90)
    print()

    print(
        df[
            [
                "date",
                "wager_type",
                "selection",
                "american_odds",
                "stake",
                "result",
                "return",
                "profit",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 90)
    print("BANKROLL")
    print("=" * 90)
    print()

    print(
        f"Starting bankroll: ${STARTING_BANKROLL:.2f}"
    )

    print(
        f"Total staked:      ${total_staked:.2f}"
    )

    print(
        f"Total returned:    ${total_return:.2f}"
    )

    print(
        f"Net profit:        ${total_profit:+.2f}"
    )

    print(
        f"Actual ROI:        {roi:+.2%}"
    )

    print(
        f"Current bankroll:  ${ending_bankroll:.2f}"
    )

    print()
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
