from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LEDGER_FILE = ROOT / "data" / "live" / "v5_live_bet_ledger.csv"
HISTORY_FILE = ROOT / "data" / "live" / "odds_h2h_history.csv"
OUTPUT_FILE = ROOT / "data" / "live" / "v5_live_clv_report.csv"

# A snapshot must be within this many minutes of kickoff
# before we are willing to call it a closing-line observation.
MAX_CLOSE_MINUTES = 60


def decimal_to_american(decimal_odds):
    try:
        d = float(decimal_odds)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(d) or d <= 1:
        return np.nan

    if d >= 2:
        return round((d - 1) * 100)

    return round(-100 / (d - 1))


def side_column(side):
    side = str(side).strip().upper()

    return {
        "HOME": "home_odds",
        "DRAW": "draw_odds",
        "AWAY": "away_odds",
    }.get(side)


def main():

    print()
    print("=" * 110)
    print("LIVE V5 CLV REPORT")
    print("=" * 110)

    ledger = pd.read_csv(
        LEDGER_FILE,
        low_memory=False,
    )

    # --------------------------------------------------------
    # EXCLUDE INVALID / AUDIT-ONLY SIGNALS
    #
    # Invalid signals remain permanently in the master ledger
    # for auditability, but they must never participate in CLV
    # analysis or model evaluation.
    # --------------------------------------------------------

    if "status" in ledger.columns:

        invalid_mask = (
            ledger["status"]
            .astype(str)
            .str.strip()
            .str.upper()
            .eq("INVALID")
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        ledger = ledger[
            ~invalid_mask
        ].copy()

        if invalid_count:

            print(
                f"Excluded invalid CLV signals: "
                f"{invalid_count}"
            )

    history = pd.read_csv(
        HISTORY_FILE,
        low_memory=False,
    )

    history["snapshot_time"] = pd.to_datetime(
        history["snapshot_time"],
        utc=True,
        errors="coerce",
    )

    history["commence_time"] = pd.to_datetime(
        history["commence_time"],
        utc=True,
        errors="coerce",
    )

    rows = []

    for _, bet in ledger.iterrows():

        event_id = str(
            bet.get("event_id", "")
        ).strip()

        side = str(
            bet.get("bet_side", "")
        ).strip().upper()

        odds_col = side_column(side)

        if not event_id or odds_col is None:
            continue

        x = history[
            history["event_id"]
            .astype(str)
            .eq(event_id)
        ].copy()

        if x.empty:
            continue

        x[odds_col] = pd.to_numeric(
            x[odds_col],
            errors="coerce",
        )

        x = x[
            x[odds_col].notna()
            & x["snapshot_time"].notna()
            & x["commence_time"].notna()
            & (
                x["snapshot_time"]
                <
                x["commence_time"]
            )
        ].copy()

        if x.empty:
            continue

        # ----------------------------------------------------
        # LAST OBSERVED MARKET SNAPSHOT
        # ----------------------------------------------------

        last_snapshot = x[
            "snapshot_time"
        ].max()

        latest = x[
            x["snapshot_time"]
            ==
            last_snapshot
        ].copy()

        kickoff = latest[
            "commence_time"
        ].iloc[0]

        minutes_to_kickoff = (
            kickoff
            -
            last_snapshot
        ).total_seconds() / 60

        latest_prices = (
            latest[odds_col]
            .dropna()
            .astype(float)
        )

        if latest_prices.empty:
            continue

        best_last_decimal = float(
            latest_prices.max()
        )

        median_last_decimal = float(
            latest_prices.median()
        )

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        signal_decimal = pd.to_numeric(
            pd.Series(
                [bet.get("bet_odds")]
            ),
            errors="coerce",
        ).iloc[0]

        if pd.isna(signal_decimal):
            continue

        signal_decimal = float(
            signal_decimal
        )

        # ----------------------------------------------------
        # TRUE-CLOSE ELIGIBILITY
        # ----------------------------------------------------

        is_true_close = (
            minutes_to_kickoff
            <=
            MAX_CLOSE_MINUTES
        )

        # Probability-space CLV against consensus.
        signal_implied = (
            1.0
            /
            signal_decimal
        )

        last_implied = (
            1.0
            /
            median_last_decimal
        )

        probability_clv = (
            last_implied
            -
            signal_implied
        )

        rows.append(
            {
                "date":
                    bet.get("date"),

                "league":
                    bet.get("league"),

                "home_team":
                    bet.get("home_team"),

                "away_team":
                    bet.get("away_team"),

                "bet_side":
                    side,

                "result":
                    bet.get("actual_outcome"),

                "won":
                    bet.get("won"),

                "signal_decimal":
                    signal_decimal,

                "signal_american":
                    decimal_to_american(
                        signal_decimal
                    ),

                "last_snapshot_time":
                    last_snapshot,

                "minutes_before_kickoff":
                    minutes_to_kickoff,

                "last_consensus_decimal":
                    median_last_decimal,

                "last_consensus_american":
                    decimal_to_american(
                        median_last_decimal
                    ),

                "last_best_decimal":
                    best_last_decimal,

                "last_best_american":
                    decimal_to_american(
                        best_last_decimal
                    ),

                "probability_clv":
                    probability_clv,

                "probability_clv_pct":
                    probability_clv * 100,

                "beat_last_consensus":
                    signal_decimal
                    >
                    median_last_decimal,

                "beat_last_best":
                    signal_decimal
                    >
                    best_last_decimal,

                "true_close_available":
                    is_true_close,
            }
        )

    report = pd.DataFrame(
        rows
    )

    report.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    if report.empty:

        print()
        print("No CLV observations available.")
        return

    show = report.copy()

    show["SIGNAL"] = (
        show["signal_american"]
        .apply(
            lambda x:
            f"{int(x):+d}"
            if pd.notna(x)
            else "-"
        )
    )

    show["LAST_CONS"] = (
        show["last_consensus_american"]
        .apply(
            lambda x:
            f"{int(x):+d}"
            if pd.notna(x)
            else "-"
        )
    )

    show["LAST_BEST"] = (
        show["last_best_american"]
        .apply(
            lambda x:
            f"{int(x):+d}"
            if pd.notna(x)
            else "-"
        )
    )

    show["CLV"] = (
        show["probability_clv_pct"]
        .map(
            lambda x:
            f"{x:+.2f}%"
        )
    )

    show["OBSERVED"] = (
        show["minutes_before_kickoff"]
        .map(
            lambda x:
            f"T-{x / 60:.1f}h"
        )
    )

    show["CLOSE?"] = np.where(
        show["true_close_available"],
        "YES",
        "NO",
    )

    print()
    print(
        show[
            [
                "league",
                "home_team",
                "away_team",
                "bet_side",
                "SIGNAL",
                "LAST_CONS",
                "LAST_BEST",
                "CLV",
                "OBSERVED",
                "CLOSE?",
            ]
        ]
        .to_string(index=False)
    )

    print()
    print("=" * 110)
    print("IMPORTANT")
    print("=" * 110)
    print()
    print(
        "SIGNAL = frozen model bet price."
    )
    print(
        "LAST_CONS = median sportsbook price at the final captured snapshot."
    )
    print(
        "LAST_BEST = best sportsbook price at the final captured snapshot."
    )
    print(
        f"CLOSE? = YES only when the final snapshot was within "
        f"{MAX_CLOSE_MINUTES} minutes of kickoff."
    )
    print()
    print(
        "A positive probability CLV means the market later "
        "assigned MORE probability to our selection than it did "
        "when we bet it."
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
