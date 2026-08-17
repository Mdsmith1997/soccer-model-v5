from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    ROOT
    / "data"
    / "live"
    / "v5_live_predictions.csv"
)

ODDS_FILE = (
    ROOT
    / "data"
    / "live"
    / "odds_snapshot.csv"
)

OUTPUT_CURRENT = (
    ROOT
    / "data"
    / "live"
    / "paper_live_current.csv"
)

OUTPUT_HISTORY = (
    ROOT
    / "data"
    / "live"
    / "paper_live_history.csv"
)


# ============================================================
# SETTINGS
#
# These are monitoring labels, NOT optimized betting rules.
# ============================================================

TRACK_EDGE = 0.025
STRONG_WATCH_EDGE = 0.10
EXTREME_WATCH_EDGE = 0.15

MIN_EV = 0.00

EPS = 1e-12


# ============================================================
# HELPERS
# ============================================================

def normalize_three_way(
    home,
    draw,
    away,
):
    probs = np.column_stack(
        [
            home,
            draw,
            away,
        ]
    ).astype(float)

    probs = np.clip(
        probs,
        EPS,
        None,
    )

    probs /= probs.sum(
        axis=1,
        keepdims=True,
    )

    return probs


def utc_now():
    return datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    )


# ============================================================
# LOAD V5 LIVE PREDICTIONS
# ============================================================

def load_predictions():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"\nMissing live prediction file:\n"
            f"{PREDICTIONS_FILE}\n\n"
            "Expected columns:\n"
            "match_id,date,league,home_team,away_team,"
            "p_home_v5,p_draw_v5,p_away_v5\n"
        )

    df = pd.read_csv(
        PREDICTIONS_FILE,
        parse_dates=[
            "date",
        ],
    )

    required = [
        "match_id",
        "date",
        "league",
        "home_team",
        "away_team",
        "p_home_v5",
        "p_draw_v5",
        "p_away_v5",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Live predictions missing columns: "
            + str(missing)
        )

    if df["match_id"].duplicated().any():
        raise ValueError(
            "Duplicate match_id values found "
            "in live predictions."
        )

    probability_cols = [
        "p_home_v5",
        "p_draw_v5",
        "p_away_v5",
    ]

    for col in probability_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    valid = (
        df[probability_cols]
        .notna()
        .all(axis=1)
    )

    df = df.loc[
        valid
    ].copy()

    if len(df) == 0:
        raise ValueError(
            "No valid live V5 prediction rows found."
        )

    probs = normalize_three_way(
        df["p_home_v5"].to_numpy(dtype=float),
        df["p_draw_v5"].to_numpy(dtype=float),
        df["p_away_v5"].to_numpy(dtype=float),
    )

    df["p_home_v5"] = probs[:, 0]
    df["p_draw_v5"] = probs[:, 1]
    df["p_away_v5"] = probs[:, 2]

    return df


# ============================================================
# LOAD ODDS SNAPSHOT
#
# Expected decimal odds.
# ============================================================

def load_odds():
    if not ODDS_FILE.exists():
        raise FileNotFoundError(
            f"\nMissing live odds snapshot:\n"
            f"{ODDS_FILE}\n\n"
            "Expected columns:\n"
            "match_id,bookmaker,home_odds,draw_odds,away_odds\n\n"
            "Optional:\n"
            "snapshot_time\n"
        )

    df = pd.read_csv(
        ODDS_FILE
    )

    required = [
        "match_id",
        "bookmaker",
        "home_odds",
        "draw_odds",
        "away_odds",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Odds snapshot missing columns: "
            + str(missing)
        )

    if "snapshot_time" not in df.columns:
        df["snapshot_time"] = utc_now()
    else:
        df["snapshot_time"] = pd.to_datetime(
            df["snapshot_time"],
            utc=True,
            errors="coerce",
        )

        missing_time = df["snapshot_time"].isna()

        if missing_time.any():
            df.loc[
                missing_time,
                "snapshot_time",
            ] = utc_now()

    for col in [
        "home_odds",
        "draw_odds",
        "away_odds",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    valid = (
        df["home_odds"].notna()
        & df["draw_odds"].notna()
        & df["away_odds"].notna()
        & np.isfinite(df["home_odds"])
        & np.isfinite(df["draw_odds"])
        & np.isfinite(df["away_odds"])
        & (df["home_odds"] > 1.0)
        & (df["draw_odds"] > 1.0)
        & (df["away_odds"] > 1.0)
    )

    df = df.loc[
        valid
    ].copy()

    if len(df) == 0:
        raise ValueError(
            "No valid decimal odds rows found."
        )

    return df


# ============================================================
# CHOOSE BEST AVAILABLE PRICE
#
# For each match and outcome, take the highest price
# available in the supplied snapshot.
# ============================================================

def build_best_prices(
    odds,
):
    rows = []

    for match_id, sub in odds.groupby(
        "match_id",
        sort=False,
    ):
        home_row = sub.loc[
            sub["home_odds"].idxmax()
        ]

        draw_row = sub.loc[
            sub["draw_odds"].idxmax()
        ]

        away_row = sub.loc[
            sub["away_odds"].idxmax()
        ]

        latest_snapshot = sub[
            "snapshot_time"
        ].max()

        rows.append(
            {
                "match_id": match_id,
                "snapshot_time": latest_snapshot,
                "home_odds": float(home_row["home_odds"]),
                "home_bookmaker": home_row["bookmaker"],
                "draw_odds": float(draw_row["draw_odds"]),
                "draw_bookmaker": draw_row["bookmaker"],
                "away_odds": float(away_row["away_odds"]),
                "away_bookmaker": away_row["bookmaker"],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# JOIN PREDICTIONS + ODDS
# ============================================================

def build_match_table(
    predictions,
    prices,
):
    df = predictions.merge(
        prices,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    if len(df) == 0:
        raise ValueError(
            "No match_id overlap between "
            "live V5 predictions and odds."
        )

    # ========================================================
    # RAW IMPLIED MARKET PROBABILITIES
    # ========================================================

    raw_home = 1.0 / df["home_odds"]
    raw_draw = 1.0 / df["draw_odds"]
    raw_away = 1.0 / df["away_odds"]

    df["market_margin"] = (
        raw_home
        + raw_draw
        + raw_away
        - 1.0
    )

    market_probs = normalize_three_way(
        raw_home,
        raw_draw,
        raw_away,
    )

    df["market_p_home"] = market_probs[:, 0]
    df["market_p_draw"] = market_probs[:, 1]
    df["market_p_away"] = market_probs[:, 2]

    return df


# ============================================================
# KELLY
#
# INFORMATIONAL ONLY.
#
# Kelly fraction:
# f = (bp - q) / b
#
# where:
# b = decimal odds - 1
# p = model probability
# q = 1 - p
# ============================================================

def kelly_fraction(
    probability,
    decimal_odds,
):
    b = decimal_odds - 1.0
    q = 1.0 - probability

    numerator = (
        b
        * probability
        - q
    )

    fraction = (
        numerator
        / b
    )

    return np.clip(
        fraction,
        0.0,
        1.0,
    )


# ============================================================
# SIGNAL CLASSIFICATION
#
# Monitoring labels only.
# ============================================================

def classify_signal(
    edge,
    ev,
):
    if (
        edge >= EXTREME_WATCH_EDGE
        and ev > MIN_EV
    ):
        return "EXTREME_WATCH"

    if (
        edge >= STRONG_WATCH_EDGE
        and ev > MIN_EV
    ):
        return "STRONG_WATCH"

    if (
        edge >= TRACK_EDGE
        and ev > MIN_EV
    ):
        return "TRACK"

    return "NONE"


# ============================================================
# BUILD LONG SIGNAL TABLE
# ============================================================

def build_signals(
    matches,
):
    rows = []

    definitions = [
        (
            "HOME",
            "p_home_v5",
            "market_p_home",
            "home_odds",
            "home_bookmaker",
        ),
        (
            "DRAW",
            "p_draw_v5",
            "market_p_draw",
            "draw_odds",
            "draw_bookmaker",
        ),
        (
            "AWAY",
            "p_away_v5",
            "market_p_away",
            "away_odds",
            "away_bookmaker",
        ),
    ]

    base_cols = [
        "match_id",
        "date",
        "league",
        "home_team",
        "away_team",
        "snapshot_time",
        "market_margin",
    ]

    for (
        side,
        model_col,
        market_col,
        odds_col,
        bookmaker_col,
    ) in definitions:
        sub = matches[
            base_cols
            + [
                model_col,
                market_col,
                odds_col,
                bookmaker_col,
            ]
        ].copy()

        sub = sub.rename(
            columns={
                model_col: "model_probability",
                market_col: "market_probability",
                odds_col: "decimal_odds",
                bookmaker_col: "bookmaker",
            }
        )

        sub["side"] = side

        # ====================================================
        # MODEL-MARKET EDGE
        # ====================================================

        sub["probability_edge"] = (
            sub["model_probability"]
            - sub["market_probability"]
        )

        # ====================================================
        # EV AT AVAILABLE PRICE
        # ====================================================

        sub["expected_value"] = (
            sub["model_probability"]
            * sub["decimal_odds"]
            - 1.0
        )

        # ====================================================
        # INFORMATIONAL KELLY
        # ====================================================

        sub["full_kelly"] = kelly_fraction(
            sub["model_probability"],
            sub["decimal_odds"],
        )

        sub["half_kelly"] = (
            0.50
            * sub["full_kelly"]
        )

        sub["quarter_kelly"] = (
            0.25
            * sub["full_kelly"]
        )

        # ====================================================
        # MONITORING LABEL
        # ====================================================

        sub["signal"] = [
            classify_signal(
                edge,
                ev,
            )
            for edge, ev in zip(
                sub["probability_edge"],
                sub["expected_value"],
            )
        ]

        rows.append(
            sub
        )

    signals = pd.concat(
        rows,
        ignore_index=True,
    )

    return signals


# ============================================================
# SAVE CURRENT SNAPSHOT
# ============================================================

def save_current(
    signals,
):
    signals = (
        signals
        .sort_values(
            [
                "date",
                "match_id",
                "probability_edge",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    signals.to_csv(
        OUTPUT_CURRENT,
        index=False,
    )


# ============================================================
# APPEND HISTORY
#
# Every run preserves the market snapshot.
# ============================================================

def append_history(
    signals,
):
    history_cols = [
        "snapshot_time",
        "match_id",
        "date",
        "league",
        "home_team",
        "away_team",
        "side",
        "bookmaker",
        "decimal_odds",
        "model_probability",
        "market_probability",
        "market_margin",
        "probability_edge",
        "expected_value",
        "full_kelly",
        "half_kelly",
        "quarter_kelly",
        "signal",
    ]

    new = signals[
        history_cols
    ].copy()

    new["snapshot_time"] = pd.to_datetime(
        new["snapshot_time"],
        utc=True,
    )

    if OUTPUT_HISTORY.exists():
        old = pd.read_csv(
            OUTPUT_HISTORY
        )

        old["snapshot_time"] = pd.to_datetime(
            old["snapshot_time"],
            utc=True,
            errors="coerce",
        )

        history = pd.concat(
            [
                old,
                new,
            ],
            ignore_index=True,
        )
    else:
        history = new.copy()

    # ========================================================
    # REMOVE TRUE DUPLICATES
    #
    # Same match / side / bookmaker / odds / snapshot.
    # ========================================================

    history = (
        history
        .drop_duplicates(
            subset=[
                "snapshot_time",
                "match_id",
                "side",
                "bookmaker",
                "decimal_odds",
            ],
            keep="last",
        )
        .sort_values(
            [
                "snapshot_time",
                "match_id",
                "side",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    history.to_csv(
        OUTPUT_HISTORY,
        index=False,
    )

    return history


# ============================================================
# DISPLAY
# ============================================================

def print_signals(
    signals,
):
    print()
    print("=" * 125)
    print("CURRENT PAPER-LIVE SIGNALS")
    print("=" * 125)

    active = signals[
        signals["signal"] != "NONE"
    ].copy()

    if len(active) == 0:
        print(
            "No V5 signals currently meet "
            "the monitoring threshold."
        )
        return

    display = active.copy()

    for col in [
        "model_probability",
        "market_probability",
        "probability_edge",
        "expected_value",
        "full_kelly",
        "quarter_kelly",
    ]:
        display[col] *= 100.0

    print(
        display[
            [
                "date",
                "league",
                "home_team",
                "away_team",
                "side",
                "signal",
                "bookmaker",
                "decimal_odds",
                "model_probability",
                "market_probability",
                "probability_edge",
                "expected_value",
                "full_kelly",
                "quarter_kelly",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )


# ============================================================
# MATCH SUMMARY
# ============================================================

def print_match_summary(
    signals,
):
    print()
    print("=" * 105)
    print("MATCH SUMMARY")
    print("=" * 105)

    rows = []

    for match_id, sub in signals.groupby(
        "match_id",
        sort=False,
    ):
        best = (
            sub
            .sort_values(
                "probability_edge",
                ascending=False,
            )
            .iloc[0]
        )

        rows.append(
            {
                "date": best["date"],
                "league": best["league"],
                "home_team": best["home_team"],
                "away_team": best["away_team"],
                "best_side": best["side"],
                "signal": best["signal"],
                "best_odds": best["decimal_odds"],
                "best_edge": best["probability_edge"] * 100,
                "best_ev": best["expected_value"] * 100,
            }
        )

    table = pd.DataFrame(
        rows
    )

    print(
        table
        .sort_values(
            [
                "date",
                "best_edge",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .round(3)
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("==============================")
    print("V5 PAPER-LIVE ENGINE")
    print("==============================")
    print()

    print(
        "Mode: prospective tracking"
    )

    print(
        "Frozen V5 probabilities"
    )

    print(
        "Kelly: informational only"
    )

    print(
        "No historical threshold "
        "optimization performed"
    )

    # ========================================================
    # LOAD
    # ========================================================

    predictions = load_predictions()

    odds = load_odds()

    prices = build_best_prices(
        odds
    )

    matches = build_match_table(
        predictions,
        prices,
    )

    signals = build_signals(
        matches
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_current(
        signals
    )

    history = append_history(
        signals
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        f"Live prediction matches: "
        f"{len(predictions):,}"
    )

    print(
        f"Odds rows supplied: "
        f"{len(odds):,}"
    )

    print(
        f"Matches with odds: "
        f"{len(matches):,}"
    )

    print(
        f"Outcome observations: "
        f"{len(signals):,}"
    )

    print(
        f"Tracked history rows: "
        f"{len(history):,}"
    )

    print_signals(
        signals
    )

    print_match_summary(
        signals
    )

    print()
    print("==============================")
    print("PAPER-LIVE SNAPSHOT COMPLETE")
    print("==============================")

    print(
        "No real stakes placed ✅"
    )

    print(
        "V5 unchanged ✅"
    )

    print(
        "Market snapshots preserved ✅"
    )

    print(
        "EV recorded ✅"
    )

    print(
        "Kelly recorded for analysis only ✅"
    )

    print()
    print(
        "Current:"
    )
    print(
        OUTPUT_CURRENT
    )

    print()
    print(
        "History:"
    )
    print(
        OUTPUT_HISTORY
    )


if __name__ == "__main__":
    main()
