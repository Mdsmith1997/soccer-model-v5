from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "v5_market_comparison.csv"
)

OUTPUT_DETAIL = (
    ROOT
    / "data"
    / "processed"
    / "v5_market_edge_diagnostics.csv"
)

OUTPUT_PERIOD = (
    ROOT
    / "data"
    / "processed"
    / "v5_market_edge_by_period.csv"
)


# ============================================================
# PERIODS
# ============================================================

PERIODS = {
    "TUNING_2122_2223": {
        "2122",
        "2223",
    },

    "VALIDATION_2324": {
        "2324",
    },

    "FINAL_2425": {
        "2425",
    },

    "LOCKED_2526": {
        "2526",
    },
}


# ============================================================
# EDGE BINS
# ============================================================

EDGE_BINS = [
    -1.00,
    -0.15,
    -0.10,
    -0.075,
    -0.05,
    -0.025,
    0.00,
    0.025,
    0.05,
    0.075,
    0.10,
    0.15,
    1.00,
]

EDGE_LABELS = [
    "<-15%",
    "-15 to -10%",
    "-10 to -7.5%",
    "-7.5 to -5%",
    "-5 to -2.5%",
    "-2.5 to 0%",
    "0 to +2.5%",
    "+2.5 to +5%",
    "+5 to +7.5%",
    "+7.5 to +10%",
    "+10 to +15%",
    "+15%+",
]


# ============================================================
# HELPERS
# ============================================================

def season_string(series):

    return (
        series
        .astype(str)
        .str.zfill(4)
    )


def actual_outcome(df):

    return np.where(
        df["home_goals"].to_numpy()
        >
        df["away_goals"].to_numpy(),
        "HOME",
        np.where(
            df["home_goals"].to_numpy()
            ==
            df["away_goals"].to_numpy(),
            "DRAW",
            "AWAY",
        ),
    )


# ============================================================
# LOAD
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "date",
        ],
    )

    df["season"] = season_string(
        df["season"]
    )

    required = [
        "match_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",

        "p_home_v5",
        "p_draw_v5",
        "p_away_v5",

        "market_nv_home",
        "market_nv_draw",
        "market_nv_away",

        "market_home_odds",
        "market_draw_odds",
        "market_away_odds",

        "market_source",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + str(missing)
        )

    # Closing odds only.
    df = df[
        df["market_source"]
        ==
        "avg_close"
    ].copy()

    df["actual_outcome"] = actual_outcome(
        df
    )

    return df


# ============================================================
# LONG FORMAT
# ============================================================

def build_long_table(df):

    rows = []

    definitions = [
        (
            "HOME",
            "p_home_v5",
            "market_nv_home",
            "market_home_odds",
        ),
        (
            "DRAW",
            "p_draw_v5",
            "market_nv_draw",
            "market_draw_odds",
        ),
        (
            "AWAY",
            "p_away_v5",
            "market_nv_away",
            "market_away_odds",
        ),
    ]

    base_cols = [
        "match_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
        "actual_outcome",
    ]

    for side, model_col, market_col, odds_col in definitions:

        sub = df[
            base_cols
            +
            [
                model_col,
                market_col,
                odds_col,
            ]
        ].copy()

        sub = sub.rename(
            columns={
                model_col:
                    "model_probability",

                market_col:
                    "market_probability",

                odds_col:
                    "decimal_odds",
            }
        )

        sub["side"] = side

        sub["edge"] = (
            sub["model_probability"]
            -
            sub["market_probability"]
        )

        sub["actual_hit"] = (
            sub["actual_outcome"]
            ==
            side
        ).astype(
            int
        )

        sub["flat_profit"] = np.where(
            sub["actual_hit"] == 1,
            sub["decimal_odds"] - 1.0,
            -1.0,
        )

        rows.append(
            sub
        )

    long_df = pd.concat(
        rows,
        ignore_index=True,
    )

    valid = (
        long_df["model_probability"].notna()
        &
        long_df["market_probability"].notna()
        &
        long_df["decimal_odds"].notna()
        &
        np.isfinite(
            long_df["decimal_odds"]
        )
        &
        (
            long_df["decimal_odds"]
            >
            1.0
        )
    )

    long_df = long_df.loc[
        valid
    ].copy()

    long_df["edge_band"] = pd.cut(
        long_df["edge"],
        bins=EDGE_BINS,
        labels=EDGE_LABELS,
        right=False,
    )

    return long_df


# ============================================================
# AGGREGATION
# ============================================================

def summarize_group(
    sub,
):

    games = len(
        sub
    )

    if games == 0:

        return None

    market_prob = (
        sub[
            "market_probability"
        ]
        .mean()
    )

    model_prob = (
        sub[
            "model_probability"
        ]
        .mean()
    )

    actual_rate = (
        sub[
            "actual_hit"
        ]
        .mean()
    )

    avg_odds = (
        sub[
            "decimal_odds"
        ]
        .mean()
    )

    flat_profit = (
        sub[
            "flat_profit"
        ]
        .sum()
    )

    roi = (
        flat_profit
        /
        games
    )

    return {
        "games":
            games,

        "avg_market_probability":
            market_prob,

        "avg_model_probability":
            model_prob,

        "actual_hit_rate":
            actual_rate,

        "avg_edge":
            sub[
                "edge"
            ].mean(),

        "market_error":
            (
                actual_rate
                -
                market_prob
            ),

        "model_error":
            (
                actual_rate
                -
                model_prob
            ),

        "v5_direction_value":
            (
                actual_rate
                -
                market_prob
            )
            *
            np.sign(
                sub[
                    "edge"
                ].mean()
            ),

        "avg_closing_odds":
            avg_odds,

        "flat_profit":
            flat_profit,

        "flat_roi":
            roi,
    }


# ============================================================
# EDGE TABLE
# ============================================================

def build_edge_table(
    long_df,
):

    rows = []

    grouped = long_df.groupby(
        [
            "side",
            "edge_band",
        ],
        observed=False,
    )

    for (
        side,
        edge_band,
    ), sub in grouped:

        if len(sub) == 0:

            continue

        metrics = summarize_group(
            sub
        )

        rows.append(
            {
                "side":
                    side,

                "edge_band":
                    str(
                        edge_band
                    ),

                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PERIOD TABLE
# ============================================================

def build_period_table(
    long_df,
):

    rows = []

    for period_name, seasons in PERIODS.items():

        period_df = long_df[
            long_df["season"]
            .isin(
                seasons
            )
        ].copy()

        grouped = period_df.groupby(
            [
                "side",
                "edge_band",
            ],
            observed=False,
        )

        for (
            side,
            edge_band,
        ), sub in grouped:

            if len(sub) == 0:

                continue

            metrics = summarize_group(
                sub
            )

            rows.append(
                {
                    "period":
                        period_name,

                    "side":
                        side,

                    "edge_band":
                        str(
                            edge_band
                        ),

                    **metrics,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT TABLE
# ============================================================

def print_diagnostic_table(
    title,
    df,
):

    print()
    print("=" * 130)
    print(title)
    print("=" * 130)

    if len(df) == 0:

        print(
            "No rows."
        )

        return

    display = df.copy()

    pct_cols = [
        "avg_market_probability",
        "avg_model_probability",
        "actual_hit_rate",
        "avg_edge",
        "market_error",
        "model_error",
        "v5_direction_value",
        "flat_roi",
    ]

    for col in pct_cols:

        if col in display.columns:

            display[
                col
            ] *= 100.0

    cols = [
        col
        for col in [
            "side",
            "edge_band",
            "games",
            "avg_market_probability",
            "avg_model_probability",
            "actual_hit_rate",
            "avg_edge",
            "market_error",
            "model_error",
            "v5_direction_value",
            "avg_closing_odds",
            "flat_profit",
            "flat_roi",
        ]
        if col in display.columns
    ]

    print(
        display[
            cols
        ]
        .round(
            3
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# POSITIVE EDGE SUMMARY
# ============================================================

def print_positive_edge_summary(
    long_df,
):

    print()
    print("=" * 120)
    print("POSITIVE V5 EDGE — SUMMARY")
    print("=" * 120)

    rows = []

    for threshold in [
        0.025,
        0.05,
        0.075,
        0.10,
        0.15,
    ]:

        for side in [
            "HOME",
            "DRAW",
            "AWAY",
        ]:

            sub = long_df[
                (
                    long_df["side"]
                    ==
                    side
                )
                &
                (
                    long_df["edge"]
                    >=
                    threshold
                )
            ].copy()

            if len(sub) == 0:

                continue

            metrics = summarize_group(
                sub
            )

            rows.append(
                {
                    "side":
                        side,

                    "min_edge":
                        threshold,

                    **metrics,
                }
            )

    table = pd.DataFrame(
        rows
    )

    for col in [
        "avg_market_probability",
        "avg_model_probability",
        "actual_hit_rate",
        "avg_edge",
        "market_error",
        "model_error",
        "v5_direction_value",
        "flat_roi",
    ]:

        table[
            col
        ] *= 100.0

    print(
        table[
            [
                "side",
                "min_edge",
                "games",
                "avg_market_probability",
                "avg_model_probability",
                "actual_hit_rate",
                "avg_edge",
                "market_error",
                "v5_direction_value",
                "avg_closing_odds",
                "flat_profit",
                "flat_roi",
            ]
        ]
        .round(
            3
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# PERIOD POSITIVE EDGE SUMMARY
# ============================================================

def print_locked_positive_edges(
    long_df,
):

    locked = long_df[
        long_df["season"]
        .isin(
            PERIODS[
                "LOCKED_2526"
            ]
        )
    ].copy()

    print()
    print("=" * 120)
    print("2025/26 LOCKED TEST — POSITIVE EDGE")
    print("=" * 120)

    rows = []

    for threshold in [
        0.025,
        0.05,
        0.075,
        0.10,
        0.15,
    ]:

        for side in [
            "HOME",
            "DRAW",
            "AWAY",
        ]:

            sub = locked[
                (
                    locked["side"]
                    ==
                    side
                )
                &
                (
                    locked["edge"]
                    >=
                    threshold
                )
            ].copy()

            if len(sub) == 0:

                continue

            metrics = summarize_group(
                sub
            )

            rows.append(
                {
                    "side":
                        side,

                    "min_edge":
                        threshold,

                    **metrics,
                }
            )

    table = pd.DataFrame(
        rows
    )

    for col in [
        "avg_market_probability",
        "avg_model_probability",
        "actual_hit_rate",
        "avg_edge",
        "market_error",
        "model_error",
        "v5_direction_value",
        "flat_roi",
    ]:

        table[
            col
        ] *= 100.0

    print(
        table[
            [
                "side",
                "min_edge",
                "games",
                "avg_market_probability",
                "avg_model_probability",
                "actual_hit_rate",
                "avg_edge",
                "market_error",
                "v5_direction_value",
                "avg_closing_odds",
                "flat_profit",
                "flat_roi",
            ]
        ]
        .round(
            3
        )
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
    print("V5 MARKET EDGE DIAGNOSTICS")
    print("==============================")
    print()

    df = load_data()

    print(
        f"Closing-market matches: "
        f"{len(df):,}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"-> "
        f"{df['date'].max().date()}"
    )

    long_df = build_long_table(
        df
    )

    print(
        f"Outcome observations: "
        f"{len(long_df):,}"
    )

    # ========================================================
    # ALL-PERIOD DIAGNOSTICS
    # ========================================================

    edge_table = build_edge_table(
        long_df
    )

    edge_table.to_csv(
        OUTPUT_DETAIL,
        index=False,
    )

    print_diagnostic_table(
        "ALL PERIODS — EDGE DIAGNOSTICS",
        edge_table,
    )

    # ========================================================
    # PERIOD DIAGNOSTICS
    # ========================================================

    period_table = build_period_table(
        long_df
    )

    period_table.to_csv(
        OUTPUT_PERIOD,
        index=False,
    )

    for period_name in PERIODS.keys():

        sub = period_table[
            period_table["period"]
            ==
            period_name
        ].copy()

        print_diagnostic_table(
            period_name,
            sub,
        )

    # ========================================================
    # POSITIVE EDGE
    # ========================================================

    print_positive_edge_summary(
        long_df
    )

    print_locked_positive_edges(
        long_df
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("==============================")
    print("EDGE DIAGNOSTICS COMPLETE")
    print("==============================")

    print(
        "Only avg_close odds used ✅"
    )

    print(
        "No new betting threshold "
        "selected ✅"
    )

    print(
        "2025/26 used only as "
        "diagnostic locked test ✅"
    )

    print()
    print(
        "Diagnostics:"
    )

    print(
        OUTPUT_DETAIL
    )

    print()

    print(
        "By period:"
    )

    print(
        OUTPUT_PERIOD
    )


if __name__ == "__main__":
    main()