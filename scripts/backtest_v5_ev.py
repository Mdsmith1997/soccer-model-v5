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

OUTPUT_THRESHOLD_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "v5_ev_threshold_results.csv"
)

OUTPUT_BETS = (
    ROOT
    / "data"
    / "processed"
    / "v5_ev_backtest_bets.csv"
)

OUTPUT_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "v5_ev_backtest_summary.csv"
)


# ============================================================
# DATA SPLITS
# ============================================================

TUNING_SEASONS = {
    "2122",
    "2223",
}

VALIDATION_SEASONS = {
    "2324",
}

FINAL_SEASONS = {
    "2425",
}

LOCKED_SEASONS = {
    "2526",
}


# ============================================================
# BETTING SETTINGS
# ============================================================

EDGE_THRESHOLDS = [
    0.00,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.075,
    0.10,
]

EV_THRESHOLDS = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
]

MIN_BETS_FOR_SELECTION = 40

FLAT_STAKE = 1.0

EPS = 1e-12


# ============================================================
# HELPERS
# ============================================================

def season_string(
    series,
):

    return (
        series
        .astype(str)
        .str.zfill(4)
    )


def actual_outcome(
    df,
):

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
        "season",
        "league",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",

        "p_home_v5",
        "p_draw_v5",
        "p_away_v5",

        "market_home_odds",
        "market_draw_odds",
        "market_away_odds",

        "market_nv_home",
        "market_nv_draw",
        "market_nv_away",

        "market_source",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: "
            + str(missing)
        )

    # Main betting test uses closing market only.
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
# BUILD BET CANDIDATES
# ============================================================

def build_bet_candidates(
    df,
):

    rows = []

    sides = [
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

    for side, model_col, market_col, odds_col in sides:

        sub = df[
            [
                "match_id",
                "date",
                "season",
                "league",
                "home_team",
                "away_team",
                "actual_outcome",

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

        sub["bet_side"] = side

        sub["probability_edge"] = (
            sub["model_probability"]
            -
            sub["market_probability"]
        )

        sub["expected_value"] = (
            sub["model_probability"]
            *
            sub["decimal_odds"]
            -
            1.0
        )

        sub["won"] = (
            sub["actual_outcome"]
            ==
            side
        ).astype(int)

        sub["profit"] = np.where(
            sub["won"] == 1,
            (
                sub["decimal_odds"]
                -
                1.0
            )
            *
            FLAT_STAKE,
            -FLAT_STAKE,
        )

        rows.append(
            sub
        )

    bets = pd.concat(
        rows,
        ignore_index=True,
    )

    valid = (
        bets["model_probability"].notna()
        &
        bets["market_probability"].notna()
        &
        bets["decimal_odds"].notna()
        &
        np.isfinite(
            bets["decimal_odds"]
        )
        &
        (bets["decimal_odds"] > 1.0)
    )

    return bets.loc[
        valid
    ].copy()


# ============================================================
# BET METRICS
# ============================================================

def evaluate_bets(
    bets,
):

    if len(bets) == 0:

        return {
            "bets": 0,
            "wins": 0,
            "win_rate": np.nan,
            "avg_odds": np.nan,
            "avg_edge": np.nan,
            "avg_ev": np.nan,
            "profit": 0.0,
            "roi": np.nan,
        }

    total_staked = (
        len(bets)
        *
        FLAT_STAKE
    )

    profit = bets[
        "profit"
    ].sum()

    return {
        "bets":
            len(bets),

        "wins":
            int(
                bets["won"].sum()
            ),

        "win_rate":
            float(
                bets["won"].mean()
            ),

        "avg_odds":
            float(
                bets["decimal_odds"].mean()
            ),

        "avg_edge":
            float(
                bets["probability_edge"].mean()
            ),

        "avg_ev":
            float(
                bets["expected_value"].mean()
            ),

        "profit":
            float(
                profit
            ),

        "roi":
            float(
                profit
                /
                total_staked
            ),
    }


# ============================================================
# APPLY THRESHOLD
# ============================================================

def filter_bets(
    bets,
    edge_threshold,
    ev_threshold,
):

    return bets[
        (
            bets["probability_edge"]
            >=
            edge_threshold
        )
        &
        (
            bets["expected_value"]
            >=
            ev_threshold
        )
    ].copy()


# ============================================================
# BUILD THRESHOLD TABLE
# ============================================================

def tune_thresholds(
    bets,
):

    tune = bets[
        bets["season"]
        .isin(
            TUNING_SEASONS
        )
    ].copy()

    rows = []

    for edge_threshold in EDGE_THRESHOLDS:

        for ev_threshold in EV_THRESHOLDS:

            selected = filter_bets(
                tune,
                edge_threshold,
                ev_threshold,
            )

            metrics = evaluate_bets(
                selected
            )

            rows.append(
                {
                    "edge_threshold":
                        edge_threshold,

                    "ev_threshold":
                        ev_threshold,

                    **metrics,
                }
            )

    results = pd.DataFrame(
        rows
    )

    eligible = results[
        results["bets"]
        >=
        MIN_BETS_FOR_SELECTION
    ].copy()

    if len(eligible) == 0:

        raise ValueError(
            "No threshold combination had enough bets."
        )

    eligible = (
        eligible
        .sort_values(
            [
                "roi",
                "profit",
                "bets",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    eligible["rank"] = (
        np.arange(
            len(eligible)
        )
        + 1
    )

    return eligible


# ============================================================
# PRINT BET METRICS
# ============================================================

def print_metrics(
    title,
    metrics,
):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

    print(
        f"Bets:       "
        f"{metrics['bets']:,}"
    )

    print(
        f"Wins:       "
        f"{metrics['wins']:,}"
    )

    print(
        f"Win rate:   "
        f"{metrics['win_rate']:.2%}"
    )

    print(
        f"Avg odds:   "
        f"{metrics['avg_odds']:.3f}"
    )

    print(
        f"Avg edge:   "
        f"{metrics['avg_edge']:.2%}"
    )

    print(
        f"Avg EV:     "
        f"{metrics['avg_ev']:.2%}"
    )

    print(
        f"Profit:     "
        f"{metrics['profit']:+.2f} units"
    )

    print(
        f"ROI:        "
        f"{metrics['roi']:+.2%}"
    )


# ============================================================
# BY SIDE
# ============================================================

def print_by_side(
    bets,
):

    print()
    print("=" * 95)
    print("BY BET TYPE")
    print("=" * 95)

    rows = []

    for side, sub in bets.groupby(
        "bet_side"
    ):

        metrics = evaluate_bets(
            sub
        )

        rows.append(
            {
                "side":
                    side,

                **metrics,
            }
        )

    table = pd.DataFrame(
        rows
    )

    for col in [
        "win_rate",
        "avg_edge",
        "avg_ev",
        "roi",
    ]:

        table[col] *= 100

    print(
        table
        .round(4)
        .to_string(
            index=False
        )
    )


# ============================================================
# BY LEAGUE
# ============================================================

def print_by_league(
    bets,
):

    print()
    print("=" * 95)
    print("BY LEAGUE")
    print("=" * 95)

    rows = []

    for league, sub in bets.groupby(
        "league"
    ):

        metrics = evaluate_bets(
            sub
        )

        rows.append(
            {
                "league":
                    league,

                **metrics,
            }
        )

    table = pd.DataFrame(
        rows
    )

    for col in [
        "win_rate",
        "avg_edge",
        "avg_ev",
        "roi",
    ]:

        table[col] *= 100

    print(
        table
        .round(4)
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
    print("V5 EV BACKTEST")
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

    bets = build_bet_candidates(
        df
    )

    print(
        f"Bet candidates: "
        f"{len(bets):,}"
    )

    # ========================================================
    # TUNE THRESHOLDS
    # ========================================================

    results = tune_thresholds(
        bets
    )

    results.to_csv(
        OUTPUT_THRESHOLD_RESULTS,
        index=False,
    )

    print()
    print("==============================")
    print("TOP 20 BETTING THRESHOLDS")
    print("==============================")
    print()

    display = (
        results
        .head(20)
        .copy()
    )

    for col in [
        "win_rate",
        "avg_edge",
        "avg_ev",
        "roi",
    ]:

        display[col] *= 100

    print(
        display[
            [
                "rank",
                "edge_threshold",
                "ev_threshold",
                "bets",
                "wins",
                "win_rate",
                "avg_odds",
                "avg_edge",
                "avg_ev",
                "profit",
                "roi",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # WINNER
    # ========================================================

    best = results.iloc[
        0
    ]

    best_edge = float(
        best[
            "edge_threshold"
        ]
    )

    best_ev = float(
        best[
            "ev_threshold"
        ]
    )

    print()
    print("==============================")
    print("SELECTED BETTING THRESHOLD")
    print("==============================")

    print(
        f"Probability edge: "
        f"{best_edge:.1%}"
    )

    print(
        f"Minimum EV: "
        f"{best_ev:.1%}"
    )

    print(
        f"Tuning ROI: "
        f"{best['roi']:.2%}"
    )

    # ========================================================
    # PERIOD TESTS
    # ========================================================

    samples = [
        (
            "TUNING — 2021/22 TO 2022/23",
            TUNING_SEASONS,
        ),
        (
            "VALIDATION — 2023/24",
            VALIDATION_SEASONS,
        ),
        (
            "FINAL CHECK — 2024/25",
            FINAL_SEASONS,
        ),
        (
            "LOCKED TEST — 2025/26",
            LOCKED_SEASONS,
        ),
    ]

    summary_rows = []

    selected_all = []

    for title, seasons in samples:

        sub = bets[
            bets["season"]
            .isin(
                seasons
            )
        ].copy()

        selected = filter_bets(
            sub,
            best_edge,
            best_ev,
        )

        selected["sample"] = title

        selected_all.append(
            selected
        )

        metrics = evaluate_bets(
            selected
        )

        print_metrics(
            title,
            metrics,
        )

        summary_rows.append(
            {
                "sample":
                    title,

                "edge_threshold":
                    best_edge,

                "ev_threshold":
                    best_ev,

                **metrics,
            }
        )

    # ========================================================
    # SAVE SELECTED BETS
    # ========================================================

    selected_bets = pd.concat(
        selected_all,
        ignore_index=True,
    )

    selected_bets.to_csv(
        OUTPUT_BETS,
        index=False,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # ========================================================
    # LOCKED TEST BREAKDOWNS
    # ========================================================

    locked = selected_bets[
        selected_bets["sample"]
        ==
        "LOCKED TEST — 2025/26"
    ].copy()

    print_by_side(
        locked
    )

    print_by_league(
        locked
    )

    # ========================================================
    # EDGE BANDS ON LOCKED TEST
    # ========================================================

    if len(locked) > 0:

        print()
        print("=" * 95)
        print("LOCKED TEST — EDGE BANDS")
        print("=" * 95)

        bins = [
            best_edge,
            0.05,
            0.075,
            0.10,
            0.15,
            1.00,
        ]

        bins = sorted(
            list(
                set(
                    [
                        x
                        for x in bins
                        if x >= best_edge
                    ]
                )
            )
        )

        if len(bins) >= 2:

            locked[
                "edge_band"
            ] = pd.cut(
                locked[
                    "probability_edge"
                ],
                bins=bins,
                include_lowest=True,
            )

            rows = []

            for band, sub in locked.groupby(
                "edge_band",
                observed=False,
            ):

                if len(sub) == 0:
                    continue

                metrics = evaluate_bets(
                    sub
                )

                rows.append(
                    {
                        "edge_band":
                            str(band),

                        **metrics,
                    }
                )

            edge_table = pd.DataFrame(
                rows
            )

            for col in [
                "win_rate",
                "avg_edge",
                "avg_ev",
                "roi",
            ]:

                edge_table[col] *= 100

            print(
                edge_table
                .round(4)
                .to_string(
                    index=False
                )
            )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("==============================")
    print("V5 EV BACKTEST COMPLETE")
    print("==============================")

    print(
        "Only avg_close odds used ✅"
    )

    print(
        "Thresholds selected using "
        "2021/22–2022/23 only ✅"
    )

    print(
        "2023/24, 2024/25 and 2025/26 "
        "not used for threshold selection ✅"
    )

    print(
        "Flat 1-unit staking used for "
        "initial EV test ✅"
    )

    print()
    print(
        "Threshold results:"
    )

    print(
        OUTPUT_THRESHOLD_RESULTS
    )

    print()

    print(
        "Selected bets:"
    )

    print(
        OUTPUT_BETS
    )

    print()

    print(
        "Summary:"
    )

    print(
        OUTPUT_SUMMARY
    )


if __name__ == "__main__":
    main()