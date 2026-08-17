from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from scripts.backtest_v5_totals_quick import build_dataset


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "v5_totals_walkforward_predictions.csv"
)

OUTPUT_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "v5_totals_walkforward_summary.csv"
)

OUTPUT_SEASONS = (
    ROOT
    / "data"
    / "processed"
    / "v5_totals_walkforward_by_season.csv"
)

OUTPUT_LEAGUES = (
    ROOT
    / "data"
    / "processed"
    / "v5_totals_walkforward_by_league.csv"
)


# ============================================================
# SETTINGS
# ============================================================

EPS = 1e-6

MIN_TRAIN_SEASONS = 2

EDGE_THRESHOLDS = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
    0.16,
    0.18,
    0.20,
]

METHODS = [
    "RAW",
    "PLATT",
    "ISOTONIC",
]

SIDES = [
    "OVER",
    "UNDER",
]


# ============================================================
# HELPERS
# ============================================================

def clip_probability(p):

    return np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        EPS,
        1.0 - EPS,
    )


def logit(p):

    p = clip_probability(
        p
    )

    return np.log(
        p
        /
        (
            1.0 - p
        )
    )


def brier_score(
    y,
    p,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    return float(
        np.mean(
            (
                p
                -
                y
            )
            ** 2
        )
    )


def binary_log_loss(
    y,
    p,
):

    return float(
        log_loss(
            y,
            clip_probability(
                p
            ),
            labels=[
                0,
                1,
            ],
        )
    )


def calibration_error(
    y,
    p,
    bins=10,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    total = len(
        y
    )

    if total == 0:
        return np.nan

    ece = 0.0

    for i in range(
        bins
    ):

        left = edges[i]
        right = edges[
            i + 1
        ]

        if i == bins - 1:

            mask = (
                (p >= left)
                &
                (p <= right)
            )

        else:

            mask = (
                (p >= left)
                &
                (p < right)
            )

        n = int(
            mask.sum()
        )

        if n == 0:
            continue

        observed = float(
            y[
                mask
            ].mean()
        )

        predicted = float(
            p[
                mask
            ].mean()
        )

        ece += (
            n
            /
            total
        ) * abs(
            observed
            -
            predicted
        )

    return float(
        ece
    )


def season_sort_key(
    season,
):

    return int(
        str(
            season
        ).zfill(4)
    )


# ============================================================
# CALIBRATORS
# ============================================================

def fit_platt(
    train_p,
    train_y,
):

    x = logit(
        train_p
    ).reshape(
        -1,
        1,
    )

    y = np.asarray(
        train_y,
        dtype=int,
    )

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
    )

    model.fit(
        x,
        y,
    )

    return model


def predict_platt(
    model,
    p,
):

    x = logit(
        p
    ).reshape(
        -1,
        1,
    )

    return model.predict_proba(
        x
    )[
        :,
        1
    ]


def fit_isotonic(
    train_p,
    train_y,
):

    model = IsotonicRegression(
        y_min=EPS,
        y_max=1.0 - EPS,
        increasing=True,
        out_of_bounds="clip",
    )

    model.fit(
        np.asarray(
            train_p,
            dtype=float,
        ),
        np.asarray(
            train_y,
            dtype=float,
        ),
    )

    return model


# ============================================================
# WALK FORWARD
# ============================================================

def build_walkforward_predictions(
    df,
):

    data = df.copy()

    data[
        "season"
    ] = (
        data[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    data[
        "actual_over"
    ] = (
        data[
            "actual_total"
        ]
        >
        2.5
    ).astype(int)

    seasons = sorted(
        data[
            "season"
        ]
        .dropna()
        .unique(),
        key=season_sort_key,
    )

    print()
    print(
        "Available seasons:",
        ", ".join(
            seasons
        ),
    )

    print()

    rows = []

    for test_index in range(
        MIN_TRAIN_SEASONS,
        len(seasons),
    ):

        train_seasons = seasons[
            :test_index
        ]

        test_season = seasons[
            test_index
        ]

        train = data[
            data[
                "season"
            ].isin(
                train_seasons
            )
        ].copy()

        test = data[
            data[
                "season"
            ].eq(
                test_season
            )
        ].copy()

        if (
            train.empty
            or
            test.empty
        ):
            continue

        print(
            f"TRAIN "
            f"{train_seasons[0]}"
            f" -> "
            f"{train_seasons[-1]}"
            f" "
            f"({len(train):,} matches)"
            f" | "
            f"TEST {test_season}"
            f" "
            f"({len(test):,})"
        )

        train_p = train[
            "p_over_v5"
        ].to_numpy(
            dtype=float
        )

        train_y = train[
            "actual_over"
        ].to_numpy(
            dtype=int
        )

        test_p = test[
            "p_over_v5"
        ].to_numpy(
            dtype=float
        )

        # ----------------------------------------------------
        # Fit calibration only on PRIOR seasons.
        # ----------------------------------------------------

        platt = fit_platt(
            train_p,
            train_y,
        )

        isotonic = fit_isotonic(
            train_p,
            train_y,
        )

        raw_over = clip_probability(
            test_p
        )

        platt_over = clip_probability(
            predict_platt(
                platt,
                test_p,
            )
        )

        isotonic_over = clip_probability(
            isotonic.predict(
                test_p
            )
        )

        out = test.copy()

        out[
            "train_start_season"
        ] = train_seasons[
            0
        ]

        out[
            "train_end_season"
        ] = train_seasons[
            -1
        ]

        out[
            "test_season"
        ] = test_season

        out[
            "p_over_raw"
        ] = raw_over

        out[
            "p_under_raw"
        ] = (
            1.0
            -
            raw_over
        )

        out[
            "p_over_platt"
        ] = platt_over

        out[
            "p_under_platt"
        ] = (
            1.0
            -
            platt_over
        )

        out[
            "p_over_isotonic"
        ] = isotonic_over

        out[
            "p_under_isotonic"
        ] = (
            1.0
            -
            isotonic_over
        )

        # ----------------------------------------------------
        # Edges / EV after calibration.
        # ----------------------------------------------------

        for method in [
            "raw",
            "platt",
            "isotonic",
        ]:

            out[
                f"over_edge_{method}"
            ] = (
                out[
                    f"p_over_{method}"
                ]
                -
                out[
                    "market_p_over"
                ]
            )

            out[
                f"under_edge_{method}"
            ] = (
                out[
                    f"p_under_{method}"
                ]
                -
                out[
                    "market_p_under"
                ]
            )

            out[
                f"over_ev_{method}"
            ] = (
                out[
                    f"p_over_{method}"
                ]
                *
                out[
                    "over_odds"
                ]
                -
                1.0
            )

            out[
                f"under_ev_{method}"
            ] = (
                out[
                    f"p_under_{method}"
                ]
                *
                out[
                    "under_odds"
                ]
                -
                1.0
            )

        rows.append(
            out
        )

    if not rows:

        raise RuntimeError(
            "No walk-forward test seasons were created."
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ============================================================
# PROBABILITY QUALITY
# ============================================================

def probability_metrics(
    df,
):

    y = df[
        "actual_over"
    ].to_numpy(
        dtype=int
    )

    rows = []

    for method in METHODS:

        key = method.lower()

        p = df[
            f"p_over_{key}"
        ].to_numpy(
            dtype=float
        )

        rows.append(
            {
                "method":
                    method,

                "matches":
                    len(df),

                "brier":
                    brier_score(
                        y,
                        p,
                    ),

                "log_loss":
                    binary_log_loss(
                        y,
                        p,
                    ),

                "ece":
                    calibration_error(
                        y,
                        p,
                    ),

                "avg_prediction":
                    float(
                        np.mean(
                            p
                        )
                    ),

                "actual_over_rate":
                    float(
                        np.mean(
                            y
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# BETTING EVALUATION
# ============================================================

def evaluate_bets(
    df,
    method,
    side,
    threshold,
):

    method_key = method.lower()
    side_key = side.lower()

    edge_col = (
        f"{side_key}_edge_"
        f"{method_key}"
    )

    ev_col = (
        f"{side_key}_ev_"
        f"{method_key}"
    )

    probability_col = (
        f"p_{side_key}_"
        f"{method_key}"
    )

    odds_col = (
        "over_odds"
        if side == "OVER"
        else
        "under_odds"
    )

    x = df[
        df[
            edge_col
        ]
        >=
        threshold
    ].copy()

    if x.empty:

        return {
            "method":
                method,

            "side":
                side,

            "threshold":
                threshold,

            "bets":
                0,

            "wins":
                0,

            "win_rate":
                np.nan,

            "avg_odds":
                np.nan,

            "avg_model_probability":
                np.nan,

            "avg_edge":
                np.nan,

            "avg_ev":
                np.nan,

            "profit":
                0.0,

            "roi":
                np.nan,
        }

    if side == "OVER":

        x[
            "won"
        ] = (
            x[
                "actual_total"
            ]
            >
            2.5
        )

    else:

        x[
            "won"
        ] = (
            x[
                "actual_total"
            ]
            <
            2.5
        )

    x[
        "bet_profit"
    ] = np.where(
        x[
            "won"
        ],
        x[
            odds_col
        ]
        -
        1.0,
        -1.0,
    )

    bets = len(
        x
    )

    profit = float(
        x[
            "bet_profit"
        ].sum()
    )

    return {
        "method":
            method,

        "side":
            side,

        "threshold":
            threshold,

        "bets":
            bets,

        "wins":
            int(
                x[
                    "won"
                ].sum()
            ),

        "win_rate":
            float(
                x[
                    "won"
                ].mean()
            ),

        "avg_odds":
            float(
                x[
                    odds_col
                ].mean()
            ),

        "avg_model_probability":
            float(
                x[
                    probability_col
                ].mean()
            ),

        "avg_edge":
            float(
                x[
                    edge_col
                ].mean()
            ),

        "avg_ev":
            float(
                x[
                    ev_col
                ].mean()
            ),

        "profit":
            profit,

        "roi":
            profit
            /
            bets,
    }


def build_betting_summary(
    df,
):

    rows = []

    for method in METHODS:

        for side in SIDES:

            for threshold in EDGE_THRESHOLDS:

                rows.append(
                    evaluate_bets(
                        df,
                        method,
                        side,
                        threshold,
                    )
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# GROUPED PERFORMANCE
# ============================================================

def grouped_performance(
    df,
    group_col,
):

    rows = []

    important_thresholds = [
        0.08,
        0.10,
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
        0.16,
    ]

    for group, g in df.groupby(
        group_col
    ):

        for method in METHODS:

            for side in SIDES:

                for threshold in important_thresholds:

                    result = evaluate_bets(
                        g,
                        method,
                        side,
                        threshold,
                    )

                    result[
                        group_col
                    ] = group

                    rows.append(
                        result
                    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DISPLAY
# ============================================================

def display_probability_metrics(
    metrics,
):

    show = metrics.copy()

    show[
        "ece"
    ] *= 100.0

    show[
        "avg_prediction"
    ] *= 100.0

    show[
        "actual_over_rate"
    ] *= 100.0

    print()
    print("=" * 120)
    print("OUT-OF-SAMPLE PROBABILITY QUALITY")
    print("=" * 120)
    print()

    print(
        show.to_string(
            index=False,
            formatters={
                "brier":
                    lambda x:
                    f"{x:.5f}",

                "log_loss":
                    lambda x:
                    f"{x:.5f}",

                "ece":
                    lambda x:
                    f"{x:.2f}%",

                "avg_prediction":
                    lambda x:
                    f"{x:.2f}%",

                "actual_over_rate":
                    lambda x:
                    f"{x:.2f}%",
            },
        )
    )


def display_betting(
    summary,
    side,
):

    x = summary[
        summary[
            "side"
        ].eq(
            side
        )
    ].copy()

    show = x.copy()

    for col in [
        "win_rate",
        "avg_model_probability",
        "avg_edge",
        "avg_ev",
        "roi",
    ]:

        show[
            col
        ] *= 100.0

    print()
    print("=" * 120)
    print(
        f"{side} 2.5 — "
        f"WALK-FORWARD THRESHOLD SCAN"
    )
    print("=" * 120)
    print()

    print(
        show[
            [
                "method",
                "threshold",
                "bets",
                "wins",
                "win_rate",
                "avg_odds",
                "avg_model_probability",
                "avg_edge",
                "avg_ev",
                "profit",
                "roi",
            ]
        ].to_string(
            index=False,
            formatters={
                "threshold":
                    lambda x:
                    f"{x:.0%}",

                "win_rate":
                    lambda x:
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "-",

                "avg_odds":
                    lambda x:
                    f"{x:.3f}"
                    if pd.notna(x)
                    else "-",

                "avg_model_probability":
                    lambda x:
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "-",

                "avg_edge":
                    lambda x:
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "-",

                "avg_ev":
                    lambda x:
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "-",

                "profit":
                    lambda x:
                    f"{x:+.2f}u",

                "roi":
                    lambda x:
                    f"{x:+.2f}%"
                    if pd.notna(x)
                    else "-",
            },
        )
    )


def display_best_regions(
    summary,
):

    print()
    print("=" * 120)
    print("POSITIVE WALK-FORWARD REGIONS")
    print("=" * 120)
    print()

    x = summary[
        (
            summary[
                "bets"
            ]
            >=
            30
        )
        &
        (
            summary[
                "roi"
            ]
            >
            0
        )
    ].copy()

    if x.empty:

        print(
            "No positive regions with "
            "at least 30 bets."
        )

        return

    x = x.sort_values(
        [
            "side",
            "roi",
        ],
        ascending=[
            True,
            False,
        ],
    )

    x[
        "roi_pct"
    ] = (
        x[
            "roi"
        ]
        *
        100.0
    )

    x[
        "avg_edge_pct"
    ] = (
        x[
            "avg_edge"
        ]
        *
        100.0
    )

    print(
        x[
            [
                "method",
                "side",
                "threshold",
                "bets",
                "wins",
                "profit",
                "roi_pct",
                "avg_edge_pct",
            ]
        ].to_string(
            index=False,
            formatters={
                "threshold":
                    lambda x:
                    f"{x:.0%}",

                "profit":
                    lambda x:
                    f"{x:+.2f}u",

                "roi_pct":
                    lambda x:
                    f"{x:+.2f}%",

                "avg_edge_pct":
                    lambda x:
                    f"{x:.2f}%",
            },
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 120)
    print("V5 TOTALS WALK-FORWARD CALIBRATION")
    print("=" * 120)

    raw = build_dataset()

    print()
    print(
        f"Full historical dataset: "
        f"{len(raw):,} matches"
    )

    wf = build_walkforward_predictions(
        raw
    )

    print()
    print(
        f"Out-of-sample matches: "
        f"{len(wf):,}"
    )

    print(
        f"Out-of-sample seasons: "
        f"{wf['test_season'].nunique()}"
    )

    print(
        "Test seasons:",
        ", ".join(
            sorted(
                wf[
                    "test_season"
                ]
                .astype(str)
                .unique(),
                key=season_sort_key,
            )
        ),
    )

    # ========================================================
    # PROBABILITY QUALITY
    # ========================================================

    metrics = probability_metrics(
        wf
    )

    display_probability_metrics(
        metrics
    )

    # ========================================================
    # BETTING PERFORMANCE
    # ========================================================

    summary = build_betting_summary(
        wf
    )

    display_betting(
        summary,
        "OVER",
    )

    display_betting(
        summary,
        "UNDER",
    )

    display_best_regions(
        summary
    )

    # ========================================================
    # SEASON / LEAGUE
    # ========================================================

    by_season = grouped_performance(
        wf,
        "test_season",
    )

    by_league = grouped_performance(
        wf,
        "league",
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PREDICTIONS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    wf.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    by_season.to_csv(
        OUTPUT_SEASONS,
        index=False,
    )

    by_league.to_csv(
        OUTPUT_LEAGUES,
        index=False,
    )

    print()
    print("=" * 120)
    print("FILES SAVED")
    print("=" * 120)
    print()

    print(
        OUTPUT_PREDICTIONS
    )

    print(
        OUTPUT_SUMMARY
    )

    print(
        OUTPUT_SEASONS
    )

    print(
        OUTPUT_LEAGUES
    )

    print()
    print("=" * 120)
    print("INTERPRETATION RULE")
    print("=" * 120)
    print()

    print(
        "Do NOT select a live totals rule "
        "from headline ROI alone."
    )

    print(
        "We want calibration improvement, "
        "positive OOS ROI across nearby thresholds, "
        "reasonable sample size, and stability "
        "across seasons/leagues."
    )


if __name__ == "__main__":
    main()
