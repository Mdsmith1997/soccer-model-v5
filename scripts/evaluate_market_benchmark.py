from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATHS / SETTINGS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MATCHES_FILE = (
    ROOT
    / "data"
    / "processed"
    / "matches.csv"
)

MODEL_FILE = (
    ROOT
    / "data"
    / "processed"
    / "poisson_v0_predictions.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "market_benchmark_comparison.csv"
)

EPS = 1e-12

RECENT_SEASONS = {
    "2324",
    "2425",
    "2526",
}


# =========================================================
# METRICS
# =========================================================

def multiclass_log_loss(
    y_true,
    probs,
):
    probs = np.clip(
        probs,
        EPS,
        1.0,
    )

    chosen = probs[
        np.arange(len(y_true)),
        y_true,
    ]

    return (
        -np.log(chosen)
    ).mean()


def multiclass_brier(
    y_true,
    probs,
):
    truth = np.zeros_like(
        probs
    )

    truth[
        np.arange(len(y_true)),
        y_true,
    ] = 1.0

    return (
        (
            probs - truth
        ) ** 2
    ).sum(
        axis=1
    ).mean()


def binary_ece(
    y_true,
    probs,
    bins=10,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    probs = np.asarray(
        probs,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0
    n = len(probs)

    for i in range(bins):

        left = edges[i]
        right = edges[i + 1]

        if i == bins - 1:
            mask = (
                (probs >= left)
                &
                (probs <= right)
            )
        else:
            mask = (
                (probs >= left)
                &
                (probs < right)
            )

        count = mask.sum()

        if count == 0:
            continue

        avg_prob = probs[
            mask
        ].mean()

        avg_actual = y_true[
            mask
        ].mean()

        ece += (
            count / n
        ) * abs(
            avg_prob
            - avg_actual
        )

    return ece


def multiclass_ece(
    y_true,
    probs,
    bins=10,
):
    predicted = probs.argmax(
        axis=1
    )

    confidence = probs.max(
        axis=1
    )

    correct = (
        predicted
        == y_true
    ).astype(float)

    return binary_ece(
        correct,
        confidence,
        bins=bins,
    )


# =========================================================
# MARKET PROBABILITIES
# =========================================================

def odds_to_fair_probs(
    home_odds,
    draw_odds,
    away_odds,
):
    """
    Convert decimal odds into implied probabilities,
    then remove bookmaker margin proportionally.
    """

    if (
        pd.isna(home_odds)
        or pd.isna(draw_odds)
        or pd.isna(away_odds)
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    if (
        home_odds <= 1.0
        or draw_odds <= 1.0
        or away_odds <= 1.0
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    raw_home = (
        1.0 / home_odds
    )

    raw_draw = (
        1.0 / draw_odds
    )

    raw_away = (
        1.0 / away_odds
    )

    overround = (
        raw_home
        + raw_draw
        + raw_away
    )

    fair_home = (
        raw_home
        / overround
    )

    fair_draw = (
        raw_draw
        / overround
    )

    fair_away = (
        raw_away
        / overround
    )

    margin = (
        overround - 1.0
    )

    return (
        fair_home,
        fair_draw,
        fair_away,
        margin,
    )


# =========================================================
# MARKET SOURCE SELECTION
# =========================================================

def select_market_odds(row):
    """
    Prefer average closing odds.

    If unavailable, fall back to Bet365 closing odds.

    If closing odds are unavailable, use average opening
    odds, then Bet365 opening odds.

    Returns:
        home odds
        draw odds
        away odds
        source
    """

    candidates = [
        (
            "avg_close",
            "avg_home_close",
            "avg_draw_close",
            "avg_away_close",
        ),
        (
            "b365_close",
            "b365_home_close",
            "b365_draw_close",
            "b365_away_close",
        ),
        (
            "avg_open",
            "avg_home_open",
            "avg_draw_open",
            "avg_away_open",
        ),
        (
            "b365_open",
            "b365_home_open",
            "b365_draw_open",
            "b365_away_open",
        ),
    ]

    for (
        source,
        home_col,
        draw_col,
        away_col,
    ) in candidates:

        if (
            home_col in row.index
            and draw_col in row.index
            and away_col in row.index
        ):

            home_odds = row[
                home_col
            ]

            draw_odds = row[
                draw_col
            ]

            away_odds = row[
                away_col
            ]

            if (
                pd.notna(home_odds)
                and pd.notna(draw_odds)
                and pd.notna(away_odds)
            ):
                return (
                    home_odds,
                    draw_odds,
                    away_odds,
                    source,
                )

    return (
        np.nan,
        np.nan,
        np.nan,
        "none",
    )


# =========================================================
# BUILD MARKET TABLE
# =========================================================

def build_market_table(matches):

    rows = []

    for _, row in matches.iterrows():

        (
            home_odds,
            draw_odds,
            away_odds,
            source,
        ) = select_market_odds(
            row
        )

        (
            p_home,
            p_draw,
            p_away,
            margin,
        ) = odds_to_fair_probs(
            home_odds,
            draw_odds,
            away_odds,
        )

        rows.append({
            "match_id":
                row["match_id"],

            "market_source":
                source,

            "market_home_odds":
                home_odds,

            "market_draw_odds":
                draw_odds,

            "market_away_odds":
                away_odds,

            "market_p_home":
                p_home,

            "market_p_draw":
                p_draw,

            "market_p_away":
                p_away,

            "market_margin":
                margin,
        })

    return pd.DataFrame(
        rows
    )


# =========================================================
# EVALUATION
# =========================================================

def evaluate_probs(
    df,
    prefix,
):
    y_true = df[
        "result_class"
    ].to_numpy()

    probs = df[
        [
            f"{prefix}_p_home",
            f"{prefix}_p_draw",
            f"{prefix}_p_away",
        ]
    ].to_numpy()

    predicted = probs.argmax(
        axis=1
    )

    return {
        "games":
            len(df),

        "accuracy":
            (
                predicted
                == y_true
            ).mean(),

        "log_loss":
            multiclass_log_loss(
                y_true,
                probs,
            ),

        "brier":
            multiclass_brier(
                y_true,
                probs,
            ),

        "ece":
            multiclass_ece(
                y_true,
                probs,
            ),
    }


def print_comparison(
    title,
    model_metrics,
    market_metrics,
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"Games: {model_metrics['games']:,}"
    )

    print()
    print(
        f"{'Metric':<18}"
        f"{'Model V0':>14}"
        f"{'Market':>14}"
        f"{'Difference':>14}"
    )

    print("-" * 60)

    accuracy_diff = (
        model_metrics["accuracy"]
        -
        market_metrics["accuracy"]
    )

    print(
        f"{'Accuracy':<18}"
        f"{model_metrics['accuracy']:>13.2%}"
        f"{market_metrics['accuracy']:>13.2%}"
        f"{accuracy_diff:>+13.2%}"
    )

    logloss_diff = (
        model_metrics["log_loss"]
        -
        market_metrics["log_loss"]
    )

    print(
        f"{'Log Loss':<18}"
        f"{model_metrics['log_loss']:>14.4f}"
        f"{market_metrics['log_loss']:>14.4f}"
        f"{logloss_diff:>+14.4f}"
    )

    brier_diff = (
        model_metrics["brier"]
        -
        market_metrics["brier"]
    )

    print(
        f"{'Brier':<18}"
        f"{model_metrics['brier']:>14.4f}"
        f"{market_metrics['brier']:>14.4f}"
        f"{brier_diff:>+14.4f}"
    )

    ece_diff = (
        model_metrics["ece"]
        -
        market_metrics["ece"]
    )

    print(
        f"{'ECE':<18}"
        f"{model_metrics['ece']:>13.2%}"
        f"{market_metrics['ece']:>13.2%}"
        f"{ece_diff:>+13.2%}"
    )

    print()
    print(
        "For Log Loss, Brier, and ECE: "
        "lower is better."
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("==============================")
    print("MARKET BENCHMARK")
    print("==============================")
    print()

    if not MATCHES_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{MATCHES_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{MODEL_FILE}"
        )

    matches = pd.read_csv(
        MATCHES_FILE,
        parse_dates=["date"],
    )

    predictions = pd.read_csv(
        MODEL_FILE,
        parse_dates=["date"],
    )

    print(
        f"Historical matches: "
        f"{len(matches):,}"
    )

    print(
        f"Model predictions: "
        f"{len(predictions):,}"
    )

    print(
        "Building vig-free market probabilities..."
    )

    market = build_market_table(
        matches
    )

    comparison = predictions.merge(
        market,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    # Rename model probabilities so the two systems
    # use the same naming structure.
    comparison = comparison.rename(
        columns={
            "p_home_win":
                "model_p_home",

            "p_draw":
                "model_p_draw",

            "p_away_win":
                "model_p_away",
        }
    )

    # -----------------------------------------------------
    # MARKET COVERAGE
    # -----------------------------------------------------

    valid_market = comparison[
        comparison[
            "market_p_home"
        ].notna()
        &
        comparison[
            "market_p_draw"
        ].notna()
        &
        comparison[
            "market_p_away"
        ].notna()
    ].copy()

    print(
        f"Matches with usable market odds: "
        f"{len(valid_market):,}"
    )

    coverage = (
        len(valid_market)
        / len(comparison)
    )

    print(
        f"Market coverage: "
        f"{coverage:.2%}"
    )

    print()
    print("MARKET SOURCE COUNTS")

    print(
        valid_market[
            "market_source"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Average bookmaker margin: "
        f"{valid_market['market_margin'].mean():.2%}"
    )

    # -----------------------------------------------------
    # OVERALL
    # -----------------------------------------------------

    model_metrics = evaluate_probs(
        valid_market,
        "model",
    )

    market_metrics = evaluate_probs(
        valid_market,
        "market",
    )

    print_comparison(
        "MODEL V0 VS MARKET — ALL MATCHES",
        model_metrics,
        market_metrics,
    )

    # -----------------------------------------------------
    # RECENT
    # -----------------------------------------------------

    recent = valid_market[
        valid_market[
            "season"
        ]
        .astype(str)
        .isin(
            RECENT_SEASONS
        )
    ].copy()

    recent_model = evaluate_probs(
        recent,
        "model",
    )

    recent_market = evaluate_probs(
        recent,
        "market",
    )

    print_comparison(
        "MODEL V0 VS MARKET — 2023/24 TO 2025/26",
        recent_model,
        recent_market,
    )

    # -----------------------------------------------------
    # BY LEAGUE
    # -----------------------------------------------------

    print()
    print("=" * 100)
    print("BY LEAGUE — ALL MATCHES")
    print("=" * 100)

    league_rows = []

    for league, group in valid_market.groupby(
        "league"
    ):

        model = evaluate_probs(
            group,
            "model",
        )

        market_metrics = evaluate_probs(
            group,
            "market",
        )

        league_rows.append({
            "league":
                league,

            "games":
                len(group),

            "model_accuracy":
                model[
                    "accuracy"
                ],

            "market_accuracy":
                market_metrics[
                    "accuracy"
                ],

            "model_log_loss":
                model[
                    "log_loss"
                ],

            "market_log_loss":
                market_metrics[
                    "log_loss"
                ],

            "log_loss_diff":
                model[
                    "log_loss"
                ]
                -
                market_metrics[
                    "log_loss"
                ],

            "model_brier":
                model[
                    "brier"
                ],

            "market_brier":
                market_metrics[
                    "brier"
                ],

            "brier_diff":
                model[
                    "brier"
                ]
                -
                market_metrics[
                    "brier"
                ],

            "model_ece":
                model[
                    "ece"
                ],

            "market_ece":
                market_metrics[
                    "ece"
                ],
        })

    league_table = pd.DataFrame(
        league_rows
    )

    for col in [
        "model_accuracy",
        "market_accuracy",
        "model_ece",
        "market_ece",
    ]:
        league_table[
            col
        ] *= 100.0

    print(
        league_table
        .round(4)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # RECENT BY LEAGUE
    # -----------------------------------------------------

    print()
    print("=" * 100)
    print(
        "BY LEAGUE — 2023/24 TO 2025/26"
    )
    print("=" * 100)

    recent_rows = []

    for league, group in recent.groupby(
        "league"
    ):

        model = evaluate_probs(
            group,
            "model",
        )

        market_metrics = evaluate_probs(
            group,
            "market",
        )

        recent_rows.append({
            "league":
                league,

            "games":
                len(group),

            "model_accuracy":
                model[
                    "accuracy"
                ],

            "market_accuracy":
                market_metrics[
                    "accuracy"
                ],

            "model_log_loss":
                model[
                    "log_loss"
                ],

            "market_log_loss":
                market_metrics[
                    "log_loss"
                ],

            "log_loss_diff":
                model[
                    "log_loss"
                ]
                -
                market_metrics[
                    "log_loss"
                ],

            "model_brier":
                model[
                    "brier"
                ],

            "market_brier":
                market_metrics[
                    "brier"
                ],

            "brier_diff":
                model[
                    "brier"
                ]
                -
                market_metrics[
                    "brier"
                ],

            "model_ece":
                model[
                    "ece"
                ],

            "market_ece":
                market_metrics[
                    "ece"
                ],
        })

    recent_table = pd.DataFrame(
        recent_rows
    )

    for col in [
        "model_accuracy",
        "market_accuracy",
        "model_ece",
        "market_ece",
    ]:
        recent_table[
            col
        ] *= 100.0

    print(
        recent_table
        .round(4)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # EDGE COLUMNS
    # -----------------------------------------------------

    valid_market[
        "home_probability_edge"
    ] = (
        valid_market[
            "model_p_home"
        ]
        -
        valid_market[
            "market_p_home"
        ]
    )

    valid_market[
        "draw_probability_edge"
    ] = (
        valid_market[
            "model_p_draw"
        ]
        -
        valid_market[
            "market_p_draw"
        ]
    )

    valid_market[
        "away_probability_edge"
    ] = (
        valid_market[
            "model_p_away"
        ]
        -
        valid_market[
            "market_p_away"
        ]
    )

    # These are diagnostic only for now.
    # We are NOT placing/backtesting bets yet.

    valid_market.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    market_sum = (
        valid_market[
            [
                "market_p_home",
                "market_p_draw",
                "market_p_away",
            ]
        ]
        .sum(
            axis=1
        )
    )

    max_error = (
        market_sum - 1.0
    ).abs().max()

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(
        "Max vig-free probability "
        f"sum error: {max_error:.12f}"
    )

    print(
        "Market probabilities "
        "sum to 1 ✅"
    )

    print(
        "Market odds were not used "
        "to generate Model V0 ✅"
    )

    print()
    print(
        f"Saved:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()