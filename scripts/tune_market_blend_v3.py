from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

V2_FILE = (
    ROOT
    / "data"
    / "processed"
    / "shot_model_v2_predictions.csv"
)

MATCHES_FILE = (
    ROOT
    / "data"
    / "processed"
    / "matches.csv"
)

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "market_blend_v3_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "market_blend_v3_predictions.csv"
)


# =========================================================
# SPLITS
# =========================================================

BLEND_TUNING_SEASON = {
    "2324",
}

VALIDATION_SEASON = {
    "2425",
}

FINAL_SEASON = {
    "2526",
}


# =========================================================
# SETTINGS
# =========================================================

EPS = 1e-12

MODEL_WEIGHTS = np.round(
    np.arange(
        0.00,
        1.01,
        0.025,
    ),
    3,
)


# =========================================================
# HELPERS
# =========================================================

def result_classes(
    home_goals,
    away_goals,
):
    return np.where(
        home_goals > away_goals,
        0,
        np.where(
            home_goals == away_goals,
            1,
            2,
        ),
    )


def fair_probs_from_odds(
    home_odds,
    draw_odds,
    away_odds,
):
    """
    Convert decimal bookmaker odds to proportional
    vig-free probabilities.
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

    raw = np.array(
        [
            1.0 / home_odds,
            1.0 / draw_odds,
            1.0 / away_odds,
        ],
        dtype=float,
    )

    overround = raw.sum()

    fair = (
        raw
        / overround
    )

    margin = (
        overround - 1.0
    )

    return (
        fair[0],
        fair[1],
        fair[2],
        margin,
    )


# =========================================================
# OPENING MARKET
# =========================================================

def build_opening_market(
    matches,
):
    """
    Use opening prices only.

    Priority:
        1. average opening odds
        2. Bet365 opening odds

    Closing prices are deliberately excluded.
    """

    rows = []

    for _, row in matches.iterrows():

        avg_available = (
            "avg_home_open" in row.index
            and "avg_draw_open" in row.index
            and "avg_away_open" in row.index
            and pd.notna(
                row["avg_home_open"]
            )
            and pd.notna(
                row["avg_draw_open"]
            )
            and pd.notna(
                row["avg_away_open"]
            )
        )

        b365_available = (
            "b365_home_open" in row.index
            and "b365_draw_open" in row.index
            and "b365_away_open" in row.index
            and pd.notna(
                row["b365_home_open"]
            )
            and pd.notna(
                row["b365_draw_open"]
            )
            and pd.notna(
                row["b365_away_open"]
            )
        )

        if avg_available:

            source = "avg_open"

            home_odds = row[
                "avg_home_open"
            ]

            draw_odds = row[
                "avg_draw_open"
            ]

            away_odds = row[
                "avg_away_open"
            ]

        elif b365_available:

            source = "b365_open"

            home_odds = row[
                "b365_home_open"
            ]

            draw_odds = row[
                "b365_draw_open"
            ]

            away_odds = row[
                "b365_away_open"
            ]

        else:

            source = "none"

            home_odds = np.nan
            draw_odds = np.nan
            away_odds = np.nan

        (
            p_home,
            p_draw,
            p_away,
            margin,
        ) = fair_probs_from_odds(
            home_odds,
            draw_odds,
            away_odds,
        )

        rows.append({
            "match_id":
                row["match_id"],

            "opening_source":
                source,

            "opening_home_odds":
                home_odds,

            "opening_draw_odds":
                draw_odds,

            "opening_away_odds":
                away_odds,

            "opening_p_home":
                p_home,

            "opening_p_draw":
                p_draw,

            "opening_p_away":
                p_away,

            "opening_margin":
                margin,
        })

    return pd.DataFrame(
        rows
    )


# =========================================================
# LOGARITHMIC OPINION POOL
# =========================================================

def blend_probabilities(
    model_probs,
    market_probs,
    model_weight,
):
    """
    Geometric/log-opinion blend:

        p_final ∝
            model_probability ^ model_weight
            *
            market_probability ^ market_weight

    where:
        market_weight = 1 - model_weight

    model_weight = 0
        pure market

    model_weight = 1
        pure V2
    """

    market_weight = (
        1.0 - model_weight
    )

    model_probs = np.clip(
        model_probs,
        EPS,
        1.0,
    )

    market_probs = np.clip(
        market_probs,
        EPS,
        1.0,
    )

    blended = (
        (
            model_probs
            ** model_weight
        )
        *
        (
            market_probs
            ** market_weight
        )
    )

    blended /= blended.sum(
        axis=1,
        keepdims=True,
    )

    return blended


# =========================================================
# METRICS
# =========================================================

def log_loss(
    y_true,
    probs,
):
    selected = probs[
        np.arange(
            len(y_true)
        ),
        y_true,
    ]

    selected = np.clip(
        selected,
        EPS,
        1.0,
    )

    return (
        -np.log(
            selected
        )
    ).mean()


def brier_score(
    y_true,
    probs,
):
    truth = np.zeros_like(
        probs
    )

    truth[
        np.arange(
            len(y_true)
        ),
        y_true,
    ] = 1.0

    return np.mean(
        np.sum(
            (
                probs
                - truth
            ) ** 2,
            axis=1,
        )
    )


def accuracy_score(
    y_true,
    probs,
):
    return (
        probs.argmax(
            axis=1
        )
        == y_true
    ).mean()


def binary_ece(
    y_true,
    probs,
    bins=10,
):
    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    n = len(probs)
    ece = 0.0

    for i in range(
        bins
    ):

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

        confidence = (
            probs[
                mask
            ].mean()
        )

        actual = (
            y_true[
                mask
            ].mean()
        )

        ece += (
            count / n
        ) * abs(
            confidence
            - actual
        )

    return ece


def multiclass_ece(
    y_true,
    probs,
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
    )


def evaluate(
    y_true,
    probs,
):
    return {
        "games":
            len(y_true),

        "accuracy":
            accuracy_score(
                y_true,
                probs,
            ),

        "log_loss":
            log_loss(
                y_true,
                probs,
            ),

        "brier":
            brier_score(
                y_true,
                probs,
            ),

        "ece":
            multiclass_ece(
                y_true,
                probs,
            ),
    }


# =========================================================
# PROBABILITY MATRICES
# =========================================================

def model_matrix(
    df,
):
    return df[
        [
            "p_home_v2",
            "p_draw_v2",
            "p_away_v2",
        ]
    ].to_numpy(
        dtype=float
    )


def market_matrix(
    df,
):
    return df[
        [
            "opening_p_home",
            "opening_p_draw",
            "opening_p_away",
        ]
    ].to_numpy(
        dtype=float
    )


# =========================================================
# DISPLAY
# =========================================================

def print_three_way(
    title,
    y,
    model_probs,
    market_probs,
    v3_probs,
):

    model = evaluate(
        y,
        model_probs,
    )

    market = evaluate(
        y,
        market_probs,
    )

    v3 = evaluate(
        y,
        v3_probs,
    )

    print()
    print("=" * 88)
    print(title)
    print("=" * 88)

    print(
        f"Games: {len(y):,}"
    )

    print()
    print(
        f"{'Metric':<18}"
        f"{'V2':>14}"
        f"{'Open Market':>14}"
        f"{'V3':>14}"
    )

    print("-" * 60)

    print(
        f"{'Accuracy':<18}"
        f"{model['accuracy']:>13.2%}"
        f"{market['accuracy']:>13.2%}"
        f"{v3['accuracy']:>13.2%}"
    )

    print(
        f"{'Log Loss':<18}"
        f"{model['log_loss']:>14.5f}"
        f"{market['log_loss']:>14.5f}"
        f"{v3['log_loss']:>14.5f}"
    )

    print(
        f"{'Brier':<18}"
        f"{model['brier']:>14.5f}"
        f"{market['brier']:>14.5f}"
        f"{v3['brier']:>14.5f}"
    )

    print(
        f"{'ECE':<18}"
        f"{model['ece']:>13.2%}"
        f"{market['ece']:>13.2%}"
        f"{v3['ece']:>13.2%}"
    )

    print()
    print(
        f"V3 LL vs market: "
        f"{v3['log_loss'] - market['log_loss']:+.5f}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("==============================")
    print("TUNING MARKET BLEND V3")
    print("==============================")
    print()

    if not V2_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{V2_FILE}"
        )

    if not MATCHES_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{MATCHES_FILE}"
        )

    v2 = pd.read_csv(
        V2_FILE,
        parse_dates=[
            "date",
        ],
    )

    matches = pd.read_csv(
        MATCHES_FILE,
        parse_dates=[
            "date",
        ],
    )

    v2[
        "season"
    ] = (
        v2[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    print(
        f"V2 predictions: "
        f"{len(v2):,}"
    )

    print(
        f"Historical matches: "
        f"{len(matches):,}"
    )

    # =====================================================
    # BUILD OPENING MARKET
    # =====================================================

    print()
    print(
        "Building vig-free "
        "OPENING market probabilities..."
    )

    opening = build_opening_market(
        matches
    )

    df = v2.merge(
        opening,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    usable = df[
        df[
            "opening_p_home"
        ].notna()
        &
        df[
            "opening_p_draw"
        ].notna()
        &
        df[
            "opening_p_away"
        ].notna()
    ].copy()

    usable[
        "result_class"
    ] = result_classes(
        usable[
            "home_goals"
        ].to_numpy(),
        usable[
            "away_goals"
        ].to_numpy(),
    )

    print(
        f"Matches with opening market: "
        f"{len(usable):,}"
    )

    print(
        f"Coverage: "
        f"{len(usable) / len(v2):.2%}"
    )

    print()
    print("OPENING SOURCE COUNTS")

    print(
        usable[
            "opening_source"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Average opening margin: "
        f"{usable['opening_margin'].mean():.2%}"
    )

    # =====================================================
    # TUNING SAMPLE
    # =====================================================

    tuning = usable[
        usable[
            "season"
        ].isin(
            BLEND_TUNING_SEASON
        )
    ].copy()

    print()
    print(
        "Blend tuning season: "
        "2023/24"
    )

    print(
        f"Tuning games: "
        f"{len(tuning):,}"
    )

    y_tune = tuning[
        "result_class"
    ].to_numpy()

    model_tune = model_matrix(
        tuning
    )

    market_tune = market_matrix(
        tuning
    )

    results = []

    for model_weight in MODEL_WEIGHTS:

        probs = blend_probabilities(
            model_tune,
            market_tune,
            model_weight,
        )

        metrics = evaluate(
            y_tune,
            probs,
        )

        results.append({
            "model_weight":
                model_weight,

            "market_weight":
                (
                    1.0
                    - model_weight
                ),

            "games":
                metrics[
                    "games"
                ],

            "accuracy":
                metrics[
                    "accuracy"
                ],

            "log_loss":
                metrics[
                    "log_loss"
                ],

            "brier":
                metrics[
                    "brier"
                ],

            "ece":
                metrics[
                    "ece"
                ],
        })

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            [
                "log_loss",
                "brier",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    results_df[
        "rank"
    ] = (
        np.arange(
            len(results_df)
        )
        + 1
    )

    results_df.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    # =====================================================
    # TOP RESULTS
    # =====================================================

    print()
    print("==============================")
    print("TOP 15 BLENDS")
    print("==============================")

    display = (
        results_df
        .head(15)
        .copy()
    )

    display[
        "accuracy"
    ] *= 100.0

    display[
        "ece"
    ] *= 100.0

    print(
        display[
            [
                "rank",
                "model_weight",
                "market_weight",
                "games",
                "log_loss",
                "brier",
                "accuracy",
                "ece",
            ]
        ]
        .round(6)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # WINNER
    # =====================================================

    best = results_df.iloc[
        0
    ]

    best_model_weight = float(
        best[
            "model_weight"
        ]
    )

    best_market_weight = (
        1.0
        - best_model_weight
    )

    print()
    print("==============================")
    print("WINNING V3 BLEND")
    print("==============================")

    print(
        f"V2 weight:      "
        f"{best_model_weight:.1%}"
    )

    print(
        f"Market weight:  "
        f"{best_market_weight:.1%}"
    )

    print(
        f"Tuning LL:      "
        f"{best['log_loss']:.5f}"
    )

    # =====================================================
    # PURE MARKET / PURE MODEL TUNING REFERENCE
    # =====================================================

    pure_model_metrics = evaluate(
        y_tune,
        model_tune,
    )

    pure_market_metrics = evaluate(
        y_tune,
        market_tune,
    )

    print()
    print(
        f"Pure V2 tuning LL:     "
        f"{pure_model_metrics['log_loss']:.5f}"
    )

    print(
        f"Pure market tuning LL: "
        f"{pure_market_metrics['log_loss']:.5f}"
    )

    # =====================================================
    # APPLY V3 TO ALL MATCHES
    # =====================================================

    all_model = model_matrix(
        usable
    )

    all_market = market_matrix(
        usable
    )

    all_v3 = blend_probabilities(
        all_model,
        all_market,
        best_model_weight,
    )

    usable[
        "p_home_v3"
    ] = all_v3[
        :,
        0,
    ]

    usable[
        "p_draw_v3"
    ] = all_v3[
        :,
        1,
    ]

    usable[
        "p_away_v3"
    ] = all_v3[
        :,
        2,
    ]

    usable[
        "v3_model_weight"
    ] = best_model_weight

    usable[
        "v3_market_weight"
    ] = best_market_weight

    # =====================================================
    # 2024/25 VALIDATION
    # =====================================================

    validation = usable[
        usable[
            "season"
        ].isin(
            VALIDATION_SEASON
        )
    ].copy()

    y_validation = validation[
        "result_class"
    ].to_numpy()

    v3_validation = validation[
        [
            "p_home_v3",
            "p_draw_v3",
            "p_away_v3",
        ]
    ].to_numpy()

    print_three_way(
        "2024/25 VALIDATION",
        y_validation,
        model_matrix(
            validation
        ),
        market_matrix(
            validation
        ),
        v3_validation,
    )

    # =====================================================
    # 2025/26 FINAL CHECK
    # =====================================================

    final = usable[
        usable[
            "season"
        ].isin(
            FINAL_SEASON
        )
    ].copy()

    y_final = final[
        "result_class"
    ].to_numpy()

    v3_final = final[
        [
            "p_home_v3",
            "p_draw_v3",
            "p_away_v3",
        ]
    ].to_numpy()

    print_three_way(
        "2025/26 FINAL CHECK",
        y_final,
        model_matrix(
            final
        ),
        market_matrix(
            final
        ),
        v3_final,
    )

    # =====================================================
    # FINAL BY LEAGUE
    # =====================================================

    print()
    print("=" * 125)
    print("2025/26 — BY LEAGUE")
    print("=" * 125)

    league_rows = []

    for league, group in final.groupby(
        "league"
    ):

        y = group[
            "result_class"
        ].to_numpy()

        v2_metrics = evaluate(
            y,
            model_matrix(
                group
            ),
        )

        market_metrics = evaluate(
            y,
            market_matrix(
                group
            ),
        )

        v3_probs = group[
            [
                "p_home_v3",
                "p_draw_v3",
                "p_away_v3",
            ]
        ].to_numpy()

        v3_metrics = evaluate(
            y,
            v3_probs,
        )

        league_rows.append({
            "league":
                league,

            "games":
                len(group),

            "v2_ll":
                v2_metrics[
                    "log_loss"
                ],

            "market_ll":
                market_metrics[
                    "log_loss"
                ],

            "v3_ll":
                v3_metrics[
                    "log_loss"
                ],

            "v3_vs_market":
                (
                    v3_metrics[
                        "log_loss"
                    ]
                    -
                    market_metrics[
                        "log_loss"
                    ]
                ),

            "v2_brier":
                v2_metrics[
                    "brier"
                ],

            "market_brier":
                market_metrics[
                    "brier"
                ],

            "v3_brier":
                v3_metrics[
                    "brier"
                ],

            "v2_acc":
                v2_metrics[
                    "accuracy"
                ],

            "market_acc":
                market_metrics[
                    "accuracy"
                ],

            "v3_acc":
                v3_metrics[
                    "accuracy"
                ],
        })

    league_table = pd.DataFrame(
        league_rows
    )

    for col in [
        "v2_acc",
        "market_acc",
        "v3_acc",
    ]:
        league_table[
            col
        ] *= 100.0

    print(
        league_table
        .round(5)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # SAVE
    # =====================================================

    usable.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    sums = usable[
        [
            "p_home_v3",
            "p_draw_v3",
            "p_away_v3",
        ]
    ].sum(
        axis=1
    )

    print()
    print("==============================")
    print("VALIDATION")
    print("==============================")

    print(
        "Max V3 probability "
        "sum error: "
        f"{(sums - 1.0).abs().max():.12f}"
    )

    print(
        "V3 probabilities sum to 1 ✅"
    )

    print(
        "Only opening market odds "
        "used as V3 market prior ✅"
    )

    print(
        "Closing prices were not used "
        "to generate V3 ✅"
    )

    print(
        "Blend selected using "
        "2023/24 only ✅"
    )

    print(
        "2025/26 was not used "
        "to select blend weight ✅"
    )

    print()
    print(
        f"Tuning results:"
        f"\n{OUTPUT_RESULTS}"
    )

    print()
    print(
        f"V3 predictions:"
        f"\n{OUTPUT_PREDICTIONS}"
    )


if __name__ == "__main__":
    main()