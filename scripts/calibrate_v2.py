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

MARKET_FILE = (
    ROOT
    / "data"
    / "processed"
    / "v2_market_comparison.csv"
)

OUTPUT_TUNING = (
    ROOT
    / "data"
    / "processed"
    / "v2_calibration_tuning.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "v2_calibrated_predictions.csv"
)


# =========================================================
# SPLITS
# =========================================================

CALIBRATION_SEASON = {
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

TEMPERATURES = np.round(
    np.arange(
        0.60,
        1.61,
        0.01,
    ),
    2,
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


def probability_matrix(
    df,
    prefix,
):
    return df[
        [
            f"{prefix}_p_home",
            f"{prefix}_p_draw",
            f"{prefix}_p_away",
        ]
    ].to_numpy(
        dtype=float
    )


# =========================================================
# TEMPERATURE SCALING
# =========================================================

def temperature_scale(
    probs,
    temperature,
):
    """
    Multiclass temperature scaling performed directly
    from probabilities.

    Equivalent to dividing logits by T:

        calibrated_i ∝ p_i ** (1 / T)

    T > 1:
        softer / less confident

    T < 1:
        sharper / more confident

    T = 1:
        unchanged
    """

    probs = np.clip(
        probs,
        EPS,
        1.0,
    )

    scaled = (
        probs
        ** (
            1.0
            / temperature
        )
    )

    scaled /= scaled.sum(
        axis=1,
        keepdims=True,
    )

    return scaled


# =========================================================
# METRICS
# =========================================================

def log_loss(
    y_true,
    probs,
):
    probs = np.clip(
        probs,
        EPS,
        1.0,
    )

    selected = probs[
        np.arange(
            len(y_true)
        ),
        y_true,
    ]

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
    total = len(probs)

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

        avg_confidence = (
            probs[
                mask
            ].mean()
        )

        avg_actual = (
            y_true[
                mask
            ].mean()
        )

        ece += (
            count
            / total
        ) * abs(
            avg_confidence
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
# DISPLAY
# =========================================================

def print_raw_vs_calibrated(
    title,
    raw,
    calibrated,
):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    print(
        f"Games: "
        f"{raw['games']:,}"
    )

    print()
    print(
        f"{'Metric':<18}"
        f"{'Raw V2':>14}"
        f"{'Calibrated':>14}"
        f"{'Change':>14}"
    )

    print("-" * 60)

    print(
        f"{'Accuracy':<18}"
        f"{raw['accuracy']:>13.2%}"
        f"{calibrated['accuracy']:>13.2%}"
        f"{calibrated['accuracy'] - raw['accuracy']:>+13.2%}"
    )

    print(
        f"{'Log Loss':<18}"
        f"{raw['log_loss']:>14.5f}"
        f"{calibrated['log_loss']:>14.5f}"
        f"{calibrated['log_loss'] - raw['log_loss']:>+14.5f}"
    )

    print(
        f"{'Brier':<18}"
        f"{raw['brier']:>14.5f}"
        f"{calibrated['brier']:>14.5f}"
        f"{calibrated['brier'] - raw['brier']:>+14.5f}"
    )

    print(
        f"{'ECE':<18}"
        f"{raw['ece']:>13.2%}"
        f"{calibrated['ece']:>13.2%}"
        f"{calibrated['ece'] - raw['ece']:>+13.2%}"
    )


def print_market_comparison(
    title,
    calibrated,
    market,
):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    print(
        f"Games: "
        f"{calibrated['games']:,}"
    )

    print()
    print(
        f"{'Metric':<18}"
        f"{'Cal V2':>14}"
        f"{'Market':>14}"
        f"{'Gap':>14}"
    )

    print("-" * 60)

    print(
        f"{'Accuracy':<18}"
        f"{calibrated['accuracy']:>13.2%}"
        f"{market['accuracy']:>13.2%}"
        f"{calibrated['accuracy'] - market['accuracy']:>+13.2%}"
    )

    print(
        f"{'Log Loss':<18}"
        f"{calibrated['log_loss']:>14.5f}"
        f"{market['log_loss']:>14.5f}"
        f"{calibrated['log_loss'] - market['log_loss']:>+14.5f}"
    )

    print(
        f"{'Brier':<18}"
        f"{calibrated['brier']:>14.5f}"
        f"{market['brier']:>14.5f}"
        f"{calibrated['brier'] - market['brier']:>+14.5f}"
    )

    print(
        f"{'ECE':<18}"
        f"{calibrated['ece']:>13.2%}"
        f"{market['ece']:>13.2%}"
        f"{calibrated['ece'] - market['ece']:>+13.2%}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("==============================")
    print("CALIBRATING SOCCER MODEL V2")
    print("==============================")
    print()

    if not V2_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{V2_FILE}"
        )

    v2 = pd.read_csv(
        V2_FILE,
        parse_dates=["date"],
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

    # -----------------------------------------------------
    # STANDARDIZE RAW PROBABILITY NAMES
    # -----------------------------------------------------

    v2[
        "raw_p_home"
    ] = v2[
        "p_home_v2"
    ]

    v2[
        "raw_p_draw"
    ] = v2[
        "p_draw_v2"
    ]

    v2[
        "raw_p_away"
    ] = v2[
        "p_away_v2"
    ]

    v2[
        "result_class"
    ] = result_classes(
        v2[
            "home_goals"
        ].to_numpy(),
        v2[
            "away_goals"
        ].to_numpy(),
    )

    print(
        f"V2 predictions loaded: "
        f"{len(v2):,}"
    )

    # =====================================================
    # CALIBRATION FIT
    # =====================================================

    calibration = v2[
        v2[
            "season"
        ].isin(
            CALIBRATION_SEASON
        )
    ].copy()

    print()
    print(
        "Calibration fit season: "
        "2023/24"
    )

    print(
        f"Calibration games: "
        f"{len(calibration):,}"
    )

    raw_cal_probs = probability_matrix(
        calibration,
        "raw",
    )

    y_cal = calibration[
        "result_class"
    ].to_numpy()

    # -----------------------------------------------------
    # TEMPERATURE GRID
    # -----------------------------------------------------

    tuning_rows = []

    for temperature in TEMPERATURES:

        scaled = temperature_scale(
            raw_cal_probs,
            temperature,
        )

        metrics = evaluate(
            y_cal,
            scaled,
        )

        tuning_rows.append({
            "temperature":
                temperature,

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

    tuning = pd.DataFrame(
        tuning_rows
    )

    tuning = (
        tuning
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

    tuning[
        "rank"
    ] = (
        np.arange(
            len(tuning)
        )
        + 1
    )

    tuning.to_csv(
        OUTPUT_TUNING,
        index=False,
    )

    # =====================================================
    # TOP TEMPERATURES
    # =====================================================

    print()
    print("==============================")
    print("TOP 15 TEMPERATURES")
    print("==============================")

    display = (
        tuning
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
                "temperature",
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

    best_temperature = float(
        tuning.iloc[
            0
        ][
            "temperature"
        ]
    )

    print()
    print("==============================")
    print("WINNING TEMPERATURE")
    print("==============================")

    print(
        f"T = "
        f"{best_temperature:.2f}"
    )

    if best_temperature > 1:

        print(
            "Interpretation: "
            "V2 was too confident."
        )

    elif best_temperature < 1:

        print(
            "Interpretation: "
            "V2 was too conservative."
        )

    else:

        print(
            "Interpretation: "
            "No scaling was preferred."
        )

    # =====================================================
    # APPLY TO ALL V2 MATCHES
    # =====================================================

    all_raw_probs = probability_matrix(
        v2,
        "raw",
    )

    all_cal_probs = temperature_scale(
        all_raw_probs,
        best_temperature,
    )

    v2[
        "cal_p_home"
    ] = all_cal_probs[
        :,
        0,
    ]

    v2[
        "cal_p_draw"
    ] = all_cal_probs[
        :,
        1,
    ]

    v2[
        "cal_p_away"
    ] = all_cal_probs[
        :,
        2,
    ]

    v2[
        "temperature"
    ] = best_temperature

    # =====================================================
    # VALIDATION 2024/25
    # =====================================================

    validation = v2[
        v2[
            "season"
        ].isin(
            VALIDATION_SEASON
        )
    ].copy()

    y_validation = validation[
        "result_class"
    ].to_numpy()

    raw_validation = evaluate(
        y_validation,
        probability_matrix(
            validation,
            "raw",
        ),
    )

    cal_validation = evaluate(
        y_validation,
        probability_matrix(
            validation,
            "cal",
        ),
    )

    print_raw_vs_calibrated(
        "2024/25 VALIDATION",
        raw_validation,
        cal_validation,
    )

    # =====================================================
    # FINAL 2025/26
    # =====================================================

    final = v2[
        v2[
            "season"
        ].isin(
            FINAL_SEASON
        )
    ].copy()

    y_final = final[
        "result_class"
    ].to_numpy()

    raw_final = evaluate(
        y_final,
        probability_matrix(
            final,
            "raw",
        ),
    )

    cal_final = evaluate(
        y_final,
        probability_matrix(
            final,
            "cal",
        ),
    )

    print_raw_vs_calibrated(
        "2025/26 FINAL CHECK",
        raw_final,
        cal_final,
    )

    # =====================================================
    # FINAL BY LEAGUE
    # =====================================================

    print()
    print("=" * 105)
    print("2025/26 CALIBRATION — BY LEAGUE")
    print("=" * 105)

    league_rows = []

    for league, group in final.groupby(
        "league"
    ):

        y = group[
            "result_class"
        ].to_numpy()

        raw_metrics = evaluate(
            y,
            probability_matrix(
                group,
                "raw",
            ),
        )

        cal_metrics = evaluate(
            y,
            probability_matrix(
                group,
                "cal",
            ),
        )

        league_rows.append({
            "league":
                league,

            "games":
                len(group),

            "raw_ll":
                raw_metrics[
                    "log_loss"
                ],

            "cal_ll":
                cal_metrics[
                    "log_loss"
                ],

            "ll_change":
                (
                    cal_metrics[
                        "log_loss"
                    ]
                    -
                    raw_metrics[
                        "log_loss"
                    ]
                ),

            "raw_brier":
                raw_metrics[
                    "brier"
                ],

            "cal_brier":
                cal_metrics[
                    "brier"
                ],

            "raw_ece":
                raw_metrics[
                    "ece"
                ],

            "cal_ece":
                cal_metrics[
                    "ece"
                ],

            "raw_acc":
                raw_metrics[
                    "accuracy"
                ],

            "cal_acc":
                cal_metrics[
                    "accuracy"
                ],
        })

    league_table = pd.DataFrame(
        league_rows
    )

    for col in [
        "raw_ece",
        "cal_ece",
        "raw_acc",
        "cal_acc",
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
    # MARKET BENCHMARK
    # =====================================================

    if MARKET_FILE.exists():

        print()
        print(
            "Loading market benchmark..."
        )

        market = pd.read_csv(
            MARKET_FILE,
            parse_dates=["date"],
        )

        market[
            "season"
        ] = (
            market[
                "season"
            ]
            .astype(str)
            .str.zfill(4)
        )

        market_keep = market[
            [
                "match_id",
                "market_source",
                "market_p_home",
                "market_p_draw",
                "market_p_away",
            ]
        ].copy()

        final_market = final.merge(
            market_keep,
            on="match_id",
            how="inner",
            validate="one_to_one",
        )

        final_market = final_market[
            final_market[
                "market_p_home"
            ].notna()
            &
            final_market[
                "market_p_draw"
            ].notna()
            &
            final_market[
                "market_p_away"
            ].notna()
        ].copy()

        y_market = final_market[
            "result_class"
        ].to_numpy()

        calibrated_market_subset = evaluate(
            y_market,
            probability_matrix(
                final_market,
                "cal",
            ),
        )

        market_metrics = evaluate(
            y_market,
            probability_matrix(
                final_market,
                "market",
            ),
        )

        print_market_comparison(
            "2025/26 — CALIBRATED V2 VS MARKET",
            calibrated_market_subset,
            market_metrics,
        )

        # -------------------------------------------------
        # CLOSING ONLY
        # -------------------------------------------------

        closing = final_market[
            final_market[
                "market_source"
            ]
            == "avg_close"
        ].copy()

        if len(closing) > 0:

            y_close = closing[
                "result_class"
            ].to_numpy()

            cal_close = evaluate(
                y_close,
                probability_matrix(
                    closing,
                    "cal",
                ),
            )

            market_close = evaluate(
                y_close,
                probability_matrix(
                    closing,
                    "market",
                ),
            )

            print_market_comparison(
                "2025/26 — CALIBRATED V2 VS CLOSING MARKET",
                cal_close,
                market_close,
            )

    # =====================================================
    # SAVE
    # =====================================================

    v2.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    raw_sum = (
        v2[
            [
                "raw_p_home",
                "raw_p_draw",
                "raw_p_away",
            ]
        ]
        .sum(
            axis=1
        )
    )

    cal_sum = (
        v2[
            [
                "cal_p_home",
                "cal_p_draw",
                "cal_p_away",
            ]
        ]
        .sum(
            axis=1
        )
    )

    print()
    print("==============================")
    print("VALIDATION")
    print("==============================")

    print(
        "Max raw probability sum error: "
        f"{(raw_sum - 1).abs().max():.12f}"
    )

    print(
        "Max calibrated probability "
        "sum error: "
        f"{(cal_sum - 1).abs().max():.12f}"
    )

    print(
        "Probabilities sum to 1 ✅"
    )

    print(
        "Market probabilities were not "
        "used to fit temperature ✅"
    )

    print(
        "Temperature selected using "
        "2023/24 only ✅"
    )

    print(
        "2025/26 was not used to "
        "select temperature ✅"
    )

    print()
    print(
        f"Temperature search saved:"
        f"\n{OUTPUT_TUNING}"
    )

    print()
    print(
        f"Calibrated predictions saved:"
        f"\n{OUTPUT_PREDICTIONS}"
    )


if __name__ == "__main__":
    main()