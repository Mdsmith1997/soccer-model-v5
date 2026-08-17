from pathlib import Path
import itertools

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
except ImportError as exc:
    raise ImportError(
        "\nscikit-learn is required for this calibration test.\n"
        "Install it inside the active venv with:\n\n"
        "pip install scikit-learn\n"
    ) from exc


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "frozen_v5_predictions.csv"
)

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "v5_calibration_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "v5_calibrated_predictions.csv"
)


# ============================================================
# DATA SPLITS
# ============================================================

CALIBRATION_FIT_SEASONS = {
    "1819",
    "1920",
    "2021",
}

TUNING_SEASONS = {
    "2122",
    "2223",
}

VALIDATION_SEASONS = {
    "2324",
}

FINAL_CHECK_SEASONS = {
    "2425",
}

LOCKED_TEST_SEASONS = {
    "2526",
}


# ============================================================
# SETTINGS
# ============================================================

EPS = 1e-12

# Temperature:
# T < 1 sharpens probabilities
# T > 1 softens probabilities
TEMPERATURES = np.round(
    np.arange(
        0.60,
        1.61,
        0.01,
    ),
    2,
)

LOGISTIC_C_VALUES = [
    0.03,
    0.10,
    0.30,
    1.00,
    3.00,
    10.00,
    30.00,
]

POWER_VALUES = [
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
]


# ============================================================
# BASIC HELPERS
# ============================================================

def season_string(
    series,
):

    return (
        series
        .astype(str)
        .str.zfill(4)
    )


def get_probs(
    df,
):

    probs = df[
        [
            "p_home_v5",
            "p_draw_v5",
            "p_away_v5",
        ]
    ].to_numpy(
        dtype=float
    )

    probs = np.clip(
        probs,
        EPS,
        1.0,
    )

    probs = (
        probs
        /
        probs.sum(
            axis=1,
            keepdims=True,
        )
    )

    return probs


def result_classes(
    df,
):

    home_goals = (
        df[
            "home_goals"
        ]
        .to_numpy()
    )

    away_goals = (
        df[
            "away_goals"
        ]
        .to_numpy()
    )

    return np.where(
        home_goals
        >
        away_goals,
        0,
        np.where(
            home_goals
            ==
            away_goals,
            1,
            2,
        ),
    )


# ============================================================
# METRICS
# ============================================================

def log_loss(
    y_true,
    probs,
):

    chosen = probs[
        np.arange(
            len(y_true)
        ),
        y_true,
    ]

    return float(
        -np.log(
            np.clip(
                chosen,
                EPS,
                1.0,
            )
        ).mean()
    )


def brier(
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

    return float(
        np.mean(
            np.sum(
                (
                    probs
                    - truth
                )
                ** 2,
                axis=1,
            )
        )
    )


def accuracy(
    y_true,
    probs,
):

    return float(
        (
            probs.argmax(
                axis=1
            )
            ==
            y_true
        ).mean()
    )


def expected_calibration_error(
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
        ==
        y_true
    ).astype(
        float
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0

    for i in range(
        bins
    ):

        lower = edges[i]
        upper = edges[i + 1]

        if i == bins - 1:

            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )

        n = mask.sum()

        if n == 0:
            continue

        bucket_conf = (
            confidence[
                mask
            ].mean()
        )

        bucket_acc = (
            correct[
                mask
            ].mean()
        )

        ece += (
            n
            /
            len(y_true)
            *
            abs(
                bucket_conf
                -
                bucket_acc
            )
        )

    return float(
        ece
    )


def evaluate(
    y_true,
    probs,
):

    return {
        "games":
            len(y_true),

        "accuracy":
            accuracy(
                y_true,
                probs,
            ),

        "log_loss":
            log_loss(
                y_true,
                probs,
            ),

        "brier":
            brier(
                y_true,
                probs,
            ),

        "ece":
            expected_calibration_error(
                y_true,
                probs,
            ),

        "avg_confidence":
            float(
                probs.max(
                    axis=1
                ).mean()
            ),
    }


# ============================================================
# RAW CALIBRATION
# ============================================================

class RawCalibrator:

    name = "RAW"

    def fit(
        self,
        probs,
        y,
    ):

        return self

    def transform(
        self,
        probs,
    ):

        return probs


# ============================================================
# TEMPERATURE CALIBRATION
# ============================================================

def temperature_transform(
    probs,
    temperature,
):

    logits = np.log(
        np.clip(
            probs,
            EPS,
            1.0,
        )
    )

    logits = (
        logits
        /
        temperature
    )

    logits = (
        logits
        -
        logits.max(
            axis=1,
            keepdims=True,
        )
    )

    output = np.exp(
        logits
    )

    output /= output.sum(
        axis=1,
        keepdims=True,
    )

    return output


class TemperatureCalibrator:

    name = "TEMPERATURE"

    def __init__(
        self,
    ):

        self.temperature = 1.0

    def fit(
        self,
        probs,
        y,
    ):

        rows = []

        for temperature in (
            TEMPERATURES
        ):

            calibrated = (
                temperature_transform(
                    probs,
                    temperature,
                )
            )

            rows.append(
                (
                    log_loss(
                        y,
                        calibrated,
                    ),
                    float(
                        temperature
                    ),
                )
            )

        rows.sort(
            key=lambda x:
                x[0]
        )

        self.temperature = (
            rows[
                0
            ][
                1
            ]
        )

        return self

    def transform(
        self,
        probs,
    ):

        return (
            temperature_transform(
                probs,
                self.temperature,
            )
        )


# ============================================================
# MULTINOMIAL LOGISTIC CALIBRATION
#
# Uses raw model probabilities as features.
# ============================================================

class MultinomialCalibrator:

    name = "MULTINOMIAL"

    def __init__(
        self,
        C=1.0,
    ):

        self.C = C
        self.model = None

    def fit(
        self,
        probs,
        y,
    ):

        self.model = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=5000,
        )

        self.model.fit(
            probs,
            y,
        )

        return self

    def transform(
        self,
        probs,
    ):

        return (
            self.model
            .predict_proba(
                probs
            )
        )


# ============================================================
# DIRICHLET-STYLE CALIBRATION
#
# Multinomial logistic regression using log probabilities.
# ============================================================

class DirichletCalibrator:

    name = "DIRICHLET"

    def __init__(
        self,
        C=1.0,
    ):

        self.C = C
        self.model = None

    def features(
        self,
        probs,
    ):

        return np.log(
            np.clip(
                probs,
                EPS,
                1.0,
            )
        )

    def fit(
        self,
        probs,
        y,
    ):

        X = self.features(
            probs
        )

        self.model = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=5000,
        )

        self.model.fit(
            X,
            y,
        )

        return self

    def transform(
        self,
        probs,
    ):

        X = self.features(
            probs
        )

        return (
            self.model
            .predict_proba(
                X
            )
        )


# ============================================================
# CLASSWISE POWER CALIBRATION
# ============================================================

def class_power_transform(
    probs,
    powers,
):

    power_array = np.array(
        powers,
        dtype=float,
    )

    output = (
        np.clip(
            probs,
            EPS,
            1.0,
        )
        **
        power_array[
            None,
            :
        ]
    )

    output /= output.sum(
        axis=1,
        keepdims=True,
    )

    return output


class ClassPowerCalibrator:

    name = "CLASS_POWER"

    def __init__(
        self,
    ):

        self.powers = (
            1.0,
            1.0,
            1.0,
        )

    def fit(
        self,
        probs,
        y,
    ):

        rows = []

        for powers in itertools.product(
            POWER_VALUES,
            repeat=3,
        ):

            calibrated = (
                class_power_transform(
                    probs,
                    powers,
                )
            )

            rows.append(
                (
                    log_loss(
                        y,
                        calibrated,
                    ),
                    powers,
                )
            )

        rows.sort(
            key=lambda x:
                x[0]
        )

        self.powers = (
            rows[
                0
            ][
                1
            ]
        )

        return self

    def transform(
        self,
        probs,
    ):

        return (
            class_power_transform(
                probs,
                self.powers,
            )
        )


# ============================================================
# BUILD DATASET
# ============================================================

def load_predictions():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{INPUT_FILE}\n\n"
            "Run backtest_frozen_v5.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "date",
        ],
    )

    df[
        "season"
    ] = season_string(
        df[
            "season"
        ]
    )

    required = [
        "p_home_v5",
        "p_draw_v5",
        "p_away_v5",
        "home_goals",
        "away_goals",
        "season",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + str(
                missing
            )
        )

    return df


def subset(
    df,
    seasons,
):

    return df[
        df[
            "season"
        ].isin(
            seasons
        )
    ].copy()


# ============================================================
# LOGISTIC C SELECTION
#
# Models fit only on calibration-fit seasons.
# C selected by tuning-period LL.
# ============================================================

def select_logistic_c(
    calibrator_class,
    fit_probs,
    fit_y,
    tune_probs,
    tune_y,
):

    rows = []

    for C in (
        LOGISTIC_C_VALUES
    ):

        calibrator = (
            calibrator_class(
                C=C
            )
        )

        calibrator.fit(
            fit_probs,
            fit_y,
        )

        tune_calibrated = (
            calibrator
            .transform(
                tune_probs
            )
        )

        rows.append(
            (
                log_loss(
                    tune_y,
                    tune_calibrated,
                ),
                brier(
                    tune_y,
                    tune_calibrated,
                ),
                C,
            )
        )

    rows.sort(
        key=lambda x:
            (
                x[0],
                x[1],
            )
    )

    return rows[
        0
    ][
        2
    ]


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_metrics_table(
    title,
    rows,
):

    print()
    print("=" * 95)
    print(title)
    print("=" * 95)

    table = pd.DataFrame(
        rows
    )

    display = table.copy()

    display[
        "accuracy"
    ] *= 100

    display[
        "ece"
    ] *= 100

    display[
        "avg_confidence"
    ] *= 100

    print(
        display[
            [
                "method",
                "games",
                "accuracy",
                "log_loss",
                "brier",
                "ece",
                "avg_confidence",
            ]
        ]
        .round(
            5
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# METHOD DESCRIPTION
# ============================================================

def calibrator_description(
    calibrator,
):

    if isinstance(
        calibrator,
        TemperatureCalibrator,
    ):

        return (
            f"T={calibrator.temperature:.3f}"
        )

    if isinstance(
        calibrator,
        MultinomialCalibrator,
    ):

        return (
            f"C={calibrator.C:g}"
        )

    if isinstance(
        calibrator,
        DirichletCalibrator,
    ):

        return (
            f"C={calibrator.C:g}"
        )

    if isinstance(
        calibrator,
        ClassPowerCalibrator,
    ):

        return (
            "powers="
            +
            "/".join(
                f"{x:.2f}"
                for x in calibrator.powers
            )
        )

    return "none"


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("CALIBRATING FROZEN V5")
    print("==============================")
    print()

    print(
        "Calibration fit: "
        "2018/19–2020/21"
    )

    print(
        "Method selection: "
        "2021/22–2022/23"
    )

    print(
        "Validation: 2023/24"
    )

    print(
        "Final check: 2024/25"
    )

    print(
        "Locked test: 2025/26"
    )

    # ========================================================
    # LOAD
    # ========================================================

    df = load_predictions()

    fit_df = subset(
        df,
        CALIBRATION_FIT_SEASONS,
    )

    tune_df = subset(
        df,
        TUNING_SEASONS,
    )

    validation_df = subset(
        df,
        VALIDATION_SEASONS,
    )

    final_df = subset(
        df,
        FINAL_CHECK_SEASONS,
    )

    locked_df = subset(
        df,
        LOCKED_TEST_SEASONS,
    )

    fit_probs = get_probs(
        fit_df
    )

    fit_y = result_classes(
        fit_df
    )

    tune_probs = get_probs(
        tune_df
    )

    tune_y = result_classes(
        tune_df
    )

    validation_probs = get_probs(
        validation_df
    )

    validation_y = result_classes(
        validation_df
    )

    final_probs = get_probs(
        final_df
    )

    final_y = result_classes(
        final_df
    )

    locked_probs = get_probs(
        locked_df
    )

    locked_y = result_classes(
        locked_df
    )

    print()
    print(
        f"Calibration-fit games: "
        f"{len(fit_df):,}"
    )

    print(
        f"Tuning games: "
        f"{len(tune_df):,}"
    )

    print(
        f"Validation games: "
        f"{len(validation_df):,}"
    )

    print(
        f"Final-check games: "
        f"{len(final_df):,}"
    )

    print(
        f"Locked-test games: "
        f"{len(locked_df):,}"
    )

    # ========================================================
    # SELECT REGULARIZATION FOR LOGISTIC METHODS
    # ========================================================

    multinomial_C = (
        select_logistic_c(
            MultinomialCalibrator,
            fit_probs,
            fit_y,
            tune_probs,
            tune_y,
        )
    )

    dirichlet_C = (
        select_logistic_c(
            DirichletCalibrator,
            fit_probs,
            fit_y,
            tune_probs,
            tune_y,
        )
    )

    print()
    print(
        f"Selected multinomial C: "
        f"{multinomial_C:g}"
    )

    print(
        f"Selected Dirichlet C: "
        f"{dirichlet_C:g}"
    )

    # ========================================================
    # FIT CANDIDATE CALIBRATORS
    # ========================================================

    candidates = [
        RawCalibrator(),

        TemperatureCalibrator(),

        MultinomialCalibrator(
            C=multinomial_C
        ),

        DirichletCalibrator(
            C=dirichlet_C
        ),

        ClassPowerCalibrator(),
    ]

    results = []

    fitted = {}

    for calibrator in candidates:

        calibrator.fit(
            fit_probs,
            fit_y,
        )

        name = calibrator.name

        fitted[
            name
        ] = calibrator

        tune_calibrated = (
            calibrator
            .transform(
                tune_probs
            )
        )

        tune_metrics = evaluate(
            tune_y,
            tune_calibrated,
        )

        results.append(
            {
                "method":
                    name,

                "parameters":
                    calibrator_description(
                        calibrator
                    ),

                "tuning_games":
                    tune_metrics[
                        "games"
                    ],

                "tuning_accuracy":
                    tune_metrics[
                        "accuracy"
                    ],

                "tuning_log_loss":
                    tune_metrics[
                        "log_loss"
                    ],

                "tuning_brier":
                    tune_metrics[
                        "brier"
                    ],

                "tuning_ece":
                    tune_metrics[
                        "ece"
                    ],
            }
        )

    results = pd.DataFrame(
        results
    )

    results = (
        results
        .sort_values(
            [
                "tuning_log_loss",
                "tuning_brier",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    results[
        "rank"
    ] = (
        np.arange(
            len(results)
        )
        + 1
    )

    # ========================================================
    # WINNER SELECTED HERE
    # ========================================================

    winning_method = (
        results.iloc[
            0
        ][
            "method"
        ]
    )

    winner = fitted[
        winning_method
    ]

    print()
    print("==============================")
    print("CALIBRATION SELECTION RESULTS")
    print("==============================")
    print()

    display = results.copy()

    display[
        "tuning_accuracy"
    ] *= 100

    display[
        "tuning_ece"
    ] *= 100

    print(
        display[
            [
                "rank",
                "method",
                "parameters",
                "tuning_games",
                "tuning_log_loss",
                "tuning_brier",
                "tuning_accuracy",
                "tuning_ece",
            ]
        ]
        .round(
            6
        )
        .to_string(
            index=False
        )
    )

    print()
    print("==============================")
    print("SELECTED CALIBRATOR")
    print("==============================")

    print(
        f"Method: "
        f"{winning_method}"
    )

    print(
        f"Parameters: "
        f"{calibrator_description(winner)}"
    )

    # ========================================================
    # COMPARE RAW + WINNER ON HELD-OUT PERIODS
    # ========================================================

    comparison_sets = [
        (
            "TUNING — 2021/22 TO 2022/23",
            tune_y,
            tune_probs,
        ),

        (
            "VALIDATION — 2023/24",
            validation_y,
            validation_probs,
        ),

        (
            "FINAL CHECK — 2024/25",
            final_y,
            final_probs,
        ),
    ]

    all_result_rows = []

    for title, y, raw_probs in (
        comparison_sets
    ):

        raw_metrics = evaluate(
            y,
            raw_probs,
        )

        calibrated_probs = (
            winner.transform(
                raw_probs
            )
        )

        calibrated_metrics = evaluate(
            y,
            calibrated_probs,
        )

        print()
        print("=" * 78)
        print(title)
        print("=" * 78)

        print(
            f"{'Metric':<16}"
            f"{'Raw V5':>14}"
            f"{'Calibrated':>14}"
            f"{'Change':>14}"
        )

        print("-" * 58)

        print(
            f"{'Accuracy':<16}"
            f"{raw_metrics['accuracy']:>13.2%}"
            f"{calibrated_metrics['accuracy']:>13.2%}"
            f"{calibrated_metrics['accuracy'] - raw_metrics['accuracy']:>+13.2%}"
        )

        print(
            f"{'Log Loss':<16}"
            f"{raw_metrics['log_loss']:>14.5f}"
            f"{calibrated_metrics['log_loss']:>14.5f}"
            f"{calibrated_metrics['log_loss'] - raw_metrics['log_loss']:>+14.5f}"
        )

        print(
            f"{'Brier':<16}"
            f"{raw_metrics['brier']:>14.5f}"
            f"{calibrated_metrics['brier']:>14.5f}"
            f"{calibrated_metrics['brier'] - raw_metrics['brier']:>+14.5f}"
        )

        print(
            f"{'ECE':<16}"
            f"{raw_metrics['ece']:>13.2%}"
            f"{calibrated_metrics['ece']:>13.2%}"
            f"{calibrated_metrics['ece'] - raw_metrics['ece']:>+13.2%}"
        )

        all_result_rows.append(
            {
                "sample":
                    title,

                "method":
                    winning_method,

                "raw_log_loss":
                    raw_metrics[
                        "log_loss"
                    ],

                "calibrated_log_loss":
                    calibrated_metrics[
                        "log_loss"
                    ],

                "raw_brier":
                    raw_metrics[
                        "brier"
                    ],

                "calibrated_brier":
                    calibrated_metrics[
                        "brier"
                    ],

                "raw_accuracy":
                    raw_metrics[
                        "accuracy"
                    ],

                "calibrated_accuracy":
                    calibrated_metrics[
                        "accuracy"
                    ],

                "raw_ece":
                    raw_metrics[
                        "ece"
                    ],

                "calibrated_ece":
                    calibrated_metrics[
                        "ece"
                    ],
            }
        )

    # ========================================================
    # LOCKED TEST
    #
    # IMPORTANT:
    # Winner already selected before this section.
    # We do NOT change method after seeing locked results.
    # ========================================================

    locked_raw_metrics = evaluate(
        locked_y,
        locked_probs,
    )

    locked_calibrated_probs = (
        winner.transform(
            locked_probs
        )
    )

    locked_cal_metrics = evaluate(
        locked_y,
        locked_calibrated_probs,
    )

    print()
    print("=" * 78)
    print("LOCKED TEST — 2025/26")
    print("=" * 78)

    print(
        f"{'Metric':<16}"
        f"{'Raw V5':>14}"
        f"{'Calibrated':>14}"
        f"{'Change':>14}"
    )

    print("-" * 58)

    print(
        f"{'Accuracy':<16}"
        f"{locked_raw_metrics['accuracy']:>13.2%}"
        f"{locked_cal_metrics['accuracy']:>13.2%}"
        f"{locked_cal_metrics['accuracy'] - locked_raw_metrics['accuracy']:>+13.2%}"
    )

    print(
        f"{'Log Loss':<16}"
        f"{locked_raw_metrics['log_loss']:>14.5f}"
        f"{locked_cal_metrics['log_loss']:>14.5f}"
        f"{locked_cal_metrics['log_loss'] - locked_raw_metrics['log_loss']:>+14.5f}"
    )

    print(
        f"{'Brier':<16}"
        f"{locked_raw_metrics['brier']:>14.5f}"
        f"{locked_cal_metrics['brier']:>14.5f}"
        f"{locked_cal_metrics['brier'] - locked_raw_metrics['brier']:>+14.5f}"
    )

    print(
        f"{'ECE':<16}"
        f"{locked_raw_metrics['ece']:>13.2%}"
        f"{locked_cal_metrics['ece']:>13.2%}"
        f"{locked_cal_metrics['ece'] - locked_raw_metrics['ece']:>+13.2%}"
    )

    all_result_rows.append(
        {
            "sample":
                "LOCKED TEST — 2025/26",

            "method":
                winning_method,

            "raw_log_loss":
                locked_raw_metrics[
                    "log_loss"
                ],

            "calibrated_log_loss":
                locked_cal_metrics[
                    "log_loss"
                ],

            "raw_brier":
                locked_raw_metrics[
                    "brier"
                ],

            "calibrated_brier":
                locked_cal_metrics[
                    "brier"
                ],

            "raw_accuracy":
                locked_raw_metrics[
                    "accuracy"
                ],

            "calibrated_accuracy":
                locked_cal_metrics[
                    "accuracy"
                ],

            "raw_ece":
                locked_raw_metrics[
                    "ece"
                ],

            "calibrated_ece":
                locked_cal_metrics[
                    "ece"
                ],
        }
    )

    # ========================================================
    # LOCKED TEST BY LEAGUE
    # ========================================================

    print()
    print("=" * 105)
    print("2025/26 LOCKED TEST — CALIBRATION BY LEAGUE")
    print("=" * 105)

    league_rows = []

    locked_copy = (
        locked_df.copy()
    )

    locked_copy[
        "cal_p_home"
    ] = (
        locked_calibrated_probs[
            :,
            0
        ]
    )

    locked_copy[
        "cal_p_draw"
    ] = (
        locked_calibrated_probs[
            :,
            1
        ]
    )

    locked_copy[
        "cal_p_away"
    ] = (
        locked_calibrated_probs[
            :,
            2
        ]
    )

    for league, sub in (
        locked_copy.groupby(
            "league"
        )
    ):

        raw = get_probs(
            sub
        )

        calibrated = sub[
            [
                "cal_p_home",
                "cal_p_draw",
                "cal_p_away",
            ]
        ].to_numpy()

        y = result_classes(
            sub
        )

        raw_metrics = evaluate(
            y,
            raw,
        )

        cal_metrics = evaluate(
            y,
            calibrated,
        )

        league_rows.append(
            {
                "league":
                    league,

                "games":
                    len(sub),

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
                    ]
                    * 100,

                "cal_ece":
                    cal_metrics[
                        "ece"
                    ]
                    * 100,
            }
        )

    league_table = pd.DataFrame(
        league_rows
    )

    print(
        league_table
        .round(
            5
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE CALIBRATED MASTER PREDICTIONS
    # ========================================================

    all_probs = get_probs(
        df
    )

    all_calibrated = (
        winner.transform(
            all_probs
        )
    )

    output = df.copy()

    output[
        "p_home_v5_cal"
    ] = (
        all_calibrated[
            :,
            0
        ]
    )

    output[
        "p_draw_v5_cal"
    ] = (
        all_calibrated[
            :,
            1
        ]
    )

    output[
        "p_away_v5_cal"
    ] = (
        all_calibrated[
            :,
            2
        ]
    )

    output[
        "calibration_method_v5"
    ] = winning_method

    output[
        "calibration_parameters_v5"
    ] = calibrator_description(
        winner
    )

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    result_table = pd.DataFrame(
        all_result_rows
    )

    result_table.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("==============================")
    print("V5 CALIBRATION COMPLETE")
    print("==============================")

    print(
        "Frozen V5 probabilities were "
        "not retuned ✅"
    )

    print(
        "Calibration fitted only on "
        "2018/19–2020/21 ✅"
    )

    print(
        "Calibration method selected "
        "using 2021/22–2022/23 only ✅"
    )

    print(
        "2023/24 and 2024/25 remained "
        "held out during selection ✅"
    )

    print(
        "2025/26 examined only after "
        "the calibrator was selected ✅"
    )

    print()
    print(
        "Results:"
    )

    print(
        OUTPUT_RESULTS
    )

    print()

    print(
        "Calibrated predictions:"
    )

    print(
        OUTPUT_PREDICTIONS
    )


if __name__ == "__main__":
    main()