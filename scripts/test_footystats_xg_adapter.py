from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"

INPUT_FILE = (
    PROCESSED
    / "footystats_understat_xg_history_matched.csv"
)

OUTPUT_RESULTS = (
    PROCESSED
    / "footystats_xg_adapter_results.csv"
)

OUTPUT_PREDICTIONS = (
    PROCESSED
    / "footystats_xg_adapter_predictions.csv"
)


# ============================================================
# SPLITS
# ============================================================

TRAIN_SEASONS = {
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

FINAL_SEASONS = {
    "2425",
}


# ============================================================
# HELPERS
# ============================================================

def corr(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    if (
        len(a) < 2
        or
        np.std(a) == 0
        or
        np.std(b) == 0
    ):
        return np.nan

    return np.corrcoef(
        a,
        b,
    )[0, 1]


def mae(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    return np.mean(
        np.abs(
            a - b
        )
    )


def rmse(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    return np.sqrt(
        np.mean(
            (
                a - b
            ) ** 2
        )
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            INPUT_FILE
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required = [
        "season",
        "date",
        "home_xg_fs",
        "away_xg_fs",
        "home_xg_us",
        "away_xg_us",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    df[
        "season"
    ] = (
        df[
            "season"
        ]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
    )

    for col in [
        "home_xg_fs",
        "away_xg_fs",
        "home_xg_us",
        "away_xg_us",
    ]:

        df[
            col
        ] = pd.to_numeric(
            df[
                col
            ],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "home_xg_fs",
            "away_xg_fs",
            "home_xg_us",
            "away_xg_us",
        ]
    ).copy()

    return df


# ============================================================
# QUANTILE MAPPING
# ============================================================

def build_quantile_mapper(
    x_train,
    y_train,
    quantiles=101,
):

    probs = np.linspace(
        0.0,
        1.0,
        quantiles,
    )

    x_q = np.quantile(
        x_train,
        probs,
    )

    y_q = np.quantile(
        y_train,
        probs,
    )

    # Remove duplicate x quantiles so np.interp is stable.
    unique_x = []

    unique_y = []

    last_x = None

    for x_val, y_val in zip(
        x_q,
        y_q,
    ):

        if (
            last_x is None
            or
            x_val != last_x
        ):

            unique_x.append(
                x_val
            )

            unique_y.append(
                y_val
            )

            last_x = x_val

    unique_x = np.asarray(
        unique_x,
        dtype=float,
    )

    unique_y = np.asarray(
        unique_y,
        dtype=float,
    )

    def mapper(
        values,
    ):

        values = np.asarray(
            values,
            dtype=float,
        )

        return np.interp(
            values,
            unique_x,
            unique_y,
            left=unique_y[0],
            right=unique_y[-1],
        )

    return mapper


# ============================================================
# METHOD BUILDERS
# ============================================================

def apply_raw(
    train,
    target,
):

    out = target.copy()

    out[
        "pred_home_xg"
    ] = out[
        "home_xg_fs"
    ]

    out[
        "pred_away_xg"
    ] = out[
        "away_xg_fs"
    ]

    return out


def fit_global_linear(
    train,
):

    x = np.concatenate(
        [
            train[
                "home_xg_fs"
            ].values,

            train[
                "away_xg_fs"
            ].values,
        ]
    ).reshape(
        -1,
        1,
    )

    y = np.concatenate(
        [
            train[
                "home_xg_us"
            ].values,

            train[
                "away_xg_us"
            ].values,
        ]
    )

    model = LinearRegression()

    model.fit(
        x,
        y,
    )

    return model


def apply_global_linear(
    model,
    target,
):

    out = target.copy()

    out[
        "pred_home_xg"
    ] = model.predict(
        out[
            [
                "home_xg_fs"
            ]
        ]
    )

    out[
        "pred_away_xg"
    ] = model.predict(
        out[
            [
                "away_xg_fs"
            ]
        ]
    )

    return out


def fit_side_linear(
    train,
):

    home_model = LinearRegression()

    away_model = LinearRegression()

    home_model.fit(
        train[
            [
                "home_xg_fs"
            ]
        ],
        train[
            "home_xg_us"
        ],
    )

    away_model.fit(
        train[
            [
                "away_xg_fs"
            ]
        ],
        train[
            "away_xg_us"
        ],
    )

    return (
        home_model,
        away_model,
    )


def apply_side_linear(
    models,
    target,
):

    home_model, away_model = (
        models
    )

    out = target.copy()

    out[
        "pred_home_xg"
    ] = home_model.predict(
        out[
            [
                "home_xg_fs"
            ]
        ]
    )

    out[
        "pred_away_xg"
    ] = away_model.predict(
        out[
            [
                "away_xg_fs"
            ]
        ]
    )

    return out


def fit_huber(
    train,
):

    home_model = HuberRegressor(
        epsilon=1.35,
        max_iter=1000,
    )

    away_model = HuberRegressor(
        epsilon=1.35,
        max_iter=1000,
    )

    home_model.fit(
        train[
            [
                "home_xg_fs"
            ]
        ],
        train[
            "home_xg_us"
        ],
    )

    away_model.fit(
        train[
            [
                "away_xg_fs"
            ]
        ],
        train[
            "away_xg_us"
        ],
    )

    return (
        home_model,
        away_model,
    )


def apply_huber(
    models,
    target,
):

    home_model, away_model = (
        models
    )

    out = target.copy()

    out[
        "pred_home_xg"
    ] = home_model.predict(
        out[
            [
                "home_xg_fs"
            ]
        ]
    )

    out[
        "pred_away_xg"
    ] = away_model.predict(
        out[
            [
                "away_xg_fs"
            ]
        ]
    )

    return out


def fit_quantile(
    train,
):

    home_mapper = build_quantile_mapper(
        train[
            "home_xg_fs"
        ].values,
        train[
            "home_xg_us"
        ].values,
    )

    away_mapper = build_quantile_mapper(
        train[
            "away_xg_fs"
        ].values,
        train[
            "away_xg_us"
        ].values,
    )

    return (
        home_mapper,
        away_mapper,
    )


def apply_quantile(
    mappers,
    target,
):

    home_mapper, away_mapper = (
        mappers
    )

    out = target.copy()

    out[
        "pred_home_xg"
    ] = home_mapper(
        out[
            "home_xg_fs"
        ].values
    )

    out[
        "pred_away_xg"
    ] = away_mapper(
        out[
            "away_xg_fs"
        ].values
    )

    return out


# ============================================================
# SHRUNK LINEAR
# ============================================================

def apply_shrunk_linear(
    target,
    linear_predictions,
    shrinkage,
):

    out = target.copy()

    out[
        "pred_home_xg"
    ] = (
        shrinkage
        *
        linear_predictions[
            "pred_home_xg"
        ]
        +
        (
            1.0
            -
            shrinkage
        )
        *
        target[
            "home_xg_fs"
        ]
    )

    out[
        "pred_away_xg"
    ] = (
        shrinkage
        *
        linear_predictions[
            "pred_away_xg"
        ]
        +
        (
            1.0
            -
            shrinkage
        )
        *
        target[
            "away_xg_fs"
        ]
    )

    return out


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    predictions,
    method,
    split_name,
):

    home_actual = predictions[
        "home_xg_us"
    ].values

    away_actual = predictions[
        "away_xg_us"
    ].values

    home_pred = predictions[
        "pred_home_xg"
    ].values

    away_pred = predictions[
        "pred_away_xg"
    ].values

    total_actual = (
        home_actual
        +
        away_actual
    )

    total_pred = (
        home_pred
        +
        away_pred
    )

    return {
        "method":
            method,

        "split":
            split_name,

        "games":
            len(
                predictions
            ),

        "home_corr":
            corr(
                home_pred,
                home_actual,
            ),

        "away_corr":
            corr(
                away_pred,
                away_actual,
            ),

        "total_corr":
            corr(
                total_pred,
                total_actual,
            ),

        "home_mae":
            mae(
                home_pred,
                home_actual,
            ),

        "away_mae":
            mae(
                away_pred,
                away_actual,
            ),

        "total_mae":
            mae(
                total_pred,
                total_actual,
            ),

        "home_rmse":
            rmse(
                home_pred,
                home_actual,
            ),

        "away_rmse":
            rmse(
                away_pred,
                away_actual,
            ),

        "total_rmse":
            rmse(
                total_pred,
                total_actual,
            ),

        "home_bias":
            np.mean(
                home_pred
                -
                home_actual
            ),

        "away_bias":
            np.mean(
                away_pred
                -
                away_actual
            ),

        "total_bias":
            np.mean(
                total_pred
                -
                total_actual
            ),

        "home_pred_mean":
            np.mean(
                home_pred
            ),

        "home_actual_mean":
            np.mean(
                home_actual
            ),

        "away_pred_mean":
            np.mean(
                away_pred
            ),

        "away_actual_mean":
            np.mean(
                away_actual
            ),

        "home_pred_std":
            np.std(
                home_pred,
                ddof=1,
            ),

        "home_actual_std":
            np.std(
                home_actual,
                ddof=1,
            ),

        "away_pred_std":
            np.std(
                away_pred,
                ddof=1,
            ),

        "away_actual_std":
            np.std(
                away_actual,
                ddof=1,
            ),
    }


# ============================================================
# BUILD ALL METHOD PREDICTIONS
# ============================================================

def build_method_predictions(
    train,
    target,
):

    outputs = {}

    # RAW
    outputs[
        "RAW"
    ] = apply_raw(
        train,
        target,
    )

    # GLOBAL LINEAR
    global_model = (
        fit_global_linear(
            train
        )
    )

    outputs[
        "GLOBAL_LINEAR"
    ] = apply_global_linear(
        global_model,
        target,
    )

    # SIDE LINEAR
    side_models = (
        fit_side_linear(
            train
        )
    )

    side_linear = (
        apply_side_linear(
            side_models,
            target,
        )
    )

    outputs[
        "SIDE_LINEAR"
    ] = side_linear

    # ROBUST HUBER
    huber_models = (
        fit_huber(
            train
        )
    )

    outputs[
        "HUBER"
    ] = apply_huber(
        huber_models,
        target,
    )

    # QUANTILE
    quantile_models = (
        fit_quantile(
            train
        )
    )

    outputs[
        "QUANTILE"
    ] = apply_quantile(
        quantile_models,
        target,
    )

    # SHRUNK SIDE LINEAR
    for shrinkage in [
        0.25,
        0.50,
        0.75,
    ]:

        name = (
            f"SHRUNK_LINEAR_"
            f"{int(shrinkage * 100)}"
        )

        outputs[
            name
        ] = apply_shrunk_linear(
            target,
            side_linear,
            shrinkage,
        )

    return outputs


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 100
    )

    print(
        "FOOTYSTATS → UNDERSTAT "
        "XG ADAPTER TEST"
    )

    print(
        "=" * 100
    )

    print()
    print(
        "TRAIN:"
    )

    print(
        sorted(
            TRAIN_SEASONS
        )
    )

    print(
        "TUNING:"
    )

    print(
        sorted(
            TUNING_SEASONS
        )
    )

    print(
        "VALIDATION:"
    )

    print(
        sorted(
            VALIDATION_SEASONS
        )
    )

    print(
        "FINAL:"
    )

    print(
        sorted(
            FINAL_SEASONS
        )
    )

    print()
    print(
        "NO V5 PARAMETERS WILL BE CHANGED."
    )

    df = load_data()

    train = df[
        df[
            "season"
        ].isin(
            TRAIN_SEASONS
        )
    ].copy()

    tuning = df[
        df[
            "season"
        ].isin(
            TUNING_SEASONS
        )
    ].copy()

    validation = df[
        df[
            "season"
        ].isin(
            VALIDATION_SEASONS
        )
    ].copy()

    final = df[
        df[
            "season"
        ].isin(
            FINAL_SEASONS
        )
    ].copy()

    print()
    print(
        "Games:"
    )

    print(
        f"Train:      {len(train):,}"
    )

    print(
        f"Tuning:     {len(tuning):,}"
    )

    print(
        f"Validation: {len(validation):,}"
    )

    print(
        f"Final:      {len(final):,}"
    )

    # ========================================================
    # FIT ONLY ON TRAIN
    # ========================================================

    tuning_methods = (
        build_method_predictions(
            train,
            tuning,
        )
    )

    tuning_results = []

    prediction_frames = []

    for method, pred in (
        tuning_methods.items()
    ):

        result = evaluate(
            pred,
            method,
            "TUNING",
        )

        tuning_results.append(
            result
        )

        temp = pred.copy()

        temp[
            "adapter_method"
        ] = method

        temp[
            "split"
        ] = "TUNING"

        prediction_frames.append(
            temp
        )

    tuning_table = pd.DataFrame(
        tuning_results
    )

    # --------------------------------------------------------
    # Rank on total MAE first,
    # then total RMSE,
    # then absolute total bias.
    # --------------------------------------------------------

    tuning_table[
        "abs_total_bias"
    ] = (
        tuning_table[
            "total_bias"
        ].abs()
    )

    tuning_table = (
        tuning_table
        .sort_values(
            [
                "total_mae",
                "total_rmse",
                "abs_total_bias",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    tuning_table[
        "rank"
    ] = (
        np.arange(
            len(
                tuning_table
            )
        )
        +
        1
    )

    print()
    print(
        "=" * 120
    )

    print(
        "TUNING RESULTS"
    )

    print(
        "=" * 120
    )

    display_cols = [
        "rank",
        "method",
        "games",
        "home_corr",
        "away_corr",
        "total_corr",
        "home_mae",
        "away_mae",
        "total_mae",
        "home_rmse",
        "away_rmse",
        "total_rmse",
        "home_bias",
        "away_bias",
        "total_bias",
    ]

    print(
        tuning_table[
            display_cols
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    selected_method = (
        tuning_table.iloc[
            0
        ][
            "method"
        ]
    )

    print()
    print(
        "=" * 100
    )

    print(
        "SELECTED ADAPTER"
    )

    print(
        "=" * 100
    )

    print(
        selected_method
    )

    # ========================================================
    # REBUILD SELECTED METHOD
    #
    # Still fit only on TRAIN.
    # Validation and final remain untouched.
    # ========================================================

    validation_methods = (
        build_method_predictions(
            train,
            validation,
        )
    )

    final_methods = (
        build_method_predictions(
            train,
            final,
        )
    )

    validation_pred = (
        validation_methods[
            selected_method
        ]
    )

    final_pred = (
        final_methods[
            selected_method
        ]
    )

    validation_result = evaluate(
        validation_pred,
        selected_method,
        "VALIDATION",
    )

    final_result = evaluate(
        final_pred,
        selected_method,
        "FINAL",
    )

    # RAW controls
    raw_validation = evaluate(
        validation_methods[
            "RAW"
        ],
        "RAW",
        "VALIDATION",
    )

    raw_final = evaluate(
        final_methods[
            "RAW"
        ],
        "RAW",
        "FINAL",
    )

    print()
    print(
        "=" * 120
    )

    print(
        "VALIDATION — 2023/24"
    )

    print(
        "=" * 120
    )

    validation_compare = pd.DataFrame(
        [
            raw_validation,
            validation_result,
        ]
    )

    print(
        validation_compare[
            [
                "method",
                "games",
                "home_corr",
                "away_corr",
                "total_corr",
                "home_mae",
                "away_mae",
                "total_mae",
                "home_rmse",
                "away_rmse",
                "total_rmse",
                "home_bias",
                "away_bias",
                "total_bias",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print(
        "=" * 120
    )

    print(
        "FINAL CHECK — 2024/25"
    )

    print(
        "=" * 120
    )

    final_compare = pd.DataFrame(
        [
            raw_final,
            final_result,
        ]
    )

    print(
        final_compare[
            [
                "method",
                "games",
                "home_corr",
                "away_corr",
                "total_corr",
                "home_mae",
                "away_mae",
                "total_mae",
                "home_rmse",
                "away_rmse",
                "total_rmse",
                "home_bias",
                "away_bias",
                "total_bias",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # SAVE EVERYTHING
    # ========================================================

    results = pd.concat(
        [
            tuning_table,
            pd.DataFrame(
                [
                    validation_result,
                    final_result,
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    results.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    validation_out = (
        validation_pred.copy()
    )

    validation_out[
        "adapter_method"
    ] = selected_method

    validation_out[
        "split"
    ] = "VALIDATION"

    final_out = (
        final_pred.copy()
    )

    final_out[
        "adapter_method"
    ] = selected_method

    final_out[
        "split"
    ] = "FINAL"

    selected_tuning = (
        tuning_methods[
            selected_method
        ].copy()
    )

    selected_tuning[
        "adapter_method"
    ] = selected_method

    selected_tuning[
        "split"
    ] = "TUNING"

    predictions = pd.concat(
        [
            selected_tuning,
            validation_out,
            final_out,
        ],
        ignore_index=True,
    )

    predictions.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "ADAPTER TEST COMPLETE"
    )

    print(
        "=" * 100
    )

    print()
    print(
        "Selection used only "
        "2021/22–2022/23 ✅"
    )

    print(
        "Validation 2023/24 "
        "not used for selection ✅"
    )

    print(
        "Final 2024/25 "
        "not used for selection ✅"
    )

    print(
        "Frozen V5 unchanged ✅"
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
        "Predictions:"
    )

    print(
        OUTPUT_PREDICTIONS
    )


if __name__ == "__main__":
    main()