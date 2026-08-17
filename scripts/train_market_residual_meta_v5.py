from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
except ImportError as exc:
    raise ImportError(
        "\nscikit-learn is required.\n\n"
        "Run:\n"
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
    / "v5_market_comparison.csv"
)

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "market_residual_meta_v5_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "market_residual_meta_v5_predictions.csv"
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

LOCKED_SEASONS = {
    "2526",
}


# ============================================================
# MODEL SETTINGS
# ============================================================

C_VALUES = [
    0.01,
    0.03,
    0.10,
    0.30,
    1.00,
    3.00,
    10.00,
    30.00,
]

EPS = 1e-12


# ============================================================
# HELPERS
# ============================================================

def season_string(series):

    return (
        series
        .astype(str)
        .str.zfill(4)
    )


def normalize_probs(probs):

    probs = np.clip(
        probs,
        EPS,
        None,
    )

    return (
        probs
        /
        probs.sum(
            axis=1,
            keepdims=True,
        )
    )


def result_classes(df):

    return np.where(
        df["home_goals"].to_numpy()
        >
        df["away_goals"].to_numpy(),
        0,
        np.where(
            df["home_goals"].to_numpy()
            ==
            df["away_goals"].to_numpy(),
            1,
            2,
        ),
    )


# ============================================================
# METRICS
# ============================================================

def log_loss(y_true, probs):

    chosen = probs[
        np.arange(len(y_true)),
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


def brier(y_true, probs):

    truth = np.zeros_like(
        probs
    )

    truth[
        np.arange(len(y_true)),
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


def accuracy(y_true, probs):

    return float(
        (
            probs.argmax(axis=1)
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
    ).astype(float)

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0

    for i in range(bins):

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

        avg_conf = (
            confidence[mask]
            .mean()
        )

        avg_acc = (
            correct[mask]
            .mean()
        )

        ece += (
            n
            /
            len(y_true)
            *
            abs(
                avg_conf
                -
                avg_acc
            )
        )

    return float(ece)


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
    }


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

    df = df[
        df["market_source"]
        ==
        "avg_close"
    ].copy()

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df):

    out = df.copy()

    # --------------------------------------------------------
    # MARKET BASE
    # --------------------------------------------------------

    out["market_home"] = (
        out["market_nv_home"]
    )

    out["market_draw"] = (
        out["market_nv_draw"]
    )

    out["market_away"] = (
        out["market_nv_away"]
    )

    # --------------------------------------------------------
    # V5 RESIDUALS
    # --------------------------------------------------------

    out["resid_home"] = (
        out["p_home_v5"]
        -
        out["market_nv_home"]
    )

    out["resid_draw"] = (
        out["p_draw_v5"]
        -
        out["market_nv_draw"]
    )

    out["resid_away"] = (
        out["p_away_v5"]
        -
        out["market_nv_away"]
    )

    # --------------------------------------------------------
    # RESIDUAL MAGNITUDE
    # --------------------------------------------------------

    out["abs_resid_home"] = (
        out["resid_home"].abs()
    )

    out["abs_resid_draw"] = (
        out["resid_draw"].abs()
    )

    out["abs_resid_away"] = (
        out["resid_away"].abs()
    )

    out["max_abs_residual"] = (
        out[
            [
                "abs_resid_home",
                "abs_resid_draw",
                "abs_resid_away",
            ]
        ]
        .max(axis=1)
    )

    # --------------------------------------------------------
    # MARKET SHAPE
    # --------------------------------------------------------

    out["market_favorite_prob"] = (
        out[
            [
                "market_home",
                "market_draw",
                "market_away",
            ]
        ]
        .max(axis=1)
    )

    out["market_entropy"] = (
        -(
            out["market_home"]
            *
            np.log(
                np.clip(
                    out["market_home"],
                    EPS,
                    1.0,
                )
            )
            +
            out["market_draw"]
            *
            np.log(
                np.clip(
                    out["market_draw"],
                    EPS,
                    1.0,
                )
            )
            +
            out["market_away"]
            *
            np.log(
                np.clip(
                    out["market_away"],
                    EPS,
                    1.0,
                )
            )
        )
    )

    # --------------------------------------------------------
    # V5 SHAPE
    # --------------------------------------------------------

    out["v5_favorite_prob"] = (
        out[
            [
                "p_home_v5",
                "p_draw_v5",
                "p_away_v5",
            ]
        ]
        .max(axis=1)
    )

    out["v5_entropy"] = (
        -(
            out["p_home_v5"]
            *
            np.log(
                np.clip(
                    out["p_home_v5"],
                    EPS,
                    1.0,
                )
            )
            +
            out["p_draw_v5"]
            *
            np.log(
                np.clip(
                    out["p_draw_v5"],
                    EPS,
                    1.0,
                )
            )
            +
            out["p_away_v5"]
            *
            np.log(
                np.clip(
                    out["p_away_v5"],
                    EPS,
                    1.0,
                )
            )
        )
    )

    # --------------------------------------------------------
    # DISAGREEMENT STRUCTURE
    # --------------------------------------------------------

    out["favorite_prob_diff"] = (
        out["v5_favorite_prob"]
        -
        out["market_favorite_prob"]
    )

    out["entropy_diff"] = (
        out["v5_entropy"]
        -
        out["market_entropy"]
    )

    # --------------------------------------------------------
    # LEAGUE INDICATOR
    #
    # PL = 1
    # Bundesliga = 0
    # --------------------------------------------------------

    out["is_premier_league"] = (
        out["league"]
        .eq(
            "Premier League"
        )
        .astype(float)
    )

    return out


# ============================================================
# FEATURES USED
# ============================================================

FEATURE_COLS = [
    "market_home",
    "market_draw",
    "market_away",

    "resid_home",
    "resid_draw",
    "resid_away",

    "abs_resid_home",
    "abs_resid_draw",
    "abs_resid_away",

    "max_abs_residual",

    "market_favorite_prob",
    "market_entropy",

    "v5_favorite_prob",
    "v5_entropy",

    "favorite_prob_diff",
    "entropy_diff",

    "is_premier_league",
]


# ============================================================
# SUBSET
# ============================================================

def subset(
    df,
    seasons,
):

    return df[
        df["season"]
        .isin(
            seasons
        )
    ].copy()


# ============================================================
# FIT MODEL
# ============================================================

def fit_model(
    train_df,
    C,
):

    X = (
        train_df[
            FEATURE_COLS
        ]
        .to_numpy(
            dtype=float
        )
    )

    y = result_classes(
        train_df
    )

    model = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=5000,
    )

    model.fit(
        X,
        y,
    )

    return model


# ============================================================
# PREDICT
# ============================================================

def predict_probs(
    model,
    df,
):

    X = (
        df[
            FEATURE_COLS
        ]
        .to_numpy(
            dtype=float
        )
    )

    probs = model.predict_proba(
        X
    )

    return normalize_probs(
        probs
    )


# ============================================================
# MARKET / V5 PROBS
# ============================================================

def market_probs(df):

    return normalize_probs(
        df[
            [
                "market_nv_home",
                "market_nv_draw",
                "market_nv_away",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )


def v5_probs(df):

    return normalize_probs(
        df[
            [
                "p_home_v5",
                "p_draw_v5",
                "p_away_v5",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )


# ============================================================
# SELECT REGULARIZATION
# ============================================================

def select_C(
    train_df,
    tune_df,
):

    rows = []

    y_tune = result_classes(
        tune_df
    )

    for C in C_VALUES:

        model = fit_model(
            train_df,
            C,
        )

        probs = predict_probs(
            model,
            tune_df,
        )

        metrics = evaluate(
            y_tune,
            probs,
        )

        rows.append(
            {
                "C":
                    C,

                **metrics,
            }
        )

    results = pd.DataFrame(
        rows
    )

    results = (
        results
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

    results["rank"] = (
        np.arange(
            len(results)
        )
        + 1
    )

    return results


# ============================================================
# PRINT MODEL COMPARISON
# ============================================================

def print_comparison(
    title,
    df,
    model,
):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    y = result_classes(
        df
    )

    market = evaluate(
        y,
        market_probs(
            df
        ),
    )

    v5 = evaluate(
        y,
        v5_probs(
            df
        ),
    )

    meta = evaluate(
        y,
        predict_probs(
            model,
            df,
        ),
    )

    table = pd.DataFrame(
        [
            {
                "model":
                    "MARKET",

                **market,
            },
            {
                "model":
                    "V5",

                **v5,
            },
            {
                "model":
                    "META",

                **meta,
            },
        ]
    )

    display = table.copy()

    display["accuracy"] *= 100
    display["ece"] *= 100

    print(
        display[
            [
                "model",
                "games",
                "accuracy",
                "log_loss",
                "brier",
                "ece",
            ]
        ]
        .round(
            6
        )
        .to_string(
            index=False
        )
    )

    return table


# ============================================================
# LOCKED BY LEAGUE
# ============================================================

def print_locked_by_league(
    locked,
    model,
):

    print()
    print("=" * 110)
    print("2025/26 LOCKED TEST — META BY LEAGUE")
    print("=" * 110)

    rows = []

    for league, sub in (
        locked.groupby(
            "league"
        )
    ):

        y = result_classes(
            sub
        )

        market = evaluate(
            y,
            market_probs(
                sub
            ),
        )

        meta = evaluate(
            y,
            predict_probs(
                model,
                sub,
            ),
        )

        rows.append(
            {
                "league":
                    league,

                "games":
                    len(sub),

                "market_ll":
                    market[
                        "log_loss"
                    ],

                "meta_ll":
                    meta[
                        "log_loss"
                    ],

                "ll_change":
                    (
                        meta[
                            "log_loss"
                        ]
                        -
                        market[
                            "log_loss"
                        ]
                    ),

                "market_brier":
                    market[
                        "brier"
                    ],

                "meta_brier":
                    meta[
                        "brier"
                    ],

                "market_acc":
                    (
                        market[
                            "accuracy"
                        ]
                        * 100
                    ),

                "meta_acc":
                    (
                        meta[
                            "accuracy"
                        ]
                        * 100
                    ),

                "market_ece":
                    (
                        market[
                            "ece"
                        ]
                        * 100
                    ),

                "meta_ece":
                    (
                        meta[
                            "ece"
                        ]
                        * 100
                    ),
            }
        )

    table = pd.DataFrame(
        rows
    )

    print(
        table
        .round(
            6
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE META PREDICTIONS
# ============================================================

def save_predictions(
    df,
    model,
    C,
):

    output = df.copy()

    probs = predict_probs(
        model,
        output,
    )

    output["meta_p_home"] = (
        probs[:, 0]
    )

    output["meta_p_draw"] = (
        probs[:, 1]
    )

    output["meta_p_away"] = (
        probs[:, 2]
    )

    output["meta_model_C"] = C

    output["meta_edge_home"] = (
        output["meta_p_home"]
        -
        output["market_nv_home"]
    )

    output["meta_edge_draw"] = (
        output["meta_p_draw"]
        -
        output["market_nv_draw"]
    )

    output["meta_edge_away"] = (
        output["meta_p_away"]
        -
        output["market_nv_away"]
    )

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("MARKET RESIDUAL META MODEL V5")
    print("==============================")
    print()

    print(
        "Training: 2018/19–2020/21"
    )

    print(
        "C selection: 2021/22–2022/23"
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

    df = load_data()

    df = build_features(
        df
    )

    train = subset(
        df,
        TRAIN_SEASONS,
    )

    tune = subset(
        df,
        TUNING_SEASONS,
    )

    validation = subset(
        df,
        VALIDATION_SEASONS,
    )

    final = subset(
        df,
        FINAL_SEASONS,
    )

    locked = subset(
        df,
        LOCKED_SEASONS,
    )

    print()
    print(
        f"Training games: "
        f"{len(train):,}"
    )

    print(
        f"Tuning games: "
        f"{len(tune):,}"
    )

    print(
        f"Validation games: "
        f"{len(validation):,}"
    )

    print(
        f"Final games: "
        f"{len(final):,}"
    )

    print(
        f"Locked games: "
        f"{len(locked):,}"
    )

    # ========================================================
    # SELECT C
    # ========================================================

    results = select_C(
        train,
        tune,
    )

    results.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    print()
    print("==============================")
    print("META MODEL C RESULTS")
    print("==============================")
    print()

    display = results.copy()

    display["accuracy"] *= 100
    display["ece"] *= 100

    print(
        display[
            [
                "rank",
                "C",
                "games",
                "log_loss",
                "brier",
                "accuracy",
                "ece",
            ]
        ]
        .round(
            6
        )
        .to_string(
            index=False
        )
    )

    best_C = float(
        results.iloc[0][
            "C"
        ]
    )

    print()
    print("==============================")
    print("SELECTED META MODEL")
    print("==============================")

    print(
        f"C: {best_C:g}"
    )

    # ========================================================
    # FIT FINAL META MODEL
    #
    # Still fit ONLY on TRAIN seasons.
    # Do not refit with validation/final/locked yet.
    # ========================================================

    model = fit_model(
        train,
        best_C,
    )

    # ========================================================
    # PERIOD COMPARISONS
    # ========================================================

    comparison_rows = []

    periods = [
        (
            "TUNING — 2021/22 TO 2022/23",
            tune,
        ),
        (
            "VALIDATION — 2023/24",
            validation,
        ),
        (
            "FINAL CHECK — 2024/25",
            final,
        ),
        (
            "LOCKED TEST — 2025/26",
            locked,
        ),
    ]

    for title, sub in periods:

        table = print_comparison(
            title,
            sub,
            model,
        )

        for _, row in table.iterrows():

            comparison_rows.append(
                {
                    "sample":
                        title,

                    **row.to_dict(),
                }
            )

    # ========================================================
    # LOCKED BY LEAGUE
    # ========================================================

    print_locked_by_league(
        locked,
        model,
    )

    # ========================================================
    # COEFFICIENTS
    # ========================================================

    print()
    print("=" * 120)
    print("META MODEL COEFFICIENTS")
    print("=" * 120)

    coef_rows = []

    for class_index, class_name in enumerate(
        [
            "HOME",
            "DRAW",
            "AWAY",
        ]
    ):

        for feature, coef in zip(
            FEATURE_COLS,
            model.coef_[
                class_index
            ],
        ):

            coef_rows.append(
                {
                    "class":
                        class_name,

                    "feature":
                        feature,

                    "coefficient":
                        coef,
                }
            )

    coef_table = pd.DataFrame(
        coef_rows
    )

    coef_table[
        "abs_coefficient"
    ] = (
        coef_table[
            "coefficient"
        ]
        .abs()
    )

    coef_table = (
        coef_table
        .sort_values(
            [
                "class",
                "abs_coefficient",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    print(
        coef_table[
            [
                "class",
                "feature",
                "coefficient",
            ]
        ]
        .round(
            6
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    save_predictions(
        df,
        model,
        best_C,
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("==============================")
    print("META MODEL TEST COMPLETE")
    print("==============================")

    print(
        "Frozen V5 probabilities unchanged ✅"
    )

    print(
        "Closing market only ✅"
    )

    print(
        "Meta model trained only on "
        "2018/19–2020/21 ✅"
    )

    print(
        "Regularization selected using "
        "2021/22–2022/23 only ✅"
    )

    print(
        "2023/24, 2024/25 and 2025/26 "
        "not used for model fitting ✅"
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