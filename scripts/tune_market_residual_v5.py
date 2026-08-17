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

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "market_residual_v5_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "market_residual_v5_predictions.csv"
)


# ============================================================
# SPLITS
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
# SETTINGS
# ============================================================

ALPHAS = np.round(
    np.arange(
        0.00,
        1.01,
        0.025,
    ),
    3,
)

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


def normalize_probs(
    probs,
):

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


def actual_classes(
    df,
):

    return np.where(
        df[
            "home_goals"
        ].to_numpy()
        >
        df[
            "away_goals"
        ].to_numpy(),
        0,
        np.where(
            df[
                "home_goals"
            ].to_numpy()
            ==
            df[
                "away_goals"
            ].to_numpy(),
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

        avg_conf = (
            confidence[
                mask
            ].mean()
        )

        avg_acc = (
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
                avg_conf
                -
                avg_acc
            )
        )

    return float(
        ece
    )


def evaluate(
    df,
    prob_cols,
):

    if len(df) == 0:

        return {
            "games":
                0,

            "accuracy":
                np.nan,

            "log_loss":
                np.nan,

            "brier":
                np.nan,

            "ece":
                np.nan,
        }

    probs = (
        df[
            prob_cols
        ]
        .to_numpy(
            dtype=float
        )
    )

    probs = normalize_probs(
        probs
    )

    y = actual_classes(
        df
    )

    return {
        "games":
            len(df),

        "accuracy":
            accuracy(
                y,
                probs,
            ),

        "log_loss":
            log_loss(
                y,
                probs,
            ),

        "brier":
            brier(
                y,
                probs,
            ),

        "ece":
            expected_calibration_error(
                y,
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
        "market_nv_home",
        "market_nv_draw",
        "market_nv_away",
        "home_goals",
        "away_goals",
        "season",
        "league",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: "
            + str(
                missing
            )
        )

    return df


# ============================================================
# RESIDUAL SHRINKAGE
# ============================================================

def apply_alpha(
    df,
    alpha,
):

    out = df.copy()

    out[
        "resid_p_home"
    ] = (
        out[
            "market_nv_home"
        ]
        +
        alpha
        *
        (
            out[
                "p_home_v5"
            ]
            -
            out[
                "market_nv_home"
            ]
        )
    )

    out[
        "resid_p_draw"
    ] = (
        out[
            "market_nv_draw"
        ]
        +
        alpha
        *
        (
            out[
                "p_draw_v5"
            ]
            -
            out[
                "market_nv_draw"
            ]
        )
    )

    out[
        "resid_p_away"
    ] = (
        out[
            "market_nv_away"
        ]
        +
        alpha
        *
        (
            out[
                "p_away_v5"
            ]
            -
            out[
                "market_nv_away"
            ]
        )
    )

    probs = normalize_probs(
        out[
            [
                "resid_p_home",
                "resid_p_draw",
                "resid_p_away",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    out[
        "resid_p_home"
    ] = probs[
        :,
        0
    ]

    out[
        "resid_p_draw"
    ] = probs[
        :,
        1
    ]

    out[
        "resid_p_away"
    ] = probs[
        :,
        2
    ]

    return out


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_comparison(
    title,
    df,
    alpha,
):

    print()
    print("=" * 95)
    print(title)
    print("=" * 95)

    market = evaluate(
        df,
        [
            "market_nv_home",
            "market_nv_draw",
            "market_nv_away",
        ],
    )

    v5 = evaluate(
        df,
        [
            "p_home_v5",
            "p_draw_v5",
            "p_away_v5",
        ],
    )

    residual_df = apply_alpha(
        df,
        alpha,
    )

    residual = evaluate(
        residual_df,
        [
            "resid_p_home",
            "resid_p_draw",
            "resid_p_away",
        ],
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
                    f"RESIDUAL alpha={alpha:.3f}",

                **residual,
            },
        ]
    )

    display = table.copy()

    display[
        "accuracy"
    ] *= 100

    display[
        "ece"
    ] *= 100

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


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("TUNING MARKET RESIDUAL V5")
    print("==============================")
    print()

    df = load_data()

    print(
        f"Matched games: "
        f"{len(df):,}"
    )

    print(
        f"Alpha settings: "
        f"{len(ALPHAS)}"
    )

    tune = df[
        df[
            "season"
        ].isin(
            TUNING_SEASONS
        )
    ].copy()

    print(
        f"Tuning games: "
        f"{len(tune):,}"
    )

    # ========================================================
    # TUNE ALPHA
    # ========================================================

    rows = []

    for alpha in (
        ALPHAS
    ):

        adjusted = apply_alpha(
            tune,
            alpha,
        )

        metrics = evaluate(
            adjusted,
            [
                "resid_p_home",
                "resid_p_draw",
                "resid_p_away",
            ],
        )

        rows.append(
            {
                "alpha":
                    float(
                        alpha
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

    results[
        "rank"
    ] = (
        np.arange(
            len(results)
        )
        + 1
    )

    results.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("==============================")
    print("TOP 20 RESIDUAL SETTINGS")
    print("==============================")
    print()

    display = (
        results
        .head(
            20
        )
        .copy()
    )

    display[
        "accuracy"
    ] *= 100

    display[
        "ece"
    ] *= 100

    print(
        display[
            [
                "rank",
                "alpha",
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

    # ========================================================
    # WINNER
    # ========================================================

    best = results.iloc[
        0
    ]

    best_alpha = float(
        best[
            "alpha"
        ]
    )

    print()
    print("==============================")
    print("WINNING RESIDUAL SHRINKAGE")
    print("==============================")

    print(
        f"Alpha: "
        f"{best_alpha:.3f}"
    )

    print(
        f"Market share: "
        f"{1-best_alpha:.1%}"
    )

    print(
        f"V5 residual share: "
        f"{best_alpha:.1%}"
    )

    print(
        f"Tuning LL: "
        f"{best['log_loss']:.5f}"
    )

    # ========================================================
    # PERIOD COMPARISONS
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

    for title, seasons in (
        samples
    ):

        sub = df[
            df[
                "season"
            ].isin(
                seasons
            )
        ].copy()

        print_comparison(
            title,
            sub,
            best_alpha,
        )

    # ========================================================
    # LOCKED TEST BY LEAGUE
    # ========================================================

    locked = df[
        df[
            "season"
        ].isin(
            LOCKED_SEASONS
        )
    ].copy()

    print()
    print("=" * 110)
    print("2025/26 LOCKED TEST — RESIDUAL BY LEAGUE")
    print("=" * 110)

    rows = []

    for league, sub in (
        locked.groupby(
            "league"
        )
    ):

        market = evaluate(
            sub,
            [
                "market_nv_home",
                "market_nv_draw",
                "market_nv_away",
            ],
        )

        residual_df = apply_alpha(
            sub,
            best_alpha,
        )

        residual = evaluate(
            residual_df,
            [
                "resid_p_home",
                "resid_p_draw",
                "resid_p_away",
            ],
        )

        rows.append(
            {
                "league":
                    league,

                "games":
                    len(
                        sub
                    ),

                "market_ll":
                    market[
                        "log_loss"
                    ],

                "residual_ll":
                    residual[
                        "log_loss"
                    ],

                "ll_change":
                    (
                        residual[
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

                "residual_brier":
                    residual[
                        "brier"
                    ],

                "market_ece":
                    market[
                        "ece"
                    ]
                    * 100,

                "residual_ece":
                    residual[
                        "ece"
                    ]
                    * 100,
            }
        )

    league_table = pd.DataFrame(
        rows
    )

    print(
        league_table
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

    output = apply_alpha(
        df,
        best_alpha,
    )

    output[
        "market_residual_alpha_v5"
    ] = best_alpha

    output[
        "market_residual_market_share_v5"
    ] = (
        1.0
        -
        best_alpha
    )

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("==============================")
    print("MARKET RESIDUAL TEST COMPLETE")
    print("==============================")

    print(
        "Alpha selected using "
        "2021/22–2022/23 only ✅"
    )

    print(
        "2023/24, 2024/25 and 2025/26 "
        "not used for alpha selection ✅"
    )

    print(
        "Frozen V5 probabilities "
        "unchanged ✅"
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