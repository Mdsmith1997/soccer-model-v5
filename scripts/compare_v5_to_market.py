from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

V5_FILE = (
    ROOT
    / "data"
    / "processed"
    / "frozen_v5_predictions.csv"
)

MARKET_FILE = (
    ROOT
    / "data"
    / "processed"
    / "market_benchmark_comparison.csv"
)

OUTPUT_COMPARISON = (
    ROOT
    / "data"
    / "processed"
    / "v5_market_comparison.csv"
)

OUTPUT_BLEND_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "v5_market_blend_tuning_results.csv"
)

OUTPUT_BLEND_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "v5_market_blend_predictions.csv"
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
# BLEND GRID
#
# model weight = 1 - market weight
# ============================================================

MARKET_WEIGHTS = np.round(
    np.arange(
        0.00,
        0.51,
        0.05,
    ),
    2,
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


def result_classes(
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

    y = result_classes(
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
    }


# ============================================================
# LOAD
# ============================================================

def load_v5():

    df = pd.read_csv(
        V5_FILE,
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

    return df


def load_market():

    df = pd.read_csv(
        MARKET_FILE,
        parse_dates=[
            "date",
        ],
    )

    if "season" in df.columns:

        df[
            "season"
        ] = season_string(
            df[
                "season"
            ]
        )

    return df


# ============================================================
# FIND JOIN KEYS
# ============================================================

def determine_join_keys(
    v5,
    market,
):

    # Best case
    if (
        "match_id"
        in v5.columns
        and
        "match_id"
        in market.columns
    ):

        return [
            "match_id",
        ]

    # Fallback
    candidate = [
        "date",
        "home_team",
        "away_team",
    ]

    if all(
        col in v5.columns
        and
        col in market.columns
        for col in candidate
    ):

        return candidate

    raise ValueError(
        "Could not determine safe join keys.\n"
        "Need either shared match_id or "
        "date/home_team/away_team."
    )


# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

def build_comparison():

    v5 = load_v5()
    market = load_market()

    join_keys = determine_join_keys(
        v5,
        market,
    )

    print(
        "Join keys:",
        join_keys,
    )

    market_keep = list(
        dict.fromkeys(
            join_keys
            +
            [
                "market_source",
                "market_home_odds",
                "market_draw_odds",
                "market_away_odds",
                "market_p_home",
                "market_p_draw",
                "market_p_away",
                "market_margin",
            ]
        )
    )

    market_keep = [
        col
        for col in market_keep
        if col in market.columns
    ]

    market_small = (
        market[
            market_keep
        ]
        .copy()
    )

    df = v5.merge(
        market_small,
        on=join_keys,
        how="inner",
        validate="one_to_one",
    )

    required = [
        "p_home_v5",
        "p_draw_v5",
        "p_away_v5",
        "market_p_home",
        "market_p_draw",
        "market_p_away",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required probability columns: "
            + str(
                missing
            )
        )

    # --------------------------------------------------------
    # NORMALIZE MARKET TO NO-VIG
    # --------------------------------------------------------

    market_probs = normalize_probs(
        df[
            [
                "market_p_home",
                "market_p_draw",
                "market_p_away",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    df[
        "market_nv_home"
    ] = market_probs[
        :,
        0
    ]

    df[
        "market_nv_draw"
    ] = market_probs[
        :,
        1
    ]

    df[
        "market_nv_away"
    ] = market_probs[
        :,
        2
    ]

    # --------------------------------------------------------
    # MODEL EDGE
    # --------------------------------------------------------

    df[
        "edge_home"
    ] = (
        df[
            "p_home_v5"
        ]
        -
        df[
            "market_nv_home"
        ]
    )

    df[
        "edge_draw"
    ] = (
        df[
            "p_draw_v5"
        ]
        -
        df[
            "market_nv_draw"
        ]
    )

    df[
        "edge_away"
    ] = (
        df[
            "p_away_v5"
        ]
        -
        df[
            "market_nv_away"
        ]
    )

    return df


# ============================================================
# BLEND
# ============================================================

def add_blend(
    df,
    market_weight,
):

    out = df.copy()

    model_weight = (
        1.0
        - market_weight
    )

    out[
        "blend_p_home"
    ] = (
        model_weight
        *
        out[
            "p_home_v5"
        ]
        +
        market_weight
        *
        out[
            "market_nv_home"
        ]
    )

    out[
        "blend_p_draw"
    ] = (
        model_weight
        *
        out[
            "p_draw_v5"
        ]
        +
        market_weight
        *
        out[
            "market_nv_draw"
        ]
    )

    out[
        "blend_p_away"
    ] = (
        model_weight
        *
        out[
            "p_away_v5"
        ]
        +
        market_weight
        *
        out[
            "market_nv_away"
        ]
    )

    return out


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_three_way(
    title,
    df,
    blend_weight=None,
):

    print()
    print("=" * 92)
    print(title)
    print("=" * 92)

    raw = evaluate(
        df,
        [
            "p_home_v5",
            "p_draw_v5",
            "p_away_v5",
        ],
    )

    market = evaluate(
        df,
        [
            "market_nv_home",
            "market_nv_draw",
            "market_nv_away",
        ],
    )

    rows = [
        {
            "model":
                "V5",

            **raw,
        },

        {
            "model":
                "MARKET",

            **market,
        },
    ]

    if blend_weight is not None:

        blended = add_blend(
            df,
            blend_weight,
        )

        blend_metrics = evaluate(
            blended,
            [
                "blend_p_home",
                "blend_p_draw",
                "blend_p_away",
            ],
        )

        rows.append(
            {
                "model":
                    (
                        f"BLEND "
                        f"{1-blend_weight:.0%}/"
                        f"{blend_weight:.0%}"
                    ),

                **blend_metrics,
            }
        )

    table = pd.DataFrame(
        rows
    )

    display = table.copy()

    display[
        "accuracy"
    ] *= 100

    print(
        display[
            [
                "model",
                "games",
                "accuracy",
                "log_loss",
                "brier",
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
    print("V5 VS MARKET")
    print("==============================")
    print()

    df = build_comparison()

    print(
        f"Matched games: "
        f"{len(df):,}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"-> "
        f"{df['date'].max().date()}"
    )

    # ========================================================
    # SAVE RAW COMPARISON
    # ========================================================

    df.to_csv(
        OUTPUT_COMPARISON,
        index=False,
    )

    # ========================================================
    # TUNING MARKET BLEND
    # ========================================================

    tune = df[
        df[
            "season"
        ].isin(
            TUNING_SEASONS
        )
    ].copy()

    rows = []

    for market_weight in (
        MARKET_WEIGHTS
    ):

        blended = add_blend(
            tune,
            market_weight,
        )

        metrics = evaluate(
            blended,
            [
                "blend_p_home",
                "blend_p_draw",
                "blend_p_away",
            ],
        )

        rows.append(
            {
                "market_weight":
                    float(
                        market_weight
                    ),

                "model_weight":
                    float(
                        1.0
                        -
                        market_weight
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
        OUTPUT_BLEND_RESULTS,
        index=False,
    )

    print()
    print("==============================")
    print("MARKET BLEND RESULTS")
    print("==============================")
    print()

    display = results.copy()

    display[
        "accuracy"
    ] *= 100

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
            ]
        ]
        .round(
            6
        )
        .to_string(
            index=False
        )
    )

    best = results.iloc[
        0
    ]

    best_market_weight = float(
        best[
            "market_weight"
        ]
    )

    print()
    print("==============================")
    print("WINNING MARKET BLEND")
    print("==============================")

    print(
        f"Model:  "
        f"{1-best_market_weight:.1%}"
    )

    print(
        f"Market: "
        f"{best_market_weight:.1%}"
    )

    print(
        f"Tuning LL: "
        f"{best['log_loss']:.5f}"
    )

    # ========================================================
    # COMPARISON PERIODS
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

    for title, seasons in samples:

        sub = df[
            df[
                "season"
            ].isin(
                seasons
            )
        ].copy()

        print_three_way(
            title,
            sub,
            best_market_weight,
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
    print("=" * 100)
    print("2025/26 LOCKED TEST — BY LEAGUE")
    print("=" * 100)

    rows = []

    for league, sub in (
        locked.groupby(
            "league"
        )
    ):

        raw = evaluate(
            sub,
            [
                "p_home_v5",
                "p_draw_v5",
                "p_away_v5",
            ],
        )

        market = evaluate(
            sub,
            [
                "market_nv_home",
                "market_nv_draw",
                "market_nv_away",
            ],
        )

        blended = add_blend(
            sub,
            best_market_weight,
        )

        blend = evaluate(
            blended,
            [
                "blend_p_home",
                "blend_p_draw",
                "blend_p_away",
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

                "v5_ll":
                    raw[
                        "log_loss"
                    ],

                "market_ll":
                    market[
                        "log_loss"
                    ],

                "blend_ll":
                    blend[
                        "log_loss"
                    ],

                "v5_brier":
                    raw[
                        "brier"
                    ],

                "market_brier":
                    market[
                        "brier"
                    ],

                "blend_brier":
                    blend[
                        "brier"
                    ],
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
    # SAVE WINNING BLEND
    # ========================================================

    output = add_blend(
        df,
        best_market_weight,
    )

    output[
        "v5_market_blend_model_weight"
    ] = (
        1.0
        -
        best_market_weight
    )

    output[
        "v5_market_blend_market_weight"
    ] = (
        best_market_weight
    )

    output.to_csv(
        OUTPUT_BLEND_PREDICTIONS,
        index=False,
    )

    # ========================================================
    # EDGE SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print("MODEL VS MARKET EDGE SUMMARY")
    print("=" * 100)

    edge_cols = [
        "edge_home",
        "edge_draw",
        "edge_away",
    ]

    for col in edge_cols:

        print(
            f"{col:<15}"
            f" mean="
            f"{df[col].mean():+.3%}"
            f"  abs_mean="
            f"{df[col].abs().mean():.3%}"
            f"  >5%="
            f"{(df[col] > 0.05).sum():,}"
        )

    print()
    print("==============================")
    print("V5 MARKET COMPARISON COMPLETE")
    print("==============================")

    print(
        "Blend selected using "
        "2021/22–2022/23 only ✅"
    )

    print(
        "2023/24, 2024/25 and 2025/26 "
        "not used for blend selection ✅"
    )

    print()
    print(
        "Comparison:"
    )

    print(
        OUTPUT_COMPARISON
    )

    print()

    print(
        "Blend results:"
    )

    print(
        OUTPUT_BLEND_RESULTS
    )

    print()

    print(
        "Blend predictions:"
    )

    print(
        OUTPUT_BLEND_PREDICTIONS
    )


if __name__ == "__main__":
    main()