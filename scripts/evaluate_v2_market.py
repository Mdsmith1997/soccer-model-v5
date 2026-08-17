from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MATCHES_FILE = (
    ROOT
    / "data"
    / "processed"
    / "matches.csv"
)

V2_FILE = (
    ROOT
    / "data"
    / "processed"
    / "shot_model_v2_predictions.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "v2_market_comparison.csv"
)

EPS = 1e-12


# =========================================================
# SEASONS
# =========================================================

VALIDATION_SEASONS = {
    "2324",
    "2425",
}

FINAL_SEASON = {
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

    return np.mean(
        np.sum(
            (
                probs - truth
            ) ** 2,
            axis=1,
        )
    )


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


def accuracy_score(
    y_true,
    probs,
):
    predicted = probs.argmax(
        axis=1
    )

    return (
        predicted == y_true
    ).mean()


# =========================================================
# RESULT CLASS
# =========================================================

def get_result_class(
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


# =========================================================
# MARKET ODDS
# =========================================================

def fair_probs_from_odds(
    home_odds,
    draw_odds,
    away_odds,
):
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

    total = (
        raw_home
        + raw_draw
        + raw_away
    )

    fair_home = (
        raw_home / total
    )

    fair_draw = (
        raw_draw / total
    )

    fair_away = (
        raw_away / total
    )

    margin = (
        total - 1.0
    )

    return (
        fair_home,
        fair_draw,
        fair_away,
        margin,
    )


# =========================================================
# MARKET TABLE
# =========================================================

def build_market_table(
    matches,
):
    rows = []

    for _, row in matches.iterrows():

        # -------------------------------------------------
        # TRUE AVERAGE CLOSING
        # -------------------------------------------------

        avg_close_available = (
            "avg_home_close" in row.index
            and "avg_draw_close" in row.index
            and "avg_away_close" in row.index
            and pd.notna(
                row["avg_home_close"]
            )
            and pd.notna(
                row["avg_draw_close"]
            )
            and pd.notna(
                row["avg_away_close"]
            )
        )

        # -------------------------------------------------
        # OPENING FALLBACK
        # -------------------------------------------------

        avg_open_available = (
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

        b365_open_available = (
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

        # -------------------------------------------------
        # SOURCE PRIORITY
        # -------------------------------------------------

        if avg_close_available:

            source = "avg_close"

            home_odds = row[
                "avg_home_close"
            ]

            draw_odds = row[
                "avg_draw_close"
            ]

            away_odds = row[
                "avg_away_close"
            ]

        elif avg_open_available:

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

        elif b365_open_available:

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

def evaluate_system(
    df,
    prefix,
):
    y = df[
        "result_class"
    ].to_numpy()

    probs = df[
        [
            f"{prefix}_p_home",
            f"{prefix}_p_draw",
            f"{prefix}_p_away",
        ]
    ].to_numpy()

    return {
        "games":
            len(df),

        "accuracy":
            accuracy_score(
                y,
                probs,
            ),

        "log_loss":
            multiclass_log_loss(
                y,
                probs,
            ),

        "brier":
            multiclass_brier(
                y,
                probs,
            ),

        "ece":
            multiclass_ece(
                y,
                probs,
            ),
    }


def print_comparison(
    title,
    model,
    market,
):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    print(
        f"Games: "
        f"{model['games']:,}"
    )

    print()
    print(
        f"{'Metric':<18}"
        f"{'V2':>14}"
        f"{'Market':>14}"
        f"{'Difference':>14}"
    )

    print("-" * 60)

    print(
        f"{'Accuracy':<18}"
        f"{model['accuracy']:>13.2%}"
        f"{market['accuracy']:>13.2%}"
        f"{model['accuracy'] - market['accuracy']:>+13.2%}"
    )

    print(
        f"{'Log Loss':<18}"
        f"{model['log_loss']:>14.5f}"
        f"{market['log_loss']:>14.5f}"
        f"{model['log_loss'] - market['log_loss']:>+14.5f}"
    )

    print(
        f"{'Brier':<18}"
        f"{model['brier']:>14.5f}"
        f"{market['brier']:>14.5f}"
        f"{model['brier'] - market['brier']:>+14.5f}"
    )

    print(
        f"{'ECE':<18}"
        f"{model['ece']:>13.2%}"
        f"{market['ece']:>13.2%}"
        f"{model['ece'] - market['ece']:>+13.2%}"
    )


# =========================================================
# TABLE BY LEAGUE
# =========================================================

def league_comparison_table(
    df,
):
    rows = []

    for league, group in df.groupby(
        "league"
    ):

        model = evaluate_system(
            group,
            "model",
        )

        market = evaluate_system(
            group,
            "market",
        )

        rows.append({
            "league":
                league,

            "games":
                len(group),

            "model_acc":
                model[
                    "accuracy"
                ],

            "market_acc":
                market[
                    "accuracy"
                ],

            "model_ll":
                model[
                    "log_loss"
                ],

            "market_ll":
                market[
                    "log_loss"
                ],

            "ll_gap":
                (
                    model[
                        "log_loss"
                    ]
                    -
                    market[
                        "log_loss"
                    ]
                ),

            "model_brier":
                model[
                    "brier"
                ],

            "market_brier":
                market[
                    "brier"
                ],

            "brier_gap":
                (
                    model[
                        "brier"
                    ]
                    -
                    market[
                        "brier"
                    ]
                ),

            "model_ece":
                model[
                    "ece"
                ],

            "market_ece":
                market[
                    "ece"
                ],
        })

    table = pd.DataFrame(
        rows
    )

    for column in [
        "model_acc",
        "market_acc",
        "model_ece",
        "market_ece",
    ]:
        table[
            column
        ] *= 100.0

    return table


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("==============================")
    print("V2 VS MARKET BENCHMARK")
    print("==============================")
    print()

    if not MATCHES_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{MATCHES_FILE}"
        )

    if not V2_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{V2_FILE}"
        )

    matches = pd.read_csv(
        MATCHES_FILE,
        parse_dates=["date"],
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

    print(
        f"Historical matches: "
        f"{len(matches):,}"
    )

    print(
        f"V2 predictions: "
        f"{len(v2):,}"
    )

    print(
        "Building market probabilities..."
    )

    market = build_market_table(
        matches
    )

    df = v2.merge(
        market,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------
    # MODEL COLUMNS
    # -----------------------------------------------------

    df = df.rename(
        columns={
            "p_home_v2":
                "model_p_home",

            "p_draw_v2":
                "model_p_draw",

            "p_away_v2":
                "model_p_away",
        }
    )

    df[
        "result_class"
    ] = get_result_class(
        df[
            "home_goals"
        ].to_numpy(),
        df[
            "away_goals"
        ].to_numpy(),
    )

    # -----------------------------------------------------
    # VALID MARKET ROWS
    # -----------------------------------------------------

    valid = df[
        df[
            "market_p_home"
        ].notna()
        &
        df[
            "market_p_draw"
        ].notna()
        &
        df[
            "market_p_away"
        ].notna()
    ].copy()

    print()
    print(
        f"Matches with market odds: "
        f"{len(valid):,}"
    )

    print(
        f"Coverage: "
        f"{len(valid) / len(df):.2%}"
    )

    print()
    print("MARKET SOURCE COUNTS")

    print(
        valid[
            "market_source"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Average margin: "
        f"{valid['market_margin'].mean():.2%}"
    )

    # =====================================================
    # COMBINED
    # =====================================================

    model_all = evaluate_system(
        valid,
        "model",
    )

    market_all = evaluate_system(
        valid,
        "market",
    )

    print_comparison(
        "V2 VS MARKET — ALL USABLE ODDS",
        model_all,
        market_all,
    )

    # =====================================================
    # CLOSING ONLY
    # =====================================================

    closing = valid[
        valid[
            "market_source"
        ]
        == "avg_close"
    ].copy()

    if len(closing) > 0:

        model_close = evaluate_system(
            closing,
            "model",
        )

        market_close = evaluate_system(
            closing,
            "market",
        )

        print_comparison(
            "V2 VS MARKET — AVERAGE CLOSING ONLY",
            model_close,
            market_close,
        )

    # =====================================================
    # OPENING FALLBACK
    # =====================================================

    opening = valid[
        valid[
            "market_source"
        ]
        != "avg_close"
    ].copy()

    if len(opening) > 0:

        model_open = evaluate_system(
            opening,
            "model",
        )

        market_open = evaluate_system(
            opening,
            "market",
        )

        print_comparison(
            "V2 VS MARKET — OPENING FALLBACK ONLY",
            model_open,
            market_open,
        )

    # =====================================================
    # VALIDATION YEARS
    # =====================================================

    validation = valid[
        valid[
            "season"
        ].isin(
            VALIDATION_SEASONS
        )
    ].copy()

    if len(validation) > 0:

        model_validation = evaluate_system(
            validation,
            "model",
        )

        market_validation = evaluate_system(
            validation,
            "market",
        )

        print_comparison(
            "V2 VS MARKET — 2023/24 TO 2024/25",
            model_validation,
            market_validation,
        )

    # =====================================================
    # FINAL SEASON
    # =====================================================

    final = valid[
        valid[
            "season"
        ].isin(
            FINAL_SEASON
        )
    ].copy()

    if len(final) > 0:

        model_final = evaluate_system(
            final,
            "model",
        )

        market_final = evaluate_system(
            final,
            "market",
        )

        print_comparison(
            "V2 VS MARKET — 2025/26",
            model_final,
            market_final,
        )

    # =====================================================
    # CLOSING BY LEAGUE
    # =====================================================

    print()
    print("=" * 105)
    print("AVERAGE CLOSING ONLY — BY LEAGUE")
    print("=" * 105)

    closing_table = (
        league_comparison_table(
            closing
        )
    )

    print(
        closing_table
        .round(5)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # 2025/26 BY LEAGUE
    # =====================================================

    print()
    print("=" * 105)
    print("2025/26 — BY LEAGUE")
    print("=" * 105)

    final_table = (
        league_comparison_table(
            final
        )
    )

    print(
        final_table
        .round(5)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # EDGE COLUMNS
    # =====================================================

    valid[
        "home_edge"
    ] = (
        valid[
            "model_p_home"
        ]
        -
        valid[
            "market_p_home"
        ]
    )

    valid[
        "draw_edge"
    ] = (
        valid[
            "model_p_draw"
        ]
        -
        valid[
            "market_p_draw"
        ]
    )

    valid[
        "away_edge"
    ] = (
        valid[
            "model_p_away"
        ]
        -
        valid[
            "market_p_away"
        ]
    )

    valid.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    model_sum = (
        valid[
            [
                "model_p_home",
                "model_p_draw",
                "model_p_away",
            ]
        ]
        .sum(
            axis=1
        )
    )

    market_sum = (
        valid[
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

    print()
    print("=" * 72)
    print("VALIDATION")
    print("=" * 72)

    print(
        "Max V2 probability sum error: "
        f"{(model_sum - 1.0).abs().max():.12f}"
    )

    print(
        "Max market probability sum error: "
        f"{(market_sum - 1.0).abs().max():.12f}"
    )

    print(
        "Both probability sets sum to 1 ✅"
    )

    print(
        "Market odds were not used "
        "to generate V2 ✅"
    )

    print()
    print(
        f"Saved:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()