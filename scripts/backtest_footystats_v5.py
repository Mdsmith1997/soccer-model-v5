from pathlib import Path

import numpy as np
import pandas as pd

import confirm_opponent_adjusted_recency_v5 as v5
import tune_overall_venue_v5 as core


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = (
    ROOT
    / "data"
    / "processed"
)

UNDERSTAT_FILE = (
    PROCESSED
    / "understat_xg_matched.csv"
)

FOOTYSTATS_MATCHED_FILE = (
    PROCESSED
    / "footystats_understat_xg_history_matched.csv"
)

OUTPUT_SUMMARY = (
    PROCESSED
    / "footystats_v5_backtest_summary.csv"
)

OUTPUT_PREDICTIONS = (
    PROCESSED
    / "footystats_v5_backtest_predictions.csv"
)


# ============================================================
# TEST SETTINGS
# ============================================================

TEST_SEASONS = {
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
}

# Frozen winning V5 overall / venue structure.
OVERALL_WEIGHT = 0.80

EPS = 1e-12


# ============================================================
# METRICS
# ============================================================

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

    chosen = np.clip(
        chosen,
        EPS,
        1.0,
    )

    return (
        -np.log(
            chosen
        )
    ).mean()


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

    return np.mean(
        np.sum(
            (
                probs
                -
                truth
            ) ** 2,
            axis=1,
        )
    )


def accuracy(
    y_true,
    probs,
):

    predicted = probs.argmax(
        axis=1
    )

    return (
        predicted
        ==
        y_true
    ).mean()


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

    for i in range(
        bins
    ):

        low = edges[i]
        high = edges[i + 1]

        if i == bins - 1:

            mask = (
                (confidence >= low)
                &
                (confidence <= high)
            )

        else:

            mask = (
                (confidence >= low)
                &
                (confidence < high)
            )

        n = int(
            mask.sum()
        )

        if n == 0:

            continue

        bin_acc = correct[
            mask
        ].mean()

        bin_conf = confidence[
            mask
        ].mean()

        ece += (
            n
            /
            len(
                y_true
            )
        ) * abs(
            bin_acc
            -
            bin_conf
        )

    return ece


# ============================================================
# NORMALIZE SEASON
# ============================================================

def normalize_season(
    series,
):

    return (
        series
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.strip()
    )


# ============================================================
# LOAD FOOTYSTATS REPLACEMENT
# ============================================================

def build_footystats_xg_source():

    print()
    print(
        "=" * 90
    )

    print(
        "BUILDING FOOTYSTATS XG REPLACEMENT"
    )

    print(
        "=" * 90
    )

    if not UNDERSTAT_FILE.exists():

        raise FileNotFoundError(
            UNDERSTAT_FILE
        )

    if not FOOTYSTATS_MATCHED_FILE.exists():

        raise FileNotFoundError(
            FOOTYSTATS_MATCHED_FILE
        )

    # --------------------------------------------------------
    # Start with the exact schema that the existing V5 xG
    # loader already expects.
    #
    # Then replace ONLY home_xg / away_xg with FootyStats xG
    # for the EPL seasons being tested.
    #
    # Auxiliary Understat columns remain present only so the
    # existing loader/schema functions continue to work.
    #
    # Frozen V5's xG attack/defense path consumes xG, not
    # these auxiliary fields.
    # --------------------------------------------------------

    understat = pd.read_csv(
        UNDERSTAT_FILE,
        low_memory=False,
    )

    fs = pd.read_csv(
        FOOTYSTATS_MATCHED_FILE,
        low_memory=False,
    )

    understat[
        "season"
    ] = normalize_season(
        understat[
            "season"
        ]
    )

    fs[
        "season"
    ] = normalize_season(
        fs[
            "season"
        ]
    )

    required_fs = [
        "match_id",
        "season",
        "home_xg_fs",
        "away_xg_fs",
    ]

    missing = [
        col
        for col in required_fs
        if col not in fs.columns
    ]

    if missing:

        raise ValueError(
            "FootyStats matched file "
            "missing columns: "
            f"{missing}"
        )

    fs = fs[
        fs[
            "season"
        ].isin(
            TEST_SEASONS
        )
    ].copy()

    fs[
        "home_xg_fs"
    ] = pd.to_numeric(
        fs[
            "home_xg_fs"
        ],
        errors="coerce",
    )

    fs[
        "away_xg_fs"
    ] = pd.to_numeric(
        fs[
            "away_xg_fs"
        ],
        errors="coerce",
    )

    fs = fs.dropna(
        subset=[
            "home_xg_fs",
            "away_xg_fs",
        ]
    )

    if fs[
        "match_id"
    ].duplicated().any():

        dupes = fs.loc[
            fs[
                "match_id"
            ].duplicated(
                keep=False
            ),
            "match_id",
        ]

        raise ValueError(
            "Duplicate FootyStats "
            "match IDs found:\n"
            +
            dupes.to_string(
                index=False
            )
        )

    replacement = fs[
        [
            "match_id",
            "home_xg_fs",
            "away_xg_fs",
        ]
    ].copy()

    synthetic = understat.merge(
        replacement,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Replace ONLY EPL overlapping xG.
    # --------------------------------------------------------

    replacement_mask = (
        synthetic[
            "home_xg_fs"
        ].notna()
        &
        synthetic[
            "away_xg_fs"
        ].notna()
    )

    replaced = int(
        replacement_mask.sum()
    )

    synthetic.loc[
        replacement_mask,
        "home_xg",
    ] = synthetic.loc[
        replacement_mask,
        "home_xg_fs",
    ]

    synthetic.loc[
        replacement_mask,
        "away_xg",
    ] = synthetic.loc[
        replacement_mask,
        "away_xg_fs",
    ]

    # --------------------------------------------------------
    # Recalculate xG-derived match columns when they exist.
    # --------------------------------------------------------

    if (
        "total_xg"
        in
        synthetic.columns
    ):

        synthetic.loc[
            replacement_mask,
            "total_xg",
        ] = (
            synthetic.loc[
                replacement_mask,
                "home_xg",
            ]
            +
            synthetic.loc[
                replacement_mask,
                "away_xg",
            ]
        )

    if (
        "xg_diff_home"
        in
        synthetic.columns
    ):

        synthetic.loc[
            replacement_mask,
            "xg_diff_home",
        ] = (
            synthetic.loc[
                replacement_mask,
                "home_xg",
            ]
            -
            synthetic.loc[
                replacement_mask,
                "away_xg",
            ]
        )

    synthetic = synthetic.drop(
        columns=[
            "home_xg_fs",
            "away_xg_fs",
        ],
        errors="ignore",
    )

    print(
        "Understat source rows:",
        f"{len(understat):,}",
    )

    print(
        "FootyStats matched EPL rows:",
        f"{len(fs):,}",
    )

    print(
        "xG matches replaced:",
        f"{replaced:,}",
    )

    if replaced != 2660:

        print()
        print(
            "WARNING:"
        )

        print(
            "Expected 2,660 EPL "
            "FootyStats replacements."
        )

    return synthetic


# ============================================================
# BUILD COMPONENT STORE FROM A SPECIFIC XG SOURCE
# ============================================================

def build_store_with_xg_source(
    source_name,
    xg_dataframe=None,
):

    print()
    print(
        "=" * 90
    )

    print(
        f"BUILDING V5 COMPONENTS — "
        f"{source_name}"
    )

    print(
        "=" * 90
    )

    # --------------------------------------------------------
    # Save original function so nothing persists.
    # --------------------------------------------------------

    original_load_xg = (
        v5.load_xg
    )

    try:

        if xg_dataframe is not None:

            def replacement_load_xg():

                return (
                    xg_dataframe.copy()
                )

            v5.load_xg = (
                replacement_load_xg
            )

        store = (
            core.build_component_store()
        )

    finally:

        v5.load_xg = (
            original_load_xg
        )

    store[
        "season"
    ] = normalize_season(
        store[
            "season"
        ]
    )

    print(
        "Component-store matches:",
        f"{len(store):,}",
    )

    return store


# ============================================================
# BUILD PREDICTIONS
# ============================================================

def score_store(
    store,
    source_name,
):

    df = store.copy()

    # --------------------------------------------------------
    # Restrict this provider comparison to EPL seasons where
    # FootyStats has complete historical xG.
    # --------------------------------------------------------

    df = df[
        (
            df[
                "league"
            ]
            ==
            "Premier League"
        )
        &
        (
            df[
                "season"
            ].isin(
                TEST_SEASONS
            )
        )
    ].copy()

    if len(df) == 0:

        raise ValueError(
            f"No EPL matches available "
            f"for {source_name}."
        )

    home_lambda, away_lambda = (
        core.build_lambdas(
            df,
            OVERALL_WEIGHT,
        )
    )

    probs = (
        core.calculate_1x2_probs(
            home_lambda,
            away_lambda,
        )
    )

    out = pd.DataFrame(
        {
            "match_id":
                df[
                    "match_id"
                ].values,

            "date":
                df[
                    "date"
                ].values,

            "season":
                df[
                    "season"
                ].values,

            "league":
                df[
                    "league"
                ].values,

            "home_team":
                df[
                    "home_team"
                ].values,

            "away_team":
                (
                    df[
                        "away_team"
                    ].values
                    if
                    "away_team"
                    in
                    df.columns
                    else
                    df[
                        "away_team_check"
                    ].values
                ),

            "home_goals":
                df[
                    "home_goals"
                ].values,

            "away_goals":
                df[
                    "away_goals"
                ].values,

            "home_lambda":
                np.asarray(
                    home_lambda
                ),

            "away_lambda":
                np.asarray(
                    away_lambda
                ),

            "p_home":
                probs[
                    :,
                    0
                ],

            "p_draw":
                probs[
                    :,
                    1
                ],

            "p_away":
                probs[
                    :,
                    2
                ],

            "xg_source":
                source_name,
        }
    )

    return out


# ============================================================
# EVALUATE
# ============================================================

def valid_prediction_mask(
    df,
):

    working = df.copy()

    probability_cols = [
        "p_home",
        "p_draw",
        "p_away",
    ]

    required_cols = [
        "home_goals",
        "away_goals",
        "home_lambda",
        "away_lambda",
        *probability_cols,
    ]

    numeric = working[
        required_cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    finite_mask = np.isfinite(
        numeric.to_numpy(
            dtype=float
        )
    ).all(
        axis=1
    )

    positive_prob_mask = (
        numeric[
            probability_cols
        ]
        >=
        0.0
    ).all(
        axis=1
    )

    probability_sum = (
        numeric[
            probability_cols
        ]
        .sum(
            axis=1
        )
    )

    valid_sum_mask = np.isclose(
        probability_sum,
        1.0,
        atol=1e-6,
    )

    return pd.Series(
        finite_mask
        &
        positive_prob_mask.to_numpy()
        &
        valid_sum_mask,
        index=working.index,
    )


def evaluate_predictions(
    df,
):

    working = df.copy()

    probability_cols = [
        "p_home",
        "p_draw",
        "p_away",
    ]

    required_cols = [
        "home_goals",
        "away_goals",
        "home_lambda",
        "away_lambda",
        *probability_cols,
    ]

    for col in required_cols:

        working[col] = pd.to_numeric(
            working[col],
            errors="coerce",
        )

    valid_mask = valid_prediction_mask(
        working
    )

    valid = working.loc[
        valid_mask
    ].copy()

    dropped = (
        len(
            working
        )
        -
        len(
            valid
        )
    )

    if len(valid) == 0:

        raise ValueError(
            "No valid finite V5 "
            "probability rows found."
        )

    y = result_classes(
        valid[
            "home_goals"
        ].values,
        valid[
            "away_goals"
        ].values,
    )

    probs = valid[
        probability_cols
    ].to_numpy(
        dtype=float
    )

    return {
        "games":
            len(
                valid
            ),

        "dropped_invalid":
            dropped,

        "accuracy":
            100.0
            *
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
            100.0
            *
            expected_calibration_error(
                y,
                probs,
            ),

        "avg_confidence":
            100.0
            *
            probs.max(
                axis=1
            ).mean(),

        "avg_home_lambda":
            valid[
                "home_lambda"
            ].mean(),

        "avg_away_lambda":
            valid[
                "away_lambda"
            ].mean(),
    }


# ============================================================
# ALIGN PROVIDERS ON IDENTICAL VALID ROWS
# ============================================================

def align_valid_probability_rows(
    understat,
    footystats,
):

    us = understat.copy()
    fs = footystats.copy()

    common_ids = (
        set(
            us[
                "match_id"
            ]
        )
        &
        set(
            fs[
                "match_id"
            ]
        )
    )

    us = (
        us[
            us[
                "match_id"
            ].isin(
                common_ids
            )
        ]
        .sort_values(
            "match_id"
        )
        .reset_index(
            drop=True
        )
    )

    fs = (
        fs[
            fs[
                "match_id"
            ].isin(
                common_ids
            )
        ]
        .sort_values(
            "match_id"
        )
        .reset_index(
            drop=True
        )
    )

    if len(us) != len(fs):

        raise ValueError(
            "Provider alignment produced "
            "different row counts."
        )

    if not (
        us[
            "match_id"
        ].astype(str).values
        ==
        fs[
            "match_id"
        ].astype(str).values
    ).all():

        raise ValueError(
            "Provider rows are not aligned "
            "on match_id."
        )

    us_valid = valid_prediction_mask(
        us
    ).to_numpy()

    fs_valid = valid_prediction_mask(
        fs
    ).to_numpy()

    common_valid = (
        us_valid
        &
        fs_valid
    )

    return (
        us.loc[
            common_valid
        ].copy().reset_index(
            drop=True
        ),
        fs.loc[
            common_valid
        ].copy().reset_index(
            drop=True
        ),
        int(
            len(us)
            -
            common_valid.sum()
        ),
    )


# ============================================================
# MERGE SOURCE PREDICTIONS
# ============================================================

def compare_predictions(
    understat,
    footystats,
):

    us = understat.rename(
        columns={
            "home_lambda":
                "us_home_lambda",

            "away_lambda":
                "us_away_lambda",

            "p_home":
                "us_p_home",

            "p_draw":
                "us_p_draw",

            "p_away":
                "us_p_away",
        }
    )

    fs = footystats[
        [
            "match_id",
            "home_lambda",
            "away_lambda",
            "p_home",
            "p_draw",
            "p_away",
        ]
    ].rename(
        columns={
            "home_lambda":
                "fs_home_lambda",

            "away_lambda":
                "fs_away_lambda",

            "p_home":
                "fs_p_home",

            "p_draw":
                "fs_p_draw",

            "p_away":
                "fs_p_away",
        }
    )

    merged = us.merge(
        fs,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    merged[
        "home_lambda_diff"
    ] = (
        merged[
            "fs_home_lambda"
        ]
        -
        merged[
            "us_home_lambda"
        ]
    )

    merged[
        "away_lambda_diff"
    ] = (
        merged[
            "fs_away_lambda"
        ]
        -
        merged[
            "us_away_lambda"
        ]
    )

    merged[
        "home_prob_diff"
    ] = (
        merged[
            "fs_p_home"
        ]
        -
        merged[
            "us_p_home"
        ]
    )

    merged[
        "draw_prob_diff"
    ] = (
        merged[
            "fs_p_draw"
        ]
        -
        merged[
            "us_p_draw"
        ]
    )

    merged[
        "away_prob_diff"
    ] = (
        merged[
            "fs_p_away"
        ]
        -
        merged[
            "us_p_away"
        ]
    )

    merged[
        "max_abs_probability_diff"
    ] = (
        merged[
            [
                "home_prob_diff",
                "draw_prob_diff",
                "away_prob_diff",
            ]
        ]
        .abs()
        .max(
            axis=1
        )
    )

    return merged


# ============================================================
# PRINT METRIC COMPARISON
# ============================================================

def print_metric_comparison(
    title,
    understat,
    footystats,
):

    original_common = len(
        set(
            understat[
                "match_id"
            ]
        )
        &
        set(
            footystats[
                "match_id"
            ]
        )
    )

    (
        understat_valid,
        footystats_valid,
        dropped_common,
    ) = align_valid_probability_rows(
        understat,
        footystats,
    )

    us = evaluate_predictions(
        understat_valid
    )

    fs = evaluate_predictions(
        footystats_valid
    )

    print()
    print(
        "=" * 100
    )

    print(
        title
    )

    print(
        "=" * 100
    )

    print(
        f"{'Metric':<22}"
        f"{'Understat V5':>16}"
        f"{'FootyStats V5':>18}"
        f"{'Change':>14}"
    )

    print(
        "-" * 70
    )

    print(
        f"{'Common Rows':<22}"
        f"{original_common:>16,}"
        f"{original_common:>18,}"
        f"{'':>14}"
    )

    print(
        f"{'Valid Games':<22}"
        f"{us['games']:>16,}"
        f"{fs['games']:>18,}"
        f"{'':>14}"
    )

    print(
        f"{'Dropped Common':<22}"
        f"{dropped_common:>16,}"
        f"{dropped_common:>18,}"
        f"{'':>14}"
    )

    print(
        f"{'Accuracy':<22}"
        f"{us['accuracy']:>15.2f}%"
        f"{fs['accuracy']:>17.2f}%"
        f"{fs['accuracy'] - us['accuracy']:>+13.2f}%"
    )

    print(
        f"{'Log Loss':<22}"
        f"{us['log_loss']:>16.5f}"
        f"{fs['log_loss']:>18.5f}"
        f"{fs['log_loss'] - us['log_loss']:>+14.5f}"
    )

    print(
        f"{'Brier':<22}"
        f"{us['brier']:>16.5f}"
        f"{fs['brier']:>18.5f}"
        f"{fs['brier'] - us['brier']:>+14.5f}"
    )

    print(
        f"{'ECE':<22}"
        f"{us['ece']:>15.2f}%"
        f"{fs['ece']:>17.2f}%"
        f"{fs['ece'] - us['ece']:>+13.2f}%"
    )

    print(
        f"{'Avg Confidence':<22}"
        f"{us['avg_confidence']:>15.2f}%"
        f"{fs['avg_confidence']:>17.2f}%"
        f"{fs['avg_confidence'] - us['avg_confidence']:>+13.2f}%"
    )

    print(
        f"{'Avg Home Lambda':<22}"
        f"{us['avg_home_lambda']:>16.4f}"
        f"{fs['avg_home_lambda']:>18.4f}"
        f"{fs['avg_home_lambda'] - us['avg_home_lambda']:>+14.4f}"
    )

    print(
        f"{'Avg Away Lambda':<22}"
        f"{us['avg_away_lambda']:>16.4f}"
        f"{fs['avg_away_lambda']:>18.4f}"
        f"{fs['avg_away_lambda'] - us['avg_away_lambda']:>+14.4f}"
    )

    return (
        us,
        fs,
        understat_valid,
        footystats_valid,
        dropped_common,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=============================="
    )

    print(
        "RAW FOOTYSTATS V5 BACKTEST"
    )

    print(
        "=============================="
    )

    print()

    print(
        "Purpose:"
    )

    print(
        "Swap only the historical xG "
        "provider from Understat to "
        "raw FootyStats."
    )

    print()

    print(
        "Frozen V5:"
    )

    print(
        "Goals: 9%"
    )

    print(
        "xG: 75%"
    )

    print(
        "Shots: 16%"
    )

    print(
        "Recency: "
        "0.975 / 0.925 / 0.850"
    )

    print(
        "Opponent strength: 0.875"
    )

    print(
        "Overall / venue: 80% / 20%"
    )

    print()

    print(
        "NO MODEL PARAMETERS "
        "ARE BEING FIT OR SELECTED."
    )

    # ========================================================
    # ORIGINAL UNDERSTAT V5
    # ========================================================

    understat_store = (
        build_store_with_xg_source(
            "UNDERSTAT"
        )
    )

    understat_predictions = (
        score_store(
            understat_store,
            "UNDERSTAT",
        )
    )

    # ========================================================
    # RAW FOOTYSTATS V5
    # ========================================================

    footystats_source = (
        build_footystats_xg_source()
    )

    footystats_store = (
        build_store_with_xg_source(
            "FOOTYSTATS_RAW",
            footystats_source,
        )
    )

    footystats_predictions = (
        score_store(
            footystats_store,
            "FOOTYSTATS_RAW",
        )
    )

    # ========================================================
    # MATCH COMMON SET
    # ========================================================

    common_ids = (
        set(
            understat_predictions[
                "match_id"
            ]
        )
        &
        set(
            footystats_predictions[
                "match_id"
            ]
        )
    )

    understat_predictions = (
        understat_predictions[
            understat_predictions[
                "match_id"
            ].isin(
                common_ids
            )
        ]
        .copy()
        .sort_values(
            "match_id"
        )
        .reset_index(
            drop=True
        )
    )

    footystats_predictions = (
        footystats_predictions[
            footystats_predictions[
                "match_id"
            ].isin(
                common_ids
            )
        ]
        .copy()
        .sort_values(
            "match_id"
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "Common EPL predictions:",
        f"{len(common_ids):,}",
    )

    # ========================================================
    # OVERALL — IDENTICAL VALID SAMPLE
    # ========================================================

    summary_rows = []

    (
        overall_us,
        overall_fs,
        overall_us_valid,
        overall_fs_valid,
        overall_dropped,
    ) = print_metric_comparison(
        "ALL EPL — 2018/19 TO 2024/25",
        understat_predictions,
        footystats_predictions,
    )

    summary_rows.append(
        {
            "period":
                "ALL_1819_2425",

            "season":
                "ALL",

            "source":
                "UNDERSTAT",

            "common_rows":
                len(common_ids),

            "dropped_common":
                overall_dropped,

            **overall_us,
        }
    )

    summary_rows.append(
        {
            "period":
                "ALL_1819_2425",

            "season":
                "ALL",

            "source":
                "FOOTYSTATS_RAW",

            "common_rows":
                len(common_ids),

            "dropped_common":
                overall_dropped,

            **overall_fs,
        }
    )

    # ========================================================
    # BY SEASON — IDENTICAL VALID SAMPLE
    # ========================================================

    season_order = [
        "1819",
        "1920",
        "2021",
        "2122",
        "2223",
        "2324",
        "2425",
    ]

    for season in season_order:

        us_sub = (
            understat_predictions[
                understat_predictions[
                    "season"
                ]
                ==
                season
            ]
            .copy()
        )

        fs_sub = (
            footystats_predictions[
                footystats_predictions[
                    "season"
                ]
                ==
                season
            ]
            .copy()
        )

        if (
            len(
                us_sub
            )
            ==
            0
            or
            len(
                fs_sub
            )
            ==
            0
        ):

            continue

        (
            us_metrics,
            fs_metrics,
            us_valid,
            fs_valid,
            dropped_common,
        ) = print_metric_comparison(
            f"SEASON {season}",
            us_sub,
            fs_sub,
        )

        common_season_rows = len(
            set(
                us_sub[
                    "match_id"
                ]
            )
            &
            set(
                fs_sub[
                    "match_id"
                ]
            )
        )

        summary_rows.append(
            {
                "period":
                    "SEASON",

                "season":
                    season,

                "source":
                    "UNDERSTAT",

                "common_rows":
                    common_season_rows,

                "dropped_common":
                    dropped_common,

                **us_metrics,
            }
        )

        summary_rows.append(
            {
                "period":
                    "SEASON",

                "season":
                    season,

                "source":
                    "FOOTYSTATS_RAW",

                "common_rows":
                    common_season_rows,

                "dropped_common":
                    dropped_common,

                **fs_metrics,
            }
        )

    # ========================================================
    # PROVIDER DIFFERENCES — VALID COMMON ROWS ONLY
    # ========================================================

    (
        understat_valid,
        footystats_valid,
        dropped_common,
    ) = align_valid_probability_rows(
        understat_predictions,
        footystats_predictions,
    )

    print()
    print(
        "Valid common rows for provider "
        "difference analysis:",
        f"{len(understat_valid):,}",
    )

    print(
        "Dropped invalid common rows:",
        f"{dropped_common:,}",
    )

    compared = compare_predictions(
        understat_valid,
        footystats_valid,
    )

    print()
    print(
        "=" * 105
    )

    print(
        "PROBABILITY / LAMBDA DIFFERENCES"
    )

    print(
        "=" * 105
    )

    print(
        f"Mean |home λ diff|: "
        f"{compared['home_lambda_diff'].abs().mean():.4f}"
    )

    print(
        f"Mean |away λ diff|: "
        f"{compared['away_lambda_diff'].abs().mean():.4f}"
    )

    print(
        f"Mean |home probability diff|: "
        f"{100 * compared['home_prob_diff'].abs().mean():.2f}%"
    )

    print(
        f"Mean |draw probability diff|: "
        f"{100 * compared['draw_prob_diff'].abs().mean():.2f}%"
    )

    print(
        f"Mean |away probability diff|: "
        f"{100 * compared['away_prob_diff'].abs().mean():.2f}%"
    )

    print(
        f"Mean max probability difference: "
        f"{100 * compared['max_abs_probability_diff'].mean():.2f}%"
    )

    print(
        f"95th percentile max probability difference: "
        f"{100 * compared['max_abs_probability_diff'].quantile(0.95):.2f}%"
    )

    # ========================================================
    # BIGGEST MODEL DIFFERENCES
    # ========================================================

    print()
    print(
        "=" * 125
    )

    print(
        "20 LARGEST V5 PROBABILITY DIFFERENCES"
    )

    print(
        "=" * 125
    )

    largest = (
        compared
        .sort_values(
            "max_abs_probability_diff",
            ascending=False,
        )
        .head(
            20
        )
    )

    print(
        largest[
            [
                "date",
                "season",
                "home_team",
                "away_team",
                "us_p_home",
                "fs_p_home",
                "us_p_draw",
                "fs_p_draw",
                "us_p_away",
                "fs_p_away",
                "max_abs_probability_diff",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # DECISION AGREEMENT
    # ========================================================

    us_probs = compared[
        [
            "us_p_home",
            "us_p_draw",
            "us_p_away",
        ]
    ].to_numpy(
        dtype=float
    )

    fs_probs = compared[
        [
            "fs_p_home",
            "fs_p_draw",
            "fs_p_away",
        ]
    ].to_numpy(
        dtype=float
    )

    same_pick = (
        us_probs.argmax(
            axis=1
        )
        ==
        fs_probs.argmax(
            axis=1
        )
    )

    print()
    print(
        "=" * 105
    )

    print(
        "MODEL AGREEMENT"
    )

    print(
        "=" * 105
    )

    print(
        "Same predicted result:",
        f"{100 * same_pick.mean():.2f}%"
    )

    print(
        "Different predicted result:",
        f"{100 * (1 - same_pick.mean()):.2f}%"
    )

    # ========================================================
    # SAVE
    # ========================================================

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    compared.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print()
    print(
        "=============================="
    )

    print(
        "FOOTYSTATS V5 TEST COMPLETE"
    )

    print(
        "=============================="
    )

    print()

    print(
        "FootyStats used RAW ✅"
    )

    print(
        "No Huber adapter used ✅"
    )

    print(
        "Goals / shots unchanged ✅"
    )

    print(
        "V5 weights unchanged ✅"
    )

    print(
        "V5 recencies unchanged ✅"
    )

    print(
        "Opponent strength unchanged ✅"
    )

    print(
        "Overall / venue unchanged ✅"
    )

    print(
        "Identical valid rows used for "
        "provider comparison ✅"
    )

    print(
        "No parameter selected using "
        "test results ✅"
    )

    print()
    print(
        "Summary:"
    )

    print(
        OUTPUT_SUMMARY
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
