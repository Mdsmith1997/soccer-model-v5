from pathlib import Path
from itertools import product
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    SplineTransformer,
    StandardScaler,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "btts_feature_store_v2.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "btts_lab_v2"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEV_YEARS = [
    2021,
    2022,
    2023,
    2024,
]

VALIDATION_YEAR = 2025

FINAL_YEAR = 2026

MIN_TRAIN = 5000


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 120)
print("BTTS EXPERIMENT LAB")
print("=" * 120)

df = pd.read_csv(
    INPUT,
    low_memory=False,
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)

df["btts_yes"] = pd.to_numeric(
    df["btts_yes"],
    errors="coerce",
)

df["poisson_btts"] = pd.to_numeric(
    df["poisson_btts"],
    errors="coerce",
)

df = df[
    df["date"].notna()
    &
    df["btts_yes"].notna()
    &
    df["poisson_btts"].notna()
].copy()

df["btts_yes"] = (
    df["btts_yes"]
    .astype(int)
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# POISSON LOGIT
# ============================================================

p = df["poisson_btts"].clip(
    0.001,
    0.999,
)

df["poisson_logit"] = np.log(
    p
    /
    (1.0 - p)
)


# ============================================================
# HINGE FEATURES
#
# Explicitly test weaker-lambda transition points.
# Example:
#   hinge_above_1.1 = max(lambda_min - 1.1, 0)
# ============================================================

LAMBDA_CUTS = np.round(
    np.arange(
        0.8,
        2.01,
        0.1,
    ),
    1,
).tolist()


for cut in LAMBDA_CUTS:

    label = str(cut).replace(
        ".",
        "_",
    )

    df[
        f"lambda_min_above_{label}"
    ] = np.maximum(
        pd.to_numeric(
            df["lambda_min"],
            errors="coerce",
        )
        -
        cut,
        0.0,
    )

    df[
        f"lambda_min_below_{label}"
    ] = np.maximum(
        cut
        -
        pd.to_numeric(
            df["lambda_min"],
            errors="coerce",
        ),
        0.0,
    )


# ============================================================
# TOTAL-LAMBDA HINGES
# ============================================================

TOTAL_CUTS = [
    2.25,
    2.50,
    2.75,
    3.00,
    3.25,
    3.50,
    4.00,
]


for cut in TOTAL_CUTS:

    label = str(cut).replace(
        ".",
        "_",
    )

    df[
        f"lambda_total_above_{label}"
    ] = np.maximum(
        pd.to_numeric(
            df["lambda_total"],
            errors="coerce",
        )
        -
        cut,
        0.0,
    )


# ============================================================
# FEATURE SETS
# ============================================================

CORE = [
    "poisson_logit",
    "lambda_min",
    "lambda_total",
    "lambda_gap",
    "lambda_balance_ratio",
    "weaker_team_score_probability",
]


MATCHUP = CORE + [
    "xg_matchup_overall_min",
    "xg_matchup_overall_balance",
    "shot_matchup_overall_min",
    "shot_matchup_overall_balance",
    "goal_matchup_overall_min",
    "goal_matchup_overall_balance",
]


ENVIRONMENT = MATCHUP + [
    "league_goal_environment",
    "league_xg_environment",
    "xg_attack_balance",
    "goal_attack_balance",
    "minimum_team_history",
]


FEATURE_SETS = {
    "core": CORE,
    "matchup": MATCHUP,
    "environment": ENVIRONMENT,
}


for key in FEATURE_SETS:

    FEATURE_SETS[key] = [
        c
        for c in FEATURE_SETS[key]
        if c in df.columns
    ]


# ============================================================
# METRICS
# ============================================================

def metrics(
    y,
    pred,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    pred = np.asarray(
        pred,
        dtype=float,
    )

    mask = (
        np.isfinite(y)
        &
        np.isfinite(pred)
    )

    y = y[mask]
    pred = pred[mask]

    pred = np.clip(
        pred,
        1e-8,
        1 - 1e-8,
    )

    return {
        "games":
            len(y),

        "brier":
            brier_score_loss(
                y,
                pred,
            ),

        "log_loss":
            log_loss(
                y,
                pred,
            ),

        "auc":
            (
                roc_auc_score(
                    y,
                    pred,
                )
                if len(
                    np.unique(y)
                ) > 1
                else np.nan
            ),

        "avg_pred":
            pred.mean(),

        "actual_rate":
            y.mean(),
    }


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_linear_model(
    features,
    C,
):

    numeric = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
        ]
    )

    league = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    prep = ColumnTransformer(
        [
            (
                "num",
                numeric,
                features,
            ),
            (
                "league",
                league,
                ["league"],
            ),
        ]
    )

    return Pipeline(
        [
            (
                "prep",
                prep,
            ),
            (
                "model",
                LogisticRegression(
                    C=C,
                    max_iter=3000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def build_spline_model(
    linear_features,
    spline_features,
    knots,
    C,
):

    linear = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
        ]
    )

    spline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "spline",
                SplineTransformer(
                    n_knots=knots,
                    degree=3,
                    include_bias=False,
                    extrapolation="linear",
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
        ]
    )

    league = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    prep = ColumnTransformer(
        [
            (
                "linear",
                linear,
                linear_features,
            ),
            (
                "spline",
                spline,
                spline_features,
            ),
            (
                "league",
                league,
                ["league"],
            ),
        ]
    )

    return Pipeline(
        [
            (
                "prep",
                prep,
            ),
            (
                "model",
                LogisticRegression(
                    C=C,
                    max_iter=4000,
                    solver="liblinear",
                ),
            ),
        ]
    )


# ============================================================
# CONFIGURATION GENERATOR
# ============================================================

configs = []


# ------------------------------------------------------------
# Linear compact models
# ------------------------------------------------------------

for feature_set, C in product(
    FEATURE_SETS.keys(),
    [
        0.01,
        0.03,
        0.05,
        0.10,
        0.20,
        0.50,
    ],
):

    configs.append(
        {
            "family":
                "linear",

            "feature_set":
                feature_set,

            "C":
                C,

            "blend":
                1.0,
        }
    )


# ------------------------------------------------------------
# Spline models
# ------------------------------------------------------------

for feature_set, knots, C, blend in product(
    FEATURE_SETS.keys(),
    [
        3,
        4,
        5,
        6,
        7,
    ],
    [
        0.01,
        0.03,
        0.05,
        0.10,
        0.20,
    ],
    [
        0.25,
        0.50,
        0.75,
        1.00,
    ],
):

    configs.append(
        {
            "family":
                "spline",

            "feature_set":
                feature_set,

            "knots":
                knots,

            "C":
                C,

            "blend":
                blend,
        }
    )


# ------------------------------------------------------------
# Weaker-lambda hinge models
# ------------------------------------------------------------

for feature_set, cut, C, blend in product(
    FEATURE_SETS.keys(),
    LAMBDA_CUTS,
    [
        0.01,
        0.03,
        0.05,
        0.10,
        0.20,
    ],
    [
        0.50,
        0.75,
        1.00,
    ],
):

    configs.append(
        {
            "family":
                "lambda_hinge",

            "feature_set":
                feature_set,

            "lambda_cut":
                cut,

            "C":
                C,

            "blend":
                blend,
        }
    )


# ------------------------------------------------------------
# Combined lambda-min + total-lambda hinge
# ------------------------------------------------------------

for lambda_cut, total_cut, C, blend in product(
    [
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
    ],
    TOTAL_CUTS,
    [
        0.03,
        0.05,
        0.10,
    ],
    [
        0.50,
        0.75,
        1.00,
    ],
):

    configs.append(
        {
            "family":
                "double_hinge",

            "feature_set":
                "environment",

            "lambda_cut":
                lambda_cut,

            "total_cut":
                total_cut,

            "C":
                C,

            "blend":
                blend,
        }
    )


# ============================================================
# IDS
# ============================================================

for i, cfg in enumerate(
    configs,
    start=1,
):

    cfg["config_id"] = (
        f"CFG_{i:04d}"
    )


print()
print(
    "Total candidate configurations:",
    len(configs),
)

print()
print(
    "Lambda-min cutoffs:",
    LAMBDA_CUTS,
)

print(
    "Total-lambda cutoffs:",
    TOTAL_CUTS,
)


# ============================================================
# GET FEATURES FOR CONFIG
# ============================================================

def get_model_and_columns(
    cfg,
):

    base = FEATURE_SETS[
        cfg["feature_set"]
    ].copy()


    if cfg["family"] == "linear":

        model = build_linear_model(
            base,
            cfg["C"],
        )

        columns = (
            base
            +
            ["league"]
        )

        return model, columns


    if cfg["family"] == "spline":

        spline_candidates = [
            "lambda_min",
            "lambda_total",
            "lambda_gap",
            "weaker_team_score_probability",
            "xg_matchup_overall_min",
            "shot_matchup_overall_min",
            "goal_matchup_overall_min",
        ]

        spline_features = [
            c
            for c in spline_candidates
            if c in base
        ]

        linear_features = [
            c
            for c in base
            if c not in spline_features
        ]

        model = build_spline_model(
            linear_features,
            spline_features,
            cfg["knots"],
            cfg["C"],
        )

        columns = (
            linear_features
            +
            spline_features
            +
            ["league"]
        )

        return model, columns


    if cfg["family"] == "lambda_hinge":

        cut = cfg[
            "lambda_cut"
        ]

        label = str(
            cut
        ).replace(
            ".",
            "_",
        )

        features = (
            base
            +
            [
                f"lambda_min_above_{label}",
                f"lambda_min_below_{label}",
            ]
        )

        features = list(
            dict.fromkeys(
                features
            )
        )

        model = build_linear_model(
            features,
            cfg["C"],
        )

        columns = (
            features
            +
            ["league"]
        )

        return model, columns


    if cfg["family"] == "double_hinge":

        lcut = str(
            cfg["lambda_cut"]
        ).replace(
            ".",
            "_",
        )

        tcut = str(
            cfg["total_cut"]
        ).replace(
            ".",
            "_",
        )

        features = (
            base
            +
            [
                f"lambda_min_above_{lcut}",
                f"lambda_min_below_{lcut}",
                f"lambda_total_above_{tcut}",
            ]
        )

        features = list(
            dict.fromkeys(
                features
            )
        )

        model = build_linear_model(
            features,
            cfg["C"],
        )

        columns = (
            features
            +
            ["league"]
        )

        return model, columns


    raise ValueError(
        cfg["family"]
    )


# ============================================================
# SINGLE YEAR OOS
# ============================================================

def predict_year(
    cfg,
    year,
):

    start = pd.Timestamp(
        year,
        1,
        1,
    )

    end = pd.Timestamp(
        year + 1,
        1,
        1,
    )

    train = df[
        df["date"]
        <
        start
    ].copy()

    test = df[
        (df["date"] >= start)
        &
        (df["date"] < end)
    ].copy()


    if len(train) < MIN_TRAIN:

        return None


    model, columns = (
        get_model_and_columns(
            cfg
        )
    )


    model.fit(
        train[columns],
        train["btts_yes"],
    )


    model_prob = (
        model.predict_proba(
            test[columns]
        )[:, 1]
    )


    blend = cfg[
        "blend"
    ]


    final_prob = (
        blend
        *
        model_prob
        +
        (
            1.0
            -
            blend
        )
        *
        test[
            "poisson_btts"
        ].to_numpy()
    )


    return (
        test,
        final_prob,
        model_prob,
    )


# ============================================================
# POISSON BASELINES
# ============================================================

baseline_rows = []


for year in (
    DEV_YEARS
    +
    [
        VALIDATION_YEAR,
        FINAL_YEAR,
    ]
):

    z = df[
        df["date"].dt.year
        == year
    ]

    r = metrics(
        z["btts_yes"],
        z["poisson_btts"],
    )

    r["year"] = year

    baseline_rows.append(
        r
    )


baseline_df = pd.DataFrame(
    baseline_rows
)


# ============================================================
# PHASE 1:
# DEVELOPMENT SEARCH — 2021-2024 ONLY
# ============================================================

print()
print("=" * 120)
print("PHASE 1 — DEVELOPMENT SEARCH")
print("YEARS: 2021-2024")
print("=" * 120)


dev_rows = []


for num, cfg in enumerate(
    configs,
    start=1,
):

    year_results = []


    for year in DEV_YEARS:

        output = predict_year(
            cfg,
            year,
        )

        if output is None:
            continue

        test, pred, _ = output

        r = metrics(
            test["btts_yes"],
            pred,
        )

        baseline = baseline_df[
            baseline_df["year"]
            == year
        ].iloc[0]


        year_results.append(
            {
                "year":
                    year,

                "games":
                    r["games"],

                "brier":
                    r["brier"],

                "log_loss":
                    r["log_loss"],

                "auc":
                    r["auc"],

                "brier_improvement":
                    (
                        baseline["brier"]
                        -
                        r["brier"]
                    ),

                "logloss_improvement":
                    (
                        baseline["log_loss"]
                        -
                        r["log_loss"]
                    ),

                "auc_improvement":
                    (
                        r["auc"]
                        -
                        baseline["auc"]
                    ),
            }
        )


    yr = pd.DataFrame(
        year_results
    )


    if len(yr) == 0:
        continue


    dev_rows.append(
        {
            **cfg,

            "dev_years":
                len(yr),

            "dev_games":
                int(
                    yr["games"].sum()
                ),

            "dev_brier":
                np.average(
                    yr["brier"],
                    weights=yr["games"],
                ),

            "dev_log_loss":
                np.average(
                    yr["log_loss"],
                    weights=yr["games"],
                ),

            "dev_auc":
                np.average(
                    yr["auc"],
                    weights=yr["games"],
                ),

            "dev_brier_improvement":
                np.average(
                    yr[
                        "brier_improvement"
                    ],
                    weights=yr["games"],
                ),

            "dev_logloss_improvement":
                np.average(
                    yr[
                        "logloss_improvement"
                    ],
                    weights=yr["games"],
                ),

            "dev_auc_improvement":
                np.average(
                    yr[
                        "auc_improvement"
                    ],
                    weights=yr["games"],
                ),

            "dev_brier_years_won":
                int(
                    (
                        yr[
                            "brier_improvement"
                        ]
                        >
                        0
                    ).sum()
                ),

            "dev_logloss_years_won":
                int(
                    (
                        yr[
                            "logloss_improvement"
                        ]
                        >
                        0
                    ).sum()
                ),

            "worst_year_brier_improvement":
                yr[
                    "brier_improvement"
                ].min(),

            "worst_year_logloss_improvement":
                yr[
                    "logloss_improvement"
                ].min(),
        }
    )


    if (
        num % 50
        == 0
        or
        num == len(configs)
    ):

        print(
            f"Completed "
            f"{num:,}/"
            f"{len(configs):,}"
        )


dev = pd.DataFrame(
    dev_rows
)


# ============================================================
# DEVELOPMENT RANK
#
# Prefer:
# 1. Better Brier
# 2. Better log loss
# 3. Stability
# 4. AUC as secondary
# ============================================================

dev["dev_score"] = (
    1000
    *
    dev[
        "dev_brier_improvement"
    ]
    +
    400
    *
    dev[
        "dev_logloss_improvement"
    ]
    +
    0.10
    *
    dev[
        "dev_brier_years_won"
    ]
    +
    0.05
    *
    dev[
        "dev_logloss_years_won"
    ]
    +
    0.5
    *
    dev[
        "dev_auc_improvement"
    ]
)


dev = dev.sort_values(
    [
        "dev_score",
        "dev_brier_improvement",
        "dev_logloss_improvement",
    ],
    ascending=False,
).reset_index(
    drop=True
)


dev[
    "dev_rank"
] = (
    np.arange(
        len(dev)
    )
    +
    1
)


dev.to_csv(
    OUT_DIR
    / "01_development_leaderboard.csv",
    index=False,
)


print()
print("=" * 120)
print("TOP 30 — DEVELOPMENT")
print("=" * 120)


cols = [
    "dev_rank",
    "config_id",
    "family",
    "feature_set",
    "lambda_cut",
    "total_cut",
    "knots",
    "C",
    "blend",
    "dev_brier_improvement",
    "dev_logloss_improvement",
    "dev_auc_improvement",
    "dev_brier_years_won",
    "dev_logloss_years_won",
    "worst_year_brier_improvement",
]


for c in cols:

    if c not in dev.columns:

        dev[c] = np.nan


print()
print(
    dev[
        cols
    ]
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# PHASE 2:
# VALIDATION 2025
#
# Only top 30 development configs get to see 2025.
# ============================================================

TOP_DEV = dev.head(
    30
).copy()


print()
print("=" * 120)
print("PHASE 2 — 2025 VALIDATION")
print("=" * 120)


validation_rows = []


config_lookup = {
    cfg["config_id"]: cfg
    for cfg in configs
}


baseline_2025 = baseline_df[
    baseline_df["year"]
    ==
    VALIDATION_YEAR
].iloc[0]


for _, row in TOP_DEV.iterrows():

    cfg = config_lookup[
        row["config_id"]
    ]

    output = predict_year(
        cfg,
        VALIDATION_YEAR,
    )

    test, pred, _ = output

    r = metrics(
        test["btts_yes"],
        pred,
    )


    validation_rows.append(
        {
            **row.to_dict(),

            "validation_games":
                r["games"],

            "validation_brier":
                r["brier"],

            "validation_log_loss":
                r["log_loss"],

            "validation_auc":
                r["auc"],

            "validation_brier_improvement":
                (
                    baseline_2025[
                        "brier"
                    ]
                    -
                    r["brier"]
                ),

            "validation_logloss_improvement":
                (
                    baseline_2025[
                        "log_loss"
                    ]
                    -
                    r["log_loss"]
                ),

            "validation_auc_improvement":
                (
                    r["auc"]
                    -
                    baseline_2025[
                        "auc"
                    ]
                ),
        }
    )


validation = pd.DataFrame(
    validation_rows
)


# ============================================================
# FINALIST SCORE
#
# Development still matters.
# 2025 validation matters heavily.
# ============================================================

validation[
    "selection_score"
] = (
    validation[
        "dev_score"
    ]
    +
    1500
    *
    validation[
        "validation_brier_improvement"
    ]
    +
    600
    *
    validation[
        "validation_logloss_improvement"
    ]
    +
    1.0
    *
    validation[
        "validation_auc_improvement"
    ]
)


# Require positive development Brier.
# Prefer finalists that also improve 2025.
eligible = validation[
    validation[
        "dev_brier_improvement"
    ]
    >
    0
].copy()


eligible[
    "validation_positive"
] = (
    eligible[
        "validation_brier_improvement"
    ]
    >
    0
)


eligible = eligible.sort_values(
    [
        "validation_positive",
        "selection_score",
        "validation_brier_improvement",
    ],
    ascending=[
        False,
        False,
        False,
    ],
).reset_index(
    drop=True
)


eligible[
    "validation_rank"
] = (
    np.arange(
        len(eligible)
    )
    +
    1
)


eligible.to_csv(
    OUT_DIR
    / "02_validation_2025_leaderboard.csv",
    index=False,
)


print()
print("TOP 15 AFTER 2025 VALIDATION")
print()


show_cols = [
    "validation_rank",
    "config_id",
    "family",
    "feature_set",
    "lambda_cut",
    "total_cut",
    "knots",
    "C",
    "blend",
    "dev_brier_improvement",
    "dev_brier_years_won",
    "validation_brier_improvement",
    "validation_logloss_improvement",
    "validation_auc_improvement",
    "selection_score",
]


for c in show_cols:

    if c not in eligible.columns:
        eligible[c] = np.nan


print(
    eligible[
        show_cols
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# PHASE 3:
# FINAL UNTOUCHED 2026
#
# ONLY top five finalists.
# ============================================================

FINALISTS = eligible.head(
    5
).copy()


print()
print("=" * 120)
print("PHASE 3 — UNTOUCHED 2026 FINAL TEST")
print("=" * 120)


baseline_2026 = baseline_df[
    baseline_df["year"]
    ==
    FINAL_YEAR
].iloc[0]


final_rows = []

prediction_frames = []


for _, row in FINALISTS.iterrows():

    cfg = config_lookup[
        row["config_id"]
    ]

    output = predict_year(
        cfg,
        FINAL_YEAR,
    )

    test, pred, model_pred = output

    r = metrics(
        test["btts_yes"],
        pred,
    )


    final_rows.append(
        {
            **row.to_dict(),

            "final_games":
                r["games"],

            "final_brier":
                r["brier"],

            "final_log_loss":
                r["log_loss"],

            "final_auc":
                r["auc"],

            "final_avg_pred":
                r["avg_pred"],

            "final_actual_rate":
                r["actual_rate"],

            "final_brier_improvement":
                (
                    baseline_2026[
                        "brier"
                    ]
                    -
                    r["brier"]
                ),

            "final_logloss_improvement":
                (
                    baseline_2026[
                        "log_loss"
                    ]
                    -
                    r["log_loss"]
                ),

            "final_auc_improvement":
                (
                    r["auc"]
                    -
                    baseline_2026[
                        "auc"
                    ]
                ),
        }
    )


    temp = test[
        [
            "date",
            "league",
            "home_team",
            "away_team",
            "season",
            "btts_yes",
            "home_goals",
            "away_goals",
            "poisson_btts",
        ]
    ].copy()

    temp["config_id"] = (
        cfg["config_id"]
    )

    temp["model_probability"] = (
        model_pred
    )

    temp["final_probability"] = (
        pred
    )

    prediction_frames.append(
        temp
    )


final = pd.DataFrame(
    final_rows
)


final = final.sort_values(
    [
        "final_brier_improvement",
        "final_logloss_improvement",
    ],
    ascending=False,
).reset_index(
    drop=True
)


final[
    "final_rank"
] = (
    np.arange(
        len(final)
    )
    +
    1
)


final.to_csv(
    OUT_DIR
    / "03_final_2026_results.csv",
    index=False,
)


final_predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

final_predictions.to_csv(
    OUT_DIR
    / "04_finalist_2026_predictions.csv",
    index=False,
)


print()
print("POISSON 2026 BASELINE")
print(
    f"Brier:   "
    f"{baseline_2026['brier']:.6f}"
)

print(
    f"LogLoss: "
    f"{baseline_2026['log_loss']:.6f}"
)

print(
    f"AUC:     "
    f"{baseline_2026['auc']:.6f}"
)


print()
print("FINALISTS — 2026")
print()


final_cols = [
    "final_rank",
    "config_id",
    "family",
    "feature_set",
    "lambda_cut",
    "total_cut",
    "knots",
    "C",
    "blend",
    "dev_brier_improvement",
    "validation_brier_improvement",
    "final_brier_improvement",
    "final_logloss_improvement",
    "final_auc_improvement",
    "final_brier",
    "final_log_loss",
    "final_auc",
]


for c in final_cols:

    if c not in final.columns:
        final[c] = np.nan


print(
    final[
        final_cols
    ].to_string(
        index=False
    )
)


# ============================================================
# WINNER LEAGUE BREAKDOWN
# ============================================================

winner_id = final.iloc[0][
    "config_id"
]

winner_predictions = (
    final_predictions[
        final_predictions[
            "config_id"
        ]
        ==
        winner_id
    ]
    .copy()
)


league_rows = []


for league in sorted(
    winner_predictions[
        "league"
    ]
    .dropna()
    .unique()
):

    z = winner_predictions[
        winner_predictions[
            "league"
        ]
        ==
        league
    ]


    if len(z) < 30:
        continue


    new = metrics(
        z["btts_yes"],
        z["final_probability"],
    )

    old = metrics(
        z["btts_yes"],
        z["poisson_btts"],
    )


    league_rows.append(
        {
            "league":
                league,

            "games":
                len(z),

            "poisson_brier":
                old["brier"],

            "winner_brier":
                new["brier"],

            "brier_improvement":
                (
                    old["brier"]
                    -
                    new["brier"]
                ),

            "poisson_log_loss":
                old["log_loss"],

            "winner_log_loss":
                new["log_loss"],

            "logloss_improvement":
                (
                    old["log_loss"]
                    -
                    new["log_loss"]
                ),

            "poisson_auc":
                old["auc"],

            "winner_auc":
                new["auc"],

            "auc_improvement":
                (
                    new["auc"]
                    -
                    old["auc"]
                ),
        }
    )


winner_leagues = pd.DataFrame(
    league_rows
).sort_values(
    "brier_improvement",
    ascending=False,
)


winner_leagues.to_csv(
    OUT_DIR
    / "05_winner_2026_by_league.csv",
    index=False,
)


print()
print("=" * 120)
print(
    f"WINNER 2026 BY LEAGUE: "
    f"{winner_id}"
)
print("=" * 120)

print()
print(
    winner_leagues.to_string(
        index=False
    )
)


# ============================================================
# SAVE CONFIG CATALOG
# ============================================================

config_df = pd.DataFrame(
    configs
)

config_df.to_csv(
    OUT_DIR
    / "00_config_catalog.csv",
    index=False,
)


with open(
    OUT_DIR
    / "experiment_metadata.json",
    "w",
) as f:

    json.dump(
        {
            "development_years":
                DEV_YEARS,

            "validation_year":
                VALIDATION_YEAR,

            "final_year":
                FINAL_YEAR,

            "total_configs":
                len(configs),

            "lambda_min_cutoffs":
                LAMBDA_CUTS,

            "lambda_total_cutoffs":
                TOTAL_CUTS,

            "finalist_count":
                len(FINALISTS),

            "winner_config":
                winner_id,
        },
        f,
        indent=2,
    )


print()
print("=" * 120)
print("FILES")
print("=" * 120)

for path in sorted(
    OUT_DIR.glob("*")
):

    print(path)


print()
print("=" * 120)
print("DONE")
print("=" * 120)
