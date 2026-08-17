from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_feature_store_v1.csv"
)

OUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_model_v2_oos_predictions.csv"
)

SUMMARY_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_model_v2_summary.csv"
)

BY_LEAGUE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_model_v2_by_league.csv"
)

BY_YEAR_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_model_v2_by_year.csv"
)

MIN_GLOBAL_TRAIN = 5000

MIN_LEAGUE_TRAIN = 600


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 115)
print("BTTS MODEL V2 — POISSON CORRECTION ARCHITECTURE")
print("=" * 115)

df = pd.read_csv(
    INPUT_FILE,
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
].copy()

df["btts_yes"] = (
    df["btts_yes"]
    .astype(int)
)


# ============================================================
# POISSON LOGIT
# ============================================================

p = df["poisson_btts"].clip(
    0.001,
    0.999,
)

df["poisson_logit"] = np.log(
    p / (1.0 - p)
)


# ============================================================
# COMPACT BTTS FEATURE SET
#
# Intentionally small.
# ============================================================

FEATURES = [
    # Strong baseline
    "poisson_logit",

    # Weak-team scoring
    "lambda_min",
    "weaker_team_score_probability",

    # Total scoring environment
    "lambda_total",

    # One-sided vs balanced match
    "lambda_gap",
    "lambda_balance_ratio",
    "score_probability_gap",

    # XG matchup
    "xg_matchup_overall_min",
    "xg_matchup_overall_balance",

    # Shot matchup
    "shot_matchup_overall_min",
    "shot_matchup_overall_balance",

    # Goal matchup
    "goal_matchup_overall_min",
    "goal_matchup_overall_balance",

    # League environment
    "league_goal_environment",
    "league_xg_environment",

    # Attack balance
    "xg_attack_balance",
    "goal_attack_balance",

    # Reliability
    "minimum_team_history",
]


FEATURES = [
    c for c in FEATURES
    if c in df.columns
]


print()
print("Rows:", len(df))
print("Compact numeric features:", len(FEATURES))

for c in FEATURES:
    print(" ", c)


# ============================================================
# MODEL BUILDERS
# ============================================================

def global_model():

    numeric_pipe = Pipeline(
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

    cat_pipe = Pipeline(
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
                numeric_pipe,
                FEATURES,
            ),
            (
                "league",
                cat_pipe,
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
                    C=0.10,
                    max_iter=3000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def league_model():

    return Pipeline(
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
            (
                "model",
                LogisticRegression(
                    C=0.10,
                    max_iter=3000,
                    solver="liblinear",
                ),
            ),
        ]
    )


# ============================================================
# EVALUATION
# ============================================================

def metrics(y, p):

    y = np.asarray(y)
    p = np.asarray(p)

    mask = (
        np.isfinite(y)
        &
        np.isfinite(p)
    )

    y = y[mask]
    p = p[mask]

    p = np.clip(
        p,
        1e-8,
        1 - 1e-8,
    )

    if len(y) == 0:
        return None

    return {
        "games":
            len(y),

        "brier":
            brier_score_loss(
                y,
                p,
            ),

        "log_loss":
            log_loss(
                y,
                p,
            ),

        "auc":
            roc_auc_score(
                y,
                p,
            )
            if len(np.unique(y)) > 1
            else np.nan,

        "avg_pred":
            p.mean(),

        "actual_rate":
            y.mean(),
    }


# ============================================================
# CHRONOLOGICAL OOS
# ============================================================

years = sorted(
    df["date"]
    .dt.year
    .unique()
)

oos_frames = []


for year in years:

    start = pd.Timestamp(
        int(year),
        1,
        1,
    )

    end = pd.Timestamp(
        int(year) + 1,
        1,
        1,
    )

    train = df[
        df["date"] < start
    ].copy()

    test = df[
        (df["date"] >= start)
        &
        (df["date"] < end)
    ].copy()


    if len(test) == 0:
        continue

    if len(train) < MIN_GLOBAL_TRAIN:

        print(
            f"{year}: skipped "
            f"(train={len(train):,})"
        )

        continue


    print()
    print("-" * 115)

    print(
        f"{year}: "
        f"train={len(train):,} "
        f"test={len(test):,}"
    )


    # ========================================================
    # GLOBAL COMPACT / CORRECTION MODEL
    # ========================================================

    gm = global_model()

    train_global = train[
        train["poisson_btts"].notna()
    ].copy()

    test_global = test.copy()


    gm.fit(
        train_global[
            FEATURES
            +
            ["league"]
        ],
        train_global["btts_yes"],
    )


    p_global = gm.predict_proba(
        test_global[
            FEATURES
            +
            ["league"]
        ]
    )[:, 1]


    result = test_global[
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


    result["test_year"] = int(
        year
    )

    result["p_btts_v2_global"] = (
        p_global
    )


    # ========================================================
    # LEAGUE-SPECIFIC CORRECTION
    # ========================================================

    result["p_btts_v2_league"] = np.nan

    result["league_train_games"] = 0

    result["league_model_used"] = False


    for league in test_global[
        "league"
    ].dropna().unique():

        league_train = train_global[
            train_global["league"]
            == league
        ].copy()

        league_test_idx = test_global[
            test_global["league"]
            == league
        ].index


        result_mask = (
            result["league"]
            == league
        )


        result.loc[
            result_mask,
            "league_train_games"
        ] = len(
            league_train
        )


        if len(league_train) < MIN_LEAGUE_TRAIN:

            # Fallback to global model
            result.loc[
                result_mask,
                "p_btts_v2_league"
            ] = result.loc[
                result_mask,
                "p_btts_v2_global"
            ]

            continue


        lm = league_model()

        lm.fit(
            league_train[
                FEATURES
            ],
            league_train[
                "btts_yes"
            ],
        )


        league_test = test_global[
            test_global["league"]
            == league
        ].copy()


        league_prob = (
            lm.predict_proba(
                league_test[
                    FEATURES
                ]
            )[:, 1]
        )


        result.loc[
            result_mask,
            "p_btts_v2_league"
        ] = league_prob


        result.loc[
            result_mask,
            "league_model_used"
        ] = True


    oos_frames.append(
        result
    )


# ============================================================
# COMBINE
# ============================================================

oos = pd.concat(
    oos_frames,
    ignore_index=True,
)


print()
print("=" * 115)
print("LEAGUE-SPECIFIC MODEL USAGE")
print("=" * 115)

usage = (
    oos.groupby("league")
    .agg(
        games=("btts_yes", "size"),
        league_specific_games=(
            "league_model_used",
            "sum",
        ),
        avg_training_games=(
            "league_train_games",
            "mean",
        ),
    )
)

usage["usage_rate"] = (
    usage["league_specific_games"]
    /
    usage["games"]
)

print()
print(
    usage.to_string()
)


# ============================================================
# OVERALL
# ============================================================

MODELS = {
    "V5_POISSON":
        "poisson_btts",

    "BTTS_V2_GLOBAL":
        "p_btts_v2_global",

    "BTTS_V2_LEAGUE":
        "p_btts_v2_league",
}


overall_rows = []


for name, col in MODELS.items():

    r = metrics(
        oos["btts_yes"],
        oos[col],
    )

    r["model"] = name

    overall_rows.append(
        r
    )


overall = pd.DataFrame(
    overall_rows
)


print()
print("=" * 115)
print("OVERALL OOS")
print("=" * 115)

d = overall[
    [
        "model",
        "games",
        "brier",
        "log_loss",
        "auc",
        "avg_pred",
        "actual_rate",
    ]
].copy()


for c in [
    "brier",
    "log_loss",
]:

    d[c] = d[c].map(
        lambda v: f"{v:.5f}"
    )


d["auc"] = d["auc"].map(
    lambda v: f"{v:.4f}"
)

for c in [
    "avg_pred",
    "actual_rate",
]:

    d[c] = d[c].map(
        lambda v: f"{v:.2%}"
    )


print()
print(
    d.to_string(
        index=False
    )
)


# ============================================================
# YEAR
# ============================================================

year_rows = []


for year in sorted(
    oos["test_year"]
    .unique()
):

    z = oos[
        oos["test_year"]
        == year
    ]

    for name, col in MODELS.items():

        r = metrics(
            z["btts_yes"],
            z[col],
        )

        r["year"] = int(year)
        r["model"] = name

        year_rows.append(
            r
        )


year_df = pd.DataFrame(
    year_rows
)


print()
print("=" * 115)
print("OOS BY YEAR")
print("=" * 115)


yd = year_df[
    [
        "year",
        "model",
        "games",
        "brier",
        "log_loss",
        "auc",
        "avg_pred",
        "actual_rate",
    ]
].copy()


yd["brier"] = yd["brier"].map(
    lambda v: f"{v:.5f}"
)

yd["log_loss"] = yd["log_loss"].map(
    lambda v: f"{v:.5f}"
)

yd["auc"] = yd["auc"].map(
    lambda v: f"{v:.4f}"
)

yd["avg_pred"] = yd["avg_pred"].map(
    lambda v: f"{v:.2%}"
)

yd["actual_rate"] = yd[
    "actual_rate"
].map(
    lambda v: f"{v:.2%}"
)


print()
print(
    yd.to_string(
        index=False
    )
)


# ============================================================
# LEAGUE
# ============================================================

league_rows = []


for league in sorted(
    oos["league"]
    .dropna()
    .unique()
):

    z = oos[
        oos["league"]
        == league
    ]

    if len(z) < 100:
        continue


    for name, col in MODELS.items():

        r = metrics(
            z["btts_yes"],
            z[col],
        )

        r["league"] = league
        r["model"] = name

        league_rows.append(
            r
        )


league_df = pd.DataFrame(
    league_rows
)


print()
print("=" * 115)
print("OOS BY LEAGUE")
print("=" * 115)


ld = league_df[
    [
        "league",
        "model",
        "games",
        "brier",
        "log_loss",
        "auc",
        "avg_pred",
        "actual_rate",
    ]
].copy()


ld["brier"] = ld["brier"].map(
    lambda v: f"{v:.5f}"
)

ld["log_loss"] = ld["log_loss"].map(
    lambda v: f"{v:.5f}"
)

ld["auc"] = ld["auc"].map(
    lambda v: f"{v:.4f}"
)

ld["avg_pred"] = ld["avg_pred"].map(
    lambda v: f"{v:.2%}"
)

ld["actual_rate"] = ld[
    "actual_rate"
].map(
    lambda v: f"{v:.2%}"
)


print()
print(
    ld.to_string(
        index=False
    )
)


# ============================================================
# IMPROVEMENT VS POISSON
# ============================================================

print()
print("=" * 115)
print("LEAGUE IMPROVEMENT VS POISSON")
print("=" * 115)


pivot_brier = league_df.pivot(
    index="league",
    columns="model",
    values="brier",
)

pivot_log = league_df.pivot(
    index="league",
    columns="model",
    values="log_loss",
)


comparison = pd.DataFrame(
    index=pivot_brier.index
)


comparison["games"] = (
    league_df[
        league_df["model"]
        == "V5_POISSON"
    ]
    .set_index("league")[
        "games"
    ]
)


comparison[
    "global_brier_improvement"
] = (
    pivot_brier["V5_POISSON"]
    -
    pivot_brier["BTTS_V2_GLOBAL"]
)


comparison[
    "league_brier_improvement"
] = (
    pivot_brier["V5_POISSON"]
    -
    pivot_brier["BTTS_V2_LEAGUE"]
)


comparison[
    "global_logloss_improvement"
] = (
    pivot_log["V5_POISSON"]
    -
    pivot_log["BTTS_V2_GLOBAL"]
)


comparison[
    "league_logloss_improvement"
] = (
    pivot_log["V5_POISSON"]
    -
    pivot_log["BTTS_V2_LEAGUE"]
)


comparison = comparison.sort_values(
    "league_brier_improvement",
    ascending=False,
)


print()
print(
    comparison.to_string()
)


# ============================================================
# SAVE
# ============================================================

oos.to_csv(
    OUT_FILE,
    index=False,
)

overall.to_csv(
    SUMMARY_FILE,
    index=False,
)

league_df.to_csv(
    BY_LEAGUE_FILE,
    index=False,
)

year_df.to_csv(
    BY_YEAR_FILE,
    index=False,
)


print()
print("=" * 115)
print("OUTPUTS")
print("=" * 115)

print()
print(OUT_FILE)
print(SUMMARY_FILE)
print(BY_LEAGUE_FILE)
print(BY_YEAR_FILE)

print()
print("DONE")
