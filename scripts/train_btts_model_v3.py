from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    SplineTransformer,
)
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

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "btts_feature_store_v1.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "btts_model_v3_oos_predictions.csv"
)

SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "btts_model_v3_summary.csv"
)

BY_YEAR = (
    ROOT
    / "data"
    / "processed"
    / "btts_model_v3_by_year.csv"
)

BY_LEAGUE = (
    ROOT
    / "data"
    / "processed"
    / "btts_model_v3_by_league.csv"
)

MIN_TRAIN = 5000


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 115)
print("BTTS MODEL V3 — NONLINEAR POISSON CORRECTION")
print("=" * 115)

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


# ============================================================
# POISSON LOGIT
# ============================================================

p = df["poisson_btts"].clip(
    0.001,
    0.999,
)

df["poisson_logit"] = np.log(
    p / (1 - p)
)


# ============================================================
# FEATURE GROUPS
# ============================================================

LINEAR_FEATURES = [
    "poisson_logit",
    "lambda_balance_ratio",
    "xg_matchup_overall_balance",
    "shot_matchup_overall_balance",
    "goal_matchup_overall_balance",
    "league_goal_environment",
    "league_xg_environment",
    "minimum_team_history",
]


SPLINE_FEATURES = [
    "lambda_min",
    "lambda_total",
    "lambda_gap",
    "weaker_team_score_probability",
    "xg_matchup_overall_min",
    "shot_matchup_overall_min",
    "goal_matchup_overall_min",
]


LINEAR_FEATURES = [
    c for c in LINEAR_FEATURES
    if c in df.columns
]

SPLINE_FEATURES = [
    c for c in SPLINE_FEATURES
    if c in df.columns
]


print()
print("Rows:", len(df))
print("Linear features:", len(LINEAR_FEATURES))
print("Spline features:", len(SPLINE_FEATURES))


# ============================================================
# PREPROCESSOR
# ============================================================

linear_pipe = Pipeline(
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


spline_pipe = Pipeline(
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
                n_knots=5,
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


league_pipe = Pipeline(
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
            linear_pipe,
            LINEAR_FEATURES,
        ),
        (
            "nonlinear",
            spline_pipe,
            SPLINE_FEATURES,
        ),
        (
            "league",
            league_pipe,
            ["league"],
        ),
    ]
)


model = Pipeline(
    [
        (
            "prep",
            prep,
        ),
        (
            "model",
            LogisticRegression(
                C=0.05,
                max_iter=4000,
                solver="liblinear",
            ),
        ),
    ]
)


FEATURE_COLUMNS = (
    LINEAR_FEATURES
    +
    SPLINE_FEATURES
    +
    ["league"]
)


# ============================================================
# METRICS
# ============================================================

def evaluate(y, p):

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

    return {
        "games": len(y),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "auc": (
            roc_auc_score(y, p)
            if len(np.unique(y)) > 1
            else np.nan
        ),
        "avg_pred": p.mean(),
        "actual_rate": y.mean(),
    }


# ============================================================
# CHRONOLOGICAL OOS
# ============================================================

years = sorted(
    df["date"]
    .dt.year
    .unique()
)

frames = []


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


    if len(train) < MIN_TRAIN:

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


    model.fit(
        train[FEATURE_COLUMNS],
        train["btts_yes"],
    )


    p_v3 = model.predict_proba(
        test[FEATURE_COLUMNS]
    )[:, 1]


    out = test[
        [
            "date",
            "league",
            "home_team",
            "away_team",
            "season",
            "btts_yes",
            "home_goals",
            "away_goals",
            "home_lambda",
            "away_lambda",
            "lambda_min",
            "lambda_total",
            "lambda_gap",
            "poisson_btts",
        ]
    ].copy()


    out["test_year"] = int(
        year
    )

    out["p_btts_v3"] = p_v3

    out["v3_adjustment"] = (
        out["p_btts_v3"]
        -
        out["poisson_btts"]
    )


    frames.append(
        out
    )


oos = pd.concat(
    frames,
    ignore_index=True,
)


# ============================================================
# OVERALL
# ============================================================

overall_rows = []


for name, col in {
    "V5_POISSON":
        "poisson_btts",

    "BTTS_V3":
        "p_btts_v3",
}.items():

    r = evaluate(
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

d["avg_pred"] = d[
    "avg_pred"
].map(
    lambda v: f"{v:.2%}"
)

d["actual_rate"] = d[
    "actual_rate"
].map(
    lambda v: f"{v:.2%}"
)


print()
print(
    d.to_string(
        index=False
    )
)


# ============================================================
# BY YEAR
# ============================================================

year_rows = []


for year in sorted(
    oos["test_year"].unique()
):

    z = oos[
        oos["test_year"]
        == year
    ]

    for name, col in {
        "V5_POISSON":
            "poisson_btts",

        "BTTS_V3":
            "p_btts_v3",
    }.items():

        r = evaluate(
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

yd["log_loss"] = yd[
    "log_loss"
].map(
    lambda v: f"{v:.5f}"
)

yd["auc"] = yd["auc"].map(
    lambda v: f"{v:.4f}"
)

yd["avg_pred"] = yd[
    "avg_pred"
].map(
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
# BY LEAGUE
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


    for name, col in {
        "V5_POISSON":
            "poisson_btts",

        "BTTS_V3":
            "p_btts_v3",
    }.items():

        r = evaluate(
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

ld["log_loss"] = ld[
    "log_loss"
].map(
    lambda v: f"{v:.5f}"
)

ld["auc"] = ld["auc"].map(
    lambda v: f"{v:.4f}"
)

ld["avg_pred"] = ld[
    "avg_pred"
].map(
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
# CORRECTION SHAPE
# ============================================================

print()
print("=" * 115)
print("V3 CORRECTION BY WEAKER LAMBDA")
print("=" * 115)


oos["lambda_band"] = pd.cut(
    oos["lambda_min"],
    [
        0,
        .7,
        .9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.7,
        2.0,
        10,
    ],
    include_lowest=True,
)


shape = (
    oos.groupby(
        "lambda_band",
        observed=True,
    )
    .agg(
        games=("btts_yes", "size"),
        actual=("btts_yes", "mean"),
        poisson=("poisson_btts", "mean"),
        v3=("p_btts_v3", "mean"),
        adjustment=("v3_adjustment", "mean"),
    )
    .reset_index()
)


shape["poisson_error"] = (
    shape["actual"]
    -
    shape["poisson"]
)

shape["v3_error"] = (
    shape["actual"]
    -
    shape["v3"]
)


sd = shape.copy()


for c in [
    "actual",
    "poisson",
    "v3",
    "adjustment",
    "poisson_error",
    "v3_error",
]:

    sd[c] = sd[c].map(
        lambda v: f"{v:+.2%}"
    )


print()
print(
    sd.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

oos.to_csv(
    OUTPUT,
    index=False,
)

overall.to_csv(
    SUMMARY,
    index=False,
)

year_df.to_csv(
    BY_YEAR,
    index=False,
)

league_df.to_csv(
    BY_LEAGUE,
    index=False,
)


shape.to_csv(
    ROOT
    / "data"
    / "processed"
    / "btts_model_v3_correction_shape.csv",
    index=False,
)


print()
print("=" * 115)
print("OUTPUTS")
print("=" * 115)

print()
print(OUTPUT)
print(SUMMARY)
print(BY_YEAR)
print(BY_LEAGUE)

print(
    ROOT
    / "data"
    / "processed"
    / "btts_model_v3_correction_shape.csv"
)

print()
print("DONE")
