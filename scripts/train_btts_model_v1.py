from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
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

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_model_v1_oos_predictions.csv"
)

SUMMARY_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_model_v1_oos_summary.csv"
)

MIN_TRAIN_GAMES = 5000


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 110)
print("BTTS MODEL V1 — CHRONOLOGICAL OOS TEST")
print("=" * 110)

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

df = (
    df[
        df["date"].notna()
        &
        df["btts_yes"].notna()
    ]
    .sort_values("date")
    .reset_index(drop=True)
)

df["btts_yes"] = (
    df["btts_yes"]
    .astype(int)
)

print()
print("Rows:", len(df))
print(
    "Date range:",
    df["date"].min(),
    "→",
    df["date"].max(),
)


# ============================================================
# FEATURES
#
# Explicit whitelist.
# No score / result / target leakage.
# ============================================================

numeric_features = [
    # V5 goal model
    "home_lambda",
    "away_lambda",
    "lambda_total",
    "lambda_min",
    "lambda_max",
    "lambda_gap",
    "lambda_balance_ratio",
    "home_score_probability",
    "away_score_probability",
    "weaker_team_score_probability",
    "stronger_team_score_probability",
    "score_probability_gap",
    "poisson_btts",

    # League environment
    "lg_home_goals",
    "lg_away_goals",
    "lg_home_xg",
    "lg_away_xg",
    "lg_home_shots",
    "lg_away_shots",
    "league_goal_environment",
    "league_xg_environment",
    "league_shot_environment",

    # Home goal strengths
    "home_final_goal_attack_overall",
    "home_final_goal_defense_overall",
    "home_final_goal_attack_venue",
    "home_final_goal_defense_venue",

    # Away goal strengths
    "away_final_goal_attack_overall",
    "away_final_goal_defense_overall",
    "away_final_goal_attack_venue",
    "away_final_goal_defense_venue",

    # XG strengths
    "home_final_xg_attack_overall",
    "home_final_xg_defense_overall",
    "home_final_xg_attack_venue",
    "home_final_xg_defense_venue",

    "away_final_xg_attack_overall",
    "away_final_xg_defense_overall",
    "away_final_xg_attack_venue",
    "away_final_xg_defense_venue",

    # Shot strengths
    "home_final_shot_attack_overall",
    "home_final_shot_defense_overall",
    "home_final_shot_attack_venue",
    "home_final_shot_defense_venue",

    "away_final_shot_attack_overall",
    "away_final_shot_defense_overall",
    "away_final_shot_attack_venue",
    "away_final_shot_defense_venue",

    # Goal matchups
    "home_goal_matchup_overall",
    "away_goal_matchup_overall",
    "home_goal_matchup_venue",
    "away_goal_matchup_venue",

    "goal_matchup_overall_min",
    "goal_matchup_overall_max",
    "goal_matchup_overall_gap",
    "goal_matchup_overall_balance",

    "goal_matchup_venue_min",
    "goal_matchup_venue_max",
    "goal_matchup_venue_gap",
    "goal_matchup_venue_balance",

    # XG matchups
    "home_xg_matchup_overall",
    "away_xg_matchup_overall",
    "home_xg_matchup_venue",
    "away_xg_matchup_venue",

    "xg_matchup_overall_min",
    "xg_matchup_overall_max",
    "xg_matchup_overall_gap",
    "xg_matchup_overall_balance",

    "xg_matchup_venue_min",
    "xg_matchup_venue_max",
    "xg_matchup_venue_gap",
    "xg_matchup_venue_balance",

    # Shot matchups
    "home_shot_matchup_overall",
    "away_shot_matchup_overall",
    "home_shot_matchup_venue",
    "away_shot_matchup_venue",

    "shot_matchup_overall_min",
    "shot_matchup_overall_max",
    "shot_matchup_overall_gap",
    "shot_matchup_overall_balance",

    "shot_matchup_venue_min",
    "shot_matchup_venue_max",
    "shot_matchup_venue_gap",
    "shot_matchup_venue_balance",

    # Balance
    "goal_attack_balance",
    "xg_attack_balance",
    "shot_attack_balance",

    # Reliability
    "prior_games",
    "minimum_team_history",
    "maximum_team_history",
    "history_balance",
]


numeric_features = [
    c for c in numeric_features
    if c in df.columns
]

categorical_features = [
    c for c in [
        "league",
        "history_class",
    ]
    if c in df.columns
]


print()
print("Numeric features:", len(numeric_features))
print("Categorical features:", categorical_features)


# ============================================================
# PREPROCESSORS
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


categorical_transformer = Pipeline(
    steps=[
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


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features,
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features,
        ),
    ]
)


# ============================================================
# LOGISTIC MODEL
# ============================================================

logistic = Pipeline(
    steps=[
        (
            "prep",
            preprocessor,
        ),
        (
            "model",
            LogisticRegression(
                C=0.25,
                max_iter=3000,
                solver="liblinear",
            ),
        ),
    ]
)


# ============================================================
# BOOSTING MODEL
#
# Numeric only so we avoid sparse/dense complications.
# ============================================================

boost_features = numeric_features.copy()

boost_model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
        (
            "model",
            HistGradientBoostingClassifier(
                learning_rate=0.035,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=80,
                l2_regularization=2.0,
                random_state=42,
            ),
        ),
    ]
)


# ============================================================
# OOS STRUCTURE
#
# Test one calendar year at a time.
# Train ONLY on dates before that year.
# ============================================================

years = sorted(
    df["date"]
    .dt.year
    .dropna()
    .unique()
)

prediction_frames = []


for year in years:

    test_start = pd.Timestamp(
        year=int(year),
        month=1,
        day=1,
    )

    test_end = pd.Timestamp(
        year=int(year) + 1,
        month=1,
        day=1,
    )

    train = df[
        df["date"]
        <
        test_start
    ].copy()

    test = df[
        (df["date"] >= test_start)
        &
        (df["date"] < test_end)
    ].copy()


    if len(test) == 0:
        continue

    if len(train) < MIN_TRAIN_GAMES:

        print()
        print(
            f"{year}: skipped — "
            f"only {len(train)} prior training games"
        )

        continue


    print()
    print("-" * 110)

    print(
        f"{year}: "
        f"train={len(train):,} "
        f"test={len(test):,}"
    )


    # --------------------------------------------------------
    # Logistic
    # --------------------------------------------------------

    X_train = train[
        numeric_features
        +
        categorical_features
    ]

    X_test = test[
        numeric_features
        +
        categorical_features
    ]

    y_train = train["btts_yes"]


    logistic.fit(
        X_train,
        y_train,
    )

    p_logit = (
        logistic.predict_proba(
            X_test
        )[:, 1]
    )


    # --------------------------------------------------------
    # HistGradientBoosting
    # --------------------------------------------------------

    boost_model.fit(
        train[boost_features],
        y_train,
    )

    p_boost = (
        boost_model.predict_proba(
            test[boost_features]
        )[:, 1]
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    pred = test[
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
            "poisson_btts",
        ]
    ].copy()

    pred["test_year"] = int(
        year
    )

    pred["train_games"] = len(
        train
    )

    pred["p_logit_v1"] = (
        p_logit
    )

    pred["p_boost_v1"] = (
        p_boost
    )

    prediction_frames.append(
        pred
    )


# ============================================================
# COMBINE OOS
# ============================================================

if not prediction_frames:

    raise RuntimeError(
        "No OOS predictions generated."
    )


oos = pd.concat(
    prediction_frames,
    ignore_index=True,
)


# ============================================================
# METRICS
# ============================================================

def evaluate(
    y,
    p,
):

    y = np.asarray(y)
    p = np.asarray(p)

    valid = (
        np.isfinite(y)
        &
        np.isfinite(p)
    )

    y = y[valid]
    p = p[valid]

    p = np.clip(
        p,
        1e-8,
        1 - 1e-8,
    )

    return {
        "games": len(y),

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
            ),

        "avg_pred":
            p.mean(),

        "actual_rate":
            y.mean(),
    }


models = {
    "V5_POISSON":
        "poisson_btts",

    "BTTS_LOGIT_V1":
        "p_logit_v1",

    "BTTS_BOOST_V1":
        "p_boost_v1",
}


summary_rows = []


print()
print("=" * 110)
print("OVERALL OOS RESULTS")
print("=" * 110)


for model_name, col in models.items():

    r = evaluate(
        oos["btts_yes"],
        oos[col],
    )

    r["model"] = model_name

    summary_rows.append(
        r
    )


summary = pd.DataFrame(
    summary_rows
)[
    [
        "model",
        "games",
        "brier",
        "log_loss",
        "auc",
        "avg_pred",
        "actual_rate",
    ]
]


display = summary.copy()

display["brier"] = (
    display["brier"]
    .map(lambda x: f"{x:.5f}")
)

display["log_loss"] = (
    display["log_loss"]
    .map(lambda x: f"{x:.5f}")
)

display["auc"] = (
    display["auc"]
    .map(lambda x: f"{x:.4f}")
)

display["avg_pred"] = (
    display["avg_pred"]
    .map(lambda x: f"{x:.2%}")
)

display["actual_rate"] = (
    display["actual_rate"]
    .map(lambda x: f"{x:.2%}")
)


print()
print(
    display.to_string(
        index=False
    )
)


# ============================================================
# BY YEAR
# ============================================================

print()
print("=" * 110)
print("OOS RESULTS BY YEAR")
print("=" * 110)


year_rows = []


for year in sorted(
    oos["test_year"]
    .unique()
):

    z = oos[
        oos["test_year"]
        == year
    ]

    for model_name, col in models.items():

        r = evaluate(
            z["btts_yes"],
            z[col],
        )

        r["year"] = int(year)
        r["model"] = model_name

        year_rows.append(
            r
        )


year_df = pd.DataFrame(
    year_rows
)


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


yd["brier"] = (
    yd["brier"]
    .map(lambda x: f"{x:.5f}")
)

yd["log_loss"] = (
    yd["log_loss"]
    .map(lambda x: f"{x:.5f}")
)

yd["auc"] = (
    yd["auc"]
    .map(lambda x: f"{x:.4f}")
)

yd["avg_pred"] = (
    yd["avg_pred"]
    .map(lambda x: f"{x:.2%}")
)

yd["actual_rate"] = (
    yd["actual_rate"]
    .map(lambda x: f"{x:.2%}")
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

print()
print("=" * 110)
print("OOS RESULTS BY LEAGUE")
print("=" * 110)


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

    for model_name, col in models.items():

        r = evaluate(
            z["btts_yes"],
            z[col],
        )

        r["league"] = league
        r["model"] = model_name

        league_rows.append(
            r
        )


league_df = pd.DataFrame(
    league_rows
)


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


ld["brier"] = (
    ld["brier"]
    .map(lambda x: f"{x:.5f}")
)

ld["log_loss"] = (
    ld["log_loss"]
    .map(lambda x: f"{x:.5f}")
)

ld["auc"] = (
    ld["auc"]
    .map(lambda x: f"{x:.4f}")
)

ld["avg_pred"] = (
    ld["avg_pred"]
    .map(lambda x: f"{x:.2%}")
)

ld["actual_rate"] = (
    ld["actual_rate"]
    .map(lambda x: f"{x:.2%}")
)


print()
print(
    ld.to_string(
        index=False
    )
)


# ============================================================
# CALIBRATION BINS
# ============================================================

print()
print("=" * 110)
print("CALIBRATION — BTTS BOOST V1")
print("=" * 110)


cal = oos[
    [
        "btts_yes",
        "p_boost_v1",
    ]
].dropna().copy()


cal["bin"] = pd.cut(
    cal["p_boost_v1"],
    bins=np.arange(
        0,
        1.0001,
        0.05,
    ),
    include_lowest=True,
)


cal_summary = (
    cal.groupby(
        "bin",
        observed=True,
    )
    .agg(
        games=("btts_yes", "size"),
        avg_pred=("p_boost_v1", "mean"),
        actual_rate=("btts_yes", "mean"),
    )
    .reset_index()
)


cal_summary["error"] = (
    cal_summary["actual_rate"]
    -
    cal_summary["avg_pred"]
)


cd = cal_summary.copy()

for c in [
    "avg_pred",
    "actual_rate",
    "error",
]:

    cd[c] = (
        cd[c]
        .map(
            lambda x: f"{x:+.2%}"
        )
    )


print()
print(
    cd.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

oos.to_csv(
    OUTPUT_FILE,
    index=False,
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)

year_df.to_csv(
    ROOT
    / "data"
    / "processed"
    / "btts_model_v1_oos_by_year.csv",
    index=False,
)

league_df.to_csv(
    ROOT
    / "data"
    / "processed"
    / "btts_model_v1_oos_by_league.csv",
    index=False,
)


print()
print("=" * 110)
print("OUTPUTS")
print("=" * 110)

print()
print(OUTPUT_FILE)
print(SUMMARY_FILE)

print(
    ROOT
    / "data"
    / "processed"
    / "btts_model_v1_oos_by_year.csv"
)

print(
    ROOT
    / "data"
    / "processed"
    / "btts_model_v1_oos_by_league.csv"
)

print()
print("DONE")
