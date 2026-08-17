from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MARKET_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_cfg0755_market_matched.csv"
)

FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_feature_store_v1.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "btts_false_edge_forensics"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


EDGE_MIN = 0.04

HIST_YEARS = [
    2021,
    2022,
    2023,
    2024,
]

FAIL_YEAR = 2025


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 120)
print("BTTS FALSE-EDGE FORENSICS LAB")
print("MLS | CFG_0755 - MARKET >= 4%")
print("=" * 120)

market = pd.read_csv(
    MARKET_FILE,
    low_memory=False,
)

features = pd.read_csv(
    FEATURE_FILE,
    low_memory=False,
)


market["date"] = pd.to_datetime(
    market["date"],
    errors="coerce",
).dt.normalize()

features["date"] = pd.to_datetime(
    features["date"],
    errors="coerce",
).dt.normalize()


market = market[
    market["league"]
    ==
    "MLS"
].copy()

features = features[
    features["league"]
    ==
    "MLS"
].copy()


# ============================================================
# SAFE MERGE
# ============================================================

keys = [
    "date",
    "league",
    "home_team",
    "away_team",
]


feature_keep = [
    c
    for c in features.columns
    if c not in market.columns
    or c in keys
]


df = market.merge(
    features[
        feature_keep
    ],
    on=keys,
    how="left",
    validate="m:1",
)


print()
print("MLS market rows:", len(market))
print("Merged rows:", len(df))


# ============================================================
# NUMERIC CLEAN
# ============================================================

for c in df.columns:

    if c in [
        "date",
        "league",
        "home_team",
        "away_team",
        "season",
    ]:
        continue

    try:
        df[c] = pd.to_numeric(
            df[c],
            errors="ignore",
        )
    except Exception:
        pass


# ============================================================
# EDGE / OUTCOME
# ============================================================

df["cfg_market_edge"] = (
    pd.to_numeric(
        df["champion_yes"],
        errors="coerce",
    )
    -
    pd.to_numeric(
        df["market_yes"],
        errors="coerce",
    )
)


false_edge = df[
    df["cfg_market_edge"]
    >= EDGE_MIN
].copy()


false_edge["btts_yes"] = pd.to_numeric(
    false_edge["btts_yes"],
    errors="coerce",
)


false_edge = false_edge[
    false_edge["btts_yes"].notna()
].copy()


false_edge["btts_yes"] = (
    false_edge["btts_yes"]
    .astype(int)
)


false_edge["period"] = np.where(
    false_edge["test_year"]
    ==
    FAIL_YEAR,
    "2025",
    "2021_2024",
)


print()
print("High-disagreement games:", len(false_edge))

print()
print(
    false_edge.groupby(
        [
            "test_year",
        ]
    )
    .agg(
        games=(
            "btts_yes",
            "size",
        ),
        actual=(
            "btts_yes",
            "mean",
        ),
        avg_cfg=(
            "champion_yes",
            "mean",
        ),
        avg_market=(
            "market_yes",
            "mean",
        ),
        avg_edge=(
            "cfg_market_edge",
            "mean",
        ),
    )
    .to_string()
)


# ============================================================
# EXPERIENCE VARIABLES
# ============================================================

if (
    "home_game_number"
    not in false_edge.columns
    or
    "away_game_number"
    not in false_edge.columns
):

    tmp = false_edge[
        [
            "test_year",
            "date",
            "home_team",
            "away_team",
        ]
    ].copy()


    home = tmp[
        [
            "test_year",
            "date",
            "home_team",
        ]
    ].rename(
        columns={
            "home_team":
                "team",
        }
    )

    away = tmp[
        [
            "test_year",
            "date",
            "away_team",
        ]
    ].rename(
        columns={
            "away_team":
                "team",
        }
    )

    all_team = pd.concat(
        [
            home.assign(
                venue="home"
            ),
            away.assign(
                venue="away"
            ),
        ],
        ignore_index=True,
    )

    all_team = all_team.sort_values(
        [
            "test_year",
            "team",
            "date",
        ]
    )

    all_team["game_number"] = (
        all_team
        .groupby(
            [
                "test_year",
                "team",
            ]
        )
        .cumcount()
        +
        1
    )


    hnum = all_team[
        all_team["venue"]
        ==
        "home"
    ][
        [
            "test_year",
            "date",
            "team",
            "game_number",
        ]
    ].rename(
        columns={
            "team":
                "home_team",
            "game_number":
                "home_game_number",
        }
    )


    anum = all_team[
        all_team["venue"]
        ==
        "away"
    ][
        [
            "test_year",
            "date",
            "team",
            "game_number",
        ]
    ].rename(
        columns={
            "team":
                "away_team",
            "game_number":
                "away_game_number",
        }
    )


    false_edge = false_edge.merge(
        hnum,
        on=[
            "test_year",
            "date",
            "home_team",
        ],
        how="left",
    )

    false_edge = false_edge.merge(
        anum,
        on=[
            "test_year",
            "date",
            "away_team",
        ],
        how="left",
    )


false_edge[
    "minimum_team_game_number"
] = np.minimum(
    false_edge[
        "home_game_number"
    ],
    false_edge[
        "away_game_number"
    ],
)


# ============================================================
# DERIVED STRUCTURAL VARIABLES
# ============================================================

for c in [
    "home_lambda",
    "away_lambda",
]:

    if c in false_edge.columns:
        false_edge[c] = pd.to_numeric(
            false_edge[c],
            errors="coerce",
        )


if (
    "home_lambda"
    in false_edge.columns
    and
    "away_lambda"
    in false_edge.columns
):

    false_edge["lambda_min_diag"] = np.minimum(
        false_edge["home_lambda"],
        false_edge["away_lambda"],
    )

    false_edge["lambda_max_diag"] = np.maximum(
        false_edge["home_lambda"],
        false_edge["away_lambda"],
    )

    false_edge["lambda_total_diag"] = (
        false_edge["home_lambda"]
        +
        false_edge["away_lambda"]
    )

    false_edge["lambda_gap_diag"] = np.abs(
        false_edge["home_lambda"]
        -
        false_edge["away_lambda"]
    )


# ============================================================
# CANDIDATE FEATURES
# ============================================================

preferred_features = [
    # probabilities / structure
    "champion_yes",
    "poisson_yes",
    "market_yes",
    "cfg_market_edge",

    "home_lambda",
    "away_lambda",
    "lambda_min",
    "lambda_total",
    "lambda_gap",
    "lambda_balance_ratio",
    "weaker_team_score_probability",

    # xG
    "xg_matchup_overall_min",
    "xg_matchup_overall_balance",
    "xg_attack_balance",
    "league_xg_environment",

    # shots
    "shot_matchup_overall_min",
    "shot_matchup_overall_balance",

    # goals
    "goal_matchup_overall_min",
    "goal_matchup_overall_balance",
    "goal_attack_balance",
    "league_goal_environment",

    # history / reliability
    "minimum_team_history",
    "minimum_team_game_number",

    # if alternate names exist
    "home_xg_for_ew",
    "away_xg_for_ew",
    "home_xg_against_ew",
    "away_xg_against_ew",

    "home_shots_for_ew",
    "away_shots_for_ew",
    "home_shots_against_ew",
    "away_shots_against_ew",

    "home_sot_for_ew",
    "away_sot_for_ew",
    "home_sot_against_ew",
    "away_sot_against_ew",
]


candidate_features = []


for c in preferred_features:

    if (
        c in false_edge.columns
        and
        c not in candidate_features
    ):

        numeric = pd.to_numeric(
            false_edge[c],
            errors="coerce",
        )

        if numeric.notna().sum() >= 30:

            false_edge[c] = numeric

            candidate_features.append(
                c
            )


# Add other useful numeric engineered columns automatically

keywords = [
    "xg",
    "shot",
    "sot",
    "goal",
    "lambda",
    "history",
    "attack",
    "defense",
    "balance",
]


for c in false_edge.columns:

    lc = c.lower()

    if c in candidate_features:
        continue

    if c in [
        "btts_yes",
        "home_goals",
        "away_goals",
        "profit",
        "won",
    ]:
        continue

    if not any(
        k in lc
        for k in keywords
    ):
        continue

    numeric = pd.to_numeric(
        false_edge[c],
        errors="coerce",
    )

    if numeric.notna().sum() < 50:
        continue

    if numeric.nunique(
        dropna=True
    ) < 3:
        continue

    false_edge[c] = numeric

    candidate_features.append(
        c
    )


print()
print(
    "Candidate forensic features:",
    len(candidate_features),
)


# ============================================================
# 1. DISTRIBUTION SHIFT:
# HISTORICAL HIGH-EDGE VS 2025 HIGH-EDGE
# ============================================================

print()
print("=" * 120)
print(
    "FEATURE SHIFT:"
    " 2021-2024 HIGH-EDGE VS 2025 HIGH-EDGE"
)
print("=" * 120)


shift_rows = []


hist = false_edge[
    false_edge["test_year"]
    .isin(
        HIST_YEARS
    )
]

y25 = false_edge[
    false_edge["test_year"]
    ==
    FAIL_YEAR
]


for c in candidate_features:

    a = pd.to_numeric(
        hist[c],
        errors="coerce",
    ).dropna()

    b = pd.to_numeric(
        y25[c],
        errors="coerce",
    ).dropna()


    if len(a) < 20 or len(b) < 10:
        continue


    pooled_sd = np.sqrt(
        (
            a.var(ddof=1)
            +
            b.var(ddof=1)
        )
        /
        2
    )


    standardized_shift = (
        (
            b.mean()
            -
            a.mean()
        )
        /
        pooled_sd
        if pooled_sd
        >
        0
        else np.nan
    )


    shift_rows.append(
        {
            "feature":
                c,

            "hist_n":
                len(a),

            "y2025_n":
                len(b),

            "hist_mean":
                a.mean(),

            "y2025_mean":
                b.mean(),

            "raw_change":
                b.mean()
                -
                a.mean(),

            "std_shift":
                standardized_shift,

            "hist_median":
                a.median(),

            "y2025_median":
                b.median(),
        }
    )


shift_df = pd.DataFrame(
    shift_rows
)


shift_df["abs_std_shift"] = (
    shift_df["std_shift"]
    .abs()
)


shift_df = shift_df.sort_values(
    "abs_std_shift",
    ascending=False,
)


print()
print(
    shift_df.head(
        40
    ).to_string(
        index=False
    )
)


# ============================================================
# 2. GOOD VS BAD DISAGREEMENTS
#
# "GOOD" = BTTS actually happened.
# "FALSE" = model had +4% edge but BTTS failed.
# ============================================================

print()
print("=" * 120)
print("GOOD VS FALSE HIGH-EDGE SIGNALS")
print("=" * 120)


outcome_rows = []


for c in candidate_features:

    yes = pd.to_numeric(
        false_edge.loc[
            false_edge["btts_yes"]
            ==
            1,
            c,
        ],
        errors="coerce",
    ).dropna()

    no = pd.to_numeric(
        false_edge.loc[
            false_edge["btts_yes"]
            ==
            0,
            c,
        ],
        errors="coerce",
    ).dropna()


    if len(yes) < 20 or len(no) < 20:
        continue


    pooled_sd = np.sqrt(
        (
            yes.var(ddof=1)
            +
            no.var(ddof=1)
        )
        /
        2
    )


    effect = (
        (
            yes.mean()
            -
            no.mean()
        )
        /
        pooled_sd
        if pooled_sd
        >
        0
        else np.nan
    )


    outcome_rows.append(
        {
            "feature":
                c,

            "btts_yes_mean":
                yes.mean(),

            "btts_no_mean":
                no.mean(),

            "raw_difference":
                yes.mean()
                -
                no.mean(),

            "standardized_effect":
                effect,

            "abs_effect":
                abs(effect)
                if np.isfinite(
                    effect
                )
                else np.nan,
        }
    )


outcome_df = pd.DataFrame(
    outcome_rows
).sort_values(
    "abs_effect",
    ascending=False,
)


print()
print(
    outcome_df.head(
        40
    ).to_string(
        index=False
    )
)


# ============================================================
# 3. 2025 FALSE EDGE VS HISTORICAL SUCCESS
#
# Direct comparison:
# Historical BTTS YES high-edge games
# vs 2025 BTTS NO high-edge games.
# ============================================================

historical_success = false_edge[
    false_edge["test_year"]
    .isin(
        HIST_YEARS
    )
    &
    (
        false_edge["btts_yes"]
        ==
        1
    )
]


failure_2025 = false_edge[
    (false_edge["test_year"] == 2025)
    &
    (
        false_edge["btts_yes"]
        ==
        0
    )
]


contrast_rows = []


for c in candidate_features:

    a = pd.to_numeric(
        historical_success[c],
        errors="coerce",
    ).dropna()

    b = pd.to_numeric(
        failure_2025[c],
        errors="coerce",
    ).dropna()


    if len(a) < 15 or len(b) < 10:
        continue


    pooled_sd = np.sqrt(
        (
            a.var(ddof=1)
            +
            b.var(ddof=1)
        )
        /
        2
    )


    effect = (
        (
            b.mean()
            -
            a.mean()
        )
        /
        pooled_sd
        if pooled_sd
        >
        0
        else np.nan
    )


    contrast_rows.append(
        {
            "feature":
                c,

            "historical_success_mean":
                a.mean(),

            "2025_failure_mean":
                b.mean(),

            "raw_change":
                b.mean()
                -
                a.mean(),

            "std_shift":
                effect,

            "abs_std_shift":
                abs(effect)
                if np.isfinite(effect)
                else np.nan,
        }
    )


contrast_df = pd.DataFrame(
    contrast_rows
).sort_values(
    "abs_std_shift",
    ascending=False,
)


print()
print("=" * 120)
print(
    "HISTORICAL SUCCESS"
    " VS 2025 FALSE EDGE"
)
print("=" * 120)

print()
print(
    contrast_df.head(
        40
    ).to_string(
        index=False
    )
)


# ============================================================
# 4. SIMPLE UNIVARIATE THRESHOLD DIAGNOSTICS
#
# Not optimization.
# Just see whether certain feature ranges
# consistently separate high-edge outcomes.
# ============================================================

print()
print("=" * 120)
print("UNIVARIATE QUARTILE DIAGNOSTICS")
print("=" * 120)


quartile_rows = []


for c in candidate_features:

    x = pd.to_numeric(
        false_edge[c],
        errors="coerce",
    )


    valid = (
        x.notna()
        &
        false_edge[
            "btts_yes"
        ].notna()
    )


    if valid.sum() < 80:
        continue


    try:

        bins = pd.qcut(
            x[valid],
            q=4,
            duplicates="drop",
        )

    except Exception:
        continue


    temp = pd.DataFrame(
        {
            "bin":
                bins,

            "y":
                false_edge.loc[
                    valid,
                    "btts_yes",
                ],

            "year":
                false_edge.loc[
                    valid,
                    "test_year",
                ],
        }
    )


    for band, z in temp.groupby(
        "bin",
        observed=True,
    ):

        quartile_rows.append(
            {
                "feature":
                    c,

                "band":
                    str(band),

                "games":
                    len(z),

                "actual_rate":
                    z["y"].mean(),

                "hist_rate":
                    z.loc[
                        z["year"]
                        <
                        2025,
                        "y",
                    ].mean(),

                "rate_2025":
                    z.loc[
                        z["year"]
                        ==
                        2025,
                        "y",
                    ].mean(),

                "games_2025":
                    int(
                        (
                            z[
                                "year"
                            ]
                            ==
                            2025
                        ).sum()
                    ),
            }
        )


quartile_df = pd.DataFrame(
    quartile_rows
)


print()
print(
    quartile_df.head(
        100
    ).to_string(
        index=False
    )
)


# ============================================================
# 5. CLASSIFIER:
# CAN FEATURES IDENTIFY 2025-LIKE HIGH-EDGE GAMES?
#
# Target = game belongs to 2025.
# This detects regime/feature shifts.
# ============================================================

model_df = false_edge[
    false_edge["test_year"]
    .isin(
        HIST_YEARS
        +
        [
            FAIL_YEAR,
        ]
    )
].copy()


X = model_df[
    candidate_features
].copy()

y_regime = (
    model_df["test_year"]
    ==
    2025
).astype(int)


usable = [
    c
    for c in X.columns
    if pd.to_numeric(
        X[c],
        errors="coerce",
    ).notna().sum()
    >=
    50
]


X = X[
    usable
].apply(
    pd.to_numeric,
    errors="coerce",
)


print()
print("=" * 120)
print("2025 REGIME CLASSIFIER")
print("=" * 120)

print(
    "Rows:",
    len(X),
    "Features:",
    len(usable),
)


if (
    len(X)
    >=
    80
    and
    y_regime.nunique()
    ==
    2
):

    rf = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=8,
                    max_features="sqrt",
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


    rf.fit(
        X,
        y_regime,
    )


    regime_prob = (
        rf.predict_proba(
            X
        )[:, 1]
    )


    regime_auc = roc_auc_score(
        y_regime,
        regime_prob,
    )


    print(
        "In-sample regime AUC:",
        f"{regime_auc:.4f}",
    )


    perm = permutation_importance(
        rf,
        X,
        y_regime,
        n_repeats=20,
        random_state=42,
        scoring="roc_auc",
        n_jobs=-1,
    )


    importance_df = pd.DataFrame(
        {
            "feature":
                usable,

            "importance_mean":
                perm[
                    "importances_mean"
                ],

            "importance_std":
                perm[
                    "importances_std"
                ],
        }
    ).sort_values(
        "importance_mean",
        ascending=False,
    )


    print()
    print("TOP REGIME FEATURES")
    print()

    print(
        importance_df.head(
            30
        ).to_string(
            index=False
        )
    )

else:

    importance_df = pd.DataFrame()


# ============================================================
# 6. OUTCOME CLASSIFIER
#
# Within high-edge games only:
# can features distinguish actual BTTS YES/NO?
#
# Uses 2021-2024 train, evaluates 2025.
# This is diagnostic, not a replacement model.
# ============================================================

train = false_edge[
    false_edge["test_year"]
    .isin(
        HIST_YEARS
    )
].copy()

test = false_edge[
    false_edge["test_year"]
    ==
    2025
].copy()


outcome_features = [
    c
    for c in usable
    if c not in [
        "market_yes",
        "champion_yes",
        "cfg_market_edge",
        "poisson_yes",
    ]
]


print()
print("=" * 120)
print(
    "HISTORICAL HIGH-EDGE"
    " OUTCOME CLASSIFIER -> 2025"
)
print("=" * 120)


if (
    len(train)
    >=
    80
    and
    len(test)
    >=
    20
    and
    len(outcome_features)
    >
    0
):

    pipeline = Pipeline(
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
                    C=0.1,
                    max_iter=3000,
                    solver="liblinear",
                ),
            ),
        ]
    )


    pipeline.fit(
        train[
            outcome_features
        ],
        train[
            "btts_yes"
        ],
    )


    p_train = (
        pipeline.predict_proba(
            train[
                outcome_features
            ]
        )[:, 1]
    )

    p_test = (
        pipeline.predict_proba(
            test[
                outcome_features
            ]
        )[:, 1]
    )


    train_auc = roc_auc_score(
        train["btts_yes"],
        p_train,
    )

    test_auc = (
        roc_auc_score(
            test["btts_yes"],
            p_test,
        )
        if test[
            "btts_yes"
        ].nunique()
        >
        1
        else np.nan
    )


    print(
        "Historical AUC:",
        f"{train_auc:.4f}",
    )

    print(
        "2025 AUC:",
        f"{test_auc:.4f}",
    )

    print(
        "2025 actual rate:",
        f"{test['btts_yes'].mean():.2%}",
    )

    print(
        "2025 classifier average:",
        f"{p_test.mean():.2%}",
    )

    print(
        "2025 CFG average:",
        f"{test['champion_yes'].mean():.2%}",
    )


    # coefficients

    feature_names = outcome_features

    coef = (
        pipeline
        .named_steps[
            "model"
        ]
        .coef_[0]
    )


    coef_df = pd.DataFrame(
        {
            "feature":
                feature_names,

            "coefficient":
                coef,

            "abs_coefficient":
                np.abs(
                    coef
                ),
        }
    ).sort_values(
        "abs_coefficient",
        ascending=False,
    )


    print()
    print("TOP HISTORICAL OUTCOME FEATURES")
    print()

    print(
        coef_df.head(
            30
        ).to_string(
            index=False
        )
    )

else:

    coef_df = pd.DataFrame()


# ============================================================
# 7. 2025 EXACT FALSE POSITIVES
# ============================================================

false_2025 = false_edge[
    (false_edge["test_year"] == 2025)
    &
    (
        false_edge["btts_yes"]
        ==
        0
    )
].copy()


display_cols = [
    c
    for c in [
        "date",
        "home_team",
        "away_team",
        "champion_yes",
        "market_yes",
        "cfg_market_edge",
        "poisson_yes",
        "odds_yes",
        "home_lambda",
        "away_lambda",
        "lambda_min",
        "lambda_total",
        "lambda_gap",
        "minimum_team_game_number",
        "xg_matchup_overall_min",
        "xg_matchup_overall_balance",
        "shot_matchup_overall_min",
        "shot_matchup_overall_balance",
        "goal_matchup_overall_min",
        "goal_matchup_overall_balance",
        "minimum_team_history",
    ]
    if c in false_2025.columns
]


print()
print("=" * 120)
print("2025 FALSE-EDGE MATCHES")
print("=" * 120)

print()
print(
    false_2025[
        display_cols
    ]
    .sort_values(
        "cfg_market_edge",
        ascending=False,
    )
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

false_edge.to_csv(
    OUT_DIR
    / "01_all_high_edge_games.csv",
    index=False,
)

shift_df.to_csv(
    OUT_DIR
    / "02_2025_feature_shift.csv",
    index=False,
)

outcome_df.to_csv(
    OUT_DIR
    / "03_good_vs_false_features.csv",
    index=False,
)

contrast_df.to_csv(
    OUT_DIR
    / "04_historical_success_vs_2025_failure.csv",
    index=False,
)

quartile_df.to_csv(
    OUT_DIR
    / "05_univariate_quartiles.csv",
    index=False,
)

importance_df.to_csv(
    OUT_DIR
    / "06_regime_feature_importance.csv",
    index=False,
)

coef_df.to_csv(
    OUT_DIR
    / "07_historical_outcome_coefficients.csv",
    index=False,
)

false_2025.to_csv(
    OUT_DIR
    / "08_2025_false_edges.csv",
    index=False,
)


print()
print("=" * 120)
print("OUTPUTS")
print("=" * 120)

for p in sorted(
    OUT_DIR.glob("*")
):

    print(p)


print()
print("DONE")
