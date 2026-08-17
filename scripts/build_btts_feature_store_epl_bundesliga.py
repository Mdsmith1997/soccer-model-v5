from pathlib import Path
import math

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
    / "btts_epl_bundesliga_v5_source.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_feature_store_epl_bundesliga.csv"
)


# ============================================================
# HELPERS
# ============================================================

def safe_numeric(df, column):

    if column not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def safe_divide(a, b):

    a = pd.to_numeric(
        a,
        errors="coerce",
    )

    b = pd.to_numeric(
        b,
        errors="coerce",
    )

    return np.where(
        np.abs(b) > 1e-9,
        a / b,
        np.nan,
    )


def poisson_btts(h, a):

    h = np.asarray(
        h,
        dtype=float,
    )

    a = np.asarray(
        a,
        dtype=float,
    )

    out = (
        1.0
        - np.exp(-h)
        - np.exp(-a)
        + np.exp(-(h + a))
    )

    return out


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 110)
print("BUILD BTTS FEATURE STORE V1")
print("=" * 110)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
)

print()
print("Input rows:", len(df))
print("Input columns:", len(df.columns))


# ============================================================
# CORE IDENTIFIERS
# ============================================================

required = [
    "date",
    "league",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "home_lambda",
    "away_lambda",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:

    print()
    print("Missing required columns:")
    print(missing)

    print()
    print("Available columns:")

    for c in df.columns:
        print(c)

    raise SystemExit


out = pd.DataFrame(
    {
        "date":
            pd.to_datetime(
                df["date"],
                errors="coerce",
            ),

        "league":
            df["league"],

        "home_team":
            df["home_team"],

        "away_team":
            df["away_team"],

        "season":
            df["season"]
            if "season" in df.columns
            else np.nan,

        "season_role":
            df["season_role"]
            if "season_role" in df.columns
            else np.nan,

        "history_class":
            df["history_class"]
            if "history_class" in df.columns
            else np.nan,

        "prior_games":
            safe_numeric(
                df,
                "prior_games",
            ),

        "home_goals":
            safe_numeric(
                df,
                "home_goals",
            ),

        "away_goals":
            safe_numeric(
                df,
                "away_goals",
            ),
    }
)


# ============================================================
# TARGET
# ============================================================

out["btts_yes"] = np.where(
    out["home_goals"].notna()
    &
    out["away_goals"].notna(),
    (
        (out["home_goals"] > 0)
        &
        (out["away_goals"] > 0)
    ).astype(int),
    np.nan,
)


# ============================================================
# BASE V5 LAMBDAS
# ============================================================

home_lambda = safe_numeric(
    df,
    "home_lambda",
)

away_lambda = safe_numeric(
    df,
    "away_lambda",
)

out["home_lambda"] = home_lambda
out["away_lambda"] = away_lambda


# ============================================================
# BTTS-SPECIFIC LAMBDA FEATURES
# ============================================================

out["lambda_total"] = (
    home_lambda
    +
    away_lambda
)

out["lambda_min"] = np.minimum(
    home_lambda,
    away_lambda,
)

out["lambda_max"] = np.maximum(
    home_lambda,
    away_lambda,
)

out["lambda_gap"] = np.abs(
    home_lambda
    -
    away_lambda
)

out["lambda_balance_ratio"] = (
    safe_divide(
        out["lambda_min"],
        out["lambda_max"],
    )
)

out["home_score_probability"] = (
    1.0
    -
    np.exp(
        -home_lambda
    )
)

out["away_score_probability"] = (
    1.0
    -
    np.exp(
        -away_lambda
    )
)

out["weaker_team_score_probability"] = np.minimum(
    out["home_score_probability"],
    out["away_score_probability"],
)

out["stronger_team_score_probability"] = np.maximum(
    out["home_score_probability"],
    out["away_score_probability"],
)

out["score_probability_gap"] = np.abs(
    out["home_score_probability"]
    -
    out["away_score_probability"]
)

out["poisson_btts"] = poisson_btts(
    home_lambda,
    away_lambda,
)


# ============================================================
# LEAGUE ENVIRONMENT
# ============================================================

for c in [
    "lg_home_goals",
    "lg_away_goals",
    "lg_home_xg",
    "lg_away_xg",
    "lg_home_shots",
    "lg_away_shots",
]:

    out[c] = safe_numeric(
        df,
        c,
    )


out["league_goal_environment"] = (
    out["lg_home_goals"]
    +
    out["lg_away_goals"]
)

out["league_xg_environment"] = (
    out["lg_home_xg"]
    +
    out["lg_away_xg"]
)

out["league_shot_environment"] = (
    out["lg_home_shots"]
    +
    out["lg_away_shots"]
)


# ============================================================
# RAW CURRENT-MATCH XG / SHOT FIELDS
#
# These may not be pre-match features in every dataset.
# We intentionally DO NOT include home_xg / away_xg or
# home_shots / away_shots here because those can describe
# the actual match that already occurred.
# ============================================================


# ============================================================
# HOME TEAM PREMATCH STRENGTHS
# ============================================================

home_feature_cols = [
    "home_final_goal_attack_overall",
    "home_final_goal_defense_overall",
    "home_final_goal_attack_venue",
    "home_final_goal_defense_venue",

    "home_final_xg_attack_overall",
    "home_final_xg_defense_overall",
    "home_final_xg_attack_venue",
    "home_final_xg_defense_venue",

    "home_final_shot_attack_overall",
    "home_final_shot_defense_overall",
    "home_final_shot_attack_venue",
    "home_final_shot_defense_venue",

    "home_adj_goal_attack_overall_games",
    "home_adj_xg_attack_overall_games",
    "home_adj_shot_attack_overall_games",

    "home_adj_goal_attack_venue_games",
    "home_adj_xg_attack_venue_games",
    "home_adj_shot_attack_venue_games",

    "home_global_xg_attack_overall_games",
]


for c in home_feature_cols:

    out[c] = safe_numeric(
        df,
        c,
    )


# ============================================================
# AWAY TEAM PREMATCH STRENGTHS
# ============================================================

away_feature_cols = [
    "away_final_goal_attack_overall",
    "away_final_goal_defense_overall",
    "away_final_goal_attack_venue",
    "away_final_goal_defense_venue",

    "away_final_xg_attack_overall",
    "away_final_xg_defense_overall",
    "away_final_xg_attack_venue",
    "away_final_xg_defense_venue",

    "away_final_shot_attack_overall",
    "away_final_shot_defense_overall",
    "away_final_shot_attack_venue",
    "away_final_shot_defense_venue",

    "away_adj_goal_attack_overall_games",
    "away_adj_xg_attack_overall_games",
    "away_adj_shot_attack_overall_games",

    "away_adj_goal_attack_venue_games",
    "away_adj_xg_attack_venue_games",
    "away_adj_shot_attack_venue_games",

    "away_global_xg_attack_overall_games",
]


for c in away_feature_cols:

    out[c] = safe_numeric(
        df,
        c,
    )


# ============================================================
# BTTS MATCHUP FEATURES
#
# Critical difference from 1X2:
# we care about EACH team's ability to score against the
# opponent, especially the weaker scoring side.
# ============================================================


# GOAL MATCHUPS

out["home_goal_matchup_overall"] = (
    out["home_final_goal_attack_overall"]
    *
    out["away_final_goal_defense_overall"]
)

out["away_goal_matchup_overall"] = (
    out["away_final_goal_attack_overall"]
    *
    out["home_final_goal_defense_overall"]
)

out["home_goal_matchup_venue"] = (
    out["home_final_goal_attack_venue"]
    *
    out["away_final_goal_defense_venue"]
)

out["away_goal_matchup_venue"] = (
    out["away_final_goal_attack_venue"]
    *
    out["home_final_goal_defense_venue"]
)


# XG MATCHUPS

out["home_xg_matchup_overall"] = (
    out["home_final_xg_attack_overall"]
    *
    out["away_final_xg_defense_overall"]
)

out["away_xg_matchup_overall"] = (
    out["away_final_xg_attack_overall"]
    *
    out["home_final_xg_defense_overall"]
)

out["home_xg_matchup_venue"] = (
    out["home_final_xg_attack_venue"]
    *
    out["away_final_xg_defense_venue"]
)

out["away_xg_matchup_venue"] = (
    out["away_final_xg_attack_venue"]
    *
    out["home_final_xg_defense_venue"]
)


# SHOT MATCHUPS

out["home_shot_matchup_overall"] = (
    out["home_final_shot_attack_overall"]
    *
    out["away_final_shot_defense_overall"]
)

out["away_shot_matchup_overall"] = (
    out["away_final_shot_attack_overall"]
    *
    out["home_final_shot_defense_overall"]
)

out["home_shot_matchup_venue"] = (
    out["home_final_shot_attack_venue"]
    *
    out["away_final_shot_defense_venue"]
)

out["away_shot_matchup_venue"] = (
    out["away_final_shot_attack_venue"]
    *
    out["home_final_shot_defense_venue"]
)


# ============================================================
# WEAKER / STRONGER MATCHUP FEATURES
#
# BTTS should be strongly influenced by the team with the
# LOWER expected attacking matchup.
# ============================================================

for family in [
    "goal_matchup_overall",
    "goal_matchup_venue",
    "xg_matchup_overall",
    "xg_matchup_venue",
    "shot_matchup_overall",
    "shot_matchup_venue",
]:

    home_col = f"home_{family}"
    away_col = f"away_{family}"

    out[f"{family}_min"] = np.minimum(
        out[home_col],
        out[away_col],
    )

    out[f"{family}_max"] = np.maximum(
        out[home_col],
        out[away_col],
    )

    out[f"{family}_gap"] = np.abs(
        out[home_col]
        -
        out[away_col]
    )

    out[f"{family}_balance"] = safe_divide(
        out[f"{family}_min"],
        out[f"{family}_max"],
    )


# ============================================================
# ATTACK / DEFENSE SYMMETRY
# ============================================================

out["goal_attack_balance"] = safe_divide(
    np.minimum(
        out["home_final_goal_attack_overall"],
        out["away_final_goal_attack_overall"],
    ),
    np.maximum(
        out["home_final_goal_attack_overall"],
        out["away_final_goal_attack_overall"],
    ),
)

out["xg_attack_balance"] = safe_divide(
    np.minimum(
        out["home_final_xg_attack_overall"],
        out["away_final_xg_attack_overall"],
    ),
    np.maximum(
        out["home_final_xg_attack_overall"],
        out["away_final_xg_attack_overall"],
    ),
)

out["shot_attack_balance"] = safe_divide(
    np.minimum(
        out["home_final_shot_attack_overall"],
        out["away_final_shot_attack_overall"],
    ),
    np.maximum(
        out["home_final_shot_attack_overall"],
        out["away_final_shot_attack_overall"],
    ),
)


# ============================================================
# HISTORY RELIABILITY
# ============================================================

out["home_goal_history_games"] = (
    out["home_adj_goal_attack_overall_games"]
)

out["away_goal_history_games"] = (
    out["away_adj_goal_attack_overall_games"]
)

out["minimum_team_history"] = np.minimum(
    out["home_goal_history_games"],
    out["away_goal_history_games"],
)

out["maximum_team_history"] = np.maximum(
    out["home_goal_history_games"],
    out["away_goal_history_games"],
)

out["history_balance"] = safe_divide(
    out["minimum_team_history"],
    out["maximum_team_history"],
)


# ============================================================
# OPTIONAL SOURCE / PROMOTION FLAGS
# ============================================================

for c in [
    "home_league_changed",
    "away_league_changed",
]:

    if c in df.columns:

        out[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )


if (
    "home_league_changed" in out.columns
    and
    "away_league_changed" in out.columns
):

    out["any_league_changed"] = (
        out[
            [
                "home_league_changed",
                "away_league_changed",
            ]
        ]
        .fillna(0)
        .max(axis=1)
    )


# ============================================================
# CLEAN TARGET ROWS
# ============================================================

out = out[
    out["btts_yes"].notna()
].copy()

out["btts_yes"] = (
    out["btts_yes"]
    .astype(int)
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

out = (
    out
    .sort_values(
        [
            "date",
            "league",
            "home_team",
            "away_team",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 110)
print("FEATURE STORE SUMMARY")
print("=" * 110)

print()
print("Rows:", len(out))
print("Columns:", len(out.columns))

print()
print(
    "Date range:",
    out["date"].min(),
    "→",
    out["date"].max(),
)

print()
print(
    f"Overall BTTS rate: {out['btts_yes'].mean():.2%}"
)


print()
print("=" * 110)
print("LEAGUE COUNTS / BTTS RATE")
print("=" * 110)

league_summary = (
    out.groupby("league")
    .agg(
        games=("btts_yes", "size"),
        btts_rate=("btts_yes", "mean"),
        avg_total_lambda=("lambda_total", "mean"),
        avg_min_lambda=("lambda_min", "mean"),
        avg_lambda_gap=("lambda_gap", "mean"),
        avg_poisson_btts=("poisson_btts", "mean"),
    )
    .sort_values(
        "games",
        ascending=False,
    )
)

display = league_summary.copy()

for c in [
    "btts_rate",
    "avg_poisson_btts",
]:

    display[c] = (
        display[c]
        .map(
            lambda x: f"{x:.2%}"
        )
    )

for c in [
    "avg_total_lambda",
    "avg_min_lambda",
    "avg_lambda_gap",
]:

    display[c] = (
        display[c]
        .map(
            lambda x: f"{x:.3f}"
        )
    )

print()
print(
    display.to_string()
)


# ============================================================
# MISSINGNESS
# ============================================================

print()
print("=" * 110)
print("MOST-MISSING FEATURES")
print("=" * 110)

feature_cols = [
    c for c in out.columns
    if c not in {
        "date",
        "league",
        "home_team",
        "away_team",
        "season",
        "season_role",
        "history_class",
        "home_goals",
        "away_goals",
        "btts_yes",
    }
]

missingness = (
    out[feature_cols]
    .isna()
    .mean()
    .sort_values(
        ascending=False
    )
)

print()

print(
    (
        missingness
        .head(30)
        * 100
    )
    .round(2)
    .astype(str)
    .add("%")
    .to_string()
)


# ============================================================
# SIMPLE FEATURE CORRELATIONS
#
# Diagnostic only.
# We are NOT selecting features from this yet.
# ============================================================

print()
print("=" * 110)
print("UNIVARIATE CORRELATION WITH BTTS TARGET")
print("=" * 110)

numeric = out[
    feature_cols
    +
    ["btts_yes"]
].select_dtypes(
    include=[np.number]
)

corr = (
    numeric
    .corr(numeric_only=True)["btts_yes"]
    .drop("btts_yes")
    .dropna()
)

corr_table = pd.DataFrame(
    {
        "correlation": corr,
        "abs_correlation": corr.abs(),
    }
).sort_values(
    "abs_correlation",
    ascending=False,
)


print()
print(
    corr_table
    .head(30)
    .to_string()
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 110)
print("OUTPUT")
print("=" * 110)

print()
print(OUTPUT_FILE)

print()
print("DONE")
