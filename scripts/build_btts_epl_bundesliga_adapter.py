from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

V5_FILE = (
    ROOT
    / "data"
    / "processed"
    / "attack_defense_weights_v5_predictions.csv"
)

XG_FILE = (
    ROOT
    / "data"
    / "processed"
    / "xg_features_v5.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_epl_bundesliga_v5_source.csv"
)


print()
print("=" * 110)
print("BUILD EPL / BUNDESLIGA BTTS V5 SOURCE")
print("=" * 110)


# ============================================================
# LOAD MATCH-LEVEL FROZEN V5 SOURCE
# ============================================================

v5 = pd.read_csv(
    V5_FILE,
    low_memory=False,
)

v5["date"] = pd.to_datetime(
    v5["date"],
    errors="coerce",
)

v5 = v5[
    v5["league"].isin(
        [
            "Premier League",
            "Bundesliga",
        ]
    )
].copy()


# ============================================================
# MAP V5 SCHEMA -> BTTS HISTORICAL SCHEMA
# ============================================================

rename = {

    "home_lambda_v5":
        "home_lambda",

    "away_lambda_v5":
        "away_lambda",


    "home_adj_goal_attack":
        "home_final_goal_attack_overall",

    "home_adj_goal_defense":
        "home_final_goal_defense_overall",

    "away_adj_goal_attack":
        "away_final_goal_attack_overall",

    "away_adj_goal_defense":
        "away_final_goal_defense_overall",


    "home_adj_shot_attack":
        "home_final_shot_attack_overall",

    "home_adj_shot_defense":
        "home_final_shot_defense_overall",

    "away_adj_shot_attack":
        "away_final_shot_attack_overall",

    "away_adj_shot_defense":
        "away_final_shot_defense_overall",


    "home_xg_attack_overall":
        "home_final_xg_attack_overall",

    "home_xg_defense_overall":
        "home_final_xg_defense_overall",

    "away_xg_attack_overall":
        "away_final_xg_attack_overall",

    "away_xg_defense_overall":
        "away_final_xg_defense_overall",


    # Frozen V5 prior team-game state.
    "home_games":
        "home_adj_goal_attack_overall_games",

    "away_games":
        "away_adj_goal_attack_overall_games",
}

v5 = v5.rename(
    columns=rename,
)


# ============================================================
# BUILD MATCH-LEVEL PRIOR LEAGUE xG ENVIRONMENT
#
# xg_features_v5 is team-level: two rows per match.
# league_prior_xg_for/against are leakage-safe league baselines.
# Collapse them to one match-level league state.
# ============================================================

xg = pd.read_csv(
    XG_FILE,
    low_memory=False,
)

xg["date"] = pd.to_datetime(
    xg["date"],
    errors="coerce",
)

xg = xg[
    xg["league"].isin(
        [
            "Premier League",
            "Bundesliga",
        ]
    )
].copy()


required_xg = [
    "date",
    "league",
    "league_prior_xg_for",
    "league_prior_xg_against",
]

missing_xg = [
    c
    for c in required_xg
    if c not in xg.columns
]

if missing_xg:
    raise RuntimeError(
        f"Missing xG fields: {missing_xg}"
    )


# One league state per league/date.
league_xg = (
    xg[
        required_xg
    ]
    .groupby(
        [
            "date",
            "league",
        ],
        as_index=False,
    )
    .agg(
        league_prior_xg_for=(
            "league_prior_xg_for",
            "median",
        ),
        league_prior_xg_against=(
            "league_prior_xg_against",
            "median",
        ),
    )
)


# BTTS builder expects lg_home_xg + lg_away_xg.
#
# EPL/Bundesliga xG architecture stores league xG in
# team-oriented FOR/AGAINST form rather than home/away form.
# Preserve the same TOTAL prior league xG environment:
#
# league_prior_xg_for + league_prior_xg_against.
#
# Split the total equally between compatibility columns so
# the existing BTTS builder can remain unchanged.

league_xg[
    "league_xg_environment_source"
] = (
    league_xg[
        "league_prior_xg_for"
    ]
    +
    league_xg[
        "league_prior_xg_against"
    ]
)

league_xg[
    "lg_home_xg"
] = (
    league_xg[
        "league_xg_environment_source"
    ]
    / 2.0
)

league_xg[
    "lg_away_xg"
] = (
    league_xg[
        "league_xg_environment_source"
    ]
    / 2.0
)


v5 = v5.merge(
    league_xg[
        [
            "date",
            "league",
            "lg_home_xg",
            "lg_away_xg",
        ]
    ],
    on=[
        "date",
        "league",
    ],
    how="left",
    validate="many_to_one",
)


# ============================================================
# OPTIONAL BTTS BUILDER COMPATIBILITY
# ============================================================

# The original BTTS builder safely tolerates absent optional
# columns, but explicitly provide these for schema consistency.

for c in [
    "lg_home_shots",
    "lg_away_shots",
]:

    if c not in v5.columns:
        v5[c] = np.nan


# ============================================================
# VALIDATION
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

    "lg_home_goals",
    "lg_away_goals",
    "lg_home_xg",
    "lg_away_xg",

    "home_final_goal_attack_overall",
    "home_final_goal_defense_overall",
    "away_final_goal_attack_overall",
    "away_final_goal_defense_overall",

    "home_final_shot_attack_overall",
    "home_final_shot_defense_overall",
    "away_final_shot_attack_overall",
    "away_final_shot_defense_overall",

    "home_final_xg_attack_overall",
    "home_final_xg_defense_overall",
    "away_final_xg_attack_overall",
    "away_final_xg_defense_overall",

    "home_adj_goal_attack_overall_games",
    "away_adj_goal_attack_overall_games",
]


missing = [
    c
    for c in required
    if c not in v5.columns
]

if missing:
    raise RuntimeError(
        f"Missing BTTS adapter fields: {missing}"
    )


print()
print("Rows:", f"{len(v5):,}")

print(
    "Dates:",
    v5["date"].min(),
    "->",
    v5["date"].max(),
)

print()
print("LEAGUES")
print(
    v5["league"]
    .value_counts()
    .to_string()
)


print()
print("=" * 110)
print("FEATURE COMPLETENESS")
print("=" * 110)

bad = False

for c in required:

    non_null = int(
        v5[c]
        .notna()
        .sum()
    )

    pct = (
        100.0
        * non_null
        / len(v5)
    )

    print(
        f"{c:45s}"
        f"{non_null:6d}/{len(v5):6d}"
        f"   {pct:6.2f}%"
    )

    if (
        c not in [
            "home_goals",
            "away_goals",
        ]
        and non_null == 0
    ):
        bad = True


print()
print("=" * 110)
print("LEAGUE xG ENVIRONMENT")
print("=" * 110)

audit = (
    v5
    .assign(
        league_xg_environment=(
            v5["lg_home_xg"]
            +
            v5["lg_away_xg"]
        )
    )
    .groupby("league")
    ["league_xg_environment"]
    .agg(
        [
            "count",
            "mean",
            "min",
            "max",
        ]
    )
)

print(
    audit.to_string()
)


if bad:
    raise RuntimeError(
        "One or more required adapter fields are entirely missing."
    )


v5.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("=" * 110)
print("EPL / BUNDESLIGA BTTS SOURCE COMPLETE")
print("=" * 110)

print("Saved:", OUTPUT_FILE)
print("Rows:", f"{len(v5):,}")
