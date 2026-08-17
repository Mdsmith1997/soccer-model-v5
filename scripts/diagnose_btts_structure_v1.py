from pathlib import Path

import numpy as np
import pandas as pd


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
    / "btts_structure_diagnostic_v1.csv"
)


print()
print("=" * 120)
print("BTTS STRUCTURE DIAGNOSTIC V1")
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
    df["btts_yes"].notna()
].copy()


# ============================================================
# GENERIC BIN DIAGNOSTIC
# ============================================================

all_results = []


def diagnose(
    feature,
    bins,
):

    if feature not in df.columns:
        return

    x = df[
        [
            feature,
            "btts_yes",
            "poisson_btts",
        ]
    ].dropna().copy()

    x["bin"] = pd.cut(
        x[feature],
        bins=bins,
        include_lowest=True,
        duplicates="drop",
    )

    z = (
        x.groupby(
            "bin",
            observed=True,
        )
        .agg(
            games=("btts_yes", "size"),
            feature_mean=(feature, "mean"),
            actual_btts=("btts_yes", "mean"),
            poisson_btts=("poisson_btts", "mean"),
        )
        .reset_index()
    )

    z["poisson_error"] = (
        z["actual_btts"]
        -
        z["poisson_btts"]
    )

    z["feature"] = feature

    all_results.append(
        z
    )

    print()
    print("=" * 120)
    print(feature.upper())
    print("=" * 120)

    show = z.copy()

    show["feature_mean"] = show[
        "feature_mean"
    ].map(
        lambda v: f"{v:.3f}"
    )

    for c in [
        "actual_btts",
        "poisson_btts",
        "poisson_error",
    ]:

        show[c] = show[c].map(
            lambda v: f"{v:+.2%}"
        )

    print()
    print(
        show.to_string(
            index=False
        )
    )


# ============================================================
# MAIN STRUCTURAL VARIABLES
# ============================================================

diagnose(
    "lambda_min",
    [
        0,
        .50,
        .70,
        .90,
        1.00,
        1.10,
        1.20,
        1.30,
        1.40,
        1.50,
        1.70,
        2.00,
        3.00,
        10,
    ],
)


diagnose(
    "lambda_total",
    [
        0,
        1.5,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        4.0,
        5.0,
        10,
    ],
)


diagnose(
    "lambda_gap",
    [
        0,
        .10,
        .20,
        .30,
        .40,
        .50,
        .70,
        1.0,
        1.5,
        2.0,
        10,
    ],
)


diagnose(
    "weaker_team_score_probability",
    np.arange(
        0,
        1.01,
        .05,
    ),
)


diagnose(
    "xg_matchup_overall_min",
    [
        0,
        .5,
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
        3.0,
        10,
    ],
)


diagnose(
    "shot_matchup_overall_min",
    [
        0,
        .5,
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
        3.0,
        10,
    ],
)


# ============================================================
# 2D LAMBDA MAP
#
# This is especially important.
# ============================================================

print()
print("=" * 120)
print("2D LAMBDA MAP")
print("=" * 120)


lambda_bins = [
    0,
    .6,
    .8,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.5,
    3.0,
    10,
]


x = df[
    [
        "home_lambda",
        "away_lambda",
        "btts_yes",
        "poisson_btts",
    ]
].dropna().copy()


x["home_bin"] = pd.cut(
    x["home_lambda"],
    lambda_bins,
    include_lowest=True,
)

x["away_bin"] = pd.cut(
    x["away_lambda"],
    lambda_bins,
    include_lowest=True,
)


grid = (
    x.groupby(
        [
            "home_bin",
            "away_bin",
        ],
        observed=True,
    )
    .agg(
        games=("btts_yes", "size"),
        actual_btts=("btts_yes", "mean"),
        poisson_btts=("poisson_btts", "mean"),
    )
    .reset_index()
)


grid["poisson_error"] = (
    grid["actual_btts"]
    -
    grid["poisson_btts"]
)


grid = grid[
    grid["games"] >= 50
].copy()


grid = grid.sort_values(
    "poisson_error",
    ascending=False,
)


gd = grid.copy()

for c in [
    "actual_btts",
    "poisson_btts",
    "poisson_error",
]:

    gd[c] = gd[c].map(
        lambda v: f"{v:+.2%}"
    )


print()
print("Largest Poisson UNDER-estimates:")
print()

print(
    gd.head(20)
    .to_string(
        index=False
    )
)


print()
print("Largest Poisson OVER-estimates:")
print()

print(
    gd.tail(20)
    .sort_values(
        "poisson_error"
    )
    .to_string(
        index=False
    )
)


# ============================================================
# LEAGUE × WEAKER LAMBDA
# ============================================================

print()
print("=" * 120)
print("LEAGUE × WEAKER-LAMBDA")
print("=" * 120)


x = df[
    [
        "league",
        "lambda_min",
        "btts_yes",
        "poisson_btts",
    ]
].dropna().copy()


x["lambda_band"] = pd.cut(
    x["lambda_min"],
    [
        0,
        .8,
        1.0,
        1.2,
        1.4,
        1.6,
        2.0,
        10,
    ],
    include_lowest=True,
)


league_grid = (
    x.groupby(
        [
            "league",
            "lambda_band",
        ],
        observed=True,
    )
    .agg(
        games=("btts_yes", "size"),
        actual_btts=("btts_yes", "mean"),
        poisson_btts=("poisson_btts", "mean"),
    )
    .reset_index()
)


league_grid["poisson_error"] = (
    league_grid["actual_btts"]
    -
    league_grid["poisson_btts"]
)


league_grid = league_grid[
    league_grid["games"] >= 75
].copy()


league_grid = league_grid.sort_values(
    "poisson_error",
    ascending=False,
)


lg = league_grid.copy()

for c in [
    "actual_btts",
    "poisson_btts",
    "poisson_error",
]:

    lg[c] = lg[c].map(
        lambda v: f"{v:+.2%}"
    )


print()
print("Largest under-estimates:")
print()

print(
    lg.head(25)
    .to_string(
        index=False
    )
)


print()
print("Largest over-estimates:")
print()

print(
    lg.tail(25)
    .sort_values(
        "poisson_error"
    )
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

if all_results:

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT,
        index=False,
    )


grid.to_csv(
    ROOT
    / "data"
    / "processed"
    / "btts_lambda_grid_v1.csv",
    index=False,
)


league_grid.to_csv(
    ROOT
    / "data"
    / "processed"
    / "btts_league_lambda_grid_v1.csv",
    index=False,
)


print()
print("=" * 120)
print("OUTPUTS")
print("=" * 120)

print()
print(OUTPUT)

print(
    ROOT
    / "data"
    / "processed"
    / "btts_lambda_grid_v1.csv"
)

print(
    ROOT
    / "data"
    / "processed"
    / "btts_league_lambda_grid_v1.csv"
)

print()
print("DONE")
