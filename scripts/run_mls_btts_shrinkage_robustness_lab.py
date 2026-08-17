from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "mls_btts_stability_diag"
    / "01_mls_games_enriched.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "mls_btts_shrinkage_robustness"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
]


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 120)
print("MLS BTTS SHRINKAGE ROBUSTNESS LAB")
print("=" * 120)

df = pd.read_csv(
    INPUT,
    low_memory=False,
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)

for c in [
    "test_year",
    "btts_yes",
    "champion_yes",
    "minimum_team_game_number",
]:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce",
    )

df = df[
    df["date"].notna()
    &
    df["btts_yes"].notna()
    &
    df["champion_yes"].notna()
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
# PRIOR CONSTRUCTIONS
# ============================================================

# ------------------------------------------------------------
# 1. ALL HISTORY PRIOR
# Everything before Jan 1 of test year.
# ------------------------------------------------------------

all_history_prior = {}

for year in YEARS:

    history = df[
        df["date"]
        <
        pd.Timestamp(
            year,
            1,
            1,
        )
    ]

    all_history_prior[year] = (
        history["btts_yes"].mean()
        if len(history)
        else np.nan
    )


# ------------------------------------------------------------
# 2. PREVIOUS SEASON ONLY
# ------------------------------------------------------------

previous_season_prior = {}

for year in YEARS:

    z = df[
        df["test_year"]
        ==
        year - 1
    ]

    previous_season_prior[year] = (
        z["btts_yes"].mean()
        if len(z)
        else np.nan
    )


# ------------------------------------------------------------
# 3. RECENT 2 SEASONS
# ------------------------------------------------------------

recent_two_prior = {}

for year in YEARS:

    z = df[
        df["test_year"].isin(
            [
                year - 2,
                year - 1,
            ]
        )
    ]

    recent_two_prior[year] = (
        z["btts_yes"].mean()
        if len(z)
        else np.nan
    )


# ------------------------------------------------------------
# 4. ROLLING LAST 365 DAYS
# Leakage-safe per match.
# ------------------------------------------------------------

rolling_prior = np.full(
    len(df),
    np.nan,
    dtype=float,
)

dates = df["date"].to_numpy()

targets = (
    df["btts_yes"]
    .to_numpy(
        dtype=float
    )
)

for i in range(len(df)):

    cutoff = df.loc[i, "date"]

    start = (
        cutoff
        -
        pd.Timedelta(
            days=365
        )
    )

    mask = (
        (df["date"] < cutoff)
        &
        (df["date"] >= start)
    )

    if mask.any():

        rolling_prior[i] = (
            df.loc[
                mask,
                "btts_yes"
            ]
            .mean()
        )


df[
    "rolling_365_prior"
] = rolling_prior


print()
print("YEAR PRIORS")
print()

for year in YEARS:

    print(
        year,
        "all_history=",
        (
            f"{all_history_prior[year]:.2%}"
            if np.isfinite(
                all_history_prior[year]
            )
            else "NA"
        ),
        "previous=",
        (
            f"{previous_season_prior[year]:.2%}"
            if np.isfinite(
                previous_season_prior[year]
            )
            else "NA"
        ),
        "recent2=",
        (
            f"{recent_two_prior[year]:.2%}"
            if np.isfinite(
                recent_two_prior[year]
            )
            else "NA"
        ),
    )


# ============================================================
# ROBUSTNESS CONFIGS
# ============================================================

configs = [
    # Baseline
    {
        "name": "BASELINE",
        "s1": 0.0,
        "s2": 0.0,
        "s3": 0.0,
        "prior": "none",
    },

    # Exact prior winner
    {
        "name": "0148_ALL_HISTORY",
        "s1": 0.6,
        "s2": 0.0,
        "s3": 0.3,
        "prior": "all_history",
    },

    # Neighboring rules
    {
        "name": "40_0_20_ALL_HISTORY",
        "s1": 0.4,
        "s2": 0.0,
        "s3": 0.2,
        "prior": "all_history",
    },

    {
        "name": "50_0_20_ALL_HISTORY",
        "s1": 0.5,
        "s2": 0.0,
        "s3": 0.2,
        "prior": "all_history",
    },

    {
        "name": "50_0_30_ALL_HISTORY",
        "s1": 0.5,
        "s2": 0.0,
        "s3": 0.3,
        "prior": "all_history",
    },

    {
        "name": "60_0_20_ALL_HISTORY",
        "s1": 0.6,
        "s2": 0.0,
        "s3": 0.2,
        "prior": "all_history",
    },

    # Isolate first 5 games only
    {
        "name": "FIRST5_40_ONLY",
        "s1": 0.4,
        "s2": 0.0,
        "s3": 0.0,
        "prior": "all_history",
    },

    {
        "name": "FIRST5_50_ONLY",
        "s1": 0.5,
        "s2": 0.0,
        "s3": 0.0,
        "prior": "all_history",
    },

    {
        "name": "FIRST5_60_ONLY",
        "s1": 0.6,
        "s2": 0.0,
        "s3": 0.0,
        "prior": "all_history",
    },

    # 11-15 only
    {
        "name": "GAME11_15_20_ONLY",
        "s1": 0.0,
        "s2": 0.0,
        "s3": 0.2,
        "prior": "all_history",
    },

    {
        "name": "GAME11_15_30_ONLY",
        "s1": 0.0,
        "s2": 0.0,
        "s3": 0.3,
        "prior": "all_history",
    },

    # Same 0148 shape, different priors
    {
        "name": "0148_PREV_SEASON",
        "s1": 0.6,
        "s2": 0.0,
        "s3": 0.3,
        "prior": "previous",
    },

    {
        "name": "0148_RECENT2",
        "s1": 0.6,
        "s2": 0.0,
        "s3": 0.3,
        "prior": "recent2",
    },

    {
        "name": "0148_ROLLING365",
        "s1": 0.6,
        "s2": 0.0,
        "s3": 0.3,
        "prior": "rolling365",
    },

    # Test mild 6-10 correction
    {
        "name": "50_10_20_ALL_HISTORY",
        "s1": 0.5,
        "s2": 0.1,
        "s3": 0.2,
        "prior": "all_history",
    },

    {
        "name": "60_10_30_ALL_HISTORY",
        "s1": 0.6,
        "s2": 0.1,
        "s3": 0.3,
        "prior": "all_history",
    },
]


# ============================================================
# PRIOR VECTOR
# ============================================================

def prior_vector(
    data,
    prior_type,
):

    base = (
        data["champion_yes"]
        .to_numpy(
            dtype=float
        )
    )


    if prior_type == "none":

        return base.copy()


    if prior_type == "rolling365":

        prior = (
            data[
                "rolling_365_prior"
            ]
            .to_numpy(
                dtype=float
            )
        )


    else:

        mapping = {
            "all_history":
                all_history_prior,

            "previous":
                previous_season_prior,

            "recent2":
                recent_two_prior,
        }[
            prior_type
        ]


        prior = np.array(
            [
                mapping[
                    int(year)
                ]
                for year
                in data[
                    "test_year"
                ]
            ],
            dtype=float,
        )


    # Never leak.
    # If no historical prior exists,
    # use original CFG probability.
    prior = np.where(
        np.isfinite(prior),
        prior,
        base,
    )


    return prior


# ============================================================
# APPLY CONFIG
# ============================================================

def apply_config(
    data,
    cfg,
):

    base = (
        data[
            "champion_yes"
        ]
        .to_numpy(
            dtype=float
        )
    )


    prior = prior_vector(
        data,
        cfg["prior"],
    )


    games = (
        data[
            "minimum_team_game_number"
        ]
        .to_numpy(
            dtype=float
        )
    )


    shrink = np.zeros(
        len(data),
        dtype=float,
    )


    shrink[
        games <= 5
    ] = cfg["s1"]


    shrink[
        (games >= 6)
        &
        (games <= 10)
    ] = cfg["s2"]


    shrink[
        (games >= 11)
        &
        (games <= 15)
    ] = cfg["s3"]


    adjusted = (
        (
            1.0
            -
            shrink
        )
        *
        base
        +
        shrink
        *
        prior
    )


    return np.clip(
        adjusted,
        0.001,
        0.999,
    )


# ============================================================
# METRICS
# ============================================================

def evaluate(
    y,
    p,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )


    mask = (
        np.isfinite(y)
        &
        np.isfinite(p)
    )


    y = y[mask]
    p = p[mask]


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
                labels=[
                    0,
                    1,
                ],
            ),

        "auc":
            (
                roc_auc_score(
                    y,
                    p,
                )
                if len(
                    np.unique(y)
                )
                >
                1
                else np.nan
            ),

        "avg_pred":
            p.mean(),

        "actual":
            y.mean(),

        "cal_error":
            y.mean()
            -
            p.mean(),
    }


# ============================================================
# YEAR-BY-YEAR
# ============================================================

print()
print("=" * 120)
print("YEAR-BY-YEAR ROBUSTNESS")
print("=" * 120)


year_rows = []


for cfg in configs:

    for year in YEARS:

        z = df[
            df["test_year"]
            ==
            year
        ].copy()


        pred = apply_config(
            z,
            cfg,
        )


        r = evaluate(
            z["btts_yes"],
            pred,
        )


        baseline = evaluate(
            z["btts_yes"],
            z["champion_yes"],
        )


        year_rows.append(
            {
                "config":
                    cfg["name"],

                "prior":
                    cfg["prior"],

                "s1":
                    cfg["s1"],

                "s2":
                    cfg["s2"],

                "s3":
                    cfg["s3"],

                "year":
                    year,

                **r,

                "brier_improvement":
                    baseline["brier"]
                    -
                    r["brier"],

                "logloss_improvement":
                    baseline["log_loss"]
                    -
                    r["log_loss"],

                "auc_improvement":
                    r["auc"]
                    -
                    baseline["auc"],
            }
        )


year_df = pd.DataFrame(
    year_rows
)


year_df.to_csv(
    OUT_DIR
    / "01_year_by_year.csv",
    index=False,
)


print()
print(
    year_df[
        [
            "config",
            "year",
            "brier_improvement",
            "logloss_improvement",
            "auc_improvement",
            "cal_error",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# AGGREGATE ROBUSTNESS
# ============================================================

summary_rows = []


for cfg in configs:

    z = year_df[
        year_df["config"]
        ==
        cfg["name"]
    ]


    # Exclude baseline from "wins" interpretation
    positive_brier_years = int(
        (
            z[
                "brier_improvement"
            ]
            >
            0
        ).sum()
    )


    positive_logloss_years = int(
        (
            z[
                "logloss_improvement"
            ]
            >
            0
        ).sum()
    )


    summary_rows.append(
        {
            "config":
                cfg["name"],

            "prior":
                cfg["prior"],

            "s1":
                cfg["s1"],

            "s2":
                cfg["s2"],

            "s3":
                cfg["s3"],

            "positive_brier_years":
                positive_brier_years,

            "positive_logloss_years":
                positive_logloss_years,

            "mean_brier_improvement":
                z[
                    "brier_improvement"
                ].mean(),

            "median_brier_improvement":
                z[
                    "brier_improvement"
                ].median(),

            "worst_brier_improvement":
                z[
                    "brier_improvement"
                ].min(),

            "best_brier_improvement":
                z[
                    "brier_improvement"
                ].max(),

            "mean_logloss_improvement":
                z[
                    "logloss_improvement"
                ].mean(),

            "mean_auc_improvement":
                z[
                    "auc_improvement"
                ].mean(),

            "mean_abs_cal_error":
                z[
                    "cal_error"
                ]
                .abs()
                .mean(),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


summary["robustness_score"] = (
    1500
    *
    summary[
        "mean_brier_improvement"
    ]
    +
    600
    *
    summary[
        "mean_logloss_improvement"
    ]
    +
    0.15
    *
    summary[
        "positive_brier_years"
    ]
    +
    0.08
    *
    summary[
        "positive_logloss_years"
    ]
    +
    500
    *
    summary[
        "worst_brier_improvement"
    ]
    +
    0.25
    *
    summary[
        "mean_auc_improvement"
    ]
)


summary = (
    summary
    .sort_values(
        [
            "robustness_score",
            "positive_brier_years",
            "mean_brier_improvement",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)


summary[
    "rank"
] = (
    np.arange(
        len(summary)
    )
    +
    1
)


summary.to_csv(
    OUT_DIR
    / "02_robustness_summary.csv",
    index=False,
)


print()
print("=" * 120)
print("ROBUSTNESS LEADERBOARD")
print("=" * 120)

print()
print(
    summary[
        [
            "rank",
            "config",
            "prior",
            "s1",
            "s2",
            "s3",
            "positive_brier_years",
            "positive_logloss_years",
            "mean_brier_improvement",
            "median_brier_improvement",
            "worst_brier_improvement",
            "mean_logloss_improvement",
            "mean_auc_improvement",
            "robustness_score",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# EXPERIENCE-BAND ROBUSTNESS
# ============================================================

print()
print("=" * 120)
print("EXPERIENCE-BAND ROBUSTNESS")
print("=" * 120)


band_rows = []


for cfg in configs:

    if cfg["name"] == "BASELINE":
        continue


    for year in YEARS:

        z_year = df[
            df["test_year"]
            ==
            year
        ].copy()


        z_year[
            "adjusted"
        ] = apply_config(
            z_year,
            cfg,
        )


        for band in [
            "1-5",
            "6-10",
            "11-15",
            "16-20",
            "21+",
        ]:

            z = z_year[
                z_year[
                    "team_experience_band"
                ]
                ==
                band
            ]


            if len(z) == 0:
                continue


            old = evaluate(
                z["btts_yes"],
                z["champion_yes"],
            )


            new = evaluate(
                z["btts_yes"],
                z["adjusted"],
            )


            band_rows.append(
                {
                    "config":
                        cfg["name"],

                    "year":
                        year,

                    "band":
                        band,

                    "games":
                        len(z),

                    "brier_improvement":
                        old["brier"]
                        -
                        new["brier"],

                    "logloss_improvement":
                        old["log_loss"]
                        -
                        new["log_loss"],

                    "original_error":
                        old["cal_error"],

                    "adjusted_error":
                        new["cal_error"],
                }
            )


band_df = pd.DataFrame(
    band_rows
)


band_df.to_csv(
    OUT_DIR
    / "03_experience_band_results.csv",
    index=False,
)


band_summary = (
    band_df
    .groupby(
        [
            "config",
            "band",
        ]
    )
    .agg(
        seasons=(
            "year",
            "nunique",
        ),

        positive_brier_years=(
            "brier_improvement",
            lambda x:
                int(
                    (
                        x > 0
                    ).sum()
                ),
        ),

        mean_brier_improvement=(
            "brier_improvement",
            "mean",
        ),

        worst_brier_improvement=(
            "brier_improvement",
            "min",
        ),

        mean_logloss_improvement=(
            "logloss_improvement",
            "mean",
        ),

        original_abs_error=(
            "original_error",
            lambda x:
                np.mean(
                    np.abs(x)
                ),
        ),

        adjusted_abs_error=(
            "adjusted_error",
            lambda x:
                np.mean(
                    np.abs(x)
                ),
        ),
    )
    .reset_index()
)


band_summary[
    "calibration_abs_improvement"
] = (
    band_summary[
        "original_abs_error"
    ]
    -
    band_summary[
        "adjusted_abs_error"
    ]
)


band_summary.to_csv(
    OUT_DIR
    / "04_experience_band_summary.csv",
    index=False,
)


print()
print(
    band_summary.to_string(
        index=False
    )
)


# ============================================================
# 2025 HIGH-EDGE DIAGNOSTIC
#
# Diagnostic only — not used for selection.
# ============================================================

print()
print("=" * 120)
print("2025 HIGH-EDGE IMPACT")
print("=" * 120)


z2025 = df[
    df["test_year"]
    ==
    2025
].copy()


high_edge = z2025[
    (
        z2025[
            "champion_yes"
        ]
        -
        z2025[
            "market_yes"
        ]
    )
    >=
    0.06
].copy()


high_edge_rows = []


for cfg in configs:

    pred = apply_config(
        high_edge,
        cfg,
    )


    r = evaluate(
        high_edge[
            "btts_yes"
        ],
        pred,
    )


    high_edge_rows.append(
        {
            "config":
                cfg["name"],

            "games":
                len(high_edge),

            "actual":
                r["actual"],

            "avg_pred":
                r["avg_pred"],

            "cal_error":
                r["cal_error"],

            "brier":
                r["brier"],

            "log_loss":
                r["log_loss"],
        }
    )


high_edge_df = pd.DataFrame(
    high_edge_rows
).sort_values(
    "brier"
)


high_edge_df.to_csv(
    OUT_DIR
    / "05_2025_high_edge_diagnostic.csv",
    index=False,
)


print()
print(
    high_edge_df.to_string(
        index=False
    )
)


# ============================================================
# OUTPUT
# ============================================================

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
