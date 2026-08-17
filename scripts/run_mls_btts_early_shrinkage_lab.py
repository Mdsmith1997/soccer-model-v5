from pathlib import Path
from itertools import product

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
    / "mls_btts_shrinkage_lab"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# EXPERIMENT SPLIT
# ============================================================

DEV_YEARS = [
    2021,
    2022,
    2023,
]

VALIDATION_YEAR = 2024

FINAL_YEAR = 2025


# ============================================================
# SHRINKAGE SEARCH
#
# Shrink toward leakage-safe MLS season prior.
# ============================================================

SHRINK_1_5 = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
]

SHRINK_6_10 = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

SHRINK_11_15 = [
    0.00,
    0.10,
    0.20,
    0.30,
]

# 16+ remains unchanged


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 120)
print("MLS BTTS EARLY-SEASON SHRINKAGE LAB")
print("CFG_0755 FROZEN")
print("=" * 120)

df = pd.read_csv(
    INPUT,
    low_memory=False,
)


df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)


numeric_cols = [
    "test_year",
    "btts_yes",
    "champion_yes",
    "market_yes",
    "poisson_yes",
    "minimum_team_game_number",
]


for c in numeric_cols:

    if c in df.columns:

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


# ============================================================
# LEAKAGE-SAFE MLS PRIOR
#
# For each test year:
# use ONLY games dated before Jan 1 of that year.
# ============================================================

year_priors = {}


for year in sorted(
    df["test_year"]
    .dropna()
    .unique()
):

    cutoff = pd.Timestamp(
        int(year),
        1,
        1,
    )

    history = df[
        df["date"]
        <
        cutoff
    ]


    if len(history) == 0:

        # No pre-season MLS outcome history exists in this
        # dataset for the first test year.
        #
        # Keep the prior undefined rather than using future
        # outcomes and creating leakage.
        prior = np.nan

    else:

        prior = (
            history["btts_yes"]
            .mean()
        )


    year_priors[
        int(year)
    ] = prior


print()
print("Leakage-safe MLS priors:")

for year, prior in year_priors.items():

    print(
        f"  {year}: "
        f"{prior:.2%}"
    )


# ============================================================
# APPLY SHRINKAGE
# ============================================================

def apply_shrinkage(
    data,
    s1,
    s2,
    s3,
):

    out = data.copy()

    base = (
        out[
            "champion_yes"
        ]
        .to_numpy(
            dtype=float
        )
    )

    games = (
        out[
            "minimum_team_game_number"
        ]
        .to_numpy(
            dtype=float
        )
    )

    priors = np.array(
        [
            year_priors[
                int(y)
            ]
            for y in out[
                "test_year"
            ]
        ],
        dtype=float,
    )


    shrink = np.zeros(
        len(out),
        dtype=float,
    )


    shrink[
        games <= 5
    ] = s1

    shrink[
        (games >= 6)
        &
        (games <= 10)
    ] = s2

    shrink[
        (games >= 11)
        &
        (games <= 15)
    ] = s3


    # If no historical MLS prior exists for the first
    # available season, fall back to the original CFG
    # probability. Algebraically this means shrinkage has
    # no effect for that season and avoids leakage.
    priors = np.where(
        np.isfinite(priors),
        priors,
        base,
    )

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
        priors
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


    p = np.clip(
        p,
        1e-8,
        1 - 1e-8,
    )


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

        "actual_rate":
            y.mean(),

        "calibration_error":
            y.mean()
            -
            p.mean(),
    }


# ============================================================
# BASELINE
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
        df["test_year"]
        ==
        year
    ]

    r = evaluate(
        z["btts_yes"],
        z["champion_yes"],
    )

    r["year"] = year

    baseline_rows.append(
        r
    )


baseline = pd.DataFrame(
    baseline_rows
)


# ============================================================
# CONFIG CATALOG
# ============================================================

configs = []


for (
    s1,
    s2,
    s3,
) in product(
    SHRINK_1_5,
    SHRINK_6_10,
    SHRINK_11_15,
):

    configs.append(
        {
            "shrink_1_5":
                s1,

            "shrink_6_10":
                s2,

            "shrink_11_15":
                s3,
        }
    )


for i, cfg in enumerate(
    configs,
    start=1,
):

    cfg["config_id"] = (
        f"SHRINK_{i:04d}"
    )


print()
print(
    "Total configurations:",
    len(configs),
)


# ============================================================
# DEVELOPMENT SEARCH
# ============================================================

print()
print("=" * 120)
print("PHASE 1 — DEVELOPMENT 2021-2023")
print("=" * 120)


dev_rows = []


for cfg in configs:

    year_rows = []


    for year in DEV_YEARS:

        z = df[
            df["test_year"]
            ==
            year
        ].copy()


        pred = apply_shrinkage(
            z,
            cfg["shrink_1_5"],
            cfg["shrink_6_10"],
            cfg["shrink_11_15"],
        )


        r = evaluate(
            z["btts_yes"],
            pred,
        )


        base = baseline[
            baseline["year"]
            ==
            year
        ].iloc[0]


        year_rows.append(
            {
                "year":
                    year,

                "games":
                    r["games"],

                "brier_improvement":
                    (
                        base["brier"]
                        -
                        r["brier"]
                    ),

                "logloss_improvement":
                    (
                        base["log_loss"]
                        -
                        r["log_loss"]
                    ),

                "auc_improvement":
                    (
                        r["auc"]
                        -
                        base["auc"]
                    ),

                "calibration_abs":
                    abs(
                        r[
                            "calibration_error"
                        ]
                    ),
            }
        )


    yr = pd.DataFrame(
        year_rows
    )


    dev_rows.append(
        {
            **cfg,

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

            "worst_brier_improvement":
                yr[
                    "brier_improvement"
                ].min(),

            "mean_abs_calibration":
                yr[
                    "calibration_abs"
                ].mean(),
        }
    )


dev = pd.DataFrame(
    dev_rows
)


# ============================================================
# DEVELOPMENT SCORE
# ============================================================

dev["dev_score"] = (
    1500
    *
    dev[
        "dev_brier_improvement"
    ]
    +
    600
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
    0.25
    *
    dev[
        "dev_auc_improvement"
    ]
    -
    0.50
    *
    dev[
        "mean_abs_calibration"
    ]
)


dev = dev.sort_values(
    [
        "dev_score",
        "dev_brier_improvement",
    ],
    ascending=False,
).reset_index(
    drop=True
)


dev["dev_rank"] = (
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
print("TOP 25 DEVELOPMENT")
print()

print(
    dev[
        [
            "dev_rank",
            "config_id",
            "shrink_1_5",
            "shrink_6_10",
            "shrink_11_15",
            "dev_brier_improvement",
            "dev_logloss_improvement",
            "dev_auc_improvement",
            "dev_brier_years_won",
            "worst_brier_improvement",
            "dev_score",
        ]
    ]
    .head(25)
    .to_string(
        index=False
    )
)


# ============================================================
# VALIDATION 2024
# ============================================================

print()
print("=" * 120)
print("PHASE 2 — VALIDATION 2024")
print("=" * 120)


TOP_DEV = dev.head(
    30
).copy()


validation_rows = []


z2024 = df[
    df["test_year"]
    ==
    VALIDATION_YEAR
].copy()


base_2024 = baseline[
    baseline["year"]
    ==
    VALIDATION_YEAR
].iloc[0]


config_lookup = {
    cfg["config_id"]:
        cfg
    for cfg in configs
}


for _, row in TOP_DEV.iterrows():

    cfg = config_lookup[
        row["config_id"]
    ]


    pred = apply_shrinkage(
        z2024,
        cfg["shrink_1_5"],
        cfg["shrink_6_10"],
        cfg["shrink_11_15"],
    )


    r = evaluate(
        z2024["btts_yes"],
        pred,
    )


    validation_rows.append(
        {
            **row.to_dict(),

            "validation_brier":
                r["brier"],

            "validation_log_loss":
                r["log_loss"],

            "validation_auc":
                r["auc"],

            "validation_calibration_error":
                r[
                    "calibration_error"
                ],

            "validation_brier_improvement":
                (
                    base_2024[
                        "brier"
                    ]
                    -
                    r["brier"]
                ),

            "validation_logloss_improvement":
                (
                    base_2024[
                        "log_loss"
                    ]
                    -
                    r["log_loss"]
                ),

            "validation_auc_improvement":
                (
                    r["auc"]
                    -
                    base_2024[
                        "auc"
                    ]
                ),
        }
    )


validation = pd.DataFrame(
    validation_rows
)


validation[
    "selection_score"
] = (
    validation[
        "dev_score"
    ]
    +
    1800
    *
    validation[
        "validation_brier_improvement"
    ]
    +
    700
    *
    validation[
        "validation_logloss_improvement"
    ]
    +
    0.5
    *
    validation[
        "validation_auc_improvement"
    ]
)


validation = validation.sort_values(
    [
        "selection_score",
        "validation_brier_improvement",
    ],
    ascending=False,
).reset_index(
    drop=True
)


validation[
    "validation_rank"
] = (
    np.arange(
        len(validation)
    )
    +
    1
)


validation.to_csv(
    OUT_DIR
    / "02_validation_2024.csv",
    index=False,
)


print()
print("TOP 15 AFTER 2024")
print()

print(
    validation[
        [
            "validation_rank",
            "config_id",
            "shrink_1_5",
            "shrink_6_10",
            "shrink_11_15",
            "dev_brier_improvement",
            "dev_brier_years_won",
            "validation_brier_improvement",
            "validation_logloss_improvement",
            "validation_auc_improvement",
            "selection_score",
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# FINALISTS — SELECTED BEFORE 2025
# ============================================================

FINALISTS = validation.head(
    5
).copy()


print()
print("=" * 120)
print("FINALISTS SELECTED BEFORE 2025")
print("=" * 120)

print()
print(
    FINALISTS[
        [
            "validation_rank",
            "config_id",
            "shrink_1_5",
            "shrink_6_10",
            "shrink_11_15",
            "dev_brier_improvement",
            "validation_brier_improvement",
            "selection_score",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL 2025 TEST
# ============================================================

print()
print("=" * 120)
print("PHASE 3 — FINAL 2025")
print("=" * 120)


z2025 = df[
    df["test_year"]
    ==
    FINAL_YEAR
].copy()


base_2025 = baseline[
    baseline["year"]
    ==
    FINAL_YEAR
].iloc[0]


final_rows = []


for _, row in FINALISTS.iterrows():

    cfg = config_lookup[
        row["config_id"]
    ]


    pred = apply_shrinkage(
        z2025,
        cfg["shrink_1_5"],
        cfg["shrink_6_10"],
        cfg["shrink_11_15"],
    )


    r = evaluate(
        z2025["btts_yes"],
        pred,
    )


    final_rows.append(
        {
            **row.to_dict(),

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

            "final_calibration_error":
                r[
                    "calibration_error"
                ],

            "final_brier_improvement":
                (
                    base_2025[
                        "brier"
                    ]
                    -
                    r["brier"]
                ),

            "final_logloss_improvement":
                (
                    base_2025[
                        "log_loss"
                    ]
                    -
                    r["log_loss"]
                ),

            "final_auc_improvement":
                (
                    r["auc"]
                    -
                    base_2025[
                        "auc"
                    ]
                ),
        }
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
    / "03_final_2025.csv",
    index=False,
)


print()
print("BASELINE CFG_0755 2025")
print(
    f"Brier:   "
    f"{base_2025['brier']:.6f}"
)

print(
    f"LogLoss: "
    f"{base_2025['log_loss']:.6f}"
)

print(
    f"AUC:     "
    f"{base_2025['auc']:.6f}"
)


print()
print("SHRINKAGE FINALISTS")
print()

print(
    final[
        [
            "final_rank",
            "config_id",
            "shrink_1_5",
            "shrink_6_10",
            "shrink_11_15",
            "final_brier",
            "final_brier_improvement",
            "final_log_loss",
            "final_logloss_improvement",
            "final_auc",
            "final_auc_improvement",
            "final_calibration_error",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# WINNER EXPERIENCE-BAND CHECK
# ============================================================

winner_id = final.iloc[0][
    "config_id"
]

winner_cfg = config_lookup[
    winner_id
]


print()
print("=" * 120)
print(
    "WINNER EXPERIENCE-BAND CHECK"
)
print("=" * 120)


band_rows = []


for year in [
    2021,
    2022,
    2023,
    2024,
    2025,
]:

    z_year = df[
        df["test_year"]
        ==
        year
    ].copy()


    pred = apply_shrinkage(
        z_year,
        winner_cfg[
            "shrink_1_5"
        ],
        winner_cfg[
            "shrink_6_10"
        ],
        winner_cfg[
            "shrink_11_15"
        ],
    )


    z_year[
        "adjusted_prob"
    ] = pred


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


        original = evaluate(
            z["btts_yes"],
            z["champion_yes"],
        )

        adjusted = evaluate(
            z["btts_yes"],
            z["adjusted_prob"],
        )


        band_rows.append(
            {
                "year":
                    year,

                "band":
                    band,

                "games":
                    len(z),

                "actual_rate":
                    z[
                        "btts_yes"
                    ].mean(),

                "original_pred":
                    z[
                        "champion_yes"
                    ].mean(),

                "adjusted_pred":
                    z[
                        "adjusted_prob"
                    ].mean(),

                "original_error":
                    original[
                        "calibration_error"
                    ],

                "adjusted_error":
                    adjusted[
                        "calibration_error"
                    ],

                "original_brier":
                    original[
                        "brier"
                    ],

                "adjusted_brier":
                    adjusted[
                        "brier"
                    ],

                "brier_improvement":
                    (
                        original[
                            "brier"
                        ]
                        -
                        adjusted[
                            "brier"
                        ]
                    ),
            }
        )


band_df = pd.DataFrame(
    band_rows
)


band_df.to_csv(
    OUT_DIR
    / "04_winner_experience_bands.csv",
    index=False,
)


print()
print(
    band_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE CATALOG
# ============================================================

pd.DataFrame(
    configs
).to_csv(
    OUT_DIR
    / "00_config_catalog.csv",
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
