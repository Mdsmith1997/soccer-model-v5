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
    / "btts_cfg0755_market_matched.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "btts_market_calibration_lab"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DEV_YEARS = [
    2021,
    2022,
    2023,
]

VALIDATION_YEAR = 2024

FINAL_YEAR = 2025


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 120)
print("BTTS CFG × MARKET CALIBRATION LAB")
print("CFG_0755 FROZEN")
print("=" * 120)

df = pd.read_csv(
    INPUT,
    low_memory=False,
)


for c in [
    "test_year",
    "btts_yes",
    "champion_yes",
    "market_yes",
    "poisson_yes",
]:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce",
    )


df = df[
    df["test_year"].isin(
        DEV_YEARS
        +
        [
            VALIDATION_YEAR,
            FINAL_YEAR,
        ]
    )
    &
    df["btts_yes"].notna()
    &
    df["champion_yes"].notna()
    &
    df["market_yes"].notna()
].copy()


df["btts_yes"] = (
    df["btts_yes"]
    .astype(int)
)


df["abs_disagreement"] = np.abs(
    df["champion_yes"]
    -
    df["market_yes"]
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

        "actual":
            y.mean(),

        "cal_error":
            y.mean()
            -
            p.mean(),
    }


# ============================================================
# BASELINES
# ============================================================

def baseline_for_year(
    year,
):

    z = df[
        df["test_year"]
        ==
        year
    ]

    cfg = evaluate(
        z["btts_yes"],
        z["champion_yes"],
    )

    market = evaluate(
        z["btts_yes"],
        z["market_yes"],
    )

    return cfg, market


# ============================================================
# CONFIGS
# ============================================================

configs = []


# ------------------------------------------------------------
# STATIC BLENDS
#
# cfg_weight = 1.0 means pure CFG
# cfg_weight = 0.0 means pure market
# ------------------------------------------------------------

for w in np.round(
    np.arange(
        0.0,
        1.01,
        0.10,
    ),
    2,
):

    configs.append(
        {
            "family":
                "static",

            "name":
                f"STATIC_CFG_{int(w*100):03d}",

            "cfg_weight":
                float(w),
        }
    )


# ------------------------------------------------------------
# DYNAMIC BLENDS
#
# Trust CFG when close to market.
# Increase market influence as disagreement grows.
#
# For example:
# <= 2% disagreement: 90% CFG
# 2-4%:              75% CFG
# 4-6%:              60% CFG
# > 6%:              40% CFG
# ------------------------------------------------------------

LOW_WEIGHTS = [
    0.80,
    0.90,
    1.00,
]

MID_WEIGHTS = [
    0.60,
    0.70,
    0.80,
]

HIGH_WEIGHTS = [
    0.30,
    0.40,
    0.50,
    0.60,
]

VERY_HIGH_WEIGHTS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]


for (
    w_low,
    w_mid,
    w_high,
    w_very,
) in product(
    LOW_WEIGHTS,
    MID_WEIGHTS,
    HIGH_WEIGHTS,
    VERY_HIGH_WEIGHTS,
):

    # enforce decreasing trust in CFG
    # as disagreement rises
    if not (
        w_low
        >=
        w_mid
        >=
        w_high
        >=
        w_very
    ):
        continue


    configs.append(
        {
            "family":
                "dynamic",

            "name":
                (
                    f"DYN_"
                    f"{int(w_low*100):02d}_"
                    f"{int(w_mid*100):02d}_"
                    f"{int(w_high*100):02d}_"
                    f"{int(w_very*100):02d}"
                ),

            "w_low":
                w_low,

            "w_mid":
                w_mid,

            "w_high":
                w_high,

            "w_very":
                w_very,
        }
    )


print()
print(
    "Total configurations:",
    len(configs),
)


# ============================================================
# APPLY CONFIG
# ============================================================

def apply_config(
    data,
    cfg,
):

    p_cfg = (
        data["champion_yes"]
        .to_numpy(
            dtype=float
        )
    )

    p_mkt = (
        data["market_yes"]
        .to_numpy(
            dtype=float
        )
    )


    if cfg["family"] == "static":

        w = cfg[
            "cfg_weight"
        ]

        return (
            w
            *
            p_cfg
            +
            (
                1.0
                -
                w
            )
            *
            p_mkt
        )


    disagreement = np.abs(
        p_cfg
        -
        p_mkt
    )


    w = np.empty(
        len(data),
        dtype=float,
    )


    w[
        disagreement
        <
        0.02
    ] = cfg[
        "w_low"
    ]


    w[
        (
            disagreement
            >=
            0.02
        )
        &
        (
            disagreement
            <
            0.04
        )
    ] = cfg[
        "w_mid"
    ]


    w[
        (
            disagreement
            >=
            0.04
        )
        &
        (
            disagreement
            <
            0.06
        )
    ] = cfg[
        "w_high"
    ]


    w[
        disagreement
        >=
        0.06
    ] = cfg[
        "w_very"
    ]


    return (
        w
        *
        p_cfg
        +
        (
            1.0
            -
            w
        )
        *
        p_mkt
    )


# ============================================================
# DEVELOPMENT
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


        pred = apply_config(
            z,
            cfg,
        )


        r = evaluate(
            z["btts_yes"],
            pred,
        )


        base_cfg, base_market = (
            baseline_for_year(
                year
            )
        )


        year_rows.append(
            {
                "year":
                    year,

                "games":
                    r["games"],

                "brier_vs_cfg":
                    (
                        base_cfg[
                            "brier"
                        ]
                        -
                        r[
                            "brier"
                        ]
                    ),

                "logloss_vs_cfg":
                    (
                        base_cfg[
                            "log_loss"
                        ]
                        -
                        r[
                            "log_loss"
                        ]
                    ),

                "auc_vs_cfg":
                    (
                        r["auc"]
                        -
                        base_cfg[
                            "auc"
                        ]
                    ),

                "brier_vs_market":
                    (
                        base_market[
                            "brier"
                        ]
                        -
                        r[
                            "brier"
                        ]
                    ),

                "logloss_vs_market":
                    (
                        base_market[
                            "log_loss"
                        ]
                        -
                        r[
                            "log_loss"
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

            "dev_brier_vs_cfg":
                np.average(
                    yr[
                        "brier_vs_cfg"
                    ],
                    weights=yr[
                        "games"
                    ],
                ),

            "dev_logloss_vs_cfg":
                np.average(
                    yr[
                        "logloss_vs_cfg"
                    ],
                    weights=yr[
                        "games"
                    ],
                ),

            "dev_auc_vs_cfg":
                np.average(
                    yr[
                        "auc_vs_cfg"
                    ],
                    weights=yr[
                        "games"
                    ],
                ),

            "dev_brier_vs_market":
                np.average(
                    yr[
                        "brier_vs_market"
                    ],
                    weights=yr[
                        "games"
                    ],
                ),

            "dev_logloss_vs_market":
                np.average(
                    yr[
                        "logloss_vs_market"
                    ],
                    weights=yr[
                        "games"
                    ],
                ),

            "dev_cfg_brier_years_won":
                int(
                    (
                        yr[
                            "brier_vs_cfg"
                        ]
                        >
                        0
                    ).sum()
                ),

            "dev_cfg_logloss_years_won":
                int(
                    (
                        yr[
                            "logloss_vs_cfg"
                        ]
                        >
                        0
                    ).sum()
                ),

            "dev_market_brier_years_won":
                int(
                    (
                        yr[
                            "brier_vs_market"
                        ]
                        >
                        0
                    ).sum()
                ),

            "worst_brier_vs_cfg":
                yr[
                    "brier_vs_cfg"
                ].min(),
        }
    )


dev = pd.DataFrame(
    dev_rows
)


# ============================================================
# DEVELOPMENT SCORE
# ============================================================

dev["dev_score"] = (
    1800
    *
    dev[
        "dev_brier_vs_cfg"
    ]
    +
    700
    *
    dev[
        "dev_logloss_vs_cfg"
    ]
    +
    0.15
    *
    dev[
        "dev_cfg_brier_years_won"
    ]
    +
    0.08
    *
    dev[
        "dev_cfg_logloss_years_won"
    ]
    +
    0.25
    *
    dev[
        "dev_auc_vs_cfg"
    ]
    +
    500
    *
    dev[
        "worst_brier_vs_cfg"
    ]
)


dev = dev.sort_values(
    [
        "dev_score",
        "dev_brier_vs_cfg",
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
    /
    "01_development_leaderboard.csv",
    index=False,
)


print()
print("TOP 30 DEVELOPMENT")
print()

cols = [
    "dev_rank",
    "name",
    "family",
    "dev_brier_vs_cfg",
    "dev_logloss_vs_cfg",
    "dev_auc_vs_cfg",
    "dev_brier_vs_market",
    "dev_cfg_brier_years_won",
    "dev_market_brier_years_won",
    "worst_brier_vs_cfg",
    "dev_score",
]


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
# VALIDATION 2024
# ============================================================

print()
print("=" * 120)
print("PHASE 2 — VALIDATION 2024")
print("=" * 120)


TOP_DEV = (
    dev
    .head(30)
    .copy()
)


z2024 = df[
    df["test_year"]
    ==
    VALIDATION_YEAR
].copy()


cfg_2024, market_2024 = (
    baseline_for_year(
        VALIDATION_YEAR
    )
)


validation_rows = []


config_lookup = {
    cfg["name"]:
        cfg
    for cfg in configs
}


for _, row in TOP_DEV.iterrows():

    cfg = config_lookup[
        row["name"]
    ]


    pred = apply_config(
        z2024,
        cfg,
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

            "validation_brier_vs_cfg":
                (
                    cfg_2024[
                        "brier"
                    ]
                    -
                    r[
                        "brier"
                    ]
                ),

            "validation_logloss_vs_cfg":
                (
                    cfg_2024[
                        "log_loss"
                    ]
                    -
                    r[
                        "log_loss"
                    ]
                ),

            "validation_auc_vs_cfg":
                (
                    r["auc"]
                    -
                    cfg_2024[
                        "auc"
                    ]
                ),

            "validation_brier_vs_market":
                (
                    market_2024[
                        "brier"
                    ]
                    -
                    r[
                        "brier"
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
    2200
    *
    validation[
        "validation_brier_vs_cfg"
    ]
    +
    900
    *
    validation[
        "validation_logloss_vs_cfg"
    ]
    +
    0.50
    *
    validation[
        "validation_auc_vs_cfg"
    ]
)


validation = validation.sort_values(
    [
        "selection_score",
        "validation_brier_vs_cfg",
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
    /
    "02_validation_2024.csv",
    index=False,
)


print()
print("TOP 15 AFTER 2024")
print()

print(
    validation[
        [
            "validation_rank",
            "name",
            "family",
            "dev_brier_vs_cfg",
            "validation_brier_vs_cfg",
            "validation_logloss_vs_cfg",
            "validation_auc_vs_cfg",
            "validation_brier_vs_market",
            "selection_score",
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# FINALISTS
# ============================================================

FINALISTS = (
    validation
    .head(5)
    .copy()
)


print()
print("=" * 120)
print("FINALISTS SELECTED BEFORE 2025")
print("=" * 120)

print()
print(
    FINALISTS[
        [
            "validation_rank",
            "name",
            "family",
            "dev_brier_vs_cfg",
            "validation_brier_vs_cfg",
            "selection_score",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL 2025
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


cfg_2025, market_2025 = (
    baseline_for_year(
        FINAL_YEAR
    )
)


final_rows = []


for _, row in FINALISTS.iterrows():

    cfg = config_lookup[
        row["name"]
    ]


    pred = apply_config(
        z2025,
        cfg,
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

            "final_brier_vs_cfg":
                (
                    cfg_2025[
                        "brier"
                    ]
                    -
                    r[
                        "brier"
                    ]
                ),

            "final_logloss_vs_cfg":
                (
                    cfg_2025[
                        "log_loss"
                    ]
                    -
                    r[
                        "log_loss"
                    ]
                ),

            "final_auc_vs_cfg":
                (
                    r["auc"]
                    -
                    cfg_2025[
                        "auc"
                    ]
                ),

            "final_brier_vs_market":
                (
                    market_2025[
                        "brier"
                    ]
                    -
                    r[
                        "brier"
                    ]
                ),

            "final_avg_pred":
                r["avg_pred"],

            "final_actual":
                r["actual"],

            "final_cal_error":
                r["cal_error"],
        }
    )


final = pd.DataFrame(
    final_rows
)


final = final.sort_values(
    [
        "final_brier",
        "final_log_loss",
    ]
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
    /
    "03_final_2025.csv",
    index=False,
)


print()
print("2025 BASELINES")
print()

print(
    "CFG Brier:",
    f"{cfg_2025['brier']:.6f}",
)

print(
    "Market Brier:",
    f"{market_2025['brier']:.6f}",
)

print(
    "CFG LogLoss:",
    f"{cfg_2025['log_loss']:.6f}",
)

print(
    "Market LogLoss:",
    f"{market_2025['log_loss']:.6f}",
)

print(
    "CFG AUC:",
    f"{cfg_2025['auc']:.6f}",
)

print(
    "Market AUC:",
    f"{market_2025['auc']:.6f}",
)


print()
print("2025 FINALISTS")
print()

print(
    final[
        [
            "final_rank",
            "name",
            "family",
            "final_brier",
            "final_brier_vs_cfg",
            "final_brier_vs_market",
            "final_log_loss",
            "final_logloss_vs_cfg",
            "final_auc",
            "final_auc_vs_cfg",
            "final_cal_error",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# DISAGREEMENT-BAND TEST
# ============================================================

print()
print("=" * 120)
print("2025 DISAGREEMENT-BAND PERFORMANCE")
print("=" * 120)


winner_name = (
    final.iloc[0][
        "name"
    ]
)

winner_cfg = config_lookup[
    winner_name
]


z2025[
    "winner_prob"
] = apply_config(
    z2025,
    winner_cfg,
)


z2025[
    "disagreement_band"
] = pd.cut(
    z2025[
        "abs_disagreement"
    ],
    bins=[
        0.00,
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        1.00,
    ],
    include_lowest=True,
)


band_rows = []


for band, z in z2025.groupby(
    "disagreement_band",
    observed=True,
):

    if len(z) < 10:
        continue


    cfg_r = evaluate(
        z["btts_yes"],
        z["champion_yes"],
    )

    mkt_r = evaluate(
        z["btts_yes"],
        z["market_yes"],
    )

    win_r = evaluate(
        z["btts_yes"],
        z["winner_prob"],
    )


    band_rows.append(
        {
            "band":
                str(band),

            "games":
                len(z),

            "actual":
                z[
                    "btts_yes"
                ].mean(),

            "cfg_pred":
                z[
                    "champion_yes"
                ].mean(),

            "market_pred":
                z[
                    "market_yes"
                ].mean(),

            "winner_pred":
                z[
                    "winner_prob"
                ].mean(),

            "cfg_brier":
                cfg_r[
                    "brier"
                ],

            "market_brier":
                mkt_r[
                    "brier"
                ],

            "winner_brier":
                win_r[
                    "brier"
                ],

            "winner_vs_cfg":
                (
                    cfg_r[
                        "brier"
                    ]
                    -
                    win_r[
                        "brier"
                    ]
                ),

            "winner_vs_market":
                (
                    mkt_r[
                        "brier"
                    ]
                    -
                    win_r[
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
    /
    "04_2025_disagreement_bands.csv",
    index=False,
)


print()
print(
    band_df.to_string(
        index=False
    )
)


# ============================================================
# MLS VS ELITESERIEN
# ============================================================

print()
print("=" * 120)
print("2025 LEAGUE SPLIT — WINNER")
print("=" * 120)


league_rows = []


for league in sorted(
    z2025["league"]
    .unique()
):

    z = z2025[
        z2025["league"]
        ==
        league
    ].copy()


    z[
        "winner_prob"
    ] = apply_config(
        z,
        winner_cfg,
    )


    cfg_r = evaluate(
        z["btts_yes"],
        z["champion_yes"],
    )

    mkt_r = evaluate(
        z["btts_yes"],
        z["market_yes"],
    )

    win_r = evaluate(
        z["btts_yes"],
        z["winner_prob"],
    )


    league_rows.append(
        {
            "league":
                league,

            "games":
                len(z),

            "cfg_brier":
                cfg_r["brier"],

            "market_brier":
                mkt_r["brier"],

            "winner_brier":
                win_r["brier"],

            "winner_vs_cfg":
                (
                    cfg_r["brier"]
                    -
                    win_r["brier"]
                ),

            "winner_vs_market":
                (
                    mkt_r["brier"]
                    -
                    win_r["brier"]
                ),

            "cfg_logloss":
                cfg_r["log_loss"],

            "market_logloss":
                mkt_r["log_loss"],

            "winner_logloss":
                win_r["log_loss"],
        }
    )


league_df = pd.DataFrame(
    league_rows
)


league_df.to_csv(
    OUT_DIR
    /
    "05_2025_league_split.csv",
    index=False,
)


print()
print(
    league_df.to_string(
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
    /
    "00_config_catalog.csv",
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
