from pathlib import Path

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
    / "btts_cfg0755_market_matched.csv"
)

V1_DEV_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_betting_lab"
    / "01_development_leaderboard.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "btts_betting_lab_v2"
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
# LOAD MARKET DATA
# ============================================================

print()
print("=" * 120)
print("BTTS BETTING LAB V2")
print("SELECTION METHODOLOGY FIX")
print("=" * 120)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
)


numeric_cols = [
    "test_year",
    "btts_yes",
    "champion_yes",
    "champion_no",
    "market_yes",
    "market_no",
    "odds_yes",
    "odds_no",
    "champion_edge_yes",
    "champion_edge_no",
    "champion_ev_yes",
    "champion_ev_no",
    "home_lambda",
    "away_lambda",
    "lambda_min",
    "lambda_total",
    "lambda_gap",
]


for c in numeric_cols:

    if c in df.columns:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )


# Recreate if needed

if "lambda_min" not in df.columns:

    df["lambda_min"] = np.minimum(
        df["home_lambda"],
        df["away_lambda"],
    )

if "lambda_total" not in df.columns:

    df["lambda_total"] = (
        df["home_lambda"]
        +
        df["away_lambda"]
    )

if "lambda_gap" not in df.columns:

    df["lambda_gap"] = np.abs(
        df["home_lambda"]
        -
        df["away_lambda"]
    )


# ============================================================
# LOAD V1 DEVELOPMENT LEADERBOARD
#
# Important:
# We are NOT searching new configurations using 2025.
# We only reconsider configs already evaluated in development.
# ============================================================

dev = pd.read_csv(
    V1_DEV_FILE,
    low_memory=False,
)


print()
print("Market rows:", len(df))
print("V1 development configurations:", len(dev))


# ============================================================
# FOCUS ON VALID DEVELOPMENT RULES
#
# Preserve V1's pre-2024 development criteria.
# ============================================================

candidates = dev[
    (dev["side"] == "YES")
    &
    (dev["dev_seasons"] >= 3)
    &
    (dev["dev_bets"] >= 75)
    &
    (dev["dev_positive_seasons"] >= 2)
    &
    (dev["dev_worst_season_roi"] > -0.30)
].copy()


print()
print(
    "Eligible pre-2024 development candidates:",
    len(candidates),
)


# ============================================================
# RULE EVALUATION
# ============================================================

def evaluate_rule(
    data,
    row,
):

    if row["side"] == "YES":

        edge_col = "champion_edge_yes"
        ev_col = "champion_ev_yes"
        odds_col = "odds_yes"

    else:

        edge_col = "champion_edge_no"
        ev_col = "champion_ev_no"
        odds_col = "odds_no"


    league_group = row["league_group"]


    if league_group == "ALL":

        allowed = [
            "MLS",
            "Eliteserien",
        ]

    elif league_group == "MLS":

        allowed = [
            "MLS",
        ]

    elif league_group == "ELITESERIEN":

        allowed = [
            "Eliteserien",
        ]

    else:

        return None


    x = data[
        data["league"].isin(
            allowed
        )
    ].copy()


    x = x[
        x[edge_col]
        >= float(
            row["edge_min"]
        )
    ]

    x = x[
        x[ev_col]
        >= float(
            row["ev_min"]
        )
    ]

    x = x[
        x[odds_col]
        >= float(
            row["odds_min"]
        )
    ]

    x = x[
        x[odds_col]
        <
        float(
            row["odds_max"]
        )
    ]

    x = x[
        x["lambda_min"]
        >= float(
            row["lambda_floor"]
        )
    ]

    x = x[
        x["lambda_min"]
        <
        float(
            row["lambda_ceiling"]
        )
    ]


    if len(x) == 0:

        return None


    if row["side"] == "YES":

        x["won"] = (
            x["btts_yes"]
            ==
            1
        )

    else:

        x["won"] = (
            x["btts_yes"]
            ==
            0
        )


    x["odds"] = x[
        odds_col
    ]


    x["profit"] = np.where(
        x["won"],
        x["odds"] - 1.0,
        -1.0,
    )


    return {
        "bets":
            len(x),

        "wins":
            int(
                x["won"].sum()
            ),

        "win_rate":
            x["won"].mean(),

        "profit_units":
            x["profit"].sum(),

        "roi":
            x["profit"].mean(),

        "avg_odds":
            x["odds"].mean(),

        "avg_edge":
            x[edge_col].mean(),

        "avg_ev":
            x[ev_col].mean(),

        "avg_lambda_min":
            x["lambda_min"].mean(),

        "avg_lambda_total":
            x["lambda_total"].mean(),
    }


# ============================================================
# RE-EVALUATE 2024
# ============================================================

print()
print("=" * 120)
print("PHASE 1 — 2024 VALIDATION RE-EVALUATION")
print("=" * 120)


validation_data = df[
    df["test_year"]
    ==
    VALIDATION_YEAR
]


validation_rows = []


for _, row in candidates.iterrows():

    result = evaluate_rule(
        validation_data,
        row,
    )


    if result is None:
        continue


    validation_rows.append(
        {
            **row.to_dict(),

            "validation_bets":
                result["bets"],

            "validation_wins":
                result["wins"],

            "validation_roi":
                result["roi"],

            "validation_profit":
                result["profit_units"],

            "validation_avg_odds":
                result["avg_odds"],

            "validation_avg_edge":
                result["avg_edge"],

            "validation_avg_ev":
                result["avg_ev"],
        }
    )


validation = pd.DataFrame(
    validation_rows
)


# ============================================================
# NEW FORWARD ELIGIBILITY
#
# No arbitrary 20-bet cliff.
#
# Requirements:
# - at least 10 bets in 2024
# - positive 2024 ROI
# - development already had >=75 bets
# - development already positive in >=2/3 seasons
# ============================================================

validation[
    "forward_eligible"
] = (
    (validation["validation_bets"] >= 10)
    &
    (validation["validation_roi"] > 0)
)


# ============================================================
# STABILITY SCORE
#
# Heavily reward:
# - positive development ROI
# - multiple positive development seasons
# - positive validation ROI
# - sufficient samples
#
# Penalize bad worst-year performance.
# ============================================================

validation[
    "stability_score"
] = (
    2.0
    *
    validation[
        "dev_pooled_roi"
    ]
    +
    1.0
    *
    validation[
        "dev_mean_season_roi"
    ]
    +
    0.50
    *
    validation[
        "dev_median_season_roi"
    ]
    +
    1.50
    *
    validation[
        "validation_roi"
    ]
    +
    0.10
    *
    validation[
        "dev_positive_seasons"
    ]
    +
    0.05
    *
    np.log1p(
        validation[
            "dev_bets"
        ]
    )
    +
    0.05
    *
    np.log1p(
        validation[
            "validation_bets"
        ]
    )
    +
    0.50
    *
    validation[
        "dev_worst_season_roi"
    ]
)


validation = validation.sort_values(
    [
        "forward_eligible",
        "stability_score",
        "validation_bets",
        "dev_bets",
    ],
    ascending=[
        False,
        False,
        False,
        False,
    ],
).reset_index(
    drop=True
)


validation[
    "v2_rank"
] = (
    np.arange(
        len(validation)
    )
    +
    1
)


validation.to_csv(
    OUT_DIR
    / "01_v2_validation_2024.csv",
    index=False,
)


print()
print("TOP 30 AFTER CORRECTED 2024 SELECTION")
print()


cols = [
    "v2_rank",
    "config_id",
    "league_group",
    "edge_min",
    "ev_min",
    "odds_min",
    "odds_max",
    "lambda_floor",
    "lambda_ceiling",
    "dev_bets",
    "dev_pooled_roi",
    "dev_positive_seasons",
    "dev_worst_season_roi",
    "validation_bets",
    "validation_roi",
    "forward_eligible",
    "stability_score",
]


print(
    validation[
        cols
    ]
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# EXPLICITLY LOCATE BET_036028 FAMILY
# ============================================================

print()
print("=" * 120)
print("BET_036028 FAMILY CHECK")
print("=" * 120)


family_mask = (
    (validation["league_group"] == "ALL")
    &
    np.isclose(
        validation["edge_min"],
        0.06,
    )
    &
    np.isclose(
        validation["ev_min"],
        0.00,
    )
    &
    np.isclose(
        validation["lambda_floor"],
        0.90,
    )
    &
    np.isclose(
        validation["lambda_ceiling"],
        1.20,
    )
)


family = validation[
    family_mask
].copy()


print()
print(
    family[
        cols
    ]
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# FINALISTS
#
# Chosen ONLY from 2021-2024.
# 2025 has not been used here.
# ============================================================

FINALISTS = (
    validation[
        validation[
            "forward_eligible"
        ]
    ]
    .head(20)
    .copy()
)


print()
print("=" * 120)
print("FINALISTS SELECTED BEFORE 2025")
print("=" * 120)

print()
print(
    FINALISTS[
        cols
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 2025 FINAL TEST
# ============================================================

print()
print("=" * 120)
print("PHASE 2 — 2025 FINAL TEST")
print("=" * 120)


final_data = df[
    df["test_year"]
    ==
    FINAL_YEAR
]


final_rows = []


for _, row in FINALISTS.iterrows():

    result = evaluate_rule(
        final_data,
        row,
    )


    if result is None:
        continue


    final_rows.append(
        {
            **row.to_dict(),

            "final_bets":
                result["bets"],

            "final_wins":
                result["wins"],

            "final_win_rate":
                result["win_rate"],

            "final_profit_units":
                result[
                    "profit_units"
                ],

            "final_roi":
                result["roi"],

            "final_avg_odds":
                result[
                    "avg_odds"
                ],

            "final_avg_edge":
                result[
                    "avg_edge"
                ],

            "final_avg_ev":
                result[
                    "avg_ev"
                ],

            "final_avg_lambda_min":
                result[
                    "avg_lambda_min"
                ],

            "final_avg_lambda_total":
                result[
                    "avg_lambda_total"
                ],
        }
    )


final = pd.DataFrame(
    final_rows
)


if len(final):

    final = final.sort_values(
        [
            "final_roi",
            "final_bets",
        ],
        ascending=[
            False,
            False,
        ],
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
    / "02_v2_final_2025.csv",
    index=False,
)


print()
print("2025 RESULTS")
print()


final_cols = [
    "final_rank",
    "config_id",
    "league_group",
    "edge_min",
    "ev_min",
    "odds_min",
    "odds_max",
    "lambda_floor",
    "lambda_ceiling",
    "dev_bets",
    "dev_pooled_roi",
    "validation_bets",
    "validation_roi",
    "final_bets",
    "final_wins",
    "final_roi",
    "final_avg_odds",
    "final_avg_edge",
    "final_avg_ev",
]


if len(final):

    print(
        final[
            final_cols
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# DIRECT BET_036028 2025 CHECK
#
# This is diagnostic only.
# It was selected by pre-2025 methodology if forward eligible.
# ============================================================

print()
print("=" * 120)
print("DIRECT BET_036028 FAMILY — 2025")
print("=" * 120)


family_final_rows = []


for _, row in family.iterrows():

    result = evaluate_rule(
        final_data,
        row,
    )


    if result is None:
        continue


    family_final_rows.append(
        {
            "config_id":
                row[
                    "config_id"
                ],

            "odds_min":
                row[
                    "odds_min"
                ],

            "odds_max":
                row[
                    "odds_max"
                ],

            "dev_bets":
                row[
                    "dev_bets"
                ],

            "dev_roi":
                row[
                    "dev_pooled_roi"
                ],

            "validation_bets":
                row[
                    "validation_bets"
                ],

            "validation_roi":
                row[
                    "validation_roi"
                ],

            "final_bets":
                result[
                    "bets"
                ],

            "final_wins":
                result[
                    "wins"
                ],

            "final_roi":
                result[
                    "roi"
                ],

            "final_profit":
                result[
                    "profit_units"
                ],

            "final_avg_odds":
                result[
                    "avg_odds"
                ],
        }
    )


family_final = pd.DataFrame(
    family_final_rows
)


family_final.to_csv(
    OUT_DIR
    / "03_bet036028_family_2025.csv",
    index=False,
)


print()
print(
    family_final.to_string(
        index=False
    )
)


# ============================================================
# FIVE-YEAR STABILITY FOR FINALISTS
# ============================================================

print()
print("=" * 120)
print("FIVE-YEAR FINALIST STABILITY")
print("=" * 120)


stability_rows = []


for _, row in FINALISTS.iterrows():

    for year in [
        2021,
        2022,
        2023,
        2024,
        2025,
    ]:

        z = df[
            df["test_year"]
            ==
            year
        ]

        result = evaluate_rule(
            z,
            row,
        )


        if result is None:
            continue


        stability_rows.append(
            {
                "config_id":
                    row[
                        "config_id"
                    ],

                "year":
                    year,

                "bets":
                    result[
                        "bets"
                    ],

                "profit":
                    result[
                        "profit_units"
                    ],

                "roi":
                    result[
                        "roi"
                    ],

                "avg_odds":
                    result[
                        "avg_odds"
                    ],
            }
        )


stability = pd.DataFrame(
    stability_rows
)


stability.to_csv(
    OUT_DIR
    / "04_finalist_year_by_year.csv",
    index=False,
)


summary = (
    stability
    .groupby(
        "config_id"
    )
    .agg(
        seasons=(
            "year",
            "nunique",
        ),

        total_bets=(
            "bets",
            "sum",
        ),

        total_profit=(
            "profit",
            "sum",
        ),

        positive_seasons=(
            "roi",
            lambda x:
                int(
                    (
                        x > 0
                    ).sum()
                ),
        ),

        mean_roi=(
            "roi",
            "mean",
        ),

        median_roi=(
            "roi",
            "median",
        ),

        worst_roi=(
            "roi",
            "min",
        ),

        best_roi=(
            "roi",
            "max",
        ),
    )
    .reset_index()
)


summary[
    "pooled_roi"
] = (
    summary[
        "total_profit"
    ]
    /
    summary[
        "total_bets"
    ]
)


summary = summary.sort_values(
    [
        "positive_seasons",
        "pooled_roi",
        "total_bets",
    ],
    ascending=[
        False,
        False,
        False,
    ],
)


summary.to_csv(
    OUT_DIR
    / "05_finalist_five_year_summary.csv",
    index=False,
)


print()
print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# OUTPUTS
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
