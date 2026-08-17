from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_cfg0755_market_matched.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "btts_betting_lab"
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
# SEARCH SPACE
# ============================================================

EDGE_THRESHOLDS = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
]


EV_THRESHOLDS = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.14,
    0.16,
]


ODDS_MIN = [
    1.00,
    1.50,
    1.60,
    1.70,
    1.80,
    1.90,
    2.00,
]


ODDS_MAX = [
    1.80,
    2.00,
    2.20,
    2.50,
    3.00,
    10.00,
]


LAMBDA_MIN_FLOORS = [
    0.00,
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
]


LAMBDA_MIN_CEILINGS = [
    1.20,
    1.30,
    1.40,
    1.50,
    1.70,
    10.00,
]


LEAGUE_GROUPS = {
    "ALL": [
        "MLS",
        "Eliteserien",
    ],
    "MLS": [
        "MLS",
    ],
    "ELITESERIEN": [
        "Eliteserien",
    ],
}


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 120)
print("BTTS BETTING EXPERIMENT LAB")
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
    "poisson_yes",
    "poisson_no",
    "poisson_edge_yes",
    "poisson_edge_no",
    "poisson_ev_yes",
    "poisson_ev_no",
    "home_lambda",
    "away_lambda",
]


for c in numeric_cols:

    if c in df.columns:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )


# ============================================================
# RECREATE STRUCTURAL VARIABLES
# ============================================================

df["lambda_min"] = np.minimum(
    df["home_lambda"],
    df["away_lambda"],
)

df["lambda_max"] = np.maximum(
    df["home_lambda"],
    df["away_lambda"],
)

df["lambda_total"] = (
    df["home_lambda"]
    +
    df["away_lambda"]
)

df["lambda_gap"] = np.abs(
    df["home_lambda"]
    -
    df["away_lambda"]
)


# ============================================================
# CLEAN
# ============================================================

df = df[
    df["test_year"].isin(
        DEV_YEARS
        +
        [
            VALIDATION_YEAR,
            FINAL_YEAR,
        ]
    )
].copy()


print()
print("Rows:", len(df))

print()
print(
    df.groupby(
        [
            "league",
            "test_year",
        ]
    )
    .size()
    .to_string()
)


# ============================================================
# BET SETUP
# ============================================================

def get_side_columns(
    side,
):

    if side == "YES":

        return {
            "prob":
                "champion_yes",

            "market":
                "market_yes",

            "odds":
                "odds_yes",

            "edge":
                "champion_edge_yes",

            "ev":
                "champion_ev_yes",
        }

    return {
        "prob":
            "champion_no",

        "market":
            "market_no",

        "odds":
            "odds_no",

        "edge":
            "champion_edge_no",

        "ev":
            "champion_ev_no",
    }


# ============================================================
# EVALUATE RULE
# ============================================================

def evaluate_rule(
    data,
    cfg,
):

    side = cfg["side"]

    cols = get_side_columns(
        side
    )


    x = data[
        data["league"].isin(
            LEAGUE_GROUPS[
                cfg["league_group"]
            ]
        )
    ].copy()


    x = x[
        x[cols["edge"]]
        >=
        cfg["edge_min"]
    ]


    x = x[
        x[cols["ev"]]
        >=
        cfg["ev_min"]
    ]


    x = x[
        x[cols["odds"]]
        >=
        cfg["odds_min"]
    ]


    x = x[
        x[cols["odds"]]
        <
        cfg["odds_max"]
    ]


    x = x[
        x["lambda_min"]
        >=
        cfg["lambda_floor"]
    ]


    x = x[
        x["lambda_min"]
        <
        cfg["lambda_ceiling"]
    ]


    if len(x) == 0:

        return None


    if side == "YES":

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


    x["odds"] = (
        x[cols["odds"]]
    )


    x["profit"] = np.where(
        x["won"],
        x["odds"]
        -
        1.0,
        -1.0,
    )


    bets = len(x)

    wins = int(
        x["won"].sum()
    )

    profit = float(
        x["profit"].sum()
    )


    return {
        "bets":
            bets,

        "wins":
            wins,

        "win_rate":
            wins / bets,

        "profit_units":
            profit,

        "roi":
            profit / bets,

        "avg_odds":
            x["odds"].mean(),

        "avg_edge":
            x[cols["edge"]].mean(),

        "avg_ev":
            x[cols["ev"]].mean(),

        "avg_lambda_min":
            x["lambda_min"].mean(),

        "avg_lambda_total":
            x["lambda_total"].mean(),
    }


# ============================================================
# CONFIG GENERATION
# ============================================================

configs = []


# ------------------------------------------------------------
# YES primary search
# ------------------------------------------------------------

for (
    league_group,
    edge_min,
    ev_min,
    odds_min,
    odds_max,
    lambda_floor,
    lambda_ceiling,
) in product(
    LEAGUE_GROUPS.keys(),
    EDGE_THRESHOLDS,
    EV_THRESHOLDS,
    ODDS_MIN,
    ODDS_MAX,
    LAMBDA_MIN_FLOORS,
    LAMBDA_MIN_CEILINGS,
):

    if odds_min >= odds_max:
        continue

    if lambda_floor >= lambda_ceiling:
        continue


    configs.append(
        {
            "side":
                "YES",

            "league_group":
                league_group,

            "edge_min":
                edge_min,

            "ev_min":
                ev_min,

            "odds_min":
                odds_min,

            "odds_max":
                odds_max,

            "lambda_floor":
                lambda_floor,

            "lambda_ceiling":
                lambda_ceiling,
        }
    )


# ------------------------------------------------------------
# NO control search
#
# Much smaller because current evidence is poor.
# ------------------------------------------------------------

for (
    league_group,
    edge_min,
    ev_min,
) in product(
    LEAGUE_GROUPS.keys(),
    [
        0.00,
        0.04,
        0.08,
        0.12,
    ],
    [
        0.00,
        0.04,
        0.08,
        0.12,
    ],
):

    configs.append(
        {
            "side":
                "NO",

            "league_group":
                league_group,

            "edge_min":
                edge_min,

            "ev_min":
                ev_min,

            "odds_min":
                1.00,

            "odds_max":
                10.00,

            "lambda_floor":
                0.00,

            "lambda_ceiling":
                10.00,
        }
    )


for i, cfg in enumerate(
    configs,
    start=1,
):

    cfg["config_id"] = (
        f"BET_{i:06d}"
    )


print()
print(
    "Candidate betting configurations:",
    len(configs),
)


# ============================================================
# DEVELOPMENT
# ============================================================

print()
print("=" * 120)
print("PHASE 1 — DEVELOPMENT")
print("2021-2023")
print("=" * 120)


dev_rows = []


for i, cfg in enumerate(
    configs,
    start=1,
):

    year_rows = []


    for year in DEV_YEARS:

        z = df[
            df["test_year"]
            ==
            year
        ]

        result = evaluate_rule(
            z,
            cfg,
        )


        if result is None:
            continue


        year_rows.append(
            {
                "year":
                    year,

                **result,
            }
        )


    if len(year_rows) == 0:
        continue


    yr = pd.DataFrame(
        year_rows
    )


    total_bets = int(
        yr["bets"].sum()
    )

    total_profit = float(
        yr["profit_units"].sum()
    )

    pooled_roi = (
        total_profit
        /
        total_bets
        if total_bets
        else np.nan
    )


    dev_rows.append(
        {
            **cfg,

            "dev_seasons":
                len(yr),

            "dev_bets":
                total_bets,

            "dev_profit_units":
                total_profit,

            "dev_pooled_roi":
                pooled_roi,

            "dev_mean_season_roi":
                yr["roi"].mean(),

            "dev_median_season_roi":
                yr["roi"].median(),

            "dev_positive_seasons":
                int(
                    (
                        yr["roi"]
                        >
                        0
                    ).sum()
                ),

            "dev_worst_season_roi":
                yr["roi"].min(),

            "dev_best_season_roi":
                yr["roi"].max(),

            "dev_avg_odds":
                np.average(
                    yr["avg_odds"],
                    weights=yr["bets"],
                ),

            "dev_avg_edge":
                np.average(
                    yr["avg_edge"],
                    weights=yr["bets"],
                ),

            "dev_avg_ev":
                np.average(
                    yr["avg_ev"],
                    weights=yr["bets"],
                ),
        }
    )


    if (
        i % 10000
        ==
        0
        or
        i == len(configs)
    ):

        print(
            f"Completed "
            f"{i:,}/"
            f"{len(configs):,}"
        )


dev = pd.DataFrame(
    dev_rows
)


# ============================================================
# DEVELOPMENT ELIGIBILITY
#
# Avoid tiny-sample jackpots.
# ============================================================

dev["dev_eligible"] = (
    (
        dev["side"]
        ==
        "YES"
    )
    &
    (
        dev["dev_seasons"]
        >=
        3
    )
    &
    (
        dev["dev_bets"]
        >=
        75
    )
    &
    (
        dev["dev_positive_seasons"]
        >=
        2
    )
    &
    (
        dev["dev_worst_season_roi"]
        >
        -0.30
    )
)


# NO control allowed only if extremely convincing

no_mask = (
    (
        dev["side"]
        ==
        "NO"
    )
    &
    (
        dev["dev_seasons"]
        >=
        3
    )
    &
    (
        dev["dev_bets"]
        >=
        100
    )
    &
    (
        dev["dev_positive_seasons"]
        >=
        2
    )
)


dev.loc[
    no_mask,
    "dev_eligible"
] = True


# ============================================================
# DEVELOPMENT SCORE
#
# ROI + consistency + sample size.
# ============================================================

dev["sample_score"] = np.log1p(
    dev["dev_bets"]
) / 10.0


dev["dev_score"] = (
    4.0
    *
    dev["dev_pooled_roi"]
    +
    2.0
    *
    dev["dev_mean_season_roi"]
    +
    1.0
    *
    dev["dev_median_season_roi"]
    +
    0.15
    *
    dev["dev_positive_seasons"]
    +
    0.10
    *
    dev["sample_score"]
    +
    0.5
    *
    dev["dev_worst_season_roi"]
)


dev = dev.sort_values(
    [
        "dev_eligible",
        "dev_score",
        "dev_bets",
    ],
    ascending=[
        False,
        False,
        False,
    ],
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
print("=" * 120)
print("TOP 40 DEVELOPMENT RULES")
print("=" * 120)


show_cols = [
    "dev_rank",
    "config_id",
    "side",
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
    "dev_score",
]


print()
print(
    dev[
        show_cols
    ]
    .head(40)
    .to_string(
        index=False
    )
)


# ============================================================
# PHASE 2 — 2024 VALIDATION
# ============================================================

print()
print("=" * 120)
print("PHASE 2 — 2024 VALIDATION")
print("=" * 120)


TOP_DEV = (
    dev[
        dev["dev_eligible"]
    ]
    .head(100)
    .copy()
)


config_lookup = {
    cfg["config_id"]:
        cfg
    for cfg in configs
}


validation_rows = []


validation_data = df[
    df["test_year"]
    ==
    VALIDATION_YEAR
]


for _, row in TOP_DEV.iterrows():

    cfg = config_lookup[
        row["config_id"]
    ]

    result = evaluate_rule(
        validation_data,
        cfg,
    )


    if result is None:
        continue


    validation_rows.append(
        {
            **row.to_dict(),

            "validation_bets":
                result["bets"],

            "validation_profit_units":
                result["profit_units"],

            "validation_roi":
                result["roi"],

            "validation_win_rate":
                result["win_rate"],

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


# Require a minimum number of 2024 bets.
validation["validation_eligible"] = (
    validation[
        "validation_bets"
    ]
    >=
    20
)


validation["validation_score"] = (
    validation["dev_score"]
    +
    5.0
    *
    validation["validation_roi"]
    +
    0.05
    *
    np.log1p(
        validation["validation_bets"]
    )
)


validation = validation.sort_values(
    [
        "validation_eligible",
        "validation_score",
        "validation_bets",
    ],
    ascending=[
        False,
        False,
        False,
    ],
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
    / "02_validation_2024_leaderboard.csv",
    index=False,
)


print()
print("TOP 25 AFTER 2024")
print()


validation_cols = [
    "validation_rank",
    "config_id",
    "side",
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
    "validation_bets",
    "validation_roi",
    "validation_score",
]


print(
    validation[
        validation_cols
    ]
    .head(25)
    .to_string(
        index=False
    )
)


# ============================================================
# PHASE 3 — FINAL UNTOUCHED 2025
# ============================================================

print()
print("=" * 120)
print("PHASE 3 — UNTOUCHED 2025 FINAL TEST")
print("=" * 120)


FINALISTS = (
    validation[
        validation[
            "validation_eligible"
        ]
    ]
    .head(10)
    .copy()
)


final_data = df[
    df["test_year"]
    ==
    FINAL_YEAR
]


final_rows = []


for _, row in FINALISTS.iterrows():

    cfg = config_lookup[
        row["config_id"]
    ]

    result = evaluate_rule(
        final_data,
        cfg,
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
                result["profit_units"],

            "final_roi":
                result["roi"],

            "final_avg_odds":
                result["avg_odds"],

            "final_avg_edge":
                result["avg_edge"],

            "final_avg_ev":
                result["avg_ev"],

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


    final["final_rank"] = (
        np.arange(
            len(final)
        )
        +
        1
    )


final.to_csv(
    OUT_DIR
    / "03_final_2025_results.csv",
    index=False,
)


print()
print("FINALISTS — 2025")
print()


final_cols = [
    "final_rank",
    "config_id",
    "side",
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

else:

    print(
        "No finalists produced bets in 2025."
    )


# ============================================================
# NEIGHBORHOOD ROBUSTNESS
#
# For the winning rule only.
# ============================================================

if len(final):

    winner_id = final.iloc[0][
        "config_id"
    ]

    winner = config_lookup[
        winner_id
    ]


    print()
    print("=" * 120)
    print(
        f"WINNER ROBUSTNESS: "
        f"{winner_id}"
    )
    print("=" * 120)


    neighborhood_rows = []


    edge_values = sorted(
        set(
            [
                max(
                    0,
                    winner[
                        "edge_min"
                    ]
                    -
                    0.02
                ),

                winner[
                    "edge_min"
                ],

                winner[
                    "edge_min"
                ]
                +
                0.02,
            ]
        )
    )


    ev_values = sorted(
        set(
            [
                max(
                    0,
                    winner[
                        "ev_min"
                    ]
                    -
                    0.02
                ),

                winner[
                    "ev_min"
                ],

                winner[
                    "ev_min"
                ]
                +
                0.02,
            ]
        )
    )


    odds_min_values = sorted(
        set(
            [
                max(
                    1.0,
                    winner[
                        "odds_min"
                    ]
                    -
                    0.10
                ),

                winner[
                    "odds_min"
                ],

                winner[
                    "odds_min"
                ]
                +
                0.10,
            ]
        )
    )


    for (
        edge_min,
        ev_min,
        odds_min,
    ) in product(
        edge_values,
        ev_values,
        odds_min_values,
    ):

        cfg = winner.copy()

        cfg["edge_min"] = (
            edge_min
        )

        cfg["ev_min"] = (
            ev_min
        )

        cfg["odds_min"] = (
            odds_min
        )


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

            result = evaluate_rule(
                z,
                cfg,
            )


            if result is None:
                continue


            neighborhood_rows.append(
                {
                    "year":
                        year,

                    "edge_min":
                        edge_min,

                    "ev_min":
                        ev_min,

                    "odds_min":
                        odds_min,

                    **result,
                }
            )


    neighborhood = pd.DataFrame(
        neighborhood_rows
    )


    neighborhood.to_csv(
        OUT_DIR
        / "04_winner_neighborhood.csv",
        index=False,
    )


    robust = (
        neighborhood
        .groupby(
            [
                "edge_min",
                "ev_min",
                "odds_min",
            ]
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
                "profit_units",
                "sum",
            ),

            mean_roi=(
                "roi",
                "mean",
            ),

            median_roi=(
                "roi",
                "median",
            ),

            positive_seasons=(
                "roi",
                lambda x:
                    int(
                        (
                            x
                            >
                            0
                        ).sum()
                    ),
            ),

            worst_roi=(
                "roi",
                "min",
            ),
        )
        .reset_index()
    )


    robust["pooled_roi"] = (
        robust["total_profit"]
        /
        robust["total_bets"]
    )


    robust = robust.sort_values(
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


    robust.to_csv(
        OUT_DIR
        / "05_winner_neighborhood_summary.csv",
        index=False,
    )


    print()
    print(
        robust.head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# BASELINE TABLE
#
# Useful reference rules
# ============================================================

print()
print("=" * 120)
print("REFERENCE RULES")
print("=" * 120)


reference_configs = [
    {
        "name":
            "MLS YES EDGE 4%",

        "side":
            "YES",

        "league_group":
            "MLS",

        "edge_min":
            0.04,

        "ev_min":
            0.00,

        "odds_min":
            1.00,

        "odds_max":
            10.00,

        "lambda_floor":
            0.00,

        "lambda_ceiling":
            10.00,
    },

    {
        "name":
            "MLS YES EDGE 6%",

        "side":
            "YES",

        "league_group":
            "MLS",

        "edge_min":
            0.06,

        "ev_min":
            0.00,

        "odds_min":
            1.00,

        "odds_max":
            10.00,

        "lambda_floor":
            0.00,

        "lambda_ceiling":
            10.00,
    },

    {
        "name":
            "MLS YES EV 8%",

        "side":
            "YES",

        "league_group":
            "MLS",

        "edge_min":
            0.00,

        "ev_min":
            0.08,

        "odds_min":
            1.00,

        "odds_max":
            10.00,

        "lambda_floor":
            0.00,

        "lambda_ceiling":
            10.00,
    },

    {
        "name":
            "MLS YES EV 12%",

        "side":
            "YES",

        "league_group":
            "MLS",

        "edge_min":
            0.00,

        "ev_min":
            0.12,

        "odds_min":
            1.00,

        "odds_max":
            10.00,

        "lambda_floor":
            0.00,

        "lambda_ceiling":
            10.00,
    },
]


reference_rows = []


for cfg in reference_configs:

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

        result = evaluate_rule(
            z,
            cfg,
        )


        if result:

            reference_rows.append(
                {
                    "rule":
                        cfg["name"],

                    "year":
                        year,

                    **result,
                }
            )


reference = pd.DataFrame(
    reference_rows
)


reference.to_csv(
    OUT_DIR
    / "06_reference_rules.csv",
    index=False,
)


print()
print(
    reference.to_string(
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
    / "00_betting_config_catalog.csv",
    index=False,
)


print()
print("=" * 120)
print("OUTPUTS")
print("=" * 120)

for path in sorted(
    OUT_DIR.glob("*")
):

    print(path)


print()
print("DONE")
