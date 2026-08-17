from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd


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
    / "btts_dynamic_betting_lab"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FROZEN PROBABILITY CALIBRATION
#
# DYN_100_60_50_40
#
# |CFG-market| < 2%  -> 100% CFG
# 2-4%               -> 60% CFG
# 4-6%               -> 50% CFG
# 6%+                 -> 40% CFG
# ============================================================

def dynamic_probability(
    cfg,
    market,
):

    cfg = np.asarray(
        cfg,
        dtype=float,
    )

    market = np.asarray(
        market,
        dtype=float,
    )

    disagreement = np.abs(
        cfg
        -
        market
    )


    weight = np.where(
        disagreement < 0.02,
        1.00,
        np.where(
            disagreement < 0.04,
            0.60,
            np.where(
                disagreement < 0.06,
                0.50,
                0.40,
            ),
        ),
    )


    return (
        weight
        *
        cfg
        +
        (
            1.0
            -
            weight
        )
        *
        market
    )


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 120)
print("BTTS DYNAMIC-CALIBRATED BETTING LAB")
print("FROZEN CALIBRATION: DYN_100_60_50_40")
print("=" * 120)

df = pd.read_csv(
    INPUT,
    low_memory=False,
)


numeric_cols = [
    "test_year",
    "btts_yes",
    "champion_yes",
    "market_yes",
    "odds_yes",
]


for c in numeric_cols:

    if c in df.columns:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )


required = [
    "test_year",
    "btts_yes",
    "champion_yes",
    "market_yes",
    "odds_yes",
]


missing = [
    c
    for c in required
    if c not in df.columns
]


if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


df = df[
    df["test_year"].isin(
        [
            2021,
            2022,
            2023,
            2024,
            2025,
        ]
    )
    &
    df["btts_yes"].notna()
    &
    df["champion_yes"].notna()
    &
    df["market_yes"].notna()
    &
    df["odds_yes"].notna()
].copy()


df["btts_yes"] = (
    df["btts_yes"]
    .astype(int)
)


# ============================================================
# CHECK FOR NO ODDS
# ============================================================

odds_no_candidates = [
    "odds_no",
    "btts_no_odds",
    "no_odds",
]


odds_no_col = None


for c in odds_no_candidates:

    if c in df.columns:

        odds_no_col = c

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

        break


print()
print("Rows:", len(df))
print("NO odds column:", odds_no_col)


# ============================================================
# BUILD FROZEN DYNAMIC PROBABILITY
# ============================================================

df[
    "dynamic_btts_yes"
] = dynamic_probability(
    df["champion_yes"],
    df["market_yes"],
)


df[
    "dynamic_btts_no"
] = (
    1.0
    -
    df[
        "dynamic_btts_yes"
    ]
)


df[
    "dynamic_edge_yes"
] = (
    df[
        "dynamic_btts_yes"
    ]
    -
    df[
        "market_yes"
    ]
)


df[
    "dynamic_ev_yes"
] = (
    df[
        "dynamic_btts_yes"
    ]
    *
    df[
        "odds_yes"
    ]
    -
    1.0
)


# ============================================================
# NO SIDE
# ============================================================

if odds_no_col is not None:

    df[
        "market_no"
    ] = (
        1.0
        -
        df[
            "market_yes"
        ]
    )


    df[
        "dynamic_edge_no"
    ] = (
        df[
            "dynamic_btts_no"
        ]
        -
        df[
            "market_no"
        ]
    )


    df[
        "dynamic_ev_no"
    ] = (
        df[
            "dynamic_btts_no"
        ]
        *
        df[
            odds_no_col
        ]
        -
        1.0
    )


# ============================================================
# SAVE PROBABILITY DATASET
# ============================================================

df.to_csv(
    OUT_DIR
    /
    "00_dynamic_probabilities.csv",
    index=False,
)


# ============================================================
# BETTING FUNCTION
#
# Flat 1-unit stakes.
# ============================================================

def bet_metrics(
    data,
    mask,
    side="YES",
):

    z = data[
        mask
    ].copy()


    if len(z) == 0:

        return {
            "bets":
                0,

            "wins":
                0,

            "win_rate":
                np.nan,

            "profit":
                0.0,

            "roi":
                np.nan,

            "avg_odds":
                np.nan,

            "avg_edge":
                np.nan,

            "avg_ev":
                np.nan,
        }


    if side == "YES":

        won = (
            z[
                "btts_yes"
            ]
            ==
            1
        )

        odds = z[
            "odds_yes"
        ]

        edge = z[
            "dynamic_edge_yes"
        ]

        ev = z[
            "dynamic_ev_yes"
        ]


    else:

        won = (
            z[
                "btts_yes"
            ]
            ==
            0
        )

        odds = z[
            odds_no_col
        ]

        edge = z[
            "dynamic_edge_no"
        ]

        ev = z[
            "dynamic_ev_no"
        ]


    profit = np.where(
        won,
        odds - 1.0,
        -1.0,
    )


    return {
        "bets":
            len(z),

        "wins":
            int(
                won.sum()
            ),

        "win_rate":
            won.mean(),

        "profit":
            profit.sum(),

        "roi":
            profit.mean(),

        "avg_odds":
            odds.mean(),

        "avg_edge":
            edge.mean(),

        "avg_ev":
            ev.mean(),
    }


# ============================================================
# SEARCH SPACE
#
# Development ONLY = 2021-2023.
#
# We're deliberately using a fairly broad grid.
# ============================================================

EDGE_THRESHOLDS = np.round(
    np.arange(
        0.000,
        0.061,
        0.005,
    ),
    3,
)


EV_THRESHOLDS = np.round(
    np.arange(
        0.000,
        0.101,
        0.005,
    ),
    3,
)


MIN_ODDS = [
    1.40,
    1.50,
    1.60,
    1.70,
    1.80,
]


MAX_ODDS = [
    1.90,
    2.00,
    2.10,
    2.20,
    2.40,
    3.00,
]


print()
print(
    "YES configurations:",
    len(
        EDGE_THRESHOLDS
    )
    *
    len(
        EV_THRESHOLDS
    )
    *
    len(
        MIN_ODDS
    )
    *
    len(
        MAX_ODDS
    ),
)


# ============================================================
# PHASE 1
# DEVELOPMENT 2021-2023
# ============================================================

print()
print("=" * 120)
print("PHASE 1 — DEVELOPMENT 2021-2023")
print("=" * 120)


dev = df[
    df[
        "test_year"
    ].isin(
        [
            2021,
            2022,
            2023,
        ]
    )
].copy()


rows = []


for (
    edge_min,
    ev_min,
    odds_min,
    odds_max,
) in product(
    EDGE_THRESHOLDS,
    EV_THRESHOLDS,
    MIN_ODDS,
    MAX_ODDS,
):

    if odds_max <= odds_min:
        continue


    overall_mask = (
        (
            dev[
                "dynamic_edge_yes"
            ]
            >=
            edge_min
        )
        &
        (
            dev[
                "dynamic_ev_yes"
            ]
            >=
            ev_min
        )
        &
        (
            dev[
                "odds_yes"
            ]
            >=
            odds_min
        )
        &
        (
            dev[
                "odds_yes"
            ]
            <=
            odds_max
        )
    )


    overall = bet_metrics(
        dev,
        overall_mask,
        "YES",
    )


    if overall[
        "bets"
    ] < 40:
        continue


    yearly = []


    for year in [
        2021,
        2022,
        2023,
    ]:

        z = dev[
            dev[
                "test_year"
            ]
            ==
            year
        ]


        mask = (
            (
                z[
                    "dynamic_edge_yes"
                ]
                >=
                edge_min
            )
            &
            (
                z[
                    "dynamic_ev_yes"
                ]
                >=
                ev_min
            )
            &
            (
                z[
                    "odds_yes"
                ]
                >=
                odds_min
            )
            &
            (
                z[
                    "odds_yes"
                ]
                <=
                odds_max
            )
        )


        r = bet_metrics(
            z,
            mask,
            "YES",
        )


        yearly.append(
            {
                "year":
                    year,

                **r,
            }
        )


    yr = pd.DataFrame(
        yearly
    )


    valid_roi = yr[
        "roi"
    ].dropna()


    positive_years = int(
        (
            valid_roi
            >
            0
        ).sum()
    )


    worst_year_roi = (
        valid_roi.min()
        if len(
            valid_roi
        )
        else np.nan
    )


    median_year_roi = (
        valid_roi.median()
        if len(
            valid_roi
        )
        else np.nan
    )


    # Robustness-oriented development score.
    # We don't want the highest raw ROI.
    score = (
        3.0
        *
        overall[
            "roi"
        ]
        +
        0.75
        *
        median_year_roi
        +
        0.20
        *
        positive_years
        +
        0.75
        *
        worst_year_roi
        +
        0.0005
        *
        min(
            overall[
                "bets"
            ],
            300,
        )
    )


    rows.append(
        {
            "edge_min":
                edge_min,

            "ev_min":
                ev_min,

            "odds_min":
                odds_min,

            "odds_max":
                odds_max,

            **{
                f"dev_{k}":
                    v
                for k, v
                in overall.items()
            },

            "positive_years":
                positive_years,

            "worst_year_roi":
                worst_year_roi,

            "median_year_roi":
                median_year_roi,

            "dev_score":
                score,
        }
    )


dev_results = pd.DataFrame(
    rows
)


dev_results = (
    dev_results
    .sort_values(
        [
            "dev_score",
            "dev_roi",
            "dev_bets",
        ],
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


dev_results[
    "dev_rank"
] = (
    np.arange(
        len(
            dev_results
        )
    )
    +
    1
)


dev_results.to_csv(
    OUT_DIR
    /
    "01_yes_development_search.csv",
    index=False,
)


print()
print("TOP 30 DEVELOPMENT RULES")
print()

print(
    dev_results[
        [
            "dev_rank",
            "edge_min",
            "ev_min",
            "odds_min",
            "odds_max",
            "dev_bets",
            "dev_win_rate",
            "dev_profit",
            "dev_roi",
            "positive_years",
            "worst_year_roi",
            "median_year_roi",
            "dev_score",
        ]
    ]
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# PHASE 2
# VALIDATION 2024
#
# Only top development candidates are allowed through.
# ============================================================

print()
print("=" * 120)
print("PHASE 2 — VALIDATION 2024")
print("=" * 120)


top_dev = (
    dev_results
    .head(100)
    .copy()
)


validation = df[
    df[
        "test_year"
    ]
    ==
    2024
].copy()


val_rows = []


for _, rule in top_dev.iterrows():

    mask = (
        (
            validation[
                "dynamic_edge_yes"
            ]
            >=
            rule[
                "edge_min"
            ]
        )
        &
        (
            validation[
                "dynamic_ev_yes"
            ]
            >=
            rule[
                "ev_min"
            ]
        )
        &
        (
            validation[
                "odds_yes"
            ]
            >=
            rule[
                "odds_min"
            ]
        )
        &
        (
            validation[
                "odds_yes"
            ]
            <=
            rule[
                "odds_max"
            ]
        )
    )


    r = bet_metrics(
        validation,
        mask,
        "YES",
    )


    # Reward validation profitability,
    # but don't allow a tiny sample to dominate.

    val_score = (
        rule[
            "dev_score"
        ]
        +
        (
            3.0
            *
            r[
                "roi"
            ]
            if np.isfinite(
                r[
                    "roi"
                ]
            )
            else -1.0
        )
        +
        0.001
        *
        min(
            r[
                "bets"
            ],
            100,
        )
    )


    val_rows.append(
        {
            **rule.to_dict(),

            **{
                f"val_{k}":
                    v
                for k, v
                in r.items()
            },

            "validation_score":
                val_score,
        }
    )


val_results = pd.DataFrame(
    val_rows
)


val_results = (
    val_results
    .sort_values(
        [
            "validation_score",
            "val_roi",
            "val_bets",
        ],
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


val_results[
    "validation_rank"
] = (
    np.arange(
        len(
            val_results
        )
    )
    +
    1
)


val_results.to_csv(
    OUT_DIR
    /
    "02_yes_validation_2024.csv",
    index=False,
)


print()
print("TOP 20 AFTER 2024")
print()

print(
    val_results[
        [
            "validation_rank",
            "edge_min",
            "ev_min",
            "odds_min",
            "odds_max",
            "dev_bets",
            "dev_roi",
            "positive_years",
            "worst_year_roi",
            "val_bets",
            "val_win_rate",
            "val_profit",
            "val_roi",
            "validation_score",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# FREEZE FINALISTS BEFORE 2025
# ============================================================

FINALISTS = (
    val_results
    .head(10)
    .copy()
)


print()
print("=" * 120)
print("FINALISTS FROZEN BEFORE 2025")
print("=" * 120)

print()

print(
    FINALISTS[
        [
            "validation_rank",
            "edge_min",
            "ev_min",
            "odds_min",
            "odds_max",
            "dev_bets",
            "dev_roi",
            "val_bets",
            "val_roi",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# PHASE 3
# UNTOUCHED 2025 TEST
# ============================================================

print()
print("=" * 120)
print("PHASE 3 — UNTOUCHED 2025 TEST")
print("=" * 120)


test = df[
    df[
        "test_year"
    ]
    ==
    2025
].copy()


test_rows = []


for _, rule in FINALISTS.iterrows():

    mask = (
        (
            test[
                "dynamic_edge_yes"
            ]
            >=
            rule[
                "edge_min"
            ]
        )
        &
        (
            test[
                "dynamic_ev_yes"
            ]
            >=
            rule[
                "ev_min"
            ]
        )
        &
        (
            test[
                "odds_yes"
            ]
            >=
            rule[
                "odds_min"
            ]
        )
        &
        (
            test[
                "odds_yes"
            ]
            <=
            rule[
                "odds_max"
            ]
        )
    )


    r = bet_metrics(
        test,
        mask,
        "YES",
    )


    test_rows.append(
        {
            **rule.to_dict(),

            **{
                f"test_{k}":
                    v
                for k, v
                in r.items()
            },
        }
    )


test_results = pd.DataFrame(
    test_rows
)


test_results.to_csv(
    OUT_DIR
    /
    "03_yes_final_2025.csv",
    index=False,
)


print()
print("2025 FINAL RESULTS")
print()

print(
    test_results[
        [
            "validation_rank",
            "edge_min",
            "ev_min",
            "odds_min",
            "odds_max",
            "dev_bets",
            "dev_roi",
            "val_bets",
            "val_roi",
            "test_bets",
            "test_win_rate",
            "test_profit",
            "test_roi",
            "test_avg_odds",
            "test_avg_edge",
            "test_avg_ev",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# YEAR-BY-YEAR FOR THE PRE-2025 #1 RULE
# ============================================================

winner = FINALISTS.iloc[0]


print()
print("=" * 120)
print("PRE-2025 #1 RULE — YEAR-BY-YEAR")
print("=" * 120)

print()

print(
    "Edge >=",
    winner[
        "edge_min"
    ],
)

print(
    "EV >=",
    winner[
        "ev_min"
    ],
)

print(
    "Odds:",
    winner[
        "odds_min"
    ],
    "to",
    winner[
        "odds_max"
    ],
)


year_rows = []


for year in [
    2021,
    2022,
    2023,
    2024,
    2025,
]:

    z = df[
        df[
            "test_year"
        ]
        ==
        year
    ]


    mask = (
        (
            z[
                "dynamic_edge_yes"
            ]
            >=
            winner[
                "edge_min"
            ]
        )
        &
        (
            z[
                "dynamic_ev_yes"
            ]
            >=
            winner[
                "ev_min"
            ]
        )
        &
        (
            z[
                "odds_yes"
            ]
            >=
            winner[
                "odds_min"
            ]
        )
        &
        (
            z[
                "odds_yes"
            ]
            <=
            winner[
                "odds_max"
            ]
        )
    )


    r = bet_metrics(
        z,
        mask,
        "YES",
    )


    year_rows.append(
        {
            "year":
                year,

            **r,
        }
    )


year_df = pd.DataFrame(
    year_rows
)


year_df.to_csv(
    OUT_DIR
    /
    "04_winner_year_by_year.csv",
    index=False,
)


print()
print(
    year_df.to_string(
        index=False
    )
)


# ============================================================
# LEAGUE SPLIT
# ============================================================

print()
print("=" * 120)
print("PRE-2025 #1 RULE — LEAGUE SPLIT")
print("=" * 120)


league_rows = []


for league in sorted(
    df["league"]
    .dropna()
    .unique()
):

    z = df[
        df[
            "league"
        ]
        ==
        league
    ]


    mask = (
        (
            z[
                "dynamic_edge_yes"
            ]
            >=
            winner[
                "edge_min"
            ]
        )
        &
        (
            z[
                "dynamic_ev_yes"
            ]
            >=
            winner[
                "ev_min"
            ]
        )
        &
        (
            z[
                "odds_yes"
            ]
            >=
            winner[
                "odds_min"
            ]
        )
        &
        (
            z[
                "odds_yes"
            ]
            <=
            winner[
                "odds_max"
            ]
        )
    )


    r = bet_metrics(
        z,
        mask,
        "YES",
    )


    league_rows.append(
        {
            "league":
                league,

            **r,
        }
    )


league_df = pd.DataFrame(
    league_rows
)


league_df.to_csv(
    OUT_DIR
    /
    "05_winner_league_split.csv",
    index=False,
)


print()
print(
    league_df.to_string(
        index=False
    )
)


# ============================================================
# NO-SIDE DIAGNOSTIC
#
# We do NOT optimize NO yet.
# First see whether the data supports it at all.
# ============================================================

if odds_no_col is not None:

    print()
    print("=" * 120)
    print("BTTS NO — SIMPLE EV DIAGNOSTIC")
    print("=" * 120)


    no_rows = []


    for ev_min in [
        0.00,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.10,
    ]:

        for year in [
            2021,
            2022,
            2023,
            2024,
            2025,
        ]:

            z = df[
                df[
                    "test_year"
                ]
                ==
                year
            ]


            mask = (
                z[
                    "dynamic_ev_no"
                ]
                >=
                ev_min
            )


            r = bet_metrics(
                z,
                mask,
                "NO",
            )


            no_rows.append(
                {
                    "ev_min":
                        ev_min,

                    "year":
                        year,

                    **r,
                }
            )


    no_df = pd.DataFrame(
        no_rows
    )


    no_df.to_csv(
        OUT_DIR
        /
        "06_no_ev_diagnostic.csv",
        index=False,
    )


    print()
    print(
        no_df.to_string(
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
