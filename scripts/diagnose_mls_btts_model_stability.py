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
    / "btts_cfg0755_market_matched.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "mls_btts_stability_diag"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 120)
print("MLS BTTS MODEL STABILITY DIAGNOSTIC")
print("CFG_0755 VS POISSON VS MARKET")
print("=" * 120)


df = pd.read_csv(
    INPUT,
    low_memory=False,
)


df = df[
    df["league"]
    ==
    "MLS"
].copy()


df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)


numeric_cols = [
    "test_year",
    "btts_yes",

    "champion_yes",
    "poisson_yes",
    "market_yes",

    "champion_edge_yes",
    "champion_ev_yes",

    "odds_yes",

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


df = df[
    df["date"].notna()
    &
    df["btts_yes"].notna()
    &
    df["champion_yes"].notna()
    &
    df["poisson_yes"].notna()
    &
    df["market_yes"].notna()
].copy()


df["btts_yes"] = (
    df["btts_yes"]
    .astype(int)
)


df["month"] = (
    df["date"]
    .dt.month
)


# ============================================================
# SEASON PHASE
# ============================================================

def season_phase(month):

    if month <= 4:
        return "EARLY"

    if month <= 7:
        return "MID"

    return "LATE"


df["season_phase"] = (
    df["month"]
    .map(season_phase)
)


# ============================================================
# GAME NUMBER BY TEAM
#
# Leakage-safe within each season:
# game 1 means first appearance of that team that season.
# ============================================================

home = df[
    [
        "test_year",
        "date",
        "home_team",
    ]
].copy()

home = home.rename(
    columns={
        "home_team":
            "team",
    }
)

home["venue"] = "HOME"


away = df[
    [
        "test_year",
        "date",
        "away_team",
    ]
].copy()

away = away.rename(
    columns={
        "away_team":
            "team",
    }
)

away["venue"] = "AWAY"


team_games = pd.concat(
    [
        home,
        away,
    ],
    ignore_index=True,
)


team_games = (
    team_games
    .sort_values(
        [
            "test_year",
            "team",
            "date",
        ]
    )
    .reset_index(
        drop=True
    )
)


team_games[
    "team_game_number"
] = (
    team_games
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


home_numbers = team_games[
    team_games["venue"]
    ==
    "HOME"
][
    [
        "test_year",
        "date",
        "team",
        "team_game_number",
    ]
].rename(
    columns={
        "team":
            "home_team",

        "team_game_number":
            "home_game_number",
    }
)


away_numbers = team_games[
    team_games["venue"]
    ==
    "AWAY"
][
    [
        "test_year",
        "date",
        "team",
        "team_game_number",
    ]
].rename(
    columns={
        "team":
            "away_team",

        "team_game_number":
            "away_game_number",
    }
)


df = df.merge(
    home_numbers,
    on=[
        "test_year",
        "date",
        "home_team",
    ],
    how="left",
)


df = df.merge(
    away_numbers,
    on=[
        "test_year",
        "date",
        "away_team",
    ],
    how="left",
)


df["minimum_team_game_number"] = np.minimum(
    df["home_game_number"],
    df["away_game_number"],
)


def experience_band(n):

    if pd.isna(n):
        return "UNKNOWN"

    if n <= 5:
        return "1-5"

    if n <= 10:
        return "6-10"

    if n <= 15:
        return "11-15"

    if n <= 20:
        return "16-20"

    return "21+"


df["team_experience_band"] = (
    df[
        "minimum_team_game_number"
    ]
    .map(
        experience_band
    )
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

        "actual_rate":
            y.mean(),

        "avg_pred":
            p.mean(),

        "calibration_error":
            y.mean()
            -
            p.mean(),

        "brier":
            brier_score_loss(
                y,
                p,
            ),

        "log_loss":
            log_loss(
                y,
                p,
                labels=[0, 1],
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
    }


MODELS = {
    "POISSON":
        "poisson_yes",

    "CFG_0755":
        "champion_yes",

    "MARKET":
        "market_yes",
}


# ============================================================
# OVERALL YEAR
# ============================================================

print()
print("=" * 120)
print("MLS MODEL QUALITY BY YEAR")
print("=" * 120)


year_rows = []


for year in sorted(
    df["test_year"]
    .dropna()
    .unique()
):

    z = df[
        df["test_year"]
        ==
        year
    ]


    for name, col in MODELS.items():

        r = evaluate(
            z["btts_yes"],
            z[col],
        )

        year_rows.append(
            {
                "year":
                    int(year),

                "model":
                    name,

                **r,
            }
        )


year_summary = pd.DataFrame(
    year_rows
)


yd = year_summary.copy()


for c in [
    "actual_rate",
    "avg_pred",
    "calibration_error",
]:

    yd[c] = yd[c].map(
        lambda x: f"{x:+.2%}"
    )


for c in [
    "brier",
    "log_loss",
]:

    yd[c] = yd[c].map(
        lambda x: f"{x:.5f}"
    )


yd["auc"] = yd["auc"].map(
    lambda x: f"{x:.4f}"
)


print()
print(
    yd.to_string(
        index=False
    )
)


# ============================================================
# PHASE × YEAR
# ============================================================

print()
print("=" * 120)
print("CFG_0755 BY MLS SEASON PHASE")
print("=" * 120)


phase_rows = []


for year in sorted(
    df["test_year"]
    .unique()
):

    for phase in [
        "EARLY",
        "MID",
        "LATE",
    ]:

        z = df[
            (df["test_year"] == year)
            &
            (
                df[
                    "season_phase"
                ]
                ==
                phase
            )
        ]


        if len(z) < 20:
            continue


        cfg = evaluate(
            z["btts_yes"],
            z["champion_yes"],
        )

        market = evaluate(
            z["btts_yes"],
            z["market_yes"],
        )


        phase_rows.append(
            {
                "year":
                    int(year),

                "phase":
                    phase,

                "games":
                    len(z),

                "actual_rate":
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

                "cfg_calibration_error":
                    cfg[
                        "calibration_error"
                    ],

                "market_calibration_error":
                    market[
                        "calibration_error"
                    ],

                "cfg_brier":
                    cfg["brier"],

                "market_brier":
                    market["brier"],

                "cfg_log_loss":
                    cfg[
                        "log_loss"
                    ],

                "market_log_loss":
                    market[
                        "log_loss"
                    ],
            }
        )


phase_df = pd.DataFrame(
    phase_rows
)


pd_show = phase_df.copy()


for c in [
    "actual_rate",
    "cfg_pred",
    "market_pred",
    "cfg_calibration_error",
    "market_calibration_error",
]:

    pd_show[c] = pd_show[c].map(
        lambda x: f"{x:+.2%}"
    )


for c in [
    "cfg_brier",
    "market_brier",
    "cfg_log_loss",
    "market_log_loss",
]:

    pd_show[c] = pd_show[c].map(
        lambda x: f"{x:.5f}"
    )


print()
print(
    pd_show.to_string(
        index=False
    )
)


# ============================================================
# TEAM EXPERIENCE BANDS
# ============================================================

print()
print("=" * 120)
print("CFG_0755 BY TEAM SEASON EXPERIENCE")
print("=" * 120)


experience_rows = []


for year in sorted(
    df["test_year"]
    .unique()
):

    for band in [
        "1-5",
        "6-10",
        "11-15",
        "16-20",
        "21+",
    ]:

        z = df[
            (df["test_year"] == year)
            &
            (
                df[
                    "team_experience_band"
                ]
                ==
                band
            )
        ]


        if len(z) < 15:
            continue


        cfg = evaluate(
            z["btts_yes"],
            z["champion_yes"],
        )

        market = evaluate(
            z["btts_yes"],
            z["market_yes"],
        )


        experience_rows.append(
            {
                "year":
                    int(year),

                "experience_band":
                    band,

                "games":
                    len(z),

                "actual_rate":
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

                "cfg_error":
                    cfg[
                        "calibration_error"
                    ],

                "market_error":
                    market[
                        "calibration_error"
                    ],

                "cfg_brier":
                    cfg[
                        "brier"
                    ],

                "market_brier":
                    market[
                        "brier"
                    ],
            }
        )


experience_df = pd.DataFrame(
    experience_rows
)


ed = experience_df.copy()


for c in [
    "actual_rate",
    "cfg_pred",
    "market_pred",
    "cfg_error",
    "market_error",
]:

    ed[c] = ed[c].map(
        lambda x: f"{x:+.2%}"
    )


for c in [
    "cfg_brier",
    "market_brier",
]:

    ed[c] = ed[c].map(
        lambda x: f"{x:.5f}"
    )


print()
print(
    ed.to_string(
        index=False
    )
)


# ============================================================
# MODEL-MARKET DISAGREEMENT BANDS
# ============================================================

df["disagreement"] = (
    df["champion_yes"]
    -
    df["market_yes"]
)


df["disagreement_band"] = pd.cut(
    df["disagreement"],
    bins=[
        -1.0,
        -0.04,
        -0.02,
        0.00,
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
        1.00,
    ],
    labels=[
        "<-4%",
        "-4 to -2%",
        "-2 to 0%",
        "0 to 2%",
        "2 to 4%",
        "4 to 6%",
        "6 to 8%",
        "8 to 10%",
        "10 to 12%",
        ">12%",
    ],
    include_lowest=True,
    right=False,
)


print()
print("=" * 120)
print("MODEL-MARKET DISAGREEMENT BY YEAR")
print("=" * 120)


disagreement_rows = []


for year in sorted(
    df["test_year"]
    .unique()
):

    z_year = df[
        df["test_year"]
        ==
        year
    ]


    for band in (
        df[
            "disagreement_band"
        ]
        .cat
        .categories
    ):

        z = z_year[
            z_year[
                "disagreement_band"
            ]
            ==
            band
        ]


        if len(z) < 10:
            continue


        disagreement_rows.append(
            {
                "year":
                    int(year),

                "band":
                    str(band),

                "games":
                    len(z),

                "avg_model":
                    z[
                        "champion_yes"
                    ].mean(),

                "avg_market":
                    z[
                        "market_yes"
                    ].mean(),

                "avg_disagreement":
                    z[
                        "disagreement"
                    ].mean(),

                "actual_rate":
                    z[
                        "btts_yes"
                    ].mean(),

                "model_error":
                    z[
                        "btts_yes"
                    ].mean()
                    -
                    z[
                        "champion_yes"
                    ].mean(),

                "market_error":
                    z[
                        "btts_yes"
                    ].mean()
                    -
                    z[
                        "market_yes"
                    ].mean(),
            }
        )


disagreement_df = pd.DataFrame(
    disagreement_rows
)


dd = disagreement_df.copy()


for c in [
    "avg_model",
    "avg_market",
    "avg_disagreement",
    "actual_rate",
    "model_error",
    "market_error",
]:

    dd[c] = dd[c].map(
        lambda x: f"{x:+.2%}"
    )


print()
print(
    dd.to_string(
        index=False
    )
)


# ============================================================
# SPECIAL: POSITIVE 6%+ DISAGREEMENT
# ============================================================

print()
print("=" * 120)
print("MLS GAMES WHERE CFG_0755 EDGE >= 6%")
print("=" * 120)


high_edge_rows = []


for year in sorted(
    df["test_year"]
    .unique()
):

    z = df[
        (df["test_year"] == year)
        &
        (
            df["champion_edge_yes"]
            >=
            0.06
        )
    ]


    if len(z) == 0:
        continue


    cfg = evaluate(
        z["btts_yes"],
        z["champion_yes"],
    )

    market = evaluate(
        z["btts_yes"],
        z["market_yes"],
    )


    high_edge_rows.append(
        {
            "year":
                int(year),

            "games":
                len(z),

            "actual_rate":
                z[
                    "btts_yes"
                ].mean(),

            "avg_model":
                z[
                    "champion_yes"
                ].mean(),

            "avg_market":
                z[
                    "market_yes"
                ].mean(),

            "avg_edge":
                z[
                    "champion_edge_yes"
                ].mean(),

            "cfg_calibration_error":
                cfg[
                    "calibration_error"
                ],

            "market_calibration_error":
                market[
                    "calibration_error"
                ],

            "cfg_brier":
                cfg["brier"],

            "market_brier":
                market["brier"],
        }
    )


high_edge_df = pd.DataFrame(
    high_edge_rows
)


hd = high_edge_df.copy()


for c in [
    "actual_rate",
    "avg_model",
    "avg_market",
    "avg_edge",
    "cfg_calibration_error",
    "market_calibration_error",
]:

    hd[c] = hd[c].map(
        lambda x: f"{x:+.2%}"
    )


for c in [
    "cfg_brier",
    "market_brier",
]:

    hd[c] = hd[c].map(
        lambda x: f"{x:.5f}"
    )


print()
print(
    hd.to_string(
        index=False
    )
)


# ============================================================
# 2025 MONTHLY
# ============================================================

print()
print("=" * 120)
print("MLS 2025 MONTHLY")
print("=" * 120)


y2025 = df[
    df["test_year"]
    ==
    2025
].copy()


month_rows = []


for month in sorted(
    y2025["month"]
    .dropna()
    .unique()
):

    z = y2025[
        y2025["month"]
        ==
        month
    ]


    cfg = evaluate(
        z["btts_yes"],
        z["champion_yes"],
    )

    market = evaluate(
        z["btts_yes"],
        z["market_yes"],
    )


    month_rows.append(
        {
            "month":
                int(month),

            "games":
                len(z),

            "actual_rate":
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

            "cfg_error":
                cfg[
                    "calibration_error"
                ],

            "market_error":
                market[
                    "calibration_error"
                ],

            "cfg_brier":
                cfg["brier"],

            "market_brier":
                market["brier"],

            "high_edge_games":
                int(
                    (
                        z[
                            "champion_edge_yes"
                        ]
                        >=
                        0.06
                    ).sum()
                ),

            "high_edge_actual":
                (
                    z.loc[
                        z[
                            "champion_edge_yes"
                        ]
                        >=
                        0.06,
                        "btts_yes",
                    ]
                    .mean()
                ),
        }
    )


month_df = pd.DataFrame(
    month_rows
)


md = month_df.copy()


for c in [
    "actual_rate",
    "cfg_pred",
    "market_pred",
    "cfg_error",
    "market_error",
    "high_edge_actual",
]:

    md[c] = md[c].map(
        lambda x:
            ""
            if pd.isna(x)
            else
            f"{x:+.2%}"
    )


for c in [
    "cfg_brier",
    "market_brier",
]:

    md[c] = md[c].map(
        lambda x: f"{x:.5f}"
    )


print()
print(
    md.to_string(
        index=False
    )
)


# ============================================================
# CALIBRATION BINS
# ============================================================

df["probability_band"] = pd.cut(
    df["champion_yes"],
    bins=np.arange(
        0.35,
        0.801,
        0.05,
    ),
    include_lowest=True,
)


cal_rows = []


for year in sorted(
    df["test_year"]
    .unique()
):

    z_year = df[
        df["test_year"]
        ==
        year
    ]


    for band in (
        df[
            "probability_band"
        ]
        .cat
        .categories
    ):

        z = z_year[
            z_year[
                "probability_band"
            ]
            ==
            band
        ]


        if len(z) < 15:
            continue


        cal_rows.append(
            {
                "year":
                    int(year),

                "band":
                    str(band),

                "games":
                    len(z),

                "avg_pred":
                    z[
                        "champion_yes"
                    ].mean(),

                "actual_rate":
                    z[
                        "btts_yes"
                    ].mean(),

                "error":
                    z[
                        "btts_yes"
                    ].mean()
                    -
                    z[
                        "champion_yes"
                    ].mean(),
            }
        )


cal_df = pd.DataFrame(
    cal_rows
)


print()
print("=" * 120)
print("CFG_0755 CALIBRATION BANDS BY YEAR")
print("=" * 120)


cd = cal_df.copy()


for c in [
    "avg_pred",
    "actual_rate",
    "error",
]:

    cd[c] = cd[c].map(
        lambda x: f"{x:+.2%}"
    )


print()
print(
    cd.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUT_DIR
    / "01_mls_games_enriched.csv",
    index=False,
)

year_summary.to_csv(
    OUT_DIR
    / "02_model_quality_by_year.csv",
    index=False,
)

phase_df.to_csv(
    OUT_DIR
    / "03_phase_by_year.csv",
    index=False,
)

experience_df.to_csv(
    OUT_DIR
    / "04_team_experience_by_year.csv",
    index=False,
)

disagreement_df.to_csv(
    OUT_DIR
    / "05_disagreement_by_year.csv",
    index=False,
)

high_edge_df.to_csv(
    OUT_DIR
    / "06_high_edge_by_year.csv",
    index=False,
)

month_df.to_csv(
    OUT_DIR
    / "07_2025_monthly.csv",
    index=False,
)

cal_df.to_csv(
    OUT_DIR
    / "08_calibration_bands.csv",
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
