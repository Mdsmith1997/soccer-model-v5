from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

FEATURE_FILE = DATA / "btts_feature_store_v1.csv"

MLS_MARKET_FILE = DATA / "mls_v5_btts_market_matched.csv"

ELI_MARKET_FILE = DATA / "eliteserien_btts_market_oos.csv"

OUT_PRED = DATA / "btts_cfg0755_oos_2021_2025.csv"

OUT_MARKET = DATA / "btts_cfg0755_market_matched.csv"

OUT_THRESH = DATA / "btts_cfg0755_market_thresholds.csv"

OUT_SEASON = DATA / "btts_cfg0755_market_by_season.csv"


TEST_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
]

MIN_TRAIN = 5000

LAMBDA_CUT = 1.1

C_VALUE = 0.01

MODEL_BLEND = 0.75
POISSON_BLEND = 0.25


EDGE_THRESHOLDS = np.round(
    np.arange(
        0.00,
        0.151,
        0.01,
    ),
    2,
)

EV_THRESHOLDS = np.round(
    np.arange(
        0.00,
        0.201,
        0.02,
    ),
    2,
)


# ============================================================
# HELPERS
# ============================================================

def normalize_team(value):

    value = str(value)

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )

    value = value.lower().strip()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def probability_metrics(y, p):

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

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
        "games": len(y),

        "brier":
            brier_score_loss(
                y,
                p,
            ),

        "log_loss":
            log_loss(
                y,
                p,
            ),

        "auc":
            roc_auc_score(
                y,
                p,
            )
            if len(np.unique(y)) > 1
            else np.nan,

        "avg_pred":
            p.mean(),

        "actual_rate":
            y.mean(),
    }


# ============================================================
# LOAD FEATURE STORE
# ============================================================

print()
print("=" * 120)
print("CFG_0755 BTTS REAL-MARKET VALIDATION")
print("=" * 120)

df = pd.read_csv(
    FEATURE_FILE,
    low_memory=False,
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
).dt.normalize()

df["btts_yes"] = pd.to_numeric(
    df["btts_yes"],
    errors="coerce",
)

df["poisson_btts"] = pd.to_numeric(
    df["poisson_btts"],
    errors="coerce",
)

df["lambda_min"] = pd.to_numeric(
    df["lambda_min"],
    errors="coerce",
)


df = df[
    df["date"].notna()
    &
    df["btts_yes"].notna()
    &
    df["poisson_btts"].notna()
].copy()

df["btts_yes"] = df["btts_yes"].astype(int)


# ============================================================
# REBUILD CFG_0755 FEATURES
# ============================================================

p = df["poisson_btts"].clip(
    0.001,
    0.999,
)

df["poisson_logit"] = np.log(
    p / (1 - p)
)


df["lambda_min_above_1_1"] = np.maximum(
    df["lambda_min"] - 1.1,
    0.0,
)

df["lambda_min_below_1_1"] = np.maximum(
    1.1 - df["lambda_min"],
    0.0,
)


ENVIRONMENT = [
    "poisson_logit",

    "lambda_min",
    "lambda_total",
    "lambda_gap",
    "lambda_balance_ratio",
    "weaker_team_score_probability",

    "xg_matchup_overall_min",
    "xg_matchup_overall_balance",

    "shot_matchup_overall_min",
    "shot_matchup_overall_balance",

    "goal_matchup_overall_min",
    "goal_matchup_overall_balance",

    "league_goal_environment",
    "league_xg_environment",

    "xg_attack_balance",
    "goal_attack_balance",

    "minimum_team_history",

    "lambda_min_above_1_1",
    "lambda_min_below_1_1",
]


FEATURES = [
    c for c in ENVIRONMENT
    if c in df.columns
]


print()
print("CFG_0755")
print("Family:       lambda_hinge")
print("Lambda cut:   1.1")
print("C:            0.01")
print("Model blend:  75%")
print("Poisson:      25%")
print("Features:", len(FEATURES))


# ============================================================
# MODEL
# ============================================================

numeric_pipe = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
        (
            "scale",
            StandardScaler(),
        ),
    ]
)


league_pipe = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
        ),
    ]
)


prep = ColumnTransformer(
    [
        (
            "num",
            numeric_pipe,
            FEATURES,
        ),
        (
            "league",
            league_pipe,
            ["league"],
        ),
    ]
)


model = Pipeline(
    [
        (
            "prep",
            prep,
        ),
        (
            "model",
            LogisticRegression(
                C=C_VALUE,
                max_iter=3000,
                solver="liblinear",
            ),
        ),
    ]
)


MODEL_COLS = (
    FEATURES
    +
    ["league"]
)


# ============================================================
# EXPANDING-WINDOW OOS PREDICTIONS
# ============================================================

frames = []


print()
print("=" * 120)
print("REBUILDING CFG_0755 OOS PROBABILITIES")
print("=" * 120)


for year in TEST_YEARS:

    start = pd.Timestamp(
        year,
        1,
        1,
    )

    end = pd.Timestamp(
        year + 1,
        1,
        1,
    )

    train = df[
        df["date"] < start
    ].copy()

    test = df[
        (df["date"] >= start)
        &
        (df["date"] < end)
    ].copy()


    if len(train) < MIN_TRAIN:

        print(
            f"{year}: skipped "
            f"train={len(train):,}"
        )

        continue


    print(
        f"{year}: "
        f"train={len(train):,} "
        f"test={len(test):,}"
    )


    model.fit(
        train[MODEL_COLS],
        train["btts_yes"],
    )


    p_model = (
        model.predict_proba(
            test[MODEL_COLS]
        )[:, 1]
    )


    p_final = (
        MODEL_BLEND
        *
        p_model
        +
        POISSON_BLEND
        *
        test["poisson_btts"].to_numpy()
    )


    out = test[
        [
            "date",
            "league",
            "home_team",
            "away_team",
            "season",
            "btts_yes",
            "home_goals",
            "away_goals",
            "poisson_btts",
            "home_lambda",
            "away_lambda",
            "lambda_min",
            "lambda_total",
            "lambda_gap",
        ]
    ].copy()


    out["test_year"] = year

    out["cfg0755_model_probability"] = (
        p_model
    )

    out["cfg0755_probability"] = (
        p_final
    )


    out["home_norm"] = (
        out["home_team"]
        .map(normalize_team)
    )

    out["away_norm"] = (
        out["away_team"]
        .map(normalize_team)
    )


    frames.append(out)


pred = pd.concat(
    frames,
    ignore_index=True,
)


pred.to_csv(
    OUT_PRED,
    index=False,
)


print()
print("Generated OOS predictions:", len(pred))


# ============================================================
# LOAD MLS MARKET
# ============================================================

mls = pd.read_csv(
    MLS_MARKET_FILE,
    low_memory=False,
)


mls["date"] = pd.to_datetime(
    mls["date"],
    errors="coerce",
).dt.normalize()


mls["home_norm"] = (
    mls["home_team"]
    .map(normalize_team)
)

mls["away_norm"] = (
    mls["away_team"]
    .map(normalize_team)
)


mls_market = pd.DataFrame(
    {
        "league":
            "MLS",

        "date":
            mls["date"],

        "home_norm":
            mls["home_norm"],

        "away_norm":
            mls["away_norm"],

        "market_year":
            pd.to_numeric(
                mls["source_year"],
                errors="coerce",
            ),

        "odds_yes":
            pd.to_numeric(
                mls["odds_btts_yes"],
                errors="coerce",
            ),

        "odds_no":
            pd.to_numeric(
                mls["odds_btts_no"],
                errors="coerce",
            ),

        "market_yes":
            pd.to_numeric(
                mls[
                    "market_btts_yes_novig"
                ],
                errors="coerce",
            ),

        "market_no":
            pd.to_numeric(
                mls[
                    "market_btts_no_novig"
                ],
                errors="coerce",
            ),
    }
)


# ============================================================
# LOAD ELITESERIEN MARKET
# ============================================================

eli = pd.read_csv(
    ELI_MARKET_FILE,
    low_memory=False,
)


eli["date"] = pd.to_datetime(
    eli["date"],
    errors="coerce",
).dt.normalize()


eli["home_norm"] = (
    eli["home_team"]
    .map(normalize_team)
)

eli["away_norm"] = (
    eli["away_team"]
    .map(normalize_team)
)


eli_market = pd.DataFrame(
    {
        "league":
            "Eliteserien",

        "date":
            eli["date"],

        "home_norm":
            eli["home_norm"],

        "away_norm":
            eli["away_norm"],

        "market_year":
            pd.to_numeric(
                eli["season"],
                errors="coerce",
            ),

        "odds_yes":
            pd.to_numeric(
                eli["odds_btts_yes"],
                errors="coerce",
            ),

        "odds_no":
            pd.to_numeric(
                eli["odds_btts_no"],
                errors="coerce",
            ),

        "market_yes":
            pd.to_numeric(
                eli["market_yes_nv"],
                errors="coerce",
            ),

        "market_no":
            pd.to_numeric(
                eli["market_no_nv"],
                errors="coerce",
            ),
    }
)


market = pd.concat(
    [
        mls_market,
        eli_market,
    ],
    ignore_index=True,
)


# ============================================================
# JOIN
# ============================================================

target_pred = pred[
    pred["league"].isin(
        [
            "MLS",
            "Eliteserien",
        ]
    )
].copy()


matched = target_pred.merge(
    market,
    on=[
        "league",
        "date",
        "home_norm",
        "away_norm",
    ],
    how="inner",
)


matched = matched[
    (matched["odds_yes"] > 1)
    &
    (matched["odds_no"] > 1)
    &
    matched["market_yes"].notna()
    &
    matched["market_no"].notna()
].copy()


print()
print("=" * 120)
print("MARKET MATCHING")
print("=" * 120)

print()
print(
    "Candidate MLS/Eliteserien OOS rows:",
    len(target_pred),
)

print(
    "Valid market matched rows:",
    len(matched),
)

print()
print(
    matched.groupby(
        [
            "league",
            "test_year",
        ]
    )
    .size()
    .to_string()
)


# ============================================================
# MODEL / MARKET VARIABLES
# ============================================================

matched["champion_yes"] = (
    matched[
        "cfg0755_probability"
    ]
)

matched["champion_no"] = (
    1.0
    -
    matched["champion_yes"]
)


matched["poisson_yes"] = (
    matched["poisson_btts"]
)

matched["poisson_no"] = (
    1.0
    -
    matched["poisson_yes"]
)


# Probability edges vs no-vig market

matched["champion_edge_yes"] = (
    matched["champion_yes"]
    -
    matched["market_yes"]
)

matched["champion_edge_no"] = (
    matched["champion_no"]
    -
    matched["market_no"]
)

matched["poisson_edge_yes"] = (
    matched["poisson_yes"]
    -
    matched["market_yes"]
)

matched["poisson_edge_no"] = (
    matched["poisson_no"]
    -
    matched["market_no"]
)


# True expected return at available price

matched["champion_ev_yes"] = (
    matched["champion_yes"]
    *
    matched["odds_yes"]
    -
    1.0
)

matched["champion_ev_no"] = (
    matched["champion_no"]
    *
    matched["odds_no"]
    -
    1.0
)

matched["poisson_ev_yes"] = (
    matched["poisson_yes"]
    *
    matched["odds_yes"]
    -
    1.0
)

matched["poisson_ev_no"] = (
    matched["poisson_no"]
    *
    matched["odds_no"]
    -
    1.0
)


matched.to_csv(
    OUT_MARKET,
    index=False,
)


# ============================================================
# PROBABILITY QUALITY VS MARKET
# ============================================================

print()
print("=" * 120)
print("PROBABILITY QUALITY ON MARKET-MATCHED GAMES")
print("=" * 120)


quality_rows = []


for league in [
    "ALL",
    "MLS",
    "Eliteserien",
]:

    if league == "ALL":

        z = matched

    else:

        z = matched[
            matched["league"]
            == league
        ]


    for name, col in {
        "POISSON":
            "poisson_yes",

        "CFG_0755":
            "champion_yes",

        "MARKET":
            "market_yes",
    }.items():

        r = probability_metrics(
            z["btts_yes"],
            z[col],
        )

        quality_rows.append(
            {
                "league":
                    league,

                "model":
                    name,

                **r,
            }
        )


quality = pd.DataFrame(
    quality_rows
)


qd = quality.copy()

qd["brier"] = qd["brier"].map(
    lambda x: f"{x:.5f}"
)

qd["log_loss"] = qd[
    "log_loss"
].map(
    lambda x: f"{x:.5f}"
)

qd["auc"] = qd["auc"].map(
    lambda x: f"{x:.4f}"
)

qd["avg_pred"] = qd[
    "avg_pred"
].map(
    lambda x: f"{x:.2%}"
)

qd["actual_rate"] = qd[
    "actual_rate"
].map(
    lambda x: f"{x:.2%}"
)


print()
print(
    qd.to_string(
        index=False
    )
)


# ============================================================
# BET EVALUATION
# ============================================================

def evaluate_bets(
    data,
    model_name,
    side,
    selector,
    threshold,
):

    prefix = (
        "champion"
        if model_name == "CFG_0755"
        else "poisson"
    )


    if side == "YES":

        odds_col = "odds_yes"

        won = (
            data["btts_yes"]
            == 1
        )

    else:

        odds_col = "odds_no"

        won = (
            data["btts_yes"]
            == 0
        )


    if selector == "EDGE":

        signal_col = (
            f"{prefix}_edge_"
            f"{side.lower()}"
        )

    else:

        signal_col = (
            f"{prefix}_ev_"
            f"{side.lower()}"
        )


    bets = data[
        data[signal_col]
        >= threshold
    ].copy()


    if len(bets) == 0:

        return None


    if side == "YES":

        bets["won"] = (
            bets["btts_yes"]
            == 1
        )

    else:

        bets["won"] = (
            bets["btts_yes"]
            == 0
        )


    bets["odds"] = (
        bets[odds_col]
    )


    bets["profit"] = np.where(
        bets["won"],
        bets["odds"] - 1.0,
        -1.0,
    )


    return {
        "model":
            model_name,

        "selector":
            selector,

        "side":
            side,

        "threshold":
            threshold,

        "bets":
            len(bets),

        "wins":
            int(
                bets["won"].sum()
            ),

        "win_rate":
            bets["won"].mean(),

        "avg_odds":
            bets["odds"].mean(),

        "avg_signal":
            bets[signal_col].mean(),

        "profit_units":
            bets["profit"].sum(),

        "roi":
            bets["profit"].mean(),
    }


# ============================================================
# OVERALL THRESHOLD SWEEP
# ============================================================

threshold_rows = []


for league in [
    "ALL",
    "MLS",
    "Eliteserien",
]:

    if league == "ALL":

        z = matched

    else:

        z = matched[
            matched["league"]
            == league
        ]


    for model_name in [
        "POISSON",
        "CFG_0755",
    ]:

        for side in [
            "YES",
            "NO",
        ]:

            for threshold in (
                EDGE_THRESHOLDS
            ):

                r = evaluate_bets(
                    z,
                    model_name,
                    side,
                    "EDGE",
                    threshold,
                )

                if r:

                    r["league"] = (
                        league
                    )

                    threshold_rows.append(
                        r
                    )


            for threshold in (
                EV_THRESHOLDS
            ):

                r = evaluate_bets(
                    z,
                    model_name,
                    side,
                    "EV",
                    threshold,
                )

                if r:

                    r["league"] = (
                        league
                    )

                    threshold_rows.append(
                        r
                    )


threshold_df = pd.DataFrame(
    threshold_rows
)


threshold_df.to_csv(
    OUT_THRESH,
    index=False,
)


# ============================================================
# PRINT USEFUL SAMPLE THRESHOLDS
# ============================================================

print()
print("=" * 120)
print("PROBABILITY EDGE — ALL MATCHED GAMES")
print("=" * 120)


edge_show = threshold_df[
    (threshold_df["league"] == "ALL")
    &
    (threshold_df["selector"] == "EDGE")
    &
    threshold_df["threshold"].isin(
        [
            0.00,
            0.02,
            0.04,
            0.06,
            0.08,
            0.10,
            0.12,
        ]
    )
].copy()


ed = edge_show[
    [
        "model",
        "side",
        "threshold",
        "bets",
        "wins",
        "win_rate",
        "avg_odds",
        "avg_signal",
        "profit_units",
        "roi",
    ]
].copy()


ed["threshold"] = ed[
    "threshold"
].map(
    lambda x: f"{x:.0%}"
)

ed["win_rate"] = ed[
    "win_rate"
].map(
    lambda x: f"{x:.2%}"
)

ed["avg_odds"] = ed[
    "avg_odds"
].map(
    lambda x: f"{x:.3f}"
)

ed["avg_signal"] = ed[
    "avg_signal"
].map(
    lambda x: f"{x:.2%}"
)

ed["profit_units"] = ed[
    "profit_units"
].map(
    lambda x: f"{x:+.2f}"
)

ed["roi"] = ed[
    "roi"
].map(
    lambda x: f"{x:+.2%}"
)


print()
print(
    ed.to_string(
        index=False
    )
)


print()
print("=" * 120)
print("EXPECTED VALUE THRESHOLDS — ALL MATCHED GAMES")
print("=" * 120)


ev_show = threshold_df[
    (threshold_df["league"] == "ALL")
    &
    (threshold_df["selector"] == "EV")
    &
    threshold_df["threshold"].isin(
        [
            0.00,
            0.02,
            0.04,
            0.06,
            0.08,
            0.10,
            0.12,
            0.14,
            0.16,
            0.18,
            0.20,
        ]
    )
].copy()


evd = ev_show[
    [
        "model",
        "side",
        "threshold",
        "bets",
        "wins",
        "win_rate",
        "avg_odds",
        "avg_signal",
        "profit_units",
        "roi",
    ]
].copy()


evd["threshold"] = evd[
    "threshold"
].map(
    lambda x: f"{x:.0%}"
)

evd["win_rate"] = evd[
    "win_rate"
].map(
    lambda x: f"{x:.2%}"
)

evd["avg_odds"] = evd[
    "avg_odds"
].map(
    lambda x: f"{x:.3f}"
)

evd["avg_signal"] = evd[
    "avg_signal"
].map(
    lambda x: f"{x:.2%}"
)

evd["profit_units"] = evd[
    "profit_units"
].map(
    lambda x: f"{x:+.2f}"
)

evd["roi"] = evd[
    "roi"
].map(
    lambda x: f"{x:+.2%}"
)


print()
print(
    evd.to_string(
        index=False
    )
)


# ============================================================
# SEASON ROBUSTNESS
# ============================================================

season_rows = []


for league in [
    "MLS",
    "Eliteserien",
]:

    z_league = matched[
        matched["league"]
        == league
    ]


    for year in sorted(
        z_league[
            "test_year"
        ].unique()
    ):

        z = z_league[
            z_league["test_year"]
            == year
        ]


        for model_name in [
            "POISSON",
            "CFG_0755",
        ]:

            for selector in [
                "EDGE",
                "EV",
            ]:

                thresholds = (
                    [
                        0.02,
                        0.04,
                        0.06,
                        0.08,
                        0.10,
                    ]
                    if selector
                    == "EDGE"
                    else
                    [
                        0.00,
                        0.04,
                        0.08,
                        0.12,
                    ]
                )


                for side in [
                    "YES",
                    "NO",
                ]:

                    for threshold in thresholds:

                        r = evaluate_bets(
                            z,
                            model_name,
                            side,
                            selector,
                            threshold,
                        )

                        if r:

                            r["league"] = (
                                league
                            )

                            r["year"] = int(
                                year
                            )

                            season_rows.append(
                                r
                            )


season_df = pd.DataFrame(
    season_rows
)


season_df.to_csv(
    OUT_SEASON,
    index=False,
)


# ============================================================
# ROBUSTNESS SUMMARY
# ============================================================

print()
print("=" * 120)
print("CFG_0755 SEASON ROBUSTNESS")
print("=" * 120)


robust = (
    season_df[
        season_df["model"]
        == "CFG_0755"
    ]
    .groupby(
        [
            "league",
            "selector",
            "side",
            "threshold",
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

        positive_seasons=(
            "roi",
            lambda x:
                int(
                    (x > 0)
                    .sum()
                ),
        ),

        mean_season_roi=(
            "roi",
            "mean",
        ),

        median_season_roi=(
            "roi",
            "median",
        ),

        worst_season_roi=(
            "roi",
            "min",
        ),

        best_season_roi=(
            "roi",
            "max",
        ),
    )
    .reset_index()
)


rd = robust.copy()


rd["threshold"] = rd[
    "threshold"
].map(
    lambda x: f"{x:.0%}"
)


for c in [
    "mean_season_roi",
    "median_season_roi",
    "worst_season_roi",
    "best_season_roi",
]:

    rd[c] = rd[c].map(
        lambda x: f"{x:+.2%}"
    )


print()
print(
    rd.to_string(
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

print()
print(OUT_PRED)
print(OUT_MARKET)
print(OUT_THRESH)
print(OUT_SEASON)

print()
print("DONE")
