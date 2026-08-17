from pathlib import Path
import math
import re
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRED_FILE = ROOT / "data/processed/footystats_multileague_v5_predictions.csv"
RAW_DIR = ROOT / "data/raw"

LEAGUE = "Eredivisie"
RAW_CODE = "N1"

SEASONS = [
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
    "2526",
]

THRESHOLDS = [
    0.08,
    0.09,
    0.10,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
]

FROZEN_THRESHOLD = 0.11


def normalize_team(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = " ".join(
        value.split()
    )

    aliases = {
        "psv eindhoven": "psv",
        "ajax amsterdam": "ajax",
        "az alkmaar": "az",
        "fc utrecht": "utrecht",
        "feyenoord rotterdam": "feyenoord",
        "sc heerenveen": "heerenveen",
        "fc groningen": "groningen",
        "fc twente": "twente",
        "pec zwolle": "zwolle",
        "rkc waalwijk": "waalwijk",
        "fortuna sittard": "sittard",
        "sparta rotterdam": "sparta rotterdam",
        "heracles almelo": "heracles",
        "nec nijmegen": "nijmegen",
        "fc volendam": "volendam",
        "excelsior rotterdam": "excelsior",
    }

    return aliases.get(
        value,
        value,
    )


def p_under_2_5(home_lambda, away_lambda):
    lam = float(home_lambda) + float(away_lambda)

    return math.exp(-lam) * (
        1.0
        + lam
        + lam ** 2 / 2.0
    )


def load_predictions():
    df = pd.read_csv(
        PRED_FILE,
        low_memory=False,
    )

    df = df[
        df["league"].astype(str).eq(LEAGUE)
    ].copy()

    df["season"] = (
        df["season"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["season"].isin(SEASONS)
    ].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.date

    for c in [
        "home_lambda",
        "away_lambda",
        "home_goals",
        "away_goals",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "date",
            "home_lambda",
            "away_lambda",
            "home_goals",
            "away_goals",
        ]
    ).copy()

    df["home_key"] = df["home_team"].map(
        normalize_team
    )

    df["away_key"] = df["away_team"].map(
        normalize_team
    )

    return df


def load_market():
    rows = []

    for season in SEASONS:
        path = RAW_DIR / f"{season}_{RAW_CODE}.csv"

        if not path.exists():
            print(
                f"WARNING: missing {path.name}"
            )
            continue

        x = pd.read_csv(
            path,
            low_memory=False,
        ).copy()

        x["date"] = pd.to_datetime(
            x["Date"],
            dayfirst=True,
            errors="coerce",
        ).dt.date

        x["home_key"] = x["HomeTeam"].map(
            normalize_team
        )

        x["away_key"] = x["AwayTeam"].map(
            normalize_team
        )

        x["over_odds"] = pd.to_numeric(
            x["Avg>2.5"],
            errors="coerce",
        )

        x["under_odds"] = pd.to_numeric(
            x["Avg<2.5"],
            errors="coerce",
        )

        x["season"] = season

        rows.append(
            x[
                [
                    "season",
                    "date",
                    "home_key",
                    "away_key",
                    "over_odds",
                    "under_odds",
                ]
            ]
        )

    if not rows:
        raise RuntimeError(
            "No Eredivisie market files found."
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def performance(df, threshold):
    x = df[
        df["under_edge"] >= threshold
    ].copy()

    if x.empty:
        return {
            "bets": 0,
            "wins": 0,
            "win_rate": np.nan,
            "avg_odds": np.nan,
            "avg_edge": np.nan,
            "avg_ev": np.nan,
            "profit": 0.0,
            "roi": np.nan,
        }

    x["profit"] = np.where(
        x["under_win"].eq(1),
        x["under_odds"] - 1.0,
        -1.0,
    )

    bets = len(x)
    wins = int(x["under_win"].sum())
    profit = float(x["profit"].sum())

    return {
        "bets": bets,
        "wins": wins,
        "win_rate": wins / bets,
        "avg_odds": x["under_odds"].mean(),
        "avg_edge": x["under_edge"].mean(),
        "avg_ev": x["under_ev"].mean(),
        "profit": profit,
        "roi": profit / bets,
    }


def print_result(label, p):
    if not p["bets"]:
        print(f"{label:<10}  no bets")
        return

    print(
        f"{label:<10} "
        f"bets={p['bets']:4d}  "
        f"wins={p['wins']:4d}  "
        f"WR={p['win_rate']:7.2%}  "
        f"odds={p['avg_odds']:.3f}  "
        f"edge={p['avg_edge']:+7.2%}  "
        f"EV={p['avg_ev']:+7.2%}  "
        f"profit={p['profit']:+8.2f}u  "
        f"ROI={p['roi']:+7.2%}"
    )


def main():
    print()
    print("=" * 120)
    print("EREDIVISIE V5 TOTALS — TRUE OOS TEST")
    print("=" * 120)

    print()
    print(
        "PRE-SELECTED RULE: "
        "RAW UNDER 2.5 EDGE >= 11%"
    )

    pred = load_predictions()
    market = load_market()

    merged = pred.merge(
        market,
        on=[
            "season",
            "date",
            "home_key",
            "away_key",
        ],
        how="inner",
    )

    merged = merged[
        merged["over_odds"].gt(1)
        & merged["under_odds"].gt(1)
    ].copy()

    print()
    print(f"Prediction rows: {len(pred):,}")
    print(f"Market rows:     {len(market):,}")
    print(f"Matched rows:    {len(merged):,}")

    if len(market):
        print(
            f"Match rate:      "
            f"{len(merged) / len(market):.2%}"
        )

    merged["model_p_under"] = [
        p_under_2_5(h, a)
        for h, a in zip(
            merged["home_lambda"],
            merged["away_lambda"],
        )
    ]

    raw_over = 1.0 / merged["over_odds"]
    raw_under = 1.0 / merged["under_odds"]

    vig_sum = raw_over + raw_under

    merged["market_p_under"] = (
        raw_under / vig_sum
    )

    merged["under_edge"] = (
        merged["model_p_under"]
        - merged["market_p_under"]
    )

    merged["under_ev"] = (
        merged["model_p_under"]
        * merged["under_odds"]
        - 1.0
    )

    merged["actual_total"] = (
        merged["home_goals"]
        + merged["away_goals"]
    )

    merged["under_win"] = (
        merged["actual_total"] < 2.5
    ).astype(int)

    print()
    print("=" * 120)
    print("THRESHOLD ROBUSTNESS")
    print("=" * 120)
    print()

    for threshold in THRESHOLDS:
        print_result(
            f">={threshold:.0%}",
            performance(
                merged,
                threshold,
            ),
        )

    print()
    print("=" * 120)
    print("FROZEN 11% RULE — BY SEASON")
    print("=" * 120)
    print()

    for season in SEASONS:
        x = merged[
            merged["season"].eq(season)
        ]

        p = performance(
            x,
            FROZEN_THRESHOLD,
        )

        if p["bets"]:
            print_result(
                season,
                p,
            )

    frozen = performance(
        merged,
        FROZEN_THRESHOLD,
    )

    print()
    print("=" * 120)
    print("EREDIVISIE — FROZEN 11% RESULT")
    print("=" * 120)
    print()

    print_result(
        "UNDER",
        frozen,
    )


if __name__ == "__main__":
    main()
