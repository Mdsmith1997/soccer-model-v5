from pathlib import Path
import math
import re
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRED_FILE = ROOT / "data/processed/footystats_multileague_v5_predictions.csv"
RAW_DIR = ROOT / "data/raw"

LEAGUE_FILES = {
    "Belgian Pro League": "B1",
}

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

        # English EFL — FootyStats -> Football-Data
        "accrington stanley": "accrington",
        "birmingham city": "birmingham",
        "bolton wanderers": "bolton",
        "bradford city": "bradford",
        "bristol rovers": "bristol rvs",
        "burton albion": "burton",
        "cambridge united": "cambridge",
        "cardiff city": "cardiff",
        "carlisle united": "carlisle",
        "charlton athletic": "charlton",
        "cheltenham town": "cheltenham",
        "colchester united": "colchester",
        "coventry city": "coventry",
        "crewe alexandra": "crewe",
        "derby county": "derby",
        "doncaster rovers": "doncaster",
        "exeter city": "exeter",
        "forest green rovers": "forest green",
        "grimsby town": "grimsby",
        "harrogate town": "harrogate",
        "hartlepool united": "hartlepool",
        "huddersfield town": "huddersfield",
        "hull city": "hull",
        "ipswich town": "ipswich",
        "lincoln city": "lincoln",
        "luton town": "luton",
        "macclesfield town": "macclesfield",
        "mansfield town": "mansfield",
        "northampton town": "northampton",
        "oldham athletic": "oldham",
        "oxford united": "oxford",
        "peterborough united": "peterboro",
        "plymouth argyle": "plymouth",
        "rotherham united": "rotherham",
        "salford city": "salford",
        "scunthorpe united": "scunthorpe",
        "sheffield wednesday": "sheffield weds",
        "shrewsbury town": "shrewsbury",
        "southend united": "southend",
        "stockport county": "stockport",
        "sutton united": "sutton",
        "swindon town": "swindon",
        "tranmere rovers": "tranmere",
        "wigan athletic": "wigan",
        "wycombe wanderers": "wycombe",

        # 2. Bundesliga — FootyStats -> Football-Data
        # Belgian Pro League — FootyStats -> Football-Data
        "as eupen": "eupen",
        "beerschot wilrijk": "beerschot va",
        "fcv dender eh": "dender",
        "kaa gent": "gent",
        "krc genk": "genk",
        "kv kortrijk": "kortrijk",
        "kv mechelen": "mechelen",
        "kv oostende": "oostende",
        "kvc westerlo": "westerlo",
        "oh leuven": "oud heverlee leuven",
        "rfc seraing": "seraing",
        "royal antwerp fc": "antwerp",
        "royal excel mouscron": "mouscron",
        "rsc anderlecht": "anderlecht",
        "rwdm": "rwd molenbeek",
        "sint truiden": "st truiden",
        "sporting charleroi": "charleroi",
        "standard liege": "standard",
        "union saint gilloise": "st gilloise",
        "zulte waregem": "waregem",

        "arminia bielefeld": "bielefeld",
        "darmstadt 98": "darmstadt",
        "dynamo dresden": "dresden",
        "eintracht braunschweig": "braunschweig",
        "hamburger sv": "hamburg",
        "hannover 96": "hannover",
        "hertha bsc": "hertha",
        "jahn regensburg": "regensburg",
        "karlsruher sc": "karlsruhe",
        "koln": "fc koln",
        "wehen wiesbaden": "wehen",
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
        df["league"]
        .astype(str)
        .isin(LEAGUE_FILES)
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

    for league, raw_code in LEAGUE_FILES.items():

        for season in SEASONS:

            path = (
                RAW_DIR
                / f"{season}_{raw_code}.csv"
            )

            if not path.exists():
                print(
                    f"WARNING: missing "
                    f"{path.name}"
                )
                continue

            x = pd.read_csv(
                path,
                low_memory=False,
            ).copy()

            required = [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "Avg>2.5",
                "Avg<2.5",
            ]

            missing = [
                c
                for c in required
                if c not in x.columns
            ]

            if missing:
                print(
                    f"WARNING: {path.name} "
                    f"missing {missing}"
                )
                continue

            x["date"] = pd.to_datetime(
                x["Date"],
                dayfirst=True,
                errors="coerce",
            ).dt.date

            x["home_key"] = (
                x["HomeTeam"]
                .map(normalize_team)
            )

            x["away_key"] = (
                x["AwayTeam"]
                .map(normalize_team)
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
            x["league"] = league

            rows.append(
                x[
                    [
                        "season",
                        "league",
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
            "No historical market files found."
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
    print("BELGIAN PRO LEAGUE V5 TOTALS — FROZEN OOS TEST")
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
            "league",
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
    print("FROZEN 11% RULE — BELGIAN PRO LEAGUE")
    print("=" * 120)
    print()

    overall = performance(
        merged,
        FROZEN_THRESHOLD,
    )

    print_result(
        "ALL",
        overall,
    )

    print()
    print("=" * 120)
    print("FROZEN 11% RULE — BY LEAGUE")
    print("=" * 120)
    print()

    for league in LEAGUE_FILES:

        x = merged[
            merged["league"].eq(league)
        ]

        p = performance(
            x,
            FROZEN_THRESHOLD,
        )

        print_result(
            league,
            p,
        )

    print()
    print("=" * 120)
    print("FROZEN 11% RULE — LEAGUE + SEASON")
    print("=" * 120)

    for league in LEAGUE_FILES:

        print()
        print("-" * 120)
        print(league.upper())
        print("-" * 120)
        print()

        league_df = merged[
            merged["league"].eq(league)
        ]

        for season in SEASONS:

            x = league_df[
                league_df["season"].eq(season)
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

    print()
    print("=" * 120)
    print("DIAGNOSTIC THRESHOLD ROBUSTNESS — BY LEAGUE")
    print("=" * 120)

    for league in LEAGUE_FILES:

        print()
        print("-" * 120)
        print(league.upper())
        print("-" * 120)
        print()

        x = merged[
            merged["league"].eq(league)
        ]

        for threshold in THRESHOLDS:

            print_result(
                f">={threshold:.0%}",
                performance(
                    x,
                    threshold,
                ),
            )


if __name__ == "__main__":
    main()
