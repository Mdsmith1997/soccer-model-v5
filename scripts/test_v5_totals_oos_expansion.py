from pathlib import Path
import math
import re
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PRED_FILE = (
    ROOT
    / "data"
    / "processed"
    / "footystats_multileague_v5_predictions.csv"
)

RAW_DIR = (
    ROOT
    / "data"
    / "raw"
)

LEAGUES = {
    "Championship": "E1",
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


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_team(value):

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        c
        for c in value
        if not unicodedata.combining(c)
    )

    value = value.replace(
        "&",
        " and ",
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
        # England - Championship / Football-Data aliases
        "birmingham":
            "birmingham city",

        "bournemouth":
            "afc bournemouth",

        "cardiff":
            "cardiff city",

        "charlton":
            "charlton athletic",

        "coventry":
            "coventry city",

        "derby":
            "derby county",

        "huddersfield":
            "huddersfield town",

        "hull":
            "hull city",

        "ipswich":
            "ipswich town",

        "leeds":
            "leeds united",

        "leicester":
            "leicester city",

        "luton":
            "luton town",

        "nott m forest":
            "nottingham forest",

        "oxford":
            "oxford united",

        "peterboro":
            "peterborough united",

        "plymouth":
            "plymouth argyle",

        "rotherham":
            "rotherham united",

        "sheffield weds":
            "sheffield wednesday",

        "wigan":
            "wigan athletic",

        "wycombe":
            "wycombe wanderers",

        # Belgium
        "standard liege":
            "standard liege",

        "standard liege fc":
            "standard liege",

        "kaa gent":
            "gent",

        "rsc anderlecht":
            "anderlecht",

        "waasland beveren":
            "beveren",

        "sk beveren":
            "beveren",

        "zulte waregem":
            "zulte waregem",

        "sv zulte waregem":
            "zulte waregem",

        "union saint gilloise":
            "union saint gilloise",

        "royale union saint gilloise":
            "union saint gilloise",

        "st truiden":
            "sint truiden",

        "sint truidense":
            "sint truiden",

        "club brugge kv":
            "club brugge",

        "cercle brugge ksv":
            "cercle brugge",

        "royal antwerp fc":
            "royal antwerp",

        "kv kortrijk":
            "kortrijk",

        "kv mechelen":
            "mechelen",

        # Championship/common England aliases
        "qpr":
            "queens park rangers",

        "sheff utd":
            "sheffield united",

        "sheffield utd":
            "sheffield united",

        "sheff wed":
            "sheffield wednesday",

        "west brom":
            "west bromwich albion",

        "blackburn":
            "blackburn rovers",

        "preston":
            "preston north end",

        "stoke":
            "stoke city",

        "swansea":
            "swansea city",

        "norwich":
            "norwich city",

        "bristol city":
            "bristol city",

        "millwall":
            "millwall",
    }

    return aliases.get(
        value,
        value,
    )


# ============================================================
# POISSON TOTALS
# ============================================================

def p_under_2_5(
    home_lambda,
    away_lambda,
):

    total_lambda = (
        float(home_lambda)
        +
        float(away_lambda)
    )

    # P(total goals <= 2)
    return (
        math.exp(-total_lambda)
        *
        (
            1.0
            +
            total_lambda
            +
            (
                total_lambda ** 2
                /
                2.0
            )
        )
    )


# ============================================================
# LOAD V5 PREDICTIONS
# ============================================================

def load_predictions():

    df = pd.read_csv(
        PRED_FILE,
        low_memory=False,
    )

    df = df[
        df["league"].isin(
            LEAGUES
        )
    ].copy()

    df["season"] = (
        df["season"]
        .astype(str)
        .str.strip()
    )

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

    df["home_key"] = (
        df["home_team"]
        .map(normalize_team)
    )

    df["away_key"] = (
        df["away_team"]
        .map(normalize_team)
    )

    return df


# ============================================================
# LOAD HISTORICAL MARKET
# ============================================================

def load_market():

    rows = []

    for league, code in LEAGUES.items():

        for season in SEASONS:

            path = (
                RAW_DIR
                / f"{season}_{code}.csv"
            )

            if not path.exists():
                continue

            x = pd.read_csv(
                path,
                low_memory=False,
            )

            required = [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
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
                    f"Skipping {path.name}; "
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

            x["market_home_goals"] = (
                pd.to_numeric(
                    x["FTHG"],
                    errors="coerce",
                )
            )

            x["market_away_goals"] = (
                pd.to_numeric(
                    x["FTAG"],
                    errors="coerce",
                )
            )

            x["league"] = league
            x["season"] = season

            rows.append(
                x[
                    [
                        "league",
                        "season",
                        "date",
                        "home_key",
                        "away_key",
                        "HomeTeam",
                        "AwayTeam",
                        "over_odds",
                        "under_odds",
                        "market_home_goals",
                        "market_away_goals",
                    ]
                ]
            )

    if not rows:
        raise RuntimeError(
            "No market files loaded."
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ============================================================
# BUILD MATCHED DATASET
# ============================================================

def build_dataset():

    pred = load_predictions()
    market = load_market()

    print()
    print("=" * 120)
    print("INPUT AUDIT")
    print("=" * 120)

    print()
    print("V5 prediction rows:")
    print(
        pred["league"]
        .value_counts()
        .to_string()
    )

    print()
    print("Historical market rows:")
    print(
        market["league"]
        .value_counts()
        .to_string()
    )

    merged = pred.merge(
        market,
        on=[
            "league",
            "season",
            "date",
            "home_key",
            "away_key",
        ],
        how="inner",
        suffixes=(
            "_pred",
            "_market",
        ),
    )

    merged = merged[
        merged["under_odds"].gt(1.0)
        &
        merged["over_odds"].gt(1.0)
    ].copy()

    print()
    print("Matched rows:")
    print(
        merged["league"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # MODEL PROBABILITY
    # ========================================================

    merged[
        "model_p_under"
    ] = [
        p_under_2_5(
            h,
            a,
        )
        for h, a in zip(
            merged["home_lambda"],
            merged["away_lambda"],
        )
    ]

    # ========================================================
    # MARKET NO-VIG PROBABILITY
    # ========================================================

    raw_over = (
        1.0
        /
        merged["over_odds"]
    )

    raw_under = (
        1.0
        /
        merged["under_odds"]
    )

    vig_sum = (
        raw_over
        +
        raw_under
    )

    merged[
        "market_p_under"
    ] = (
        raw_under
        /
        vig_sum
    )

    # ========================================================
    # EDGE / EV
    # ========================================================

    merged[
        "under_edge"
    ] = (
        merged["model_p_under"]
        -
        merged["market_p_under"]
    )

    merged[
        "under_ev"
    ] = (
        merged["model_p_under"]
        *
        merged["under_odds"]
        -
        1.0
    )

    merged[
        "actual_total"
    ] = (
        merged["home_goals"]
        +
        merged["away_goals"]
    )

    merged[
        "under_win"
    ] = (
        merged["actual_total"]
        <
        2.5
    ).astype(int)

    return merged


# ============================================================
# PERFORMANCE
# ============================================================

def performance(
    df,
    threshold,
):

    x = df[
        df["under_edge"]
        >= threshold
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

    x[
        "profit"
    ] = np.where(
        x["under_win"].eq(1),
        x["under_odds"] - 1.0,
        -1.0,
    )

    bets = len(x)

    wins = int(
        x["under_win"].sum()
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

        "avg_odds":
            x["under_odds"].mean(),

        "avg_edge":
            x["under_edge"].mean(),

        "avg_ev":
            x["under_ev"].mean(),

        "profit":
            profit,

        "roi":
            profit / bets,
    }


# ============================================================
# DISPLAY
# ============================================================

def fmt_pct(x):

    if pd.isna(x):
        return "-"

    return f"{x:+.2%}"


def print_threshold_scan(df):

    rows = []

    for threshold in THRESHOLDS:

        p = performance(
            df,
            threshold,
        )

        rows.append(
            {
                "threshold":
                    threshold,

                **p,
            }
        )

    out = pd.DataFrame(
        rows
    )

    show = out.copy()

    show[
        "threshold"
    ] = show[
        "threshold"
    ].map(
        lambda x:
            f"{x:.0%}"
    )

    for c in [
        "win_rate",
        "avg_edge",
        "avg_ev",
        "roi",
    ]:

        show[c] = show[c].map(
            fmt_pct
        )

    show[
        "avg_odds"
    ] = show[
        "avg_odds"
    ].map(
        lambda x:
            "-"
            if pd.isna(x)
            else f"{x:.3f}"
    )

    show[
        "profit"
    ] = show[
        "profit"
    ].map(
        lambda x:
            f"{x:+.2f}u"
    )

    print(
        show.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 120)
    print("V5 TOTALS — OUT-OF-SAMPLE LEAGUE EXPANSION")
    print("=" * 120)

    print()
    print(
        "PRIMARY PRE-SELECTED RULE: "
        "RAW UNDER 2.5 EDGE >= 11%"
    )

    print(
        "These leagues were NOT used to choose "
        "the 11% threshold."
    )

    df = build_dataset()

    print()
    print("=" * 120)
    print("AGGREGATE THRESHOLD ROBUSTNESS")
    print("=" * 120)
    print()

    print_threshold_scan(
        df
    )

    # ========================================================
    # FROZEN 11% BY LEAGUE
    # ========================================================

    print()
    print("=" * 120)
    print("FROZEN 11% RULE — BY LEAGUE")
    print("=" * 120)
    print()

    league_rows = []

    for league, x in df.groupby(
        "league"
    ):

        p = performance(
            x,
            FROZEN_THRESHOLD,
        )

        league_rows.append(
            {
                "league":
                    league,

                **p,
            }
        )

    league_df = pd.DataFrame(
        league_rows
    )

    if not league_df.empty:

        show = league_df.copy()

        for c in [
            "win_rate",
            "avg_edge",
            "avg_ev",
            "roi",
        ]:

            show[c] = show[c].map(
                fmt_pct
            )

        show[
            "avg_odds"
        ] = show[
            "avg_odds"
        ].map(
            lambda x:
                "-"
                if pd.isna(x)
                else f"{x:.3f}"
        )

        show[
            "profit"
        ] = show[
            "profit"
        ].map(
            lambda x:
                f"{x:+.2f}u"
        )

        print(
            show.to_string(
                index=False
            )
        )

    # ========================================================
    # FROZEN 11% BY SEASON + LEAGUE
    # ========================================================

    print()
    print("=" * 120)
    print("FROZEN 11% RULE — BY SEASON")
    print("=" * 120)
    print()

    season_rows = []

    for (
        league,
        season,
    ), x in df.groupby(
        [
            "league",
            "season",
        ]
    ):

        p = performance(
            x,
            FROZEN_THRESHOLD,
        )

        if p["bets"] == 0:
            continue

        season_rows.append(
            {
                "league":
                    league,

                "season":
                    season,

                **p,
            }
        )

    season_df = pd.DataFrame(
        season_rows
    )

    if season_df.empty:

        print(
            "No >=11% signals."
        )

    else:

        show = season_df.copy()

        for c in [
            "win_rate",
            "avg_edge",
            "avg_ev",
            "roi",
        ]:

            show[c] = show[c].map(
                fmt_pct
            )

        show[
            "avg_odds"
        ] = show[
            "avg_odds"
        ].map(
            lambda x:
                f"{x:.3f}"
        )

        show[
            "profit"
        ] = show[
            "profit"
        ].map(
            lambda x:
                f"{x:+.2f}u"
        )

        print(
            show.to_string(
                index=False
            )
        )

    # ========================================================
    # DECISION SUMMARY
    # ========================================================

    frozen = performance(
        df,
        FROZEN_THRESHOLD,
    )

    print()
    print("=" * 120)
    print("OUT-OF-SAMPLE 11% RESULT")
    print("=" * 120)
    print()

    print(
        f"Bets:      {frozen['bets']}"
    )

    print(
        f"Wins:      {frozen['wins']}"
    )

    print(
        f"Win rate:  "
        f"{frozen['win_rate']:.2%}"
        if frozen["bets"]
        else "Win rate:  -"
    )

    print(
        f"Profit:    "
        f"{frozen['profit']:+.2f}u"
    )

    print(
        f"ROI:       "
        f"{frozen['roi']:+.2%}"
        if frozen["bets"]
        else "ROI:       -"
    )

    print()
    print(
        "Interpretation rule:"
    )

    print(
        "Do NOT choose a different threshold just "
        "because it has the highest ROI here."
    )

    print(
        "11% is the genuine out-of-sample test. "
        "Nearby thresholds are only a robustness check."
    )


if __name__ == "__main__":
    main()
