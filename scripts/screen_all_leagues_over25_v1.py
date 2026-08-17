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

RAW_DIR = ROOT / "data" / "raw"

# ============================================================
# FROZEN LEAGUE UNIVERSE
# Same 11 leagues used in BTTS all-league investigation.
# ============================================================

LEAGUE_FILES = {
    "2. Bundesliga": "D2",
    "Belgian Pro League": "B1",
    "Championship": "E1",
    "Eredivisie": "N1",
    "League One": "E2",
    "League Two": "E3",
    "Primeira Liga": "P1",
    "Segunda División": "SP2",
    "Serie A": "I1",
    "Super Lig": "T1",
    "Swiss Super League": "SWZ",
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
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.11,
    0.12,
    0.14,
    0.16,
]

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "over25_all_leagues_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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

        # Eredivisie
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

        # English EFL
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

        # Belgium
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

        # 2 Bundesliga
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


# ============================================================
# RAW V5 OVER 2.5 PROBABILITY
# ============================================================

def p_under_25(home_lambda, away_lambda):

    lam = (
        float(home_lambda)
        +
        float(away_lambda)
    )

    return math.exp(-lam) * (
        1.0
        +
        lam
        +
        lam ** 2 / 2.0
    )


def p_over_25(home_lambda, away_lambda):

    return (
        1.0
        -
        p_under_25(
            home_lambda,
            away_lambda,
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
        df["season"]
        .isin(SEASONS)
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

    df["home_key"] = (
        df["home_team"]
        .map(normalize_team)
    )

    df["away_key"] = (
        df["away_team"]
        .map(normalize_team)
    )

    df["raw_over_prob"] = [
        p_over_25(h, a)
        for h, a in zip(
            df["home_lambda"],
            df["away_lambda"],
        )
    ]

    df["over_win"] = (
        (
            df["home_goals"]
            +
            df["away_goals"]
        )
        > 2
    ).astype(int)

    return df


# ============================================================
# LOAD HISTORICAL MARKET
# ============================================================

def load_market():

    rows = []

    missing_files = []

    for league, raw_code in LEAGUE_FILES.items():

        for season in SEASONS:

            path = (
                RAW_DIR
                / f"{season}_{raw_code}.csv"
            )

            if not path.exists():

                missing_files.append(
                    (
                        league,
                        season,
                        path.name,
                    )
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
                    f"missing columns {missing}"
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

    print()
    print("=" * 120)
    print("MISSING RAW MARKET FILES")
    print("=" * 120)

    if not missing_files:
        print("None")
    else:
        for league, season, name in missing_files:
            print(
                f"{league:<22} "
                f"{season:<6} "
                f"{name}"
            )

    if not rows:
        raise RuntimeError(
            "No historical totals market data found."
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ============================================================
# PERFORMANCE
# ============================================================

def performance(df, threshold):

    x = df[
        df["raw_over_edge"]
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

    x["profit"] = np.where(
        x["over_win"].eq(1),
        x["over_odds"] - 1.0,
        -1.0,
    )

    bets = len(x)
    wins = int(
        x["over_win"].sum()
    )

    profit = float(
        x["profit"].sum()
    )

    return {
        "bets": bets,
        "wins": wins,
        "win_rate": wins / bets,
        "avg_odds": x["over_odds"].mean(),
        "avg_edge": x["raw_over_edge"].mean(),
        "avg_ev": x["raw_over_ev"].mean(),
        "profit": profit,
        "roi": profit / bets,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 120)
    print("ALL-LEAGUE RAW OVER 2.5 — V1 DISCOVERY SCREEN")
    print("=" * 120)

    print()
    print(
        "League universe:",
        len(LEAGUE_FILES),
    )

    for league in LEAGUE_FILES:
        print(" ", league)

    pred = load_predictions()
    market = load_market()

    print()
    print("=" * 120)
    print("V5 PREDICTION COUNTS")
    print("=" * 120)

    print(
        pred.groupby("league")
        .size()
        .to_string()
    )

    print()
    print("=" * 120)
    print("MARKET COUNTS")
    print("=" * 120)

    print(
        market.groupby("league")
        .size()
        .to_string()
    )

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
        validate="one_to_one",
    )

    merged = merged.dropna(
        subset=[
            "raw_over_prob",
            "over_odds",
            "under_odds",
            "over_win",
        ]
    ).copy()

    # No-vig market probabilities
    merged["over_imp"] = (
        1.0
        /
        merged["over_odds"]
    )

    merged["under_imp"] = (
        1.0
        /
        merged["under_odds"]
    )

    denom = (
        merged["over_imp"]
        +
        merged["under_imp"]
    )

    merged["market_over_nv"] = (
        merged["over_imp"]
        /
        denom
    )

    merged["market_under_nv"] = (
        merged["under_imp"]
        /
        denom
    )

    merged["raw_over_edge"] = (
        merged["raw_over_prob"]
        -
        merged["market_over_nv"]
    )

    merged["raw_over_ev"] = (
        merged["raw_over_prob"]
        *
        merged["over_odds"]
        -
        1.0
    )

    print()
    print("=" * 120)
    print("MATCHED COUNTS")
    print("=" * 120)

    matched_counts = (
        merged.groupby("league")
        .size()
        .reindex(
            LEAGUE_FILES.keys()
        )
        .fillna(0)
        .astype(int)
    )

    print(
        matched_counts.to_string()
    )

    print()
    print("=" * 120)
    print("MATCH RATES")
    print("=" * 120)

    pred_counts = (
        pred.groupby("league")
        .size()
        .reindex(
            LEAGUE_FILES.keys()
        )
        .fillna(0)
    )

    rates = (
        matched_counts
        /
        pred_counts.replace(0, np.nan)
    )

    for league in LEAGUE_FILES:

        print(
            f"{league:<22} "
            f"{matched_counts.loc[league]:>5} / "
            f"{int(pred_counts.loc[league]):>5} "
            f"= {rates.loc[league]:>7.2%}"
        )

    # Save match-level dataset
    merged.to_csv(
        OUT_DIR
        / "01_matches.csv",
        index=False,
    )

    # ========================================================
    # AGGREGATE THRESHOLD SCREEN
    # ========================================================

    rows = []

    for league, g in merged.groupby(
        "league",
        sort=True,
    ):

        for threshold in THRESHOLDS:

            p = performance(
                g,
                threshold,
            )

            rows.append(
                {
                    "league": league,
                    "threshold": threshold,
                    **p,
                }
            )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        OUT_DIR
        / "02_threshold_screen.csv",
        index=False,
    )

    print()
    print("=" * 120)
    print("RAW OVER 2.5 — THRESHOLD SCREEN")
    print("=" * 120)

    for league in LEAGUE_FILES:

        print()
        print("-" * 120)
        print(league)
        print("-" * 120)

        x = summary[
            summary["league"].eq(league)
        ]

        for _, r in x.iterrows():

            if not r["bets"]:
                print(
                    f"{r['threshold']:>5.0%} "
                    f"no bets"
                )
                continue

            print(
                f"{r['threshold']:>5.0%} | "
                f"N={int(r['bets']):>4} | "
                f"WR={r['win_rate']:>7.2%} | "
                f"odds={r['avg_odds']:.3f} | "
                f"edge={r['avg_edge']:>7.2%} | "
                f"EV={r['avg_ev']:>7.2%} | "
                f"P/L={r['profit']:>+8.2f}u | "
                f"ROI={r['roi']:>+7.2%}"
            )

    # ========================================================
    # YEAR-BY-YEAR
    # ========================================================

    year_rows = []

    for (
        league,
        season
    ), g in merged.groupby(
        [
            "league",
            "season",
        ],
        sort=True,
    ):

        for threshold in THRESHOLDS:

            p = performance(
                g,
                threshold,
            )

            year_rows.append(
                {
                    "league": league,
                    "season": season,
                    "threshold": threshold,
                    **p,
                }
            )

    by_year = pd.DataFrame(
        year_rows
    )

    by_year.to_csv(
        OUT_DIR
        / "03_by_season.csv",
        index=False,
    )

    print()
    print("=" * 120)
    print("FILES SAVED")
    print("=" * 120)

    print(
        OUT_DIR
        / "01_matches.csv"
    )

    print(
        OUT_DIR
        / "02_threshold_screen.csv"
    )

    print(
        OUT_DIR
        / "03_by_season.csv"
    )

    print()
    print("=" * 120)
    print("DISCOVERY SCREEN COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
