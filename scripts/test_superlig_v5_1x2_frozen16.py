from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PRED_FILE = (
    ROOT / "data" / "processed"
    / "footystats_multileague_v5_predictions.csv"
)

MARKET_FILE = (
    ROOT / "data" / "processed"
    / "v5_1x2_football_data.csv"
)

OUT_FILE = (
    ROOT / "data" / "processed"
    / "v5_1x2_with_superlig_frozen16_research.csv"
)

THRESHOLD = 0.16


# ============================================================
# NORMALIZATION
# ============================================================

ALIASES = {
    'ac milan': 'milan',
    'accrington stanley': 'accrington',
    'ad alcorcon': 'alcorcon',
    'ado den haag': 'den haag',
    'albacete balompie': 'albacete',
    'arminia bielefeld': 'bielefeld',
    'as eupen': 'eupen',
    'athletic club bilbao': 'ath bilbao',
    'atletico madrid': 'ath madrid',
    'az': 'az alkmaar',
    'beerschot wilrijk': 'beerschot va',
    'birmingham': 'birmingham city',
    'blackburn rovers': 'blackburn',
    'boavista fc': 'boavista',
    'bolton wanderers': 'bolton',
    'bournemouth': 'afc bournemouth',
    'bradford city': 'bradford',
    'bristol rovers': 'bristol rvs',
    'burgos cf': 'burgos',
    'burton albion': 'burton',
    'ca osasuna': 'osasuna',
    'cambridge united': 'cambridge',
    'cardiff': 'cardiff city',
    'carlisle united': 'carlisle',
    'cd castellon': 'castellon',
    'cd eldense': 'eldense',
    'cd nacional': 'nacional',
    'cd tenerife': 'tenerife',
    'cd tondela': 'tondela',
    'celta de vigo': 'celta',
    'charlton': 'charlton athletic',
    'cheltenham town': 'cheltenham',
    'colchester united': 'colchester',
    'coventry': 'coventry city',
    'crewe alexandra': 'crewe',
    'cs maritimo': 'maritimo',
    'cultural y deportiva leonesa': 'cultural leonesa',
    'darmstadt 98': 'darmstadt',
    'de graafschap': 'graafschap',
    'deportivo alaves': 'alaves',
    'deportivo la coruna': 'la coruna',
    'derby': 'derby county',
    'doncaster rovers': 'doncaster',
    'dynamo dresden': 'dresden',
    'eintracht braunschweig': 'braunschweig',
    'elche cf': 'elche',
    'emmen': 'fc emmen',
    'estrela amadora': 'estrela',
    'exeter city': 'exeter',
    'fc andorra': 'andorra',
    'fc arouca': 'arouca',
    'fc barcelona': 'barcelona',
    'fc cartagena': 'cartagena',
    'fc vizela': 'vizela',
    'fcv dender eh': 'dender',
    'forest green rovers': 'forest green',
    'fortuna sittard': 'for sittard',
    'gd chaves': 'chaves',
    'gd estoril praia': 'estoril',
    'getafe cf': 'getafe',
    'girona fc': 'girona',
    'granada cf': 'granada',
    'grimsby town': 'grimsby',
    'hamburger sv': 'hamburg',
    'hannover 96': 'hannover',
    'harrogate town': 'harrogate',
    'hartlepool united': 'hartlepool',
    'hellas verona': 'verona',
    'hertha bsc': 'hertha',
    'huddersfield': 'huddersfield town',
    'hull': 'hull city',
    'inter milan': 'inter',
    'ipswich': 'ipswich town',
    'jahn regensburg': 'regensburg',
    'kaa gent': 'gent',
    'karlsruher sc': 'karlsruhe',
    'koln': 'fc koln',
    'krc genk': 'genk',
    'ksc lokeren': 'lokeren',
    'kv kortrijk': 'kortrijk',
    'kv mechelen': 'mechelen',
    'kv oostende': 'oostende',
    'kvc westerlo': 'westerlo',
    'leeds': 'leeds united',
    'leicester': 'leicester city',
    'levante ud': 'levante',
    'lincoln city': 'lincoln',
    'luton': 'luton town',
    'macclesfield town': 'macclesfield',
    'malaga cf': 'malaga',
    'mansfield town': 'mansfield',
    'moreirense fc': 'moreirense',
    'msv duisburg': 'duisburg',
    'nec': 'nijmegen',
    'northampton town': 'northampton',
    'norwich city': 'norwich',
    'nott m forest': 'nottingham forest',
    'oh leuven': 'oud heverlee leuven',
    'oldham athletic': 'oldham',
    'oxford': 'oxford united',
    'pacos de ferreira': 'pacos ferreira',
    'pec zwolle': 'zwolle',
    'peterboro': 'peterborough united',
    'plymouth': 'plymouth argyle',
    'preston north end': 'preston',
    'psv': 'psv eindhoven',
    'queens park rangers': 'qpr',
    'racing club de ferrol': 'ferrol',
    'racing santander': 'santander',
    'rayo vallecano': 'vallecano',
    'rcd espanyol': 'espanol',
    'rcd mallorca': 'mallorca',
    'real betis': 'betis',
    'real oviedo': 'oviedo',
    'real sociedad': 'sociedad',
    'real sociedad ii': 'sociedad b',
    'real valladolid': 'valladolid',
    'real zaragoza': 'zaragoza',
    'rfc seraing': 'seraing',
    'rio ave fc': 'rio ave',
    'rkc waalwijk': 'waalwijk',
    'rotherham': 'rotherham united',
    'royal antwerp fc': 'antwerp',
    'royal excel mouscron': 'mouscron',
    'rsc anderlecht': 'anderlecht',
    'rwdm': 'rwd molenbeek',
    'salford city': 'salford',
    'scunthorpe united': 'scunthorpe',
    'sd amorebieta': 'amorebieta',
    'sd eibar': 'eibar',
    'sd huesca': 'huesca',
    'sevilla fc': 'sevilla',
    'sheffield weds': 'sheffield wednesday',
    'shrewsbury town': 'shrewsbury',
    'sint truiden': 'st truiden',
    'sk beveren': 'beveren',
    'southend united': 'southend',
    'sporting braga': 'sp braga',
    'sporting charleroi': 'charleroi',
    'sporting cp': 'sp lisbon',
    'sporting gijon': 'sp gijon',
    'standard liege': 'standard',
    'standard liege fc': 'standard liege',
    'stockport county': 'stockport',
    'stoke city': 'stoke',
    'sutton united': 'sutton',
    'sv zulte waregem': 'zulte waregem',
    'swansea city': 'swansea',
    'swindon town': 'swindon',
    'tranmere rovers': 'tranmere',
    'ud las palmas': 'las palmas',
    'union saint gilloise': 'st gilloise',
    'valencia cf': 'valencia',
    'villarreal ii': 'villarreal b',
    'vitoria guimaraes': 'guimaraes',
    'vvv': 'vvv venlo',
    'waasland beveren': 'beveren',
    'wehen wiesbaden': 'wehen',
    'west bromwich albion': 'west brom',
    'wigan': 'wigan athletic',
    'wycombe': 'wycombe wanderers',
    'yeovil town': 'yeovil',
    'zulte waregem': 'waregem',
}


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

    value = value.replace("&", " and ")

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = " ".join(value.split())

    return ALIASES.get(value, value)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    df = pd.read_csv(PRED_FILE)

    print("=" * 115)
    print("PREDICTION FILE")
    print("=" * 115)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    # Flexible column detection.
    home_col = next(
        c for c in [
            "home_team", "HomeTeam", "home"
        ]
        if c in df.columns
    )

    away_col = next(
        c for c in [
            "away_team", "AwayTeam", "away"
        ]
        if c in df.columns
    )

    date_col = next(
        c for c in [
            "date", "Date"
        ]
        if c in df.columns
    )

    league_col = next(
        c for c in [
            "league", "League"
        ]
        if c in df.columns
    )

    p_home_col = next(
        c for c in [
            "p_home", "prob_home",
            "home_prob", "pred_home_prob"
        ]
        if c in df.columns
    )

    p_draw_col = next(
        c for c in [
            "p_draw", "prob_draw",
            "draw_prob", "pred_draw_prob"
        ]
        if c in df.columns
    )

    p_away_col = next(
        c for c in [
            "p_away", "prob_away",
            "away_prob", "pred_away_prob"
        ]
        if c in df.columns
    )

    keep = [
        league_col,
        date_col,
        home_col,
        away_col,
        p_home_col,
        p_draw_col,
        p_away_col,
    ]

    if "season" in df.columns:
        keep.append("season")

    x = df[keep].copy()

    rename = {
        league_col: "league",
        date_col: "date",
        home_col: "home_team",
        away_col: "away_team",
        p_home_col: "p_home",
        p_draw_col: "p_draw",
        p_away_col: "p_away",
    }

    x = x.rename(columns=rename)

    # Research expansion:
    # FootyStats = "Super Lig"
    # Historical market = "Süper Lig"
    x["league"] = x["league"].replace({
        "Super Lig": "Süper Lig",
    })

    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce",
    ).dt.normalize()

    for c in ["p_home", "p_draw", "p_away"]:
        x[c] = pd.to_numeric(
            x[c],
            errors="coerce",
        )

    x = x[
        x["date"].notna()
        & x["home_team"].notna()
        & x["away_team"].notna()
        & x[
            ["p_home", "p_draw", "p_away"]
        ].notna().all(axis=1)
    ].copy()

    x["home_key"] = x["home_team"].map(
        normalize_team
    )

    x["away_key"] = x["away_team"].map(
        normalize_team
    )

    # Sanity check probability sums.
    x["prob_sum"] = (
        x["p_home"]
        + x["p_draw"]
        + x["p_away"]
    )

    print("\nProbability sum:")
    print(x["prob_sum"].describe().to_string())

    print("\nRows by league:")
    print(
        x["league"]
        .value_counts()
        .to_string()
    )

    return x


# ============================================================
# LOAD MARKET
# ============================================================

def load_market():

    df = pd.read_csv(MARKET_FILE)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.normalize()

    for c in [
        "home_odds",
        "draw_odds",
        "away_odds",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df["home_key"] = df["home_team"].map(
        normalize_team
    )

    df["away_key"] = df["away_team"].map(
        normalize_team
    )

    return df


# ============================================================
# MATCH
# ============================================================

def build_dataset():

    pred = load_predictions()
    market = load_market()

    common = sorted(
        set(pred["league"].dropna())
        & set(market["league"].dropna())
    )

    print("\n" + "=" * 115)
    print("COMMON LEAGUES")
    print("=" * 115)

    for league in common:
        print(league)

    pred = pred[
        pred["league"].isin(common)
    ].copy()

    market = market[
        market["league"].isin(common)
    ].copy()

    key = [
        "league",
        "date",
        "home_key",
        "away_key",
    ]

    # Avoid accidental duplicate joins.
    pred_dupes = pred.duplicated(
        key,
        keep=False,
    )

    market_dupes = market.duplicated(
        key,
        keep=False,
    )

    print("\nPrediction duplicate keys:",
          int(pred_dupes.sum()))

    print("Market duplicate keys:",
          int(market_dupes.sum()))

    pred = pred.drop_duplicates(
        key,
        keep="first",
    )

    market = market.drop_duplicates(
        key,
        keep="first",
    )

    # Coverage audit from prediction side.
    coverage = pred.merge(
        market[
            key + [
                "home_odds",
                "draw_odds",
                "away_odds",
            ]
        ],
        on=key,
        how="left",
        indicator=True,
    )

    print("\n" + "=" * 115)
    print("MATCH COVERAGE BY LEAGUE")
    print("=" * 115)

    rows = []

    for league, g in coverage.groupby("league"):

        matched = g["_merge"].eq("both")

        valid_odds = (
            matched
            & g[
                [
                    "home_odds",
                    "draw_odds",
                    "away_odds",
                ]
            ].notna().all(axis=1)
        )

        rows.append({
            "league": league,
            "predictions": len(g),
            "matched": int(matched.sum()),
            "valid_odds": int(valid_odds.sum()),
            "match_pct": matched.mean(),
            "valid_pct": valid_odds.mean(),
        })

    cov = pd.DataFrame(rows)

    if len(cov):
        print(
            cov.sort_values(
                "predictions",
                ascending=False,
            ).to_string(
                index=False,
                formatters={
                    "match_pct":
                        lambda x: f"{x:.2%}",
                    "valid_pct":
                        lambda x: f"{x:.2%}",
                },
            )
        )

    # Exact matched dataset.
    merged = pred.merge(
        market,
        on=key,
        how="inner",
        suffixes=("_pred", "_market"),
    )

    merged = merged[
        merged[
            [
                "home_odds",
                "draw_odds",
                "away_odds",
            ]
        ].notna().all(axis=1)
    ].copy()

    # Recalculate no-vig probabilities ourselves.
    merged["imp_home"] = 1 / merged["home_odds"]
    merged["imp_draw"] = 1 / merged["draw_odds"]
    merged["imp_away"] = 1 / merged["away_odds"]

    total = (
        merged["imp_home"]
        + merged["imp_draw"]
        + merged["imp_away"]
    )

    merged["mkt_home"] = (
        merged["imp_home"] / total
    )

    merged["mkt_draw"] = (
        merged["imp_draw"] / total
    )

    merged["mkt_away"] = (
        merged["imp_away"] / total
    )

    merged["edge_home"] = (
        merged["p_home"]
        - merged["mkt_home"]
    )

    merged["edge_draw"] = (
        merged["p_draw"]
        - merged["mkt_draw"]
    )

    merged["edge_away"] = (
        merged["p_away"]
        - merged["mkt_away"]
    )

    return merged, cov


# ============================================================
# FROZEN BET RULE
# ============================================================

def create_bets(df):

    rows = []

    for _, r in df.iterrows():

        candidates = [
            (
                "H",
                r["edge_home"],
                r["home_odds"],
                r["p_home"],
                r["mkt_home"],
            ),
            (
                "D",
                r["edge_draw"],
                r["draw_odds"],
                r["p_draw"],
                r["mkt_draw"],
            ),
            (
                "A",
                r["edge_away"],
                r["away_odds"],
                r["p_away"],
                r["mkt_away"],
            ),
        ]

        # ONE BET PER MATCH:
        # choose largest RAW V5 edge.
        selection, edge, odds, model_p, market_p = max(
            candidates,
            key=lambda x: x[1],
        )

        if edge < THRESHOLD:
            continue

        actual = r["result"]

        win = selection == actual

        profit = (
            odds - 1.0
            if win
            else -1.0
        )

        rows.append({
            "league": r["league"],
            "date": r["date"],
            "home_team": r["home_team_market"],
            "away_team": r["away_team_market"],
            "selection": selection,
            "actual": actual,
            "odds": odds,
            "model_prob": model_p,
            "market_prob": market_p,
            "raw_edge": edge,
            "win": int(win),
            "profit": profit,
        })

    return pd.DataFrame(rows)


# ============================================================
# PERFORMANCE
# ============================================================

def max_drawdown(profits):

    if len(profits) == 0:
        return np.nan

    equity = np.cumsum(profits)
    peak = np.maximum.accumulate(
        np.r_[0.0, equity]
    )[1:]

    dd = equity - peak

    return float(dd.min())


def bootstrap_positive_roi(
    profits,
    n_boot=10000,
    seed=42,
):

    profits = np.asarray(
        profits,
        dtype=float,
    )

    if len(profits) == 0:
        return np.nan

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(profits),
        size=(n_boot, len(profits)),
    )

    means = profits[idx].mean(axis=1)

    return float(
        np.mean(means > 0)
    )


def summarize(bets):

    rows = []

    for league, g in bets.groupby("league"):

        bets_n = len(g)
        wins = int(g["win"].sum())
        profit = g["profit"].sum()
        roi = profit / bets_n

        seasons = (
            g.assign(
                season=g["date"].dt.year
            )
            .groupby("season")
            .agg(
                bets=("profit", "size"),
                profit=("profit", "sum"),
            )
        )

        seasons["roi"] = (
            seasons["profit"]
            / seasons["bets"]
        )

        rows.append({
            "league": league,
            "bets": bets_n,
            "wins": wins,
            "losses": bets_n - wins,
            "win_pct": wins / bets_n,
            "avg_odds": g["odds"].mean(),
            "avg_edge": g["raw_edge"].mean(),
            "profit": profit,
            "roi": roi,
            "max_drawdown":
                max_drawdown(
                    g.sort_values("date")["profit"]
                ),
            "p_roi_gt_0":
                bootstrap_positive_roi(
                    g["profit"].values
                ),
            "positive_years":
                int((seasons["roi"] > 0).sum()),
            "years":
                len(seasons),
            "worst_year_roi":
                seasons["roi"].min(),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    df, coverage = build_dataset()

    print("\n" + "=" * 115)
    print("MATCHED V5 + FOOTBALL-DATA DATASET")
    print("=" * 115)

    print("Rows:", len(df))
    print("Leagues:", df["league"].nunique())

    bets = create_bets(df)

    print("\n" + "=" * 115)
    print("FROZEN V5 1X2 RULE")
    print("=" * 115)

    print("RAW edge threshold: >= 16%")
    print("One selection per match")
    print("Selection = highest RAW V5 edge")
    print("Flat stake = 1 unit")
    print("Market = Football-Data B365 opening 1X2")
    print("No threshold optimization")

    if bets.empty:
        print("\nNO QUALIFYING BETS")
        return

    OUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bets.to_csv(
        OUT_FILE,
        index=False,
    )

    summary = summarize(bets)

    summary = summary.sort_values(
        ["roi", "bets"],
        ascending=[False, False],
    )

    print("\n" + "=" * 115)
    print("ALL-LEAGUE RESULTS")
    print("=" * 115)

    print(
        summary.to_string(
            index=False,
            formatters={
                "win_pct":
                    lambda x: f"{x:.2%}",
                "avg_odds":
                    lambda x: f"{x:.3f}",
                "avg_edge":
                    lambda x: f"{x:.2%}",
                "profit":
                    lambda x: f"{x:+.2f}",
                "roi":
                    lambda x: f"{x:+.2%}",
                "max_drawdown":
                    lambda x: f"{x:.2f}",
                "p_roi_gt_0":
                    lambda x: f"{x:.1%}",
                "worst_year_roi":
                    lambda x: f"{x:+.2%}",
            },
        )
    )

    print("\n" + "=" * 115)
    print("SEASON / YEAR DETAIL")
    print("=" * 115)

    for league in summary["league"]:

        g = bets[
            bets["league"].eq(league)
        ].copy()

        # Calendar year is displayed only as an initial
        # robustness check. We'll convert to proper league
        # seasons in the final validator if necessary.
        g["year"] = g["date"].dt.year

        s = (
            g.groupby("year")
            .agg(
                bets=("profit", "size"),
                wins=("win", "sum"),
                profit=("profit", "sum"),
            )
        )

        s["roi"] = (
            s["profit"] / s["bets"]
        )

        print(f"\n{league}")
        print(
            s.to_string(
                formatters={
                    "profit":
                        lambda x: f"{x:+.2f}",
                    "roi":
                        lambda x: f"{x:+.2%}",
                }
            )
        )

    print("\nSaved bets:")
    print(OUT_FILE)


if __name__ == "__main__":
    main()
