from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata
import re

ROOT = Path("data/processed")

PRED_FILE = ROOT / "footystats_multileague_v5_predictions.csv"
MARKET_FILE = ROOT / "v5_1x2_football_data.csv"
OUT_FILE = ROOT / "eredivisie_v5_1x2_frozen16_corrected.csv"

LEAGUE = "Eredivisie"
EDGE_THRESHOLD = 0.16


def norm_team(x):
    x = str(x).strip().lower()
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = x.replace("&", "and")
    x = re.sub(r"[^a-z0-9]+", "", x)

    aliases = {
        # ADO Den Haag
        "adodenhaag": "denhaag",
        "denhaag": "denhaag",

        # AZ
        "az": "az",
        "azalkmaar": "az",

        # De Graafschap
        "degraafschap": "graafschap",
        "graafschap": "graafschap",

        # Emmen
        "emmen": "emmen",
        "fcemmen": "emmen",

        # Fortuna Sittard
        "fortunasittard": "fortunasittard",
        "forsittard": "fortunasittard",

        # NEC Nijmegen
        "nec": "nec",
        "nijmegen": "nec",

        # PEC Zwolle
        "peczwolle": "zwolle",
        "zwolle": "zwolle",

        # PSV
        "psv": "psv",
        "psveindhoven": "psv",

        # RKC Waalwijk
        "rkcwaalwijk": "waalwijk",
        "waalwijk": "waalwijk",

        # VVV Venlo
        "vvv": "vvv",
        "vvvvenlo": "vvv",
    }

    return aliases.get(x, x)


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


pred = pd.read_csv(PRED_FILE, low_memory=False)
market = pd.read_csv(MARKET_FILE, low_memory=False)

pred = pred[
    pred["league"].astype(str).str.strip().eq(LEAGUE)
].copy()

market = market[
    market["league"].astype(str).str.strip().eq(LEAGUE)
].copy()

print("=" * 120)
print("EREDIVISIE — CORRECTED FROZEN V5 1X2 ≥16%")
print("=" * 120)

print("Prediction rows:", len(pred))
print("Market rows:    ", len(market))

ph = find_col(pred, ["p_home", "home_win_prob", "prob_home"])
pd_ = find_col(pred, ["p_draw", "draw_prob", "prob_draw"])
pa = find_col(pred, ["p_away", "away_win_prob", "prob_away"])

mh = find_col(market, ["home_odds", "B365H", "b365_home"])
md = find_col(market, ["draw_odds", "B365D", "b365_draw"])
ma = find_col(market, ["away_odds", "B365A", "b365_away"])

result_col = find_col(
    market,
    ["actual_result", "result", "FTR"]
)

required = {
    "pred home probability": ph,
    "pred draw probability": pd_,
    "pred away probability": pa,
    "market home odds": mh,
    "market draw odds": md,
    "market away odds": ma,
    "market result": result_col,
}

missing = [k for k, v in required.items() if v is None]

if missing:
    print("Pred columns:", pred.columns.tolist())
    print("Market columns:", market.columns.tolist())
    raise RuntimeError("Missing columns: " + ", ".join(missing))


for df in [pred, market]:
    df["date_key"] = pd.to_datetime(
        df["date"], errors="coerce"
    ).dt.normalize()

    df["home_key"] = df["home_team"].map(norm_team)
    df["away_key"] = df["away_team"].map(norm_team)


market_cols = [
    "date_key",
    "home_key",
    "away_key",
    mh,
    md,
    ma,
    result_col,
]

merged = pred.merge(
    market[market_cols],
    on=["date_key", "home_key", "away_key"],
    how="left",
    indicator=True,
)

exact = merged["_merge"].eq("both")

print()
print("Exact matches:", int(exact.sum()))
print("Exact rate:   ", f"{exact.mean():.2%}")
print("Unmatched:    ", int((~exact).sum()))


for c in [ph, pd_, pa, mh, md, ma]:
    merged[c] = pd.to_numeric(
        merged[c],
        errors="coerce"
    )


valid = merged[
    exact
    & merged[[ph, pd_, pa, mh, md, ma]].notna().all(axis=1)
    & (merged[[mh, md, ma]] > 1.0).all(axis=1)
].copy()

print("Valid odds:   ", len(valid))
print("Coverage:     ", f"{len(valid)/len(pred):.2%}")


# Coverage by year
merged["year"] = merged["date_key"].dt.year
merged["exact"] = exact

print()
print("=" * 120)
print("COVERAGE BY YEAR")
print("=" * 120)

cov = (
    merged.groupby("year")
    .agg(
        predictions=("exact", "size"),
        exact_matches=("exact", "sum"),
    )
)

cov["exact_pct"] = (
    cov["exact_matches"] / cov["predictions"]
)

print(
    cov.to_string(
        formatters={
            "exact_pct": lambda x: f"{x:.2%}"
        }
    )
)


# Three-way proportional de-vig
valid["imp_h"] = 1 / valid[mh]
valid["imp_d"] = 1 / valid[md]
valid["imp_a"] = 1 / valid[ma]

valid["book_sum"] = (
    valid["imp_h"]
    + valid["imp_d"]
    + valid["imp_a"]
)

valid["nv_h"] = valid["imp_h"] / valid["book_sum"]
valid["nv_d"] = valid["imp_d"] / valid["book_sum"]
valid["nv_a"] = valid["imp_a"] / valid["book_sum"]


# Frozen raw V5 edge
valid["edge_h"] = valid[ph] - valid["nv_h"]
valid["edge_d"] = valid[pd_] - valid["nv_d"]
valid["edge_a"] = valid[pa] - valid["nv_a"]

edges = valid[
    ["edge_h", "edge_d", "edge_a"]
].to_numpy()

best_idx = np.argmax(edges, axis=1)

labels = np.array(["H", "D", "A"])

valid["selection"] = labels[best_idx]

valid["raw_edge"] = edges[
    np.arange(len(valid)),
    best_idx
]

valid["odds"] = np.select(
    [
        valid["selection"].eq("H"),
        valid["selection"].eq("D"),
        valid["selection"].eq("A"),
    ],
    [
        valid[mh],
        valid[md],
        valid[ma],
    ],
    default=np.nan,
)


bets = valid[
    valid["raw_edge"] >= EDGE_THRESHOLD
].copy()

bets["actual_result"] = (
    bets[result_col]
    .astype(str)
    .str.upper()
    .str.strip()
)

bets["win"] = (
    bets["selection"] == bets["actual_result"]
).astype(int)

bets["profit"] = np.where(
    bets["win"].eq(1),
    bets["odds"] - 1,
    -1.0,
)

bets["year"] = bets["date_key"].dt.year

bets.to_csv(OUT_FILE, index=False)


def show(label, g):
    print()
    print(label)

    if len(g) == 0:
        print("NO BETS")
        return

    print("Bets:      ", len(g))
    print("Wins:      ", int(g["win"].sum()))
    print("Win rate:  ", f"{g['win'].mean():.2%}")
    print("Avg odds:  ", f"{g['odds'].mean():.3f}")
    print("Avg edge:  ", f"{g['raw_edge'].mean():.2%}")
    print("Profit:    ", f"{g['profit'].sum():+.2f}u")
    print("ROI:       ", f"{g['profit'].mean():+.2%}")


print()
print("=" * 120)
print("BROAD ≥16%")
print("=" * 120)

show("ALL", bets)


print()
print("=" * 120)
print("BY SELECTION")
print("=" * 120)

for side in ["H", "D", "A"]:
    show(side, bets[bets["selection"].eq(side)])


print()
print("=" * 120)
print("BY YEAR")
print("=" * 120)

yr = (
    bets.groupby("year")
    .agg(
        bets=("profit", "size"),
        wins=("win", "sum"),
        profit=("profit", "sum"),
        avg_odds=("odds", "mean"),
        avg_edge=("raw_edge", "mean"),
    )
)

yr["roi"] = yr["profit"] / yr["bets"]

print(
    yr.to_string(
        formatters={
            "profit": lambda x: f"{x:+.2f}",
            "avg_odds": lambda x: f"{x:.3f}",
            "avg_edge": lambda x: f"{x:.2%}",
            "roi": lambda x: f"{x:+.2%}",
        }
    )
)


print()
print("=" * 120)
print("SELECTION × YEAR")
print("=" * 120)

sy = (
    bets.groupby(["year", "selection"])
    .agg(
        bets=("profit", "size"),
        wins=("win", "sum"),
        profit=("profit", "sum"),
    )
)

sy["roi"] = sy["profit"] / sy["bets"]

print(
    sy.to_string(
        formatters={
            "profit": lambda x: f"{x:+.2f}",
            "roi": lambda x: f"{x:+.2%}",
        }
    )
)

print()
print("Saved:", OUT_FILE)
print("=" * 120)
