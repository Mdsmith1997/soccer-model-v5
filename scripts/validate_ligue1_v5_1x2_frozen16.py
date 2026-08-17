from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata
import re

ROOT = Path("data/processed")

PRED_FILE = ROOT / "footystats_ligue1_v5_predictions_research.csv"
MARKET_FILE = ROOT / "v5_1x2_football_data.csv"
OUT_FILE = ROOT / "ligue1_v5_1x2_frozen16_research.csv"

EDGE_THRESHOLD = 0.16


def norm_team(x):
    x = str(x).strip().lower()
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = x.replace("&", "and")
    x = re.sub(r"[^a-z0-9]+", "", x)

    aliases = {
        # PSG
        "parissaintgermainfc": "psg",
        "parissaintgermain": "psg",
        "parissg": "psg",
        "psg": "psg",

        # Marseille
        "olympiquedemarseille": "marseille",
        "olympiquemarseille": "marseille",
        "marseille": "marseille",

        # Lyon
        "olympiquelyonnais": "lyon",
        "lyon": "lyon",

        # Monaco
        "asmonacofc": "monaco",
        "monaco": "monaco",

        # Lille
        "losclille": "lille",
        "lille": "lille",

        # Saint-Etienne
        "asstetienne": "saintetienne",
        "saintetienne": "saintetienne",
        "stetienne": "saintetienne",

        # Amiens
        "amienssc": "amiens",
        "amiens": "amiens",

        # Angers
        "angerssco": "angers",
        "angers": "angers",

        # Paris FC — NOT PSG
        "paris": "parisfc",
        "parisfc": "parisfc",
    }

    return aliases.get(x, x)


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def summarize(g):
    n = len(g)

    if n == 0:
        return pd.Series({
            "bets": 0,
            "wins": 0,
            "win_pct": np.nan,
            "avg_odds": np.nan,
            "avg_edge": np.nan,
            "profit": 0.0,
            "roi": np.nan,
        })

    return pd.Series({
        "bets": n,
        "wins": int(g["win"].sum()),
        "win_pct": g["win"].mean(),
        "avg_odds": g["odds"].mean(),
        "avg_edge": g["raw_edge"].mean(),
        "profit": g["profit"].sum(),
        "roi": g["profit"].sum() / n,
    })


pred = pd.read_csv(PRED_FILE, low_memory=False)
market = pd.read_csv(MARKET_FILE, low_memory=False)

pred = pred[
    pred["league"].astype(str).str.strip().eq("Ligue 1")
].copy()

market = market[
    market["league"].astype(str).str.strip().eq("Ligue 1")
].copy()

print("=" * 120)
print("LIGUE 1 — CORRECTED FROZEN V5 1X2 ≥16%")
print("=" * 120)

print("Prediction rows:", len(pred))
print("Market rows:    ", len(market))


ph = find_col(pred, ["p_home", "home_win_prob", "prob_home"])
pd_ = find_col(pred, ["p_draw", "draw_prob", "prob_draw"])
pa = find_col(pred, ["p_away", "away_win_prob", "prob_away"])

mh = find_col(market, ["B365H", "b365_home", "home_odds"])
md = find_col(market, ["B365D", "b365_draw", "draw_odds"])
ma = find_col(market, ["B365A", "b365_away", "away_odds"])

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


pred["date_key"] = pd.to_datetime(
    pred["date"], errors="coerce"
).dt.normalize()

market["date_key"] = pd.to_datetime(
    market["date"], errors="coerce"
).dt.normalize()

pred["home_key"] = pred["home_team"].map(norm_team)
pred["away_key"] = pred["away_team"].map(norm_team)

market["home_key"] = market["home_team"].map(norm_team)
market["away_key"] = market["away_team"].map(norm_team)

pred = pred.dropna(subset=["date_key"])
market = market.dropna(subset=["date_key"])


merged = pred.merge(
    market[
        [
            "date_key",
            "home_key",
            "away_key",
            mh,
            md,
            ma,
            result_col,
        ]
    ],
    on=["date_key", "home_key", "away_key"],
    how="left",
    indicator=True,
)

exact_matches = int(
    (merged["_merge"] == "both").sum()
)

print()
print("Exact matched:", exact_matches)
print("Exact rate:   ", f"{exact_matches / len(pred):.2%}")
print("Unmatched:    ", len(pred) - exact_matches)


for c in [ph, pd_, pa, mh, md, ma]:
    merged[c] = pd.to_numeric(
        merged[c],
        errors="coerce"
    )


valid = merged[
    (merged["_merge"] == "both")
    & merged[[ph, pd_, pa, mh, md, ma]].notna().all(axis=1)
    & (merged[[mh, md, ma]] > 1.0).all(axis=1)
].copy()

print("Valid odds:   ", len(valid))
print("Coverage:     ", f"{len(valid) / len(pred):.2%}")


print()
print("=" * 120)
print("COVERAGE BY SEASON")
print("=" * 120)

coverage = (
    merged.assign(
        exact=merged["_merge"].eq("both")
    )
    .groupby("season")
    .agg(
        predictions=("exact", "size"),
        exact_matches=("exact", "sum"),
    )
    .reset_index()
)

coverage["exact_pct"] = (
    coverage["exact_matches"]
    / coverage["predictions"]
)

print(
    coverage.to_string(
        index=False,
        formatters={
            "exact_pct": lambda v: f"{v:.2%}"
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


# Frozen 16% threshold
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
    bets["selection"]
    == bets["actual_result"]
).astype(int)

bets["profit"] = np.where(
    bets["win"].eq(1),
    bets["odds"] - 1,
    -1.0,
)

bets.to_csv(OUT_FILE, index=False)


print()
print("=" * 120)
print("BROAD ≥16%")
print("=" * 120)

broad = summarize(bets)

for k, v in broad.items():

    if k in ["win_pct", "avg_edge", "roi"]:
        print(f"{k:12s}: {v:.2%}")

    elif k in ["avg_odds", "profit"]:
        print(f"{k:12s}: {v:.3f}")

    else:
        print(f"{k:12s}: {v}")


print()
print("=" * 120)
print("BY SELECTION")
print("=" * 120)

side = (
    bets.groupby("selection")
    .apply(summarize, include_groups=False)
    .reset_index()
)

print(
    side.to_string(
        index=False,
        formatters={
            "win_pct": lambda v: f"{v:.2%}",
            "avg_edge": lambda v: f"{v:.2%}",
            "roi": lambda v: f"{v:+.2%}",
            "avg_odds": lambda v: f"{v:.3f}",
            "profit": lambda v: f"{v:+.2f}",
        }
    )
)


print()
print("=" * 120)
print("BROAD BY SEASON")
print("=" * 120)

season = (
    bets.groupby("season")
    .apply(summarize, include_groups=False)
    .reset_index()
)

print(
    season.to_string(
        index=False,
        formatters={
            "win_pct": lambda v: f"{v:.2%}",
            "avg_edge": lambda v: f"{v:.2%}",
            "roi": lambda v: f"{v:+.2%}",
            "avg_odds": lambda v: f"{v:.3f}",
            "profit": lambda v: f"{v:+.2f}",
        }
    )
)


print()
print("=" * 120)
print("SELECTION × SEASON")
print("=" * 120)

ss = (
    bets.groupby(["selection", "season"])
    .apply(summarize, include_groups=False)
    .reset_index()
)

print(
    ss.to_string(
        index=False,
        formatters={
            "win_pct": lambda v: f"{v:.2%}",
            "avg_edge": lambda v: f"{v:.2%}",
            "roi": lambda v: f"{v:+.2%}",
            "avg_odds": lambda v: f"{v:.3f}",
            "profit": lambda v: f"{v:+.2f}",
        }
    )
)

print()
print("Saved:", OUT_FILE)
print("=" * 120)
