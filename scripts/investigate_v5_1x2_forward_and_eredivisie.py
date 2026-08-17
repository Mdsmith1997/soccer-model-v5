from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BET_FILE = (
    ROOT / "data" / "processed"
    / "v5_1x2_all_leagues_frozen16_bets.csv"
)

PRED_FILE = (
    ROOT / "data" / "processed"
    / "footystats_multileague_v5_predictions.csv"
)

MARKET_FILE = (
    ROOT / "data" / "processed"
    / "v5_1x2_football_data.csv"
)


# ============================================================
# HELPERS
# ============================================================

def perf(g):
    if len(g) == 0:
        return {
            "bets": 0,
            "wins": 0,
            "profit": 0.0,
            "roi": np.nan,
            "avg_odds": np.nan,
            "avg_edge": np.nan,
        }

    return {
        "bets": len(g),
        "wins": int(g["win"].sum()),
        "profit": float(g["profit"].sum()),
        "roi": float(g["profit"].mean()),
        "avg_odds": float(g["odds"].mean()),
        "avg_edge": float(g["raw_edge"].mean()),
    }


def fmt_perf(label, g):
    p = perf(g)

    print(
        f"{label:<28} "
        f"bets={p['bets']:>3} | "
        f"wins={p['wins']:>3} | "
        f"profit={p['profit']:+7.2f}u | "
        f"ROI={p['roi']:+8.2%} | "
        f"odds={p['avg_odds']:.3f} | "
        f"edge={p['avg_edge']:+.2%}"
    )


def normalize_date(df):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )
    return df


# ============================================================
# LOAD
# ============================================================

bets = normalize_date(pd.read_csv(BET_FILE))
pred = normalize_date(pd.read_csv(PRED_FILE))
market = normalize_date(pd.read_csv(MARKET_FILE))

for c in [
    "odds",
    "model_prob",
    "market_prob",
    "raw_edge",
    "profit",
    "win",
]:
    if c in bets.columns:
        bets[c] = pd.to_numeric(
            bets[c],
            errors="coerce",
        )


print("=" * 130)
print("PART 1 — LEAGUE TWO HOME TRUE TEMPORAL VALIDATION")
print("=" * 130)

lt = bets[
    bets["league"].eq("League Two")
    & bets["selection"].eq("H")
].copy()

lt = lt.sort_values("date").reset_index(drop=True)

print("\nFull candidate:")
fmt_perf("ALL", lt)

print("\nDate range:")
print("First:", lt["date"].min())
print("Last: ", lt["date"].max())


# ============================================================
# A. FIXED CALENDAR CUTOFFS
# ============================================================

print("\n" + "=" * 130)
print("FIXED CALENDAR CUTOFFS")
print("=" * 130)

years = sorted(
    lt["date"].dt.year
    .dropna()
    .unique()
)

rows = []

for cutoff in years[1:-1]:

    train = lt[
        lt["date"].dt.year <= cutoff
    ]

    forward = lt[
        lt["date"].dt.year > cutoff
    ]

    if len(train) < 10 or len(forward) < 10:
        continue

    a = perf(train)
    b = perf(forward)

    rows.append({
        "cutoff": int(cutoff),
        "train_bets": a["bets"],
        "train_roi": a["roi"],
        "forward_bets": b["bets"],
        "forward_roi": b["roi"],
        "forward_profit": b["profit"],
    })

cut = pd.DataFrame(rows)

if len(cut):
    print(
        cut.to_string(
            index=False,
            formatters={
                "train_roi": lambda x: f"{x:+.2%}",
                "forward_roi": lambda x: f"{x:+.2%}",
                "forward_profit": lambda x: f"{x:+.2f}",
            },
        )
    )


# ============================================================
# B. PURE CHRONOLOGICAL SPLITS
# ============================================================

print("\n" + "=" * 130)
print("PURE CHRONOLOGICAL HOLDOUTS")
print("=" * 130)

for frac in [0.50, 0.60, 0.67, 0.75]:

    split = int(len(lt) * frac)

    discovery = lt.iloc[:split]
    holdout = lt.iloc[split:]

    print(
        f"\nDISCOVERY {frac:.0%} / "
        f"HOLDOUT {1-frac:.0%}"
    )

    fmt_perf("Discovery", discovery)
    fmt_perf("Forward holdout", holdout)

    if len(holdout):
        print(
            "Holdout dates:",
            holdout["date"].min().date(),
            "to",
            holdout["date"].max().date(),
        )


# ============================================================
# C. YEARLY SIGNAL COUNTS + PERFORMANCE
# ============================================================

print("\n" + "=" * 130)
print("LEAGUE TWO HOME — SIGNALS BY YEAR")
print("=" * 130)

lt["year"] = lt["date"].dt.year

yearly = (
    lt.groupby("year")
    .agg(
        bets=("profit", "size"),
        wins=("win", "sum"),
        avg_odds=("odds", "mean"),
        avg_edge=("raw_edge", "mean"),
        profit=("profit", "sum"),
    )
)

yearly["roi"] = (
    yearly["profit"]
    / yearly["bets"]
)

print(
    yearly.to_string(
        formatters={
            "avg_odds": lambda x: f"{x:.3f}",
            "avg_edge": lambda x: f"{x:+.2%}",
            "profit": lambda x: f"{x:+.2f}",
            "roi": lambda x: f"{x:+.2%}",
        }
    )
)


# ============================================================
# D. ROLLING 20-BET WINDOWS
# ============================================================

print("\n" + "=" * 130)
print("LEAGUE TWO HOME — ROLLING 20-BET WINDOWS")
print("=" * 130)

if len(lt) >= 20:

    roll = []

    for end in range(20, len(lt) + 1):

        g = lt.iloc[end-20:end]
        p = perf(g)

        roll.append({
            "start_date": g["date"].iloc[0].date(),
            "end_date": g["date"].iloc[-1].date(),
            "profit": p["profit"],
            "roi": p["roi"],
        })

    roll = pd.DataFrame(roll)

    print("Windows:", len(roll))
    print(
        "Positive:",
        int((roll["roi"] > 0).sum()),
        "/",
        len(roll),
        f"({(roll['roi'] > 0).mean():.1%})",
    )

    print(
        "Median ROI:",
        f"{roll['roi'].median():+.2%}",
    )

    print(
        "Worst ROI:",
        f"{roll['roi'].min():+.2%}",
    )

    print(
        "Best ROI:",
        f"{roll['roi'].max():+.2%}",
    )

    print("\nLAST 10 WINDOWS")

    print(
        roll.tail(10).to_string(
            index=False,
            formatters={
                "profit": lambda x: f"{x:+.2f}",
                "roi": lambda x: f"{x:+.2%}",
            },
        )
    )


# ============================================================
# PART 2 — EREDIVISIE INVESTIGATION
# ============================================================

print("\n\n" + "=" * 130)
print("PART 2 — WHY DOES EREDIVISIE AWAY HAVE ONLY 18 BETS?")
print("=" * 130)

eb = bets[
    bets["league"].eq("Eredivisie")
].copy()

ep = pred[
    pred["league"].eq("Eredivisie")
].copy()

em = market[
    market["league"].eq("Eredivisie")
].copy()

print("\nDataset inventory:")
print("V5 predictions:", len(ep))
print("Market rows:", len(em))
print("Frozen >=16% bets:", len(eb))

print("\nPrediction date range:")
print(ep["date"].min(), "->", ep["date"].max())

print("\nMarket date range:")
print(em["date"].min(), "->", em["date"].max())

print("\nBet date range:")
print(eb["date"].min(), "->", eb["date"].max())


# ============================================================
# A. ALL FROZEN EREDIVISIE BETS BY YEAR + SIDE
# ============================================================

print("\n" + "=" * 130)
print("ALL EREDIVISIE >=16% BETS — YEAR × SIDE")
print("=" * 130)

eb["year"] = eb["date"].dt.year

pivot = pd.crosstab(
    eb["year"],
    eb["selection"],
)

print(pivot.to_string())


# ============================================================
# B. AWAY BET DETAILS
# ============================================================

print("\n" + "=" * 130)
print("THE 18 EREDIVISIE AWAY BETS")
print("=" * 130)

ea = eb[
    eb["selection"].eq("A")
].copy()

show_cols = [
    c for c in [
        "date",
        "season",
        "home_team",
        "away_team",
        "selection",
        "model_prob",
        "market_prob",
        "raw_edge",
        "odds",
        "win",
        "profit",
    ]
    if c in ea.columns
]

print(
    ea[show_cols]
    .sort_values("date")
    .to_string(
        index=False,
        formatters={
            "model_prob":
                lambda x: f"{x:.2%}",
            "market_prob":
                lambda x: f"{x:.2%}",
            "raw_edge":
                lambda x: f"{x:+.2%}",
            "odds":
                lambda x: f"{x:.3f}",
            "profit":
                lambda x: f"{x:+.2f}",
        },
    )
)


# ============================================================
# C. PREDICTION / MARKET COVERAGE BY YEAR
# ============================================================

print("\n" + "=" * 130)
print("EREDIVISIE SOURCE COVERAGE BY YEAR")
print("=" * 130)

ep["year"] = ep["date"].dt.year
em["year"] = em["date"].dt.year

pred_counts = (
    ep.groupby("year")
    .size()
    .rename("prediction_rows")
)

market_counts = (
    em.groupby("year")
    .size()
    .rename("market_rows")
)

bet_counts = (
    eb.groupby("year")
    .size()
    .rename("bets_ge16")
)

away_counts = (
    ea.groupby(ea["date"].dt.year)
    .size()
    .rename("away_bets_ge16")
)

coverage = pd.concat(
    [
        pred_counts,
        market_counts,
        bet_counts,
        away_counts,
    ],
    axis=1,
).fillna(0)

print(coverage.to_string())


# ============================================================
# D. RAW EDGE DISTRIBUTION FROM FROZEN BET POPULATION
# ============================================================

print("\n" + "=" * 130)
print("EREDIVISIE FROZEN BET EDGE DISTRIBUTION — YEAR × SIDE")
print("=" * 130)

edge_summary = (
    eb.groupby(["year", "selection"])
    .agg(
        bets=("raw_edge", "size"),
        min_edge=("raw_edge", "min"),
        median_edge=("raw_edge", "median"),
        mean_edge=("raw_edge", "mean"),
        max_edge=("raw_edge", "max"),
        avg_odds=("odds", "mean"),
    )
)

print(
    edge_summary.to_string(
        formatters={
            "min_edge": lambda x: f"{x:+.2%}",
            "median_edge": lambda x: f"{x:+.2%}",
            "mean_edge": lambda x: f"{x:+.2%}",
            "max_edge": lambda x: f"{x:+.2%}",
            "avg_odds": lambda x: f"{x:.3f}",
        }
    )
)


# ============================================================
# E. CHECK WHETHER >=16% SIGNALS EXIST AFTER 2018 AT ALL
# ============================================================

print("\n" + "=" * 130)
print("POST-2018 EREDIVISIE >=16% BET INVENTORY")
print("=" * 130)

post = eb[
    eb["date"].dt.year > 2018
].copy()

if post.empty:
    print("NONE")
else:

    print(
        post[
            [
                c for c in show_cols
                if c in post.columns
            ]
        ]
        .sort_values("date")
        .to_string(
            index=False,
            formatters={
                "model_prob":
                    lambda x: f"{x:.2%}",
                "market_prob":
                    lambda x: f"{x:.2%}",
                "raw_edge":
                    lambda x: f"{x:+.2%}",
                "odds":
                    lambda x: f"{x:.3f}",
                "profit":
                    lambda x: f"{x:+.2f}",
            },
        )
    )


# ============================================================
# F. SEASON DISTRIBUTION IN RAW V5 PREDICTIONS
# ============================================================

print("\n" + "=" * 130)
print("EREDIVISIE V5 PREDICTION INVENTORY BY YEAR")
print("=" * 130)

pred_inventory = (
    ep.groupby("year")
    .size()
)

print(pred_inventory.to_string())


# ============================================================
# G. MARKET INVENTORY BY YEAR
# ============================================================

print("\n" + "=" * 130)
print("EREDIVISIE FOOTBALL-DATA MARKET INVENTORY BY YEAR")
print("=" * 130)

market_inventory = (
    em.groupby("year")
    .size()
)

print(market_inventory.to_string())


print("\n" + "=" * 130)
print("DIAGNOSTIC COMPLETE")
print("=" * 130)

print("""
INTERPRETATION GUIDE

If prediction_rows and market_rows remain healthy after 2018,
but away_bets_ge16 collapse:
    -> the V5 model simply stopped producing >=16% AWAY edges.

If prediction_rows collapse:
    -> V5 source/prediction coverage is the problem.

If market_rows collapse:
    -> Football-Data odds coverage is the problem.

If total >=16% bets remain high but AWAY disappears:
    -> the edge shifted toward HOME selections.

If both predictions and market data are healthy but all
>=16% signals collapse:
    -> investigate probability/edge distribution by season
       and possible model/data regime changes.
""")
