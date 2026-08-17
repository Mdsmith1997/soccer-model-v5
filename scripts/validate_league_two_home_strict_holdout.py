from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILE = (
    ROOT
    / "data"
    / "processed"
    / "v5_1x2_all_leagues_frozen16_bets_history_audited.csv"
)

df = pd.read_csv(FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)

for c in [
    "profit",
    "win",
    "odds",
    "raw_edge",
]:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce",
    )

g = df[
    df["league"].eq("League Two")
    & df["selection"].eq("H")
    & df["history_class"].eq("BOTH_SAME_LEAGUE")
].copy()

g = g.sort_values("date").reset_index(drop=True)


def perf(x):
    n = len(x)

    if n == 0:
        return {
            "bets": 0,
            "wins": 0,
            "profit": 0.0,
            "roi": np.nan,
            "avg_odds": np.nan,
            "avg_edge": np.nan,
        }

    return {
        "bets": n,
        "wins": int(x["win"].sum()),
        "profit": float(x["profit"].sum()),
        "roi": float(x["profit"].mean()),
        "avg_odds": float(x["odds"].mean()),
        "avg_edge": float(x["raw_edge"].mean()),
    }


def show(label, x):
    p = perf(x)

    print(
        f"{label:<30}"
        f" bets={p['bets']:>3}"
        f" | wins={p['wins']:>3}"
        f" | profit={p['profit']:+7.2f}u"
        f" | ROI={p['roi']:+8.2%}"
        f" | odds={p['avg_odds']:.3f}"
        f" | edge={p['avg_edge']:+.2%}"
    )


def bootstrap(x, n=20000, seed=42):
    vals = x["profit"].to_numpy(dtype=float)

    if len(vals) == 0:
        return None

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(vals),
        size=(n, len(vals)),
    )

    rois = vals[idx].mean(axis=1)

    return {
        "p_gt_0": float((rois > 0).mean()),
        "p05": float(np.quantile(rois, 0.05)),
        "p50": float(np.quantile(rois, 0.50)),
        "p95": float(np.quantile(rois, 0.95)),
    }


print("=" * 125)
print("LEAGUE TWO HOME >=16% — STRICT HISTORY-VALID TEMPORAL HOLDOUT")
print("=" * 125)

print("\nRULE:")
print("League Two")
print("HOME only")
print("raw V5 edge >= 16%")
print("history_class = BOTH_SAME_LEAGUE")
print("flat 1u")
print("no odds filter")
print("no threshold tuning")

print("\nFull sample:")
show("ALL", g)

print("\nDate range:")
print(g["date"].min(), "->", g["date"].max())


# ============================================================
# YEARLY
# ============================================================

g["year"] = g["date"].dt.year

print("\n" + "=" * 125)
print("YEAR BY YEAR")
print("=" * 125)

for year, y in g.groupby("year"):
    show(str(int(year)), y)


# ============================================================
# FIXED YEAR CUTOFFS
# ============================================================

print("\n" + "=" * 125)
print("FIXED YEAR HOLDOUTS")
print("=" * 125)

rows = []

years = sorted(g["year"].unique())

for cutoff in years[:-1]:

    train = g[
        g["year"] <= cutoff
    ]

    hold = g[
        g["year"] > cutoff
    ]

    if len(train) < 10 or len(hold) < 5:
        continue

    a = perf(train)
    b = perf(hold)

    rows.append({
        "cutoff": int(cutoff),
        "train_bets": a["bets"],
        "train_roi": a["roi"],
        "holdout_bets": b["bets"],
        "holdout_profit": b["profit"],
        "holdout_roi": b["roi"],
    })

cut = pd.DataFrame(rows)

print(
    cut.to_string(
        index=False,
        formatters={
            "train_roi": lambda x: f"{x:+.2%}",
            "holdout_profit": lambda x: f"{x:+.2f}",
            "holdout_roi": lambda x: f"{x:+.2%}",
        },
    )
)


# ============================================================
# PURE CHRONOLOGICAL SPLITS
# ============================================================

print("\n" + "=" * 125)
print("PURE CHRONOLOGICAL SPLITS")
print("=" * 125)

for frac in [0.50, 0.60, 0.67, 0.70, 0.75, 0.80]:

    split = int(len(g) * frac)

    train = g.iloc[:split]
    hold = g.iloc[split:]

    print(
        f"\nDISCOVERY {frac:.0%} / "
        f"HOLDOUT {1-frac:.0%}"
    )

    show("Discovery", train)
    show("Holdout", hold)

    if len(hold):
        print(
            "Holdout dates:",
            hold["date"].min().date(),
            "->",
            hold["date"].max().date(),
        )


# ============================================================
# RECENT WINDOWS
# ============================================================

print("\n" + "=" * 125)
print("RECENT WINDOWS")
print("=" * 125)

for n in [10, 15, 20, 25, 30]:

    if len(g) >= n:
        show(
            f"Recent {n}",
            g.tail(n),
        )


# ============================================================
# ROLLING 15 / 20 BET WINDOWS
# ============================================================

for window in [15, 20]:

    print("\n" + "=" * 125)
    print(f"ROLLING {window}-BET WINDOWS")
    print("=" * 125)

    if len(g) < window:
        continue

    vals = []

    for end in range(window, len(g) + 1):

        x = g.iloc[end-window:end]

        p = perf(x)

        vals.append({
            "start": x["date"].iloc[0],
            "end": x["date"].iloc[-1],
            "roi": p["roi"],
            "profit": p["profit"],
        })

    r = pd.DataFrame(vals)

    print("Windows:", len(r))
    print(
        "Positive:",
        int((r["roi"] > 0).sum()),
        "/",
        len(r),
        f"({(r['roi'] > 0).mean():.1%})",
    )
    print(
        "Median ROI:",
        f"{r['roi'].median():+.2%}",
    )
    print(
        "Worst ROI:",
        f"{r['roi'].min():+.2%}",
    )
    print(
        "Best ROI:",
        f"{r['roi'].max():+.2%}",
    )

    print("\nLast 10:")
    print(
        r.tail(10).to_string(
            index=False,
            formatters={
                "roi": lambda x: f"{x:+.2%}",
                "profit": lambda x: f"{x:+.2f}",
            },
        )
    )


# ============================================================
# REMOVE BEST WINNERS
# ============================================================

print("\n" + "=" * 125)
print("WINNER CONCENTRATION")
print("=" * 125)

show("Original", g)

winners = g[
    g["profit"] > 0
].sort_values(
    "profit",
    ascending=False,
)

for k in [1, 2, 3]:

    drop = winners.head(k).index

    x = g.drop(index=drop)

    show(
        f"Remove top {k}",
        x,
    )


# ============================================================
# BOOTSTRAP
# ============================================================

print("\n" + "=" * 125)
print("BOOTSTRAP")
print("=" * 125)

b = bootstrap(g)

if b:
    print(
        "P(ROI > 0):",
        f"{b['p_gt_0']:.1%}",
    )
    print(
        "5th percentile:",
        f"{b['p05']:+.2%}",
    )
    print(
        "Median:",
        f"{b['p50']:+.2%}",
    )
    print(
        "95th percentile:",
        f"{b['p95']:+.2%}",
    )


# ============================================================
# STRICT POST-2018 VIEW
# ============================================================

print("\n" + "=" * 125)
print("POST-2018 ONLY")
print("=" * 125)

post18 = g[
    g["year"] >= 2019
]

show("2019+", post18)

b2 = bootstrap(
    post18,
    seed=99,
)

if b2:
    print(
        "P(ROI > 0):",
        f"{b2['p_gt_0']:.1%}",
    )
    print(
        "5th percentile:",
        f"{b2['p05']:+.2%}",
    )
    print(
        "Median:",
        f"{b2['p50']:+.2%}",
    )
    print(
        "95th percentile:",
        f"{b2['p95']:+.2%}",
    )


# ============================================================
# SIMPLE PROMOTION CHECK
# ============================================================

print("\n" + "=" * 125)
print("PROMOTION CHECK")
print("=" * 125)

criteria = {}

criteria["full_positive"] = perf(g)["roi"] > 0
criteria["post2018_positive"] = perf(post18)["roi"] > 0
criteria["recent20_positive"] = (
    perf(g.tail(20))["roi"] > 0
    if len(g) >= 20
    else False
)

criteria["remove_best_positive"] = (
    perf(
        g.drop(
            index=winners.head(1).index
        )
    )["roi"] > 0
)

criteria["bootstrap_gt_80"] = (
    b["p_gt_0"] >= 0.80
    if b
    else False
)

for k, v in criteria.items():
    print(
        f"{k:<30}",
        "PASS" if v else "FAIL",
    )

print(
    "\nPassed:",
    sum(criteria.values()),
    "/",
    len(criteria),
)


print("\n" + "=" * 125)
print("DONE")
print("=" * 125)
