from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("data/processed")
FILE = ROOT / "eredivisie_v5_1x2_frozen16_corrected.csv"

N_BOOT = 20000
SEED = 42

df = pd.read_csv(FILE, low_memory=False)

df["date"] = pd.to_datetime(
    df["date_key"] if "date_key" in df.columns else df["date"],
    errors="coerce"
)

df = df.sort_values("date").reset_index(drop=True)

print("=" * 110)
print("EREDIVISIE BROAD — FROZEN V5 ≥16% ROBUSTNESS")
print("=" * 110)

def perf(g):
    if len(g) == 0:
        return None

    return {
        "bets": len(g),
        "wins": int(g["win"].sum()),
        "profit": float(g["profit"].sum()),
        "roi": float(g["profit"].mean()),
        "avg_odds": float(g["odds"].mean()),
        "avg_edge": float(g["raw_edge"].mean()),
    }

def show(label, g):
    p = perf(g)

    if p is None:
        print(f"{label:<28} NO BETS")
        return

    print(
        f"{label:<28}"
        f"bets={p['bets']:>3} | "
        f"wins={p['wins']:>3} | "
        f"profit={p['profit']:+7.2f}u | "
        f"ROI={p['roi']:+8.2%} | "
        f"odds={p['avg_odds']:.3f} | "
        f"edge={p['avg_edge']:.2%}"
    )


print()
print("FULL SAMPLE")
show("All", df)


# ============================================================
# YEAR
# ============================================================

df["year"] = df["date"].dt.year

print()
print("=" * 110)
print("BY YEAR")
print("=" * 110)

for year, g in df.groupby("year"):
    show(str(year), g)


# ============================================================
# FIRST / SECOND HALF
# ============================================================

mid = len(df) // 2

print()
print("=" * 110)
print("FIRST / SECOND HALF")
print("=" * 110)

show("First half", df.iloc[:mid])
show("Second half", df.iloc[mid:])


# ============================================================
# RECENT WINDOWS
# ============================================================

print()
print("=" * 110)
print("RECENT WINDOWS")
print("=" * 110)

for n in [20, 25, 40, 50]:
    if len(df) >= n:
        show(f"Recent {n}", df.tail(n))


# ============================================================
# CHRONOLOGICAL HOLDOUTS
# ============================================================

print()
print("=" * 110)
print("CHRONOLOGICAL HOLDOUTS")
print("=" * 110)

for frac in [0.50, 0.60, 0.67, 0.70, 0.75, 0.80]:

    cut = int(len(df) * frac)

    train = df.iloc[:cut]
    test = df.iloc[cut:]

    print()
    print(
        f"{int(frac*100):>2}% / "
        f"{100-int(frac*100):>2}% split"
    )

    show("Train", train)
    show("Holdout", test)


# ============================================================
# LEAVE ONE YEAR OUT
# ============================================================

print()
print("=" * 110)
print("LEAVE-ONE-YEAR-OUT")
print("=" * 110)

for year in sorted(df["year"].dropna().unique()):

    g = df[df["year"] != year]

    show(f"Remove {int(year)}", g)


# ============================================================
# WINNER DEPENDENCY
# ============================================================

print()
print("=" * 110)
print("REMOVE TOP WINNERS")
print("=" * 110)

show("Original", df)

wins = df[
    df["profit"] > 0
].sort_values(
    "profit",
    ascending=False
)

for n in [1, 2, 3, 5]:

    remove_idx = wins.head(n).index

    g = df.drop(remove_idx)

    show(f"Remove top {n}", g)


print()
print("TOP 10 WINNERS")

cols = [
    c for c in [
        "date",
        "home_team",
        "away_team",
        "selection",
        "odds",
        "raw_edge",
        "profit",
    ]
    if c in wins.columns
]

print(
    wins[cols]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# BOOTSTRAP
# ============================================================

print()
print("=" * 110)
print("BOOTSTRAP")
print("=" * 110)

rng = np.random.default_rng(SEED)

profits = df["profit"].to_numpy()
n = len(profits)

boot_roi = np.empty(N_BOOT)

for i in range(N_BOOT):

    sample = rng.choice(
        profits,
        size=n,
        replace=True
    )

    boot_roi[i] = sample.mean()

print("Iterations:       ", N_BOOT)
print("Observed ROI:     ", f"{profits.mean():+.2%}")
print("P(ROI > 0):       ", f"{(boot_roi > 0).mean():.2%}")
print("5th percentile:   ", f"{np.percentile(boot_roi,5):+.2%}")
print("Median:           ", f"{np.percentile(boot_roi,50):+.2%}")
print("95th percentile:  ", f"{np.percentile(boot_roi,95):+.2%}")


# ============================================================
# MAX DRAWDOWN / LOSING STREAK
# ============================================================

print()
print("=" * 110)
print("RISK")
print("=" * 110)

cum = df["profit"].cumsum()
peak = cum.cummax()
drawdown = cum - peak

print("Max drawdown:", f"{drawdown.min():+.2f}u")

max_losing = 0
current = 0

for p in df["profit"]:

    if p < 0:
        current += 1
        max_losing = max(
            max_losing,
            current
        )
    else:
        current = 0

print("Max losing streak:", max_losing)


# ============================================================
# SIDE CONTRIBUTION
# ============================================================

print()
print("=" * 110)
print("SIDE CONTRIBUTION")
print("=" * 110)

for side in ["H", "D", "A"]:
    show(
        side,
        df[df["selection"].eq(side)]
    )


# ============================================================
# HOME-ONLY WITHOUT AWAY CONTRIBUTION
# ============================================================

print()
print("=" * 110)
print("BROAD WITHOUT AWAY BETS")
print("=" * 110)

show(
    "Home/Draw only",
    df[~df["selection"].eq("A")]
)

print()
print("=" * 110)
print("DONE")
print("=" * 110)
