import pandas as pd
import numpy as np

PATH = "data/processed/eliteserien_v5_t24_totals_games.csv"

df = pd.read_csv(PATH)
df = df[df["has_exact_25"] == True].copy()

# Derive OVER quantities from existing UNDER quantities.
df["model_over_prob"] = 1 - df["model_under_prob"]
df["market_over_prob"] = 1 - df["consensus_market_under_prob"]
df["over_edge"] = df["model_over_prob"] - df["market_over_prob"]
df["over_win"] = (df["actual_total"] > 2.5).astype(int)

# Best over price stored from same bookmaker as best-under execution.
df["over_odds"] = pd.to_numeric(
    df["best_over_odds_same_book"], errors="coerce"
)

df = df[df["over_odds"].notna()].copy()

df["over_profit"] = np.where(
    df["over_win"] == 1,
    df["over_odds"] - 1,
    -1.0,
)

def summarize(x):
    if len(x) == 0:
        return None
    return {
        "bets": len(x),
        "wins": int(x["over_win"].sum()),
        "win_rate": x["over_win"].mean(),
        "avg_odds": x["over_odds"].mean(),
        "avg_edge": x["over_edge"].mean(),
        "profit": x["over_profit"].sum(),
        "roi": x["over_profit"].mean(),
    }

def show(label, x):
    s = summarize(x)
    if s is None:
        print(label, "| no bets")
        return
    print(
        f"{label:<28} "
        f"bets={s['bets']:>3} | "
        f"wins={s['wins']:>3} | "
        f"WR={s['win_rate']:.2%} | "
        f"odds={s['avg_odds']:.3f} | "
        f"edge={s['avg_edge']:.2%} | "
        f"profit={s['profit']:+.2f}u | "
        f"ROI={s['roi']:+.2%}"
    )

sig = df[df["over_edge"] >= .11].copy()

print("=" * 120)
print("ELITESERIEN O2.5 >=11% ROBUSTNESS AUDIT")
print("=" * 120)

show("FULL SAMPLE", sig)

print("\n" + "=" * 120)
print("LEAVE-ONE-SEASON-OUT")
print("=" * 120)

for season in sorted(sig["season"].unique()):
    show(
        f"exclude {season}",
        sig[sig["season"] != season]
    )

print("\n" + "=" * 120)
print("2024 PRICE BANDS")
print("=" * 120)

x = sig[sig["season"] == 2024].copy()

price_bins = [
    (0, 1.60),
    (1.60, 1.70),
    (1.70, 1.80),
    (1.80, 1.90),
    (1.90, 99),
]

for lo, hi in price_bins:
    y = x[
        (x["over_odds"] >= lo) &
        (x["over_odds"] < hi)
    ]
    show(f"{lo:.2f}-{hi:.2f}", y)

print("\n" + "=" * 120)
print("2024 EDGE BANDS")
print("=" * 120)

edge_bins = [
    (.11, .13),
    (.13, .15),
    (.15, .17),
    (.17, .20),
    (.20, 1),
]

for lo, hi in edge_bins:
    y = x[
        (x["over_edge"] >= lo) &
        (x["over_edge"] < hi)
    ]
    show(f"{lo:.0%}-{hi:.0%}", y)

print("\n" + "=" * 120)
print(">=11% BY TEAM — ALL SEASONS")
print("=" * 120)

rows = []

teams = sorted(
    set(sig["home_team"]).union(sig["away_team"])
)

for team in teams:
    y = sig[
        (sig["home_team"] == team) |
        (sig["away_team"] == team)
    ]

    if len(y) >= 5:
        s = summarize(y)
        rows.append({
            "team": team,
            "bets": s["bets"],
            "wins": s["wins"],
            "roi": s["roi"],
            "profit": s["profit"],
        })

out = pd.DataFrame(rows).sort_values(
    ["roi", "bets"],
    ascending=[False, False]
)

print(
    out.to_string(
        index=False,
        formatters={
            "roi": lambda x: f"{x:+.2%}",
            "profit": lambda x: f"{x:+.2f}u",
        }
    )
)

print("\n" + "=" * 120)
print(">=11% — 2024 TEAM EXPOSURE")
print("=" * 120)

rows = []

for team in sorted(
    set(x["home_team"]).union(x["away_team"])
):
    y = x[
        (x["home_team"] == team) |
        (x["away_team"] == team)
    ]

    if len(y) >= 2:
        s = summarize(y)
        rows.append({
            "team": team,
            "bets": s["bets"],
            "wins": s["wins"],
            "roi": s["roi"],
            "profit": s["profit"],
        })

out = pd.DataFrame(rows).sort_values(
    ["roi", "bets"],
    ascending=[False, False]
)

print(
    out.to_string(
        index=False,
        formatters={
            "roi": lambda x: f"{x:+.2%}",
            "profit": lambda x: f"{x:+.2f}u",
        }
    )
)
