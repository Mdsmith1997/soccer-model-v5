import pandas as pd
import numpy as np

PATH = "data/processed/mls_v5_t24_totals_games.csv"

df = pd.read_csv(PATH)

# Exact O/U 2.5 market only
df = df[
    (df["has_exact_25"] == True) &
    df["consensus_market_under_prob"].notna() &
    df["best_under_odds"].notna() &
    df["best_over_odds_same_book"].notna()
].copy()

# Reconstruct Over side
df["model_over_prob"] = 1.0 - df["model_under_prob"]
df["consensus_market_over_prob"] = 1.0 - df["consensus_market_under_prob"]
df["consensus_over_edge"] = (
    df["model_over_prob"] - df["consensus_market_over_prob"]
)

df["over_25_win"] = df["actual_total"] > 2.5

thresholds = [
    0.03, 0.04, 0.05, 0.06, 0.07,
    0.08, 0.09, 0.10, 0.11, 0.12,
    0.13, 0.14, 0.15
]

def summarize(data, win_col, odds_col):
    if len(data) == 0:
        return {
            "bets": 0,
            "wins": 0,
            "win_rate": np.nan,
            "avg_odds": np.nan,
            "profit": 0,
            "roi": np.nan
        }

    wins = data[win_col].astype(bool)

    profit = np.where(
        wins,
        data[odds_col] - 1.0,
        -1.0
    ).sum()

    return {
        "bets": len(data),
        "wins": int(wins.sum()),
        "win_rate": wins.mean(),
        "avg_odds": data[odds_col].mean(),
        "profit": profit,
        "roi": profit / len(data)
    }


print("=" * 110)
print("MLS O/U 2.5 SPECIALIST DISCOVERY")
print("=" * 110)

print(f"\nExact 2.5 games analyzed: {len(df)}")


# ============================================================
# UNDER SWEEP
# ============================================================

print("\n" + "=" * 110)
print("UNDER 2.5 — CONSENSUS EDGE SWEEP")
print("=" * 110)

under_results = []

for t in thresholds:

    bets = df[df["consensus_under_edge"] >= t].copy()

    s = summarize(
        bets,
        "under_25_win",
        "best_under_odds"
    )

    under_results.append({
        "threshold": t,
        **s
    })

under_table = pd.DataFrame(under_results)

print(
    under_table.to_string(
        index=False,
        formatters={
            "threshold": lambda x: f"{x:.0%}",
            "win_rate": lambda x: f"{x:.2%}" if pd.notna(x) else "-",
            "avg_odds": lambda x: f"{x:.3f}" if pd.notna(x) else "-",
            "profit": lambda x: f"{x:+.2f}u",
            "roi": lambda x: f"{x:+.2%}" if pd.notna(x) else "-"
        }
    )
)


# ============================================================
# OVER SWEEP
# ============================================================

print("\n" + "=" * 110)
print("OVER 2.5 — CONSENSUS EDGE SWEEP")
print("=" * 110)

over_results = []

for t in thresholds:

    bets = df[df["consensus_over_edge"] >= t].copy()

    s = summarize(
        bets,
        "over_25_win",
        "best_over_odds_same_book"
    )

    over_results.append({
        "threshold": t,
        **s
    })

over_table = pd.DataFrame(over_results)

print(
    over_table.to_string(
        index=False,
        formatters={
            "threshold": lambda x: f"{x:.0%}",
            "win_rate": lambda x: f"{x:.2%}" if pd.notna(x) else "-",
            "avg_odds": lambda x: f"{x:.3f}" if pd.notna(x) else "-",
            "profit": lambda x: f"{x:+.2f}u",
            "roi": lambda x: f"{x:+.2%}" if pd.notna(x) else "-"
        }
    )
)


# ============================================================
# SEASON BREAKDOWN FOR EVERY THRESHOLD >= 8%
# ============================================================

for side in ["UNDER", "OVER"]:

    print("\n\n" + "#" * 110)
    print(f"{side} 2.5 — SEASON STABILITY")
    print("#" * 110)

    for t in thresholds:

        if t < 0.08:
            continue

        if side == "UNDER":
            bets = df[df["consensus_under_edge"] >= t].copy()
            win_col = "under_25_win"
            odds_col = "best_under_odds"
        else:
            bets = df[df["consensus_over_edge"] >= t].copy()
            win_col = "over_25_win"
            odds_col = "best_over_odds_same_book"

        if len(bets) == 0:
            continue

        print(f"\n{side} >= {t:.0%} EDGE")

        rows = []

        for season, g in bets.groupby("season"):

            s = summarize(
                g,
                win_col,
                odds_col
            )

            rows.append({
                "season": season,
                **s
            })

        table = pd.DataFrame(rows)

        print(
            table.to_string(
                index=False,
                formatters={
                    "win_rate": lambda x: f"{x:.2%}",
                    "avg_odds": lambda x: f"{x:.3f}",
                    "profit": lambda x: f"{x:+.2f}u",
                    "roi": lambda x: f"{x:+.2%}"
                }
            )
        )


# ============================================================
# EDGE BANDS — NON-OVERLAPPING
# ============================================================

print("\n\n" + "=" * 110)
print("NON-OVERLAPPING EDGE BANDS")
print("=" * 110)

bands = [
    (0.03, 0.05),
    (0.05, 0.07),
    (0.07, 0.09),
    (0.09, 0.11),
    (0.11, 0.13),
    (0.13, 0.15),
    (0.15, 1.00)
]

for side in ["UNDER", "OVER"]:

    print(f"\n{side} 2.5")

    rows = []

    edge_col = (
        "consensus_under_edge"
        if side == "UNDER"
        else "consensus_over_edge"
    )

    win_col = (
        "under_25_win"
        if side == "UNDER"
        else "over_25_win"
    )

    odds_col = (
        "best_under_odds"
        if side == "UNDER"
        else "best_over_odds_same_book"
    )

    for lo, hi in bands:

        bets = df[
            (df[edge_col] >= lo) &
            (df[edge_col] < hi)
        ].copy()

        s = summarize(
            bets,
            win_col,
            odds_col
        )

        label = (
            f"{lo:.0%}-{hi:.0%}"
            if hi < 1
            else f"{lo:.0%}+"
        )

        rows.append({
            "band": label,
            **s
        })

    table = pd.DataFrame(rows)

    print(
        table.to_string(
            index=False,
            formatters={
                "win_rate": lambda x: f"{x:.2%}" if pd.notna(x) else "-",
                "avg_odds": lambda x: f"{x:.3f}" if pd.notna(x) else "-",
                "profit": lambda x: f"{x:+.2f}u",
                "roi": lambda x: f"{x:+.2%}" if pd.notna(x) else "-"
            }
        )
    )

