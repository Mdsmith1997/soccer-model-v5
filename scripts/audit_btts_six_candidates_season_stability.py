from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

STABILITY_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_all_leagues_season_stability.csv"
)

RESULT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_all_leagues_threshold_results.csv"
)

CANDIDATES = {
    ("Swiss Super League", "YES"): [0.04, 0.06, 0.08, 0.10],
    ("Super Lig", "YES"): [0.04, 0.06, 0.08, 0.10, 0.12],
    ("Segunda División", "YES"): [0.00, 0.02, 0.04, 0.06],
    ("Championship", "NO"): [0.06, 0.08, 0.10, 0.12],
    ("Super Lig", "NO"): [0.02, 0.04, 0.06, 0.08, 0.10],
    ("Serie A", "YES"): [0.10, 0.12, 0.14, 0.16],
}

stab = pd.read_csv(STABILITY_FILE, low_memory=False)
res = pd.read_csv(RESULT_FILE, low_memory=False)

for c in ["edge", "roi", "profit_u", "win_rate", "avg_odds"]:
    if c in stab.columns:
        stab[c] = pd.to_numeric(stab[c], errors="coerce")
    if c in res.columns:
        res[c] = pd.to_numeric(res[c], errors="coerce")

for c in ["bets", "wins", "season"]:
    if c in stab.columns:
        stab[c] = pd.to_numeric(stab[c], errors="coerce")

print("=" * 130)
print("BTTS SIX-CANDIDATE SEASON ROBUSTNESS AUDIT")
print("=" * 130)

summary_rows = []

for (league, side), thresholds in CANDIDATES.items():

    print()
    print("=" * 130)
    print(f"{league} — BTTS {side}")
    print("=" * 130)

    for threshold in thresholds:

        overall = res[
            res["league"].eq(league)
            & res["side"].eq(side)
            & np.isclose(res["edge"], threshold)
        ].copy()

        seas = stab[
            stab["league"].eq(league)
            & stab["side"].eq(side)
            & np.isclose(stab["edge"], threshold)
        ].copy()

        seas = seas.sort_values("season")

        print()
        print(f"EDGE >= {threshold:.0%}")
        print("-" * 90)

        if overall.empty:
            print("NO OVERALL SAMPLE")
            continue

        o = overall.iloc[0]

        print(
            f"OVERALL | bets={int(o['bets'])} "
            f"| profit={o['profit_u']:+.2f}u "
            f"| ROI={o['roi']:+.2%} "
            f"| avg odds={o['avg_odds']:.2f}"
        )

        if seas.empty:
            print("NO SEASON DATA")
            continue

        print()
        print(
            seas[
                [
                    "season",
                    "bets",
                    "wins",
                    "profit_u",
                    "roi",
                    "avg_odds",
                ]
            ]
            .to_string(
                index=False,
                formatters={
                    "profit_u": lambda x: f"{x:+.2f}",
                    "roi": lambda x: f"{x:+.2%}",
                    "avg_odds": lambda x: f"{x:.2f}",
                },
            )
        )

        positive = int((seas["roi"] > 0).sum())
        negative = int((seas["roi"] < 0).sum())
        flat = int((seas["roi"] == 0).sum())

        profitable_bets = seas.loc[
            seas["roi"] > 0,
            "bets"
        ].sum()

        total_bets = seas["bets"].sum()

        profit_by_season = seas["profit_u"].to_numpy()

        total_profit = profit_by_season.sum()

        max_season_profit = (
            profit_by_season.max()
            if len(profit_by_season)
            else np.nan
        )

        winner_dependency = (
            max_season_profit / total_profit
            if total_profit > 0
            else np.nan
        )

        recent = seas.iloc[-1]
        recent2 = seas.tail(2)

        recent2_profit = recent2["profit_u"].sum()
        recent2_bets = recent2["bets"].sum()

        recent2_roi = (
            recent2_profit / recent2_bets
            if recent2_bets
            else np.nan
        )

        print()
        print(
            f"Positive seasons: {positive}/{len(seas)} "
            f"| Negative: {negative} "
            f"| Flat: {flat}"
        )

        print(
            f"Profitable-season bet share: "
            f"{profitable_bets / total_bets:.2%}"
            if total_bets
            else "Profitable-season bet share: NA"
        )

        print(
            f"Latest season: "
            f"{int(recent['season'])} "
            f"| bets={int(recent['bets'])} "
            f"| ROI={recent['roi']:+.2%}"
        )

        print(
            f"Last 2 seasons combined: "
            f"bets={int(recent2_bets)} "
            f"| profit={recent2_profit:+.2f}u "
            f"| ROI={recent2_roi:+.2%}"
        )

        if pd.notna(winner_dependency):
            print(
                f"Largest winning-season share of total profit: "
                f"{winner_dependency:.2%}"
            )

        summary_rows.append({
            "league": league,
            "side": side,
            "edge": threshold,
            "bets": int(o["bets"]),
            "overall_roi": float(o["roi"]),
            "overall_profit": float(o["profit_u"]),
            "seasons": len(seas),
            "positive_seasons": positive,
            "negative_seasons": negative,
            "positive_season_rate": (
                positive / len(seas)
                if len(seas)
                else np.nan
            ),
            "latest_season_roi": float(recent["roi"]),
            "last2_roi": float(recent2_roi),
            "winner_dependency": float(winner_dependency)
                if pd.notna(winner_dependency)
                else np.nan,
        })

summary = pd.DataFrame(summary_rows)

print()
print("=" * 130)
print("COMPACT CANDIDATE SUMMARY")
print("=" * 130)

if not summary.empty:

    show = summary.copy()

    for c in [
        "overall_roi",
        "positive_season_rate",
        "latest_season_roi",
        "last2_roi",
        "winner_dependency",
    ]:
        show[c] = show[c].map(
            lambda x: (
                f"{x:+.2%}"
                if pd.notna(x)
                else "NA"
            )
        )

    show["edge"] = show["edge"].map(
        lambda x: f">={x:.0%}"
    )

    show["overall_profit"] = show["overall_profit"].map(
        lambda x: f"{x:+.2f}u"
    )

    print(show.to_string(index=False))

print()
print("=" * 130)
print("AUDIT COMPLETE — BOARD UNCHANGED")
print("=" * 130)
