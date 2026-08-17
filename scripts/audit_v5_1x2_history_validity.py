from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

BET_FILE = ROOT / "data/processed/v5_1x2_all_leagues_frozen16_bets.csv"
PRED_FILE = ROOT / "data/processed/footystats_multileague_v5_predictions.csv"

bets = pd.read_csv(BET_FILE)
pred = pd.read_csv(PRED_FILE)

bets["date"] = pd.to_datetime(bets["date"], errors="coerce")
pred["date"] = pd.to_datetime(pred["date"], errors="coerce")

for c in ["profit", "win", "odds", "raw_edge"]:
    if c in bets:
        bets[c] = pd.to_numeric(bets[c], errors="coerce")


# ------------------------------------------------------------
# MERGE HISTORY INFORMATION ONTO BETS
# ------------------------------------------------------------

# Prefer exact team/date/league matching.
hist_cols = [
    "league",
    "date",
    "home_team",
    "away_team",
    "history_class",
    "home_history_source",
    "away_history_source",
    "prior_games",
    "home_adj_goal_attack_overall_games",
    "away_adj_goal_attack_overall_games",
    "home_global_xg_attack_overall_games",
    "away_global_xg_attack_overall_games",
]

hist = pred[hist_cols].drop_duplicates(
    ["league", "date", "home_team", "away_team"]
)

x = bets.merge(
    hist,
    on=["league", "date", "home_team", "away_team"],
    how="left",
    validate="many_to_one",
)

print("=" * 120)
print("V5 1X2 HISTORY VALIDITY AUDIT")
print("=" * 120)

print("Frozen bets:", len(bets))
print("Matched to prediction history:", x["history_class"].notna().sum())
print("Unmatched:", x["history_class"].isna().sum())


def perf(g):
    n = len(g)

    if not n:
        return {
            "bets": 0,
            "wins": 0,
            "profit": 0,
            "roi": np.nan,
            "odds": np.nan,
            "edge": np.nan,
        }

    return {
        "bets": n,
        "wins": int(g["win"].sum()),
        "profit": float(g["profit"].sum()),
        "roi": float(g["profit"].mean()),
        "odds": float(g["odds"].mean()),
        "edge": float(g["raw_edge"].mean()),
    }


def show(label, g):
    p = perf(g)

    print(
        f"{label:<30}"
        f" bets={p['bets']:>4}"
        f" | wins={p['wins']:>3}"
        f" | profit={p['profit']:+8.2f}u"
        f" | ROI={p['roi']:+8.2%}"
        f" | odds={p['odds']:.3f}"
        f" | edge={p['edge']:+.2%}"
    )


# ------------------------------------------------------------
# GLOBAL
# ------------------------------------------------------------

print("\n" + "=" * 120)
print("ALL FROZEN BETS BY HISTORY CLASS")
print("=" * 120)

show("ALL", x)

for cls in [
    "BOTH_SAME_LEAGUE",
    "HAS_TRANSFERRED",
    "HAS_NEUTRAL",
]:
    show(
        cls,
        x[x["history_class"].eq(cls)]
    )


# ------------------------------------------------------------
# HISTORY SOURCE COMBINATIONS
# ------------------------------------------------------------

print("\n" + "=" * 120)
print("HOME/AWAY HISTORY SOURCE COMBINATIONS")
print("=" * 120)

for (h, a), g in x.groupby(
    ["home_history_source", "away_history_source"],
    dropna=False,
):
    show(f"{h} / {a}", g)


# ------------------------------------------------------------
# CANDIDATES
# ------------------------------------------------------------

candidates = [
    ("League Two HOME", "League Two", "H"),
    ("Serie A AWAY", "Serie A", "A"),
    ("2 Bundesliga AWAY", "2. Bundesliga", "A"),
    ("Eredivisie AWAY", "Eredivisie", "A"),
]

print("\n" + "=" * 120)
print("CANDIDATE HISTORY VALIDITY")
print("=" * 120)

for label, league, side in candidates:

    g = x[
        x["league"].eq(league)
        & x["selection"].eq(side)
    ].copy()

    print("\n" + "-" * 120)
    print(label)
    print("-" * 120)

    show("ALL", g)

    for cls in [
        "BOTH_SAME_LEAGUE",
        "HAS_TRANSFERRED",
        "HAS_NEUTRAL",
    ]:
        show(
            cls,
            g[g["history_class"].eq(cls)]
        )


# ------------------------------------------------------------
# LEAGUE TWO HOME DETAIL
# ------------------------------------------------------------

lt = x[
    x["league"].eq("League Two")
    & x["selection"].eq("H")
].copy()

lt["year"] = lt["date"].dt.year

print("\n" + "=" * 120)
print("LEAGUE TWO HOME — YEAR × HISTORY CLASS")
print("=" * 120)

for year in sorted(lt["year"].dropna().unique()):

    y = lt[lt["year"].eq(year)]

    print(f"\nYEAR {int(year)}")

    show("ALL", y)

    for cls in [
        "BOTH_SAME_LEAGUE",
        "HAS_TRANSFERRED",
        "HAS_NEUTRAL",
    ]:
        show(
            cls,
            y[y["history_class"].eq(cls)]
        )


# ------------------------------------------------------------
# NON-NEUTRAL VIEW
# ------------------------------------------------------------

print("\n" + "=" * 120)
print("CANDIDATES — EXCLUDING NEUTRAL HISTORY")
print("=" * 120)

for label, league, side in candidates:

    g = x[
        x["league"].eq(league)
        & x["selection"].eq(side)
    ]

    valid = g[
        ~g["history_class"].eq("HAS_NEUTRAL")
    ]

    print(f"\n{label}")
    show("Original", g)
    show("Non-neutral", valid)


# ------------------------------------------------------------
# NEUTRAL BET DETAILS
# ------------------------------------------------------------

print("\n" + "=" * 120)
print("LEAGUE TWO HOME — NEUTRAL BET DETAILS")
print("=" * 120)

neutral_lt = lt[
    lt["history_class"].eq("HAS_NEUTRAL")
]

cols = [
    "date",
    "home_team",
    "away_team",
    "selection",
    "odds",
    "raw_edge",
    "win",
    "profit",
    "prior_games",
    "home_history_source",
    "away_history_source",
]

print(
    neutral_lt[cols]
    .sort_values("date")
    .to_string(index=False)
)


print("\n" + "=" * 120)
print("AUDIT COMPLETE")
print("=" * 120)

print("""
IMPORTANT:

This is NOT threshold optimization.

We are measuring the effect of a model-validity condition:
whether V5 actually possessed team-specific history when
generating the probability.

Do not change the frozen >=16% rule from this output.
""")
