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


# ============================================================
# HELPERS
# ============================================================

def perf(g):
    n = len(g)

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
        "wins": int(g["win"].sum()),
        "profit": float(g["profit"].sum()),
        "roi": float(g["profit"].mean()),
        "avg_odds": float(g["odds"].mean()),
        "avg_edge": float(g["raw_edge"].mean()),
    }


def print_perf(label, g):
    p = perf(g)

    print(
        f"{label:<35} "
        f"bets={p['bets']:>4} | "
        f"wins={p['wins']:>4} | "
        f"profit={p['profit']:+8.2f}u | "
        f"ROI={p['roi']:+8.2%} | "
        f"odds={p['avg_odds']:.3f} | "
        f"edge={p['avg_edge']:+.2%}"
    )


def season_start_info(df):
    """
    Determine first observed date for each league-season.
    """

    starts = (
        df.groupby(
            ["league", "season"],
            as_index=False,
        )["date"]
        .min()
        .rename(
            columns={
                "date": "season_start"
            }
        )
    )

    return starts


# ============================================================
# LOAD
# ============================================================

bets = pd.read_csv(BET_FILE)
pred = pd.read_csv(PRED_FILE)

bets["date"] = pd.to_datetime(
    bets["date"],
    errors="coerce",
)

pred["date"] = pd.to_datetime(
    pred["date"],
    errors="coerce",
)

for c in [
    "model_prob",
    "market_prob",
    "raw_edge",
    "odds",
    "profit",
    "win",
]:
    if c in bets.columns:
        bets[c] = pd.to_numeric(
            bets[c],
            errors="coerce",
        )


print("=" * 135)
print("V5 1X2 — COLD-START / EARLY-SEASON AUDIT")
print("=" * 135)

print("\nFrozen >=16% bets:", len(bets))
print("Prediction rows:", len(pred))


# ============================================================
# 1. DETERMINE DAYS SINCE SEASON START
# ============================================================

# ============================================================
# DERIVE SEASON START WITHOUT REQUIRING `season` IN BET FILE
#
# The frozen bet export does not contain a season column.
# For this audit, define the season start as the earliest V5
# prediction date for that league in the same football season.
#
# European seasons normally begin Jul/Aug and cross calendar
# years, so assign July-Dec to that calendar year's season and
# Jan-Jun to the previous calendar year's season.
# ============================================================

def season_year_from_date(s):
    return np.where(
        s.dt.month >= 7,
        s.dt.year,
        s.dt.year - 1,
    )

pred["season_year"] = season_year_from_date(
    pred["date"]
)

bets["season_year"] = season_year_from_date(
    bets["date"]
)

starts = (
    pred.dropna(subset=["date"])
    .groupby(
        ["league", "season_year"],
        as_index=False,
    )["date"]
    .min()
    .rename(
        columns={
            "date": "season_start"
        }
    )
)

bets = bets.merge(
    starts,
    on=["league", "season_year"],
    how="left",
)

bets["days_since_start"] = (
    bets["date"]
    - bets["season_start"]
).dt.days

missing_start = int(
    bets["season_start"].isna().sum()
)

print(
    "Bets without inferred season start:",
    missing_start,
)

if missing_start:
    print(
        bets.loc[
            bets["season_start"].isna(),
            [
                "league",
                "date",
                "season_year",
            ],
        ]
        .drop_duplicates()
        .head(30)
        .to_string(index=False)
    )


print("\n" + "=" * 135)
print("DAYS-SINCE-SEASON-START COVERAGE")
print("=" * 135)

print(
    bets["days_since_start"]
    .describe()
    .to_string()
)


# ============================================================
# 2. PERFORMANCE BY DAYS SINCE SEASON START
# ============================================================

bins = [
    -1,
    7,
    14,
    30,
    60,
    90,
    180,
    10000,
]

labels = [
    "0-7 days",
    "8-14 days",
    "15-30 days",
    "31-60 days",
    "61-90 days",
    "91-180 days",
    "181+ days",
]

bets["season_age_band"] = pd.cut(
    bets["days_since_start"],
    bins=bins,
    labels=labels,
)

print("\n" + "=" * 135)
print("ALL LEAGUES — PERFORMANCE BY SEASON AGE")
print("=" * 135)

for band, g in bets.groupby(
    "season_age_band",
    observed=True,
):
    print_perf(str(band), g)


# ============================================================
# 3. EARLY VS ESTABLISHED
# ============================================================

print("\n" + "=" * 135)
print("EARLY VS ESTABLISHED")
print("=" * 135)

for days in [7, 14, 30, 45, 60]:

    early = bets[
        bets["days_since_start"] <= days
    ]

    later = bets[
        bets["days_since_start"] > days
    ]

    print(f"\nCUTOFF = {days} DAYS")

    print_perf("Early", early)
    print_perf("Later", later)


# ============================================================
# 4. BY LEAGUE — FIRST 30 DAYS VS LATER
# ============================================================

print("\n" + "=" * 135)
print("BY LEAGUE — FIRST 30 DAYS VS LATER")
print("=" * 135)

for league in sorted(
    bets["league"].dropna().unique()
):

    g = bets[
        bets["league"].eq(league)
    ]

    early = g[
        g["days_since_start"] <= 30
    ]

    later = g[
        g["days_since_start"] > 30
    ]

    print(f"\n{league}")
    print_perf("First 30 days", early)
    print_perf("After 30 days", later)


# ============================================================
# 5. BY LEAGUE / YEAR — EARLY SIGNAL COUNTS
# ============================================================

print("\n" + "=" * 135)
print("EARLY SIGNAL COUNTS — LEAGUE × YEAR")
print("=" * 135)

bets["year"] = bets["date"].dt.year

early30 = bets[
    bets["days_since_start"] <= 30
]

counts = (
    early30.groupby(
        ["league", "year"]
    )
    .size()
    .unstack(fill_value=0)
)

print(counts.to_string())


# ============================================================
# 6. REPEATED MODEL PROBABILITIES
# ============================================================

print("\n" + "=" * 135)
print("REPEATED MODEL PROBABILITIES — SAME LEAGUE / DATE / SIDE")
print("=" * 135)

tmp = bets.copy()

tmp["model_prob_round"] = (
    tmp["model_prob"]
    .round(4)
)

repeat = (
    tmp.groupby(
        [
            "league",
            "date",
            "selection",
            "model_prob_round",
        ],
        as_index=False,
    )
    .agg(
        count=("model_prob", "size"),
        avg_odds=("odds", "mean"),
        avg_edge=("raw_edge", "mean"),
        profit=("profit", "sum"),
    )
)

repeat = repeat[
    repeat["count"] >= 2
].copy()

repeat = repeat.sort_values(
    ["count", "date"],
    ascending=[False, True],
)

if repeat.empty:
    print("NONE")
else:
    print(
        repeat.head(100).to_string(
            index=False,
            formatters={
                "model_prob_round":
                    lambda x: f"{x:.4f}",
                "avg_odds":
                    lambda x: f"{x:.3f}",
                "avg_edge":
                    lambda x: f"{x:+.2%}",
                "profit":
                    lambda x: f"{x:+.2f}",
            },
        )
    )


# ============================================================
# 7. SHOW EXACT REPEATED-PROBABILITY BETS
# ============================================================

print("\n" + "=" * 135)
print("DETAILS OF REPEATED PROBABILITY GROUPS")
print("=" * 135)

if not repeat.empty:

    top_groups = repeat.head(20)

    for _, r in top_groups.iterrows():

        g = tmp[
            tmp["league"].eq(r["league"])
            & tmp["date"].eq(r["date"])
            & tmp["selection"].eq(
                r["selection"]
            )
            & tmp["model_prob_round"].eq(
                r["model_prob_round"]
            )
        ]

        print(
            "\n",
            r["league"],
            "|",
            r["date"].date(),
            "|",
            r["selection"],
            "| probability",
            f"{r['model_prob_round']:.4f}",
            "| count",
            int(r["count"]),
        )

        cols = [
            c for c in [
                "home_team",
                "away_team",
                "model_prob",
                "market_prob",
                "raw_edge",
                "odds",
                "win",
                "profit",
                "days_since_start",
            ]
            if c in g.columns
        ]

        print(
            g[cols].to_string(
                index=False
            )
        )


# ============================================================
# 8. EREDIVISIE 2018 SPECIFIC
# ============================================================

print("\n" + "=" * 135)
print("EREDIVISIE 2018 — COLD START DETAIL")
print("=" * 135)

er18 = bets[
    bets["league"].eq("Eredivisie")
    & bets["date"].dt.year.eq(2018)
].copy()

for side in ["H", "A"]:

    g = er18[
        er18["selection"].eq(side)
    ]

    print(f"\nSIDE = {side}")

    print_perf("All", g)

    for days in [7, 14, 30, 60, 90]:

        x = g[
            g["days_since_start"] <= days
        ]

        print_perf(
            f"First {days} days",
            x,
        )


# ============================================================
# 9. LEAGUE TWO HOME SPECIFIC
# ============================================================

print("\n" + "=" * 135)
print("LEAGUE TWO HOME — COLD START TEST")
print("=" * 135)

lt = bets[
    bets["league"].eq("League Two")
    & bets["selection"].eq("H")
].copy()

print_perf("ALL", lt)

for days in [7, 14, 30, 45, 60, 90]:

    early = lt[
        lt["days_since_start"] <= days
    ]

    later = lt[
        lt["days_since_start"] > days
    ]

    print(f"\nCUTOFF = {days} DAYS")
    print_perf("Early", early)
    print_perf("Later", later)


# ============================================================
# 10. LEAGUE TWO HOME BY YEAR + SEASON AGE
# ============================================================

print("\n" + "=" * 135)
print("LEAGUE TWO HOME — YEAR × EARLY/LATE")
print("=" * 135)

lt["period"] = np.where(
    lt["days_since_start"] <= 30,
    "FIRST_30",
    "AFTER_30",
)

for year in sorted(
    lt["year"].dropna().unique()
):

    print(f"\nYEAR {int(year)}")

    y = lt[
        lt["year"].eq(year)
    ]

    for period in [
        "FIRST_30",
        "AFTER_30",
    ]:

        print_perf(
            period,
            y[
                y["period"].eq(period)
            ],
        )


# ============================================================
# 11. 2018 VS POST-2018
# ============================================================

print("\n" + "=" * 135)
print("2018 VS POST-2018 — ALL LEAGUES")
print("=" * 135)

print_perf(
    "2018",
    bets[bets["year"].eq(2018)],
)

print_perf(
    "2019+",
    bets[bets["year"] >= 2019],
)


print("\n" + "=" * 135)
print("2018 VS POST-2018 — LEAGUE TWO HOME")
print("=" * 135)

print_perf(
    "2018",
    lt[lt["year"].eq(2018)],
)

print_perf(
    "2019+",
    lt[lt["year"] >= 2019],
)


# ============================================================
# 12. MOST COMMON MODEL PROBABILITIES
# ============================================================

print("\n" + "=" * 135)
print("MOST COMMON MODEL PROBABILITIES — ALL FROZEN BETS")
print("=" * 135)

freq = (
    tmp.groupby(
        "model_prob_round"
    )
    .size()
    .sort_values(
        ascending=False
    )
    .head(30)
)

print(freq.to_string())


print("\n" + "=" * 135)
print("MOST COMMON MODEL PROBABILITIES — 2018 ONLY")
print("=" * 135)

freq18 = (
    tmp[
        tmp["year"].eq(2018)
    ]
    .groupby(
        "model_prob_round"
    )
    .size()
    .sort_values(
        ascending=False
    )
    .head(30)
)

print(freq18.to_string())


print("\n" + "=" * 135)
print("AUDIT COMPLETE")
print("=" * 135)

print("""
WHAT WE ARE TESTING

1. Are >=16% signals disproportionately concentrated
   at the beginning of seasons?

2. Do early-season signals materially underperform
   later-season signals?

3. Are repeated model probabilities common during
   cold-start periods?

4. Is 2018 uniquely abnormal?

5. Does League Two HOME remain profitable after the
   suspected cold-start period?

IMPORTANT:
Do NOT turn any day cutoff from this diagnostic into
a betting rule yet. This is a causal/data-quality
investigation, not threshold optimization.
""")
