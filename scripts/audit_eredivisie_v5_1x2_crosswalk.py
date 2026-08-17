from pathlib import Path
from collections import Counter
import pandas as pd
import unicodedata
import re

ROOT = Path("data/processed")

PRED_FILE = ROOT / "footystats_multileague_v5_predictions.csv"
MARKET_FILE = ROOT / "v5_1x2_football_data.csv"

LEAGUE = "Eredivisie"


def norm_team(x):
    x = str(x).strip().lower()

    x = unicodedata.normalize("NFKD", x)
    x = "".join(
        c for c in x
        if not unicodedata.combining(c)
    )

    x = x.replace("&", "and")
    x = re.sub(r"[^a-z0-9]+", "", x)

    return x


pred = pd.read_csv(PRED_FILE, low_memory=False)
market = pd.read_csv(MARKET_FILE, low_memory=False)

pred = pred[
    pred["league"].astype(str).str.strip().eq(LEAGUE)
].copy()

market = market[
    market["league"].astype(str).str.strip().eq(LEAGUE)
].copy()


for df in [pred, market]:
    df["date_key"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    ).dt.normalize()

    df["home_key"] = df["home_team"].map(norm_team)
    df["away_key"] = df["away_team"].map(norm_team)


print("=" * 130)
print("EREDIVISIE — V5 1X2 CROSSWALK AUDIT")
print("=" * 130)

print("Prediction rows:", len(pred))
print("Market rows:    ", len(market))

print()
print(
    "Prediction dates:",
    pred["date_key"].min(),
    "->",
    pred["date_key"].max()
)

print(
    "Market dates:    ",
    market["date_key"].min(),
    "->",
    market["date_key"].max()
)


# ============================================================
# EXACT CROSSWALK
# ============================================================

market_keys = market[
    [
        "date_key",
        "home_key",
        "away_key",
        "home_odds",
        "draw_odds",
        "away_odds",
    ]
].copy()

merged = pred.merge(
    market_keys,
    on=[
        "date_key",
        "home_key",
        "away_key",
    ],
    how="left",
    indicator=True,
)

merged["exact_match"] = merged["_merge"].eq("both")

print()
print("=" * 130)
print("EXACT MATCH")
print("=" * 130)

print("Exact matches:", int(merged["exact_match"].sum()))
print("Exact rate:   ", f"{merged['exact_match'].mean():.2%}")
print("Unmatched:    ", int((~merged["exact_match"]).sum()))


# ============================================================
# VALID ODDS
# ============================================================

for c in ["home_odds", "draw_odds", "away_odds"]:
    merged[c] = pd.to_numeric(
        merged[c],
        errors="coerce"
    )

merged["valid_odds"] = (
    merged["exact_match"]
    &
    merged[
        ["home_odds", "draw_odds", "away_odds"]
    ].notna().all(axis=1)
    &
    (
        merged[
            ["home_odds", "draw_odds", "away_odds"]
        ] > 1.0
    ).all(axis=1)
)

print()
print(
    "Exact but invalid odds:",
    int(
        (
            merged["exact_match"]
            &
            ~merged["valid_odds"]
        ).sum()
    )
)

print(
    "Valid market matches:  ",
    int(merged["valid_odds"].sum())
)

print(
    "Valid coverage:         ",
    f"{merged['valid_odds'].mean():.2%}"
)


# ============================================================
# COVERAGE BY YEAR
# ============================================================

merged["year"] = merged["date_key"].dt.year

coverage = (
    merged.groupby("year")
    .agg(
        predictions=("exact_match", "size"),
        exact_matches=("exact_match", "sum"),
        valid_odds=("valid_odds", "sum"),
    )
    .reset_index()
)

coverage["exact_pct"] = (
    coverage["exact_matches"]
    / coverage["predictions"]
)

coverage["valid_pct"] = (
    coverage["valid_odds"]
    / coverage["predictions"]
)

print()
print("=" * 130)
print("COVERAGE BY YEAR")
print("=" * 130)

print(
    coverage.to_string(
        index=False,
        formatters={
            "exact_pct": lambda x: f"{x:.2%}",
            "valid_pct": lambda x: f"{x:.2%}",
        }
    )
)


# ============================================================
# TEAM INVENTORIES
# ============================================================

pred_names = sorted(
    set(pred["home_team"].dropna())
    |
    set(pred["away_team"].dropna())
)

market_names = sorted(
    set(market["home_team"].dropna())
    |
    set(market["away_team"].dropna())
)

print()
print("=" * 130)
print("PREDICTION TEAM NAMES")
print("=" * 130)

for team in pred_names:
    print(f"{team:<35} => {norm_team(team)}")

print()
print("=" * 130)
print("MARKET TEAM NAMES")
print("=" * 130)

for team in market_names:
    print(f"{team:<35} => {norm_team(team)}")


pred_keys = (
    set(pred["home_key"])
    |
    set(pred["away_key"])
)

market_team_keys = (
    set(market["home_key"])
    |
    set(market["away_key"])
)

print()
print("=" * 130)
print("NORMALIZED KEYS ONLY IN PREDICTIONS")
print("=" * 130)

print(sorted(pred_keys - market_team_keys))

print()
print("=" * 130)
print("NORMALIZED KEYS ONLY IN MARKET")
print("=" * 130)

print(sorted(market_team_keys - pred_keys))


# ============================================================
# UNMATCHED DIAGNOSTICS
# ============================================================

unmatched = merged[
    ~merged["exact_match"]
].copy()

print()
print("=" * 130)
print("UNMATCHED BY YEAR")
print("=" * 130)

if len(unmatched):
    print(
        unmatched.groupby("year")
        .size()
        .rename("unmatched")
        .to_string()
    )
else:
    print("NONE")


# Same date, one team agrees
market_by_date = {
    d: x
    for d, x in market.groupby("date_key")
}

candidates = []

for _, row in unmatched.iterrows():

    day = market_by_date.get(row["date_key"])

    if day is None:
        continue

    for _, m in day.iterrows():

        home_match = row["home_key"] == m["home_key"]
        away_match = row["away_key"] == m["away_key"]

        if home_match ^ away_match:

            candidates.append({
                "date": row["date_key"],
                "pred_home": row["home_team"],
                "pred_away": row["away_team"],
                "market_home": m["home_team"],
                "market_away": m["away_team"],
                "home_matches": home_match,
                "away_matches": away_match,
            })


print()
print("=" * 130)
print("SAME DATE — EXACTLY ONE TEAM MATCHES")
print("=" * 130)

print("Candidates:", len(candidates))

if candidates:
    print(
        pd.DataFrame(candidates)
        .head(100)
        .to_string(index=False)
    )


# Same teams but nearby date
pair_dates = (
    market.groupby(
        ["home_key", "away_key"]
    )["date_key"]
    .apply(list)
    .to_dict()
)

offsets = Counter()
examples = []

for _, row in unmatched.iterrows():

    dates = pair_dates.get(
        (row["home_key"], row["away_key"]),
        []
    )

    diffs = [
        (d - row["date_key"]).days
        for d in dates
        if pd.notna(d)
    ]

    if not diffs:
        continue

    nearest = min(
        diffs,
        key=lambda x: abs(x)
    )

    if abs(nearest) <= 7:

        offsets[nearest] += 1

        if len(examples) < 40:
            examples.append({
                "pred_date": row["date_key"],
                "home": row["home_team"],
                "away": row["away_team"],
                "offset_days": nearest,
            })


print()
print("=" * 130)
print("SAME TEAMS — MARKET DATE WITHIN ±7 DAYS")
print("=" * 130)

if offsets:
    for k in sorted(offsets):
        print(f"{k:+d} days: {offsets[k]}")
else:
    print("NONE")

if examples:
    print()
    print(pd.DataFrame(examples).to_string(index=False))


# ============================================================
# MOST COMMON TEAMS AMONG UNMATCHED
# ============================================================

if len(unmatched):

    teams = pd.concat(
        [
            unmatched[["home_team"]]
            .rename(columns={"home_team": "team"}),

            unmatched[["away_team"]]
            .rename(columns={"away_team": "team"}),
        ],
        ignore_index=True
    )

    print()
    print("=" * 130)
    print("MOST COMMON TEAMS IN UNMATCHED MATCHES")
    print("=" * 130)

    print(
        teams["team"]
        .value_counts()
        .head(40)
        .to_string()
    )


print()
print("=" * 130)
print("AUDIT COMPLETE — NOTHING MODIFIED")
print("=" * 130)
