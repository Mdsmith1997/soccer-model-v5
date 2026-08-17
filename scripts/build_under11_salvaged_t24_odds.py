from pathlib import Path
import json
import pandas as pd
import numpy as np

MAP = Path("data/processed/under11_wave1_event_map.csv")
ROOT = Path("data/raw/odds_api/under11_wave1_t24")

OUT = Path(
    "data/processed/"
    "under11_salvaged_t24_odds.csv"
)

# ============================================================
# LOAD EVENT MAP
# ============================================================

df = pd.read_csv(MAP, low_memory=False)

df["odds_api_kickoff"] = pd.to_datetime(
    df["odds_api_kickoff"],
    errors="coerce",
    utc=True,
)

lookup = {}

for _, r in df.iterrows():

    eid = r.get("odds_api_event_id")

    if pd.isna(eid):
        continue

    if pd.isna(r["odds_api_kickoff"]):
        continue

    lookup[str(eid)] = r.to_dict()

# ============================================================
# EXTRACT EVERY EXACT 2.5 OBSERVATION
# ============================================================

rows = []

for p in ROOT.rglob("*.json"):

    try:
        x = json.loads(p.read_text())
    except Exception:
        continue

    if not isinstance(x, dict):
        continue

    snap = pd.to_datetime(
        x.get("timestamp")
        or x.get("requested_timestamp"),
        errors="coerce",
        utc=True,
    )

    if pd.isna(snap):
        continue

    data = x.get("data", [])

    if not isinstance(data, list):
        continue

    for event in data:

        if not isinstance(event, dict):
            continue

        eid = str(event.get("id", ""))

        if eid not in lookup:
            continue

        info = lookup[eid]

        kickoff = pd.to_datetime(
            info["odds_api_kickoff"],
            utc=True,
        )

        hours_before = (
            kickoff - snap
        ).total_seconds() / 3600

        # Never use post-kickoff data
        if hours_before < 0:
            continue

        under = []
        over = []

        for bm in event.get("bookmakers", []):

            for market in bm.get("markets", []):

                if market.get("key") != "totals":
                    continue

                for o in market.get("outcomes", []):

                    point = o.get("point")
                    price = o.get("price")

                    if point is None or price is None:
                        continue

                    try:
                        point = float(point)
                        price = float(price)
                    except Exception:
                        continue

                    if abs(point - 2.5) > 1e-9:
                        continue

                    name = str(
                        o.get("name", "")
                    ).lower()

                    if name == "under":
                        under.append(price)

                    elif name == "over":
                        over.append(price)

        if not under or not over:
            continue

        # Consensus market price:
        # median protects against one rogue bookmaker.
        under_med = float(np.median(under))
        over_med = float(np.median(over))

        rows.append({
            "event_id": eid,
            "league": info["league"],
            "season": info["season"],
            "date": info["date"],
            "home_team": info["home_team"],
            "away_team": info["away_team"],
            "home_goals": info["home_goals"],
            "away_goals": info["away_goals"],

            # Model inputs/results already generated
            "home_lambda": info["home_lambda"],
            "away_lambda": info["away_lambda"],

            "kickoff": kickoff,
            "snapshot": snap,
            "hours_before_kickoff": hours_before,
            "distance_from_t24": abs(
                hours_before - 24
            ),

            "under_odds": under_med,
            "over_odds": over_med,

            "under_books": len(under),
            "over_books": len(over),
        })

obs = pd.DataFrame(rows)

if obs.empty:
    raise SystemExit(
        "No exact 2.5 observations found."
    )

# ============================================================
# BEST AVAILABLE SNAPSHOT CLOSEST TO TRUE T-24
# ============================================================

best = (
    obs
    .sort_values(
        [
            "event_id",
            "distance_from_t24",
            "snapshot",
        ]
    )
    .drop_duplicates(
        "event_id",
        keep="first",
    )
    .reset_index(drop=True)
)

# Timing bands
best["within_4h"] = (
    best["distance_from_t24"] <= 4
)

best["within_6h"] = (
    best["distance_from_t24"] <= 6
)

best["within_8h"] = (
    best["distance_from_t24"] <= 8
)

best["within_12h"] = (
    best["distance_from_t24"] <= 12
)

best.to_csv(
    OUT,
    index=False,
)

print("=" * 110)
print("SALVAGED T-24 EXACT 2.5 DATASET")
print("=" * 110)

print("Exact-2.5 games:", len(best))

print()

for h in [4, 6, 8, 12]:

    sub = best[
        best["distance_from_t24"] <= h
    ]

    print(
        f"T-24 ±{h:2}h: "
        f"{len(sub):4} games"
    )

print()
print("BY LEAGUE / SEASON — ±6h")

s6 = best[
    best["distance_from_t24"] <= 6
]

print(
    s6.groupby(
        ["league", "season"]
    )
    .size()
    .rename("games")
    .to_string()
)

print()
print("Under odds summary — ±6h")

print(
    s6["under_odds"]
    .describe()
    .to_string()
)

print()
print("Bookmakers per game — ±6h")

print(
    s6["under_books"]
    .describe()
    .to_string()
)

print()
print("Saved:")
print(OUT)
