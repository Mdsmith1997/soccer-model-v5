import os
import requests
import pandas as pd

BASE_URL = "https://api.football-data-api.com"
KEY = os.environ["FOOTYSTATS_API_KEY"]

# Regular MLS only
MLS_SEASONS = {
    2020: 4473,
    2021: 5674,
    2022: 6969,
    2023: 8777,
    2024: 10977,
    2025: 13973,
}

TEST_YEAR = 2025
SEASON_ID = MLS_SEASONS[TEST_YEAR]


def get_json(endpoint, params):
    url = f"{BASE_URL}/{endpoint}"

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    print()
    print("URL:")
    print(response.url)

    print()
    print("STATUS:")
    print(response.status_code)

    if response.status_code != 200:
        print()
        print("RESPONSE:")
        print(response.text[:3000])
        return None

    return response.json()


print()
print("=" * 100)
print("REGULAR MLS FOOTYSTATS TEST")
print("=" * 100)

print()
print("Year:", TEST_YEAR)
print("Season ID:", SEASON_ID)

payload = get_json(
    "league-matches",
    {
        "key": KEY,
        "season_id": SEASON_ID,
        "max_per_page": 1000,
    },
)

if not payload:
    raise SystemExit

print()
print("API success:", payload.get("success"))

data = payload.get("data", [])
df = pd.DataFrame(data)

print()
print("=" * 100)
print("MLS RESPONSE")
print("=" * 100)

print()
print("Rows:", len(df))
print("Columns:", len(df.columns))

if len(df) == 0:
    print("No matches returned.")
    raise SystemExit


# ============================================================
# FIND ODDS / BTTS COLUMNS
# ============================================================

odds_cols = []

for col in df.columns:
    c = str(col).lower()

    if (
        "odds" in c
        or "btts" in c
        or "both" in c
    ):
        odds_cols.append(col)


print()
print("=" * 100)
print("ODDS / BTTS COLUMNS")
print("=" * 100)

for col in odds_cols:
    print(col)


# ============================================================
# PRINT VALUES FROM ODDS COLUMNS
# ============================================================

print()
print("=" * 100)
print("ODDS COLUMN SAMPLE VALUES")
print("=" * 100)

for col in odds_cols:
    vals = (
        df[col]
        .dropna()
        .head(5)
        .tolist()
    )

    print()
    print(col)
    print(vals)


# ============================================================
# MATCH SAMPLE
# ============================================================

base_cols = []

for col in [
    "id",
    "date_unix",
    "home_name",
    "away_name",
    "homeGoalCount",
    "awayGoalCount",
]:
    if col in df.columns:
        base_cols.append(col)

sample_cols = base_cols + [
    c for c in odds_cols
    if c not in base_cols
]

print()
print("=" * 100)
print("FIRST 5 MLS MATCHES")
print("=" * 100)

print(
    df[sample_cols]
    .head(5)
    .to_string(index=False)
)


print()
print("=" * 100)
print("TEST COMPLETE")
print("=" * 100)
