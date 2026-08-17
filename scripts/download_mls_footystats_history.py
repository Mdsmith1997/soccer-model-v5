from pathlib import Path
import os
import time

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "footystats_mls"

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BASE_URL = "https://api.football-data-api.com"

KEY = os.environ["FOOTYSTATS_API_KEY"]

MLS_SEASONS = {
    2020: 4473,
    2021: 5674,
    2022: 6969,
    2023: 8777,
    2024: 10977,
    2025: 13973,
}


# ============================================================
# REQUEST
# ============================================================

def download_season(year, season_id):

    url = f"{BASE_URL}/league-matches"

    response = requests.get(
        url,
        params={
            "key": KEY,
            "season_id": season_id,
            "max_per_page": 1000,
        },
        timeout=60,
    )

    print(
        f"{year}: HTTP {response.status_code}"
    )

    if response.status_code != 200:

        print(
            response.text[:1000]
        )

        return None

    payload = response.json()

    if not payload.get(
        "success",
        False,
    ):

        print(
            f"{year}: API success=False"
        )

        print(
            payload.get(
                "message",
                ""
            )
        )

        return None

    data = payload.get(
        "data",
        [],
    )

    return pd.DataFrame(data)


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 100)
print("DOWNLOADING MLS FOOTYSTATS HISTORY")
print("=" * 100)

summary = []

for year, season_id in MLS_SEASONS.items():

    print()
    print("-" * 100)

    print(
        f"MLS {year} | season_id={season_id}"
    )

    df = download_season(
        year,
        season_id,
    )

    if df is None:

        summary.append(
            {
                "season": year,
                "season_id": season_id,
                "rows": 0,
                "columns": 0,
                "btts_yes": 0,
                "btts_no": 0,
                "status": "FAILED",
            }
        )

        time.sleep(1)

        continue

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    rows = len(df)
    columns = len(df.columns)

    yes_count = 0
    no_count = 0

    if "odds_btts_yes" in df.columns:

        yes = pd.to_numeric(
            df["odds_btts_yes"],
            errors="coerce",
        )

        yes_count = int(
            (yes > 1).sum()
        )

    if "odds_btts_no" in df.columns:

        no = pd.to_numeric(
            df["odds_btts_no"],
            errors="coerce",
        )

        no_count = int(
            (no > 1).sum()
        )

    # --------------------------------------------------------
    # Add identifiers
    # --------------------------------------------------------

    df.insert(
        0,
        "download_league",
        "MLS",
    )

    df.insert(
        1,
        "download_season",
        year,
    )

    df.insert(
        2,
        "download_season_id",
        season_id,
    )

    # --------------------------------------------------------
    # Save raw season
    # --------------------------------------------------------

    output = (
        RAW_DIR
        / f"mls_{year}_footystats.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    print(
        f"Rows: {rows:,}"
    )

    print(
        f"Columns: {columns}"
    )

    print(
        f"BTTS YES prices > 1: {yes_count:,}"
    )

    print(
        f"BTTS NO prices > 1:  {no_count:,}"
    )

    print(
        f"Saved: {output}"
    )

    summary.append(
        {
            "season": year,
            "season_id": season_id,
            "rows": rows,
            "columns": columns,
            "btts_yes": yes_count,
            "btts_no": no_count,
            "status": "OK",
        }
    )

    time.sleep(1)


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary
)

summary_file = (
    RAW_DIR
    / "mls_download_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False,
)

print()
print("=" * 100)
print("MLS DOWNLOAD SUMMARY")
print("=" * 100)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print(
    "Total matches:",
    summary_df["rows"].sum(),
)

print(
    "Total BTTS YES prices:",
    summary_df["btts_yes"].sum(),
)

print(
    "Total BTTS NO prices:",
    summary_df["btts_no"].sum(),
)

print()
print(
    f"Summary saved: {summary_file}"
)

print()
print("DONE")
