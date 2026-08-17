from pathlib import Path
import json
import os
import time

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "footystats"
    / "expansion"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)

API_KEY = os.environ["FOOTYSTATS_API_KEY"]

URL = "https://api.football-data-api.com/league-matches"


# ============================================================
# FROZEN SEASON MAP
# ============================================================

LEAGUES = {
    "la_liga": {
        "display": "La Liga",
        "country": "Spain",
        "seasons": {
            "2021": 4944,   # 2020/21
            "2122": 6211,
            "2223": 7665,
            "2324": 9665,
            "2425": 12316,
            "2526": 14956,
        },
    },

    "eliteserien": {
        "display": "Eliteserien",
        "country": "Norway",
        "seasons": {
            "2020": 3695,
            "2021": 5496,
            "2022": 7048,
            "2023": 8739,
            "2024": 17353,
            "2025": 16260,
        },
    },
}


def download_season(
    league_key,
    league_info,
    season_label,
    season_id,
):

    league_dir = (
        RAW_DIR
        / league_key
    )

    league_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        league_dir
        / f"{season_label}_{season_id}.json"
    )

    # ----------------------------------------
    # CACHE
    # ----------------------------------------

    if raw_path.exists():

        print("CACHE:", raw_path.name)

        payload = json.loads(
            raw_path.read_text()
        )

        return payload

    # ----------------------------------------
    # API
    # ----------------------------------------

    r = requests.get(
        URL,
        params={
            "key": API_KEY,
            "season_id": season_id,
            "max_per_page": 1000,
        },
        timeout=60,
    )

    print(
        league_info["display"],
        season_label,
        "| ID:",
        season_id,
        "| HTTP:",
        r.status_code,
    )

    if r.status_code != 200:

        print(
            "BODY:",
            r.text[:1000],
        )

        r.raise_for_status()

    payload = r.json()

    if not payload.get("success", False):

        raise RuntimeError(
            f"API returned success=false for "
            f"{league_info['display']} "
            f"{season_label}: {payload}"
        )

    raw_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
    )

    time.sleep(0.15)

    return payload


def main():

    print()
    print("=" * 115)
    print(
        "FOOTYSTATS EXPANSION — "
        "LA LIGA + ELITESERIEN"
    )
    print("=" * 115)

    all_rows = []

    summary = []

    for league_key, info in LEAGUES.items():

        print()
        print("=" * 115)
        print(info["display"].upper())
        print("=" * 115)

        for season_label, season_id in (
            info["seasons"].items()
        ):

            payload = download_season(
                league_key,
                info,
                season_label,
                season_id,
            )

            data = payload.get(
                "data",
                [],
            )

            df = pd.DataFrame(data)

            if df.empty:

                print(
                    season_label,
                    "-> NO MATCHES"
                )

                summary.append(
                    {
                        "league": info["display"],
                        "season": season_label,
                        "season_id": season_id,
                        "rows": 0,
                        "complete": 0,
                    }
                )

                continue

            # Preserve every original API field.
            df.insert(
                0,
                "model_league",
                league_key,
            )

            df.insert(
                1,
                "model_season",
                season_label,
            )

            df.insert(
                2,
                "footystats_season_id",
                season_id,
            )

            df.insert(
                3,
                "country_name",
                info["country"],
            )

            complete = (
                df["status"]
                .astype(str)
                .str.lower()
                .eq("complete")
                .sum()
                if "status" in df.columns
                else 0
            )

            print(
                f"{season_label:<6}",
                f"rows={len(df):4d}",
                f"complete={complete:4d}",
                f"columns={len(df.columns):4d}",
            )

            summary.append(
                {
                    "league": info["display"],
                    "season": season_label,
                    "season_id": season_id,
                    "rows": len(df),
                    "complete": int(complete),
                    "columns": len(df.columns),
                }
            )

            all_rows.append(df)

    if not all_rows:

        raise RuntimeError(
            "No match data downloaded."
        )

    combined = pd.concat(
        all_rows,
        ignore_index=True,
        sort=False,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_path = (
        PROCESSED_DIR
        / "laliga_eliteserien_footystats_raw.csv"
    )

    summary_path = (
        PROCESSED_DIR
        / "laliga_eliteserien_download_summary.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    pd.DataFrame(
        summary
    ).to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 115)
    print("DOWNLOAD SUMMARY")
    print("=" * 115)

    summary_df = pd.DataFrame(
        summary
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "TOTAL ROWS:",
        len(combined),
    )

    print(
        "TOTAL COLUMNS:",
        len(combined.columns),
    )

    print()
    print("=" * 115)
    print("IMPORTANT V5 INPUT FIELDS")
    print("=" * 115)

    keywords = [
        "goal",
        "xg",
        "shot",
        "date",
        "team",
        "home",
        "away",
        "status",
    ]

    for c in combined.columns:

        if any(
            k in c.lower()
            for k in keywords
        ):
            print(c)

    print()
    print("=" * 115)
    print("MISSINGNESS — CORE CANDIDATES")
    print("=" * 115)

    candidate_cols = [
        "homeGoalCount",
        "awayGoalCount",
        "team_a_xg",
        "team_b_xg",
        "team_a_shots",
        "team_b_shots",
        "date_unix",
        "home_name",
        "away_name",
        "status",
    ]

    for c in candidate_cols:

        if c not in combined.columns:
            print(
                f"{c:<25} NOT PRESENT"
            )
            continue

        nonnull = (
            combined[c]
            .notna()
            .sum()
        )

        print(
            f"{c:<25}"
            f"{nonnull:6d}/{len(combined):6d}"
            f"  {nonnull / len(combined):7.2%}"
        )

    print()
    print("Saved combined data:")
    print(combined_path)

    print()
    print("Saved summary:")
    print(summary_path)

    print()
    print("Raw JSON:")
    print(RAW_DIR)


if __name__ == "__main__":
    main()
