from pathlib import Path
import json
import os
import time

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

PRED_PATH = (
    ROOT
    / "data"
    / "processed"
    / "footystats_ligue1_v5_predictions_research.csv"
)

RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "footystats"
    / "btts_ligue1"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)

OUT_PATH = (
    PROCESSED_DIR
    / "ligue1_btts_footystats_raw.csv"
)

SUMMARY_PATH = (
    PROCESSED_DIR
    / "ligue1_btts_download_summary.csv"
)

API_KEY = os.environ["FOOTYSTATS_API_KEY"]

URL = "https://api.football-data-api.com/league-matches"


def download_season(season, season_id):

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        RAW_DIR
        / f"{season}_{season_id}.json"
    )

    if raw_path.exists():

        print(
            "CACHE:",
            season,
            season_id,
        )

        return json.loads(
            raw_path.read_text()
        )

    r = requests.get(
        URL,
        params={
            "key": API_KEY,
            "season_id": int(season_id),
            "max_per_page": 1000,
        },
        timeout=60,
    )

    print(
        season,
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

    if not payload.get(
        "success",
        False,
    ):

        raise RuntimeError(
            f"API success=false: "
            f"{season} {season_id}"
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
    print("=" * 110)
    print("LIGUE 1 — BTTS MARKET HISTORY DOWNLOAD")
    print("=" * 110)

    pred = pd.read_csv(
        PRED_PATH,
        low_memory=False,
    )

    required = {
        "league",
        "season",
        "footystats_season_id",
    }

    missing = required - set(pred.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    season_map = (
        pred[
            [
                "league",
                "season",
                "footystats_season_id",
            ]
        ]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    season_map = season_map[
        season_map["league"]
        .astype(str)
        .eq("Ligue 1")
    ].copy()

    season_map["season"] = pd.to_numeric(
        season_map["season"],
        errors="raise",
    ).astype(int)

    season_map[
        "footystats_season_id"
    ] = pd.to_numeric(
        season_map[
            "footystats_season_id"
        ],
        errors="raise",
    ).astype(int)

    # Completed historical seasons only.
    season_map = season_map[
        season_map["season"] <= 2526
    ].copy()

    season_map = (
        season_map
        .sort_values("season")
        .reset_index(drop=True)
    )

    print()
    print("DOWNLOAD PLAN")
    print("-" * 110)

    print(
        season_map.to_string(
            index=False
        )
    )

    print()
    print(
        "Seasons:",
        len(season_map),
    )

    all_rows = []
    summary = []

    for row in season_map.itertuples(
        index=False
    ):

        season = row.season
        season_id = row.footystats_season_id

        payload = download_season(
            season,
            season_id,
        )

        data = payload.get(
            "data",
            [],
        )

        df = pd.DataFrame(data)

        if df.empty:

            print(
                "NO MATCHES:",
                season,
            )

            summary.append({
                "season": season,
                "season_id": season_id,
                "rows": 0,
                "complete": 0,
                "valid_btts_prices": 0,
            })

            continue

        df.insert(
            0,
            "model_league",
            "Ligue 1",
        )

        df.insert(
            1,
            "model_season",
            season,
        )

        df.insert(
            2,
            "footystats_season_id",
            season_id,
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

        if (
            "odds_btts_yes" in df.columns
            and "odds_btts_no" in df.columns
        ):

            yes = pd.to_numeric(
                df["odds_btts_yes"],
                errors="coerce",
            )

            no = pd.to_numeric(
                df["odds_btts_no"],
                errors="coerce",
            )

            valid_btts = (
                yes.gt(1)
                & no.gt(1)
            ).sum()

        else:

            valid_btts = 0

        print(
            f"{season} | "
            f"rows={len(df):4d} | "
            f"complete={complete:4d} | "
            f"BTTS={valid_btts:4d}"
        )

        summary.append({
            "season": season,
            "season_id": season_id,
            "rows": len(df),
            "complete": int(complete),
            "valid_btts_prices": int(
                valid_btts
            ),
        })

        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(
            "No Ligue 1 data downloaded."
        )

    combined = pd.concat(
        all_rows,
        ignore_index=True,
        sort=False,
    )

    summary_df = pd.DataFrame(
        summary
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUT_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print("=" * 110)
    print("DOWNLOAD SUMMARY")
    print("=" * 110)

    print(
        summary_df.to_string(
            index=False
        )
    )

    total_rows = summary_df["rows"].sum()

    total_btts = (
        summary_df[
            "valid_btts_prices"
        ].sum()
    )

    coverage = (
        total_btts / total_rows
        if total_rows
        else 0
    )

    print()
    print("Total rows:", total_rows)
    print(
        "Valid two-sided BTTS:",
        total_btts,
    )
    print(
        "BTTS price coverage:",
        f"{coverage:.2%}",
    )

    print()
    print("Saved:")
    print(OUT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
