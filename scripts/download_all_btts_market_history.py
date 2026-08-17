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
    / "footystats_multileague_v5_predictions.csv"
)

RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "footystats"
    / "btts_all_leagues"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)

OUT_PATH = (
    PROCESSED_DIR
    / "btts_all_leagues_footystats_raw.csv"
)

SUMMARY_PATH = (
    PROCESSED_DIR
    / "btts_all_leagues_download_summary.csv"
)

API_KEY = os.environ["FOOTYSTATS_API_KEY"]

URL = "https://api.football-data-api.com/league-matches"


# ============================================================
# LEAGUES TO DOWNLOAD
#
# EPL + Bundesliga are intentionally handled separately because
# they are not represented in the multileague V5 prediction file.
#
# MLS already has a dedicated market dataset, but leaving it in
# here would unnecessarily redownload data we already possess.
#
# La Liga + Eliteserien are also already downloaded and validated.
# ============================================================

SKIP = {
    "MLS",
    "La Liga",
    "Eliteserien",
}

TARGETS = {
    "Ligue 1",
    "2. Bundesliga",
    "Belgian Pro League",
    "Championship",
    "Eredivisie",
    "League One",
    "League Two",
    "National League",
    "Primeira Liga",
    "Segunda División",
    "Serie A",
    "Super Lig",
    "Swiss Super League",
}


def slugify(x):
    return (
        str(x)
        .lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("ó", "o")
        .replace("í", "i")
    )


def download_season(
    league,
    season,
    season_id,
):
    league_dir = (
        RAW_DIR
        / slugify(league)
    )

    league_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        league_dir
        / f"{season}_{season_id}.json"
    )

    if raw_path.exists():
        print(
            "CACHE:",
            league,
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
        league,
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
            f"{league} {season} "
            f"{season_id}"
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
    print("=" * 120)
    print(
        "BTTS MARKET HISTORY — "
        "ALL REMAINING FOOTYSTATS LEAGUES"
    )
    print("=" * 120)

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
            f"Missing prediction columns: {missing}"
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

    season_map[
        "footystats_season_id"
    ] = pd.to_numeric(
        season_map[
            "footystats_season_id"
        ],
        errors="raise",
    ).astype(int)

    season_map = season_map[
        season_map["league"].isin(
            TARGETS
        )
    ].copy()

    # Do not download future/current 2026/27 seasons for a
    # historical completed-market backtest.
    season_map = season_map[
        pd.to_numeric(
            season_map["season"],
            errors="coerce",
        ) <= 2526
    ].copy()

    season_map = season_map.sort_values(
        [
            "league",
            "season",
        ]
    )

    print()
    print("SEASON DOWNLOAD PLAN")
    print("-" * 120)

    print(
        season_map.to_string(
            index=False
        )
    )

    print()
    print(
        "Leagues:",
        season_map["league"].nunique(),
    )

    print(
        "League-seasons:",
        len(season_map),
    )

    all_rows = []
    summary = []

    for row in season_map.itertuples(
        index=False
    ):
        league = row.league
        season = row.season
        season_id = row.footystats_season_id

        payload = download_season(
            league,
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
                league,
                season,
            )

            summary.append({
                "league": league,
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
            league,
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
            f"{league:<24}",
            f"{str(season):<6}",
            f"rows={len(df):4d}",
            f"complete={complete:4d}",
            f"BTTS={valid_btts:4d}",
        )

        summary.append({
            "league": league,
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

    combined.to_csv(
        OUT_PATH,
        index=False,
    )

    summary_df = pd.DataFrame(
        summary
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print("=" * 120)
    print("DOWNLOAD SUMMARY")
    print("=" * 120)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("BTTS COVERAGE BY LEAGUE")
    print("=" * 120)

    league_summary = (
        summary_df
        .groupby(
            "league",
            as_index=False,
        )
        .agg(
            seasons=("season", "nunique"),
            rows=("rows", "sum"),
            complete=("complete", "sum"),
            valid_btts_prices=(
                "valid_btts_prices",
                "sum",
            ),
        )
    )

    league_summary[
        "btts_price_coverage"
    ] = (
        league_summary[
            "valid_btts_prices"
        ]
        / league_summary[
            "rows"
        ].replace(0, pd.NA)
    )

    print(
        league_summary.to_string(
            index=False,
            formatters={
                "btts_price_coverage":
                    lambda x: (
                        f"{x:.2%}"
                        if pd.notna(x)
                        else "NA"
                    )
            },
        )
    )

    print()
    print("Saved:")
    print(OUT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
