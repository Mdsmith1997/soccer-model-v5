from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

from scripts.build_footystats_multileague_history import (
    fetch_season,
)

OUT = (
    ROOT
    / "data/processed"
    / "footystats_mls_history.csv"
)

SUMMARY_OUT = (
    ROOT
    / "data/processed"
    / "footystats_mls_history_summary.csv"
)

SEASONS = {
    "2019": 1846,
    "2020": 4473,
    "2021": 5674,
    "2022": 6969,
    "2023": 8777,
    "2024": 10977,
    "2025": 13973,
    "2026": 16504,
}


def main():

    print()
    print("=" * 100)
    print("BUILD FOOTYSTATS MLS HISTORY")
    print("=" * 100)

    frames = []
    summary = []

    for season, season_id in SEASONS.items():

        df = fetch_season(
            "MLS",
            season,
            season_id,
        )

        if df.empty:

            summary.append(
                {
                    "season": season,
                    "season_id": season_id,
                    "matches": 0,
                }
            )

            continue

        frames.append(df)

        summary.append(
            {
                "season": season,
                "season_id": season_id,
                "matches": len(df),
                "min_date": df["date"].min(),
                "max_date": df["date"].max(),
                "home_goals_mean":
                    df["home_goals"].mean(),
                "away_goals_mean":
                    df["away_goals"].mean(),
                "home_xg_mean":
                    df["home_xg"].mean(),
                "away_xg_mean":
                    df["away_xg"].mean(),
            }
        )

    if not frames:
        raise RuntimeError(
            "No MLS history returned."
        )

    history = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    dupes = history.duplicated(
        subset=[
            "league",
            "footystats_match_id",
        ],
        keep=False,
    )

    if dupes.any():

        print()
        print("DUPLICATES DETECTED")
        print(
            history.loc[
                dupes,
                [
                    "season",
                    "date",
                    "home_team",
                    "away_team",
                    "footystats_match_id",
                ],
            ].to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Duplicate MLS match IDs detected."
        )

    history = history.sort_values(
        [
            "date",
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.to_csv(
        OUT,
        index=False,
    )

    summary_df = pd.DataFrame(
        summary
    )

    summary_df.to_csv(
        SUMMARY_OUT,
        index=False,
    )

    print()
    print("=" * 100)
    print("MLS HISTORY SUMMARY")
    print("=" * 100)
    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("Total rows:", len(history))

    print(
        "Date range:",
        history["date"].min(),
        "->",
        history["date"].max(),
    )

    print()
    print("Saved:")
    print(OUT)
    print(SUMMARY_OUT)


if __name__ == "__main__":
    main()
