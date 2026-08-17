from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

from scripts.backtest_footystats_multileague_v5 import (
    build_predictions,
)

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "footystats_mls_pregame_v2.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "footystats_mls_v5_predictions.csv"
)


def main():

    print()
    print("=" * 110)
    print("MLS — FROZEN V5 PREDICTIONS")
    print("=" * 110)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    print()
    print("Input rows:", len(df))

    print(
        "Seasons:",
        sorted(
            df["season"]
            .astype(str)
            .unique()
            .tolist()
        ),
    )

    print(
        "Leagues:",
        df["league"]
        .value_counts()
        .to_dict(),
    )

    # --------------------------------------------------------
    # EXACT EXISTING FROZEN V5 ENGINE
    # --------------------------------------------------------

    pred = build_predictions(
        df
    )

    # MLS calendar-year role labels.
    # Reporting only — these do NOT affect predictions.
    role_map = {
        "2019": "HISTORY",
        "2020": "HISTORY",
        "2021": "HISTORY",
        "2022": "HISTORY",
        "2023": "DEVELOPMENT",
        "2024": "VALIDATION",
        "2025": "FINAL_HOLDOUT",
        "2026": "CURRENT_FORWARD",
    }

    pred["season_role"] = (
        pred["season"]
        .astype(str)
        .map(role_map)
        .fillna("UNKNOWN")
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 110)
    print("MLS V5 OUTPUT")
    print("=" * 110)

    print()
    print("Prediction rows:", len(pred))

    print()
    print("By season:")
    print(
        pred["season"]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Season roles:")
    print(
        pred["season_role"]
        .value_counts()
        .to_string()
    )

    print()
    print("Lambda summary:")
    print(
        pred[
            [
                "home_lambda",
                "away_lambda",
            ]
        ]
        .describe()
        .to_string()
    )

    print()
    print("1X2 probability sums:")

    probability_sum = (
        pred["p_home"]
        + pred["p_draw"]
        + pred["p_away"]
    )

    print(
        probability_sum
        .describe()
        .to_string()
    )

    print()
    print("Missing values:")

    for col in [
        "home_lambda",
        "away_lambda",
        "p_home",
        "p_draw",
        "p_away",
    ]:

        print(
            f"{col:<15}",
            int(
                pred[col]
                .isna()
                .sum()
            ),
        )

    print()
    print("Saved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
