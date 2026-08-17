from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "opponent_adjusted_v3_features.csv"
)


FEATURES = [
    "raw_attack_goals",
    "raw_defense_goals",

    "raw_attack_shots",
    "raw_defense_shots",

    "raw_attack_sot",
    "raw_defense_sot",

    "v3_attack_goals",
    "v3_defense_goals",

    "v3_attack_shots",
    "v3_defense_shots",

    "v3_attack_sot",
    "v3_defense_sot",
]


def main():

    print()
    print("==============================")
    print("DIAGNOSING V3 FEATURES")
    print("==============================")
    print()

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    usable = df[
        df["pregame_games"] >= 5
    ].copy()

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Usable rows: {len(usable):,}"
    )

    # =====================================================
    # MISSINGNESS
    # =====================================================

    print()
    print("==============================")
    print("MISSING DATA")
    print("==============================")

    missing_columns = [
        "shots_for",
        "shots_against",
        "shots_on_target_for",
        "shots_on_target_against",
        "lg_team_shots",
        "lg_team_sot",
    ]

    for column in missing_columns:

        missing = (
            usable[column]
            .isna()
            .mean()
        )

        print(
            f"{column:<32}"
            f"{missing:>8.2%}"
        )

    # =====================================================
    # BASELINE RANGE
    # =====================================================

    print()
    print("==============================")
    print("LEAGUE BASELINE RANGES")
    print("==============================")

    for column in [
        "lg_team_goals",
        "lg_team_shots",
        "lg_team_sot",
    ]:

        values = usable[
            column
        ].dropna()

        print()
        print(column)

        print(
            values.describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.50,
                    0.95,
                    0.99,
                ]
            )
            .round(4)
            .to_string()
        )

    # =====================================================
    # FEATURE QUANTILES
    # =====================================================

    print()
    print("==============================")
    print("FEATURE DISTRIBUTIONS")
    print("==============================")

    for feature in FEATURES:

        values = (
            usable[
                feature
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        print()
        print(feature)

        print(
            values.describe(
                percentiles=[
                    0.001,
                    0.01,
                    0.05,
                    0.50,
                    0.95,
                    0.99,
                    0.999,
                ]
            )
            .round(4)
            .to_string()
        )

    # =====================================================
    # EXTREME RAW SHOT RATINGS
    # =====================================================

    print()
    print("==============================")
    print("EXTREME RAW SHOT RATINGS")
    print("==============================")

    extreme = usable[
        (
            usable[
                "raw_attack_shots"
            ] > 3.0
        )
        |
        (
            usable[
                "raw_defense_shots"
            ] > 3.0
        )
        |
        (
            usable[
                "raw_attack_sot"
            ] > 3.0
        )
        |
        (
            usable[
                "raw_defense_sot"
            ] > 3.0
        )
    ].copy()

    print(
        f"Extreme rows: "
        f"{len(extreme):,}"
    )

    if len(extreme) > 0:

        columns = [
            "date",
            "season",
            "league",
            "team",
            "opponent",

            "ew_shots_for",
            "lg_team_shots",
            "raw_attack_shots",

            "ew_shots_on_target_for",
            "lg_team_sot",
            "raw_attack_sot",
        ]

        print()
        print(
            extreme[
                columns
            ]
            .sort_values(
                "raw_attack_shots",
                ascending=False,
            )
            .head(30)
            .round(4)
            .to_string(
                index=False
            )
        )

    # =====================================================
    # MISSING SHOTS BY LEAGUE / SEASON
    # =====================================================

    print()
    print("==============================")
    print("SHOT COVERAGE BY LEAGUE / SEASON")
    print("==============================")

    coverage = (
        df
        .groupby(
            [
                "league",
                "season",
            ]
        )
        .agg(
            rows=(
                "match_id",
                "size",
            ),

            shot_missing=(
                "shots_for",
                lambda x:
                    x.isna().mean(),
            ),

            sot_missing=(
                "shots_on_target_for",
                lambda x:
                    x.isna().mean(),
            ),
        )
        .reset_index()
    )

    coverage[
        "shot_missing"
    ] *= 100

    coverage[
        "sot_missing"
    ] *= 100

    print(
        coverage
        .round(2)
        .to_string(
            index=False
        )
    )

    print()
    print("==============================")
    print("DIAGNOSIS COMPLETE")
    print("==============================")


if __name__ == "__main__":
    main()