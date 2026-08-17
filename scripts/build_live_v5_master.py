from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"

CORE = LIVE / "v5_live_predictions_core.csv"
FOOTYSTATS = LIVE / "v5_live_predictions_footystats.csv"
OUT = LIVE / "v5_live_predictions_master.csv"

EXPECTED_CORE = 19
EXPECTED_FOOTYSTATS = 130
EXPECTED_TOTAL = 149

REQUIRED = [
    "match_id",
    "date",
    "league",
    "home_team",
    "away_team",
    "home_lambda_v5",
    "away_lambda_v5",
    "p_home_v5",
    "p_draw_v5",
    "p_away_v5",
]


def validate_source(df, name, expected_rows):

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    print("Rows:", len(df))

    if len(df) != expected_rows:
        raise ValueError(
            f"{name}: expected {expected_rows} rows, "
            f"found {len(df)}"
        )

    missing = [
        col for col in REQUIRED
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name}: missing columns: {missing}"
        )

    if df["match_id"].isna().any():
        raise ValueError(
            f"{name}: missing match_id values"
        )

    if df["match_id"].duplicated().any():

        bad = df.loc[
            df["match_id"].duplicated(False),
            [
                "match_id",
                "league",
                "home_team",
                "away_team",
            ],
        ]

        raise ValueError(
            f"{name}: duplicate match IDs:\n"
            + bad.to_string(index=False)
        )

    numeric = [
        "home_lambda_v5",
        "away_lambda_v5",
        "p_home_v5",
        "p_draw_v5",
        "p_away_v5",
    ]

    for col in numeric:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    if df[numeric].isna().any().any():

        bad = df.loc[
            df[numeric].isna().any(axis=1),
            REQUIRED,
        ]

        raise ValueError(
            f"{name}: missing numeric predictions:\n"
            + bad.to_string(index=False)
        )

    probability_sum = (
        df[
            [
                "p_home_v5",
                "p_draw_v5",
                "p_away_v5",
            ]
        ]
        .sum(axis=1)
    )

    if not np.allclose(
        probability_sum,
        1.0,
        atol=1e-6,
    ):

        bad = df.loc[
            ~np.isclose(
                probability_sum,
                1.0,
                atol=1e-6,
            ),
            [
                "match_id",
                "league",
                "home_team",
                "away_team",
                "p_home_v5",
                "p_draw_v5",
                "p_away_v5",
            ],
        ]

        raise ValueError(
            f"{name}: probabilities do not sum to 1:\n"
            + bad.to_string(index=False)
        )

    print()
    print("By league:")
    print(
        df["league"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Validation: PASS ✅")

    return df


def main():

    print()
    print("=" * 100)
    print("BUILD LIVE V5 MASTER BOARD")
    print("=" * 100)

    for path in [CORE, FOOTYSTATS]:
        if not path.exists():
            raise FileNotFoundError(path)

    core = pd.read_csv(
        CORE,
        low_memory=False,
    )

    footystats = pd.read_csv(
        FOOTYSTATS,
        low_memory=False,
    )

    core = validate_source(
        core,
        "CORE V5",
        EXPECTED_CORE,
    )

    footystats = validate_source(
        footystats,
        "FOOTYSTATS V5",
        EXPECTED_FOOTYSTATS,
    )

    core["prediction_provider"] = "CORE_V5"
    footystats["prediction_provider"] = "FOOTYSTATS_V5"

    all_cols = sorted(
        set(core.columns)
        |
        set(footystats.columns)
    )

    core = core.reindex(
        columns=all_cols
    )

    footystats = footystats.reindex(
        columns=all_cols
    )

    master = pd.concat(
        [
            core,
            footystats,
        ],
        ignore_index=True,
    )

    if len(master) != EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL} master rows; "
            f"found {len(master)}"
        )

    if master["match_id"].duplicated().any():

        bad = master.loc[
            master["match_id"].duplicated(False),
            [
                "match_id",
                "league",
                "home_team",
                "away_team",
                "prediction_provider",
            ],
        ]

        raise ValueError(
            "Duplicate match IDs across providers:\n"
            + bad.to_string(index=False)
        )

    probability_sum = (
        master[
            [
                "p_home_v5",
                "p_draw_v5",
                "p_away_v5",
            ]
        ]
        .sum(axis=1)
    )

    if not np.allclose(
        probability_sum,
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Master probabilities failed integrity check."
        )

    master["date"] = pd.to_datetime(
        master["date"],
        errors="coerce",
    )

    master = (
        master
        .sort_values(
            [
                "date",
                "league",
                "home_team",
            ]
        )
        .reset_index(drop=True)
    )

    master.to_csv(
        OUT,
        index=False,
    )

    print()
    print("=" * 100)
    print("MASTER BOARD COMPLETE")
    print("=" * 100)

    print("Rows:", len(master))
    print(
        "Unique match IDs:",
        master["match_id"].nunique(),
    )

    print()
    print("BY PROVIDER")
    print(
        master["prediction_provider"]
        .value_counts()
        .to_string()
    )

    print()
    print("BY LEAGUE")
    print(
        master["league"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("89 / 89 fixtures ✅")
    print("No duplicate match IDs ✅")
    print("No missing core probabilities ✅")
    print("Probability sums validated ✅")

    print()
    print("Saved:")
    print(OUT)


if __name__ == "__main__":
    main()
