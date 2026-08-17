from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = (
    ROOT
    / "data"
    / "processed"
)

INPUT_FILE = (
    PROCESSED
    / "footystats_multileague_history.csv"
)

OUTPUT_SEASON = (
    PROCESSED
    / "footystats_multileague_stability_by_season.csv"
)

OUTPUT_TRANSITIONS = (
    PROCESSED
    / "footystats_multileague_stability_transitions.csv"
)

OUTPUT_FLAGS = (
    PROCESSED
    / "footystats_multileague_stability_flags.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# These are intentionally conservative.
#
# A flag does NOT automatically mean bad data.
# It means "inspect this season transition before live use."

RELATIVE_CHANGE_WARN = 0.20

RELATIVE_CHANGE_HIGH = 0.35

HOME_ADVANTAGE_CHANGE_WARN = 0.30

MIN_MATCHES_FULL_SEASON = 150


# ============================================================
# HELPERS
# ============================================================

def safe_pct_change(
    new,
    old,
):

    if (
        pd.isna(new)
        or
        pd.isna(old)
        or
        old == 0
    ):

        return np.nan

    return (
        new
        -
        old
    ) / abs(
        old
    )


def season_sort_key(
    season,
):

    value = str(
        season
    )

    mapping = {
        "1819": 2018,
        "1920": 2019,
        "2021": 2020,
        "2122": 2021,
        "2223": 2022,
        "2324": 2023,
        "2425": 2024,
        "2526": 2025,
        "2627": 2026,
    }

    return mapping.get(
        value,
        9999,
    )


# ============================================================
# LOAD
# ============================================================

def load_history():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            INPUT_FILE
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    required = [
        "date",
        "season",
        "league",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "home_shots",
        "away_shots",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: "
            f"{missing}"
        )

    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        errors="coerce",
    )

    df[
        "season"
    ] = (
        df[
            "season"
        ]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.strip()
    )

    numeric_cols = [
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "home_shots",
        "away_shots",
    ]

    for col in numeric_cols:

        df[
            col
        ] = pd.to_numeric(
            df[
                col
            ],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "date",
            "season",
            "league",
            *numeric_cols,
        ]
    ).copy()

    return df


# ============================================================
# SEASON SUMMARY
# ============================================================

def build_season_summary(
    df,
):

    rows = []

    for (
        league,
        season
    ), sub in df.groupby(
        [
            "league",
            "season",
        ]
    ):

        home_goals = (
            sub[
                "home_goals"
            ].mean()
        )

        away_goals = (
            sub[
                "away_goals"
            ].mean()
        )

        home_xg = (
            sub[
                "home_xg"
            ].mean()
        )

        away_xg = (
            sub[
                "away_xg"
            ].mean()
        )

        home_shots = (
            sub[
                "home_shots"
            ].mean()
        )

        away_shots = (
            sub[
                "away_shots"
            ].mean()
        )

        rows.append(
            {
                "league":
                    league,

                "season":
                    season,

                "season_sort":
                    season_sort_key(
                        season
                    ),

                "games":
                    len(
                        sub
                    ),

                "first_date":
                    sub[
                        "date"
                    ].min(),

                "last_date":
                    sub[
                        "date"
                    ].max(),

                "home_goals_mean":
                    home_goals,

                "away_goals_mean":
                    away_goals,

                "total_goals_mean":
                    (
                        home_goals
                        +
                        away_goals
                    ),

                "home_xg_mean":
                    home_xg,

                "away_xg_mean":
                    away_xg,

                "total_xg_mean":
                    (
                        home_xg
                        +
                        away_xg
                    ),

                "home_shots_mean":
                    home_shots,

                "away_shots_mean":
                    away_shots,

                "total_shots_mean":
                    (
                        home_shots
                        +
                        away_shots
                    ),

                "goal_home_advantage":
                    (
                        home_goals
                        -
                        away_goals
                    ),

                "xg_home_advantage":
                    (
                        home_xg
                        -
                        away_xg
                    ),

                "shot_home_advantage":
                    (
                        home_shots
                        -
                        away_shots
                    ),

                "shots_per_xg_home":
                    (
                        home_shots
                        /
                        home_xg
                        if home_xg > 0
                        else np.nan
                    ),

                "shots_per_xg_away":
                    (
                        away_shots
                        /
                        away_xg
                        if away_xg > 0
                        else np.nan
                    ),

                "goals_per_xg_home":
                    (
                        home_goals
                        /
                        home_xg
                        if home_xg > 0
                        else np.nan
                    ),

                "goals_per_xg_away":
                    (
                        away_goals
                        /
                        away_xg
                        if away_xg > 0
                        else np.nan
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary = summary.sort_values(
        [
            "league",
            "season_sort",
        ]
    ).reset_index(
        drop=True
    )

    return summary


# ============================================================
# TRANSITION ANALYSIS
# ============================================================

def build_transitions(
    summary,
):

    rows = []

    metrics = [
        "home_goals_mean",
        "away_goals_mean",
        "total_goals_mean",
        "home_xg_mean",
        "away_xg_mean",
        "total_xg_mean",
        "home_shots_mean",
        "away_shots_mean",
        "total_shots_mean",
        "goal_home_advantage",
        "xg_home_advantage",
        "shot_home_advantage",
        "shots_per_xg_home",
        "shots_per_xg_away",
        "goals_per_xg_home",
        "goals_per_xg_away",
    ]

    for league, sub in summary.groupby(
        "league"
    ):

        sub = sub.sort_values(
            "season_sort"
        ).reset_index(
            drop=True
        )

        for i in range(
            1,
            len(
                sub
            ),
        ):

            previous = sub.iloc[
                i - 1
            ]

            current = sub.iloc[
                i
            ]

            row = {
                "league":
                    league,

                "previous_season":
                    previous[
                        "season"
                    ],

                "season":
                    current[
                        "season"
                    ],

                "previous_games":
                    previous[
                        "games"
                    ],

                "games":
                    current[
                        "games"
                    ],
            }

            for metric in metrics:

                old = previous[
                    metric
                ]

                new = current[
                    metric
                ]

                row[
                    f"{metric}_previous"
                ] = old

                row[
                    metric
                ] = new

                row[
                    f"{metric}_change"
                ] = (
                    new
                    -
                    old
                )

                row[
                    f"{metric}_pct_change"
                ] = safe_pct_change(
                    new,
                    old,
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FLAGGING
# ============================================================

def build_flags(
    transitions,
):

    flags = []

    for row in transitions.itertuples(
        index=False
    ):

        # Current season may simply be incomplete.
        incomplete_current = (
            row.games
            <
            MIN_MATCHES_FULL_SEASON
        )

        check_metrics = [
            (
                "total_xg",
                row.total_xg_mean_pct_change,
            ),
            (
                "total_shots",
                row.total_shots_mean_pct_change,
            ),
            (
                "total_goals",
                row.total_goals_mean_pct_change,
            ),
            (
                "shots_per_xg_home",
                row.shots_per_xg_home_pct_change,
            ),
            (
                "shots_per_xg_away",
                row.shots_per_xg_away_pct_change,
            ),
        ]

        for metric, change in (
            check_metrics
        ):

            if pd.isna(
                change
            ):

                continue

            abs_change = abs(
                change
            )

            if (
                abs_change
                >=
                RELATIVE_CHANGE_HIGH
            ):

                severity = "HIGH"

            elif (
                abs_change
                >=
                RELATIVE_CHANGE_WARN
            ):

                severity = "WARN"

            else:

                continue

            flags.append(
                {
                    "league":
                        row.league,

                    "previous_season":
                        row.previous_season,

                    "season":
                        row.season,

                    "metric":
                        metric,

                    "pct_change":
                        change,

                    "severity":
                        severity,

                    "current_incomplete":
                        incomplete_current,
                }
            )

        home_adv_checks = [
            (
                "goal_home_advantage",
                row.goal_home_advantage_previous,
                row.goal_home_advantage,
            ),
            (
                "xg_home_advantage",
                row.xg_home_advantage_previous,
                row.xg_home_advantage,
            ),
            (
                "shot_home_advantage",
                row.shot_home_advantage_previous,
                row.shot_home_advantage,
            ),
        ]

        for (
            metric,
            old,
            new
        ) in home_adv_checks:

            if (
                pd.isna(old)
                or
                pd.isna(new)
            ):

                continue

            denominator = max(
                abs(
                    old
                ),
                0.25,
            )

            relative_change = (
                new
                -
                old
            ) / denominator

            if (
                abs(
                    relative_change
                )
                <
                HOME_ADVANTAGE_CHANGE_WARN
            ):

                continue

            severity = (
                "HIGH"
                if
                abs(
                    relative_change
                )
                >=
                0.60
                else
                "WARN"
            )

            flags.append(
                {
                    "league":
                        row.league,

                    "previous_season":
                        row.previous_season,

                    "season":
                        row.season,

                    "metric":
                        metric,

                    "pct_change":
                        relative_change,

                    "severity":
                        severity,

                    "current_incomplete":
                        incomplete_current,
                }
            )

    return pd.DataFrame(
        flags
    )


# ============================================================
# PRINTING
# ============================================================

def print_season_summary(
    summary,
):

    print()
    print(
        "=" * 150
    )

    print(
        "SEASON LEVEL STABILITY"
    )

    print(
        "=" * 150
    )

    cols = [
        "league",
        "season",
        "games",
        "total_goals_mean",
        "total_xg_mean",
        "total_shots_mean",
        "goal_home_advantage",
        "xg_home_advantage",
        "shot_home_advantage",
        "shots_per_xg_home",
        "shots_per_xg_away",
    ]

    print(
        summary[
            cols
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


def print_transitions(
    transitions,
):

    print()
    print(
        "=" * 150
    )

    print(
        "SEASON-TO-SEASON CHANGES"
    )

    print(
        "=" * 150
    )

    cols = [
        "league",
        "previous_season",
        "season",
        "games",
        "total_goals_mean_pct_change",
        "total_xg_mean_pct_change",
        "total_shots_mean_pct_change",
        "shots_per_xg_home_pct_change",
        "shots_per_xg_away_pct_change",
    ]

    print(
        transitions[
            cols
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{100 * x:.1f}%"
                if
                abs(
                    x
                )
                < 10
                else
                f"{x:.3f}",
        )
    )


def print_flags(
    flags,
):

    print()
    print(
        "=" * 120
    )

    print(
        "STABILITY FLAGS"
    )

    print(
        "=" * 120
    )

    if len(
        flags
    ) == 0:

        print(
            "No material discontinuities "
            "flagged."
        )

        return

    display = (
        flags
        .sort_values(
            [
                "severity",
                "league",
                "season",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .copy()
    )

    display[
        "pct_change"
    ] = (
        100.0
        *
        display[
            "pct_change"
        ]
    )

    print(
        display.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.1f}%",
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 90
    )

    print(
        "FOOTYSTATS MULTI-LEAGUE "
        "DATA STABILITY AUDIT"
    )

    print(
        "=" * 90
    )

    print()
    print(
        "Purpose:"
    )

    print(
        "Detect season-to-season provider "
        "or definition shifts before the "
        "historical data is used by live V5."
    )

    df = load_history()

    print()
    print(
        "Historical matches:",
        f"{len(df):,}",
    )

    print(
        "Leagues:",
        df[
            "league"
        ].nunique(),
    )

    print(
        "Date range:",
        df[
            "date"
        ].min(),
        "->",
        df[
            "date"
        ].max(),
    )

    summary = build_season_summary(
        df
    )

    transitions = build_transitions(
        summary
    )

    flags = build_flags(
        transitions
    )

    print_season_summary(
        summary
    )

    print_transitions(
        transitions
    )

    print_flags(
        flags
    )

    # --------------------------------------------------------
    # DATA QUALITY CHECKS
    # --------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "CORE SIGNAL CORRELATIONS"
    )

    print(
        "=" * 100
    )

    correlation_rows = []

    for league, sub in df.groupby(
        "league"
    ):

        correlation_rows.append(
            {
                "league":
                    league,

                "home_xg_goal_corr":
                    sub[
                        [
                            "home_xg",
                            "home_goals",
                        ]
                    ]
                    .corr()
                    .iloc[
                        0,
                        1
                    ],

                "away_xg_goal_corr":
                    sub[
                        [
                            "away_xg",
                            "away_goals",
                        ]
                    ]
                    .corr()
                    .iloc[
                        0,
                        1
                    ],

                "home_shot_xg_corr":
                    sub[
                        [
                            "home_shots",
                            "home_xg",
                        ]
                    ]
                    .corr()
                    .iloc[
                        0,
                        1
                    ],

                "away_shot_xg_corr":
                    sub[
                        [
                            "away_shots",
                            "away_xg",
                        ]
                    ]
                    .corr()
                    .iloc[
                        0,
                        1
                    ],
            }
        )

    correlations = pd.DataFrame(
        correlation_rows
    )

    print(
        correlations.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    summary.to_csv(
        OUTPUT_SEASON,
        index=False,
    )

    transitions.to_csv(
        OUTPUT_TRANSITIONS,
        index=False,
    )

    flags.to_csv(
        OUTPUT_FLAGS,
        index=False,
    )

    print()
    print(
        "=" * 90
    )

    print(
        "AUDIT COMPLETE"
    )

    print(
        "=" * 90
    )

    print()
    print(
        "Season summary:"
    )

    print(
        OUTPUT_SEASON
    )

    print()
    print(
        "Transitions:"
    )

    print(
        OUTPUT_TRANSITIONS
    )

    print()
    print(
        "Flags:"
    )

    print(
        OUTPUT_FLAGS
    )

    print()
    print(
        "NO V5 PARAMETERS CHANGED ✅"
    )


if __name__ == "__main__":
    main()