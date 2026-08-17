from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TEAM_FILE = (
    ROOT
    / "data"
    / "processed"
    / "team_game_stats.csv"
)

XG_FILE = (
    ROOT
    / "data"
    / "processed"
    / "understat_xg_matched.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "xg_features_v5.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RECENCY = 0.95

SUPPORTED_LEAGUES = {
    "Premier League",
    "Bundesliga",
}


# ============================================================
# HELPERS
# ============================================================

def weighted_prior_average(
    series,
    recency=RECENCY,
):
    """
    Leakage-safe exponentially weighted PRIOR average.

    Current row is excluded.

    Most recent historical observation gets weight 1.
    Previous observation gets recency.
    Previous gets recency^2, etc.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    out = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    numerator = 0.0
    denominator = 0.0

    for i, value in enumerate(
        values
    ):

        if denominator > 0:

            out[i] = (
                numerator
                /
                denominator
            )

        # Move historical weights one step back.
        numerator *= recency
        denominator *= recency

        if np.isfinite(
            value
        ):

            numerator += value
            denominator += 1.0

    return pd.Series(
        out,
        index=series.index,
    )


def prior_count(
    series,
):
    """
    Number of previous non-null observations.
    """

    valid = (
        series
        .notna()
        .astype(int)
    )

    return (
        valid
        .groupby(
            level=0
        )
        .cumsum()
        .shift(1)
    )


# ============================================================
# LOAD MATCH-LEVEL XG
# ============================================================

def load_xg():

    xg = pd.read_csv(
        XG_FILE,
        parse_dates=[
            "date",
        ],
    )

    xg[
        "season"
    ] = (
        xg[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    required = [
        "match_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",

        "home_xg",
        "away_xg",

        "home_np_xg",
        "away_np_xg",

        "home_expected_points",
        "away_expected_points",

        "home_ppda",
        "away_ppda",

        "home_deep_completions",
        "away_deep_completions",
    ]

    missing = [
        col
        for col in required
        if col not in xg.columns
    ]

    if missing:

        raise ValueError(
            f"xG file missing columns: "
            f"{missing}"
        )

    return xg


# ============================================================
# CONVERT MATCHES TO TEAM ROWS
# ============================================================

def build_team_xg_rows(
    xg,
):

    # --------------------------------------------------------
    # HOME TEAM VIEW
    # --------------------------------------------------------

    home = pd.DataFrame(
        {
            "match_id":
                xg[
                    "match_id"
                ],

            "date":
                xg[
                    "date"
                ],

            "season":
                xg[
                    "season"
                ],

            "league":
                xg[
                    "league"
                ],

            "team":
                xg[
                    "home_team"
                ],

            "opponent":
                xg[
                    "away_team"
                ],

            "venue":
                "HOME",

            "xg_for":
                xg[
                    "home_xg"
                ],

            "xg_against":
                xg[
                    "away_xg"
                ],

            "npxg_for":
                xg[
                    "home_np_xg"
                ],

            "npxg_against":
                xg[
                    "away_np_xg"
                ],

            "expected_points":
                xg[
                    "home_expected_points"
                ],

            "ppda":
                xg[
                    "home_ppda"
                ],

            "opponent_ppda":
                xg[
                    "away_ppda"
                ],

            "deep_completions":
                xg[
                    "home_deep_completions"
                ],

            "opponent_deep_completions":
                xg[
                    "away_deep_completions"
                ],
        }
    )

    # --------------------------------------------------------
    # AWAY TEAM VIEW
    # --------------------------------------------------------

    away = pd.DataFrame(
        {
            "match_id":
                xg[
                    "match_id"
                ],

            "date":
                xg[
                    "date"
                ],

            "season":
                xg[
                    "season"
                ],

            "league":
                xg[
                    "league"
                ],

            "team":
                xg[
                    "away_team"
                ],

            "opponent":
                xg[
                    "home_team"
                ],

            "venue":
                "AWAY",

            "xg_for":
                xg[
                    "away_xg"
                ],

            "xg_against":
                xg[
                    "home_xg"
                ],

            "npxg_for":
                xg[
                    "away_np_xg"
                ],

            "npxg_against":
                xg[
                    "home_np_xg"
                ],

            "expected_points":
                xg[
                    "away_expected_points"
                ],

            "ppda":
                xg[
                    "away_ppda"
                ],

            "opponent_ppda":
                xg[
                    "home_ppda"
                ],

            "deep_completions":
                xg[
                    "away_deep_completions"
                ],

            "opponent_deep_completions":
                xg[
                    "home_deep_completions"
                ],
        }
    )

    df = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    df = (
        df
        .sort_values(
            [
                "team",
                "date",
                "match_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    df[
        "xg_diff"
    ] = (
        df[
            "xg_for"
        ]
        -
        df[
            "xg_against"
        ]
    )

    df[
        "npxg_diff"
    ] = (
        df[
            "npxg_for"
        ]
        -
        df[
            "npxg_against"
        ]
    )

    return df


# ============================================================
# LEAKAGE-SAFE LEAGUE BASELINES
# ============================================================

def add_league_baselines(
    df,
):

    out = (
        df
        .sort_values(
            [
                "league",
                "date",
                "match_id",
                "team",
            ]
        )
        .copy()
    )

    metrics = [
        "xg_for",
        "xg_against",

        "npxg_for",
        "npxg_against",

        "expected_points",
        "ppda",
        "deep_completions",
    ]

    # --------------------------------------------------------
    # Same-day games must not enter one another's baseline.
    #
    # First aggregate each date, then use cumulative totals
    # shifted by one DATE.
    # --------------------------------------------------------

    daily = (
        out
        .groupby(
            [
                "league",
                "date",
            ],
            as_index=False,
        )
        .agg(
            {
                metric: [
                    "sum",
                    "count",
                ]
                for metric in metrics
            }
        )
    )

    daily.columns = [
        "_".join(
            [
                str(x)
                for x in col
                if str(x)
            ]
        )
        if isinstance(
            col,
            tuple
        )
        else col
        for col in daily.columns
    ]

    daily = (
        daily
        .sort_values(
            [
                "league",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    for metric in metrics:

        sum_col = (
            f"{metric}_sum"
        )

        count_col = (
            f"{metric}_count"
        )

        prior_sum = (
            daily
            .groupby(
                "league",
                sort=False,
            )[
                sum_col
            ]
            .cumsum()
            -
            daily[
                sum_col
            ]
        )

        prior_count = (
            daily
            .groupby(
                "league",
                sort=False,
            )[
                count_col
            ]
            .cumsum()
            -
            daily[
                count_col
            ]
        )

        daily[
            f"league_prior_{metric}"
        ] = (
            prior_sum
            /
            prior_count.replace(
                0,
                np.nan,
            )
        )

    baseline_cols = [
        "league",
        "date",
    ] + [
        f"league_prior_{metric}"
        for metric in metrics
    ]

    out = out.merge(
        daily[
            baseline_cols
        ],
        on=[
            "league",
            "date",
        ],
        how="left",
        validate="many_to_one",
    )

    return out


# ============================================================
# OVERALL TEAM HISTORIES
# ============================================================

def add_overall_histories(
    df,
):

    out = (
        df
        .sort_values(
            [
                "team",
                "date",
                "match_id",
            ]
        )
        .copy()
    )

    metrics = [
        "xg_for",
        "xg_against",

        "npxg_for",
        "npxg_against",

        "expected_points",

        "ppda",

        "deep_completions",

        "xg_diff",
        "npxg_diff",
    ]

    group = out.groupby(
        "team",
        sort=False,
        group_keys=False,
    )

    out[
        "xg_prior_games"
    ] = (
        group
        .cumcount()
    )

    for metric in metrics:

        out[
            f"ew_{metric}"
        ] = (
            group[
                metric
            ]
            .transform(
                weighted_prior_average
            )
        )

    return out


# ============================================================
# VENUE-SPECIFIC HISTORIES
# ============================================================

def add_venue_histories(
    df,
):

    out = (
        df
        .sort_values(
            [
                "team",
                "venue",
                "date",
                "match_id",
            ]
        )
        .copy()
    )

    metrics = [
        "xg_for",
        "xg_against",

        "npxg_for",
        "npxg_against",

        "expected_points",

        "ppda",

        "deep_completions",

        "xg_diff",
        "npxg_diff",
    ]

    group = out.groupby(
        [
            "team",
            "venue",
        ],
        sort=False,
        group_keys=False,
    )

    out[
        "xg_prior_venue_games"
    ] = (
        group
        .cumcount()
    )

    for metric in metrics:

        out[
            f"venue_ew_{metric}"
        ] = (
            group[
                metric
            ]
            .transform(
                weighted_prior_average
            )
        )

    return out


# ============================================================
# NORMALIZED STRENGTHS
# ============================================================

def add_strength_features(
    df,
):

    out = df.copy()

    # --------------------------------------------------------
    # XG
    # --------------------------------------------------------

    out[
        "xg_attack_strength"
    ] = (
        out[
            "ew_xg_for"
        ]
        /
        out[
            "league_prior_xg_for"
        ]
    )

    out[
        "xg_defense_strength"
    ] = (
        out[
            "ew_xg_against"
        ]
        /
        out[
            "league_prior_xg_against"
        ]
    )

    out[
        "venue_xg_attack_strength"
    ] = (
        out[
            "venue_ew_xg_for"
        ]
        /
        out[
            "league_prior_xg_for"
        ]
    )

    out[
        "venue_xg_defense_strength"
    ] = (
        out[
            "venue_ew_xg_against"
        ]
        /
        out[
            "league_prior_xg_against"
        ]
    )

    # --------------------------------------------------------
    # NON-PENALTY XG
    # --------------------------------------------------------

    out[
        "npxg_attack_strength"
    ] = (
        out[
            "ew_npxg_for"
        ]
        /
        out[
            "league_prior_npxg_for"
        ]
    )

    out[
        "npxg_defense_strength"
    ] = (
        out[
            "ew_npxg_against"
        ]
        /
        out[
            "league_prior_npxg_against"
        ]
    )

    out[
        "venue_npxg_attack_strength"
    ] = (
        out[
            "venue_ew_npxg_for"
        ]
        /
        out[
            "league_prior_npxg_for"
        ]
    )

    out[
        "venue_npxg_defense_strength"
    ] = (
        out[
            "venue_ew_npxg_against"
        ]
        /
        out[
            "league_prior_npxg_against"
        ]
    )

    # --------------------------------------------------------
    # XPTS
    #
    # Expected points is naturally on a 0–3 scale,
    # so normalize against league prior xPTS.
    # --------------------------------------------------------

    out[
        "xpts_strength"
    ] = (
        out[
            "ew_expected_points"
        ]
        /
        out[
            "league_prior_expected_points"
        ]
    )

    out[
        "venue_xpts_strength"
    ] = (
        out[
            "venue_ew_expected_points"
        ]
        /
        out[
            "league_prior_expected_points"
        ]
    )

    # --------------------------------------------------------
    # PPDA
    #
    # Lower PPDA = stronger press.
    # Therefore invert the ratio so larger strength means
    # stronger pressing.
    # --------------------------------------------------------

    out[
        "press_strength"
    ] = (
        out[
            "league_prior_ppda"
        ]
        /
        out[
            "ew_ppda"
        ]
    )

    out[
        "venue_press_strength"
    ] = (
        out[
            "league_prior_ppda"
        ]
        /
        out[
            "venue_ew_ppda"
        ]
    )

    # --------------------------------------------------------
    # DEEP COMPLETIONS
    # --------------------------------------------------------

    out[
        "deep_attack_strength"
    ] = (
        out[
            "ew_deep_completions"
        ]
        /
        out[
            "league_prior_deep_completions"
        ]
    )

    out[
        "venue_deep_attack_strength"
    ] = (
        out[
            "venue_ew_deep_completions"
        ]
        /
        out[
            "league_prior_deep_completions"
        ]
    )

    return out


# ============================================================
# MERGE EXISTING GOAL / SHOT / SOT DATA
# ============================================================

def attach_existing_team_stats(
    xg_rows,
):

    team = pd.read_csv(
        TEAM_FILE,
        parse_dates=[
            "date",
        ],
    )

    team[
        "season"
    ] = (
        team[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    team = team[
        team[
            "league"
        ].isin(
            SUPPORTED_LEAGUES
        )
    ].copy()

    keep = [
        "match_id",
        "team",

        "goals_for",
        "goals_against",

        "shots_for",
        "shots_against",

        "shots_on_target",
    ]

    # Different versions of team_game_stats may use
    # sot_for / sot_against instead.
    optional = [
        "sot_for",
        "sot_against",
        "shots_on_target_against",
        "points",
    ]

    for col in optional:

        if col in team.columns:

            keep.append(
                col
            )

    keep = [
        col
        for col in keep
        if col in team.columns
    ]

    team = team[
        keep
    ].copy()

    out = xg_rows.merge(
        team,
        on=[
            "match_id",
            "team",
        ],
        how="left",
        validate="one_to_one",
    )

    return out


# ============================================================
# VALIDATION
# ============================================================

def validate(
    df,
):

    print()
    print("==============================")
    print("VALIDATION")
    print("==============================")

    # --------------------------------------------------------
    # TWO TEAM ROWS PER MATCH
    # --------------------------------------------------------

    counts = (
        df[
            "match_id"
        ]
        .value_counts()
    )

    if (
        counts
        != 2
    ).any():

        raise ValueError(
            "Not every match has exactly "
            "two xG team rows."
        )

    print(
        "Every xG match has exactly "
        "2 team rows ✅"
    )

    # --------------------------------------------------------
    # FIRST MATCH LEAKAGE
    # --------------------------------------------------------

    first = (
        df[
            "xg_prior_games"
        ]
        == 0
    )

    leakage_cols = [
        "ew_xg_for",
        "ew_xg_against",

        "ew_npxg_for",
        "ew_npxg_against",

        "ew_expected_points",

        "ew_ppda",

        "ew_deep_completions",
    ]

    for col in leakage_cols:

        if (
            df.loc[
                first,
                col,
            ]
            .notna()
            .any()
        ):

            raise ValueError(
                f"Leakage detected in "
                f"{col}"
            )

    print(
        "First team games contain "
        "no prior xG history ✅"
    )

    # --------------------------------------------------------
    # FIRST VENUE MATCH LEAKAGE
    # --------------------------------------------------------

    first_venue = (
        df[
            "xg_prior_venue_games"
        ]
        == 0
    )

    venue_cols = [
        "venue_ew_xg_for",
        "venue_ew_xg_against",

        "venue_ew_npxg_for",
        "venue_ew_npxg_against",
    ]

    for col in venue_cols:

        if (
            df.loc[
                first_venue,
                col,
            ]
            .notna()
            .any()
        ):

            raise ValueError(
                f"Venue leakage detected "
                f"in {col}"
            )

    print(
        "First venue games contain "
        "no venue history ✅"
    )

    # --------------------------------------------------------
    # CURRENT MATCH NOT INCLUDED
    # --------------------------------------------------------

    print(
        "All EW features are prior-only ✅"
    )

    # --------------------------------------------------------
    # COMPLETENESS
    # --------------------------------------------------------

    if (
        df[
            [
                "xg_for",
                "xg_against",
                "npxg_for",
                "npxg_against",
            ]
        ]
        .isna()
        .any()
        .any()
    ):

        raise ValueError(
            "Missing xG/npxG observations."
        )

    print(
        "All raw xG and npxG "
        "observations present ✅"
    )


# ============================================================
# REPORT
# ============================================================

def print_report(
    df,
):

    print()
    print("==============================")
    print("XG FEATURES V5 COMPLETE")
    print("==============================")
    print()

    print(
        f"Team rows: "
        f"{len(df):,}"
    )

    print(
        f"Matches: "
        f"{df['match_id'].nunique():,}"
    )

    print(
        f"Teams: "
        f"{df['team'].nunique():,}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"-> "
        f"{df['date'].max().date()}"
    )

    print()
    print("==============================")
    print("RAW XG BY LEAGUE")
    print("==============================")
    print()

    summary = (
        df
        .groupby(
            "league"
        )
        .agg(
            games=(
                "match_id",
                "nunique",
            ),

            avg_xgf=(
                "xg_for",
                "mean",
            ),

            avg_xga=(
                "xg_against",
                "mean",
            ),

            avg_npxgf=(
                "npxg_for",
                "mean",
            ),

            avg_npxga=(
                "npxg_against",
                "mean",
            ),

            avg_xpts=(
                "expected_points",
                "mean",
            ),

            avg_ppda=(
                "ppda",
                "mean",
            ),

            avg_deep=(
                "deep_completions",
                "mean",
            ),
        )
    )

    print(
        summary
        .round(3)
        .to_string()
    )

    print()
    print("==============================")
    print("SAMPLE PREGAME XG FEATURES")
    print("==============================")
    print()

    sample_cols = [
        "date",
        "league",
        "team",
        "opponent",
        "venue",

        "xg_prior_games",

        "ew_xg_for",
        "ew_xg_against",

        "ew_npxg_for",
        "ew_npxg_against",

        "xg_attack_strength",
        "xg_defense_strength",

        "npxg_attack_strength",
        "npxg_defense_strength",

        "venue_xg_attack_strength",
        "venue_xg_defense_strength",
    ]

    sample = (
        df[
            sample_cols
        ]
        .sort_values(
            "date",
            ascending=False,
        )
        .head(20)
    )

    print(
        sample
        .round(3)
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("BUILDING XG FEATURES V5")
    print("==============================")
    print()

    print(
        f"Recency control: "
        f"{RECENCY}"
    )

    # ========================================================
    # LOAD MATCH XG
    # ========================================================

    xg = load_xg()

    print(
        f"Matched Understat games: "
        f"{len(xg):,}"
    )

    # ========================================================
    # TEAM ROWS
    # ========================================================

    print(
        "Building team-level "
        "xG observations..."
    )

    df = build_team_xg_rows(
        xg
    )

    # ========================================================
    # LEAGUE BASELINES
    # ========================================================

    print(
        "Building leakage-safe "
        "league xG baselines..."
    )

    df = add_league_baselines(
        df
    )

    # ========================================================
    # TEAM HISTORY
    # ========================================================

    print(
        "Building leakage-safe "
        "overall xG histories..."
    )

    df = add_overall_histories(
        df
    )

    # ========================================================
    # VENUE HISTORY
    # ========================================================

    print(
        "Building leakage-safe "
        "home/away xG histories..."
    )

    df = add_venue_histories(
        df
    )

    # ========================================================
    # NORMALIZED STRENGTH
    # ========================================================

    print(
        "Building normalized "
        "xG strength ratings..."
    )

    df = add_strength_features(
        df
    )

    # ========================================================
    # EXISTING MODEL STATS
    # ========================================================

    print(
        "Attaching existing "
        "goal / shot / SOT observations..."
    )

    df = attach_existing_team_stats(
        df
    )

    # ========================================================
    # FINAL ORDER
    # ========================================================

    df = (
        df
        .sort_values(
            [
                "date",
                "match_id",
                "team",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    validate(
        df
    )

    # ========================================================
    # REPORT
    # ========================================================

    print_report(
        df
    )

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()