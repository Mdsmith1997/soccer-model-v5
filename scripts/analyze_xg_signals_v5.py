from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

XG_FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "xg_features_v5.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "xg_signal_analysis_v5.csv"
)


# ============================================================
# SETTINGS
# ============================================================

MIN_HISTORY = 5


# ============================================================
# HELPERS
# ============================================================

def safe_corr(
    x,
    y,
):

    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(
                x,
                errors="coerce",
            ),

            "y": pd.to_numeric(
                y,
                errors="coerce",
            ),
        }
    ).dropna()

    if len(pair) < 3:

        return np.nan

    if (
        pair[
            "x"
        ].std()
        == 0
        or
        pair[
            "y"
        ].std()
        == 0
    ):

        return np.nan

    return pair[
        "x"
    ].corr(
        pair[
            "y"
        ]
    )


def correlation_row(
    df,
    feature,
    target,
    label,
):

    sub = df[
        [
            feature,
            target,
        ]
    ].dropna()

    return {
        "analysis":
            label,

        "feature":
            feature,

        "target":
            target,

        "rows":
            len(sub),

        "correlation":
            safe_corr(
                sub[
                    feature
                ],
                sub[
                    target
                ],
            ),
    }


# ============================================================
# LOAD
# ============================================================

def load_data():

    df = pd.read_csv(
        XG_FEATURE_FILE,
        parse_dates=[
            "date",
        ],
    )

    df[
        "season"
    ] = (
        df[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    return (
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


# ============================================================
# OBSERVED MATCH SIGNAL CORRELATIONS
# ============================================================

def observed_signal_correlations(
    df,
):

    print()
    print("=" * 80)
    print("CURRENT-MATCH SIGNAL CORRELATIONS")
    print("=" * 80)
    print()

    candidates = [
        "goals_for",
        "shots_for",
        "shots_on_target",

        "xg_for",
        "npxg_for",

        "expected_points",
        "ppda",
        "deep_completions",
    ]

    candidates = [
        col
        for col in candidates
        if col in df.columns
    ]

    corr = (
        df[
            candidates
        ]
        .corr()
    )

    print(
        corr
        .round(3)
        .to_string()
    )

    return corr


# ============================================================
# FUTURE PERFORMANCE
# ============================================================

def add_future_targets(
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

    group = out.groupby(
        "team",
        sort=False,
    )

    targets = [
        "goals_for",
        "goals_against",

        "xg_for",
        "xg_against",

        "npxg_for",
        "npxg_against",
    ]

    for col in targets:

        if col not in out.columns:
            continue

        # ----------------------------------------------------
        # NEXT MATCH
        # ----------------------------------------------------

        out[
            f"next_{col}"
        ] = (
            group[
                col
            ]
            .shift(-1)
        )

        # ----------------------------------------------------
        # NEXT 3 MATCHES
        #
        # We deliberately use future observations only.
        # These are evaluation targets, not model features.
        # ----------------------------------------------------

        shifted_1 = (
            group[
                col
            ]
            .shift(-1)
        )

        shifted_2 = (
            group[
                col
            ]
            .shift(-2)
        )

        shifted_3 = (
            group[
                col
            ]
            .shift(-3)
        )

        future_frame = pd.concat(
            [
                shifted_1,
                shifted_2,
                shifted_3,
            ],
            axis=1,
        )

        out[
            f"next3_{col}"
        ] = (
            future_frame
            .mean(
                axis=1,
                skipna=False,
            )
        )

        # ----------------------------------------------------
        # NEXT 5 MATCHES
        # ----------------------------------------------------

        future = []

        for lag in range(
            1,
            6,
        ):

            future.append(
                group[
                    col
                ]
                .shift(
                    -lag
                )
            )

        future_frame = pd.concat(
            future,
            axis=1,
        )

        out[
            f"next5_{col}"
        ] = (
            future_frame
            .mean(
                axis=1,
                skipna=False,
            )
        )

    return out


# ============================================================
# PREDICTIVE CORRELATIONS
# ============================================================

def predictive_signal_analysis(
    df,
):

    print()
    print("=" * 80)
    print("PREGAME SIGNAL → FUTURE PERFORMANCE")
    print("=" * 80)
    print()

    sub = df[
        df[
            "xg_prior_games"
        ]
        >= MIN_HISTORY
    ].copy()

    rows = []

    # ========================================================
    # ATTACK FEATURES
    # ========================================================

    attack_features = [
        "ew_xg_for",
        "ew_npxg_for",

        "ew_expected_points",
        "ew_ppda",
        "ew_deep_completions",

        "xg_attack_strength",
        "npxg_attack_strength",

        "venue_ew_xg_for",
        "venue_ew_npxg_for",

        "venue_xg_attack_strength",
        "venue_npxg_attack_strength",
    ]

    # Add existing model signals if available.
    existing_attack = [
        "goals_for",
        "shots_for",
        "shots_on_target",
    ]

    attack_features.extend(
        [
            col
            for col in existing_attack
            if col in sub.columns
        ]
    )

    attack_targets = [
        "next_goals_for",
        "next3_goals_for",
        "next5_goals_for",

        "next_xg_for",
        "next3_xg_for",
        "next5_xg_for",
    ]

    for feature in attack_features:

        if feature not in sub.columns:
            continue

        for target in attack_targets:

            if target not in sub.columns:
                continue

            rows.append(
                correlation_row(
                    sub,
                    feature,
                    target,
                    "ATTACK",
                )
            )

    # ========================================================
    # DEFENSE FEATURES
    # ========================================================

    defense_features = [
        "ew_xg_against",
        "ew_npxg_against",

        "xg_defense_strength",
        "npxg_defense_strength",

        "venue_ew_xg_against",
        "venue_ew_npxg_against",

        "venue_xg_defense_strength",
        "venue_npxg_defense_strength",
    ]

    existing_defense = [
        "goals_against",
        "shots_against",
        "sot_against",
        "shots_on_target_against",
    ]

    defense_features.extend(
        [
            col
            for col in existing_defense
            if col in sub.columns
        ]
    )

    defense_targets = [
        "next_goals_against",
        "next3_goals_against",
        "next5_goals_against",

        "next_xg_against",
        "next3_xg_against",
        "next5_xg_against",
    ]

    for feature in defense_features:

        if feature not in sub.columns:
            continue

        for target in defense_targets:

            if target not in sub.columns:
                continue

            rows.append(
                correlation_row(
                    sub,
                    feature,
                    target,
                    "DEFENSE",
                )
            )

    results = pd.DataFrame(
        rows
    )

    results[
        "abs_correlation"
    ] = (
        results[
            "correlation"
        ]
        .abs()
    )

    results = (
        results
        .sort_values(
            [
                "target",
                "abs_correlation",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # PRINT FUTURE GOALS
    # ========================================================

    for target in [
        "next_goals_for",
        "next3_goals_for",
        "next5_goals_for",
        "next_goals_against",
        "next3_goals_against",
        "next5_goals_against",
    ]:

        current = results[
            results[
                "target"
            ]
            == target
        ].copy()

        if current.empty:
            continue

        print()
        print(
            f"TARGET: {target}"
        )

        print(
            "-" * 80
        )

        print(
            current[
                [
                    "feature",
                    "rows",
                    "correlation",
                ]
            ]
            .head(15)
            .round(4)
            .to_string(
                index=False
            )
        )

    return results


# ============================================================
# BY LEAGUE
# ============================================================

def league_predictive_analysis(
    df,
):

    print()
    print("=" * 80)
    print("XG PREDICTIVE VALUE BY LEAGUE")
    print("=" * 80)

    rows = []

    for league, sub in df.groupby(
        "league"
    ):

        sub = sub[
            sub[
                "xg_prior_games"
            ]
            >= MIN_HISTORY
        ].copy()

        candidates = [
            "ew_xg_for",
            "ew_npxg_for",

            "xg_attack_strength",
            "npxg_attack_strength",

            "venue_ew_xg_for",
            "venue_ew_npxg_for",
        ]

        targets = [
            "next3_goals_for",
            "next5_goals_for",
        ]

        for feature in candidates:

            for target in targets:

                rows.append(
                    {
                        "league":
                            league,

                        "feature":
                            feature,

                        "target":
                            target,

                        "rows":
                            sub[
                                [
                                    feature,
                                    target,
                                ]
                            ]
                            .dropna()
                            .shape[0],

                        "correlation":
                            safe_corr(
                                sub[
                                    feature
                                ],
                                sub[
                                    target
                                ],
                            ),
                    }
                )

    results = pd.DataFrame(
        rows
    )

    for league in sorted(
        results[
            "league"
        ].unique()
    ):

        print()
        print(
            league
        )

        print(
            "-" * 80
        )

        league_rows = results[
            results[
                "league"
            ]
            == league
        ]

        print(
            league_rows
            .round(4)
            .to_string(
                index=False
            )
        )

    return results


# ============================================================
# XG VS NPXG DIRECT COMPARISON
# ============================================================

def compare_xg_npxg(
    df,
):

    print()
    print("=" * 80)
    print("XG VS NON-PENALTY XG")
    print("=" * 80)
    print()

    sub = df[
        df[
            "xg_prior_games"
        ]
        >= MIN_HISTORY
    ].copy()

    pairs = [
        (
            "ew_xg_for",
            "ew_npxg_for",
            "next3_goals_for",
        ),

        (
            "ew_xg_for",
            "ew_npxg_for",
            "next5_goals_for",
        ),

        (
            "xg_attack_strength",
            "npxg_attack_strength",
            "next3_goals_for",
        ),

        (
            "xg_attack_strength",
            "npxg_attack_strength",
            "next5_goals_for",
        ),

        (
            "ew_xg_against",
            "ew_npxg_against",
            "next3_goals_against",
        ),

        (
            "ew_xg_against",
            "ew_npxg_against",
            "next5_goals_against",
        ),
    ]

    rows = []

    for xg_feature, npxg_feature, target in pairs:

        xg_corr = safe_corr(
            sub[
                xg_feature
            ],
            sub[
                target
            ],
        )

        npxg_corr = safe_corr(
            sub[
                npxg_feature
            ],
            sub[
                target
            ],
        )

        rows.append(
            {
                "target":
                    target,

                "xg_feature":
                    xg_feature,

                "xg_corr":
                    xg_corr,

                "npxg_feature":
                    npxg_feature,

                "npxg_corr":
                    npxg_corr,

                "winner":
                    (
                        "xG"
                        if abs(
                            xg_corr
                        )
                        >
                        abs(
                            npxg_corr
                        )
                        else
                        "npxG"
                    ),
            }
        )

    results = pd.DataFrame(
        rows
    )

    print(
        results
        .round(4)
        .to_string(
            index=False
        )
    )

    return results


# ============================================================
# SIGNAL REDUNDANCY
# ============================================================

def redundancy_analysis(
    df,
):

    print()
    print("=" * 80)
    print("SIGNAL REDUNDANCY")
    print("=" * 80)
    print()

    cols = [
        "ew_xg_for",
        "ew_npxg_for",

        "ew_expected_points",
        "ew_ppda",
        "ew_deep_completions",

        "xg_attack_strength",
        "npxg_attack_strength",

        "venue_ew_xg_for",
        "venue_ew_npxg_for",
    ]

    cols = [
        col
        for col in cols
        if col in df.columns
    ]

    sub = df[
        df[
            "xg_prior_games"
        ]
        >= MIN_HISTORY
    ]

    corr = (
        sub[
            cols
        ]
        .corr()
    )

    print(
        corr
        .round(3)
        .to_string()
    )

    return corr


# ============================================================
# CURRENT MATCH DESCRIPTIVE CORRELATIONS
# ============================================================

def match_level_quality(
    df,
):

    print()
    print("=" * 80)
    print("CURRENT-MATCH GOALS VS PROCESS SIGNALS")
    print("=" * 80)
    print()

    attack = [
        "goals_for",
        "xg_for",
        "npxg_for",
        "shots_for",
        "shots_on_target",
        "expected_points",
        "ppda",
        "deep_completions",
    ]

    attack = [
        c
        for c in attack
        if c in df.columns
    ]

    if (
        "goals_for"
        not in attack
    ):

        print(
            "goals_for unavailable."
        )

        return pd.DataFrame()

    rows = []

    for feature in attack:

        if feature == "goals_for":
            continue

        rows.append(
            {
                "feature":
                    feature,

                "target":
                    "goals_for",

                "rows":
                    df[
                        [
                            feature,
                            "goals_for",
                        ]
                    ]
                    .dropna()
                    .shape[0],

                "correlation":
                    safe_corr(
                        df[
                            feature
                        ],
                        df[
                            "goals_for"
                        ],
                    ),
            }
        )

    results = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "correlation",
            ascending=False,
        )
    )

    print(
        results
        .round(4)
        .to_string(
            index=False
        )
    )

    return results


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_results(
    predictive,
    league_results,
    xg_npxg,
    current_match,
):

    frames = []

    # --------------------------------------------------------
    # MAIN PREDICTIVE
    # --------------------------------------------------------

    p = predictive.copy()

    p[
        "section"
    ] = "predictive"

    frames.append(
        p
    )

    # --------------------------------------------------------
    # LEAGUE
    # --------------------------------------------------------

    l = league_results.copy()

    l[
        "section"
    ] = "league"

    frames.append(
        l
    )

    # --------------------------------------------------------
    # XG VS NPXG
    # --------------------------------------------------------

    x = xg_npxg.copy()

    x[
        "section"
    ] = "xg_vs_npxg"

    frames.append(
        x
    )

    # --------------------------------------------------------
    # CURRENT MATCH
    # --------------------------------------------------------

    c = current_match.copy()

    if not c.empty:

        c[
            "section"
        ] = "current_match"

        frames.append(
            c
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("ANALYZING XG SIGNALS V5")
    print("==============================")
    print()

    df = load_data()

    print(
        f"Team rows: "
        f"{len(df):,}"
    )

    print(
        f"Matches: "
        f"{df['match_id'].nunique():,}"
    )

    print(
        f"Minimum history: "
        f"{MIN_HISTORY} matches"
    )

    # ========================================================
    # FUTURE TARGETS
    # ========================================================

    print(
        "Building future evaluation targets..."
    )

    df = add_future_targets(
        df
    )

    # ========================================================
    # ANALYSES
    # ========================================================

    observed_signal_correlations(
        df
    )

    current_match = (
        match_level_quality(
            df
        )
    )

    redundancy_analysis(
        df
    )

    predictive = (
        predictive_signal_analysis(
            df
        )
    )

    league_results = (
        league_predictive_analysis(
            df
        )
    )

    xg_npxg = compare_xg_npxg(
        df
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_results(
        predictive,
        league_results,
        xg_npxg,
        current_match,
    )

    print()
    print("==============================")
    print("XG SIGNAL ANALYSIS COMPLETE")
    print("==============================")
    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()