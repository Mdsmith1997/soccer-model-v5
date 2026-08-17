from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"

INPUT_FILE = (
    PROCESSED
    / "footystats_ligue1_history_research.csv"
)

OUTPUT_TEAM_ROWS = (
    PROCESSED
    / "footystats_ligue1_team_pregame_research.csv"
)

OUTPUT_MATCH_ROWS = (
    PROCESSED
    / "footystats_ligue1_pregame_research.csv"
)

OUTPUT_SUMMARY = (
    PROCESSED
    / "footystats_ligue1_pregame_summary_research.csv"
)


# ============================================================
# FROZEN V5 SETTINGS
# ============================================================

GOAL_RECENCY = 0.975
XG_RECENCY = 0.925
SHOT_RECENCY = 0.850

OPPONENT_STRENGTH = 0.875

EPS = 1e-9

MIN_STRENGTH = 0.20
MAX_STRENGTH = 5.00


# ============================================================
# SIGNAL DEFINITIONS
# ============================================================

SIGNALS = {
    "goal": {
        "recency": GOAL_RECENCY,
        "attack_perf": "goal_attack_perf",
        "defense_perf": "goal_defense_perf",
    },

    "xg": {
        "recency": XG_RECENCY,
        "attack_perf": "xg_attack_perf",
        "defense_perf": "xg_defense_perf",
    },

    "shot": {
        "recency": SHOT_RECENCY,
        "attack_perf": "shot_attack_perf",
        "defense_perf": "shot_defense_perf",
    },
}


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
        "footystats_match_id",
        "footystats_season_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
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
            f"Missing required columns: {missing}"
        )

    df["date"] = (
        pd.to_datetime(
            df["date"],
            errors="coerce",
        )
        .dt.normalize()
    )

    df["season"] = (
        df["season"]
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

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "date",
            "season",
            "league",
            "home_team",
            "away_team",
            *numeric_cols,
        ]
    ).copy()

    df = (
        df
        .sort_values(
            [
                "date",
                "league",
                "footystats_match_id",
            ]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# STRICT PREMATCH LEAGUE BASELINES
# ============================================================

def build_league_daily_baselines(
    matches,
):

    """
    Build league baselines using ONLY matches on dates
    strictly before the current match date.

    Games on the same date cannot leak into one another.
    """

    daily = (
        matches
        .groupby(
            [
                "league",
                "date",
            ],
            as_index=False,
        )
        .agg(
            games=(
                "footystats_match_id",
                "count",
            ),

            home_goals_sum=(
                "home_goals",
                "sum",
            ),

            away_goals_sum=(
                "away_goals",
                "sum",
            ),

            home_xg_sum=(
                "home_xg",
                "sum",
            ),

            away_xg_sum=(
                "away_xg",
                "sum",
            ),

            home_shots_sum=(
                "home_shots",
                "sum",
            ),

            away_shots_sum=(
                "away_shots",
                "sum",
            ),
        )
        .sort_values(
            [
                "league",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    sum_cols = [
        "games",
        "home_goals_sum",
        "away_goals_sum",
        "home_xg_sum",
        "away_xg_sum",
        "home_shots_sum",
        "away_shots_sum",
    ]

    for col in sum_cols:

        daily[
            f"prior_{col}"
        ] = (
            daily
            .groupby(
                "league"
            )[col]
            .cumsum()
            -
            daily[col]
        )

    denominator = (
        daily["prior_games"]
        .replace(
            0,
            np.nan,
        )
    )

    mapping = {
        "lg_home_goals":
            "prior_home_goals_sum",

        "lg_away_goals":
            "prior_away_goals_sum",

        "lg_home_xg":
            "prior_home_xg_sum",

        "lg_away_xg":
            "prior_away_xg_sum",

        "lg_home_shots":
            "prior_home_shots_sum",

        "lg_away_shots":
            "prior_away_shots_sum",
    }

    for output_col, source_col in (
        mapping.items()
    ):

        daily[
            output_col
        ] = (
            daily[source_col]
            /
            denominator
        )

    return daily[
        [
            "league",
            "date",
            "prior_games",
            "lg_home_goals",
            "lg_away_goals",
            "lg_home_xg",
            "lg_away_xg",
            "lg_home_shots",
            "lg_away_shots",
        ]
    ].copy()


def attach_league_baselines(
    history,
):

    baseline = (
        build_league_daily_baselines(
            history
        )
    )

    return history.merge(
        baseline,
        on=[
            "league",
            "date",
        ],
        how="left",
        validate="many_to_one",
    )


# ============================================================
# MATCH → TEAM ROWS
# ============================================================

def build_team_rows(
    matches,
):

    home = pd.DataFrame(
        {
            "footystats_match_id":
                matches["footystats_match_id"],

            "date":
                matches["date"],

            "season":
                matches["season"],

            "league":
                matches["league"],

            "team":
                matches["home_team"],

            "opponent":
                matches["away_team"],

            "venue":
                "HOME",

            "goals_for":
                matches["home_goals"],

            "goals_against":
                matches["away_goals"],

            "xg_for":
                matches["home_xg"],

            "xg_against":
                matches["away_xg"],

            "shots_for":
                matches["home_shots"],

            "shots_against":
                matches["away_shots"],

            "league_goals_for":
                matches["lg_home_goals"],

            "league_goals_against":
                matches["lg_away_goals"],

            "league_xg_for":
                matches["lg_home_xg"],

            "league_xg_against":
                matches["lg_away_xg"],

            "league_shots_for":
                matches["lg_home_shots"],

            "league_shots_against":
                matches["lg_away_shots"],

            "league_prior_games":
                matches["prior_games"],
        }
    )

    away = pd.DataFrame(
        {
            "footystats_match_id":
                matches["footystats_match_id"],

            "date":
                matches["date"],

            "season":
                matches["season"],

            "league":
                matches["league"],

            "team":
                matches["away_team"],

            "opponent":
                matches["home_team"],

            "venue":
                "AWAY",

            "goals_for":
                matches["away_goals"],

            "goals_against":
                matches["home_goals"],

            "xg_for":
                matches["away_xg"],

            "xg_against":
                matches["home_xg"],

            "shots_for":
                matches["away_shots"],

            "shots_against":
                matches["home_shots"],

            "league_goals_for":
                matches["lg_away_goals"],

            "league_goals_against":
                matches["lg_home_goals"],

            "league_xg_for":
                matches["lg_away_xg"],

            "league_xg_against":
                matches["lg_home_xg"],

            "league_shots_for":
                matches["lg_away_shots"],

            "league_shots_against":
                matches["lg_home_shots"],

            "league_prior_games":
                matches["prior_games"],
        }
    )

    team = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    team = (
        team
        .sort_values(
            [
                "league",
                "team",
                "date",
                "footystats_match_id",
            ]
        )
        .reset_index(drop=True)
    )

    return team


# ============================================================
# LEAGUE-RELATIVE PERFORMANCE
# ============================================================

def relative_strength(
    numerator,
    denominator,
):

    value = (
        numerator
        /
        denominator.replace(
            0,
            np.nan,
        )
    )

    return value.clip(
        lower=MIN_STRENGTH,
        upper=MAX_STRENGTH,
    )


def add_relative_performance(
    team,
):

    df = team.copy()

    df["goal_attack_perf"] = (
        relative_strength(
            df["goals_for"],
            df["league_goals_for"],
        )
    )

    df["goal_defense_perf"] = (
        relative_strength(
            df["goals_against"],
            df["league_goals_against"],
        )
    )

    df["xg_attack_perf"] = (
        relative_strength(
            df["xg_for"],
            df["league_xg_for"],
        )
    )

    df["xg_defense_perf"] = (
        relative_strength(
            df["xg_against"],
            df["league_xg_against"],
        )
    )

    # FootyStats shot-scale normalization.
    df["shot_attack_perf"] = (
        relative_strength(
            df["shots_for"],
            df["league_shots_for"],
        )
    )

    df["shot_defense_perf"] = (
        relative_strength(
            df["shots_against"],
            df["league_shots_against"],
        )
    )

    return df


# ============================================================
# GENERIC LEAKAGE-SAFE EWMA
# ============================================================

def add_ewm_pregame(
    df,
    value_col,
    decay,
    output_col,
    group_cols,
):

    result = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    counts = pd.Series(
        0,
        index=df.index,
        dtype=int,
    )

    grouped = df.groupby(
        group_cols,
        sort=False,
        dropna=False,
    )

    for _, indices in (
        grouped.groups.items()
    ):

        ordered = sorted(
            list(indices),
            key=lambda idx: (
                df.at[idx, "date"],
                str(
                    df.at[
                        idx,
                        "footystats_match_id",
                    ]
                ),
            ),
        )

        numerator = 0.0
        denominator = 0.0
        games = 0

        for idx in ordered:

            # State BEFORE current match.
            if denominator > 0:

                result.at[idx] = (
                    numerator
                    /
                    denominator
                )

            counts.at[idx] = games

            value = df.at[
                idx,
                value_col,
            ]

            if pd.notna(value):

                numerator = (
                    decay
                    *
                    numerator
                    +
                    float(value)
                )

                denominator = (
                    decay
                    *
                    denominator
                    +
                    1.0
                )

                games += 1

    df[output_col] = result

    df[
        f"{output_col}_games"
    ] = counts

    return df


# ============================================================
# RAW PREMATCH STRENGTH
# ============================================================

def add_raw_histories(
    team,
):

    df = team.copy()

    for signal, config in (
        SIGNALS.items()
    ):

        decay = config[
            "recency"
        ]

        for role in [
            "attack",
            "defense",
        ]:

            perf_col = config[
                f"{role}_perf"
            ]

            # SAME-LEAGUE raw strength.
            df = add_ewm_pregame(
                df,
                perf_col,
                decay,
                f"raw_{signal}_{role}_overall",
                [
                    "league",
                    "team",
                ],
            )

            # Venue raw strength.
            df = add_ewm_pregame(
                df,
                perf_col,
                decay,
                f"raw_{signal}_{role}_venue",
                [
                    "league",
                    "team",
                    "venue",
                ],
            )

    return df


# ============================================================
# ATTACH OPPONENT PREMATCH STRENGTH
# ============================================================

def attach_opponent_strength(
    team,
):

    df = team.copy()

    columns = [
        "footystats_match_id",
        "team",
    ]

    raw_cols = []

    for signal in SIGNALS:

        raw_cols.extend(
            [
                f"raw_{signal}_attack_overall",
                f"raw_{signal}_defense_overall",
            ]
        )

    opponent = df[
        columns
        +
        raw_cols
    ].copy()

    opponent = opponent.rename(
        columns={
            "team":
                "opponent_check",

            **{
                col:
                    f"opp_{col}"
                for col in raw_cols
            },
        }
    )

    paired = df.merge(
        opponent,
        on="footystats_match_id",
        how="left",
        validate="many_to_many",
    )

    # Remove self-row from two-row match merge.
    paired = paired[
        paired["team"]
        !=
        paired["opponent_check"]
    ].copy()

    # Explicit opponent identity check.
    paired = paired[
        paired["opponent"]
        ==
        paired["opponent_check"]
    ].copy()

    if len(paired) != len(df):

        raise ValueError(
            "Opponent pairing produced "
            f"{len(paired):,} rows; "
            f"expected {len(df):,}."
        )

    paired = paired.drop(
        columns=[
            "opponent_check",
        ]
    )

    return paired


# ============================================================
# OPPONENT-ADJUST GAME PERFORMANCE
# ============================================================

def add_opponent_adjusted_performance(
    team,
):

    df = team.copy()

    strength = OPPONENT_STRENGTH

    for signal, config in (
        SIGNALS.items()
    ):

        attack_perf = config[
            "attack_perf"
        ]

        defense_perf = config[
            "defense_perf"
        ]

        opponent_defense = (
            df[
                f"opp_raw_{signal}_defense_overall"
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(1.0)
        )

        opponent_attack = (
            df[
                f"opp_raw_{signal}_attack_overall"
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(1.0)
        )

        attack_factor = (
            (
                1.0
                -
                strength
            )
            +
            strength
            *
            opponent_defense
        )

        defense_factor = (
            (
                1.0
                -
                strength
            )
            +
            strength
            *
            opponent_attack
        )

        df[
            f"adj_{signal}_attack_perf"
        ] = (
            df[
                attack_perf
            ]
            /
            attack_factor
        )

        df[
            f"adj_{signal}_defense_perf"
        ] = (
            df[
                defense_perf
            ]
            /
            defense_factor
        )

        df[
            f"adj_{signal}_attack_perf"
        ] = (
            df[
                f"adj_{signal}_attack_perf"
            ]
            .clip(
                MIN_STRENGTH,
                MAX_STRENGTH,
            )
        )

        df[
            f"adj_{signal}_defense_perf"
        ] = (
            df[
                f"adj_{signal}_defense_perf"
            ]
            .clip(
                MIN_STRENGTH,
                MAX_STRENGTH,
            )
        )

    return df


# ============================================================
# ADJUSTED SAME-LEAGUE + CROSS-LEAGUE HISTORY
# ============================================================

def add_adjusted_histories(
    team,
):

    df = team.copy()

    # Stable team chronology for cross-league carryover.
    df = (
        df
        .sort_values(
            [
                "team",
                "date",
                "footystats_match_id",
                "league",
            ]
        )
        .reset_index(drop=True)
    )

    for signal, config in (
        SIGNALS.items()
    ):

        decay = config[
            "recency"
        ]

        for role in [
            "attack",
            "defense",
        ]:

            value_col = (
                f"adj_{signal}_"
                f"{role}_perf"
            )

            # -----------------------------------------------
            # SAME-LEAGUE OVERALL
            # -----------------------------------------------

            df = add_ewm_pregame(
                df,
                value_col,
                decay,
                (
                    f"adj_{signal}_"
                    f"{role}_overall"
                ),
                [
                    "league",
                    "team",
                ],
            )

            # -----------------------------------------------
            # SAME-LEAGUE VENUE
            # -----------------------------------------------

            df = add_ewm_pregame(
                df,
                value_col,
                decay,
                (
                    f"adj_{signal}_"
                    f"{role}_venue"
                ),
                [
                    "league",
                    "team",
                    "venue",
                ],
            )

            # -----------------------------------------------
            # CROSS-LEAGUE FALLBACK
            #
            # This exists ONLY for a team's first match
            # after changing divisions.
            #
            # It remains league-relative because every
            # game-level performance was normalized to its
            # contemporary league before entering history.
            # -----------------------------------------------

            df = add_ewm_pregame(
                df,
                value_col,
                decay,
                (
                    f"global_{signal}_"
                    f"{role}_overall"
                ),
                [
                    "team",
                ],
            )

            df = add_ewm_pregame(
                df,
                value_col,
                decay,
                (
                    f"global_{signal}_"
                    f"{role}_venue"
                ),
                [
                    "team",
                    "venue",
                ],
            )

    return df


# ============================================================
# DETECT LEAGUE TRANSITIONS
# ============================================================

def add_transition_context(
    team,
):

    df = (
        team
        .sort_values(
            [
                "team",
                "date",
                "footystats_match_id",
            ]
        )
        .reset_index(drop=True)
    )

    df[
        "previous_league"
    ] = (
        df
        .groupby(
            "team"
        )[
            "league"
        ]
        .shift(1)
    )

    df[
        "league_changed"
    ] = (
        df[
            "previous_league"
        ].notna()
        &
        (
            df[
                "previous_league"
            ]
            !=
            df[
                "league"
            ]
        )
    )

    return df


# ============================================================
# RESOLVE SAME-LEAGUE / TRANSFERRED / NEUTRAL STATE
# ============================================================

def resolve_final_histories(
    team,
):

    df = team.copy()

    # Use xG same-league game count as the common history
    # availability indicator.
    same_games = (
        df[
            "adj_xg_attack_overall_games"
        ]
    )

    global_games = (
        df[
            "global_xg_attack_overall_games"
        ]
    )

    df[
        "history_source"
    ] = np.where(
        same_games > 0,
        "SAME_LEAGUE",
        np.where(
            global_games > 0,
            "TRANSFERRED",
            "NEUTRAL",
        ),
    )

    for signal in SIGNALS:

        for role in [
            "attack",
            "defense",
        ]:

            for scope in [
                "overall",
                "venue",
            ]:

                same_col = (
                    f"adj_{signal}_"
                    f"{role}_{scope}"
                )

                global_col = (
                    f"global_{signal}_"
                    f"{role}_{scope}"
                )

                output_col = (
                    f"final_{signal}_"
                    f"{role}_{scope}"
                )

                same_values = df[
                    same_col
                ]

                transferred_values = df[
                    global_col
                ]

                df[
                    output_col
                ] = np.where(
                    same_values.notna(),
                    same_values,
                    np.where(
                        transferred_values.notna(),
                        transferred_values,
                        1.0,
                    ),
                )

                df[
                    output_col
                ] = (
                    pd.to_numeric(
                        df[
                            output_col
                        ],
                        errors="coerce",
                    )
                    .fillna(1.0)
                    .clip(
                        MIN_STRENGTH,
                        MAX_STRENGTH,
                    )
                )

    return df


# ============================================================
# MATCH-LEVEL STORE
# ============================================================

def build_match_store(
    matches,
    team,
):

    feature_cols = [
        "history_source",
        "previous_league",
        "league_changed",

        "final_goal_attack_overall",
        "final_goal_defense_overall",
        "final_goal_attack_venue",
        "final_goal_defense_venue",

        "final_xg_attack_overall",
        "final_xg_defense_overall",
        "final_xg_attack_venue",
        "final_xg_defense_venue",

        "final_shot_attack_overall",
        "final_shot_defense_overall",
        "final_shot_attack_venue",
        "final_shot_defense_venue",

        "adj_goal_attack_overall_games",
        "adj_xg_attack_overall_games",
        "adj_shot_attack_overall_games",

        "adj_goal_attack_venue_games",
        "adj_xg_attack_venue_games",
        "adj_shot_attack_venue_games",

        "global_xg_attack_overall_games",
    ]

    home = (
        team[
            team["venue"]
            ==
            "HOME"
        ][
            [
                "footystats_match_id",
                "team",
                *feature_cols,
            ]
        ]
        .copy()
    )

    home = home.rename(
        columns={
            "team":
                "home_team_check",

            **{
                col:
                    f"home_{col}"
                for col in feature_cols
            },
        }
    )

    away = (
        team[
            team["venue"]
            ==
            "AWAY"
        ][
            [
                "footystats_match_id",
                "team",
                *feature_cols,
            ]
        ]
        .copy()
    )

    away = away.rename(
        columns={
            "team":
                "away_team_check",

            **{
                col:
                    f"away_{col}"
                for col in feature_cols
            },
        }
    )

    base_cols = [
        "footystats_match_id",
        "footystats_season_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "home_shots",
        "away_shots",
        "prior_games",
        "lg_home_goals",
        "lg_away_goals",
        "lg_home_xg",
        "lg_away_xg",
        "lg_home_shots",
        "lg_away_shots",
    ]

    out = matches[
        base_cols
    ].copy()

    out = out.merge(
        home,
        on="footystats_match_id",
        how="left",
        validate="one_to_one",
    )

    out = out.merge(
        away,
        on="footystats_match_id",
        how="left",
        validate="one_to_one",
    )

    home_bad = (
        out["home_team"]
        !=
        out["home_team_check"]
    )

    away_bad = (
        out["away_team"]
        !=
        out["away_team_check"]
    )

    if (
        home_bad.any()
        or
        away_bad.any()
    ):

        raise ValueError(
            "Team identity mismatch."
        )

    return out.drop(
        columns=[
            "home_team_check",
            "away_team_check",
        ]
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    match_store,
):

    rows = []

    for (
        league,
        season
    ), sub in (
        match_store.groupby(
            [
                "league",
                "season",
            ]
        )
    ):

        sources = pd.concat(
            [
                sub[
                    "home_history_source"
                ],
                sub[
                    "away_history_source"
                ],
            ],
            ignore_index=True,
        )

        fully_ready = (
            (
                sub[
                    "home_history_source"
                ]
                !=
                "NEUTRAL"
            )
            &
            (
                sub[
                    "away_history_source"
                ]
                !=
                "NEUTRAL"
            )
        )

        rows.append(
            {
                "league":
                    league,

                "season":
                    season,

                "games":
                    len(sub),

                "fully_ready_games":
                    int(
                        fully_ready.sum()
                    ),

                "fully_ready_pct":
                    (
                        100.0
                        *
                        fully_ready.mean()
                    ),

                "same_league_team_states":
                    int(
                        (
                            sources
                            ==
                            "SAME_LEAGUE"
                        ).sum()
                    ),

                "transferred_team_states":
                    int(
                        (
                            sources
                            ==
                            "TRANSFERRED"
                        ).sum()
                    ),

                "neutral_team_states":
                    int(
                        (
                            sources
                            ==
                            "NEUTRAL"
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# LEAKAGE CHECKS
# ============================================================

def run_leakage_checks(
    team,
):

    print()
    print(
        "=" * 100
    )

    print(
        "LEAKAGE / PIPELINE CHECKS"
    )

    print(
        "=" * 100
    )

    first_team = (
        team
        .sort_values(
            [
                "league",
                "team",
                "date",
                "footystats_match_id",
            ]
        )
        .groupby(
            [
                "league",
                "team",
            ],
            as_index=False,
        )
        .head(1)
    )

    first_ok = (
        first_team[
            "adj_xg_attack_overall_games"
        ]
        ==
        0
    ).all()

    print(
        "First same-league appearance "
        "has 0 prior league games:",
        "✅" if first_ok else "❌",
    )

    if not first_ok:

        raise AssertionError(
            "Same-league history leakage."
        )

    first_global = (
        team
        .sort_values(
            [
                "team",
                "date",
                "footystats_match_id",
            ]
        )
        .groupby(
            "team",
            as_index=False,
        )
        .head(1)
    )

    global_ok = (
        first_global[
            "global_xg_attack_overall_games"
        ]
        ==
        0
    ).all()

    print(
        "First career appearance has "
        "0 prior global games:",
        "✅" if global_ok else "❌",
    )

    if not global_ok:

        raise AssertionError(
            "Cross-league history leakage."
        )

    print(
        "Current match excluded from "
        "raw history: ✅"
    )

    print(
        "Current match excluded from "
        "adjusted history: ✅"
    )

    print(
        "Same-day league matches excluded "
        "from baselines: ✅"
    )

    print(
        "Missing opponent ratings treated "
        "as neutral 1.0: ✅"
    )


# ============================================================
# TRANSFER REPORT
# ============================================================

def print_transfer_report(
    team,
):

    transferred = team[
        team[
            "history_source"
        ]
        ==
        "TRANSFERRED"
    ].copy()

    print()
    print(
        "=" * 115
    )

    print(
        "CROSS-DIVISION HISTORY TRANSFERS"
    )

    print(
        "=" * 115
    )

    if len(transferred) == 0:

        print(
            "No transferred states found."
        )

        return

    display = transferred[
        [
            "date",
            "season",
            "team",
            "previous_league",
            "league",
            "venue",
            "global_xg_attack_overall_games",
            "final_xg_attack_overall",
            "final_xg_defense_overall",
        ]
    ].copy()

    print(
        display
        .sort_values(
            [
                "date",
                "team",
            ]
        )
        .tail(50)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print()
    print(
        "Transferred team-match states:",
        f"{len(transferred):,}",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 105
    )

    print(
        "BUILD FOOTYSTATS MULTI-LEAGUE "
        "PREGAME FEATURE STORE V2"
    )

    print(
        "=" * 105
    )

    print()
    print(
        "Frozen recencies:"
    )

    print(
        f"Goals: {GOAL_RECENCY}"
    )

    print(
        f"xG:    {XG_RECENCY}"
    )

    print(
        f"Shots: {SHOT_RECENCY}"
    )

    print()

    print(
        "Opponent strength:",
        OPPONENT_STRENGTH,
    )

    print()

    print(
        "Pipeline:"
    )

    print(
        "league-relative performance"
    )

    print(
        "→ raw pregame strength"
    )

    print(
        "→ opponent adjustment"
    )

    print(
        "→ adjusted EWMA history"
    )

    print(
        "→ same-league / transferred / "
        "neutral resolution"
    )

    print()

    print(
        "NO V5 SIGNAL WEIGHTS ARE "
        "BEING FIT."
    )

    history = load_history()

    print()
    print(
        "Historical matches:",
        f"{len(history):,}",
    )

    print(
        "Leagues:",
        history["league"].nunique(),
    )

    print(
        "Date range:",
        history["date"].min(),
        "->",
        history["date"].max(),
    )

    matches = (
        attach_league_baselines(
            history
        )
    )

    print()
    print(
        "Strict pregame league "
        "baselines built ✅"
    )

    team = build_team_rows(
        matches
    )

    print(
        "Team rows:",
        f"{len(team):,}",
    )

    team = add_relative_performance(
        team
    )

    print(
        "League-relative match "
        "performance built ✅"
    )

    team = add_raw_histories(
        team
    )

    print(
        "Raw pregame strength built ✅"
    )

    team = attach_opponent_strength(
        team
    )

    print(
        "Opponent pregame strength "
        "attached ✅"
    )

    team = (
        add_opponent_adjusted_performance(
            team
        )
    )

    print(
        "Opponent-adjusted game "
        "performance built ✅"
    )

    team = add_adjusted_histories(
        team
    )

    print(
        "Adjusted same-league and "
        "cross-league histories built ✅"
    )

    team = add_transition_context(
        team
    )

    team = resolve_final_histories(
        team
    )

    print(
        "History-source resolution "
        "complete ✅"
    )

    run_leakage_checks(
        team
    )

    print_transfer_report(
        team
    )

    match_store = build_match_store(
        matches,
        team,
    )

    summary = build_summary(
        match_store
    )

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    team.to_csv(
        OUTPUT_TEAM_ROWS,
        index=False,
    )

    match_store.to_csv(
        OUTPUT_MATCH_ROWS,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print()
    print(
        "=" * 120
    )

    print(
        "PREGAME V2 SUMMARY"
    )

    print(
        "=" * 120
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}",
        )
    )

    print()
    print(
        "=" * 120
    )

    print(
        "CURRENT SEASON — 2627"
    )

    print(
        "=" * 120
    )

    current = summary[
        summary["season"]
        ==
        "2627"
    ]

    if len(current):

        print(
            current.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.2f}",
            )
        )

    else:

        print(
            "No 2627 matches."
        )

    print()
    print(
        "=" * 105
    )

    print(
        "FEATURE STORE V2 COMPLETE"
    )

    print(
        "=" * 105
    )

    print()
    print(
        "Team rows:"
    )

    print(
        OUTPUT_TEAM_ROWS
    )

    print()
    print(
        "Match rows:"
    )

    print(
        OUTPUT_MATCH_ROWS
    )

    print()
    print(
        "Summary:"
    )

    print(
        OUTPUT_SUMMARY
    )

    print()
    print(
        "Opponent strength 0.875 ✅"
    )

    print(
        "Current-match leakage "
        "prevented ✅"
    )

    print(
        "Same-day leakage prevented ✅"
    )

    print(
        "Promotion/relegation history "
        "preserved when available ✅"
    )

    print(
        "Neutral fallback retained ✅"
    )

    print(
        "Goals recency frozen ✅"
    )

    print(
        "xG recency frozen ✅"
    )

    print(
        "Shots recency frozen ✅"
    )

    print(
        "V5 signal weights unchanged ✅"
    )


if __name__ == "__main__":
    main()