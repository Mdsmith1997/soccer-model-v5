from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

V3_FILE = (
    ROOT
    / "data"
    / "processed"
    / "opponent_strength_v3_predictions.csv"
)

XG_MATCH_FILE = (
    ROOT
    / "data"
    / "processed"
    / "understat_xg_matched.csv"
)

TEAM_FILE = (
    ROOT
    / "data"
    / "processed"
    / "team_game_stats.csv"
)

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "signal_recency_v5_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "signal_recency_v5_predictions.csv"
)


# ============================================================
# FROZEN V5 SETTINGS
# ============================================================

GOAL_WEIGHT = 0.09
XG_WEIGHT = 0.75
SHOT_WEIGHT = 0.16
SOT_WEIGHT = 0.00

OVERALL_WEIGHT = 0.75
VENUE_WEIGHT = 0.25


# ============================================================
# RECENCY GRID
# ============================================================

GOAL_RECENCIES = [
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
]

XG_RECENCIES = [
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
]

SHOT_RECENCIES = [
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
]


# ============================================================
# SPLITS
# ============================================================

TUNING_SEASONS = {
    "2122",
    "2223",
}

VALIDATION_SEASONS = {
    "2324",
}

FINAL_SEASONS = {
    "2425",
}

SUPPORTED_LEAGUES = {
    "Premier League",
    "Bundesliga",
}


# ============================================================
# POISSON SETTINGS
# ============================================================

MAX_GOALS = 10
EPS = 1e-12

FACTORIALS = np.array(
    [
        math.factorial(k)
        for k in range(
            MAX_GOALS + 1
        )
    ],
    dtype=float,
)


# ============================================================
# EW PRIOR
# ============================================================

def weighted_prior_average(
    series,
    recency,
):

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


# ============================================================
# POISSON
# ============================================================

def poisson_probabilities(
    lambdas,
):

    lambdas = np.asarray(
        lambdas,
        dtype=float,
    )

    goals = np.arange(
        MAX_GOALS + 1
    )

    probs = (
        np.exp(
            -lambdas[:, None]
        )
        *
        (
            lambdas[:, None]
            ** goals[None, :]
        )
        /
        FACTORIALS[
            None,
            :
        ]
    )

    probs /= probs.sum(
        axis=1,
        keepdims=True,
    )

    return probs


def calculate_1x2_probs(
    home_lambda,
    away_lambda,
):

    hp = poisson_probabilities(
        home_lambda
    )

    ap = poisson_probabilities(
        away_lambda
    )

    away_cdf = np.cumsum(
        ap,
        axis=1,
    )

    home_win = np.zeros(
        len(home_lambda)
    )

    for h in range(
        1,
        MAX_GOALS + 1,
    ):

        home_win += (
            hp[:, h]
            *
            away_cdf[
                :,
                h - 1
            ]
        )

    home_cdf = np.cumsum(
        hp,
        axis=1,
    )

    away_win = np.zeros(
        len(home_lambda)
    )

    for a in range(
        1,
        MAX_GOALS + 1,
    ):

        away_win += (
            ap[:, a]
            *
            home_cdf[
                :,
                a - 1
            ]
        )

    draw = (
        hp
        * ap
    ).sum(
        axis=1
    )

    total = (
        home_win
        + draw
        + away_win
    )

    return np.column_stack(
        [
            home_win / total,
            draw / total,
            away_win / total,
        ]
    )


# ============================================================
# METRICS
# ============================================================

def result_classes(
    home_goals,
    away_goals,
):

    return np.where(
        home_goals > away_goals,
        0,
        np.where(
            home_goals
            == away_goals,
            1,
            2,
        ),
    )


def log_loss(
    y_true,
    probs,
):

    chosen = probs[
        np.arange(
            len(y_true)
        ),
        y_true,
    ]

    chosen = np.clip(
        chosen,
        EPS,
        1.0,
    )

    return (
        -np.log(
            chosen
        )
    ).mean()


def brier(
    y_true,
    probs,
):

    truth = np.zeros_like(
        probs
    )

    truth[
        np.arange(
            len(y_true)
        ),
        y_true,
    ] = 1.0

    return np.mean(
        np.sum(
            (
                probs
                - truth
            ) ** 2,
            axis=1,
        )
    )


def accuracy(
    y_true,
    probs,
):

    return (
        probs.argmax(
            axis=1
        )
        == y_true
    ).mean()


# ============================================================
# LOAD V3 MATCHES
# ============================================================

def load_v3():

    df = pd.read_csv(
        V3_FILE,
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

    df = df[
        df[
            "league"
        ].isin(
            SUPPORTED_LEAGUES
        )
    ].copy()

    return df


# ============================================================
# LOAD TEAM GAME DATA
# ============================================================

def load_team_games():

    df = pd.read_csv(
        TEAM_FILE,
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

    df = df[
        df[
            "league"
        ].isin(
            SUPPORTED_LEAGUES
        )
    ].copy()

    # --------------------------------------------------------
    # NORMALIZE SOT NAMES
    # --------------------------------------------------------

    if (
        "sot_for"
        not in df.columns
        and
        "shots_on_target"
        in df.columns
    ):

        df[
            "sot_for"
        ] = df[
            "shots_on_target"
        ]

    if (
        "sot_against"
        not in df.columns
        and
        "shots_on_target_against"
        in df.columns
    ):

        df[
            "sot_against"
        ] = df[
            "shots_on_target_against"
        ]

    return df


# ============================================================
# LOAD UNDERSTAT XG
# ============================================================

def load_xg_matches():

    df = pd.read_csv(
        XG_MATCH_FILE,
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

    return df


# ============================================================
# BUILD XG TEAM ROWS
# ============================================================

def build_xg_team_rows(
    xg,
):

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
        }
    )

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
        }
    )

    return (
        pd.concat(
            [
                home,
                away,
            ],
            ignore_index=True,
        )
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
# LEAGUE BASELINES
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
        "goals_for",
        "goals_against",
        "shots_for",
        "shots_against",
        "xg_for",
        "xg_against",
    ]

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
            f"lg_{metric}"
        ] = (
            prior_sum
            /
            prior_count.replace(
                0,
                np.nan,
            )
        )

    keep = [
        "league",
        "date",
    ] + [
        f"lg_{metric}"
        for metric in metrics
    ]

    return out.merge(
        daily[
            keep
        ],
        on=[
            "league",
            "date",
        ],
        how="left",
        validate="many_to_one",
    )


# ============================================================
# BUILD SIGNAL HISTORIES
# ============================================================

def add_histories(
    df,
    goal_recency,
    xg_recency,
    shot_recency,
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

    overall = out.groupby(
        "team",
        sort=False,
        group_keys=False,
    )

    venue = out.groupby(
        [
            "team",
            "venue",
        ],
        sort=False,
        group_keys=False,
    )

    recency_map = {
        "goals_for":
            goal_recency,

        "goals_against":
            goal_recency,

        "shots_for":
            shot_recency,

        "shots_against":
            shot_recency,

        "xg_for":
            xg_recency,

        "xg_against":
            xg_recency,
    }

    for metric, recency in (
        recency_map.items()
    ):

        out[
            f"ew_{metric}"
        ] = (
            overall[
                metric
            ]
            .transform(
                lambda s:
                    weighted_prior_average(
                        s,
                        recency,
                    )
            )
        )

        out[
            f"venue_ew_{metric}"
        ] = (
            venue[
                metric
            ]
            .transform(
                lambda s:
                    weighted_prior_average(
                        s,
                        recency,
                    )
            )
        )

    return out


# ============================================================
# BUILD NORMALIZED STRENGTHS
# ============================================================

def add_strengths(
    df,
):

    out = df.copy()

    # --------------------------------------------------------
    # GOALS
    # --------------------------------------------------------

    out[
        "goal_attack"
    ] = (
        OVERALL_WEIGHT
        * (
            out[
                "ew_goals_for"
            ]
            /
            out[
                "lg_goals_for"
            ]
        )
        +
        VENUE_WEIGHT
        * (
            out[
                "venue_ew_goals_for"
            ]
            /
            out[
                "lg_goals_for"
            ]
        )
    )

    out[
        "goal_defense"
    ] = (
        OVERALL_WEIGHT
        * (
            out[
                "ew_goals_against"
            ]
            /
            out[
                "lg_goals_against"
            ]
        )
        +
        VENUE_WEIGHT
        * (
            out[
                "venue_ew_goals_against"
            ]
            /
            out[
                "lg_goals_against"
            ]
        )
    )

    # --------------------------------------------------------
    # SHOTS
    # --------------------------------------------------------

    out[
        "shot_attack"
    ] = (
        OVERALL_WEIGHT
        * (
            out[
                "ew_shots_for"
            ]
            /
            out[
                "lg_shots_for"
            ]
        )
        +
        VENUE_WEIGHT
        * (
            out[
                "venue_ew_shots_for"
            ]
            /
            out[
                "lg_shots_for"
            ]
        )
    )

    out[
        "shot_defense"
    ] = (
        OVERALL_WEIGHT
        * (
            out[
                "ew_shots_against"
            ]
            /
            out[
                "lg_shots_against"
            ]
        )
        +
        VENUE_WEIGHT
        * (
            out[
                "venue_ew_shots_against"
            ]
            /
            out[
                "lg_shots_against"
            ]
        )
    )

    # --------------------------------------------------------
    # XG
    # --------------------------------------------------------

    out[
        "xg_attack"
    ] = (
        OVERALL_WEIGHT
        * (
            out[
                "ew_xg_for"
            ]
            /
            out[
                "lg_xg_for"
            ]
        )
        +
        VENUE_WEIGHT
        * (
            out[
                "venue_ew_xg_for"
            ]
            /
            out[
                "lg_xg_for"
            ]
        )
    )

    out[
        "xg_defense"
    ] = (
        OVERALL_WEIGHT
        * (
            out[
                "ew_xg_against"
            ]
            /
            out[
                "lg_xg_against"
            ]
        )
        +
        VENUE_WEIGHT
        * (
            out[
                "venue_ew_xg_against"
            ]
            /
            out[
                "lg_xg_against"
            ]
        )
    )

    return out


# ============================================================
# BUILD TEAM SIGNAL TABLE FOR ONE RECENCY SET
# ============================================================

def build_signal_table(
    team,
    xg_team,
    goal_recency,
    xg_recency,
    shot_recency,
):

    base = team[
        [
            "match_id",
            "date",
            "season",
            "league",
            "team",
            "venue",

            "goals_for",
            "goals_against",

            "shots_for",
            "shots_against",
        ]
    ].copy()

    base = base.merge(
        xg_team[
            [
                "match_id",
                "team",
                "xg_for",
                "xg_against",
            ]
        ],
        on=[
            "match_id",
            "team",
        ],
        how="inner",
        validate="one_to_one",
    )

    base = add_league_baselines(
        base
    )

    base = add_histories(
        base,
        goal_recency,
        xg_recency,
        shot_recency,
    )

    base = add_strengths(
        base
    )

    return base


# ============================================================
# MATCH TABLE
# ============================================================

def build_match_table(
    signals,
    v3,
):

    home = signals[
        signals[
            "venue"
        ]
        == "HOME"
    ][
        [
            "match_id",
            "goal_attack",
            "goal_defense",
            "shot_attack",
            "shot_defense",
            "xg_attack",
            "xg_defense",
        ]
    ].rename(
        columns={
            "goal_attack":
                "home_goal_attack",

            "goal_defense":
                "home_goal_defense",

            "shot_attack":
                "home_shot_attack",

            "shot_defense":
                "home_shot_defense",

            "xg_attack":
                "home_xg_attack",

            "xg_defense":
                "home_xg_defense",
        }
    )

    away = signals[
        signals[
            "venue"
        ]
        == "AWAY"
    ][
        [
            "match_id",
            "goal_attack",
            "goal_defense",
            "shot_attack",
            "shot_defense",
            "xg_attack",
            "xg_defense",
        ]
    ].rename(
        columns={
            "goal_attack":
                "away_goal_attack",

            "goal_defense":
                "away_goal_defense",

            "shot_attack":
                "away_shot_attack",

            "shot_defense":
                "away_shot_defense",

            "xg_attack":
                "away_xg_attack",

            "xg_defense":
                "away_xg_defense",
        }
    )

    match = home.merge(
        away,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    match = v3.merge(
        match,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    return match


# ============================================================
# LAMBDAS
# ============================================================

def build_lambdas(
    df,
):

    home_attack = (
        GOAL_WEIGHT
        * df[
            "home_goal_attack"
        ]
        +
        XG_WEIGHT
        * df[
            "home_xg_attack"
        ]
        +
        SHOT_WEIGHT
        * df[
            "home_shot_attack"
        ]
    )

    away_attack = (
        GOAL_WEIGHT
        * df[
            "away_goal_attack"
        ]
        +
        XG_WEIGHT
        * df[
            "away_xg_attack"
        ]
        +
        SHOT_WEIGHT
        * df[
            "away_shot_attack"
        ]
    )

    home_defense = (
        GOAL_WEIGHT
        * df[
            "home_goal_defense"
        ]
        +
        XG_WEIGHT
        * df[
            "home_xg_defense"
        ]
        +
        SHOT_WEIGHT
        * df[
            "home_shot_defense"
        ]
    )

    away_defense = (
        GOAL_WEIGHT
        * df[
            "away_goal_defense"
        ]
        +
        XG_WEIGHT
        * df[
            "away_xg_defense"
        ]
        +
        SHOT_WEIGHT
        * df[
            "away_shot_defense"
        ]
    )

    home_lambda = (
        df[
            "lg_home_goals"
        ]
        *
        home_attack
        *
        away_defense
    ).clip(
        lower=0.15,
        upper=4.50,
    )

    away_lambda = (
        df[
            "lg_away_goals"
        ]
        *
        away_attack
        *
        home_defense
    ).clip(
        lower=0.15,
        upper=4.50,
    )

    return (
        home_lambda,
        away_lambda,
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    df,
    seasons,
):

    season = (
        df[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    sub = df[
        season.isin(
            seasons
        )
    ].copy()

    (
        home_lambda,
        away_lambda,
    ) = build_lambdas(
        sub
    )

    valid = (
        home_lambda.notna()
        &
        away_lambda.notna()
    )

    sub = sub.loc[
        valid
    ].copy()

    home_lambda = (
        home_lambda.loc[
            valid
        ]
        .to_numpy()
    )

    away_lambda = (
        away_lambda.loc[
            valid
        ]
        .to_numpy()
    )

    probs = calculate_1x2_probs(
        home_lambda,
        away_lambda,
    )

    y = result_classes(
        sub[
            "home_goals"
        ].to_numpy(),
        sub[
            "away_goals"
        ].to_numpy(),
    )

    return {
        "games":
            len(sub),

        "accuracy":
            accuracy(
                y,
                probs,
            ),

        "log_loss":
            log_loss(
                y,
                probs,
            ),

        "brier":
            brier(
                y,
                probs,
            ),
    }


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_comparison(
    title,
    baseline,
    candidate,
):

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    print(
        f"Games: "
        f"{candidate['games']:,}"
    )

    print()

    print(
        f"{'Metric':<15}"
        f"{'0.95 Control':>14}"
        f"{'Winner':>14}"
        f"{'Change':>14}"
    )

    print("-" * 57)

    print(
        f"{'Accuracy':<15}"
        f"{baseline['accuracy']:>13.2%}"
        f"{candidate['accuracy']:>13.2%}"
        f"{candidate['accuracy'] - baseline['accuracy']:>+13.2%}"
    )

    print(
        f"{'Log Loss':<15}"
        f"{baseline['log_loss']:>14.5f}"
        f"{candidate['log_loss']:>14.5f}"
        f"{candidate['log_loss'] - baseline['log_loss']:>+14.5f}"
    )

    print(
        f"{'Brier':<15}"
        f"{baseline['brier']:>14.5f}"
        f"{candidate['brier']:>14.5f}"
        f"{candidate['brier'] - baseline['brier']:>+14.5f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("TUNING SIGNAL RECENCY V5")
    print("==============================")
    print()

    print(
        "Frozen signal weights:"
    )

    print(
        f"Goals: "
        f"{GOAL_WEIGHT:.0%}"
    )

    print(
        f"xG: "
        f"{XG_WEIGHT:.0%}"
    )

    print(
        f"Shots: "
        f"{SHOT_WEIGHT:.0%}"
    )

    print(
        f"SOT: "
        f"{SOT_WEIGHT:.0%}"
    )

    v3 = load_v3()

    team = load_team_games()

    xg = load_xg_matches()

    xg_team = build_xg_team_rows(
        xg
    )

    print()
    print(
        f"V3 PL/Bundesliga matches: "
        f"{len(v3):,}"
    )

    print(
        f"Team rows: "
        f"{len(team):,}"
    )

    print(
        f"xG matches: "
        f"{len(xg):,}"
    )

    total_combinations = (
        len(
            GOAL_RECENCIES
        )
        *
        len(
            XG_RECENCIES
        )
        *
        len(
            SHOT_RECENCIES
        )
    )

    print()
    print(
        f"Recency combinations: "
        f"{total_combinations:,}"
    )

    # ========================================================
    # CACHE SIGNAL TABLES
    #
    # This is still manageable at only 125 combinations.
    # ========================================================

    rows = []

    tested = 0

    best_match = None

    for goal_recency in (
        GOAL_RECENCIES
    ):

        for xg_recency in (
            XG_RECENCIES
        ):

            for shot_recency in (
                SHOT_RECENCIES
            ):

                signals = (
                    build_signal_table(
                        team,
                        xg_team,
                        goal_recency,
                        xg_recency,
                        shot_recency,
                    )
                )

                match = build_match_table(
                    signals,
                    v3,
                )

                metrics = evaluate(
                    match,
                    TUNING_SEASONS,
                )

                rows.append(
                    {
                        "goal_recency":
                            goal_recency,

                        "xg_recency":
                            xg_recency,

                        "shot_recency":
                            shot_recency,

                        "games":
                            metrics[
                                "games"
                            ],

                        "accuracy":
                            metrics[
                                "accuracy"
                            ],

                        "log_loss":
                            metrics[
                                "log_loss"
                            ],

                        "brier":
                            metrics[
                                "brier"
                            ],
                    }
                )

                tested += 1

                if (
                    tested % 25
                    == 0
                ):

                    print(
                        f"Tested "
                        f"{tested:,}/"
                        f"{total_combinations:,}"
                    )

    results = pd.DataFrame(
        rows
    )

    results = (
        results
        .sort_values(
            [
                "log_loss",
                "brier",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    results[
        "rank"
    ] = (
        np.arange(
            len(results)
        )
        + 1
    )

    results.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    # ========================================================
    # TOP 20
    # ========================================================

    print()
    print("==============================")
    print("TOP 20 RECENCY SETTINGS")
    print("==============================")

    display = (
        results
        .head(20)
        .copy()
    )

    display[
        "accuracy"
    ] *= 100.0

    print(
        display[
            [
                "rank",
                "goal_recency",
                "xg_recency",
                "shot_recency",

                "games",
                "log_loss",
                "brier",
                "accuracy",
            ]
        ]
        .round(6)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # WINNER
    # ========================================================

    best = results.iloc[
        0
    ]

    best_goal_recency = float(
        best[
            "goal_recency"
        ]
    )

    best_xg_recency = float(
        best[
            "xg_recency"
        ]
    )

    best_shot_recency = float(
        best[
            "shot_recency"
        ]
    )

    print()
    print("==============================")
    print("WINNING RECENCY SETTINGS")
    print("==============================")

    print(
        f"Goals: "
        f"{best_goal_recency:.3f}"
    )

    print(
        f"xG: "
        f"{best_xg_recency:.3f}"
    )

    print(
        f"Shots: "
        f"{best_shot_recency:.3f}"
    )

    print(
        f"Tuning LL: "
        f"{best['log_loss']:.5f}"
    )

    # ========================================================
    # BUILD CONTROL
    # ========================================================

    control_signals = (
        build_signal_table(
            team,
            xg_team,
            0.95,
            0.95,
            0.95,
        )
    )

    control_match = (
        build_match_table(
            control_signals,
            v3,
        )
    )

    # ========================================================
    # BUILD WINNER
    # ========================================================

    winner_signals = (
        build_signal_table(
            team,
            xg_team,
            best_goal_recency,
            best_xg_recency,
            best_shot_recency,
        )
    )

    winner_match = (
        build_match_table(
            winner_signals,
            v3,
        )
    )

    # ========================================================
    # TUNING
    # ========================================================

    baseline_tune = evaluate(
        control_match,
        TUNING_SEASONS,
    )

    winner_tune = evaluate(
        winner_match,
        TUNING_SEASONS,
    )

    print_comparison(
        "TUNING — 2021/22 TO 2022/23",
        baseline_tune,
        winner_tune,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    baseline_validation = evaluate(
        control_match,
        VALIDATION_SEASONS,
    )

    winner_validation = evaluate(
        winner_match,
        VALIDATION_SEASONS,
    )

    print_comparison(
        "VALIDATION — 2023/24",
        baseline_validation,
        winner_validation,
    )

    # ========================================================
    # FINAL
    # ========================================================

    baseline_final = evaluate(
        control_match,
        FINAL_SEASONS,
    )

    winner_final = evaluate(
        winner_match,
        FINAL_SEASONS,
    )

    print_comparison(
        "FINAL CHECK — 2024/25",
        baseline_final,
        winner_final,
    )

    # ========================================================
    # SAVE WINNING PREDICTIONS
    # ========================================================

    (
        home_lambda,
        away_lambda,
    ) = build_lambdas(
        winner_match
    )

    valid = (
        home_lambda.notna()
        &
        away_lambda.notna()
    )

    output = winner_match.loc[
        valid
    ].copy()

    output[
        "home_lambda_v5"
    ] = (
        home_lambda.loc[
            valid
        ]
        .to_numpy()
    )

    output[
        "away_lambda_v5"
    ] = (
        away_lambda.loc[
            valid
        ]
        .to_numpy()
    )

    probs = calculate_1x2_probs(
        output[
            "home_lambda_v5"
        ].to_numpy(),

        output[
            "away_lambda_v5"
        ].to_numpy(),
    )

    output[
        "p_home_v5"
    ] = probs[
        :,
        0
    ]

    output[
        "p_draw_v5"
    ] = probs[
        :,
        1
    ]

    output[
        "p_away_v5"
    ] = probs[
        :,
        2
    ]

    output[
        "goal_recency_v5"
    ] = best_goal_recency

    output[
        "xg_recency_v5"
    ] = best_xg_recency

    output[
        "shot_recency_v5"
    ] = best_shot_recency

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print()
    print("==============================")
    print("SIGNAL RECENCY TUNING COMPLETE")
    print("==============================")

    print(
        "Recencies selected using "
        "2021/22–2022/23 only ✅"
    )

    print(
        "2023/24 and 2024/25 held out "
        "from parameter selection ✅"
    )

    print()

    print(
        "Tuning results:"
    )

    print(
        OUTPUT_RESULTS
    )

    print()

    print(
        "Predictions:"
    )

    print(
        OUTPUT_PREDICTIONS
    )


if __name__ == "__main__":
    main()