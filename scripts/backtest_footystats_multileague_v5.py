from pathlib import Path
import math

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
    / "footystats_multileague_pregame_v2.csv"
)

OUTPUT_PREDICTIONS = (
    PROCESSED
    / "footystats_multileague_v5_predictions.csv"
)

OUTPUT_SUMMARY = (
    PROCESSED
    / "footystats_multileague_v5_summary.csv"
)

OUTPUT_BY_SEASON = (
    PROCESSED
    / "footystats_multileague_v5_by_season.csv"
)

OUTPUT_BY_HISTORY_SOURCE = (
    PROCESSED
    / "footystats_multileague_v5_by_history_source.csv"
)


# ============================================================
# FROZEN V5 SETTINGS
# ============================================================

GOAL_WEIGHT = 0.09
XG_WEIGHT = 0.75
SHOT_WEIGHT = 0.16

OVERALL_WEIGHT = 0.80
VENUE_WEIGHT = 0.20

# Already baked into the V2 feature store.
OPPONENT_STRENGTH = 0.875

MAX_GOALS = 10
EPS = 1e-12


# ============================================================
# HOLDOUT LABELS
# ============================================================

SEASON_LABELS = {
    # Split-year European seasons
    "1819": "HISTORY",
    "1920": "HISTORY",
    "2021": "HISTORY",
    "2122": "DEVELOPMENT",
    "2223": "DEVELOPMENT",
    "2324": "VALIDATION",
    "2425": "FINAL_HOLDOUT",
    "2526": "POST_HOLDOUT_CHECK",
    "2627": "CURRENT",

    # Calendar-year leagues
    "2022": "DEVELOPMENT",
    "2023": "VALIDATION",
    "2024": "FINAL_HOLDOUT",
    "2025": "POST_HOLDOUT_CHECK",
}


ROLE_SEASONS = {
    "VALIDATION": [
        "2324",
        "2023",
    ],
    "FINAL_HOLDOUT": [
        "2425",
        "2024",
    ],
    "POST_HOLDOUT_CHECK": [
        "2526",
        "2025",
    ],
}


IMPORTANT_SEASONS = [
    season
    for seasons in ROLE_SEASONS.values()
    for season in seasons
]


# ============================================================
# POISSON
# ============================================================

FACTORIALS = np.array(
    [
        math.factorial(k)
        for k in range(
            MAX_GOALS + 1
        )
    ],
    dtype=float,
)


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
        +
        draw
        +
        away_win
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
            home_goals == away_goals,
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
                -
                truth
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
        ==
        y_true
    ).mean()


def expected_calibration_error(
    y_true,
    probs,
    bins=10,
):

    predicted = probs.argmax(
        axis=1
    )

    confidence = probs.max(
        axis=1
    )

    correct = (
        predicted
        ==
        y_true
    ).astype(float)

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0

    for i in range(
        bins
    ):

        low = edges[i]
        high = edges[i + 1]

        if i == bins - 1:

            mask = (
                (confidence >= low)
                &
                (confidence <= high)
            )

        else:

            mask = (
                (confidence >= low)
                &
                (confidence < high)
            )

        n = int(
            mask.sum()
        )

        if n == 0:

            continue

        bin_accuracy = correct[
            mask
        ].mean()

        bin_confidence = confidence[
            mask
        ].mean()

        ece += (
            n
            /
            len(y_true)
        ) * abs(
            bin_accuracy
            -
            bin_confidence
        )

    return ece


# ============================================================
# LOAD
# ============================================================

def load_data():

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
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "lg_home_goals",
        "lg_away_goals",

        "home_history_source",
        "away_history_source",

        "home_final_goal_attack_overall",
        "home_final_goal_defense_overall",
        "home_final_goal_attack_venue",
        "home_final_goal_defense_venue",

        "away_final_goal_attack_overall",
        "away_final_goal_defense_overall",
        "away_final_goal_attack_venue",
        "away_final_goal_defense_venue",

        "home_final_xg_attack_overall",
        "home_final_xg_defense_overall",
        "home_final_xg_attack_venue",
        "home_final_xg_defense_venue",

        "away_final_xg_attack_overall",
        "away_final_xg_defense_overall",
        "away_final_xg_attack_venue",
        "away_final_xg_defense_venue",

        "home_final_shot_attack_overall",
        "home_final_shot_defense_overall",
        "home_final_shot_attack_venue",
        "home_final_shot_defense_venue",

        "away_final_shot_attack_overall",
        "away_final_shot_defense_overall",
        "away_final_shot_attack_venue",
        "away_final_shot_defense_venue",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required V2 columns:\n"
            +
            "\n".join(
                f" - {col}"
                for col in missing
            )
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
        col
        for col in required
        if (
            "final_"
            in col
            or
            col
            in
            [
                "home_goals",
                "away_goals",
                "lg_home_goals",
                "lg_away_goals",
            ]
        )
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

    return df


# ============================================================
# COMPONENT BLEND
# ============================================================

def blend_component(
    df,
    overall_col,
    venue_col,
):

    return (
        OVERALL_WEIGHT
        *
        df[
            overall_col
        ]
        +
        VENUE_WEIGHT
        *
        df[
            venue_col
        ]
    )


# ============================================================
# BUILD FROZEN V5 LAMBDAS
# ============================================================

def build_lambdas(
    df,
):

    # ========================================================
    # GOALS
    # ========================================================

    home_goal_attack = blend_component(
        df,
        "home_final_goal_attack_overall",
        "home_final_goal_attack_venue",
    )

    home_goal_defense = blend_component(
        df,
        "home_final_goal_defense_overall",
        "home_final_goal_defense_venue",
    )

    away_goal_attack = blend_component(
        df,
        "away_final_goal_attack_overall",
        "away_final_goal_attack_venue",
    )

    away_goal_defense = blend_component(
        df,
        "away_final_goal_defense_overall",
        "away_final_goal_defense_venue",
    )

    # ========================================================
    # XG
    # ========================================================

    home_xg_attack = blend_component(
        df,
        "home_final_xg_attack_overall",
        "home_final_xg_attack_venue",
    )

    home_xg_defense = blend_component(
        df,
        "home_final_xg_defense_overall",
        "home_final_xg_defense_venue",
    )

    away_xg_attack = blend_component(
        df,
        "away_final_xg_attack_overall",
        "away_final_xg_attack_venue",
    )

    away_xg_defense = blend_component(
        df,
        "away_final_xg_defense_overall",
        "away_final_xg_defense_venue",
    )

    # ========================================================
    # SHOTS
    # ========================================================

    home_shot_attack = blend_component(
        df,
        "home_final_shot_attack_overall",
        "home_final_shot_attack_venue",
    )

    home_shot_defense = blend_component(
        df,
        "home_final_shot_defense_overall",
        "home_final_shot_defense_venue",
    )

    away_shot_attack = blend_component(
        df,
        "away_final_shot_attack_overall",
        "away_final_shot_attack_venue",
    )

    away_shot_defense = blend_component(
        df,
        "away_final_shot_defense_overall",
        "away_final_shot_defense_venue",
    )

    # ========================================================
    # FROZEN V5 SIGNAL BLEND
    # ========================================================

    home_attack = (
        GOAL_WEIGHT
        *
        home_goal_attack
        +
        XG_WEIGHT
        *
        home_xg_attack
        +
        SHOT_WEIGHT
        *
        home_shot_attack
    )

    home_defense = (
        GOAL_WEIGHT
        *
        home_goal_defense
        +
        XG_WEIGHT
        *
        home_xg_defense
        +
        SHOT_WEIGHT
        *
        home_shot_defense
    )

    away_attack = (
        GOAL_WEIGHT
        *
        away_goal_attack
        +
        XG_WEIGHT
        *
        away_xg_attack
        +
        SHOT_WEIGHT
        *
        away_shot_attack
    )

    away_defense = (
        GOAL_WEIGHT
        *
        away_goal_defense
        +
        XG_WEIGHT
        *
        away_xg_defense
        +
        SHOT_WEIGHT
        *
        away_shot_defense
    )

    # ========================================================
    # EXPECTED GOALS
    # ========================================================

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
# HISTORY CLASS
# ============================================================

def add_history_class(
    df,
):

    out = df.copy()

    home = out[
        "home_history_source"
    ].astype(str)

    away = out[
        "away_history_source"
    ].astype(str)

    out[
        "history_class"
    ] = np.select(
        [
            (
                (home == "SAME_LEAGUE")
                &
                (away == "SAME_LEAGUE")
            ),

            (
                (home == "NEUTRAL")
                |
                (away == "NEUTRAL")
            ),

            (
                (home == "TRANSFERRED")
                |
                (away == "TRANSFERRED")
            ),
        ],
        [
            "BOTH_SAME_LEAGUE",
            "HAS_NEUTRAL",
            "HAS_TRANSFERRED",
        ],
        default="OTHER",
    )

    return out


# ============================================================
# SCORE
# ============================================================

def build_predictions(
    df,
):

    out = df.copy()

    home_lambda, away_lambda = (
        build_lambdas(
            out
        )
    )

    out[
        "home_lambda"
    ] = home_lambda

    out[
        "away_lambda"
    ] = away_lambda

    probs = calculate_1x2_probs(
        home_lambda.to_numpy(
            dtype=float
        ),
        away_lambda.to_numpy(
            dtype=float
        ),
    )

    out[
        "p_home"
    ] = probs[
        :,
        0
    ]

    out[
        "p_draw"
    ] = probs[
        :,
        1
    ]

    out[
        "p_away"
    ] = probs[
        :,
        2
    ]

    out[
        "predicted_result"
    ] = np.array(
        [
            "HOME",
            "DRAW",
            "AWAY",
        ]
    )[
        probs.argmax(
            axis=1
        )
    ]

    out[
        "actual_result"
    ] = np.where(
        out[
            "home_goals"
        ]
        >
        out[
            "away_goals"
        ],
        "HOME",
        np.where(
            out[
                "home_goals"
            ]
            ==
            out[
                "away_goals"
            ],
            "DRAW",
            "AWAY",
        ),
    )

    out[
        "season_role"
    ] = out[
        "season"
    ].map(
        SEASON_LABELS
    ).fillna(
        "UNKNOWN"
    )

    out = add_history_class(
        out
    )

    return out


# ============================================================
# VALID ROW MASK
# ============================================================

def valid_prediction_mask(
    df,
):

    required = [
        "home_goals",
        "away_goals",
        "home_lambda",
        "away_lambda",
        "p_home",
        "p_draw",
        "p_away",
    ]

    numeric = (
        df[
            required
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    finite = np.isfinite(
        numeric
    ).all(
        axis=1
    )

    probs = df[
        [
            "p_home",
            "p_draw",
            "p_away",
        ]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    nonnegative = (
        probs
        >=
        0.0
    ).all(
        axis=1
    ).to_numpy()

    sums = probs.sum(
        axis=1
    ).to_numpy(
        dtype=float
    )

    valid_sum = np.isclose(
        sums,
        1.0,
        atol=1e-6,
    )

    return (
        finite
        &
        nonnegative
        &
        valid_sum
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    df,
):

    common_games = len(
        df
    )

    valid_mask = valid_prediction_mask(
        df
    )

    valid = df.loc[
        valid_mask
    ].copy()

    if len(valid) == 0:

        return {
            "common_games":
                common_games,

            "games":
                0,

            "dropped_invalid":
                common_games,

            "accuracy":
                np.nan,

            "log_loss":
                np.nan,

            "brier":
                np.nan,

            "ece":
                np.nan,

            "avg_confidence":
                np.nan,

            "avg_home_lambda":
                np.nan,

            "avg_away_lambda":
                np.nan,
        }

    y = result_classes(
        valid[
            "home_goals"
        ].to_numpy(),
        valid[
            "away_goals"
        ].to_numpy(),
    )

    probs = valid[
        [
            "p_home",
            "p_draw",
            "p_away",
        ]
    ].to_numpy(
        dtype=float
    )

    return {
        "common_games":
            common_games,

        "games":
            len(
                valid
            ),

        "dropped_invalid":
            (
                common_games
                -
                len(valid)
            ),

        "accuracy":
            100.0
            *
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

        "ece":
            100.0
            *
            expected_calibration_error(
                y,
                probs,
            ),

        "avg_confidence":
            100.0
            *
            probs.max(
                axis=1
            ).mean(),

        "avg_home_lambda":
            valid[
                "home_lambda"
            ].mean(),

        "avg_away_lambda":
            valid[
                "away_lambda"
            ].mean(),
    }


# ============================================================
# SUMMARY HELPERS
# ============================================================

def build_group_summary(
    df,
    group_cols,
):

    rows = []

    if isinstance(
        group_cols,
        str,
    ):

        group_cols = [
            group_cols
        ]

    for keys, sub in df.groupby(
        group_cols,
        dropna=False,
    ):

        if not isinstance(
            keys,
            tuple,
        ):

            keys = (
                keys,
            )

        row = {
            col:
                key
            for col, key in zip(
                group_cols,
                keys,
            )
        }

        row.update(
            evaluate(
                sub
            )
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT
# ============================================================

def print_metrics(
    title,
    df,
):

    metrics = evaluate(
        df
    )

    print()
    print(
        "=" * 85
    )

    print(
        title
    )

    print(
        "=" * 85
    )

    print(
        "Rows:          ",
        f"{metrics['common_games']:,}",
    )

    print(
        "Valid games:   ",
        f"{metrics['games']:,}",
    )

    print(
        "Dropped:       ",
        f"{metrics['dropped_invalid']:,}",
    )

    print(
        "Accuracy:      ",
        f"{metrics['accuracy']:.2f}%",
    )

    print(
        "Log Loss:      ",
        f"{metrics['log_loss']:.5f}",
    )

    print(
        "Brier:         ",
        f"{metrics['brier']:.5f}",
    )

    print(
        "ECE:           ",
        f"{metrics['ece']:.2f}%",
    )

    print(
        "Avg Confidence:",
        f"{metrics['avg_confidence']:.2f}%",
    )

    print(
        "Avg Home λ:    ",
        f"{metrics['avg_home_lambda']:.4f}",
    )

    print(
        "Avg Away λ:    ",
        f"{metrics['avg_away_lambda']:.4f}",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 100
    )

    print(
        "FOOTYSTATS MULTI-LEAGUE "
        "FROZEN V5 TRANSFER BACKTEST"
    )

    print(
        "=" * 100
    )

    print()
    print(
        "Frozen V5 settings:"
    )

    print(
        "Goals: 9%"
    )

    print(
        "xG:    75%"
    )

    print(
        "Shots: 16%"
    )

    print(
        "Overall / venue: 80% / 20%"
    )

    print(
        "Opponent strength: 0.875 "
        "(already baked into V2 features)"
    )

    print()
    print(
        "NO PARAMETERS WILL BE FIT "
        "OR SELECTED."
    )

    # ========================================================
    # LOAD
    # ========================================================

    df = load_data()

    print()
    print(
        "Loaded matches:",
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

    # ========================================================
    # SCORE
    # ========================================================

    predictions = build_predictions(
        df
    )

    predictions.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # ========================================================
    # ALL
    # ========================================================

    print_metrics(
        "ALL V5 LEAGUES",
        predictions,
    )

    # ========================================================
    # HOLDOUT PERIODS
    # ========================================================

    for role, label in [
        (
            "VALIDATION",
            "VALIDATION — 2023/24 + 2023 CALENDAR",
        ),
        (
            "FINAL_HOLDOUT",
            "FINAL HOLDOUT — 2024/25 + 2024 CALENDAR",
        ),
        (
            "POST_HOLDOUT_CHECK",
            "POST-HOLDOUT CHECK — 2025/26 + 2025 CALENDAR",
        ),
    ]:

        sub = predictions[
            predictions[
                "season"
            ].isin(
                ROLE_SEASONS[
                    role
                ]
            )
        ]

        print_metrics(
            label,
            sub,
        )

    # ========================================================
    # BY LEAGUE
    # ========================================================

    by_league = build_group_summary(
        predictions,
        "league",
    )

    print()
    print(
        "=" * 120
    )

    print(
        "BY LEAGUE — ALL AVAILABLE HISTORY"
    )

    print(
        "=" * 120
    )

    print(
        by_league[
            [
                "league",
                "games",
                "accuracy",
                "log_loss",
                "brier",
                "ece",
                "avg_confidence",
            ]
        ]
        .sort_values(
            "log_loss"
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # BY LEAGUE + SEASON
    # ========================================================

    by_season = build_group_summary(
        predictions,
        [
            "league",
            "season",
        ],
    )

    by_season[
        "season_role"
    ] = by_season[
        "season"
    ].map(
        SEASON_LABELS
    )

    by_season.to_csv(
        OUTPUT_BY_SEASON,
        index=False,
    )

    print()
    print(
        "=" * 140
    )

    print(
        "VALIDATION / HOLDOUT / "
        "POST-HOLDOUT BY LEAGUE"
    )

    print(
        "=" * 140
    )

    important = by_season[
        by_season[
            "season"
        ].isin(
            IMPORTANT_SEASONS
        )
    ].copy()

    print(
        important[
            [
                "league",
                "season",
                "season_role",
                "games",
                "accuracy",
                "log_loss",
                "brier",
                "ece",
                "avg_confidence",
            ]
        ]
        .sort_values(
            [
                "season",
                "log_loss",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # HISTORY SOURCE
    # ========================================================

    by_history = build_group_summary(
        predictions,
        "history_class",
    )

    by_history.to_csv(
        OUTPUT_BY_HISTORY_SOURCE,
        index=False,
    )

    print()
    print(
        "=" * 110
    )

    print(
        "BY HISTORY SOURCE"
    )

    print(
        "=" * 110
    )

    print(
        by_history[
            [
                "history_class",
                "games",
                "accuracy",
                "log_loss",
                "brier",
                "ece",
                "avg_confidence",
            ]
        ]
        .sort_values(
            "log_loss"
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # TRANSFERRED ONLY BY LEAGUE
    # ========================================================

    transferred = predictions[
        predictions[
            "history_class"
        ]
        ==
        "HAS_TRANSFERRED"
    ].copy()

    if len(
        transferred
    ):

        transferred_by_league = (
            build_group_summary(
                transferred,
                "league",
            )
        )

        print()
        print(
            "=" * 110
        )

        print(
            "TRANSFERRED-HISTORY MATCHES "
            "BY LEAGUE"
        )

        print(
            "=" * 110
        )

        print(
            transferred_by_league[
                [
                    "league",
                    "games",
                    "accuracy",
                    "log_loss",
                    "brier",
                    "ece",
                ]
            ]
            .sort_values(
                "log_loss"
            )
            .to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    # ========================================================
    # HOLDOUT LEAGUE STABILITY
    # ========================================================

    print()
    print(
        "=" * 145
    )

    print(
        "THREE-SEASON HOLDOUT STABILITY"
    )

    print(
        "=" * 145
    )

    holdout = important.pivot_table(
        index="league",
        columns="season_role",
        values=[
            "accuracy",
            "log_loss",
            "brier",
            "ece",
        ],
    )

    print(
        holdout.to_string(
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # SIMPLE DEPLOYMENT AUDIT
    #
    # IMPORTANT:
    # THIS IS DESCRIPTIVE ONLY.
    # IT DOES NOT CHANGE V5 OR SELECT PARAMETERS.
    # ========================================================

    print()
    print(
        "=" * 110
    )

    print(
        "DESCRIPTIVE DEPLOYMENT AUDIT"
    )

    print(
        "=" * 110
    )

    print()
    print(
        "This section does NOT tune V5."
    )

    print(
        "It simply exposes unstable leagues "
        "before live paper deployment."
    )

    deployment_rows = []

    for league in sorted(
        predictions[
            "league"
        ].unique()
    ):

        league_rows = important[
            important[
                "league"
            ]
            ==
            league
        ].copy()

        if len(
            league_rows
        ) == 0:

            continue

        deployment_rows.append(
            {
                "league":
                    league,

                "holdout_seasons":
                    len(
                        league_rows
                    ),

                "mean_log_loss":
                    league_rows[
                        "log_loss"
                    ].mean(),

                "worst_log_loss":
                    league_rows[
                        "log_loss"
                    ].max(),

                "mean_brier":
                    league_rows[
                        "brier"
                    ].mean(),

                "mean_ece":
                    league_rows[
                        "ece"
                    ].mean(),

                "mean_accuracy":
                    league_rows[
                        "accuracy"
                    ].mean(),

                "log_loss_std":
                    league_rows[
                        "log_loss"
                    ].std(
                        ddof=0
                    ),
            }
        )

    deployment = pd.DataFrame(
        deployment_rows
    )

    print(
        deployment
        .sort_values(
            "mean_log_loss"
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # SAVE MAIN SUMMARY
    # ========================================================

    overall_rows = []

    overall_rows.append(
        {
            "scope":
                "ALL",

            "league":
                "ALL",

            "season":
                "ALL",

            **evaluate(
                predictions
            ),
        }
    )

    for role in [
        "VALIDATION",
        "FINAL_HOLDOUT",
        "POST_HOLDOUT_CHECK",
    ]:

        seasons = ROLE_SEASONS[
            role
        ]

        sub = predictions[
            predictions[
                "season"
            ].isin(
                seasons
            )
        ]

        overall_rows.append(
            {
                "scope":
                    role,

                "league":
                    "ALL",

                "season":
                    "+".join(
                        seasons
                    ),

                **evaluate(
                    sub
                ),
            }
        )

    for row in (
        by_league
        .to_dict(
            orient="records"
        )
    ):

        overall_rows.append(
            {
                "scope":
                    "LEAGUE",

                "season":
                    "ALL",

                **row,
            }
        )

    summary = pd.DataFrame(
        overall_rows
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        "FROZEN V5 TRANSFER BACKTEST COMPLETE"
    )

    print(
        "=" * 100
    )

    print()
    print(
        "Goals 9% unchanged ✅"
    )

    print(
        "xG 75% unchanged ✅"
    )

    print(
        "Shots 16% unchanged ✅"
    )

    print(
        "Overall / venue 80/20 unchanged ✅"
    )

    print(
        "Opponent strength 0.875 unchanged ✅"
    )

    print(
        "No holdout parameter fitting ✅"
    )

    print(
        "No odds used ✅"
    )

    print(
        "No betting threshold optimized ✅"
    )

    print()
    print(
        "Predictions:"
    )

    print(
        OUTPUT_PREDICTIONS
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
        "By season:"
    )

    print(
        OUTPUT_BY_SEASON
    )

    print()
    print(
        "By history source:"
    )

    print(
        OUTPUT_BY_HISTORY_SOURCE
    )


if __name__ == "__main__":
    main()