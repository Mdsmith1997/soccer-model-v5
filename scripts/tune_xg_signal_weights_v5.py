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

XG_FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "xg_features_v5.csv"
)

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "xg_signal_weights_v5_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "xg_signal_weights_v5_predictions.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SUPPORTED_LEAGUES = {
    "Premier League",
    "Bundesliga",
}

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

MAX_GOALS = 10
EPS = 1e-12

# Frozen structural settings
OVERALL_WEIGHT = 0.75
VENUE_WEIGHT = 0.25

# Current V2/V3 signal blend
BASE_GOAL_WEIGHT = 0.70
BASE_SHOT_WEIGHT = 0.15
BASE_SOT_WEIGHT = 0.15


# ============================================================
# WEIGHT GRID
# ============================================================

# Coarse first pass.
# All weights must sum to 1.
GOAL_WEIGHTS = np.round(
    np.arange(
        0.05,
        0.151,
        0.01,
    ),
    3,
)

XG_WEIGHTS = np.round(
    np.arange(
        0.68,
        0.821,
        0.01,
    ),
    3,
)

SOT_WEIGHTS = [
    0.00,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
]


# ============================================================
# POISSON HELPERS
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
# LOAD V3
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
# LOAD XG FEATURES
# ============================================================

def load_xg_features():

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

    return df


# ============================================================
# BUILD MATCH-LEVEL XG STRENGTHS
# ============================================================

def build_xg_match_table(
    xg,
):

    # --------------------------------------------------------
    # BLEND OVERALL + VENUE
    # --------------------------------------------------------

    xg[
        "xg_attack_blend"
    ] = (
        OVERALL_WEIGHT
        * xg[
            "xg_attack_strength"
        ]
        +
        VENUE_WEIGHT
        * xg[
            "venue_xg_attack_strength"
        ]
    )

    xg[
        "xg_defense_blend"
    ] = (
        OVERALL_WEIGHT
        * xg[
            "xg_defense_strength"
        ]
        +
        VENUE_WEIGHT
        * xg[
            "venue_xg_defense_strength"
        ]
    )

    xg[
        "npxg_attack_blend"
    ] = (
        OVERALL_WEIGHT
        * xg[
            "npxg_attack_strength"
        ]
        +
        VENUE_WEIGHT
        * xg[
            "venue_npxg_attack_strength"
        ]
    )

    xg[
        "npxg_defense_blend"
    ] = (
        OVERALL_WEIGHT
        * xg[
            "npxg_defense_strength"
        ]
        +
        VENUE_WEIGHT
        * xg[
            "venue_npxg_defense_strength"
        ]
    )

    # --------------------------------------------------------
    # HOME
    # --------------------------------------------------------

    home = xg[
        xg[
            "venue"
        ]
        == "HOME"
    ].copy()

    home = home[
        [
            "match_id",
            "team",

            "xg_attack_blend",
            "xg_defense_blend",

            "npxg_attack_blend",
            "npxg_defense_blend",
        ]
    ].rename(
        columns={
            "team":
                "home_team_xg",

            "xg_attack_blend":
                "home_xg_attack",

            "xg_defense_blend":
                "home_xg_defense",

            "npxg_attack_blend":
                "home_npxg_attack",

            "npxg_defense_blend":
                "home_npxg_defense",
        }
    )

    # --------------------------------------------------------
    # AWAY
    # --------------------------------------------------------

    away = xg[
        xg[
            "venue"
        ]
        == "AWAY"
    ].copy()

    away = away[
        [
            "match_id",
            "team",

            "xg_attack_blend",
            "xg_defense_blend",

            "npxg_attack_blend",
            "npxg_defense_blend",
        ]
    ].rename(
        columns={
            "team":
                "away_team_xg",

            "xg_attack_blend":
                "away_xg_attack",

            "xg_defense_blend":
                "away_xg_defense",

            "npxg_attack_blend":
                "away_npxg_attack",

            "npxg_defense_blend":
                "away_npxg_defense",
        }
    )

    match = home.merge(
        away,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    return match


# ============================================================
# ATTACH XG TO V3 MATCHES
# ============================================================

def merge_features(
    v3,
    xg_match,
):

    out = v3.merge(
        xg_match,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    return out


# ============================================================
# DETECT EXISTING V3 SIGNAL COLUMNS
# ============================================================

def detect_existing_signals(
    df,
):

    required = [
        "home_adj_goal_attack",
        "home_adj_goal_defense",

        "home_adj_shot_attack",
        "home_adj_shot_defense",

        "home_adj_sot_attack",
        "home_adj_sot_defense",

        "home_adj_venue_goal_attack",
        "home_adj_venue_goal_defense",

        "home_adj_venue_shot_attack",
        "home_adj_venue_shot_defense",

        "home_adj_venue_sot_attack",
        "home_adj_venue_sot_defense",

        "away_adj_goal_attack",
        "away_adj_goal_defense",

        "away_adj_shot_attack",
        "away_adj_shot_defense",

        "away_adj_sot_attack",
        "away_adj_sot_defense",

        "away_adj_venue_goal_attack",
        "away_adj_venue_goal_defense",

        "away_adj_venue_shot_attack",
        "away_adj_venue_shot_defense",

        "away_adj_venue_sot_attack",
        "away_adj_venue_sot_defense",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing V3 component columns:\n"
            f"{missing}"
        )

    # ========================================================
    # GOALS
    # ========================================================

    df[
        "home_goal_attack"
    ] = (
        OVERALL_WEIGHT
        * df[
            "home_adj_goal_attack"
        ]
        +
        VENUE_WEIGHT
        * df[
            "home_adj_venue_goal_attack"
        ]
    )

    df[
        "home_goal_defense"
    ] = (
        OVERALL_WEIGHT
        * df[
            "home_adj_goal_defense"
        ]
        +
        VENUE_WEIGHT
        * df[
            "home_adj_venue_goal_defense"
        ]
    )

    df[
        "away_goal_attack"
    ] = (
        OVERALL_WEIGHT
        * df[
            "away_adj_goal_attack"
        ]
        +
        VENUE_WEIGHT
        * df[
            "away_adj_venue_goal_attack"
        ]
    )

    df[
        "away_goal_defense"
    ] = (
        OVERALL_WEIGHT
        * df[
            "away_adj_goal_defense"
        ]
        +
        VENUE_WEIGHT
        * df[
            "away_adj_venue_goal_defense"
        ]
    )

    # ========================================================
    # SHOTS
    # ========================================================

    df[
        "home_shot_attack"
    ] = (
        OVERALL_WEIGHT
        * df[
            "home_adj_shot_attack"
        ]
        +
        VENUE_WEIGHT
        * df[
            "home_adj_venue_shot_attack"
        ]
    )

    df[
        "home_shot_defense"
    ] = (
        OVERALL_WEIGHT
        * df[
            "home_adj_shot_defense"
        ]
        +
        VENUE_WEIGHT
        * df[
            "home_adj_venue_shot_defense"
        ]
    )

    df[
        "away_shot_attack"
    ] = (
        OVERALL_WEIGHT
        * df[
            "away_adj_shot_attack"
        ]
        +
        VENUE_WEIGHT
        * df[
            "away_adj_venue_shot_attack"
        ]
    )

    df[
        "away_shot_defense"
    ] = (
        OVERALL_WEIGHT
        * df[
            "away_adj_shot_defense"
        ]
        +
        VENUE_WEIGHT
        * df[
            "away_adj_venue_shot_defense"
        ]
    )

    # ========================================================
    # SHOTS ON TARGET
    # ========================================================

    df[
        "home_sot_attack"
    ] = (
        OVERALL_WEIGHT
        * df[
            "home_adj_sot_attack"
        ]
        +
        VENUE_WEIGHT
        * df[
            "home_adj_venue_sot_attack"
        ]
    )

    df[
        "home_sot_defense"
    ] = (
        OVERALL_WEIGHT
        * df[
            "home_adj_sot_defense"
        ]
        +
        VENUE_WEIGHT
        * df[
            "home_adj_venue_sot_defense"
        ]
    )

    df[
        "away_sot_attack"
    ] = (
        OVERALL_WEIGHT
        * df[
            "away_adj_sot_attack"
        ]
        +
        VENUE_WEIGHT
        * df[
            "away_adj_venue_sot_attack"
        ]
    )

    df[
        "away_sot_defense"
    ] = (
        OVERALL_WEIGHT
        * df[
            "away_adj_sot_defense"
        ]
        +
        VENUE_WEIGHT
        * df[
            "away_adj_venue_sot_defense"
        ]
    )

    return df

    raise ValueError(
        "V3 predictions do not contain the "
        "component goal/shot/SOT strengths needed "
        "for xG weight tuning.\n"
        f"Missing columns: {missing}\n\n"
        "Run:\n"
        "python - <<'PY'\n"
        "import pandas as pd\n"
        "df = pd.read_csv('data/processed/opponent_strength_v3_predictions.csv', nrows=1)\n"
        "print(df.columns.tolist())\n"
        "PY\n"
        "and send me the output."
    )


# ============================================================
# GENERATE VALID WEIGHT COMBINATIONS
# ============================================================

def build_weight_grid():

    rows = []

    for goal_w in GOAL_WEIGHTS:

        for xg_w in XG_WEIGHTS:

            for sot_w in SOT_WEIGHTS:

                shot_w = (
                    1.0
                    - goal_w
                    - xg_w
                    - sot_w
                )

                if (
                    shot_w < -1e-9
                    or
                    shot_w > 1.0 + 1e-9
                ):
                    continue

                shot_w = round(
                    shot_w,
                    3,
                )

                # Keep shots in the region supported
                # by the coarse search.
                if not (
                    0.05
                    <= shot_w
                    <= 0.25
                ):
                    continue

                rows.append(
                    (
                        goal_w,
                        xg_w,
                        shot_w,
                        sot_w,
                    )
                )

    return rows


# ============================================================
# BUILD LAMBDAS
# ============================================================

def build_lambdas(
    df,
    goal_weight,
    xg_weight,
    shot_weight,
    sot_weight,
    xg_family,
):

    if xg_family == "xg":

        home_xg_attack = (
            df[
                "home_xg_attack"
            ]
        )

        home_xg_defense = (
            df[
                "home_xg_defense"
            ]
        )

        away_xg_attack = (
            df[
                "away_xg_attack"
            ]
        )

        away_xg_defense = (
            df[
                "away_xg_defense"
            ]
        )

    elif xg_family == "npxg":

        home_xg_attack = (
            df[
                "home_npxg_attack"
            ]
        )

        home_xg_defense = (
            df[
                "home_npxg_defense"
            ]
        )

        away_xg_attack = (
            df[
                "away_npxg_attack"
            ]
        )

        away_xg_defense = (
            df[
                "away_npxg_defense"
            ]
        )

    else:

        raise ValueError(
            f"Unknown xG family: "
            f"{xg_family}"
        )

    # --------------------------------------------------------
    # COMBINED ATTACK
    # --------------------------------------------------------

    home_attack = (
        goal_weight
        * df[
            "home_goal_attack"
        ]
        +
        xg_weight
        * home_xg_attack
        +
        shot_weight
        * df[
            "home_shot_attack"
        ]
        +
        sot_weight
        * df[
            "home_sot_attack"
        ]
    )

    away_attack = (
        goal_weight
        * df[
            "away_goal_attack"
        ]
        +
        xg_weight
        * away_xg_attack
        +
        shot_weight
        * df[
            "away_shot_attack"
        ]
        +
        sot_weight
        * df[
            "away_sot_attack"
        ]
    )

    # --------------------------------------------------------
    # COMBINED DEFENSE
    # --------------------------------------------------------

    home_defense = (
        goal_weight
        * df[
            "home_goal_defense"
        ]
        +
        xg_weight
        * home_xg_defense
        +
        shot_weight
        * df[
            "home_shot_defense"
        ]
        +
        sot_weight
        * df[
            "home_sot_defense"
        ]
    )

    away_defense = (
        goal_weight
        * df[
            "away_goal_defense"
        ]
        +
        xg_weight
        * away_xg_defense
        +
        shot_weight
        * df[
            "away_shot_defense"
        ]
        +
        sot_weight
        * df[
            "away_sot_defense"
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
    goal_weight,
    xg_weight,
    shot_weight,
    sot_weight,
    xg_family,
):

    season = (
        df[
            "season"
        ]
        .astype(str)
        .str.zfill(4)
    )

    mask = season.isin(
        seasons
    )

    sub = df.loc[
        mask
    ].copy()

    (
        home_lambda,
        away_lambda,
    ) = build_lambdas(
        sub,
        goal_weight,
        xg_weight,
        shot_weight,
        sot_weight,
        xg_family,
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
        f"{'Baseline':>14}"
        f"{'V5':>14}"
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
    print("TUNING XG SIGNAL WEIGHTS V5")
    print("==============================")
    print()

    v3 = load_v3()

    xg = load_xg_features()

    xg_match = (
        build_xg_match_table(
            xg
        )
    )

    df = merge_features(
        v3,
        xg_match,
    )

    print(
        f"Eligible PL/Bundesliga matches: "
        f"{len(df):,}"
    )

    print()

    print(
        "Checking existing signal columns..."
    )

    df = detect_existing_signals(
        df
    )

    # --------------------------------------------------------
    # BUILD GRID
    # --------------------------------------------------------

    grid = build_weight_grid()

    print(
        f"Goal range: "
        f"{GOAL_WEIGHTS.min():.0%} "
        f"to "
        f"{GOAL_WEIGHTS.max():.0%}"
    )

    print(
        f"xG range: "
        f"{XG_WEIGHTS.min():.0%} "
        f"to "
        f"{XG_WEIGHTS.max():.0%}"
    )

    print(
        f"SOT range: "
        f"{min(SOT_WEIGHTS):.0%} "
        f"to "
        f"{max(SOT_WEIGHTS):.0%}"
    )

    print(
        f"Combinations per xG family: "
        f"{len(grid):,}"
    )

    print(
        f"Total tests: "
        f"{len(grid) * 2:,}"
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_tune = evaluate(
        df,
        TUNING_SEASONS,
        BASE_GOAL_WEIGHT,
        0.0,
        BASE_SHOT_WEIGHT,
        BASE_SOT_WEIGHT,
        "xg",
    )

    print()
    print("==============================")
    print("CURRENT BASELINE")
    print("==============================")

    print(
        f"Goals: "
        f"{BASE_GOAL_WEIGHT:.0%}"
    )

    print(
        f"Shots: "
        f"{BASE_SHOT_WEIGHT:.0%}"
    )

    print(
        f"SOT: "
        f"{BASE_SOT_WEIGHT:.0%}"
    )

    print(
        f"Tuning LL: "
        f"{baseline_tune['log_loss']:.5f}"
    )

    # --------------------------------------------------------
    # TUNE
    # --------------------------------------------------------

    rows = []

    total = (
        len(grid)
        * 2
    )

    tested = 0

    for family in [
        "xg",
        "npxg",
    ]:

        for (
            goal_w,
            xg_w,
            shot_w,
            sot_w,
        ) in grid:

            metrics = evaluate(
                df,
                TUNING_SEASONS,
                goal_w,
                xg_w,
                shot_w,
                sot_w,
                family,
            )

            rows.append(
                {
                    "xg_family":
                        family,

                    "goal_weight":
                        goal_w,

                    "xg_weight":
                        xg_w,

                    "shot_weight":
                        shot_w,

                    "sot_weight":
                        sot_w,

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
                tested % 500
                == 0
            ):

                print(
                    f"Tested "
                    f"{tested:,}/"
                    f"{total:,}"
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

    # --------------------------------------------------------
    # TOP 25
    # --------------------------------------------------------

    print()
    print("==============================")
    print("TOP 25 XG SIGNAL BLENDS")
    print("==============================")

    display = (
        results
        .head(25)
        .copy()
    )

    display[
        "accuracy"
    ] *= 100.0

    print(
        display[
            [
                "rank",
                "xg_family",

                "goal_weight",
                "xg_weight",
                "shot_weight",
                "sot_weight",

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

    # --------------------------------------------------------
    # BEST
    # --------------------------------------------------------

    best = results.iloc[
        0
    ]

    best_family = str(
        best[
            "xg_family"
        ]
    )

    best_goal = float(
        best[
            "goal_weight"
        ]
    )

    best_xg = float(
        best[
            "xg_weight"
        ]
    )

    best_shot = float(
        best[
            "shot_weight"
        ]
    )

    best_sot = float(
        best[
            "sot_weight"
        ]
    )

    print()
    print("==============================")
    print("WINNING V5 SIGNAL BLEND")
    print("==============================")

    print(
        f"xG family: "
        f"{best_family}"
    )

    print(
        f"Goals: "
        f"{best_goal:.1%}"
    )

    print(
        f"{best_family}: "
        f"{best_xg:.1%}"
    )

    print(
        f"Shots: "
        f"{best_shot:.1%}"
    )

    print(
        f"SOT: "
        f"{best_sot:.1%}"
    )

    print(
        f"Tuning LL: "
        f"{best['log_loss']:.5f}"
    )

    # --------------------------------------------------------
    # TUNING COMPARISON
    # --------------------------------------------------------

    winner_tune = evaluate(
        df,
        TUNING_SEASONS,
        best_goal,
        best_xg,
        best_shot,
        best_sot,
        best_family,
    )

    print_comparison(
        "TUNING — 2021/22 TO 2022/23",
        baseline_tune,
        winner_tune,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    baseline_validation = evaluate(
        df,
        VALIDATION_SEASONS,
        BASE_GOAL_WEIGHT,
        0.0,
        BASE_SHOT_WEIGHT,
        BASE_SOT_WEIGHT,
        "xg",
    )

    winner_validation = evaluate(
        df,
        VALIDATION_SEASONS,
        best_goal,
        best_xg,
        best_shot,
        best_sot,
        best_family,
    )

    print_comparison(
        "VALIDATION — 2023/24",
        baseline_validation,
        winner_validation,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    baseline_final = evaluate(
        df,
        FINAL_SEASONS,
        BASE_GOAL_WEIGHT,
        0.0,
        BASE_SHOT_WEIGHT,
        BASE_SOT_WEIGHT,
        "xg",
    )

    winner_final = evaluate(
        df,
        FINAL_SEASONS,
        best_goal,
        best_xg,
        best_shot,
        best_sot,
        best_family,
    )

    print_comparison(
        "FINAL CHECK — 2024/25",
        baseline_final,
        winner_final,
    )

    # --------------------------------------------------------
    # SAVE WINNING PREDICTIONS
    # --------------------------------------------------------

    (
        home_lambda,
        away_lambda,
    ) = build_lambdas(
        df,
        best_goal,
        best_xg,
        best_shot,
        best_sot,
        best_family,
    )

    valid = (
        home_lambda.notna()
        &
        away_lambda.notna()
    )

    output = df.loc[
        valid
    ].copy()

    output[
        "home_lambda_v5"
    ] = home_lambda.loc[
        valid
    ].to_numpy()

    output[
        "away_lambda_v5"
    ] = away_lambda.loc[
        valid
    ].to_numpy()

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
        "xg_family_v5"
    ] = best_family

    output[
        "goal_weight_v5"
    ] = best_goal

    output[
        "xg_weight_v5"
    ] = best_xg

    output[
        "shot_weight_v5"
    ] = best_shot

    output[
        "sot_weight_v5"
    ] = best_sot

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print()
    print("==============================")
    print("XG SIGNAL TUNING COMPLETE")
    print("==============================")

    print(
        "Weights selected using "
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