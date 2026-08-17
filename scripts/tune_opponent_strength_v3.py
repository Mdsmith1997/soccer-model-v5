from pathlib import Path
import time

import numpy as np
import pandas as pd

import build_opponent_adjusted_v3 as builder
import tune_opponent_adjustment_v3 as model


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "team_game_stats.csv"
)

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "opponent_strength_v3_tuning_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "opponent_strength_v3_predictions.csv"
)


# =========================================================
# SPLITS
# =========================================================

TUNING_SEASONS = {
    "2122",
    "2223",
}

VALIDATION_SEASONS = {
    "2324",
    "2425",
}

FINAL_SEASON = {
    "2526",
}


# =========================================================
# STRENGTH GRID
# =========================================================

OPPONENT_STRENGTHS = np.round(
    np.arange(
        0.70,
        0.951,
        0.025,
    ),
    3,
)


# =========================================================
# SAFE GAME-LEVEL OPPONENT ADJUSTMENT
# =========================================================

def add_adjusted_game_performance(
    df,
    strength,
):
    """
    Adjust each match performance for opponent quality.

    Missing opponent ratings are treated as neutral = 1.0.

    This is important because early-history matches should
    not disappear simply because the opponent did not yet
    have enough pregame information.
    """

    out = df.copy()

    opponent_columns = [
        "opp_attack_goals",
        "opp_defense_goals",
        "opp_attack_shots",
        "opp_defense_shots",
        "opp_attack_sot",
        "opp_defense_sot",
    ]

    for col in opponent_columns:
        out[col] = (
            out[col]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(1.0)
        )

    # -----------------------------------------------------
    # OFFENSE
    # -----------------------------------------------------

    goal_defense_factor = (
        (
            1.0 - strength
        )
        +
        strength
        * out[
            "opp_defense_goals"
        ]
    )

    shot_defense_factor = (
        (
            1.0 - strength
        )
        +
        strength
        * out[
            "opp_defense_shots"
        ]
    )

    sot_defense_factor = (
        (
            1.0 - strength
        )
        +
        strength
        * out[
            "opp_defense_sot"
        ]
    )

    out[
        "adj_goals_for"
    ] = (
        out[
            "goals_for"
        ]
        /
        goal_defense_factor
    )

    out[
        "adj_shots_for"
    ] = (
        out[
            "shots_for"
        ]
        /
        shot_defense_factor
    )

    out[
        "adj_sot_for"
    ] = (
        out[
            "shots_on_target_for"
        ]
        /
        sot_defense_factor
    )

    # -----------------------------------------------------
    # DEFENSE
    # -----------------------------------------------------

    goal_attack_factor = (
        (
            1.0 - strength
        )
        +
        strength
        * out[
            "opp_attack_goals"
        ]
    )

    shot_attack_factor = (
        (
            1.0 - strength
        )
        +
        strength
        * out[
            "opp_attack_shots"
        ]
    )

    sot_attack_factor = (
        (
            1.0 - strength
        )
        +
        strength
        * out[
            "opp_attack_sot"
        ]
    )

    out[
        "adj_goals_against"
    ] = (
        out[
            "goals_against"
        ]
        /
        goal_attack_factor
    )

    out[
        "adj_shots_against"
    ] = (
        out[
            "shots_against"
        ]
        /
        shot_attack_factor
    )

    out[
        "adj_sot_against"
    ] = (
        out[
            "shots_on_target_against"
        ]
        /
        sot_attack_factor
    )

    return out


# =========================================================
# BUILD FEATURE STORE FOR ONE STRENGTH
# =========================================================

def build_strength_features(
    base,
    venue_baselines,
    strength,
):

    df = base.copy()

    # -----------------------------------------------------
    # GAME-LEVEL OPPONENT ADJUSTMENT
    # -----------------------------------------------------

    df = add_adjusted_game_performance(
        df,
        strength,
    )

    # -----------------------------------------------------
    # ROLL ADJUSTED HISTORY
    # -----------------------------------------------------

    df = builder.add_adjusted_history(
        df
    )

    df = builder.add_v3_strengths(
        df
    )

    # -----------------------------------------------------
    # HOME/AWAY SHOT BASELINES
    # -----------------------------------------------------

    df = df.merge(
        venue_baselines,
        on=[
            "league_code",
            "date",
        ],
        how="left",
        validate="many_to_one",
    )

    # -----------------------------------------------------
    # HOME/AWAY HISTORIES
    # -----------------------------------------------------

    df = model.add_venue_histories(
        df
    )

    df = model.build_venue_strengths(
        df
    )

    df = model.add_league_transition(
        df
    )

    # -----------------------------------------------------
    # MATCH TABLE
    # -----------------------------------------------------

    matches = model.build_match_table(
        df
    )

    return matches


# =========================================================
# PRINT COMPARISON
# =========================================================

def print_comparison(
    title,
    baseline,
    candidate,
):

    print()
    print("=" * 62)
    print(title)
    print("=" * 62)

    print(
        f"Games: "
        f"{candidate['games']:,}"
    )

    print()

    print(
        f"{'Metric':<15}"
        f"{'Strength 0':>14}"
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


# =========================================================
# MAIN
# =========================================================

def main():

    start = time.time()

    print()
    print("==============================")
    print("TUNING V3 OPPONENT STRENGTH")
    print("==============================")
    print()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
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

    df = (
        df
        .sort_values(
            [
                "date",
                "match_id",
                "is_home",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Team rows loaded: "
        f"{len(df):,}"
    )

    print()
    print(
        "Frozen model structure:"
    )

    print(
        "Recency:             0.95"
    )

    print(
        "Goals / Shots / SOT: "
        "70% / 15% / 15%"
    )

    print(
        "Overall / Venue:     "
        "75% / 25%"
    )

    print(
        "Adjusted feature use: 100%"
    )

    # =====================================================
    # BASE FEATURES THAT DO NOT CHANGE WITH STRENGTH
    # =====================================================

    print()
    print(
        "Building common leakage-safe "
        "league baselines..."
    )

    league_baselines = (
        builder.build_league_baselines(
            df
        )
    )

    df = df.merge(
        league_baselines,
        on=[
            "league_code",
            "date",
        ],
        how="left",
        validate="many_to_one",
    )

    print(
        "Building raw pregame histories..."
    )

    df = builder.build_basic_history(
        df
    )

    df = builder.add_raw_strengths(
        df
    )

    print(
        "Attaching opponent pregame snapshots..."
    )

    base = (
        builder.attach_opponent_pregame_strength(
            df
        )
    )

    # -----------------------------------------------------
    # Venue league baselines are also strength-independent.
    # -----------------------------------------------------

    print(
        "Building home/away shot baselines..."
    )

    venue_baselines = (
        model.build_venue_league_baselines(
            base
        )
    )

    # =====================================================
    # GRID SEARCH
    # =====================================================

    print()
    print(
        f"Testing "
        f"{len(OPPONENT_STRENGTHS)} "
        f"opponent strengths..."
    )

    results = []

    cached_matches = {}

    for strength in OPPONENT_STRENGTHS:

        print(
            f"  Strength "
            f"{strength:.1f}..."
        )

        matches = build_strength_features(
            base,
            venue_baselines,
            strength,
        )

        cached_matches[
            float(strength)
        ] = matches

        # Always use 100% of the resulting adjusted
        # strength features.
        metrics = model.calculate_model(
            matches,
            adjustment_weight=1.0,
            seasons=TUNING_SEASONS,
        )

        results.append({
            "opponent_strength":
                strength,

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
        })

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
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

    results_df[
        "rank"
    ] = (
        np.arange(
            len(results_df)
        )
        + 1
    )

    results_df.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    # =====================================================
    # RESULTS TABLE
    # =====================================================

    print()
    print("==============================")
    print("OPPONENT STRENGTH RESULTS")
    print("==============================")

    display = results_df.copy()

    display[
        "accuracy"
    ] *= 100.0

    print(
        display[
            [
                "rank",
                "opponent_strength",
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

    # =====================================================
    # WINNER
    # =====================================================

    best = (
        results_df
        .iloc[0]
    )

    best_strength = float(
        best[
            "opponent_strength"
        ]
    )

    print()
    print("==============================")
    print("WINNING OPPONENT STRENGTH")
    print("==============================")

    print(
        f"Strength:      "
        f"{best_strength:.2f}"
    )

    print(
        f"Tuning LL:     "
        f"{best['log_loss']:.5f}"
    )

    # Current production-like V3 was built at 0.50.
    current_row = results_df[
        np.isclose(
            results_df[
                "opponent_strength"
            ],
            0.50,
        )
    ]

    if len(current_row) > 0:

        current_ll = float(
            current_row.iloc[
                0
            ][
                "log_loss"
            ]
        )

        print(
            f"Strength 0.50: "
            f"{current_ll:.5f}"
        )

        print(
            f"Improvement:   "
            f"{best['log_loss'] - current_ll:+.5f}"
        )

    # =====================================================
    # BASELINE = NO OPPONENT CORRECTION
    # =====================================================

    zero_matches = build_strength_features(
    base,
    venue_baselines,
    0.0,
)

    best_matches = build_strength_features(
    base,
    venue_baselines,
    best_strength,
)

    zero_tune = model.calculate_model(
        zero_matches,
        adjustment_weight=1.0,
        seasons=TUNING_SEASONS,
    )

    best_tune = model.calculate_model(
        best_matches,
        adjustment_weight=1.0,
        seasons=TUNING_SEASONS,
    )

    print_comparison(
        "TUNING — 2021/22 TO 2022/23",
        zero_tune,
        best_tune,
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    zero_validation = (
        model.calculate_model(
            zero_matches,
            adjustment_weight=1.0,
            seasons=VALIDATION_SEASONS,
        )
    )

    best_validation = (
        model.calculate_model(
            best_matches,
            adjustment_weight=1.0,
            seasons=VALIDATION_SEASONS,
        )
    )

    print_comparison(
        "VALIDATION — 2023/24 TO 2024/25",
        zero_validation,
        best_validation,
    )

    # =====================================================
    # 2025/26 CONFIRMATION
    # =====================================================

    zero_final = (
        model.calculate_model(
            zero_matches,
            adjustment_weight=1.0,
            seasons=FINAL_SEASON,
        )
    )

    best_final = (
        model.calculate_model(
            best_matches,
            adjustment_weight=1.0,
            seasons=FINAL_SEASON,
        )
    )

    print_comparison(
        "2025/26 CONFIRMATION",
        zero_final,
        best_final,
    )

    # =====================================================
    # 2025/26 BY LEAGUE
    # =====================================================

    print()
    print("=" * 90)
    print("2025/26 — BY LEAGUE")
    print("=" * 90)

    league_rows = []

    leagues = sorted(
        best_matches[
            "league"
        ].dropna().unique()
    )

    for league in leagues:

        zero_group = zero_matches[
            zero_matches[
                "league"
            ]
            == league
        ]

        best_group = best_matches[
            best_matches[
                "league"
            ]
            == league
        ]

        baseline = model.calculate_model(
            zero_group,
            adjustment_weight=1.0,
            seasons=FINAL_SEASON,
        )

        candidate = model.calculate_model(
            best_group,
            adjustment_weight=1.0,
            seasons=FINAL_SEASON,
        )

        league_rows.append({
            "league":
                league,

            "games":
                candidate[
                    "games"
                ],

            "strength0_ll":
                baseline[
                    "log_loss"
                ],

            "winner_ll":
                candidate[
                    "log_loss"
                ],

            "ll_change":
                (
                    candidate[
                        "log_loss"
                    ]
                    -
                    baseline[
                        "log_loss"
                    ]
                ),

            "strength0_brier":
                baseline[
                    "brier"
                ],

            "winner_brier":
                candidate[
                    "brier"
                ],

            "strength0_acc":
                baseline[
                    "accuracy"
                ],

            "winner_acc":
                candidate[
                    "accuracy"
                ],
        })

    league_table = pd.DataFrame(
        league_rows
    )

    for col in [
        "strength0_acc",
        "winner_acc",
    ]:

        league_table[
            col
        ] *= 100.0

    print(
        league_table
        .round(5)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # SAVE WINNING PREDICTIONS
    # =====================================================

    predictions = model.build_predictions(
        best_matches,
        adjustment_weight=1.0,
    )

    predictions[
        "opponent_strength_v3"
    ] = best_strength

    predictions.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    print()
    print("==============================")
    print("V3 STRENGTH TUNING COMPLETE")
    print("==============================")

    if best_strength == 0.0:

        print(
            "Opponent correction rejected."
        )

    elif best_strength == 1.0:

        print(
            "Winning strength reached the "
            "upper search boundary."
        )

        print(
            "Do not increase it further yet; "
            "inspect validation first."
        )

    else:

        print(
            "Winning strength lies inside "
            "the search range."
        )

    print()
    print(
        "Opponent strength selected using "
        "2021/22–2022/23 only ✅"
    )

    print(
        "2023/24 onward used for "
        "confirmation, not selection ✅"
    )

    print()

    elapsed = (
        time.time()
        - start
    )

    print(
        f"Runtime: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print(
        f"Tuning results:"
        f"\n{OUTPUT_RESULTS}"
    )

    print()
    print(
        f"Winning V3 predictions:"
        f"\n{OUTPUT_PREDICTIONS}"
    )


if __name__ == "__main__":
    main()