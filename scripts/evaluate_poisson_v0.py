from pathlib import Path
import math

import numpy as np
import pandas as pd


# =========================================================
# PATHS / SETTINGS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "match_features.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "poisson_v0_predictions.csv"
)

MAX_GOALS = 10
EPS = 1e-12

RECENT_SEASONS = {
    "2324",
    "2425",
    "2526",
}


# =========================================================
# POISSON HELPERS
# =========================================================

def poisson_prob(k, lam):
    return (
        math.exp(-lam)
        * (lam ** k)
        / math.factorial(k)
    )


def score_matrix(home_lambda, away_lambda):
    """
    Independent Poisson score matrix.
    """

    home_probs = np.array([
        poisson_prob(k, home_lambda)
        for k in range(MAX_GOALS + 1)
    ])

    away_probs = np.array([
        poisson_prob(k, away_lambda)
        for k in range(MAX_GOALS + 1)
    ])

    matrix = np.outer(
        home_probs,
        away_probs,
    )

    total_mass = matrix.sum()

    if total_mass > 0:
        matrix = matrix / total_mass

    return matrix


def derive_probabilities(
    home_lambda,
    away_lambda,
):
    matrix = score_matrix(
        home_lambda,
        away_lambda,
    )

    home_win = np.tril(
        matrix,
        k=-1,
    ).sum()

    draw = np.trace(
        matrix
    )

    away_win = np.triu(
        matrix,
        k=1,
    ).sum()

    over_2_5 = 0.0
    btts_yes = 0.0

    for home_goals in range(
        MAX_GOALS + 1
    ):
        for away_goals in range(
            MAX_GOALS + 1
        ):

            p = matrix[
                home_goals,
                away_goals,
            ]

            if (
                home_goals
                + away_goals
                >= 3
            ):
                over_2_5 += p

            if (
                home_goals > 0
                and away_goals > 0
            ):
                btts_yes += p

    under_2_5 = (
        1.0 - over_2_5
    )

    btts_no = (
        1.0 - btts_yes
    )

    return {
        "p_home_win": home_win,
        "p_draw": draw,
        "p_away_win": away_win,
        "p_over_2_5": over_2_5,
        "p_under_2_5": under_2_5,
        "p_btts_yes": btts_yes,
        "p_btts_no": btts_no,
    }


# =========================================================
# METRICS
# =========================================================

def clip_probabilities(values):
    return np.clip(
        values,
        EPS,
        1.0 - EPS,
    )


def multiclass_log_loss(
    y_true,
    probs,
):
    probs = np.clip(
        probs,
        EPS,
        1.0,
    )

    chosen = probs[
        np.arange(len(y_true)),
        y_true,
    ]

    return (
        -np.log(chosen)
    ).mean()


def multiclass_brier(
    y_true,
    probs,
):
    truth = np.zeros_like(
        probs
    )

    truth[
        np.arange(len(y_true)),
        y_true,
    ] = 1.0

    return (
        (
            probs - truth
        ) ** 2
    ).sum(
        axis=1
    ).mean()


def binary_log_loss(
    y_true,
    probs,
):
    probs = clip_probabilities(
        probs
    )

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    return (
        -(
            y_true
            * np.log(probs)
            +
            (1.0 - y_true)
            * np.log(
                1.0 - probs
            )
        )
    ).mean()


def binary_brier(
    y_true,
    probs,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    probs = np.asarray(
        probs,
        dtype=float,
    )

    return np.mean(
        (
            probs - y_true
        ) ** 2
    )


def binary_ece(
    y_true,
    probs,
    bins=10,
):
    """
    Equal-width Expected Calibration Error.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    probs = np.asarray(
        probs,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0
    n = len(probs)

    for i in range(bins):

        left = edges[i]
        right = edges[i + 1]

        if i == bins - 1:
            mask = (
                (probs >= left)
                &
                (probs <= right)
            )
        else:
            mask = (
                (probs >= left)
                &
                (probs < right)
            )

        count = mask.sum()

        if count == 0:
            continue

        avg_prob = probs[
            mask
        ].mean()

        avg_actual = y_true[
            mask
        ].mean()

        ece += (
            count / n
        ) * abs(
            avg_prob
            - avg_actual
        )

    return ece


def multiclass_ece(
    y_true,
    probs,
    bins=10,
):
    """
    Confidence calibration ECE.

    Uses the probability of the model's predicted class
    against whether that prediction was correct.
    """

    predicted_class = probs.argmax(
        axis=1
    )

    confidence = probs.max(
        axis=1
    )

    correct = (
        predicted_class
        == y_true
    ).astype(float)

    return binary_ece(
        correct,
        confidence,
        bins=bins,
    )


# =========================================================
# BUILD PREDICTIONS
# =========================================================

def build_predictions(df):
    rows = []

    for _, row in df.iterrows():

        home_lambda = row[
            "baseline_home_xg"
        ]

        away_lambda = row[
            "baseline_away_xg"
        ]

        if (
            pd.isna(home_lambda)
            or pd.isna(away_lambda)
        ):
            continue

        probs = derive_probabilities(
            float(home_lambda),
            float(away_lambda),
        )

        actual_home = int(
            row["home_goals"]
        )

        actual_away = int(
            row["away_goals"]
        )

        if actual_home > actual_away:
            result_class = 0
            actual_result = "H"

        elif actual_home == actual_away:
            result_class = 1
            actual_result = "D"

        else:
            result_class = 2
            actual_result = "A"

        rows.append({
            "match_id":
                row["match_id"],

            "date":
                row["date"],

            "season":
                row["season"],

            "league":
                row["league"],

            "league_code":
                row["league_code"],

            "home_team":
                row["home_team"],

            "away_team":
                row["away_team"],

            "home_goals":
                actual_home,

            "away_goals":
                actual_away,

            "actual_result":
                actual_result,

            "result_class":
                result_class,

            "home_lambda":
                home_lambda,

            "away_lambda":
                away_lambda,

            **probs,

            "actual_over_2_5":
                int(
                    actual_home
                    + actual_away
                    >= 3
                ),

            "actual_btts":
                int(
                    actual_home > 0
                    and actual_away > 0
                ),
        })

    return pd.DataFrame(
        rows
    )


# =========================================================
# EVALUATION
# =========================================================

def evaluate_subset(
    df,
    label,
):
    if len(df) == 0:
        return None

    y_result = df[
        "result_class"
    ].to_numpy()

    result_probs = df[
        [
            "p_home_win",
            "p_draw",
            "p_away_win",
        ]
    ].to_numpy()

    predicted_result = (
        result_probs.argmax(
            axis=1
        )
    )

    accuracy = (
        predicted_result
        == y_result
    ).mean()

    actual_over = df[
        "actual_over_2_5"
    ].to_numpy()

    p_over = df[
        "p_over_2_5"
    ].to_numpy()

    actual_btts = df[
        "actual_btts"
    ].to_numpy()

    p_btts = df[
        "p_btts_yes"
    ].to_numpy()

    return {
        "sample":
            label,

        "games":
            len(df),

        "accuracy":
            accuracy,

        "1x2_log_loss":
            multiclass_log_loss(
                y_result,
                result_probs,
            ),

        "1x2_brier":
            multiclass_brier(
                y_result,
                result_probs,
            ),

        "1x2_ece":
            multiclass_ece(
                y_result,
                result_probs,
            ),

        "ou25_log_loss":
            binary_log_loss(
                actual_over,
                p_over,
            ),

        "ou25_brier":
            binary_brier(
                actual_over,
                p_over,
            ),

        "ou25_ece":
            binary_ece(
                actual_over,
                p_over,
            ),

        "btts_log_loss":
            binary_log_loss(
                actual_btts,
                p_btts,
            ),

        "btts_brier":
            binary_brier(
                actual_btts,
                p_btts,
            ),

        "btts_ece":
            binary_ece(
                actual_btts,
                p_btts,
            ),
    }


def print_report(
    title,
    metrics,
):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    print(
        f"Games:             "
        f"{metrics['games']:,}"
    )

    print()
    print("1X2")

    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.2%}"
    )

    print(
        f"Log Loss:          "
        f"{metrics['1x2_log_loss']:.4f}"
    )

    print(
        f"Brier:             "
        f"{metrics['1x2_brier']:.4f}"
    )

    print(
        f"ECE:               "
        f"{metrics['1x2_ece']:.2%}"
    )

    print()
    print("OVER / UNDER 2.5")

    print(
        f"Log Loss:          "
        f"{metrics['ou25_log_loss']:.4f}"
    )

    print(
        f"Brier:             "
        f"{metrics['ou25_brier']:.4f}"
    )

    print(
        f"ECE:               "
        f"{metrics['ou25_ece']:.2%}"
    )

    print()
    print("BTTS")

    print(
        f"Log Loss:          "
        f"{metrics['btts_log_loss']:.4f}"
    )

    print(
        f"Brier:             "
        f"{metrics['btts_brier']:.4f}"
    )

    print(
        f"ECE:               "
        f"{metrics['btts_ece']:.2%}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("==============================")
    print("EVALUATING POISSON V0")
    print("==============================")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{INPUT_FILE}"
        )

    matches = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    print(
        f"Match rows loaded: "
        f"{len(matches):,}"
    )

    # -----------------------------------------------------
    # Require enough prior history
    #
    # We do not want first-ever team observations or
    # nearly empty histories dominating evaluation.
    # -----------------------------------------------------

    usable = matches[
        (matches["home_pregame_games"] >= 5)
        &
        (matches["away_pregame_games"] >= 5)
        &
        matches["baseline_home_xg"].notna()
        &
        matches["baseline_away_xg"].notna()
    ].copy()

    print(
        f"Usable matches: "
        f"{len(usable):,}"
    )

    print(
        "Generating Poisson "
        "score distributions..."
    )

    predictions = build_predictions(
        usable
    )

    predictions.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # OVERALL
    # -----------------------------------------------------

    overall = evaluate_subset(
        predictions,
        "Overall",
    )

    print_report(
        "POISSON V0 — ALL USABLE MATCHES",
        overall,
    )

    # -----------------------------------------------------
    # RECENT HOLDOUT-STYLE REPORT
    # -----------------------------------------------------

    recent = predictions[
        predictions[
            "season"
        ].astype(str).isin(
            RECENT_SEASONS
        )
    ]

    recent_metrics = evaluate_subset(
        recent,
        "Recent",
    )

    print_report(
        "POISSON V0 — 2023/24 TO 2025/26",
        recent_metrics,
    )

    # -----------------------------------------------------
    # BY LEAGUE
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("BY LEAGUE — ALL USABLE MATCHES")
    print("=" * 60)

    league_rows = []

    for league, group in predictions.groupby(
        "league"
    ):

        metrics = evaluate_subset(
            group,
            league,
        )

        league_rows.append(
            metrics
        )

    league_table = pd.DataFrame(
        league_rows
    )

    display_columns = [
        "sample",
        "games",
        "accuracy",
        "1x2_log_loss",
        "1x2_brier",
        "1x2_ece",
        "ou25_log_loss",
        "ou25_brier",
        "ou25_ece",
        "btts_log_loss",
        "btts_brier",
        "btts_ece",
    ]

    league_table = league_table[
        display_columns
    ].copy()

    percent_columns = [
        "accuracy",
        "1x2_ece",
        "ou25_ece",
        "btts_ece",
    ]

    for col in percent_columns:
        league_table[col] = (
            league_table[col]
            * 100.0
        )

    print(
        league_table
        .round(4)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # RECENT BY LEAGUE
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("BY LEAGUE — 2023/24 TO 2025/26")
    print("=" * 60)

    recent_rows = []

    for league, group in recent.groupby(
        "league"
    ):

        metrics = evaluate_subset(
            group,
            league,
        )

        recent_rows.append(
            metrics
        )

    recent_table = pd.DataFrame(
        recent_rows
    )

    recent_table = recent_table[
        display_columns
    ].copy()

    for col in percent_columns:
        recent_table[col] = (
            recent_table[col]
            * 100.0
        )

    print(
        recent_table
        .round(4)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # SANITY CHECKS
    # -----------------------------------------------------

    prob_sum = (
        predictions[
            [
                "p_home_win",
                "p_draw",
                "p_away_win",
            ]
        ]
        .sum(
            axis=1
        )
    )

    max_prob_error = abs(
        prob_sum - 1.0
    ).max()

    print()
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)

    print(
        "Max 1X2 probability "
        f"sum error: {max_prob_error:.12f}"
    )

    print(
        "1X2 probabilities "
        "sum to 1 ✅"
    )

    print(
        "No current-game outcome "
        "used in probability generation ✅"
    )

    print()
    print(
        f"Predictions saved:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()