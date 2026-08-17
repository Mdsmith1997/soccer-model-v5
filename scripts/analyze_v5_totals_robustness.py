from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from scripts.backtest_v5_totals_quick import build_dataset


OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "v5_totals_robustness.csv"
)


THRESHOLDS = [
    0.00,
    0.04,
    0.06,
    0.08,
    0.10,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
    0.16,
    0.18,
    0.20,
    0.22,
    0.25,
]


def evaluate(df, side, threshold):

    edge_col = (
        "over_edge"
        if side == "OVER"
        else "under_edge"
    )

    ev_col = (
        "over_ev"
        if side == "OVER"
        else "under_ev"
    )

    odds_col = (
        "over_odds"
        if side == "OVER"
        else "under_odds"
    )

    prob_col = (
        "p_over_v5"
        if side == "OVER"
        else "p_under_v5"
    )

    x = df[
        df[edge_col] >= threshold
    ].copy()

    if x.empty:
        return None

    if side == "OVER":

        x["won"] = (
            x["home_goals"]
            +
            x["away_goals"]
            >
            2.5
        )

    else:

        x["won"] = (
            x["home_goals"]
            +
            x["away_goals"]
            <
            2.5
        )

    x["profit"] = np.where(
        x["won"],
        x[odds_col] - 1.0,
        -1.0,
    )

    return {
        "side": side,
        "threshold": threshold,
        "bets": len(x),
        "wins": int(x["won"].sum()),
        "win_rate": x["won"].mean(),
        "avg_odds": x[odds_col].mean(),
        "avg_model_probability": x[prob_col].mean(),
        "avg_edge": x[edge_col].mean(),
        "avg_ev": x[ev_col].mean(),
        "profit": x["profit"].sum(),
        "roi": x["profit"].sum() / len(x),
    }


def grouped_evaluate(
    df,
    group_col,
    side,
    threshold,
):

    rows = []

    for group, g in df.groupby(group_col):

        result = evaluate(
            g,
            side,
            threshold,
        )

        if result is None:
            continue

        result["group_type"] = group_col
        result["group"] = str(group)

        rows.append(result)

    return rows


def print_table(df, title):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)
    print()

    if df.empty:
        print("No observations.")
        return

    show = df.copy()

    for col in [
        "win_rate",
        "avg_model_probability",
        "avg_edge",
        "avg_ev",
        "roi",
    ]:
        if col in show.columns:
            show[col] *= 100.0

    print(
        show.to_string(
            index=False,
            formatters={
                "threshold":
                    lambda x: f"{x:.0%}",

                "win_rate":
                    lambda x: f"{x:.2f}%",

                "avg_model_probability":
                    lambda x: f"{x:.2f}%",

                "avg_edge":
                    lambda x: f"{x:.2f}%",

                "avg_ev":
                    lambda x: f"{x:.2f}%",

                "avg_odds":
                    lambda x: f"{x:.3f}",

                "profit":
                    lambda x: f"{x:+.2f}u",

                "roi":
                    lambda x: f"{x:+.2f}%",
            },
        )
    )


def main():

    print()
    print("=" * 120)
    print("V5 TOTALS ROBUSTNESS ANALYSIS")
    print("=" * 120)

    df = build_dataset()

    print()
    print(f"Matched matches: {len(df):,}")

    rows = []

    for side in [
        "OVER",
        "UNDER",
    ]:

        for threshold in THRESHOLDS:

            result = evaluate(
                df,
                side,
                threshold,
            )

            if result is None:
                continue

            result["group_type"] = "ALL"
            result["group"] = "ALL"

            rows.append(result)

    results = pd.DataFrame(rows)

    print_table(
        results[
            results["side"].eq("OVER")
        ],
        "OVER 2.5 — THRESHOLD SCAN",
    )

    print_table(
        results[
            results["side"].eq("UNDER")
        ],
        "UNDER 2.5 — THRESHOLD SCAN",
    )

    # --------------------------------------------------------
    # Detailed stability analysis at the interesting thresholds.
    # --------------------------------------------------------

    detail_thresholds = [
        0.10,
        0.12,
        0.14,
        0.15,
        0.16,
        0.18,
        0.20,
    ]

    detail_rows = []

    for side in [
        "OVER",
        "UNDER",
    ]:

        for threshold in detail_thresholds:

            detail_rows.extend(
                grouped_evaluate(
                    df,
                    "league",
                    side,
                    threshold,
                )
            )

            detail_rows.extend(
                grouped_evaluate(
                    df,
                    "season",
                    side,
                    threshold,
                )
            )

    detail = pd.DataFrame(
        detail_rows
    )

    if not detail.empty:

        print_table(
            detail[
                detail["group_type"].eq(
                    "league"
                )
            ].sort_values(
                [
                    "side",
                    "threshold",
                    "group",
                ]
            ),
            "LEAGUE STABILITY",
        )

        print_table(
            detail[
                detail["group_type"].eq(
                    "season"
                )
            ].sort_values(
                [
                    "side",
                    "threshold",
                    "group",
                ]
            ),
            "SEASON STABILITY",
        )

    output = pd.concat(
        [
            results,
            detail,
        ],
        ignore_index=True,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 120)
    print("WHAT WE ARE LOOKING FOR")
    print("=" * 120)
    print()
    print(
        "1. Positive ROI across a RANGE of nearby thresholds."
    )
    print(
        "2. Similar behavior in both leagues."
    )
    print(
        "3. Positive performance across multiple seasons."
    )
    print(
        "4. Enough bets that results are not driven by a tiny sample."
    )
    print(
        "5. OVER and UNDER evaluated independently."
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
