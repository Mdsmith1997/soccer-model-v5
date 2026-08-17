from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from scripts.analyze_v5_totals_historical_clv import (
    build_clv_dataset,
)


OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "v5_totals_price_sensitivity.csv"
)


RAW_THRESHOLDS = [
    0.10,
    0.11,
    0.12,
    0.13,
]


ODDS_BUCKETS = [
    (1.80, 2.00, "1.80-1.99"),
    (2.00, 2.20, "2.00-2.19"),
    (2.20, 2.40, "2.20-2.39"),
    (2.40, 2.60, "2.40-2.59"),
    (2.60, 2.80, "2.60-2.79"),
    (2.80, 3.00, "2.80-2.99"),
    (3.00, np.inf, "3.00+"),
]


def make_base(df, threshold):

    x = df[
        df["under_edge_raw"] >= threshold
    ].copy()

    x["won"] = (
        x["actual_total"] < 2.5
    )

    x["profit"] = np.where(
        x["won"],
        x["under_odds"] - 1.0,
        -1.0,
    )

    x["clv_probability"] = (
        x["close_p_under"]
        -
        x["market_p_under"]
    )

    x["beat_close"] = (
        x["under_odds"]
        >
        x["close_under_odds"]
    )

    return x


def summarize(
    x,
    threshold,
    bucket,
):

    if x.empty:
        return None

    bets = len(x)

    profit = float(
        x["profit"].sum()
    )

    return {
        "threshold":
            threshold,

        "odds_bucket":
            bucket,

        "bets":
            bets,

        "wins":
            int(
                x["won"].sum()
            ),

        "win_rate":
            float(
                x["won"].mean()
            ),

        "avg_odds":
            float(
                x["under_odds"].mean()
            ),

        "profit":
            profit,

        "roi":
            profit / bets,

        "avg_clv":
            float(
                x["clv_probability"].mean()
            ),

        "median_clv":
            float(
                x["clv_probability"].median()
            ),

        "beat_close_rate":
            float(
                x["beat_close"].mean()
            ),
    }


def main():

    print()
    print("=" * 120)
    print("V5 TOTALS — PRICE SENSITIVITY")
    print("=" * 120)

    df = build_clv_dataset()

    rows = []

    for threshold in RAW_THRESHOLDS:

        base = make_base(
            df,
            threshold,
        )

        print()
        print("=" * 120)
        print(
            f"RAW UNDER 2.5 >= {threshold:.0%}"
        )
        print("=" * 120)
        print()

        overall = summarize(
            base,
            threshold,
            "ALL",
        )

        if overall:
            rows.append(
                overall
            )

        for low, high, label in ODDS_BUCKETS:

            bucket = base[
                (
                    base["under_odds"] >= low
                )
                &
                (
                    base["under_odds"] < high
                )
            ].copy()

            result = summarize(
                bucket,
                threshold,
                label,
            )

            if result:
                rows.append(
                    result
                )

        out = pd.DataFrame(
            [
                r
                for r in rows
                if r["threshold"] == threshold
            ]
        )

        show = out.copy()

        for col in [
            "win_rate",
            "roi",
            "avg_clv",
            "median_clv",
            "beat_close_rate",
        ]:
            show[col] *= 100.0

        print(
            show.to_string(
                index=False,
                formatters={
                    "threshold":
                        lambda x: f"{x:.0%}",

                    "win_rate":
                        lambda x: f"{x:.2f}%",

                    "avg_odds":
                        lambda x: f"{x:.3f}",

                    "profit":
                        lambda x: f"{x:+.2f}u",

                    "roi":
                        lambda x: f"{x:+.2f}%",

                    "avg_clv":
                        lambda x: f"{x:+.2f}%",

                    "median_clv":
                        lambda x: f"{x:+.2f}%",

                    "beat_close_rate":
                        lambda x: f"{x:.2f}%",
                },
            )
        )

    results = pd.DataFrame(
        rows
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 120)
    print("ROBUSTNESS CHECK")
    print("=" * 120)
    print()

    # Require a minimum sample so tiny buckets
    # do not dominate interpretation.
    useful = results[
        (
            results["odds_bucket"] != "ALL"
        )
        &
        (
            results["bets"] >= 10
        )
    ].copy()

    if useful.empty:

        print(
            "No odds buckets have at least 10 bets."
        )

    else:

        useful[
            "positive_roi"
        ] = (
            useful["roi"] > 0
        )

        useful[
            "positive_clv"
        ] = (
            useful["avg_clv"] > 0
        )

        print(
            useful[
                [
                    "threshold",
                    "odds_bucket",
                    "bets",
                    "roi",
                    "avg_clv",
                    "beat_close_rate",
                    "positive_roi",
                    "positive_clv",
                ]
            ].to_string(
                index=False,
                formatters={
                    "threshold":
                        lambda x: f"{x:.0%}",

                    "roi":
                        lambda x: f"{x:+.2%}",

                    "avg_clv":
                        lambda x: f"{x:+.2%}",

                    "beat_close_rate":
                        lambda x: f"{x:.2%}",
                },
            )
        )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
