from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from scripts.walkforward_calibrate_v5_totals import (
    build_walkforward_predictions,
)

from scripts.backtest_v5_totals_quick import (
    build_dataset,
    canonical_team,
    season_string,
    LEAGUE_FILES,
    SEASONS,
)


OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "v5_totals_historical_clv.csv"
)


# ============================================================
# CANDIDATE RULES
# ============================================================

RULES = [
    {
        "name": "PLATT_UNDER_8",
        "method": "platt",
        "side": "UNDER",
        "threshold": 0.08,
    },
    {
        "name": "PLATT_UNDER_10",
        "method": "platt",
        "side": "UNDER",
        "threshold": 0.10,
    },
    {
        "name": "RAW_UNDER_11",
        "method": "raw",
        "side": "UNDER",
        "threshold": 0.11,
    },
    {
        "name": "RAW_UNDER_12",
        "method": "raw",
        "side": "UNDER",
        "threshold": 0.12,
    },
]


# ============================================================
# LOAD HISTORICAL CLOSING TOTALS ODDS
# ============================================================

def load_closing_market():

    frames = []

    for season in SEASONS:

        for league, code in LEAGUE_FILES.items():

            path = (
                ROOT
                / "data"
                / "raw"
                / f"{season}_{code}.csv"
            )

            if not path.exists():
                continue

            df = pd.read_csv(
                path,
                low_memory=False,
            )

            required = [
                "Date",
                "HomeTeam",
                "AwayTeam",
            ]

            if any(
                c not in df.columns
                for c in required
            ):
                continue

            # Prefer average closing market.
            # Fall back to Bet365 closing if needed.
            if (
                "AvgC>2.5" in df.columns
                and
                "AvgC<2.5" in df.columns
            ):

                over_col = "AvgC>2.5"
                under_col = "AvgC<2.5"

            elif (
                "B365C>2.5" in df.columns
                and
                "B365C<2.5" in df.columns
            ):

                over_col = "B365C>2.5"
                under_col = "B365C<2.5"

            else:
                continue

            x = df[
                [
                    "Date",
                    "HomeTeam",
                    "AwayTeam",
                    over_col,
                    under_col,
                ]
            ].copy()

            x = x.rename(
                columns={
                    "Date":
                        "close_date",

                    "HomeTeam":
                        "close_home",

                    "AwayTeam":
                        "close_away",

                    over_col:
                        "close_over_odds",

                    under_col:
                        "close_under_odds",
                }
            )

            x["season"] = season
            x["league"] = league

            x["close_date"] = pd.to_datetime(
                x["close_date"],
                dayfirst=True,
                errors="coerce",
            ).dt.date

            x["home_key"] = x[
                "close_home"
            ].map(
                canonical_team
            )

            x["away_key"] = x[
                "close_away"
            ].map(
                canonical_team
            )

            x["close_over_odds"] = pd.to_numeric(
                x["close_over_odds"],
                errors="coerce",
            )

            x["close_under_odds"] = pd.to_numeric(
                x["close_under_odds"],
                errors="coerce",
            )

            x["season"] = season_string(
                x["season"]
            )

            frames.append(
                x
            )

    if not frames:

        raise RuntimeError(
            "No closing O/U 2.5 odds found."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# DE-VIG CLOSING MARKET
# ============================================================

def add_closing_probabilities(df):

    x = df.copy()

    raw_over = (
        1.0
        /
        x["close_over_odds"]
    )

    raw_under = (
        1.0
        /
        x["close_under_odds"]
    )

    total = (
        raw_over
        +
        raw_under
    )

    x["close_p_over"] = (
        raw_over
        /
        total
    )

    x["close_p_under"] = (
        raw_under
        /
        total
    )

    return x


# ============================================================
# BUILD FULL WALK-FORWARD DATASET + CLOSE
# ============================================================

def build_clv_dataset():

    raw = build_dataset()

    wf = build_walkforward_predictions(
        raw
    )

    close = load_closing_market()

    merged = wf.merge(
        close,
        left_on=[
            "season",
            "league",
            "date",
            "home_key",
            "away_key",
        ],
        right_on=[
            "season",
            "league",
            "close_date",
            "home_key",
            "away_key",
        ],
        how="left",
        suffixes=(
            "",
            "_close",
        ),
    )

    merged = merged.dropna(
        subset=[
            "close_over_odds",
            "close_under_odds",
        ]
    ).copy()

    merged = merged[
        (
            merged["close_over_odds"] > 1.0
        )
        &
        (
            merged["close_under_odds"] > 1.0
        )
    ].copy()

    merged = add_closing_probabilities(
        merged
    )

    return merged


# ============================================================
# RULE EVALUATION
# ============================================================

def evaluate_rule(
    df,
    rule,
):

    method = rule["method"]
    side = rule["side"]
    threshold = rule["threshold"]

    side_key = side.lower()

    edge_col = (
        f"{side_key}_edge_{method}"
    )

    odds_col = (
        "under_odds"
        if side == "UNDER"
        else "over_odds"
    )

    close_odds_col = (
        "close_under_odds"
        if side == "UNDER"
        else "close_over_odds"
    )

    entry_market_prob_col = (
        "market_p_under"
        if side == "UNDER"
        else "market_p_over"
    )

    close_market_prob_col = (
        "close_p_under"
        if side == "UNDER"
        else "close_p_over"
    )

    x = df[
        df[edge_col] >= threshold
    ].copy()

    if x.empty:
        return x

    # Probability-space CLV:
    # positive = closing market assigned MORE probability
    # to our selection than entry market did.
    x["clv_probability"] = (
        x[close_market_prob_col]
        -
        x[entry_market_prob_col]
    )

    # Odds-based CLV:
    # positive = our entry price was better than closing price.
    x["beat_close"] = (
        x[odds_col]
        >
        x[close_odds_col]
    )

    x["same_as_close"] = np.isclose(
        x[odds_col],
        x[close_odds_col],
        rtol=0,
        atol=1e-9,
    )

    if side == "UNDER":

        x["won"] = (
            x["actual_total"] < 2.5
        )

    else:

        x["won"] = (
            x["actual_total"] > 2.5
        )

    x["profit"] = np.where(
        x["won"],
        x[odds_col] - 1.0,
        -1.0,
    )

    x["rule"] = rule["name"]
    x["rule_method"] = method.upper()
    x["rule_side"] = side
    x["rule_threshold"] = threshold
    x["entry_odds"] = x[odds_col]
    x["closing_odds"] = x[close_odds_col]

    return x


def summarize_rule(x):

    if x.empty:
        return None

    return {
        "rule":
            x["rule"].iloc[0],

        "method":
            x["rule_method"].iloc[0],

        "side":
            x["rule_side"].iloc[0],

        "threshold":
            x["rule_threshold"].iloc[0],

        "bets":
            len(x),

        "wins":
            int(
                x["won"].sum()
            ),

        "profit":
            float(
                x["profit"].sum()
            ),

        "roi":
            float(
                x["profit"].sum()
                /
                len(x)
            ),

        "avg_entry_odds":
            float(
                x["entry_odds"].mean()
            ),

        "avg_closing_odds":
            float(
                x["closing_odds"].mean()
            ),

        "avg_probability_clv":
            float(
                x["clv_probability"].mean()
            ),

        "median_probability_clv":
            float(
                x["clv_probability"].median()
            ),

        "beat_close_count":
            int(
                x["beat_close"].sum()
            ),

        "beat_close_rate":
            float(
                x["beat_close"].mean()
            ),
    }


# ============================================================
# DISPLAY
# ============================================================

def print_summary(summary):

    print()
    print("=" * 120)
    print("HISTORICAL TOTALS CLV — RULE SUMMARY")
    print("=" * 120)
    print()

    show = summary.copy()

    for col in [
        "roi",
        "avg_probability_clv",
        "median_probability_clv",
        "beat_close_rate",
    ]:

        show[col] *= 100.0

    print(
        show.to_string(
            index=False,
            formatters={
                "threshold":
                    lambda x: f"{x:.0%}",

                "profit":
                    lambda x: f"{x:+.2f}u",

                "roi":
                    lambda x: f"{x:+.2f}%",

                "avg_entry_odds":
                    lambda x: f"{x:.3f}",

                "avg_closing_odds":
                    lambda x: f"{x:.3f}",

                "avg_probability_clv":
                    lambda x: f"{x:+.2f}%",

                "median_probability_clv":
                    lambda x: f"{x:+.2f}%",

                "beat_close_rate":
                    lambda x: f"{x:.2f}%",
            },
        )
    )


def grouped_summary(
    bets,
    group_col,
):

    rows = []

    for (
        rule,
        group,
    ), g in bets.groupby(
        [
            "rule",
            group_col,
        ]
    ):

        rows.append(
            {
                "rule":
                    rule,

                group_col:
                    group,

                "bets":
                    len(g),

                "profit":
                    g["profit"].sum(),

                "roi":
                    (
                        g["profit"].sum()
                        /
                        len(g)
                    ),

                "avg_clv":
                    g[
                        "clv_probability"
                    ].mean(),

                "beat_close_rate":
                    g[
                        "beat_close"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def print_grouped(
    df,
    group_col,
    title,
):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)
    print()

    show = df.copy()

    show["roi"] *= 100.0
    show["avg_clv"] *= 100.0
    show["beat_close_rate"] *= 100.0

    print(
        show.to_string(
            index=False,
            formatters={
                "profit":
                    lambda x: f"{x:+.2f}u",

                "roi":
                    lambda x: f"{x:+.2f}%",

                "avg_clv":
                    lambda x: f"{x:+.2f}%",

                "beat_close_rate":
                    lambda x: f"{x:.2f}%",
            },
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 120)
    print("V5 TOTALS — HISTORICAL CLOSING LINE VALUE")
    print("=" * 120)

    df = build_clv_dataset()

    print()
    print(
        f"Walk-forward matches with closing totals odds: "
        f"{len(df):,}"
    )

    all_bets = []
    summaries = []

    for rule in RULES:

        bets = evaluate_rule(
            df,
            rule,
        )

        if bets.empty:
            continue

        all_bets.append(
            bets
        )

        summaries.append(
            summarize_rule(
                bets
            )
        )

    if not all_bets:

        raise RuntimeError(
            "No candidate totals bets found."
        )

    bets = pd.concat(
        all_bets,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summaries
    )

    print_summary(
        summary
    )

    by_season = grouped_summary(
        bets,
        "test_season",
    )

    by_league = grouped_summary(
        bets,
        "league",
    )

    print_grouped(
        by_season,
        "test_season",
        "CLV BY WALK-FORWARD SEASON",
    )

    print_grouped(
        by_league,
        "league",
        "CLV BY LEAGUE",
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bets.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 120)
    print("INTERPRETATION")
    print("=" * 120)
    print()

    print(
        "Positive probability CLV means the closing "
        "market moved toward our selection."
    )

    print(
        "Beat-close rate above 50% is directionally useful, "
        "but average CLV magnitude matters too."
    )

    print(
        "We want positive ROI AND positive CLV across "
        "multiple seasons/leagues before freezing a rule."
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
