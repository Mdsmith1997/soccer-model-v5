from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BET_FILE = (
    ROOT
    / "data"
    / "processed"
    / "v5_1x2_all_leagues_frozen16_bets.csv"
)

OUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "v5_1x2_side_candidate_robustness.csv"
)

CANDIDATES = [
    ("League Two", "H"),
    ("Serie A", "A"),
    ("2. Bundesliga", "A"),
    ("Eredivisie", "A"),
]


# ============================================================
# HELPERS
# ============================================================

def basic(g):
    n = len(g)

    if n == 0:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "win_pct": np.nan,
            "avg_odds": np.nan,
            "avg_edge": np.nan,
            "profit": 0.0,
            "roi": np.nan,
        }

    wins = int(g["win"].sum())
    profit = float(g["profit"].sum())

    return {
        "bets": n,
        "wins": wins,
        "losses": n - wins,
        "win_pct": wins / n,
        "avg_odds": g["odds"].mean(),
        "avg_edge": g["raw_edge"].mean(),
        "profit": profit,
        "roi": profit / n,
    }


def max_drawdown(profits):
    profits = np.asarray(profits, dtype=float)

    if len(profits) == 0:
        return np.nan

    equity = np.cumsum(profits)
    peak = np.maximum.accumulate(
        np.r_[0.0, equity]
    )[1:]

    dd = equity - peak

    return float(dd.min())


def longest_losing_streak(wins):
    longest = 0
    current = 0

    for x in wins:
        if int(x) == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def bootstrap_roi(
    profits,
    n_boot=20000,
    seed=42,
):
    profits = np.asarray(profits, dtype=float)

    if len(profits) == 0:
        return {
            "p_roi_gt_0": np.nan,
            "roi_p05": np.nan,
            "roi_p50": np.nan,
            "roi_p95": np.nan,
        }

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(profits),
        size=(n_boot, len(profits)),
    )

    rois = profits[idx].mean(axis=1)

    return {
        "p_roi_gt_0": float(np.mean(rois > 0)),
        "roi_p05": float(np.quantile(rois, 0.05)),
        "roi_p50": float(np.quantile(rois, 0.50)),
        "roi_p95": float(np.quantile(rois, 0.95)),
    }


def remove_best_winner(g):
    winners = g[g["profit"] > 0].copy()

    if winners.empty:
        return {
            "removed_profit": np.nan,
            "remaining_bets": len(g),
            "remaining_profit": g["profit"].sum(),
            "remaining_roi": (
                g["profit"].sum() / len(g)
                if len(g)
                else np.nan
            ),
        }

    idx = winners["profit"].idxmax()

    removed = float(g.loc[idx, "profit"])

    x = g.drop(index=idx)

    return {
        "removed_profit": removed,
        "remaining_bets": len(x),
        "remaining_profit": float(x["profit"].sum()),
        "remaining_roi": (
            float(x["profit"].sum()) / len(x)
            if len(x)
            else np.nan
        ),
    }


def print_stats(label, d):
    print(label)

    for k, v in d.items():

        if isinstance(v, float):

            if "roi" in k or "pct" in k or k.startswith("p_"):
                if pd.isna(v):
                    print(f"  {k}: NA")
                else:
                    print(f"  {k}: {v:+.2%}")

            elif "odds" in k or "edge" in k:
                if pd.isna(v):
                    print(f"  {k}: NA")
                elif "edge" in k:
                    print(f"  {k}: {v:+.2%}")
                else:
                    print(f"  {k}: {v:.3f}")

            else:
                print(f"  {k}: {v:+.2f}")

        else:
            print(f"  {k}: {v}")


# ============================================================
# MAIN
# ============================================================

def main():

    df = pd.read_csv(BET_FILE)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    for c in [
        "odds",
        "model_prob",
        "market_prob",
        "raw_edge",
        "profit",
        "win",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    rows = []

    print("=" * 125)
    print("V5 1X2 — SIDE-SPECIFIC CANDIDATE ROBUSTNESS")
    print("=" * 125)

    print("Frozen rule remains:")
    print("RAW V5 edge >= 16%")
    print("No threshold tuning")
    print("No odds filter")
    print("Flat 1u")
    print()

    for league, selection in CANDIDATES:

        g = df[
            df["league"].eq(league)
            & df["selection"].eq(selection)
        ].copy()

        g = g.sort_values("date").reset_index(drop=True)

        print("\n" + "=" * 125)
        print(f"{league} — {selection}")
        print("=" * 125)

        # ----------------------------------------------------
        # OVERALL
        # ----------------------------------------------------

        overall = basic(g)

        print_stats(
            "\nOVERALL",
            overall,
        )

        # ----------------------------------------------------
        # YEAR-BY-YEAR
        # ----------------------------------------------------

        g["year"] = g["date"].dt.year

        yearly = (
            g.groupby("year")
            .agg(
                bets=("profit", "size"),
                wins=("win", "sum"),
                profit=("profit", "sum"),
            )
        )

        yearly["roi"] = (
            yearly["profit"]
            / yearly["bets"]
        )

        print("\nYEAR-BY-YEAR")
        print(
            yearly.to_string(
                formatters={
                    "profit":
                        lambda x: f"{x:+.2f}",
                    "roi":
                        lambda x: f"{x:+.2%}",
                }
            )
        )

        # ----------------------------------------------------
        # FIRST HALF / SECOND HALF
        # ----------------------------------------------------

        midpoint = len(g) // 2

        first = g.iloc[:midpoint]
        second = g.iloc[midpoint:]

        print_stats(
            "\nFIRST HALF",
            basic(first),
        )

        print_stats(
            "\nSECOND HALF",
            basic(second),
        )

        # ----------------------------------------------------
        # RECENT 25 / 50
        # ----------------------------------------------------

        recent25 = g.tail(25)
        recent50 = g.tail(50)

        print_stats(
            "\nRECENT 25",
            basic(recent25),
        )

        print_stats(
            "\nRECENT 50",
            basic(recent50),
        )

        # ----------------------------------------------------
        # LEAVE-ONE-YEAR-OUT
        # ----------------------------------------------------

        print("\nLEAVE-ONE-YEAR-OUT")

        loo_rows = []

        for year in sorted(g["year"].dropna().unique()):

            x = g[
                ~g["year"].eq(year)
            ]

            p = basic(x)

            loo_rows.append({
                "removed_year": int(year),
                "bets": p["bets"],
                "profit": p["profit"],
                "roi": p["roi"],
            })

        loo = pd.DataFrame(loo_rows)

        print(
            loo.to_string(
                index=False,
                formatters={
                    "profit":
                        lambda x: f"{x:+.2f}",
                    "roi":
                        lambda x: f"{x:+.2%}",
                }
            )
        )

        # ----------------------------------------------------
        # BOOTSTRAP
        # ----------------------------------------------------

        boot = bootstrap_roi(
            g["profit"].values
        )

        print_stats(
            "\nBOOTSTRAP",
            boot,
        )

        # ----------------------------------------------------
        # DRAWDOWN / LOSING STREAK
        # ----------------------------------------------------

        dd = max_drawdown(
            g["profit"].values
        )

        streak = longest_losing_streak(
            g["win"].values
        )

        print("\nRISK")
        print(f"  max_drawdown: {dd:+.2f}u")
        print(f"  longest_losing_streak: {streak}")

        # ----------------------------------------------------
        # REMOVE SINGLE BEST WINNER
        # ----------------------------------------------------

        rbw = remove_best_winner(g)

        print_stats(
            "\nREMOVE SINGLE BEST WINNER",
            rbw,
        )

        # ----------------------------------------------------
        # REMOVE TOP 2 WINNERS
        # ----------------------------------------------------

        winners = g[
            g["profit"] > 0
        ].sort_values(
            "profit",
            ascending=False,
        )

        drop_idx = winners.head(2).index

        x2 = g.drop(index=drop_idx)

        print_stats(
            "\nREMOVE TOP 2 WINNERS",
            {
                "remaining_bets": len(x2),
                "remaining_profit":
                    float(x2["profit"].sum()),
                "remaining_roi":
                    float(x2["profit"].sum()) / len(x2)
                    if len(x2)
                    else np.nan,
            },
        )

        # ----------------------------------------------------
        # ODDS DISTRIBUTION
        # ----------------------------------------------------

        print("\nODDS DISTRIBUTION")
        print(
            g["odds"]
            .describe(
                percentiles=[
                    .10, .25, .50, .75, .90
                ]
            )
            .to_string()
        )

        # ----------------------------------------------------
        # RESULT ROW
        # ----------------------------------------------------

        positive_years = int(
            (yearly["roi"] > 0).sum()
        )

        years = len(yearly)

        all_loo_positive = bool(
            (loo["roi"] > 0).all()
        ) if len(loo) else False

        row = {
            "league": league,
            "selection": selection,
            **overall,
            **boot,
            "positive_years": positive_years,
            "years": years,
            "max_drawdown": dd,
            "longest_losing_streak": streak,
            "first_half_roi": basic(first)["roi"],
            "second_half_roi": basic(second)["roi"],
            "recent25_roi": basic(recent25)["roi"],
            "recent50_roi": basic(recent50)["roi"],
            "remove_best_roi":
                rbw["remaining_roi"],
            "all_loo_positive":
                all_loo_positive,
            "worst_loo_roi":
                loo["roi"].min()
                if len(loo)
                else np.nan,
        }

        rows.append(row)

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    out = pd.DataFrame(rows)

    print("\n\n" + "=" * 125)
    print("FINAL CANDIDATE COMPARISON")
    print("=" * 125)

    print(
        out[
            [
                "league",
                "selection",
                "bets",
                "profit",
                "roi",
                "p_roi_gt_0",
                "roi_p05",
                "roi_p50",
                "roi_p95",
                "positive_years",
                "years",
                "max_drawdown",
                "longest_losing_streak",
                "first_half_roi",
                "second_half_roi",
                "recent25_roi",
                "recent50_roi",
                "remove_best_roi",
                "all_loo_positive",
                "worst_loo_roi",
            ]
        ]
        .sort_values(
            "roi",
            ascending=False,
        )
        .to_string(
            index=False,
            formatters={
                "profit":
                    lambda x: f"{x:+.2f}",
                "roi":
                    lambda x: f"{x:+.2%}",
                "p_roi_gt_0":
                    lambda x: f"{x:.1%}",
                "roi_p05":
                    lambda x: f"{x:+.2%}",
                "roi_p50":
                    lambda x: f"{x:+.2%}",
                "roi_p95":
                    lambda x: f"{x:+.2%}",
                "max_drawdown":
                    lambda x: f"{x:+.2f}",
                "first_half_roi":
                    lambda x: f"{x:+.2%}",
                "second_half_roi":
                    lambda x: f"{x:+.2%}",
                "recent25_roi":
                    lambda x: f"{x:+.2%}",
                "recent50_roi":
                    lambda x: f"{x:+.2%}",
                "remove_best_roi":
                    lambda x: f"{x:+.2%}",
                "worst_loo_roi":
                    lambda x: f"{x:+.2%}",
            },
        )
    )

    OUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUT_FILE,
        index=False,
    )

    print("\nSaved:")
    print(OUT_FILE)


if __name__ == "__main__":
    main()
