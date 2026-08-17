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
    / "v5_1x2_frozen16_structure.csv"
)


# ============================================================
# HELPERS
# ============================================================

def performance(g):

    n = len(g)

    if n == 0:
        return {
            "bets": 0,
            "wins": 0,
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
        "win_pct": wins / n,
        "avg_odds": g["odds"].mean(),
        "avg_edge": g["raw_edge"].mean(),
        "profit": profit,
        "roi": profit / n,
    }


def print_table(df):

    if df.empty:
        print("NO ROWS")
        return

    fmt = {}

    for c in df.columns:

        if c in {
            "win_pct",
            "avg_edge",
            "roi",
        }:
            fmt[c] = lambda x: (
                ""
                if pd.isna(x)
                else f"{x:+.2%}"
            )

        elif c == "avg_odds":
            fmt[c] = lambda x: (
                ""
                if pd.isna(x)
                else f"{x:.3f}"
            )

        elif c == "profit":
            fmt[c] = lambda x: f"{x:+.2f}"

    print(
        df.to_string(
            index=False,
            formatters=fmt,
        )
    )


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

    print("=" * 125)
    print("V5 1X2 FROZEN 16% — STRUCTURAL PERFORMANCE ANALYSIS")
    print("=" * 125)

    print("Rows:", len(df))
    print("Leagues:", df["league"].nunique())

    print("\nFrozen population only:")
    print("RAW V5 edge >= 16%")
    print("No threshold tuning")
    print("No new bets created")
    print("No odds filters applied")

    # ========================================================
    # ODDS BANDS
    # ========================================================

    bins = [
        1.00,
        1.75,
        2.00,
        2.50,
        3.00,
        4.00,
        5.00,
        7.50,
        np.inf,
    ]

    labels = [
        "1.00-1.74",
        "1.75-1.99",
        "2.00-2.49",
        "2.50-2.99",
        "3.00-3.99",
        "4.00-4.99",
        "5.00-7.49",
        "7.50+",
    ]

    df["odds_band"] = pd.cut(
        df["odds"],
        bins=bins,
        labels=labels,
        right=False,
    )

    # ========================================================
    # 1. OVERALL BY SELECTION
    # ========================================================

    print("\n" + "=" * 125)
    print("OVERALL — BY SELECTION")
    print("=" * 125)

    rows = []

    for selection, g in df.groupby("selection"):

        p = performance(g)

        rows.append({
            "selection": selection,
            **p,
        })

    out = pd.DataFrame(rows)

    print_table(
        out.sort_values(
            "roi",
            ascending=False,
        )
    )

    # ========================================================
    # 2. OVERALL BY ODDS BAND
    # ========================================================

    print("\n" + "=" * 125)
    print("OVERALL — BY ODDS BAND")
    print("=" * 125)

    rows = []

    for band, g in df.groupby(
        "odds_band",
        observed=True,
    ):

        p = performance(g)

        rows.append({
            "odds_band": str(band),
            **p,
        })

    out = pd.DataFrame(rows)

    print_table(out)

    # ========================================================
    # 3. SELECTION × ODDS BAND
    # ========================================================

    print("\n" + "=" * 125)
    print("OVERALL — SELECTION × ODDS BAND")
    print("=" * 125)

    rows = []

    for (selection, band), g in df.groupby(
        ["selection", "odds_band"],
        observed=True,
    ):

        p = performance(g)

        rows.append({
            "selection": selection,
            "odds_band": str(band),
            **p,
        })

    cross = pd.DataFrame(rows)

    print_table(
        cross.sort_values(
            ["selection", "odds_band"]
        )
    )

    # ========================================================
    # 4. LEAGUE × SELECTION
    # ========================================================

    print("\n" + "=" * 125)
    print("LEAGUE × SELECTION")
    print("=" * 125)

    league_selection_rows = []

    for (league, selection), g in df.groupby(
        ["league", "selection"]
    ):

        p = performance(g)

        league_selection_rows.append({
            "league": league,
            "selection": selection,
            **p,
        })

    league_selection = pd.DataFrame(
        league_selection_rows
    )

    for league in sorted(df["league"].unique()):

        print(f"\n{league}")
        print("-" * 125)

        x = league_selection[
            league_selection["league"].eq(league)
        ]

        print_table(
            x.sort_values(
                "roi",
                ascending=False,
            )
        )

    # ========================================================
    # 5. LEAGUE × ODDS BAND
    # ========================================================

    print("\n" + "=" * 125)
    print("LEAGUE × ODDS BAND")
    print("=" * 125)

    league_odds_rows = []

    for (league, band), g in df.groupby(
        ["league", "odds_band"],
        observed=True,
    ):

        p = performance(g)

        league_odds_rows.append({
            "league": league,
            "odds_band": str(band),
            **p,
        })

    league_odds = pd.DataFrame(
        league_odds_rows
    )

    for league in sorted(df["league"].unique()):

        print(f"\n{league}")
        print("-" * 125)

        x = league_odds[
            league_odds["league"].eq(league)
        ]

        print_table(x)

    # ========================================================
    # 6. LEAGUE × SELECTION × ODDS
    # Only print cells with >=5 bets
    # ========================================================

    print("\n" + "=" * 125)
    print("LEAGUE × SELECTION × ODDS BAND — MINIMUM 5 BETS")
    print("=" * 125)

    detailed_rows = []

    for (league, selection, band), g in df.groupby(
        [
            "league",
            "selection",
            "odds_band",
        ],
        observed=True,
    ):

        if len(g) < 5:
            continue

        p = performance(g)

        detailed_rows.append({
            "league": league,
            "selection": selection,
            "odds_band": str(band),
            **p,
        })

    detailed = pd.DataFrame(
        detailed_rows
    )

    if not detailed.empty:

        detailed = detailed.sort_values(
            ["roi", "bets"],
            ascending=[False, False],
        )

        print_table(detailed)

    # ========================================================
    # 7. WHERE DO PROFITS COME FROM?
    # ========================================================

    print("\n" + "=" * 125)
    print("PROFIT CONTRIBUTION BY LEAGUE / SELECTION")
    print("=" * 125)

    contrib = (
        df.groupby(
            ["league", "selection"],
            as_index=False,
        )
        .agg(
            bets=("profit", "size"),
            profit=("profit", "sum"),
        )
    )

    contrib["roi"] = (
        contrib["profit"]
        / contrib["bets"]
    )

    print_table(
        contrib.sort_values(
            "profit",
            ascending=False,
        )
    )

    # ========================================================
    # 8. TOP / BOTTOM CELLS
    # Require >=10 bets to avoid tiny-sample nonsense.
    # ========================================================

    print("\n" + "=" * 125)
    print("BEST STRUCTURAL CELLS — MINIMUM 10 BETS")
    print("=" * 125)

    stable = detailed[
        detailed["bets"] >= 10
    ].copy()

    if stable.empty:
        print("No cells with >=10 bets.")
    else:
        print_table(
            stable.sort_values(
                "roi",
                ascending=False,
            ).head(20)
        )

    print("\n" + "=" * 125)
    print("WORST STRUCTURAL CELLS — MINIMUM 10 BETS")
    print("=" * 125)

    if stable.empty:
        print("No cells with >=10 bets.")
    else:
        print_table(
            stable.sort_values(
                "roi",
                ascending=True,
            ).head(20)
        )

    # ========================================================
    # SAVE
    # ========================================================

    detailed.to_csv(
        OUT_FILE,
        index=False,
    )

    print("\nSaved:")
    print(OUT_FILE)


if __name__ == "__main__":
    main()
