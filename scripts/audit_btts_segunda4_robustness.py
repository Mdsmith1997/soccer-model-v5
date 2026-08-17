from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_all_leagues_market_oos.csv"
)

N_BOOT = 20000
SEED = 42

CANDIDATES = {
    ("Segunda División", "YES"): 0.04,
}

df = pd.read_csv(FILE, low_memory=False)

# Find the available chronological column.
date_col = None

for candidate in [
    "date",
    "date_unix",
    "timestamp",
]:
    if candidate in df.columns:
        date_col = candidate
        break

if date_col == "date":
    df["_sort_date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

elif date_col in {
    "date_unix",
    "timestamp",
}:
    df["_sort_date"] = pd.to_datetime(
        df[date_col],
        unit="s",
        errors="coerce",
    )

else:
    # season + existing row order still gives us a deterministic
    # chronological approximation if no date field was retained.
    df["_sort_date"] = pd.NaT

for c in [
    "edge_yes",
    "edge_no",
    "odds_btts_yes",
    "odds_btts_no",
    "actual_yes",
    "season_num",
]:
    if c in df.columns:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )


def build_bets(g, side, threshold):

    edge_col = (
        "edge_yes"
        if side == "YES"
        else "edge_no"
    )

    odds_col = (
        "odds_btts_yes"
        if side == "YES"
        else "odds_btts_no"
    )

    b = g[
        g[edge_col] >= threshold
    ].copy()

    b["edge"] = b[edge_col]
    b["odds"] = b[odds_col]

    b["win"] = (
        b["actual_yes"].eq(1)
        if side == "YES"
        else b["actual_yes"].eq(0)
    ).astype(int)

    b["profit"] = np.where(
        b["win"].eq(1),
        b["odds"] - 1.0,
        -1.0,
    )

    sort_cols = []

    if b["_sort_date"].notna().any():
        sort_cols.append("_sort_date")

    if "season_num" in b.columns:
        if not sort_cols:
            sort_cols.append("season_num")

    if sort_cols:
        b = b.sort_values(
            sort_cols,
            kind="stable",
        )

    return b.reset_index(drop=True)


def perf(g):

    if len(g) == 0:
        return None

    return {
        "bets": len(g),
        "wins": int(g["win"].sum()),
        "profit": float(g["profit"].sum()),
        "roi": float(g["profit"].mean()),
        "avg_odds": float(g["odds"].mean()),
        "avg_edge": float(g["edge"].mean()),
    }


def show(label, g):

    p = perf(g)

    if p is None:
        print(f"{label:<30} NO BETS")
        return

    print(
        f"{label:<30}"
        f"bets={p['bets']:>4} | "
        f"wins={p['wins']:>4} | "
        f"profit={p['profit']:+8.2f}u | "
        f"ROI={p['roi']:+8.2%} | "
        f"odds={p['avg_odds']:.3f} | "
        f"edge={p['avg_edge']:.2%}"
    )


def max_drawdown(profits):

    if len(profits) == 0:
        return np.nan

    cum = np.cumsum(profits)

    # Include starting bankroll/equity of zero.
    equity = np.concatenate(
        [[0.0], cum]
    )

    peak = np.maximum.accumulate(equity)

    dd = equity - peak

    return float(dd.min())


def max_losing_streak(profits):

    best = 0
    current = 0

    for x in profits:

        if x < 0:
            current += 1
            best = max(best, current)

        else:
            current = 0

    return best


summary_rows = []

print("=" * 130)
print("SEGUNDA DIVISIÓN BTTS YES >=4% — HARD ROBUSTNESS AUDIT")
print("=" * 130)

print()
print("Input:", FILE)
print("Rows:", len(df))
print("Chronological source:", date_col)


for (league, side), threshold in CANDIDATES.items():

    base = df[
        df["audit_league"]
        .astype(str)
        .eq(league)
    ].copy()

    bets = build_bets(
        base,
        side,
        threshold,
    )

    print()
    print("=" * 130)
    print(
        f"{league} — BTTS {side} "
        f"EDGE >= {threshold:.0%}"
    )
    print("=" * 130)

    show("FULL SAMPLE", bets)

    if bets.empty:
        continue

    # ========================================================
    # EXACT EDGE BUCKETS
    # ========================================================

    print()
    print("EDGE BUCKETS")
    print("-" * 130)

    for lo, hi in [
        (threshold, threshold + 0.02),
        (threshold + 0.02, threshold + 0.04),
        (threshold + 0.04, threshold + 0.06),
        (threshold + 0.06, threshold + 0.08),
        (threshold + 0.08, np.inf),
    ]:

        if np.isinf(hi):

            g = bets[
                bets["edge"] >= lo
            ]

            label = f"{lo:.0%}+"

        else:

            g = bets[
                (bets["edge"] >= lo)
                & (bets["edge"] < hi)
            ]

            label = (
                f"{lo:.0%}-{hi:.0%}"
            )

        show(label, g)

    # ========================================================
    # ODDS BANDS
    # ========================================================

    print()
    print("ODDS BANDS")
    print("-" * 130)

    for lo, hi in [
        (1.00, 1.75),
        (1.75, 2.00),
        (2.00, 2.25),
        (2.25, 2.50),
        (2.50, 3.00),
        (3.00, np.inf),
    ]:

        if np.isinf(hi):

            g = bets[
                bets["odds"] >= lo
            ]

            label = f"{lo:.2f}+"

        else:

            g = bets[
                (bets["odds"] >= lo)
                & (bets["odds"] < hi)
            ]

            label = (
                f"{lo:.2f}-{hi:.2f}"
            )

        show(label, g)

    # ========================================================
    # SEASONS
    # ========================================================

    print()
    print("SEASON PERFORMANCE")
    print("-" * 130)

    season_stats = []

    for season, g in bets.groupby(
        "season_num"
    ):

        p = perf(g)

        season_stats.append({
            "season": season,
            **p,
        })

        show(
            str(int(season))
            if pd.notna(season)
            else "NA",
            g,
        )

    season_df = pd.DataFrame(
        season_stats
    )

    # ========================================================
    # REMOVE BEST SEASON
    # ========================================================

    print()
    print("REMOVE BEST SEASON")
    print("-" * 130)

    best_season = np.nan
    without_best = bets.iloc[0:0]

    if not season_df.empty:

        best_season = (
            season_df
            .sort_values(
                "profit",
                ascending=False,
            )
            .iloc[0]["season"]
        )

        without_best = bets[
            ~bets["season_num"].eq(
                best_season
            )
        ]

        print(
            "Best season removed:",
            int(best_season),
        )

        show(
            "WITHOUT BEST SEASON",
            without_best,
        )

    # ========================================================
    # CHRONOLOGICAL SPLITS
    # ========================================================

    print()
    print("CHRONOLOGICAL SPLITS")
    print("-" * 130)

    for frac in [
        0.50,
        0.60,
        0.67,
        0.70,
        0.75,
        0.80,
    ]:

        cut = int(
            len(bets) * frac
        )

        early = bets.iloc[:cut]
        holdout = bets.iloc[cut:]

        print()
        print(
            f"{int(frac * 100)} / "
            f"{100 - int(frac * 100)}"
        )

        show("Early sample", early)
        show("Later holdout", holdout)

    # ========================================================
    # RECENT WINDOWS
    # ========================================================

    print()
    print("RECENT WINDOWS")
    print("-" * 130)

    for n in [
        10,
        20,
        30,
        40,
        50,
        75,
        100,
    ]:

        if len(bets) >= n:

            show(
                f"Recent {n}",
                bets.tail(n),
            )

    # ========================================================
    # TEAM DEPENDENCY
    # ========================================================

    print()
    print("TEAM DEPENDENCY")
    print("-" * 130)

    home_col = next(
        (
            c for c in [
                "home_team",
                "home_name",
            ]
            if c in bets.columns
        ),
        None,
    )

    away_col = next(
        (
            c for c in [
                "away_team",
                "away_name",
            ]
            if c in bets.columns
        ),
        None,
    )

    if home_col and away_col:

        teams = pd.concat(
            [
                bets[
                    [
                        home_col,
                        "profit",
                    ]
                ].rename(
                    columns={
                        home_col: "team"
                    }
                ),

                bets[
                    [
                        away_col,
                        "profit",
                    ]
                ].rename(
                    columns={
                        away_col: "team"
                    }
                ),
            ],
            ignore_index=True,
        )

        team_stats = (
            teams
            .groupby(
                "team",
                as_index=False,
            )
            .agg(
                appearances=(
                    "profit",
                    "size",
                ),
                profit=(
                    "profit",
                    "sum",
                ),
            )
            .sort_values(
                "profit",
                ascending=False,
            )
        )

        print(
            team_stats
            .head(10)
            .to_string(
                index=False,
                formatters={
                    "profit":
                        lambda x: f"{x:+.2f}",
                },
            )
        )

        if not team_stats.empty:

            best_team = str(
                team_stats.iloc[0][
                    "team"
                ]
            )

            without_team = bets[
                ~bets[home_col]
                .astype(str)
                .eq(best_team)
                &
                ~bets[away_col]
                .astype(str)
                .eq(best_team)
            ]

            print()
            print(
                "Best-profit team removed:",
                best_team,
            )

            show(
                "WITHOUT BEST TEAM",
                without_team,
            )

    else:

        print(
            "Team columns unavailable — "
            "team dependency skipped."
        )

    # ========================================================
    # WINNER DEPENDENCY
    # ========================================================

    print()
    print("WINNER DEPENDENCY")
    print("-" * 130)

    winners = (
        bets[
            bets["profit"] > 0
        ]
        .sort_values(
            "profit",
            ascending=False,
        )
    )

    show("Original", bets)

    for n in [1, 2, 3, 5]:

        remove_idx = (
            winners
            .head(n)
            .index
        )

        show(
            f"Remove top {n}",
            bets.drop(
                index=remove_idx
            ),
        )

    # ========================================================
    # BOOTSTRAP
    # ========================================================

    print()
    print("BOOTSTRAP")
    print("-" * 130)

    rng = np.random.default_rng(
        SEED
    )

    profits = (
        bets["profit"]
        .to_numpy(
            dtype=float
        )
    )

    n = len(profits)

    # Chunked bootstrap avoids creating one enormous matrix.
    chunk = 1000
    boot_roi = []

    remaining = N_BOOT

    while remaining > 0:

        k = min(
            chunk,
            remaining,
        )

        idx = rng.integers(
            0,
            n,
            size=(k, n),
        )

        samples = profits[idx]

        boot_roi.extend(
            samples.mean(
                axis=1
            )
        )

        remaining -= k

    boot_roi = np.asarray(
        boot_roi
    )

    p_positive = float(
        (boot_roi > 0).mean()
    )

    p5, p50, p95 = np.percentile(
        boot_roi,
        [5, 50, 95],
    )

    print(
        "Iterations:",
        N_BOOT,
    )

    print(
        "Observed ROI:",
        f"{profits.mean():+.2%}",
    )

    print(
        "P(ROI > 0):",
        f"{p_positive:.2%}",
    )

    print(
        "5th percentile:",
        f"{p5:+.2%}",
    )

    print(
        "Median:",
        f"{p50:+.2%}",
    )

    print(
        "95th percentile:",
        f"{p95:+.2%}",
    )

    # ========================================================
    # RISK
    # ========================================================

    print()
    print("RISK")
    print("-" * 130)

    mdd = max_drawdown(
        profits
    )

    losing = max_losing_streak(
        profits
    )

    print(
        "Max drawdown:",
        f"{mdd:+.2f}u",
    )

    print(
        "Max losing streak:",
        losing,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    full = perf(bets)

    without_best_roi = np.nan

    if len(without_best):

        without_best_roi = float(
            without_best[
                "profit"
            ].mean()
        )

    summary_rows.append({
        "league": league,
        "side": side,
        "threshold": threshold,
        "bets": full["bets"],
        "profit": full["profit"],
        "roi": full["roi"],
        "positive_seasons": int(
            (season_df["roi"] > 0)
            .sum()
        )
        if not season_df.empty
        else 0,
        "seasons": len(
            season_df
        ),
        "without_best_season_roi":
            without_best_roi,
        "bootstrap_p_positive":
            p_positive,
        "bootstrap_p5":
            float(p5),
        "max_drawdown":
            mdd,
        "max_losing_streak":
            losing,
    })


print()
print("=" * 130)
print("SEGUNDA >=4% ROBUSTNESS SUMMARY")
print("=" * 130)

summary = pd.DataFrame(
    summary_rows
)

display = summary.copy()

display["threshold"] = (
    display["threshold"]
    .map(
        lambda x: f">={x:.0%}"
    )
)

display["profit"] = (
    display["profit"]
    .map(
        lambda x: f"{x:+.2f}u"
    )
)

for c in [
    "roi",
    "without_best_season_roi",
    "bootstrap_p_positive",
    "bootstrap_p5",
]:

    display[c] = (
        display[c]
        .map(
            lambda x: (
                f"{x:+.2%}"
                if pd.notna(x)
                else "NA"
            )
        )
    )

display["max_drawdown"] = (
    display["max_drawdown"]
    .map(
        lambda x: f"{x:+.2f}u"
    )
)

print(
    display.to_string(
        index=False
    )
)

print()
print("=" * 130)
print("AUDIT COMPLETE — BOARD UNCHANGED")
print("=" * 130)
