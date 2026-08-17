from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_all_leagues_market_oos.csv"
)

TARGETS = {
    "Super Lig": 0.10,
    "Segunda División": 0.04,
}

df = pd.read_csv(FILE, low_memory=False)

for c in [
    "actual_yes",
    "p_raw",
    "p_yes_cal",
    "market_yes_nv",
    "odds_btts_yes",
    "edge_yes",
    "season_num",
]:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce",
    )

df["raw_edge_yes"] = (
    df["p_raw"]
    - df["market_yes_nv"]
)

df["calibration_shift"] = (
    df["p_yes_cal"]
    - df["p_raw"]
)

df["profit_yes"] = np.where(
    df["actual_yes"].eq(1),
    df["odds_btts_yes"] - 1.0,
    -1.0,
)


def summarize(label, g):
    if g.empty:
        print(f"{label:<32} NO BETS")
        return

    print(
        f"{label:<32}"
        f"N={len(g):>4} | "
        f"Raw={g['p_raw'].mean():6.2%} | "
        f"Cal={g['p_yes_cal'].mean():6.2%} | "
        f"Mkt={g['market_yes_nv'].mean():6.2%} | "
        f"RawEdge={g['raw_edge_yes'].mean():6.2%} | "
        f"CalEdge={g['edge_yes'].mean():6.2%} | "
        f"Shift={g['calibration_shift'].mean():+6.2%} | "
        f"Actual={g['actual_yes'].mean():6.2%} | "
        f"ROI={g['profit_yes'].mean():+7.2%}"
    )


print("=" * 130)
print("BTTS RAW vs CALIBRATED EDGE FORENSIC AUDIT")
print("=" * 130)

for league, threshold in TARGETS.items():

    league_df = df[
        df["audit_league"]
        .astype(str)
        .eq(league)
    ].copy()

    qualified = league_df[
        league_df["edge_yes"] >= threshold
    ].copy()

    print()
    print("=" * 130)
    print(
        f"{league} — CURRENT RULE "
        f"CALIBRATED EDGE >= {threshold:.0%}"
    )
    print("=" * 130)

    summarize("ALL OOS", league_df)
    summarize("CURRENT QUALIFIERS", qualified)

    print()
    print("HOW MANY CURRENT QUALIFIERS ALSO CLEAR RAW-EDGE LEVELS?")
    print("-" * 130)

    for raw_cut in [
        0.00,
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
    ]:
        g = qualified[
            qualified["raw_edge_yes"] >= raw_cut
        ]

        pct = (
            len(g) / len(qualified)
            if len(qualified)
            else np.nan
        )

        print(
            f"Raw edge >= {raw_cut:.0%}: "
            f"{len(g):>3}/{len(qualified):<3} "
            f"({pct:.1%})"
        )

    print()
    print("CURRENT QUALIFIERS BY RAW EDGE BUCKET")
    print("-" * 130)

    for lo, hi in [
        (-1.00, 0.00),
        (0.00, 0.02),
        (0.02, 0.04),
        (0.04, 0.06),
        (0.06, 0.08),
        (0.08, 0.10),
        (0.10, np.inf),
    ]:

        if np.isinf(hi):
            g = qualified[
                qualified["raw_edge_yes"] >= lo
            ]
            label = f"{lo:.0%}+"
        else:
            g = qualified[
                (qualified["raw_edge_yes"] >= lo)
                & (qualified["raw_edge_yes"] < hi)
            ]
            label = f"{lo:.0%} to {hi:.0%}"

        summarize(label, g)

    print()
    print("CALIBRATION SHIFT BUCKETS")
    print("-" * 130)

    for lo, hi in [
        (-1.00, 0.00),
        (0.00, 0.02),
        (0.02, 0.04),
        (0.04, 0.06),
        (0.06, 0.08),
        (0.08, np.inf),
    ]:

        if np.isinf(hi):
            g = qualified[
                qualified["calibration_shift"] >= lo
            ]
            label = f"{lo:.0%}+"
        else:
            g = qualified[
                (qualified["calibration_shift"] >= lo)
                & (qualified["calibration_shift"] < hi)
            ]
            label = f"{lo:.0%} to {hi:.0%}"

        summarize(label, g)

    print()
    print("SEASON-BY-SEASON MAPPING")
    print("-" * 130)

    rows = []

    for season, g in qualified.groupby(
        "season_num",
        sort=True,
    ):
        rows.append({
            "season": int(season),
            "bets": len(g),
            "raw": g["p_raw"].mean(),
            "cal": g["p_yes_cal"].mean(),
            "market": g["market_yes_nv"].mean(),
            "raw_edge": g["raw_edge_yes"].mean(),
            "cal_edge": g["edge_yes"].mean(),
            "shift": g["calibration_shift"].mean(),
            "actual": g["actual_yes"].mean(),
            "profit": g["profit_yes"].sum(),
            "roi": g["profit_yes"].mean(),
        })

    season_df = pd.DataFrame(rows)

    print(
        season_df.to_string(
            index=False,
            formatters={
                "raw": lambda x: f"{x:.2%}",
                "cal": lambda x: f"{x:.2%}",
                "market": lambda x: f"{x:.2%}",
                "raw_edge": lambda x: f"{x:.2%}",
                "cal_edge": lambda x: f"{x:.2%}",
                "shift": lambda x: f"{x:+.2%}",
                "actual": lambda x: f"{x:.2%}",
                "profit": lambda x: f"{x:+.2f}u",
                "roi": lambda x: f"{x:+.2%}",
            },
        )
    )

    print()
    print("SAME-MATCH RAW vs CALIBRATED RANK RELATIONSHIP")
    print("-" * 130)

    corr = qualified[
        [
            "raw_edge_yes",
            "edge_yes",
        ]
    ].corr().iloc[0, 1]

    spearman = qualified[
        [
            "raw_edge_yes",
            "edge_yes",
        ]
    ].corr(
        method="spearman"
    ).iloc[0, 1]

    print(
        "Pearson raw-edge vs calibrated-edge:",
        f"{corr:.4f}",
    )

    print(
        "Spearman raw-edge vs calibrated-edge:",
        f"{spearman:.4f}",
    )

    print()
    print("RAW-EDGE SIGN TEST")
    print("-" * 130)

    positive_raw = qualified[
        qualified["raw_edge_yes"] > 0
    ]

    nonpositive_raw = qualified[
        qualified["raw_edge_yes"] <= 0
    ]

    summarize(
        "Raw edge > 0",
        positive_raw,
    )

    summarize(
        "Raw edge <= 0",
        nonpositive_raw,
    )

    print()
    print("=" * 130)

print()
print("FORENSIC AUDIT COMPLETE")
