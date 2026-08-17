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

print("=" * 125)
print("SUPER LIG + SEGUNDA — BTTS SIGNAL INVESTIGATION")
print("=" * 125)
print("Input:", FILE)
print("Rows:", len(df))
print()

print("AVAILABLE COLUMNS")
print("-" * 125)
print("\n".join(df.columns))
print()

numeric = [
    "actual_yes",
    "p_raw",
    "p_yes_cal",
    "market_yes_nv",
    "odds_btts_yes",
    "odds_btts_no",
    "edge_yes",
    "season_num",
    "home_lambda",
    "away_lambda",
]

for c in numeric:
    if c in df.columns:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )


def perf(g):
    if len(g) == 0:
        return None

    profit = np.where(
        g["actual_yes"].eq(1),
        g["odds_btts_yes"] - 1,
        -1.0,
    )

    return {
        "bets": len(g),
        "wins": int(g["actual_yes"].sum()),
        "actual": g["actual_yes"].mean(),
        "model": g["p_yes_cal"].mean(),
        "raw": g["p_raw"].mean(),
        "market": g["market_yes_nv"].mean(),
        "edge": g["edge_yes"].mean(),
        "odds": g["odds_btts_yes"].mean(),
        "profit": profit.sum(),
        "roi": profit.mean(),
    }


def show(label, g):
    p = perf(g)

    if p is None:
        print(f"{label:<30} NO BETS")
        return

    print(
        f"{label:<30} "
        f"N={p['bets']:>4} | "
        f"W={p['wins']:>3} | "
        f"Actual={p['actual']:>6.2%} | "
        f"Raw={p['raw']:>6.2%} | "
        f"Cal={p['model']:>6.2%} | "
        f"Mkt={p['market']:>6.2%} | "
        f"Edge={p['edge']:>6.2%} | "
        f"Odds={p['odds']:.3f} | "
        f"ROI={p['roi']:+7.2%}"
    )


for league, threshold in TARGETS.items():

    league_df = df[
        df["audit_league"]
        .astype(str)
        .eq(league)
    ].copy()

    bets = league_df[
        league_df["edge_yes"] >= threshold
    ].copy()

    print()
    print("=" * 125)
    print(
        f"{league} — BTTS YES >= {threshold:.0%}"
    )
    print("=" * 125)

    show("ALL LEAGUE OOS", league_df)
    show("QUALIFYING", bets)

    # -------------------------------------------------------
    # MODEL vs MARKET DISAGREEMENT
    # -------------------------------------------------------

    print()
    print("MODEL vs MARKET DISAGREEMENT")
    print("-" * 125)

    disagreement_bins = [
        (threshold, threshold + .02),
        (threshold + .02, threshold + .04),
        (threshold + .04, threshold + .06),
        (threshold + .06, np.inf),
    ]

    for lo, hi in disagreement_bins:

        if np.isinf(hi):
            g = bets[
                bets["edge_yes"] >= lo
            ]
            label = f"{lo:.0%}+"
        else:
            g = bets[
                (bets["edge_yes"] >= lo)
                & (bets["edge_yes"] < hi)
            ]
            label = f"{lo:.0%}-{hi:.0%}"

        show(label, g)

    # -------------------------------------------------------
    # RAW MODEL PROBABILITY
    # -------------------------------------------------------

    print()
    print("RAW MODEL PROBABILITY BANDS")
    print("-" * 125)

    for lo, hi in [
        (.35, .45),
        (.45, .50),
        (.50, .55),
        (.55, .60),
        (.60, .65),
        (.65, 1.01),
    ]:
        g = bets[
            (bets["p_raw"] >= lo)
            & (bets["p_raw"] < hi)
        ]

        show(
            f"{lo:.0%}-{hi:.0%}",
            g,
        )

    # -------------------------------------------------------
    # CALIBRATED MODEL PROBABILITY
    # -------------------------------------------------------

    print()
    print("CALIBRATED PROBABILITY BANDS")
    print("-" * 125)

    for lo, hi in [
        (.35, .45),
        (.45, .50),
        (.50, .55),
        (.55, .60),
        (.60, .65),
        (.65, 1.01),
    ]:
        g = bets[
            (bets["p_yes_cal"] >= lo)
            & (bets["p_yes_cal"] < hi)
        ]

        show(
            f"{lo:.0%}-{hi:.0%}",
            g,
        )

    # -------------------------------------------------------
    # MARKET PROBABILITY
    # -------------------------------------------------------

    print()
    print("MARKET PROBABILITY BANDS")
    print("-" * 125)

    for lo, hi in [
        (.30, .35),
        (.35, .40),
        (.40, .45),
        (.45, .50),
        (.50, .55),
        (.55, .60),
    ]:
        g = bets[
            (bets["market_yes_nv"] >= lo)
            & (bets["market_yes_nv"] < hi)
        ]

        show(
            f"{lo:.0%}-{hi:.0%}",
            g,
        )

    # -------------------------------------------------------
    # ODDS
    # -------------------------------------------------------

    print()
    print("ODDS BANDS")
    print("-" * 125)

    for lo, hi in [
        (1.50, 1.75),
        (1.75, 2.00),
        (2.00, 2.25),
        (2.25, 2.50),
        (2.50, 2.75),
        (2.75, 3.01),
    ]:
        g = bets[
            (bets["odds_btts_yes"] >= lo)
            & (bets["odds_btts_yes"] < hi)
        ]

        show(
            f"{lo:.2f}-{hi:.2f}",
            g,
        )

    # -------------------------------------------------------
    # HOME / AWAY TEAM DEPENDENCY
    # -------------------------------------------------------

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

    if home_col:

        print()
        print("HOME TEAM — MINIMUM 3 BETS")
        print("-" * 125)

        rows = []

        for team, g in bets.groupby(home_col):

            if len(g) < 3:
                continue

            p = perf(g)

            rows.append({
                "team": team,
                **p,
            })

        if rows:
            out = (
                pd.DataFrame(rows)
                .sort_values(
                    ["bets", "roi"],
                    ascending=[False, False],
                )
            )

            print(
                out[
                    [
                        "team",
                        "bets",
                        "wins",
                        "actual",
                        "model",
                        "market",
                        "edge",
                        "odds",
                        "roi",
                    ]
                ].to_string(
                    index=False,
                    formatters={
                        "actual": lambda x: f"{x:.2%}",
                        "model": lambda x: f"{x:.2%}",
                        "market": lambda x: f"{x:.2%}",
                        "edge": lambda x: f"{x:.2%}",
                        "odds": lambda x: f"{x:.3f}",
                        "roi": lambda x: f"{x:+.2%}",
                    },
                )
            )

    if away_col:

        print()
        print("AWAY TEAM — MINIMUM 3 BETS")
        print("-" * 125)

        rows = []

        for team, g in bets.groupby(away_col):

            if len(g) < 3:
                continue

            p = perf(g)

            rows.append({
                "team": team,
                **p,
            })

        if rows:
            out = (
                pd.DataFrame(rows)
                .sort_values(
                    ["bets", "roi"],
                    ascending=[False, False],
                )
            )

            print(
                out[
                    [
                        "team",
                        "bets",
                        "wins",
                        "actual",
                        "model",
                        "market",
                        "edge",
                        "odds",
                        "roi",
                    ]
                ].to_string(
                    index=False,
                    formatters={
                        "actual": lambda x: f"{x:.2%}",
                        "model": lambda x: f"{x:.2%}",
                        "market": lambda x: f"{x:.2%}",
                        "edge": lambda x: f"{x:.2%}",
                        "odds": lambda x: f"{x:.3f}",
                        "roi": lambda x: f"{x:+.2%}",
                    },
                )
            )

    # -------------------------------------------------------
    # HOME / AWAY LAMBDA STRUCTURE
    # -------------------------------------------------------

    if {
        "home_lambda",
        "away_lambda",
    }.issubset(bets.columns):

        print()
        print("EXPECTED-GOAL STRUCTURE")
        print("-" * 125)

        bets["_lambda_total"] = (
            bets["home_lambda"]
            + bets["away_lambda"]
        )

        bets["_lambda_min"] = bets[
            [
                "home_lambda",
                "away_lambda",
            ]
        ].min(axis=1)

        for lo, hi in [
            (0.0, 2.25),
            (2.25, 2.50),
            (2.50, 2.75),
            (2.75, 3.00),
            (3.00, np.inf),
        ]:

            if np.isinf(hi):
                g = bets[
                    bets["_lambda_total"] >= lo
                ]
                label = f"total {lo:.2f}+"
            else:
                g = bets[
                    (bets["_lambda_total"] >= lo)
                    & (bets["_lambda_total"] < hi)
                ]
                label = f"total {lo:.2f}-{hi:.2f}"

            show(label, g)

        print()
        print("WEAKER TEAM LAMBDA")
        print("-" * 125)

        for lo, hi in [
            (0.0, .80),
            (.80, 1.00),
            (1.00, 1.20),
            (1.20, 1.40),
            (1.40, np.inf),
        ]:

            if np.isinf(hi):
                g = bets[
                    bets["_lambda_min"] >= lo
                ]
                label = f"min {lo:.2f}+"
            else:
                g = bets[
                    (bets["_lambda_min"] >= lo)
                    & (bets["_lambda_min"] < hi)
                ]
                label = f"min {lo:.2f}-{hi:.2f}"

            show(label, g)

    # -------------------------------------------------------
    # SEASON EVOLUTION
    # -------------------------------------------------------

    print()
    print("SEASON EVOLUTION")
    print("-" * 125)

    for season, g in bets.groupby(
        "season_num",
        sort=True,
    ):
        show(
            str(int(season)),
            g,
        )

    # -------------------------------------------------------
    # WIN vs LOSS CHARACTERISTICS
    # -------------------------------------------------------

    print()
    print("WINNER vs LOSER CHARACTERISTICS")
    print("-" * 125)

    winners = bets[
        bets["actual_yes"].eq(1)
    ]

    losers = bets[
        bets["actual_yes"].eq(0)
    ]

    for label, g in [
        ("WINNERS", winners),
        ("LOSERS", losers),
    ]:

        print()
        print(label)

        for c in [
            "p_raw",
            "p_yes_cal",
            "market_yes_nv",
            "edge_yes",
            "odds_btts_yes",
            "home_lambda",
            "away_lambda",
        ]:
            if c in g.columns:
                print(
                    f"  {c:<20}"
                    f"mean={g[c].mean():.4f} | "
                    f"median={g[c].median():.4f}"
                )

    print()
    print("=" * 125)


print()
print("INVESTIGATION COMPLETE")
