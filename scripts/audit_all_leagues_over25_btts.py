from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

V5_FILE = (
    ROOT / "data/processed/"
    "footystats_multileague_v5_predictions.csv"
)

TOTALS_FILE = (
    ROOT / "data/processed/"
    "v5_totals_walkforward_predictions.csv"
)


# ============================================================
# HELPERS
# ============================================================

def poisson_over25(h, a):
    lam = float(h) + float(a)
    return 1.0 - (
        math.exp(-lam)
        * (
            1.0
            + lam
            + (lam ** 2) / 2.0
        )
    )


def poisson_btts_yes(h, a):
    h = float(h)
    a = float(a)

    return (
        1.0
        - math.exp(-h)
        - math.exp(-a)
        + math.exp(-(h + a))
    )


def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return np.mean((p - y) ** 2)


def logloss(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    p = np.clip(
        p,
        1e-12,
        1 - 1e-12,
    )

    return -np.mean(
        y * np.log(p)
        + (1 - y) * np.log(1 - p)
    )


def calibration_error(y, p, bins=10):
    x = pd.DataFrame(
        {
            "y": np.asarray(y, dtype=float),
            "p": np.asarray(p, dtype=float),
        }
    )

    x["bin"] = pd.cut(
        x["p"],
        bins=np.linspace(0, 1, bins + 1),
        include_lowest=True,
    )

    total = len(x)

    if not total:
        return np.nan

    ece = 0.0

    for _, g in x.groupby(
        "bin",
        observed=False,
    ):
        if len(g) == 0:
            continue

        ece += (
            len(g) / total
            * abs(
                g["p"].mean()
                - g["y"].mean()
            )
        )

    return ece


def flat_profit(win, odds):
    return np.where(
        win,
        odds - 1.0,
        -1.0,
    )


def print_table(df):
    if df.empty:
        print("NO DATA")
        return

    print(
        df.to_string(
            index=False
        )
    )


# ============================================================
# PART 1 — OVER 2.5 BETTING AUDIT
# ============================================================

print("=" * 125)
print("ALL-LEAGUE V5 — OVER 2.5 + BTTS AUDIT")
print("=" * 125)

print("\n" + "=" * 125)
print("PART 1 — OVER 2.5 BETTING / MARKET AUDIT")
print("=" * 125)

if not TOTALS_FILE.exists():
    print("Missing:", TOTALS_FILE)

else:
    t = pd.read_csv(
        TOTALS_FILE,
        low_memory=False,
    )

    required = [
        "league",
        "season",
        "over_odds",
        "p_over_raw",
        "market_p_over",
        "over_edge_raw",
        "actual_over",
    ]

    missing = [
        c for c in required
        if c not in t.columns
    ]

    if missing:
        raise ValueError(
            "Totals file missing: "
            + ", ".join(missing)
        )

    for c in [
        "over_odds",
        "p_over_raw",
        "market_p_over",
        "over_edge_raw",
        "actual_over",
    ]:
        t[c] = pd.to_numeric(
            t[c],
            errors="coerce",
        )

    t = t[
        t["over_odds"].gt(1)
        & t["p_over_raw"].between(0, 1)
        & t["market_p_over"].between(0, 1)
        & t["actual_over"].isin([0, 1])
    ].copy()

    print(
        "\nMarket-matched games:",
        len(t),
    )

    print(
        "Leagues:",
        t["league"].nunique(),
    )

    print("\nGames by league:")

    counts = (
        t.groupby("league")
        .size()
        .sort_values(ascending=False)
    )

    print(counts.to_string())


    # --------------------------------------------------------
    # Threshold scan
    # --------------------------------------------------------

    thresholds = [
        0.00,
        0.03,
        0.05,
        0.07,
        0.09,
        0.11,
        0.13,
        0.15,
    ]

    rows = []

    for league, g in t.groupby("league"):

        for threshold in thresholds:

            b = g[
                g["over_edge_raw"]
                >= threshold
            ].copy()

            if len(b) == 0:
                continue

            profit = flat_profit(
                b["actual_over"].eq(1),
                b["over_odds"],
            )

            rows.append(
                {
                    "league": league,
                    "edge": f">={threshold:.0%}",
                    "bets": len(b),
                    "wins": int(
                        b["actual_over"].sum()
                    ),
                    "win_rate": (
                        b["actual_over"].mean()
                    ),
                    "avg_model_p": (
                        b["p_over_raw"].mean()
                    ),
                    "avg_market_p": (
                        b["market_p_over"].mean()
                    ),
                    "avg_odds": (
                        b["over_odds"].mean()
                    ),
                    "profit_u": profit.sum(),
                    "roi": profit.mean(),
                }
            )

    over_scan = pd.DataFrame(rows)

    if len(over_scan):

        over_scan = over_scan.sort_values(
            [
                "edge",
                "roi",
                "bets",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )

        fmt = over_scan.copy()

        for c in [
            "win_rate",
            "avg_model_p",
            "avg_market_p",
            "roi",
        ]:
            fmt[c] = fmt[c].map(
                lambda x: f"{x:+.2%}"
                if c == "roi"
                else f"{x:.2%}"
            )

        for c in [
            "avg_odds",
            "profit_u",
        ]:
            fmt[c] = fmt[c].map(
                lambda x: f"{x:+.2f}"
                if c == "profit_u"
                else f"{x:.2f}"
            )

        print(
            "\n" + "-" * 125
        )
        print(
            "OVER 2.5 — RAW EDGE THRESHOLD SCAN"
        )
        print(
            "-" * 125
        )

        print_table(fmt)


    # --------------------------------------------------------
    # More conservative candidate screen
    # --------------------------------------------------------

    candidate = over_scan[
        (over_scan["bets"] >= 20)
        & (over_scan["roi"] > 0)
    ].copy()

    print(
        "\n" + "-" * 125
    )
    print(
        "OVER 2.5 — POSITIVE ROI WITH >=20 BETS"
    )
    print(
        "-" * 125
    )

    if len(candidate):

        candidate = candidate.sort_values(
            [
                "roi",
                "bets",
            ],
            ascending=[
                False,
                False,
            ],
        )

        x = candidate.copy()

        for c in [
            "win_rate",
            "avg_model_p",
            "avg_market_p",
            "roi",
        ]:
            x[c] = x[c].map(
                lambda z: (
                    f"{z:+.2%}"
                    if c == "roi"
                    else f"{z:.2%}"
                )
            )

        x["avg_odds"] = (
            x["avg_odds"]
            .map(lambda z: f"{z:.2f}")
        )

        x["profit_u"] = (
            x["profit_u"]
            .map(lambda z: f"{z:+.2f}")
        )

        print_table(x)

    else:
        print("NONE")


# ============================================================
# PART 2 — BTTS MODEL QUALITY
# ============================================================

print(
    "\n" + "=" * 125
)
print(
    "PART 2 — BTTS YES MODEL QUALITY"
)
print(
    "=" * 125
)

if not V5_FILE.exists():
    raise FileNotFoundError(V5_FILE)

v = pd.read_csv(
    V5_FILE,
    low_memory=False,
)

required = [
    "league",
    "season",
    "home_goals",
    "away_goals",
    "home_lambda",
    "away_lambda",
    "history_class",
]

missing = [
    c for c in required
    if c not in v.columns
]

if missing:
    raise ValueError(
        "V5 file missing: "
        + ", ".join(missing)
    )

for c in [
    "home_goals",
    "away_goals",
    "home_lambda",
    "away_lambda",
]:
    v[c] = pd.to_numeric(
        v[c],
        errors="coerce",
    )

v = v[
    v["home_goals"].notna()
    & v["away_goals"].notna()
    & v["home_lambda"].gt(0)
    & v["away_lambda"].gt(0)
].copy()

v["actual_btts"] = (
    (v["home_goals"] > 0)
    & (v["away_goals"] > 0)
).astype(int)

v["p_btts_yes"] = [
    poisson_btts_yes(h, a)
    for h, a in zip(
        v["home_lambda"],
        v["away_lambda"],
    )
]

v["actual_over25"] = (
    (
        v["home_goals"]
        + v["away_goals"]
    )
    > 2
).astype(int)

v["p_over25"] = [
    poisson_over25(h, a)
    for h, a in zip(
        v["home_lambda"],
        v["away_lambda"],
    )
]


# ------------------------------------------------------------
# League quality
# ------------------------------------------------------------

rows = []

for league, g in v.groupby("league"):

    n = len(g)

    if n == 0:
        continue

    pred = (
        g["p_btts_yes"]
        >= 0.50
    ).astype(int)

    rows.append(
        {
            "league": league,
            "games": n,
            "actual_btts_rate":
                g["actual_btts"].mean(),
            "avg_model_p":
                g["p_btts_yes"].mean(),
            "accuracy":
                (
                    pred
                    == g["actual_btts"]
                ).mean(),
            "brier":
                brier(
                    g["actual_btts"],
                    g["p_btts_yes"],
                ),
            "log_loss":
                logloss(
                    g["actual_btts"],
                    g["p_btts_yes"],
                ),
            "ece":
                calibration_error(
                    g["actual_btts"],
                    g["p_btts_yes"],
                ),
        }
    )

quality = pd.DataFrame(rows)

quality = quality.sort_values(
    "brier"
)

q = quality.copy()

for c in [
    "actual_btts_rate",
    "avg_model_p",
    "accuracy",
    "ece",
]:
    q[c] = q[c].map(
        lambda x: f"{x:.2%}"
    )

for c in [
    "brier",
    "log_loss",
]:
    q[c] = q[c].map(
        lambda x: f"{x:.4f}"
    )

print(
    "\n" + "-" * 125
)
print(
    "BTTS YES — BY LEAGUE"
)
print(
    "-" * 125
)

print_table(q)


# ------------------------------------------------------------
# SAME-LEAGUE history only
# ------------------------------------------------------------

clean = v[
    v["history_class"]
    .astype(str)
    .eq("BOTH_SAME_LEAGUE")
].copy()

rows = []

for league, g in clean.groupby("league"):

    pred = (
        g["p_btts_yes"]
        >= 0.50
    ).astype(int)

    rows.append(
        {
            "league": league,
            "games": len(g),
            "actual_btts_rate":
                g["actual_btts"].mean(),
            "avg_model_p":
                g["p_btts_yes"].mean(),
            "accuracy":
                (
                    pred
                    == g["actual_btts"]
                ).mean(),
            "brier":
                brier(
                    g["actual_btts"],
                    g["p_btts_yes"],
                ),
            "log_loss":
                logloss(
                    g["actual_btts"],
                    g["p_btts_yes"],
                ),
            "ece":
                calibration_error(
                    g["actual_btts"],
                    g["p_btts_yes"],
                ),
        }
    )

clean_quality = pd.DataFrame(rows)

if len(clean_quality):

    clean_quality = (
        clean_quality
        .sort_values("brier")
    )

    x = clean_quality.copy()

    for c in [
        "actual_btts_rate",
        "avg_model_p",
        "accuracy",
        "ece",
    ]:
        x[c] = x[c].map(
            lambda z: f"{z:.2%}"
        )

    for c in [
        "brier",
        "log_loss",
    ]:
        x[c] = x[c].map(
            lambda z: f"{z:.4f}"
        )

    print(
        "\n" + "-" * 125
    )
    print(
        "BTTS YES — BOTH_SAME_LEAGUE HISTORY ONLY"
    )
    print(
        "-" * 125
    )

    print_table(x)


# ------------------------------------------------------------
# BTTS probability bands
# ------------------------------------------------------------

bins = [
    0.00,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    1.00,
]

v["btts_band"] = pd.cut(
    v["p_btts_yes"],
    bins=bins,
    right=False,
)

band = (
    v.groupby(
        "btts_band",
        observed=True,
    )
    .agg(
        games=("actual_btts", "size"),
        avg_model_p=("p_btts_yes", "mean"),
        actual_rate=("actual_btts", "mean"),
    )
    .reset_index()
)

band["calibration_gap"] = (
    band["avg_model_p"]
    - band["actual_rate"]
)

for c in [
    "avg_model_p",
    "actual_rate",
    "calibration_gap",
]:
    band[c] = band[c].map(
        lambda x: (
            f"{x:+.2%}"
            if c == "calibration_gap"
            else f"{x:.2%}"
        )
    )

print(
    "\n" + "-" * 125
)
print(
    "BTTS YES — PROBABILITY BANDS"
)
print(
    "-" * 125
)

print_table(band)


# ------------------------------------------------------------
# Season stability
# ------------------------------------------------------------

season_rows = []

for (league, season), g in v.groupby(
    ["league", "season"]
):

    if len(g) < 20:
        continue

    season_rows.append(
        {
            "league": league,
            "season": season,
            "games": len(g),
            "actual_btts_rate":
                g["actual_btts"].mean(),
            "avg_model_p":
                g["p_btts_yes"].mean(),
            "brier":
                brier(
                    g["actual_btts"],
                    g["p_btts_yes"],
                ),
            "calibration_gap":
                (
                    g["p_btts_yes"].mean()
                    - g["actual_btts"].mean()
                ),
        }
    )

season_df = pd.DataFrame(
    season_rows
)

if len(season_df):

    season_df = season_df.sort_values(
        [
            "league",
            "season",
        ]
    )

    x = season_df.copy()

    for c in [
        "actual_btts_rate",
        "avg_model_p",
        "calibration_gap",
    ]:
        x[c] = x[c].map(
            lambda z: (
                f"{z:+.2%}"
                if c == "calibration_gap"
                else f"{z:.2%}"
            )
        )

    x["brier"] = (
        x["brier"]
        .map(lambda z: f"{z:.4f}")
    )

    print(
        "\n" + "-" * 125
    )
    print(
        "BTTS YES — BY LEAGUE / SEASON"
    )
    print(
        "-" * 125
    )

    print_table(x)


print(
    "\n" + "=" * 125
)
print("DONE")
print("=" * 125)

print(
    "\nIMPORTANT:"
)
print(
    "OVER 2.5 section is a betting/market audit."
)
print(
    "BTTS section is probability-quality only because "
    "historical BTTS market prices have not yet been verified."
)
