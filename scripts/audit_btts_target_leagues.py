from pathlib import Path
import math
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

MULTI = ROOT / "data/processed/footystats_multileague_v5_predictions.csv"
MLS = ROOT / "data/processed/footystats_mls_v5_predictions.csv"


TARGETS = {
    "Championship": [
        "championship",
    ],
    "Eredivisie": [
        "eredivisie",
    ],
    "Primeira Liga": [
        "primeira",
        "portugal",
        "liga portugal",
    ],
    "Super Lig": [
        "super lig",
        "süper lig",
        "turkey",
    ],
}


def btts_prob(h, a):
    h = float(h)
    a = float(a)

    return (
        1
        - math.exp(-h)
        - math.exp(-a)
        + math.exp(-(h + a))
    )


def season_start(x):
    s = str(x)

    # Handles 2018/19, 2018, 1819, etc.
    try:
        if "/" in s:
            return int(s.split("/")[0])

        n = int(float(s))

        if 2000 <= n <= 2100:
            return n

        if 1000 <= n <= 9999:
            # 1819 -> 2018
            first = n // 100
            if first < 50:
                return 2000 + first
            return 1900 + first

    except Exception:
        pass

    return np.nan


def ece(y, p, bins=10):

    y = np.asarray(y)
    p = np.asarray(p)

    edges = np.linspace(0, 1, bins + 1)

    total = len(y)
    score = 0.0

    for lo, hi in zip(edges[:-1], edges[1:]):

        if hi == 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        if not mask.any():
            continue

        score += (
            mask.mean()
            * abs(
                y[mask].mean()
                - p[mask].mean()
            )
        )

    return score


def prepare(df):

    df = df.copy()

    for c in [
        "home_goals",
        "away_goals",
        "home_lambda",
        "away_lambda",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = df[
        df["home_goals"].notna()
        & df["away_goals"].notna()
        & df["home_lambda"].gt(0)
        & df["away_lambda"].gt(0)
    ].copy()

    df["actual_btts"] = (
        df["home_goals"].gt(0)
        & df["away_goals"].gt(0)
    ).astype(int)

    df["p_raw"] = [
        btts_prob(h, a)
        for h, a in zip(
            df["home_lambda"],
            df["away_lambda"],
        )
    ]

    df["season_num"] = (
        df["season"]
        .map(season_start)
    )

    return df


def walk_forward(df):

    seasons = sorted(
        df["season_num"]
        .dropna()
        .unique()
    )

    parts = []

    for season in seasons[1:]:

        train = df[
            df["season_num"] < season
        ].copy()

        test = df[
            df["season_num"] == season
        ].copy()

        if (
            len(train) < 150
            or len(test) < 40
            or train["actual_btts"].nunique() < 2
        ):
            continue

        model = LogisticRegression(
            solver="lbfgs"
        )

        model.fit(
            train[["p_raw"]],
            train["actual_btts"],
        )

        test["p_cal"] = (
            model.predict_proba(
                test[["p_raw"]]
            )[:, 1]
        )

        parts.append(test)

    if not parts:
        return pd.DataFrame()

    return pd.concat(
        parts,
        ignore_index=True,
    )


def audit(name, df):

    df = prepare(df)
    oos = walk_forward(df)

    print("\n" + "=" * 110)
    print(name.upper())
    print("=" * 110)

    print("Historical games:", len(df))
    print(
        "Seasons:",
        sorted(
            df["season_num"]
            .dropna()
            .astype(int)
            .unique()
        )
    )

    if oos.empty:
        print("NO VALID WALK-FORWARD SAMPLE")
        return None

    y = oos["actual_btts"].values
    raw = oos["p_raw"].values
    cal = oos["p_cal"].values

    print("\nOOS games:", len(oos))
    print(f"Actual BTTS rate: {y.mean():.2%}")

    print("\nRAW")
    print(f"Mean probability : {raw.mean():.2%}")
    print(f"Log loss         : {log_loss(y, raw):.4f}")
    print(f"Brier            : {brier_score_loss(y, raw):.4f}")
    print(f"ECE              : {ece(y, raw):.2%}")

    try:
        raw_auc = roc_auc_score(y, raw)
        print(f"AUC              : {raw_auc:.4f}")
    except Exception:
        raw_auc = np.nan

    print("\nCALIBRATED")
    print(f"Mean probability : {cal.mean():.2%}")
    print(f"Log loss         : {log_loss(y, cal):.4f}")
    print(f"Brier            : {brier_score_loss(y, cal):.4f}")
    print(f"ECE              : {ece(y, cal):.2%}")

    try:
        cal_auc = roc_auc_score(y, cal)
        print(f"AUC              : {cal_auc:.4f}")
    except Exception:
        cal_auc = np.nan

    # --------------------------------------------------------
    # Probability buckets
    # --------------------------------------------------------

    print("\nCALIBRATED PROBABILITY BUCKETS")

    buckets = [
        (0.00, 0.50),
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 1.01),
    ]

    rows = []

    for lo, hi in buckets:

        b = oos[
            (oos["p_cal"] >= lo)
            & (oos["p_cal"] < hi)
        ]

        if not len(b):
            continue

        rows.append({
            "bucket": f"{lo:.0%}-{min(hi,1):.0%}",
            "games": len(b),
            "avg_pred": b["p_cal"].mean(),
            "actual": b["actual_btts"].mean(),
            "gap": (
                b["actual_btts"].mean()
                - b["p_cal"].mean()
            ),
        })

    b = pd.DataFrame(rows)

    if len(b):
        for c in [
            "avg_pred",
            "actual",
            "gap",
        ]:
            b[c] = b[c].map(
                lambda x: f"{x:+.2%}"
            )

        print(b.to_string(index=False))

    return {
        "league": name,
        "games": len(df),
        "oos": len(oos),
        "actual": y.mean(),
        "raw_logloss": log_loss(y, raw),
        "cal_logloss": log_loss(y, cal),
        "raw_brier": brier_score_loss(y, raw),
        "cal_brier": brier_score_loss(y, cal),
        "raw_ece": ece(y, raw),
        "cal_ece": ece(y, cal),
        "auc": cal_auc,
    }


# ============================================================
# LOAD MULTILEAGUE
# ============================================================

multi = pd.read_csv(
    MULTI,
    low_memory=False,
)

print("=" * 110)
print("BTTS TARGET LEAGUE PROBABILITY AUDIT")
print("=" * 110)

print("\nAvailable multileague values:")

print(
    multi["league"]
    .astype(str)
    .value_counts()
    .to_string()
)

results = []


# ============================================================
# FOUR MULTILEAGUE TARGETS
# ============================================================

for name, aliases in TARGETS.items():

    mask = pd.Series(
        False,
        index=multi.index,
    )

    s = (
        multi["league"]
        .astype(str)
        .str.lower()
    )

    for alias in aliases:
        mask |= s.str.contains(
            alias,
            regex=False,
            na=False,
        )

    x = multi[mask].copy()

    if x.empty:
        print(
            f"\n{name}: NO MATCHING "
            f"LEAGUE ROWS FOUND"
        )
        continue

    r = audit(
        name,
        x,
    )

    if r:
        results.append(r)


# ============================================================
# MLS
# ============================================================

if MLS.exists():

    mls = pd.read_csv(
        MLS,
        low_memory=False,
    )

    r = audit(
        "MLS",
        mls,
    )

    if r:
        results.append(r)

else:
    print("\nMLS V5 FILE MISSING")


# ============================================================
# LEADERBOARD
# ============================================================

print("\n" + "=" * 110)
print("BTTS PROBABILITY LEADERBOARD")
print("=" * 110)

if results:

    r = pd.DataFrame(results)

    r["logloss_improvement"] = (
        r["raw_logloss"]
        - r["cal_logloss"]
    )

    r["brier_improvement"] = (
        r["raw_brier"]
        - r["cal_brier"]
    )

    r = r.sort_values(
        [
            "cal_brier",
            "cal_logloss",
        ]
    )

    show = r[
        [
            "league",
            "games",
            "oos",
            "actual",
            "cal_logloss",
            "cal_brier",
            "cal_ece",
            "auc",
            "logloss_improvement",
        ]
    ].copy()

    for c in [
        "actual",
        "cal_ece",
    ]:
        show[c] = show[c].map(
            lambda x: f"{x:.2%}"
        )

    for c in [
        "cal_logloss",
        "cal_brier",
        "auc",
        "logloss_improvement",
    ]:
        show[c] = show[c].map(
            lambda x: f"{x:.4f}"
        )

    print(
        show.to_string(
            index=False
        )
    )

print("\n" + "=" * 110)
print("IMPORTANT")
print("=" * 110)

print("""
This audit measures BTTS probability quality only.

It does NOT prove betting profitability.

Promotion still requires:
1. historical BTTS Yes + No prices,
2. no-vig market comparison,
3. OOS edge-threshold ROI,
4. season stability,
5. neighboring-threshold robustness.

The strongest leagues here become the priority
for historical BTTS market-data acquisition.
""")
