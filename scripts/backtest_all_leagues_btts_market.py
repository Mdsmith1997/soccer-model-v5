from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]

ODDS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_all_leagues_footystats_raw.csv"
)

V5_FILE = (
    ROOT
    / "data"
    / "processed"
    / "footystats_multileague_v5_predictions.csv"
)

OUT_OOS = (
    ROOT
    / "data"
    / "processed"
    / "btts_all_leagues_market_oos.csv"
)

OUT_RESULTS = (
    ROOT
    / "data"
    / "processed"
    / "btts_all_leagues_threshold_results.csv"
)

OUT_STABILITY = (
    ROOT
    / "data"
    / "processed"
    / "btts_all_leagues_season_stability.csv"
)

OUT_AUDIT = (
    ROOT
    / "data"
    / "processed"
    / "btts_all_leagues_match_audit.csv"
)


# ============================================================
# FROZEN CROSS-LEAGUE THRESHOLD GRID
# ============================================================

THRESHOLDS = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.14,
    0.16,
    0.18,
    0.20,
]


def btts_prob(h, a):
    h = float(h)
    a = float(a)

    return (
        1
        - math.exp(-h)
        - math.exp(-a)
        + math.exp(-(h + a))
    )


def season_start(s):
    try:
        x = str(s)

        if "/" in x:
            return int(x.split("/")[0])

        n = int(float(x))

        if n >= 2000:
            return n

        if 1000 <= n <= 9999:
            s4 = f"{n:04d}"

            # Season codes such as 2122, 2223, 2324
            # represent 2021/22, 2022/23, 2023/24.
            first_two = int(s4[:2])

            if 18 <= first_two <= 99:
                return 2000 + first_two

            return n

        return n

    except Exception:
        return np.nan


def settle(b, side, odds_col):
    wins = (
        b["actual_yes"].eq(1)
        if side == "YES"
        else b["actual_yes"].eq(0)
    )

    profit = np.where(
        wins,
        b[odds_col] - 1.0,
        -1.0,
    )

    return wins, profit


print("=" * 130)
print(
    "ALL-LEAGUE BTTS — "
    "EXACT-ID WALK-FORWARD MARKET BACKTEST"
)
print("=" * 130)


# ============================================================
# LOAD
# ============================================================

odds = pd.read_csv(
    ODDS_FILE,
    low_memory=False,
)

v5 = pd.read_csv(
    V5_FILE,
    low_memory=False,
)

print("\nRaw market rows:", len(odds))
print("V5 prediction rows:", len(v5))


# ============================================================
# CLEAN
# ============================================================

odds["id"] = pd.to_numeric(
    odds["id"],
    errors="coerce",
).astype("Int64")

v5["footystats_match_id"] = pd.to_numeric(
    v5["footystats_match_id"],
    errors="coerce",
).astype("Int64")

for c in [
    "odds_btts_yes",
    "odds_btts_no",
]:
    odds[c] = pd.to_numeric(
        odds[c],
        errors="coerce",
    )

for c in [
    "home_goals",
    "away_goals",
    "home_lambda",
    "away_lambda",
]:
    v5[c] = pd.to_numeric(
        v5[c],
        errors="coerce",
    )

odds = odds[
    odds["id"].notna()
    & odds["odds_btts_yes"].gt(1)
    & odds["odds_btts_no"].gt(1)
].copy()

v5 = v5[
    v5["footystats_match_id"].notna()
    & v5["home_goals"].notna()
    & v5["away_goals"].notna()
    & v5["home_lambda"].gt(0)
    & v5["away_lambda"].gt(0)
].copy()


# ============================================================
# LEAGUE AUDIT / EXACT-ID MATCH
# ============================================================

market_leagues = sorted(
    odds["model_league"]
    .dropna()
    .astype(str)
    .unique()
)

print("\nMarket leagues:")
for x in market_leagues:
    print(" ", x)

all_oos = []
audit_rows = []
stability_rows = []
threshold_rows = []


for league in market_leagues:

    print()
    print("=" * 130)
    print("LEAGUE:", league)
    print("=" * 130)

    lo = odds[
        odds["model_league"]
        .astype(str)
        .eq(league)
    ].copy()

    lv = v5[
        v5["league"]
        .astype(str)
        .eq(league)
    ].copy()

    keep = [
        "id",
        "model_season",
        "footystats_season_id",
        "home_name",
        "away_name",
        "odds_btts_yes",
        "odds_btts_no",
    ]

    lo_small = (
        lo[keep]
        .drop_duplicates("id")
        .copy()
    )

    m = lv.merge(
        lo_small,
        left_on="footystats_match_id",
        right_on="id",
        how="inner",
        validate="one_to_one",
    )

    v5_games = len(lv)
    priced_games = len(lo_small)
    matched_games = len(m)

    v5_coverage = (
        matched_games / v5_games
        if v5_games
        else np.nan
    )

    market_coverage = (
        matched_games / priced_games
        if priced_games
        else np.nan
    )

    print("V5 usable games:", v5_games)
    print("Priced market games:", priced_games)
    print("Exact-ID matched:", matched_games)

    print(
        "V5 coverage:",
        f"{v5_coverage:.2%}"
        if pd.notna(v5_coverage)
        else "NA",
    )

    print(
        "Market coverage:",
        f"{market_coverage:.2%}"
        if pd.notna(market_coverage)
        else "NA",
    )

    audit = {
        "league": league,
        "v5_games": v5_games,
        "priced_games": priced_games,
        "matched_games": matched_games,
        "v5_coverage": v5_coverage,
        "market_coverage": market_coverage,
        "matched_seasons": 0,
        "oos_games": 0,
        "oos_seasons": 0,
        "status": "",
    }

    if matched_games == 0:
        audit["status"] = "NO_MATCHES"
        audit_rows.append(audit)
        continue

    # --------------------------------------------------------
    # OUTCOME / RAW BTTS
    # --------------------------------------------------------

    m["actual_yes"] = (
        (m["home_goals"] > 0)
        & (m["away_goals"] > 0)
    ).astype(int)

    m["p_raw"] = [
        btts_prob(h, a)
        for h, a in zip(
            m["home_lambda"],
            m["away_lambda"],
        )
    ]

    m["season_num"] = (
        m["model_season"]
        .map(season_start)
    )

    seasons = sorted(
        m["season_num"]
        .dropna()
        .unique()
    )

    audit["matched_seasons"] = len(seasons)

    print("Matched seasons:", seasons)

    # --------------------------------------------------------
    # WALK-FORWARD PLATT CALIBRATION
    # --------------------------------------------------------

    parts = []

    for season in seasons:

        train = m[
            m["season_num"] < season
        ].copy()

        test = m[
            m["season_num"] == season
        ].copy()

        if len(train) < 150:
            print(
                f"Skipping {season}: "
                f"only {len(train)} prior games"
            )
            continue

        if len(test) < 50:
            print(
                f"Skipping {season}: "
                f"only {len(test)} test games"
            )
            continue

        if train["actual_yes"].nunique() < 2:
            print(
                f"Skipping {season}: "
                "training outcome has one class"
            )
            continue

        model = LogisticRegression(
            solver="lbfgs"
        )

        model.fit(
            train[["p_raw"]],
            train["actual_yes"],
        )

        test["p_yes_cal"] = (
            model.predict_proba(
                test[["p_raw"]]
            )[:, 1]
        )

        print(
            f"{season}: "
            f"train={len(train)} "
            f"test={len(test)}"
        )

        parts.append(test)

    if not parts:
        print(
            "NO VALID WALK-FORWARD OOS SEASONS"
        )

        audit["status"] = (
            "INSUFFICIENT_HISTORY"
        )

        audit_rows.append(audit)
        continue

    oos = pd.concat(
        parts,
        ignore_index=True,
    )

    oos["p_no_cal"] = (
        1 - oos["p_yes_cal"]
    )

    # --------------------------------------------------------
    # NO-VIG MARKET
    # --------------------------------------------------------

    oos["imp_yes"] = (
        1 / oos["odds_btts_yes"]
    )

    oos["imp_no"] = (
        1 / oos["odds_btts_no"]
    )

    oos["overround"] = (
        oos["imp_yes"]
        + oos["imp_no"]
    )

    oos["market_yes_nv"] = (
        oos["imp_yes"]
        / oos["overround"]
    )

    oos["market_no_nv"] = (
        oos["imp_no"]
        / oos["overround"]
    )

    oos["edge_yes"] = (
        oos["p_yes_cal"]
        - oos["market_yes_nv"]
    )

    oos["edge_no"] = (
        oos["p_no_cal"]
        - oos["market_no_nv"]
    )

    oos["audit_league"] = league

    all_oos.append(oos)

    audit["oos_games"] = len(oos)

    audit["oos_seasons"] = (
        oos["season_num"].nunique()
    )

    audit["status"] = "OK"

    audit_rows.append(audit)

    print("OOS market games:", len(oos))

    print(
        "Mean calibrated YES:",
        f"{oos['p_yes_cal'].mean():.2%}",
    )

    print(
        "Actual YES:",
        f"{oos['actual_yes'].mean():.2%}",
    )

    print(
        "Market YES no-vig:",
        f"{oos['market_yes_nv'].mean():.2%}",
    )

    print(
        "Median overround:",
        f"{oos['overround'].median():.3f}",
    )

    # --------------------------------------------------------
    # STANDARDIZED THRESHOLD AUDIT
    # --------------------------------------------------------

    for side in ["YES", "NO"]:

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

        for threshold in THRESHOLDS:

            b = oos[
                oos[edge_col] >= threshold
            ].copy()

            if b.empty:
                continue

            wins, profit = settle(
                b,
                side,
                odds_col,
            )

            threshold_rows.append({
                "league": league,
                "side": side,
                "edge": threshold,
                "bets": len(b),
                "wins": int(wins.sum()),
                "win_rate": wins.mean(),
                "avg_odds": b[odds_col].mean(),
                "avg_edge": b[edge_col].mean(),
                "profit_u": profit.sum(),
                "roi": profit.mean(),
            })

            # -----------------------------------------------
            # SEASON STABILITY AT SAME FROZEN GRID
            # -----------------------------------------------

            for season, g in oos.groupby(
                "season_num"
            ):

                sb = g[
                    g[edge_col] >= threshold
                ].copy()

                if sb.empty:
                    continue

                swins, sprofit = settle(
                    sb,
                    side,
                    odds_col,
                )

                stability_rows.append({
                    "league": league,
                    "season": season,
                    "side": side,
                    "edge": threshold,
                    "bets": len(sb),
                    "wins": int(swins.sum()),
                    "win_rate": swins.mean(),
                    "avg_odds": sb[
                        odds_col
                    ].mean(),
                    "profit_u": (
                        sprofit.sum()
                    ),
                    "roi": sprofit.mean(),
                })


# ============================================================
# SAVE
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)

result_df = pd.DataFrame(
    threshold_rows
)

stability_df = pd.DataFrame(
    stability_rows
)

if all_oos:
    oos_df = pd.concat(
        all_oos,
        ignore_index=True,
        sort=False,
    )

    oos_df.to_csv(
        OUT_OOS,
        index=False,
    )
else:
    oos_df = pd.DataFrame()

audit_df.to_csv(
    OUT_AUDIT,
    index=False,
)

result_df.to_csv(
    OUT_RESULTS,
    index=False,
)

stability_df.to_csv(
    OUT_STABILITY,
    index=False,
)


# ============================================================
# PRINT MATCH / OOS AUDIT
# ============================================================

print()
print("=" * 130)
print("MASTER MATCH / OOS AUDIT")
print("=" * 130)

audit_show = audit_df.copy()

for c in [
    "v5_coverage",
    "market_coverage",
]:
    if c in audit_show.columns:
        audit_show[c] = (
            audit_show[c]
            .map(
                lambda x: (
                    f"{x:.2%}"
                    if pd.notna(x)
                    else "NA"
                )
            )
        )

print(
    audit_show.to_string(
        index=False
    )
)


# ============================================================
# PRINT MASTER THRESHOLD TABLE
# ============================================================

print()
print("=" * 130)
print("MASTER BTTS THRESHOLD RESULTS")
print("=" * 130)

if not result_df.empty:

    show = result_df.copy()

    show["edge"] = (
        show["edge"]
        .map(lambda x: f">={x:.0%}")
    )

    for c in [
        "win_rate",
        "avg_edge",
        "roi",
    ]:
        show[c] = show[c].map(
            lambda x: f"{x:+.2%}"
        )

    show["avg_odds"] = (
        show["avg_odds"]
        .map(lambda x: f"{x:.2f}")
    )

    show["profit_u"] = (
        show["profit_u"]
        .map(lambda x: f"{x:+.2f}")
    )

    print(
        show.to_string(
            index=False
        )
    )


# ============================================================
# COMPACT ROI MATRIX
# ============================================================

print()
print("=" * 130)
print("ROI MATRIX")
print("=" * 130)

if not result_df.empty:

    matrix = result_df.pivot_table(
        index=[
            "league",
            "side",
        ],
        columns="edge",
        values="roi",
        aggfunc="first",
    )

    matrix = matrix.reindex(
        columns=THRESHOLDS
    )

    matrix.columns = [
        f">={x:.0%}"
        for x in matrix.columns
    ]

    formatted = matrix.copy()

    for c in formatted.columns:
        formatted[c] = formatted[c].map(
            lambda x: (
                f"{x:+.1%}"
                if pd.notna(x)
                else "-"
            )
        )

    print(formatted.to_string())


# ============================================================
# BET-COUNT MATRIX
# ============================================================

print()
print("=" * 130)
print("BET COUNT MATRIX")
print("=" * 130)

if not result_df.empty:

    counts = result_df.pivot_table(
        index=[
            "league",
            "side",
        ],
        columns="edge",
        values="bets",
        aggfunc="first",
    )

    counts = counts.reindex(
        columns=THRESHOLDS
    )

    counts.columns = [
        f">={x:.0%}"
        for x in counts.columns
    ]

    print(
        counts.fillna(0)
        .astype(int)
        .to_string()
    )


print()
print("=" * 130)
print("SAVED")
print("=" * 130)

for p in [
    OUT_OOS,
    OUT_RESULTS,
    OUT_STABILITY,
    OUT_AUDIT,
]:
    print(p)
