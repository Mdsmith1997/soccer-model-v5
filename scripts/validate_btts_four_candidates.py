from pathlib import Path
import math
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


ROOT = Path(__file__).resolve().parents[1]

HIST_FILE = (
    ROOT
    / "data"
    / "processed"
    / "footystats_multileague_v5_predictions.csv"
)

LIVE_FILE = (
    ROOT
    / "data"
    / "live"
    / "v5_live_predictions_master.csv"
)

CANDIDATES = [
    "Swiss Super League",
    "MLS",
    "2. Bundesliga",
    "Eliteserien",
]


# ============================================================
# HELPERS
# ============================================================

def p_btts(home_lambda, away_lambda):

    h = float(home_lambda)
    a = float(away_lambda)

    return (
        1.0
        - math.exp(-h)
        - math.exp(-a)
        + math.exp(-(h + a))
    )


def fair_decimal(p):

    if pd.isna(p) or p <= 0:
        return np.nan

    return 1.0 / p


def ece(y, p, bins=10):

    z = pd.DataFrame({
        "y": np.asarray(y, dtype=float),
        "p": np.asarray(p, dtype=float),
    })

    z["bin"] = pd.cut(
        z["p"],
        bins=np.linspace(0, 1, bins + 1),
        include_lowest=True,
    )

    out = 0.0

    for _, g in z.groupby(
        "bin",
        observed=True,
    ):

        if len(g) == 0:
            continue

        out += (
            len(g) / len(z)
            * abs(
                g["p"].mean()
                - g["y"].mean()
            )
        )

    return out


def metrics(y, p):

    y = np.asarray(y, dtype=int)
    p = np.clip(
        np.asarray(p, dtype=float),
        1e-6,
        1 - 1e-6,
    )

    pred = (p >= 0.50).astype(int)

    return {
        "games": len(y),
        "actual_rate": y.mean(),
        "avg_probability": p.mean(),
        "accuracy": (pred == y).mean(),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "ece": ece(y, p),
    }


def print_metrics(label, x):

    print(
        f"{label:<20}"
        f" games={x['games']:>5}"
        f" | actual={x['actual_rate']:.2%}"
        f" | model={x['avg_probability']:.2%}"
        f" | acc={x['accuracy']:.2%}"
        f" | brier={x['brier']:.4f}"
        f" | logloss={x['log_loss']:.4f}"
        f" | ECE={x['ece']:.2%}"
    )


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

df = pd.read_csv(
    HIST_FILE,
    low_memory=False,
)

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
    df["league"].isin(CANDIDATES)
    & df["home_goals"].notna()
    & df["away_goals"].notna()
    & df["home_lambda"].gt(0)
    & df["away_lambda"].gt(0)
].copy()

df["actual_btts"] = (
    (df["home_goals"] > 0)
    & (df["away_goals"] > 0)
).astype(int)

df["p_raw"] = [
    p_btts(h, a)
    for h, a in zip(
        df["home_lambda"],
        df["away_lambda"],
    )
]

df["season_num"] = pd.to_numeric(
    df["season"],
    errors="coerce",
)


print("=" * 125)
print("BTTS — FOUR-CANDIDATE WALK-FORWARD VALIDATION")
print("=" * 125)

print("\nCandidates:")
for x in CANDIDATES:
    print(" ", x)


# ============================================================
# WALK-FORWARD CALIBRATION
#
# For each season:
# calibrator sees ONLY earlier seasons.
# ============================================================

all_oos = []

for league in CANDIDATES:

    g = (
        df[df["league"].eq(league)]
        .sort_values(
            ["season_num", "date"]
        )
        .copy()
    )

    seasons = sorted(
        g["season_num"]
        .dropna()
        .unique()
    )

    print("\n\n" + "=" * 125)
    print(league)
    print("=" * 125)

    print_metrics(
        "RAW FULL",
        metrics(
            g["actual_btts"],
            g["p_raw"],
        )
    )

    league_oos = []

    for season in seasons[1:]:

        train = g[
            g["season_num"] < season
        ].copy()

        test = g[
            g["season_num"].eq(season)
        ].copy()

        if (
            len(train) < 150
            or len(test) < 50
            or train["actual_btts"].nunique() < 2
        ):
            continue

        X_train = (
            train["p_raw"]
            .to_numpy()
            .reshape(-1, 1)
        )

        X_test = (
            test["p_raw"]
            .to_numpy()
            .reshape(-1, 1)
        )

        y_train = (
            train["actual_btts"]
            .to_numpy()
        )

        # ----------------------------------------------------
        # PLATT
        # ----------------------------------------------------

        platt = LogisticRegression(
            solver="lbfgs",
        )

        platt.fit(
            X_train,
            y_train,
        )

        test["p_platt"] = (
            platt.predict_proba(
                X_test
            )[:, 1]
        )

        # ----------------------------------------------------
        # ISOTONIC
        # ----------------------------------------------------

        iso = IsotonicRegression(
            y_min=0.01,
            y_max=0.99,
            out_of_bounds="clip",
        )

        iso.fit(
            train["p_raw"],
            train["actual_btts"],
        )

        test["p_iso"] = iso.predict(
            test["p_raw"]
        )

        league_oos.append(test)

    if not league_oos:
        print("Not enough data for walk-forward.")
        continue

    oos = pd.concat(
        league_oos,
        ignore_index=True,
    )

    all_oos.append(oos)

    print("\nWALK-FORWARD OOS")

    raw_m = metrics(
        oos["actual_btts"],
        oos["p_raw"],
    )

    platt_m = metrics(
        oos["actual_btts"],
        oos["p_platt"],
    )

    iso_m = metrics(
        oos["actual_btts"],
        oos["p_iso"],
    )

    print_metrics(
        "RAW",
        raw_m,
    )

    print_metrics(
        "PLATT",
        platt_m,
    )

    print_metrics(
        "ISOTONIC",
        iso_m,
    )

    # --------------------------------------------------------
    # SEASON RESULTS
    # --------------------------------------------------------

    print("\nBY OOS SEASON")

    rows = []

    for season, s in oos.groupby(
        "season_num"
    ):

        r = metrics(
            s["actual_btts"],
            s["p_raw"],
        )

        p = metrics(
            s["actual_btts"],
            s["p_platt"],
        )

        i = metrics(
            s["actual_btts"],
            s["p_iso"],
        )

        rows.append({
            "season": int(season),
            "games": len(s),
            "actual_btts": s["actual_btts"].mean(),
            "raw_brier": r["brier"],
            "platt_brier": p["brier"],
            "iso_brier": i["brier"],
            "raw_ece": r["ece"],
            "platt_ece": p["ece"],
            "iso_ece": i["ece"],
        })

    season_table = pd.DataFrame(rows)

    print(
        season_table.to_string(
            index=False,
            formatters={
                "actual_btts":
                    lambda x: f"{x:.2%}",
                "raw_brier":
                    lambda x: f"{x:.4f}",
                "platt_brier":
                    lambda x: f"{x:.4f}",
                "iso_brier":
                    lambda x: f"{x:.4f}",
                "raw_ece":
                    lambda x: f"{x:.2%}",
                "platt_ece":
                    lambda x: f"{x:.2%}",
                "iso_ece":
                    lambda x: f"{x:.2%}",
            }
        )
    )

    # --------------------------------------------------------
    # CALIBRATED PROBABILITY BANDS
    # --------------------------------------------------------

    print("\nPLATT PROBABILITY BANDS")

    bins = [
        0.00,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        1.00,
    ]

    oos["band"] = pd.cut(
        oos["p_platt"],
        bins=bins,
        right=False,
    )

    band = (
        oos.groupby(
            "band",
            observed=True,
        )
        .agg(
            games=("actual_btts", "size"),
            avg_probability=("p_platt", "mean"),
            actual_rate=("actual_btts", "mean"),
        )
        .reset_index()
    )

    band["gap"] = (
        band["avg_probability"]
        - band["actual_rate"]
    )

    print(
        band.to_string(
            index=False,
            formatters={
                "avg_probability":
                    lambda x: f"{x:.2%}",
                "actual_rate":
                    lambda x: f"{x:.2%}",
                "gap":
                    lambda x: f"{x:+.2%}",
            }
        )
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n" + "=" * 125)
print("FINAL OOS COMPARISON")
print("=" * 125)

summary = []

if all_oos:

    combined = pd.concat(
        all_oos,
        ignore_index=True,
    )

    for league, g in combined.groupby(
        "league"
    ):

        raw = metrics(
            g["actual_btts"],
            g["p_raw"],
        )

        platt = metrics(
            g["actual_btts"],
            g["p_platt"],
        )

        iso = metrics(
            g["actual_btts"],
            g["p_iso"],
        )

        summary.append({
            "league": league,
            "games": len(g),

            "raw_brier":
                raw["brier"],

            "platt_brier":
                platt["brier"],

            "iso_brier":
                iso["brier"],

            "raw_ece":
                raw["ece"],

            "platt_ece":
                platt["ece"],

            "iso_ece":
                iso["ece"],

            "platt_improvement":
                raw["brier"]
                - platt["brier"],

            "iso_improvement":
                raw["brier"]
                - iso["brier"],
        })

summary = pd.DataFrame(summary)

if len(summary):

    summary = summary.sort_values(
        "platt_brier"
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "raw_brier":
                    lambda x: f"{x:.4f}",
                "platt_brier":
                    lambda x: f"{x:.4f}",
                "iso_brier":
                    lambda x: f"{x:.4f}",

                "raw_ece":
                    lambda x: f"{x:.2%}",
                "platt_ece":
                    lambda x: f"{x:.2%}",
                "iso_ece":
                    lambda x: f"{x:.2%}",

                "platt_improvement":
                    lambda x: f"{x:+.4f}",
                "iso_improvement":
                    lambda x: f"{x:+.4f}",
            }
        )
    )


# ============================================================
# TODAY / UPCOMING LIVE FAIR PRICES
# ============================================================

print("\n\n" + "=" * 125)
print("CURRENT / UPCOMING BTTS FAIR-PRICE BOARD")
print("=" * 125)

if not LIVE_FILE.exists():

    print("No live predictions file.")

else:

    live = pd.read_csv(
        LIVE_FILE,
        low_memory=False,
    )

    live = live[
        live["league"].isin(CANDIDATES)
    ].copy()

    if live.empty:

        print(
            "No current/upcoming fixtures "
            "for the four candidate leagues."
        )

    else:

        # Train final Platt calibrator per league
        # using ALL completed historical data.
        out = []

        for league, games in live.groupby(
            "league"
        ):

            hist = df[
                df["league"].eq(league)
            ].copy()

            if (
                len(hist) < 150
                or hist["actual_btts"].nunique() < 2
            ):
                continue

            platt = LogisticRegression(
                solver="lbfgs"
            )

            platt.fit(
                hist["p_raw"]
                .to_numpy()
                .reshape(-1, 1),

                hist["actual_btts"]
                .to_numpy(),
            )

            for _, r in games.iterrows():

                raw = (
                    r["p_btts_yes_v5"]
                    if "p_btts_yes_v5" in r
                    and pd.notna(
                        r["p_btts_yes_v5"]
                    )
                    else p_btts(
                        r["home_lambda_v5"],
                        r["away_lambda_v5"],
                    )
                )

                calibrated = (
                    platt.predict_proba(
                        np.array(
                            [[float(raw)]]
                        )
                    )[0, 1]
                )

                out.append({
                    "date": r["date"],
                    "league": league,
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],

                    "raw_btts_yes":
                        float(raw),

                    "calibrated_btts_yes":
                        calibrated,

                    "raw_fair_odds":
                        fair_decimal(raw),

                    "calibrated_fair_odds":
                        fair_decimal(
                            calibrated
                        ),

                    "home_history":
                        r.get(
                            "home_history_source",
                            np.nan,
                        ),

                    "away_history":
                        r.get(
                            "away_history_source",
                            np.nan,
                        ),
                })

        board = pd.DataFrame(out)

        if board.empty:

            print("No fair-price rows.")

        else:

            board = board.sort_values(
                [
                    "date",
                    "calibrated_btts_yes",
                ],
                ascending=[
                    True,
                    False,
                ],
            )

            print(
                board.to_string(
                    index=False,
                    formatters={
                        "raw_btts_yes":
                            lambda x: f"{x:.2%}",

                        "calibrated_btts_yes":
                            lambda x: f"{x:.2%}",

                        "raw_fair_odds":
                            lambda x: f"{x:.2f}",

                        "calibrated_fair_odds":
                            lambda x: f"{x:.2f}",
                    }
                )
            )


print("\n" + "=" * 125)
print("IMPORTANT")
print("=" * 125)

print("""
This does NOT create a live BTTS betting strategy.

The historical test is probability validation only because
we do not yet have verified historical BTTS sportsbook prices.

The live board therefore reports CALIBRATED FAIR ODDS.

Example:
calibrated probability = 60%
fair decimal odds = 1.67

A sportsbook price ABOVE 1.67 is better than model fair value,
but we still need a validated betting-edge threshold before
calling it a frozen strategy.
""")
