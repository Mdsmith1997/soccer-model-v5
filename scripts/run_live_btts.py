from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import math
import sys

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_FILE = (
    ROOT
    / "data"
    / "processed"
    / "btts_feature_store_v2.csv"
)

ODDS_FILE = (
    ROOT
    / "data"
    / "live"
    / "odds_markets_snapshot.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "live"
    / "btts_live_predictions.csv"
)

LEDGER_FILE = (
    ROOT
    / "data"
    / "live"
    / "btts_prediction_history.csv"
)

LIVE_V5_SCRIPT = (
    ROOT
    / "scripts"
    / "build_live_v5_predictions.py"
)


TRANSFER_STORE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "footystats_multileague_team_pregame_v2.csv"
)


# ============================================================
# FROZEN CFG_0755 DEFINITION
# ============================================================

MODEL_WEIGHT = 0.75
POISSON_WEIGHT = 0.25

LOGISTIC_C = 0.01
LAMBDA_CUT = 1.1


CFG_FEATURES = [
    "poisson_logit",

    "lambda_min",
    "lambda_total",
    "lambda_gap",
    "lambda_balance_ratio",
    "weaker_team_score_probability",

    "xg_matchup_overall_min",
    "xg_matchup_overall_balance",

    "shot_matchup_overall_min",
    "shot_matchup_overall_balance",

    "goal_matchup_overall_min",
    "goal_matchup_overall_balance",

    "league_goal_environment",
    "league_xg_environment",

    "xg_attack_balance",
    "goal_attack_balance",

    "minimum_team_history",

    "lambda_min_above_1_1",
    "lambda_min_below_1_1",
]


# ============================================================
# MULTI-LEAGUE FOOTYSTATS LIVE CFG BRIDGE
# ============================================================

FOOTYSTATS_LIVE_FILE = (
    ROOT
    / "data"
    / "live"
    / "v5_live_predictions_footystats.csv"
)


# ============================================================
# LIVE BETTING POLICY — V1
# ============================================================

# Fraction of full Kelly actually used.
KELLY_FRACTION = 0.25

# Hard cap on bankroll exposure per wager.
MAX_STAKE_PCT = 0.015

# Minimum model-vs-market requirements.
MIN_BET_EV = 0.03
MIN_BET_EDGE = 0.02

# Avoid extremely short / long BTTS prices in V1.
MIN_BET_ODDS = 1.50
MAX_BET_ODDS = 3.50

# Optional bankroll in dollars.
#
# Leave as None to output stake percentages only.
# Example:
# BANKROLL = 1000.0
BANKROLL = None


# ============================================================
# HELPERS
# ============================================================

def banner(title):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def safe_numeric(df, column):
    if column not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def safe_divide(a, b):
    a = pd.to_numeric(
        a,
        errors="coerce",
    )

    b = pd.to_numeric(
        b,
        errors="coerce",
    )

    return np.where(
        np.abs(b) > 1e-9,
        a / b,
        np.nan,
    )


def poisson_btts(h, a):
    h = np.asarray(
        h,
        dtype=float,
    )

    a = np.asarray(
        a,
        dtype=float,
    )

    return (
        1.0
        - np.exp(-h)
        - np.exp(-a)
        + np.exp(-(h + a))
    )


def logit_clip(p):
    p = np.asarray(
        p,
        dtype=float,
    )

    p = np.clip(
        p,
        1e-6,
        1.0 - 1e-6,
    )

    return np.log(
        p / (1.0 - p)
    )


def normalize_team(x):
    if pd.isna(x):
        return ""

    return (
        str(x)
        .strip()
        .lower()
        .replace("&", "and")
    )


def normalize_league(x):
    if pd.isna(x):
        return ""

    return (
        str(x)
        .strip()
        .lower()
    )


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# HISTORICAL CFG FEATURE ENRICHMENT
# ============================================================

def add_cfg_features(df):
    """
    Reproduce CFG_0755 derived features.

    Historical btts_feature_store_v1.csv already contains most
    of these, but this function guarantees the production
    definition is explicit.
    """

    x = df.copy()


    # --------------------------------------------------------
    # Lambdas
    # --------------------------------------------------------

    h = safe_numeric(
        x,
        "home_lambda",
    )

    a = safe_numeric(
        x,
        "away_lambda",
    )

    x["lambda_total"] = h + a

    x["lambda_min"] = np.minimum(
        h,
        a,
    )

    x["lambda_max"] = np.maximum(
        h,
        a,
    )

    x["lambda_gap"] = np.abs(
        h - a
    )

    x["lambda_balance_ratio"] = safe_divide(
        x["lambda_min"],
        x["lambda_max"],
    )


    # --------------------------------------------------------
    # Scoring probabilities
    # --------------------------------------------------------

    x["home_score_probability"] = (
        1.0
        -
        np.exp(-h)
    )

    x["away_score_probability"] = (
        1.0
        -
        np.exp(-a)
    )

    x["weaker_team_score_probability"] = np.minimum(
        x["home_score_probability"],
        x["away_score_probability"],
    )

    x["poisson_btts"] = poisson_btts(
        h,
        a,
    )

    p = pd.to_numeric(
        x["poisson_btts"],
        errors="coerce",
    ).clip(
        0.001,
        0.999,
    )

    x["poisson_logit"] = np.log(
        p / (1.0 - p)
    )


    # --------------------------------------------------------
    # Frozen CFG_0755 lambda hinge
    # --------------------------------------------------------

    x["lambda_min_above_1_1"] = np.maximum(
        x["lambda_min"] - LAMBDA_CUT,
        0.0,
    )

    x["lambda_min_below_1_1"] = np.maximum(
        LAMBDA_CUT - x["lambda_min"],
        0.0,
    )


    # --------------------------------------------------------
    # Matchup features
    # --------------------------------------------------------

    families = [
        (
            "goal",
            "final_goal",
        ),
        (
            "xg",
            "final_xg",
        ),
        (
            "shot",
            "final_shot",
        ),
    ]


    for short, source in families:

        home_attack = safe_numeric(
            x,
            f"home_{source}_attack_overall",
        )

        away_attack = safe_numeric(
            x,
            f"away_{source}_attack_overall",
        )

        home_defense = safe_numeric(
            x,
            f"home_{source}_defense_overall",
        )

        away_defense = safe_numeric(
            x,
            f"away_{source}_defense_overall",
        )


        home_matchup = (
            home_attack
            *
            away_defense
        )

        away_matchup = (
            away_attack
            *
            home_defense
        )


        prefix = (
            f"{short}_matchup_overall"
        )


        x[
            f"home_{prefix}"
        ] = home_matchup

        x[
            f"away_{prefix}"
        ] = away_matchup


        x[
            f"{prefix}_min"
        ] = np.minimum(
            home_matchup,
            away_matchup,
        )

        x[
            f"{prefix}_max"
        ] = np.maximum(
            home_matchup,
            away_matchup,
        )

        x[
            f"{prefix}_gap"
        ] = np.abs(
            home_matchup
            -
            away_matchup
        )

        x[
            f"{prefix}_balance"
        ] = safe_divide(
            x[
                f"{prefix}_min"
            ],
            x[
                f"{prefix}_max"
            ],
        )


    # --------------------------------------------------------
    # Attack symmetry
    # --------------------------------------------------------

    x["goal_attack_balance"] = safe_divide(
        np.minimum(
            safe_numeric(
                x,
                "home_final_goal_attack_overall",
            ),
            safe_numeric(
                x,
                "away_final_goal_attack_overall",
            ),
        ),
        np.maximum(
            safe_numeric(
                x,
                "home_final_goal_attack_overall",
            ),
            safe_numeric(
                x,
                "away_final_goal_attack_overall",
            ),
        ),
    )

    x["xg_attack_balance"] = safe_divide(
        np.minimum(
            safe_numeric(
                x,
                "home_final_xg_attack_overall",
            ),
            safe_numeric(
                x,
                "away_final_xg_attack_overall",
            ),
        ),
        np.maximum(
            safe_numeric(
                x,
                "home_final_xg_attack_overall",
            ),
            safe_numeric(
                x,
                "away_final_xg_attack_overall",
            ),
        ),
    )


    # --------------------------------------------------------
    # League environments
    # --------------------------------------------------------

    x["league_goal_environment"] = (
        safe_numeric(
            x,
            "lg_home_goals",
        )
        +
        safe_numeric(
            x,
            "lg_away_goals",
        )
    )


    x["league_xg_environment"] = (
        safe_numeric(
            x,
            "lg_home_xg",
        )
        +
        safe_numeric(
            x,
            "lg_away_xg",
        )
    )


    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Team history
    #
    # Historical V2:
    #   EPL/Bundesliga adapter maps home_games / away_games
    #   to the historical *_overall_games compatibility fields.
    #
    # Live frozen V5 exposes the underlying prior-only counters
    #   directly as home_games / away_games.
    #
    # Prefer the native live counters when present, while
    # preserving historical V2 compatibility.
    # --------------------------------------------------------

    if (
        "home_games" in x.columns
        and
        "away_games" in x.columns
    ):
        home_history = safe_numeric(
            x,
            "home_games",
        )

        away_history = safe_numeric(
            x,
            "away_games",
        )

    else:
        home_history = safe_numeric(
            x,
            "home_adj_goal_attack_overall_games",
        )

        away_history = safe_numeric(
            x,
            "away_adj_goal_attack_overall_games",
        )

    x["minimum_team_history"] = np.minimum(
        home_history,
        away_history,
    )


    return x


# ============================================================
# CFG MODEL MATRIX
# ============================================================

def build_model():
    """
    Production CFG logistic regression.

    Numeric variables are median-imputed and standardized.
    League is one-hot encoded.
    """

    numeric_features = CFG_FEATURES


    numeric_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )


    categorical_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )


    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipe,
                numeric_features,
            ),
            (
                "league",
                categorical_pipe,
                ["league"],
            ),
        ],
        remainder="drop",
    )


    model = LogisticRegression(
        C=LOGISTIC_C,
        max_iter=3000,
        solver="liblinear",
    )


    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )



# ============================================================
# EXACT LIVE BTTS HISTORICAL STATE
# ============================================================

def attach_live_btts_state(live, historical):
    """
    Attach the historical fields required by CFG_0755 that are
    not directly exposed by the frozen V5 live component frame.

    Team strengths/history:
        latest PREMATCH team state from the FootyStats transfer
        store strictly before the live fixture date.

    League xG:
        latest leakage-safe league home/away xG baseline from
        the historical BTTS store strictly before fixture date.
    """

    if not TRANSFER_STORE_FILE.exists():
        raise FileNotFoundError(
            TRANSFER_STORE_FILE
        )

    x = live.copy()

    transfer = pd.read_csv(
        TRANSFER_STORE_FILE,
        low_memory=False,
    )

    transfer["date"] = pd.to_datetime(
        transfer["date"],
        errors="coerce",
    )

    historical = historical.copy()

    historical["date"] = pd.to_datetime(
        historical["date"],
        errors="coerce",
    )


    # --------------------------------------------------------
    # Determine fixture date
    # --------------------------------------------------------

    fixture_date_col = None

    for candidate in [
        "date",
        "commence_time",
        "kickoff",
    ]:
        if candidate in x.columns:
            fixture_date_col = candidate
            break

    if fixture_date_col is None:
        raise RuntimeError(
            "Cannot determine live fixture date."
        )

    x["_fixture_date"] = pd.to_datetime(
        x[fixture_date_col],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)


    # --------------------------------------------------------
    # Exact transfer-store team state
    # --------------------------------------------------------

    strength_cols = [
        "final_goal_attack_overall",
        "final_goal_defense_overall",
        "final_shot_attack_overall",
        "final_shot_defense_overall",
        "final_xg_attack_overall",
        "final_xg_defense_overall",
        "adj_goal_attack_overall_games",
    ]


    def latest_team_state(
        team,
        venue,
        fixture_date,
    ):

        if pd.isna(fixture_date):
            return None

        q = transfer[
            (
                transfer["team"]
                .astype(str)
                .map(normalize_team)
                ==
                normalize_team(team)
            )
            &
            (
                transfer["venue"]
                .astype(str)
                .str.upper()
                ==
                venue
            )
            &
            (
                transfer["date"]
                <
                fixture_date
            )
        ].copy()

        if q.empty:
            return None

        return (
            q.sort_values("date")
            .iloc[-1]
        )


    for side, venue in [
        ("home", "HOME"),
        ("away", "AWAY"),
    ]:

        records = []

        for _, row in x.iterrows():

            state = latest_team_state(
                row[f"{side}_team"],
                venue,
                row["_fixture_date"],
            )

            if state is None:

                records.append(
                    {
                        c: np.nan
                        for c in strength_cols
                    }
                )

            else:

                records.append(
                    {
                        c: state.get(
                            c,
                            np.nan,
                        )
                        for c in strength_cols
                    }
                )

        state_df = pd.DataFrame(
            records,
            index=x.index,
        )

        for c in strength_cols:

            x[
                f"{side}_{c}"
            ] = pd.to_numeric(
                state_df[c],
                errors="coerce",
            )


    # --------------------------------------------------------
    # League xG environment
    # --------------------------------------------------------

    league_state = (
        historical[
            [
                "date",
                "league",
                "lg_home_xg",
                "lg_away_xg",
            ]
        ]
        .dropna(
            subset=[
                "date",
                "league",
            ]
        )
        .copy()
    )

    league_state["_league_norm"] = (
        league_state["league"]
        .map(normalize_league)
    )


    home_xg = []
    away_xg = []


    for _, row in x.iterrows():

        q = league_state[
            (
                league_state["_league_norm"]
                ==
                normalize_league(
                    row["league"]
                )
            )
            &
            (
                league_state["date"]
                <
                row["_fixture_date"]
            )
        ].copy()

        if q.empty:

            home_xg.append(np.nan)
            away_xg.append(np.nan)

            continue

        latest_date = q["date"].max()

        q = q[
            q["date"]
            ==
            latest_date
        ]

        home_xg.append(
            pd.to_numeric(
                q["lg_home_xg"],
                errors="coerce",
            ).median()
        )

        away_xg.append(
            pd.to_numeric(
                q["lg_away_xg"],
                errors="coerce",
            ).median()
        )


    x["lg_home_xg"] = home_xg
    x["lg_away_xg"] = away_xg


    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("EXACT BTTS STATE ATTACHMENT")
    print("=" * 110)

    audit_cols = [
        "home_final_goal_attack_overall",
        "away_final_goal_attack_overall",

        "home_final_xg_attack_overall",
        "away_final_xg_attack_overall",

        "home_final_shot_attack_overall",
        "away_final_shot_attack_overall",

        "home_adj_goal_attack_overall_games",
        "away_adj_goal_attack_overall_games",

        "lg_home_xg",
        "lg_away_xg",
    ]

    for c in audit_cols:

        print(
            f"{c:45s} "
            f"{int(x[c].notna().sum()):3d}"
            f"/{len(x):3d}"
        )


    return x


# ============================================================
# LIVE BTTS FEATURE ADAPTER
# ============================================================

def build_live_btts_features(
    components,
    predictions,
):
    """
    Convert frozen V5 live components into the exact BTTS
    feature definitions used by the historical feature store.
    """

    comp = components.copy()
    pred = predictions.copy()


    # --------------------------------------------------------
    # Identify match key
    # --------------------------------------------------------

    candidate_keys = [
        "match_id",
        "event_id",
        "fixture_id",
    ]


    join_key = None


    for key in candidate_keys:

        if (
            key in comp.columns
            and
            key in pred.columns
        ):
            join_key = key
            break


    if join_key is not None:

        live = comp.merge(
            pred,
            on=join_key,
            how="inner",
            suffixes=(
                "",
                "_prediction",
            ),
        )

    else:

        keys = [
            c
            for c in [
                "league",
                "home_team",
                "away_team",
            ]
            if (
                c in comp.columns
                and
                c in pred.columns
            )
        ]


        if len(keys) < 3:

            raise RuntimeError(
                "Cannot identify a safe key for joining "
                "live V5 components to predictions."
            )


        live = comp.merge(
            pred,
            on=keys,
            how="inner",
            suffixes=(
                "",
                "_prediction",
            ),
        )


    # --------------------------------------------------------
    # Frozen lambdas
    # --------------------------------------------------------

    lambda_aliases = {
        "home_lambda": [
            "home_lambda",
            "home_lambda_v5",
            "home_lambda_prediction",
            "home_lambda_v5_prediction",
        ],

        "away_lambda": [
            "away_lambda",
            "away_lambda_v5",
            "away_lambda_prediction",
            "away_lambda_v5_prediction",
        ],
    }


    for target, candidates in lambda_aliases.items():

        found = None

        for c in candidates:

            if c in live.columns:

                found = c
                break


        if found is None:

            raise RuntimeError(
                f"Missing live {target}. "
                f"Tried: {candidates}"
            )


        live[target] = pd.to_numeric(
            live[found],
            errors="coerce",
        )


    # --------------------------------------------------------
    # Exact historical BTTS state
    # --------------------------------------------------------

    live = attach_live_btts_state(
        live,
        hist,
    )


    # --------------------------------------------------------
    # LIVE V5 -> HISTORICAL BTTS STRENGTH PARITY
    #
    # These are the actual frozen V5 component values used to
    # construct the current fixture. Do NOT replace them with
    # separately looked-up transfer-store strength rows.
    # --------------------------------------------------------

    strength_aliases = {

        "home_final_goal_attack_overall":
            "home_adj_goal_attack",

        "home_final_goal_defense_overall":
            "home_adj_goal_defense",

        "away_final_goal_attack_overall":
            "away_adj_goal_attack",

        "away_final_goal_defense_overall":
            "away_adj_goal_defense",


        "home_final_shot_attack_overall":
            "home_adj_shot_attack",

        "home_final_shot_defense_overall":
            "home_adj_shot_defense",

        "away_final_shot_attack_overall":
            "away_adj_shot_attack",

        "away_final_shot_defense_overall":
            "away_adj_shot_defense",


        "home_final_xg_attack_overall":
            "home_xg_attack_overall",

        "home_final_xg_defense_overall":
            "home_xg_defense_overall",

        "away_final_xg_attack_overall":
            "away_xg_attack_overall",

        "away_final_xg_defense_overall":
            "away_xg_defense_overall",
    }


    for target, source in strength_aliases.items():

        if source not in live.columns:

            raise RuntimeError(
                f"Missing frozen V5 component: {source}"
            )

        live[target] = pd.to_numeric(
            live[source],
            errors="coerce",
        )


    # --------------------------------------------------------
    # Exact historical BTTS transformations
    # --------------------------------------------------------

    live = add_cfg_features(
        live
    )


    return live



# ============================================================
# FOOTYSTATS LIVE -> CFG_0755
# ============================================================

def build_footystats_live_btts_features(
    footy,
    hist,
):
    """
    Convert the existing live FootyStats V5 output into the
    exact CFG_0755 feature schema.

    The FootyStats scorer already exposes the frozen V5
    goal/xG/shot strengths and lambdas used for each fixture.
    No model is refit and no live strength is approximated here.
    """

    x = footy.copy()

    # --------------------------------------------------------
    # Required identifiers
    # --------------------------------------------------------

    required = [
        "match_id",
        "date",
        "league",
        "home_team",
        "away_team",
        "lg_home_goals",
        "lg_away_goals",
        "home_adj_goal_attack",
        "home_adj_goal_defense",
        "away_adj_goal_attack",
        "away_adj_goal_defense",
        "home_adj_xg_attack",
        "home_adj_xg_defense",
        "away_adj_xg_attack",
        "away_adj_xg_defense",
        "home_adj_shot_attack",
        "home_adj_shot_defense",
        "away_adj_shot_attack",
        "away_adj_shot_defense",
        "home_games",
        "away_games",
        "home_lambda_v5",
        "away_lambda_v5",
    ]

    missing = [
        c
        for c in required
        if c not in x.columns
    ]

    if missing:
        raise RuntimeError(
            "FootyStats CFG bridge missing columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Normalize numerics
    # --------------------------------------------------------

    numeric = [
        c
        for c in required
        if c not in [
            "match_id",
            "date",
            "league",
            "home_team",
            "away_team",
        ]
    ]

    for c in numeric:
        x[c] = pd.to_numeric(
            x[c],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Match the naming expected by the historical CFG builder
    # --------------------------------------------------------

    x["home_xg_attack_overall"] = (
        x["home_adj_xg_attack"]
    )

    x["home_xg_defense_overall"] = (
        x["home_adj_xg_defense"]
    )

    x["away_xg_attack_overall"] = (
        x["away_adj_xg_attack"]
    )

    x["away_xg_defense_overall"] = (
        x["away_adj_xg_defense"]
    )

    # Lambda aliases expected by build_live_btts_features().
    x["home_lambda"] = x["home_lambda_v5"]
    x["away_lambda"] = x["away_lambda_v5"]

    # --------------------------------------------------------
    # Attach exact leakage-safe historical league xG state.
    #
    # This function also attempts team-state attachment, but
    # the native frozen live strengths below deliberately
    # overwrite those strength fields afterward.
    # --------------------------------------------------------

    x = attach_live_btts_state(
        x,
        hist,
    )

    # --------------------------------------------------------
    # Frozen live V5 strength parity
    # --------------------------------------------------------

    aliases = {
        "home_final_goal_attack_overall":
            "home_adj_goal_attack",

        "home_final_goal_defense_overall":
            "home_adj_goal_defense",

        "away_final_goal_attack_overall":
            "away_adj_goal_attack",

        "away_final_goal_defense_overall":
            "away_adj_goal_defense",

        "home_final_shot_attack_overall":
            "home_adj_shot_attack",

        "home_final_shot_defense_overall":
            "home_adj_shot_defense",

        "away_final_shot_attack_overall":
            "away_adj_shot_attack",

        "away_final_shot_defense_overall":
            "away_adj_shot_defense",

        "home_final_xg_attack_overall":
            "home_xg_attack_overall",

        "home_final_xg_defense_overall":
            "home_xg_defense_overall",

        "away_final_xg_attack_overall":
            "away_xg_attack_overall",

        "away_final_xg_defense_overall":
            "away_xg_defense_overall",
    }

    for target, source in aliases.items():

        x[target] = pd.to_numeric(
            x[source],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Exact historical CFG transformations
    # --------------------------------------------------------

    x = add_cfg_features(x)

    return x


# ============================================================
# MARKET
# ============================================================

def build_btts_market(odds):
    x = odds.copy()


    x = x[
        x["market"]
        .astype(str)
        .str.upper()
        .eq("BTTS")
    ].copy()


    if x.empty:
        return x


    x["selection_norm"] = (
        x["selection"]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    x["decimal_odds"] = pd.to_numeric(
        x["decimal_odds"],
        errors="coerce",
    )


    x = x[
        x["selection_norm"]
        .isin(
            [
                "YES",
                "NO",
            ]
        )
    ].copy()


    x = x[
        x["decimal_odds"]
        >
        1.0
    ].copy()


    key_cols = [
        c
        for c in [
            "match_id",
            "league",
            "home_team",
            "away_team",
            "commence_time",
            "bookmaker",
            "bookmaker_key",
        ]
        if c in x.columns
    ]


    wide = (
        x.pivot_table(
            index=key_cols,
            columns="selection_norm",
            values="decimal_odds",
            aggfunc="last",
        )
        .reset_index()
    )


    if (
        "YES" not in wide.columns
        or
        "NO" not in wide.columns
    ):
        return pd.DataFrame()


    wide = wide[
        wide["YES"].notna()
        &
        wide["NO"].notna()
    ].copy()


    wide["yes_implied"] = (
        1.0
        /
        wide["YES"]
    )

    wide["no_implied"] = (
        1.0
        /
        wide["NO"]
    )


    total_implied = (
        wide["yes_implied"]
        +
        wide["no_implied"]
    )


    wide["market_yes"] = (
        wide["yes_implied"]
        /
        total_implied
    )


    wide["market_no"] = (
        wide["no_implied"]
        /
        total_implied
    )


    wide = wide.rename(
        columns={
            "YES":
                "yes_odds",

            "NO":
                "no_odds",
        }
    )


    return wide


# ============================================================
# DYN_100_60_50_40
# ============================================================

def dynamic_probability(
    champion_yes,
    market_yes,
):

    champion_yes = np.asarray(
        champion_yes,
        dtype=float,
    )

    market_yes = np.asarray(
        market_yes,
        dtype=float,
    )


    gap = np.abs(
        champion_yes
        -
        market_yes
    )


    model_weight = np.select(
        [
            gap < 0.02,
            gap < 0.04,
            gap < 0.06,
        ],
        [
            1.00,
            0.60,
            0.50,
        ],
        default=0.40,
    )


    final = (
        model_weight
        *
        champion_yes
        +
        (
            1.0
            -
            model_weight
        )
        *
        market_yes
    )


    return (
        final,
        model_weight,
        gap,
    )


# ============================================================
# MAIN
# ============================================================

banner(
    "LIVE BTTS — CFG_0755 + DYN_100_60_50_40"
)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

if not HISTORICAL_FILE.exists():

    raise FileNotFoundError(
        HISTORICAL_FILE
    )


hist = pd.read_csv(
    HISTORICAL_FILE,
    low_memory=False,
)


hist["date"] = pd.to_datetime(
    hist["date"],
    errors="coerce",
)


hist = hist[
    hist["btts_yes"].notna()
].copy()


hist["btts_yes"] = (
    pd.to_numeric(
        hist["btts_yes"],
        errors="coerce",
    )
    .astype(int)
)


hist = add_cfg_features(
    hist
)


required_model_features = (
    CFG_FEATURES
    +
    [
        "league",
        "btts_yes",
        "poisson_btts",
    ]
)


missing = [
    c
    for c in required_model_features
    if c not in hist.columns
]


if missing:

    raise RuntimeError(
        "Historical CFG features missing: "
        + ", ".join(
            missing
        )
    )


print(
    f"Historical training matches: "
    f"{len(hist):,}"
)

print(
    "Historical through:",
    hist["date"].max(),
)


# ============================================================
# TRAIN FROZEN CFG
# ============================================================

model = build_model()


X_train = hist[
    CFG_FEATURES
    +
    ["league"]
].copy()


y_train = hist[
    "btts_yes"
].copy()


model.fit(
    X_train,
    y_train,
)


print(
    "CFG_0755 production model trained."
)


# ============================================================
# BUILD LIVE V5 COMPONENTS
# ============================================================

if not LIVE_V5_SCRIPT.exists():

    raise FileNotFoundError(
        LIVE_V5_SCRIPT
    )


live_v5 = load_module(
    LIVE_V5_SCRIPT,
    "live_v5",
)


if not hasattr(
    live_v5,
    "build_live_components",
):

    raise RuntimeError(
        "build_live_components() missing from "
        "build_live_v5_predictions.py"
    )


if not hasattr(
    live_v5,
    "build_predictions",
):

    raise RuntimeError(
        "build_predictions() missing from "
        "build_live_v5_predictions.py"
    )


banner(
    "BUILDING LIVE V5 COMPONENTS"
)


# ------------------------------------------------------------
# CORE V5 PROVIDER
#
# EPL/Bundesliga use the core V5 provider. A rolling 72-hour
# window can legitimately contain zero core fixtures. That is
# not a pipeline error; expansion leagues must still continue.
# ------------------------------------------------------------

try:

    fixtures = (
        live_v5
        .load_fixtures()
    )

except ValueError as exc:

    if "No supported core V5 fixtures" not in str(exc):
        raise

    print()
    print(
        "No EPL/Bundesliga fixtures in current 72h window."
    )
    print(
        "Skipping core V5 provider."
    )

    fixtures = pd.DataFrame()


if len(fixtures):

    print(
        "Loaded core live fixtures:",
        len(fixtures),
    )

    components = (
        live_v5
        .build_live_components(
            fixtures
        )
    )

    predictions = (
        live_v5
        .build_predictions(
            fixtures,
            components,
        )
    )

else:

    # Empty placeholders. Expansion-league predictions are
    # appended later in this script.
    components = pd.DataFrame()
    predictions = pd.DataFrame()

    print(
        "Core component rows: 0"
    )

    print(
        "Core prediction rows: 0"
    )


print(
    "Component rows:",
    len(components),
)

print(
    "Prediction rows:",
    len(predictions),
)

print()
print("=" * 110)
print("LIVE COMPONENT SCHEMA")
print("=" * 110)

for i, c in enumerate(components.columns, 1):
    print(
        f"{i:3d}. {c}"
    )

print()
print("=" * 110)
print("LIVE PREDICTION SCHEMA")
print("=" * 110)

for i, c in enumerate(predictions.columns, 1):
    print(
        f"{i:3d}. {c}"
    )


# ============================================================
# LIVE BTTS FEATURES
# ============================================================

if (
    len(components)
    and
    len(predictions)
):

    live_core = build_live_btts_features(
        components,
        predictions,
    )

else:

    print()
    print(
        "No core EPL/Bundesliga BTTS fixtures "
        "in current 72h window."
    )
    print(
        "Skipping core BTTS feature adapter."
    )

    live_core = pd.DataFrame()

# ------------------------------------------------------------
# Expansion-league FootyStats live route
# ------------------------------------------------------------

if not FOOTYSTATS_LIVE_FILE.exists():

    raise FileNotFoundError(
        FOOTYSTATS_LIVE_FILE
    )


footy_live_raw = pd.read_csv(
    FOOTYSTATS_LIVE_FILE,
    low_memory=False,
)


live_footy = build_footystats_live_btts_features(
    footy_live_raw,
    hist,
)


print()
print("=" * 110)
print("MULTI-LEAGUE CFG BRIDGE")
print("=" * 110)

print(
    "Core CFG rows:",
    len(live_core),
)

print(
    "FootyStats CFG rows:",
    len(live_footy),
)


# Keep the union of columns. CFG completeness below remains
# strict and determines which rows are actually scoreable.

live = pd.concat(
    [
        live_core,
        live_footy,
    ],
    ignore_index=True,
    sort=False,
)


# Guard against accidental route overlap.

duplicate_match_ids = (
    live["match_id"]
    .astype(str)
    .duplicated(
        keep=False,
    )
)

if duplicate_match_ids.any():

    dup = (
        live.loc[
            duplicate_match_ids,
            [
                "match_id",
                "league",
                "home_team",
                "away_team",
            ],
        ]
        .sort_values(
            [
                "match_id",
                "league",
            ]
        )
    )

    raise RuntimeError(
        "Core / FootyStats live route overlap detected:\n"
        + dup.to_string(index=False)
    )


print(
    "Combined live CFG rows:",
    len(live),
)


missing_live = [
    c
    for c in (
        CFG_FEATURES
        +
        ["league"]
    )
    if c not in live.columns
]


if missing_live:

    raise RuntimeError(
        "Live BTTS features missing: "
        + ", ".join(
            missing_live
        )
    )


# ============================================================
# STRICT FEATURE COMPLETENESS
# ============================================================

numeric_required = CFG_FEATURES


live[
    "cfg_feature_complete"
] = (
    live[
        numeric_required
    ]
    .notna()
    .all(axis=1)
)


print()
print(
    "Live fixtures:",
    len(live),
)

print(
    "CFG feature-complete:",
    int(
        live[
            "cfg_feature_complete"
        ].sum()
    ),
)


# ============================================================
# CFG LIVE FEATURE DIAGNOSTIC
# ============================================================

print()
print("=" * 110)
print("CFG LIVE FEATURE DIAGNOSTIC")
print("=" * 110)

diagnostic_rows = []

for c in CFG_FEATURES:

    if c not in live.columns:

        diagnostic_rows.append(
            {
                "feature": c,
                "exists": False,
                "non_null": 0,
                "missing": len(live),
                "missing_pct": 100.0,
            }
        )

        continue

    z = pd.to_numeric(
        live[c],
        errors="coerce",
    )

    non_null = int(
        z.notna().sum()
    )

    missing = int(
        z.isna().sum()
    )

    diagnostic_rows.append(
        {
            "feature": c,
            "exists": True,
            "non_null": non_null,
            "missing": missing,
            "missing_pct":
                100.0 * missing / len(live),
        }
    )


diagnostic = pd.DataFrame(
    diagnostic_rows
)

print(
    diagnostic.to_string(
        index=False
    )
)


print()
print("-" * 110)
print("MISSING CFG FEATURES BY FIXTURE")
print("-" * 110)

for _, row in live.iterrows():

    missing_features = [
        c
        for c in CFG_FEATURES
        if (
            c not in live.columns
            or
            pd.isna(
                pd.to_numeric(
                    pd.Series(
                        [row.get(c)]
                    ),
                    errors="coerce",
                ).iloc[0]
            )
        )
    ]

    print(
        f"{row.get('league', '')} | "
        f"{row.get('home_team', '')} vs "
        f"{row.get('away_team', '')} | "
        f"missing={len(missing_features)} | "
        +
        (
            ", ".join(
                missing_features
            )
            if missing_features
            else "COMPLETE"
        )
    )


score = live[
    live[
        "cfg_feature_complete"
    ]
].copy()


if score.empty:

    banner(
        "NO CFG-ELIGIBLE LIVE FIXTURES"
    )

    print(
        "No live fixture currently has the complete "
        "historical CFG_0755 feature set."
    )

    sys.exit(0)


# ============================================================
# CFG PROBABILITIES
# ============================================================

X_live = score[
    CFG_FEATURES
    +
    ["league"]
].copy()


score[
    "cfg0755_model_probability"
] = (
    model.predict_proba(
        X_live
    )[:, 1]
)


score[
    "cfg0755_probability"
] = (
    MODEL_WEIGHT
    *
    score[
        "cfg0755_model_probability"
    ]
    +
    POISSON_WEIGHT
    *
    score[
        "poisson_btts"
    ]
)


score[
    "champion_yes"
] = score[
    "cfg0755_probability"
]


# ============================================================
# LOAD LIVE BTTS MARKET
# ============================================================

if not ODDS_FILE.exists():

    raise FileNotFoundError(
        ODDS_FILE
    )


# The odds fetcher can legitimately produce an empty market snapshot
# when no totals/BTTS prices are returned. An empty file must not crash
# the live pipeline.
if ODDS_FILE.stat().st_size <= 1:
    print(
        "BTTS market snapshot is empty — no market prices returned."
    )
    odds = pd.DataFrame()
else:
    try:
        odds = pd.read_csv(
            ODDS_FILE,
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        print(
            "BTTS market snapshot contains no readable rows."
        )
        odds = pd.DataFrame()


if odds.empty:
    market = pd.DataFrame()
else:
    market = build_btts_market(
        odds
    )


banner(
    "LIVE BTTS MARKET"
)


print(
    "Complete YES/NO bookmaker pairs:",
    len(market),
)


if market.empty:

    print(
        "No complete BTTS YES/NO market pairs available."
    )

    sys.exit(0)


# ============================================================
# MATCH MODEL TO MARKET
# ============================================================

score["home_norm"] = (
    score["home_team"]
    .map(
        normalize_team
    )
)

score["away_norm"] = (
    score["away_team"]
    .map(
        normalize_team
    )
)

score["league_norm"] = (
    score["league"]
    .map(
        normalize_league
    )
)


market["home_norm"] = (
    market["home_team"]
    .map(
        normalize_team
    )
)

market["away_norm"] = (
    market["away_team"]
    .map(
        normalize_team
    )
)

market["league_norm"] = (
    market["league"]
    .map(
        normalize_league
    )
)


final = market.merge(
    score,
    on=[
        "home_norm",
        "away_norm",
        "league_norm",
    ],
    how="inner",
    suffixes=(
        "_market",
        "_model",
    ),
)


# ------------------------------------------------------------
# RESTORE CANONICAL MATCH ID AFTER MODEL / MARKET MERGE
# ------------------------------------------------------------

if "match_id_model" in final.columns:

    final["match_id"] = (
        final["match_id_model"]
        .astype(str)
    )

elif "match_id" not in final.columns:

    if "match_id_market" in final.columns:

        final["match_id"] = (
            final["match_id_market"]
            .astype(str)
        )

    else:

        raise RuntimeError(
            "No match_id available after model / market merge."
        )


banner(
    "MODEL / MARKET MATCHING"
)


print(
    "Matched bookmaker rows:",
    len(final),
)


if final.empty:

    print(
        "No live BTTS market currently matches a "
        "CFG-eligible fixture."
    )

    sys.exit(0)


# ============================================================
# DYNAMIC CALIBRATION
# ============================================================

(
    final["final_yes_probability"],
    final["dynamic_model_weight"],
    final["model_market_gap"],
) = dynamic_probability(
    final["champion_yes"],
    final["market_yes"],
)


# ============================================================
# EDGE / EV
# ============================================================

final["yes_edge"] = (
    final["final_yes_probability"]
    -
    final["market_yes"]
)


final["yes_ev"] = (
    final["final_yes_probability"]
    *
    final["yes_odds"]
    -
    1.0
)


final["no_probability"] = (
    1.0
    -
    final["final_yes_probability"]
)


final["no_edge"] = (
    final["no_probability"]
    -
    final["market_no"]
)


final["no_ev"] = (
    final["no_probability"]
    *
    final["no_odds"]
    -
    1.0
)


final["best_side"] = np.where(
    final["yes_ev"]
    >=
    final["no_ev"],
    "YES",
    "NO",
)


final["best_ev"] = np.maximum(
    final["yes_ev"],
    final["no_ev"],
)


# ============================================================
# LIVE BET SELECTION / KELLY SIZING
# ============================================================

# ------------------------------------------------------------
# Put the chosen side onto one common probability / odds axis.
# ------------------------------------------------------------

final["best_probability"] = np.where(
    final["best_side"].eq("YES"),
    final["final_yes_probability"],
    final["no_probability"],
)

final["best_odds"] = np.where(
    final["best_side"].eq("YES"),
    final["yes_odds"],
    final["no_odds"],
)

final["best_edge_selected"] = np.where(
    final["best_side"].eq("YES"),
    final["yes_edge"],
    final["no_edge"],
)

# ------------------------------------------------------------
# FULL KELLY
#
# Decimal odds:
#
#     b = odds - 1
#     f* = (b*p - q) / b
#
# where q = 1 - p.
# ------------------------------------------------------------

b = (
    final["best_odds"]
    -
    1.0
)

p_win = final[
    "best_probability"
]

q_lose = (
    1.0
    -
    p_win
)

final["kelly_full"] = np.where(
    b > 0,
    (
        b * p_win
        -
        q_lose
    )
    /
    b,
    0.0,
)

# Kelly cannot recommend a negative wager.
final["kelly_full"] = (
    pd.to_numeric(
        final["kelly_full"],
        errors="coerce",
    )
    .fillna(0.0)
    .clip(lower=0.0)
)

# ------------------------------------------------------------
# FRACTIONAL KELLY
# ------------------------------------------------------------

final["kelly_fractional"] = (
    final["kelly_full"]
    *
    KELLY_FRACTION
)

# Hard bankroll exposure cap.
final["recommended_stake_pct"] = (
    final["kelly_fractional"]
    .clip(
        lower=0.0,
        upper=MAX_STAKE_PCT,
    )
)

# ------------------------------------------------------------
# BASE ELIGIBILITY
# ------------------------------------------------------------

final["passes_ev"] = (
    final["best_ev"]
    >=
    MIN_BET_EV
)

final["passes_edge"] = (
    final["best_edge_selected"]
    >=
    MIN_BET_EDGE
)

final["passes_odds"] = (
    final["best_odds"]
    .between(
        MIN_BET_ODDS,
        MAX_BET_ODDS,
        inclusive="both",
    )
)

final["base_bet_eligible"] = (
    final["passes_ev"]
    &
    final["passes_edge"]
    &
    final["passes_odds"]
    &
    (
        final["recommended_stake_pct"]
        >
        0
    )
)

# ------------------------------------------------------------
# BEST PRICE PER FIXTURE / SIDE
#
# Never recommend multiple wagers on the same BTTS side merely
# because several sportsbooks offer it.
# ------------------------------------------------------------

final["best_price_rank"] = np.nan

eligible_index = final.index[
    final["base_bet_eligible"]
]

if len(eligible_index):

    ranked = (
        final.loc[
            eligible_index
        ]
        .groupby(
            [
                "match_id",
                "best_side",
            ],
            dropna=False,
        )[
            "best_odds"
        ]
        .rank(
            method="first",
            ascending=False,
        )
    )

    final.loc[
        eligible_index,
        "best_price_rank",
    ] = ranked


final["best_price_available"] = (
    final["best_price_rank"]
    ==
    1
)

# ------------------------------------------------------------
# FINAL BET DECISION
# ------------------------------------------------------------

final["bet_eligible"] = (
    final["base_bet_eligible"]
    &
    final["best_price_available"]
)

final["bet_status"] = np.where(
    final["bet_eligible"],
    "BET",
    "PASS",
)

# PASS rows receive no recommended exposure.
final.loc[
    ~final["bet_eligible"],
    "recommended_stake_pct",
] = 0.0

# ------------------------------------------------------------
# OPTIONAL DOLLAR STAKE
# ------------------------------------------------------------

if BANKROLL is None:

    final["recommended_stake"] = np.nan

else:

    final["recommended_stake"] = (
        final["recommended_stake_pct"]
        *
        float(BANKROLL)
    )

final["prediction_time"] = (
    datetime.now(
        timezone.utc
    ).isoformat()
)


# ============================================================
# OUTPUT
# ============================================================

output_cols = [
    c
    for c in [
        "prediction_time",
        "match_id",
        "commence_time",
        "league_market",
        "home_team_market",
        "away_team_market",
        "bookmaker",
        "bookmaker_key",
        "home_lambda",
        "away_lambda",
        "poisson_btts",
        "cfg0755_model_probability",
        "cfg0755_probability",
        "champion_yes",
        "market_yes",
        "yes_odds",
        "no_odds",
        "model_market_gap",
        "dynamic_model_weight",
        "final_yes_probability",
        "yes_edge",
        "yes_ev",
        "no_probability",
        "no_edge",
        "no_ev",
        "best_side",
        "best_probability",
        "best_odds",
        "best_edge_selected",
        "best_ev",
        "kelly_full",
        "kelly_fractional",
        "recommended_stake_pct",
        "recommended_stake",
        "passes_ev",
        "passes_edge",
        "passes_odds",
        "best_price_available",
        "bet_eligible",
        "bet_status",
    ]
    if c in final.columns
]


result = final[
    output_cols
].copy()


result = result.sort_values(
    "best_ev",
    ascending=False,
)


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


result.to_csv(
    OUTPUT_FILE,
    index=False,
)


if LEDGER_FILE.exists():

    old = pd.read_csv(
        LEDGER_FILE,
        low_memory=False,
    )

    ledger = pd.concat(
        [
            old,
            result,
        ],
        ignore_index=True,
    )

else:

    ledger = result.copy()


ledger.to_csv(
    LEDGER_FILE,
    index=False,
)


banner(
    "LIVE BTTS OUTPUT"
)


display = result.copy()


pct_cols = [
    "poisson_btts",
    "cfg0755_model_probability",
    "cfg0755_probability",
    "market_yes",
    "final_yes_probability",
    "yes_edge",
    "yes_ev",
    "no_edge",
    "no_ev",
    "best_ev",
    "best_probability",
    "best_edge_selected",
    "kelly_full",
    "kelly_fractional",
    "recommended_stake_pct",
]


for c in pct_cols:

    if c in display.columns:

        display[c] = (
            pd.to_numeric(
                display[c],
                errors="coerce",
            )
            .map(
                lambda x:
                f"{x:.2%}"
                if pd.notna(x)
                else ""
            )
        )


print(
    display.to_string(
        index=False
    )
)


print()
print(
    "Current predictions:",
    OUTPUT_FILE,
)

print(
    "Permanent history:",
    LEDGER_FILE,
)

print()
print("DONE")
