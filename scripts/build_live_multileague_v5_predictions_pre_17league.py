from pathlib import Path
import math
import re
import unicodedata
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
LIVE = ROOT / "data" / "live"

TEAM_FILE = PROCESSED / "footystats_multileague_team_pregame_v2.csv"
HISTORY_FILE = PROCESSED / "footystats_multileague_history.csv"
FIXTURES_FILE = LIVE / "upcoming_fixtures.csv"

OUT = LIVE / "v5_live_predictions_footystats.csv"
OUT_PAPER = LIVE / "v5_live_predictions_core.csv"
OUT_AUDIT = LIVE / "v5_live_predictions_footystats_audit.csv"

GOAL_WEIGHT = 0.09
XG_WEIGHT = 0.75
SHOT_WEIGHT = 0.16
GOAL_RECENCY = 0.975
XG_RECENCY = 0.925
SHOT_RECENCY = 0.850
OPPONENT_STRENGTH = 0.875
OVERALL_WEIGHT = 0.80
VENUE_WEIGHT = 0.20

MIN_STRENGTH = 0.20
MAX_STRENGTH = 5.00
MAX_GOALS = 12

LEAGUES = {
    "Championship",
    "League One",
    "League Two",
    "2. Bundesliga",
    "Belgian Pro League",
    "La Liga",
}

CORE_V5_FILE = (
    PROCESSED
    / "frozen_v5_predictions.csv"
)

TIERS = {
    "Belgian Pro League": "A",
    "League One": "A",
    "La Liga": "A",
    "Championship": "B",
    "2. Bundesliga": "B",
    "League Two": "C",
}

ALIASES = {
    # England
    "qpr": "queens park rangers",
    "mk dons": "milton keynes dons",
    "luton": "luton town",
    "west ham united": "west ham",

    # Germany
    "hertha berlin": "hertha bsc",

    # Belgium
    "genk": "krc genk",
    "westerlo": "kvc westerlo",
    "leuven": "oh leuven",
    "beveren": "waasland beveren",
    "anderlecht": "rsc anderlecht",
    "gent": "kaa gent",
    "lommel": "lommel united",
    "charleroi": "sporting charleroi",

        # Spain
    "alaves": "deportivo alaves",
    "getafe": "getafe cf",
    "espanyol": "rcd espanyol",
    "levante": "levante ud",
    "athletic bilbao": "athletic club bilbao",
    "valencia": "valencia cf",
    "celta vigo": "celta de vigo",
    "barcelona": "fc barcelona",
    "sevilla": "sevilla fc",
        # Spain promoted-team donor aliases
    "real racing club de santander": "racing santander",
    "malaga": "malaga cf",
}

SIGNALS = {
    "goal": GOAL_RECENCY,
    "xg": XG_RECENCY,
    "shot": SHOT_RECENCY,
}

FACTORIALS = np.array(
    [math.factorial(i) for i in range(MAX_GOALS + 1)],
    dtype=float,
)


def norm_team(value):
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = text.lower().replace("&", " and ")

    # Remove common club-name prefixes/suffixes
    text = re.sub(
        r"\b(fc|afc|sv|vfl|sk)\b",
        " ",
        text,
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = " ".join(text.split())

    # Handles names such as "1. FC Nürnberg",
    # "1. FC Magdeburg", etc.
    text = re.sub(
        r"^1\s+",
        "",
        text,
    )

    return ALIASES.get(text, text)

def final_ewma(values, decay):
    num = 0.0
    den = 0.0
    n = 0
    for value in values:
        if not np.isfinite(value):
            continue
        num = decay * num + float(value)
        den = decay * den + 1.0
        n += 1
    if den <= 0:
        return np.nan, 0
    return num / den, n


def poisson(lam):
    goals = np.arange(MAX_GOALS + 1, dtype=float)
    p = np.exp(-lam) * (lam ** goals) / FACTORIALS
    return p / p.sum()


def market_probs(home_lambda, away_lambda):
    hp = poisson(home_lambda)
    ap = poisson(away_lambda)
    matrix = np.outer(hp, ap)

    p_home = np.tril(matrix, -1).sum()
    p_draw = np.trace(matrix)
    p_away = np.triu(matrix, 1).sum()
    z = p_home + p_draw + p_away

    total_grid = (
        np.arange(MAX_GOALS + 1)[:, None]
        +
        np.arange(MAX_GOALS + 1)[None, :]
    )

    out = {
        "p_home_v5": p_home / z,
        "p_draw_v5": p_draw / z,
        "p_away_v5": p_away / z,
        "p_btts_yes_v5": matrix[1:, 1:].sum(),
        "p_btts_no_v5": (
            matrix[0, :].sum()
            + matrix[:, 0].sum()
            - matrix[0, 0]
        ),
    }

    for line in (1.5, 2.5, 3.5, 4.5):
        tag = str(line).replace(".", "_")
        out[f"p_over_{tag}_v5"] = matrix[total_grid > line].sum()
        out[f"p_under_{tag}_v5"] = matrix[total_grid < line].sum()

    return out


def load_inputs():
    for path in (TEAM_FILE, HISTORY_FILE, FIXTURES_FILE):
        if not path.exists():
            raise FileNotFoundError(path)

    team = pd.read_csv(TEAM_FILE, low_memory=False)
    history = pd.read_csv(HISTORY_FILE, low_memory=False)
    fixtures = pd.read_csv(FIXTURES_FILE, low_memory=False)

    needed_team = [
        "footystats_match_id", "date", "league",
        "team", "venue",
        "adj_goal_attack_perf", "adj_goal_defense_perf",
        "adj_xg_attack_perf", "adj_xg_defense_perf",
        "adj_shot_attack_perf", "adj_shot_defense_perf",
    ]
    missing = [c for c in needed_team if c not in team.columns]
    if missing:
        raise ValueError(f"Missing V2 team columns: {missing}")

    needed_fix = [
        "match_id", "date", "league",
        "home_team", "away_team",
    ]
    missing = [c for c in needed_fix if c not in fixtures.columns]
    if missing:
        raise ValueError(f"Missing fixture columns: {missing}")

    team["date"] = pd.to_datetime(team["date"], errors="coerce")
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    fixtures["date"] = pd.to_datetime(
        fixtures["date"], errors="coerce"
    ).dt.normalize()

    for col in [
        "adj_goal_attack_perf", "adj_goal_defense_perf",
        "adj_xg_attack_perf", "adj_xg_defense_perf",
        "adj_shot_attack_perf", "adj_shot_defense_perf",
    ]:
        team[col] = pd.to_numeric(team[col], errors="coerce")

    for col in ["home_goals", "away_goals"]:
        history[col] = pd.to_numeric(history[col], errors="coerce")

    team["team_norm"] = team["team"].map(norm_team)
    fixtures["home_team_norm"] = fixtures["home_team"].map(norm_team)
    fixtures["away_team_norm"] = fixtures["away_team"].map(norm_team)

    return team, history, fixtures


def build_state_store(team):
    rows = []

    for team_norm, club in team.groupby("team_norm", sort=False):
        club = club.sort_values(["date", "footystats_match_id"])
        display = club["team"].iloc[-1]
        last_date = club["date"].max()

        global_values = {}
        for signal, decay in SIGNALS.items():
            for role in ("attack", "defense"):
                col = f"adj_{signal}_{role}_perf"

                global_values[(signal, role, "overall")] = final_ewma(
                    club[col].to_numpy(dtype=float),
                    decay,
                )

                for venue in ("HOME", "AWAY"):
                    vals = club.loc[
                        club["venue"] == venue,
                        col,
                    ].to_numpy(dtype=float)

                    global_values[(signal, role, venue)] = final_ewma(
                        vals,
                        decay,
                    )

        for league, league_club in club.groupby("league", sort=False):
            league_club = league_club.sort_values(
                ["date", "footystats_match_id"]
            )

            row = {
                "team_norm": team_norm,
                "team_history_name": display,
                "league": league,
                "last_completed_date": last_date,
                "same_league_games": len(league_club),
                "global_games": len(club),
            }

            for signal, decay in SIGNALS.items():
                for role in ("attack", "defense"):
                    col = f"adj_{signal}_{role}_perf"

                    val, n = final_ewma(
                        league_club[col].to_numpy(dtype=float),
                        decay,
                    )
                    row[f"same_{signal}_{role}_overall"] = val
                    row[f"same_{signal}_{role}_overall_games"] = n

                    gval, gn = global_values[
                        (signal, role, "overall")
                    ]
                    row[f"global_{signal}_{role}_overall"] = gval
                    row[f"global_{signal}_{role}_overall_games"] = gn

                    for venue in ("HOME", "AWAY"):
                        slug = venue.lower()
                        vals = league_club.loc[
                            league_club["venue"] == venue,
                            col,
                        ].to_numpy(dtype=float)

                        v, vn = final_ewma(vals, decay)
                        row[f"same_{signal}_{role}_{slug}"] = v
                        row[
                            f"same_{signal}_{role}_{slug}_games"
                        ] = vn

                        gv, gvn = global_values[
                            (signal, role, venue)
                        ]
                        row[f"global_{signal}_{role}_{slug}"] = gv
                        row[
                            f"global_{signal}_{role}_{slug}_games"
                        ] = gvn

            rows.append(row)

    return pd.DataFrame(rows)

def build_core_v5_transfer_states():
    """
    Build fallback states from the frozen core V5 prediction store.

    These states are used ONLY when a live expansion-league team
    has no history anywhere in the FootyStats expansion state store.

    Example:
        West Ham Premier League history
            -> Championship live fixture
            -> TRANSFERRED_CORE_V5

    No V5 parameters are refit.
    """

    if not CORE_V5_FILE.exists():
        print()
        print(
            "WARNING: core V5 transfer file missing:"
        )
        print(
            CORE_V5_FILE
        )

        return pd.DataFrame()

    df = pd.read_csv(
        CORE_V5_FILE,
        low_memory=False,
    )

    required = [
        "date",

        "home_team",
        "away_team",

        "home_games",
        "away_games",

        "home_venue_games",
        "away_venue_games",

        "home_adj_goal_attack",
        "home_adj_goal_defense",
        "home_adj_venue_goal_attack",
        "home_adj_venue_goal_defense",

        "away_adj_goal_attack",
        "away_adj_goal_defense",
        "away_adj_venue_goal_attack",
        "away_adj_venue_goal_defense",

        "home_xg_attack_overall",
        "home_xg_defense_overall",
        "home_xg_attack_venue",
        "home_xg_defense_venue",

        "away_xg_attack_overall",
        "away_xg_defense_overall",
        "away_xg_attack_venue",
        "away_xg_defense_venue",

        "home_adj_shot_attack",
        "home_adj_shot_defense",
        "home_adj_venue_shot_attack",
        "home_adj_venue_shot_defense",

        "away_adj_shot_attack",
        "away_adj_shot_defense",
        "away_adj_venue_shot_attack",
        "away_adj_venue_shot_defense",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Core V5 transfer store missing columns:\n"
            +
            "\n".join(
                f" - {col}"
                for col in missing
            )
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    # ========================================================
    # HOME TEAM STATES
    # ========================================================

    home = pd.DataFrame(
        {
            "date":
                df["date"],

            "team":
                df["home_team"],

            "games":
                df["home_games"],

            "venue_games":
                df["home_venue_games"],

            "goal_attack_overall":
                df["home_adj_goal_attack"],

            "goal_defense_overall":
                df["home_adj_goal_defense"],

            "goal_attack_venue":
                df["home_adj_venue_goal_attack"],

            "goal_defense_venue":
                df["home_adj_venue_goal_defense"],

            "xg_attack_overall":
                df["home_xg_attack_overall"],

            "xg_defense_overall":
                df["home_xg_defense_overall"],

            "xg_attack_venue":
                df["home_xg_attack_venue"],

            "xg_defense_venue":
                df["home_xg_defense_venue"],

            "shot_attack_overall":
                df["home_adj_shot_attack"],

            "shot_defense_overall":
                df["home_adj_shot_defense"],

            "shot_attack_venue":
                df["home_adj_venue_shot_attack"],

            "shot_defense_venue":
                df["home_adj_venue_shot_defense"],
        }
    )

    # ========================================================
    # AWAY TEAM STATES
    # ========================================================

    away = pd.DataFrame(
        {
            "date":
                df["date"],

            "team":
                df["away_team"],

            "games":
                df["away_games"],

            "venue_games":
                df["away_venue_games"],

            "goal_attack_overall":
                df["away_adj_goal_attack"],

            "goal_defense_overall":
                df["away_adj_goal_defense"],

            "goal_attack_venue":
                df["away_adj_venue_goal_attack"],

            "goal_defense_venue":
                df["away_adj_venue_goal_defense"],

            "xg_attack_overall":
                df["away_xg_attack_overall"],

            "xg_defense_overall":
                df["away_xg_defense_overall"],

            "xg_attack_venue":
                df["away_xg_attack_venue"],

            "xg_defense_venue":
                df["away_xg_defense_venue"],

            "shot_attack_overall":
                df["away_adj_shot_attack"],

            "shot_defense_overall":
                df["away_adj_shot_defense"],

            "shot_attack_venue":
                df["away_adj_venue_shot_attack"],

            "shot_defense_venue":
                df["away_adj_venue_shot_defense"],
        }
    )

    core = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    core["team_norm"] = (
        core["team"].map(
            norm_team
        )
    )

    numeric_cols = [
        "games",
        "venue_games",

        "goal_attack_overall",
        "goal_defense_overall",
        "goal_attack_venue",
        "goal_defense_venue",

        "xg_attack_overall",
        "xg_defense_overall",
        "xg_attack_venue",
        "xg_defense_venue",

        "shot_attack_overall",
        "shot_defense_overall",
        "shot_attack_venue",
        "shot_defense_venue",
    ]

    for col in numeric_cols:
        core[col] = pd.to_numeric(
            core[col],
            errors="coerce",
        )

    core = core.dropna(
        subset=[
            "date",
            "team",
            "team_norm",
        ]
    ).copy()

    # Latest legitimate frozen-V5 state available for each club.
    core = (
        core
        .sort_values(
            [
                "team_norm",
                "date",
            ]
        )
        .groupby(
            "team_norm",
            as_index=False,
        )
        .tail(1)
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "Core V5 fallback teams:",
        f"{len(core):,}",
    )

    return core


def build_baselines(history):
    return (
        history.loc[history["league"].isin(LEAGUES)]
        .groupby("league", as_index=False)
        .agg(
            completed_games=("footystats_match_id", "count"),
            lg_home_goals=("home_goals", "mean"),
            lg_away_goals=("away_goals", "mean"),
        )
    )


def resolve_state(
    states,
    team_name,
    league,
    venue,
    core_states=None,
):
    team_norm = norm_team(
        team_name
    )

    candidates = states.loc[
        states["team_norm"]
        ==
        team_norm
    ].copy()

    # ========================================================
    # FOOTYSTATS HISTORY EXISTS
    # ========================================================

    if not candidates.empty:

        same = candidates.loc[
            candidates["league"]
            ==
            league
        ]

        if not same.empty:
            row = (
                same
                .sort_values(
                    "last_completed_date"
                )
                .iloc[-1]
            )

            prefix = "same"
            source = "SAME_LEAGUE"

        else:
            row = (
                candidates
                .sort_values(
                    "last_completed_date"
                )
                .iloc[-1]
            )

            prefix = "global"
            source = "TRANSFERRED"

        slug = venue.lower()

        out = {
            "resolved":
                True,

            "history_source":
                source,

            "history_team":
                row[
                    "team_history_name"
                ],

            "team_norm":
                team_norm,

            "same_league_games":
                int(
                    row[
                        "same_league_games"
                    ]
                ),

            "global_games":
                int(
                    row[
                        "global_games"
                    ]
                ),
        }

        for signal in SIGNALS:

            for role in (
                "attack",
                "defense",
            ):

                overall = row.get(
                    (
                        f"{prefix}_"
                        f"{signal}_"
                        f"{role}_overall"
                    ),
                    np.nan,
                )

                venue_value = row.get(
                    (
                        f"{prefix}_"
                        f"{signal}_"
                        f"{role}_{slug}"
                    ),
                    np.nan,
                )

                if not np.isfinite(
                    overall
                ):
                    overall = 1.0

                if not np.isfinite(
                    venue_value
                ):

                    venue_value = row.get(
                        (
                            f"global_"
                            f"{signal}_"
                            f"{role}_{slug}"
                        ),
                        np.nan,
                    )

                if not np.isfinite(
                    venue_value
                ):
                    venue_value = 1.0

                out[
                    f"{signal}_{role}_overall"
                ] = float(
                    np.clip(
                        overall,
                        MIN_STRENGTH,
                        MAX_STRENGTH,
                    )
                )

                out[
                    f"{signal}_{role}_venue"
                ] = float(
                    np.clip(
                        venue_value,
                        MIN_STRENGTH,
                        MAX_STRENGTH,
                    )
                )

        return out

    # ========================================================
    # CORE V5 FALLBACK
    # ========================================================

    if (
        core_states is not None
        and
        len(core_states)
    ):

        core_match = core_states.loc[
            core_states[
                "team_norm"
            ]
            ==
            team_norm
        ]

        if not core_match.empty:

            row = (
                core_match
                .sort_values(
                    "date"
                )
                .iloc[-1]
            )

            out = {
                "resolved":
                    True,

                "history_source":
                    "TRANSFERRED_CORE_V5",

                "history_team":
                    row[
                        "team"
                    ],

                "team_norm":
                    team_norm,

                # There is no same-league Championship
                # history in this branch.
                "same_league_games":
                    0,

                "global_games":
                    int(
                        row[
                            "games"
                        ]
                    )
                    if
                    pd.notna(
                        row[
                            "games"
                        ]
                    )
                    else
                    0,
            }

            for signal in SIGNALS:

                for role in (
                    "attack",
                    "defense",
                ):

                    overall = row[
                        (
                            f"{signal}_"
                            f"{role}_overall"
                        )
                    ]

                    venue_value = row[
                        (
                            f"{signal}_"
                            f"{role}_venue"
                        )
                    ]

                    if not np.isfinite(
                        overall
                    ):
                        overall = 1.0

                    if not np.isfinite(
                        venue_value
                    ):
                        venue_value = overall

                    out[
                        f"{signal}_{role}_overall"
                    ] = float(
                        np.clip(
                            overall,
                            MIN_STRENGTH,
                            MAX_STRENGTH,
                        )
                    )

                    out[
                        f"{signal}_{role}_venue"
                    ] = float(
                        np.clip(
                            venue_value,
                            MIN_STRENGTH,
                            MAX_STRENGTH,
                        )
                    )

            return out

    # ========================================================
    # NOTHING LEGITIMATE FOUND
    # ========================================================

    return {
        "resolved":
            False,

        "history_source":
            "UNRESOLVED",

        "history_team":
            None,

        "team_norm":
            team_norm,
    }


def blend(overall, venue):
    return (
        OVERALL_WEIGHT * overall
        + VENUE_WEIGHT * venue
    )


def score_fixture(fixture, home, away, baseline):
    c = {}

    for signal in SIGNALS:
        c[f"h_{signal}_a"] = blend(
            home[f"{signal}_attack_overall"],
            home[f"{signal}_attack_venue"],
        )
        c[f"h_{signal}_d"] = blend(
            home[f"{signal}_defense_overall"],
            home[f"{signal}_defense_venue"],
        )
        c[f"a_{signal}_a"] = blend(
            away[f"{signal}_attack_overall"],
            away[f"{signal}_attack_venue"],
        )
        c[f"a_{signal}_d"] = blend(
            away[f"{signal}_defense_overall"],
            away[f"{signal}_defense_venue"],
        )

    h_attack = (
        GOAL_WEIGHT * c["h_goal_a"]
        + XG_WEIGHT * c["h_xg_a"]
        + SHOT_WEIGHT * c["h_shot_a"]
    )
    h_defense = (
        GOAL_WEIGHT * c["h_goal_d"]
        + XG_WEIGHT * c["h_xg_d"]
        + SHOT_WEIGHT * c["h_shot_d"]
    )
    a_attack = (
        GOAL_WEIGHT * c["a_goal_a"]
        + XG_WEIGHT * c["a_xg_a"]
        + SHOT_WEIGHT * c["a_shot_a"]
    )
    a_defense = (
        GOAL_WEIGHT * c["a_goal_d"]
        + XG_WEIGHT * c["a_xg_d"]
        + SHOT_WEIGHT * c["a_shot_d"]
    )

    home_lambda = float(np.clip(
        baseline["lg_home_goals"]
        * h_attack
        * a_defense,
        0.15,
        4.50,
    ))

    away_lambda = float(np.clip(
        baseline["lg_away_goals"]
        * a_attack
        * h_defense,
        0.15,
        4.50,
    ))

    probs = market_probs(
        home_lambda,
        away_lambda,
    )

    return {
        "match_id": fixture["match_id"],
        "date": fixture["date"],
        "league": fixture["league"],
        "home_team": fixture["home_team"],
        "away_team": fixture["away_team"],
        "xg_provider": "FOOTYSTATS",
        "deployment_tier": TIERS.get(
            fixture["league"], "UNRATED"
        ),
        "home_history_source": home["history_source"],
        "away_history_source": away["history_source"],
        "home_history_team": home["history_team"],
        "away_history_team": away["history_team"],
        "home_history_games": (
            home["same_league_games"]
            if home["history_source"] == "SAME_LEAGUE"
            else home["global_games"]
        ),
        "away_history_games": (
            away["same_league_games"]
            if away["history_source"] == "SAME_LEAGUE"
            else away["global_games"]
        ),
        "lg_home_goals": baseline["lg_home_goals"],
        "lg_away_goals": baseline["lg_away_goals"],

        # ====================================================
        # EXPOSE FROZEN V5 COMPONENTS FOR CFG_0755
        #
        # These are the exact 80% overall / 20% venue blended
        # strengths already used above to construct the V5
        # lambdas. No new modeling or fitting is performed.
        # ====================================================

        "home_adj_goal_attack": c["h_goal_a"],
        "home_adj_goal_defense": c["h_goal_d"],
        "away_adj_goal_attack": c["a_goal_a"],
        "away_adj_goal_defense": c["a_goal_d"],

        "home_adj_xg_attack": c["h_xg_a"],
        "home_adj_xg_defense": c["h_xg_d"],
        "away_adj_xg_attack": c["a_xg_a"],
        "away_adj_xg_defense": c["a_xg_d"],

        "home_adj_shot_attack": c["h_shot_a"],
        "home_adj_shot_defense": c["h_shot_d"],
        "away_adj_shot_attack": c["a_shot_a"],
        "away_adj_shot_defense": c["a_shot_d"],

        # Native history counters used by the CFG bridge.
        "home_games": (
            home["same_league_games"]
            if home["history_source"] == "SAME_LEAGUE"
            else home["global_games"]
        ),
        "away_games": (
            away["same_league_games"]
            if away["history_source"] == "SAME_LEAGUE"
            else away["global_games"]
        ),

        "home_lambda_v5": home_lambda,
        "away_lambda_v5": away_lambda,
        **probs,
    }


def main():
    print()
    print("=" * 100)
    print("BUILD LIVE FOOTYSTATS MULTI-LEAGUE V5 PREDICTIONS")
    print("=" * 100)
    print()
    print("Frozen V5:")
    print("Goals 9% | xG 75% | Shots 16%")
    print(
        "Recency:"
        f" goals={GOAL_RECENCY}"
        f" xG={XG_RECENCY}"
        f" shots={SHOT_RECENCY}"
    )
    print("Opponent strength:", OPPONENT_STRENGTH)
    print("Overall / venue: 80% / 20%")
    print("No parameters fitted.")

    team, history, fixtures = load_inputs()

    expansion = fixtures.loc[
        fixtures["league"].isin(LEAGUES)
    ].copy()

    reserved = fixtures.loc[
        ~fixtures["league"].isin(LEAGUES)
    ].copy()

    print()
    print("All upcoming fixtures:", len(fixtures))
    print("FootyStats fixtures:", len(expansion))
    print("Reserved for Understat:", len(reserved))

    states = build_state_store(team)
    core_states = build_core_v5_transfer_states()
    baselines = build_baselines(history)

    baseline_lookup = {
        row["league"]: row
        for _, row in baselines.iterrows()
    }

    predictions = []
    audit = []

    for _, fixture in expansion.iterrows():
        league = fixture["league"]
        baseline = baseline_lookup.get(league)

        home = resolve_state(
            states,
            fixture["home_team"],
            league,
            "HOME",
            core_states,
        )
        away = resolve_state(
            states,
            fixture["away_team"],
            league,
            "AWAY",
            core_states,
        )

        issues = []
        if baseline is None:
            issues.append("NO_LEAGUE_BASELINE")
        if not home["resolved"]:
            issues.append("HOME_TEAM_UNRESOLVED")
        if not away["resolved"]:
            issues.append("AWAY_TEAM_UNRESOLVED")

        if issues:
            audit.append({
                "match_id": fixture["match_id"],
                "date": fixture["date"],
                "league": league,
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "status": "BLOCKED",
                "issues": "|".join(issues),
                "home_history_match": home.get("history_team"),
                "away_history_match": away.get("history_team"),
            })
            continue

        predictions.append(
            score_fixture(
                fixture,
                home,
                away,
                baseline,
            )
        )

        audit.append({
            "match_id": fixture["match_id"],
            "date": fixture["date"],
            "league": league,
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "status": "SCORED",
            "issues": "",
            "home_history_match": home["history_team"],
            "away_history_match": away["history_team"],
            "home_history_source": home["history_source"],
            "away_history_source": away["history_source"],
        })

    pred = pd.DataFrame(predictions)
    audit_df = pd.DataFrame(audit)

    LIVE.mkdir(parents=True, exist_ok=True)

    pred.to_csv(OUT, index=False)
    pred.to_csv(OUT_PAPER, index=False)
    audit_df.to_csv(OUT_AUDIT, index=False)

    scored = (
        int((audit_df["status"] == "SCORED").sum())
        if len(audit_df)
        else 0
    )
    blocked = (
        int((audit_df["status"] == "BLOCKED").sum())
        if len(audit_df)
        else 0
    )

    print()
    print("=" * 100)
    print("LIVE SCORING RESULTS")
    print("=" * 100)
    print("Scored:", scored, "/", len(expansion))
    print("Blocked:", blocked)

    if len(audit_df):
        print()
        print("BY LEAGUE")
        print(
            audit_df.groupby(
                ["league", "status"]
            )
            .size()
            .unstack(fill_value=0)
            .to_string()
        )

    if blocked:
        print()
        print("BLOCKED FIXTURES")
        print(
            audit_df.loc[
                audit_df["status"] == "BLOCKED",
                [
                    "league",
                    "home_team",
                    "away_team",
                    "issues",
                    "home_history_match",
                    "away_history_match",
                ],
            ].to_string(index=False)
        )

    if len(pred):
        print()
        print("=" * 135)
        print("LIVE V5 PREVIEW")
        print("=" * 135)
        cols = [
            "date", "league", "home_team", "away_team",
            "deployment_tier",
            "home_history_source", "away_history_source",
            "home_lambda_v5", "away_lambda_v5",
            "p_home_v5", "p_draw_v5", "p_away_v5",
            "p_over_2_5_v5", "p_btts_yes_v5",
        ]
        print(
            pred[cols]
            .sort_values(
                ["date", "league", "home_team"]
            )
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    print()
    print("FootyStats predictions:", OUT)
    print("Paper-live input:", OUT_PAPER)
    print("Audit:", OUT_AUDIT)
    print()
    print(
    "NOTE: this step scores the six "
    "FootyStats expansion leagues."
)
print(
    "Premier League and Bundesliga "
    "remain with the core V5 provider step."
)


if __name__ == "__main__":
    main()
