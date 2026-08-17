from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS))

import tune_overall_venue_v5 as ov


# ============================================================
# FROZEN LIVE TRANSITION SETTINGS
# ============================================================

PROMOTION_ADJUSTMENT = 0.205
RELEGATION_ADJUSTMENT = 0.135


# ============================================================
# HELPERS
# ============================================================

def result_classes(
    home_goals,
    away_goals,
):
    return np.where(
        home_goals > away_goals,
        0,
        np.where(
            home_goals == away_goals,
            1,
            2,
        ),
    )


def log_loss(
    y,
    probs,
):
    chosen = probs[
        np.arange(
            len(y)
        ),
        y,
    ]

    chosen = np.clip(
        chosen,
        1e-15,
        1.0,
    )

    return float(
        -np.mean(
            np.log(
                chosen
            )
        )
    )


def brier(
    y,
    probs,
):
    one_hot = np.zeros_like(
        probs,
        dtype=float,
    )

    one_hot[
        np.arange(
            len(y)
        ),
        y,
    ] = 1.0

    return float(
        np.mean(
            np.sum(
                (
                    probs
                    -
                    one_hot
                )
                ** 2,
                axis=1,
            )
        )
    )


def accuracy(
    y,
    probs,
):
    return float(
        np.mean(
            probs.argmax(
                axis=1
            )
            ==
            y
        )
    )


# ============================================================
# LEAGUE TRANSITION DETECTION
# ============================================================

def add_transition_flags(
    df,
):

    out = df.copy()

    # ========================================================
    # HISTORICAL MULTI-LEAGUE MEMBERSHIP SOURCE
    # ========================================================

    history_file = (
        ROOT
        / "data"
        / "processed"
        / "footystats_multileague_history.csv"
    )

    if not history_file.exists():

        raise FileNotFoundError(
            "FootyStats multi-league history not found:\n"
            f"{history_file}"
        )

    history = pd.read_csv(
        history_file,
        low_memory=False,
    )

    required_history = [
        "season",
        "league",
        "date",
        "home_team",
        "away_team",
    ]

    missing = [
        c
        for c in required_history
        if c not in history.columns
    ]

    if missing:

        raise ValueError(
            "FootyStats history missing columns: "
            + str(missing)
        )

    # ========================================================
    # CLEAN HISTORY
    # ========================================================

    history["date"] = pd.to_datetime(
        history["date"],
        errors="coerce",
    )

    history["season"] = (
        history["season"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.zfill(4)
    )

    history["league"] = (
        history["league"]
        .astype(str)
        .str.strip()
    )

    history["home_team"] = (
        history["home_team"]
        .astype(str)
        .str.strip()
    )

    history["away_team"] = (
        history["away_team"]
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # MATCH LEVEL -> TEAM / SEASON / LEAGUE
    # ========================================================

    home = history[
        [
            "season",
            "league",
            "date",
            "home_team",
        ]
    ].rename(
        columns={
            "home_team":
                "team",
        }
    )

    away = history[
        [
            "season",
            "league",
            "date",
            "away_team",
        ]
    ].rename(
        columns={
            "away_team":
                "team",
        }
    )

    team_history = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    team_history["team"] = (
        team_history["team"]
        .astype(str)
        .str.strip()
    )

    team_history = team_history.loc[
        team_history["date"].notna()
    ].copy()

    team_seasons = (
        team_history
        .groupby(
            [
                "team",
                "season",
                "league",
            ],
            as_index=False,
        )
        .agg(
            last_date=(
                "date",
                "max",
            )
        )
    )

    # ========================================================
    # LEAGUE PYRAMIDS
    # ========================================================

    league_levels = {

        # England
        "Premier League": 1,
        "Championship": 2,
        "League One": 3,
        "League Two": 4,
        "National League": 5,

        # Germany
        "Bundesliga": 1,
        "2. Bundesliga": 2,

        # Spain
        "La Liga": 1,
        "Segunda División": 2,
    }

    league_countries = {

        "Premier League":
            "England",

        "Championship":
            "England",

        "League One":
            "England",

        "League Two":
            "England",

        "National League":
            "England",

        "Bundesliga":
            "Germany",

        "2. Bundesliga":
            "Germany",

        "La Liga":
            "Spain",

        "Segunda División":
            "Spain",
    }

    # ========================================================
    # CORE V5 CURRENT-LEAGUE MAPPING
    # ========================================================

    league_code_to_name = {
        "E0":
            "Premier League",

        "D1":
            "Bundesliga",
    }

    # ========================================================
    # HISTORICAL NAME ALIASES
    # ========================================================

    history_aliases = {

        # England
        "Brighton":
            "Brighton and Hove Albion",

        "Man City":
            "Manchester City",

        "Man United":
            "Manchester United",

        "Newcastle":
            "Newcastle United",

        "Nott'm Forest":
            "Nottingham Forest",

        "Tottenham":
            "Tottenham Hotspur",

        # Germany
        "FC Koln":
            "1. FC Köln",

        "Leverkusen":
            "Bayer Leverkusen",

        "Dortmund":
            "Borussia Dortmund",

        "M'gladbach":
            "Borussia Monchengladbach",

        "Ein Frankfurt":
            "Eintracht Frankfurt",

        "Mainz":
            "FSV Mainz 05",

        "Hamburg":
            "Hamburger SV",

        "Freiburg":
            "SC Freiburg",

        "Paderborn":
            "SC Paderborn",

        "Hoffenheim":
            "TSG Hoffenheim",

        "Stuttgart":
            "VfB Stuttgart",

        "Schalke 04":
            "FC Schalke 04",
    }

    history_team_set = set(
        team_seasons[
            "team"
        ]
        .dropna()
        .astype(str)
    )

    # ========================================================
    # TEAM NAME RESOLUTION
    # ========================================================

    def resolve_history_name(
        team,
    ):

        team = str(
            team
        ).strip()

        if team in history_team_set:
            return team

        alias = history_aliases.get(
            team
        )

        if (
            alias is not None
            and alias in history_team_set
        ):
            return alias

        return team

    # ========================================================
    # PREVIOUS SEASON
    # ========================================================

    def previous_season(
        season,
    ):

        season = str(
            season
        ).zfill(4)

        start = int(
            season[
                :2
            ]
        )

        end = int(
            season[
                2:
            ]
        )

        return (
            f"{start - 1:02d}"
            f"{end - 1:02d}"
        )

    # ========================================================
    # PREVIOUS LEAGUE LOOKUP
    # ========================================================

    def get_previous_league(
        team,
        season,
        current_league,
    ):

        resolved_team = (
            resolve_history_name(
                team
            )
        )

        prev = previous_season(
            season
        )

        candidates = team_seasons.loc[
            (
                team_seasons[
                    "team"
                ]
                ==
                resolved_team
            )
            &
            (
                team_seasons[
                    "season"
                ]
                ==
                prev
            )
        ].copy()

        if candidates.empty:

            return (
                resolved_team,
                None,
            )

        current_country = (
            league_countries.get(
                current_league
            )
        )

        if current_country is not None:

            candidates[
                "country"
            ] = (
                candidates[
                    "league"
                ]
                .map(
                    league_countries
                )
            )

            same_country = candidates.loc[
                candidates[
                    "country"
                ]
                ==
                current_country
            ]

            if not same_country.empty:

                candidates = (
                    same_country.copy()
                )

        candidates = candidates.sort_values(
            "last_date"
        )

        previous = str(
            candidates.iloc[
                -1
            ][
                "league"
            ]
        )

        return (
            resolved_team,
            previous,
        )

    # ========================================================
    # NORMALIZE BACKTEST SEASON
    # ========================================================

    out["season_norm"] = (
        out["season"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.zfill(4)
    )

    # ========================================================
    # BUILD FLAGS
    # ========================================================

    rows = []

    for fixture in out.itertuples(
        index=False
    ):

        current_league = (
            league_code_to_name.get(
                str(
                    fixture.league_code
                )
            )
        )

        season = str(
            fixture.season_norm
        ).zfill(4)

        home_team = str(
            fixture.home_team
        ).strip()

        away_team = str(
            fixture.away_team
        ).strip()

        (
            home_history_team,
            home_previous_league,
        ) = get_previous_league(
            home_team,
            season,
            current_league,
        )

        (
            away_history_team,
            away_previous_league,
        ) = get_previous_league(
            away_team,
            season,
            current_league,
        )

        current_level = (
            league_levels.get(
                current_league
            )
        )

        home_promoted = 0
        away_promoted = 0

        home_relegated = 0
        away_relegated = 0

        # ----------------------------------------------------
        # HOME
        # ----------------------------------------------------

        if (
            current_level is not None
            and
            home_previous_league
            in league_levels
        ):

            previous_level = (
                league_levels[
                    home_previous_league
                ]
            )

            previous_country = (
                league_countries.get(
                    home_previous_league
                )
            )

            current_country = (
                league_countries.get(
                    current_league
                )
            )

            if (
                previous_country
                ==
                current_country
            ):

                if (
                    previous_level
                    >
                    current_level
                ):

                    home_promoted = 1

                elif (
                    previous_level
                    <
                    current_level
                ):

                    home_relegated = 1

        # ----------------------------------------------------
        # AWAY
        # ----------------------------------------------------

        if (
            current_level is not None
            and
            away_previous_league
            in league_levels
        ):

            previous_level = (
                league_levels[
                    away_previous_league
                ]
            )

            previous_country = (
                league_countries.get(
                    away_previous_league
                )
            )

            current_country = (
                league_countries.get(
                    current_league
                )
            )

            if (
                previous_country
                ==
                current_country
            ):

                if (
                    previous_level
                    >
                    current_level
                ):

                    away_promoted = 1

                elif (
                    previous_level
                    <
                    current_level
                ):

                    away_relegated = 1

        rows.append(
            {
                "match_id":
                    fixture.match_id,

                "home_history_team":
                    home_history_team,

                "away_history_team":
                    away_history_team,

                "home_previous_league":
                    home_previous_league,

                "away_previous_league":
                    away_previous_league,

                "home_promoted":
                    home_promoted,

                "away_promoted":
                    away_promoted,

                "home_relegated":
                    home_relegated,

                "away_relegated":
                    away_relegated,
            }
        )

    flags = pd.DataFrame(
        rows
    )

    flags[
        "transition_applied"
    ] = (
        flags[
            [
                "home_promoted",
                "away_promoted",
                "home_relegated",
                "away_relegated",
            ]
        ]
        .sum(
            axis=1
        )
        >
        0
    ).astype(int)

    out = out.merge(
        flags,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    return out

# ============================================================
# APPLY EXACT LIVE TRANSITION RULES
# ============================================================

def apply_transition(
    df,
    home_lambda,
    away_lambda,
):
    home = pd.Series(
        np.asarray(
            home_lambda,
            dtype=float,
        ),
        index=df.index,
    )

    away = pd.Series(
        np.asarray(
            away_lambda,
            dtype=float,
        ),
        index=df.index,
    )

    # --------------------------------------------------------
    # HOME PROMOTED
    # --------------------------------------------------------

    mask = (
        df[
            "home_promoted"
        ]
        .astype(bool)
    )

    home.loc[
        mask
    ] *= (
        1.0
        -
        PROMOTION_ADJUSTMENT
    )

    away.loc[
        mask
    ] *= (
        1.0
        +
        PROMOTION_ADJUSTMENT
    )

    # --------------------------------------------------------
    # AWAY PROMOTED
    # --------------------------------------------------------

    mask = (
        df[
            "away_promoted"
        ]
        .astype(bool)
    )

    away.loc[
        mask
    ] *= (
        1.0
        -
        PROMOTION_ADJUSTMENT
    )

    home.loc[
        mask
    ] *= (
        1.0
        +
        PROMOTION_ADJUSTMENT
    )

    # --------------------------------------------------------
    # HOME RELEGATED
    # --------------------------------------------------------

    mask = (
        df[
            "home_relegated"
        ]
        .astype(bool)
    )

    home.loc[
        mask
    ] *= (
        1.0
        +
        RELEGATION_ADJUSTMENT
    )

    away.loc[
        mask
    ] *= (
        1.0
        -
        RELEGATION_ADJUSTMENT
    )

    # --------------------------------------------------------
    # AWAY RELEGATED
    # --------------------------------------------------------

    mask = (
        df[
            "away_relegated"
        ]
        .astype(bool)
    )

    away.loc[
        mask
    ] *= (
        1.0
        +
        RELEGATION_ADJUSTMENT
    )

    home.loc[
        mask
    ] *= (
        1.0
        -
        RELEGATION_ADJUSTMENT
    )

    home = home.clip(
        lower=0.15,
        upper=4.50,
    )

    away = away.clip(
        lower=0.15,
        upper=4.50,
    )

    return (
        home.to_numpy(),
        away.to_numpy(),
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_subset(
    df,
    raw_probs,
    transition_probs,
    mask,
    label,
):
    sub = df.loc[
        mask
    ].copy()

    if sub.empty:
        return {
            "segment":
                label,

            "games":
                0,

            "raw_accuracy":
                np.nan,

            "transition_accuracy":
                np.nan,

            "raw_log_loss":
                np.nan,

            "transition_log_loss":
                np.nan,

            "raw_brier":
                np.nan,

            "transition_brier":
                np.nan,
        }

    idx = np.flatnonzero(
        mask.to_numpy()
    )

    raw = raw_probs[
        idx
    ]

    transition = transition_probs[
        idx
    ]

    y = result_classes(
        sub[
            "home_goals"
        ].to_numpy(),

        sub[
            "away_goals"
        ].to_numpy(),
    )

    return {
        "segment":
            label,

        "games":
            len(sub),

        "raw_accuracy":
            accuracy(
                y,
                raw,
            ),

        "transition_accuracy":
            accuracy(
                y,
                transition,
            ),

        "raw_log_loss":
            log_loss(
                y,
                raw,
            ),

        "transition_log_loss":
            log_loss(
                y,
                transition,
            ),

        "raw_brier":
            brier(
                y,
                raw,
            ),

        "transition_brier":
            brier(
                y,
                transition,
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print("BACKTEST LIVE TRANSITION V5")
    print("=" * 110)

    print()
    print(
        "Promotion adjustment:",
        PROMOTION_ADJUSTMENT,
    )

    print(
        "Relegation adjustment:",
        RELEGATION_ADJUSTMENT,
    )

    # ========================================================
    # BUILD ORIGINAL HISTORICAL V5 COMPONENT STORE
    # ========================================================

    print()
    print(
        "Building historical V5 component store..."
    )

    df = (
        ov.build_component_store()
        .copy()
    )

    required = [
        "match_id",
        "date",
        "season",
        "league_code",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Historical V5 component store missing columns: "
            + str(
                missing
            )
        )

    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        errors="coerce",
    )

    # ========================================================
    # TRANSITION FLAGS
    # ========================================================

    print(
        "Building historical transition flags..."
    )

    df = add_transition_flags(
        df
    )

    print()
    print(
        "Historical rows:",
        len(df),
    )

    print(
        "Transition rows:",
        int(
            df[
                "transition_applied"
            ].sum()
        ),
    )

    print(
        "Home promoted:",
        int(
            df[
                "home_promoted"
            ].sum()
        ),
    )

    print(
        "Away promoted:",
        int(
            df[
                "away_promoted"
            ].sum()
        ),
    )

    print(
        "Home relegated:",
        int(
            df[
                "home_relegated"
            ].sum()
        ),
    )

    print(
        "Away relegated:",
        int(
            df[
                "away_relegated"
            ].sum()
        ),
    )

    # ========================================================
    # RAW V5
    # ========================================================

    print()
    print(
        "Building raw V5 lambdas..."
    )

    (
        raw_home_lambda,
        raw_away_lambda,
    ) = ov.build_lambdas(
        df,
        0.80,
    )

    raw_probs = (
        ov.calculate_1x2_probs(
            np.asarray(
                raw_home_lambda,
                dtype=float,
            ),

            np.asarray(
                raw_away_lambda,
                dtype=float,
            ),
        )
    )

    # ========================================================
    # TRANSITION V5
    # ========================================================

    print(
        "Applying live transition rules..."
    )

    (
        transition_home_lambda,
        transition_away_lambda,
    ) = apply_transition(
        df,
        raw_home_lambda,
        raw_away_lambda,
    )

    transition_probs = (
        ov.calculate_1x2_probs(
            transition_home_lambda,
            transition_away_lambda,
        )
    )

    # ========================================================
    # SEASONS
    # ========================================================

    season = (
        df[
            "season"
        ]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.zfill(4)
    )

    df[
        "season_norm"
    ] = season

    # --------------------------------------------------------
    # Evaluate the years we previously used as validation
    # and final holdout.
    # --------------------------------------------------------

    season_masks = {
        "2023/24":
            season.eq(
                "2324"
            ),

        "2024/25":
            season.eq(
                "2425"
            ),

        "2025/26":
            season.eq(
                "2526"
            ),
    }

    results = []

    # ========================================================
    # FULL SEASON RESULTS
    # ========================================================

    for label, mask in season_masks.items():

        results.append(
            evaluate_subset(
                df,
                raw_probs,
                transition_probs,
                mask,
                label,
            )
        )

        results.append(
            evaluate_subset(
                df,
                raw_probs,
                transition_probs,
                (
                    mask
                    &
                    df[
                        "transition_applied"
                    ].eq(1)
                ),
                label
                +
                " TRANSITIONS",
            )
        )

    # ========================================================
    # TRANSITION TYPE BREAKDOWN
    # ========================================================

    transition_types = {
        "ALL TRANSITIONS":
            df[
                "transition_applied"
            ].eq(1),

        "HOME PROMOTED":
            df[
                "home_promoted"
            ].eq(1),

        "AWAY PROMOTED":
            df[
                "away_promoted"
            ].eq(1),

        "HOME RELEGATED":
            df[
                "home_relegated"
            ].eq(1),

        "AWAY RELEGATED":
            df[
                "away_relegated"
            ].eq(1),
    }

    for label, mask in transition_types.items():

        results.append(
            evaluate_subset(
                df,
                raw_probs,
                transition_probs,
                mask,
                label,
            )
        )

    results = pd.DataFrame(
        results
    )

    # ========================================================
    # DELTAS
    # ========================================================

    results[
        "accuracy_delta"
    ] = (
        results[
            "transition_accuracy"
        ]
        -
        results[
            "raw_accuracy"
        ]
    )

    results[
        "log_loss_delta"
    ] = (
        results[
            "transition_log_loss"
        ]
        -
        results[
            "raw_log_loss"
        ]
    )

    results[
        "brier_delta"
    ] = (
        results[
            "transition_brier"
        ]
        -
        results[
            "raw_brier"
        ]
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 150)
    print("RAW V5 vs LIVE TRANSITION V5")
    print("=" * 150)

    display = results.copy()

    for col in [
        "raw_accuracy",
        "transition_accuracy",
        "accuracy_delta",
    ]:
        display[
            col
        ] *= 100.0

    print(
        display[
            [
                "segment",
                "games",

                "raw_accuracy",
                "transition_accuracy",
                "accuracy_delta",

                "raw_log_loss",
                "transition_log_loss",
                "log_loss_delta",

                "raw_brier",
                "transition_brier",
                "brier_delta",
            ]
        ]
        .round(5)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    out = (
        ROOT
        / "data"
        / "processed"
        / "backtest_transition_v5.csv"
    )

    results.to_csv(
        out,
        index=False,
    )

    print()
    print(
        "Saved:",
        out,
    )


if __name__ == "__main__":
    main()