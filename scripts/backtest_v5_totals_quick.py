from pathlib import Path
from math import exp

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PRED_FILE = (
    ROOT
    / "data"
    / "processed"
    / "overall_venue_v5_predictions.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "v5_totals_quick_backtest.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Test only thresholds. We are NOT selecting a live threshold
# until we inspect season-by-season stability.
EDGE_THRESHOLDS = [
    0.00,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.10,
    0.12,
    0.15,
]

# Football-data league codes.
LEAGUE_FILES = {
    "Premier League": "E0",
    "Bundesliga": "D1",
}

SEASONS = [
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
    "2526",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_team(value):

    if pd.isna(value):
        return ""

    return (
        str(value)
        .lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .replace("&", "and")
        .replace("  ", " ")
        .strip()
    )


TEAM_ALIASES = {
    "man city":
        "manchester city",

    "man united":
        "manchester united",

    "newcastle":
        "newcastle united",

    "tottenham":
        "tottenham hotspur",

    "west ham":
        "west ham united",

    "wolves":
        "wolverhampton wanderers",

    "nott'm forest":
        "nottingham forest",

    "brighton":
        "brighton and hove albion",

    "leicester":
        "leicester city",

    "leeds":
        "leeds united",

    "sheffield utd":
        "sheffield united",

    "west brom":
        "west bromwich albion",

    "bayern munich":
        "bayern munich",

    "dortmund":
        "borussia dortmund",

    "leverkusen":
        "bayer leverkusen",

    "ein frankfurt":
        "eintracht frankfurt",

    "mgladbach":
        "borussia monchengladbach",

    "mainz":
        "mainz 05",

    "koln":
        "fc koln",

    "fc koln":
        "fc koln",

    "stuttgart":
        "vfb stuttgart",

    "freiburg":
        "sc freiburg",

    "hoffenheim":
        "tsg hoffenheim",
}


def canonical_team(value):

    x = normalize_team(
        value
    )

    return TEAM_ALIASES.get(
        x,
        x,
    )


def season_string(series):

    return (
        series
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.zfill(4)
    )


def over_2_5_probability(
    home_lambda,
    away_lambda,
):

    total_lambda = (
        home_lambda
        +
        away_lambda
    )

    p0 = exp(
        -total_lambda
    )

    p1 = (
        p0
        *
        total_lambda
    )

    p2 = (
        p1
        *
        total_lambda
        /
        2.0
    )

    under = (
        p0
        +
        p1
        +
        p2
    )

    return 1.0 - under


# ============================================================
# LOAD HISTORICAL TOTALS ODDS
# ============================================================

def load_market_data():

    frames = []

    for season in SEASONS:

        for league, code in LEAGUE_FILES.items():

            path = (
                ROOT
                / "data"
                / "raw"
                / f"{season}_{code}.csv"
            )

            if not path.exists():
                continue

            df = pd.read_csv(
                path,
                low_memory=False,
            )

            # Older files may not contain totals odds.
            required = [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
            ]

            if any(
                c not in df.columns
                for c in required
            ):
                continue

            # ------------------------------------------------
            # Prefer average market odds.
            # Fall back to Bet365 if needed.
            # ------------------------------------------------

            if (
                "Avg>2.5" in df.columns
                and
                "Avg<2.5" in df.columns
            ):

                over_col = "Avg>2.5"
                under_col = "Avg<2.5"

            elif (
                "B365>2.5" in df.columns
                and
                "B365<2.5" in df.columns
            ):

                over_col = "B365>2.5"
                under_col = "B365<2.5"

            else:
                continue

            x = df[
                [
                    "Date",
                    "HomeTeam",
                    "AwayTeam",
                    "FTHG",
                    "FTAG",
                    over_col,
                    under_col,
                ]
            ].copy()

            x = x.rename(
                columns={
                    "Date":
                        "market_date",

                    "HomeTeam":
                        "market_home",

                    "AwayTeam":
                        "market_away",

                    "FTHG":
                        "home_goals",

                    "FTAG":
                        "away_goals",

                    over_col:
                        "over_odds",

                    under_col:
                        "under_odds",
                }
            )

            x[
                "season"
            ] = season

            x[
                "league"
            ] = league

            x[
                "market_date"
            ] = pd.to_datetime(
                x[
                    "market_date"
                ],
                dayfirst=True,
                errors="coerce",
            ).dt.date

            x[
                "home_key"
            ] = x[
                "market_home"
            ].map(
                canonical_team
            )

            x[
                "away_key"
            ] = x[
                "market_away"
            ].map(
                canonical_team
            )

            x[
                "over_odds"
            ] = pd.to_numeric(
                x[
                    "over_odds"
                ],
                errors="coerce",
            )

            x[
                "under_odds"
            ] = pd.to_numeric(
                x[
                    "under_odds"
                ],
                errors="coerce",
            )

            x[
                "home_goals"
            ] = pd.to_numeric(
                x[
                    "home_goals"
                ],
                errors="coerce",
            )

            x[
                "away_goals"
            ] = pd.to_numeric(
                x[
                    "away_goals"
                ],
                errors="coerce",
            )

            frames.append(
                x
            )

    if not frames:

        raise ValueError(
            "No historical O/U 2.5 market data found."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# LOAD V5 PREDICTIONS
# ============================================================

def load_predictions():

    if not PRED_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{PRED_FILE}"
        )

    df = pd.read_csv(
        PRED_FILE,
        low_memory=False,
    )

    required = [
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
        "home_lambda_v5",
        "away_lambda_v5",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Prediction file missing:\n"
            +
            "\n".join(
                missing
            )
        )

    df[
        "season"
    ] = season_string(
        df[
            "season"
        ]
    )

    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        errors="coerce",
    ).dt.date

    df[
        "home_key"
    ] = df[
        "home_team"
    ].map(
        canonical_team
    )

    df[
        "away_key"
    ] = df[
        "away_team"
    ].map(
        canonical_team
    )

    df[
        "home_lambda_v5"
    ] = pd.to_numeric(
        df[
            "home_lambda_v5"
        ],
        errors="coerce",
    )

    df[
        "away_lambda_v5"
    ] = pd.to_numeric(
        df[
            "away_lambda_v5"
        ],
        errors="coerce",
    )

    df = df[
        df[
            "league"
        ].isin(
            LEAGUE_FILES
        )
    ].copy()

    return df


# ============================================================
# MATCH PREDICTIONS TO ODDS
# ============================================================

def build_dataset():

    pred = load_predictions()

    market = load_market_data()

    merged = pred.merge(
        market,
        left_on=[
            "season",
            "league",
            "date",
            "home_key",
            "away_key",
        ],
        right_on=[
            "season",
            "league",
            "market_date",
            "home_key",
            "away_key",
        ],
        how="inner",
        suffixes=(
            "",
            "_market",
        ),
    )

    merged = merged.dropna(
        subset=[
            "home_lambda_v5",
            "away_lambda_v5",
            "over_odds",
            "under_odds",
            "home_goals",
            "away_goals",
        ]
    ).copy()

    merged = merged[
        (
            merged[
                "over_odds"
            ]
            >
            1.0
        )
        &
        (
            merged[
                "under_odds"
            ]
            >
            1.0
        )
    ].copy()

    # ========================================================
    # V5 TOTALS PROBABILITIES
    # ========================================================

    merged[
        "p_over_v5"
    ] = [
        over_2_5_probability(
            h,
            a,
        )
        for h, a in zip(
            merged[
                "home_lambda_v5"
            ],
            merged[
                "away_lambda_v5"
            ],
        )
    ]

    merged[
        "p_under_v5"
    ] = (
        1.0
        -
        merged[
            "p_over_v5"
        ]
    )

    # ========================================================
    # NO-VIG MARKET PROBABILITIES
    # ========================================================

    raw_over = (
        1.0
        /
        merged[
            "over_odds"
        ]
    )

    raw_under = (
        1.0
        /
        merged[
            "under_odds"
        ]
    )

    vig_sum = (
        raw_over
        +
        raw_under
    )

    merged[
        "market_p_over"
    ] = (
        raw_over
        /
        vig_sum
    )

    merged[
        "market_p_under"
    ] = (
        raw_under
        /
        vig_sum
    )

    # ========================================================
    # EDGES / EV
    # ========================================================

    merged[
        "over_edge"
    ] = (
        merged[
            "p_over_v5"
        ]
        -
        merged[
            "market_p_over"
        ]
    )

    merged[
        "under_edge"
    ] = (
        merged[
            "p_under_v5"
        ]
        -
        merged[
            "market_p_under"
        ]
    )

    merged[
        "over_ev"
    ] = (
        merged[
            "p_over_v5"
        ]
        *
        merged[
            "over_odds"
        ]
        -
        1.0
    )

    merged[
        "under_ev"
    ] = (
        merged[
            "p_under_v5"
        ]
        *
        merged[
            "under_odds"
        ]
        -
        1.0
    )

    merged[
        "actual_total"
    ] = (
        merged[
            "home_goals"
        ]
        +
        merged[
            "away_goals"
        ]
    )

    return merged


# ============================================================
# BUILD ONE-BET-PER-MATCH CANDIDATES
# ============================================================

def build_candidates(
    df,
):

    over = df.copy()

    over[
        "selection"
    ] = "OVER"

    over[
        "model_probability"
    ] = over[
        "p_over_v5"
    ]

    over[
        "market_probability"
    ] = over[
        "market_p_over"
    ]

    over[
        "edge"
    ] = over[
        "over_edge"
    ]

    over[
        "odds"
    ] = over[
        "over_odds"
    ]

    over[
        "model_ev"
    ] = over[
        "over_ev"
    ]

    over[
        "won"
    ] = (
        over[
            "actual_total"
        ]
        >
        2.5
    ).astype(
        int
    )

    under = df.copy()

    under[
        "selection"
    ] = "UNDER"

    under[
        "model_probability"
    ] = under[
        "p_under_v5"
    ]

    under[
        "market_probability"
    ] = under[
        "market_p_under"
    ]

    under[
        "edge"
    ] = under[
        "under_edge"
    ]

    under[
        "odds"
    ] = under[
        "under_odds"
    ]

    under[
        "model_ev"
    ] = under[
        "under_ev"
    ]

    under[
        "won"
    ] = (
        under[
            "actual_total"
        ]
        <
        2.5
    ).astype(
        int
    )

    candidates = pd.concat(
        [
            over,
            under,
        ],
        ignore_index=True,
    )

    candidates[
        "profit"
    ] = np.where(
        candidates[
            "won"
        ]
        ==
        1,
        candidates[
            "odds"
        ]
        -
        1.0,
        -1.0,
    )

    return candidates


# ============================================================
# THRESHOLD TEST
# ============================================================

def evaluate_threshold(
    candidates,
    threshold,
):

    x = candidates[
        candidates[
            "edge"
        ]
        >=
        threshold
    ].copy()

    # One totals bet per fixture.
    x = (
        x
        .sort_values(
            [
                "match_id",
                "edge",
                "model_ev",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "match_id",
            ],
            keep="first",
        )
    )

    n = len(
        x
    )

    if n == 0:

        return {
            "threshold":
                threshold,

            "bets":
                0,

            "wins":
                0,

            "win_rate":
                np.nan,

            "avg_odds":
                np.nan,

            "avg_edge":
                np.nan,

            "avg_ev":
                np.nan,

            "profit":
                0.0,

            "roi":
                np.nan,
        }

    profit = float(
        x[
            "profit"
        ].sum()
    )

    return {
        "threshold":
            threshold,

        "bets":
            n,

        "wins":
            int(
                x[
                    "won"
                ].sum()
            ),

        "win_rate":
            x[
                "won"
            ].mean(),

        "avg_odds":
            x[
                "odds"
            ].mean(),

        "avg_edge":
            x[
                "edge"
            ].mean(),

        "avg_ev":
            x[
                "model_ev"
            ].mean(),

        "profit":
            profit,

        "roi":
            profit
            /
            n,
    }


# ============================================================
# SEASON TEST FOR INTERESTING THRESHOLDS
# ============================================================

def season_breakdown(
    candidates,
    threshold,
):

    x = candidates[
        candidates[
            "edge"
        ]
        >=
        threshold
    ].copy()

    x = (
        x
        .sort_values(
            [
                "match_id",
                "edge",
                "model_ev",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .drop_duplicates(
            "match_id"
        )
    )

    rows = []

    for season, g in x.groupby(
        "season"
    ):

        n = len(
            g
        )

        profit = float(
            g[
                "profit"
            ].sum()
        )

        rows.append(
            {
                "season":
                    season,

                "bets":
                    n,

                "wins":
                    int(
                        g[
                            "won"
                        ].sum()
                    ),

                "win_rate":
                    g[
                        "won"
                    ].mean(),

                "avg_odds":
                    g[
                        "odds"
                    ].mean(),

                "profit":
                    profit,

                "roi":
                    profit
                    /
                    n,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 120
    )

    print(
        "QUICK V5 OVER/UNDER 2.5 BACKTEST"
    )

    print(
        "=" * 120
    )

    print()
    print(
        "Historical market: average O/U 2.5 odds"
    )

    print(
        "Market probabilities: no-vig"
    )

    print(
        "Stake: flat 1 unit"
    )

    print(
        "Maximum: one totals bet per match"
    )

    df = build_dataset()

    print()
    print(
        "Matched historical games:",
        f"{len(df):,}",
    )

    print(
        "Date range:",
        df[
            "date"
        ].min(),
        "->",
        df[
            "date"
        ].max(),
    )

    print()

    print(
        "By league:"
    )

    print(
        df[
            "league"
        ]
        .value_counts()
        .to_string()
    )

    candidates = build_candidates(
        df
    )

    # ========================================================
    # THRESHOLDS
    # ========================================================

    rows = []

    for threshold in EDGE_THRESHOLDS:

        rows.append(
            evaluate_threshold(
                candidates,
                threshold,
            )
        )

    results = pd.DataFrame(
        rows
    )

    display = results.copy()

    for c in [
        "threshold",
        "win_rate",
        "avg_edge",
        "avg_ev",
        "roi",
    ]:

        display[
            c
        ] *= 100.0

    print()
    print(
        "=" * 120
    )

    print(
        "EDGE THRESHOLD RESULTS"
    )

    print(
        "=" * 120
    )

    print()

    print(
        display
        .round(
            3
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SHOW 5%, 7%, 8%, 10% BY SEASON
    # ========================================================

    for threshold in [
        0.05,
        0.07,
        0.08,
        0.10,
    ]:

        season = season_breakdown(
            candidates,
            threshold,
        )

        print()
        print(
            "=" * 120
        )

        print(
            f"BY SEASON — EDGE >= {threshold:.0%}"
        )

        print(
            "=" * 120
        )

        if season.empty:

            print(
                "No bets."
            )

            continue

        show = season.copy()

        for c in [
            "win_rate",
            "roi",
        ]:

            show[
                c
            ] *= 100.0

        print(
            show
            .round(
                3
            )
            .to_string(
                index=False
            )
        )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()