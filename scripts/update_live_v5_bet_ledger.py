from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EV_BOARD_FILE = (
    ROOT
    / "data"
    / "live"
    / "v5_live_ev_board.csv"
)

ODDS_HISTORY_FILE = (
    ROOT
    / "data"
    / "live"
    / "odds_h2h_history.csv"
)

LEDGER_FILE = (
    ROOT
    / "data"
    / "live"
    / "v5_live_bet_ledger.csv"
)


# ============================================================
# LOCKED LIVE STRATEGY
# ============================================================

RAW_V5_EDGE_THRESHOLD = 0.16

ONE_BET_PER_MATCH = True

FREEZE_FIRST_QUALIFYING_SIGNAL = True


# ============================================================
# STAKING
# ============================================================

STARTING_BANKROLL = 100.0

UNIT_SIZE_FRACTION = 0.01

FLAT_STAKE_UNITS = 1.0

# Kelly is recorded for research only.
USE_KELLY_STAKING = False


# ============================================================
# ACTIVE LEAGUE UNIVERSE
# ============================================================

VALIDATED_LEAGUES = {
    "Premier League",
    "Bundesliga",
}

EXPANDED_LIVE_LEAGUES = {
    "Championship",
    "League One",
    "League Two",
    "La Liga",
    "Belgian Pro League",
    "2. Bundesliga",
}

ACTIVE_LIVE_LEAGUES = (
    VALIDATED_LEAGUES
    |
    EXPANDED_LIVE_LEAGUES
)


# ============================================================
# LEDGER SCHEMA
# ============================================================

LEDGER_COLUMNS = [
    "ledger_id",
    "match_id",
    "event_id",

    "date",
    "commence_time",

    "league",
    "home_team",
    "away_team",

    "deployment_tier",
    "deployment_status",
    "eligibility_reason",
    "history_type",

    "bet_side",

    "model_probability",
    "signal_market_probability",
    "signal_edge",
    "signal_ev",

    "bet_odds",
    "bet_book",

    "bet_snapshot_time",
    "bet_recorded_time",

    "kelly_full",
    "kelly_fraction",

    "stake_units",
    "stake_bankroll_fraction",
    "stake_amount",

    "bankroll_before",
    "bankroll_after",

    "closing_odds",
    "closing_market_probability",
    "closing_snapshot_time",

    "price_clv",
    "probability_clv",
    "beat_close",

    "actual_outcome",
    "won",
    "profit_units",

    "status",
]


# ============================================================
# HELPERS
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_float(
    value,
):

    try:

        x = float(
            value
        )

        if np.isfinite(
            x
        ):
            return x

    except (
        TypeError,
        ValueError,
    ):
        pass

    return np.nan


def normalize_side(
    value,
):

    if pd.isna(
        value
    ):
        return None

    side = (
        str(
            value
        )
        .strip()
        .upper()
    )

    if side in {
        "HOME",
        "DRAW",
        "AWAY",
    }:

        return side

    return None


def side_probability_column(
    side,
):

    return {
        "HOME":
            "p_home_v5",

        "DRAW":
            "p_draw_v5",

        "AWAY":
            "p_away_v5",
    }[
        side
    ]


def side_market_probability_column(
    side,
):

    return {
        "HOME":
            "market_p_home",

        "DRAW":
            "market_p_draw",

        "AWAY":
            "market_p_away",
    }[
        side
    ]


def side_odds_column(
    side,
):

    return {
        "HOME":
            "best_home_odds",

        "DRAW":
            "best_draw_odds",

        "AWAY":
            "best_away_odds",
    }[
        side
    ]


def side_book_column(
    side,
):

    return {
        "HOME":
            "best_home_book",

        "DRAW":
            "best_draw_book",

        "AWAY":
            "best_away_book",
    }[
        side
    ]


def side_kelly_full_column(
    side,
):

    return {
        "HOME":
            "home_kelly_full",

        "DRAW":
            "draw_kelly_full",

        "AWAY":
            "away_kelly_full",
    }[
        side
    ]


def side_kelly_column(
    side,
):

    return {
        "HOME":
            "home_kelly",

        "DRAW":
            "draw_kelly",

        "AWAY":
            "away_kelly",
    }[
        side
    ]


def odds_history_side_column(
    side,
):

    return {
        "HOME":
            "home_odds",

        "DRAW":
            "draw_odds",

        "AWAY":
            "away_odds",
    }[
        side
    ]


# ============================================================
# STAKE HELPERS
# ============================================================

def calculate_stake_amount(
    bankroll_before,
):

    if not np.isfinite(
        bankroll_before
    ):

        return np.nan

    unit_amount = (
        bankroll_before
        *
        UNIT_SIZE_FRACTION
    )

    return (
        unit_amount
        *
        FLAT_STAKE_UNITS
    )


# ============================================================
# LOAD LEDGER
# ============================================================

def load_ledger():

    if not LEDGER_FILE.exists():

        return pd.DataFrame(
            columns=LEDGER_COLUMNS
        )

    ledger = pd.read_csv(
        LEDGER_FILE,
        low_memory=False,
    )

    for col in LEDGER_COLUMNS:

        if col not in ledger.columns:

            ledger[
                col
            ] = np.nan

    return ledger[
        LEDGER_COLUMNS
    ].copy()


# ============================================================
# VALIDATE EV BOARD
# ============================================================

def validate_ev_board(
    df,
):

    required = {
        "match_id",
        "date",
        "league",
        "home_team",
        "away_team",

        "odds_matched",
        "odds_event_id",
        "odds_snapshot_time",

        "p_home_v5",
        "p_draw_v5",
        "p_away_v5",

        "market_p_home",
        "market_p_draw",
        "market_p_away",

        "best_home_odds",
        "best_draw_odds",
        "best_away_odds",

        "best_home_book",
        "best_draw_book",
        "best_away_book",

        "home_kelly_full",
        "draw_kelly_full",
        "away_kelly_full",

        "home_kelly",
        "draw_kelly",
        "away_kelly",
    }

    missing = sorted(
        required
        -
        set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            "EV board missing required columns:\n"
            +
            "\n".join(
                missing
            )
        )


# ============================================================
# ACTIVE LEAGUE FILTER
# ============================================================

def filter_active_leagues(
    board,
):

    out = board.copy()

    out[
        "league"
    ] = (
        out[
            "league"
        ]
        .astype(str)
        .str.strip()
    )

    return out[
        out[
            "league"
        ]
        .isin(
            ACTIVE_LIVE_LEAGUES
        )
    ].copy()


# ============================================================
# FIND TRUE BEST RAW V5 EDGE
# ============================================================

def choose_qualifying_side(
    row,
):

    """
    Frozen Betting Signal V1.

    RAW V5 EDGE:
        V5 probability
        -
        no-vig market probability

    Qualification:
        edge >= 16%

    We deliberately do NOT trust:
        paper_bet
        bet_status
        best_side
        best_edge
        best_ev

    The signal is reconstructed directly.
    """

    # --------------------------------------------------------
    # PRE-MATCH SAFETY GUARD
    #
    # Defense-in-depth:
    # even if an upstream EV-board bug allows in-play odds
    # through, the permanent model ledger must never freeze
    # a signal at or after T-1 minute.
    # --------------------------------------------------------

    snapshot_time = pd.to_datetime(
        row.get(
            "odds_snapshot_time"
        ),
        utc=True,
        errors="coerce",
    )

    commence_time = pd.to_datetime(
        row.get(
            "odds_commence_time"
        ),
        utc=True,
        errors="coerce",
    )

    if (
        pd.isna(snapshot_time)
        or
        pd.isna(commence_time)
        or
        snapshot_time
        >=
        (
            commence_time
            -
            pd.Timedelta(minutes=1)
        )
    ):
        return None

    candidates = []

    for side in (
        "HOME",
        "DRAW",
        "AWAY",
    ):

        model_p = safe_float(
            row.get(
                side_probability_column(
                    side
                )
            )
        )

        market_p = safe_float(
            row.get(
                side_market_probability_column(
                    side
                )
            )
        )

        odds = safe_float(
            row.get(
                side_odds_column(
                    side
                )
            )
        )

        if (
            not np.isfinite(
                model_p
            )
            or
            not np.isfinite(
                market_p
            )
            or
            not np.isfinite(
                odds
            )
            or
            odds <= 1.0
        ):

            continue

        raw_edge = (
            model_p
            -
            market_p
        )

        raw_ev = (
            model_p
            *
            odds
            -
            1.0
        )

        candidates.append(
            {
                "side":
                    side,

                "model_probability":
                    model_p,

                "market_probability":
                    market_p,

                "edge":
                    raw_edge,

                "ev":
                    raw_ev,

                "odds":
                    odds,

                "book":
                    row.get(
                        side_book_column(
                            side
                        )
                    ),

                "kelly_full":
                    safe_float(
                        row.get(
                            side_kelly_full_column(
                                side
                            )
                        )
                    ),

                "kelly_fraction":
                    safe_float(
                        row.get(
                            side_kelly_column(
                                side
                            )
                        )
                    ),
            }
        )

    if not candidates:

        return None

    # --------------------------------------------------------
    # Select highest raw V5 edge.
    #
    # Tie-break:
    # highest raw EV.
    # --------------------------------------------------------

    candidates = sorted(
        candidates,
        key=lambda x: (
            x[
                "edge"
            ],
            x[
                "ev"
            ],
        ),
        reverse=True,
    )

    best = candidates[
        0
    ]

    if (
        best[
            "edge"
        ]
        <
        RAW_V5_EDGE_THRESHOLD
    ):

        return None

    return best


# ============================================================
# CLASSIFY LIVE SIGNAL
# ============================================================

def classify_live_signal(
    row,
):

    league = (
        str(
            row.get(
                "league",
                ""
            )
        )
        .strip()
    )

    home_source = (
        str(
            row.get(
                "home_history_source",
                ""
            )
        )
        .strip()
        .upper()
    )

    away_source = (
        str(
            row.get(
                "away_history_source",
                ""
            )
        )
        .strip()
        .upper()
    )

    transition_applied = safe_float(
        row.get(
            "transition_applied"
        )
    )

    prediction_provider = (
        str(
            row.get(
                "prediction_provider",
                ""
            )
        )
        .strip()
        .upper()
    )

    # ========================================================
    # DEPLOYMENT STATUS
    # ========================================================

    if league in VALIDATED_LEAGUES:

        deployment_status = (
            "VALIDATED"
        )

        eligibility_reason = (
            "HISTORICAL_WALKFORWARD"
        )

    elif league in EXPANDED_LIVE_LEAGUES:

        deployment_status = (
            "EXPANDED_LIVE"
        )

        eligibility_reason = (
            "EXPANDED_LIVE_UNIVERSE"
        )

    else:

        deployment_status = (
            "PAPER_ONLY"
        )

        eligibility_reason = (
            "UNSUPPORTED_LIVE_LEAGUE"
        )

    # ========================================================
    # HISTORY TYPE
    # ========================================================

    sources = (
        home_source
        +
        " "
        +
        away_source
    )

    if (
        np.isfinite(
            transition_applied
        )
        and
        transition_applied
        ==
        1
    ):

        history_type = (
            "TRANSITION"
        )

    elif (
        "TRANSFER"
        in
        sources
    ):

        history_type = (
            "TRANSFERRED"
        )

    elif (
        home_source
        ==
        "SAME_LEAGUE"
        and
        away_source
        ==
        "SAME_LEAGUE"
    ):

        history_type = (
            "SAME_LEAGUE"
        )

    elif (
        prediction_provider
        ==
        "CORE_V5"
    ):

        history_type = (
            "CORE"
        )

    else:

        history_type = (
            "OTHER"
        )

    return (
        deployment_status,
        eligibility_reason,
        history_type,
    )


# ============================================================
# CREATE LEDGER ROW
# ============================================================

def build_ledger_row(
    row,
    signal,
):

    match_id = str(
        row[
            "match_id"
        ]
    )

    side = signal[
        "side"
    ]

    (
        deployment_status,
        eligibility_reason,
        history_type,
    ) = classify_live_signal(
        row
    )

    ledger_id = (
        f"{match_id}_{side}"
    )

    return {
        "ledger_id":
            ledger_id,

        "match_id":
            match_id,

        "event_id":
            row.get(
                "odds_event_id",
                np.nan,
            ),

        "date":
            row.get(
                "date",
                np.nan,
            ),

        "commence_time":
            row.get(
                "commence_time",
                np.nan,
            ),

        "league":
            row.get(
                "league",
                np.nan,
            ),

        "home_team":
            row.get(
                "home_team",
                np.nan,
            ),

        "away_team":
            row.get(
                "away_team",
                np.nan,
            ),

        "deployment_tier":
            row.get(
                "deployment_tier",
                np.nan,
            ),

        "deployment_status":
            deployment_status,

        "eligibility_reason":
            eligibility_reason,

        "history_type":
            history_type,

        "bet_side":
            side,

        "model_probability":
            signal[
                "model_probability"
            ],

        "signal_market_probability":
            signal[
                "market_probability"
            ],

        "signal_edge":
            signal[
                "edge"
            ],

        "signal_ev":
            signal[
                "ev"
            ],

        "bet_odds":
            signal[
                "odds"
            ],

        "bet_book":
            signal[
                "book"
            ],

        "bet_snapshot_time":
            row.get(
                "odds_snapshot_time",
                np.nan,
            ),

        "bet_recorded_time":
            utc_now(),

        "kelly_full":
            signal[
                "kelly_full"
            ],

        "kelly_fraction":
            signal[
                "kelly_fraction"
            ],

        "stake_units":
            FLAT_STAKE_UNITS,

        "stake_bankroll_fraction":
            UNIT_SIZE_FRACTION,

        "stake_amount":
            np.nan,

        "bankroll_before":
            np.nan,

        "bankroll_after":
            np.nan,

        "closing_odds":
            np.nan,

        "closing_market_probability":
            np.nan,

        "closing_snapshot_time":
            np.nan,

        "price_clv":
            np.nan,

        "probability_clv":
            np.nan,

        "beat_close":
            np.nan,

        "actual_outcome":
            np.nan,

        "won":
            np.nan,

        "profit_units":
            np.nan,

        "status":
            "OPEN",
    }


# ============================================================
# ADD NEW QUALIFYING BETS
# ============================================================

def add_new_bets(
    board,
    ledger,
):

    existing_matches = set(
        ledger[
            "match_id"
        ]
        .dropna()
        .astype(str)
    )

    new_rows = []

    board = filter_active_leagues(
        board
    )

    if "odds_matched" in board.columns:

        board = board[
            board[
                "odds_matched"
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
        ].copy()

    for _, row in board.iterrows():

        match_id = str(
            row[
                "match_id"
            ]
        )

        if (
            ONE_BET_PER_MATCH
            and
            match_id
            in
            existing_matches
        ):

            continue

        signal = choose_qualifying_side(
            row
        )

        if signal is None:

            continue

        ledger_row = build_ledger_row(
            row,
            signal,
        )

        new_rows.append(
            ledger_row
        )

        existing_matches.add(
            match_id
        )

    if new_rows:

        new_df = pd.DataFrame(
            new_rows
        )

        ledger = pd.concat(
            [
                ledger,
                new_df,
            ],
            ignore_index=True,
        )

    return (
        ledger,
        new_rows,
    )


# ============================================================
# UPDATE CLOSING PRICES / CLV
# ============================================================

def update_closing_prices(
    ledger,
    odds_history,
):

    """
    Closing line:

    1. Match must have kicked off.
    2. Use latest H2H snapshot strictly BEFORE kickoff.
    3. Calculate average consensus no-vig close probability.
    4. Compare frozen bet odds with average closing odds.

    price_clv:
        bet_odds / closing_odds - 1

    Positive:
        our bet price beat the close.
    """

    if ledger.empty:

        return ledger

    if odds_history.empty:

        return ledger

    hist = odds_history.copy()

    required = {
        "match_id",
        "snapshot_time",
        "commence_time",
        "home_odds",
        "draw_odds",
        "away_odds",
    }

    missing = (
        required
        -
        set(
            hist.columns
        )
    )

    if missing:

        print()
        print(
            "WARNING: Odds history missing "
            "required CLV fields:"
        )

        for col in sorted(
            missing
        ):

            print(
                " ",
                col,
            )

        return ledger

    hist[
        "snapshot_time"
    ] = pd.to_datetime(
        hist[
            "snapshot_time"
        ],
        utc=True,
        errors="coerce",
    )

    hist[
        "commence_time"
    ] = pd.to_datetime(
        hist[
            "commence_time"
        ],
        utc=True,
        errors="coerce",
    )

    now = pd.Timestamp.now(
        tz="UTC"
    )

    for idx, bet in ledger.iterrows():

        status = (
            str(
                bet.get(
                    "status",
                    ""
                )
            )
            .strip()
            .upper()
        )

        if status not in {
            "OPEN",
            "CLOSED_LINE",
        }:

            continue

        match_id = str(
            bet[
                "match_id"
            ]
        )

        side = normalize_side(
            bet[
                "bet_side"
            ]
        )

        if side is None:

            continue

        match_hist = hist[
            hist[
                "match_id"
            ]
            .astype(str)
            .eq(
                match_id
            )
        ].copy()

        if match_hist.empty:

            continue

        commence_values = (
            match_hist[
                "commence_time"
            ]
            .dropna()
        )

        if commence_values.empty:

            continue

        commence_time = (
            commence_values.iloc[
                0
            ]
        )

        ledger.at[
            idx,
            "commence_time"
        ] = (
            commence_time
            .isoformat()
        )

        # ----------------------------------------------------
        # Do not create closing line before kickoff.
        # ----------------------------------------------------

        if (
            now
            <
            commence_time
        ):

            continue

        eligible = match_hist[
            match_hist[
                "snapshot_time"
            ]
            <
            commence_time
        ].copy()

        if eligible.empty:

            continue

        final_snapshot = (
            eligible[
                "snapshot_time"
            ]
            .max()
        )

        close = eligible[
            eligible[
                "snapshot_time"
            ]
            .eq(
                final_snapshot
            )
        ].copy()

        if close.empty:

            continue

        for col in (
            "home_odds",
            "draw_odds",
            "away_odds",
        ):

            close[
                col
            ] = pd.to_numeric(
                close[
                    col
                ],
                errors="coerce",
            )

        close = close[
            (
                close[
                    "home_odds"
                ]
                > 1.0
            )
            &
            (
                close[
                    "draw_odds"
                ]
                > 1.0
            )
            &
            (
                close[
                    "away_odds"
                ]
                > 1.0
            )
        ].copy()

        if close.empty:

            continue

        # ----------------------------------------------------
        # De-vig each bookmaker's final 1X2 market.
        # ----------------------------------------------------

        inv_home = (
            1.0
            /
            close[
                "home_odds"
            ]
        )

        inv_draw = (
            1.0
            /
            close[
                "draw_odds"
            ]
        )

        inv_away = (
            1.0
            /
            close[
                "away_odds"
            ]
        )

        totals = (
            inv_home
            +
            inv_draw
            +
            inv_away
        )

        close[
            "_nv_home"
        ] = (
            inv_home
            /
            totals
        )

        close[
            "_nv_draw"
        ] = (
            inv_draw
            /
            totals
        )

        close[
            "_nv_away"
        ] = (
            inv_away
            /
            totals
        )

        probability_column = {
            "HOME":
                "_nv_home",

            "DRAW":
                "_nv_draw",

            "AWAY":
                "_nv_away",
        }[
            side
        ]

        odds_column = (
            odds_history_side_column(
                side
            )
        )

        closing_market_probability = (
            close[
                probability_column
            ]
            .mean()
        )

        closing_odds = (
            close[
                odds_column
            ]
            .mean()
        )

        bet_odds = safe_float(
            bet[
                "bet_odds"
            ]
        )

        signal_market_probability = safe_float(
            bet[
                "signal_market_probability"
            ]
        )

        if (
            not np.isfinite(
                closing_odds
            )
            or
            closing_odds
            <=
            1.0
            or
            not np.isfinite(
                bet_odds
            )
            or
            not np.isfinite(
                closing_market_probability
            )
            or
            not np.isfinite(
                signal_market_probability
            )
        ):

            continue

        # ----------------------------------------------------
        # CLV
        # ----------------------------------------------------

        price_clv = (
            bet_odds
            /
            closing_odds
            -
            1.0
        )

        probability_clv = (
            closing_market_probability
            -
            signal_market_probability
        )

        beat_close = int(
            price_clv
            >
            0
        )

        ledger.at[
            idx,
            "closing_odds"
        ] = closing_odds

        ledger.at[
            idx,
            "closing_market_probability"
        ] = closing_market_probability

        ledger.at[
            idx,
            "closing_snapshot_time"
        ] = (
            final_snapshot
            .isoformat()
        )

        ledger.at[
            idx,
            "price_clv"
        ] = price_clv

        ledger.at[
            idx,
            "probability_clv"
        ] = probability_clv

        ledger.at[
            idx,
            "beat_close"
        ] = beat_close

        if status == "OPEN":

            ledger.at[
                idx,
                "status"
            ] = (
                "CLOSED_LINE"
            )

    return ledger


# ============================================================
# SETTLEMENT
# ============================================================

def settle_bet(
    ledger,
    match_id,
    actual_outcome,
):

    actual_outcome = normalize_side(
        actual_outcome
    )

    if actual_outcome is None:

        raise ValueError(
            "actual_outcome must be "
            "HOME, DRAW, or AWAY."
        )

    mask = (
        ledger[
            "match_id"
        ]
        .astype(str)
        .eq(
            str(
                match_id
            )
        )
    )

    if not mask.any():

        raise ValueError(
            f"No ledger bet found for "
            f"match_id={match_id}"
        )

    idx = ledger.index[
        mask
    ][
        0
    ]

    bet_side = normalize_side(
        ledger.at[
            idx,
            "bet_side"
        ]
    )

    odds = safe_float(
        ledger.at[
            idx,
            "bet_odds"
        ]
    )

    if (
        bet_side is None
        or
        not np.isfinite(
            odds
        )
        or
        odds <= 1.0
    ):

        raise ValueError(
            "Cannot settle bet because "
            "bet side or odds are invalid."
        )

    won = int(
        bet_side
        ==
        actual_outcome
    )

    if won:

        profit_units = (
            odds
            -
            1.0
        )

    else:

        profit_units = (
            -1.0
        )

    ledger.at[
        idx,
        "actual_outcome"
    ] = actual_outcome

    ledger.at[
        idx,
        "won"
    ] = won

    ledger.at[
        idx,
        "profit_units"
    ] = profit_units

    ledger.at[
        idx,
        "status"
    ] = "SETTLED"

    return ledger


# ============================================================
# BANKROLL
# ============================================================

def rebuild_bankroll(
    ledger,
):

    if ledger.empty:

        return ledger

    out = ledger.copy()

    out[
        "_recorded_sort"
    ] = pd.to_datetime(
        out[
            "bet_recorded_time"
        ],
        utc=True,
        errors="coerce",
    )

    out = (
        out
        .sort_values(
            [
                "_recorded_sort",
                "ledger_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    bankroll = float(
        STARTING_BANKROLL
    )

    for idx, row in out.iterrows():

        out.at[
            idx,
            "bankroll_before"
        ] = bankroll

        stake_units = safe_float(
            row.get(
                "stake_units"
            )
        )

        if not np.isfinite(
            stake_units
        ):

            stake_units = (
                FLAT_STAKE_UNITS
            )

            out.at[
                idx,
                "stake_units"
            ] = stake_units

        stake_fraction = safe_float(
            row.get(
                "stake_bankroll_fraction"
            )
        )

        if not np.isfinite(
            stake_fraction
        ):

            stake_fraction = (
                UNIT_SIZE_FRACTION
            )

            out.at[
                idx,
                "stake_bankroll_fraction"
            ] = stake_fraction

        stake_amount = (
            bankroll
            *
            stake_fraction
            *
            stake_units
        )

        out.at[
            idx,
            "stake_amount"
        ] = stake_amount

        status = (
            str(
                row.get(
                    "status",
                    ""
                )
            )
            .strip()
            .upper()
        )

        profit_units = safe_float(
            row.get(
                "profit_units"
            )
        )

        # ----------------------------------------------------
        # Only SETTLED bets change bankroll.
        # ----------------------------------------------------

        if (
            status
            ==
            "SETTLED"
            and
            np.isfinite(
                profit_units
            )
        ):

            bankroll_change = (
                stake_amount
                *
                profit_units
            )

            bankroll += (
                bankroll_change
            )

        out.at[
            idx,
            "bankroll_after"
        ] = bankroll

    out = out.drop(
        columns=[
            "_recorded_sort",
        ],
        errors="ignore",
    )

    return out


# ============================================================
# SAVE LEDGER
# ============================================================

def save_ledger(
    ledger,
):

    LEDGER_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ledger = ledger.copy()

    for col in LEDGER_COLUMNS:

        if col not in ledger.columns:

            ledger[
                col
            ] = np.nan

    ledger = ledger[
        LEDGER_COLUMNS
    ]

    ledger.to_csv(
        LEDGER_FILE,
        index=False,
    )


# ============================================================
# DISPLAY — NEW BETS
# ============================================================

def print_new_bets(
    new_rows,
):

    if not new_rows:

        return

    print()
    print(
        "=" * 150
    )

    print(
        "NEW QUALIFYING BETS"
    )

    print(
        "=" * 150
    )

    print()

    new_df = pd.DataFrame(
        new_rows
    )

    display_columns = [
        "date",
        "league",

        "deployment_status",
        "history_type",

        "home_team",
        "away_team",

        "bet_side",

        "model_probability",
        "signal_market_probability",
        "signal_edge",
        "signal_ev",

        "bet_odds",
        "bet_book",

        "stake_units",
    ]

    display = new_df[
        display_columns
    ].copy()

    for col in (
        "model_probability",
        "signal_market_probability",
        "signal_edge",
        "signal_ev",
    ):

        display[
            col
        ] = (
            pd.to_numeric(
                display[
                    col
                ],
                errors="coerce",
            )
            *
            100.0
        )

    print(
        display
        .round(
            {
                "model_probability":
                    2,

                "signal_market_probability":
                    2,

                "signal_edge":
                    2,

                "signal_ev":
                    2,

                "bet_odds":
                    3,

                "stake_units":
                    2,
            }
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# DISPLAY — DEPLOYMENT
# ============================================================

def print_deployment_summary(
    ledger,
):

    if ledger.empty:

        return

    print()
    print(
        "=" * 100
    )

    print(
        "DEPLOYMENT STATUS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        ledger[
            "deployment_status"
        ]
        .fillna(
            "UNKNOWN"
        )
        .value_counts()
        .to_string()
    )

    print()
    print(
        "HISTORY TYPE"
    )

    print()

    print(
        ledger[
            "history_type"
        ]
        .fillna(
            "UNKNOWN"
        )
        .value_counts()
        .to_string()
    )

    print()
    print(
        "LEAGUE BREAKDOWN"
    )

    print()

    print(
        ledger[
            "league"
        ]
        .fillna(
            "UNKNOWN"
        )
        .value_counts()
        .to_string()
    )


# ============================================================
# DISPLAY — CLV
# ============================================================

def print_clv_summary(
    ledger,
):

    clv = pd.to_numeric(
        ledger[
            "price_clv"
        ],
        errors="coerce",
    )

    completed_clv = (
        clv
        .dropna()
    )

    if completed_clv.empty:

        return

    print()
    print(
        "=" * 100
    )

    print(
        "CLV"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"CLV observations: "
        f"{len(completed_clv)}"
    )

    print(
        f"Average price CLV: "
        f"{completed_clv.mean():+.2%}"
    )

    print(
        f"Median price CLV: "
        f"{completed_clv.median():+.2%}"
    )

    probability_clv = pd.to_numeric(
        ledger[
            "probability_clv"
        ],
        errors="coerce",
    ).dropna()

    if not probability_clv.empty:

        print(
            f"Average probability CLV: "
            f"{probability_clv.mean():+.2%}"
        )

    beat_close = pd.to_numeric(
        ledger[
            "beat_close"
        ],
        errors="coerce",
    ).dropna()

    if not beat_close.empty:

        print(
            f"Beat close: "
            f"{beat_close.mean():.2%}"
        )


# ============================================================
# DISPLAY — BANKROLL
# ============================================================

def print_bankroll_summary(
    ledger,
):

    if ledger.empty:

        return

    bankroll_after = pd.to_numeric(
        ledger[
            "bankroll_after"
        ],
        errors="coerce",
    ).dropna()

    if bankroll_after.empty:

        return

    current_bankroll = (
        bankroll_after.iloc[
            -1
        ]
    )

    settled = ledger[
        ledger[
            "status"
        ]
        .astype(str)
        .str.upper()
        .eq(
            "SETTLED"
        )
    ].copy()

    open_bets = ledger[
        ledger[
            "status"
        ]
        .astype(str)
        .str.upper()
        .isin(
            [
                "OPEN",
                "CLOSED_LINE",
            ]
        )
    ].copy()

    print()
    print(
        "=" * 100
    )

    print(
        "BANKROLL"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Starting bankroll: "
        f"{STARTING_BANKROLL:.2f}"
    )

    print(
        f"Current bankroll: "
        f"{current_bankroll:.2f}"
    )

    print(
        f"Unit size: "
        f"{UNIT_SIZE_FRACTION:.2%}"
    )

    print(
        f"Flat stake: "
        f"{FLAT_STAKE_UNITS:.2f} unit"
    )

    print(
        f"Settled bets: "
        f"{len(settled)}"
    )

    print(
        f"Open bets: "
        f"{len(open_bets)}"
    )

    if len(
        settled
    ) > 0:

        total_profit_units = pd.to_numeric(
            settled[
                "profit_units"
            ],
            errors="coerce",
        ).sum()

        wins = pd.to_numeric(
            settled[
                "won"
            ],
            errors="coerce",
        ).sum()

        total_bets = len(
            settled
        )

        print(
            f"Settled wins: "
            f"{int(wins)}"
        )

        print(
            f"Win rate: "
            f"{wins / total_bets:.2%}"
        )

        print(
            f"Profit units: "
            f"{total_profit_units:+.2f}"
        )

        print(
            f"Bankroll return: "
            f"{(current_bankroll / STARTING_BANKROLL) - 1:+.2%}"
        )


# ============================================================
# DISPLAY — FULL SUMMARY
# ============================================================

def print_summary(
    ledger,
    new_rows,
):

    print()
    print(
        "=" * 100
    )

    print(
        "LIVE V5 BET LEDGER"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Locked raw V5 edge threshold: "
        f"{RAW_V5_EDGE_THRESHOLD:.1%}"
    )

    print(
        "One bet per match: "
        f"{'YES' if ONE_BET_PER_MATCH else 'NO'}"
    )

    print(
        "First qualifying signal frozen: "
        f"{'YES' if FREEZE_FIRST_QUALIFYING_SIGNAL else 'NO'}"
    )

    print(
        "Kelly staking active: "
        f"{'YES' if USE_KELLY_STAKING else 'NO'}"
    )

    print()

    print(
        "Validated leagues:"
    )

    for league in sorted(
        VALIDATED_LEAGUES
    ):

        print(
            f"  {league}"
        )

    print()

    print(
        "Expanded-live leagues:"
    )

    for league in sorted(
        EXPANDED_LIVE_LEAGUES
    ):

        print(
            f"  {league}"
        )

    print()

    print(
        f"New bets recorded: "
        f"{len(new_rows)}"
    )

    print(
        f"Total ledger bets: "
        f"{len(ledger)}"
    )

    print_new_bets(
        new_rows
    )

    if not ledger.empty:

        print()
        print(
            "=" * 100
        )

        print(
            "LEDGER STATUS"
        )

        print(
            "=" * 100
        )

        print()

        print(
            ledger[
                "status"
            ]
            .fillna(
                "UNKNOWN"
            )
            .value_counts()
            .to_string()
        )

        print_deployment_summary(
            ledger
        )

        print_bankroll_summary(
            ledger
        )

        print_clv_summary(
            ledger
        )

    print()

    print(
        "Saved:"
    )

    print(
        LEDGER_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 100
    )

    print(
        "UPDATE LIVE V5 BET LEDGER"
    )

    print(
        "=" * 100
    )

    if not EV_BOARD_FILE.exists():

        raise FileNotFoundError(
            f"Missing EV board:\n"
            f"{EV_BOARD_FILE}"
        )

    board = pd.read_csv(
        EV_BOARD_FILE,
        low_memory=False,
    )

    validate_ev_board(
        board
    )

    active_board = filter_active_leagues(
        board
    )

    print()
    print(
        f"EV board rows: "
        f"{len(board):,}"
    )

    print(
        f"Active-league rows: "
        f"{len(active_board):,}"
    )

    print(
        f"Odds matched overall: "
        f"{board['odds_matched'].fillna(False).astype(bool).sum():,}"
    )

    print(
        f"Odds matched active leagues: "
        f"{active_board['odds_matched'].fillna(False).astype(bool).sum():,}"
    )

    ledger = load_ledger()

    print(
        f"Existing ledger bets: "
        f"{len(ledger):,}"
    )

    # --------------------------------------------------------
    # RECORD NEW QUALIFYING SIGNALS
    # --------------------------------------------------------

    (
        ledger,
        new_rows,
    ) = add_new_bets(
        board,
        ledger,
    )

    # --------------------------------------------------------
    # UPDATE CLOSING PRICES / CLV
    # --------------------------------------------------------

    if ODDS_HISTORY_FILE.exists():

        odds_history = pd.read_csv(
            ODDS_HISTORY_FILE,
            low_memory=False,
        )

        ledger = update_closing_prices(
            ledger,
            odds_history,
        )

    else:

        print()
        print(
            "WARNING: odds_h2h_history.csv "
            "not found."
        )

        print(
            "New bets will still be recorded, "
            "but CLV cannot yet be updated."
        )

    # --------------------------------------------------------
    # REBUILD BANKROLL
    # --------------------------------------------------------

    ledger = rebuild_bankroll(
        ledger
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_ledger(
        ledger
    )

    print_summary(
        ledger,
        new_rows,
    )


if __name__ == "__main__":

    main()