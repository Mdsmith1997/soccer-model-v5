from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LIVE = ROOT / "data" / "live"

PREDICTIONS_FILE = (
    LIVE / "v5_live_predictions_master.csv"
)

ODDS_FILE = (
    LIVE / "odds_h2h_history.csv"
)

CURRENT_EVENTS_FILE = (
    LIVE / "odds_events_snapshot.csv"
)

OUTPUT_FILE = (
    LIVE / "v5_live_ev_board.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# ============================================================
# PAPER-LIVE BET SELECTION
# ============================================================

# Minimum expected return required before a selection
# can enter the paper-live portfolio.
MIN_EV = 0.05

# Minimum probability advantage over the de-vigged
# sportsbook consensus.
MIN_EDGE = 0.03

# Extremely large disagreements with the market are useful
# research signals, but are not automatically trusted as
# deployable bets.
MAX_EDGE = 0.15

# Existing bankroll protection.
MAX_BANKROLL_FRACTION = 0.02
# Fractional Kelly protects the bankroll from model error.
KELLY_FRACTION = 0.25

# Never recommend more than 2% bankroll on one outcome.
MAX_BANKROLL_FRACTION = 0.02


# ============================================================
# TEAM NORMALIZATION
# ============================================================

ALIASES = {
    # England
    "brighton and hove albion": "brighton",
    "manchester city": "man city",
    "manchester united": "man united",
    "newcastle united": "newcastle",
    "nottingham forest": "nottm forest",
    "tottenham hotspur": "tottenham",
    "west ham united": "west ham",
    "wolverhampton wanderers": "wolves",

    # Germany
    "1 koln": "koln",
    "koln": "koln",
    "fc koln": "koln",
    "bayer leverkusen": "leverkusen",
    "borussia dortmund": "dortmund",
    "borussia monchengladbach": "m gladbach",
    "eintracht frankfurt": "ein frankfurt",
    "fsv mainz 05": "mainz",
    "hamburger sv": "hamburg",
    "sc freiburg": "freiburg",
    "sc paderborn": "paderborn",
    "fc schalke 04": "schalke 04",
    "tsg hoffenheim": "hoffenheim",
    "vfb stuttgart": "stuttgart",

    # Spain
    "atletico madrid": "atletico madrid",
    "athletic club bilbao": "athletic bilbao",
    "deportivo alaves": "alaves",
    "fc barcelona": "barcelona",
    "getafe cf": "getafe",
    "girona fc": "girona",
    "rcd espanyol": "espanyol",
    "rcd mallorca": "mallorca",
    "valencia cf": "valencia",

    # Belgium
    "kaa gent": "gent",
    "krc genk": "genk",
    "kvc westerlo": "westerlo",
    "rsc anderlecht": "anderlecht",
}


def norm_team(value):

    if pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    text = "".join(
        c
        for c in text
        if not unicodedata.combining(c)
    )

    text = text.lower().replace(
        "&",
        " and ",
    )

    text = re.sub(
        r"\b(fc|afc|sv|vfl|sk)\b",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = " ".join(
        text.split()
    )

    text = re.sub(
        r"^1\s+",
        "",
        text,
    )

    return ALIASES.get(
        text,
        text,
    )


# ============================================================
# LOAD
# ============================================================

def load_inputs():

    for path in (
        PREDICTIONS_FILE,
        ODDS_FILE,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        low_memory=False,
    )

    odds = pd.read_csv(
        ODDS_FILE,
        low_memory=False,
    )

    # --------------------------------------------------------
    # CURRENT LIVE SCAN UNIVERSE
    #
    # The odds fetcher already defines the official NOW -> T+72h
    # window and saves those events to odds_events_snapshot.csv.
    #
    # Restrict the actionable EV board to those CURRENT event IDs.
    # Historical odds remain stored on disk; previously frozen
    # ledger bets are not deleted or modified here.
    # --------------------------------------------------------

    if not CURRENT_EVENTS_FILE.exists():

        raise FileNotFoundError(
            CURRENT_EVENTS_FILE
        )

    current_events = pd.read_csv(
        CURRENT_EVENTS_FILE,
        low_memory=False,
    )

    if "event_id" not in current_events.columns:

        raise RuntimeError(
            "Current odds event snapshot has no event_id column."
        )

    current_ids = set(
        current_events[
            "event_id"
        ]
        .dropna()
        .astype(str)
    )

    print(
        "Current T+72 event IDs:",
        len(current_ids),
    )

    # Predictions may use event_id or match_id.
    prediction_id_col = (
        "event_id"
        if "event_id" in predictions.columns
        else "match_id"
    )

    predictions[
        prediction_id_col
    ] = (
        predictions[
            prediction_id_col
        ]
        .astype(str)
    )

    before_predictions = len(
        predictions
    )

    predictions = predictions[
        predictions[
            prediction_id_col
        ]
        .isin(current_ids)
    ].copy()

    print(
        "Predictions restricted to current scan:",
        before_predictions,
        "->",
        len(predictions),
    )

    # Historical H2H odds should also only contribute rows for
    # events belonging to the current scan.
    if "event_id" in odds.columns:

        odds[
            "event_id"
        ] = (
            odds[
                "event_id"
            ]
            .astype(str)
        )

        before_odds = len(
            odds
        )

        odds = odds[
            odds[
                "event_id"
            ]
            .isin(current_ids)
        ].copy()

        print(
            "Odds rows restricted to current scan:",
            before_odds,
            "->",
            len(odds),
        )

    predictions["date"] = pd.to_datetime(
        predictions["date"],
        errors="coerce",
    ).dt.normalize()

    odds["commence_time"] = pd.to_datetime(
        odds["commence_time"],
        errors="coerce",
        utc=True,
    )

    # We only need the calendar date for fixture matching.
    odds["date"] = (
        odds["commence_time"]
        .dt.tz_convert(None)
        .dt.normalize()
    )

    predictions["home_norm"] = (
        predictions["home_team"]
        .map(norm_team)
    )

    predictions["away_norm"] = (
        predictions["away_team"]
        .map(norm_team)
    )

    odds["home_norm"] = (
        odds["home_team"]
        .map(norm_team)
    )

    odds["away_norm"] = (
        odds["away_team"]
        .map(norm_team)
    )

    for col in (
        "p_home_v5",
        "p_draw_v5",
        "p_away_v5",
    ):
        predictions[col] = pd.to_numeric(
            predictions[col],
            errors="coerce",
        )

    for col in (
        "home_odds",
        "draw_odds",
        "away_odds",
    ):
        odds[col] = pd.to_numeric(
            odds[col],
            errors="coerce",
        )

    return predictions, odds


# ============================================================
# MATCH ODDS EVENTS
# ============================================================

def build_fixture_odds(predictions, odds):

    rows = []

    for fixture in predictions.itertuples(index=False):

        candidates = odds.loc[
            (odds["date"] == fixture.date)
            &
            (odds["home_norm"] == fixture.home_norm)
            &
            (odds["away_norm"] == fixture.away_norm)
        ].copy()

        # Some providers can report UTC on the adjacent
        # calendar date. Team identity is sufficiently strong
        # for a fallback.
        if candidates.empty:

            candidates = odds.loc[
                (odds["home_norm"] == fixture.home_norm)
                &
                (odds["away_norm"] == fixture.away_norm)
            ].copy()

        if candidates.empty:

            rows.append({
                "match_id": fixture.match_id,
                "odds_matched": False,
            })

            continue

        # ----------------------------------------------------
        # USE LATEST SNAPSHOT
        # ----------------------------------------------------

        candidates["snapshot_dt"] = pd.to_datetime(
            candidates["snapshot_time"],
            errors="coerce",
            utc=True,
        )

        latest = candidates["snapshot_dt"].max()

        current = candidates.loc[
            candidates["snapshot_dt"] == latest
        ].copy()

        # ----------------------------------------------------
        # VALID COMPLETE 1X2 BOOKS
        #
        # Market consensus must be calculated from complete
        # sportsbook markets, not synthetic best prices.
        # ----------------------------------------------------

        complete = current.loc[
            current[
                [
                    "home_odds",
                    "draw_odds",
                    "away_odds",
                ]
            ]
            .notna()
            .all(axis=1)
        ].copy()

        complete = complete.loc[
            (complete["home_odds"] > 1.0)
            &
            (complete["draw_odds"] > 1.0)
            &
            (complete["away_odds"] > 1.0)
        ].copy()

        if complete.empty:

            rows.append({
                "match_id": fixture.match_id,
                "odds_matched": False,
            })

            continue

        # ----------------------------------------------------
        # BEST EXECUTION PRICES
        #
        # These are the actual prices we could bet.
        # ----------------------------------------------------

        home_row = complete.loc[
            complete["home_odds"].idxmax()
        ]

        draw_row = complete.loc[
            complete["draw_odds"].idxmax()
        ]

        away_row = complete.loc[
            complete["away_odds"].idxmax()
        ]

        # ----------------------------------------------------
        # DE-VIG EACH SPORTSBOOK INDIVIDUALLY
        # ----------------------------------------------------

        complete["imp_home"] = (
            1.0 / complete["home_odds"]
        )

        complete["imp_draw"] = (
            1.0 / complete["draw_odds"]
        )

        complete["imp_away"] = (
            1.0 / complete["away_odds"]
        )

        complete["book_overround"] = (
            complete["imp_home"]
            +
            complete["imp_draw"]
            +
            complete["imp_away"]
        )

        complete["fair_home"] = (
            complete["imp_home"]
            /
            complete["book_overround"]
        )

        complete["fair_draw"] = (
            complete["imp_draw"]
            /
            complete["book_overround"]
        )

        complete["fair_away"] = (
            complete["imp_away"]
            /
            complete["book_overround"]
        )

        # ----------------------------------------------------
        # MARKET CONSENSUS
        #
        # Median is deliberately used instead of mean.
        # One stale/outlier sportsbook therefore cannot
        # materially distort the market benchmark.
        # ----------------------------------------------------

        market_p_home = complete["fair_home"].median()
        market_p_draw = complete["fair_draw"].median()
        market_p_away = complete["fair_away"].median()

        # Medians independently calculated may not sum to
        # exactly 1, so normalize once more.

        market_total = (
            market_p_home
            +
            market_p_draw
            +
            market_p_away
        )

        market_p_home /= market_total
        market_p_draw /= market_total
        market_p_away /= market_total

        # ----------------------------------------------------
        # PRE-MATCH SAFETY GUARD
        #
        # Never allow in-play odds onto the EV board.
        # Require the odds snapshot to be at least one minute
        # before scheduled kickoff.
        # ----------------------------------------------------

        commence_time = pd.to_datetime(
            current["commence_time"].iloc[0],
            utc=True,
            errors="coerce",
        )

        snapshot_time = pd.to_datetime(
            latest,
            utc=True,
            errors="coerce",
        )

        if (
            pd.isna(commence_time)
            or
            pd.isna(snapshot_time)
            or
            snapshot_time
            >=
            (
                commence_time
                -
                pd.Timedelta(minutes=1)
            )
        ):
            continue

        rows.append({

            "match_id":
                fixture.match_id,

            "odds_matched":
                True,

            "odds_event_id":
                current["event_id"].iloc[0],

            "odds_snapshot_time":
                latest,

            "odds_commence_time":
                current["commence_time"].iloc[0],

            # -----------------------------------------------
            # BEST EXECUTION PRICES
            # -----------------------------------------------

            "best_home_odds":
                home_row["home_odds"],

            "best_home_book":
                home_row["bookmaker"],

            "best_draw_odds":
                draw_row["draw_odds"],

            "best_draw_book":
                draw_row["bookmaker"],

            "best_away_odds":
                away_row["away_odds"],

            "best_away_book":
                away_row["bookmaker"],

            # -----------------------------------------------
            # MARKET CONSENSUS
            # -----------------------------------------------

            "market_p_home":
                market_p_home,

            "market_p_draw":
                market_p_draw,

            "market_p_away":
                market_p_away,

            "market_overround":
                complete[
                    "book_overround"
                ].median(),

            "books_available":
                complete[
                    "bookmaker_key"
                ].nunique(),
        })

    matched = pd.DataFrame(rows)

    return predictions.merge(
        matched,
        on="match_id",
        how="left",
        validate="one_to_one",
    )


# ============================================================
# MARKET PROBABILITIES
# ============================================================

def add_market_probabilities(df):

    required = [
        "market_p_home",
        "market_p_draw",
        "market_p_away",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing market consensus columns: "
            + str(missing)
        )

    probability_sum = (
        df[
            required
        ]
        .sum(axis=1)
    )

    valid = (
        df["odds_matched"]
        .fillna(False)
    )

    if not np.allclose(
        probability_sum.loc[valid],
        1.0,
        atol=1e-8,
    ):

        raise ValueError(
            "Consensus market probabilities "
            "do not sum to 1."
        )

    return df


# ============================================================
# EV + KELLY
# ============================================================

def kelly_fraction(
    probability,
    decimal_odds,
):

    b = decimal_odds - 1.0

    if (
        pd.isna(probability)
        or pd.isna(decimal_odds)
        or b <= 0
    ):
        return np.nan

    q = 1.0 - probability

    full = (
        (
            b * probability
            - q
        )
        / b
    )

    return max(
        0.0,
        full,
    )


def add_betting_metrics(df):

    definitions = [
        (
            "home",
            "p_home_v5",
            "market_p_home",
            "best_home_odds",
        ),
        (
            "draw",
            "p_draw_v5",
            "market_p_draw",
            "best_draw_odds",
        ),
        (
            "away",
            "p_away_v5",
            "market_p_away",
            "best_away_odds",
        ),
    ]

    for (
        side,
        model_col,
        market_col,
        odds_col,
    ) in definitions:

        # Probability edge
        df[
            f"{side}_edge"
        ] = (
            df[model_col]
            -
            df[market_col]
        )

        # Expected profit per $1 staked.
        df[
            f"{side}_ev"
        ] = (
            df[model_col]
            * df[odds_col]
            - 1.0
        )

        full_kelly = [
            kelly_fraction(p, o)
            for p, o in zip(
                df[model_col],
                df[odds_col],
            )
        ]

        df[
            f"{side}_kelly_full"
        ] = full_kelly

        df[
            f"{side}_kelly"
        ] = np.minimum(
            np.asarray(
                full_kelly,
                dtype=float,
            )
            * KELLY_FRACTION,
            MAX_BANKROLL_FRACTION,
        )

    return df


# ============================================================
# BEST BET PER MATCH
# ============================================================

def add_best_bet(df):

    ev_cols = [
        "home_ev",
        "draw_ev",
        "away_ev",
    ]

    available = (
        df[ev_cols]
        .notna()
        .any(axis=1)
    )

    # ========================================================
    # OUTPUT COLUMNS
    # ========================================================

    df["best_side"] = None
    df["best_ev"] = np.nan
    df["best_edge"] = np.nan
    df["best_odds"] = np.nan
    df["best_book"] = None
    df["kelly_fraction"] = np.nan

    df["paper_bet"] = False
    df["bet_status"] = None

    # ========================================================
    # BEST EV OUTCOME PER FIXTURE
    # ========================================================

    for idx in df.index[available]:

        side = (
            df.loc[
                idx,
                ev_cols,
            ]
            .astype(float)
            .idxmax()
            .replace(
                "_ev",
                "",
            )
        )

        best_ev = df.loc[
            idx,
            f"{side}_ev",
        ]

        best_edge = df.loc[
            idx,
            f"{side}_edge",
        ]

        best_odds = df.loc[
            idx,
            f"best_{side}_odds",
        ]

        best_book = df.loc[
            idx,
            f"best_{side}_book",
        ]

        kelly = df.loc[
            idx,
            f"{side}_kelly",
        ]

        # ----------------------------------------------------
        # STORE BEST AVAILABLE OPPORTUNITY
        # ----------------------------------------------------

        df.loc[
            idx,
            "best_side"
        ] = side.upper()

        df.loc[
            idx,
            "best_ev"
        ] = best_ev

        df.loc[
            idx,
            "best_edge"
        ] = best_edge

        df.loc[
            idx,
            "best_odds"
        ] = best_odds

        df.loc[
            idx,
            "best_book"
        ] = best_book

        df.loc[
            idx,
            "kelly_fraction"
        ] = kelly

        # ====================================================
        # PAPER-LIVE QUALITY GATE
        # ====================================================

        if (
            pd.isna(best_ev)
            or pd.isna(best_edge)
            or pd.isna(best_odds)
        ):

            df.loc[
                idx,
                "bet_status"
            ] = "NO_DATA"

            continue

        # ----------------------------------------------------
        # Negative / insufficient EV
        # ----------------------------------------------------

        if best_ev < MIN_EV:

            df.loc[
                idx,
                "bet_status"
            ] = "EV_TOO_LOW"

            continue

        # ----------------------------------------------------
        # Model advantage is too small.
        # ----------------------------------------------------

        if best_edge < MIN_EDGE:

            df.loc[
                idx,
                "bet_status"
            ] = "EDGE_TOO_LOW"

            continue

        # ----------------------------------------------------
        # Very large model-market disagreement.
        #
        # Do not automatically deploy these. Preserve them
        # for research/model review.
        # ----------------------------------------------------

        if best_edge > MAX_EDGE:

            df.loc[
                idx,
                "bet_status"
            ] = "MODEL_REVIEW"

            continue

        # ----------------------------------------------------
        # Passed all deployment rules.
        # ----------------------------------------------------

        df.loc[
            idx,
            "paper_bet"
        ] = True

        df.loc[
            idx,
            "bet_status"
        ] = "PAPER_BET"

    return df


# ============================================================
# DISPLAY
# ============================================================

def print_summary(df):

    print()
    print("=" * 110)
    print("LIVE V5 EV BOARD")
    print("=" * 110)

    print(
        "Prediction fixtures:",
        len(df),
    )

    matched = int(
        df["odds_matched"]
        .fillna(False)
        .sum()
    )

    print(
        "Odds matched:",
        matched,
        "/",
        len(df),
    )

    print(
        "Unmatched:",
        len(df) - matched,
    )

    bets = df.loc[
        df["paper_bet"] == True
    ].copy()

    print(
        "Paper bets:",
        len(bets),
    )

    if len(bets):

        display = bets[
            [
                "date",
                "league",
                "home_team",
                "away_team",
                "best_side",
                "best_odds",
                "best_book",
                "best_edge",
                "best_ev",
                "kelly_fraction",
            ]
        ].copy()

        for col in [
            "best_edge",
            "best_ev",
            "kelly_fraction",
        ]:
            display[col] *= 100.0

        print()
        print(
            display
            .sort_values(
                "best_ev",
                ascending=False,
            )
            .round(3)
            .to_string(index=False)
        )

    unmatched = df.loc[
        ~df[
            "odds_matched"
        ].fillna(False).astype(bool).fillna(False),
        [
            "date",
            "league",
            "home_team",
            "away_team",
        ],
    ]

    if len(unmatched):

        print()
        print("=" * 110)
        print("UNMATCHED ODDS FIXTURES")
        print("=" * 110)

        print(
            unmatched.to_string(
                index=False
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print("BUILD LIVE V5 EV BOARD")
    print("=" * 110)

    predictions, odds = (
        load_inputs()
    )

    print()
    print(
        "Predictions:",
        len(predictions),
    )

    print(
        "Historical bookmaker rows:",
        len(odds),
    )

    df = build_fixture_odds(
        predictions,
        odds,
    )

    df = add_market_probabilities(
        df
    )

    df = add_betting_metrics(
        df
    )

    df = add_best_bet(
        df
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print_summary(
        df
    )

    print()
    print("=" * 110)
    print("EV BOARD COMPLETE")
    print("=" * 110)

    print(
        "3-way market de-vigged ✅"
    )

    print(
        "Model probability edge calculated ✅"
    )

    print(
        "Expected value calculated ✅"
    )

    print(
        "Quarter Kelly calculated ✅"
    )

    print(
        "2% bankroll cap applied ✅"
    )

    print(
    "Paper-live EV threshold:",
    f"{MIN_EV:.1%}",
)

    print(
    "Paper-live edge range:",
    f"{MIN_EDGE:.1%}",
    "->",
    f"{MAX_EDGE:.1%}",
)

    print(
    "Kelly bankroll cap:",
    f"{MAX_BANKROLL_FRACTION:.1%}",
)

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()