from pathlib import Path
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
LIVE = ROOT / "data" / "live"

LEDGER_FILE = LIVE / "v5_live_bet_ledger.csv"
ACTUAL_FILE = LIVE / "v5_actual_wagers.csv"
CLV_FILE = LIVE / "v5_live_clv_report.csv"
BTTS_FILE = LIVE / "btts_live_predictions.csv"


# ============================================================
# DAILY PIPELINE
# ============================================================

# Always collect BTTS prices needed by the cemented specialist board.
# This changes live market collection only — NOT frozen model parameters.
os.environ.setdefault("FETCH_BTTS", "1")
os.environ.setdefault("BTTS_ONLY_MODEL_COVERAGE", "1")
os.environ.setdefault("MAX_BTTS_EVENTS", "100")


PIPELINE = [
    (
        "FETCH CURRENT U.S. SOCCER ODDS",
        "fetch_us_soccer_odds.py",
    ),
    (
        "BUILD LIVE FIXTURES",
        "build_live_multileague_fixtures.py",
    ),
    (
        "BUILD LIVE CORE V5 PREDICTIONS",
        "build_live_v5_predictions.py",
    ),
    (
        "BUILD LIVE FOOTYSTATS V5 PREDICTIONS",
        "build_live_multileague_v5_predictions.py",
    ),
    (
        "BUILD LIVE V5 MASTER BOARD",
        "build_live_v5_master.py",
    ),
    (
        "BUILD LIVE FROZEN BTTS BOARD",
        "run_live_btts.py",
    ),
    (
        "FREEZE NEW QUALIFYING BTTS SPECIALIST SIGNALS",
        "update_live_btts_specialist_ledger.py",
    ),

    # ========================================================
    # 1X2
    # ========================================================

    (
        "BUILD LIVE V5 1X2 EV BOARD",
        "build_live_v5_ev_board.py",
    ),

    # ========================================================
    # LEAGUE TWO HOME — FROZEN PROSPECTIVE STRATEGY
    #
    # HOME ML
    # RAW V5 edge >= 16%
    # BOTH teams SAME_LEAGUE history
    # Flat 1u
    # No odds filter
    #
    # This logger is prospective-only and independently
    # records every qualifying League Two HOME signal.
    # ========================================================

    (
        "UPDATE LEAGUE TWO HOME LIVE-FORWARD LEDGER",
        "update_league_two_home_live_forward.py",
    ),

    (
        "FREEZE NEW QUALIFYING 1X2 SIGNALS",
        "update_live_v5_bet_ledger.py",
    ),
    (
        "BACKFILL 1X2 LEDGER KICKOFF TIMES",
        "backfill_v5_ledger_commence_times.py",
    ),

    # ========================================================
    # TOTALS
    #
    # Frozen forward-test rule:
    # RAW V5 Under 2.5 edge >= 11%
    # ========================================================

    (
        "BUILD LIVE V5 TOTALS EV BOARD",
        "build_live_v5_totals_ev_board.py",
    ),
    (
        "FREEZE NEW QUALIFYING TOTALS SIGNALS",
        "update_live_v5_totals_bet_ledger.py",
    ),

    # Capture any available closing lines before settlement.
    (
        "UPDATE TOTALS CLOSING LINES / CLV",
        "update_live_v5_totals_clv.py",
    ),

    # ========================================================
    # RESULTS / SETTLEMENT
    # ========================================================

    (
        "FETCH COMPLETED SOCCER RESULTS",
        "fetch_live_soccer_results.py",
    ),
    (
        "SETTLE 1X2 MODEL BETS",
        "settle_live_v5_bets.py",
    ),
    (
        "SETTLE BTTS SPECIALIST MODEL BETS",
        "settle_live_btts_specialist_bets.py",
    ),
    (
        "SETTLE TOTALS MODEL BETS",
        "settle_live_v5_totals_bets.py",
    ),

    # ========================================================
    # FINAL REFRESH
    # ========================================================

    (
        "REFRESH 1X2 MODEL LEDGER",
        "update_live_v5_bet_ledger.py",
    ),
    (
        "REFRESH 1X2 CLV REPORT",
        "build_live_v5_clv_report.py",
    ),
]




# ============================================================
# CEMENTED V5 BOARD
# ============================================================

BTTS_SPECIALISTS = {
    "Swiss Super League": 0.06,
    "Super Lig": 0.10,
    "Segunda División": 0.04,
}

TOTALS_SPECIALISTS = {
    ("Premier League", "UNDER"): 0.11,
    ("Bundesliga", "UNDER"): 0.11,
    ("Belgian Pro League", "OVER"): 0.15,
    ("Eliteserien", "OVER"): 0.11,
}


def decimal_to_american(decimal_odds):
    try:
        d = float(decimal_odds)
    except Exception:
        return None

    if d <= 1:
        return None

    if d >= 2.0:
        return int(round((d - 1.0) * 100))

    return int(round(-100 / (d - 1.0)))


def live_window():
    now = pd.Timestamp.now(
        tz="America/Detroit"
    )

    end = now + pd.Timedelta(
        hours=72
    )

    return now, end


def print_cemented_board():
    print()
    print("#" * 110)
    print("COMPLETE CEMENTED V5 BOARD — NEXT 72 HOURS")
    print("#" * 110)

    window_start, window_end = live_window()

    print(
        "Window:",
        window_start.strftime(
            "%Y-%m-%d %I:%M %p %Z"
        ),
        "->",
        window_end.strftime(
            "%Y-%m-%d %I:%M %p %Z"
        ),
    )

    official_count = 0

    # --------------------------------------------------------
    # GLOBAL 1X2
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("1X2 — GLOBAL RAW V5 EDGE >= 16%")
    print("=" * 110)

    if LEDGER_FILE.exists():
        x = pd.read_csv(
            LEDGER_FILE,
            low_memory=False,
        )

        if "commence_time" in x.columns:

            x["_kickoff"] = pd.to_datetime(
                x["commence_time"],
                errors="coerce",
                utc=True,
            ).dt.tz_convert(
                "America/Detroit"
            )

            mask = (
                x["_kickoff"].ge(window_start)
                &
                x["_kickoff"].le(window_end)
            )

            if "status" in x.columns:
                mask &= (
                    ~x["status"]
                    .astype(str)
                    .str.upper()
                    .eq("INVALID")
                )

            today_x = x[
                mask
            ].copy()

            if len(today_x):

                today_x["T_MINUS_H"] = (
                    (
                        today_x["_kickoff"]
                        -
                        window_start
                    )
                    .dt.total_seconds()
                    /
                    3600
                )
                cols = [
                    c for c in [
                        "league",
                        "home_team",
                        "away_team",
                        "bet_side",
                        "bet_odds",
                        "bet_book",
                        "edge",
                        "history_type",
                        "T_MINUS_H",
                        "status",
                    ]
                    if c in today_x.columns
                ]

                print(
                    today_x[cols]
                    .to_string(index=False)
                )
                official_count += len(today_x)
            else:
                print("NONE")
        else:
            print("NONE")
    else:
        print("NONE")

    # --------------------------------------------------------
    # LEAGUE TWO HOME SPECIALIST
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("1X2 SPECIALIST — LEAGUE TWO HOME >= 16%")
    print("=" * 110)

    league2 = pd.DataFrame()

    if LEDGER_FILE.exists():
        x = pd.read_csv(
            LEDGER_FILE,
            low_memory=False,
        )

        if "commence_time" in x.columns:

            x["_kickoff"] = pd.to_datetime(
                x["commence_time"],
                errors="coerce",
                utc=True,
            ).dt.tz_convert(
                "America/Detroit"
            )

            masks = [
                x["_kickoff"].ge(window_start),
                x["_kickoff"].le(window_end),
                x["league"].astype(str).eq("League Two"),
                x["bet_side"].astype(str).str.upper().eq("HOME"),
            ]

            if "status" in x.columns:
                masks.append(
                    ~x["status"]
                    .astype(str)
                    .str.upper()
                    .eq("INVALID")
                )

            mask = masks[0]
            for m in masks[1:]:
                mask &= m

            league2 = x[mask].copy()

            if len(league2):
                league2["T_MINUS_H"] = (
                    (
                        league2["_kickoff"]
                        -
                        window_start
                    )
                    .dt.total_seconds()
                    /
                    3600
                )

    if len(league2):
        cols = [
            c for c in [
                "home_team",
                "away_team",
                "bet_side",
                "bet_odds",
                "bet_book",
                "edge",
                "T_MINUS_H",
                "status",
            ]
            if c in league2.columns
        ]

        print(
            league2[cols]
            .to_string(index=False)
        )
    else:
        print("NONE")

    # Do NOT increment official_count:
    # League Two HOME may already be present in Global 1X2.
    # We never double-count the same underlying signal.

    # --------------------------------------------------------
    # BTTS SPECIALISTS
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("BTTS YES SPECIALISTS")
    print("=" * 110)

    btts_qualifiers = []

    if BTTS_FILE.exists():
        b = pd.read_csv(
            BTTS_FILE,
            low_memory=False,
        )

        if "commence_time" in b.columns:

            b["_kickoff"] = pd.to_datetime(
                b["commence_time"],
                errors="coerce",
                utc=True,
            ).dt.tz_convert(
                "America/Detroit"
            )

            b = b[
                b["_kickoff"].ge(window_start)
                &
                b["_kickoff"].le(window_end)
            ].copy()

            if len(b):
                b["T_MINUS_H"] = (
                    (
                        b["_kickoff"]
                        -
                        window_start
                    )
                    .dt.total_seconds()
                    /
                    3600
                )

        if len(b):
            b["yes_edge"] = pd.to_numeric(
                b.get("yes_edge"),
                errors="coerce",
            )

            # One best price / bookmaker row per fixture.
            b["yes_odds"] = pd.to_numeric(
                b.get("yes_odds"),
                errors="coerce",
            )

            for league, threshold in BTTS_SPECIALISTS.items():
                q = b[
                    b["league_market"]
                    .astype(str)
                    .eq(league)
                    &
                    b["yes_edge"]
                    .ge(threshold)
                ].copy()

                if len(q):
                    q = (
                        q.sort_values(
                            ["match_id", "yes_odds"],
                            ascending=[True, False],
                        )
                        .drop_duplicates(
                            subset=["match_id"],
                            keep="first",
                        )
                    )

                    q["frozen_threshold"] = threshold
                    btts_qualifiers.append(q)

    if btts_qualifiers:
        q = pd.concat(
            btts_qualifiers,
            ignore_index=True,
        )

        q["american_odds"] = q[
            "yes_odds"
        ].map(decimal_to_american)

        cols = [
            c for c in [
                "league_market",
                "home_team_market",
                "away_team_market",
                "american_odds",
                "final_yes_probability",
                "market_yes",
                "yes_edge",
                "yes_ev",
                "bookmaker",
                "T_MINUS_H",
                "frozen_threshold",
            ]
            if c in q.columns
        ]

        print(
            q[cols]
            .to_string(index=False)
        )

        official_count += len(q)
    else:
        print("NONE")

    # --------------------------------------------------------
    # TOTALS SPECIALISTS
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("O/U 2.5 SPECIALISTS")
    print("=" * 110)

    totals_file = (
        LIVE
        /
        "v5_live_totals_ev_board.csv"
    )

    totals_qualifiers = []

    if totals_file.exists():
        t = pd.read_csv(
            totals_file,
            low_memory=False,
        )

        kickoff_col = (
            "commence_time"
            if "commence_time" in t.columns
            else None
        )

        if kickoff_col is not None:

            t["_kickoff"] = pd.to_datetime(
                t[kickoff_col],
                errors="coerce",
                utc=True,
            ).dt.tz_convert(
                "America/Detroit"
            )

            t = t[
                t["_kickoff"].ge(window_start)
                &
                t["_kickoff"].le(window_end)
            ].copy()

            if len(t):
                t["T_MINUS_H"] = (
                    (
                        t["_kickoff"]
                        -
                        window_start
                    )
                    .dt.total_seconds()
                    /
                    3600
                )

        elif "date" in t.columns:
            # Fallback only if totals board has no kickoff timestamp.
            t["_date"] = pd.to_datetime(
                t["date"],
                errors="coerce",
            ).dt.date

            allowed_dates = {
                window_start.date(),
                (window_start + pd.Timedelta(days=1)).date(),
                (window_start + pd.Timedelta(days=2)).date(),
                (window_start + pd.Timedelta(days=3)).date(),
            }

            t = t[
                t["_date"].isin(
                    allowed_dates
                )
            ].copy()

        if len(t):
            t["edge"] = pd.to_numeric(
                t.get("edge"),
                errors="coerce",
            )

            t["decimal_odds"] = pd.to_numeric(
                t.get("decimal_odds"),
                errors="coerce",
            )

            # Normalize bet labels such as OVER / Over 2.5.
            bet_text = (
                t["bet"]
                .astype(str)
                .str.upper()
            )

            for (
                league,
                side,
            ), threshold in TOTALS_SPECIALISTS.items():

                side_mask = bet_text.str.contains(
                    side,
                    regex=False,
                )

                q = t[
                    t["league"].astype(str).eq(league)
                    &
                    side_mask
                    &
                    t["edge"].ge(threshold)
                ].copy()

                if len(q):
                    q = (
                        q.sort_values(
                            [
                                "match_id",
                                "decimal_odds",
                            ],
                            ascending=[
                                True,
                                False,
                            ],
                        )
                        .drop_duplicates(
                            subset=["match_id"],
                            keep="first",
                        )
                    )

                    q["frozen_threshold"] = threshold
                    totals_qualifiers.append(q)

    if totals_qualifiers:
        q = pd.concat(
            totals_qualifiers,
            ignore_index=True,
        )

        cols = [
            c for c in [
                "league",
                "home_team",
                "away_team",
                "bet",
                "decimal_odds",
                "model_probability",
                "market_probability",
                "edge",
                "ev",
                "bookmaker",
                "T_MINUS_H",
                "frozen_threshold",
            ]
            if c in q.columns
        ]

        print(
            q[cols]
            .to_string(index=False)
        )

        official_count += len(q)
    else:
        print("NONE")

    print()
    print("=" * 110)
    print(
        f"UNIQUE OFFICIAL/CEMENTED SIGNALS NEXT 72H: "
        f"{official_count}"
    )
    print("=" * 110)


# ============================================================
# HELPERS
# ============================================================

def american_odds(decimal_odds):

    try:
        d = float(decimal_odds)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(d) or d <= 1:
        return None

    if d >= 2:
        return int(
            round(
                (d - 1)
                * 100
            )
        )

    return int(
        round(
            -100
            /
            (d - 1)
        )
    )


def american_string(decimal_odds):

    a = american_odds(
        decimal_odds
    )

    if a is None:
        return "-"

    return f"{a:+d}"


def pct_string(value):

    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"

    if not np.isfinite(x):
        return "-"

    # Existing model fields are generally stored as decimals.
    if abs(x) <= 1:
        x *= 100

    return f"{x:.2f}%"


def money(value):

    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"

    if not np.isfinite(x):
        return "-"

    return f"${x:,.2f}"


def run_script(
    title,
    filename,
):

    path = (
        SCRIPTS
        /
        filename
    )

    print()
    print("=" * 110)
    print(title)
    print("=" * 110)
    print()

    if not path.exists():

        print(
            f"SKIPPED — script not found: "
            f"{path}"
        )

        return False

    # Core V5 only supports Premier League and Bundesliga.
    # A live window can legitimately contain zero core fixtures while
    # FootyStats still has fixtures to score. In that case, skip Core
    # instead of stopping the entire live pipeline.
    if filename == "build_live_v5_predictions.py":
        fixtures_file = LIVE / "upcoming_fixtures.csv"

        if fixtures_file.exists():
            fixtures = pd.read_csv(fixtures_file, low_memory=False)
            core_leagues = {"Premier League", "Bundesliga"}

            if (
                "league" in fixtures.columns
                and not fixtures["league"].isin(core_leagues).any()
            ):
                # Clear any previous Core predictions so Master cannot
                # ingest stale fixtures from an earlier live window.
                core_output = LIVE / "v5_live_predictions_core.csv"

                if core_output.exists():
                    stale_core = pd.read_csv(
                        core_output,
                        low_memory=False,
                    )
                    stale_core.iloc[0:0].to_csv(
                        core_output,
                        index=False,
                    )
                    print(
                        "Cleared stale Core prediction rows: "
                        f"{len(stale_core)}"
                    )

                print(
                    "SKIPPED — no Premier League or Bundesliga "
                    "fixtures in current live window."
                )
                return True

    result = subprocess.run(
        [
            sys.executable,
            str(path),
        ],
        cwd=str(ROOT),
    )

    if result.returncode != 0:

        print()
        print(
            f"PIPELINE STOPPED: "
            f"{filename} failed with "
            f"exit code {result.returncode}."
        )

        raise SystemExit(
            result.returncode
        )

    return True


def local_today():

    return datetime.now(
        ZoneInfo(
            "America/Detroit"
        )
    ).date()


# ============================================================
# TODAY'S MODEL BETS
# ============================================================

def print_today_model_bets():

    print()
    print("=" * 110)
    print("TODAY'S OFFICIAL V5 1X2 SIGNALS")
    print("=" * 110)
    print()

    if not LEDGER_FILE.exists():

        print(
            "No model ledger found."
        )

        return

    df = pd.read_csv(
        LEDGER_FILE,
        low_memory=False,
    )

    if df.empty:

        print(
            "Model ledger is empty."
        )

        return

    today = local_today()

    df["_date"] = pd.to_datetime(
        df.get(
            "date"
        ),
        errors="coerce",
    ).dt.date

    today_df = df[
        df["_date"].eq(
            today
        )
    ].copy()

    # INVALID rows are audit records, not official model signals.
    if "status" in today_df.columns:
        today_df = today_df[
            ~today_df["status"]
            .astype(str)
            .str.upper()
            .eq("INVALID")
        ].copy()

    if today_df.empty:

        print(
            f"No official V5 signals "
            f"for {today}."
        )

        return

    today_df[
        "ODDS"
    ] = today_df[
        "bet_odds"
    ].map(
        american_string
    )

    if (
        "signal_edge"
        in
        today_df.columns
    ):

        today_df[
            "EDGE"
        ] = today_df[
            "signal_edge"
        ].map(
            pct_string
        )

    else:

        today_df[
            "EDGE"
        ] = "-"

    if (
        "model_probability"
        in
        today_df.columns
    ):

        today_df[
            "MODEL"
        ] = today_df[
            "model_probability"
        ].map(
            pct_string
        )

    else:

        today_df[
            "MODEL"
        ] = "-"

    if (
        "status"
        not in
        today_df.columns
    ):

        today_df[
            "status"
        ] = "OPEN"

    cols = [
        c
        for c in [
            "league",
            "home_team",
            "away_team",
            "bet_side",
            "ODDS",
            "bet_book",
            "MODEL",
            "EDGE",
            "history_type",
            "deployment_status",
            "status",
        ]
        if c in today_df.columns
    ]

    print(
        today_df[
            cols
        ].to_string(
            index=False
        )
    )


# ============================================================
# OPEN MODEL BETS
# ============================================================

def print_open_model_bets():

    print()
    print("=" * 110)
    print("UPCOMING FROZEN MODEL BETS")
    print("=" * 110)
    print()

    if not LEDGER_FILE.exists():
        print("No model ledger found.")
        return

    df = pd.read_csv(
        LEDGER_FILE,
        low_memory=False,
    )

    if (
        df.empty
        or
        "status"
        not in
        df.columns
    ):
        print("No open model bets.")
        return

    x = df[
        df[
            "status"
        ]
        .astype(str)
        .str.upper()
        .eq("OPEN")
    ].copy()

    if x.empty:

        print(
            "No open model bets."
        )

        return

    if (
        "commence_time"
        in
        x.columns
    ):

        x[
            "commence_time"
        ] = pd.to_datetime(
            x[
                "commence_time"
            ],
            utc=True,
            errors="coerce",
        )

        x = x.sort_values(
            "commence_time"
        )

    x[
        "ODDS"
    ] = x[
        "bet_odds"
    ].map(
        american_string
    )

    if (
        "signal_edge"
        in
        x.columns
    ):

        x[
            "EDGE"
        ] = x[
            "signal_edge"
        ].map(
            pct_string
        )

    else:

        x[
            "EDGE"
        ] = "-"

    cols = [
        c
        for c in [
            "date",
            "league",
            "home_team",
            "away_team",
            "bet_side",
            "ODDS",
            "bet_book",
            "EDGE",
            "history_type",
            "commence_time",
        ]
        if c in x.columns
    ]

    print(
        x[
            cols
        ].to_string(
            index=False
        )
    )


# ============================================================
# INVALID / AUDIT SIGNALS
# ============================================================

def print_invalid_signals():

    print()
    print("=" * 110)
    print("INVALID / AUDIT-ONLY SIGNALS")
    print("=" * 110)
    print()

    if not LEDGER_FILE.exists():
        print("No model ledger found.")
        return

    df = pd.read_csv(
        LEDGER_FILE,
        low_memory=False,
    )

    if (
        df.empty
        or
        "status" not in df.columns
    ):
        print("No invalid signals.")
        return

    x = df[
        df["status"]
        .astype(str)
        .str.upper()
        .eq("INVALID")
    ].copy()

    if x.empty:
        print("No invalid signals.")
        return

    x["ODDS"] = x[
        "bet_odds"
    ].map(
        american_string
    )

    cols = [
        c
        for c in [
            "date",
            "league",
            "home_team",
            "away_team",
            "bet_side",
            "ODDS",
            "bet_book",
            "invalid_reason",
        ]
        if c in x.columns
    ]

    print(
        x[
            cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "These rows are retained for audit purposes "
        "and are excluded from model performance."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def print_model_performance():

    print()
    print("=" * 110)
    print("V5 MODEL FORWARD TEST")
    print("=" * 110)
    print()

    if not LEDGER_FILE.exists():

        print(
            "No model ledger found."
        )

        return

    df = pd.read_csv(
        LEDGER_FILE,
        low_memory=False,
    )

    if (
        df.empty
        or
        "status"
        not in
        df.columns
    ):

        print(
            "No settled model bets."
        )

        return

    settled = df[
        df[
            "status"
        ]
        .astype(str)
        .str.upper()
        .eq("SETTLED")
    ].copy()

    if settled.empty:

        print(
            "No settled model bets yet."
        )

        return

    settled[
        "won"
    ] = pd.to_numeric(
        settled.get(
            "won"
        ),
        errors="coerce",
    ).fillna(0)

    settled[
        "profit_units"
    ] = pd.to_numeric(
        settled.get(
            "profit_units"
        ),
        errors="coerce",
    ).fillna(0)

    bets = len(
        settled
    )

    wins = int(
        settled[
            "won"
        ].sum()
    )

    losses = (
        bets
        -
        wins
    )

    profit = float(
        settled[
            "profit_units"
        ].sum()
    )

    roi = (
        profit
        /
        bets
        if bets
        else np.nan
    )

    print(
        f"Settled bets:   {bets}"
    )

    print(
        f"Record:         {wins}-{losses}"
    )

    print(
        f"Win rate:       "
        f"{wins / bets:.2%}"
    )

    print(
        f"Profit:         "
        f"{profit:+.2f}u"
    )

    print(
        f"Flat-stake ROI: "
        f"{roi:+.2%}"
    )

    if (
        "history_type"
        in
        settled.columns
    ):

        print()
        print(
            "PERFORMANCE BY HISTORY TYPE"
        )
        print()

        rows = []

        for (
            history_type,
            group,
        ) in settled.groupby(
            "history_type",
            dropna=False,
        ):

            n = len(group)

            w = int(
                group[
                    "won"
                ].sum()
            )

            p = float(
                group[
                    "profit_units"
                ].sum()
            )

            rows.append(
                {
                    "history_type":
                        history_type,

                    "bets":
                        n,

                    "record":
                        f"{w}-{n-w}",

                    "profit_units":
                        round(
                            p,
                            2,
                        ),

                    "roi":
                        f"{p / n:+.2%}",
                }
            )

        print(
            pd.DataFrame(
                rows
            ).to_string(
                index=False
            )
        )



# ============================================================
# TOTALS FORWARD TEST
# ============================================================

def print_totals_forward_test():

    totals_file = (
        LIVE
        / "v5_live_totals_bet_ledger.csv"
    )

    print()
    print("=" * 110)
    print("V5 TOTALS FORWARD TEST")
    print("=" * 110)
    print()

    print(
        "Frozen strategy: "
        "RAW V5 UNDER 2.5 EDGE >= 11.0%"
    )

    print(
        "Validated leagues: "
        "Premier League, Bundesliga"
    )

    if not totals_file.exists():

        print()
        print(
            "No totals ledger found."
        )

        return

    df = pd.read_csv(
        totals_file,
        low_memory=False,
    )

    if df.empty:

        print()
        print(
            "No qualifying totals signals "
            "have been recorded yet."
        )

        print()
        print(
            "Settled bets:   0"
        )

        print(
            "Profit:         +0.00u"
        )

        print(
            "Flat-stake ROI: -"
        )

        print()
        print(
            "Closing observations: 0"
        )

        return

    # ========================================================
    # TODAY'S SIGNALS
    # ========================================================

    today = local_today()

    df["_date"] = pd.to_datetime(
        df.get(
            "date"
        ),
        errors="coerce",
    ).dt.date

    today_df = df[
        df["_date"].eq(
            today
        )
    ].copy()

    print()
    print("-" * 110)
    print("TODAY'S TOTALS SIGNALS")
    print("-" * 110)
    print()

    if today_df.empty:

        print(
            "No qualifying totals signals today."
        )

    else:

        today_df[
            "ODDS"
        ] = today_df[
            "decimal_odds"
        ].map(
            american_string
        )

        today_df[
            "MODEL"
        ] = today_df[
            "model_probability"
        ].map(
            pct_string
        )

        today_df[
            "MARKET"
        ] = today_df[
            "market_probability"
        ].map(
            pct_string
        )

        today_df[
            "EDGE"
        ] = today_df[
            "edge"
        ].map(
            pct_string
        )

        cols = [
            c
            for c in [
                "league",
                "home_team",
                "away_team",
                "selection",
                "ODDS",
                "bookmaker",
                "MODEL",
                "MARKET",
                "EDGE",
                "validation_status",
                "status",
            ]
            if c in today_df.columns
        ]

        print(
            today_df[
                cols
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # OPEN TOTALS BETS
    # ========================================================

    print()
    print("-" * 110)
    print("UPCOMING TOTALS BETS")
    print("-" * 110)
    print()

    if "status" not in df.columns:

        print(
            "No status column."
        )

    else:

        open_bets = df[
            df["status"]
            .astype(str)
            .str.upper()
            .eq("OPEN")
        ].copy()

        if open_bets.empty:

            print(
                "No open totals bets."
            )

        else:

            open_bets[
                "ODDS"
            ] = open_bets[
                "decimal_odds"
            ].map(
                american_string
            )

            open_bets[
                "EDGE"
            ] = open_bets[
                "edge"
            ].map(
                pct_string
            )

            if (
                "commence_time"
                in
                open_bets.columns
            ):

                open_bets[
                    "commence_time"
                ] = pd.to_datetime(
                    open_bets[
                        "commence_time"
                    ],
                    utc=True,
                    errors="coerce",
                )

                open_bets = (
                    open_bets
                    .sort_values(
                        "commence_time"
                    )
                )

            cols = [
                c
                for c in [
                    "date",
                    "league",
                    "home_team",
                    "away_team",
                    "selection",
                    "ODDS",
                    "bookmaker",
                    "EDGE",
                    "validation_status",
                    "commence_time",
                ]
                if c in open_bets.columns
            ]

            print(
                open_bets[
                    cols
                ].to_string(
                    index=False
                )
            )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    print()
    print("-" * 110)
    print("TOTALS PERFORMANCE")
    print("-" * 110)
    print()

    settled = df[
        df["status"]
        .astype(str)
        .str.upper()
        .eq("SETTLED")
    ].copy()

    if settled.empty:

        print(
            "Settled bets:   0"
        )

        print(
            "Profit:         +0.00u"
        )

        print(
            "Flat-stake ROI: -"
        )

    else:

        settled[
            "won"
        ] = pd.to_numeric(
            settled[
                "won"
            ],
            errors="coerce",
        ).fillna(0)

        settled[
            "profit_units"
        ] = pd.to_numeric(
            settled[
                "profit_units"
            ],
            errors="coerce",
        ).fillna(0)

        bets = len(
            settled
        )

        wins = int(
            settled[
                "won"
            ].sum()
        )

        losses = (
            bets
            -
            wins
        )

        profit = float(
            settled[
                "profit_units"
            ].sum()
        )

        roi = (
            profit
            /
            bets
            if bets
            else np.nan
        )

        print(
            f"Settled bets:   {bets}"
        )

        print(
            f"Record:         "
            f"{wins}-{losses}"
        )

        print(
            f"Win rate:       "
            f"{wins / bets:.2%}"
        )

        print(
            f"Profit:         "
            f"{profit:+.2f}u"
        )

        print(
            f"Flat-stake ROI: "
            f"{roi:+.2%}"
        )

    # ========================================================
    # CLV
    # ========================================================

    print()
    print("-" * 110)
    print("TOTALS CLOSING LINE VALUE")
    print("-" * 110)
    print()

    captured = df[
        pd.to_numeric(
            df[
                "closing_decimal_odds"
            ],
            errors="coerce",
        ).notna()
    ].copy()

    if captured.empty:

        print(
            "Closing observations: 0"
        )

    else:

        captured[
            "price_clv"
        ] = pd.to_numeric(
            captured[
                "price_clv"
            ],
            errors="coerce",
        )

        captured[
            "probability_clv"
        ] = pd.to_numeric(
            captured[
                "probability_clv"
            ],
            errors="coerce",
        )

        captured[
            "beat_close"
        ] = pd.to_numeric(
            captured[
                "beat_close"
            ],
            errors="coerce",
        )

        print(
            f"Closing observations: "
            f"{len(captured)}"
        )

        print(
            f"Average price CLV:    "
            f"{captured['price_clv'].mean():+.2%}"
        )

        print(
            f"Average prob. CLV:    "
            f"{captured['probability_clv'].mean():+.2%}"
        )

        print(
            f"Beat close:           "
            f"{captured['beat_close'].mean():.2%}"
        )


# ============================================================
# ACTUAL WAGERING
# ============================================================

def print_actual_bankroll():

    print()
    print("=" * 110)
    print("ACTUAL WAGERING BANKROLL")
    print("=" * 110)
    print()

    if not ACTUAL_FILE.exists():

        print(
            "No actual-wager ledger found."
        )

        return

    df = pd.read_csv(
        ACTUAL_FILE,
        low_memory=False,
    )

    if df.empty:

        print(
            "Actual-wager ledger is empty."
        )

        return

    starting = pd.to_numeric(
        df.get(
            "starting_bankroll"
        ),
        errors="coerce",
    ).dropna()

    if starting.empty:

        starting_bankroll = np.nan

    else:

        starting_bankroll = float(
            starting.iloc[0]
        )

    stake = pd.to_numeric(
        df.get(
            "stake"
        ),
        errors="coerce",
    ).fillna(0)

    profit = pd.to_numeric(
        df.get(
            "profit"
        ),
        errors="coerce",
    ).fillna(0)

    total_staked = float(
        stake.sum()
    )

    total_profit = float(
        profit.sum()
    )

    if np.isfinite(
        starting_bankroll
    ):

        current = (
            starting_bankroll
            +
            total_profit
        )

    else:

        current = np.nan

    roi = (
        total_profit
        /
        total_staked
        if total_staked > 0
        else np.nan
    )

    print(
        f"Starting bankroll: "
        f"{money(starting_bankroll)}"
    )

    print(
        f"Total staked:      "
        f"{money(total_staked)}"
    )

    print(
        f"Net profit:        "
        f"${total_profit:+,.2f}"
    )

    if np.isfinite(
        roi
    ):

        print(
            f"Actual ROI:        "
            f"{roi:+.2%}"
        )

    print(
        f"Current bankroll:  "
        f"{money(current)}"
    )

    print()
    print(
        "ACTUAL WAGERS"
    )
    print()

    cols = [
        c
        for c in [
            "date",
            "wager_type",
            "selection",
            "american_odds",
            "stake",
            "result",
            "return",
            "profit",
        ]
        if c in df.columns
    ]

    print(
        df[
            cols
        ]
        .tail(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# CLV SUMMARY
# ============================================================

def print_clv():

    print()
    print("=" * 110)
    print("CLOSING LINE VALUE")
    print("=" * 110)
    print()

    if not CLV_FILE.exists():

        print(
            "No CLV report found."
        )

        return

    df = pd.read_csv(
        CLV_FILE,
        low_memory=False,
    )

    if df.empty:

        print(
            "No CLV observations yet."
        )

        return

    def format_american(
        value,
    ):

        try:
            x = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return "-"

        if not np.isfinite(
            x
        ):
            return "-"

        return f"{int(round(x)):+d}"

    if (
        "signal_american"
        in
        df.columns
    ):

        df[
            "SIGNAL"
        ] = df[
            "signal_american"
        ].map(
            format_american
        )

    if (
        "last_consensus_american"
        in
        df.columns
    ):

        df[
            "MARKET"
        ] = df[
            "last_consensus_american"
        ].map(
            format_american
        )

    if (
        "last_best_american"
        in
        df.columns
    ):

        df[
            "BEST"
        ] = df[
            "last_best_american"
        ].map(
            format_american
        )

    if (
        "probability_clv_pct"
        in
        df.columns
    ):

        df[
            "CLV"
        ] = pd.to_numeric(
            df[
                "probability_clv_pct"
            ],
            errors="coerce",
        ).map(
            lambda x:
            f"{x:+.2f}%"
            if pd.notna(x)
            else "-"
        )

    if (
        "true_close_available"
        in
        df.columns
    ):

        close_bool = (
            df[
                "true_close_available"
            ]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        )

        df[
            "CLOSE"
        ] = np.where(
            close_bool,
            "YES",
            "NO",
        )

    cols = [
        c
        for c in [
            "league",
            "home_team",
            "away_team",
            "bet_side",
            "SIGNAL",
            "MARKET",
            "BEST",
            "CLV",
            "CLOSE",
        ]
        if c in df.columns
    ]

    print(
        df[
            cols
        ]
        .tail(20)
        .to_string(
            index=False
        )
    )

    true_close_count = 0

    if (
        "true_close_available"
        in
        df.columns
    ):

        true_close_count = int(
            df[
                "true_close_available"
            ]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
            .sum()
        )

    print()
    print(
        f"True closing-line observations: "
        f"{true_close_count}"
    )

    print(
        "Near-kickoff collection is handled "
        "automatically by launchd."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    local_now = datetime.now(
        ZoneInfo(
            "America/Detroit"
        )
    )

    print()
    print("#" * 110)
    print("V5 LIVE SOCCER — DAILY RUNNER")
    print("#" * 110)
    print()

    print(
        "Local time:",
        local_now.strftime(
            "%Y-%m-%d %I:%M:%S %p %Z"
        ),
    )

    print(
        "Python:",
        sys.executable,
    )

    print()

    print(
        "Model selection rule: "
        "RAW V5 EDGE >= 16.0%"
    )

    print(
        "Staking in model ledger: "
        "flat 1 unit"
    )

    print(
        "Displayed prices: "
        "AMERICAN ODDS"
    )

    # ========================================================
    # RUN PIPELINE
    # ========================================================

    for (
        title,
        filename,
    ) in PIPELINE:

        run_script(
            title,
            filename,
        )

    # ========================================================
    # FINAL DASHBOARD
    # ========================================================

    print()
    print()
    print("#" * 110)
    print("V5 LIVE DASHBOARD")
    print("#" * 110)

    print_today_model_bets()

    print_open_model_bets()

    print_invalid_signals()

    print_model_performance()

    print_totals_forward_test()

    print_actual_bankroll()

    print_clv()

    # ========================================================
    # COMPLETE CEMENTED BOARD
    # ========================================================

    print_cemented_board()

    print()
    print("#" * 110)
    print("DAILY RUN COMPLETE")
    print("#" * 110)
    print()

    print(
        "Model ledger and actual wagering ledger "
        "remain separate."
    )

    print(
        "Closing-line snapshots continue "
        "automatically in the background."
    )

    print()


if __name__ == "__main__":
    main()
