from pathlib import Path
import importlib.util
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            low_memory=False,
        )
    except Exception:
        return pd.DataFrame()


def event_id_col(df):
    if "event_id" in df.columns:
        return "event_id"

    if "match_id" in df.columns:
        return "match_id"

    return None


def build_model_map():
    """
    Discover current V5 prediction files and build one model row
    per event ID.
    """

    rows = []

    for path in LIVE.glob("*v5*prediction*.csv"):

        x = load_csv(path)

        if x.empty:
            continue

        id_col = event_id_col(x)

        if id_col is None:
            continue

        x = x.copy()

        x["_event_id"] = (
            x[id_col]
            .astype(str)
        )

        keep = [
            c for c in [
                "_event_id",
                "league",
                "home_team",
                "away_team",
                "p_home_v5",
                "p_draw_v5",
                "p_away_v5",
                "p_btts_yes_v5",
                "p_btts_no_v5",
                "p_over_2_5_v5",
                "p_under_2_5_v5",
            ]
            if c in x.columns
        ]

        rows.append(
            x[keep]
        )

    if not rows:
        return pd.DataFrame(
            columns=["_event_id"]
        )

    model = pd.concat(
        rows,
        ignore_index=True,
        sort=False,
    )

    model = (
        model
        .drop_duplicates(
            "_event_id",
            keep="last",
        )
    )

    return model


def complete_market_ids(
    markets,
    market_name,
    required_selections,
):
    """
    Return event IDs for which at least one bookmaker has
    every required side of a market.
    """

    if markets.empty:
        return set()

    required_cols = {
        "event_id",
        "bookmaker",
        "market",
        "selection",
    }

    if not required_cols.issubset(
        markets.columns
    ):
        return set()

    x = markets[
        markets["market"]
        .astype(str)
        .str.upper()
        .eq(market_name.upper())
    ].copy()

    if x.empty:
        return set()

    x["_selection"] = (
        x["selection"]
        .astype(str)
        .str.upper()
    )

    wanted = {
        str(v).upper()
        for v in required_selections
    }

    x = x[
        x["_selection"].isin(
            wanted
        )
    ].copy()

    if x.empty:
        return set()

    counts = (
        x.groupby(
            [
                "event_id",
                "bookmaker",
            ]
        )["_selection"]
        .nunique()
    )

    complete = (
        counts[
            counts >= len(wanted)
        ]
        .reset_index()
    )

    return set(
        complete["event_id"]
        .astype(str)
    )


def model_value(
    model,
    event_id,
    column,
):
    if (
        model.empty
        or
        column not in model.columns
    ):
        return None

    x = model[
        model["_event_id"]
        .astype(str)
        .eq(str(event_id))
    ]

    if x.empty:
        return None

    value = pd.to_numeric(
        x[column],
        errors="coerce",
    ).iloc[0]

    if pd.isna(value):
        return None

    return float(value)


def fmt_pct(value):
    if value is None:
        return "N/A"

    return f"{value:.2%}"


def print_status(
    league,
    home,
    away,
    model_text,
    status,
):
    matchup = f"{home} vs {away}"

    print(
        f"{str(league):<20} | "
        f"{matchup:<42} | "
        f"{model_text:<22} | "
        f"{status}"
    )


# ============================================================
# PRINT EXISTING OFFICIAL / CEMENTED BOARD
# ============================================================

runner_path = (
    ROOT
    / "scripts"
    / "run_live_v5.py"
)

spec = importlib.util.spec_from_file_location(
    "run_live_v5",
    runner_path,
)

runner = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    runner
)

runner.print_cemented_board()


# ============================================================
# LOAD CURRENT DATA
# ============================================================

events = load_csv(
    LIVE / "odds_events_snapshot.csv"
)

markets = load_csv(
    LIVE / "odds_markets_snapshot.csv"
)

model = build_model_map()


print()
print("#" * 118)
print(
    "MARKET PRICE-COVERAGE AUDIT — CURRENT SCAN"
)
print("#" * 118)


if events.empty:

    print(
        "No current odds event snapshot available."
    )

    raise SystemExit(0)


if "event_id" not in events.columns:

    print(
        "Current event snapshot has no event_id column."
    )

    raise SystemExit(0)


events["_event_id"] = (
    events["event_id"]
    .astype(str)
)


# ============================================================
# COMPLETE MARKET COVERAGE SETS
# ============================================================

# 1X2 may be represented as H2H in the markets snapshot.
# If the current pipeline stores 1X2 elsewhere, we also use the
# bookmaker snapshot / H2H history below as a fallback.

btts_complete = complete_market_ids(
    markets,
    "BTTS",
    [
        "YES",
        "NO",
    ],
)

totals_complete = complete_market_ids(
    markets,
    "TOTALS",
    [
        "OVER",
        "UNDER",
    ],
)


# ============================================================
# 1X2 COVERAGE
# ============================================================

print()
print("=" * 118)
print("1X2 PRICE COVERAGE")
print("=" * 118)

h2h = load_csv(
    LIVE / "odds_h2h_history.csv"
)

h2h_complete = set()

if (
    not h2h.empty
    and
    "event_id" in h2h.columns
):

    required = [
        "home_odds",
        "draw_odds",
        "away_odds",
    ]

    if all(
        c in h2h.columns
        for c in required
    ):

        y = h2h.copy()

        for c in required:
            y[c] = pd.to_numeric(
                y[c],
                errors="coerce",
            )

        y = y[
            y[required]
            .notna()
            .all(axis=1)
        ].copy()

        h2h_complete = set(
            y["event_id"]
            .astype(str)
        )


for row in events.itertuples(
    index=False
):

    event_id = str(
        row.event_id
    )

    p_home = model_value(
        model,
        event_id,
        "p_home_v5",
    )

    p_draw = model_value(
        model,
        event_id,
        "p_draw_v5",
    )

    p_away = model_value(
        model,
        event_id,
        "p_away_v5",
    )

    model_ready = all(
        p is not None
        for p in [
            p_home,
            p_draw,
            p_away,
        ]
    )

    priced = (
        event_id
        in
        h2h_complete
    )

    if not model_ready:

        status = (
            "MODEL UNAVAILABLE ❌"
        )

    elif priced:

        status = (
            "PRICE CAPTURED ✅"
        )

    else:

        status = (
            "MANUAL 1X2 PRICE CHECK ⚠️"
        )

    model_text = (
        "Model 1X2: "
        f"{fmt_pct(p_home)}/"
        f"{fmt_pct(p_draw)}/"
        f"{fmt_pct(p_away)}"
    )

    print_status(
        row.league,
        row.home_team,
        row.away_team,
        model_text,
        status,
    )


# ============================================================
# BTTS SPECIALIST COVERAGE
# ============================================================

print()
print("=" * 118)
print("BTTS SPECIALIST PRICE COVERAGE")
print("=" * 118)

BTTS_LEAGUES = {
    "Swiss Super League",
    "Super Lig",
    "Segunda División",
}

btts_events = events[
    events["league"]
    .isin(BTTS_LEAGUES)
].copy()

if btts_events.empty:

    print(
        "No BTTS specialist fixtures "
        "in current scan."
    )

else:

    for row in btts_events.itertuples(
        index=False
    ):

        event_id = str(
            row.event_id
        )

        p_yes = model_value(
            model,
            event_id,
            "p_btts_yes_v5",
        )

        if p_yes is None:

            status = (
                "MODEL UNAVAILABLE ❌"
            )

        elif event_id in btts_complete:

            status = (
                "PRICE CAPTURED ✅"
            )

        else:

            status = (
                "MANUAL BTTS PRICE CHECK ⚠️"
            )

        print_status(
            row.league,
            row.home_team,
            row.away_team,
            (
                "Model YES: "
                f"{fmt_pct(p_yes)}"
            ),
            status,
        )


# ============================================================
# TOTALS SPECIALIST COVERAGE
# ============================================================

print()
print("=" * 118)
print("O/U 2.5 SPECIALIST PRICE COVERAGE")
print("=" * 118)

TOTALS_LEAGUES = {
    "Premier League",
    "Bundesliga",
    "Belgian Pro League",
    "Eliteserien",
}

totals_events = events[
    events["league"]
    .isin(TOTALS_LEAGUES)
].copy()

if totals_events.empty:

    print(
        "No O/U 2.5 specialist fixtures "
        "in current scan."
    )

else:

    for row in totals_events.itertuples(
        index=False
    ):

        event_id = str(
            row.event_id
        )

        p_over = model_value(
            model,
            event_id,
            "p_over_2_5_v5",
        )

        p_under = model_value(
            model,
            event_id,
            "p_under_2_5_v5",
        )

        model_ready = (
            p_over is not None
            and
            p_under is not None
        )

        priced = (
            event_id
            in
            totals_complete
        )

        if not model_ready:

            status = (
                "MODEL UNAVAILABLE ❌"
            )

        elif priced:

            status = (
                "PRICE CAPTURED ✅"
            )

        else:

            status = (
                "MANUAL O/U 2.5 PRICE CHECK ⚠️"
            )

        model_text = (
            "O:"
            f"{fmt_pct(p_over)} "
            "U:"
            f"{fmt_pct(p_under)}"
        )

        print_status(
            row.league,
            row.home_team,
            row.away_team,
            model_text,
            status,
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("#" * 118)
print("PRICE-COVERAGE AUDIT COMPLETE")
print("#" * 118)
print(
    "✅ PRICE CAPTURED = automatic market evaluation available"
)
print(
    "⚠️ MANUAL PRICE CHECK = model exists but API price is incomplete/missing"
)
print(
    "❌ MODEL UNAVAILABLE = price alone cannot produce an official model bet"
)
print()
print(
    "Warnings do NOT create bets or change any cemented thresholds."
)
