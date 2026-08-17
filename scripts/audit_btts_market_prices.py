from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "btts_cfg0755_market_matched.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "btts_market_price_audit"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def banner(title):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================
# LOAD MAIN DATASET
# ============================================================

banner("BTTS MARKET PRICE / CLV AUDIT")

df = pd.read_csv(
    INPUT,
    low_memory=False,
)

print("Input:", INPUT)
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 1. COLUMN INVENTORY
# ============================================================

banner("MARKET DATASET COLUMN INVENTORY")

keywords = [
    "odds",
    "price",
    "market",
    "book",
    "sportsbook",
    "source",
    "open",
    "opening",
    "close",
    "closing",
    "timestamp",
    "time",
    "updated",
    "commence",
    "snapshot",
    "provider",
]


interesting_cols = [
    c
    for c in df.columns
    if any(
        k in c.lower()
        for k in keywords
    )
]


for c in interesting_cols:
    print(c)


# ============================================================
# 2. IDENTIFY CORE BTTS PRICE COLUMNS
# ============================================================

required = [
    "odds_yes",
    "odds_no",
    "market_yes",
]

missing = [
    c
    for c in required
    if c not in df.columns
]

if missing:
    raise SystemExit(
        f"Missing required market columns: {missing}"
    )


df["odds_yes"] = safe_numeric(
    df["odds_yes"]
)

df["odds_no"] = safe_numeric(
    df["odds_no"]
)

df["market_yes"] = safe_numeric(
    df["market_yes"]
)


if "market_no" in df.columns:
    df["market_no"] = safe_numeric(
        df["market_no"]
    )
else:
    df["market_no"] = (
        1.0
        -
        df["market_yes"]
    )


# ============================================================
# 3. RAW IMPLIED PROBABILITIES + VIG
# ============================================================

df["raw_imp_yes"] = (
    1.0
    /
    df["odds_yes"]
)

df["raw_imp_no"] = (
    1.0
    /
    df["odds_no"]
)

df["raw_overround"] = (
    df["raw_imp_yes"]
    +
    df["raw_imp_no"]
)

df["book_margin"] = (
    df["raw_overround"]
    -
    1.0
)

df["calc_novig_yes"] = (
    df["raw_imp_yes"]
    /
    df["raw_overround"]
)

df["calc_novig_no"] = (
    df["raw_imp_no"]
    /
    df["raw_overround"]
)

df["market_yes_diff"] = (
    df["market_yes"]
    -
    df["calc_novig_yes"]
)

df["market_no_diff"] = (
    df["market_no"]
    -
    df["calc_novig_no"]
)


banner("PRICE CONSTRUCTION CHECK")

print(
    df[
        [
            "odds_yes",
            "odds_no",
            "raw_imp_yes",
            "raw_imp_no",
            "raw_overround",
            "book_margin",
            "market_yes",
            "calc_novig_yes",
            "market_yes_diff",
        ]
    ]
    .describe()
    .round(6)
    .to_string()
)


print()
print(
    "Max abs difference between stored market_yes "
    "and calculated proportional no-vig:"
)

print(
    f"{df['market_yes_diff'].abs().max():.10f}"
)


# ============================================================
# 4. YEAR / LEAGUE PRICE CHARACTERISTICS
# ============================================================

if "test_year" in df.columns:
    df["test_year"] = safe_numeric(
        df["test_year"]
    )


group_cols = []

if "test_year" in df.columns:
    group_cols.append(
        "test_year"
    )

if "league" in df.columns:
    group_cols.append(
        "league"
    )


if group_cols:

    price_summary = (
        df
        .groupby(
            group_cols
        )
        .agg(
            games=(
                "odds_yes",
                "size",
            ),

            avg_yes_odds=(
                "odds_yes",
                "mean",
            ),

            median_yes_odds=(
                "odds_yes",
                "median",
            ),

            avg_no_odds=(
                "odds_no",
                "mean",
            ),

            median_no_odds=(
                "odds_no",
                "median",
            ),

            avg_margin=(
                "book_margin",
                "mean",
            ),

            median_margin=(
                "book_margin",
                "median",
            ),

            min_margin=(
                "book_margin",
                "min",
            ),

            max_margin=(
                "book_margin",
                "max",
            ),

            avg_market_yes=(
                "market_yes",
                "mean",
            ),
        )
        .reset_index()
    )


    banner("PRICE CHARACTERISTICS BY YEAR / LEAGUE")

    print(
        price_summary
        .round(6)
        .to_string(
            index=False
        )
    )

else:
    price_summary = pd.DataFrame()


# ============================================================
# 5. ODDS PRECISION / ROUNDING SIGNATURE
#
# This can sometimes reveal whether prices came directly from
# sportsbooks, consensus feeds, converted decimals, etc.
# ============================================================

banner("ODDS ROUNDING / PRECISION SIGNATURE")

for col in [
    "odds_yes",
    "odds_no",
]:

    x = (
        df[col]
        .dropna()
        .astype(float)
    )

    print()
    print(col)

    print(
        "Unique values:",
        x.nunique()
    )

    print(
        "Most common values:"
    )

    print(
        x.value_counts()
        .head(20)
        .to_string()
    )


# ============================================================
# 6. DUPLICATE MATCH CHECK
#
# Multiple rows per match could imply multiple snapshots,
# bookmakers or timestamps.
# ============================================================

match_keys = [
    c
    for c in [
        "date",
        "league",
        "home_team",
        "away_team",
    ]
    if c in df.columns
]


if len(match_keys) >= 3:

    dup_counts = (
        df.groupby(
            match_keys,
            dropna=False,
        )
        .size()
        .reset_index(
            name="rows_per_match"
        )
    )


    banner("MULTIPLE PRICE ROWS PER MATCH")

    print(
        dup_counts[
            "rows_per_match"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Maximum rows for one match:",
        dup_counts[
            "rows_per_match"
        ].max()
    )

else:
    dup_counts = pd.DataFrame()


# ============================================================
# 7. TIMESTAMP / SOURCE METADATA CHECK
# ============================================================

metadata_candidates = [
    c
    for c in df.columns
    if any(
        k in c.lower()
        for k in [
            "timestamp",
            "updated",
            "commence",
            "source",
            "provider",
            "book",
            "sportsbook",
            "open",
            "close",
            "snapshot",
            "time",
        ]
    )
]


banner("TIMESTAMP / SOURCE METADATA")

if metadata_candidates:

    for c in metadata_candidates:

        print()
        print("COLUMN:", c)

        series = df[c]

        print(
            "Non-null:",
            series.notna().sum(),
        )

        print(
            "Unique:",
            series.nunique(
                dropna=True
            ),
        )

        print(
            "Sample:"
        )

        print(
            series
            .dropna()
            .astype(str)
            .head(10)
            .to_string(
                index=False
            )
        )

else:

    print(
        "No obvious timestamp/source/opening/closing "
        "metadata in processed market file."
    )


# ============================================================
# 8. PROJECT-WIDE ODDS FILE DISCOVERY
#
# Scan data directory headers only.
# We are looking for source files that might preserve
# opening/closing/timestamps/bookmaker fields.
# ============================================================

banner("PROJECT-WIDE ODDS / MARKET FILE DISCOVERY")

DATA_ROOT = (
    ROOT
    /
    "data"
)

candidate_files = []


name_terms = [
    "odd",
    "market",
    "price",
    "book",
    "btts",
]


for p in DATA_ROOT.rglob("*"):

    if not p.is_file():
        continue

    if p.suffix.lower() not in [
        ".csv",
        ".parquet",
    ]:
        continue

    lower_name = p.name.lower()

    if any(
        t in lower_name
        for t in name_terms
    ):

        candidate_files.append(
            p
        )


print(
    "Candidate files discovered:",
    len(candidate_files),
)


file_inventory = []


column_terms = [
    "odds",
    "price",
    "market",
    "book",
    "sportsbook",
    "source",
    "provider",
    "open",
    "opening",
    "close",
    "closing",
    "timestamp",
    "updated",
    "commence",
    "snapshot",
]


for p in sorted(
    candidate_files
):

    try:

        if p.suffix.lower() == ".csv":

            temp = pd.read_csv(
                p,
                nrows=3,
                low_memory=False,
            )

        else:

            temp = pd.read_parquet(
                p
            ).head(3)


        cols = list(
            temp.columns
        )


        market_cols = [
            c
            for c in cols
            if any(
                term in c.lower()
                for term in column_terms
            )
        ]


        file_inventory.append(
            {
                "file":
                    str(
                        p.relative_to(
                            ROOT
                        )
                    ),

                "num_columns":
                    len(cols),

                "market_related_columns":
                    " | ".join(
                        market_cols
                    ),

                "all_columns":
                    " | ".join(
                        cols
                    ),
            }
        )


    except Exception as exc:

        file_inventory.append(
            {
                "file":
                    str(
                        p.relative_to(
                            ROOT
                        )
                    ),

                "num_columns":
                    np.nan,

                "market_related_columns":
                    f"READ_ERROR: {exc}",

                "all_columns":
                    "",
            }
        )


inventory_df = pd.DataFrame(
    file_inventory
)


if len(inventory_df):

    for _, row in inventory_df.iterrows():

        print()
        print(
            row[
                "file"
            ]
        )

        print(
            "  market cols:",
            row[
                "market_related_columns"
            ]
        )

else:

    print(
        "No candidate odds/market files found."
    )


# ============================================================
# 9. CLASSIFY CURRENT PRICE DATA
# ============================================================

banner("AUTOMATIC PRICE-PROVENANCE ASSESSMENT")


has_open_col = any(
    re.search(
        r"(^|_)open(ing)?($|_)",
        c.lower(),
    )
    for c in df.columns
)

has_close_col = any(
    re.search(
        r"(^|_)clos(e|ing)($|_)",
        c.lower(),
    )
    for c in df.columns
)

has_timestamp_col = any(
    any(
        t in c.lower()
        for t in [
            "timestamp",
            "updated",
            "commence",
            "snapshot",
        ]
    )
    for c in df.columns
)

has_book_col = any(
    any(
        t in c.lower()
        for t in [
            "bookmaker",
            "sportsbook",
            "book_name",
            "provider",
        ]
    )
    for c in df.columns
)


if has_open_col and has_close_col:

    classification = (
        "OPENING + CLOSING PRICE FIELDS FOUND"
    )

elif has_close_col:

    classification = (
        "CLOSING-LABELLED PRICE FIELD FOUND"
    )

elif has_open_col:

    classification = (
        "OPENING-LABELLED PRICE FIELD FOUND"
    )

elif has_timestamp_col:

    classification = (
        "TIMESTAMPED PRICE DATA MAY EXIST"
    )

else:

    classification = (
        "SINGLE UNLABELLED MARKET SNAPSHOT"
    )


print(
    "Processed dataset classification:"
)

print(
    classification
)


print()
print(
    "Bookmaker/provider metadata:",
    has_book_col,
)

print(
    "Timestamp metadata:",
    has_timestamp_col,
)

print(
    "Opening fields:",
    has_open_col,
)

print(
    "Closing fields:",
    has_close_col,
)


# ============================================================
# 10. CLV READINESS
# ============================================================

banner("CLV READINESS")


true_clv_ready = (
    has_close_col
    or
    (
        has_timestamp_col
        and
        (
            dup_counts[
                "rows_per_match"
            ].max()
            >
            1
            if len(
                dup_counts
            )
            else False
        )
    )
)


if true_clv_ready:

    print(
        "TRUE CLV ANALYSIS MAY BE POSSIBLE "
        "FROM EXISTING DATA."
    )

else:

    print(
        "TRUE CLV CANNOT YET BE PROVEN FROM "
        "THE PROCESSED MARKET FILE."
    )

    print()

    print(
        "For real CLV we need, per match:"
    )

    print(
        "  1. Price available when our model issued the pick"
    )

    print(
        "  2. Closing market price"
    )

    print(
        "  3. Timestamp for both"
    )

    print(
        "  4. Preferably bookmaker or consensus source"
    )


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUT_DIR
    /
    "01_market_rows_enriched.csv",
    index=False,
)


if len(
    price_summary
):

    price_summary.to_csv(
        OUT_DIR
        /
        "02_price_summary_by_year_league.csv",
        index=False,
    )


if len(
    dup_counts
):

    dup_counts.to_csv(
        OUT_DIR
        /
        "03_rows_per_match.csv",
        index=False,
    )


inventory_df.to_csv(
    OUT_DIR
    /
    "04_project_market_file_inventory.csv",
    index=False,
)


assessment = pd.DataFrame(
    [
        {
            "classification":
                classification,

            "has_bookmaker_metadata":
                has_book_col,

            "has_timestamp_metadata":
                has_timestamp_col,

            "has_opening_fields":
                has_open_col,

            "has_closing_fields":
                has_close_col,

            "true_clv_ready":
                true_clv_ready,

            "rows":
                len(df),

            "avg_book_margin":
                df[
                    "book_margin"
                ].mean(),

            "median_book_margin":
                df[
                    "book_margin"
                ].median(),

            "max_market_yes_novig_diff":
                df[
                    "market_yes_diff"
                ]
                .abs()
                .max(),
        }
    ]
)


assessment.to_csv(
    OUT_DIR
    /
    "05_price_provenance_assessment.csv",
    index=False,
)


banner("OUTPUTS")

for p in sorted(
    OUT_DIR.glob("*")
):
    print(p)


print()
print("DONE")
