from pathlib import Path
import math
import re
import unicodedata

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
    / "footystats_multileague_v5_predictions.csv"
)

MARKET_DIR = (
    ROOT
    / "data"
    / "raw"
    / "footystats_mls"
)

OUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "mls_v5_btts_market_matched.csv"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_team(value):

    value = str(value)

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )

    value = value.lower()

    value = value.replace(
        "&",
        " and ",
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    aliases = {
        "atlanta united fc": "atlanta united",
        "austin fc": "austin",
        "charlotte fc": "charlotte",
        "chicago fire fc": "chicago fire",
        "colorado rapids": "colorado rapids",
        "columbus crew": "columbus crew",
        "dc united": "dc united",
        "d c united": "dc united",
        "fc cincinnati": "cincinnati",
        "fc dallas": "dallas",
        "houston dynamo": "houston dynamo",
        "houston dynamo fc": "houston dynamo",
        "inter miami": "inter miami",
        "inter miami cf": "inter miami",
        "la galaxy": "la galaxy",
        "los angeles galaxy": "la galaxy",
        "los angeles fc": "los angeles fc",
        "lafc": "los angeles fc",
        "minnesota united": "minnesota united",
        "minnesota united fc": "minnesota united",
        "cf montreal": "montreal impact",
        "cf montr al": "montreal impact",
        "montreal impact": "montreal impact",
        "nashville sc": "nashville",
        "nashville": "nashville",
        "new england revolution": "new england revolution",
        "new york city": "new york city",
        "new york city fc": "new york city",
        "new york red bulls": "new york red bulls",
        "orlando city": "orlando city",
        "orlando city sc": "orlando city",
        "philadelphia union": "philadelphia union",
        "portland timbers": "portland timbers",
        "real salt lake": "real salt lake",
        "san jose earthquakes": "san jose earthquakes",
        "seattle sounders": "seattle sounders",
        "seattle sounders fc": "seattle sounders",
        "sporting kansas city": "sporting kansas city",
        "st louis city": "st louis city",
        "st louis city sc": "st louis city",
        "toronto fc": "toronto",
        "toronto": "toronto",
        "vancouver whitecaps": "vancouver whitecaps",
        "vancouver whitecaps fc": "vancouver whitecaps",
    }

    return aliases.get(
        value,
        value,
    )


def p_btts(home_lambda, away_lambda):

    h = float(home_lambda)
    a = float(away_lambda)

    return (
        1.0
        - math.exp(-h)
        - math.exp(-a)
        + math.exp(-(h + a))
    )


def find_column(df, candidates):

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower:

            return lower[
                candidate.lower()
            ]

    return None


# ============================================================
# LOAD V5
# ============================================================

print()
print("=" * 100)
print("MLS V5 → FOOTYSTATS BTTS MARKET MATCH")
print("=" * 100)

pred = pd.read_csv(
    PRED_FILE,
    low_memory=False,
)

print()
print("Prediction rows:", len(pred))
print("Prediction columns:", len(pred.columns))


league_col = find_column(
    pred,
    [
        "league",
        "competition",
    ],
)

date_col = find_column(
    pred,
    [
        "date",
        "match_date",
    ],
)

home_col = find_column(
    pred,
    [
        "home_team",
        "home_name",
        "home",
    ],
)

away_col = find_column(
    pred,
    [
        "away_team",
        "away_name",
        "away",
    ],
)

home_lambda_col = find_column(
    pred,
    [
        "home_lambda",
        "lambda_home",
        "pred_home_lambda",
    ],
)

away_lambda_col = find_column(
    pred,
    [
        "away_lambda",
        "lambda_away",
        "pred_away_lambda",
    ],
)


required = {
    "league": league_col,
    "date": date_col,
    "home": home_col,
    "away": away_col,
    "home_lambda": home_lambda_col,
    "away_lambda": away_lambda_col,
}


print()
print("Detected V5 columns:")

for key, value in required.items():

    print(
        f"{key:15s}: {value}"
    )


missing = [
    key
    for key, value in required.items()
    if value is None
]


if missing:

    print()
    print(
        "Missing required V5 columns:",
        missing,
    )

    print()
    print("Available columns:")

    for col in pred.columns:

        print(col)

    raise SystemExit


# ============================================================
# FILTER MLS
# ============================================================

mls = pred[
    pred[league_col]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("mls")
].copy()


print()
print("MLS V5 rows:", len(mls))


if len(mls) == 0:

    print()
    print("No MLS rows found.")

    print()
    print(
        pred[league_col]
        .value_counts(dropna=False)
        .head(50)
        .to_string()
    )

    raise SystemExit


mls["match_date"] = pd.to_datetime(
    mls[date_col],
    errors="coerce",
).dt.normalize()


mls["home_norm"] = (
    mls[home_col]
    .map(normalize_team)
)

mls["away_norm"] = (
    mls[away_col]
    .map(normalize_team)
)


# ============================================================
# CALCULATE MODEL BTTS
# ============================================================

mls["model_btts_yes"] = mls.apply(
    lambda r: p_btts(
        r[home_lambda_col],
        r[away_lambda_col],
    )
    if (
        pd.notna(r[home_lambda_col])
        and pd.notna(r[away_lambda_col])
    )
    else np.nan,
    axis=1,
)


# ============================================================
# LOAD FOOTYSTATS MARKET
# ============================================================

market_frames = []

for year in range(
    2020,
    2026,
):

    path = (
        MARKET_DIR
        / f"mls_{year}_footystats.csv"
    )

    if not path.exists():

        print()
        print(
            "Missing market file:",
            path,
        )

        continue

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    df["source_year"] = year

    market_frames.append(df)


if not market_frames:

    print()
    print("No MLS market files found.")

    raise SystemExit


market = pd.concat(
    market_frames,
    ignore_index=True,
)


print()
print("Market rows:", len(market))


# ============================================================
# MARKET CLEANING
# ============================================================

market["match_date"] = pd.to_datetime(
    market["date_unix"],
    unit="s",
    errors="coerce",
    utc=True,
).dt.tz_convert(None).dt.normalize()


market["home_norm"] = (
    market["home_name"]
    .map(normalize_team)
)

market["away_norm"] = (
    market["away_name"]
    .map(normalize_team)
)


market["odds_btts_yes"] = pd.to_numeric(
    market["odds_btts_yes"],
    errors="coerce",
)

market["odds_btts_no"] = pd.to_numeric(
    market["odds_btts_no"],
    errors="coerce",
)


market["valid_btts_market"] = (
    (market["odds_btts_yes"] > 1)
    &
    (market["odds_btts_no"] > 1)
)


# ============================================================
# NO-VIG MARKET PROBABILITY
# ============================================================

market["raw_imp_yes"] = (
    1.0
    / market["odds_btts_yes"]
)

market["raw_imp_no"] = (
    1.0
    / market["odds_btts_no"]
)

market["market_overround"] = (
    market["raw_imp_yes"]
    +
    market["raw_imp_no"]
)

market["market_btts_yes_novig"] = (
    market["raw_imp_yes"]
    /
    market["market_overround"]
)

market["market_btts_no_novig"] = (
    market["raw_imp_no"]
    /
    market["market_overround"]
)


# ============================================================
# EXACT MATCH
# ============================================================

pred_match = mls.reset_index(
    drop=False
).rename(
    columns={
        "index": "pred_index"
    }
)


market_match = market.reset_index(
    drop=False
).rename(
    columns={
        "index": "market_index"
    }
)


exact = pred_match.merge(
    market_match,
    on=[
        "match_date",
        "home_norm",
        "away_norm",
    ],
    how="inner",
    suffixes=(
        "_pred",
        "_market",
    ),
)


matched_pred = set(
    exact["pred_index"]
)

matched_market = set(
    exact["market_index"]
)


print()
print("=" * 100)
print("EXACT MATCH")
print("=" * 100)

print()
print("Exact matches:", len(exact))


# ============================================================
# ±1 DAY FALLBACK
# ============================================================

fallback_rows = []

unmatched_pred = pred_match[
    ~pred_match["pred_index"].isin(
        matched_pred
    )
].copy()


unmatched_market = market_match[
    ~market_match["market_index"].isin(
        matched_market
    )
].copy()


market_lookup = {}

for row in unmatched_market.itertuples(
    index=False
):

    key = (
        row.home_norm,
        row.away_norm,
    )

    market_lookup.setdefault(
        key,
        [],
    ).append(row)


for prow in unmatched_pred.itertuples(
    index=False
):

    key = (
        prow.home_norm,
        prow.away_norm,
    )

    candidates = market_lookup.get(
        key,
        [],
    )

    best = None
    best_days = None

    for mrow in candidates:

        if (
            pd.isna(prow.match_date)
            or pd.isna(mrow.match_date)
        ):

            continue

        delta = abs(
            (
                prow.match_date
                -
                mrow.match_date
            ).days
        )

        if delta <= 1:

            if (
                best_days is None
                or delta < best_days
            ):

                best = mrow
                best_days = delta

    if best is not None:

        fallback_rows.append(
            {
                "pred_index":
                    prow.pred_index,

                "market_index":
                    best.market_index,

                "date_delta_days":
                    best_days,
            }
        )


fallback = pd.DataFrame(
    fallback_rows
)


if len(fallback):

    matched_pred.update(
        fallback["pred_index"]
    )

    matched_market.update(
        fallback["market_index"]
    )


print()
print("±1 day fallback matches:", len(fallback))

print()
print(
    "Total unique V5 rows matched:",
    len(matched_pred),
)

print(
    "Total unique market rows matched:",
    len(matched_market),
)


# ============================================================
# BUILD FINAL MATCH TABLE
# ============================================================

pair_rows = []


for row in exact.itertuples(
    index=False
):

    pair_rows.append(
        {
            "pred_index":
                row.pred_index,

            "market_index":
                row.market_index,

            "match_type":
                "exact",

            "date_delta_days":
                0,
        }
    )


if len(fallback):

    for row in fallback.itertuples(
        index=False
    ):

        pair_rows.append(
            {
                "pred_index":
                    row.pred_index,

                "market_index":
                    row.market_index,

                "match_type":
                    "date_fallback",

                "date_delta_days":
                    row.date_delta_days,
            }
        )


pairs = pd.DataFrame(
    pair_rows
)


final = pairs.merge(
    pred_match,
    on="pred_index",
    how="left",
)


market_keep = market_match[
    [
        "market_index",
        "source_year",
        "id",
        "match_date",
        "home_name",
        "away_name",
        "homeGoalCount",
        "awayGoalCount",
        "odds_btts_yes",
        "odds_btts_no",
        "valid_btts_market",
        "market_overround",
        "market_btts_yes_novig",
        "market_btts_no_novig",
        "btts",
    ]
].copy()


market_keep = market_keep.rename(
    columns={
        "match_date":
            "market_match_date",

        "home_name":
            "market_home",

        "away_name":
            "market_away",

        "btts":
            "actual_btts_market",
    }
)


final = final.merge(
    market_keep,
    on="market_index",
    how="left",
)


# ============================================================
# MODEL EDGE
# ============================================================

final["btts_yes_edge"] = (
    final["model_btts_yes"]
    -
    final["market_btts_yes_novig"]
)


final["btts_no_edge"] = (
    (1.0 - final["model_btts_yes"])
    -
    final["market_btts_no_novig"]
)


# ============================================================
# SAVE
# ============================================================

OUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


final.to_csv(
    OUT_FILE,
    index=False,
)


# ============================================================
# DIAGNOSTICS
# ============================================================

print()
print("=" * 100)
print("MATCH DIAGNOSTICS")
print("=" * 100)


print()
print(
    "MLS V5 rows:",
    len(pred_match),
)

print(
    "MLS market rows:",
    len(market_match),
)

print(
    "Matched rows:",
    len(final),
)

print(
    "Valid BTTS market rows:",
    int(
        final["valid_btts_market"]
        .fillna(False)
        .sum()
    ),
)


if len(pred_match):

    print(
        "V5 match rate:",
        f"{len(matched_pred) / len(pred_match):.2%}",
    )


if len(market_match):

    print(
        "Market match rate:",
        f"{len(matched_market) / len(market_match):.2%}",
    )


print()
print("Match types:")

if len(final):

    print(
        final["match_type"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )


# ============================================================
# MARKET QUALITY
# ============================================================

valid = final[
    final["valid_btts_market"]
    .fillna(False)
].copy()


print()
print("=" * 100)
print("BTTS MARKET QUALITY")
print("=" * 100)


if len(valid):

    print()
    print(
        "Average overround:",
        f"{valid['market_overround'].mean():.4f}",
    )

    print(
        "Median overround:",
        f"{valid['market_overround'].median():.4f}",
    )

    print(
        "Average BTTS YES odds:",
        f"{valid['odds_btts_yes'].mean():.3f}",
    )

    print(
        "Average BTTS NO odds:",
        f"{valid['odds_btts_no'].mean():.3f}",
    )

    print(
        "Average no-vig YES probability:",
        f"{valid['market_btts_yes_novig'].mean():.3%}",
    )


# ============================================================
# UNMATCHED V5 SAMPLE
# ============================================================

remaining_pred = pred_match[
    ~pred_match["pred_index"].isin(
        matched_pred
    )
].copy()


print()
print("=" * 100)
print("UNMATCHED V5 SAMPLE")
print("=" * 100)

print()
print("Unmatched V5 rows:", len(remaining_pred))


if len(remaining_pred):

    print(
        remaining_pred[
            [
                date_col,
                home_col,
                away_col,
                "home_norm",
                "away_norm",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# UNMATCHED MARKET SAMPLE
# ============================================================

remaining_market = market_match[
    ~market_match["market_index"].isin(
        matched_market
    )
].copy()


print()
print("=" * 100)
print("UNMATCHED MARKET SAMPLE")
print("=" * 100)

print()
print(
    "Unmatched market rows:",
    len(remaining_market),
)


if len(remaining_market):

    print(
        remaining_market[
            [
                "source_year",
                "match_date",
                "home_name",
                "away_name",
                "home_norm",
                "away_norm",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


print()
print("=" * 100)
print("OUTPUT")
print("=" * 100)

print()
print(OUT_FILE)

print()
print("DONE")
