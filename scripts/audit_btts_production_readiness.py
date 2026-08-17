from pathlib import Path
import ast
import json
import re

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "btts_production_readiness"
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


def relative(path):
    try:
        return str(
            path.relative_to(ROOT)
        )
    except Exception:
        return str(path)


def read_text_safe(path):
    try:
        return path.read_text(
            errors="ignore"
        )
    except Exception:
        return ""


def csv_schema(path):
    try:
        x = pd.read_csv(
            path,
            nrows=5,
            low_memory=False,
        )

        return {
            "rows_sampled": len(x),
            "columns": list(x.columns),
            "error": None,
        }

    except Exception as exc:

        return {
            "rows_sampled": 0,
            "columns": [],
            "error": str(exc),
        }


def parquet_schema(path):
    try:
        x = pd.read_parquet(
            path
        ).head(5)

        return {
            "rows_sampled": len(x),
            "columns": list(x.columns),
            "error": None,
        }

    except Exception as exc:

        return {
            "rows_sampled": 0,
            "columns": [],
            "error": str(exc),
        }


# ============================================================
# 1. PROJECT MAP
# ============================================================

banner(
    "BTTS PRODUCTION READINESS AUDIT"
)

print(
    "Project root:",
    ROOT,
)


important_dirs = [
    ROOT / "scripts",
    ROOT / "data" / "live",
    ROOT / "data" / "processed",
    ROOT / "models",
    ROOT / "src",
]


banner(
    "PROJECT DIRECTORY STATUS"
)

for p in important_dirs:

    print(
        f"{relative(p):40s}",
        "EXISTS"
        if p.exists()
        else "MISSING",
    )


# ============================================================
# 2. SCRIPT DISCOVERY
# ============================================================

banner(
    "BTTS / ODDS / LIVE SCRIPT DISCOVERY"
)


script_terms = [
    "btts",
    "odd",
    "market",
    "live",
    "fixture",
    "predict",
    "feature",
    "xg",
]


script_rows = []


scripts_dir = (
    ROOT
    / "scripts"
)


if scripts_dir.exists():

    for p in sorted(
        scripts_dir.glob(
            "*.py"
        )
    ):

        name = (
            p.name.lower()
        )

        text = (
            read_text_safe(
                p
            )
        )

        matched_terms = [
            term
            for term in script_terms
            if (
                term in name
                or
                term in text.lower()
            )
        ]


        if not matched_terms:
            continue


        function_names = []

        try:

            tree = ast.parse(
                text
            )

            function_names = [
                node.name
                for node in ast.walk(
                    tree
                )
                if isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                )
            ]

        except Exception:
            pass


        input_files = sorted(
            set(
                re.findall(
                    r'''["']([^"']+\.(?:csv|parquet|joblib|pkl|json))["']''',
                    text,
                    flags=re.I,
                )
            )
        )


        script_rows.append(
            {
                "script":
                    relative(p),

                "matched_terms":
                    ",".join(
                        matched_terms
                    ),

                "functions":
                    ",".join(
                        function_names[
                            :40
                        ]
                    ),

                "referenced_files":
                    " | ".join(
                        input_files[
                            :40
                        ]
                    ),

                "has_main":
                    (
                        "__main__"
                        in text
                    ),

                "uses_requests":
                    (
                        "requests."
                        in text
                        or
                        "httpx."
                        in text
                    ),

                "uses_odds_api":
                    (
                        "the-odds-api"
                        in text.lower()
                        or
                        "api.the-odds-api"
                        in text.lower()
                        or
                        "odds_api"
                        in text.lower()
                    ),

                "uses_champion_yes":
                    (
                        "champion_yes"
                        in text
                    ),

                "uses_market_yes":
                    (
                        "market_yes"
                        in text
                    ),
            }
        )


script_df = pd.DataFrame(
    script_rows
)


if len(
    script_df
):

    print(
        script_df[
            [
                "script",
                "matched_terms",
                "has_main",
                "uses_odds_api",
                "uses_champion_yes",
                "uses_market_yes",
            ]
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "No matching scripts found."
    )


# ============================================================
# 3. LIVE DATA SCHEMA
# ============================================================

banner(
    "LIVE DATA FILE SCHEMAS"
)


live_dir = (
    ROOT
    / "data"
    / "live"
)


live_rows = []


if live_dir.exists():

    for p in sorted(
        live_dir.glob(
            "*"
        )
    ):

        if not p.is_file():
            continue

        if p.suffix.lower() == ".csv":

            schema = csv_schema(
                p
            )

        elif p.suffix.lower() == ".parquet":

            schema = parquet_schema(
                p
            )

        else:
            continue


        cols = schema[
            "columns"
        ]


        live_rows.append(
            {
                "file":
                    relative(p),

                "columns":
                    " | ".join(
                        cols
                    ),

                "has_snapshot_time":
                    "snapshot_time"
                    in cols,

                "has_commence_time":
                    "commence_time"
                    in cols,

                "has_bookmaker":
                    (
                        "bookmaker"
                        in cols
                        or
                        "bookmaker_key"
                        in cols
                    ),

                "has_market":
                    "market"
                    in cols,

                "has_decimal_odds":
                    "decimal_odds"
                    in cols,

                "has_yes_no":
                    any(
                        (
                            "yes"
                            in c.lower()
                            or
                            "no"
                            in c.lower()
                        )
                        for c in cols
                    ),

                "error":
                    schema[
                        "error"
                    ],
            }
        )


live_df = pd.DataFrame(
    live_rows
)


if len(
    live_df
):

    print(
        live_df[
            [
                "file",
                "has_snapshot_time",
                "has_commence_time",
                "has_bookmaker",
                "has_market",
                "has_decimal_odds",
                "has_yes_no",
            ]
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "No live CSV/parquet files found."
    )


# ============================================================
# 4. INSPECT ODDS MARKETS HISTORY
# ============================================================

banner(
    "ODDS MARKETS HISTORY DETAILS"
)


history_file = (
    ROOT
    / "data"
    / "live"
    / "odds_markets_history.csv"
)


history_summary = {}


if history_file.exists():

    hist = pd.read_csv(
        history_file,
        low_memory=False,
    )


    print(
        "Rows:",
        len(hist),
    )

    print(
        "Columns:"
    )

    print(
        list(
            hist.columns
        )
    )


    for c in [
        "snapshot_time",
        "commence_time",
        "market_last_update",
    ]:

        if c in hist.columns:

            hist[c] = pd.to_datetime(
                hist[c],
                errors="coerce",
                utc=True,
            )


    if "market" in hist.columns:

        print()
        print(
            "Markets:"
        )

        print(
            hist[
                "market"
            ]
            .value_counts(
                dropna=False
            )
            .head(50)
            .to_string()
        )


    if "bookmaker" in hist.columns:

        print()
        print(
            "Bookmakers:"
        )

        print(
            hist[
                "bookmaker"
            ]
            .value_counts()
            .head(30)
            .to_string()
        )


    if "snapshot_time" in hist.columns:

        print()
        print(
            "Snapshot range:"
        )

        print(
            hist[
                "snapshot_time"
            ].min(),
            "->",
            hist[
                "snapshot_time"
            ].max(),
        )


    history_summary = {
        "rows":
            len(hist),

        "columns":
            list(
                hist.columns
            ),

        "markets":
            (
                hist[
                    "market"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
                if
                "market"
                in hist.columns
                else []
            ),

        "bookmakers":
            (
                hist[
                    "bookmaker"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
                if
                "bookmaker"
                in hist.columns
                else []
            ),
    }


    # --------------------------------------------------------
    # Search specifically for BTTS-like markets
    # --------------------------------------------------------

    possible_btts = pd.DataFrame()


    if "market" in hist.columns:

        mask = (
            hist[
                "market"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                r"btts|both.*teams|both_teams",
                regex=True,
                na=False,
            )
        )


        possible_btts = (
            hist[
                mask
            ]
            .copy()
        )


        print()
        print(
            "BTTS-like market rows:",
            len(
                possible_btts
            ),
        )


        if len(
            possible_btts
        ):

            print(
                possible_btts
                .head(30)
                .to_string(
                    index=False
                )
            )


else:

    print(
        "Missing:",
        history_file,
    )


# ============================================================
# 5. SNAPSHOT DATA DETAILS
# ============================================================

banner(
    "CURRENT ODDS SNAPSHOT DETAILS"
)


snapshot_file = (
    ROOT
    / "data"
    / "live"
    / "odds_markets_snapshot.csv"
)


if snapshot_file.exists():

    snap = pd.read_csv(
        snapshot_file,
        low_memory=False,
    )


    print(
        "Rows:",
        len(snap),
    )

    print(
        "Columns:",
        list(
            snap.columns
        ),
    )


    if "market" in snap.columns:

        print()
        print(
            "Current market types:"
        )

        print(
            snap[
                "market"
            ]
            .value_counts()
            .head(50)
            .to_string()
        )


else:

    print(
        "Missing:",
        snapshot_file,
    )


# ============================================================
# 6. BTTS MODEL FILE INVENTORY
# ============================================================

banner(
    "BTTS MODEL / FEATURE FILE INVENTORY"
)


processed_dir = (
    ROOT
    / "data"
    / "processed"
)


processed_terms = [
    "btts",
    "xg",
    "feature",
    "lambda",
]


processed_rows = []


if processed_dir.exists():

    for p in sorted(
        processed_dir.glob(
            "*"
        )
    ):

        if not p.is_file():
            continue

        if not any(
            term
            in p.name.lower()
            for term in processed_terms
        ):
            continue


        if p.suffix.lower() == ".csv":

            schema = csv_schema(
                p
            )

        elif p.suffix.lower() == ".parquet":

            schema = parquet_schema(
                p
            )

        else:
            continue


        cols = schema[
            "columns"
        ]


        processed_rows.append(
            {
                "file":
                    relative(p),

                "num_columns":
                    len(cols),

                "has_home_away":
                    (
                        "home_team"
                        in cols
                        and
                        "away_team"
                        in cols
                    ),

                "has_date":
                    "date"
                    in cols,

                "has_lambda":
                    any(
                        "lambda"
                        in c.lower()
                        for c in cols
                    ),

                "has_cfg_prob":
                    (
                        "champion_yes"
                        in cols
                    ),

                "has_target":
                    (
                        "btts_yes"
                        in cols
                    ),

                "columns":
                    " | ".join(
                        cols
                    ),
            }
        )


processed_df = pd.DataFrame(
    processed_rows
)


if len(
    processed_df
):

    print(
        processed_df[
            [
                "file",
                "num_columns",
                "has_home_away",
                "has_date",
                "has_lambda",
                "has_cfg_prob",
                "has_target",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# 7. MODEL ARTIFACT DISCOVERY
# ============================================================

banner(
    "SERIALIZED MODEL ARTIFACTS"
)


model_extensions = [
    ".joblib",
    ".pkl",
    ".pickle",
    ".json",
]


model_rows = []


for base in [
    ROOT / "models",
    ROOT / "data" / "processed",
]:

    if not base.exists():
        continue


    for p in base.rglob(
        "*"
    ):

        if not p.is_file():
            continue

        if p.suffix.lower() not in model_extensions:
            continue


        name = (
            p.name.lower()
        )


        if not any(
            term
            in name
            for term in [
                "btts",
                "cfg",
                "model",
                "calib",
            ]
        ):
            continue


        model_rows.append(
            {
                "file":
                    relative(p),

                "size_bytes":
                    p.stat().st_size,
            }
        )


model_df = pd.DataFrame(
    model_rows
)


if len(
    model_df
):

    print(
        model_df.to_string(
            index=False
        )
    )

else:

    print(
        "No obvious serialized BTTS model artifacts found."
    )


# ============================================================
# 8. FIND CURRENT PREDICTION ENTRY POINTS
# ============================================================

banner(
    "POSSIBLE PREDICTION ENTRY POINTS"
)


entry_rows = []


if len(
    script_df
):

    for _, row in script_df.iterrows():

        p = (
            ROOT
            / row[
                "script"
            ]
        )


        text = (
            read_text_safe(
                p
            )
        )


        score = 0


        signals = []


        for term, weight in [
            ("predict_proba", 3),
            ("predict(", 2),
            ("fixtures", 2),
            ("fixture", 2),
            ("odds_markets_snapshot", 3),
            ("champion_yes", 3),
            ("home_team", 1),
            ("away_team", 1),
            ("commence_time", 2),
        ]:

            if term in text:

                score += weight

                signals.append(
                    term
                )


        if score > 0:

            entry_rows.append(
                {
                    "script":
                        row[
                            "script"
                        ],

                    "score":
                        score,

                    "signals":
                        ", ".join(
                            signals
                        ),
                }
            )


entry_df = pd.DataFrame(
    entry_rows
)


if len(
    entry_df
):

    entry_df = (
        entry_df
        .sort_values(
            "score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    print(
        entry_df
        .head(30)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No clear prediction entry point found."
    )


# ============================================================
# 9. REQUIRED LIVE PIPELINE COMPONENTS
# ============================================================

banner(
    "LIVE BTTS COMPONENT CHECK"
)


def script_contains(
    terms,
):

    if not scripts_dir.exists():
        return False

    for p in scripts_dir.glob(
        "*.py"
    ):

        text = (
            read_text_safe(
                p
            ).lower()
        )

        if all(
            term.lower()
            in text
            for term in terms
        ):
            return True

    return False


checks = []


def add_check(
    name,
    status,
    evidence,
):

    checks.append(
        {
            "component":
                name,

            "status":
                (
                    "READY"
                    if status
                    else "MISSING"
                ),

            "evidence":
                evidence,
        }
    )


add_check(
    "Live odds snapshot",
    snapshot_file.exists(),
    relative(
        snapshot_file
    ),
)


add_check(
    "Historical odds snapshots",
    history_file.exists(),
    relative(
        history_file
    ),
)


add_check(
    "Bookmaker metadata",
    (
        len(
            live_df
        )
        and
        live_df[
            "has_bookmaker"
        ].any()
    ),
    "data/live",
)


add_check(
    "Snapshot timestamps",
    (
        len(
            live_df
        )
        and
        live_df[
            "has_snapshot_time"
        ].any()
    ),
    "data/live",
)


add_check(
    "Kickoff timestamps",
    (
        len(
            live_df
        )
        and
        live_df[
            "has_commence_time"
        ].any()
    ),
    "data/live",
)


add_check(
    "BTTS feature store",
    (
        ROOT
        / "data"
        / "processed"
        / "btts_feature_store_v1.csv"
    ).exists(),
    "btts_feature_store_v1.csv",
)


add_check(
    "CFG historical probabilities",
    (
        ROOT
        / "data"
        / "processed"
        / "btts_cfg0755_oos_2021_2025.csv"
    ).exists(),
    "btts_cfg0755_oos_2021_2025.csv",
)


add_check(
    "Dynamic market calibration implemented",
    script_contains(
        [
            "dynamic_probability",
            "champion_yes",
            "market_yes",
        ]
    ),
    "market calibration scripts",
)


add_check(
    "Fixture prediction script",
    (
        len(
            entry_df
        )
        and
        entry_df[
            "score"
        ].max()
        >=
        8
    ),
    (
        entry_df.iloc[0][
            "script"
        ]
        if len(
            entry_df
        )
        else ""
    ),
)


add_check(
    "BTTS market present in live odds history",
    (
        "possible_btts"
        in locals()
        and
        len(
            possible_btts
        )
        >
        0
    ),
    "odds_markets_history.csv",
)


checks_df = pd.DataFrame(
    checks
)


print(
    checks_df.to_string(
        index=False
    )
)


# ============================================================
# 10. READINESS SCORE
# ============================================================

ready_count = (
    checks_df[
        "status"
    ]
    ==
    "READY"
).sum()


total_count = len(
    checks_df
)


readiness_pct = (
    ready_count
    /
    total_count
    *
    100
)


banner(
    "READINESS SUMMARY"
)


print(
    f"Ready components: "
    f"{ready_count}/{total_count}"
)

print(
    f"Readiness: "
    f"{readiness_pct:.1f}%"
)


missing = checks_df[
    checks_df[
        "status"
    ]
    ==
    "MISSING"
]


if len(
    missing
):

    print()
    print(
        "MISSING COMPONENTS:"
    )

    print(
        missing[
            [
                "component",
                "evidence",
            ]
        ]
        .to_string(
            index=False
        )
    )

else:

    print()
    print(
        "All major components appear present."
    )


# ============================================================
# 11. RECOMMENDED NEXT BUILD
# ============================================================

banner(
    "RECOMMENDED NEXT BUILD"
)


if (
    snapshot_file.exists()
    and
    history_file.exists()
):

    print(
        "Build one unified live BTTS runner that:"
    )

    print(
        "  1. Reads upcoming fixtures."
    )

    print(
        "  2. Builds leakage-safe BTTS features."
    )

    print(
        "  3. Produces frozen CFG_0755 probability."
    )

    print(
        "  4. Reads current bookmaker BTTS YES/NO prices."
    )

    print(
        "  5. Converts market prices to no-vig probability."
    )

    print(
        "  6. Applies DYN_100_60_50_40 calibration."
    )

    print(
        "  7. Calculates final edge + EV."
    )

    print(
        "  8. Writes every prediction to a permanent log."
    )

    print(
        "  9. Continues saving market snapshots until kickoff."
    )

    print(
        " 10. Calculates CLV after close."
    )

else:

    print(
        "Live odds infrastructure must be completed first."
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

script_df.to_csv(
    OUT_DIR
    / "01_script_inventory.csv",
    index=False,
)


live_df.to_csv(
    OUT_DIR
    / "02_live_data_inventory.csv",
    index=False,
)


processed_df.to_csv(
    OUT_DIR
    / "03_processed_btts_inventory.csv",
    index=False,
)


model_df.to_csv(
    OUT_DIR
    / "04_model_artifacts.csv",
    index=False,
)


entry_df.to_csv(
    OUT_DIR
    / "05_prediction_entry_candidates.csv",
    index=False,
)


checks_df.to_csv(
    OUT_DIR
    / "06_component_check.csv",
    index=False,
)


summary = {
    "ready_components":
        int(
            ready_count
        ),

    "total_components":
        int(
            total_count
        ),

    "readiness_pct":
        readiness_pct,

    "missing_components":
        missing[
            "component"
        ].tolist(),

    "history_summary":
        history_summary,
}


with open(
    OUT_DIR
    / "07_readiness_summary.json",
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        default=str,
    )


banner(
    "OUTPUTS"
)


for p in sorted(
    OUT_DIR.glob(
        "*"
    )
):

    print(
        p
    )


print()
print("DONE")
