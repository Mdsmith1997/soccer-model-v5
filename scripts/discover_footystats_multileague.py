from pathlib import Path
import os
import re

import pandas as pd
import requests


# ============================================================
# SETTINGS
# ============================================================

BASE = "https://api.football-data-api.com"

API_KEY = (
    os.getenv("FOOTYSTATS_API_KEY")
    or
    os.getenv("FOOTYSTATS_KEY")
    or
    "example"
)

TARGETS = {
    "Championship": {
        "country_terms": ["england"],
        "name_terms": [
            "championship",
            "efl championship",
        ],
    },

    "League One": {
        "country_terms": ["england"],
        "name_terms": [
            "league one",
            "league 1",
            "efl league one",
            "efl league 1",
        ],
    },

    "League Two": {
        "country_terms": ["england"],
        "name_terms": [
            "league two",
            "league 2",
            "efl league two",
            "efl league 2",
        ],
    },

    "2. Bundesliga": {
        "country_terms": ["germany"],
        "name_terms": [
            "2 bundesliga",
            "2. bundesliga",
            "bundesliga 2",
        ],
    },

    "Belgian Pro League": {
        "country_terms": ["belgium"],
        "name_terms": [
            "pro league",
            "first division a",
            "jupiler",
        ],
    },

    "Eredivisie": {
        "country_terms": ["netherlands"],
        "name_terms": [
            "eredivisie",
        ],
    },

    "MLS": {
        "country_terms": [
            "usa",
            "united states",
        ],
        "name_terms": [
            "mls",
            "major league soccer",
        ],
    },

    "Eliteserien": {
        "country_terms": ["norway"],
        "name_terms": [
            "eliteserien",
        ],
    },
}

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "footystats_multileague_discovery.csv"
)


# ============================================================
# HELPERS
# ============================================================

def normalize(value):

    value = str(
        value
        if value is not None
        else ""
    ).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return " ".join(
        value.split()
    )


def request_json(
    endpoint,
    params,
):

    response = requests.get(
        f"{BASE}/{endpoint}",
        params=params,
        timeout=60,
    )

    print(
        f"{endpoint}: "
        f"status={response.status_code}"
    )

    if response.status_code != 200:

        print(
            response.text[:500]
        )

        return None

    try:

        return response.json()

    except Exception:

        print(
            "Could not decode JSON."
        )

        return None


def get_seasons(
    league,
):

    seasons = (
        league.get("season")
        or
        league.get("seasons")
        or
        []
    )

    if isinstance(
        seasons,
        dict,
    ):

        seasons = [
            seasons
        ]

    return seasons


def league_blob(
    league,
):

    values = [
        league.get("name"),
        league.get("league_name"),
        league.get("country"),
        league.get("country_name"),
        league.get("competition_name"),
    ]

    return normalize(
        " ".join(
            str(x)
            for x in values
            if x is not None
        )
    )


def season_label(
    season,
):

    for key in [
        "year",
        "name",
        "season",
        "display_name",
    ]:

        value = season.get(
            key
        )

        if value is not None:

            return str(
                value
            )

    return ""


# ============================================================
# LEAGUE LIST
# ============================================================

print()
print("=" * 90)
print("FOOTYSTATS MULTI-LEAGUE DISCOVERY")
print("=" * 90)

print(
    "API key source:",
    (
        "environment"
        if API_KEY != "example"
        else "example"
    ),
)

payload = request_json(
    "league-list",
    {
        "key":
            API_KEY,
    },
)

if not payload:

    raise SystemExit


leagues = payload.get(
    "data",
    [],
)

print(
    "League-list entries:",
    len(leagues),
)


# ============================================================
# FIND TARGET LEAGUES
# ============================================================

matched_leagues = {}

for target, rules in (
    TARGETS.items()
):

    candidates = []

    for league in leagues:

        blob = league_blob(
            league
        )

        country_ok = any(
            normalize(term)
            in blob
            for term
            in rules[
                "country_terms"
            ]
        )

        name_ok = any(
            normalize(term)
            in blob
            for term
            in rules[
                "name_terms"
            ]
        )

        if (
            country_ok
            and
            name_ok
        ):

            candidates.append(
                league
            )

    matched_leagues[
        target
    ] = candidates


# ============================================================
# DISPLAY DISCOVERY
# ============================================================

print()
print("=" * 90)
print("TARGET LEAGUES FOUND")
print("=" * 90)

for target, candidates in (
    matched_leagues.items()
):

    print()
    print(
        target
    )

    print(
        "-" * 60
    )

    if not candidates:

        print(
            "NO MATCH"
        )

        continue

    for i, league in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"Candidate {i}:"
        )

        print(
            " name:",
            league.get("name"),
        )

        print(
            " league_name:",
            league.get(
                "league_name"
            ),
        )

        print(
            " country:",
            league.get(
                "country"
            ),
        )

        seasons = get_seasons(
            league
        )

        print(
            " seasons:",
            len(seasons),
        )

        for season in seasons:

            print(
                "   ",
                season.get("id"),
                "|",
                season_label(
                    season
                ),
            )


# ============================================================
# BUILD TEST LIST
# ============================================================

rows = []

for target, candidates in (
    matched_leagues.items()
):

    for candidate_index, league in (
        enumerate(
            candidates,
            start=1,
        )
    ):

        for season in get_seasons(
            league
        ):

            season_id = season.get(
                "id"
            )

            if season_id is None:

                continue

            rows.append(
                {
                    "target_league":
                        target,

                    "candidate_index":
                        candidate_index,

                    "league_name":
                        (
                            league.get(
                                "name"
                            )
                            or
                            league.get(
                                "league_name"
                            )
                        ),

                    "country":
                        league.get(
                            "country"
                        ),

                    "season_id":
                        season_id,

                    "season_label":
                        season_label(
                            season
                        ),
                }
            )


discovery = pd.DataFrame(
    rows
)

if len(discovery) == 0:

    print()
    print(
        "No target season IDs discovered."
    )

    raise SystemExit


# ============================================================
# PICK RECENT / RELEVANT SEASONS
# ============================================================

def season_sort_value(
    value,
):

    numbers = re.findall(
        r"\d{4}",
        str(value),
    )

    if numbers:

        return int(
            numbers[0]
        )

    return -1


discovery[
    "sort_year"
] = discovery[
    "season_label"
].map(
    season_sort_value
)


# Keep historical seasons likely relevant to V5.
testable = discovery[
    discovery[
        "sort_year"
    ].between(
        2018,
        2026,
    )
].copy()


# If labels were not parseable, keep everything.
if len(testable) == 0:

    testable = discovery.copy()


# ============================================================
# ACCESS TEST
# ============================================================

print()
print("=" * 90)
print("TESTING HISTORICAL MATCH ACCESS")
print("=" * 90)

results = []

for row in testable.itertuples(
    index=False
):

    print()
    print(
        row.target_league,
        "|",
        row.season_label,
        "| ID",
        row.season_id,
    )

    payload = request_json(
        "league-matches",
        {
            "key":
                API_KEY,

            "season_id":
                int(
                    row.season_id
                ),

            "max_per_page":
                1000,
        },
    )

    if not payload:

        results.append(
            {
                **row._asdict(),

                "accessible":
                    False,

                "matches":
                    0,

                "completed":
                    0,

                "xg_rows":
                    0,

                "shot_rows":
                    0,

                "goal_rows":
                    0,
            }
        )

        continue


    if not payload.get(
        "success",
        False,
    ):

        print(
            "success=False"
        )

        results.append(
            {
                **row._asdict(),

                "accessible":
                    False,

                "matches":
                    0,

                "completed":
                    0,

                "xg_rows":
                    0,

                "shot_rows":
                    0,

                "goal_rows":
                    0,
            }
        )

        continue


    data = payload.get(
        "data",
        [],
    )

    df = pd.DataFrame(
        data
    )

    if len(df) == 0:

        results.append(
            {
                **row._asdict(),

                "accessible":
                    True,

                "matches":
                    0,

                "completed":
                    0,

                "xg_rows":
                    0,

                "shot_rows":
                    0,

                "goal_rows":
                    0,
            }
        )

        continue


    if "status" in df.columns:

        completed_mask = (
            df[
                "status"
            ]
            .astype(str)
            .str.lower()
            ==
            "complete"
        )

    else:

        completed_mask = pd.Series(
            True,
            index=df.index,
        )


    completed = df[
        completed_mask
    ].copy()


    def valid_pair(
        left,
        right,
    ):

        if (
            left not in completed.columns
            or
            right not in completed.columns
        ):

            return 0

        a = pd.to_numeric(
            completed[
                left
            ],
            errors="coerce",
        )

        b = pd.to_numeric(
            completed[
                right
            ],
            errors="coerce",
        )

        return int(
            (
                a.notna()
                &
                b.notna()
            ).sum()
        )


    xg_rows = valid_pair(
        "team_a_xg",
        "team_b_xg",
    )

    shot_rows = valid_pair(
        "team_a_shots",
        "team_b_shots",
    )

    goal_rows = valid_pair(
        "homeGoalCount",
        "awayGoalCount",
    )


    print(
        "Matches:",
        len(df),
    )

    print(
        "Completed:",
        len(completed),
    )

    print(
        "Goals:",
        goal_rows,
        "| Shots:",
        shot_rows,
        "| xG:",
        xg_rows,
    )


    results.append(
        {
            **row._asdict(),

            "accessible":
                True,

            "matches":
                len(df),

            "completed":
                len(completed),

            "xg_rows":
                xg_rows,

            "shot_rows":
                shot_rows,

            "goal_rows":
                goal_rows,
        }
    )


# ============================================================
# SUMMARY
# ============================================================

result = pd.DataFrame(
    results
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT,
    index=False,
)

print()
print("=" * 120)
print("FOOTYSTATS ACCESS SUMMARY")
print("=" * 120)

display_cols = [
    "target_league",
    "season_label",
    "season_id",
    "accessible",
    "matches",
    "completed",
    "goal_rows",
    "shot_rows",
    "xg_rows",
]

print(
    result[
        display_cols
    ]
    .sort_values(
        [
            "target_league",
            "sort_year",
        ]
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 90)
print("LEAGUE COVERAGE")
print("=" * 90)

for league in TARGETS:

    sub = result[
        result[
            "target_league"
        ]
        ==
        league
    ]

    accessible = sub[
        (
            sub[
                "accessible"
            ]
            ==
            True
        )
        &
        (
            sub[
                "xg_rows"
            ]
            >
            0
        )
        &
        (
            sub[
                "shot_rows"
            ]
            >
            0
        )
    ]

    print(
        f"{league:<24}"
        f"{len(accessible):>3} "
        f"usable seasons"
    )


print()
print(
    "Saved:"
)

print(
    OUTPUT
)