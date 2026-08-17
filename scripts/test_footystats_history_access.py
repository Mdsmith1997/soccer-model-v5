import requests
import pandas as pd


BASE = "https://api.football-data-api.com"
KEY = "example"


# ============================================================
# HELPERS
# ============================================================

def get_json(endpoint, params):

    url = f"{BASE}/{endpoint}"

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    print(
        f"GET {endpoint} "
        f"status={response.status_code}"
    )

    if response.status_code != 200:

        print(
            response.text[:1000]
        )

        return None

    try:

        return response.json()

    except ValueError:

        print(
            "Invalid JSON response"
        )

        return None


# ============================================================
# LEAGUE LIST
# ============================================================

print()
print("=" * 90)
print("FOOTYSTATS HISTORICAL ACCESS TEST")
print("=" * 90)

payload = get_json(
    "league-list",
    {
        "key": KEY,
    },
)

if not payload:

    raise SystemExit


leagues = payload.get(
    "data",
    [],
)

print(
    "League entries returned:",
    len(leagues),
)


# ============================================================
# FIND PREMIER LEAGUE
# ============================================================

pl_candidates = []

for league in leagues:

    name_blob = " ".join(
        [
            str(
                league.get(
                    "name",
                    ""
                )
            ),
            str(
                league.get(
                    "league_name",
                    ""
                )
            ),
            str(
                league.get(
                    "country",
                    ""
                )
            ),
        ]
    ).lower()

    if (
        "premier league"
        in
        name_blob
        and
        "england"
        in
        name_blob
    ):

        pl_candidates.append(
            league
        )


print()
print("=" * 90)
print("PREMIER LEAGUE CANDIDATES")
print("=" * 90)

if not pl_candidates:

    print(
        "No English Premier League "
        "entry found."
    )

    raise SystemExit


for league in pl_candidates:

    print()
    print(
        "Name:",
        league.get(
            "name"
        ),
    )

    print(
        "League name:",
        league.get(
            "league_name"
        ),
    )

    print(
        "Country:",
        league.get(
            "country"
        ),
    )

    print(
        "Seasons:"
    )

    seasons = (
        league.get(
            "season",
            []
        )
        or
        league.get(
            "seasons",
            []
        )
    )

    for season in seasons:

        print(
            " ",
            season,
        )


# ============================================================
# BUILD SEASON TEST LIST
# ============================================================

season_rows = []

for league in pl_candidates:

    seasons = (
        league.get(
            "season",
            []
        )
        or
        league.get(
            "seasons",
            []
        )
    )

    for season in seasons:

        season_id = (
            season.get(
                "id"
            )
        )

        year = (
            season.get(
                "year"
            )
        )

        if season_id is None:

            continue

        season_rows.append(
            {
                "season_id":
                    season_id,

                "year":
                    year,
            }
        )


season_df = pd.DataFrame(
    season_rows
).drop_duplicates()


if len(season_df) == 0:

    print(
        "No season IDs found."
    )

    raise SystemExit


# Sort oldest -> newest where possible.
season_df[
    "year_numeric"
] = pd.to_numeric(
    season_df[
        "year"
    ],
    errors="coerce",
)

season_df = (
    season_df
    .sort_values(
        [
            "year_numeric",
            "season_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


print()
print("=" * 90)
print("SEASONS DISCOVERED")
print("=" * 90)

print(
    season_df[
        [
            "season_id",
            "year",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# TEST EACH SEASON
# ============================================================

results = []

print()
print("=" * 90)
print("TESTING LEAGUE-MATCHES ACCESS")
print("=" * 90)


for row in season_df.itertuples(
    index=False
):

    season_id = int(
        row.season_id
    )

    year = row.year

    print()
    print(
        "-" * 90
    )

    print(
        f"Season ID: {season_id} "
        f"Year: {year}"
    )

    payload = get_json(
        "league-matches",
        {
            "key":
                KEY,

            "season_id":
                season_id,

            "max_per_page":
                1000,
        },
    )

    if not payload:

        results.append(
            {
                "season_id":
                    season_id,

                "year":
                    year,

                "accessible":
                    False,

                "matches":
                    0,

                "xg_rows":
                    0,

                "first_home_xg":
                    None,

                "first_away_xg":
                    None,
            }
        )

        continue


    success = payload.get(
        "success",
        False,
    )

    data = payload.get(
        "data",
        [],
    )

    if not success:

        print(
            "success=False"
        )

        results.append(
            {
                "season_id":
                    season_id,

                "year":
                    year,

                "accessible":
                    False,

                "matches":
                    0,

                "xg_rows":
                    0,

                "first_home_xg":
                    None,

                "first_away_xg":
                    None,
            }
        )

        continue


    df = pd.DataFrame(
        data
    )


    if len(df) == 0:

        print(
            "Accessible but no matches."
        )

        results.append(
            {
                "season_id":
                    season_id,

                "year":
                    year,

                "accessible":
                    True,

                "matches":
                    0,

                "xg_rows":
                    0,

                "first_home_xg":
                    None,

                "first_away_xg":
                    None,
            }
        )

        continue


    has_home_xg = (
        "team_a_xg"
        in
        df.columns
    )

    has_away_xg = (
        "team_b_xg"
        in
        df.columns
    )


    if (
        has_home_xg
        and
        has_away_xg
    ):

        df[
            "team_a_xg"
        ] = pd.to_numeric(
            df[
                "team_a_xg"
            ],
            errors="coerce",
        )

        df[
            "team_b_xg"
        ] = pd.to_numeric(
            df[
                "team_b_xg"
            ],
            errors="coerce",
        )

        valid_xg = (
            df[
                [
                    "team_a_xg",
                    "team_b_xg",
                ]
            ]
            .notna()
            .all(
                axis=1
            )
        )

        # Ignore 0/0 rows if they appear to be
        # placeholders on incomplete fixtures.
        if "status" in df.columns:

            completed = (
                df[
                    "status"
                ]
                .astype(str)
                .str.lower()
                ==
                "complete"
            )

        else:

            completed = pd.Series(
                True,
                index=df.index,
            )


        meaningful_xg = (
            valid_xg
            &
            completed
            &
            (
                (
                    df[
                        "team_a_xg"
                    ]
                    >
                    0
                )
                |
                (
                    df[
                        "team_b_xg"
                    ]
                    >
                    0
                )
            )
        )

        xg_rows = int(
            meaningful_xg.sum()
        )


        if xg_rows:

            first = df[
                meaningful_xg
            ].iloc[
                0
            ]

            first_home_xg = (
                first[
                    "team_a_xg"
                ]
            )

            first_away_xg = (
                first[
                    "team_b_xg"
                ]
            )

        else:

            first_home_xg = None
            first_away_xg = None

    else:

        xg_rows = 0
        first_home_xg = None
        first_away_xg = None


    print(
        "Matches:",
        len(df),
    )

    print(
        "Completed rows with "
        "meaningful xG:",
        xg_rows,
    )

    if xg_rows:

        print(
            "Sample xG:",
            first_home_xg,
            first_away_xg,
        )


    results.append(
        {
            "season_id":
                season_id,

            "year":
                year,

            "accessible":
                True,

            "matches":
                len(df),

            "xg_rows":
                xg_rows,

            "first_home_xg":
                first_home_xg,

            "first_away_xg":
                first_away_xg,
        }
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

result_df = pd.DataFrame(
    results
)


print()
print("=" * 90)
print("HISTORICAL ACCESS SUMMARY")
print("=" * 90)

print(
    result_df
    .to_string(
        index=False
    )
)


accessible = result_df[
    result_df[
        "accessible"
    ]
    ==
    True
]

with_xg = result_df[
    result_df[
        "xg_rows"
    ]
    >
    0
]


print()
print("=" * 90)
print("RESULT")
print("=" * 90)

print(
    "Seasons discovered:",
    len(
        result_df
    ),
)

print(
    "Seasons accessible:",
    len(
        accessible
    ),
)

print(
    "Seasons with match xG:",
    len(
        with_xg
    ),
)

if len(
    with_xg
) >= 3:

    print()
    print(
        "GOOD: We have enough historical "
        "seasons to build an out-of-sample "
        "FootyStats → Understat compatibility "
        "mapping."
    )

elif len(
    with_xg
) > 0:

    print()
    print(
        "LIMITED: Some historical xG is "
        "accessible, but we should be careful "
        "about fitting and validating a provider "
        "mapping."
    )

else:

    print()
    print(
        "NO USEFUL HISTORICAL XG ACCESS "
        "WITH THE EXAMPLE KEY."
    )