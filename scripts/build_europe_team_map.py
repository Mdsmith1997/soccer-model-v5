from pathlib import Path
from difflib import SequenceMatcher
import re
import unicodedata

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DOMESTIC_FILE = (
    ROOT
    / "data"
    / "processed"
    / "team_game_stats.csv"
)

EUROPE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "european_fixtures.csv"
)

OUTPUT_SUGGESTIONS = (
    ROOT
    / "data"
    / "processed"
    / "europe_team_mapping_suggestions.csv"
)

OUTPUT_MAP = (
    ROOT
    / "data"
    / "processed"
    / "europe_team_map.csv"
)


# ============================================================
# KNOWN ALIASES
#
# European/OpenFootball name -> domestic model name
# ============================================================

KNOWN_ALIASES = {

    # --------------------------------------------------------
    # ENGLAND
    # --------------------------------------------------------

    "Arsenal FC":
        "Arsenal",

    "Aston Villa FC":
        "Aston Villa",

    "Aston Villa":
        "Aston Villa",

    "Brighton & Hove Albion":
        "Brighton",

    "Chelsea FC":
        "Chelsea",

    "Everton FC":
        "Everton",

    "Liverpool FC":
        "Liverpool",

    "Manchester City FC":
        "Man City",

    "Manchester United FC":
        "Man United",

    "Newcastle United FC":
        "Newcastle",

    "Tottenham Hotspur FC":
        "Tottenham",

    "West Ham United":
        "West Ham",

    "West Ham United FC":
        "West Ham",

    "Leicester City FC":
        "Leicester",

    "Wolverhampton Wanderers FC":
        "Wolves",

    "Southampton FC":
        "Southampton",

    "Burnley FC":
        "Burnley",

    # --------------------------------------------------------
    # GERMANY
    # --------------------------------------------------------

    "FC Bayern München":
        "Bayern Munich",

    "Borussia Dortmund":
        "Dortmund",

    "Bayer 04 Leverkusen":
        "Leverkusen",

    "RB Leipzig":
        "RB Leipzig",

    "VfL Wolfsburg":
        "Wolfsburg",

    "Borussia Mönchengladbach":
        "M'gladbach",

    "Eintracht Frankfurt":
        "Ein Frankfurt",

    "SC Freiburg":
        "Freiburg",

    "TSG 1899 Hoffenheim":
        "Hoffenheim",

    "1. FC Union Berlin":
        "Union Berlin",

    "VfB Stuttgart":
        "Stuttgart",

    "FC Schalke 04":
        "Schalke 04",

    "1. FC Köln":
        "FC Koln",

    "Hertha BSC":
        "Hertha",

    # --------------------------------------------------------
    # NETHERLANDS
    # --------------------------------------------------------

    "AFC Ajax":
        "Ajax",

    "PSV":
        "PSV Eindhoven",

    "Feyenoord Rotterdam":
        "Feyenoord",

    "AZ Alkmaar":
        "AZ Alkmaar",

    "FC Twente":
        "Twente",

    "Vitesse":
        "Vitesse",

    # --------------------------------------------------------
    # BELGIUM
    # --------------------------------------------------------

    "Club Brugge KV":
        "Club Brugge",

    "Royal Antwerp FC":
        "Antwerp",

    "KAA Gent":
        "Gent",

    "RSC Anderlecht":
        "Anderlecht",

    "Standard Liège":
        "Standard",

    "KRC Genk":
        "Genk",

    "Union Saint-Gilloise":
    "St. Gilloise",


    # --------------------------------------------------------
    # ADDITIONAL VERIFIED ALIASES
    # --------------------------------------------------------

    # Belgium
    "Cercle Brugge":
        "Cercle Brugge",

    "Royale Union Saint-Gilloise":
        "St. Gilloise",

    "Union Saint-Gilloise":
        "St. Gilloise",

    # Germany
    "Bayern München":
        "Bayern Munich",

    "1899 Hoffenheim":
        "Hoffenheim",

    "Bayer Leverkusen":
        "Leverkusen",

    "1. FC Heidenheim 1846":
        "Heidenheim",

    "Bor. Mönchengladbach":
        "M'gladbach",

    # Netherlands
    "Feyenoord":
        "Feyenoord",

    "PSV Eindhoven":
        "PSV Eindhoven",

    # England
    "Leicester City":
        "Leicester",

    "Manchester United":
        "Man United",

    "Manchester City":
        "Man City",

    "Tottenham Hotspur":
        "Tottenham",

}




# ============================================================
# NORMALIZATION
# ============================================================

REMOVE_WORDS = {
    "fc",
    "afc",
    "cf",
    "sc",
    "sv",
    "club",
    "football",
    "calcio",
}


def strip_accents(text):

    normalized = unicodedata.normalize(
        "NFKD",
        str(text),
    )

    return "".join(
        char
        for char in normalized
        if not unicodedata.combining(char)
    )


def normalize_name(name):

    name = strip_accents(
        name
    ).lower()

    name = name.replace(
        "&",
        " and ",
    )

    name = re.sub(
        r"[^a-z0-9\s]",
        " ",
        name,
    )

    words = [
        word
        for word in name.split()
        if word not in REMOVE_WORDS
    ]

    return " ".join(
        words
    )


# ============================================================
# SIMILARITY
# ============================================================

def similarity(
    a,
    b,
):

    return SequenceMatcher(
        None,
        normalize_name(a),
        normalize_name(b),
    ).ratio()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("BUILDING EUROPE TEAM MAP")
    print("==============================")
    print()

    domestic = pd.read_csv(
        DOMESTIC_FILE,
        usecols=[
            "team",
            "league",
        ],
    )

    europe = pd.read_csv(
        EUROPE_FILE,
    )

    domestic_teams = (
        domestic[
            [
                "team",
                "league",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "league",
                "team",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    european_names = sorted(
        set(
            europe[
                "home_team"
            ].dropna()
        )
        |
        set(
            europe[
                "away_team"
            ].dropna()
        )
    )

    print(
        f"Domestic team names: "
        f"{domestic_teams['team'].nunique():,}"
    )

    print(
        f"Unique European team names: "
        f"{len(european_names):,}"
    )

    # ========================================================
    # FIND BEST CANDIDATES
    # ========================================================

    rows = []

    domestic_name_list = (
        domestic_teams[
            "team"
        ]
        .drop_duplicates()
        .tolist()
    )

    league_lookup = (
        domestic_teams
        .drop_duplicates(
            "team"
        )
        .set_index(
            "team"
        )[
            "league"
        ]
        .to_dict()
    )

    for europe_team in european_names:

        # ----------------------------------------------------
        # MANUAL ALIAS
        # ----------------------------------------------------

        if europe_team in KNOWN_ALIASES:

            mapped = (
                KNOWN_ALIASES[
                    europe_team
                ]
            )

            exists = (
                mapped
                in domestic_name_list
            )

            rows.append(
                {
                    "europe_team":
                        europe_team,

                    "candidate_team":
                        mapped,

                    "candidate_league":
                        league_lookup.get(
                            mapped
                        ),

                    "score":
                        1.0,

                    "match_method":
                        "manual_alias",

                    "candidate_exists":
                        exists,
                }
            )

            continue

        # ----------------------------------------------------
        # FUZZY CANDIDATES
        # ----------------------------------------------------

        scored = []

        for domestic_team in domestic_name_list:

            score = similarity(
                europe_team,
                domestic_team,
            )

            scored.append(
                (
                    score,
                    domestic_team,
                )
            )

        scored.sort(
            reverse=True
        )

        # Keep top 3.
        for rank, (
            score,
            candidate,
        ) in enumerate(
            scored[:3],
            start=1,
        ):

            rows.append(
                {
                    "europe_team":
                        europe_team,

                    "candidate_team":
                        candidate,

                    "candidate_league":
                        league_lookup.get(
                            candidate
                        ),

                    "score":
                        score,

                    "match_method":
                        f"fuzzy_{rank}",

                    "candidate_exists":
                        True,
                }
            )

    suggestions = pd.DataFrame(
        rows
    )

    suggestions.to_csv(
        OUTPUT_SUGGESTIONS,
        index=False,
    )

    # ========================================================
    # ACCEPT ONLY VERY SAFE MATCHES
    # ========================================================

    accepted = suggestions[
        (
            suggestions[
                "match_method"
            ]
            == "manual_alias"
        )
        &
        (
            suggestions[
                "candidate_exists"
            ]
            == True
        )
    ].copy()

    accepted = accepted[
        [
            "europe_team",
            "candidate_team",
            "candidate_league",
            "match_method",
        ]
    ]

    accepted = accepted.rename(
        columns={
            "candidate_team":
                "domestic_team",

            "candidate_league":
                "domestic_league",
        }
    )

    accepted.to_csv(
        OUTPUT_MAP,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("==============================")
    print("MANUAL ALIASES ACCEPTED")
    print("==============================")
    print()

    print(
        accepted
        .sort_values(
            [
                "domestic_league",
                "domestic_team",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Accepted mappings: "
        f"{len(accepted):,}"
    )

    # --------------------------------------------------------
    # UEFA FIXTURES INVOLVING ACCEPTED TEAMS
    # --------------------------------------------------------

    accepted_names = set(
        accepted[
            "europe_team"
        ]
    )

    relevant = europe[
        europe[
            "home_team"
        ].isin(
            accepted_names
        )
        |
        europe[
            "away_team"
        ].isin(
            accepted_names
        )
    ]

    print(
        f"European fixtures currently "
        f"matched to domestic clubs: "
        f"{len(relevant):,}"
    )

    # ========================================================
    # HIGH-SIMILARITY UNMAPPED
    # ========================================================

    manual_names = set(
        accepted[
            "europe_team"
        ]
    )

    top_fuzzy = suggestions[
        (
            suggestions[
                "match_method"
            ]
            == "fuzzy_1"
        )
        &
        (
            ~suggestions[
                "europe_team"
            ].isin(
                manual_names
            )
        )
    ].copy()

    top_fuzzy = (
        top_fuzzy
        .sort_values(
            "score",
            ascending=False,
        )
    )

    print()
    print("==============================")
    print("TOP UNMAPPED CANDIDATES")
    print("==============================")
    print()

    print(
        top_fuzzy[
            [
                "europe_team",
                "candidate_team",
                "candidate_league",
                "score",
            ]
        ]
        .head(80)
        .round(
            {
                "score": 3
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print("==============================")
    print("FILES SAVED")
    print("==============================")

    print()
    print(
        OUTPUT_MAP
    )

    print()
    print(
        OUTPUT_SUGGESTIONS
    )


if __name__ == "__main__":
    main()