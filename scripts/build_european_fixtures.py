from pathlib import Path
import re
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EUROPE_DIR = (
    ROOT
    / "data"
    / "raw"
    / "europe"
    / "openfootball"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "european_fixtures.csv"
)


# ============================================================
# SETTINGS
# ============================================================

START_SEASON = 2015
END_SEASON = 2025

COMPETITIONS = {
    "cl.txt": "Champions League",
    "el.txt": "Europa League",
    "conf.txt": "Conference League",
}

MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


# ============================================================
# TEAM NAME CLEANING
# ============================================================

COUNTRY_SUFFIX = re.compile(
    r"\s*\([A-Z]{3}\)\s*$"
)


def clean_team_name(name):
    """
    Remove OpenFootball country suffixes and normalize spacing.
    Example:
        Manchester City FC (ENG)
        -> Manchester City FC
    """

    name = COUNTRY_SUFFIX.sub(
        "",
        name.strip(),
    )

    name = re.sub(
        r"\s+",
        " ",
        name,
    )

    return name.strip()


# ============================================================
# DATE PARSING
# ============================================================

DATE_PATTERN = re.compile(
    r"^\s*"
    r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
    r"\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+"
    r"(\d{1,2})"
    r"(?:\s+(\d{4}))?"
    r"\s*$"
)


def parse_date_line(
    line,
    season_start,
    current_year,
):
    """
    Parse OpenFootball date lines.

    Examples:
        Tue Sep 19 2023
        Wed Sep 20
        Tue Feb 13
    """

    match = DATE_PATTERN.match(line)

    if not match:
        return None, current_year

    month_name = match.group(2)
    day = int(match.group(3))
    explicit_year = match.group(4)

    month = MONTHS[month_name]

    if explicit_year:
        year = int(explicit_year)

    else:
        # UEFA seasons cross calendar years.
        # Jul-Dec belong to season start year.
        # Jan-Jun belong to season end year.
        if month >= 7:
            year = season_start
        else:
            year = season_start + 1

    date = pd.Timestamp(
        year=year,
        month=month,
        day=day,
    )

    return date, year


# ============================================================
# MATCH PARSING
# ============================================================

SCORE_PATTERN = re.compile(
    r"\s+"
    r"(\d+)"
    r"\s*-\s*"
    r"(\d+)"
    r"(?:\s+\([^)]*\))?"
    r"\s*$"
)


def parse_match_line(line):
    """
    Parse a completed OpenFootball fixture.

    Handles spacing variations like:

        Manchester City FC (ENG) v Arsenal FC (ENG)
        Maccabi Tel Aviv(ISR) v PSV Eindhoven (NED)
        Paris Saint-Germain(FRA) v Bayern München (GER)

    We use the country-code structure to identify
    home and away teams safely.
    """

    score_match = SCORE_PATTERN.search(line)

    if not score_match:
        return None

    before_score = line[
        :score_match.start()
    ].strip()

    # Remove optional kickoff time.
    before_score = re.sub(
        r"^\d{1,2}:\d{2}\s+",
        "",
        before_score,
    )

    # -----------------------------------------------------
    # PRIMARY PARSER
    #
    # Home team + country code
    #          v
    # Away team + country code
    # -----------------------------------------------------

    match = re.match(
        r"^\s*"
        r"(.+?)"
        r"\s*\(([A-Z]{3})\)"
        r"\s*v\s*"
        r"(.+?)"
        r"\s*\(([A-Z]{3})\)"
        r"\s*$",
        before_score,
    )

    if match:

        home_team = clean_team_name(
            match.group(1)
        )

        away_team = clean_team_name(
            match.group(3)
        )

        if (
            home_team
            and away_team
        ):
            return (
                home_team,
                away_team,
            )

    # -----------------------------------------------------
    # FALLBACK
    #
    # Some historical rows may not contain both country
    # suffixes consistently. Use a stricter v separator.
    # -----------------------------------------------------

    split = re.split(
        r"\s{1,}v\s{1,}",
        before_score,
        maxsplit=1,
    )

    if len(split) != 2:
        return None

    home_team = clean_team_name(
        split[0]
    )

    away_team = clean_team_name(
        split[1]
    )

    if (
        not home_team
        or not away_team
    ):
        return None

    return (
        home_team,
        away_team,
    )


# ============================================================
# FILE PARSER
# ============================================================

def parse_competition_file(
    path,
    season_start,
    competition,
):

    rows = []

    current_date = None
    current_year = season_start

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for raw_line in file:

            line = raw_line.rstrip()

            parsed_date, current_year = (
                parse_date_line(
                    line,
                    season_start,
                    current_year,
                )
            )

            if parsed_date is not None:
                current_date = parsed_date
                continue

            if current_date is None:
                continue

            parsed_match = parse_match_line(
                line
            )

            if parsed_match is None:
                continue

            home_team, away_team = (
                parsed_match
            )

            rows.append(
                {
                    "date": current_date,
                    "season": (
                        f"{season_start}-"
                        f"{str(season_start + 1)[-2:]}"
                    ),
                    "competition": competition,
                    "home_team": home_team,
                    "away_team": away_team,
                    "source_file": str(
                        path.relative_to(
                            EUROPE_DIR
                        )
                    ),
                }
            )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("BUILDING EUROPEAN FIXTURES")
    print("==============================")
    print()

    rows = []

    files_found = 0

    for season_start in range(
        START_SEASON,
        END_SEASON + 1,
    ):

        season_folder = (
            EUROPE_DIR
            / (
                f"{season_start}-"
                f"{str(season_start + 1)[-2:]}"
            )
        )

        if not season_folder.exists():
            print(
                "Missing season folder:",
                season_folder.name,
            )
            continue

        for filename, competition in (
            COMPETITIONS.items()
        ):

            path = (
                season_folder
                / filename
            )

            if not path.exists():
                continue

            files_found += 1

            parsed = (
                parse_competition_file(
                    path,
                    season_start,
                    competition,
                )
            )

            rows.extend(parsed)

            print(
                f"{season_folder.name:<8} "
                f"{competition:<20} "
                f"{len(parsed):>4} matches"
            )

    if not rows:
        raise RuntimeError(
            "No European fixtures parsed."
        )

    df = pd.DataFrame(rows)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    # --------------------------------------------------------
    # CLEAN / VALIDATE
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset=[
                "date",
                "competition",
                "home_team",
                "away_team",
            ]
        )
        .sort_values(
            [
                "date",
                "competition",
                "home_team",
            ]
        )
        .reset_index(drop=True)
    )

    invalid = df[
        df["home_team"]
        ==
        df["away_team"]
    ]

    if len(invalid):
        raise RuntimeError(
            "Found fixtures where home == away."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("==============================")
    print("EUROPEAN FIXTURES COMPLETE")
    print("==============================")
    print()

    print(
        f"Files parsed: {files_found}"
    )

    print(
        f"Fixtures: {len(df):,}"
    )

    print(
        "Date range:",
        df["date"].min().date(),
        "->",
        df["date"].max().date(),
    )

    print()

    print("FIXTURES BY COMPETITION")

    print(
        df[
            "competition"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print("FIXTURES BY SEASON / COMPETITION")

    summary = (
        df
        .groupby(
            [
                "season",
                "competition",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(
        summary.to_string()
    )

    print()

    print("ENGLISH CLUB EXAMPLES")

    english_pattern = (
        "Manchester|Arsenal|Liverpool|"
        "Chelsea|Tottenham|West Ham|"
        "Brighton|Aston Villa|"
        "Newcastle|Manchester United"
    )

    examples = df[
        df["home_team"].str.contains(
            english_pattern,
            case=False,
            regex=True,
        )
        |
        df["away_team"].str.contains(
            english_pattern,
            case=False,
            regex=True,
        )
    ]

    print(
        examples[
            [
                "date",
                "competition",
                "home_team",
                "away_team",
            ]
        ]
        .tail(25)
        .to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()