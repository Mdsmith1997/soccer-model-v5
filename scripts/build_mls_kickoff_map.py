from pathlib import Path
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

PRED_PATH = (
    ROOT / "data" / "processed"
    / "footystats_mls_v5_predictions.csv"
)

OUT_PATH = (
    ROOT / "data" / "processed"
    / "mls_v5_with_kickoff.csv"
)

FD_URL = (
    "https://www.football-data.co.uk/new/USA.csv"
)


ALIASES = {
    "sj earthquakes": "san jose earthquakes",
    "san jose earthquakes": "san jose earthquakes",

    "la galaxy": "los angeles galaxy",
    "los angeles galaxy": "los angeles galaxy",

    "lafc": "los angeles fc",
    "los angeles fc": "los angeles fc",

    "new york rb": "new york red bulls",
    "newyork rb": "new york red bulls",
    "ny red bulls": "new york red bulls",
    "new york red bulls": "new york red bulls",

    "new york city": "new york city",
    "newyork city": "new york city",
    "new york city fc": "new york city",

    "dc united": "dc united",
    "d c united": "dc united",

    "inter miami": "inter miami",
    "inter miami cf": "inter miami",

    "columbus crew": "columbus crew",
    "columbus crew sc": "columbus crew",

    "seattle sounders": "seattle sounders",
    "seattle sounders fc": "seattle sounders",

    "vancouver whitecaps": "vancouver whitecaps",
    "vancouver whitecaps fc": "vancouver whitecaps",

    "sporting kc": "sporting kansas city",
    "sporting kansas city": "sporting kansas city",

    "st louis city": "st louis city",
    "st louiscity": "st louis city",
    "st louis city sc": "st louis city",

    "austin": "austin",
    "austin fc": "austin",

    "toronto": "toronto",
    "toronto fc": "toronto",

    "cf montreal": "montreal",
    "montreal impact": "montreal",
    "montreal": "montreal",

    "atlanta united fc": "atlanta united",
    "atlanta united": "atlanta united",

    "fc cincinnati": "cincinnati",
    "cincinnati": "cincinnati",

    "orlando city": "orlando city",
    "orlando city sc": "orlando city",

    "nashville sc": "nashville",
    "nashville": "nashville",

    "charlotte fc": "charlotte",
    "charlotte": "charlotte",

    "minnesota united fc": "minnesota united",
    "minnesota united": "minnesota united",

    "philadelphia union": "philadelphia union",

    "chicago fire fc": "chicago fire",
    "chicago fire": "chicago fire",

    "houston dynamo fc": "houston dynamo",
    "houston dynamo": "houston dynamo",

    "portland timbers": "portland timbers",

    "real salt lake": "real salt lake",

    "colorado rapids": "colorado rapids",

    "fc dallas": "fc dallas",

    "new england revolution":
        "new england revolution",
}


def normalize_team(value):

    s = str(value)

    s = unicodedata.normalize(
        "NFKD",
        s,
    )

    s = "".join(
        c for c in s
        if not unicodedata.combining(c)
    )

    s = s.lower()

    # Fix joined words seen in FootyStats.
    replacements = {
        "newyork": "new york",
        "st.": "st ",
        "stlouis": "st louis",
        "chicagofire": "chicago fire",
        "columbuscrew": "columbus crew",
        "losangeles": "los angeles",
    }

    for a, b in replacements.items():
        s = s.replace(a, b)

    s = s.replace("&", " and ")

    s = re.sub(
        r"[^a-z0-9]+",
        " ",
        s,
    )

    s = " ".join(
        s.split()
    )

    return ALIASES.get(
        s,
        s,
    )


def similarity(a, b):

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def main():

    print()
    print("=" * 110)
    print("MLS V5 — IMPROVED KICKOFF MAP")
    print("=" * 110)

    # ========================================================
    # V5
    # ========================================================

    pred = pd.read_csv(
        PRED_PATH,
        low_memory=False,
    )

    pred["model_date"] = pd.to_datetime(
        pred["date"],
        errors="coerce",
        utc=True,
    )

    pred = pred.dropna(
        subset=[
            "model_date",
            "home_team",
            "away_team",
        ]
    ).copy()

    pred["year"] = (
        pred["model_date"].dt.year
    )

    pred = pred[
        pred["year"].between(
            2020,
            2025,
        )
    ].copy()

    pred["home_key"] = (
        pred["home_team"]
        .map(normalize_team)
    )

    pred["away_key"] = (
        pred["away_team"]
        .map(normalize_team)
    )

    # ========================================================
    # FOOTBALL-DATA
    # ========================================================

    r = requests.get(
        FD_URL,
        timeout=60,
    )

    print("Football-Data HTTP:", r.status_code)

    r.raise_for_status()

    raw_path = (
        ROOT / "data" / "raw"
        / "football_data_usa.csv"
    )

    raw_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path.write_bytes(
        r.content
    )

    fd = pd.read_csv(
        raw_path,
        low_memory=False,
        encoding="utf-8-sig",
    )

    fd = fd[
        fd["League"]
        .astype(str)
        .str.upper()
        .eq("MLS")
    ].copy()

    fd["fd_date"] = pd.to_datetime(
        fd["Date"],
        format="%d/%m/%Y",
        errors="coerce",
    )

    fd = fd[
        fd["fd_date"]
        .dt.year
        .between(
            2020,
            2025,
        )
    ].copy()

    fd["home_key"] = (
        fd["Home"]
        .map(normalize_team)
    )

    fd["away_key"] = (
        fd["Away"]
        .map(normalize_team)
    )

    # ========================================================
    # MATCH EACH V5 GAME
    # ========================================================

    results = []

    for i, row in pred.iterrows():

        target_date = (
            row["model_date"]
            .tz_convert(None)
            .normalize()
        )

        # MLS/UTC date handling:
        # permit same day +/- 1 day.
        candidates = fd[
            (
                fd["fd_date"]
                >= target_date
                - pd.Timedelta(days=1)
            )
            &
            (
                fd["fd_date"]
                <= target_date
                + pd.Timedelta(days=1)
            )
        ].copy()

        exact = candidates[
            (
                candidates["home_key"]
                == row["home_key"]
            )
            &
            (
                candidates["away_key"]
                == row["away_key"]
            )
        ]

        match = None
        method = None
        score = None

        if len(exact) == 1:

            match = exact.iloc[0]
            method = "exact"
            score = 1.0

        elif len(exact) > 1:

            # Pick nearest date.
            exact = exact.copy()

            exact["date_diff"] = (
                exact["fd_date"]
                - target_date
            ).abs()

            match = (
                exact
                .sort_values("date_diff")
                .iloc[0]
            )

            method = "exact_multi"
            score = 1.0

        else:

            # Fuzzy fallback.
            best = None

            for _, cand in candidates.iterrows():

                hs = similarity(
                    row["home_key"],
                    cand["home_key"],
                )

                aws = similarity(
                    row["away_key"],
                    cand["away_key"],
                )

                # Both teams must be similar.
                combined = (
                    hs + aws
                ) / 2

                if (
                    hs >= 0.72
                    and aws >= 0.72
                ):

                    if (
                        best is None
                        or combined
                        > best[0]
                    ):
                        best = (
                            combined,
                            cand,
                        )

            if best is not None:

                score = best[0]
                match = best[1]
                method = "fuzzy"

        result = row.to_dict()

        if match is None:

            result.update(
                {
                    "kickoff_matched":
                        False,
                    "match_method":
                        None,
                    "match_score":
                        None,
                    "fd_date":
                        None,
                    "fd_time":
                        None,
                    "fd_home":
                        None,
                    "fd_away":
                        None,
                }
            )

        else:

            result.update(
                {
                    "kickoff_matched":
                        True,
                    "match_method":
                        method,
                    "match_score":
                        score,
                    "fd_date":
                        match["Date"],
                    "fd_time":
                        match["Time"],
                    "fd_home":
                        match["Home"],
                    "fd_away":
                        match["Away"],
                }
            )

        results.append(result)

    out = pd.DataFrame(results)

    # ========================================================
    # REPORT
    # ========================================================

    matched = (
        out["kickoff_matched"]
        .eq(True)
    )

    print()
    print("=" * 110)
    print("MATCH RESULTS")
    print("=" * 110)

    print("V5 rows:", len(out))
    print(
        "Matched:",
        int(matched.sum()),
    )

    print(
        "Match rate:",
        f"{matched.mean():.2%}",
    )

    print()
    print("MATCH METHOD")

    print(
        out["match_method"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print("BY YEAR")

    for year, g in out.groupby("year"):

        ok = (
            g["kickoff_matched"]
            .eq(True)
        )

        print(
            year,
            "| rows:",
            len(g),
            "| matched:",
            int(ok.sum()),
            "| rate:",
            f"{ok.mean():.2%}",
        )

    print()
    print("=" * 110)
    print("LOWEST FUZZY MATCHES")
    print("=" * 110)

    fuzzy = out[
        out["match_method"]
        .eq("fuzzy")
    ].copy()

    if len(fuzzy):

        fuzzy = fuzzy.sort_values(
            "match_score"
        )

        cols = [
            "model_date",
            "home_team",
            "away_team",
            "fd_date",
            "fd_time",
            "fd_home",
            "fd_away",
            "match_score",
        ]

        print(
            fuzzy[
                cols
            ]
            .head(40)
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 110)
    print("UNMATCHED")
    print("=" * 110)

    cols = [
        "model_date",
        "home_team",
        "away_team",
        "home_key",
        "away_key",
    ]

    print(
        out.loc[
            ~matched,
            cols,
        ]
        .head(60)
        .to_string(
            index=False
        )
    )

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUT_PATH,
        index=False,
    )

    print()
    print("Saved:")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
