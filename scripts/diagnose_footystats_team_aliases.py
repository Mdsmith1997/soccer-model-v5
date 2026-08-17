from pathlib import Path
import re
import unicodedata

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

HISTORY_FILE = (
    ROOT
    / "data"
    / "processed"
    / "footystats_multileague_history.csv"
)

AUDIT_FILE = (
    ROOT
    / "data"
    / "live"
    / "v5_live_predictions_footystats_audit.csv"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_team(value):

    if pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = text.lower()

    text = text.replace(
        "&",
        " and ",
    )

    text = re.sub(
        r"\b(fc|afc|sv|vfl|sk|kv|raal)\b",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


# ============================================================
# TOKEN SIMILARITY
# ============================================================

def token_score(a, b):

    a_tokens = set(
        normalize_team(a).split()
    )

    b_tokens = set(
        normalize_team(b).split()
    )

    if (
        not a_tokens
        or
        not b_tokens
    ):
        return 0.0

    intersection = len(
        a_tokens
        &
        b_tokens
    )

    union = len(
        a_tokens
        |
        b_tokens
    )

    return (
        intersection
        /
        union
    )


# ============================================================
# CHARACTER SIMILARITY
# ============================================================

def char_score(a, b):

    from difflib import SequenceMatcher

    return SequenceMatcher(
        None,
        normalize_team(a),
        normalize_team(b),
    ).ratio()


# ============================================================
# COMBINED SCORE
# ============================================================

def combined_score(a, b):

    token = token_score(
        a,
        b,
    )

    char = char_score(
        a,
        b,
    )

    return (
        0.65
        *
        char
        +
        0.35
        *
        token
    )


# ============================================================
# MAIN
# ============================================================

def main():

    history = pd.read_csv(
        HISTORY_FILE,
        low_memory=False,
    )

    audit = pd.read_csv(
        AUDIT_FILE,
        low_memory=False,
    )

    blocked = audit[
        audit[
            "status"
        ]
        ==
        "BLOCKED"
    ].copy()

    print()
    print(
        "=" * 110
    )

    print(
        "FOOTYSTATS BLOCKED TEAM "
        "ALIAS DIAGNOSTIC"
    )

    print(
        "=" * 110
    )

    print()
    print(
        "Blocked fixtures:",
        len(blocked),
    )

    for _, row in blocked.iterrows():

        league = row[
            "league"
        ]

        print()
        print(
            "=" * 110
        )

        print(
            f"{league}: "
            f"{row['home_team']} "
            f"vs "
            f"{row['away_team']}"
        )

        print(
            "=" * 110
        )

        league_history = history[
            history[
                "league"
            ]
            ==
            league
        ]

        historical_teams = sorted(
            set(
                league_history[
                    "home_team"
                ]
                .dropna()
                .tolist()
            )
            |
            set(
                league_history[
                    "away_team"
                ]
                .dropna()
                .tolist()
            )
        )

        for side in [
            "home",
            "away",
        ]:

            issue_token = (
                "HOME_TEAM_UNRESOLVED"
                if side == "home"
                else
                "AWAY_TEAM_UNRESOLVED"
            )

            if (
                issue_token
                not in
                str(
                    row[
                        "issues"
                    ]
                )
            ):
                continue

            target = row[
                f"{side}_team"
            ]

            ranked = []

            for candidate in historical_teams:

                ranked.append(
                    (
                        combined_score(
                            target,
                            candidate,
                        ),
                        candidate,
                    )
                )

            ranked = sorted(
                ranked,
                reverse=True,
            )

            print()
            print(
                f"UNRESOLVED {side.upper()}: "
                f"{target}"
            )

            print(
                "Normalized:",
                normalize_team(
                    target
                ),
            )

            print()
            print(
                "Top historical candidates:"
            )

            for score, candidate in ranked[
                :12
            ]:

                print(
                    f"  {score:6.3f}  "
                    f"{candidate}"
                )

    print()
    print(
        "=" * 110
    )

    print(
        "DIAGNOSTIC COMPLETE"
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":
    main()