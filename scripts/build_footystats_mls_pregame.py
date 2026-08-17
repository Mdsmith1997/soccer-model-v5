from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

import scripts.build_footystats_multileague_pregame as v2


# ============================================================
# MLS-ONLY PATH OVERRIDES
# ============================================================

PROCESSED = (
    ROOT
    / "data"
    / "processed"
)

v2.INPUT_FILE = (
    PROCESSED
    / "footystats_mls_history.csv"
)

v2.OUTPUT_TEAM_ROWS = (
    PROCESSED
    / "footystats_mls_team_pregame_v2.csv"
)

v2.OUTPUT_MATCH_ROWS = (
    PROCESSED
    / "footystats_mls_pregame_v2.csv"
)

v2.OUTPUT_SUMMARY = (
    PROCESSED
    / "footystats_mls_pregame_v2_summary.csv"
)


def main():

    print()
    print("=" * 105)
    print("MLS — FROZEN PRE-GAME FEATURE STORE V2")
    print("=" * 105)

    print()
    print("Input:")
    print(v2.INPUT_FILE)

    print()
    print("Outputs:")
    print(v2.OUTPUT_TEAM_ROWS)
    print(v2.OUTPUT_MATCH_ROWS)
    print(v2.OUTPUT_SUMMARY)

    print()
    print(
        "Using existing frozen multileague V2 "
        "feature engine."
    )

    print(
        "No feature weights or recencies "
        "are being changed."
    )

    v2.main()


if __name__ == "__main__":
    main()
