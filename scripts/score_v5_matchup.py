from pathlib import Path
import sys
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_live_v5_predictions as live


def btts_prob(home_lambda, away_lambda):
    h = float(home_lambda)
    a = float(away_lambda)

    return (
        1
        - math.exp(-h)
        - math.exp(-a)
        + math.exp(-(h + a))
    )


if len(sys.argv) < 5:
    raise SystemExit(
        '\nUsage:\n'
        'python scripts/score_v5_matchup.py '
        '"League" "Home Team" "Away Team" YYYY-MM-DD\n'
    )


league = sys.argv[1]
home = sys.argv[2]
away = sys.argv[3]
date = sys.argv[4]

# Unique research-only ID.
match_id = (
    "RESEARCH_"
    + home.upper().replace(" ", "_")
    + "_VS_"
    + away.upper().replace(" ", "_")
)

fixtures = pd.DataFrame(
    [{
        "match_id": match_id,
        "date": pd.Timestamp(date),
        "season": 2627,
        "league": league,
        "home_team": home,
        "away_team": away,
    }]
)

print("=" * 100)
print("V5 ONE-OFF MATCHUP SCORER")
print("=" * 100)

print("\nFixture:")
print(
    fixtures[
        [
            "date",
            "league",
            "home_team",
            "away_team",
        ]
    ].to_string(index=False)
)

print("\nBuilding components using frozen V5...")

components = live.build_live_components(
    fixtures
)

print(
    "Component rows:",
    len(components),
)

if components.empty:
    raise RuntimeError(
        "V5 could not construct components for this matchup."
    )

print("\nBuilding prediction...")

output = live.build_predictions(
    fixtures,
    components,
)

if output.empty:
    raise RuntimeError(
        "V5 returned no prediction."
    )

row = output.iloc[0]

# Locate the actual lambda fields produced by V5.
home_col = (
    "home_lambda_v5"
    if "home_lambda_v5" in output.columns
    else "home_lambda"
)

away_col = (
    "away_lambda_v5"
    if "away_lambda_v5" in output.columns
    else "away_lambda"
)

h = float(row[home_col])
a = float(row[away_col])

p_btts = btts_prob(h, a)

print("\n" + "=" * 100)
print("V5 RESULT")
print("=" * 100)

print(f"\n{home} vs {away}")
print(f"Competition label : {league}")
print(f"Date              : {date}")

print(f"\nHome lambda       : {h:.4f}")
print(f"Away lambda       : {a:.4f}")
print(f"Total lambda      : {h + a:.4f}")

print(f"\nRAW BTTS YES      : {p_btts:.2%}")
print(f"RAW BTTS NO       : {1-p_btts:.2%}")

print(
    "Raw BTTS fair odds:",
    f"{1/p_btts:.2f}"
    if p_btts > 0
    else "N/A",
)

# Show useful V5 metadata if available.
meta = [
    "home_history_source",
    "away_history_source",
    "home_history_league",
    "away_history_league",
    "home_lambda_v5_raw",
    "away_lambda_v5_raw",
    "home_lambda_v5_transition",
    "away_lambda_v5_transition",
    "p_home_v5",
    "p_draw_v5",
    "p_away_v5",
]

available = [
    c for c in meta
    if c in output.columns
]

if available:

    print("\nV5 metadata:")

    for c in available:
        print(
            f"{c:<30}: "
            f"{row[c]}"
        )

print("\n" + "=" * 100)
print("IMPORTANT")
print("=" * 100)

print("""
This is a research-only V5 score.

It does NOT:
- modify upcoming_fixtures.csv
- modify the live prediction file
- create a live signal
- write a bet to a ledger
- change any frozen strategy

BTTS remains research-only until its market strategy
passes historical validation.
""")

# ============================================================
# NEUTRAL-SITE V5
# ============================================================

print("\n" + "=" * 100)
print("NEUTRAL-SITE V5")
print("=" * 100)

neutral_components = components.copy()

# Remove league-level home advantage while preserving
# the league's total scoring environment.
neutral_baseline = (
    neutral_components["lg_home_goals"]
    + neutral_components["lg_away_goals"]
) / 2

neutral_components["lg_home_goals"] = neutral_baseline
neutral_components["lg_away_goals"] = neutral_baseline

# overall_weight=1.0 removes team-specific venue influence.
neutral_home, neutral_away = live.ov.build_lambdas(
    neutral_components,
    1.0,
)

nh = float(neutral_home.iloc[0])
na = float(neutral_away.iloc[0])

neutral_btts = btts_prob(nh, na)

print(f"\nNeutral league baseline : {float(neutral_baseline.iloc[0]):.4f}")

print(f"\nArsenal lambda          : {nh:.4f}")
print(f"Manchester City lambda  : {na:.4f}")
print(f"Total lambda            : {nh + na:.4f}")

print(f"\nRAW BTTS YES            : {neutral_btts:.2%}")
print(f"RAW BTTS NO             : {1-neutral_btts:.2%}")

print(
    "Raw BTTS fair odds     :",
    f"{1/neutral_btts:.2f}"
    if neutral_btts > 0
    else "N/A",
)

print("\nComparison:")
print(f"Normal venue BTTS       : {p_btts:.2%}")
print(f"Neutral-site BTTS       : {neutral_btts:.2%}")
print(
    f"Neutral adjustment      : "
    f"{(neutral_btts-p_btts)*100:+.2f} pts"
)
