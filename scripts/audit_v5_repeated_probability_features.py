from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PRED_FILE = (
    ROOT / "data" / "processed"
    / "footystats_multileague_v5_predictions.csv"
)

df = pd.read_csv(PRED_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)

# ------------------------------------------------------------
# TARGET GROUPS
# ------------------------------------------------------------

targets = [
    {
        "label": "Championship 2018 opening AWAY",
        "league": "Championship",
        "date": "2018-08-04",
        "side": "A",
        "prob": 0.6057,
    },
    {
        "label": "League Two 2018 repeated HOME",
        "league": "League Two",
        "date": "2018-08-11",
        "side": "H",
        "prob": 0.5679,
    },
    {
        "label": "Eredivisie 2018 opening AWAY",
        "league": "Eredivisie",
        "date": "2018-08-11",
        "side": "A",
        "prob": 0.5852,
    },
    {
        "label": "2 Bundesliga 2018 extreme AWAY",
        "league": "2. Bundesliga",
        "date": "2018-08-04",
        "side": "A",
        "prob": 0.9270,
    },
    {
        "label": "La Liga 2018 opening AWAY",
        "league": "La Liga",
        "date": "2018-08-18",
        "side": "A",
        "prob": 0.7276,
    },
    {
        "label": "Serie A 2020 opening HOME",
        "league": "Serie A",
        "date": "2020-09-20",
        "side": "H",
        "prob": 0.8241,
    },
]


# ------------------------------------------------------------
# FEATURES WE CARE ABOUT
# ------------------------------------------------------------

meta_cols = [
    "date",
    "season",
    "league",
    "home_team",
    "away_team",
    "prior_games",
    "season_role",
    "history_class",

    "home_history_source",
    "home_previous_league",
    "home_league_changed",

    "away_history_source",
    "away_previous_league",
    "away_league_changed",

    "home_lambda",
    "away_lambda",
    "p_home",
    "p_draw",
    "p_away",
]


feature_cols = [
    # league priors
    "lg_home_goals",
    "lg_away_goals",
    "lg_home_xg",
    "lg_away_xg",
    "lg_home_shots",
    "lg_away_shots",

    # home goal strengths
    "home_final_goal_attack_overall",
    "home_final_goal_defense_overall",
    "home_final_goal_attack_venue",
    "home_final_goal_defense_venue",

    # home xG strengths
    "home_final_xg_attack_overall",
    "home_final_xg_defense_overall",
    "home_final_xg_attack_venue",
    "home_final_xg_defense_venue",

    # home shot strengths
    "home_final_shot_attack_overall",
    "home_final_shot_defense_overall",
    "home_final_shot_attack_venue",
    "home_final_shot_defense_venue",

    # home sample sizes
    "home_adj_goal_attack_overall_games",
    "home_adj_xg_attack_overall_games",
    "home_adj_shot_attack_overall_games",
    "home_adj_goal_attack_venue_games",
    "home_adj_xg_attack_venue_games",
    "home_adj_shot_attack_venue_games",
    "home_global_xg_attack_overall_games",

    # away goal strengths
    "away_final_goal_attack_overall",
    "away_final_goal_defense_overall",
    "away_final_goal_attack_venue",
    "away_final_goal_defense_venue",

    # away xG strengths
    "away_final_xg_attack_overall",
    "away_final_xg_defense_overall",
    "away_final_xg_attack_venue",
    "away_final_xg_defense_venue",

    # away shot strengths
    "away_final_shot_attack_overall",
    "away_final_shot_defense_overall",
    "away_final_shot_attack_venue",
    "away_final_shot_defense_venue",

    # away sample sizes
    "away_adj_goal_attack_overall_games",
    "away_adj_xg_attack_overall_games",
    "away_adj_shot_attack_overall_games",
    "away_adj_goal_attack_venue_games",
    "away_adj_xg_attack_venue_games",
    "away_adj_shot_attack_venue_games",
    "away_global_xg_attack_overall_games",
]


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def nunique_with_nan(s):
    return s.astype(str).nunique(dropna=False)


def format_value(x):
    if pd.isna(x):
        return "NaN"

    if isinstance(x, (float, np.floating)):
        return f"{x:.6f}"

    return str(x)


# ------------------------------------------------------------
# AUDIT
# ------------------------------------------------------------

print("=" * 145)
print("V5 REPEATED-PROBABILITY FEATURE AUDIT")
print("=" * 145)


for target in targets:

    league = target["league"]
    date = pd.Timestamp(target["date"])
    side = target["side"]
    expected = target["prob"]

    prob_col = (
        "p_home"
        if side == "H"
        else "p_away"
    )

    day = df[
        df["league"].eq(league)
        & df["date"].eq(date)
    ].copy()

    group = day[
        np.isclose(
            day[prob_col],
            expected,
            atol=0.0001,
        )
    ].copy()

    print("\n\n" + "=" * 145)
    print(target["label"])
    print("=" * 145)

    print("League:", league)
    print("Date:", date.date())
    print("Target side:", side)
    print("Probability column:", prob_col)
    print("Expected repeated probability:", expected)
    print("Fixtures on date:", len(day))
    print("Matching repeated rows:", len(group))

    if group.empty:
        print("\nNO MATCHING ROWS FOUND")
        continue

    # --------------------------------------------------------
    # FIXTURE / HISTORY SUMMARY
    # --------------------------------------------------------

    print("\n" + "-" * 145)
    print("FIXTURE + HISTORY SUMMARY")
    print("-" * 145)

    cols = [
        c for c in meta_cols
        if c in group.columns
    ]

    print(
        group[cols]
        .sort_values(
            ["home_team", "away_team"]
        )
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # WHICH FEATURES ARE IDENTICAL?
    # --------------------------------------------------------

    print("\n" + "-" * 145)
    print("FEATURE UNIQUENESS INSIDE REPEATED-PROBABILITY GROUP")
    print("-" * 145)

    rows = []

    for c in feature_cols:

        if c not in group.columns:
            continue

        n = nunique_with_nan(group[c])

        vals = (
            group[c]
            .drop_duplicates()
            .tolist()
        )

        rows.append(
            {
                "feature": c,
                "unique_values": n,
                "identical_all_rows": n == 1,
                "values": ", ".join(
                    format_value(v)
                    for v in vals[:8]
                ),
            }
        )

    u = pd.DataFrame(rows)

    print(
        u.sort_values(
            [
                "identical_all_rows",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
        ).to_string(index=False)
    )

    # --------------------------------------------------------
    # IDENTICAL FEATURE COUNT
    # --------------------------------------------------------

    identical = u[
        u["identical_all_rows"]
    ]

    varying = u[
        ~u["identical_all_rows"]
    ]

    print("\nIDENTICAL FEATURES:")
    print(
        f"{len(identical)} / {len(u)}"
    )

    print("\nVARYING FEATURES:")
    print(
        f"{len(varying)} / {len(u)}"
    )

    if len(varying):

        print("\nVARYING FEATURE NAMES:")

        for c in varying["feature"]:
            print("  ", c)

    # --------------------------------------------------------
    # LAMBDA UNIQUENESS
    # --------------------------------------------------------

    print("\n" + "-" * 145)
    print("LAMBDA / PROBABILITY UNIQUENESS")
    print("-" * 145)

    for c in [
        "home_lambda",
        "away_lambda",
        "p_home",
        "p_draw",
        "p_away",
    ]:

        if c not in group.columns:
            continue

        print(
            f"{c:<20}",
            "unique=",
            nunique_with_nan(group[c]),
            "| values=",
            ", ".join(
                format_value(v)
                for v in
                group[c]
                .drop_duplicates()
                .tolist()[:10]
            ),
        )

    # --------------------------------------------------------
    # HISTORY SOURCE DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "-" * 145)
    print("HISTORY SOURCE / CLASS DISTRIBUTION")
    print("-" * 145)

    for c in [
        "season_role",
        "history_class",
        "home_history_source",
        "away_history_source",
        "home_previous_league",
        "away_previous_league",
        "home_league_changed",
        "away_league_changed",
        "prior_games",
    ]:

        if c not in group.columns:
            continue

        print(f"\n{c}")

        print(
            group[c]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    # --------------------------------------------------------
    # COMPARE AGAINST OTHER FIXTURES SAME DAY
    # --------------------------------------------------------

    other = day[
        ~day.index.isin(group.index)
    ].copy()

    print("\n" + "-" * 145)
    print("OTHER FIXTURES ON SAME DATE")
    print("-" * 145)

    print("Other fixtures:", len(other))

    if len(other):

        compare_cols = [
            "home_team",
            "away_team",
            "prior_games",
            "season_role",
            "history_class",
            "home_history_source",
            "away_history_source",
            "home_lambda",
            "away_lambda",
            "p_home",
            "p_draw",
            "p_away",
        ]

        compare_cols = [
            c for c in compare_cols
            if c in other.columns
        ]

        print(
            other[compare_cols]
            .to_string(index=False)
        )


# ============================================================
# GLOBAL TEST:
# Are identical probabilities associated with history classes?
# ============================================================

print("\n\n" + "=" * 145)
print("GLOBAL REPEATED-PROBABILITY AUDIT")
print("=" * 145)

long = []

for side, pcol in [
    ("H", "p_home"),
    ("A", "p_away"),
]:

    x = df[
        [
            "league",
            "date",
            "season",
            "prior_games",
            "season_role",
            "history_class",
            "home_history_source",
            "away_history_source",
            pcol,
        ]
    ].copy()

    x["side"] = side
    x["prob_round"] = (
        x[pcol].round(4)
    )

    counts = (
        x.groupby(
            [
                "league",
                "date",
                "side",
                "prob_round",
            ],
            dropna=False,
        )[pcol]
        .transform("size")
    )

    x["same_prob_count"] = counts

    long.append(x)

long = pd.concat(
    long,
    ignore_index=True,
)

long["repeated_prob"] = (
    long["same_prob_count"] >= 2
)

print("\nRepeated-probability rows:")
print(
    long["repeated_prob"]
    .value_counts()
    .to_string()
)

print("\nRepeated probability rate:")
print(
    f"{long['repeated_prob'].mean():.2%}"
)


for c in [
    "season_role",
    "history_class",
    "home_history_source",
    "away_history_source",
]:

    print("\n" + "-" * 100)
    print(c)
    print("-" * 100)

    tab = pd.crosstab(
        long[c].fillna("NaN"),
        long["repeated_prob"],
    )

    tab.columns = [
        "not_repeated",
        "repeated",
    ]

    tab["total"] = (
        tab["not_repeated"]
        + tab["repeated"]
    )

    tab["repeat_rate"] = (
        tab["repeated"]
        / tab["total"]
    )

    print(
        tab.sort_values(
            "repeat_rate",
            ascending=False,
        ).to_string(
            formatters={
                "repeat_rate":
                    lambda x: f"{x:.2%}"
            }
        )
    )


print("\n" + "=" * 145)
print("AUDIT COMPLETE")
print("=" * 145)

print("""
WHAT TO LOOK FOR

A) If the repeated groups have identical final strength
   features and zero/very-low game counts:
      -> cold-start fallback confirmed.

B) If final strengths vary but lambdas are identical:
      -> inspect the lambda construction.

C) If lambdas vary but probabilities are identical:
      -> inspect the Poisson/probability conversion.

D) If repeated rows cluster in one history_class or
   history_source:
      -> that specific fallback path is the likely cause.

E) The Serie A 2020 group is included deliberately.
   If it shows the same mechanism, this is NOT merely a
   2018-data issue. It is a general V5 cold-start behavior.
""")
