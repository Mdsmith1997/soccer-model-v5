from pathlib import Path
import pandas as pd

from fetch_us_soccer_odds import api_get, TARGET_LEAGUES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "live" / "soccer_results.csv"

rows = []

print()
print("=" * 90)
print("FETCH LIVE SOCCER RESULTS")
print("=" * 90)

for league, info in TARGET_LEAGUES.items():

    sport_key = info["sport_key"]

    print()
    print(f"Checking {league}...")

    try:
        events = api_get(
            f"/sports/{sport_key}/scores/",
            params={
                "daysFrom": 3,
                "dateFormat": "iso",
            },
        )
    except Exception as exc:
        print(f"  FAILED: {exc}")
        continue

    print(f"  Events returned: {len(events)}")

    for event in events:

        scores = event.get("scores") or []

        score_map = {}

        for score in scores:
            try:
                score_map[score["name"]] = int(score["score"])
            except (KeyError, TypeError, ValueError):
                pass

        home = event.get("home_team")
        away = event.get("away_team")

        home_score = score_map.get(home)
        away_score = score_map.get(away)

        completed = bool(event.get("completed"))

        result = ""

        if (
            completed
            and home_score is not None
            and away_score is not None
        ):
            if home_score > away_score:
                result = "HOME"
            elif home_score < away_score:
                result = "AWAY"
            else:
                result = "DRAW"

        rows.append(
            {
                "event_id": event.get("id"),
                "league": league,
                "sport_key": sport_key,
                "commence_time": event.get("commence_time"),
                "home_team": home,
                "away_team": away,
                "completed": completed,
                "home_score": home_score,
                "away_score": away_score,
                "result": result,
            }
        )

df = pd.DataFrame(rows)

# ------------------------------------------------------------
# Preserve previously fetched results/history.
#
# The Odds API scores endpoint only returns a limited lookback.
# Without this merge, older completed matches disappear from
# soccer_results.csv and any still-open ledger row can no longer
# be settled automatically.
# ------------------------------------------------------------

if OUT.exists():
    previous = pd.read_csv(
        OUT,
        low_memory=False,
    )

    if not previous.empty:
        df = pd.concat(
            [
                previous,
                df,
            ],
            ignore_index=True,
            sort=False,
        )

        if "event_id" in df.columns:
            df = (
                df
                .drop_duplicates(
                    subset=["event_id"],
                    keep="last",
                )
                .reset_index(drop=True)
            )

df.to_csv(
    OUT,
    index=False,
)

completed = df[
    (df["completed"] == True)
    & df["home_score"].notna()
    & df["away_score"].notna()
].copy()

print()
print("=" * 90)
print("COMPLETED MATCHES")
print("=" * 90)
print()

if completed.empty:
    print("No completed matches returned.")
else:
    print(
        completed[
            [
                "league",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "result",
            ]
        ].to_string(index=False)
    )

print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"Total events: {len(df)}")
print(f"Completed: {len(completed)}")
print(f"Saved: {OUT}")
