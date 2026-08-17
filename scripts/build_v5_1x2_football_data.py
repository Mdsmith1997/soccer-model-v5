from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "data" / "processed" / "v5_1x2_football_data.csv"

BASE = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

LEAGUES = {
    "E0":  "Premier League",
    "E1":  "Championship",
    "E2":  "League One",
    "E3":  "League Two",
    "D1":  "Bundesliga",
    "D2":  "2. Bundesliga",
    "SP1": "La Liga",
    "SP2": "Segunda División",
    "I1":  "Serie A",
    "I2":  "Serie B",
    "N1":  "Eredivisie",
    "P1":  "Primeira Liga",
    "B1":  "Belgian Pro League",
    "T1":  "Süper Lig",
    "SC0": "Scottish Premiership",
    "G1":  "Super League Greece",
    "F1":  "Ligue 1",
    "F2":  "Ligue 2",
}

SEASONS = [
    "1516", "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2425", "2526",
]


def season_label(s):
    return f"20{s[:2]}-{s[2:]}"


def download(session, season, code):
    url = BASE.format(season=season, code=code)

    for attempt in range(4):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()

            df = pd.read_csv(io.BytesIO(r.content))

            if df.empty:
                raise ValueError("empty CSV")

            return df

        except Exception:
            if attempt == 3:
                return None

            time.sleep(2 ** attempt)


def main():

    OUT.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 soccer-model-research/1.0"
    })

    frames = []

    print("=" * 110)
    print("BUILDING V5 FOOTBALL-DATA 1X2 MARKET STORE")
    print("=" * 110)

    for code, league in LEAGUES.items():

        league_frames = []

        for season in SEASONS:

            df = download(session, season, code)

            if df is None:
                print(f"{league:25} {season}: unavailable")
                continue

            needed = [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
                "FTR",
                "B365H",
                "B365D",
                "B365A",
            ]

            if not all(c in df.columns for c in needed):
                print(f"{league:25} {season}: missing required columns")
                continue

            x = df[needed].copy()

            x.columns = [
                "date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "result",
                "home_odds",
                "draw_odds",
                "away_odds",
            ]

            x["date"] = pd.to_datetime(
                x["date"],
                dayfirst=True,
                errors="coerce",
            )

            for c in [
                "home_goals",
                "away_goals",
                "home_odds",
                "draw_odds",
                "away_odds",
            ]:
                x[c] = pd.to_numeric(x[c], errors="coerce")

            x = x[
                x["date"].notna()
                & x["home_team"].notna()
                & x["away_team"].notna()
                & x["result"].isin(["H", "D", "A"])
            ].copy()

            x["league"] = league
            x["league_code"] = code
            x["season_code"] = season
            x["season"] = season_label(season)
            x["market_source"] = "football_data_b365_open"

            league_frames.append(x)

        if league_frames:

            lg = pd.concat(
                league_frames,
                ignore_index=True,
            )

            complete = lg[
                ["home_odds", "draw_odds", "away_odds"]
            ].notna().all(axis=1)

            print(
                f"{league:25} "
                f"games={len(lg):5d} "
                f"complete_1x2={complete.sum():5d} "
                f"coverage={complete.mean():7.2%}"
            )

            frames.append(lg)

    if not frames:
        raise RuntimeError("No Football-Data rows loaded.")

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Market probabilities
    # ---------------------------------------------------------

    out["raw_home_prob"] = 1 / out["home_odds"]
    out["raw_draw_prob"] = 1 / out["draw_odds"]
    out["raw_away_prob"] = 1 / out["away_odds"]

    overround = (
        out["raw_home_prob"]
        + out["raw_draw_prob"]
        + out["raw_away_prob"]
    )

    out["market_home_prob"] = out["raw_home_prob"] / overround
    out["market_draw_prob"] = out["raw_draw_prob"] / overround
    out["market_away_prob"] = out["raw_away_prob"] / overround

    out["market_overround"] = overround

    out = out.sort_values(
        ["league", "date", "home_team"]
    ).reset_index(drop=True)

    out.to_csv(
        OUT,
        index=False,
    )

    print("\n" + "=" * 110)
    print("FINAL DATASET")
    print("=" * 110)

    print("Rows:", len(out))
    print("Leagues:", out["league"].nunique())
    print("Date:", out["date"].min(), "->", out["date"].max())

    print("\nROWS BY LEAGUE")
    print(
        out.groupby("league")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nCOMPLETE B365 1X2 BY LEAGUE")

    complete = out[
        ["home_odds", "draw_odds", "away_odds"]
    ].notna().all(axis=1)

    print(
        out.assign(complete=complete)
        .groupby("league")["complete"]
        .agg(["sum", "count", "mean"])
        .sort_values("count", ascending=False)
        .to_string()
    )

    print("\nSaved:")
    print(OUT)


if __name__ == "__main__":
    main()
