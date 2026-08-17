from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

LEAGUES = {
    "E0": "Premier League",
    "E1": "Championship",
    "D1": "Bundesliga",
    "N1": "Eredivisie",
    "B1": "Belgian Pro League",
}

SEASONS = [
    "1516", "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2425", "2526",
]

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Core football information.
CORE_MAP = {
    "Div": "league_code",
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR": "ht_result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
}

# Keep market data in matches.csv for later EV/CLV work,
# but DO NOT use these columns as model predictors.
ODDS_MAP = {
    # Bet365 1X2
    "B365H": "b365_home_open",
    "B365D": "b365_draw_open",
    "B365A": "b365_away_open",

    # Market averages / maximums
    "AvgH": "avg_home_open",
    "AvgD": "avg_draw_open",
    "AvgA": "avg_away_open",
    "MaxH": "max_home_open",
    "MaxD": "max_draw_open",
    "MaxA": "max_away_open",

    # Closing 1X2 where available
    "B365CH": "b365_home_close",
    "B365CD": "b365_draw_close",
    "B365CA": "b365_away_close",
    "AvgCH": "avg_home_close",
    "AvgCD": "avg_draw_close",
    "AvgCA": "avg_away_close",
    "MaxCH": "max_home_close",
    "MaxCD": "max_draw_close",
    "MaxCA": "max_away_close",

    # Over/Under 2.5
    "B365>2.5": "b365_over25_open",
    "B365<2.5": "b365_under25_open",
    "Avg>2.5": "avg_over25_open",
    "Avg<2.5": "avg_under25_open",
    "B365C>2.5": "b365_over25_close",
    "B365C<2.5": "b365_under25_close",
    "AvgC>2.5": "avg_over25_close",
    "AvgC<2.5": "avg_under25_close",

    # Asian handicap line + prices
    "AHh": "asian_handicap_home_line_open",
    "B365AHH": "b365_ah_home_open",
    "B365AHA": "b365_ah_away_open",
    "AHCh": "asian_handicap_home_line_close",
    "B365CAHH": "b365_ah_home_close",
    "B365CAHA": "b365_ah_away_close",
}

KEEP_MAP = {**CORE_MAP, **ODDS_MAP}


def download_csv(session: requests.Session, season: str, league: str) -> pd.DataFrame:
    url = BASE_URL.format(season=season, league=league)
    last_error = None

    for attempt in range(1, 5):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content))
            if df.empty:
                raise ValueError("Downloaded CSV is empty")
            return df
        except Exception as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to download {url}: {last_error}")


def normalize(df: pd.DataFrame, season: str, league: str) -> pd.DataFrame:
    available = {c: KEEP_MAP[c] for c in df.columns if c in KEEP_MAP}
    out = df[list(available)].rename(columns=available).copy()

    out["season"] = f"{season[:2]}{season[2:]}"
    out["league_code"] = league
    out["league"] = LEAGUES[league]

    # Football-Data uses day-first dates.
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")

    # Drop malformed/footer rows but retain matches even if some odds are unavailable.
    out = out[
        out["date"].notna()
        & out["home_team"].notna()
        & out["away_team"].notna()
        & out["home_goals"].notna()
        & out["away_goals"].notna()
    ].copy()

    integer_cols = [
        "home_goals", "away_goals", "ht_home_goals", "ht_away_goals",
        "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
        "home_fouls", "away_fouls", "home_corners", "away_corners",
        "home_yellow", "away_yellow", "home_red", "away_red",
    ]
    for c in integer_cols:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    numeric_cols = [c for c in out.columns if any(
        token in c for token in ("_open", "_close", "_line")
    )]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["total_goals"] = out["home_goals"] + out["away_goals"]
    out["home_win"] = (out["result"] == "H").astype(int)
    out["draw"] = (out["result"] == "D").astype(int)
    out["away_win"] = (out["result"] == "A").astype(int)
    out["over_2_5"] = (out["total_goals"] > 2.5).astype(int)
    out["btts"] = ((out["home_goals"] > 0) & (out["away_goals"] > 0)).astype(int)

    return out


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 soccer-model-research/1.0"
    })

    frames = []
    failures = []

    for season in SEASONS:
        for league in LEAGUES:
            print(f"Downloading {season} {league} ({LEAGUES[league]})...")
            try:
                df = download_csv(session, season, league)

                raw_path = RAW_DIR / f"{season}_{league}.csv"
                df.to_csv(raw_path, index=False)

                clean = normalize(df, season, league)
                frames.append(clean)
                print(f"  -> {len(clean):,} matches")
                time.sleep(0.5)

            except Exception as exc:
                failures.append((season, league, str(exc)))
                print(f"  !! FAILED: {exc}")

    if not frames:
        raise RuntimeError("No league files downloaded successfully.")

    matches = pd.concat(frames, ignore_index=True, sort=False)
    matches = matches.sort_values(
        ["date", "league_code", "home_team", "away_team"]
    ).reset_index(drop=True)

    # Stable unique match id.
    matches.insert(
        0,
        "match_id",
        matches["league_code"]
        + "_"
        + matches["date"].dt.strftime("%Y%m%d")
        + "_"
        + matches["home_team"].str.replace(r"\W+", "", regex=True)
        + "_"
        + matches["away_team"].str.replace(r"\W+", "", regex=True),
    )

    output = PROCESSED_DIR / "matches.csv"
    matches.to_csv(output, index=False)

    print("\n==============================")
    print("SOCCER DATABASE COMPLETE")
    print("==============================")
    print(f"Matches: {len(matches):,}")
    print(f"Date range: {matches['date'].min().date()} -> {matches['date'].max().date()}")
    print("\nBy league:")
    print(matches.groupby("league").size().sort_values(ascending=False).to_string())
    print("\nBy season:")
    print(matches.groupby("season").size().to_string())
    print(f"\nSaved: {output}")

    if failures:
        print("\nDownloads that failed:")
        for season, league, err in failures:
            print(f"  {season} {league}: {err}")


if __name__ == "__main__":
    main()
