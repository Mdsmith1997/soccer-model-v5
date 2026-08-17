# Soccer Model V1

Initial historical database for:

- Premier League (`E0`)
- Championship (`E1`)
- Bundesliga (`D1`)
- Eredivisie (`N1`)
- Belgian Pro League (`B1`)

Seasons: 2015/16 through 2025/26.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows activation:

```bash
.venv\Scripts\activate
```

## Build the match database

```bash
python scripts/download_matches.py
```

Output:

```text
data/processed/matches.csv
```

Raw source CSVs are also retained under `data/raw/`.

## Modeling rule

Sportsbook odds are retained in `matches.csv` for later market evaluation, EV, ROI,
and CLV analysis. They must not be used as predictive features in the initial
football model.

## Next step

Build leakage-safe pregame team features using only information available before
each kickoff:

- attack strength
- defensive strength
- home/away splits
- weighted recent goals
- weighted shots / shots on target where available
- points/form
- rest
- promoted/new-team handling
- league baseline scoring rates
