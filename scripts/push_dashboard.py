from pathlib import Path
from datetime import datetime, timezone
import os
import re
import subprocess
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
SHOW_BOARD = ROOT / "scripts" / "show_board.py"

WEB_APP_URL = os.environ.get("V5_DASHBOARD_URL", "").strip()
TOKEN = os.environ.get("V5_DASHBOARD_TOKEN", "").strip()

if not WEB_APP_URL:
    raise SystemExit("Missing V5_DASHBOARD_URL")

if not TOKEN:
    raise SystemExit("Missing V5_DASHBOARD_TOKEN")

if not SHOW_BOARD.exists():
    raise SystemExit(f"Missing: {SHOW_BOARD}")

result = subprocess.run(
    [sys.executable, str(SHOW_BOARD)],
    cwd=ROOT,
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    raise SystemExit(
        f"show_board.py failed with exit code {result.returncode}"
    )

board = result.stdout.strip()

official_count = 0

m = re.search(
    r"UNIQUE OFFICIAL/CEMENTED SIGNALS NEXT\s*72H:\s*(\d+)",
    board,
    re.I,
)

if m:
    official_count = int(m.group(1))

manual_count = len(
    re.findall(
        r"MANUAL .*?PRICE CHECK",
        board,
        re.I,
    )
)

captured_count = len(
    re.findall(
        r"PRICE CAPTURED",
        board,
        re.I,
    )
)

unavailable_count = 0

coverage_match = re.search(
    r"1X2 PRICE COVERAGE(.*?)(?:BTTS SPECIALIST PRICE COVERAGE|\\Z)",
    board,
    re.I | re.S,
)

if coverage_match:
    unavailable_lines = [
        line
        for line in coverage_match.group(1).splitlines()
        if "MODEL UNAVAILABLE" in line.upper()
    ]

    excluded_leagues = {
        "Denmark Superliga",
        "Liga MX",
    }

    unresolved_lines = []

    for line in unavailable_lines:
        league = line.split("|", 1)[0].strip()

        if league not in excluded_leagues:
            unresolved_lines.append(line)

    unavailable_count = len(unresolved_lines)

payload = {
    "token": TOKEN,
    "dashboard": {
        "pushed_at": datetime.now(timezone.utc).isoformat(),
        "official_count": official_count,
        "manual_count": manual_count,
        "captured_count": captured_count,
        "unavailable_count": unavailable_count,
        "raw_board": board,
    },
}

print("=" * 80)
print("PUSHING V5 DASHBOARD")
print("=" * 80)
print("Official plays:", official_count)
print("Manual checks:", manual_count)
print("Prices captured:", captured_count)
print("Model unavailable:", unavailable_count)

response = requests.post(
    WEB_APP_URL,
    json=payload,
    timeout=30,
)

print("HTTP:", response.status_code)
print("Response:", response.text[:500])

response.raise_for_status()

data = response.json()

if not data.get("ok"):
    raise SystemExit(
        f"Dashboard rejected update: {data}"
    )

print("✅ PHONE DASHBOARD UPDATED")
