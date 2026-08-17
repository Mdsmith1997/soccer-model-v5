# Championship V5 Under 2.5 — Strategy Freeze

## Strategy A — Official Baseline

Status:
FROZEN FOR FORWARD TESTING

Rule:

RAW V5 Under 2.5 Edge >= 11%

Historical validation:

148 bets
73 wins
75 losses
+5.34 units
+3.61% ROI

Recent 2023/24-2025/26:

128 bets
+8.63 units
+6.74% ROI

Historical market coverage:

3,864 / 3,864 Football-Data matches
100.00%

Bootstrap:

Median ROI: +3.65%
95% interval: -13.45% to +20.89%
P(ROI > 0): 65.95%

Risk:

Maximum drawdown: -9.34 units
Longest losing streak: 6

This replaces the earlier incomplete 33-bet / +8.45% result.

Strategy A has NO odds ceiling.

---

## Strategy B — Research Challenger

Status:
FROZEN RESEARCH CHALLENGER

Rule:

RAW V5 Under 2.5 Edge >= 11%

AND

Under 2.5 odds < 2.25

Historical observation:

91 bets
54 wins
37 losses
+15.46 units
+16.99% ROI

High-odds comparison:

Odds >= 2.25
57 bets
19 wins
38 losses
-10.12 units
-17.75% ROI

IMPORTANT:

The 2.25 cutoff was discovered after inspecting Championship
historical results.

Therefore the +16.99% historical result is NOT independent
validation and must NOT be presented as proven expected ROI.

League One did not independently confirm the same odds effect.

Strategy B must earn promotion through unseen forward testing.

---

## Forward-Test Rules

Track both strategies simultaneously.

Strategy A records every Championship match where:

RAW Under Edge >= 11%

Strategy B records the subset where:

RAW Under Edge >= 11%
AND Under odds < 2.25

For every signal record:

- kickoff time
- teams
- V5 Under probability
- vig-free market probability
- RAW edge
- available Under odds
- bookmaker
- bet timestamp
- closing Under odds
- closing market probability
- CLV
- result
- profit/loss

Evaluate:

- number of bets
- ROI
- units
- win rate
- average CLV
- positive CLV rate
- maximum drawdown
- longest losing streak

Do not modify either strategy during the forward test.

Any future rule change creates a new strategy version.
