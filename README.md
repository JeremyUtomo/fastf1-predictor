# fastf1-predictor

A dark-mode Streamlit dashboard built on top of [FastF1](https://github.com/theOehrly/Fast-F1):
Formula 1 race, sprint and qualifying predictions, plus point-and-click analysis
of any session since 2018: results, lap times, tyre strategy, telemetry,
weather and race control messages.

> Not affiliated with Formula 1, the FIA or any bookmaker. Predictions are
> statistical estimates, not betting advice.

## Setup

Python 3.10 or newer (3.13 is tested).

```bash
git clone https://github.com/jeremyutomo/fastf1-predictor.git
cd fastf1-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501. Run it from the repository folder so the dark theme
in `.streamlit/config.toml` is picked up.

Data is downloaded on first use and cached in `~/.cache/fastf1` (override with
the `FASTF1_CACHE` environment variable), so later loads are quick. The
Predictions page needs about 30 s the first time to collect several seasons of
results and practice data.

## Pages

| Page | What it shows |
| --- | --- |
| Season | Calendar and driver / constructor standings |
| Predictions | Qualifying, sprint and race probabilities, results vs. prediction, comparison with bookmaker odds |
| Track record | Saved predictions scored against results, a backtest of every event since 2020, and what the model learned (calibration) |
| Session analysis | Pick a session in the sidebar: Overview, Laps, Strategy, Telemetry, Weather, Race control and Data tabs |

## Predictions

**Markets**: pole, front row, top 3, Q3 and Q2 for qualifying; win, podium,
top 6, points and retirement for sprints and races; winning team, head-to-heads
and teammate battles. Each probability comes with *fair odds* (1 ÷ probability).

**Inputs**
- Starting grid from the official FIA starting grid (penalties, pit-lane starts),
  or the qualifying result until that's published; editable. Before qualifying the
  grid is estimated from recent form plus this weekend's pace: Practice 3 on
  normal weekends, Sprint Qualifying on sprint weekends.
- Recent finishing and qualifying positions, weighted towards recent races.
- **Qualifying time gaps** (in % to the fastest time), not only positions: P2 by
  0.03 s and P2 by 0.8 s are very different.
- **Driver vs. teammate**: how much faster a driver is than their teammate in the
  same car, with a long memory, to separate driver skill from car performance.
- **Race pace**: each driver's fastest race lap relative to the fastest of the race.
- Team qualifying pace and its trend over the last five races.
- The driver's record at the circuit, how much positions usually change there,
  and driver/team retirement rates.
- Rain forecast for the session (Open-Meteo) and, optionally, each driver's
  wet-weather record relative to their teammate.
- Manual adjustments for anything no data covers: penalties, upgrade
  packages, withdrawals.

**Before and after qualifying**: until qualifying is final the race grid is
estimated. Afterwards the race view uses the real grid and shows how each
driver's chances changed.

**After a session**: the real result appears next to the prediction, with a
*Result vs. prediction* summary (actual winner and the chance the model gave
them, where the favourite finished, top-3 hits, average position error) and a
predicted-vs-actual chart.

**Method**: a Plackett-Luce ranking model fitted on the finishing order of
recent seasons, using only information available before each event, then
20,000 simulated sessions. Which inputs and settings are used is decided by a
walk-forward research run (below) and stored in `model_config.json`. Sprint and
rain effects are rules of thumb; overall confidence is learned from results.

**Accuracy**: every session view and the **Track record** page show a
walk-forward backtest of every event since 2020: what the model would have
predicted at the time vs. what happened. Bookmaker odds include a margin, so
don't expect these numbers to beat them reliably.

### Research and tuning

`python evaluate.py --tune` re-predicts every race, sprint and qualifying
session from 2020 on (each with a model fitted only on earlier events), tests
candidate improvements one by one on 2020–2025, keeps those that clearly help,
checks the result once on 2026, and writes `model_config.json` plus
[`research_report.md`](research_report.md). Re-run it to re-tune the model; the
app picks the new settings up without code changes.

What the latest run found (winner/pole log loss, lower is better, 2020–2025):

| Model | Before | After |
| --- | --- | --- |
| Race, after qualifying | 1.252 | 1.192 |
| Race, before qualifying | 1.608 | 1.498 |
| Qualifying | 1.554 | 1.521 |
| Sprint | 1.208 | 1.120 |

- Adopted: qualifying **time gaps**, **driver vs. teammate** (both suggested by
  research separating driver skill from constructor strength, e.g. van Kesteren &
  Bergkamp 2023) and **race pace**; qualifying also trains on six seasons.
- Not adopted: time-weighting older races (helpful in Henderson & Kirrane 2018,
  but our form features already emphasise recent races), longer training windows
  for races, a grid × circuit interaction, and this weekend's pole gap.
- The 2026 check (16 events, new rules) shows small differences in both
  directions that are within noise; details per season are in the report.

### Learning from results

The model improves as the season goes on in two ways:

- **Retraining**: every prediction is fitted on all events before it, so each new
  result is automatically part of the next prediction.
- **Self-calibration**: every event of the last seasons is re-predicted using only
  earlier data and compared with what happened. From that the model learns, per
  session type, whether it was too cautious or too confident, and corrects its
  probabilities (shrunk towards no correction while there are few events). This
  is recalculated after every new result. Currently: qualifying needs no
  correction, races are sharpened slightly, sprints considerably (they're more
  predictable than a full race).

Predictions shown before a session starts are saved to
`~/.cache/fastf1/dashboard/prediction_log.csv`. The **Track record** page scores
them against the real results and shows a calibration chart (does a predicted 30%
happen about 30% of the time?).

### Declared upgrades

Before every Grand Prix each team declares its new parts to the FIA, which
publishes them as a PDF. `upgrades.py` downloads these from fia.com (respecting
the 10 s crawl delay in its robots.txt), keeps them in
`~/.cache/fastf1/fia_upgrades/` and parses them per part. The Predictions page
lists them for the selected Grand Prix.

They are not used by the model: in a walk-forward test on 40 events of
2025–2026, adding upgrade counts made qualifying and race predictions slightly
worse, as their effect already shows up in practice pace and the team trend.
To build the history (about 20 minutes because of the crawl delay):

```bash
python upgrades.py 2024 2026
```

### Bookmaker odds

Odds are not scraped. Copy a market from your bookmaker's page and paste it
under *Paste odds*: driver names and decimal odds (2.50 or 2,50) are recognised
and compared with the model, including the bookmaker's margin.

A high "edge" on a long shot is usually model error, not value: in the backtest,
drivers the raw model gave 1–5% won 2–4 times less often than that (the
favourite–longshot bias), and a fair 1-in-50 bet still loses 49 times out of 50.
So each bet shows:

- **Hits ~1 in** — how rarely it comes in;
- **Edge** — expected return per €1 if the model's probability is right;
- **Cautious edge** — the same using the low end of how often predictions in that
  probability range actually came true in the backtest;
- **Verdict** — *Value* only when even the cautious edge is positive.

Win probabilities for races and sprints are recalibrated with a correction learned
from the backtest (it fixes favourites and pulls long shots down); pole and podium
probabilities were already well calibrated.

### Starting grid

After qualifying, the race and sprint use the **official FIA starting grid**
(provisional, then final), so grid penalties and pit-lane starts are included.
Until it's published the qualifying result is used; the grid can always be
edited under *Adjustments*.

## Data sources and credits

- [FastF1](https://github.com/theOehrly/Fast-F1) (MIT) for F1 timing data,
  telemetry and the Ergast interface.
- [Jolpica-F1](https://github.com/jolpica/jolpica-f1), the Ergast-compatible API,
  for historical results.
- [Open-Meteo](https://open-meteo.com/) for weather forecasts
  ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- [FIA](https://www.fia.com/documents) car presentation documents for declared upgrades.
- Research: Henderson & Kirrane (2018), [time-weighted Plackett–Luce models for
  F1](https://projecteuclid.org/journals/bayesian-analysis/volume-13/issue-2/A-Comparison-of-Truncated-and-Time-Weighted-PlackettLuce-Models-for/10.1214/17-BA1048.full);
  van Kesteren & Bergkamp (2023), [driver skill vs. constructor
  advantage](https://arxiv.org/abs/2203.08489v2).

## License

MIT, see [LICENSE](LICENSE).
