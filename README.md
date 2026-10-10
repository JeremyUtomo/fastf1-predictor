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
| Session analysis | Pick a session in the sidebar: Overview, Laps, Strategy, Telemetry, Weather, Race control and Data tabs |

## Predictions

**Markets**: pole, front row, top 3, Q3 and Q2 for qualifying; win, podium,
top 6, points and retirement for sprints and races; winning team, head-to-heads
and teammate battles. Each probability comes with *fair odds* (1 ÷ probability).

**Inputs**
- Starting grid from qualifying (editable for penalties). Before qualifying the
  grid is estimated from recent form plus this weekend's pace: Practice 3 on
  normal weekends, Sprint Qualifying on sprint weekends.
- Recent finishing and qualifying positions, weighted towards recent races.
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

**Method**: a Plackett-Luce ranking model fitted on the finishing order of the
last four seasons, using only information available before each event, then
20,000 simulated sessions. Sprint and rain effects are rules of thumb, not fitted.

**Accuracy**: every session view includes a walk-forward backtest. After
qualifying the model is about as good as trusting the grid for winner and
podium markets and clearly better for the rest of the order. Before
qualifying it is less accurate. Bookmaker odds include a margin, so don't
expect these numbers to beat them reliably.

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

## Data sources and credits

- [FastF1](https://github.com/theOehrly/Fast-F1) (MIT) for F1 timing data,
  telemetry and the Ergast interface.
- [Jolpica-F1](https://github.com/jolpica/jolpica-f1), the Ergast-compatible API,
  for historical results.
- [Open-Meteo](https://open-meteo.com/) for weather forecasts
  ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- [FIA](https://www.fia.com/documents) car presentation documents for declared upgrades.

## License

MIT, see [LICENSE](LICENSE).
