"""How it works page: the data, the features, how much each one weighs and how
strengths become probabilities. Weights and calibration values are read from the
models fitted for the next weekend, so the page never goes out of date."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import data
import model
import predict_data as pdata
import predict_tab
import ui

# features that measure the same thing are grouped: their individual weights can
# offset each other (e.g. grid and recent qualifying), the group's total can't
GROUPS = {
    "Starting grid": ["grid", "grid_log"],
    "Car": ["team_gap", "team_pace", "team_trend"],
    "Recent form": ["form", "quali_form", "quali_gap", "race_pace"],
    "Driver vs. teammate": ["driver_edge"],
    "Practice this weekend": ["fp", "fp_gap", "sq"],
    "Circuit history": ["circuit_aff"],
}
_H, _DH = model.HALF_LIFE, model.DRIVER_HALF_LIFE
BUILT = {
    "grid": "Grid slot 1–22 (pit-lane start = 22)",
    "grid_log": "Logarithm of the grid slot",
    "form": f"Last 6 finishes, half-life {_H:g} races; retirements left out",
    "quali_form": f"Last 6 qualifying positions, half-life {_H:g} races",
    "quali_gap": f"Last 6 qualifying gaps to the fastest time, half-life {_H:g} races",
    "race_pace": f"Last 6 fastest-lap gaps to the race's fastest lap, half-life {_H:g} races",
    "team_gap": f"Both cars' qualifying gap, last 6 races, half-life {_H:g} races",
    "team_pace": f"Both cars' qualifying position, last 6 races, half-life {_H:g} races",
    "team_trend": "Slope of the team's qualifying position over the last 5 races",
    "driver_edge": f"Last 30 teammate comparisons, half-life {_DH:g} races; "
                   "pulled towards 0 for drivers with few comparisons",
    "circuit_aff": "Past results here vs. the form at the time; pulled towards 0",
    "fp": "Rank of the best lap in Practice 3",
    "fp_gap": "Gap of the best lap to the fastest in Practice 3",
    "sq": "Sprint Qualifying position",
}
LABEL = {"grid": "Grid position", "grid_log": "Grid position (log)", "form": "Recent finishes",
         "quali_form": "Recent qualifying", "quali_gap": "Qualifying gap %",
         "race_pace": "Race pace %", "team_gap": "Team qualifying gap %",
         "team_pace": "Team qualifying position", "team_trend": "Team trend",
         "driver_edge": "Driver edge %", "circuit_aff": "Circuit history",
         "fp": "Practice 3 rank", "fp_gap": "Practice 3 gap %", "sq": "Sprint Qualifying"}


def _influence(fit: model.RankFit, rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per group: how much it separates drivers (spread of its share of the strength
    over the training rows), as a share of all groups and as a head-to-head chance."""
    rows = rows.dropna(subset=fit.features)
    Z = (rows[fit.features].to_numpy(float) - fit.mean) / fit.std
    out = []
    for name, cols in GROUPS.items():
        idx = [i for i, f in enumerate(fit.features) if f in cols]
        if idx:
            out.append((name, float(np.std(Z[:, idx] @ fit.beta[idx]))))
    g = pd.DataFrame(out, columns=["Factor", "spread"])
    g["Share"] = g["spread"] / g["spread"].sum()
    # P(A ahead of B) = 1 / (1 + exp(-(sA - sB))) for two drivers one spread apart
    g["Head-to-head"] = 1 / (1 + np.exp(-g["spread"]))
    g = g.sort_values("Share", ascending=False)
    w = fit.weight_table()
    detail = pd.DataFrame({
        "Factor": [next(k for k, v in GROUPS.items() if f in v) for f in w["feature"]],
        "Feature": w["feature"].map(LABEL), "How it's built": w["feature"].map(BUILT),
        "Weight": w["weight"].round(2)})
    order = {f: i for i, f in enumerate(g["Factor"])}
    detail = detail.sort_values("Factor", key=lambda s: s.map(order), kind="stable")
    return g, detail


def _show_model(fit: model.RankFit, rows: pd.DataFrame, seasons: int, note: str) -> None:
    groups, detail = _influence(fit, rows)
    st.caption(f"Fitted on {fit.n_races} events from the last {seasons} seasons. {note}")
    shown = groups[["Factor", "Share", "Head-to-head"]].assign(
        Share=groups["Share"] * 100, **{"Head-to-head": groups["Head-to-head"] * 100})
    st.dataframe(
        shown, hide_index=True, width="stretch",
        height=ui.table_height(len(groups)), row_height=ui.ROW_HEIGHT,
        column_config={
            "Share": st.column_config.ProgressColumn(
                "Share of influence", format="%.0f%%", min_value=0.0, max_value=100.0,
                help="How much this factor separates drivers, relative to the other factors"),
            "Head-to-head": st.column_config.NumberColumn(
                "Typical head-to-head", format="%.0f%%",
                help="Two drivers equal on everything else, one typical step apart on this "
                     "factor: how often the better one finishes ahead")})
    with st.expander("Every feature and its weight"):
        st.table(detail.set_index("Factor").style.format({"Weight": "{:+.2f}"}))
        st.caption(
            "Weights are per standard deviation of the feature; positive = a better value "
            "helps. Features within a factor overlap (the grid already reflects recent "
            "qualifying), so one can carry a negative weight that corrects another: read the "
            "factor totals above, not a single weight.")


def render() -> None:
    year = data.current_season()
    now = data.now_utc()
    rnd = predict_tab._next_round(year, now)
    if rnd is None:  # season over: the models for the final round
        rnd = int(data.get_schedule(year)["RoundNumber"].max())
    name = data.get_schedule(year).set_index("RoundNumber").loc[rnd, "EventName"]
    st.caption(f"The numbers on this page are from the models fitted for the {year} {name}. "
               "Every weekend the models are fitted again on the results before it.")

    st.markdown("##### In short")
    st.markdown(f"""
1. **Collect results.** Every race, qualifying and sprint since {pdata.DATA_START}:
   positions, lap-time gaps and retirements.
2. **Describe each driver** before the session with a few numbers: recent form, the car's pace,
   the driver against their teammate, history at this circuit, and the grid or practice pace
   once they're known.
3. **Weigh them.** A ranking model learns how much each number matters by looking at which
   driver beat which in past sessions. The result is one *strength* per driver.
4. **Simulate the session 20,000 times.** Each run adds luck and random retirements to the
   strengths. Win, podium and other probabilities are how often each outcome happened.
5. **Correct the confidence** with what past predictions got right and wrong.
""")

    st.markdown("##### Data used")
    st.markdown(f"""
| Source | What | Used for |
|---|---|---|
| Jolpica (Ergast) via FastF1 | Race, qualifying and sprint results since {pdata.DATA_START}: positions, Q1–Q3 times, fastest laps, retirements | Every feature; what the model learns from |
| F1 live timing via FastF1 | Practice lap times, entry lists, weather on track | Practice pace, which past races were wet |
| FIA starting grid documents | Provisional and final grid | Grid penalties and pit-lane starts |
| Open-Meteo | Rain forecast at the circuit | Rain probability (editable) |
| FIA car presentation documents | Declared upgrades | Shown only: they did not improve the predictions in testing |
""")
    st.caption("Lap-time gaps are in % of the fastest time and capped at 5%, so a wet "
               "session or a lap without a time can't dominate.")

    st.markdown("##### How much each factor weighs")
    st.markdown(
        "*Share of influence* is how much a factor separates the drivers, compared with the "
        "others. *Typical head-to-head* makes it concrete: take two drivers who are equal on "
        "everything else and a typical step apart on this factor; this is how often the better "
        "one finishes ahead, before luck and retirements are added.")
    cfg = model.CONFIG
    _, hist, _ = predict_tab._prepared(year, rnd)
    fit_post, fit_pre, _, _ = predict_tab._fits(year, rnd)
    race_rows = model.training_rows(hist, year, rnd, cfg["race_post"]["train_seasons"])
    tabs = st.tabs(["Race after qualifying", "Race before qualifying", "Qualifying"])
    with tabs[0]:
        _show_model(fit_post, race_rows[race_rows["classified"]],
                    cfg["race_post"]["train_seasons"],
                    "Sprints use the same model.")
    with tabs[1]:
        _show_model(fit_pre, race_rows[race_rows["classified"]],
                    cfg["race_pre"]["train_seasons"],
                    f"Practice pace is blended into recent qualifying "
                    f"({predict_tab.FP_BLEND:.0%}); on sprint weekends the grid is estimated "
                    "from Sprint Qualifying and the after-qualifying model is used.")
    with tabs[2]:
        qrows = predict_tab._quali_data(year, rnd)
        qrows = model.training_rows(qrows, year, rnd, cfg["quali"]["train_seasons"])
        _show_model(predict_tab._quali_fit(year, rnd, "practice"), qrows,
                    cfg["quali"]["train_seasons"],
                    "Normal weekend after Practice 3. Without practice data the same model runs "
                    "without that factor; on sprint weekends Sprint Qualifying takes its place.")

    st.markdown("##### From strengths to probabilities")
    d = model.DNF
    st.markdown(f"""
- **Order.** In each of the 20,000 simulated sessions every driver gets their strength plus a
  random amount of luck; the order of the totals is the result. A driver 1 strength point
  ahead of another finishes ahead about 73% of the time.
- **Retirements.** Each driver's chance is {1 - d['driver_weight']:.0%} the team's record over
  its last {d['team_window']} starts and {d['driver_weight']:.0%} the driver's own over their last
  {d['driver_window']}, both pulled strongly towards the field average (as if every record had
  {d['prior']} extra average starts): most retirements are bad luck, not a pattern.
  Teammates' retirements are linked (correlation {d['teammate_rho']}), so both cars out is more
  likely than pure chance. Sprints use {predict_tab.SPRINT_DNF:.0%} of the race rate.
- **Circuit.** At tracks where positions usually change a lot, the luck part is larger
  (×0.75 to ×1.35, from past races there).
- **Rain** (rules of thumb, not fitted): at 100% rain, luck +{predict_tab.RAIN_RANDOMNESS:.0%} and
  retirements +{predict_tab.RAIN_DNF:.0%}. A driver's wet record against their teammate moves
  them by at most {predict_tab.WET_CAP} places.
- **Before qualifying** the grid is unknown. The model then leans on the car, recent pace and
  practice; on sprint weekends it simulates many plausible grids from Sprint Qualifying.
""")

    st.markdown("##### Confidence correction")
    temps = predict_tab._temperatures(year, rnd)
    recal, _ = predict_tab._betting_calibration(year, rnd, "race")
    rows = [(predict_tab._KIND_LABEL.get(k, k), t["temperature"], t["n"])
            for k, t in temps.items()]
    st.markdown(
        "The model is re-run on past events using only data from before each one, and the "
        "amount of luck is tuned so its probabilities match how often things actually "
        "happened. Below 1 means the raw model was too cautious and its favourites are "
        "made stronger.")
    st.dataframe(pd.DataFrame(rows, columns=["Session", "Luck factor", "Past events"]),
                 hide_index=True, width="stretch", height=ui.table_height(len(rows)),
                 row_height=ui.ROW_HEIGHT,
                 column_config={"Luck factor": st.column_config.NumberColumn(format="%.2f")})
    a, b = recal
    st.markdown(
        f"Race and sprint **win** probabilities get one more correction, fitted the same way "
        f"(slope {b:.2f}, shift {a:+.2f}). "
        + ("A slope above 1 means favourites won more often than predicted and long shots "
           "less, so favourites are raised and long shots lowered."
           if b > 1 else "A slope below 1 means favourites won less often than predicted, "
           "so favourites are lowered and long shots raised.")
        + " Qualifying needs no extra correction.")

    st.markdown("##### What the model doesn't know")
    st.markdown("""
- Set-up changes after practice, engine modes, tyre allocation, track evolution.
- Strategy calls, safety cars and red flags beyond what a circuit's history implies.
- News: a driver feeling unwell or a team's known problem. Use *Adjustments* on the
  Predictions page for these.
- Tested and **not** used because they didn't improve past predictions: upgrade counts,
  weighting recent seasons more, this weekend's pole gap, a team's retirement rate this season.
""")
    st.caption("How well it has done is on the Track record page: saved predictions against "
               "results, and a backtest of every event since 2020.")
