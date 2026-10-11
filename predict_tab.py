"""Streamlit UI for the Predictions tab."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import calibration
import charts
import data
import evaluate
import grid_docs
import model
import predict_data as pdata
import ui
import upgrades

# Heuristic factors (not fitted - there is too little sprint/wet data to fit them)
SPRINT_DNF = 0.4           # roughly a third of the distance
RAIN_RANDOMNESS = 0.35     # +35% randomness at 100% rain probability
RAIN_DNF = 0.25            # +25% retirement chance at 100% rain probability
WET_RACE = 0.15            # share of rainy weather samples for a race to count as wet
WET_SHRINK = 6             # wet races needed before a driver's wet record counts half
WET_CAP = 1.5              # max places a wet record can move a driver (few wet races = weak evidence)
FP_BLEND = 0.5             # weight of this weekend's practice pace vs. recent qualifying
MARKETS = ["Win", "Podium", "Top 6", "Points (top 10)"]
QUALI_MARKETS = ["Pole", "Front row", "Top 3", "Q3 (top 10)", "Q2 (top 15)"]
PRACTICE_ORDER = ["Practice 3", "Practice 2", "Practice 1"]


# ----------------------------------------------------------------- helpers
def _sessions(event: pd.Series) -> dict[str, pd.Timestamp]:
    out = {}
    for i in range(1, 6):
        name, start = event.get(f"Session{i}"), event.get(f"Session{i}DateUtc")
        if isinstance(name, str) and name and pd.notna(start):
            out[name] = start
    return out


def _next_round(year: int, now: pd.Timestamp) -> int | None:
    """The next weekend: the first round whose race hasn't finished (start + 3 h).
    Only its predictions are saved, so browsing later weekends doesn't log them."""
    schedule = data.get_schedule(year)
    later = schedule[(schedule["RoundNumber"] > 0)
                     & (schedule["Session5DateUtc"] > now - pd.Timedelta(hours=3))]
    return int(later["RoundNumber"].iloc[0]) if len(later) else None


def _quali_name(sessions: dict, target: str) -> str | None:
    if target == "Race":
        return "Qualifying" if "Qualifying" in sessions else None
    for name in ("Sprint Qualifying", "Sprint Shootout"):
        if name in sessions:
            return name
    return None


@st.cache_resource(show_spinner=False, max_entries=6)
def _prepared(year: int, rnd: int):
    races, quali = pdata.history(year, start=pdata.DATA_START)
    hist, builder = model.build_history(races, quali, stop_before=(year, rnd))
    return races, hist, builder


@st.cache_resource(show_spinner=False, max_entries=6)
def _grid_estimator(year: int, rnd: int):
    """Qualifying ~ Sprint Qualifying + recent quali form, fitted on past sprint weekends."""
    _, quali = pdata.history(year, start=pdata.DATA_START)
    _, hist, _ = _prepared(year, rnd)
    sprints = pdata.sprints_range(pdata.DATA_START, year)
    sprints = sprints[(sprints["season"] < year) | (sprints["round"] < rnd)]
    return model.fit_grid_estimate(hist, sprints, quali)


@st.cache_resource(show_spinner=False, max_entries=6)
def _fits(year: int, rnd: int):
    """Race model after qualifying, before qualifying, and for sprints (settings from
    model_config.json, chosen by `python evaluate.py --tune`)."""
    _, hist, _ = _prepared(year, rnd)
    sprint_feats = [f for f in model.CONFIG["race_post"]["features"] if f != "qgap_now"]
    return (model.fit_family(hist, "race_post", year, rnd),
            model.fit_family(hist, "race_pre", year, rnd),
            model.fit_family(hist, "race_post", year, rnd, features=sprint_feats),
            model.qgap_lookup(hist))


def _id_maps(races: pd.DataFrame) -> tuple[dict, dict]:
    """Abbreviation -> driverId and driverId -> constructorId from the latest results."""
    latest = races.sort_values(["season", "round"]).drop_duplicates("driverId", keep="last")
    return (dict(zip(latest["driverCode"], latest["driverId"])),
            dict(zip(latest["driverId"], latest["constructorId"])))


def _entries(event, sessions: dict, year: int, rnd: int, races: pd.DataFrame) -> pd.DataFrame:
    """Entry list of this weekend (falls back to the latest race's entry list)."""
    abbr_to_id, drv_team = _id_maps(races)
    now = data.now_utc()
    for name, start in reversed([(n, s) for n, s in sessions.items() if s < now]):
        try:
            res = pdata.load_weekend_session(year, rnd, name, False, start).results
        except Exception:
            continue
        if res is None or res.empty:
            continue
        e = pd.DataFrame({"Driver": res["Abbreviation"], "Team": res["TeamName"],
                          "TeamColor": res["TeamColor"], "LastName": res["LastName"],
                          "driverId": res["DriverId"].replace("", np.nan)})
        e["driverId"] = e["driverId"].fillna(e["Driver"].map(abbr_to_id))
        e["driverId"] = e["driverId"].fillna(e["Driver"].str.lower())  # rookie, no history
        team_ids = res["TeamId"].replace("", np.nan)
        e["constructorId"] = team_ids.fillna(e["driverId"].map(drv_team)).values
        # a new driver inherits the constructor of a teammate with history
        by_team = e.dropna(subset=["constructorId"]).drop_duplicates("Team").set_index("Team")
        e["constructorId"] = e["constructorId"].fillna(e["Team"].map(by_team["constructorId"]))
        e["constructorId"] = e["constructorId"].fillna(e["Team"].str.lower())
        return e.reset_index(drop=True)
    last = races[races["season"] == races["season"].max()]
    last = last[last["round"] == last["round"].max()]
    return pd.DataFrame({"Driver": last["driverCode"], "Team": last["constructorName"],
                         "TeamColor": "", "LastName": last["familyName"],
                         "driverId": last["driverId"],
                         "constructorId": last["constructorId"]}).reset_index(drop=True)


def _quali_gaps(sessions: dict, target: str, year: int, rnd: int) -> pd.Series | None:
    """This weekend's qualifying gap to pole (%), abbreviation-indexed, once it's final."""
    name = _quali_name(sessions, target)
    if _grid(sessions, target, year, rnd) is None:
        return None
    try:
        res = pdata.load_weekend_session(year, rnd, name, True, sessions[name]).results
        best = res.set_index("Abbreviation")[["Q1", "Q2", "Q3"]].min(axis=1)
        return pdata._gap_pct(best)
    except Exception:
        return None


def _grid(sessions: dict, target: str, year: int, rnd: int) -> pd.Series | None:
    """Qualifying classification (abbreviation -> position) if it's available."""
    name = _quali_name(sessions, target)
    if not name or sessions[name] > data.now_utc():
        return None
    try:
        session = pdata.load_weekend_session(year, rnd, name, True, sessions[name])
    except Exception:
        return None
    if not pdata.is_settled(sessions[name]) and not pdata.session_finished(session):
        return None  # still running: a partial classification (e.g. after Q1) isn't the grid
    try:
        pos = session.results.set_index("Abbreviation")["Position"].dropna()
    except Exception:  # results not available yet
        return None
    return pos if len(pos) >= 10 else None


def _practice_rank(sessions: dict, year: int, rnd: int) -> pd.Series | None:
    """Average rank of each driver's best lap over this weekend's practice sessions."""
    ranks = []
    for name, start in sessions.items():
        if not name.startswith("Practice") or start > data.now_utc():
            continue
        try:
            laps = pdata.load_weekend_session(year, rnd, name, True, start).laps
        except Exception:
            continue
        best = laps.dropna(subset=["LapTime"]).groupby("Driver")["LapTime"].min()
        if len(best) >= 10:
            ranks.append(best.rank())
    return pd.concat(ranks, axis=1).mean(axis=1) if ranks else None


def _wet_gain(hist: pd.DataFrame, year: int) -> pd.Series:
    """Extra places a driver gains on their teammate in wet races vs. dry races.

    Comparing with the teammate removes the car, and avoids the ceiling effect of
    grid-vs-finish (a driver starting P1 can't gain places). Shrunk and capped
    because there are only a handful of wet races.
    """
    recent = hist[hist["season"] >= year - 1][["season", "round"]].drop_duplicates()
    progress = st.progress(0.0, text="Checking which recent races were wet…")
    wet = {}
    for i, (s, r) in enumerate(recent.itertuples(index=False)):
        frac = pdata.race_rain_fraction(int(s), int(r))
        wet[(s, r)] = frac is not None and frac > WET_RACE
        progress.progress((i + 1) / len(recent), text="Checking which recent races were wet…")
    progress.empty()
    h = hist[hist["classified"] & (hist["season"] >= year - 1)].copy()
    pairs = h.groupby(["season", "round", "constructorId"])["finish"].transform("sum")
    n_cars = h.groupby(["season", "round", "constructorId"])["finish"].transform("size")
    h = h[n_cars == 2].copy()
    h["vs_mate"] = (pairs[n_cars == 2] - h["finish"]) - h["finish"]  # mate's pos - own pos
    h["wet"] = [wet.get((s, r), False) for s, r in zip(h["season"], h["round"])]
    out = {}
    for drv, g in h.groupby("driverId"):
        w, d = g[g["wet"]]["vs_mate"], g[~g["wet"]]["vs_mate"]
        if len(w) == 0 or len(d) == 0:
            continue
        gain = (w.mean() - d.mean()) / 2 * len(w) / (len(w) + WET_SHRINK)  # half: the mate moves too
        out[drv] = float(np.clip(gain, -WET_CAP, WET_CAP))
    st.session_state["n_wet_races"] = int(sum(wet.values()))
    return pd.Series(out, dtype=float)


def _colors(entries: pd.DataFrame) -> list[str]:
    return [f"#{c}" if isinstance(c, str) and c else charts.FALLBACK_COLORS[i % 6]
            for i, c in enumerate(entries["TeamColor"])]


def _pct(df: pd.DataFrame, cols: list[str]) -> dict:
    """Column config: probabilities shown as whole percentages with a bar."""
    return {c: st.column_config.ProgressColumn(c, format="%.0f%%", min_value=0, max_value=100)
            for c in cols if c in df}


def _as_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Display copy with probabilities (0–1) scaled to percent for `_pct`."""
    out = df.copy()
    for c in cols:
        if c in out:
            out[c] = out[c] * 100
    return out


# ------------------------------------------------------------------- render
def render() -> None:
    st.caption("Model estimates based on historical results and this weekend's sessions. "
               "Not betting advice.")
    with st.expander("How to use this page", expanded=False):
        st.markdown("""
1. **Select a Grand Prix**, then a session: *Qualifying*, *Race* or *Sprint*.
2. **Read the main table.** Probabilities per driver for each market. *Fair odds* are
   1 ÷ probability, i.e. the price without a bookmaker margin.
3. **Race before and after qualifying.** Until qualifying is final the grid is estimated.
   Afterwards the race view uses the official FIA starting grid (grid penalties and pit-lane
   starts included) or, until that's published, the qualifying result.
   Reload the page once qualifying has finished.
4. **Inputs.** The rain probability is pre-filled from the forecast. Use *Adjustments* for
   grid penalties, upgrade packages or withdrawals.
5. **Declared upgrades** lists the parts each team registered with the FIA. Shown for
   reference only: in testing, adding them did not improve the predictions.
6. **Compare with bookmaker odds.** Open *Paste odds*, copy a market from the bookmaker's
   page and paste it. Each bet gets an *edge*, a *cautious edge* (based on how often similar
   predictions actually came true) and a verdict. Only **Value** bets stay positive even in
   the cautious case. *Hits ~1 in* shows how rarely a bet comes in: long shots lose most of
   the time even when the price is fair.
7. **Check *Model accuracy*** at the bottom of each session before relying on a number.

What the model uses and how much each factor weighs: [How it works](/how-it-works).
""")

    c0, c1, c3 = st.columns([1, 4, 1])
    with c0:
        seasons = list(range(data.current_season(), data.FIRST_SEASON + 3, -1))
        year = st.selectbox("Season", seasons, key="pred_season")
    schedule = data.get_schedule(year)
    schedule = schedule[schedule["RoundNumber"] > 0]
    next_rnd = _next_round(year, data.now_utc())
    rounds = list(schedule["RoundNumber"].astype(int))
    with c1:
        rnd = st.selectbox("Grand Prix", rounds,
                           index=rounds.index(next_rnd) if next_rnd in rounds else len(rounds) - 1,
                           format_func=lambda r: f"R{r} · {schedule.set_index('RoundNumber').loc[r, 'EventName']}",
                           key=f"pred_round_{year}")
    event = schedule[schedule["RoundNumber"] == rnd].iloc[0]
    sessions = _sessions(event)
    with c3:
        st.write("")
        if st.button("Refresh data", help="Re-download results; use after a session has finished."):
            for fn in (pdata.season_races, pdata.season_quali, pdata.season_sprints,
                       pdata.weekend_session, pdata._fresh_session, _prepared, _fits,
                       _grid_estimator, _quali_data, _quali_fit, _calibration_data,
                       _temperatures):
                fn.clear()
            st.rerun()

    try:
        with st.spinner("Collecting results from the last seasons (first time ~30 s)…"):
            races, hist, builder = _prepared(year, int(rnd))
            info = pdata.race_info(year, int(rnd))
            entries = _entries(event, sessions, year, int(rnd), races)
    except Exception as exc:
        st.error(f"Could not collect the data needed for predictions: {exc}")
        return
    if hist.empty or info is None:
        st.info("Not enough historical data for this race.")
        return

    _upgrades_panel(year, int(rnd), event)

    order = [t for t in ("Qualifying", "Race", "Sprint") if t in sessions]
    for tab, target in zip(st.tabs(order), order):
        with tab:
            if target == "Qualifying":
                _render_quali(event, sessions, year, int(rnd), builder, info, entries)
            else:
                _render_race(target, event, sessions, year, int(rnd), races, hist, builder,
                             info, entries)


def _render_race(target: str, event, sessions: dict, year: int, rnd: int, races: pd.DataFrame,
                 hist: pd.DataFrame, builder, info: dict, entries: pd.DataFrame) -> None:
    now = data.now_utc()
    fit_post, fit_pre, fit_sprint, lookup = _fits(year, rnd)
    if target == "Sprint":
        fit_post = fit_sprint
    circuit = info["circuitId"]
    quali_name = _quali_name(sessions, target)
    grid = _grid(sessions, target, year, rnd)
    official = _official_grid(year, rnd, event["EventName"], target, sessions) \
        if grid is not None else None
    if official is not None:  # penalties and pit-lane starts change the order
        grid = official["grid"]
    sq = _grid(sessions, "Sprint", year, rnd) if target == "Race" else None
    estimator = _grid_estimator(year, rnd) if sq is not None else None
    if estimator is None:
        sq = None
    fp_rank = _practice_rank(sessions, year, rnd) if sq is None else None
    start = sessions.get(target)
    fc = (pdata.forecast(info["lat"], info["lon"], start)
          if start is not None and now - pd.Timedelta(hours=3) < start < now + pd.Timedelta(days=15)
          else None)
    chaos = builder.chaos(circuit)
    last_race = hist.sort_values(["season", "round"]).iloc[-1]

    if quali_name:
        q_start = sessions[quali_name]
        if official is not None:
            notes = "".join(f"\n- {n}" for n in official["notes"])
            st.success(f"**Starting grid from the FIA {official['source']} starting grid**, "
                       f"so grid penalties and pit-lane starts are included. The *before* "
                       f"column shows the prediction without any grid.{notes}")
        elif grid is not None:
            st.success(f"**Starting grid from the {quali_name} results** (the FIA starting grid "
                       "with penalties isn't published yet; edit *Grid* under Adjustments if "
                       "you know of a penalty). The *before* column shows what the model "
                       "predicted without it.")
        elif q_start > now:
            st.info(f"**{quali_name} starts {q_start:%a %d %b %H:%M} UTC.** Until then the grid "
                    "is estimated. Once the results are final this switches to the real grid "
                    "automatically (re-checked every 5 minutes; reload the page to update).")
        else:
            st.warning(f"**{quali_name} is running or its results aren't final yet**, so the "
                       "grid is still estimated. Re-checked every 5 minutes; reload the page "
                       "to update.")

    # ---------------------------------------------------------------- inputs
    st.markdown("##### Inputs")
    k = st.columns(4)
    k[0].metric("Grid", "Known" if grid is not None else ("Estimated" if sq is not None else "Not yet"),
                "from qualifying" if grid is not None else
                ("from sprint quali + form" if sq is not None else
                 ("estimated: practice + form" if fp_rank is not None else "estimated: form")),
                delta_color="off", delta_arrow="off")
    k[1].metric("Rain forecast", f"{fc['rain_prob']:.0%}" if fc else "n/a",
                f"{fc['temp']:.0f}°C · {fc['rain_mm']:.1f} mm" if fc else "only ≤ 15 days ahead",
                delta_color="off", delta_arrow="off")
    k[2].metric("Circuit chaos", f"{chaos:.2f}×",
                "above average" if chaos > 1.05 else ("processional" if chaos < 0.95 else "average"),
                delta_color="off", delta_arrow="off",
                help="How much finishing positions differ from the grid at this circuit "
                     "(overtaking, safety cars, incidents) relative to the average circuit.")
    k[3].metric("Results up to", f"{int(last_race['season'])} R{int(last_race['round'])}",
                f"{len(hist['season'].unique())} seasons", delta_color="off",
                delta_arrow="off")

    w1, w2 = st.columns([2, 3])
    with w1:
        rain = st.slider("Rain probability during the session", 0.0, 1.0,
                         float(round(fc["rain_prob"], 2)) if fc else 0.0, 0.05,
                         key=f"rain_{rnd}_{target}",
                         help="Pre-filled from the Open-Meteo forecast; adjust if you know better.")
    with w2:
        use_wet = st.toggle("Use drivers' wet-weather record", value=rain >= 0.3,
                            key=f"wet_{rnd}_{target}",
                            help="Places gained on the teammate in wet vs. dry races over the "
                                 "last two seasons. First run downloads weather for ~45 races.")

    wet_gain = _wet_gain(hist, year) if (use_wet and rain > 0) else pd.Series(dtype=float)

    feats = pd.DataFrame([builder.features(d, t, circuit)
                          for d, t in zip(entries["driverId"], entries["constructorId"])])
    table = entries[["Driver", "Team"]].copy()
    table["Grid"] = table["Driver"].map(grid) if grid is not None else np.nan
    if grid is not None:  # anyone missing from the official grid starts at the back
        table["Grid"] = table["Grid"].fillna(len(table))
    table["Adjust (places)"] = 0.0
    table["Out"] = False
    with st.expander("Adjustments: grid penalties, upgrades, withdrawals", expanded=False):
        st.caption("**Grid**: edit for penalties or fill it in yourself before qualifying "
                   "(all drivers needed). **Adjust**: expected places gained (+) or lost (−), "
                   "e.g. for a big upgrade package or a known problem. **Out**: driver won't start.")
        edited = st.data_editor(
            table, hide_index=True, width="stretch",
            key=f"pred_edit_{rnd}_{target}_{grid is not None}",
            disabled=["Driver", "Team"],
            column_config={"Grid": st.column_config.NumberColumn(min_value=1, max_value=22, step=1),
                           "Adjust (places)": st.column_config.NumberColumn(min_value=-10.0, max_value=10.0, step=0.5)})

    keep = ~edited["Out"].to_numpy()
    shift = edited["Adjust (places)"].to_numpy(float) \
        + entries["driverId"].map(wet_gain).fillna(0.0).to_numpy() * rain
    p_dnf = feats["p_dnf"].to_numpy()[keep] * (1 + RAIN_DNF * rain)
    calib = _temperatures(year, rnd)["sprint" if target == "Sprint" else "race"]
    randomness = chaos * (1 + RAIN_RANDOMNESS * rain) * calib["temperature"]
    if target == "Sprint":
        p_dnf = p_dnf * SPRINT_DNF
    drivers = list(entries["Driver"][keep])
    teams_kept = entries["constructorId"][keep].to_numpy()

    qgaps = _quali_gaps(sessions, target, year, rnd) if grid is not None else None

    def run(how: str):
        """Simulate with the known grid ("grid"), a Sprint-Qualifying-based estimate
        ("estimate") or without any grid ("pre"). Returns (fit, features, positions)."""
        X = feats.copy()
        if how == "grid":
            g = edited["Grid"].to_numpy(float) - shift
            qg = entries["Driver"].map(qgaps).to_numpy(float) if qgaps is not None else None
            X = model.with_grid(X, g, chaos, qgap=qg, qgap_lookup=lookup)
            if qg is not None:  # drivers without a time: typical gap for their grid slot
                X["qgap_now"] = np.where(np.isnan(qg), lookup(X["grid"].to_numpy()), qg)
            return fit_post, X, model.simulate_rank(fit_post.strength(X)[keep], p_dnf, randomness,
                                                    teams=teams_kept)
        if how == "estimate":
            # Sprint Qualifying + recent form -> expected qualifying position; its
            # uncertainty is simulated (see model.simulate_uncertain_grid)
            w, est_sd = estimator
            sq_pos = entries["Driver"].map(sq).fillna(21).to_numpy(float)
            est = w[0] + w[1] * sq_pos + w[2] * X["quali_form"].to_numpy() - shift
            X = model.with_grid(X, est, chaos, qgap_lookup=lookup)
            return fit_post, X, model.simulate_uncertain_grid(
                fit_post, X[keep], est[keep], est_sd, p_dnf, randomness, chaos=chaos,
                lookup=lookup, teams=teams_kept)
        if fp_rank is not None:
            fp = entries["Driver"].map(fp_rank).fillna(X["quali_form"]).to_numpy()
            X["quali_form"] = (1 - FP_BLEND) * X["quali_form"] + FP_BLEND * fp
        X = _apply_shift(X, shift)
        return fit_pre, X, model.simulate_rank(fit_pre.strength(X)[keep], p_dnf, randomness,
                                               teams=teams_kept)

    pre_how = "estimate" if sq is not None else "pre"
    how = "grid" if edited["Grid"].notna().all() else pre_how
    fit, X, pos = run(how)
    mode = {"grid": "after qualifying" if grid is not None else "with the grid you entered",
            "estimate": "before qualifying, grid estimated from Sprint Qualifying",
            "pre": "before qualifying"}[how]

    recal, evidence = _betting_calibration(year, rnd, "race")
    result = model.summarise(pos, drivers)
    result["Win"] = calibration.recalibrate_win(result["Win"].to_numpy(), recal)
    result.insert(1, "Team", list(entries["Team"][keep]))
    if how == "grid":  # show the real starting position, not the adjusted model input
        result.insert(2, "Grid", edited["Grid"].to_numpy()[keep].astype(int))
    elif how == "estimate":
        result.insert(2, "Est. grid", np.round(X["grid"].to_numpy()[keep], 1))
    if how == "grid" and grid is not None:
        # what qualifying changed: the same model without the real grid
        before = model.summarise(run(pre_how)[2], drivers)
        before["Win"] = calibration.recalibrate_win(before["Win"].to_numpy(), recal)
        before = before.set_index("Driver")
        result.insert(4, "Win before quali", result["Driver"].map(before["Win"]))
        result.insert(5, "Change", (result["Win"] - result["Win before quali"]) * 100)
    result["Fair odds (win)"] = (1 / result["Win"]).where(result["Win"] > 0.001)
    result = result.sort_values("Win", ascending=False).reset_index(drop=True)

    if start is not None and start > now and rnd == _next_round(year, now):
        calibration.log_prediction(year, rnd, event["EventName"], target, mode, result,
                                   "Win", "Podium")
    actual = _results(sessions, target, year, rnd, races)
    if actual is not None:
        result = _with_result(result, actual)

    # --------------------------------------------------------------- output
    st.markdown(f"#### {event['EventName']} · {target}")
    st.caption(f"Prediction {mode} · {len(pos):,} simulated races · trained on {fit.n_races} races"
               + _calib_note(calib, "sprints" if target == "Sprint" else "races"))
    if actual is not None:
        _result_check(result, actual, "Win", "Podium", "winner", entries, key=target)
    if "Change" in result:
        movers = result.sort_values("Change")
        down, up = movers.head(2), movers.tail(2).iloc[::-1]
        st.caption("Biggest changes after qualifying: "
                   + ", ".join(f"{r.Driver} {r.Change:+.0f} pp" for r in up.itertuples())
                   + " · " + ", ".join(f"{r.Driver} {r.Change:+.0f} pp" for r in down.itertuples()))
    left, right = st.columns([3, 2])
    with left:
        pct_cols = MARKETS + ["DNF", "Win before quali"]
        st.dataframe(_as_pct(result, pct_cols), hide_index=True, width="stretch",
                     height=ui.table_height(len(result)), row_height=ui.ROW_HEIGHT,
                     column_config={**_pct(result, pct_cols),
                                    "Change": st.column_config.NumberColumn(
                                        "Win Δ (pp)", format="%+.0f",
                                        help="Change in win probability, percentage points"),
                                    "Expected pos.": st.column_config.NumberColumn(format="%.1f"),
                                    "Fair odds (win)": st.column_config.NumberColumn(format="%.1f")})
        st.caption("**Fair odds** = 1 / probability, i.e. without bookmaker margin. "
                   "Bookmaker odds include a margin, so they are always lower than fair odds "
                   "when the bookmaker agrees with the model.")
    with right:
        top = result.head(10).iloc[::-1]
        color_of = dict(zip(entries["Driver"], _colors(entries)))
        fig = go.Figure(go.Bar(x=top["Win"], y=top["Driver"], orientation="h", name="Now",
                               marker_color=[color_of.get(d, "#888") for d in top["Driver"]],
                               text=[f"{p:.0%}" for p in top["Win"]], textposition="outside",
                               cliponaxis=False, hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        if "Win before quali" in top:
            fig.add_trace(go.Scatter(x=top["Win before quali"], y=top["Driver"], mode="markers",
                                     name="Before quali", marker=dict(symbol="line-ns-open",
                                                                      size=18, color="#888",
                                                                      line=dict(width=2)),
                                     hovertemplate="%{y} before quali: %{x:.1%}<extra></extra>"))
            fig.update_layout(legend=dict(orientation="h", y=-0.12))
        fig.update_layout(title=dict(text="Win probability", font=dict(size=14)), height=380,
                          margin=dict(l=10, r=40, t=40, b=10), xaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig, width="stretch", key=f"win_chart_{target}")
        teams = (result.groupby("Team")["Win"].sum().sort_values(ascending=False).reset_index())
        teams["Fair odds"] = (1 / teams["Win"]).where(teams["Win"] > 0.001)
        st.markdown("**Winning team**")
        st.dataframe(_as_pct(teams.head(6), ["Win"]), hide_index=True, width="stretch",
                     column_config={**_pct(teams, ["Win"]),
                                    "Fair odds": st.column_config.NumberColumn(format="%.1f")})

    _head_to_head(pos, drivers, entries[keep].reset_index(drop=True), key=target,
                  record=_season_record(races, year, rnd) if target == "Race" else None)
    _value_check(result, MARKETS, key=target, entries=entries, evidence=evidence)
    _explain(fit, X[keep], feats[keep], entries[keep], wet_gain, rain, chaos, target, year, rnd,
             sq)


def _head_to_head(pos: np.ndarray, drivers: list[str], entries: pd.DataFrame, key: str,
                  record: dict | None = None) -> None:
    st.markdown("##### Head-to-head")
    a_col, b_col, res_col = st.columns([1, 1, 3])
    with a_col:
        a = st.selectbox("Driver A", drivers, index=0, key=f"h2h_a_{key}")
    with b_col:
        b = st.selectbox("Driver B", drivers, index=min(1, len(drivers) - 1), key=f"h2h_b_{key}")
    with res_col:
        if a != b:
            pa, pb, both = model.head_to_head(pos, drivers.index(a), drivers.index(b))
            st.markdown(f"**{a}** ahead: **{pa:.0%}** (fair odds {1 / max(pa, 1e-3):.2f}) · "
                        f"**{b}** ahead: **{pb:.0%}** (fair odds {1 / max(pb, 1e-3):.2f})  \n"
                        f"Both retire: {both:.0%}. A driver who retires counts as behind one who "
                        "finishes; check your bookmaker's rules for this case.")
    rows = []
    race = (pos == 99).any() or record is not None
    for team, g in entries.groupby("Team"):
        if len(g) != 2:
            continue
        i, j = g.index
        pi, pj, _ = model.head_to_head(pos, i, j)
        row = {"Team": team, "Driver 1": drivers[i], "P(1 ahead)": pi,
               "Driver 2": drivers[j], "P(2 ahead)": pj}
        if race:
            fin = (pos[:, i] != 99) & (pos[:, j] != 99)
            row.update({"Both finish": fin.mean(),
                        "Both in points": ((pos[:, i] <= 10) & (pos[:, j] <= 10)).mean(),
                        "Both retire": ((pos[:, i] == 99) & (pos[:, j] == 99)).mean()})
            row["Fair odds (both finish)"] = 1 / max(row["Both finish"], 1e-3)
            cid = g["constructorId"].iloc[0] if "constructorId" in g else None
            if record is not None and cid in record:
                r = record[cid]
                row["This season"] = (f"both finished {r['both_finish']} of {r['n']}, "
                                      f"both retired {r['both_dnf']}")
        rows.append(row)
    if rows:
        with st.expander("Teammate battles", expanded=False):
            tm = pd.DataFrame(rows)
            pcts = ["P(1 ahead)", "P(2 ahead)", "Both finish", "Both in points", "Both retire"]
            st.dataframe(_as_pct(tm, pcts), hide_index=True, width="stretch",
                         height=ui.table_height(len(tm)), row_height=ui.ROW_HEIGHT,
                         column_config={**_pct(tm, pcts), "Fair odds (both finish)":
                                        st.column_config.NumberColumn(format="%.2f")})
            if race:
                st.caption("*Both finish* = both cars classified (completing 90% of the race). "
                           "Teammates' retirements are modelled as linked, because team-wide "
                           "problems and collisions make a double retirement more likely than "
                           "chance. The model's retirement rates are long-run averages, so a team "
                           "in a reliability crisis can be worse than the model says: compare "
                           "with *This season*.")


def _season_record(races: pd.DataFrame, year: int, rnd: int) -> dict:
    """Per team, this season before `rnd`: races where both cars finished / both retired."""
    r = races[(races["season"] == year) & (races["round"] < rnd)]
    r = r.assign(dnf=~r["positionText"].astype(str).str.isdigit())
    out = {}
    for cid, g in r.groupby("constructorId"):
        per = g.groupby("round")["dnf"].agg(["sum", "size"])
        per = per[per["size"] == 2]
        out[cid] = dict(n=len(per), both_finish=int((per["sum"] == 0).sum()),
                        both_dnf=int((per["sum"] == 2).sum()))
    return out


_ODDS = re.compile(r"(?<![\d.,])(\d{1,4}[.,]\d{1,2})(?![\d.,])")


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def parse_odds(text: str, entries: pd.DataFrame) -> dict[str, float]:
    """Read 'driver name ... decimal odds' pairs from text copied off a bookmaker page.

    Works line by line: a line naming a driver (surname, full name or 3-letter
    code) starts that driver; the next decimal number (2.50 or 2,50) is their odds.
    """
    keys = {}
    for abbr, last in zip(entries["Driver"], entries.get("LastName", entries["Driver"])):
        if isinstance(last, str) and last:
            keys[_fold(last)] = abbr
    out, current = {}, None
    for line in text.splitlines():
        folded = _fold(line)
        hits = [(len(k), a) for k, a in keys.items() if re.search(rf"\b{re.escape(k)}\b", folded)]
        hits += [(3, a) for a in entries["Driver"] if re.search(rf"\b{a}\b", line)]
        rest = line
        if hits:
            current = max(hits)[1]
            for k in keys:  # don't read digits inside the name part
                rest = re.sub(re.escape(k), " ", _fold(rest))
        numbers = [float(n.replace(",", ".")) for n in _ODDS.findall(rest)]
        numbers = [n for n in numbers if 1.01 <= n <= 2000]
        if current and numbers and current not in out:
            out[current] = numbers[0]
            current = None
    return out


_EVIDENCE_MARKET = {"Win": "first", "Pole": "first", "Podium": "top3", "Top 3": "top3"}


def _bet_row(p: float, odds: float, ev: pd.DataFrame | None) -> dict:
    """Edge with a realistic range: how often this kind of probability came true before."""
    out = {"Hits ~1 in": round(1 / p) if p > 0 else np.nan, "Edge": p * odds - 1,
           "Cautious edge": np.nan, "Past hit rate": "no backtest data", "Verdict": ""}
    band = None if ev is None or ev.empty else ev[(ev["left"] <= p) & (p < ev["right"])]
    if band is not None and len(band):
        b = band.iloc[0]
        # scale this driver's probability by how the band actually performed
        p_lo = min(1.0, p * b["lo"] / b["predicted"]) if b["predicted"] > 0 else 0.0
        out["Cautious edge"] = p_lo * odds - 1
        out["Past hit rate"] = (f"{b['lo']:.1%}–{b['hi']:.1%} when the model said "
                                f"{b['left']:.0%}–{min(b['right'], 1):.0%} "
                                f"({int(b['hits'])} of {int(b['n'])})")
    if out["Edge"] <= 0:
        out["Verdict"] = "No value"
    elif out["Cautious edge"] > 0:
        out["Verdict"] = "Value"
    else:
        out["Verdict"] = "Uncertain"
    return out


def _value_check(result: pd.DataFrame, markets: list[str], key: str,
                 entries: pd.DataFrame | None = None,
                 evidence: dict[str, pd.DataFrame] | None = None) -> None:
    st.markdown("##### Compare with bookmaker odds")
    st.caption("**Edge** is the expected return per €1 *if* the model's probability is right. "
               "**Cautious edge** uses the low end of how often the model's predictions in the "
               "same probability range actually came true in the backtest. A bet is marked "
               "**Value** only when even the cautious edge is positive. **Hits ~1 in** shows "
               "how rarely it comes in: a fair 1-in-50 bet loses 49 times out of 50, so a "
               "positive edge on a long shot only pays off over many bets.")
    parsed: dict[str, float] = {}
    paste_market = markets[0]
    if entries is not None:
        with st.expander("Paste odds", expanded=False):
            st.caption("On the bookmaker's page, select the list of drivers with their odds for "
                       "one market, copy it (Ctrl/Cmd+C) and paste it here. Names and decimal "
                       "odds (2.50 or 2,50) are recognised automatically.")
            c1, c2 = st.columns([1, 3])
            with c1:
                paste_market = st.selectbox("Market", markets, key=f"paste_market_{key}")
            with c2:
                text = st.text_area("Pasted odds", height=120, key=f"paste_{key}",
                                    placeholder="Max Verstappen\n2.50\nCharles Leclerc\n4,75\n…")
            if text.strip():
                parsed = parse_odds(text, entries)
                missing = [d for d in result["Driver"] if d not in parsed]
                show_missing = parsed and missing and len(missing) <= len(result) // 2
                st.caption(f"Recognised {len(parsed)} drivers"
                           + (f"; not found: {', '.join(missing)}" if show_missing else "")
                           + ("" if parsed else " — check that names and odds were copied"))
    if parsed:
        start = pd.DataFrame({"Driver": list(parsed), "Market": paste_market,
                              "Odds": list(parsed.values())})
    else:
        start = pd.DataFrame({"Driver": pd.Series([None] * 3, dtype="object"),
                              "Market": pd.Series([markets[0]] * 3, dtype="object"),
                              "Odds": pd.Series([np.nan] * 3, dtype=float)})
    bets = st.data_editor(
        start, num_rows="dynamic", hide_index=True, width="stretch",
        key=f"pred_odds_{key}_{hash(tuple(parsed.items()))}",
        column_config={"Driver": st.column_config.SelectboxColumn(options=list(result["Driver"])),
                       "Market": st.column_config.SelectboxColumn(options=markets),
                       "Odds": st.column_config.NumberColumn(min_value=1.01, step=0.05, format="%.2f")})
    bets = bets.dropna(subset=["Driver", "Market", "Odds"])
    if bets.empty:
        return
    probs = result.set_index("Driver")
    bets = bets.assign(
        **{"Model prob.": [probs.at[d, m] for d, m in zip(bets["Driver"], bets["Market"])]})
    bets["Implied prob."] = 1 / bets["Odds"]
    extra = [_bet_row(p, o, (evidence or {}).get(_EVIDENCE_MARKET.get(m, "")))
             for p, o, m in zip(bets["Model prob."], bets["Odds"], bets["Market"])]
    bets = pd.concat([bets.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    order = {"Value": 0, "Uncertain": 1, "No value": 2}
    bets = bets.sort_values(["Verdict", "Edge"], key=lambda c: c.map(order) if c.name == "Verdict"
                            else -c).reset_index(drop=True)
    margin = bets.groupby("Market")["Implied prob."].sum()
    colour = {"Value": "color: #2fb380; font-weight: 600", "Uncertain": "color: #e0a526",
              "No value": "color: #d73027"}
    st.dataframe(
        bets[["Driver", "Market", "Odds", "Model prob.", "Hits ~1 in", "Implied prob.", "Edge",
              "Cautious edge", "Verdict", "Past hit rate"]]
        .style.map(lambda v: colour.get(v, ""), subset=["Verdict"])
        .format({"Model prob.": "{:.1%}", "Implied prob.": "{:.1%}", "Edge": "{:+.0%}",
                 "Cautious edge": lambda v: "" if pd.isna(v) else f"{v:+.0%}",
                 "Hits ~1 in": lambda v: "" if pd.isna(v) else f"{v:,.0f}", "Odds": "{:.2f}"}),
        hide_index=True, width="stretch", height=ui.table_height(len(bets)),
        row_height=ui.ROW_HEIGHT)
    full = {"Win": 1, "Pole": 1, "Podium": 3, "Top 3": 3, "Front row": 2}
    for m, total in margin.items():
        n = int((bets["Market"] == m).sum())
        if m in full and n >= len(result) - 2:
            st.caption(f"{m}: the bookmaker's probabilities add up to {total:.0%} instead of "
                       f"{full[m]:.0%}, so its margin is about {total / full[m] - 1:.0%}.")


def _explain(fit, X: pd.DataFrame, feats: pd.DataFrame, entries: pd.DataFrame,
             wet_gain: pd.Series, rain: float, chaos: float, target: str,
             year: int, rnd: int, sq: pd.Series | None) -> None:
    with st.expander("Model inputs"):
        st.markdown("**Weights the model learned** (fitted on past finishing orders; bigger = "
                    "matters more; features are standardised so weights are comparable)")
        st.dataframe(fit.weight_table().round(3), hide_index=True, width="stretch")
        shown = pd.DataFrame({
            "Driver": entries["Driver"].values,
            "Grid used": X["grid"].round(1).values if "grid" in X else np.nan,
            "Recent finishes": feats["form"].round(1).values,
            "Recent quali": X["quali_form"].round(1).values,
            "Quali gap %": X["quali_gap"].round(2).values,
            "Driver edge %": feats["driver_edge"].round(2).values,
            "Race pace %": X["race_pace"].round(2).values,
            "Team quali pace": feats["team_pace"].round(1).values,
            "Team trend /race": feats["team_trend"].round(2).values,
            "Circuit history": feats["circuit_aff"].round(2).values,
            "Races here": feats["n_circuit"].values,
            "Retire chance": feats["p_dnf"].values,
            "Wet gain": entries["driverId"].map(wet_gain).round(2).values,
            "Sprint quali": entries["Driver"].map(sq).values if sq is not None else np.nan,
        })
        st.markdown("**Inputs per driver** (positions and % gaps; lower = better, except "
                    "*Driver edge*: % faster than the teammate in qualifying, i.e. driver skill "
                    "separate from the car, and *Wet gain*: places gained on the teammate in wet "
                    "races vs. dry. *Team trend* < 0 = improving. *Circuit history* < 0 = usually "
                    "does better here than elsewhere.)")
        st.dataframe(shown, hide_index=True, width="stretch",
                     column_config={"Retire chance": st.column_config.NumberColumn(format="percent")})
        st.markdown(
            f"**Adjustments applied on top**: circuit randomness ×{chaos:.2f}; rain {rain:.0%} "
            f"→ randomness +{RAIN_RANDOMNESS * rain:.0%}, retirements +{RAIN_DNF * rain:.0%}"
            + (f", wet record shifts drivers by up to "
               f"{(wet_gain.abs().max() if len(wet_gain) else 0) * rain:.1f} places"
               f" (based on {st.session_state.get('n_wet_races', 0)} wet races)" if len(wet_gain) else "")
            + (f"; sprint retirements ×{SPRINT_DNF}" if target == "Sprint" else "")
            + ". Rain factors are rules of thumb; the overall confidence is learned from past "
              "results (see the Track record page).")
        if sq is not None:
            st.markdown("**Grid estimate**: no qualifying yet, so each driver's qualifying "
                        "position is estimated from this weekend's Sprint Qualifying and recent "
                        "qualifying form (weights fitted on past sprint weekends). The estimate's "
                        "uncertainty (± a few places) is included in the simulation.")
        st.markdown(
            "**Not in the model** (use *Adjustments* for them): upgrades — the FIA upgrade lists "
            "were tested and made predictions slightly worse, as their effect already shows "
            "in *Team trend* and practice pace — tyre "
            "degradation and strategy, grid penalties announced after qualifying, team orders, "
            "driver illness/injury, and specific safety-car probabilities.")

    _accuracy_panel(year, rnd, ("sprint",) if target == "Sprint" else ("race", "race_pre"))


# --------------------------------------------------------------- qualifying
@st.cache_resource(show_spinner=False, max_entries=4)
def _quali_data(year: int, rnd: int) -> pd.DataFrame:
    """Past qualifying entries with pre-weekend features and weekend signals."""
    _, quali = pdata.history(year, start=pdata.DATA_START)
    _, hist, _ = _prepared(year, rnd)
    sprints = pdata.sprints_range(pdata.DATA_START, year)
    sprints = sprints[(sprints["season"] < year) | (sprints["round"] < rnd)]
    sprint_weekends = set(map(tuple, sprints[["season", "round"]].drop_duplicates().values))
    first = year - model.CONFIG["quali"]["train_seasons"] + 1
    weekends = hist[hist["season"] >= first][["season", "round"]].drop_duplicates()
    fp = {}
    text = "Loading practice pace of past weekends (first time ~2 min)…"
    progress = st.progress(0.0, text=text)
    for i, (s, r) in enumerate(weekends.itertuples(index=False)):
        if (s, r) not in sprint_weekends:
            pace = pdata.practice_pace(int(s), int(r), "Practice 3")
            if pace is not None:
                fp[(s, r)] = pace
        progress.progress((i + 1) / len(weekends), text=text)
    progress.empty()
    return model.quali_rows(hist, quali, sprints, fp)


@st.cache_resource(show_spinner=False, max_entries=12)
def _quali_fit(year: int, rnd: int, variant: str):
    return model.fit_family(_quali_data(year, rnd), "quali", year, rnd,
                            features=model.quali_features(variant))


def _weekend_signal(sessions: dict, year: int, rnd: int) -> tuple[str, object, str]:
    """(model variant, abbreviation-indexed signal, source session) for this weekend:
    Sprint Qualifying positions (Series) or practice rank and gap (DataFrame)."""
    now = data.now_utc()
    for name in ("Sprint Qualifying", "Sprint Shootout"):
        if name in sessions and sessions[name] < now:
            sq = _grid(sessions, "Sprint", year, rnd)
            if sq is not None:
                return "sprint quali", sq, name
    for name in PRACTICE_ORDER:
        if name in sessions and sessions[name] + pd.Timedelta(hours=1) < now:
            try:
                laps = pdata.load_weekend_session(year, rnd, name, True, sessions[name]).laps
            except Exception:
                continue
            pace = pdata.laps_pace(laps)
            if pace is not None:
                return "practice", pace, name
    return "form", None, "no session yet"


def _render_quali(event, sessions: dict, year: int, rnd: int, builder, info: dict,
                  entries: pd.DataFrame) -> None:
    now = data.now_utc()
    try:
        with st.spinner("Preparing qualifying history…"):
            rows = _quali_data(year, rnd)
    except Exception as exc:
        st.error(f"Could not collect qualifying history: {exc}")
        return
    variant, signal, source = _weekend_signal(sessions, year, rnd)
    start = sessions["Qualifying"]
    fc = (pdata.forecast(info["lat"], info["lon"], start, hours=1)
          if now - pd.Timedelta(hours=2) < start < now + pd.Timedelta(days=15) else None)

    st.markdown("##### Inputs")
    k = st.columns(3)
    short = {"Sprint Qualifying": "Sprint Quali", "Sprint Shootout": "Shootout",
             "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3"}
    k[0].metric("Weekend pace from", short.get(source, source) if variant != "form" else "Form only",
                "combined with recent form" if variant != "form" else "no session yet",
                delta_color="off", delta_arrow="off")
    k[1].metric("Rain forecast", f"{fc['rain_prob']:.0%}" if fc else "n/a",
                f"{fc['temp']:.0f}°C · {fc['rain_mm']:.1f} mm" if fc else "only ≤ 15 days ahead",
                delta_color="off", delta_arrow="off")
    n_fp = rows["fp"].notna().groupby([rows["season"], rows["round"]]).any().sum()
    k[2].metric("Trained on", f"{rows.groupby(['season', 'round']).ngroups} sessions",
                f"{n_fp} with practice data", delta_color="off", delta_arrow="off")
    if variant == "practice" and source != "Practice 3":
        st.caption(f"Note: the model learned practice pace from Practice 3; {source} is used "
                   "as a stand-in, which is less representative (different programmes and "
                   "fuel loads).")
    rain = st.slider("Rain probability during qualifying", 0.0, 1.0,
                     float(round(fc["rain_prob"], 2)) if fc else 0.0, 0.05, key=f"quali_rain_{rnd}",
                     help="Pre-filled from the Open-Meteo forecast. Rain makes the order "
                          "more random.")

    table = entries[["Driver", "Team"]].copy()
    table["Adjust (places)"] = 0.0
    table["Out"] = False
    with st.expander("Adjustments: upgrades, problems, withdrawals", expanded=False):
        st.caption("**Adjust**: expected places gained (+) or lost (−), e.g. for a new "
                   "upgrade or a car problem. **Out**: driver won't take part.")
        edited = st.data_editor(
            table, hide_index=True, width="stretch", key=f"quali_edit_{rnd}",
            disabled=["Driver", "Team"],
            column_config={"Adjust (places)": st.column_config.NumberColumn(
                min_value=-10.0, max_value=10.0, step=0.5)})
    keep = ~edited["Out"].to_numpy()
    shift = edited["Adjust (places)"].to_numpy(float)

    feats = model.quali_features(variant)
    X = pd.DataFrame([builder.features(d, t, info["circuitId"])
                      for d, t in zip(entries["driverId"], entries["constructorId"])])
    weekend_col = {"sprint quali": "sq",
                   "practice": model.CONFIG["quali"]["practice"]}.get(variant)
    X["sq"] = X["fp"] = X["fp_gap"] = np.nan
    if variant == "sprint quali":
        X["sq"] = entries["Driver"].map(signal).to_numpy()
    elif variant == "practice":
        X["fp"] = entries["Driver"].map(signal["rank"]).to_numpy()
        X["fp_gap"] = entries["Driver"].map(signal["gap"]).to_numpy()
    X = model.fill_missing(X, feats)
    X = _apply_shift(X, shift, extra_rank=("sq", "fp"))
    fit = _quali_fit(year, rnd, variant)
    drivers = list(entries["Driver"][keep])
    calib = _temperatures(year, rnd)["quali"]
    pos = model.simulate_rank(fit.strength(X)[keep], np.zeros(int(keep.sum())),
                              randomness=(1 + RAIN_RANDOMNESS * rain) * calib["temperature"])
    result = pd.DataFrame({"Driver": drivers, "Team": list(entries["Team"][keep]),
                           "Pole": (pos == 1).mean(0), "Front row": (pos <= 2).mean(0),
                           "Top 3": (pos <= 3).mean(0), "Q3 (top 10)": (pos <= 10).mean(0),
                           "Q2 (top 15)": (pos <= 15).mean(0), "Expected pos.": pos.mean(0)})
    result["Fair odds (pole)"] = (1 / result["Pole"]).where(result["Pole"] > 0.001)
    result = result.sort_values("Pole", ascending=False).reset_index(drop=True)
    if start > now and rnd == _next_round(year, now):
        calibration.log_prediction(year, rnd, event["EventName"], "Qualifying",
                                   f"form + {source}" if variant != "form" else "form only",
                                   result, "Pole", "Top 3")
    actual = _results(sessions, "Qualifying", year, rnd)
    if actual is not None:
        result = _with_result(result, actual)

    st.markdown(f"#### {event['EventName']} · Qualifying")
    st.caption(f"Based on recent form{' and ' + source if variant != 'form' else ''} · "
               f"{len(pos):,} simulated sessions · trained on {fit.n_races} sessions"
               + _calib_note(calib, "qualifying sessions"))
    if actual is not None:
        _result_check(result, actual, "Pole", "Top 3", "pole", entries, key="quali")
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(_as_pct(result, QUALI_MARKETS), hide_index=True, width="stretch",
                     height=ui.table_height(len(result)), row_height=ui.ROW_HEIGHT,
                     column_config={**_pct(result, QUALI_MARKETS),
                                    "Expected pos.": st.column_config.NumberColumn(format="%.1f"),
                                    "Fair odds (pole)": st.column_config.NumberColumn(format="%.1f")})
    with right:
        top = result.head(10).iloc[::-1]
        color_of = dict(zip(entries["Driver"], _colors(entries)))
        fig = go.Figure(go.Bar(x=top["Pole"], y=top["Driver"], orientation="h",
                               marker_color=[color_of.get(d, "#888") for d in top["Driver"]],
                               text=[f"{p:.0%}" for p in top["Pole"]], textposition="outside",
                               cliponaxis=False, hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        fig.update_layout(title=dict(text="Pole probability", font=dict(size=14)), height=360,
                          margin=dict(l=10, r=40, t=40, b=10), xaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig, width="stretch", key="pole_chart")
        teams = (pd.DataFrame({"Team": entries["Team"][keep].values, "Pole": (pos == 1).mean(0)})
                 .groupby("Team")["Pole"].sum().sort_values(ascending=False).reset_index())
        st.markdown("**Team on pole**")
        st.dataframe(_as_pct(teams.head(6), ["Pole"]), hide_index=True, width="stretch",
                     column_config=_pct(teams, ["Pole"]))

    _head_to_head(pos, drivers, entries[keep].reset_index(drop=True), key="quali")
    _value_check(result, QUALI_MARKETS, key="quali", entries=entries,
                 evidence=_betting_calibration(year, rnd, "quali")[1])

    with st.expander("Model inputs"):
        st.markdown("**Weights the model learned** (fitted on past qualifying orders)")
        st.dataframe(fit.weight_table().round(3), hide_index=True, width="stretch")
        shown = pd.DataFrame({"Driver": entries["Driver"].values,
                              "Recent quali": X["quali_form"].round(1).values,
                              "Recent finishes": X["form"].round(1).values,
                              "Team quali pace": X["team_pace"].round(1).values,
                              "Team trend /race": X["team_trend"].round(2).values})
        if weekend_col:
            shown[source] = X[weekend_col].round(1).values
        st.dataframe(shown[keep], hide_index=True, width="stretch")
        st.markdown(f"Rain {rain:.0%} → order randomness +{RAIN_RANDOMNESS * rain:.0%} "
                    "(rule of thumb). **Not in the model**: track evolution, red flags, "
                    "traffic, deleted laps, engine modes, set-up changes after practice.")

    _accuracy_panel(year, rnd, ("quali",))


def _event_upgrades(year: int, event_name: str) -> pd.DataFrame | None:
    """Declared upgrades of one event, or None if the FIA document isn't downloaded yet."""
    done = upgrades.events_done()
    names = list(done.loc[done["season"] == year, "event"])
    match = upgrades.get_close_matches(upgrades._norm(event_name),
                                       [upgrades._norm(n) for n in names], n=1, cutoff=0.5)
    if not match:
        return None
    fia_name = names[[upgrades._norm(n) for n in names].index(match[0])]
    items = upgrades.load_items()
    return items[(items["season"] == year) & (items["event"] == fia_name)]


def _upgrades_panel(year: int, rnd: int, event: pd.Series) -> None:
    with st.expander("Declared upgrades (FIA car presentation)", expanded=False):
        items = _event_upgrades(year, event["EventName"])
        if items is None:
            st.caption("Teams declare every new part to the FIA before each Grand Prix "
                       "(published on the Thursday/Friday of the race weekend).")
            if st.button("Download from fia.com (~30 s)", key=f"fetch_upg_{year}_{rnd}"):
                with st.spinner("Downloading the FIA car presentation document…"):
                    status = upgrades.fetch_current(year, event["EventName"], log=lambda m: None)
                _event_upgrades.clear()
                if status == "no document":
                    st.info("The FIA hasn't published it for this event yet.")
                else:
                    st.rerun()
            return
        if items.empty:
            st.info("No team declared any upgrades for this event.")
            return
        summary = (items.assign(n=1).pivot_table(index="team", columns="reason", values="n",
                                                   aggfunc="sum", fill_value=0))
        summary["Total"] = summary.sum(axis=1)
        summary = summary.sort_values("Total", ascending=False).reset_index()
        st.dataframe(summary.rename(columns={"team": "Team"}), hide_index=True, width="stretch")
        st.caption("**Performance** parts aim to make the car faster; **Circuit specific** parts "
                   "suit this track (cooling, downforce level); **Reliability** fixes problems.")
        st.dataframe(items[["team", "component", "reason", "change"]].rename(columns={
            "team": "Team", "component": "Part", "reason": "Reason", "change": "What changed"}),
            hide_index=True, width="stretch")


# ------------------------------------------------------------ actual results
_STATUS_CODES = {"R": "DNF", "N": "DNF", "D": "DSQ", "E": "DSQ", "W": "DNS", "F": "DNS"}


def _results(sessions: dict, name: str, year: int, rnd: int,
             races: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Official result of a finished session: Driver, Pos, Result label, classified.

    Uses FastF1 (available minutes after the session); for a race whose timing
    data can't be loaded it falls back to the Ergast results. None while the
    session hasn't finished.
    """
    start = sessions.get(name)
    if start is None or start > data.now_utc():
        return None
    res, laps = None, None
    try:
        session = pdata.load_weekend_session(year, rnd, name, True, start)
        if pdata.is_settled(start) or pdata.session_finished(session):
            res, laps = session.results, session.laps
    except Exception:
        pass
    if res is not None and res["Position"].notna().sum() >= 10:
        codes = res["ClassifiedPosition"].astype(str).str.strip() if "ClassifiedPosition" in res \
            else pd.Series("", index=res.index)
        labels = [_STATUS_CODES.get(c) or (f"P{int(p)}" if pd.notna(p) else "No time")
                  for p, c in zip(res["Position"], codes)]
        if name in ("Race", "Sprint") and not set(labels) & {"DNF", "DSQ", "DNS"} and len(laps):
            # classification not published yet (timing data only): a car that covered
            # less than 90% of the winner's distance is not classified
            done = laps.groupby("Driver")["LapNumber"].max()
            labels = ["DNF" if done.get(d, 0) < 0.9 * done.max() else lab
                      for d, lab in zip(res["Abbreviation"], labels)]
        out = pd.DataFrame({"Driver": res["Abbreviation"].values, "Pos": res["Position"].values,
                            "Result": labels})
    elif name == "Race" and races is not None:
        done = races[(races["season"] == year) & (races["round"] == rnd)]
        if done.empty:
            return None
        out = pd.DataFrame({"Driver": done["driverCode"].values, "Pos": done["position"].values,
                            "Result": [f"P{t}" if str(t).isdigit() else _STATUS_CODES.get(t, t)
                                       for t in done["positionText"]]})
    else:
        return None
    out["classified"] = out["Result"].str.startswith("P")
    return out


@st.cache_resource(show_spinner=False, max_entries=4)
def _temperatures(year: int, rnd: int) -> dict:
    """Confidence correction per session type, learned from past predictions vs results."""
    return _calibration_data(year, rnd)[0]


@st.cache_resource(show_spinner=False, max_entries=4)
def _calibration_data(year: int, rnd: int):
    _, hist, _ = _prepared(year, rnd)
    D = dict(hist=hist, qrows=_quali_data(year, rnd),
             sprints=pdata.sprints_range(pdata.DATA_START, year))
    text = "Re-predicting past events with only earlier data (first time ~1 min)…"
    bar = st.progress(0.0, text=text)
    events = calibration.walk_forward_events(D, (year, rnd),
                                             progress=lambda f: bar.progress(f, text=text))
    bar.empty()
    return calibration.temperatures(events), events


GAP_PER_PLACE = 0.15  # % qualifying gap that corresponds to roughly one grid place


def _apply_shift(X: pd.DataFrame, shift: np.ndarray, extra_rank=()) -> pd.DataFrame:
    """Apply manual/wet adjustments (in places) to position- and gap-type features."""
    X = X.copy()
    for c in ("quali_form", "form", *extra_rank):
        if c in X:
            X[c] = X[c] - shift
    for c in ("quali_gap", "race_pace", "fp_gap"):
        if c in X:
            X[c] = X[c] - shift * GAP_PER_PLACE
    return X


_KIND_LABEL = {"race": "Race (after qualifying)", "race_pre": "Race (before qualifying)",
               "sprint": "Sprint", "quali": "Qualifying"}


def _accuracy_panel(year: int, rnd: int, kinds: tuple) -> None:
    """Walk-forward accuracy per season, from the shared backtest store."""
    with st.expander("Model accuracy (walk-forward backtest)"):
        _calibration_data(year, rnd)  # keeps the store up to date
        events = [e for e in evaluate.load_store() if e["kind"] in kinds
                  and (e["season"], e["round"]) < (year, rnd)]
        if not events:
            st.info("No re-predicted events yet.")
            return
        tab = evaluate.season_summary(events)
        tab["kind"] = tab["kind"].map(_KIND_LABEL)
        st.markdown("Every event since 2020 was predicted using only data from before it, with "
                    "the confidence correction learned from earlier seasons, and compared with "
                    "the real result. Event-by-event detail is on the **Track record** page.")
        st.dataframe(
            tab.rename(columns={"season": "Season", "kind": "Session", "events": "Events",
                                "fav_hit": "Favourite won", "top3_hits": "Top 3 correct",
                                "logloss": "Log loss ↓", "spearman": "Order correlation",
                                "pos_mae": "Position error ↓"}),
            hide_index=True, width="stretch",
            column_config={"Season": st.column_config.NumberColumn(format="%d"),
                           "Favourite won": st.column_config.NumberColumn(format="percent"),
                           "Top 3 correct": st.column_config.NumberColumn(format="%.2f"),
                           "Log loss ↓": st.column_config.NumberColumn(format="%.3f"),
                           "Order correlation": st.column_config.NumberColumn(format="%.2f"),
                           "Position error ↓": st.column_config.NumberColumn(format="%.2f")})
        st.caption("*Order correlation* compares the predicted and actual finishing order "
                   "(1 = identical). Bookmakers use similar information plus insider knowledge "
                   "and a margin, so treat these numbers as a second opinion, not a betting signal.")


def _calib_note(calib: dict, what: str) -> str:
    if not calib.get("learned"):
        return ""
    t = calib["temperature"]
    if abs(t - 1) < 0.03:
        return f" · calibration from {calib['n']} past {what}: no correction needed"
    direction = "sharpened" if t < 1 else "widened"
    return f" · {direction} by {abs(1 / t - 1):.0%} based on {calib['n']} past {what}"


@st.cache_resource(show_spinner=False, max_entries=6, ttl=600)
def _betting_calibration(year: int, rnd: int, family: str):
    """(win recalibration, {'first'|'top3': backtest band evidence}) from past events."""
    _calibration_data(year, rnd)  # keeps the store up to date
    events = [e for e in evaluate.load_store() if (e["season"], e["round"]) < (year, rnd)]
    if family == "quali":
        kinds, recal = ("quali",), None  # pole probabilities are already well calibrated
    else:
        kinds = ("race", "sprint")
        recal = calibration.fit_win_recalibration(events, kinds)
    evidence = {"first": calibration.band_evidence(events, kinds, "first", recal),
                "top3": calibration.band_evidence(events, kinds, "top3")}
    return (recal or (0.0, 1.0)), evidence


@st.cache_data(show_spinner="Checking the FIA starting grid (penalties, pit-lane starts)…",
               ttl=600)
def _official_grid(year: int, rnd: int, event_name: str, target: str, sessions: dict):
    """Official FIA starting grid as abbreviation -> slot, or None if not published yet."""
    try:
        doc = grid_docs.starting_grid(year, event_name, sprint=(target == "Sprint"))
    except Exception:
        return None
    if doc is None:
        return None
    name = _quali_name(sessions, target)
    try:
        res = pdata.load_weekend_session(year, rnd, name, False, sessions[name]).results
        number_to_abbr = dict(zip(res["DriverNumber"].astype(str), res["Abbreviation"]))
    except Exception:
        return None
    grid = pd.Series({number_to_abbr[n]: slot for n, slot in doc["positions"].items()
                      if n in number_to_abbr}, dtype=float)
    if len(grid) < 10:
        return None
    return dict(grid=grid, source=doc["source"], notes=doc["notes"])


def _with_result(result: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Add the real result as a column right after driver/team (and grid), so the
    outcome sits next to the prediction without scrolling."""
    out = result.copy()
    res = out["Driver"].map(dict(zip(actual["Driver"], actual["Result"]))).fillna("—")
    after = 3 if "Grid" in out or "Est. grid" in out else 2
    out.insert(after, "Result", res)
    return out


def _result_check(result: pd.DataFrame, actual: pd.DataFrame, top_col: str, top3_col: str,
                  what: str, entries: pd.DataFrame, key: str) -> None:
    """Compact comparison of the prediction with the real result."""
    st.markdown("##### Result vs. prediction")
    merged = result.merge(actual, on="Driver", how="inner")
    winner = merged.loc[merged["Pos"] == 1]
    fav = result.iloc[0]["Driver"]
    fav_result = actual.set_index("Driver")["Result"].get(fav, "—")
    pred_top3 = set(result.sort_values(top3_col, ascending=False).head(3)["Driver"])
    real_top3 = set(actual.loc[actual["classified"] & (actual["Pos"] <= 3), "Driver"])
    fin = merged[merged["classified"]]
    mae = float((fin["Expected pos."] - fin["Pos"]).abs().mean()) if len(fin) else float("nan")

    c = st.columns(4)
    if len(winner):
        w = winner.iloc[0]
        rank = int((result[top_col] > w[top_col]).sum()) + 1
        c[0].metric(f"Actual {what}", w["Driver"], f"{w[top_col]:.0%} · pick #{rank}",
                    delta_color="off", delta_arrow="off",
                    help=f"Probability the model gave the actual {what}, and its rank among all drivers.")
    c[1].metric("Model favourite", fav, f"finished {fav_result}", delta_color="off",
                delta_arrow="off")
    c[2].metric("Top 3 correct", f"{len(pred_top3 & real_top3)} of 3",
                help=f"Drivers in the model's top 3 by {top3_col.lower()} probability who "
                     "actually finished in the top 3.")
    c[3].metric("Position error", f"{mae:.1f}", "avg. places", delta_color="off",
                delta_arrow="off", help="Average gap between expected and actual position, "
                                        "classified drivers only.")

    if len(fin) >= 5:
        color_of = dict(zip(entries["Driver"], _colors(entries)))
        top = int(max(fin["Pos"].max(), fin["Expected pos."].max())) + 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, top], y=[1, top], mode="lines", hoverinfo="skip",
                                 line=dict(color="rgba(139,147,161,.45)", width=1, dash="dot"),
                                 showlegend=False))
        fig.add_trace(go.Scatter(
            x=fin["Expected pos."], y=fin["Pos"], mode="markers+text", text=fin["Driver"],
            textposition="middle right", textfont=dict(size=10), showlegend=False,
            marker=dict(size=10, color=[color_of.get(d, "#888") for d in fin["Driver"]],
                        line=dict(color="#0D0F13", width=2)),
            hovertemplate="%{text}: expected P%{x:.1f}, finished P%{y}<extra></extra>"))
        fig.update_xaxes(title_text="Predicted (expected position)", range=[0.5, top],
                         dtick=2)
        fig.update_yaxes(title_text="Actual position", range=[top, 0.5], dtick=2)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(fig, width="stretch", key=f"result_scatter_{key}")
        with right:
            st.caption("Each dot is a classified driver. On the dotted line the prediction was "
                       "exact; above it the driver finished better than expected, below it "
                       "worse. Retirements are listed in the *Result* column of the table.")
