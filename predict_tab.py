"""Streamlit UI for the Predictions tab."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import data
import model
import predict_data as pdata
import ui
import upgrades

# Heuristic factors (not fitted - there is too little sprint/wet data to fit them)
SPRINT_RANDOMNESS = 0.85   # shorter race, fewer strategy swings
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


def _quali_name(sessions: dict, target: str) -> str | None:
    if target == "Race":
        return "Qualifying" if "Qualifying" in sessions else None
    for name in ("Sprint Qualifying", "Sprint Shootout"):
        if name in sessions:
            return name
    return None


@st.cache_resource(show_spinner=False, max_entries=6)
def _prepared(year: int, rnd: int):
    races, quali = pdata.history(year)
    hist, builder = model.build_history(races, quali, stop_before=(year, rnd))
    return races, hist, builder


@st.cache_resource(show_spinner=False, max_entries=6)
def _grid_estimator(year: int, rnd: int):
    """Qualifying ~ Sprint Qualifying + recent quali form, fitted on past sprint weekends."""
    _, quali = pdata.history(year)
    _, hist, _ = _prepared(year, rnd)
    years = range(year - pdata.HISTORY_SEASONS + 1, year + 1)
    sprints = pd.concat([pdata.season_sprints(y) for y in years], ignore_index=True)
    sprints = sprints[(sprints["season"] < year) | (sprints["round"] < rnd)]
    return model.fit_grid_estimate(hist, sprints, quali)


@st.cache_resource(show_spinner=False, max_entries=6)
def _fits(year: int, rnd: int):
    _, hist, _ = _prepared(year, rnd)
    return model.fit_rank(hist, model.FEATURES_POST), model.fit_rank(hist, model.FEATURES_PRE)


@st.cache_data(show_spinner=False, max_entries=6)
def _backtest(year: int, rnd: int) -> pd.DataFrame:
    _, hist, _ = _prepared(year, rnd)
    season = year if (hist["season"] == year).any() else year - 1
    post = model.backtest(hist, season)
    pre = model.backtest(hist, season, features=model.FEATURES_PRE)
    post["model"] = post["model"].map({"model": "This model, after qualifying",
                                       "grid only": "Grid order only (baseline)"})
    pre = pre[pre["model"] == "model"].assign(model="This model, before qualifying")
    out = pd.concat([post, pre])
    out["season"] = season
    return out


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
    pos = session.results.set_index("Abbreviation")["Position"].dropna()
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
   Afterwards the race view uses the actual grid and shows the change per driver.
   Reload the page once qualifying has finished.
4. **Inputs.** The rain probability is pre-filled from the forecast. Use *Adjustments* for
   grid penalties, upgrade packages or withdrawals.
5. **Declared upgrades** lists the parts each team registered with the FIA. Shown for
   reference only: in testing, adding them did not improve the predictions.
6. **Compare with bookmaker odds.** Open *Paste odds*, copy a market from the bookmaker's
   page and paste it. A positive *edge* means the price is higher than the model's fair odds;
   small edges are within the model's error.
7. **Check *Model accuracy*** at the bottom of each session before relying on a number.
""")

    c0, c1, c3 = st.columns([1, 4, 1])
    with c0:
        seasons = list(range(data.current_season(), data.FIRST_SEASON + 3, -1))
        year = st.selectbox("Season", seasons, key="pred_season")
    schedule = data.get_schedule(year)
    schedule = schedule[schedule["RoundNumber"] > 0]
    now = data.now_utc()
    upcoming = [i for i, d in enumerate(schedule["Session5DateUtc"]) if pd.notna(d) and d > now - pd.Timedelta(hours=3)]
    rounds = list(schedule["RoundNumber"].astype(int))
    with c1:
        rnd = st.selectbox("Grand Prix", rounds, index=upcoming[0] if upcoming else len(rounds) - 1,
                           format_func=lambda r: f"R{r} · {schedule.set_index('RoundNumber').loc[r, 'EventName']}",
                           key=f"pred_round_{year}")
    event = schedule[schedule["RoundNumber"] == rnd].iloc[0]
    sessions = _sessions(event)
    with c3:
        st.write("")
        if st.button("Refresh data", help="Re-download results; use after a session has finished."):
            for fn in (pdata.season_races, pdata.season_quali, pdata.season_sprints,
                       pdata.weekend_session, pdata._fresh_session, _prepared, _fits, _backtest,
                       _grid_estimator, _quali_data, _quali_fit, _quali_backtest):
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
    fit_post, fit_pre = _fits(year, rnd)
    circuit = info["circuitId"]
    quali_name = _quali_name(sessions, target)
    grid = _grid(sessions, target, year, rnd)
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
        if grid is not None:
            st.success(f"**Starting grid from the {quali_name} results.** The *before* column "
                       "shows what the model predicted without it.")
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
    randomness = chaos * (1 + RAIN_RANDOMNESS * rain)
    if target == "Sprint":
        p_dnf, randomness = p_dnf * SPRINT_DNF, randomness * SPRINT_RANDOMNESS
    drivers = list(entries["Driver"][keep])

    def run(how: str):
        """Simulate with the known grid ("grid"), a Sprint-Qualifying-based estimate
        ("estimate") or without any grid ("pre"). Returns (fit, features, positions)."""
        X = feats.copy()
        if how == "grid":
            g = np.clip(edited["Grid"].to_numpy(float) - shift, 1, 22)
            X["grid"], X["grid_log"] = g, np.log(g)
            return fit_post, X, model.simulate_rank(fit_post.strength(X)[keep], p_dnf, randomness)
        if how == "estimate":
            # Sprint Qualifying + recent form -> expected qualifying position; its
            # uncertainty is simulated (see model.simulate_uncertain_grid)
            w, est_sd = estimator
            sq_pos = entries["Driver"].map(sq).fillna(21).to_numpy(float)
            est = w[0] + w[1] * sq_pos + w[2] * X["quali_form"].to_numpy() - shift
            X["grid"], X["grid_log"] = est, np.log(np.clip(est, 1, 22))
            return fit_post, X, model.simulate_uncertain_grid(fit_post, X[keep], est[keep],
                                                              est_sd, p_dnf, randomness)
        if fp_rank is not None:
            fp = entries["Driver"].map(fp_rank).fillna(X["quali_form"]).to_numpy()
            X["quali_form"] = (1 - FP_BLEND) * X["quali_form"] + FP_BLEND * fp
        X["quali_form"] -= shift
        X["form"] -= shift
        return fit_pre, X, model.simulate_rank(fit_pre.strength(X)[keep], p_dnf, randomness)

    pre_how = "estimate" if sq is not None else "pre"
    how = "grid" if edited["Grid"].notna().all() else pre_how
    fit, X, pos = run(how)
    mode = {"grid": "after qualifying" if grid is not None else "with the grid you entered",
            "estimate": "before qualifying, grid estimated from Sprint Qualifying",
            "pre": "before qualifying"}[how]

    result = model.summarise(pos, drivers)
    result.insert(1, "Team", list(entries["Team"][keep]))
    if how == "grid":  # show the real starting position, not the adjusted model input
        result.insert(2, "Grid", edited["Grid"].to_numpy()[keep].astype(int))
    elif how == "estimate":
        result.insert(2, "Est. grid", np.round(X["grid"].to_numpy()[keep], 1))
    if how == "grid" and grid is not None:
        # what qualifying changed: the same model without the real grid
        before = model.summarise(run(pre_how)[2], drivers).set_index("Driver")
        result.insert(4, "Win before quali", result["Driver"].map(before["Win"]))
        result.insert(5, "Change", (result["Win"] - result["Win before quali"]) * 100)
    result["Fair odds (win)"] = (1 / result["Win"]).where(result["Win"] > 0.001)
    result = result.sort_values("Win", ascending=False).reset_index(drop=True)

    actual = _results(sessions, target, year, rnd, races)
    if actual is not None:
        result = _with_result(result, actual)

    # --------------------------------------------------------------- output
    st.markdown(f"#### {event['EventName']} · {target}")
    st.caption(f"Prediction {mode} · {len(pos):,} simulated races · trained on {fit.n_races} races")
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
        teams = (pd.DataFrame({"Team": entries["Team"][keep].values,
                               "Win": (pos == 1).mean(0)}).groupby("Team")["Win"].sum()
                 .sort_values(ascending=False).reset_index())
        teams["Fair odds"] = (1 / teams["Win"]).where(teams["Win"] > 0.001)
        st.markdown("**Winning team**")
        st.dataframe(_as_pct(teams.head(6), ["Win"]), hide_index=True, width="stretch",
                     column_config={**_pct(teams, ["Win"]),
                                    "Fair odds": st.column_config.NumberColumn(format="%.1f")})

    _head_to_head(pos, drivers, entries[keep].reset_index(drop=True), key=target)
    _value_check(result, MARKETS, key=target, entries=entries)
    _explain(fit, X[keep], feats[keep], entries[keep], wet_gain, rain, chaos, target, year, rnd,
             sq)


def _head_to_head(pos: np.ndarray, drivers: list[str], entries: pd.DataFrame, key: str) -> None:
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
    for team, g in entries.groupby("Team"):
        if len(g) != 2:
            continue
        i, j = g.index
        pi, pj, _ = model.head_to_head(pos, i, j)
        rows.append({"Team": team, "Driver 1": drivers[i], "P(1 ahead)": pi,
                     "Driver 2": drivers[j], "P(2 ahead)": pj})
    if rows:
        with st.expander("Teammate battles"):
            tm = pd.DataFrame(rows)
            st.dataframe(_as_pct(tm, ["P(1 ahead)", "P(2 ahead)"]), hide_index=True,
                         width="stretch", column_config=_pct(tm, ["P(1 ahead)", "P(2 ahead)"]))


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


def _value_check(result: pd.DataFrame, markets: list[str], key: str,
                 entries: pd.DataFrame | None = None) -> None:
    st.markdown("##### Compare with bookmaker odds")
    st.caption("**Edge** is the expected return per €1 if the model's probability were correct; "
               "positive = the bookmaker pays more than the model thinks fair. The model has its "
               "own error (see the backtest), so small edges are noise.")
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
    bets["Edge"] = bets["Model prob."] * bets["Odds"] - 1
    bets = bets.sort_values("Edge", ascending=False)
    margin = bets.groupby("Market")["Implied prob."].sum()
    st.dataframe(bets.style.map(lambda v: "color: #1a9850; font-weight: 600" if v > 0.05 else
                                ("color: #d73027" if v < 0 else ""), subset=["Edge"])
                 .format({"Model prob.": "{:.1%}", "Implied prob.": "{:.1%}", "Edge": "{:+.0%}",
                          "Odds": "{:.2f}"}),
                 hide_index=True, width="stretch")
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
            "Team quali pace": feats["team_pace"].round(1).values,
            "Team trend /race": feats["team_trend"].round(2).values,
            "Circuit history": feats["circuit_aff"].round(2).values,
            "Races here": feats["n_circuit"].values,
            "Retire chance": feats["p_dnf"].values,
            "Wet gain": entries["driverId"].map(wet_gain).round(2).values,
            "Sprint quali": entries["Driver"].map(sq).values if sq is not None else np.nan,
        })
        st.markdown("**Inputs per driver** (positions; lower = better, except *Wet gain*: "
                    "places gained on the teammate in wet races vs. dry. *Team trend* < 0 = improving, e.g. upgrades. "
                    "*Circuit history* < 0 = usually does better here than elsewhere.)")
        st.dataframe(shown, hide_index=True, width="stretch",
                     column_config={"Retire chance": st.column_config.NumberColumn(format="percent")})
        st.markdown(
            f"**Adjustments applied on top**: circuit randomness ×{chaos:.2f}; rain {rain:.0%} "
            f"→ randomness +{RAIN_RANDOMNESS * rain:.0%}, retirements +{RAIN_DNF * rain:.0%}"
            + (f", wet record shifts drivers by up to "
               f"{(wet_gain.abs().max() if len(wet_gain) else 0) * rain:.1f} places"
               f" (based on {st.session_state.get('n_wet_races', 0)} wet races)" if len(wet_gain) else "")
            + (f"; sprint: randomness ×{SPRINT_RANDOMNESS}, retirements ×{SPRINT_DNF}"
               if target == "Sprint" else "") + ". These factors are rules of thumb, not fitted.")
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

    with st.expander("Model accuracy (walk-forward backtest)"):
        with st.spinner("Re-predicting every race of the season using only earlier data…"):
            bt = _backtest(year, rnd)
        if bt.empty:
            st.info("Not enough races yet to backtest.")
            return
        season = int(bt["season"].iloc[0])
        summary = (bt.groupby("model")[["winner_logloss", "top_pick_won", "podium_brier", "pos_mae"]]
                   .mean().reset_index().rename(columns={
                       "model": "Approach", "winner_logloss": "Winner log-loss ↓",
                       "top_pick_won": "Favourite won", "podium_brier": "Podium Brier ↓",
                       "pos_mae": "Avg. position error ↓"}))
        st.markdown(f"Every {season} race before this one was predicted using only data from "
                    f"before it ({bt['round'].nunique()} races). Lower ↓ is better.")
        st.dataframe(summary, hide_index=True, width="stretch",
                     column_config={"Favourite won": st.column_config.NumberColumn(format="percent"),
                                    "Winner log-loss ↓": st.column_config.NumberColumn(format="%.3f"),
                                    "Podium Brier ↓": st.column_config.NumberColumn(format="%.3f"),
                                    "Avg. position error ↓": st.column_config.NumberColumn(format="%.2f")})
        st.markdown(
            "**How to read this**: after qualifying, this model is about as good as simply "
            "trusting the grid for picking the winner and podium, and clearly better at "
            "predicting the rest of the order (head-to-heads, points finishes). Before "
            "qualifying it is less accurate, as expected. Bookmakers use similar information plus "
            "insider knowledge and price in a margin, so consistently beating their odds with "
            "this model is unlikely. Treat the numbers as a second opinion, not a signal to bet.")


# --------------------------------------------------------------- qualifying
@st.cache_resource(show_spinner=False, max_entries=4)
def _quali_data(year: int, rnd: int) -> pd.DataFrame:
    """Past qualifying entries with pre-weekend features and weekend signals."""
    _, quali = pdata.history(year)
    _, hist, _ = _prepared(year, rnd)
    years = range(year - pdata.HISTORY_SEASONS + 1, year + 1)
    sprints = pd.concat([pdata.season_sprints(y) for y in years], ignore_index=True)
    sprints = sprints[(sprints["season"] < year) | (sprints["round"] < rnd)]
    sprint_weekends = set(map(tuple, sprints[["season", "round"]].drop_duplicates().values))
    weekends = hist[hist["season"] >= year - 2][["season", "round"]].drop_duplicates()
    fp = {}
    text = "Loading practice pace of past weekends (first time ~2 min)…"
    progress = st.progress(0.0, text=text)
    for i, (s, r) in enumerate(weekends.itertuples(index=False)):
        if (s, r) not in sprint_weekends:
            rank = pdata.practice_rank(int(s), int(r), "Practice 3")
            if rank is not None:
                fp[(s, r)] = rank
        progress.progress((i + 1) / len(weekends), text=text)
    progress.empty()
    return model.quali_rows(hist, quali, sprints, fp)


@st.cache_resource(show_spinner=False, max_entries=12)
def _quali_fit(year: int, rnd: int, variant: str):
    feats = model.QUALI_VARIANTS[variant]
    return model.fit_rank(_quali_data(year, rnd).dropna(subset=feats), feats)


@st.cache_data(show_spinner=False, max_entries=4)
def _quali_backtest(year: int, rnd: int) -> pd.DataFrame:
    rows = _quali_data(year, rnd)
    season = year if (rows["season"] == year).any() else year - 1
    return pd.concat([model.quali_backtest(rows, s) for s in (season - 1, season)])


def _weekend_signal(sessions: dict, year: int, rnd: int) -> tuple[str, pd.Series | None, str]:
    """(model variant, abbreviation-indexed signal, source session) for this weekend."""
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
            best = laps.dropna(subset=["LapTime"]).groupby("Driver")["LapTime"].min()
            if len(best) >= 10:
                return "practice", best.rank(), name
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

    feats = model.QUALI_VARIANTS[variant]
    X = pd.DataFrame([builder.features(d, t, info["circuitId"])
                      for d, t in zip(entries["driverId"], entries["constructorId"])])
    weekend_col = {"sprint quali": "sq", "practice": "fp"}.get(variant)
    X["sq"] = X["fp"] = np.nan
    if weekend_col:
        X[weekend_col] = entries["Driver"].map(signal).to_numpy()
    X = model.fill_missing(X, feats)
    for col in ["quali_form", "form"] + ([weekend_col] if weekend_col else []):
        X[col] = X[col] - shift
    fit = _quali_fit(year, rnd, variant)
    drivers = list(entries["Driver"][keep])
    pos = model.simulate_rank(fit.strength(X)[keep], np.zeros(int(keep.sum())),
                              randomness=1 + RAIN_RANDOMNESS * rain)
    result = pd.DataFrame({"Driver": drivers, "Team": list(entries["Team"][keep]),
                           "Pole": (pos == 1).mean(0), "Front row": (pos <= 2).mean(0),
                           "Top 3": (pos <= 3).mean(0), "Q3 (top 10)": (pos <= 10).mean(0),
                           "Q2 (top 15)": (pos <= 15).mean(0), "Expected pos.": pos.mean(0)})
    result["Fair odds (pole)"] = (1 / result["Pole"]).where(result["Pole"] > 0.001)
    result = result.sort_values("Pole", ascending=False).reset_index(drop=True)
    actual = _results(sessions, "Qualifying", year, rnd)
    if actual is not None:
        result = _with_result(result, actual)

    st.markdown(f"#### {event['EventName']} · Qualifying")
    st.caption(f"Based on recent form{' and ' + source if variant != 'form' else ''} · "
               f"{len(pos):,} simulated sessions · trained on {fit.n_races} sessions")
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
    _value_check(result, QUALI_MARKETS, key="quali", entries=entries)

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

    with st.expander("Model accuracy (walk-forward backtest)"):
        with st.spinner("Re-predicting past qualifying sessions using only earlier data…"):
            bt = _quali_backtest(year, rnd)
        if bt.empty:
            st.info("Not enough sessions to backtest.")
            return
        names = {"form": "Recent form only", "practice": "Form + practice pace",
                 "sprint quali": "Form + Sprint Qualifying"}
        kinds = {"practice": "Normal weekends", "sprint quali": "Sprint weekends",
                 "form": "No weekend data"}
        summary = (bt.assign(model=bt["model"].map(names), weekend=bt["weekend"].map(kinds))
                   .groupby(["weekend", "model"])
                   .agg(Sessions=("round", "size"),
                        **{"Pole log-loss ↓": ("pole_logloss", "mean"),
                           "Favourite took pole": ("fav_on_pole", "mean"),
                           "Top-3 Brier ↓": ("top3_brier", "mean"),
                           "Avg. position error ↓": ("pos_mae", "mean")})
                   .reset_index().rename(columns={"weekend": "Weekend", "model": "Approach"}))
        st.dataframe(summary, hide_index=True, width="stretch",
                     column_config={
                         "Favourite took pole": st.column_config.NumberColumn(format="percent"),
                         "Pole log-loss ↓": st.column_config.NumberColumn(format="%.3f"),
                         "Top-3 Brier ↓": st.column_config.NumberColumn(format="%.3f"),
                         "Avg. position error ↓": st.column_config.NumberColumn(format="%.2f")})
        st.markdown("Pole is hard to call: small gaps, track evolution and mistakes decide it. "
                    "This weekend's pace (practice or Sprint Qualifying) clearly improves the "
                    "prediction over recent form alone.")


# ---------------------------------------------------------------- upgrades
@st.cache_data(show_spinner=False, ttl=600)
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
