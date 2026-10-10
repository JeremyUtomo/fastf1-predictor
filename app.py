"""FastF1 Explorer - a point-and-click interface for FastF1 data.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import data
import predict_tab
import track_record
import ui

st.set_page_config(page_title="FastF1 Explorer", page_icon=":material/speed:", layout="wide")
ui.inject_css()


# ------------------------------------------------------------------ sidebar
def sidebar() -> dict:
    ui.sidebar_brand()

    years = list(range(data.current_season(), data.FIRST_SEASON - 1, -1))
    year = st.sidebar.selectbox("Season", years)

    try:
        schedule = data.get_schedule(year)
    except Exception as exc:  # network / API problems
        st.sidebar.error(f"Could not load the {year} schedule: {exc}")
        st.stop()

    now = data.now_utc()
    schedule = schedule[schedule["Session1DateUtc"] < now]  # events that have started
    if schedule.empty:
        st.sidebar.warning("No events have taken place in this season yet.")
        st.stop()
    labels = {int(r.RoundNumber): ("Testing" if r.RoundNumber == 0 else f"R{int(r.RoundNumber)}")
              + f" · {r.EventName}" for r in schedule.itertuples()}
    rounds = sorted(labels, reverse=True)
    rnd = st.sidebar.selectbox("Event", rounds, format_func=labels.get)
    event = schedule[schedule["RoundNumber"] == rnd].iloc[0]

    sessions = data.event_sessions(event)
    available = [n for n, done in sessions if done] or [sessions[0][0]]
    default = available.index("Race") if "Race" in available else len(available) - 1
    session_name = st.sidebar.selectbox("Session", available, index=default)

    telemetry = st.sidebar.toggle(
        "Include telemetry", value=True,
        help="Car and position telemetry (speed, throttle, track map...). "
             "Much larger download on first load; cached on disk afterwards.")
    load = st.sidebar.button("Load session", type="primary", width="stretch")
    st.sidebar.divider()
    st.sidebar.caption(f"Data cache: `{data._CACHE_DIR}`")
    return dict(year=year, round=int(rnd), event=event, session=session_name,
                telemetry=telemetry, load=load)


# ------------------------------------------------------------------- tabs
FORMATS = {"conventional": "Standard", "sprint_qualifying": "Sprint", "sprint_shootout": "Sprint",
           "sprint": "Sprint", "testing": "Testing"}


def tab_season(year: int, schedule_event: pd.Series):
    st.subheader(f"{year} season")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("##### Calendar")
        sched = data.get_schedule(year)
        now = data.now_utc()
        view = pd.DataFrame({
            "Round": sched["RoundNumber"], "Event": sched["EventName"],
            "Location": sched["Location"] + ", " + sched["Country"],
            "Date": sched["EventDate"].dt.date,
            "Format": sched["EventFormat"].map(FORMATS).fillna(sched["EventFormat"]),
            "Status": ["Completed" if d < now else "Upcoming" for d in sched["EventDate"]],
        })
        st.dataframe(view, hide_index=True, width="stretch", height=600,
                     column_config={"Round": st.column_config.NumberColumn(width="small"),
                                    "Date": st.column_config.DateColumn(format="D MMM")})
    with right:
        st.markdown("##### Championship standings")
        try:
            drivers, constructors, rnd = data.get_standings(year)
        except Exception:
            st.info("Standings are not available for this season yet.")
            return
        which = st.segmented_control("Standings", ["Drivers", "Constructors"], default="Drivers",
                                     label_visibility="collapsed", key="standings_view")
        st.caption(f"After round {rnd}")
        if which == "Constructors":
            df = constructors.rename(columns={"constructorName": "Team"})
            label = "Team"
        else:
            df = drivers.copy()
            df["Driver"] = df["givenName"] + " " + df["familyName"]
            df["Team"] = df["constructorNames"].map(lambda x: x[-1] if len(x) else "")
            label = "Driver"
        st.plotly_chart(charts.standings_bar(df.head(20), label), width="stretch")
        cols = ["position", label] + (["Team"] if label == "Driver" else []) + ["points", "wins"]
        st.dataframe(df[cols].rename(columns={"position": "Pos", "points": "Points", "wins": "Wins"}),
                     hide_index=True, width="stretch")


def tab_overview(session, kind: str):
    res = session.results.copy()
    laps = session.laps
    st.subheader(f"{session.event['EventName']} — {session.name}")
    st.caption(f"{session.event['Location']}, {session.event['Country']} · "
               f"{session.date:%d %b %Y} · {session.event['OfficialEventName']}")

    fastest = laps.pick_fastest() if len(laps) else None
    cols = st.columns(4)
    if len(res) and kind != "practice" and pd.notna(res.iloc[0]["Position"]):
        top = res.sort_values("Position").iloc[0]
        cols[0].metric("Pole" if kind == "qualifying" else "Winner", top["Abbreviation"],
                       top["TeamName"], delta_color="off", delta_arrow="off")
    if fastest is not None:
        cols[1].metric("Fastest lap", data.fmt_laptime(fastest["LapTime"]),
                       f"{fastest['Driver']} · lap {int(fastest['LapNumber'])}",
                       delta_color="off", delta_arrow="off")
    cols[2].metric("Laps recorded", int(laps["LapNumber"].max()) if len(laps) else 0)
    w = session.weather_data
    if w is not None and len(w):
        rain = "Wet" if w["Rainfall"].any() else "Dry"
        cols[3].metric("Track / air temp", f"{w['TrackTemp'].mean():.0f}° / {w['AirTemp'].mean():.0f}°C",
                       rain, delta_color="off", delta_arrow="off")

    st.markdown("##### Classification")
    show = pd.DataFrame({
        "Pos": res["Position"], "Photo": res["HeadshotUrl"], "No.": res["DriverNumber"],
        "Driver": res["FullName"], "Team": res["TeamName"],
    })
    if kind == "race":
        show["Grid"] = res["GridPosition"]
        show["Result"] = [data.fmt_laptime(t) if i == 0 and pd.notna(t) else
                          (f"+{t.total_seconds():.3f}s" if pd.notna(t) else s)
                          for i, (t, s) in enumerate(zip(res["Time"], res["Status"]))]
        show["Points"] = res["Points"]
    elif kind == "qualifying":
        for q in ("Q1", "Q2", "Q3"):
            show[q] = res[q].map(data.fmt_laptime)
    else:
        best = laps.groupby("Driver")["LapTime"].min()
        show["Best lap"] = res["Abbreviation"].map(best).map(data.fmt_laptime)
        show["Laps"] = res["Abbreviation"].map(laps.groupby("Driver")["LapNumber"].count())
        show = show.assign(_s=res["Abbreviation"].map(best)).sort_values("_s").drop(columns="_s")
        show["Pos"] = range(1, len(show) + 1)
    show = show.dropna(axis=1, how="all")
    st.dataframe(show, hide_index=True, width="stretch", height=ui.table_height(len(show)), row_height=ui.ROW_HEIGHT,
                 column_config={"Photo": st.column_config.ImageColumn("", width="small"),
                                "Pos": st.column_config.NumberColumn(format="%d")})


def _driver_picker(session, key: str, default_n: int = 5, max_n: int | None = None):
    res = session.results.sort_values("Position")
    options = [d for d in res["Abbreviation"] if d in set(session.laps["Driver"])]
    return st.multiselect("Drivers", options, default=options[:default_n], key=key,
                          max_selections=max_n)


def tab_laps(session, kind: str, styles: dict):
    laps = session.laps
    if laps.empty:
        st.info("No lap data for this session.")
        return
    c1, c2 = st.columns([3, 2])
    with c1:
        drivers = _driver_picker(session, "laps_drivers")
    with c2:
        quick = st.checkbox("Only representative laps", value=True,
                            help="Hides in/out laps, safety-car laps and anything slower than "
                                 "107% of the fastest lap, so the pace trend stays readable.")
    sel = laps[laps["Driver"].isin(drivers)]
    if quick:
        sel = sel.pick_quicklaps(1.07).pick_wo_box()
    st.markdown("##### Lap times")
    st.plotly_chart(charts.lap_time_chart(sel, styles, drivers), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("##### Pace distribution")
        st.plotly_chart(charts.pace_box(sel, styles, drivers), width="stretch")
    with right:
        if kind == "race":
            st.markdown("##### Position by lap")
            st.plotly_chart(charts.position_chart(laps, styles, drivers), width="stretch")
        else:
            st.markdown("##### Best lap per driver")
            best = (laps.dropna(subset=["LapTime"]).sort_values("LapTime")
                    .groupby("Driver", as_index=False).first().sort_values("LapTime"))
            best["Lap time"] = best["LapTime"].map(data.fmt_laptime)
            best["Gap"] = (best["LapTime"] - best["LapTime"].iloc[0]).dt.total_seconds().map("+{:.3f}".format)
            st.dataframe(best[["Driver", "Team", "Lap time", "Gap", "Compound", "LapNumber"]],
                         hide_index=True, width="stretch")

    with st.expander("Sector times and speed traps (best per driver)"):
        d = laps[laps["Driver"].isin(drivers)]
        agg = d.groupby("Driver").agg(
            S1=("Sector1Time", "min"), S2=("Sector2Time", "min"), S3=("Sector3Time", "min"),
            **{"Speed trap": ("SpeedST", "max"), "Finish line": ("SpeedFL", "max"),
               "Int. 1": ("SpeedI1", "max"), "Int. 2": ("SpeedI2", "max")})
        for col in ("S1", "S2", "S3"):
            agg[col] = agg[col].dt.total_seconds().round(3)
        st.dataframe(agg, width="stretch")


def tab_strategy(session, styles: dict):
    laps = session.laps
    if laps.empty or laps["Compound"].isna().all():
        st.info("No tyre data for this session.")
        return
    order = list(session.results.sort_values("Position")["Abbreviation"])
    order = [d for d in order if d in set(laps["Driver"])]
    st.markdown("##### Tyre strategy" + (" (in finishing order)" if session.name in ("Race", "Sprint") else ""))
    st.plotly_chart(charts.strategy_chart(laps, order), width="stretch")
    pits = (laps.dropna(subset=["PitInTime"]).groupby("Driver").size().reindex(order)
            .fillna(0).astype(int).rename("Pit stops"))
    with st.expander("Pit stop counts"):
        st.dataframe(pits, width="stretch")


def _lap_choice(session, drv: str, key: str):
    d = session.laps.pick_drivers(drv).dropna(subset=["LapTime"])
    fastest = d.pick_fastest()
    opts = ["Fastest"] + [int(n) for n in d["LapNumber"]]
    choice = st.selectbox(f"{drv} lap", opts, key=key,
                          format_func=lambda x: f"Fastest ({data.fmt_laptime(fastest['LapTime'])})"
                          if x == "Fastest" else f"Lap {x}")
    return fastest if choice == "Fastest" else d[d["LapNumber"] == choice].iloc[0]


def tab_telemetry(session, styles: dict):
    laps = session.laps
    if laps.empty:
        st.info("No lap data for this session.")
        return
    res = session.results.sort_values("Position")
    options = [d for d in res["Abbreviation"] if d in set(laps["Driver"])]
    drivers = st.multiselect("Compare drivers (the first one is the reference)", options,
                             default=options[:2], max_selections=4, key="tel_drivers")
    if not drivers:
        st.info("Pick at least one driver.")
        return
    cols = st.columns(len(drivers))
    chosen = []
    for col, drv in zip(cols, drivers):
        with col:
            chosen.append((drv, _lap_choice(session, drv, f"tel_lap_{drv}")))

    traces = []
    for drv, lap in chosen:
        try:
            tel = lap.get_telemetry().add_distance()
        except Exception:
            st.warning(f"No telemetry available for {drv} on that lap.")
            continue
        label = f"{drv} L{int(lap['LapNumber'])}"
        traces.append(dict(label=label, tel=tel, **styles[drv]))
    if not traces:
        return
    traces = charts.align_distances(traces)

    channels = st.multiselect(
        "Channels", ["Speed", "Throttle", "Brake", "nGear", "RPM", "DRS"],
        default=["Speed", "Throttle", "Brake", "nGear"],
        format_func=lambda c: {"nGear": "Gear"}.get(c, c))
    try:
        corners = session.get_circuit_info().corners
    except Exception:
        corners = None
    st.plotly_chart(charts.telemetry_chart(traces, corners, channels), width="stretch")

    m1, m2 = st.columns(2)
    with m1:
        st.plotly_chart(charts.track_speed_map(traces[0]["tel"],
                                               f"{traces[0]['label']} — speed around the lap"),
                        width="stretch")
    with m2:
        if len(traces) > 1:
            st.plotly_chart(charts.dominance_map(traces), width="stretch")
        else:
            st.info("Select a second driver to see who is quickest in each part of the track.")


def tab_weather(session):
    w = session.weather_data
    if w is None or w.empty:
        st.info("No weather data for this session.")
        return
    c = st.columns(4)
    c[0].metric("Air temp", f"{w['AirTemp'].mean():.1f}°C", f"{w['AirTemp'].min():.1f}–{w['AirTemp'].max():.1f}", delta_color="off", delta_arrow="off")
    c[1].metric("Track temp", f"{w['TrackTemp'].mean():.1f}°C", f"{w['TrackTemp'].min():.1f}–{w['TrackTemp'].max():.1f}", delta_color="off", delta_arrow="off")
    c[2].metric("Humidity", f"{w['Humidity'].mean():.0f}%")
    c[3].metric("Rain", "Yes" if w["Rainfall"].any() else "No",
                f"{w['Rainfall'].mean():.0%} of samples" if w["Rainfall"].any() else None, delta_color="off", delta_arrow="off")
    st.plotly_chart(charts.weather_chart(w), width="stretch")
    st.caption("Blue bands mark periods with rainfall.")


def tab_race_control(session):
    msgs = session.race_control_messages
    status = session.track_status
    c = st.columns(4)
    count = lambda pat: int(status["Message"].str.contains(pat, case=False, na=False).sum()) if len(status) else 0
    c[0].metric("Yellow flag periods", count("Yellow"))
    c[1].metric("Safety car", count("^SCDeployed"))
    c[2].metric("Virtual SC", count("^VSCDeployed"))
    c[3].metric("Red flags", count("Red"))

    st.markdown("##### Race control messages")
    if msgs is None or msgs.empty:
        st.info("No race control messages.")
    else:
        cats = sorted(msgs["Category"].dropna().unique())
        pick = st.multiselect("Category", cats, default=cats)
        text = st.text_input("Search messages", placeholder="e.g. penalty, track limits, investigation")
        view = msgs[msgs["Category"].isin(pick)]
        if text:
            view = view[view["Message"].str.contains(text, case=False, na=False)]
        st.dataframe(view[["Time", "Lap", "Category", "Flag", "Message"]], hide_index=True,
                     width="stretch", height=420)
    with st.expander("Track status changes"):
        st.dataframe(status, hide_index=True, width="stretch")


def tab_raw(session):
    st.markdown("Browse and download everything FastF1 returned for this session.")
    datasets = {
        "Results": session.results, "Laps": session.laps,
        "Weather": session.weather_data, "Race control messages": session.race_control_messages,
        "Track status": session.track_status, "Session status": session.session_status,
    }
    name = st.selectbox("Dataset", list(datasets))
    df = datasets[name]
    if df is None or df.empty:
        st.info("This dataset is empty.")
        return
    st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
    if name == "Laps" and "Driver" in df:
        pick = st.multiselect("Filter by driver", sorted(df["Driver"].unique()))
        if pick:
            df = df[df["Driver"].isin(pick)]
    st.dataframe(df, hide_index=True, width="stretch", height=520)
    st.download_button("Download CSV", df.to_csv(index=False).encode(),
                       file_name=f"{session.event['EventName']}_{session.name}_{name}.csv".replace(" ", "_"),
                       mime="text/csv")


# --------------------------------------------------------------------- main
def season_page():
    ui.page_header("Season")
    c, _ = st.columns([1, 5])
    with c:
        year = st.selectbox("Season", list(range(data.current_season(), data.FIRST_SEASON - 1, -1)),
                            key="season_year")
    tab_season(year, None)


def predictions_page():
    ui.page_header("Predictions")
    predict_tab.render()


def track_record_page():
    ui.page_header("Track record")
    track_record.render()


def session_page():
    sel = sidebar()
    key = (sel["year"], sel["round"], sel["session"], sel["telemetry"])
    if sel["load"]:
        st.session_state["loaded_key"] = key

    header = st.empty()
    ui.page_header("Session analysis", target=header)
    tabs = st.tabs(["Overview", "Laps", "Strategy", "Telemetry", "Weather", "Race control", "Data"])

    loaded = st.session_state.get("loaded_key")
    if loaded is None:
        for t in tabs:
            with t:
                st.info("Select an event and session in the sidebar, then choose **Load session**.")
        return

    year, rnd, name, telemetry = loaded
    try:
        with st.spinner(f"Loading {year} round {rnd} · {name}. The first load downloads data "
                        "and can take a minute; later loads come from the cache."):
            session = data.load_session(year, rnd, name, telemetry)
    except Exception as exc:
        st.session_state.pop("loaded_key")
        for t in tabs:
            with t:
                st.error(f"Could not load this session: {exc}")
        return

    if key != loaded:
        st.sidebar.info("Selection changed — press **Load session** to switch.")
    ui.page_header(f"{session.event['EventName']} {year} · {session.name}", target=header)

    kind = data.session_kind(session.name)
    styles = charts.driver_styles(session)
    with tabs[0]:
        tab_overview(session, kind)
    with tabs[1]:
        tab_laps(session, kind, styles)
    with tabs[2]:
        tab_strategy(session, styles)
    with tabs[3]:
        if telemetry:
            tab_telemetry(session, styles)
        else:
            st.info("Telemetry was switched off for this load. Turn on **Include telemetry** "
                    "in the sidebar and press **Load session** again.")
    with tabs[4]:
        tab_weather(session)
    with tabs[5]:
        tab_race_control(session)
    with tabs[6]:
        tab_raw(session)


# Season and Predictions are full-width pages; only Session analysis has the sidebar
st.navigation([
    st.Page(season_page, title="Season", url_path="season"),
    st.Page(predictions_page, title="Predictions", url_path="predictions"),
    st.Page(track_record_page, title="Track record", url_path="track-record"),
    st.Page(session_page, title="Session analysis", url_path="session"),
], position="top").run()
