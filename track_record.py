"""Track record page: saved predictions vs. results, and what the model learned."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import calibration
import data
import evaluate
import predict_data as pdata
import predict_tab

KIND_LABEL = {"quali": "Qualifying", "sprint": "Sprint", "race": "Race"}
BACKTEST_KINDS = {"Qualifying": "quali", "Race (after qualifying)": "race",
                  "Race (before qualifying)": "race_pre", "Sprint": "sprint"}
# approximate team colours, so drivers keep a consistent colour in historical charts
TEAM_COLORS = {"mercedes": "27F4D2", "red_bull": "3671C6", "ferrari": "E8002D",
               "mclaren": "FF8000", "aston_martin": "229971", "alpine": "0093CC",
               "williams": "64C4FF", "rb": "6692FF", "alphatauri": "5E8FAA",
               "toro_rosso": "469BFF", "sauber": "52E252", "alfa": "C92D4B", "haas": "B6BABD",
               "racing_point": "F596C8", "force_india": "F596C8", "renault": "FFF500",
               "audi": "BB0A30", "cadillac": "909090"}
TEAM_NAMES = {"red_bull": "Red Bull", "rb": "RB", "mclaren": "McLaren", "alphatauri": "AlphaTauri",
              "toro_rosso": "Toro Rosso", "alfa": "Alfa Romeo", "force_india": "Force India"}


def _team_name(cid: str) -> str:
    return TEAM_NAMES.get(cid, str(cid).replace("_", " ").title())


def _score_log() -> pd.DataFrame:
    """One row per saved prediction whose session has a result."""
    log = calibration.read_log()
    rows = []
    for (season, rnd, session, mode), g in log.groupby(["season", "round", "session", "mode"],
                                                       sort=False):
        schedule = data.get_schedule(int(season))
        ev = schedule[schedule["RoundNumber"] == rnd]
        if ev.empty:
            continue
        sessions = predict_tab._sessions(ev.iloc[0])
        races = pdata.season_races(int(season)) if session == "Race" else None
        actual = predict_tab._results(sessions, session, int(season), int(rnd), races)
        row = dict(Season=int(season), Round=int(rnd), Event=g["event"].iloc[0],
                   Session=session, Prediction=mode, Saved=g["saved_utc"].iloc[0])
        if actual is None:
            rows.append({**row, "Status": "Awaiting result"})
            continue
        p = g.set_index("driver")
        res = actual.set_index("Driver")
        first = res.index[res["Pos"] == 1]
        fav = p["p_first"].idxmax()
        top3_pred = set(p["p_top3"].nlargest(3).index)
        top3_real = set(res.index[res["classified"] & (res["Pos"] <= 3)])
        p_actual = float(p["p_first"].get(first[0], np.nan)) if len(first) else np.nan
        rows.append({**row, "Status": "Scored",
                     "Winner / pole": first[0] if len(first) else "—",
                     "Model gave": p_actual,
                     "Favourite": fav, "Favourite result": res["Result"].get(fav, "—"),
                     "Top 3 correct": len(top3_pred & top3_real),
                     "Log loss": -np.log(max(p_actual, 1e-3)) if p_actual == p_actual else np.nan})
    return pd.DataFrame(rows)


def render() -> None:
    st.caption("How the predictions have done against real results, and what the model "
               "learned from that.")
    year = data.current_season()
    try:
        with st.spinner("Loading history…"):
            temps, events = predict_tab._calibration_data(year, 99)
    except Exception as exc:
        st.error(f"Could not load the history: {exc}")
        return

    # ------------------------------------------------------------ saved predictions
    st.markdown("##### Saved predictions")
    scored = _score_log()
    if scored.empty:
        st.info("No saved predictions yet. A prediction is saved automatically whenever the "
                "Predictions page shows a session that hasn't started yet; it's scored here "
                "once the result is in.")
    else:
        done = scored[scored["Status"] == "Scored"]
        c = st.columns(4)
        c[0].metric("Sessions scored", len(done), f"{(scored['Status'] != 'Scored').sum()} awaiting",
                    delta_color="off", delta_arrow="off")
        if len(done):
            fav_won = (done["Favourite result"] == "P1").mean()
            c[1].metric("Favourite won", f"{fav_won:.0%}")
            c[2].metric("Top 3 correct", f"{done['Top 3 correct'].mean():.1f} of 3", "average",
                        delta_color="off", delta_arrow="off")
            c[3].metric("Avg. log loss", f"{done['Log loss'].mean():.2f}", "lower is better",
                        delta_color="off", delta_arrow="off")
        st.dataframe(scored.sort_values(["Season", "Round"], ascending=False), hide_index=True,
                     width="stretch",
                     column_config={"Model gave": st.column_config.NumberColumn(format="percent"),
                                    "Log loss": st.column_config.NumberColumn(format="%.2f")})

    _backtest_section()

    # ------------------------------------------------------------ calibration
    st.markdown("##### What the model learned from past results")
    st.caption("Every event of the last seasons is re-predicted using only data from before "
               "it. From those re-predictions the model learns how confident it should be per "
               "session type, and updates this after every new result.")
    rows = []
    for kind in calibration.KINDS:
        t = temps[kind]
        if t["learned"]:
            change = 1 / t["temperature"] - 1
            verdict = ("well calibrated, no correction" if abs(change) < 0.03 else
                       f"was too cautious: probabilities sharpened by {change:.0%}" if change > 0
                       else f"was too confident: probabilities widened by {-change:.0%}")
            rows.append({"Session": KIND_LABEL[kind], "Events": t["n"], "Finding": verdict,
                         "Log loss before": t["logloss_default"], "Log loss after": t["logloss_learned"]})
        else:
            rows.append({"Session": KIND_LABEL[kind], "Events": t["n"],
                         "Finding": "not enough events yet", "Log loss before": np.nan,
                         "Log loss after": np.nan})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Log loss before": st.column_config.NumberColumn(format="%.3f"),
                                "Log loss after": st.column_config.NumberColumn(format="%.3f")})

    table = calibration.calibration_table(events, temps)
    if not table.empty:
        left, right = st.columns([3, 2])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 0.6], y=[0, 0.6], mode="lines", hoverinfo="skip",
                                     line=dict(color="rgba(139,147,161,.45)", width=1, dash="dot"),
                                     name="Perfect calibration"))
            fig.add_trace(go.Scatter(
                x=table["predicted"], y=table["actual"], mode="lines+markers", name="Model",
                line=dict(color="#5B8DEF", width=2), marker=dict(size=9),
                customdata=table["n"],
                hovertemplate="Predicted %{x:.0%} → happened %{y:.0%} (%{customdata} cases)"
                              "<extra></extra>"))
            fig.update_xaxes(title_text="Predicted chance to win / take pole", tickformat=".0%")
            fig.update_yaxes(title_text="How often it happened", tickformat=".0%")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                              legend=dict(orientation="h", y=1.08, x=0))
            st.plotly_chart(fig, width="stretch", key="calibration_chart")
        with right:
            st.caption(f"Based on {len(events)} re-predicted sessions. If the line follows the "
                       "dotted diagonal, a predicted 30% happens about 30% of the time. Points "
                       "above the diagonal mean the model was too cautious, below too confident.")
            show = table.assign(bin=[f"{iv.left:.0%}–{min(iv.right, 1):.0%}" for iv in table["bin"]])
            st.dataframe(show.rename(columns={"bin": "Range", "predicted": "Predicted",
                                              "actual": "Happened", "n": "Cases"}),
                         hide_index=True, width="stretch",
                         column_config={"Predicted": st.column_config.NumberColumn(format="percent"),
                                        "Happened": st.column_config.NumberColumn(format="percent")})


def _event_names(season: int) -> dict:
    try:
        sched = data.get_schedule(season)
        return dict(zip(sched["RoundNumber"].astype(int), sched["EventName"]))
    except Exception:
        return {}


def _backtest_section() -> None:
    st.markdown("##### Backtest: predictions vs. results since 2020")
    st.caption("Every qualifying session, sprint and race since 2020 was predicted using only "
               "data from before it (what the model would have said at the time, including the "
               "confidence correction learned from earlier seasons) and compared with the real "
               "result.")
    events = evaluate.load_store()
    if not events:
        st.info("The backtest hasn't been computed yet; open the Predictions page once.")
        return
    label = st.segmented_control("Session type", list(BACKTEST_KINDS), default="Race (after qualifying)",
                                 key="bt_kind", label_visibility="collapsed")
    kind = BACKTEST_KINDS[label or "Race (after qualifying)"]
    evs = sorted((e for e in events if e["kind"] == kind),
                 key=lambda e: (e["season"], e["round"]), reverse=True)
    if not evs:
        st.info("No events of this type.")
        return

    summary = evaluate.season_summary(evs)
    overall = evaluate.season_summary([{**e, "season": 0} for e in evs])
    table = pd.concat([summary, overall]).drop(columns="kind")
    table["season"] = table["season"].map(lambda v: "All" if v == 0 else str(int(v)))
    st.dataframe(
        table.rename(columns={"season": "Season", "events": "Events", "fav_hit": "Favourite won",
                              "top3_hits": "Top 3 correct", "logloss": "Log loss ↓",
                              "spearman": "Order correlation", "pos_mae": "Position error ↓"}),
        hide_index=True, width="stretch",
        column_config={"Favourite won": st.column_config.NumberColumn(format="percent"),
                       "Top 3 correct": st.column_config.NumberColumn(format="%.2f"),
                       "Log loss ↓": st.column_config.NumberColumn(format="%.3f"),
                       "Order correlation": st.column_config.NumberColumn(format="%.2f"),
                       "Position error ↓": st.column_config.NumberColumn(format="%.2f")})

    names = {s: _event_names(s) for s in {e["season"] for e in evs}}
    pick = st.selectbox(
        "Event", range(len(evs)), key=f"bt_event_{kind}",
        format_func=lambda i: f"{evs[i]['season']} R{evs[i]['round']} · "
                              f"{names[evs[i]['season']].get(evs[i]['round'], evs[i]['circuit'])}")
    e = evs[pick]
    prefix = "cal_" if "cal_p_first" in e else ""
    first_col, top3_col = ("Pole", "Top 3") if kind == "quali" else ("Win", "Podium")
    codes = [c if isinstance(c, str) and c else d for c, d in zip(e["codes"], e["drivers"])]
    result = pd.DataFrame({"Driver": codes, "Team": [_team_name(t) for t in e["teams"]],
                           first_col: e[prefix + "p_first"], top3_col: e[prefix + "p_top3"],
                           "Expected pos.": e[prefix + "exp_pos"]})
    labels = [f"P{int(f)}" if c else "DNF" for f, c in zip(e["finish"], e["classified"])]
    actual = pd.DataFrame({"Driver": codes, "Pos": e["finish"], "Result": labels,
                           "classified": e["classified"]})
    result = result.sort_values(first_col, ascending=False).reset_index(drop=True)
    entries = pd.DataFrame({"Driver": codes,
                            "TeamColor": [TEAM_COLORS.get(t, "") for t in e["teams"]]})
    predict_tab._result_check(result, actual, first_col, top3_col,
                              "pole" if kind == "quali" else "winner", entries, key=f"bt_{kind}")
    shown = predict_tab._with_result(result, actual)
    st.dataframe(predict_tab._as_pct(shown, [first_col, top3_col]), hide_index=True,
                 width="stretch", height=predict_tab.ui.table_height(len(shown)),
                 row_height=predict_tab.ui.ROW_HEIGHT,
                 column_config={**predict_tab._pct(shown, [first_col, top3_col]),
                                "Expected pos.": st.column_config.NumberColumn(format="%.1f")})
