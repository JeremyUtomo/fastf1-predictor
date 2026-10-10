"""Track record page: saved predictions vs. results, and what the model learned."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import calibration
import data
import predict_data as pdata
import predict_tab

KIND_LABEL = {"quali": "Qualifying", "sprint": "Sprint", "race": "Race"}


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
