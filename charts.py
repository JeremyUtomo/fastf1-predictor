"""Plotly figure builders for the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FALLBACK_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]

COMPOUND_COLORS = {
    "SOFT": "#E8002D",
    "MEDIUM": "#FFD12E",
    "HARD": "#EBEBEB",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0067FF",
    "UNKNOWN": "#8A8A8A",
    "TEST_UNKNOWN": "#8A8A8A",
}

LINE_WIDTH = 2
ACCENT = "#5B8DEF"   # neutral single-series colour (red is reserved for the UI accent)
_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
               legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))


def driver_styles(session, drivers=None) -> dict[str, dict]:
    """Map driver abbreviation -> {'color', 'dash'}.

    Colour follows the team, so a driver keeps the same colour across every
    chart; the second driver of a team gets a dashed line to stay distinguishable.
    """
    res = session.results
    styles: dict[str, dict] = {}
    seen_teams: dict[str, int] = {}
    for i, (_, row) in enumerate(res.iterrows()):
        abbr = row["Abbreviation"]
        color = row.get("TeamColor")
        color = f"#{color}" if isinstance(color, str) and color else FALLBACK_COLORS[i % 6]
        n = seen_teams.get(row["TeamName"], 0)
        seen_teams[row["TeamName"]] = n + 1
        styles[abbr] = {"color": color, "dash": "solid" if n == 0 else "dash"}
    if drivers is not None:
        for d in drivers:
            styles.setdefault(d, {"color": "#8A8A8A", "dash": "solid"})
    return styles


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# --------------------------------------------------------------------- laps
def lap_time_chart(laps: pd.DataFrame, styles: dict, drivers: list[str]) -> go.Figure:
    fig = go.Figure()
    for drv in drivers:
        d = laps[laps["Driver"] == drv].dropna(subset=["LapTime"])
        if d.empty:
            continue
        st = styles[drv]
        fig.add_trace(go.Scatter(
            x=d["LapNumber"], y=d["LapTime"].dt.total_seconds(), name=drv,
            mode="lines+markers",
            line=dict(color=st["color"], dash=st["dash"], width=LINE_WIDTH),
            marker=dict(size=7, color=[
                COMPOUND_COLORS.get(c, "#8A8A8A") for c in d["Compound"].fillna("UNKNOWN")],
                line=dict(color=st["color"], width=2)),
            customdata=np.stack([
                d["LapTime"].map(_fmt).values, d["Compound"].fillna("?").values,
                d["TyreLife"].fillna(0).astype(int).values], axis=-1),
            hovertemplate="%{customdata[0]} · %{customdata[1]} (%{customdata[2]} laps old)",
        ))
    if not fig.data:
        return _empty("No timed laps for this selection")
    fig.update_layout(**_LAYOUT, height=480, xaxis_title="Lap",
                      yaxis_title="Lap time (s)", yaxis_tickformat=".1f")
    fig.add_annotation(xref="paper", yref="paper", x=1, y=1.12, showarrow=False,
                       text="Marker fill = tyre compound", font=dict(size=11),
                       xanchor="right")
    return fig


def pace_box(laps: pd.DataFrame, styles: dict, drivers: list[str]) -> go.Figure:
    data = laps[laps["Driver"].isin(drivers)].dropna(subset=["LapTime"])
    if data.empty:
        return _empty("No timed laps for this selection")
    order = (data.groupby("Driver")["LapTime"].median().sort_values().index.tolist())
    fig = go.Figure()
    for drv in order:
        d = data[data["Driver"] == drv]
        fig.add_trace(go.Box(
            y=d["LapTime"].dt.total_seconds(), name=drv, boxmean=True,
            line=dict(color=styles[drv]["color"], width=LINE_WIDTH),
            fillcolor="rgba(0,0,0,0)", showlegend=False, marker=dict(size=4)))
    fig.update_layout(margin=_LAYOUT["margin"], height=420, yaxis_title="Lap time (s)",
                      xaxis_title="Driver (fastest median pace → slowest)")
    return fig


def position_chart(laps: pd.DataFrame, styles: dict, drivers: list[str]) -> go.Figure:
    fig = go.Figure()
    for drv in drivers:
        d = laps[laps["Driver"] == drv].dropna(subset=["Position"])
        if d.empty:
            continue
        st = styles[drv]
        fig.add_trace(go.Scatter(
            x=d["LapNumber"], y=d["Position"], name=drv, mode="lines",
            line=dict(color=st["color"], dash=st["dash"], width=LINE_WIDTH)))
    if not fig.data:
        return _empty("No position data")
    fig.update_layout(**_LAYOUT, height=480, xaxis_title="Lap",
                      yaxis=dict(title="Position", autorange="reversed", dtick=1))
    return fig


def strategy_chart(laps: pd.DataFrame, order: list[str]) -> go.Figure:
    fig = go.Figure()
    stints = (laps.groupby(["Driver", "Stint", "Compound"], dropna=False)
              .agg(start=("LapNumber", "min"), end=("LapNumber", "max")).reset_index())
    if stints.empty:
        return _empty("No stint data")
    stints["Compound"] = stints["Compound"].fillna("UNKNOWN")
    stints["length"] = stints["end"] - stints["start"] + 1
    shown = set()
    for comp, g in stints.groupby("Compound"):
        fig.add_trace(go.Bar(
            y=g["Driver"], x=g["length"], base=g["start"] - 1, orientation="h",
            name=comp.title(), legendgroup=comp, showlegend=comp not in shown,
            marker=dict(color=COMPOUND_COLORS.get(comp, "#8A8A8A"),
                        line=dict(color="rgba(128,128,128,.8)", width=1)),
            customdata=np.stack([g["start"], g["end"], g["length"]], axis=-1),
            hovertemplate="Laps %{customdata[0]}–%{customdata[1]} "
                          "(%{customdata[2]} laps)<extra>" + comp.title() + "</extra>"))
        shown.add(comp)
    fig.update_layout(
        barmode="overlay", height=max(360, 26 * len(order) + 120),
        margin=_LAYOUT["margin"], legend=_LAYOUT["legend"],
        xaxis_title="Lap", yaxis=dict(categoryorder="array", categoryarray=order,
                                      autorange="reversed"))
    return fig


# ---------------------------------------------------------------- telemetry
_CHANNELS = [
    ("Speed", "Speed (km/h)", 3), ("Throttle", "Throttle (%)", 1.4),
    ("Brake", "Brake", 1.0), ("nGear", "Gear", 1.2), ("RPM", "RPM", 1.4),
    ("DRS", "DRS", 0.9),
]


def align_distances(traces: list[dict]) -> list[dict]:
    """Rescale every trace's Distance to the first trace's lap length.

    Distance is integrated from speed, so each lap drifts by a few metres. Without this,
    overlays don't line up and the time delta gets spurious spikes in slow corners.
    """
    ref_len = traces[0]["tel"]["Distance"].max()
    for t in traces[1:]:
        tel = t["tel"].copy()
        tel["Distance"] = tel["Distance"] * ref_len / tel["Distance"].max()
        t["tel"] = tel
    return traces


def _delta(ref: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    """Time gap (s) of `other` to `ref` at each reference distance point."""
    t_ref = ref["Time"].dt.total_seconds().values
    t_oth = other["Time"].dt.total_seconds().values
    return np.interp(ref["Distance"].values, other["Distance"].values, t_oth) - t_ref


def telemetry_chart(traces: list[dict], corners: pd.DataFrame | None,
                    channels: list[str]) -> go.Figure:
    """`traces`: dicts with keys label, tel, color, dash. First one is the reference."""
    rows = [("Delta", f"Δ vs {traces[0]['label']} (s)<br>+ = ahead", 1.6)] + [c for c in _CHANNELS if c[0] in channels]
    rows = rows if len(traces) > 1 else rows[1:]
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[r[2] for r in rows])
    ref = traces[0]["tel"]
    legend_shown: set[str] = set()
    for r, (key, title, _) in enumerate(rows, start=1):
        for i, t in enumerate(traces):
            tel = t["tel"]
            if key == "Delta":
                if i == 0:
                    continue
                x, y = ref["Distance"], -_delta(ref, tel)  # positive = ahead of reference
            elif key not in tel:
                continue
            else:
                x = tel["Distance"]
                y = (tel[key] >= 10).astype(int) if key == "DRS" else tel[key].astype(float)
            fig.add_trace(go.Scatter(
                x=x, y=y, name=t["label"], legendgroup=t["label"],
                showlegend=t["label"] not in legend_shown, mode="lines",
                line=dict(color=t["color"], dash=t["dash"], width=LINE_WIDTH,
                          shape="hv" if key in ("Brake", "DRS", "nGear") else "linear")),
                row=r, col=1)
            legend_shown.add(t["label"])
        fig.update_yaxes(title_text=title, row=r, col=1, title_font=dict(size=11))
    if len(traces) > 1:
        fig.add_hline(y=0, line=dict(color="rgba(128,128,128,.6)", width=1), row=1, col=1)
    if corners is not None and len(corners):
        for _, c in corners.iterrows():
            fig.add_vline(x=c["Distance"], line=dict(color="rgba(128,128,128,.35)",
                                                     width=1, dash="dot"))
            fig.add_annotation(x=c["Distance"], y=1, xref="x", yref="paper", yshift=8,
                               text=f"{c['Number']}{c['Letter']}", showarrow=False,
                               font=dict(size=10))
    fig.update_xaxes(title_text="Distance (m)", row=len(rows), col=1)
    fig.update_layout(height=170 * len(rows) + 90, hovermode="x unified",
                      margin=dict(l=10, r=10, t=70, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0))
    return fig


def track_speed_map(tel: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=tel["X"], y=tel["Y"], mode="markers",
        marker=dict(size=6, color=tel["Speed"], colorscale="Viridis", showscale=True,
                    colorbar=dict(title="km/h", thickness=12)),
        customdata=tel["Speed"], hovertemplate="%{customdata:.0f} km/h<extra></extra>"))
    return _map_layout(fig, title)


def dominance_map(traces: list[dict], n_segments: int = 25) -> go.Figure:
    """Colour each mini-sector by the driver with the highest mean speed in it."""
    ref = traces[0]["tel"]
    edges = np.linspace(0, ref["Distance"].max() + 1e-6, n_segments + 1)
    speeds = []
    for t in traces:
        tel = t["tel"]
        seg = pd.cut(tel["Distance"], edges, labels=False, include_lowest=True)
        speeds.append(tel.groupby(seg)["Speed"].mean().reindex(range(n_segments)))
    best = np.nanargmax(np.vstack([s.values for s in speeds]), axis=0)
    ref_seg = pd.cut(ref["Distance"], edges, labels=False, include_lowest=True)
    fig = go.Figure()
    for idx, t in enumerate(traces):
        xs, ys = [], []
        for seg in range(n_segments):
            if best[seg] != idx:
                continue
            part = ref[ref_seg == seg]
            # overlap by one point so adjacent segments join up visually
            nxt = ref[ref_seg == seg + 1].head(1)
            part = pd.concat([part, nxt])
            xs += list(part["X"]) + [None]
            ys += list(part["Y"]) + [None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=t["label"],
                                 line=dict(color=t["color"], width=6), hoverinfo="skip"))
    return _map_layout(fig, "Quickest per mini-sector (mean speed)")


def _map_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=460,
                      margin=dict(l=10, r=10, t=50, b=40),
                      legend=dict(orientation="h", yanchor="top", y=-0.02, x=0.5, xanchor="center"))
    return fig


# ------------------------------------------------------------------ weather
def weather_chart(weather: pd.DataFrame) -> go.Figure:
    w = weather.copy()
    w["minutes"] = w["Time"].dt.total_seconds() / 60
    panels = [
        ("Temperature (°C)", [("AirTemp", "Air", "#4C78A8"), ("TrackTemp", "Track", "#E45756")]),
        ("Humidity (%)", [("Humidity", "Humidity", "#54A24B")]),
        ("Wind speed (m/s)", [("WindSpeed", "Wind", "#B279A2")]),
        ("Pressure (mbar)", [("Pressure", "Pressure", "#F58518")]),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[p[0] for p in panels],
                        vertical_spacing=0.16)
    for i, (_, series) in enumerate(panels):
        r, c = divmod(i, 2)
        for col, name, color in series:
            fig.add_trace(go.Scatter(
                x=w["minutes"], y=w[col], name=name, mode="lines",
                line=dict(color=color, width=LINE_WIDTH), showlegend=len(series) > 1),
                row=r + 1, col=c + 1)
    # shade rain periods
    if w["Rainfall"].any():
        rain = w["Rainfall"].astype(bool)
        start = None
        for t, wet in zip(w["minutes"], rain):
            if wet and start is None:
                start = t
            if not wet and start is not None:
                fig.add_vrect(x0=start, x1=t, fillcolor="#0067FF", opacity=0.15, line_width=0)
                start = None
        if start is not None:
            fig.add_vrect(x0=start, x1=w["minutes"].iloc[-1], fillcolor="#0067FF",
                          opacity=0.15, line_width=0)
    fig.update_xaxes(title_text="Minutes since session data start")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10),
                      hovermode="x unified", legend=_LAYOUT["legend"])
    return fig


# ---------------------------------------------------------------- standings
def standings_bar(df: pd.DataFrame, label_col: str, colors: list[str] | None = None,
                  title: str = "") -> go.Figure:
    d = df.sort_values("points")
    fig = go.Figure(go.Bar(
        x=d["points"], y=d[label_col], orientation="h",
        marker=dict(color=colors if colors else ACCENT),
        text=d["points"], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x} pts<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(size=14)) if title else None,
                      height=max(300, 24 * len(d) + 60),
                      margin=dict(l=10, r=40, t=40 if title else 8, b=10),
                      xaxis_title="Points")
    return fig


def _fmt(td) -> str:
    if pd.isna(td):
        return ""
    s = td.total_seconds()
    return f"{int(s // 60)}:{s % 60:06.3f}"
