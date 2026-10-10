"""Cached data access for the dashboard.

Everything that talks to FastF1 lives here so the UI code stays free of
loading/caching concerns.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

import fastf1
from fastf1.ergast import Ergast

FIRST_SEASON = 2018  # first season with full timing/telemetry coverage

_CACHE_DIR = Path(os.environ.get("FASTF1_CACHE", Path.home() / ".cache" / "fastf1"))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
if hasattr(fastf1.Cache, "configure"):  # FastF1 >= 3.9
    fastf1.Cache.configure(cache_dir=str(_CACHE_DIR))
else:
    fastf1.Cache.enable_cache(str(_CACHE_DIR))
fastf1.logger.set_log_level("ERROR")


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)


def current_season() -> int:
    return now_utc().year


@st.cache_data(show_spinner=False, ttl=3600)
def get_schedule(year: int) -> pd.DataFrame:
    return fastf1.get_event_schedule(year, include_testing=True)


def event_sessions(event: pd.Series) -> list[tuple[str, bool]]:
    """Sessions of an event as (name, has_already_happened) tuples."""
    now = now_utc()
    out = []
    for i in range(1, 6):
        name = event.get(f"Session{i}")
        if not isinstance(name, str) or not name:
            continue
        date = event.get(f"Session{i}DateUtc")
        done = pd.notna(date) and date < now
        out.append((name, bool(done)))
    return out


@st.cache_resource(show_spinner=False, max_entries=4)
def load_session(year: int, round_number: int, session_name: str, telemetry: bool):
    session = fastf1.get_session(year, round_number, session_name)
    start = session.date  # scheduled start, UTC
    if pd.notna(start) and now_utc() < pd.Timestamp(start) + pd.Timedelta(hours=3):
        # FastF1's disk cache never expires; don't let it keep a still-incomplete session
        with fastf1.Cache.disabled():
            session.load(laps=True, telemetry=telemetry, weather=True, messages=True)
    else:
        session.load(laps=True, telemetry=telemetry, weather=True, messages=True)
    return session


@st.cache_data(show_spinner=False, ttl=3600)
def get_standings(year: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Driver standings, constructor standings, and the round they are after."""
    ergast = Ergast()
    drivers = ergast.get_driver_standings(season=year)
    constructors = ergast.get_constructor_standings(season=year)
    rnd = int(drivers.description["round"].iloc[0])
    return drivers.content[0], constructors.content[0], rnd


def session_kind(name: str) -> str:
    """Coarse session type: race, qualifying or practice."""
    lowered = name.lower()
    if lowered in ("race", "sprint"):
        return "race"
    if "qualifying" in lowered or "shootout" in lowered:
        return "qualifying"
    return "practice"


def fmt_laptime(value) -> str:
    """Format a Timedelta/seconds value as m:ss.mmm (empty if missing)."""
    if pd.isna(value):
        return ""
    seconds = value.total_seconds() if hasattr(value, "total_seconds") else float(value)
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes)}:{sec:06.3f}"
