"""Data collection for the prediction model.

Historical results come from the Jolpica-Ergast API (light, one request per 100
rows); wet/dry race detection and the current weekend come from FastF1 timing
data; the weather forecast comes from Open-Meteo (free, no API key).
"""
from __future__ import annotations

import time

import pandas as pd
import requests
import streamlit as st

import fastf1
from fastf1.ergast import Ergast
from fastf1.exceptions import ErgastInvalidRequestError

import data  # noqa: F401  (configures the FastF1 cache)

HISTORY_SEASONS = 4  # target season + 3 previous ones


def _retry(call, attempts: int = 5):
    """Call `call()`, backing off when the Jolpica rate limit is hit."""
    for i in range(attempts):
        try:
            return call()
        except ErgastInvalidRequestError as exc:
            if "Too Many Requests" not in str(exc) or i == attempts - 1:
                raise
            time.sleep(2 ** (i + 1))


def _all_pages(response) -> pd.DataFrame:
    """Flatten a paginated multi-race Ergast response into one row per entry."""
    frames = []
    while True:
        for (_, race), content in zip(response.description.iterrows(), response.content):
            frame = content.copy()
            for col in ("season", "round", "circuitId", "raceName", "raceDate", "lat", "long"):
                if col in race:
                    frame[col] = race[col]
            frames.append(frame)
        try:
            # note: `is_complete` is False for every page after the first one
            response = _retry(response.get_next_result_page)
        except ValueError:  # no more pages
            break
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # a race can be split over two pages; keep each entry once
    return out.drop_duplicates(subset=["season", "round", "driverId"])


@st.cache_data(show_spinner=False, ttl=3 * 3600)
def season_races(year: int) -> pd.DataFrame:
    ergast = Ergast(result_type="pandas", auto_cast=True, limit=100)
    df = _all_pages(_retry(lambda: ergast.get_race_results(season=year)))
    if df.empty:
        return df
    keep = ["season", "round", "circuitId", "raceName", "raceDate", "lat", "long", "driverId",
            "driverCode", "givenName", "familyName", "constructorId", "constructorName",
            "grid", "position", "positionText", "status", "points"]
    return df[keep]


@st.cache_data(show_spinner=False, ttl=3 * 3600)
def season_quali(year: int) -> pd.DataFrame:
    ergast = Ergast(result_type="pandas", auto_cast=True, limit=100)
    df = _all_pages(_retry(lambda: ergast.get_qualifying_results(season=year)))
    if df.empty:
        return df
    return df[["season", "round", "driverId", "constructorId", "position"]].rename(
        columns={"position": "quali"})


@st.cache_data(show_spinner=False, ttl=3 * 3600)
def season_sprints(year: int) -> pd.DataFrame:
    """Sprint results; the sprint grid is the Sprint Qualifying classification."""
    ergast = Ergast(result_type="pandas", auto_cast=True, limit=100)
    df = _all_pages(_retry(lambda: ergast.get_sprint_results(season=year)))
    if df.empty:
        return df
    return df[["season", "round", "driverId", "grid", "position", "positionText"]].rename(
        columns={"grid": "sq", "position": "sprint"})


def history(target_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Race and qualifying results for the target season and the ones before it."""
    years = range(target_year - HISTORY_SEASONS + 1, target_year + 1)
    races = pd.concat([season_races(y) for y in years], ignore_index=True)
    quali = pd.concat([season_quali(y) for y in years], ignore_index=True)
    return races, quali


@st.cache_data(show_spinner=False, ttl=30 * 24 * 3600)
def race_rain_fraction(year: int, round_number: int) -> float | None:
    """Share of weather samples with rain during a race (None if unavailable)."""
    try:
        session = fastf1.get_session(year, round_number, "Race")
        session.load(laps=False, telemetry=False, weather=True, messages=False)
        return float(session.weather_data["Rainfall"].mean())
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=1800)
def forecast(lat: float, lon: float, start_utc: pd.Timestamp, hours: int = 2) -> dict | None:
    """Hourly forecast summary for the race window, or None if out of range/unavailable."""
    day = start_utc.date()
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=dict(latitude=lat, longitude=lon, timezone="UTC",
                        hourly="precipitation_probability,precipitation,temperature_2m,"
                               "wind_speed_10m",
                        start_date=str(day), end_date=str(day)),
            timeout=10)
        resp.raise_for_status()
        hourly = pd.DataFrame(resp.json()["hourly"])
    except Exception:
        return None
    hourly["time"] = pd.to_datetime(hourly["time"])
    window = hourly[(hourly["time"] >= start_utc.floor("h"))
                    & (hourly["time"] <= start_utc + pd.Timedelta(hours=hours))]
    if window.empty or window["precipitation_probability"].isna().all():
        return None
    return dict(rain_prob=float(window["precipitation_probability"].max()) / 100,
                rain_mm=float(window["precipitation"].sum()),
                temp=float(window["temperature_2m"].mean()),
                wind=float(window["wind_speed_10m"].mean()))


@st.cache_data(show_spinner=False, ttl=30 * 24 * 3600)
def practice_rank(year: int, round_number: int, name: str) -> pd.Series | None:
    """Rank of each driver's best lap in a practice session, indexed by driverId.

    Only used for past sessions, so results are kept on disk and restarts are fast.
    """
    store = _RANK_DIR / f"{year}_{round_number}_{name.replace(' ', '')}.csv"
    if store.exists():
        saved = pd.read_csv(store, index_col=0).iloc[:, 0]
        return saved if len(saved) else None
    rank = _practice_rank(year, round_number, name)
    (rank if rank is not None else pd.Series(dtype=float)).to_csv(store)
    return rank


_RANK_DIR = data._CACHE_DIR / "dashboard" / "practice_rank"
_RANK_DIR.mkdir(parents=True, exist_ok=True)


def _practice_rank(year: int, round_number: int, name: str) -> pd.Series | None:
    try:
        session = fastf1.get_session(year, round_number, name)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception:
        return None
    best = session.laps.dropna(subset=["LapTime"]).groupby("Driver")["LapTime"].min()
    if len(best) < 10:
        return None
    ids = session.results.set_index("Abbreviation")["DriverId"]
    rank = best.rank()
    rank.index = rank.index.map(lambda a: ids.get(a) or a.lower())
    return rank


@st.cache_resource(show_spinner=False, max_entries=8)
def weekend_session(year: int, round_number: int, name: str, laps: bool):
    session = fastf1.get_session(year, round_number, name)
    # race control messages are needed to compute quali results from timing data
    # (deleted laps) when Ergast hasn't published the classification yet
    session.load(laps=laps, telemetry=False, weather=False, messages=laps)
    return session


FRESH_HOURS = 3   # a session's data can still change this long after it started
FRESH_EVERY = 300  # seconds between re-downloads of a recent session


def is_settled(start: pd.Timestamp | None) -> bool:
    return start is None or data.now_utc() > start + pd.Timedelta(hours=FRESH_HOURS)


@st.cache_resource(show_spinner=False, max_entries=6, ttl=FRESH_EVERY)
def _fresh_session(year: int, round_number: int, name: str, laps: bool, bucket: int):
    # FastF1's disk cache never expires: a session loaded while it's running or
    # shortly after would be stored incomplete for good, so bypass it entirely.
    session = fastf1.get_session(year, round_number, name)
    with fastf1.Cache.disabled():
        session.load(laps=laps, telemetry=False, weather=False, messages=laps)
    return session


def load_weekend_session(year: int, round_number: int, name: str, laps: bool,
                         start: pd.Timestamp | None):
    """Session of the current weekend; recent sessions are re-downloaded every few minutes."""
    if is_settled(start):
        return weekend_session(year, round_number, name, laps)
    bucket = int(data.now_utc().timestamp() // FRESH_EVERY)
    return _fresh_session(year, round_number, name, laps, bucket)


def session_finished(session) -> bool:
    """True once the official session status says it's over (results are final)."""
    status = getattr(session, "session_status", None)
    return status is not None and status["Status"].isin(["Finished", "Finalised", "Ends"]).any()


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def race_info(year: int, round_number: int) -> dict | None:
    """Circuit id and coordinates of a round."""
    schedule = _retry(lambda: Ergast().get_race_schedule(season=year))
    row = schedule[schedule["round"] == round_number]
    if row.empty:
        return None
    row = row.iloc[0]
    return dict(circuitId=row["circuitId"], circuitName=row["circuitName"],
                lat=float(row["lat"]), lon=float(row["long"]))
