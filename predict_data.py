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

HISTORY_SEASONS = 4  # default training window: target season + 3 previous ones
DATA_START = 2018    # first season with FastF1 timing data; history is built from here
GAP_CAP = 5.0        # % — wet sessions / no-time laps are capped so they don't dominate

_RESULTS_DIR = data._CACHE_DIR / "dashboard" / "results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def _stored(kind: str, year: int, fetch) -> pd.DataFrame:
    """Completed seasons never change: keep them on disk instead of re-downloading."""
    path = _RESULTS_DIR / f"{kind}_{year}.pkl"
    complete = year < data.current_season()
    if complete and path.exists():
        return pd.read_pickle(path)
    df = fetch()
    if complete and len(df):
        df.to_pickle(path)
    return df


def _gap_pct(times: pd.Series) -> pd.Series:
    """Gap to the fastest time in %, capped (NaN where there is no time)."""
    secs = pd.to_timedelta(times).dt.total_seconds()
    return ((secs / secs.min() - 1) * 100).clip(upper=GAP_CAP)


@st.cache_data(show_spinner=False, ttl=3 * 3600)
def season_races(year: int) -> pd.DataFrame:
    def fetch():
        ergast = Ergast(result_type="pandas", auto_cast=True, limit=100)
        df = _all_pages(_retry(lambda: ergast.get_race_results(season=year)))
        if df.empty:
            return df
        keep = ["season", "round", "circuitId", "raceName", "raceDate", "lat", "long",
                "driverId", "driverCode", "givenName", "familyName", "constructorId",
                "constructorName", "grid", "position", "positionText", "status", "points",
                "fastestLapTime"]
        df = df[[c for c in keep if c in df]].copy()
        # race pace: each driver's fastest lap relative to the fastest lap of the race
        df["flap_gap"] = (df.groupby("round")["fastestLapTime"].transform(_gap_pct)
                          if "fastestLapTime" in df else float("nan"))
        return df.drop(columns=["fastestLapTime"], errors="ignore")
    return _stored("races", year, fetch)


@st.cache_data(show_spinner=False, ttl=3 * 3600)
def season_quali(year: int) -> pd.DataFrame:
    def fetch():
        ergast = Ergast(result_type="pandas", auto_cast=True, limit=100)
        df = _all_pages(_retry(lambda: ergast.get_qualifying_results(season=year)))
        if df.empty:
            return df
        segs = [c for c in ("Q1", "Q2", "Q3") if c in df]
        df = df.copy()
        # one-lap pace: each driver's best qualifying time vs. the session's best
        df["best"] = df[segs].apply(lambda r: pd.to_timedelta(r).min(), axis=1)
        df["qgap"] = df.groupby("round")["best"].transform(_gap_pct)
        return df[["season", "round", "driverId", "constructorId", "position", "qgap"]].rename(
            columns={"position": "quali"})
    return _stored("quali", year, fetch)


@st.cache_data(show_spinner=False, ttl=3 * 3600)
def season_sprints(year: int) -> pd.DataFrame:
    """Sprint results; the sprint grid is the Sprint Qualifying classification."""
    def fetch():
        ergast = Ergast(result_type="pandas", auto_cast=True, limit=100)
        df = _all_pages(_retry(lambda: ergast.get_sprint_results(season=year)))
        if df.empty:
            return df
        return df[["season", "round", "driverId", "grid", "position", "positionText"]].rename(
            columns={"grid": "sq", "position": "sprint"})
    return _stored("sprints", year, fetch)


def history(target_year: int, start: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Race and qualifying results from `start` (default: the training window) up to
    and including the target season."""
    first = start if start is not None else target_year - HISTORY_SEASONS + 1
    years = range(max(first, 1950), target_year + 1)
    races = pd.concat([season_races(y) for y in years], ignore_index=True)
    quali = pd.concat([season_quali(y) for y in years], ignore_index=True)
    return races, quali


def sprints_range(first: int, last: int) -> pd.DataFrame:
    years = range(max(first, 2021), last + 1)  # sprints exist from 2021
    frames = [season_sprints(y) for y in years]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["season", "round", "driverId", "sq", "sprint", "positionText"])


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
def practice_pace(year: int, round_number: int, name: str) -> pd.DataFrame | None:
    """Best-lap rank and gap (%) per driver in a practice session, indexed by driverId.

    Only used for past sessions, so results are kept on disk and restarts are fast.
    """
    store = _RANK_DIR / f"{year}_{round_number}_{name.replace(' ', '')}.csv"
    if store.exists():
        saved = pd.read_csv(store, index_col=0)
        if "gap" in saved:  # files from before gaps were stored are refreshed below
            return saved if len(saved) else None
    pace = _practice_pace(year, round_number, name)
    (pace if pace is not None else pd.DataFrame(columns=["rank", "gap"])).to_csv(store)
    return pace


def practice_rank(year: int, round_number: int, name: str) -> pd.Series | None:
    pace = practice_pace(year, round_number, name)
    return None if pace is None else pace["rank"]


_RANK_DIR = data._CACHE_DIR / "dashboard" / "practice_rank"
_RANK_DIR.mkdir(parents=True, exist_ok=True)


def laps_pace(laps: pd.DataFrame) -> pd.DataFrame | None:
    """Rank and gap (%) of each driver's best lap, indexed by abbreviation."""
    best = laps.dropna(subset=["LapTime"]).groupby("Driver")["LapTime"].min()
    if len(best) < 10:
        return None
    return pd.DataFrame({"rank": best.rank(), "gap": _gap_pct(best)})


def _practice_pace(year: int, round_number: int, name: str) -> pd.DataFrame | None:
    try:
        session = fastf1.get_session(year, round_number, name)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        pace = laps_pace(session.laps)
    except Exception:  # no such session (e.g. 2020's two-day weekends) or no lap data
        return None
    if pace is None:
        return None
    ids = session.results.set_index("Abbreviation")["DriverId"]
    pace.index = pace.index.map(lambda a: ids.get(a) or a.lower())
    return pace


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
    try:
        status = session.session_status
    except Exception:  # not published yet (e.g. session still running): not finished
        return False
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
