"""Learning from results: calibration and the prediction log.

Self-calibration
    Every past event is re-predicted walk-forward (a model fitted only on earlier
    events) and compared with what happened. From that, one "temperature" per
    session type is learned: below 1 means the model was too cautious (results
    were more predictable than it thought), above 1 too confident. It is shrunk
    towards 1 while there are few events, and recalculated whenever new results
    come in. Re-predictions are stored on disk, so only new events cost time.

Prediction log
    Predictions shown before a session starts are saved, so they can be compared
    with the real result afterwards (see the Track record page).
"""
from __future__ import annotations

import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import data
import model

_DIR = data._CACHE_DIR / "dashboard"
_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_FILE = _DIR / "calibration_events.pkl"
LOG_FILE = _DIR / "prediction_log.csv"

KINDS = ("quali", "sprint", "race")
DEFAULT_TEMPERATURE = {"quali": 1.0, "sprint": 0.85, "race": 1.0}
TEMPS = np.round(np.arange(0.5, 2.01, 0.05), 2)
SHRINK = 5           # events needed before the learned value counts half
MIN_EVENTS = 6
SEASONS_BACK = 2     # learn from the current and the two previous seasons


# ------------------------------------------------------------ re-predictions
def _load_events() -> list[dict]:
    try:
        return pickle.loads(EVENTS_FILE.read_bytes())
    except (OSError, pickle.PickleError, EOFError):
        return []


def walk_forward_events(hist: pd.DataFrame, qrows: pd.DataFrame, sprints: pd.DataFrame,
                        until: tuple[int, int], progress=None) -> list[dict]:
    """Walk-forward re-predictions of all events before `until` (cached on disk).

    Each event stores the model strengths (before temperature) and which driver
    actually won / took pole, which is all that's needed to learn a temperature.
    """
    events = _load_events()
    have = {(e["kind"], e["season"], e["round"]) for e in events}
    first = until[0] - SEASONS_BACK
    keys = (hist[hist["season"] >= first][["season", "round"]].drop_duplicates()
            .sort_values(["season", "round"]))
    todo = [(int(s), int(r)) for s, r in keys.itertuples(index=False)
            if (s, r) < until and any((k, s, r) not in have for k in KINDS)]
    added = False
    for i, (season, rnd) in enumerate(todo):
        if progress:
            progress((i + 1) / len(todo))
        before = (hist["season"] < season) | ((hist["season"] == season) & (hist["round"] < rnd))
        past = hist[before]
        if past.groupby(["season", "round"]).ngroups < 10:
            continue
        race = hist[(hist["season"] == season) & (hist["round"] == rnd) & hist["started"]]
        fit = None
        if ("race", season, rnd) not in have and (race["finish"] == 1).any():
            fit = model.fit_rank(past, model.FEATURES_POST)
            events.append(dict(kind="race", season=season, round=rnd,
                               strength=fit.strength(race) / race["chaos"].to_numpy(),
                               drivers=list(race["driverId"]),
                               first=int(np.argmax(race["finish"].to_numpy() == 1))))
        sp = sprints[(sprints["season"] == season) & (sprints["round"] == rnd)]
        if ("sprint", season, rnd) not in have and len(sp) >= 15:
            x = race.merge(sp[["driverId", "sq", "sprint"]], on="driverId")
            x = x[x["sq"] > 0]
            if (x["sprint"] == 1).any():
                fit = fit or model.fit_rank(past, model.FEATURES_POST)
                x = x.assign(grid=x["sq"].astype(float), grid_log=np.log(x["sq"].astype(float)))
                events.append(dict(kind="sprint", season=season, round=rnd,
                                   strength=fit.strength(x) / x["chaos"].to_numpy(),
                                   drivers=list(x["driverId"]),
                                   first=int(np.argmax(x["sprint"].to_numpy() == 1))))
        wk = qrows[(qrows["season"] == season) & (qrows["round"] == rnd)]
        if ("quali", season, rnd) not in have and len(wk) >= 15:
            feats = model.QUALI_VARIANTS[model.weekend_variant(wk)]
            qbefore = (qrows["season"] < season) | ((qrows["season"] == season) & (qrows["round"] < rnd))
            train = qrows[qbefore].dropna(subset=feats)
            if train.groupby(["season", "round"]).ngroups >= 8:
                f = model.fit_rank(train, feats)
                events.append(dict(kind="quali", season=season, round=rnd,
                                   strength=f.strength(model.fill_missing(wk, feats)),
                                   drivers=list(wk["driverId"]),
                                   first=int(np.argmin(wk["finish"].to_numpy()))))
        added = True
    if added:
        EVENTS_FILE.write_bytes(pickle.dumps(events))
    return [e for e in events if (e["season"], e["round"]) < until and e["season"] >= first]


def _first_logp(strength: np.ndarray, idx: int, temp: float) -> float:
    s = strength / temp
    s = s - s.max()
    return float(s[idx] - np.log(np.exp(s).sum()))


def win_probs(strength: np.ndarray, temp: float) -> np.ndarray:
    s = strength / temp
    e = np.exp(s - s.max())
    return e / e.sum()


def temperatures(events: list[dict]) -> dict[str, dict]:
    """Learned temperature per session type, with the evidence behind it."""
    out = {}
    for kind in KINDS:
        ev = [e for e in events if e["kind"] == kind]
        default = DEFAULT_TEMPERATURE[kind]
        if len(ev) < MIN_EVENTS:
            out[kind] = dict(temperature=default, n=len(ev), learned=False)
            continue
        losses = [-np.mean([_first_logp(e["strength"], e["first"], t) for e in ev]) for t in TEMPS]
        best = float(TEMPS[int(np.argmin(losses))])
        temp = 1 + (best - 1) * len(ev) / (len(ev) + SHRINK)
        out[kind] = dict(temperature=round(temp, 3), raw=best, n=len(ev), learned=True,
                         logloss_default=float(-np.mean([_first_logp(e["strength"], e["first"],
                                                                     default) for e in ev])),
                         logloss_learned=float(-np.mean([_first_logp(e["strength"], e["first"],
                                                                     temp) for e in ev])))
    return out


def calibration_table(events: list[dict], temps: dict[str, dict], bins=(0, .05, .15, .3, .5, 1.01)
                      ) -> pd.DataFrame:
    """Predicted win/pole chance vs. how often it happened, over all re-predicted events."""
    rows = []
    for e in events:
        p = win_probs(e["strength"], temps[e["kind"]]["temperature"])
        for i, pi in enumerate(p):
            rows.append(dict(kind=e["kind"], p=pi, hit=float(i == e["first"])))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["bin"] = pd.cut(df["p"], bins, right=False)
    return (df.groupby("bin", observed=True)
            .agg(predicted=("p", "mean"), actual=("hit", "mean"), n=("hit", "size"))
            .reset_index())


# ------------------------------------------------------------- prediction log
LOG_COLUMNS = ["saved_utc", "season", "round", "event", "session", "mode", "driver",
               "p_first", "p_top3", "expected_pos"]


def log_prediction(season: int, rnd: int, event: str, session: str, mode: str,
                   result: pd.DataFrame, first_col: str, top3_col: str) -> None:
    """Store (or refresh) the pre-session prediction for this session and mode."""
    snap = pd.DataFrame({
        "saved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "season": season, "round": rnd, "event": event, "session": session, "mode": mode,
        "driver": result["Driver"].values, "p_first": result[first_col].round(4).values,
        "p_top3": result[top3_col].round(4).values,
        "expected_pos": result["Expected pos."].round(2).values})
    log = read_log()
    if len(log):
        same = ((log["season"] == season) & (log["round"] == rnd) & (log["session"] == session)
                & (log["mode"] == mode))
        old = log[same]
        if len(old) and old.set_index("driver")["p_first"].round(3).equals(
                snap.set_index("driver")["p_first"].round(3).reindex(old["driver"])):
            return  # unchanged
        log = log[~same]
    pd.concat([log, snap], ignore_index=True)[LOG_COLUMNS].to_csv(LOG_FILE, index=False)


def read_log() -> pd.DataFrame:
    try:
        return pd.read_csv(LOG_FILE)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=LOG_COLUMNS)
