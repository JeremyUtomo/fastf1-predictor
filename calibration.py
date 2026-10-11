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

from datetime import datetime, timezone

import numpy as np
import pandas as pd

import data

_DIR = data._CACHE_DIR / "dashboard"
_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = _DIR / "prediction_log.csv"

KINDS = ("quali", "sprint", "race")
DEFAULT_TEMPERATURE = {"quali": 1.0, "sprint": 0.85, "race": 1.0, "race_pre": 1.0}
TEMPS = np.round(np.arange(0.5, 2.01, 0.05), 2)
SHRINK = 5           # events needed before the learned value counts half
MIN_EVENTS = 6
SEASONS_BACK = 2     # learn from the current and the two previous seasons


# ------------------------------------------------------------ re-predictions
def walk_forward_events(D: dict, until: tuple[int, int], progress=None) -> list[dict]:
    """Walk-forward re-predictions of all events before `until`.

    They live in the backtest store of `evaluate.py` (computed once, then only new
    events are added), so calibration and the Track record page use the same data.
    """
    import evaluate

    first = until[0] - SEASONS_BACK
    return [e for e in evaluate.update_store(D, until=until, progress=progress)
            if e["season"] >= first]


def _first_logp(strength: np.ndarray, idx: int, temp: float) -> float:
    s = strength / temp
    s = s - s.max()
    return float(s[idx] - np.log(np.exp(s).sum()))


def win_probs(strength: np.ndarray, temp: float) -> np.ndarray:
    s = strength / temp
    e = np.exp(s - s.max())
    return e / e.sum()


def temperatures(events: list[dict], kinds=KINDS) -> dict[str, dict]:
    """Learned temperature per session type, with the evidence behind it."""
    out = {}
    for kind in kinds:
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
        if e["kind"] not in temps:
            continue
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


# ------------------------------------------------- long shots and favourites
# Even after the temperature correction, the backtest shows the classic
# favourite-longshot bias in win probabilities: drivers given 1–5% won less often,
# favourites slightly more often. A logistic recalibration learned from the
# backtest fixes most of it (tested walk-forward on 2021–2026). Pole and podium
# probabilities were already well calibrated, so they are left alone.
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def fit_win_recalibration(events: list[dict], kinds=("race", "sprint")) -> tuple[float, float]:
    """(a, b) so that P(win) ≈ sigmoid(a + b·logit(p)), fitted on re-predicted events."""
    from scipy.optimize import minimize

    z, y = [], []
    for e in events:
        if e["kind"] in kinds:
            z.extend(_logit(e.get("cal_p_first", e["p_first"])))
            y.extend((np.asarray(e["finish"]) == 1).astype(float))
    if sum(y) < 30:
        return 0.0, 1.0
    z, y = np.asarray(z), np.asarray(y)

    def nll(w):
        q = np.clip(1 / (1 + np.exp(-(w[0] + w[1] * z))), 1e-9, 1 - 1e-9)
        return -float(np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)))

    a, b = minimize(nll, [0.0, 1.0], method="Nelder-Mead").x
    return float(a), float(b)


def recalibrate_win(p, w: tuple[float, float]) -> np.ndarray:
    """Apply the win recalibration and renormalise so the field sums to 1."""
    q = 1 / (1 + np.exp(-(w[0] + w[1] * _logit(p))))
    return q / q.sum()


BANDS = [0, .01, .02, .05, .10, .20, .40, 1.01]


def _wilson(k: float, n: int, z: float = 1.645) -> tuple[float, float]:
    """90% interval for a hit rate k/n (works for very few hits)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def band_evidence(events: list[dict], kinds, market: str,
                  win_recal: tuple[float, float] | None = None) -> pd.DataFrame:
    """How often predictions in each probability band actually came true in the backtest.

    market: 'first' (win / pole) or 'top3' (podium / top 3).
    """
    rows = []
    for e in events:
        if e["kind"] not in kinds:
            continue
        finish, cls = np.asarray(e["finish"]), np.asarray(e["classified"])
        if market == "first":
            p = np.asarray(e.get("cal_p_first", e["p_first"]))
            if win_recal is not None:
                p = recalibrate_win(p, win_recal)
            y = finish == 1
        else:
            p = np.asarray(e.get("cal_p_top3", e["p_top3"]))
            y = (finish <= 3) & cls
        rows += [(pi, float(yi)) for pi, yi in zip(p, y)]
    d = pd.DataFrame(rows, columns=["p", "y"])
    if d.empty:
        return d
    d["band"] = pd.cut(d["p"], BANDS, right=False)
    out = d.groupby("band", observed=True).agg(predicted=("p", "mean"), n=("y", "size"),
                                               hits=("y", "sum")).reset_index()
    out[["lo", "hi"]] = [_wilson(k, n) for k, n in zip(out["hits"], out["n"])]
    out["left"] = [b.left for b in out["band"]]
    out["right"] = [b.right for b in out["band"]]
    return out
