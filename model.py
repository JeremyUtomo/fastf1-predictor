"""Race outcome model.

1. Walk through past races in order and compute, for every entry, features that
   only use information available *before* that race (form, car pace, circuit
   history, ...).
2. Fit a Plackett-Luce ranking model on the finishing order: each driver gets a
   "strength" and P(finishing ahead of the remaining field) ∝ exp(strength).
   Its weights show what actually mattered historically.
3. Turn strengths into market probabilities with a Monte-Carlo simulation that
   adds per-driver retirement chances and extra randomness for chaotic circuits
   and rain.

Everything here is plain pandas/numpy so it can be tested and backtested
without Streamlit.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_POST = ["grid", "grid_log", "form", "quali_form", "team_pace", "team_trend", "circuit_aff"]
FEATURES_PRE = ["form", "quali_form", "team_pace", "team_trend", "circuit_aff"]
FEATURE_INFO = {
    "grid": "Starting position (qualifying result, or the grid you enter)",
    "grid_log": "Starting position on a log scale: P1 vs P3 matters more than P11 vs P13",
    "form": "Recent finishing positions, recent races weigh more",
    "quali_form": "Recent qualifying positions (one-lap pace)",
    "team_pace": "Team's recent qualifying positions, both cars (car performance)",
    "team_trend": "Team's qualifying trend over the last 5 races — captures upgrades",
    "circuit_aff": "Driver's results at this circuit vs. their usual level",
    "quali_gap": "Recent qualifying pace: gap to the fastest time in % (recent weigh more)",
    "team_gap": "Team's recent qualifying pace in % gap, both cars (car performance)",
    "driver_edge": "Driver vs. own teammate in qualifying (% faster), long memory: driver skill",
    "race_pace": "Recent race pace: fastest race lap vs. the fastest of the race, in %",
    "grid_chaos": "Starting position × how much positions change at this circuit",
    "qgap_now": "This weekend's qualifying gap to pole in % (how dominant the pole lap was)",
}
MID_FIELD = 11.0
HALF_LIFE = 3.0          # races, for recent form
DRIVER_HALF_LIFE = 10.0  # races: driver skill changes slowly, the car quickly
GAP_DEFAULT = 1.5        # % gap assumed without history (roughly midfield)


def _ew_mean(values: list[float], n: int, default: float, half_life: float = HALF_LIFE) -> float:
    if not values:
        return default
    v = np.asarray(values[-n:], dtype=float)
    w = 0.5 ** (np.arange(len(v))[::-1] / half_life)
    return float(np.sum(v * w) / np.sum(w))


def classified(position_text) -> bool:
    return str(position_text).isdigit()


@dataclass
class FeatureBuilder:
    """Keeps running per-driver/team histories while walking through races in order."""
    drv_finish: dict = field(default_factory=lambda: defaultdict(list))
    drv_quali: dict = field(default_factory=lambda: defaultdict(list))
    team_quali: dict = field(default_factory=lambda: defaultdict(list))  # per race: mean of cars
    drv_dnf: dict = field(default_factory=lambda: defaultdict(list))
    team_dnf: dict = field(default_factory=lambda: defaultdict(list))
    circuit_resid: dict = field(default_factory=lambda: defaultdict(list))
    circuit_shuffle: dict = field(default_factory=lambda: defaultdict(list))
    all_shuffle: list = field(default_factory=list)
    all_dnf: list = field(default_factory=list)
    drv_qgap: dict = field(default_factory=lambda: defaultdict(list))
    team_qgap: dict = field(default_factory=lambda: defaultdict(list))
    drv_edge: dict = field(default_factory=lambda: defaultdict(list))
    drv_rpace: dict = field(default_factory=lambda: defaultdict(list))

    def features(self, driver: str, team: str, circuit: str) -> dict:
        team_hist = self.team_quali[team]
        team_pace = _ew_mean(team_hist, 6, MID_FIELD)
        if len(team_hist) >= 3:
            recent = np.asarray(team_hist[-5:], dtype=float)
            trend = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
        else:
            trend = 0.0
        quali_form = _ew_mean(self.drv_quali[driver], 6, team_pace)
        form = _ew_mean(self.drv_finish[driver], 6, team_pace + 1)
        resid = self.circuit_resid[(driver, circuit)]
        circuit_aff = float(np.sum(resid) / (len(resid) + 2))  # shrunk towards 0
        team_gap = _ew_mean(self.team_qgap[team], 6, GAP_DEFAULT)
        edges = self.drv_edge[driver]
        driver_edge = (_ew_mean(edges, 30, 0.0, DRIVER_HALF_LIFE) * len(edges[-30:])
                       / (len(edges[-30:]) + 3))  # shrunk towards 0 for new drivers
        return dict(form=form, quali_form=quali_form, team_pace=team_pace,
                    team_trend=trend, circuit_aff=circuit_aff,
                    quali_gap=_ew_mean(self.drv_qgap[driver], 6, team_gap),
                    team_gap=team_gap, driver_edge=driver_edge,
                    race_pace=_ew_mean(self.drv_rpace[driver], 6, team_gap),
                    n_circuit=len(resid), p_dnf=self.dnf_prob(driver, team))

    def dnf_prob(self, driver: str, team: str) -> float:
        base = np.mean(self.all_dnf[-400:]) if self.all_dnf else 0.1
        d, t = self.drv_dnf[driver][-12:], self.team_dnf[team][-24:]
        p_d = (np.sum(d) + 3 * base) / (len(d) + 3)
        p_t = (np.sum(t) + 3 * base) / (len(t) + 3)
        return float(0.5 * p_d + 0.5 * p_t)

    def chaos(self, circuit: str) -> float:
        """How much positions change at this circuit relative to average (0.75–1.35)."""
        here = self.circuit_shuffle[circuit]
        if len(here) < 10 or not self.all_shuffle:
            return 1.0
        return float(np.clip(np.mean(here) / np.mean(self.all_shuffle), 0.75, 1.35))

    def update(self, race: pd.DataFrame, quali: pd.DataFrame, feats: pd.DataFrame) -> None:
        """Add one race (all entries) to the histories."""
        q = quali.set_index("driverId")["quali"] if len(quali) else pd.Series(dtype=float)
        qgap = (quali.set_index("driverId")["qgap"] if len(quali) and "qgap" in quali
                else pd.Series(dtype=float))
        circuit = race["circuitId"].iloc[0]
        team_q = defaultdict(list)
        team_gaps = defaultdict(dict)
        for row in race.itertuples():
            ok = classified(row.positionText)
            started = row.positionText != "W"
            qpos = q.get(row.driverId, np.nan)
            if pd.isna(qpos) and row.grid > 0:
                qpos = row.grid
            if pd.notna(qpos):
                self.drv_quali[row.driverId].append(float(qpos))
                team_q[row.constructorId].append(float(qpos))
            gap = qgap.get(row.driverId, np.nan)
            if pd.notna(gap):
                self.drv_qgap[row.driverId].append(float(gap))
                team_gaps[row.constructorId][row.driverId] = float(gap)
            rpace = getattr(row, "flap_gap", np.nan)
            if pd.notna(rpace):
                self.drv_rpace[row.driverId].append(float(rpace))
            if started:
                self.drv_dnf[row.driverId].append(not ok)
                self.team_dnf[row.constructorId].append(not ok)
                self.all_dnf.append(not ok)
            if ok:
                self.drv_finish[row.driverId].append(float(row.position))
                form_then = feats.loc[row.driverId, "form"] if row.driverId in feats.index else None
                if form_then is not None:
                    self.circuit_resid[(row.driverId, circuit)].append(float(row.position) - form_then)
                if row.grid > 0:
                    shuffle = abs(float(row.position) - float(row.grid))
                    self.circuit_shuffle[circuit].append(shuffle)
                    self.all_shuffle.append(shuffle)
        for team, vals in team_q.items():
            self.team_quali[team].append(float(np.mean(vals)))
        for team, gaps in team_gaps.items():
            self.team_qgap[team].append(float(np.mean(list(gaps.values()))))
            if len(gaps) == 2:  # driver vs. teammate: same car, so the gap is the driver
                (a, ga), (b, gb) = gaps.items()
                self.drv_edge[a].append(gb - ga)
                self.drv_edge[b].append(ga - gb)


def build_history(races: pd.DataFrame, quali: pd.DataFrame,
                  stop_before: tuple[int, int] | None = None) -> tuple[pd.DataFrame, FeatureBuilder]:
    """Feature rows for every past entry (target = finishing position).

    Stops before race (season, round) if given, so nothing from it leaks in.
    """
    builder = FeatureBuilder()
    rows = []
    keys = races[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    for season, rnd in keys.itertuples(index=False):
        if stop_before and (season, rnd) >= stop_before:
            break
        race = races[(races["season"] == season) & (races["round"] == rnd)]
        rq = quali[(quali["season"] == season) & (quali["round"] == rnd)]
        qgap_now = (rq.set_index("driverId")["qgap"] if "qgap" in rq and len(rq)
                    else pd.Series(dtype=float))
        circuit = race["circuitId"].iloc[0]
        feats = {}
        chaos = builder.chaos(circuit)
        for row in race.itertuples():
            f = builder.features(row.driverId, row.constructorId, circuit)
            # this weekend's qualifying gap: known before the race, like the grid
            f["qgap_now"] = float(qgap_now.get(row.driverId, np.nan))
            f.update(season=season, round=rnd, circuitId=circuit, driverId=row.driverId,
                     driverCode=row.driverCode, constructorId=row.constructorId,
                     grid=float(row.grid) if row.grid > 0 else 21.0,
                     grid_log=float(np.log(row.grid if row.grid > 0 else 21.0)),
                     grid_chaos=float(np.log(row.grid if row.grid > 0 else 21.0)) * (chaos - 1),
                     finish=float(row.position), classified=classified(row.positionText),
                     started=row.positionText != "W", chaos=chaos)
            feats[row.driverId] = f
            rows.append(f)
        builder.update(race, rq, pd.DataFrame.from_dict(feats, orient="index"))
    return pd.DataFrame(rows), builder


def summarise(pos: np.ndarray, drivers: list[str]) -> pd.DataFrame:
    finished = np.where(pos == 99, np.nan, pos)
    return pd.DataFrame({
        "Driver": drivers,
        "Win": (pos == 1).mean(0),
        "Podium": (pos <= 3).mean(0),
        "Top 6": (pos <= 6).mean(0),
        "Points (top 10)": (pos <= 10).mean(0),
        "DNF": (pos == 99).mean(0),
        "Expected pos.": np.nanmean(finished, 0),
    })


def head_to_head(pos: np.ndarray, i: int, j: int) -> tuple[float, float, float]:
    """P(i ahead), P(j ahead), P(both retire) — a retired driver is behind a finisher."""
    a, b = pos[:, i], pos[:, j]
    both_out = (a == 99) & (b == 99)
    return float(((a < b) & ~both_out).mean()), float(((b < a) & ~both_out).mean()), \
        float(both_out.mean())


# ------------------------------------------------- Plackett-Luce ranking model
@dataclass
class RankFit:
    """Plackett-Luce model: P(i finishes ahead of the rest) ∝ exp(strength_i).

    Fitted on the finishing order of classified drivers (top `top_k` places), so
    win/podium probabilities are calibrated directly instead of via a noise model.
    top_k=3 / l2=1 were chosen by walk-forward testing on 2025 and checked on 2026.
    """
    features: list[str]
    beta: np.ndarray        # per standardised feature
    mean: np.ndarray
    std: np.ndarray
    n_races: int

    def strength(self, X: pd.DataFrame) -> np.ndarray:
        return ((X[self.features].to_numpy(float) - self.mean) / self.std) @ self.beta

    def weight_table(self) -> pd.DataFrame:
        # negative beta on a "lower is better" feature means it helps
        return pd.DataFrame({"feature": self.features, "weight": -self.beta,
                             "meaning": [FEATURE_INFO[f] for f in self.features]})


def fit_rank(hist: pd.DataFrame, features: list[str], top_k: int = 3, l2: float = 1.0,
             min_races_seen: int = 3, half_life: float | None = None) -> RankFit:
    """Maximum-likelihood Plackett-Luce fit on the top `top_k` places of every event.

    `half_life` (in events) down-weights older events, as in time-weighted
    Plackett-Luce models (Henderson & Kirrane 2018); None weighs all equally.
    """
    seen = hist.groupby(["season", "round"]).ngroup()
    train = hist[hist["classified"] & (seen >= min_races_seen)].dropna(subset=features)
    groups = [g[features].to_numpy(float) for _, g in
              train.sort_values(["season", "round", "finish"]).groupby(["season", "round"], sort=True)]
    return fit_arrays(groups, features, top_k=top_k, l2=l2, half_life=half_life)


def fit_arrays(groups: list[np.ndarray], features: list[str], top_k: int = 3, l2: float = 1.0,
               half_life: float | None = None, beta0: np.ndarray | None = None) -> RankFit:
    """Core fit on per-event feature arrays (rows in finishing order, oldest event first)."""
    from scipy.optimize import minimize

    X_all = np.vstack(groups)
    mean, std = X_all.mean(0), X_all.std(0) + 1e-9
    R, F = len(groups), len(features)
    N = max(len(g) for g in groups)
    X = np.zeros((R, N, F))
    n = np.array([len(g) for g in groups])
    for i, g in enumerate(groups):
        X[i, :len(g)] = (g - mean) / std
    mask = np.arange(N)[None, :] < n[:, None]
    age = R - 1 - np.arange(R)
    w = 0.5 ** (age / half_life) if half_life else np.ones(R)
    w = w * R / w.sum()  # mean weight 1, so l2 means the same for every half-life

    def nll(beta):
        s = np.where(mask, X @ beta, -np.inf)
        total, grad = 0.0, np.zeros(F)
        for r in range(top_k):
            valid = r < n - 1
            if not valid.any():
                break
            tail = s[:, r:]
            m = np.max(tail, axis=1)
            m = np.where(np.isfinite(m), m, 0.0)
            e = np.exp(tail - m[:, None])
            z = e.sum(1)
            zs = np.where(z > 0, z, 1.0)
            ll = np.where(valid, s[:, r] - (m + np.log(zs)), 0.0)
            wv = w * valid
            total -= float(wv @ ll)
            p = np.where(valid[:, None], e / zs[:, None], 0.0)
            exp_x = np.einsum("rn,rnf->rf", p, X[:, r:, :])
            grad -= (wv[:, None] * (X[:, r, :] - exp_x)).sum(0)
        total += 0.5 * l2 * beta @ beta
        grad += l2 * beta
        return total, grad

    start = beta0 if beta0 is not None and len(beta0) == F else np.zeros(F)
    res = minimize(nll, start, jac=True, method="L-BFGS-B")
    return RankFit(list(features), res.x, mean, std, R)


def simulate_rank(strength: np.ndarray, p_dnf: np.ndarray, randomness: float = 1.0,
                  n: int = 20000, seed: int = 7) -> np.ndarray:
    """Plackett-Luce samples via the Gumbel trick. `randomness` > 1 = more chaos."""
    rng = np.random.default_rng(seed)
    d = len(strength)
    util = strength[None, :] / randomness + rng.gumbel(size=(n, d))
    dnf = rng.random((n, d)) < p_dnf[None, :]
    util = np.where(dnf, -1e6 + util, util)
    order = np.argsort(-util, axis=1)
    pos = np.empty_like(order)
    pos[np.arange(n)[:, None], order] = np.arange(1, d + 1)
    return np.where(dnf, 99, pos)


# --------------------------------------------- grid estimated before qualifying
SQ_FEATURES = ["sq", "quali_form"]


def fit_grid_estimate(hist: pd.DataFrame, sprints: pd.DataFrame,
                      quali: pd.DataFrame) -> tuple[np.ndarray, float] | None:
    """Linear model: qualifying position ~ sprint-qualifying position + recent quali form.

    Fitted on past sprint weekends (the sprint grid is the SQ classification).
    Returns (coefficients incl. intercept, residual std) or None without data.
    """
    d = hist.merge(sprints[sprints["sq"] > 0], on=["season", "round", "driverId"]).merge(
        quali, on=["season", "round", "driverId"])
    if d.groupby(["season", "round"]).ngroups < 4:
        return None
    X = np.column_stack([np.ones(len(d))] + [d[c].astype(float) for c in SQ_FEATURES])
    y = d["quali"].astype(float)
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return w, float(np.std(y - X @ w))


def with_grid(X: pd.DataFrame, grid: np.ndarray, chaos: float,
              qgap: np.ndarray | None = None, qgap_lookup=None) -> pd.DataFrame:
    """Set the starting grid and the features derived from it."""
    X = X.copy()
    g = np.clip(np.asarray(grid, dtype=float), 1, 22)
    X["grid"], X["grid_log"] = g, np.log(g)
    X["grid_chaos"] = X["grid_log"] * (chaos - 1)
    if qgap is None and qgap_lookup is not None:
        qgap = qgap_lookup(g)
    if qgap is not None:
        X["qgap_now"] = qgap
    return X


def qgap_lookup(hist: pd.DataFrame):
    """Typical qualifying gap (%) for a grid position, from history (for estimated grids)."""
    med = hist.groupby(hist["grid"].round().clip(1, 22))["qgap_now"].median()
    med = med.reindex(np.arange(1, 23)).interpolate().bfill().ffill()
    return lambda g: np.interp(g, med.index.to_numpy(float), med.to_numpy(float))


def simulate_uncertain_grid(fit: "RankFit", X: pd.DataFrame, est_grid: np.ndarray, sd: float,
                            p_dnf: np.ndarray, randomness: float = 1.0, draws: int = 20,
                            n_per_draw: int = 1000, seed: int = 7, chaos: float = 1.0,
                            lookup=None) -> np.ndarray:
    """Simulate with a grid that is only estimated: each draw samples a plausible
    qualifying order (estimate + noise), then simulates races from that grid."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(draws):
        noisy = est_grid + rng.normal(0, sd, len(est_grid))
        grid = noisy.argsort().argsort() + 1.0  # rank -> positions 1..n
        Xk = with_grid(X, grid, chaos, qgap_lookup=lookup)
        out.append(simulate_rank(fit.strength(Xk), p_dnf, randomness, n=n_per_draw,
                                 seed=seed + k))
    return np.vstack(out)


# ------------------------------------------------------------- qualifying model
QUALI_BASE = ["quali_form", "team_pace", "team_trend", "form", "circuit_aff"]
QUALI_VARIANTS = {"form": QUALI_BASE, "practice": QUALI_BASE + ["fp"],
                  "sprint quali": QUALI_BASE + ["sq"]}
FEATURE_INFO.update({
    "fp": "Rank of the driver's best lap in the last practice session this weekend",
    "fp_gap": "Gap (%) of the driver's best lap to the fastest in the last practice session",
    "sq": "Sprint Qualifying position this weekend",
})


def quali_rows(hist: pd.DataFrame, quali: pd.DataFrame, sprints: pd.DataFrame,
               fp: dict) -> pd.DataFrame:
    """One row per qualifying entry: pre-weekend features, weekend signals, result.

    `fp` maps (season, round) -> practice rank Series indexed by driverId.
    The result goes into `finish` so the ranking model can be reused unchanged.
    """
    q = hist.drop(columns=["finish", "classified"]).merge(
        quali, on=["season", "round", "driverId", "constructorId"])
    sq = sprints.assign(sq=sprints["sq"].where(sprints["sq"] > 0, 21))
    q = q.merge(sq[["season", "round", "driverId", "sq"]], on=["season", "round", "driverId"],
                how="left")
    def pace(s, r, d, col):
        p = fp.get((s, r))
        if p is None:
            return np.nan
        if isinstance(p, pd.Series):  # rank only
            return p.get(d, np.nan) if col == "rank" else np.nan
        return p[col].get(d, np.nan) if col in p else np.nan

    keys = list(zip(q["season"], q["round"], q["driverId"]))
    q["fp"] = [pace(s, r, d, "rank") for s, r, d in keys]
    q["fp_gap"] = [pace(s, r, d, "gap") for s, r, d in keys]
    q["finish"], q["classified"] = q["quali"].astype(float), True
    return q


def weekend_variant(rows: pd.DataFrame) -> str:
    """Which qualifying model a weekend's data supports."""
    if rows["sq"].notna().mean() > 0.7:
        return "sprint quali"
    if rows["fp"].notna().mean() > 0.7:
        return "practice"
    return "form"


def fill_missing(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Drivers without a weekend signal (e.g. no lap time) are put at the back."""
    X = X.copy()
    for c in features:
        X[c] = X[c].fillna(X[c].max() if X[c].notna().any() else 21)
    return X


# ---------------------------------------------------------------- upgrades
FEATURE_INFO.update({
    "upg_perf": "Performance upgrades the team declared to the FIA for this event",
    "upg_perf_prev": "Performance upgrades the team brought in its previous 3 events",
})


def add_upgrades(rows: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Add upgrade features per entry from FIA counts (season, round, constructorId).

    Rows from events the FIA data doesn't cover get NaN, so models using these
    features train only on covered events.
    """
    c = counts.sort_values(["constructorId", "season", "round"]).copy()
    c["upg_perf_prev"] = (c.groupby("constructorId")["upg_perf"]
                          .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum()))
    out = rows.drop(columns=[x for x in ("upg_perf", "upg_perf_prev") if x in rows], errors="ignore")
    return out.merge(c[["season", "round", "constructorId", "upg_perf", "upg_perf_prev"]],
                     on=["season", "round", "constructorId"], how="left")


# ------------------------------------------------------------ model config
# Feature sets and settings per model family. `model_config.json` (written by
# `python evaluate.py --tune`) overrides these defaults, so re-tuning the model
# needs no code change.
CONFIG_FILE = Path(__file__).with_name("model_config.json")
DEFAULT_CONFIG = {
    "race_post": dict(features=list(FEATURES_POST), top_k=3, l2=1.0, half_life=None,
                      train_seasons=4),
    "race_pre": dict(features=list(FEATURES_PRE), top_k=3, l2=1.0, half_life=None,
                     train_seasons=4),
    "quali": dict(base=list(QUALI_BASE), practice="fp", top_k=3, l2=1.0, half_life=None,
                  train_seasons=4),
}


def load_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        saved = json.loads(CONFIG_FILE.read_text())
    except (OSError, ValueError):
        return cfg
    for family in cfg:
        cfg[family].update(saved.get(family, {}))
    cfg["_meta"] = saved.get("_meta", {})
    return cfg


CONFIG = load_config()


def quali_features(variant: str, cfg: dict | None = None) -> list[str]:
    q = (cfg or CONFIG)["quali"]
    extra = {"form": [], "practice": [q["practice"]], "sprint quali": ["sq"]}[variant]
    return list(q["base"]) + extra


def training_rows(rows: pd.DataFrame, season: int, rnd: int, seasons: int) -> pd.DataFrame:
    """Rows before (season, rnd), within the last `seasons` seasons."""
    before = (rows["season"] < season) | ((rows["season"] == season) & (rows["round"] < rnd))
    return rows[before & (rows["season"] >= season - seasons + 1)]


def fit_family(rows: pd.DataFrame, family: str, season: int, rnd: int,
               features: list[str] | None = None, cfg: dict | None = None) -> RankFit:
    """Fit a model family on everything before (season, rnd) with its configured settings."""
    c = (cfg or CONFIG)[family]
    feats = features or c["features"]
    train = training_rows(rows, season, rnd, c["train_seasons"]).dropna(subset=feats)
    return fit_rank(train, feats, top_k=c["top_k"], l2=c["l2"], half_life=c["half_life"])
