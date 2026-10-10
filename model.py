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

from collections import defaultdict
from dataclasses import dataclass, field

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
}
MID_FIELD = 11.0
HALF_LIFE = 3.0  # races


def _ew_mean(values: list[float], n: int, default: float) -> float:
    if not values:
        return default
    v = np.asarray(values[-n:], dtype=float)
    w = 0.5 ** (np.arange(len(v))[::-1] / HALF_LIFE)
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
        return dict(form=form, quali_form=quali_form, team_pace=team_pace,
                    team_trend=trend, circuit_aff=circuit_aff,
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
        circuit = race["circuitId"].iloc[0]
        team_q = defaultdict(list)
        for row in race.itertuples():
            ok = classified(row.positionText)
            started = row.positionText != "W"
            qpos = q.get(row.driverId, np.nan)
            if pd.isna(qpos) and row.grid > 0:
                qpos = row.grid
            if pd.notna(qpos):
                self.drv_quali[row.driverId].append(float(qpos))
                team_q[row.constructorId].append(float(qpos))
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
        circuit = race["circuitId"].iloc[0]
        feats = {}
        chaos = builder.chaos(circuit)
        for row in race.itertuples():
            f = builder.features(row.driverId, row.constructorId, circuit)
            f.update(season=season, round=rnd, circuitId=circuit, driverId=row.driverId,
                     driverCode=row.driverCode, constructorId=row.constructorId,
                     grid=float(row.grid) if row.grid > 0 else 21.0,
                     grid_log=float(np.log(row.grid if row.grid > 0 else 21.0)),
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
             min_races_seen: int = 3) -> RankFit:
    from scipy.optimize import minimize

    seen = hist.groupby(["season", "round"]).ngroup()
    train = hist[hist["classified"] & (seen >= min_races_seen)]
    X_all = train[features].to_numpy(float)
    mean, std = X_all.mean(0), X_all.std(0) + 1e-9
    races = []
    for _, race in train.sort_values("finish").groupby(["season", "round"], sort=False):
        races.append((race[features].to_numpy(float) - mean) / std)

    def nll(beta):
        total, grad = 0.0, np.zeros_like(beta)
        for X in races:
            s = X @ beta
            n = len(s)
            for r in range(min(top_k, n - 1)):
                tail = s[r:]
                m = tail.max()
                p = np.exp(tail - m)
                z = p.sum()
                total -= s[r] - (m + np.log(z))
                grad -= X[r] - (p / z) @ X[r:]
        total += 0.5 * l2 * beta @ beta
        grad += l2 * beta
        return total, grad

    res = minimize(nll, np.zeros(len(features)), jac=True, method="L-BFGS-B")
    return RankFit(features, res.x, mean, std, len(races))


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


def backtest(hist: pd.DataFrame, season: int, features: list[str] | None = None,
                  n_sims: int = 4000, use_chaos: bool = True) -> pd.DataFrame:
    """Walk-forward test of the ranking model vs. a grid-only ranking model."""
    rows = []
    races = hist[hist["season"] == season][["season", "round"]].drop_duplicates()
    for _, rnd in races.itertuples(index=False):
        past = hist[(hist["season"] < season) | ((hist["season"] == season) & (hist["round"] < rnd))]
        race = hist[(hist["season"] == season) & (hist["round"] == rnd) & hist["started"]]
        winner = race.loc[race["finish"] == 1, "driverId"]
        if len(race) < 10 or winner.empty:
            continue
        w_idx = list(race["driverId"]).index(winner.iloc[0])
        actual = race["finish"].to_numpy()
        chaos = race["chaos"].to_numpy() if use_chaos else 1.0
        for name, feats in (("model", features or FEATURES_POST), ("grid only", ["grid_log"])):
            f = fit_rank(past, feats)
            s = f.strength(race)
            pos = simulate_rank(s / chaos, race["p_dnf"].to_numpy(), n=n_sims, seed=int(rnd))
            p_win, p_pod = (pos == 1).mean(0), (pos <= 3).mean(0)
            fin = race["classified"].to_numpy()
            exp_pos = np.array([np.mean(pos[:, i][pos[:, i] < 99]) for i in range(len(race))])
            rows.append(dict(
                round=rnd, model=name,
                winner_logloss=-np.log(max(p_win[w_idx], 1e-3)),
                top_pick_won=bool(np.argmax(p_win) == w_idx),
                podium_brier=float(np.mean((p_pod - (actual <= 3)) ** 2)),
                pos_mae=float(np.mean(np.abs(exp_pos[fin] - actual[fin]))),
            ))
    return pd.DataFrame(rows)


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


def simulate_uncertain_grid(fit: "RankFit", X: pd.DataFrame, est_grid: np.ndarray, sd: float,
                            p_dnf: np.ndarray, randomness: float = 1.0, draws: int = 20,
                            n_per_draw: int = 1000, seed: int = 7) -> np.ndarray:
    """Simulate with a grid that is only estimated: each draw samples a plausible
    qualifying order (estimate + noise), then simulates races from that grid."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(draws):
        noisy = est_grid + rng.normal(0, sd, len(est_grid))
        grid = noisy.argsort().argsort() + 1.0  # rank -> positions 1..n
        Xk = X.copy()
        Xk["grid"], Xk["grid_log"] = grid, np.log(grid)
        out.append(simulate_rank(fit.strength(Xk), p_dnf, randomness, n=n_per_draw,
                                 seed=seed + k))
    return np.vstack(out)


# ------------------------------------------------------------- qualifying model
QUALI_BASE = ["quali_form", "team_pace", "team_trend", "form", "circuit_aff"]
QUALI_VARIANTS = {"form": QUALI_BASE, "practice": QUALI_BASE + ["fp"],
                  "sprint quali": QUALI_BASE + ["sq"]}
FEATURE_INFO.update({
    "fp": "Rank of the driver's best lap in the last practice session this weekend",
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
    q["fp"] = [fp[(s, r)].get(d, np.nan) if (s, r) in fp else np.nan
               for s, r, d in zip(q["season"], q["round"], q["driverId"])]
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


def quali_backtest(rows: pd.DataFrame, season: int, n_sims: int = 5000) -> pd.DataFrame:
    """Walk-forward test: weekend-data model vs. form only, per weekend type."""
    out = []
    for rnd in sorted(rows[rows["season"] == season]["round"].unique()):
        past = rows[(rows["season"] < season) | ((rows["season"] == season) & (rows["round"] < rnd))]
        wk = rows[(rows["season"] == season) & (rows["round"] == rnd)]
        if len(wk) < 15:
            continue
        variant = weekend_variant(wk)
        pole = int(np.argmin(wk["finish"].to_numpy()))
        for name in dict.fromkeys(["form", variant]):
            feats = QUALI_VARIANTS[name]
            train = past.dropna(subset=feats)
            if train.groupby(["season", "round"]).ngroups < 5:
                continue
            f = fit_rank(train, feats)
            pos = simulate_rank(f.strength(fill_missing(wk, feats)), np.zeros(len(wk)),
                                n=n_sims, seed=int(rnd))
            actual = wk["finish"].to_numpy()
            out.append(dict(round=rnd, weekend=variant, model=name,
                            pole_logloss=-np.log(max((pos == 1).mean(0)[pole], 1e-3)),
                            fav_on_pole=bool(np.argmax((pos == 1).mean(0)) == pole),
                            top3_brier=float(np.mean(((pos <= 3).mean(0) - (actual <= 3)) ** 2)),
                            pos_mae=float(np.mean(np.abs(pos.mean(0) - actual)))))
    return pd.DataFrame(out)


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
