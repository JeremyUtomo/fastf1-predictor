"""Walk-forward evaluation, model research and the backtest store.

Every race, sprint and qualifying session since 2020 is predicted with a model
fitted only on earlier events, and compared with what actually happened.

    python evaluate.py --tune    # test the candidate improvements, choose settings on
                                 # 2020–2025, check once on 2026, write
                                 # model_config.json + research_report.md
    python evaluate.py --store   # rebuild the backtest store for the current config

The store (per-event, per-driver predictions and results) feeds the Track record
page and the self-calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import data
import model
import predict_data as pdata

EVAL_FROM = 2020   # first season that is re-predicted
TUNE_UNTIL = 2025  # settings are chosen on EVAL_FROM..TUNE_UNTIL
HOLDOUT = 2026     # ...and checked once on this season
N_SIMS = 4000
MIN_GAIN = 0.003   # log-loss improvement needed to adopt a candidate (noise threshold)
STORE_DIR = data._CACHE_DIR / "dashboard"
REPORT_FILE = model.CONFIG_FILE.with_name("research_report.md")
KINDS = ("race", "race_pre", "sprint", "quali")


# --------------------------------------------------------------------- data
def load_all(last_season: int | None = None, log=print) -> dict:
    """Everything needed to re-predict events: features from 2018 onwards."""
    last = last_season or data.current_season()
    races, quali = pdata.history(last, start=pdata.DATA_START)
    sprints = pdata.sprints_range(pdata.DATA_START, last)
    hist, _ = model.build_history(races, quali)
    sprint_weekends = set(map(tuple, sprints[["season", "round"]].drop_duplicates().values))
    fp = {}
    weekends = hist[["season", "round"]].drop_duplicates().itertuples(index=False)
    for s, r in weekends:
        if (s, r) not in sprint_weekends:
            pace = pdata.practice_pace(int(s), int(r), "Practice 3")
            if pace is not None:
                fp[(s, r)] = pace
    log(f"data: {hist['season'].min()}–{hist['season'].max()}, "
        f"{hist.groupby(['season', 'round']).ngroups} races, practice for {len(fp)} weekends")
    return dict(hist=hist, qrows=model.quali_rows(hist, quali, sprints, fp), sprints=sprints,
                races=races)


def sprint_rows(D: dict, season: int, rnd: int) -> pd.DataFrame:
    """Race-weekend features with the sprint grid (= Sprint Qualifying) and result."""
    race = D["hist"][(D["hist"]["season"] == season) & (D["hist"]["round"] == rnd)]
    sp = D["sprints"][(D["sprints"]["season"] == season) & (D["sprints"]["round"] == rnd)]
    if len(sp) < 15 or race.empty:
        return pd.DataFrame()
    x = race.drop(columns=["finish", "classified", "grid", "grid_log", "grid_chaos"]).merge(
        sp[["driverId", "sq", "sprint", "positionText"]], on="driverId")
    x = x[x["sq"] > 0].copy()
    x["grid"] = x["sq"].astype(float)
    x["grid_log"] = np.log(x["grid"])
    x["grid_chaos"] = x["grid_log"] * (x["chaos"] - 1)
    x["finish"] = x["sprint"].astype(float)
    x["classified"] = x["positionText"].astype(str).str.isdigit()
    x["qgap_now"] = np.nan  # Sprint Qualifying times aren't in the historical results
    return x


def event_rows(D: dict, kind: str, season: int, rnd: int) -> pd.DataFrame:
    if kind in ("race", "race_pre"):
        h = D["hist"]
        return h[(h["season"] == season) & (h["round"] == rnd) & h["started"]]
    if kind == "sprint":
        return sprint_rows(D, season, rnd)
    q = D["qrows"]
    return q[(q["season"] == season) & (q["round"] == rnd)]


def events_list(D: dict, kind: str, first: int = EVAL_FROM, until: tuple | None = None) -> list:
    src = D["sprints"] if kind == "sprint" else (D["qrows"] if kind == "quali" else D["hist"])
    keys = src[src["season"] >= first][["season", "round"]].drop_duplicates()
    keys = sorted((int(s), int(r)) for s, r in keys.itertuples(index=False))
    return [k for k in keys if until is None or k < until]


# ------------------------------------------------------------------- fitting
class NotEnoughData(ValueError):
    pass


class Fitter:
    """Fits models on precomputed per-event arrays (fast enough for research runs)."""

    def __init__(self, D: dict):
        self.D = D
        self._groups: dict = {}
        self._warm: dict = {}

    def _event_arrays(self, source: str, feats: tuple) -> list:
        key = (source, feats)
        if key not in self._groups:
            rows = self.D["qrows"] if source == "quali" else self.D["hist"]
            rows = rows[rows["classified"]].dropna(subset=list(feats))
            out = []
            for (s, r), g in rows.sort_values(["season", "round", "finish"]).groupby(
                    ["season", "round"], sort=True):
                out.append((int(s), int(r), g[list(feats)].to_numpy(float)))
            self._groups[key] = out
        return self._groups[key]

    def fit(self, source: str, feats: list[str], c: dict, season: int, rnd: int) -> model.RankFit:
        groups = [g for s, r, g in self._event_arrays(source, tuple(feats))
                  if (s, r) < (season, rnd) and s >= season - c["train_seasons"] + 1]
        if len(groups) < 5:
            raise NotEnoughData(f"{len(groups)} training events for {feats}")
        key = (source, tuple(feats), c["top_k"], c["l2"], c["half_life"], c["train_seasons"])
        fit = model.fit_arrays(groups, feats, top_k=c["top_k"], l2=c["l2"],
                               half_life=c["half_life"], beta0=self._warm.get(key))
        self._warm[key] = fit.beta
        return fit


def family_of(kind: str) -> tuple[str, str]:
    """(config family, training rows) for an event kind."""
    return {"race": ("race_post", "races"), "sprint": ("race_post", "races"),
            "race_pre": ("race_pre", "races"), "quali": ("quali", "quali")}[kind]


def features_for(kind: str, cfg: dict, rows: pd.DataFrame) -> list[str]:
    if kind == "quali":
        return model.quali_features(model.weekend_variant(rows), cfg)
    feats = list(cfg[family_of(kind)[0]]["features"])
    if kind == "sprint":  # no historical Sprint Qualifying times
        feats = [f for f in feats if f != "qgap_now"]
    return feats


def predict_event(fitter: Fitter, kind: str, season: int, rnd: int, cfg: dict,
                  features: list[str] | None = None) -> dict | None:
    rows = event_rows(fitter.D, kind, season, rnd)
    if len(rows) < 10 or not (rows["finish"] == 1).any():
        return None
    family, source = family_of(kind)
    feats = features or features_for(kind, cfg, rows)
    src = "quali" if source == "quali" else "races"
    try:
        fit = fitter.fit(src, feats, cfg[family], season, rnd)
    except NotEnoughData:
        if kind != "quali" or features:
            return None
        # e.g. the first sprint weekend: no earlier weekend of that type to learn from
        feats = model.quali_features("form", cfg)
        try:
            fit = fitter.fit(src, feats, cfg[family], season, rnd)
        except NotEnoughData:
            return None
    X = model.fill_missing(rows, feats)
    strength = fit.strength(X)
    if kind != "quali":
        strength = strength / rows["chaos"].to_numpy()
    p_dnf = np.zeros(len(rows)) if kind == "quali" else rows["p_dnf"].to_numpy() * (
        0.4 if kind == "sprint" else 1.0)
    ev = dict(kind=kind, season=season, round=rnd, circuit=rows["circuitId"].iloc[0],
              drivers=list(rows["driverId"]), codes=list(rows["driverCode"]),
              teams=list(rows["constructorId"]), strength=strength, p_dnf=p_dnf,
              finish=rows["finish"].to_numpy(float), classified=rows["classified"].to_numpy(bool),
              first=int(np.argmax(rows["finish"].to_numpy() == 1)), n_train=fit.n_races)
    ev.update(probabilities(ev, 1.0))
    return ev


def probabilities(ev: dict, temperature: float, seed: int = 11) -> dict:
    pos = model.simulate_rank(ev["strength"], ev["p_dnf"], randomness=temperature, n=N_SIMS,
                              seed=seed + ev["round"])
    finished = np.where(pos == 99, np.nan, pos)
    with np.errstate(all="ignore"):
        exp_pos = np.nanmean(finished, 0)
    return dict(p_first=(pos == 1).mean(0), p_top3=(pos <= 3).mean(0), exp_pos=exp_pos)


# ------------------------------------------------------------------- scoring
def score(ev: dict, prefix: str = "") -> dict:
    p_first, p_top3, exp_pos = (ev[prefix + "p_first"], ev[prefix + "p_top3"],
                                ev[prefix + "exp_pos"])
    finish, cls, first = ev["finish"], ev["classified"], ev["first"]
    top3 = (finish <= 3) & cls
    ok = cls & ~np.isnan(exp_pos)
    rho = (pd.Series(exp_pos[ok]).rank().corr(pd.Series(finish[ok]).rank())
           if ok.sum() >= 5 else np.nan)
    return dict(kind=ev["kind"], season=ev["season"], round=ev["round"],
                logloss=-np.log(max(p_first[first], 1e-3)),
                fav_hit=float(np.argmax(p_first) == first),
                top3_hits=len(set(np.argsort(-p_top3)[:3]) & set(np.where(top3)[0])),
                top3_brier=float(np.mean((p_top3 - top3) ** 2)),
                spearman=rho, pos_mae=float(np.nanmean(np.abs(exp_pos[ok] - finish[ok]))))


def run(D: dict, kind: str, cfg: dict, fitter: Fitter | None = None,
        features: list[str] | None = None, until: tuple | None = None) -> list[dict]:
    fitter = fitter or Fitter(D)
    out = []
    for season, rnd in events_list(D, kind, until=until):
        ev = predict_event(fitter, kind, season, rnd, cfg, features)
        if ev is not None:
            out.append(ev)
    return out


def summarise(scores: pd.DataFrame) -> pd.Series:
    return scores[["logloss", "fav_hit", "top3_hits", "top3_brier", "spearman", "pos_mae"]].mean()


def split(evs: list[dict]) -> tuple[pd.Series, pd.Series]:
    sc = pd.DataFrame([score(e) for e in evs])
    return summarise(sc[sc["season"] <= TUNE_UNTIL]), summarise(sc[sc["season"] == HOLDOUT])


# -------------------------------------------------------------------- tuning
def tune(D: dict, log=print) -> tuple[dict, list[dict]]:
    """Greedy search: test candidates in order, keep those that clearly help on
    2020–2025. Returns the chosen config and a log of every test."""
    cfg = json.loads(json.dumps(model.DEFAULT_CONFIG))
    fitter = Fitter(D)
    trials = []

    def evaluate(family: str, trial_cfg: dict, label: str) -> tuple[float, pd.Series, pd.Series]:
        kind = {"race_post": "race", "race_pre": "race_pre", "quali": "quali"}[family]
        t0 = time.time()
        tune_s, hold_s = split(run(D, kind, trial_cfg, fitter))
        trials.append(dict(family=family, candidate=label, tune_logloss=tune_s["logloss"],
                           tune_top3_brier=tune_s["top3_brier"], tune_spearman=tune_s["spearman"],
                           holdout_logloss=hold_s["logloss"]))
        log(f"  {family:9s} {label:42s} tune logloss {tune_s['logloss']:.4f} "
            f"brier {tune_s['top3_brier']:.4f} rho {tune_s['spearman']:.3f}  ({time.time() - t0:.0f}s)")
        return tune_s["logloss"], tune_s, hold_s

    def variant(family: str, **changes) -> dict:
        c = json.loads(json.dumps(cfg))
        c[family].update(changes)
        return c

    for family in ("race_post", "race_pre", "quali"):
        log(f"\n== {family}")
        best, base_s, _ = evaluate(family, cfg, "current model")
        best_brier = base_s["top3_brier"]

        def try_change(label: str, **changes) -> bool:
            nonlocal best, best_brier
            ll, s, _ = evaluate(family, variant(family, **changes), label)
            if ll < best - MIN_GAIN and s["top3_brier"] <= best_brier + 0.0005:
                cfg[family].update(changes)
                best, best_brier = ll, s["top3_brier"]
                log(f"     -> adopted ({label})")
                return True
            return False

        # 1. time weighting, 2. training window
        hl_results = {}
        for hl in (80, 40, 20, 10):
            ll, s, _ = evaluate(family, variant(family, half_life=hl), f"half-life {hl} events")
            hl_results[hl] = (ll, s["top3_brier"])
        hl, (ll, brier) = min(hl_results.items(), key=lambda kv: kv[1][0])
        if ll < best - MIN_GAIN and brier <= best_brier + 0.0005:
            cfg[family]["half_life"], best, best_brier = hl, ll, brier
            log(f"     -> adopted half-life {hl}")
        for seasons in (6, 9):
            try_change(f"train on {seasons} seasons", train_seasons=seasons)

        # 3.–6. features
        key = "base" if family == "quali" else "features"
        feats = list(cfg[family][key])

        def try_features(label: str, new_feats: list[str]) -> None:
            nonlocal feats
            if try_change(label, **{key: new_feats}):
                feats = new_feats

        try_features("+ driver edge (driver vs teammate)", feats + ["driver_edge"])
        try_features("+ qualifying time gaps", feats + ["quali_gap", "team_gap"])
        swapped = [{"quali_form": "quali_gap", "team_pace": "team_gap"}.get(f, f) for f in feats]
        if swapped != feats:
            try_features("time gaps instead of positions", list(dict.fromkeys(swapped)))
        if family == "race_post":
            try_features("+ grid × circuit chaos", feats + ["grid_chaos"])
            try_features("+ this weekend's quali gap", feats + ["qgap_now"])
        if family != "quali":
            try_features("+ race pace (fastest-lap gap)", feats + ["race_pace"])
        if family == "quali":
            try_change("practice as time gap", practice="fp_gap")

        # 7. likelihood depth and regularisation
        grid = [(k, l2) for k in (3, 5, 10) for l2 in (0.3, 1.0, 3.0)
                if (k, l2) != (cfg[family]["top_k"], cfg[family]["l2"])]
        results = {}
        for k, l2 in grid:
            ll, s, _ = evaluate(family, variant(family, top_k=k, l2=l2), f"top_k {k}, l2 {l2}")
            results[(k, l2)] = (ll, s["top3_brier"])
        (k, l2), (ll, brier) = min(results.items(), key=lambda kv: kv[1][0])
        if ll < best - MIN_GAIN and brier <= best_brier + 0.0005:
            cfg[family].update(top_k=k, l2=l2)
            best, best_brier = ll, brier
            log(f"     -> adopted top_k {k}, l2 {l2}")
    return cfg, trials


def holdout_report(D: dict, cfg: dict, log=print) -> pd.DataFrame:
    """Final comparison on every kind: old model, new model, simple baseline."""
    old = json.loads(json.dumps(model.DEFAULT_CONFIG))
    baselines = {"race": ["grid_log"], "sprint": ["grid_log"], "race_pre": ["quali_form"],
                 "quali": ["quali_form"]}
    rows = []
    for kind in ("race", "sprint", "race_pre", "quali"):
        fitter = Fitter(D)
        for name, c, feats in (("baseline", old, baselines[kind]), ("previous model", old, None),
                               ("new model", cfg, None)):
            sc = pd.DataFrame([score(e) for e in run(D, kind, c, fitter, features=feats)])
            for period, part in (("2020–2025", sc[sc["season"] <= TUNE_UNTIL]),
                                 (str(HOLDOUT), sc[sc["season"] == HOLDOUT])):
                if len(part):
                    rows.append(dict(kind=kind, model=name, period=period, events=len(part),
                                     **summarise(part).round(4).to_dict()))
    rep = pd.DataFrame(rows)
    log("\n" + rep.to_string(index=False))
    return rep


# --------------------------------------------------------------------- store
def config_hash(cfg: dict) -> str:
    core = {k: cfg[k] for k in ("race_post", "race_pre", "quali")}
    return hashlib.sha1(json.dumps(core, sort_keys=True).encode()).hexdigest()[:10]


def store_path(cfg: dict | None = None):
    return STORE_DIR / f"backtest_{config_hash(cfg or model.CONFIG)}.pkl"


def load_store(cfg: dict | None = None) -> list[dict]:
    try:
        return pickle.loads(store_path(cfg).read_bytes())
    except (OSError, pickle.PickleError, EOFError):
        return []


def update_store(D: dict, until: tuple | None = None, cfg: dict | None = None,
                 progress=None) -> list[dict]:
    """Add every missing walk-forward prediction before `until` to the store."""
    cfg = cfg or model.CONFIG
    events = load_store(cfg)
    have = {(e["kind"], e["season"], e["round"]) for e in events}
    todo = [(k, s, r) for k in KINDS for s, r in events_list(D, k, until=until)
            if (k, s, r) not in have]
    fitter = Fitter(D)
    for i, (kind, season, rnd) in enumerate(todo):
        if progress:
            progress((i + 1) / len(todo))
        ev = predict_event(fitter, kind, season, rnd, cfg)
        if ev is not None:
            events.append(ev)
    if todo:
        _add_calibrated(events)
        store_path(cfg).write_bytes(pickle.dumps(events))
    return [e for e in events if until is None or (e["season"], e["round"]) < until]


def _add_calibrated(events: list[dict]) -> None:
    """Walk-forward calibrated probabilities: each season uses the temperature learned
    from earlier events of the same kind only (what the app would have shown)."""
    import calibration

    for kind in ("race", "sprint", "quali", "race_pre"):
        evs = sorted((e for e in events if e["kind"] == kind), key=lambda e: (e["season"], e["round"]))
        for season in sorted({e["season"] for e in evs}):
            prior = [e for e in evs if season - 3 < e["season"] < season]
            temp = calibration.temperatures(prior, kinds=(kind,))[kind]["temperature"]
            for e in evs:
                if e["season"] == season and e.get("cal_temperature") != temp:
                    e["cal_temperature"] = temp
                    e.update({"cal_" + k: v for k, v in probabilities(e, temp).items()})


def season_summary(events: list[dict], calibrated: bool = True) -> pd.DataFrame:
    rows = []
    for e in events:
        prefix = "cal_" if calibrated and "cal_p_first" in e else ""
        rows.append(score(e, prefix))
    sc = pd.DataFrame(rows)
    if sc.empty:
        return sc
    return (sc.groupby(["season", "kind"])
            .agg(events=("round", "size"), fav_hit=("fav_hit", "mean"),
                 top3_hits=("top3_hits", "mean"), logloss=("logloss", "mean"),
                 spearman=("spearman", "mean"), pos_mae=("pos_mae", "mean"))
            .reset_index())


# --------------------------------------------------------------- consistency
def consistency(D: dict, cfg: dict) -> pd.DataFrame:
    """Per-season log-loss change (new − previous model), 2026 bootstrap interval, and
    the same comparison after the app's walk-forward calibration."""
    import calibration

    old = json.loads(json.dumps(model.DEFAULT_CONFIG))
    rows = []
    for kind in ("race", "race_pre", "quali", "sprint"):
        res = {}
        for name, c in (("old", old), ("new", cfg)):
            evs = run(D, kind, c, Fitter(D))
            for season in sorted({e["season"] for e in evs}):
                prior = [e for e in evs if season - 3 < e["season"] < season]
                t = calibration.temperatures(prior, kinds=(kind,))[kind]["temperature"]
                for e in evs:
                    if e["season"] == season:
                        e.update({"cal_" + k: v for k, v in probabilities(e, t).items()})
            res[name] = pd.DataFrame([{**score(e), "cal": score(e, "cal_")["logloss"]} for e in evs])
        m = res["old"].merge(res["new"], on=["season", "round"], suffixes=("_old", "_new"))
        m["d"] = m["logloss_new"] - m["logloss_old"]
        per = m.groupby("season")["d"].mean()
        tune_seasons = per[per.index <= TUNE_UNTIL]
        hold = m[m["season"] == HOLDOUT]["d"]
        rng = np.random.default_rng(0)
        boot = [rng.choice(hold.to_numpy(), len(hold)).mean() for _ in range(2000)] if len(hold) else [np.nan]
        rows.append({"kind": kind,
                     "better seasons": f"{(tune_seasons < 0).sum()}/{len(tune_seasons)}",
                     "Δ per season": ", ".join(f"{int(k)}: {v:+.3f}" for k, v in per.items()),
                     f"{HOLDOUT} Δ": round(hold.mean(), 3) if len(hold) else np.nan,
                     f"{HOLDOUT} 95% interval": f"{np.percentile(boot, 2.5):+.3f} to {np.percentile(boot, 97.5):+.3f}",
                     "calibrated tune old→new": f"{m[m.season <= TUNE_UNTIL]['cal_old'].mean():.3f} → "
                                                f"{m[m.season <= TUNE_UNTIL]['cal_new'].mean():.3f}",
                     f"calibrated {HOLDOUT} old→new": f"{m[m.season == HOLDOUT]['cal_old'].mean():.3f} → "
                                                      f"{m[m.season == HOLDOUT]['cal_new'].mean():.3f}"})
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


# ----------------------------------------------------------------------- CLI
def write_outputs(cfg: dict, trials: list[dict], report: pd.DataFrame,
                  consist: pd.DataFrame | None = None) -> None:
    meta = dict(created=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                tuned_on=f"{EVAL_FROM}–{TUNE_UNTIL}", holdout=HOLDOUT,
                note="Written by `python evaluate.py --tune`. Re-run to re-tune.")
    model.CONFIG_FILE.write_text(json.dumps({**{k: cfg[k] for k in ("race_post", "race_pre", "quali")},
                                             "_meta": meta}, indent=2))
    tr = pd.DataFrame(trials)
    lines = [
        "# Model research report", "",
        f"Generated {meta['created']} by `python evaluate.py --tune`.", "",
        "Every event is predicted with a model fitted only on earlier events (walk-forward).",
        f"Choices are made on {EVAL_FROM}–{TUNE_UNTIL}; {HOLDOUT} is used once, as a check.",
        f"A candidate is adopted when it lowers the winner/pole log loss by at least {MIN_GAIN}"
        " without making the top-3 Brier score worse.", "",
        "## Chosen settings", "", "```json", json.dumps({k: cfg[k] for k in ("race_post", "race_pre", "quali")},
                                                          indent=2), "```", "",
        "## Candidates tested (2020–2025)", "", _md_table(tr.drop(columns=["holdout_logloss"])), "",
        "## Final comparison", "",
        "Lower is better for logloss, top3_brier and pos_mae; higher for fav_hit, top3_hits "
        "and spearman. Baselines: grid order (race, sprint) and recent qualifying form "
        "(qualifying, race before qualifying). Probabilities are uncalibrated here; the app "
        "adds the learned calibration on top.", "",
        _md_table(report), "",
    ]
    if consist is not None:
        lines += [
            "## Is the improvement consistent?", "",
            "Winner/pole log-loss change of the new vs. the previous model per season "
            f"(negative = better), the {HOLDOUT} change with a bootstrap 95% interval, and the "
            "same comparison after the app's walk-forward calibration.", "",
            _md_table(consist), "",
        ]
    REPORT_FILE.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", action="store_true", help="research run: choose settings")
    ap.add_argument("--store", action="store_true", help="rebuild the backtest store")
    args = ap.parse_args()
    D = load_all()
    if args.tune:
        cfg, trials = tune(D)
        report = holdout_report(D, cfg)
        consist = consistency(D, cfg)
        print("\n" + consist.to_string(index=False))
        write_outputs(cfg, trials, report, consist)
        print(f"\nwrote {model.CONFIG_FILE.name} and {REPORT_FILE.name}")
        model.CONFIG = model.load_config()
    if args.store or args.tune:
        events = update_store(D)
        print(f"store: {len(events)} events -> {store_path()}")
        print(season_summary(events).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
