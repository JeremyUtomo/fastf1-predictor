"""FIA "Car Presentation Submissions": the official list of car upgrades per race.

Since 2023 every team must declare, before each Grand Prix, which parts are new and
why (performance, circuit specific, reliability). The FIA publishes this as a PDF
per event. This module downloads those PDFs politely (fia.com asks for 10 s
between requests in its robots.txt), stores them on disk so each one is fetched
only once, and parses them into one row per declared upgrade.

    python upgrades.py 2024 2026     # backfill seasons (slow: ~10 s per request)
"""
from __future__ import annotations

import os
import re
import sys
import time
import unicodedata
from difflib import get_close_matches
from pathlib import Path
from urllib.parse import quote, unquote

import pandas as pd
import requests

BASE = "https://www.fia.com"
DOCS = BASE + "/documents/championships/fia-formula-one-world-championship-14"
USER_AGENT = "fastf1-predictor/0.1 (personal research; +https://github.com/jeremyutomo/fastf1-predictor)"
CRAWL_DELAY = 10  # seconds, from fia.com/robots.txt

CACHE = Path(os.environ.get("FASTF1_CACHE", Path.home() / ".cache" / "fastf1")) / "fia_upgrades"
CACHE.mkdir(parents=True, exist_ok=True)
ITEMS_CSV = CACHE / "upgrade_items.csv"
DONE_CSV = CACHE / "events_done.csv"

# FIA team names -> Jolpica/Ergast constructorId
TEAM_KEYS = [("mclaren", "mclaren"), ("mercedes", "mercedes"), ("red bull racing", "red_bull"),
             ("racing bulls", "rb"), ("visa cash app", "rb"), ("ferrari", "ferrari"),
             ("williams", "williams"), ("aston martin", "aston_martin"), ("haas", "haas"),
             ("alpine", "alpine"), ("audi", "audi"), ("sauber", "sauber"), ("cadillac", "cadillac")]
REASONS = ["Performance", "Circuit specific", "Reliability"]

_last_request = 0.0


def _get(url: str) -> requests.Response:
    """GET with the crawl delay fia.com asks for."""
    global _last_request
    wait = CRAWL_DELAY - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    return resp


def team_id(name: str) -> str | None:
    low = name.lower()
    return next((cid for key, cid in TEAM_KEYS if key in low), None)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s).replace("grand prix of ", "").replace("grand prix", "").strip()


# ----------------------------------------------------------------- crawling
def season_events(year: int) -> list[str]:
    """Event page paths of a season (one request)."""
    html = _get(f"{DOCS}").text
    season = re.search(rf'/season/season-{year}-\d+', html)
    if not season:
        return []
    html = _get(DOCS + season.group(0)).text
    paths = sorted(set(re.findall(rf'{re.escape(season.group(0))}/event/[^"\']+', html)))
    return [p for p in paths if "Grand%20Prix" in p or "Grand Prix" in unquote(p)]


def event_pdf_url(event_path: str) -> str | None:
    html = _get(DOCS + event_path).text
    # newer documents live under /system/files/ (snake_case names), older ones under
    # /sites/default/files/ (names with spaces)
    links = re.findall(r'href="((?:/system|/sites/default)/files/decision-document/'
                       r'[^"]*car[ _-]presentation[^"]*\.pdf)"', html, flags=re.I)
    return BASE + quote(links[-1]) if links else None


_REASON = re.compile(r"^(performance|circuit specific|reliability)", re.I)
_REASON_ANY = re.compile(r"(performance|circuit specific|reliability)", re.I)


def _clean(text) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("–", "-").replace("—", "-")).strip()


def parse_pdf(path: Path) -> list[dict]:
    """One dict per declared upgrade: team, component, reason, description.

    Handles both layouts the FIA has used: the team name directly under a
    "Car Presentation" header (2025+), or as the first line of each team page
    (2024). Table rows that wrap onto continuation rows are merged.
    """
    import pdfplumber

    items, team = [], None

    def finish(cells: list[str]) -> None:
        texts = [c for c in cells[1:] if c]
        reason = next((c for c in texts if _REASON.match(c)), "")
        if not reason:  # wrapped text can push the reason mid-cell ("Load Performance - Flow")
            reason = next((c for c in texts if _REASON_ANY.search(c)), "")
        rest = [c for c in texts if c != reason]
        found = _REASON_ANY.search(reason)
        kind = found.group(1).capitalize() if found else "Other"
        items.append(dict(team=team, constructorId=team_id(team), component=rest[0] if rest else "",
                          reason=kind,
                          reason_detail=reason, change=rest[1] if len(rest) > 1 else "",
                          description=" ".join(rest[2:])[:400]))

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[1:]:  # page 1 is the FIA cover note
            lines = [ln.strip() for ln in (page.extract_text() or "").splitlines() if ln.strip()]
            header = [ln for ln in lines[:2] if not ln.lower().startswith("car presentation")]
            if header and team_id(header[0]):
                team = header[0]
            if not team:
                continue
            for table in page.extract_tables():
                current = None
                for row in table:
                    cells = [_clean(c) for c in row]
                    if cells and cells[0].isdigit():
                        if current:
                            finish(current)
                        current = cells
                    elif current and any(cells):
                        current = [f"{a} {b}".strip() for a, b in
                                   zip(current + [""] * (len(cells) - len(current)),
                                       cells + [""] * (len(current) - len(cells)))]
                if current:
                    finish(current)
    return items


def reparse_all() -> None:
    """Rebuild the parsed list from the PDFs already on disk (no downloads)."""
    done, frames = _done(), []
    for season, event in done[done["status"].str.contains("upgrades")][["season", "event"]].values:
        pdf = CACHE / f"{season}_{re.sub(r'[^A-Za-z0-9]+', '_', event)}.pdf"
        if pdf.exists():
            items = pd.DataFrame(parse_pdf(pdf))
            if len(items):
                items.insert(0, "event", event)
                items.insert(0, "season", season)
                frames.append(items)
    if frames:
        pd.concat(frames).to_csv(ITEMS_CSV, index=False)


def _done() -> pd.DataFrame:
    return pd.read_csv(DONE_CSV) if DONE_CSV.exists() else pd.DataFrame(
        columns=["season", "event", "status"])


def fetch_event(year: int, event_path: str, log=print) -> str:
    """Download + parse one event's document (skipped if already done)."""
    event = unquote(event_path.rsplit("/event/", 1)[-1])
    done = _done()
    if ((done["season"] == year) & (done["event"] == event)).any():
        return "cached"
    url = event_pdf_url(event_path)
    if not url:
        # not recorded as done: the document may simply not be published yet
        log(f"{year} {event}: no document (yet)")
        return "no document"
    if url:
        pdf = CACHE / f"{year}_{re.sub(r'[^A-Za-z0-9]+', '_', event)}.pdf"
        pdf.write_bytes(_get(url).content)
        items = pd.DataFrame(parse_pdf(pdf))
        if len(items):
            items.insert(0, "event", event)
            items.insert(0, "season", year)
            items.to_csv(ITEMS_CSV, mode="a", header=not ITEMS_CSV.exists(), index=False)
        status = f"{len(items)} upgrades"
    pd.concat([done, pd.DataFrame([dict(season=year, event=event, status=status)])]).to_csv(
        DONE_CSV, index=False)
    log(f"{year} {event}: {status}")
    return status


def fetch_current(year: int, event_name: str, log=print) -> str:
    """Fetch the document of one event by its (FastF1) name: two requests at most."""
    html = _get(DOCS).text
    season = re.search(rf'/season/season-{year}-\d+', html)
    if not season:
        return "season not found"
    path = f"{season.group(0)}/event/{quote(fia_event_name(year, event_name, html))}"
    return fetch_event(year, path, log)


def fia_event_name(year: int, event_name: str, html: str | None = None) -> str:
    """FIA's name for an event (they differ slightly, e.g. 'Grand Prix of Japan')."""
    html = html or _get(DOCS).text
    names = sorted(set(unquote(n) for n in re.findall(r'/event/([^"\']+)', html)))
    by_norm = {_norm(n): n for n in names}
    match = get_close_matches(_norm(event_name), list(by_norm), n=1, cutoff=0.5)
    return by_norm[match[0]] if match else event_name


def backfill(years: list[int], log=print) -> None:
    done = _done()
    for year in years:
        for path in season_events(year):
            event = unquote(path.rsplit("/event/", 1)[-1])
            if ((done["season"] == year) & (done["event"] == event)).any():
                continue
            try:
                fetch_event(year, path, log)
            except Exception as exc:  # keep going; a failed event is retried next run
                log(f"{year} {event}: failed ({exc})")


# ------------------------------------------------------------------ reading
def load_items() -> pd.DataFrame:
    if not ITEMS_CSV.exists():
        return pd.DataFrame(columns=["season", "event", "team", "constructorId", "component",
                                     "reason", "reason_detail", "change", "description"])
    return pd.read_csv(ITEMS_CSV).fillna("")


def events_done() -> pd.DataFrame:
    return _done()


def per_team_round(schedules: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Upgrade counts per (season, round, constructorId), rounds matched by event name.

    `schedules` maps season -> FastF1 event schedule. Events covered by the FIA
    data but without any upgrades for a team get explicit zeros.
    """
    items, done = load_items(), _done()
    rows = []
    for season, sched in schedules.items():
        sched = sched[sched["RoundNumber"] > 0]
        norm_to_round = {_norm(n): int(r) for n, r in zip(sched["EventName"], sched["RoundNumber"])}
        covered = done[(done["season"] == season) & done["status"].str.contains("upgrades")]
        for event in covered["event"]:
            m = get_close_matches(_norm(event), list(norm_to_round), n=1, cutoff=0.5)
            if not m:
                continue
            ev = items[(items["season"] == season) & (items["event"] == event)]
            for cid, _ in TEAM_KEYS:
                t = ev[ev["constructorId"] == cid]
                rows.append(dict(season=season, round=norm_to_round[m[0]], constructorId=cid,
                                 upg_total=len(t), upg_perf=int((t["reason"] == "Performance").sum()),
                                 upg_circuit=int((t["reason"] == "Circuit specific").sum())))
    return pd.DataFrame(rows).drop_duplicates(["season", "round", "constructorId"])


if __name__ == "__main__":
    if sys.argv[1:] == ["--reparse"]:
        reparse_all()
        sys.exit()
    args = [int(a) for a in sys.argv[1:]] or [2024, 2025, 2026]
    years = list(range(args[0], args[-1] + 1))
    backfill(years)
