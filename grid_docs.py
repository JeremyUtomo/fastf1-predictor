"""Official starting grids from the FIA (penalties and pit-lane starts included).

The qualifying classification is not the starting grid: grid penalties (e.g. for
extra power-unit parts) and pit-lane starts change it. The stewards publish a
"Provisional Starting Grid" and later a "Final Starting Grid" PDF before every
race and sprint. This module fetches them politely via `upgrades._get` (10 s
crawl delay, as fia.com's robots.txt asks) and caches the parsed grid on disk.
"""
from __future__ import annotations

import io
import json
import re
import time
from urllib.parse import quote

import upgrades

CACHE = upgrades.CACHE.parent / "dashboard" / "grids"
CACHE.mkdir(parents=True, exist_ok=True)
RECHECK = 30 * 60  # s: a provisional (or missing) grid is re-checked after this

_ROW = re.compile(r"^(PL|\d{1,2})\s+(\d{1,3})\s+\S")


def _doc_links(html: str, sprint: bool) -> dict[str, str]:
    """{'final'|'provisional': pdf path} for the race or the sprint grid."""
    out = {}
    for path in re.findall(r'href="((?:/system|/sites/default)/files/decision-document/[^"]+\.pdf)"',
                           html):
        name = path.lower().replace("%20", " ")
        if "starting" not in name or "grid" not in name or ("sprint" in name) != sprint:
            continue
        for kind in ("final", "provisional"):
            if re.search(kind + r"[ _-]+(sprint[ _-]+)?starting[ _-]+grid", name):
                out[kind] = path  # the last link wins: newer versions come later
    return out


def parse_grid(pdf_bytes: bytes) -> dict:
    """{'positions': {car_number: grid_slot}, 'pit_lane': [car numbers], 'notes': [...]}"""
    import pdfplumber

    rows, notes, in_notes = [], [], False
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages[1:]:  # page 1 is the stewards' cover note
            for line in (page.extract_text() or "").splitlines():
                line = line.strip()
                if line.upper().startswith(("NOTES", "* PENALTIES", "PENALTIES")):
                    in_notes = True
                    continue
                if line.startswith("©"):
                    in_notes = False
                    continue
                if in_notes and line.lower().startswith("car "):
                    notes.append(line)
                    continue
                m = _ROW.match(line)
                if m and not in_notes:
                    rows.append((m.group(1), int(m.group(2))))
    positions, pit_lane = {}, []
    for slot, car in rows:
        if slot == "PL":
            pit_lane.append(car)
        else:
            positions[car] = int(slot)
    last = max(positions.values(), default=0)
    for i, car in enumerate(pit_lane, start=1):
        positions[car] = last + i  # pit-lane starters line up behind the grid
    pit_lane += [int(n) for note in notes if re.search(r"pit ?lane", note, re.I)
                 for n in re.findall(r"Car (\d+)", note)]
    return dict(positions={str(k): v for k, v in positions.items()},
                pit_lane=sorted(set(pit_lane)), notes=notes)


def starting_grid(year: int, event_name: str, sprint: bool = False) -> dict | None:
    """Latest official starting grid for a race or sprint, or None if not published.

    Returns dict(source='final'|'provisional', positions={car: slot}, pit_lane, notes).
    """
    key = re.sub(r"\W+", "_", f"{year}_{event_name}_{'sprint' if sprint else 'race'}")
    path = CACHE / f"{key}.json"
    try:
        cached = json.loads(path.read_text())
    except (OSError, ValueError):
        cached = None
    if cached and (cached.get("source") == "final" or time.time() - cached["checked"] < RECHECK):
        return cached if cached.get("positions") else None

    html = upgrades._get(upgrades.DOCS).text
    season = re.search(rf"/season/season-{year}-\d+", html)
    result = {"checked": time.time(), "source": None}
    if season:
        fia_name = upgrades.fia_event_name(year, event_name, html)
        page = upgrades._get(f"{upgrades.DOCS}{season.group(0)}/event/{quote(fia_name)}").text
        links = _doc_links(page, sprint)
        kind = "final" if "final" in links else ("provisional" if "provisional" in links else None)
        if kind:
            parsed = parse_grid(upgrades._get(upgrades.BASE + quote(links[kind])).content)
            if len(parsed["positions"]) >= 10:
                result.update(parsed, source=kind)
    path.write_text(json.dumps(result))
    return result if result.get("positions") else None
