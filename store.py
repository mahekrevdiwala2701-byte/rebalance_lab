"""
Loading the built-in data and saving the user's setup.

The original web page synced its setup (ticked holdings, benchmark choice, added files) across
devices through claude.ai. Here the setup is saved to data/saved_setup.json on the machine running
Streamlit, so everyone who opens the same app sees the same saved setup.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from engine import Asset

HERE = os.path.dirname(os.path.abspath(__file__))
BUILTIN = os.path.join(HERE, "data", "builtin.json")
SAVED = os.path.join(HERE, "data", "saved_setup.json")


def _decode_days(deltas: list[int]) -> np.ndarray:
    return np.cumsum(np.asarray(deltas, dtype=np.int64))


def _encode_days(days: np.ndarray) -> list[int]:
    d = [int(x) for x in days]
    return [d[0]] + [d[i] - d[i - 1] for i in range(1, len(d))]


def load_builtin() -> list[Asset]:
    with open(BUILTIN, encoding="utf-8") as f:
        items = json.load(f)
    return [Asset(id=f"b-{a['name']}-{a['port']}", name=a["name"], port=a["port"], cat=a.get("cat", ""),
                  builtin=True, on=a.get("on", True) is not False, also=bool(a.get("also", False)),
                  days=_decode_days(a["d"]), vals=np.asarray(a["v"], dtype=float))
            for a in items]


def load_assets() -> list[Asset]:
    """Built-in assets with the saved setup applied, plus any saved uploads."""
    assets = load_builtin()
    if not os.path.exists(SAVED):
        return assets
    try:
        with open(SAVED, encoding="utf-8") as f:
            s = json.load(f)
    except (OSError, ValueError):
        return assets
    for u in s.get("uploads", []):
        try:
            assets.append(Asset(id=u["id"], name=u["name"], port=u["port"], builtin=False,
                                on=u.get("on", True), days=_decode_days(u["d"]),
                                vals=np.asarray(u["v"], dtype=float)))
        except (KeyError, ValueError):
            continue
    off = set(s.get("off", []))
    for a in assets:
        if a.builtin and a.port != "bench":
            a.on = a.id not in off
    also = s.get("also")
    if isinstance(also, list):
        for a in assets:
            if a.port == "bench":
                a.also = a.id in also
    bench = s.get("bench")
    if bench and any(a.id == bench for a in assets):
        for a in assets:
            if a.port == "bench":
                a.on = a.id == bench
    ensure_bench(assets)
    return assets


def ensure_bench(assets: list[Asset]) -> None:
    """Exactly one benchmark is selected whenever any exists."""
    benches = [a for a in assets if a.port == "bench"]
    on = [a for a in benches if a.on]
    if not on and benches:
        first = next((a for a in benches if a.builtin), benches[0])
        first.on = True
    elif len(on) > 1:
        for a in on[1:]:
            a.on = False


def save_setup(assets: list[Asset]) -> str | None:
    """Write the setup to disk. Returns an error message, or None on success."""
    body = {
        "off": [a.id for a in assets if a.builtin and a.port != "bench" and not a.on],
        "bench": next((a.id for a in assets if a.port == "bench" and a.on), None),
        "also": [a.id for a in assets if a.port == "bench" and a.also],
        "uploads": [{"id": a.id, "name": a.name, "port": a.port, "on": a.on,
                     "d": _encode_days(a.days), "v": [float(f"{x:.6g}") for x in a.vals]}
                    for a in assets if not a.builtin],
    }
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(SAVED), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, SAVED)
        return None
    except OSError as e:
        return f"Couldn't save the setup: {e}"


def reset_setup() -> None:
    if os.path.exists(SAVED):
        os.remove(SAVED)
