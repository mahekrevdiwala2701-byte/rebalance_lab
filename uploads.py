"""
Reading uploaded Excel/CSV files into price series ("Add asset").

Port of the upload parser in the original web app: finds every Date + Close/NAV column pair,
parses many date formats, and removes isolated one-day bad values ("spikes").
"""
from __future__ import annotations

import csv
import io
import math
import re
from datetime import date, datetime

import numpy as np
import pandas as pd

from engine import from_ymd

DMY_RE = re.compile(r"^\s*(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{2,4})")
ISO_RE = re.compile(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})")
NUM_RE = re.compile(r"^\d+(\.\d+)?$")
DATE_HEAD = re.compile(r"\b(date|time|day)\b|^date|^time", re.I)
VAL_HEAD = re.compile(r"close|nav|price|value|adj|ltp|index", re.I)


def _is_blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def _excel_serial(n: float) -> float:
    return round(n) - 25569 if 20000 < n < 80000 else float("nan")


def _safe_ymd(y: int, m: int, d: int) -> float:
    try:
        return from_ymd(y, m, d)
    except ValueError:
        return float("nan")


def parse_dates(raw: list) -> list[float]:
    """Day numbers (or NaN). Handles Excel serials, dates, ISO, d/m/y or m/d/y, month names."""
    dmy = None
    for v in raw:
        if not isinstance(v, str):
            continue
        m = DMY_RE.match(v)
        if not m or len(m.group(1)) == 4:
            continue
        if int(m.group(1)) > 12:
            dmy = True
            break
        if int(m.group(2)) > 12:
            dmy = False
            break
    if dmy is None:
        dmy = True

    out = []
    for v in raw:
        if _is_blank(v):
            out.append(float("nan"))
        elif isinstance(v, (datetime, date, pd.Timestamp)):
            out.append(_safe_ymd(v.year, v.month, v.day))
        elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out.append(_excel_serial(float(v)))
        else:
            s = str(v).strip()
            m = ISO_RE.match(s)
            if m:
                out.append(_safe_ymd(int(m.group(1)), int(m.group(2)), int(m.group(3))))
                continue
            m = DMY_RE.match(s)
            if m:
                a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if y < 100:
                    y += 2000 if y < 50 else 1900
                out.append(_safe_ymd(y, b, a) if dmy else _safe_ymd(y, a, b))
                continue
            if NUM_RE.match(s):
                out.append(_excel_serial(float(s)))
                continue
            t = pd.to_datetime(s, errors="coerce")
            out.append(_safe_ymd(t.year, t.month, t.day) if not pd.isna(t) else float("nan"))
    return out


def _num(v) -> float:
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v)
    try:
        return float(re.sub(r"[,₹\s]", "", str(v if v is not None else "")))
    except ValueError:
        return float("nan")


def remove_spikes(days: list, vals: list) -> int:
    """Drop isolated one-day spikes in place. 10% threshold for equities, tighter for calm series."""
    r = sorted(abs(vals[i] / vals[i - 1] - 1) for i in range(1, len(vals)))
    med = r[len(r) >> 1] if r else 0
    thr = max(0.02, 10 * med) if med < 0.003 else 0.1
    spikes = 0
    for i in range(len(vals) - 2, 0, -1):
        a, b, c = vals[i - 1], vals[i], vals[i + 1]
        if abs(c / a - 1) < thr / 2 and abs(b / a - 1) > thr and abs(b / c - 1) > thr:
            del days[i]
            del vals[i]
            spikes += 1
    return spikes


def extract(rows: list[list], fallback_name: str) -> list[dict]:
    """Find every Date/value column pair in a sheet."""
    found = []
    hr = -1
    for i in range(min(len(rows), 15)):
        if any(DATE_HEAD.search(str(c)) for c in rows[i] if not _is_blank(c)):
            hr = i
            break
    pairs = []
    if hr < 0:
        pairs.append({"dc": 0, "vc": 1, "name": fallback_name, "start": 0})
    else:
        h = rows[hr]
        dcs = [j for j, c in enumerate(h) if not _is_blank(c) and DATE_HEAD.search(str(c))]
        for k, dc in enumerate(dcs):
            lim = dcs[k + 1] if k + 1 < len(dcs) else len(h)
            vc = next((j for j in range(dc + 1, lim) if re.search(r"close|nav", str(h[j] or ""), re.I)), -1)
            if vc < 0:
                vc = next((j for j in range(dc + 1, lim) if VAL_HEAD.search(str(h[j] or ""))), -1)
            if vc < 0:
                vc = dc + 1
            name = ""
            above = rows[hr - 1] if hr > 0 else []
            for j in range(dc, vc + 1):
                if j < len(above) and not _is_blank(above[j]):
                    name = str(above[j]).strip()
                    break
            if not name:
                name = f"{fallback_name} {k + 1}" if len(dcs) > 1 else fallback_name
            name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip() or name
            pairs.append({"dc": dc, "vc": vc, "name": name, "start": hr + 1})

    for pr in pairs:
        body = rows[pr["start"]:]
        cell = lambda r, j: r[j] if r is not None and j < len(r) else None  # noqa: E731
        days = parse_dates([cell(r, pr["dc"]) for r in body])
        vals = [_num(cell(r, pr["vc"])) for r in body]
        pts = sorted((d, v) for d, v in zip(days, vals)
                     if math.isfinite(d) and math.isfinite(v) and v > 0)
        D, V = [], []
        for d, v in pts:
            if D and D[-1] == d:
                V[-1] = v
            else:
                D.append(int(d))
                V.append(v)
        spikes = remove_spikes(D, V)
        if len(D) >= 30:
            found.append({"name": pr["name"], "days": np.array(D, dtype=np.int64),
                          "vals": np.array(V, dtype=float), "spikes": spikes})
    return found


def _sheet_rows(df: pd.DataFrame) -> list[list]:
    rows = []
    for r in df.itertuples(index=False):
        row = [None if _is_blank(v) or (not isinstance(v, str) and pd.isna(v)) else v for v in r]
        if any(v is not None for v in row):
            rows.append(row)
    return rows


def read_file(name: str, data: bytes) -> list[dict]:
    """Return every usable series in an uploaded .xlsx/.xls/.csv/.txt file."""
    base = re.sub(r"_+", " ", re.sub(r"\.[^.]+$", "", name)).strip()
    ext = name.lower().rsplit(".", 1)[-1]
    sheets: dict[str, list[list]] = {}
    if ext in ("xlsx", "xlsm", "xls"):
        book = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
        for sn, df in book.items():
            sheets[str(sn)] = _sheet_rows(df)
    else:
        text = data.decode("utf-8-sig", errors="replace")
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = [[c if c.strip() else None for c in r] for r in csv.reader(io.StringIO(text), dialect)]
        sheets["Sheet1"] = [r for r in rows if any(c is not None for c in r)]
    found = []
    for sn, rows in sheets.items():
        found += extract(rows, f"{base} {sn}" if len(sheets) > 1 else base)
    return found


def guess_port(name: str, default: str) -> str:
    return "bench" if re.search(r"\btri\b|total return|benchmark|sensex", name, re.I) else default
