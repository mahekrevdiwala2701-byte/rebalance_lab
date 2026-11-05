"""
Backtest engine for Rebalance Lab.

A direct port of the engine in the original web app (web/src/app.js), vectorised with numpy.
Dates are stored as "day numbers": whole days since 1970-01-01.
"""
from __future__ import annotations

import calendar
import itertools
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

RF = 0.06  # risk-free rate used for Sharpe
EPOCH = date(1970, 1, 1)

STRATS = [
    {"key": "bh", "label": "No rebalancing", "m": 0},
    {"key": "m6", "label": "Every 6 months", "m": 6},
    {"key": "m12", "label": "Every 12 months", "m": 12},
    {"key": "m18", "label": "Every 18 months", "m": 18},
    {"key": "m24", "label": "Every 24 months", "m": 24},
]
STRAT_BY_KEY = {s["key"]: s for s in STRATS}
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------- date helpers
def to_date(day: int) -> date:
    return EPOCH + timedelta(days=int(day))


def to_ymd(day: int) -> tuple[int, int, int]:
    d = to_date(day)
    return d.year, d.month, d.day


def from_ymd(y: int, m: int, d: int) -> int:
    return (date(y, m, d) - EPOCH).days


def days_in_month(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]


def add_months(day: int, k: int) -> int:
    y, m, d = to_ymd(day)
    m0 = m - 1 + k
    y += m0 // 12
    m = m0 % 12 + 1
    return from_ymd(y, m, min(d, days_in_month(y, m)))


def fmt_day(day: int) -> str:
    y, m, d = to_ymd(day)
    return f"{d} {MON[m - 1]} {y}"


def fmt_mon(day: int) -> str:
    y, m, _ = to_ymd(day)
    return f"{MON[m - 1]} {y}"


def month_key(day: int) -> str:
    y, m, _ = to_ymd(day)
    return f"{y}-{m:02d}"


def parse_month(s: str | None, end: bool) -> int | None:
    if not s:
        return None
    y, m = (int(x) for x in s.split("-"))
    return from_ymd(y, m, days_in_month(y, m)) if end else from_ymd(y, m, 1)


# ---------------------------------------------------------------- formatting
def _ok(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating)) and math.isfinite(x)


def pct(x, dp: int = 2) -> str:
    return f"{x * 100:.{dp}f}%" if _ok(x) else "–"


def pp(x, dp: int = 2) -> str:
    if not _ok(x):
        return "–"
    return f"{'+' if x >= 0 else '−'}{abs(x * 100):.{dp}f} pp"


def inr(x) -> str:
    """Indian digit grouping: ₹12,34,567."""
    if not _ok(x):
        return "–"
    n = int(round(x))
    neg, s = n < 0, str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return ("-" if neg else "") + "₹" + s


def fmt_num(x, dp: int = 3) -> str:
    return f"{x:.{dp}f}" if _ok(x) else "–"


# ---------------------------------------------------------------- data types
@dataclass
class Asset:
    id: str
    name: str
    port: str  # "index", "mf" or "bench"
    days: np.ndarray
    vals: np.ndarray
    cat: str = ""
    builtin: bool = True
    on: bool = True
    also: bool = False

    def signature(self) -> tuple:
        return (self.id, self.name, self.on, self.also, len(self.days))


@dataclass
class Px:
    """Aligned prices: one row per asset, one column per date."""
    days: np.ndarray
    cols: np.ndarray  # shape (k, n)


def align(assets: list[Asset], frm: float, to: float) -> Px | None:
    """Union of dates in range, forward-filled; starts when every asset has a value."""
    if not assets:
        return None
    parts = [a.days[(a.days >= frm) & (a.days <= to)] for a in assets]
    days = np.unique(np.concatenate(parts)) if parts else np.array([], dtype=np.int64)
    if len(days) == 0:
        return None
    cols = np.empty((len(assets), len(days)))
    for r, a in enumerate(assets):
        idx = np.searchsorted(a.days, days, side="right") - 1
        col = np.where(idx >= 0, a.vals[np.clip(idx, 0, None)], np.nan)
        cols[r] = col
    ok = np.all(np.isfinite(cols), axis=0)
    if not ok.any():
        return None
    s = int(np.argmax(ok))
    if len(days) - s < 30:
        return None
    return Px(days[s:], cols[:, s:])


def slice_px(px: Px | None, frm: float, to: float) -> Px | None:
    if px is None:
        return None
    i = int(np.searchsorted(px.days, frm, side="left"))
    j = int(np.searchsorted(px.days, to, side="right")) - 1
    if j - i < 1:
        return None
    return Px(px.days[i:j + 1], px.cols[:, i:j + 1])


# ---------------------------------------------------------------- simulation
@dataclass
class Sim:
    nav: np.ndarray
    nrb: int
    turn: float


def _rebalance_points(days: np.ndarray, months: int) -> list[int]:
    """Indices where a rebalance happens: first trading day on/after each scheduled date."""
    if not months:
        return []
    n, out, cur = len(days), [], 0
    nxt = add_months(int(days[0]), months)
    while True:
        idx = max(int(np.searchsorted(days, nxt, side="left")), cur + 1)
        if idx >= n:
            break
        out.append(idx)
        cur = idx
        nxt = add_months(nxt, months)
    return out


def simulate(px: Px, months: int, cost: float) -> Sim:
    """Equal-weight portfolio, reset to equal weights every `months` (0 = never).
    `cost` is charged on both the sale and the purchase side of each trade."""
    k, n = px.cols.shape
    w = 1.0 / k
    units = w / px.cols[:, 0]
    nav = np.empty(n)
    start, nrb, turn = 0, 0, 0.0
    for e in _rebalance_points(px.days, months):
        nav[start:e] = units @ px.cols[:, start:e]
        v = units * px.cols[:, e]
        tot = float(v.sum())
        traded = float(np.abs(tot * w - v).sum()) / 2
        turn += traded / tot
        tot -= traded * cost * 2
        units = tot * w / px.cols[:, e]
        nav[e] = tot
        nrb += 1
        start = e + 1
    if start < n:
        nav[start:] = units @ px.cols[:, start:]
    return Sim(nav, nrb, turn / nrb if nrb else 0.0)


def stats(days: np.ndarray, nav: np.ndarray) -> dict:
    n = len(nav)
    yrs = (days[-1] - days[0]) / 365.25
    final = nav[-1] / nav[0]
    cagr = final ** (1 / yrs) - 1 if yrs > 0 else float("nan")
    r = nav[1:] / nav[:-1] - 1
    vol = float(np.std(r, ddof=1) * math.sqrt(252)) if n > 2 else float("nan")
    peak = np.maximum.accumulate(nav)
    mdd = min(0.0, float(np.min(nav / peak - 1)))
    sharpe = (cagr - RF) / vol if vol else float("nan")
    calmar = cagr / abs(mdd) if mdd else float("inf")
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "mdd": mdd, "calmar": calmar,
            "yrs": yrs, "final": final}


def cagr_of(days: np.ndarray, v: np.ndarray) -> float:
    return (v[-1] / v[0]) ** (365.25 / (days[-1] - days[0])) - 1


def drawdown(nav: np.ndarray) -> np.ndarray:
    return nav / np.maximum.accumulate(nav) - 1


def _cover(bs: Px | None, s: int, e: int) -> bool:
    return bs is not None and len(bs.days) >= 50 and bs.days[0] <= s + 10 and bs.days[-1] >= e - 10


def rolling(px: Px | None, years: int, cost: float, bpx: Px | None, extras: list[Px]) -> list[dict]:
    """Every portfolio started on the 1st of a month and held `years` years."""
    out: list[dict] = []
    if px is None:
        return out
    y0, m0, _ = to_ymd(int(px.days[0]))
    s = from_ymd(y0, m0, 1)
    last = int(px.days[-1])
    while True:
        e = add_months(s, years * 12)
        if e - 10 > last:
            break
        sub = slice_px(px, s, e)
        if sub is not None and len(sub.days) >= 50 and sub.days[-1] >= e - 10:
            row = {"start": s, "bench": float("nan")}
            for st in STRATS:
                nav = simulate(sub, st["m"], cost).nav
                row[st["key"]] = cagr_of(sub.days, nav)
            bs = slice_px(bpx, s, e)
            if _cover(bs, s, e):
                row["bench"] = cagr_of(bs.days, bs.cols[0])
            ex = []
            for xp in extras:
                xs = slice_px(xp, s, e)
                ex.append(cagr_of(xs.days, xs.cols[0]) if _cover(xs, s, e) else float("nan"))
            row["ex"] = ex
            out.append(row)
        s = add_months(s, 1)
    return out


# ---------------------------------------------------------------- full run
def run_backtest(assets: list[Asset], port: str, cost: float, years: int,
                 frm: float, to: float, same_dates: bool) -> dict:
    pxs: dict[str, Px | None] = {port: align([a for a in assets if a.port == port and a.on], frm, to)}
    bench_a = next((a for a in assets if a.port == "bench" and a.on), None)
    pxs["bench"] = align([bench_a], frm, to) if bench_a else None
    bench_full = align([bench_a], -1e9, 1e9) if bench_a else None
    also = [(a, align([a], -1e9, 1e9)) for a in assets if a.port == "bench" and not a.on and a.also]
    also = [(a, p) for a, p in also if p is not None]

    if same_dates:
        ks = [k for k, v in pxs.items() if v is not None]
        if len(ks) > 1:
            f = max(pxs[k].days[0] for k in ks)
            t = min(pxs[k].days[-1] for k in ks)
            for k in ks:
                pxs[k] = slice_px(pxs[k], f, t)

    R: dict = {}
    px = pxs[port]
    if px is None:
        R[port] = None
    else:
        res = {"px": px, "strats": {}}
        for st in STRATS:
            sim = simulate(px, st["m"], cost)
            res["strats"][st["key"]] = {"nav": sim.nav, "nrb": sim.nrb, "turn": sim.turn,
                                        **stats(px.days, sim.nav)}
        roll = rolling(px, years, cost, bench_full, [p for _, p in also])

        extra = []
        for k, (a, bp) in enumerate(also):
            f = max(px.days[0], bp.days[0])
            t = min(px.days[-1], bp.days[-1])
            sub, bs = slice_px(px, f, t), slice_px(bp, f, t)
            if sub is None or bs is None or len(sub.days) < 30:
                continue
            sts = {s["key"]: stats(sub.days, simulate(sub, s["m"], cost).nav) for s in STRATS}
            bnav = simulate(bs, 0, 0).nav
            extra.append({"k": k, "name": a.name, "from": int(sub.days[0]), "to": int(sub.days[-1]),
                          "st": sts, "bpx": bs, "bnav": bnav, "b": stats(bs.days, bnav)})
        res["extra"] = extra

        res["bstats"] = None
        if bench_full is not None:
            bs = slice_px(bench_full, px.days[0], px.days[-1])
            if bs is not None and bs.days[0] <= px.days[0] + 10:
                res["bstats"] = stats(bs.days, simulate(bs, 0, 0).nav)

        best_count = {s["key"]: 0 for s in STRATS}
        for w in roll:
            b = max(STRATS, key=lambda s: w[s["key"]])  # first wins on ties, as in the original
            best_count[b["key"]] += 1
        res["roll"] = {"rows": roll, "n": len(roll), "best_count": best_count}
        res["assets"] = [a for a in assets if a.port == port and a.on]
        R[port] = res

    if pxs["bench"] is not None:
        nav = simulate(pxs["bench"], 0, 0).nav
        R["bench"] = {"px": pxs["bench"], "name": bench_a.name, "nav": nav, **stats(pxs["bench"].days, nav)}
    else:
        R["bench"] = None
    return R


# ---------------------------------------------------------------- best combination
def run_combos(pool: list[Asset], bench: Asset | None, cost: float, min_n: int,
               frm: float, to: float, progress=None) -> dict:
    """Test every equal-weight mix of the pool. Returns {'error': ...} or the results."""
    if len(pool) < 1:
        return {"error": "Tick at least one asset."}
    if len(pool) > 12:
        return {"error": "Tick 12 assets or fewer (that is already 4,095 mixes)."}
    last_min = min(int(a.days[-1]) for a in pool)
    px = align(pool, frm, min(to, last_min))
    if px is None:
        return {"error": "These assets don't share enough dates. Untick the one with the shortest history."}
    bst = None
    if bench is not None:
        bf = align([bench], -1e9, 1e9)
        bs = slice_px(bf, px.days[0], px.days[-1]) if bf else None
        if bs is not None and bs.days[0] <= px.days[0] + 10:
            bst = stats(bs.days, simulate(bs, 0, 0).nav)
    combos = [c for r in range(max(1, min_n), len(pool) + 1) for c in itertools.combinations(range(len(pool)), r)]
    if not combos:
        return {"error": "Tick more assets or lower the number of assets."}
    rows = []
    for i, idx in enumerate(combos):
        sub = Px(px.days, px.cols[list(idx)])
        row = {"names": [pool[k] for k in idx], "n": len(idx)}
        for s in STRATS:
            row[s["key"]] = stats(px.days, simulate(sub, s["m"], cost).nav)
        rows.append(row)
        if progress and (i % 25 == 0 or i == len(combos) - 1):
            progress(i + 1, len(combos))
    limiter = max(pool, key=lambda a: a.days[0])
    return {"rows": rows, "px": px, "bst": bst, "limiter": limiter, "pool_n": len(pool), "tested": len(combos)}


# ---------------------------------------------------------------- lump sum / XIRR
def xirr(flows: list[tuple[int, float]]) -> float:
    d0 = flows[0][0]

    def f(r):
        return sum(c * (1 + r) ** (-(d - d0) / 365) for d, c in flows)

    lo, hi = -0.99, 5.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) < 1e-9:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def median(xs) -> float:
    xs = [x for x in xs if _ok(x)]
    return float(np.median(xs)) if xs else float("nan")
