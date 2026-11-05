"""
Rebalance Lab — Streamlit edition.

Equal-weight portfolio rebalancing backtest for Indian markets. Compares rebalancing every
6, 12, 18 or 24 months with never rebalancing, for an index portfolio and a mutual fund
portfolio, against a benchmark index.

Run:  streamlit run app.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E
from engine import STRATS, STRAT_BY_KEY, fmt_day, fmt_mon, inr, pct, pp
from store import ensure_bench, load_assets, reset_setup, save_setup
from uploads import guess_port, read_file

st.set_page_config(page_title="Rebalance Lab", page_icon="⚖️", layout="wide")

# ------------------------------------------------------------------ constants
COLORS = {"index": "#2a78d6", "mf": "#eb6834", "bench": "#5f6b7a", "b2": "#6250d6"}
GREEN, RED = "#1a7f4b", "#c0392b"
VIEWS = {
    "index": {
        "label": "Index portfolio", "short": "Index portfolio",
        "eyebrow": "EQUAL-WEIGHT INDEX BACKTEST",
        "lede": "Holds Nifty 50, Midcap 150, Smallcap 100, the Nifty G-Sec Composite bond index and gold in "
                "equal parts, rebalanced every 6 to 24 months or never, and measures it against a benchmark "
                "index. Add an index, or swap the benchmark, and every number reruns.",
    },
    "mf": {
        "label": "Mutual fund portfolio", "short": "Fund portfolio",
        "eyebrow": "EQUAL-WEIGHT MUTUAL FUND BACKTEST",
        "lede": "Holds HDFC Large Cap, HDFC Mid Cap, DSP Small Cap, a rupee gold ETF and the Nifty G-Sec "
                "Composite bond index in equal parts, rebalanced every 6 to 24 months or never, and measures "
                "it against a benchmark index. Add a fund, or swap the benchmark, and every number reruns.",
    },
}
PORT_LABEL = {"index": "Index portfolio", "mf": "Mutual fund portfolio", "bench": "Benchmark"}
SHORT_STRAT = {"bh": "No rebalancing", "m6": "6 months", "m12": "12 months", "m18": "18 months", "m24": "24 months"}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .rl-eyebrow {font-size: 12px; letter-spacing: .08em; color: #6b7280; font-weight: 600;}
      .rl-sw {display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:8px;}
      .rl-sw.dash {background:none !important; border:2px dashed #5f6b7a; width:8px; height:8px;}
      .rl-chip {display:inline-block; padding:2px 10px; margin:0 6px 6px 0; border-radius:999px;
                border:1px solid rgba(128,128,128,.35); font-size:13px;}
      .rl-sub {color:#6b7280; font-size:13.5px;}
      .rl-big {font-size:34px; font-weight:700; font-variant-numeric: tabular-nums;}
      .rl-pos {color:#1a7f4b;} .rl-neg {color:#c0392b;}
      .rl-best {font-size:11px; background:#1a7f4b; color:#fff; border-radius:4px; padding:1px 6px; margin-left:6px;}
      .rl-note {font-size:13px; color:#6b7280; margin-top:6px;}
      table td, table th {font-variant-numeric: tabular-nums;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ state
def init_state():
    ss = st.session_state
    if "assets" not in ss:
        ss.assets = load_assets()
    ss.setdefault("pending", [])          # uploaded series waiting for Add / Discard
    ss.setdefault("pid", 0)
    ss.setdefault("uploader_key", 0)
    ss.setdefault("cpool_off", set())     # assets unticked in "Best combination"
    ss.setdefault("combo", None)
    ss.setdefault("memo", {})


init_state()
ss = st.session_state


def assets() -> list[E.Asset]:
    return ss.assets


def persist(msg: str | None = None):
    err = save_setup(assets())
    if err:
        st.toast(err, icon="⚠️")
    elif msg:
        st.toast(msg)


def swatch(port_or_color: str, dashed=False) -> str:
    c = COLORS.get(port_or_color, port_or_color)
    if dashed:
        return f'<span class="rl-sw dash" style="border-color:{c}"></span>'
    return f'<span class="rl-sw" style="background:{c}"></span>'


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ------------------------------------------------------------------ header
view = st.radio("Portfolio", list(VIEWS), format_func=lambda k: VIEWS[k]["label"], horizontal=True,
                key="view", label_visibility="collapsed")
V = VIEWS[view]
st.markdown(f'<div class="rl-eyebrow">{V["eyebrow"]}</div>', unsafe_allow_html=True)
st.title("Rebalance Lab")
st.markdown(f'<p class="rl-sub" style="font-size:16px">{V["lede"]}</p>', unsafe_allow_html=True)

if ss.get("_last_view") != view:      # switching view clears the combination results
    ss.combo = None
    ss._last_view = view

# ------------------------------------------------------------------ settings (sidebar)
port_assets = [a for a in assets() if a.port == view]
if port_assets:
    lo = min(int(a.days[0]) for a in port_assets)
    hi = max(int(a.days[-1]) for a in port_assets)
else:
    lo, hi = E.from_ymd(2000, 1, 1), E.from_ymd(2026, 1, 1)
months = []
y, m, _ = E.to_ymd(lo)
y1, m1, _ = E.to_ymd(hi)
while (y, m) <= (y1, m1):
    months.append(f"{y}-{m:02d}")
    m += 1
    if m > 12:
        y, m = y + 1, 1
range_key = f"range_{view}"
cur = ss.get(range_key)
if not cur or cur[0] not in months or cur[1] not in months:
    ss[range_key] = (months[0], months[-1])


def month_label(s: str) -> str:
    y, m = s.split("-")
    return f"{E.MON[int(m) - 1]} {y}"


def full_history():
    ss[range_key] = (months[0], months[-1])


with st.sidebar:
    st.header("Settings")
    frm_s, to_s = st.select_slider("Date range", options=months, key=range_key, format_func=month_label)
    st.button("Full history", on_click=full_history, width="stretch")
    same = st.checkbox("Match the benchmark's dates", value=True, key="same")
    cost_pct = st.number_input("Cost per trade %", min_value=0.0, max_value=5.0, value=0.1, step=0.05,
                               format="%.2f", key="cost")
    years = st.selectbox("Rolling window", [3, 5, 10], index=2, format_func=lambda y: f"{y} years", key="win")
    sk = st.selectbox("Charts show", [s["key"] for s in STRATS], index=2,
                      format_func=lambda k: STRAT_BY_KEY[k]["label"], key="strat")
    st.divider()
    st.caption("Holdings, benchmark and added files are saved to `data/saved_setup.json` on the server.")
    if st.button("Restore built-in setup", width="stretch"):
        reset_setup()
        ss.assets = load_assets()
        ss.memo = {}
        ss.combo = None
        st.rerun()

cost = max(0.0, float(cost_pct or 0)) / 100
frm = E.parse_month(frm_s, False)
to = E.parse_month(to_s, True)
sl = STRAT_BY_KEY[sk]["label"].lower()


# ------------------------------------------------------------------ run backtest (memoised)
def get_result():
    key = (view, cost, years, frm, to, same, tuple(a.signature() for a in assets()))
    if key not in ss.memo:
        if len(ss.memo) > 20:
            ss.memo.clear()
        ss.memo[key] = E.run_backtest(assets(), view, cost, years, frm, to, same)
    return ss.memo[key]


with st.spinner("Running backtest…"):
    R = get_result()
r = R[view]
B = R["bench"]


# ------------------------------------------------------------------ helpers: tables & charts
def show_table(header: list[str], rows: list[list], classes: list[list[str]] | None = None):
    """Static table. classes: per-cell 'win' / 'pos' / 'neg' / ''."""
    # column names must be unique for pandas styling
    seen, cols = {}, []
    for h in header:
        h = h or " "
        while h in seen:
            h += " "
        seen[h] = 1
        cols.append(h)
    df = pd.DataFrame(rows, columns=cols)
    css = {"win": f"color:{GREEN}; font-weight:700; background-color:rgba(26,127,75,.10)",
           "pos": f"color:{GREEN}", "neg": f"color:{RED}", "": ""}
    if classes is None:
        classes = [[""] * len(cols) for _ in rows]
    grid = pd.DataFrame([[css.get(c, "") for c in row] for row in classes], columns=cols)
    sty = (df.style.apply(lambda _: grid, axis=None)
           .hide(axis="index")
           .set_properties(subset=cols[1:], **{"text-align": "right"})
           .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]))
    st.table(sty)


def down(days, vals, mx=1200):
    n = len(days)
    step = max(1, math.ceil(n / mx))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return np.asarray(days)[idx], np.asarray(vals)[idx]


def to_dt(days):
    return pd.to_datetime(np.asarray(days, dtype="int64"), unit="D")


def line_chart(series: list[dict], *, height=340, log=False, zero=False, area=False, yaxis: dict | None = None,
               hover_fmt=lambda v: f"{v:.2f}", xlabel_fmt=fmt_day, key=None):
    fig = go.Figure()
    for s in series:
        x, yv = s["x"], np.asarray(s["y"], dtype=float)
        dash = s.get("dash")
        line = dict(color=s["color"], width=2, dash={True: "dash", "dot": "dot"}.get(dash, "solid"))
        custom = [f"{xlabel_fmt(int(d))}: {hover_fmt(v)}" for d, v in zip(x, yv)]
        fig.add_trace(go.Scatter(
            x=to_dt(x), y=yv, name=s["name"], mode="lines", line=line, customdata=custom,
            hovertemplate="%{customdata}<extra>" + esc(s["name"]) + "</extra>",
            fill="tozeroy" if area and not dash else None,
            fillcolor=_rgba(s["color"], 0.10) if area and not dash else None,
        ))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      template="plotly_white")
    ya = dict(type="log" if log else "linear", gridcolor="rgba(128,128,128,.15)")
    if zero:
        ya.update(zeroline=True, zerolinecolor="rgba(128,128,128,.6)", zerolinewidth=1)
    if yaxis:
        ya.update(yaxis)
    fig.update_yaxes(**ya)
    fig.update_xaxes(gridcolor="rgba(128,128,128,.10)")
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def log_ticks(lo_v: float, hi_v: float) -> list[float]:
    out = []
    for e in range(math.floor(math.log10(lo_v)), math.ceil(math.log10(hi_v)) + 1):
        for b in (1, 2, 5):
            v = b * 10 ** e
            if lo_v * 0.999 <= v <= hi_v * 1.001:
                out.append(v)
    if len(out) > 8:
        out = [v for v in out if str(v).startswith("1")]
    return out


def rupee_axis(v: float) -> str:
    if v >= 1e7:
        return f"₹{v / 1e7:.{0 if v >= 1e8 else 1}f}Cr"
    return f"₹{v / 1e5:.{1 if v < 1e5 else 0}f}L"


# ------------------------------------------------------------------ summary
c1, c2 = st.columns([1, 1], gap="large")
with c1:
    with st.container(border=True):
        st.markdown(f"#### {swatch(view)}{V['label']}", unsafe_allow_html=True)
        if r is None:
            st.markdown('<div class="rl-sub">Tick at least one holding with data in this date range.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown("".join(f'<span class="rl-chip">{esc(a.name)}</span>' for a in r["assets"]),
                        unsafe_allow_html=True)
            st.markdown(f'<div class="rl-sub">{fmt_day(r["px"].days[0])} to {fmt_day(r["px"].days[-1])} · '
                        f'{r["strats"]["bh"]["yrs"]:.1f} years</div>', unsafe_allow_html=True)
            best = max(STRATS, key=lambda s: r["strats"][s["key"]]["cagr"])
            html = ['<table style="width:100%;margin-top:8px"><tr><th style="text-align:left">Strategy</th>'
                    '<th style="text-align:right">CAGR</th><th style="text-align:right">Max fall</th></tr>']
            for s in STRATS:
                x = r["strats"][s["key"]]
                tag = '<span class="rl-best">Best</span>' if s is best else ""
                html.append(f'<tr><td>{s["label"]}{tag}</td><td style="text-align:right"><b>{pct(x["cagr"])}</b></td>'
                            f'<td style="text-align:right;color:#6b7280">{pct(x["mdd"], 1)}</td></tr>')
            html.append("</table>")
            st.markdown("".join(html), unsafe_allow_html=True)
            rl = r["roll"]
            if rl["n"]:
                order = sorted(STRATS, key=lambda s: -rl["best_count"][s["key"]])
                parts = []
                for i, s in enumerate(order):
                    share = round(rl["best_count"][s["key"]] / rl["n"] * 100)
                    t = f'{s["label"].replace("Every ", "").replace("No rebalancing", "no rebalancing")} {share}%'
                    parts.append(f"<b>{t}</b>" if i == 0 else t)
                st.markdown(f'<div class="rl-note">Most often the best in {rl["n"]} rolling {years}-year periods: '
                            f'{", ".join(parts)}.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="rl-note">Not enough history for a rolling {years}-year comparison.</div>',
                            unsafe_allow_html=True)

with c2:
    with st.container(border=True):
        if r is not None and B is not None:
            BB = r["bstats"] or B
            g = r["strats"][sk]["cagr"] - BB["cagr"]
            st.markdown(f'<div class="rl-eyebrow">PORTFOLIO MINUS {esc(B["name"]).upper()} CAGR</div>'
                        f'<div class="rl-big {"rl-pos" if g >= 0 else "rl-neg"}">{pp(g, 1)}</div>'
                        f'<div class="rl-sub">a year, portfolio {sl}, over the portfolio\'s dates. Benchmark: '
                        f'{pct(BB["cagr"])} CAGR, max fall {pct(BB["mdd"], 1)}.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="rl-eyebrow">PORTFOLIO MINUS BENCHMARK CAGR</div><div class="rl-big">–</div>'
                        '<div class="rl-sub">Choose a benchmark under Holdings to compare.</div>',
                        unsafe_allow_html=True)
        if r is not None:
            rb = [s for s in STRATS if s["m"]]
            vals = [r["strats"][s["key"]]["cagr"] for s in rb]
            top, bot = rb[int(np.argmax(vals))], rb[int(np.argmin(vals))]
            hi_v, lo_v = max(vals), min(vals)
            st.markdown(f'<div class="rl-eyebrow" style="margin-top:16px">EFFECT OF REBALANCING FREQUENCY</div>'
                        f'<div class="rl-sub">Across 6, 12, 18 and 24 months, CAGR runs from {pct(lo_v)} '
                        f'({bot["label"].lower()}) to {pct(hi_v)} ({top["label"].lower()}), a spread of '
                        f'{pp(hi_v - lo_v, 2)[1:]}. Not rebalancing: {pct(r["strats"]["bh"]["cagr"])}.</div>',
                        unsafe_allow_html=True)

# ------------------------------------------------------------------ against the benchmark
BENCH_COLS = ["", "CAGR", "vs benchmark", "Volatility", "Sharpe", "Max fall",
              f"Beat it (rolling {years} yr)", "Median rolling gap"]


def _cmp_row(label, x, bref, d):
    win = (sum(1 for v in d if v > 0) / len(d)) if d else float("nan")
    med = E.median(d)
    ex = x["cagr"] - bref["cagr"]
    row = [label, pct(x["cagr"]), pp(ex, 2), pct(x["vol"]), E.fmt_num(x["sharpe"]), pct(x["mdd"], 1),
           f"{round(win * 100)}% of {len(d)}" if d else "–", pp(med, 2)]
    cls = ["", "", "pos" if ex >= 0 else "neg", "", "pos" if x["sharpe"] >= bref["sharpe"] else "neg",
           "pos" if x["mdd"] >= bref["mdd"] else "neg",
           ("pos" if win >= 0.5 else "neg") if d else "",
           ("pos" if med >= 0 else "neg") if math.isfinite(med) else ""]
    return row, cls, win


if B is not None:
    with st.container(border=True):
        st.subheader(f"Against the benchmark: {B['name']}")
        st.markdown(f'<div class="rl-sub">Portfolio {sl} · {fmt_mon(B["px"].days[0])} to '
                    f'{fmt_mon(B["px"].days[-1])}</div>', unsafe_allow_html=True)
        rel_series = []
        rows, classes = [], []
        if r is not None:
            x = r["strats"][sk]
            wins = [w for w in r["roll"]["rows"] if math.isfinite(w["bench"])]
            d = [w[sk] - w["bench"] for w in wins]
            row, cls, win = _cmp_row(V["label"], x, r["bstats"] or B, d)
            rows.append(row)
            classes.append(cls)
            if wins:
                rel_series.append({"name": f"vs {B['name']}", "color": COLORS[view], "x": [w["start"] for w in wins],
                                   "y": d, "win": win, "n": len(d), "vs": B["name"]})
        rows.append([B["name"], pct(B["cagr"]), "–", pct(B["vol"]), E.fmt_num(B["sharpe"]), pct(B["mdd"], 1), "–", "–"])
        classes.append([""] * 8)
        show_table(BENCH_COLS, rows, classes)

        tri = any(t in B["name"].lower() for t in ("tri", "total"))
        note = ""
        if tri:
            if view == "mf":
                note = (f"{B['name']} includes reinvested dividends, and so do fund NAVs, so this is a like-for-like "
                        "comparison. Fund NAVs are also after the fund's expenses. The three equity funds were picked "
                        "knowing they did well, which flatters the result. ")
            else:
                note = (f"{B['name']} includes reinvested dividends. In the default portfolio, Nifty 50, Midcap 150 and "
                        "Smallcap 100 are price indices without dividends, which holds the portfolio back by roughly "
                        "0.8–1.1 pp a year here. The G-Sec Composite includes its interest income. ")
        note += ("Green means better than the benchmark over the same dates as that portfolio: higher return, higher "
                 "Sharpe, or a smaller max fall.")
        if not same:
            note += " The benchmark row shows its own full history."
        st.markdown(f'<div class="rl-note">{esc(note)}</div>', unsafe_allow_html=True)

        if r is not None:
            for x in r["extra"]:
                st.markdown(f"##### Also compared: {x['name']} · {fmt_mon(x['from'])} to {fmt_mon(x['to'])}")
                y = x["st"][sk]
                wins = [w for w in r["roll"]["rows"] if math.isfinite(w["ex"][x["k"]])]
                d = [w[sk] - w["ex"][x["k"]] for w in wins]
                row, cls, win = _cmp_row(V["label"] + " (same dates)", y, x["b"], d)
                b = x["b"]
                show_table(BENCH_COLS, [row, [x["name"], pct(b["cagr"]), "–", pct(b["vol"]), E.fmt_num(b["sharpe"]),
                                              pct(b["mdd"], 1), "–", "–"]], [cls, [""] * 8])
                st.markdown(f'<div class="rl-note">{esc(x["name"])} has data only from {fmt_day(x["from"])}, so this '
                            f'compares a portfolio started that day, {sl}, with {esc(x["name"])} over the same dates. '
                            "It doesn't change the dates used anywhere else.</div>", unsafe_allow_html=True)
                if wins:
                    rel_series.append({"name": f"vs {x['name']}", "color": COLORS["b2"], "dash": "dot",
                                       "x": [w["start"] for w in wins], "y": d, "win": win, "n": len(d),
                                       "vs": x["name"]})

        st.markdown(f"##### Portfolio minus benchmark CAGR, rolling {years} years")
        if rel_series:
            st.markdown(" · ".join(f"Portfolio beat {esc(s['vs'])} in <b>{round(s['win'] * 100)}%</b> of {s['n']} "
                                   "periods" for s in rel_series), unsafe_allow_html=True)
            line_chart([{**s, "y": np.asarray(s["y"]) * 100} for s in rel_series], height=280, zero=True,
                       yaxis=dict(ticksuffix="pp", tickformat="+.0f"),
                       hover_fmt=lambda v: pp(v / 100, 2), xlabel_fmt=lambda d: "Started " + fmt_mon(d), key="brel")
        else:
            st.info("Not enough overlapping history for this window.")
        st.caption("Each point is a portfolio started that month and held for the rolling window. "
                   "Above zero, the portfolio beat the benchmark.")

# ------------------------------------------------------------------ best combination
with st.container(border=True):
    st.subheader("Best combination")
    st.markdown('<div class="rl-sub">Tests every mix of the assets you tick below, each split equally.</div>',
                unsafe_allow_html=True)
    pool_all = [a for a in assets() if a.port == view]
    st.markdown("**Assets to test**")
    pcols = st.columns(min(4, max(1, len(pool_all))))
    for i, a in enumerate(pool_all):
        with pcols[i % len(pcols)]:
            on = st.checkbox(a.name, value=a.id not in ss.cpool_off, key=f"cpool_{view}_{a.id}")
            (ss.cpool_off.discard if on else ss.cpool_off.add)(a.id)
    cc1, cc2, cc3 = st.columns([1, 1, 1])
    with cc1:
        min_n = st.selectbox("Number of assets", [2, 1, 3, 4],
                             format_func=lambda n: "Any, including 1" if n == 1 else f"{n} or more", key="csize")
    with cc2:
        rank_by = st.selectbox("Rank by", ["cagr", "sharpe", "calmar"], key="crank",
                               format_func=lambda k: {"cagr": "CAGR", "sharpe": "Sharpe",
                                                      "calmar": "Return per unit of max fall (Calmar)"}[k])
    with cc3:
        st.write("")
        go_btn = st.button("Find best combination", type="primary", width="stretch")

    if go_btn:
        pool = [a for a in pool_all if a.id not in ss.cpool_off]
        bar = st.progress(0.0, text="Testing mixes…")

        def prog(i, n):
            bar.progress(i / n, text=f"Testing {i:,} of {n:,} mixes…")

        bench_a = next((a for a in assets() if a.port == "bench" and a.on), None)
        res = E.run_combos(pool, bench_a, cost, min_n, frm, to, prog)
        bar.empty()
        ss.combo = res

    res = ss.combo
    if res is not None:
        if "error" in res:
            st.warning(res["error"])
        else:
            px, bst = res["px"], res["bst"]
            rl_name = {"cagr": "CAGR", "sharpe": "Sharpe", "calmar": "Calmar"}[rank_by]
            st.success(f"Tested {res['tested']:,} mixes.")
            st.markdown(f'<div class="rl-sub">{fmt_day(px.days[0])} to {fmt_day(px.days[-1])} '
                        f'({(px.days[-1] - px.days[0]) / 365.25:.1f} years), the dates all {res["pool_n"]} ticked '
                        f'assets share. {esc(res["limiter"].name)} has the shortest history and sets the start. '
                        f'Ranked by {rl_name}, {sl}.</div>', unsafe_allow_html=True)
            ranked = sorted(res["rows"], key=lambda row: -row[sk][rank_by])

            def combo_table(items, first):
                header = [first, "Mix", "CAGR", "vs benchmark", "Volatility", "Sharpe", "Max fall"] + \
                         [f'{s["m"]} months' if s["m"] else "No rebal." for s in STRATS]
                rows, classes = [], []
                for lab, row in items:
                    x = row[sk]
                    ex = x["cagr"] - bst["cagr"] if bst else float("nan")
                    rows.append([lab, ", ".join(a.name for a in row["names"]), pct(x["cagr"]),
                                 pp(ex, 2) if bst else "–", pct(x["vol"]), E.fmt_num(x["sharpe"]), pct(x["mdd"], 1)] +
                                [pct(row[s["key"]]["cagr"]) for s in STRATS])
                    classes.append(["", "", "", ("pos" if ex >= 0 else "neg") if bst else ""] + [""] * (len(header) - 4))
                show_table(header, rows, classes)

            combo_table([(f"#{i + 1}", row) for i, row in enumerate(ranked[:15])], "Rank")
            st.markdown("##### Best mix for each number of assets")
            by = {}
            for row in ranked:
                by.setdefault(row["n"], row)
            combo_table([(f"{n} asset{'s' if n > 1 else ''}", by[n]) for n in sorted(by)], "Size")
            note = (f"Benchmark over the same dates: {pct(bst['cagr'])} CAGR, Sharpe {E.fmt_num(bst['sharpe'])}, "
                    f"max fall {pct(bst['mdd'], 1)}. " if bst else "")
            note += ("The top mix is the best one in hindsight for this exact period. Mixes with fewer assets usually "
                     "top a CAGR ranking because they hold only the past winners, and they rarely stay on top in the "
                     "next period. Check the rolling charts with that mix ticked before relying on it. If you mix "
                     "price indices with TRI files, the TRI ones have an advantage of roughly 1–1.5 pp a year from "
                     "dividends.")
            st.markdown(f'<div class="rl-note">{esc(note)}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ charts
extras = r["extra"] if r is not None else []
with st.container(border=True):
    st.subheader("Growth of ₹1,00,000")
    if r is None:
        st.info("Tick at least one holding with data in this date range.")
    else:
        nav = r["strats"][sk]["nav"]
        g = []
        x, yv = down(r["px"].days, nav * 1e5)
        g.append({"name": V["short"], "color": COLORS[view], "x": x, "y": yv})
        if B is not None:
            x, yv = down(B["px"].days, B["nav"] * 1e5)
            g.append({"name": B["name"], "color": COLORS["bench"], "dash": True, "x": x, "y": yv})
        for xx in extras:
            i = int(np.searchsorted(r["px"].days, xx["from"], side="left"))
            i = min(i, len(nav) - 1)
            x, yv = down(xx["bpx"].days, xx["bnav"] * nav[i] * 1e5)
            g.append({"name": f"{xx['name']} (from {fmt_mon(xx['from'])})", "color": COLORS["b2"], "dash": "dot",
                      "x": x, "y": yv})
        allv = np.concatenate([np.asarray(s["y"]) for s in g])
        ticks = log_ticks(float(allv.min()), float(allv.max()))
        line_chart(g, log=True, height=360, hover_fmt=inr,
                   yaxis=dict(tickvals=ticks, ticktext=[rupee_axis(v) for v in ticks]), key="growth")
        st.caption("Log scale, so equal vertical distances mean equal percentage moves.")

d1, d2 = st.columns(2, gap="large")
with d1:
    with st.container(border=True):
        st.subheader("Drawdown from peak")
        if r is not None:
            dd = []
            x, yv = down(r["px"].days, E.drawdown(r["strats"][sk]["nav"]))
            dd.append({"name": V["short"], "color": COLORS[view], "x": x, "y": yv})
            if B is not None:
                x, yv = down(B["px"].days, E.drawdown(B["nav"]))
                dd.append({"name": B["name"], "color": COLORS["bench"], "dash": True, "x": x, "y": yv})
            for xx in extras:
                x, yv = down(xx["bpx"].days, E.drawdown(xx["bnav"]))
                dd.append({"name": xx["name"], "color": COLORS["b2"], "dash": "dot", "x": x, "y": yv})
            line_chart(dd, area=True, height=300, yaxis=dict(tickformat=".0%", range=None),
                       hover_fmt=lambda v: pct(v, 1), key="dd")

with d2:
    with st.container(border=True):
        k1, k2 = st.columns(2)
        with k1:
            ka = st.selectbox("Compare", [s["key"] for s in STRATS], index=1, format_func=SHORT_STRAT.get, key="cmpa")
        with k2:
            kb = st.selectbox("Against", [s["key"] for s in STRATS], index=2, format_func=SHORT_STRAT.get, key="cmpb")
        nm = lambda k: f'{STRAT_BY_KEY[k]["m"]}-month' if STRAT_BY_KEY[k]["m"] else "No rebalancing"  # noqa: E731
        nm_b = f'{STRAT_BY_KEY[kb]["m"]}-month' if STRAT_BY_KEY[kb]["m"] else "no rebalancing"
        st.markdown(f"##### {nm(ka)} minus {nm_b} CAGR, rolling {years} years")
        if r is not None and r["roll"]["n"]:
            rows_ = r["roll"]["rows"]
            diff = [w[ka] - w[kb] for w in rows_]
            n = len(diff)
            winr = sum(1 for v in diff if v > 0) / n
            st.markdown(f"{nm(ka)} won <b>{round(winr * 100)}%</b> of {n} periods · median gap "
                        f"<b>{pp(E.median(diff), 2)}</b> a year", unsafe_allow_html=True)
            line_chart([{"name": V["short"], "color": COLORS[view], "x": [w["start"] for w in rows_],
                         "y": np.asarray(diff) * 100}], height=262, zero=True,
                       yaxis=dict(ticksuffix="pp", tickformat="+.2f"), hover_fmt=lambda v: pp(v / 100, 2),
                       xlabel_fmt=lambda d: "Started " + fmt_mon(d), key="roll")
        else:
            st.info("Not enough history for this window.")
        st.caption("Each point is a fresh portfolio started that month. Above zero, the first choice won.")

# ------------------------------------------------------------------ lump sum (XIRR)
with st.container(border=True):
    st.subheader("Lump-sum returns (XIRR)")
    amt = st.number_input("Lump sum ₹", min_value=1000, value=100000, step=10000, key="sipamt")
    amt = max(1000, float(amt or 100000))
    if r is None:
        st.info("Tick at least one holding.")
    else:
        d0, d1_ = int(r["px"].days[0]), int(r["px"].days[-1])
        st.markdown(f'<div class="rl-sub">{inr(amt)} invested on {fmt_day(d0)}, valued on {fmt_day(d1_)}</div>',
                    unsafe_allow_html=True)
        bx = bf = float("nan")
        if B is not None:
            bs = E.slice_px(B["px"], d0, d1_)
            if bs is not None and bs.days[0] <= d0 + 10:
                bf = bs.cols[0][-1] / bs.cols[0][0]
                bx = E.xirr([(int(bs.days[0]), -amt), (int(bs.days[-1]), amt * bf)])
        items = []
        for s in STRATS:
            f = r["strats"][s["key"]]["final"]
            items.append((s, amt * f, E.xirr([(d0, -amt), (d1_, amt * f)])))
        best_x = max(items, key=lambda t: t[2] if math.isfinite(t[2]) else -1e9)
        rows, classes = [], []
        for s, val, xr in items:
            ex = xr - bx
            rows.append([s["label"], inr(amt), inr(val), inr(val - amt), pct(xr), pp(ex, 2) if math.isfinite(ex) else "–"])
            classes.append(["", "", "", "", "win" if s is best_x[0] else "",
                            ("pos" if ex >= 0 else "neg") if math.isfinite(ex) else ""])
        if math.isfinite(bx):
            rows.append([f"{B['name']} (benchmark)", inr(amt), inr(amt * bf), inr(amt * bf - amt), pct(bx), "–"])
            classes.append([""] * 6)
        for xx in extras:
            f = xx["bnav"][-1] / xx["bnav"][0]
            xxr = E.xirr([(xx["from"], -amt), (xx["to"], amt * f)])
            pv = amt * xx["st"][sk]["final"]
            px2 = E.xirr([(xx["from"], -amt), (xx["to"], pv)])
            rows.append([f"{xx['name']} (from {fmt_day(xx['from'])})", inr(amt), inr(amt * f), inr(amt * f - amt),
                         pct(xxr), f"Portfolio {pct(px2)} from same date"])
            classes.append([""] * 6)
        show_table(["Strategy", "Invested", "Value today", "Gain", "XIRR", "XIRR vs benchmark"], rows, classes)
        st.markdown('<div class="rl-note">XIRR is the annual return on one investment made at the start and held to '
                    'the end date, counting a year as 365 days. For a single lump sum it equals CAGR (CAGR above '
                    'counts 365.25 days, so the two can differ in the second decimal). Green marks the best '
                    'strategy.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ full comparison
with st.container(border=True):
    st.subheader("Full comparison")
    st.markdown(f'<div class="rl-sub">Cost {cost * 100:.2f}% per trade side. Green marks the best of the five '
                'strategies.</div>', unsafe_allow_html=True)
    cols = ["Strategy", "₹1L becomes", "CAGR", "Volatility", "Sharpe", "Max fall", "Calmar", "Rebalances", "Avg moved"]
    if r is None:
        st.markdown(f"**{V['label']}** · no data")
    else:
        st.markdown(f"{swatch(view)}**{V['label']}** · {fmt_mon(r['px'].days[0])} to {fmt_mon(r['px'].days[-1])}",
                    unsafe_allow_html=True)
        best_of = {f: max(STRATS, key=lambda s: r["strats"][s["key"]][f])["key"]
                   for f in ("cagr", "sharpe", "mdd", "calmar")}
        rows, classes = [], []
        for s in STRATS:
            x = r["strats"][s["key"]]
            rows.append([s["label"], inr(x["final"] * 1e5), pct(x["cagr"]), pct(x["vol"]), E.fmt_num(x["sharpe"]),
                         pct(x["mdd"], 1), E.fmt_num(x["calmar"]), str(x["nrb"]), pct(x["turn"], 1) if s["m"] else "–"])
            w = lambda f: "win" if best_of[f] == s["key"] else ""  # noqa: E731
            classes.append(["", "", w("cagr"), "", w("sharpe"), w("mdd"), w("calmar"), "", ""])
        show_table(cols, rows, classes)
    if B is not None:
        st.markdown(f"{swatch('bench', dashed=True)}**Benchmark: {esc(B['name'])}** · {fmt_mon(B['px'].days[0])} to "
                    f"{fmt_mon(B['px'].days[-1])}", unsafe_allow_html=True)
        show_table(cols, [["Index as published", inr(B["final"] * 1e5), pct(B["cagr"]), pct(B["vol"]),
                           E.fmt_num(B["sharpe"]), pct(B["mdd"], 1), E.fmt_num(B["calmar"]), "–", "–"]])
    for xx in extras:
        b = xx["b"]
        st.markdown(f"**Also compared: {esc(xx['name'])}** · {fmt_mon(xx['from'])} to {fmt_mon(xx['to'])}")
        show_table(cols, [["Index as published", inr(xx["bnav"][-1] / xx["bnav"][0] * 1e5), pct(b["cagr"]),
                           pct(b["vol"]), E.fmt_num(b["sharpe"]), pct(b["mdd"], 1), E.fmt_num(b["calmar"]), "–", "–"]])


# ------------------------------------------------------------------ holdings & benchmark
def toggle_holding(aid: str, wkey: str):
    a = next(x for x in assets() if x.id == aid)
    a.on = bool(ss[wkey])
    persist()


def remove_asset(aid: str):
    a = next(x for x in assets() if x.id == aid)
    ss.assets = [x for x in assets() if x.id != aid]
    if a.port == "bench" and a.on:
        ensure_bench(ss.assets)
    persist(f"Removed {a.name}.")


def pick_bench(wkey: str):
    chosen = ss[wkey]
    for a in assets():
        if a.port == "bench":
            a.on = a.id == chosen
    persist()


def toggle_also(aid: str, wkey: str):
    a = next(x for x in assets() if x.id == aid)
    a.also = bool(ss[wkey])
    persist()


def meta_line(a: E.Asset) -> str:
    s = E.stats(a.days, a.vals)
    return f"{fmt_mon(a.days[0])} to {fmt_mon(a.days[-1])} · CAGR {pct(s['cagr'])} · max fall {pct(s['mdd'], 1)}"


with st.container(border=True):
    st.subheader("Holdings and benchmark")
    st.markdown('<div class="rl-sub">The portfolio splits money equally across its ticked holdings.</div>',
                unsafe_allow_html=True)
    h1, h2 = st.columns(2, gap="large")
    with h1:
        st.markdown(f"{swatch(view)}**{V['label']}**", unsafe_allow_html=True)
        plist = [a for a in assets() if a.port == view]
        n_on = sum(a.on for a in plist)
        for a in plist:
            wkey = f"hold_{a.id}"
            ss[wkey] = a.on
            a1, a2, a3 = st.columns([5, 2, 3])
            with a1:
                st.checkbox(a.name + ("" if a.builtin else "  · Added"), key=wkey,
                            on_change=toggle_holding, args=(a.id, wkey))
                st.caption(meta_line(a))
            with a2:
                st.markdown(f"<div style='text-align:right;padding-top:6px'>"
                            f"{(100 / n_on if a.on and n_on else 0):.1f}%</div>", unsafe_allow_html=True)
            with a3:
                if not a.builtin:
                    st.button("Remove", key=f"rm_{a.id}", on_click=remove_asset, args=(a.id,))
    with h2:
        st.markdown(f"{swatch('bench', dashed=True)}**Benchmark**", unsafe_allow_html=True)
        blist = [a for a in assets() if a.port == "bench"]
        if blist:
            bkey = "bench_pick_" + str(abs(hash(tuple(a.id for a in blist))))
            ss[bkey] = next((a.id for a in blist if a.on), blist[0].id)
            st.radio("Main benchmark", [a.id for a in blist], key=bkey, on_change=pick_bench, args=(bkey,),
                     format_func=lambda i: next(a.name + ("" if a.builtin else "  · Added")
                                                for a in blist if a.id == i))
            for a in blist:
                b1, b2, b3 = st.columns([4, 4, 3])
                with b1:
                    st.markdown(f"**{esc(a.name)}**" + (" · Main" if a.on else ""))
                    st.caption(meta_line(a))
                with b2:
                    if not a.on:
                        akey = f"also_{a.id}"
                        ss[akey] = a.also
                        st.checkbox("Compare too", key=akey, on_change=toggle_also, args=(a.id, akey))
                with b3:
                    if not a.builtin:
                        st.button("Remove", key=f"rmb_{a.id}", on_click=remove_asset, args=(a.id,))
        st.caption("The selected benchmark is the main one. Tick Compare too to also show another one beside it. "
                   "To add one, add a file and choose Benchmark.")

    # ---- add asset
    st.markdown("---")
    st.markdown("**Add asset**")
    st.markdown('<div class="rl-sub">Upload an Excel or CSV file with a date column and a close or NAV column. Use a '
                'TRI file for a dividend-inclusive comparison. A sheet with several Date/Close pairs adds each '
                'one.</div>', unsafe_allow_html=True)
    files = st.file_uploader("Add asset", type=["xlsx", "xls", "csv", "txt"], accept_multiple_files=True,
                             key=f"uploader_{ss.uploader_key}", label_visibility="collapsed")
    if files:
        for f in files:
            try:
                found = read_file(f.name, f.getvalue())
            except Exception:  # noqa: BLE001 - any unreadable file gets the same message
                st.toast(f"Couldn't read {f.name}. Save it as .xlsx or .csv and try again.", icon="⚠️")
                continue
            if not found:
                st.toast(f"No usable series in {f.name}. The file needs a date column and a numeric close or NAV "
                         "column with at least 30 rows.", icon="⚠️")
            for s in found:
                ss.pid += 1
                ss.pending.append({"pid": ss.pid, **s})
        ss.uploader_key += 1   # clears the uploader so the same files aren't read twice
        st.rerun()

    def add_pending(pid: int):
        p = next(x for x in ss.pending if x["pid"] == pid)
        name = (ss.get(f"pn_{pid}") or "").strip() or p["name"]
        port = ss.get(f"pp_{pid}") or guess_port(p["name"], view)
        if port == "bench":
            for b in assets():
                if b.port == "bench":
                    b.on = False
        a = E.Asset(id=f"u-{name}-{port}-{pid}-{len(p['days'])}", name=name, port=port, builtin=False, on=True,
                    days=p["days"], vals=p["vals"])
        ss.assets.append(a)
        ss.pending = [x for x in ss.pending if x["pid"] != pid]
        if port == "bench":
            msg = f"{name} is now the benchmark. Results updated."
        else:
            msg = (f"Added {name} to the {'mutual fund' if port == 'mf' else 'index'} portfolio." +
                   (" Results updated." if port == view else " Switch to that view to see it."))
        persist(msg)

    def discard_pending(pid: int):
        ss.pending = [x for x in ss.pending if x["pid"] != pid]

    for p in ss.pending:
        pid = p["pid"]
        s = E.stats(p["days"], p["vals"])
        info = (f"{len(p['days']):,} rows · {fmt_day(p['days'][0])} to {fmt_day(p['days'][-1])} · CAGR {pct(s['cagr'])}" +
                (f" · {p['spikes']} bad one-day value{'s' if p['spikes'] > 1 else ''} removed" if p["spikes"] else ""))
        with st.container(border=True):
            q1, q2, q3, q4 = st.columns([3, 2, 1, 1])
            with q1:
                st.text_input("Name", value=p["name"], key=f"pn_{pid}")
            with q2:
                opts = ["index", "mf", "bench"]
                st.selectbox("Add to", opts, index=opts.index(guess_port(p["name"], view)),
                             format_func=PORT_LABEL.get, key=f"pp_{pid}")
            with q3:
                st.write("")
                st.button("Add", key=f"padd_{pid}", type="primary", on_click=add_pending, args=(pid,),
                          width="stretch")
            with q4:
                st.write("")
                st.button("Discard", key=f"pdis_{pid}", on_click=discard_pending, args=(pid,),
                          width="stretch")
            st.caption(info)

# ------------------------------------------------------------------ footer
st.markdown(
    '<div class="rl-note" style="margin-top:24px">'
    "<p>Method: on the start date the money is split equally. Units are left alone until each rebalance date "
    "(every 6, 12, 18 or 24 months from the start, or the next trading day), when every holding is reset to an "
    "equal share. Cost is charged on both the sale and the purchase. Sharpe uses a 6% risk-free rate. Dates missing "
    "in one series take that series' last known value.</p>"
    "<p>Index portfolio data is the price index, so dividends are not included. The benchmark is used exactly as "
    "published, with bad one-day values removed: Nifty 500 TRI on 3 Sep 2015, 3 Dec 2015, 21 Aug 2019 and 1 Feb "
    "2020, and Nifty G-Sec Composite on 12 Sep 2022. Each was far from the days on both sides, which agreed with each "
    "other. Gold is the international price in US dollars per ounce (TradingView TVC:GOLD), not converted to rupees, "
    "so it leaves out the rupee's fall against the dollar (roughly 3% a year since 2005) and understates what an "
    "Indian buyer earned. Fund data (mutual fund view) is regular-plan growth NAV, so expenses are already deducted "
    "and dividends are included. The gold ETF in that view is its split-adjusted NAV in rupees, after expenses. Not "
    "included: exit loads, capital gains tax, and the one or two days a real switch takes. This is a historical "
    "backtest, not financial advice.</p></div>",
    unsafe_allow_html=True,
)
