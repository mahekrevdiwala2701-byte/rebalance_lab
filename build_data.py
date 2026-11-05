"""
Build data/builtin.json (the built-in data Rebalance Lab ships with) from the raw files in data/raw/.

Usage (from the project root):
    python build_data.py

What it does for every series:
  1. Reads the date and close/NAV columns (each source has its own layout, handled below).
  2. Sorts by date and drops duplicate dates.
  3. Removes isolated one-day bad values ("spikes"): a value far from BOTH neighbours while the
     neighbours agree with each other. 10% threshold for equities; for low-volatility series
     (median daily move < 0.3%, e.g. bonds) the threshold is max(2%, 10 x median daily move).
  4. Encodes dates as day numbers (days since 1970-01-01), delta-encoded, and values to 6
     significant digits, to keep the page small.

The same spike rule runs in uploads.py on files uploaded with "Add asset".
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "builtin.json")
EPOCH = pd.Timestamp("1970-01-01")

# Bad prints confirmed by hand that the automatic rule does not catch on its own.
MANUAL_DROPS = {"nifty500_tri": ["2015-12-03"]}


def raw(name):
    return os.path.join(RAW, name)


def clean(s):
    """Sort, de-duplicate and remove isolated one-day spikes."""
    s = s.dropna()
    s = s[~s.index.duplicated()].sort_index()
    v, idx = list(s.values), list(s.index)
    med = float(np.median(np.abs(np.diff(v) / np.array(v[:-1]))))
    thr = max(0.02, 10 * med) if med < 0.003 else 0.1
    for i in range(len(v) - 2, 0, -1):
        a, b, c = v[i - 1], v[i], v[i + 1]
        if abs(c / a - 1) < thr / 2 and abs(b / a - 1) > thr and abs(b / c - 1) > thr:
            del v[i]
            del idx[i]
    return pd.Series(v, index=pd.DatetimeIndex(idx))


def encode(s, name, port, **extra):
    d = ((s.index - EPOCH).days).tolist()
    item = {"name": name, "cat": extra.pop("cat", ""), "port": port}
    item.update(extra)
    item["d"] = [d[0]] + [d[i] - d[i - 1] for i in range(1, len(d))]
    item["v"] = [float(f"{x:.6g}") for x in s.values]
    return item


# ---------- readers, one per source layout ----------

def read_tradingview(path):
    """TradingView export: 'time' (dd-mm-yyyy or yyyy-mm-dd) and 'close'."""
    df = pd.read_csv(path)
    t = df["time"].astype(str)
    iso = t.str.match(r"^\d{4}-")
    d = pd.to_datetime(t.where(iso), format="%Y-%m-%d", errors="coerce").fillna(
        pd.to_datetime(t.where(~iso), format="%d-%m-%Y", errors="coerce"))
    return pd.Series(pd.to_numeric(df["close"], errors="coerce").values, index=pd.DatetimeIndex(d))


def read_investing(path):
    """Investing.com-style export: 'Date' (dd-mm-yyyy) and 'Price' with thousands commas."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    vals = pd.to_numeric(df["Price"].astype(str).str.replace(",", ""), errors="coerce")
    return pd.Series(vals.values, index=pd.to_datetime(df["Date"], format="%d-%m-%Y"))


def read_two_col_xlsx(path):
    """Excel sheet with a date column and a value column (header row first)."""
    df = pd.read_excel(path)
    df.columns = ["d", "v"]
    idx = pd.DatetimeIndex(pd.to_datetime(df["d"], errors="coerce")).normalize()
    return pd.Series(pd.to_numeric(df["v"], errors="coerce").values, index=idx)


def read_mf_navs(path):
    """Three Date/Close pairs side by side (fund name in the row above the header)."""
    raw_df = pd.read_csv(path, header=1)
    out = {}
    for name, (dc, cc) in {"HDFC Large Cap": (6, 7), "HDFC Mid Cap": (3, 4), "DSP Small Cap": (0, 1)}.items():
        d = pd.to_datetime(raw_df.iloc[:, dc], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        s = pd.Series(pd.to_numeric(raw_df.iloc[:, cc], errors="coerce").values, index=pd.DatetimeIndex(d).normalize())
        out[name] = s
    return out


def main():
    nifty50 = clean(read_tradingview(raw("nifty50_price.csv")))
    mid150 = clean(read_tradingview(raw("nifty_midcap150_price.csv")))
    small100 = clean(read_tradingview(raw("nifty_smallcap100_price.csv")))
    gsec = clean(read_tradingview(raw("nifty_gsec_composite.csv")))
    gold_usd = read_tradingview(raw("gold_usd_tvc.csv"))
    gold_usd = gold_usd[~gold_usd.index.duplicated()].sort_index()["2000-01-01":]
    gold_usd = clean(gold_usd)

    tri = read_investing(raw("nifty500_tri.csv"))
    tri = tri.drop(pd.to_datetime(MANUAL_DROPS["nifty500_tri"]), errors="ignore")
    tri = clean(tri)

    mf = read_mf_navs(raw("mutual_fund_navs.csv"))
    # Fund NAVs are kept as provided (no spike filter): they were checked by hand and are clean.
    mf = {k: v.dropna().pipe(lambda s: s[~s.index.duplicated()].sort_index()) for k, v in mf.items()}
    gold_etf = read_two_col_xlsx(raw("gold_etf_inr_nav.xlsx")).dropna().sort_index()
    bench2 = read_two_col_xlsx(raw("benchmark2.xlsx")).dropna().sort_index()

    items = [
        encode(nifty50, "Nifty 50", "index"),
        encode(mid150, "Nifty Midcap 150", "index"),
        encode(small100, "Nifty Smallcap 100", "index"),
        encode(gsec, "Nifty G-Sec Composite", "index"),
        encode(gold_usd, "Gold (USD)", "index"),
        encode(mf["HDFC Large Cap"], "HDFC Large Cap", "mf", cat="Large"),
        encode(mf["HDFC Mid Cap"], "HDFC Mid Cap", "mf", cat="Mid"),
        encode(mf["DSP Small Cap"], "DSP Small Cap", "mf", cat="Small"),
        encode(gold_etf, "Gold ETF (₹)", "mf"),
        encode(gsec, "Nifty G-Sec Composite", "mf"),
        encode(tri, "Nifty 500 TRI", "bench"),
        encode(bench2, "Benchmark 2", "bench", on=False, also=True),
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, separators=(",", ":"))
    for it in items:
        print(f"{it['name']:<24} {it['port']:<6} {len(it['v']):>6} days")
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
