# Rebalance Lab (Streamlit)

An equal-weight portfolio rebalancing backtest for Indian markets, rebuilt in Python with Streamlit.
It compares rebalancing every 6, 12, 18 or 24 months with never rebalancing, for two portfolios:

- **Index portfolio:** Nifty 50, Nifty Midcap 150, Nifty Smallcap 100, Nifty G-Sec Composite, Gold (USD)
- **Mutual fund portfolio:** HDFC Large Cap, HDFC Mid Cap, DSP Small Cap, Gold ETF (₹), Nifty G-Sec Composite

Both are measured against **Nifty 500 TRI** (main benchmark) and **Benchmark 2** (compared alongside, from 2013).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Files

```
rebalance-lab-streamlit/
├── app.py            the Streamlit page: settings, summary, charts, tables, holdings, uploads
├── engine.py         backtest engine: alignment, rebalancing simulation, stats, rolling windows,
│                     best-combination search, XIRR
├── uploads.py        reads uploaded Excel/CSV files (finds Date + Close/NAV columns, removes bad spikes)
├── store.py          loads the built-in data; saves your setup to data/saved_setup.json
├── build_data.py     rebuilds data/builtin.json from the files in data/raw/
├── data/
│   ├── builtin.json  built-in price/NAV history (dates delta-encoded)
│   └── raw/          the source files the data was built from
├── requirements.txt
└── .streamlit/config.toml
```

## What's in the app

- **Settings (sidebar):** date range, "Full history", match the benchmark's dates, cost per trade,
  rolling window (3/5/10 years), and which strategy the charts show.
- **Summary:** CAGR and max fall for each strategy, how often each one won across rolling windows,
  and the gap to the benchmark.
- **Against the benchmark:** CAGR, volatility, Sharpe, max fall, share of rolling periods beaten,
  and a rolling excess-return chart. Benchmark 2 is compared on its own dates.
- **Best combination:** tests every equal-weight mix of the ticked assets (up to 12 assets = 4,095 mixes),
  ranked by CAGR, Sharpe or Calmar.
- **Charts:** growth of ₹1,00,000 (log scale), drawdown from peak, and rolling CAGR difference
  between any two strategies.
- **Lump-sum returns (XIRR)** and a **full comparison** table.
- **Holdings and benchmark:** tick/untick holdings, pick the main benchmark, "Compare too", and
  **Add asset** from your own Excel/CSV files.

## Differences from the web version

- The web page synced its setup across devices through claude.ai. Here the setup (ticked holdings,
  benchmark choice, added files) is saved to `data/saved_setup.json` on the machine running Streamlit,
  so everyone using that server shares it. "Restore built-in setup" in the sidebar deletes it.
- The engine is a line-by-line port of the JavaScript engine and gives identical numbers.

## Rebuilding the data

Only needed when files in `data/raw/` change:

```bash
python build_data.py
```

This is a historical backtest, not financial advice.
