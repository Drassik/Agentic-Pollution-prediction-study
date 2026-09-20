# PM2.5 Reporting Agent — Turin Air Quality

An LLM-powered agent (tool-use + retrieval) built on top of a custom PM2.5 prediction
pipeline for Turin, Italy. The agent answers questions about historical air quality,
generates short-term forecasts, and (in progress) grounds its answers in regulatory
context such as WHO thresholds.

A core data/prediction tools is built, tested against real
data, and wired into a working tool-calling agent. Retrieval, robustness testing, and
a formal evaluation harness are in progress (see Roadmap).

## Why this project

Built to demonstrate practical experience with agentic LLM systems — tool-use over a
real ML pipeline, reasoning under uncertainty, and (eventually) rigorous multi-turn
evaluation — rather than a toy demo. Originally scoped to encompass a more evolved tool to query a public API for current weather and establish predictions but narrowed down for faster implementation. 

## Architecture

```
notebooks/          Original exploration & modeling notebooks (source of truth for
                     feature engineering, kept but no longer executed at runtime)
  Analysis.ipynb       Raw data cleaning, outlier removal, feature engineering
  prediction.ipynb     Model training (PolynomialFeatures + LassoCV), evaluation

global_dataset.csv  Exported dataset: date + 21 features (10 weather, 11 pollution/
                     calendar) + PM2.5 target — generated once from prediction.ipynb
pm25_model.pkl       Trained model, saved via joblib — loaded by tools at runtime

tools.py            4 tools wrapping the pipeline as LangChain @tool functions
agent.py            Agent construction (create_agent + ChatOllama)
```

Data flows one way: notebooks → `global_dataset.csv` / `pm25_model.pkl` → `tools.py` →
`agent.py`. The agent never touches raw CSVs or re-trains anything at runtime.

## Tools implemented 

| Tool | Purpose | Notes |
|---|---|---|
| `get_pm25_data(start_date, end_date)` | Historical PM2.5 values over a date range | Returns explicit `no_data` status if the range is out of bounds, instead of failing silently |
| `predict_pm25(horizon_days)` | Recursive multi-day forecast | Weather features are held constant at their last known value (no future weather forecast available) — reliability degrades noticeably beyond ~2-3 days, and this is surfaced in the tool's own return value |
| `compute_stat(metric, period)` | `mean`, `seasonal_mean`, `r2`, `mae`, `rmse` | Model metrics computed on the winter 2025–2026 holdout set |
| `plot_trend(period)` | PM2.5 trend chart over a period | Returns a base64-encoded PNG |

Each tool returns a structured dict with an explicit `status` field (`ok` / `error` /
`no_data`) rather than raising bare exceptions — this is what lets the agent admit
"I don't have that data" instead of guessing, which matters for the Day 4 evaluation.

## Model

- Pipeline: `PolynomialFeatures(degree=2)` + `LassoCV`
- Features: 10 weather variables (temperature, precipitation, wind, humidity, solar
  radiation) + 11 calendar/lag/rolling features (day of year, lags at 1/7/30/365 days,
  7/30-day rolling means, seasonal sin/cos encoding)
- Test set: winter 2025–2026 (chronological holdout, no data leakage)
- Current performance: **R² = 0.541**, evaluated on the holdout set. The model
  captures seasonality well (winter mean 30.4 µg/m³ vs. summer 12.1 µg/m³) but is
  weaker on sharp, short-lived spikes.

## Setup

```bash
uv sync                          # installs all dependencies from uv.lock
ollama pull qwen3:8b              # local model used by the agent
ollama serve                      # must be running before agent.py
uv run python tools.py            # sanity check: runs all 4 tools against real data
uv run python agent.py            # runs the agent end-to-end
```

No API key required — the agent runs entirely on a local Ollama model
(`qwen3:8b`), chosen for a reasonable balance of tool-calling reliability and
resource usage on consumer hardware (tested on 8–12 GB VRAM).

## Known limitations

- **No real future weather data.** `predict_pm25` assumes the last known weather
  persists for the whole forecast horizon. A production system would pull forecasts
  from a weather API instead — identified as the clearest next improvement.
- **Model R² of 0.541** leaves room for improvement, particularly on pollution spikes.
- **Observed hallucination risk:** in early testing, the agent added an unsupported
  causal explanation ("likely due to weather conditions") for a data point, even
  though no tool provides causal attribution. The system prompt was tightened to
  explicitly forbid inventing explanations not returned by a tool. 

## Roadmap
- [x] — Core tools (`get_pm25_data`, `predict_pm25`, `compute_stat`,
      `plot_trend`) extracted from notebooks, tested against real data, wrapped as
      LangChain tools, wired into a working `create_agent` loop on Ollama.
- [ ] — Building a tool to query a public API to get the weather for a current date outside of the original dataset and improve the predictions.
- [ ] — Conversation memory, structured JSON logging of every interaction,
      and testing against edge cases (out-of-scope questions, missing data,
      ambiguous follow-ups).
- [ ] — Lightweight manual evaluation: 5–6 hand-picked questions across factual,
      out-of-scope, and missing-data categories, scored by hand on two criteria
      (factually grounded in a tool result? hallucinated anything?). Not a full
      automated LLM-judge harness — scoped down to fit available time, but enough
      to surface and document real failure modes rather than assume none exist.

## Stack

Python, LangChain / LangGraph (`create_agent`), Ollama (`qwen3:8b` / `qwen3:14b`),
FAISS, pandas, scikit-learn, `uv` for dependency management.
