# API (FastAPI)

Backend for the web UI. Reuses the `equity_research` core.

## Run
    uv sync --extra web
    uv run uvicorn api.main:app --port 8000

Requires Ollama running with the configured model and `nomic-embed-text` pulled.

## Endpoints
- `GET /api/health`
- `GET /api/analyze?ticker=AAPL`  (SSE: `agent` events, then `verdict`)
- `POST /api/ingest` `{ "ticker": "AAPL" }`
- `GET /api/backtest`  (SSE: `progress` events, then `report`)
- `GET /api/backtest/config`

CORS allows `http://localhost:5173` (the Vite dev origin) by default.
