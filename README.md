# HEAN Trading Engine Backend

Event-driven crypto trading system for Bybit Testnet. All trades execute on testnet with virtual funds.

This repository contains the Python backend extracted from the [HEAN monorepo](https://github.com/nadirzhon/HEAN).

## Quick Start

```bash
pip install -e ".[dev]"
make test           # Run full test suite (~783 tests)
make test-quick     # Exclude Bybit connection tests
make lint           # ruff check + mypy
make run            # Start trading engine
```

Use `python3` (not `python`) on macOS.

## Architecture

```
Bybit WebSocket (ticks, orderbook, funding)
         |
         v
   +-------------+
   |  Event Bus   | <-- Priority queues (Critical/Normal/Low) + fast-path dispatch
   +------+------+
          |
    +-----+--------------------------------------+
    |     |     |         |         |       |     |
    v     v     v         v         v       v     v
Strategies Risk Execution Portfolio Physics Brain Oracle
```

**Signal chain:** TICK -> Strategy -> filter cascade -> SIGNAL -> RiskGovernor -> ORDER_REQUEST -> ExecutionRouter -> Bybit HTTP -> ORDER_FILLED -> Position update

### Core Components

- **EventBus** (`src/hean/core/bus.py`) -- Async event bus with multi-priority queues and circuit breaker. All components communicate through events.
- **Strategies** (`src/hean/strategies/`) -- 11 trading strategies including ImpulseEngine (momentum with 12-layer filter cascade), FundingHarvester, BasisArbitrage, MomentumTrader, and more. Each gated by a settings flag.
- **Risk** (`src/hean/risk/`) -- RiskGovernor state machine (NORMAL -> SOFT_BRAKE -> QUARANTINE -> HARD_STOP), KillSwitch (>20% drawdown protection), PositionSizer, KellyCriterion, DepositProtector.
- **Execution** (`src/hean/execution/`) -- BybitOnly production router with idempotency, position reconciliation with exchange state.
- **Exchange** (`src/hean/exchange/bybit/`) -- HTTP REST client with instrument/leverage caching, public and private WebSocket feeds.
- **Oracle** (`src/hean/core/intelligence/`) -- Hybrid 4-source signal fusion (TCN price reversal, FinBERT sentiment, Ollama local LLM, Claude Brain analysis).
- **Physics** (`src/hean/physics/`) -- Market thermodynamics: temperature, entropy, phase detection, Szilard engine.

### Microservices (Docker)

When running via Docker, the system decomposes into independent services communicating via Redis Streams:

| Service | Description | Port |
|---------|-------------|------|
| `api` | FastAPI gateway (REST + WebSocket) | 8000 |
| `symbiont-testnet` | Core trading logic container | -- |
| `redis` | Central message broker and state store | 6379 |
| `collector` | Bybit WebSocket market data ingestion | -- |
| `physics` | Market thermodynamics calculations | -- |
| `brain` | AI-based decision making via Claude | -- |
| `risk-svc` | Dedicated risk management | -- |
| `oracle` | Hybrid price+narrative AI signals | -- |

Each microservice lives in `services/<name>/` with its own `main.py`.

## Docker Usage

```bash
docker-compose up -d --build                  # Build and start (API + Redis)
docker-compose --profile training up -d mlflow # Start MLflow tracking server
docker-compose logs -f                        # View logs
```

## Configuration

Settings are managed via Pydantic `BaseSettings` in `src/hean/config.py`, loaded from `.env`:

```bash
BYBIT_API_KEY=...          # Exchange credentials
BYBIT_API_SECRET=...
BYBIT_TESTNET=true         # Always testnet
INITIAL_CAPITAL=300        # Starting USDT
ANTHROPIC_API_KEY=...      # Optional: enables Claude Brain
BRAIN_ENABLED=true
```

See `.env.example` for the full list of configuration options.

## Test Suite

The test suite contains ~783 tests using pytest with `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed).

```bash
pytest tests/test_api.py -v                    # Single file
pytest tests/test_api.py::test_health -v       # Single function
pytest -k "impulse" -v                         # Pattern match
```

Bybit connection tests (`test_bybit_http.py`, `test_bybit_websocket.py`) are excluded from CI as they require live testnet credentials.

## License

See [LICENSE](LICENSE) for details.
