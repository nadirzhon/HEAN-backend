# HEAN Profit Optimizer — Persistent Memory

## Key File Locations (Monorepo)
- Config: `packages/hean-core/src/hean/config.py` (HEANSettings, ~1550 lines)
- PositionSizer: `packages/hean-risk/src/hean/risk/position_sizer.py`
- ExecutionRouter: `packages/hean-execution/src/hean/execution/router_bybit_only.py`
- ImpulseEngine: `packages/hean-strategies/src/hean/strategies/impulse_engine.py`
- FundingHarvester: `packages/hean-strategies/src/hean/strategies/funding_harvester.py`
- BybitHTTP: `packages/hean-exchange/src/hean/exchange/bybit/http.py`

## Critical Configuration Values (verified 2026-02-19)
- `min_notional_usd = 100.0` — hardcoded in PositionSizer (lines 225 and 302), NOT driven by exchange API
- `maker_ttl_ms = 150` — config.py line 845 (comment says "was 8000ms")
- `backtest_taker_fee = 0.0003` (0.03%) — config.py line 832; real Bybit taker = 0.055%
- `backtest_maker_fee = 0.00005` (0.005%) — config.py line 826; real Bybit maker = -0.01% rebate
- `allow_taker_fallback = True` — config.py line 849
- Taker fallback edge check uses `backtest_taker_fee` (wrong value, line 800 of router)
- `min_edge_for_fallback_bps = 2.0` — hardcoded in router (line 814)
- Adaptive TTL: router has `_calculate_optimal_ttl()` that adjusts base TTL, clamped 50–500ms
- BybitHTTP fetches real `minNotional` from exchange API (line 760, fallback default 5.0)

## ImpulseEngine TP/SL
- Standard: SL=0.3%, TP=1.5%, TP1=0.7% (R:R = 1:5)
- Breakout mode: SL=0.2%, TP=2.5%, TP1=1.2% (R:R = 1:12.5)
- Stop distance used for position sizing: 0.003 (0.3%) standard

## FundingHarvester TP/SL
- SL=1.5%, TP=0.8%, TP1=0.3% (asymmetric: risk > reward — problematic)
- Min funding threshold: 0.01% (0.0001 decimal)

## Known Bugs (Session 2026-02-19)
See execution-bugs.md for full analysis.

## Architecture Notes
- PositionSizer has TWO methods: `calculate_size` and `calculate_size_v2` — both hardcode min_notional_usd=100
- Router adaptive TTL overrides `maker_ttl_ms` dynamically but uses it as the base
- Bybit real minNotional is fetched from API in http.py (~$5 for most perpetuals)
- backtest_taker_fee is used in ONE place: taker fallback edge gate (router line 800)
