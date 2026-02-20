# Execution Bugs — Verified Analysis (2026-02-19)

## Bug 1: min_notional_usd = 100.0 in PositionSizer

**File:** `packages/hean-risk/src/hean/risk/position_sizer.py`
**Lines:** 225 (calculate_size) and 302 (calculate_size_v2)

**Exact code:**
```
min_notional_usd = 100.0  # $100 minimum order value (Bybit mainnet requirement)
absolute_min = min_notional_usd / current_price
```

**Reality:** Bybit TESTNET perpetuals have minNotional of ~$5 (fetched dynamically in http.py line 760).
The comment says "Bybit mainnet requirement" but: (a) this is testnet, (b) Bybit perpetuals actual
minimum is closer to $5-10, not $100.

**Impact math at $300 capital:**
- max_trade_risk_pct = 1% → risk_amount = $3
- With SL=0.3% (ImpulseEngine standard): position_size = $3 / 0.3% = $1000 notional
- BUT if stop is wider (e.g., 2% default): position_size = $3 / 2% = $150 notional
- If calculated size < $100 notional: system forces UP to $100 = 3.3x+ overrisk
- Worst case: a $300 equity account gets forced to risk 33% of equity ($100) on a single trade
- This violates the 1% risk rule catastrophically on low-conviction, wide-stop signals

**Recommended fix:** Change to $5.0 (matches real Bybit minimum, or read from exchange API).
Alternatively, if exchange API minNotional is already fetched in http.py, pass it through to
PositionSizer so the floor is exchange-accurate and not arbitrarily inflated.

## Bug 2: maker_ttl_ms = 150ms — Too Aggressive for Maker Strategy

**File:** `packages/hean-core/src/hean/config.py`
**Line:** 845

**Exact code:**
```python
maker_ttl_ms: int = Field(
    default=150,  # Phase 2 Optimization: 150ms (was 8000ms) for HFT execution
    ...
)
```

**The router's adaptive TTL system** (`_calculate_optimal_ttl` in router, lines 981-1048):
- Takes `settings.maker_ttl_ms` as the BASE
- Applies multipliers: spread (1.0–2.0x), volatility (0.5–1.2x), fill rate (0.9–1.5x)
- Clamps result to 50–500ms
- So with base=150ms: actual range in normal conditions is ~67ms (high vol) to ~270ms

**The problem:** At 150ms base, the adaptive system rarely produces TTLs long enough for
Bybit limit orders to fill. Bybit order roundtrip alone is ~50-100ms (network + matching).
A 150ms TTL gives the order roughly 50-100ms of "live" time after placement confirmation.

**Bybit maker fill time empirics:**
- BTC spread ~1-3 bps: orders at mid typically fill in 200-800ms in normal conditions
- At 150ms TTL, majority expire before fill
- Expired orders → taker fallback (+6 bps penalty entry vs maker rebate of -1 bp = 7 bps swing)

**Recommended value:** 800ms (conservative) to 1500ms (balanced).
- At 800ms: enough time for most limit orders at mid to fill without stale signal risk
- The adaptive system will still shorten to 400-600ms in high-vol and lengthen to 1000ms+ in wide-spread markets
- Signal decay risk: ImpulseEngine signals are momentum-based; 0.8-1.5s holding is fine since
  SL/TP are set at signal time, not order-time

## Bug 3: backtest_taker_fee = 0.0003 (0.03%) vs Real 0.055%

**File:** `packages/hean-core/src/hean/config.py`
**Line:** 832

**Exact code:**
```python
backtest_taker_fee: float = Field(
    default=0.0003,
    description="Taker fee for backtesting (default 0.03% = 3 bps, reduced from 0.06%)"
)
```

The comment "reduced from 0.06%" means someone intentionally lowered it — wrong direction.

**Real Bybit Testnet fees (USDT perpetuals, standard tier):**
- Taker: 0.055% (5.5 bps)
- Maker: -0.01% (negative = rebate of 1 bp)

**Where this bug matters:**
1. **Taker fallback edge gate** (router line 800):
   ```
   taker_fee_bps = settings.backtest_taker_fee * 10000  # = 3.0 bps (WRONG)
   ```
   Real cost should be 5.5 bps. The gate adds 5 bps slippage = 8 bps total.
   With wrong value: uses 3 + 5 = 8 bps as cost gate.
   With correct value: uses 5.5 + 5 = 10.5 bps as cost gate.
   Result: system approves taker fallbacks with net_edge between 8-10.5 bps that are
   actually unprofitable. Every such trade burns 2.5 bps.

2. **Backtest strategy selection** — any strategy backtested with 3 bps taker fee shows
   50% higher apparent profitability than reality on taker-heavy signals. Strategies that
   only work with 3 bps fees but not 5.5 bps are incorrectly promoted.

**Recommended fix:** Change to 0.00055 (0.055%). Also fix `min_edge_for_fallback_bps`
from 2.0 to at least 5.0 to ensure taker fallbacks only execute when genuinely profitable
after real costs.

## Bug 4 (Secondary): FundingHarvester Inverted R:R

**File:** `packages/hean-strategies/src/hean/strategies/funding_harvester.py`
**Lines:** 371-373

**Exact code:**
```python
stop_loss=entry_price * (0.985 if side == "buy" else 1.015),  # 1.5% stop
take_profit=entry_price * (1.008 if side == "buy" else 0.992),  # 0.8% target
take_profit_1=entry_price * (1.003 if side == "buy" else 0.997),  # 0.3% first TP
```

**Problem:** R:R = 0.8% reward / 1.5% risk = 0.53:1. Need break-even win rate of 65%+
to be profitable. This only makes sense if funding income is the primary profit driver
(position expected to collect funding payments before hitting TP), but at 0.01% threshold
and $300 capital, funding per payment is ~$0.03. The 1.5% SL means one loss wipes out
50+ funding payments.

**Recommended fix:** Either tighten SL to 0.5% (matching the 0.8% TP for ~1.6:1 R:R)
or widen TP to 2.0% for genuine momentum capture.
