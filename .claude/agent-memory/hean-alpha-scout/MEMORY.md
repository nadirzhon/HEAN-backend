# HEAN Alpha Scout — Persistent Memory

## Scaling Alpha Research (Session: 2026-02-19)

### Key Data Points Verified

**Funding Rate Arbitrage (Delta-Neutral):**
- 2025 average: 0.015%/8h = 0.045%/day = ~16.4%/year on notional
- With $300 capital at 2x leverage = $600 notional → ~$0.27/day raw; after fees ~$0.10/day net
- Research source (ScienceDirect 2025): up to 19.26% annualized when rates elevated
- Key risk: funding rate flipping negative erodes returns

**Hyperliquid HLP Vault:**
- Historical APY: ~17-23% base, spikes to 165%+ during liquidation events
- Sharpe ratio lifetime: 2.89, recently 5.2
- $300 at 20% APY = $0.16/day (baseline) — too low standalone
- No minimum deposit; no IL risk; USDC-denominated

**GMX GLP:**
- APY range: 5-55% highly variable (trader counterparty risk)
- $300 at 20% APY midpoint = $0.16/day

**Uniswap v3 Concentrated Liquidity:**
- Realistic APY with active management: 8-50%+ depending on range and volume
- Requires active rebalancing or automation; IL risk amplified by narrow ranges
- Not suitable for $300 standalone — gas costs eat returns on Ethereum mainnet
- Better on L2 (Arbitrum, Base) where gas is minimal

**Bybit Options / Covered Calls:**
- BTC IV typically 50-80% annualized; selling 1-week OTM calls yields ~1-2% premium/week
- $300 in BTC spot + sell weekly call = ~$3-6/week potential; execution min size is 0.01 BTC (~$950 at current prices)
- Barrier: minimum contract size makes this impractical at $300 capital
- Bybit discontinued USDC options Feb 26 2025; only USDT-settled available

**Statistical Arbitrage (ETH/BTC pairs):**
- Sharpe ratio: 1.0-1.53 from academic studies (2019-2025 data)
- 16% annualized return documented
- Can be implemented within HEAN CorrelationArbitrage strategy framework
- Key: cointegration breaks during crypto-wide deleveraging events

**MEV:**
- Sandwich attack average net: ~$3/attack; declining sharply (from $10M/month to $2.5M/month late 2025)
- Requires significant Rust/Solidity engineering + gas capital
- NOT viable for $300 capital; entry barrier ~$10,000+ for competitive bots

**EigenLayer Restaking:**
- ~4.24% EIGEN + ~2.5% ETH staking = ~6-8% total APY
- Gas fees for $300 ETH position would take months to recoup
- Minimum practical: $2,000+

**Bybit Copy Trading (Master Trader path):**
- Requirements: Silver tier, 30-day ROI >= 10%, Sharpe >= 0.5, MaxDD <= 20%
- Profit share: up to 30% of followers' net profit (Classic up to 15%)
- Up to 300 followers; max AUM limited by Bybit per tier
- THIS IS THE HIGHEST LEVERAGE OPPORTUNITY: if HEAN can demonstrate track record,
  managing 100 followers at $500 each = $50,000 AUM, 30% of 10% return = $1,500/month

**Basis Trading (Cash-and-Carry):**
- BTC futures basis in bull markets: 15-30% annualized
- Institutional implementations reaching 20.71% (Securitize + BlackRock BUIDL)
- HEAN already has BasisArbitrage strategy — needs enhancement for pure carry
- $300 capital limitation: minimum order sizes on perpetuals (~$10 min)

### Integration Notes for HEAN

- `FundingHarvester` already implements funding rate positioning — extend for delta-neutral spot hedge
- `BasisArbitrage` already tracks mark/index spread — extend for pure carry (hold spot + short perp)
- `CorrelationArbitrage` strategy exists — extend for cointegration-based pairs trading
- New EventType needed: `FUNDING_CARRY_SIGNAL` for carry trade coordination
- Copy Trading path requires LIVE (not testnet) track record — migration prerequisite

### Ideas Rejected

- MEV bots: too capital-intensive, requires deep Solidity/Rust expertise, declining profits
- EigenLayer: gas economics broken at $300 scale
- Uniswap v3 on Ethereum mainnet: gas costs prohibitive at small scale

### Priority Ranking (for $300 → $600+/day scaling)

1. **Copy Trading Master path** — nonlinear scaling (AUM multiplication)
2. **Enhanced Funding Carry** — delta-neutral, ~15-20% annual, enhance existing strategy
3. **Options Wheel Strategy** — when capital reaches $1,000+ (min BTC option contract)
4. **Cross-Exchange Stat Arb** — Sharpe 1.5, can add to CorrelationArbitrage
5. **HLP/GMX DeFi yield** — 15-20% APY, good for idle capital
