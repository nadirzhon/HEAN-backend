"""Configuration management using Pydantic v2."""

import json
import os
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# ---------------------------------------------------------------------------
# Hot-reload whitelists
# ---------------------------------------------------------------------------

# Keys in this set may be updated at runtime via the ConfigWatcher / Redis
# pub/sub channel without restarting the process.  Only non-sensitive,
# operationally safe parameters are included.
SAFE_RELOAD_KEYS: frozenset[str] = frozenset(
    {
        # Strategy enables
        "impulse_engine_enabled",
        "funding_harvester_enabled",
        "basis_arbitrage_enabled",
        "hf_scalping_enabled",
        "enhanced_grid_enabled",
        "momentum_trader_enabled",
        "correlation_arb_enabled",
        "sentiment_strategy_enabled",
        "inventory_neutral_mm_enabled",
        "rebate_farmer_enabled",
        "liquidity_sweep_enabled",
        # Risk thresholds
        "max_daily_drawdown_pct",
        "max_trade_risk_pct",
        "max_open_positions",
        "killswitch_drawdown_pct",
        "consecutive_losses_limit",
        # Intervals
        "brain_analysis_interval",
        "ollama_sentiment_interval",
        # Feature flags
        "symbiont_x_enabled",
        "risk_sentinel_enabled",
        "intelligence_gate_enabled",
        "intelligence_gate_reject_on_contradiction",
    }
)

# Keys that must NEVER be updated via hot-reload regardless of any whitelist.
# This set acts as a hard security boundary.
BLOCKED_KEYS: frozenset[str] = frozenset(
    {
        "bybit_api_key",
        "bybit_api_secret",
        "anthropic_api_key",
        "gemini_api_key",
        "openai_api_key",
        "openrouter_api_key",
        "api_auth_key",
        "jwt_secret",
        "redis_url",
        "initial_capital",
    }
)


def parse_list_env(value: Any) -> Any:
    """Parse list values from env (JSON array, comma-separated, or single item)."""
    if value is None:
        return value
    if isinstance(value, str):
        if value.strip() == "":
            return value
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items if items else value
    return value


def _find_env_file() -> str:
    """Find .env file: check project root first, then CWD."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        return env_path
    return ".env"


class HEANSettings(BaseSettings):
    """Main configuration for HEAN system."""

    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        enable_decoding=False,
        env_ignore_empty=True,
    )

    # Absolute+ / Meta-Learning controls (disabled by default for safety in containerized runs)
    absolute_plus_enabled: bool = Field(
        default=False,
        description="Enable Absolute+ meta-learning / patching systems",
    )
    meta_learning_auto_patch: bool = Field(
        default=False,
        description="Allow meta-learning engine to patch source code (requires writable volume)",
    )
    meta_learning_rate: int = Field(
        default=0,
        description="Simulation rate hint for meta-learning engine",
    )
    meta_learning_max_workers: int = Field(
        default=0,
        description="Max concurrent simulations for meta-learning",
    )

    # Environment
    environment: Literal["development", "production"] = Field(
        default="development",
        description="Application environment (development or production)"
    )

    # Trading Mode (BYBIT TESTNET ONLY - NO PAPER TRADING)
    live_confirm: str = Field(default="YES", description="Must be 'YES' to enable live trading")
    trading_mode: Literal["live"] = Field(default="live", description="Trading mode - ALWAYS LIVE (testnet)")

    # Capital Management
    initial_capital: float = Field(default=300.0, gt=0, description="Initial capital in USDT")
    reinvest_rate: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Profit reinvestment rate (legacy, use smart_reinvest_base_rate)",
    )
    cash_reserve_rate: float = Field(
        default=0.2, ge=0, le=1, description="Cash reserve rate (not allocated)"
    )
    capital_allocation_mode: str = Field(
        default="adaptive",
        description="Capital allocation mode: 'equal' (equal split), 'adaptive' (performance-based)",
    )
    force_equal_allocation: bool = Field(
        default=False,
        description="Force equal capital allocation regardless of performance (1/N for N strategies)",
    )

    # Profit Targets
    profit_target_daily_usd: float = Field(
        default=100.0, gt=0, description="Daily profit target in USD (minimum $100/day)"
    )

    # Smart Reinvestment
    smart_reinvest_base_rate: float = Field(
        default=0.85,
        ge=0,
        le=1,
        description="Base reinvestment rate for smart reinvestor (default 85%)",
    )
    smart_reinvest_min_reserve_pct: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description="Minimum reserve percentage to always maintain (default 10%)",
    )
    smart_reinvest_drawdown_threshold: float = Field(
        default=15.0, gt=0, description="Drawdown threshold for reducing reinvestment (default 15%)"
    )

    # Risk Management
    max_daily_drawdown_pct: float = Field(
        default=15.0,
        gt=0,
        le=100,
        description="Maximum daily drawdown percentage (Testnet: 15% allows normal trading activity)",
    )
    max_trade_risk_pct: float = Field(
        default=1.0,
        gt=0,
        le=100,
        description="Maximum risk per trade percentage (HEAN v2 Iron Rule: -1% per trade)",
    )
    max_open_positions: int = Field(
        default=10, gt=0, description="Maximum number of open positions (anti-runaway guard)"
    )
    max_open_orders: int = Field(
        default=20,
        gt=0,
        description="Maximum number of open orders (prevents runaway order creation)",
    )
    max_hold_seconds: int = Field(
        default=900,
        gt=0,
        description="Maximum time to hold a position before force-closing (anti-stuck TTL, default 15m)",
    )
    position_monitor_check_interval: int = Field(
        default=30,
        gt=0,
        description="How often to check for stale positions (seconds, default 30s)",
    )
    position_monitor_enabled: bool = Field(
        default=True,
        description="Enable automatic force-closing of stale positions",
    )

    # Liquidity Sweep Detector Strategy
    liquidity_sweep_enabled: bool = Field(
        default=True,
        description="Enable Liquidity Sweep Detector strategy for catching institutional sweeps",
    )
    liquidity_sweep_threshold_pct: float = Field(
        default=0.003,
        description="Sweep threshold as decimal (0.003 = 0.3%)",
    )
    liquidity_sweep_cooldown_minutes: int = Field(
        default=15,
        ge=5,
        description="Cooldown between trades per symbol (minutes)",
    )

    max_concurrent_risk_pct: float = Field(
        default=20.0,
        gt=0,
        le=100,
        description="Maximum concurrent risk percentage across all positions",
    )
    max_leverage: float = Field(
        default=3.0, gt=0, le=100, description="Maximum leverage (capped at 3x for safety)"
    )
    max_exposure_multiplier: float = Field(
        default=3.0,
        gt=0,
        le=20,
        description="Maximum total notional exposure as multiple of equity",
    )

    # Multi-Level Protection
    max_strategy_loss_pct: float = Field(
        default=7.0,
        gt=0,
        le=100,
        description="Maximum loss per strategy as % of initial capital (default 7%)",
    )
    max_hourly_loss_pct: float = Field(
        default=15.0,
        gt=0,
        le=100,
        description="Maximum loss per hour as % of initial capital (default 15%)",
    )
    consecutive_losses_limit: int = Field(
        default=3, ge=1, description="Number of consecutive losses before pause (HEAN v2 Iron Rule: 3)"
    )
    consecutive_losses_cooldown_hours: int = Field(
        default=2, gt=0, description="Hours to pause after consecutive losses (HEAN v2 Iron Rule: 2h)"
    )

    # Minimum Risk:Reward Ratio (HEAN v2 Iron Rule #5)
    min_risk_reward_ratio: float = Field(
        default=2.0,
        gt=0,
        description="Minimum Risk:Reward ratio to enter a trade (HEAN v2 Iron Rule: 1:2)",
    )

    # Deposit Protection
    deposit_protection_active: bool = Field(
        default=True,
        description="Enable deposit protection (never allow equity below initial capital)",
    )
    killswitch_drawdown_pct: float = Field(
        default=30.0,
        gt=0,
        description="Killswitch drawdown percentage from initial capital (Testnet: 30% to prevent premature stops)",
    )

    # Capital Preservation Mode
    capital_preservation_drawdown_threshold: float = Field(
        default=12.0,
        gt=0,
        description="Drawdown % to activate capital preservation mode (default 12%)",
    )
    capital_preservation_pf_threshold: float = Field(
        default=0.9,
        gt=0,
        description="PF threshold to activate capital preservation mode (default 0.9)",
    )
    capital_preservation_consecutive_losses_threshold: int = Field(
        default=3,
        ge=1,
        description="Consecutive losses to activate capital preservation mode (default 3)",
    )

    # Risk-First Architecture
    risk_sentinel_enabled: bool = Field(
        default=True,
        description="Enable RiskSentinel pre-trade risk envelope computation",
    )
    risk_sentinel_update_interval_ms: int = Field(
        default=1000,
        ge=100,
        description="Minimum interval between RiskEnvelope recomputations (ms)",
    )
    intelligence_gate_enabled: bool = Field(
        default=True,
        description="Enable IntelligenceGate signal enrichment (Brain+Oracle+Physics)",
    )
    intelligence_gate_reject_on_contradiction: bool = Field(
        default=False,
        description="If True, IntelligenceGate rejects signals contradicting Brain/Oracle consensus",
    )

    # Smart Leverage
    min_edge_for_leverage_2x: float = Field(
        default=25.0, ge=0, description="Minimum edge in bps for 2x leverage (default 25)"
    )
    min_edge_for_leverage_3x: float = Field(
        default=35.0, ge=0, description="Minimum edge in bps for 3x leverage (default 35)"
    )
    min_edge_for_leverage_4x: float = Field(
        default=50.0, ge=0, description="Minimum edge in bps for 4x leverage (default 50)"
    )
    min_pf_for_leverage: float = Field(
        default=1.2, gt=0, description="Minimum PF for leverage > 1x (default 1.2)"
    )
    max_leverage_on_drawdown_10pct: float = Field(
        default=2.0, gt=0, description="Max leverage when drawdown 10-15% (default 2.0)"
    )

    # Strategy Settings
    funding_harvester_enabled: bool = Field(
        default=True, description="Enable funding harvester strategy"
    )
    basis_arbitrage_enabled: bool = Field(
        default=True, description="Enable basis arbitrage strategy"
    )
    impulse_engine_enabled: bool = Field(default=True, description="Enable impulse engine strategy")

    # Dormant Strategies (AFO-Director Phase 5 - NOW ENABLED for maximum trading activity)
    hf_scalping_enabled: bool = Field(
        default=True, description="Enable high-frequency scalping strategy (40-60 trades/day, 0.2-0.4% TP)"
    )
    enhanced_grid_enabled: bool = Field(
        default=True, description="Enable enhanced grid trading strategy (range-bound markets only)"
    )
    momentum_trader_enabled: bool = Field(
        default=True, description="Enable momentum trader strategy (trend following)"
    )
    inventory_neutral_mm_enabled: bool = Field(
        default=True, description="Enable inventory-neutral market making strategy"
    )
    correlation_arb_enabled: bool = Field(
        default=True, description="Enable correlation arbitrage strategy"
    )
    rebate_farmer_enabled: bool = Field(
        default=True, description="Enable rebate farmer strategy (maker fee capture)"
    )
    sentiment_strategy_enabled: bool = Field(
        default=True, description="Enable sentiment-based trading strategy"
    )

    # Trading Symbols
    trading_symbols: list[str] = Field(
        default=[
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "DOTUSDT", "LTCUSDT", "AVAXUSDT", "LINKUSDT",
            "BCHUSDT", "MATICUSDT", "ATOMUSDT", "ETCUSDT", "BNBUSDT",
            "FILUSDT", "NEARUSDT", "OPUSDT", "ARBUSDT", "APTUSDT",
            "SUIUSDT", "TRXUSDT", "XLMUSDT", "UNIUSDT", "AAVEUSDT",
            "MKRUSDT", "INJUSDT", "RNDRUSDT", "SEIUSDT", "RUNEUSDT",
            "ICPUSDT", "ALGOUSDT", "EOSUSDT", "FTMUSDT", "GALAUSDT",
            "SANDUSDT", "AXSUSDT", "CHZUSDT", "CRVUSDT", "KAVAUSDT",
            "GMXUSDT", "SNXUSDT", "ZILUSDT", "DYDXUSDT", "COMPUSDT",
            "1INCHUSDT", "LDOUSDT", "NEOUSDT", "XTZUSDT", "APEUSDT",
        ],
        description="List of trading symbols to monitor and trade (e.g., ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])",
    )

    # Multi-Symbol Support (AFO-Director feature)
    multi_symbol_enabled: bool = Field(
        default=True,
        description="Enable multi-symbol scanning and trading. Scans all symbols in SYMBOLS list for maximum opportunities.",
    )
    symbols: list[str] = Field(
        default=[
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "TONUSDT",
        ],
        description="List of symbols for multi-symbol scanning (default: 10 symbols). Used when MULTI_SYMBOL_ENABLED=true.",
    )

    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, v: Any) -> Any:
        """Parse symbols from env, handling empty strings and JSON."""
        if v is None or v == "":
            return [
                "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
                "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "TONUSDT",
            ]
        return parse_list_env(v)

    # Paper mode data source
    paper_use_live_feed: bool = Field(
        default=False,
        description="Use Bybit public market data in paper mode. Required for ticks in paper mode.",
    )

    # Triangular arbitrage (Phase 2 Profit Doubling: Optimized for Bybit)
    triangular_arb_enabled: bool = Field(
        default=True,
        description="Enable triangular arbitrage scanner",
    )
    triangular_fee_buffer: float = Field(
        default=0.0006,  # Optimized: 0.06% (Bybit maker: -0.01%, taker: 0.055%)
        ge=0,
        description="Fee buffer ratio for triangular arbitrage scanning",
    )
    triangular_min_profit_bps: float = Field(
        default=3.0,  # Optimized: Lower threshold = more opportunities (0.03%)
        ge=0,
        description="Minimum profit in bps required for triangular arbitrage",
    )

    @field_validator("trading_symbols", mode="before")
    @classmethod
    def parse_trading_symbols(cls, v: Any) -> Any:
        """Parse trading_symbols from env, handling empty strings and JSON."""
        if v is None or v == "":
            return ["BTCUSDT", "ETHUSDT"]  # Default value
        return parse_list_env(v)

    @field_validator("impulse_allowed_hours", mode="before")
    @classmethod
    def parse_impulse_allowed_hours(cls, v: Any) -> Any:
        """Parse impulse_allowed_hours from env values."""
        if v == "":
            return None
        return parse_list_env(v)

    # Impulse Engine Specific
    impulse_max_attempts_per_day: int = Field(
        default=120,
        gt=0,
        description="Maximum attempts per day for impulse engine (increased for higher frequency)",
    )
    impulse_cooldown_after_losses: int = Field(
        default=3, ge=0, description="Cooldown period after losses"
    )
    impulse_cooldown_minutes: int = Field(
        default=2,
        ge=1,
        description="Cooldown minutes between trades (reduced from 5 to 2 for more trades)",
    )
    impulse_max_spread_bps: int = Field(
        default=12,
        gt=0,
        description="Maximum spread in basis points for impulse engine (no-trade zone, increased from 8 to 12 for more opportunities). "
        "In Aggressive Mode (DEBUG_MODE=True), this is effectively doubled via multiplier.",
    )
    impulse_max_volatility_spike: float = Field(
        default=0.02,
        gt=0,
        description="Maximum volatility spike threshold (no-trade zone, increased from 0.01 to 0.02 for more opportunities)",
    )
    impulse_max_time_in_trade_sec: int = Field(
        default=300,
        gt=0,
        description="Maximum time in trade in seconds (force exit, increased from 200 to 300 to give trades more time)",
    )
    impulse_maker_edge_threshold_bps: float = Field(
        default=3.0, ge=0, description="Minimum maker edge in basis points (skip if below)"
    )
    impulse_allowed_hours: list[str] | None = Field(
        default=["07:00-11:00", "12:00-16:00", "17:00-21:00"],
        description="Allowed trading hours in UTC (list of time ranges as strings, e.g., ['08:00-12:00', '13:00-17:00'])",
    )
    impulse_vol_expansion_ratio: float = Field(
        default=1.03,
        gt=0,
        description="Required volatility expansion ratio (short-term / long-term, reduced from 1.08 to 1.03 for more opportunities)",
    )

    # Impulse regime / sizing controls
    impulse_allow_normal: bool = Field(
        default=True,
        description="Allow impulse engine to trade in NORMAL regime (with reduced size multiplier)",
    )
    impulse_normal_size_multiplier: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Size multiplier for impulse trades in NORMAL regime (<= 1.0)",
    )
    volatility_hard_block_percentile: float = Field(
        default=99.0,
        ge=0,
        le=100,
        description="Percentile of recent volatility above which trades are hard-blocked (P99 - extreme levels only)",
    )

    # Exchange selection (multi-exchange abstraction layer)
    exchange: str = Field(
        default="bybit",
        description=(
            "Exchange to use for trading. Currently supported: 'bybit'. "
            "Register additional exchanges via ExchangeFactory.register() before startup."
        ),
    )

    # Bybit API (for future live trading)
    bybit_api_key: str = Field(default="", description="Bybit API key")
    bybit_api_secret: str = Field(default="", description="Bybit API secret")
    bybit_testnet: bool = Field(default=True, description="Use Bybit testnet (default True for safety)")

    # Safety: require explicit confirmation for LIVE trading
    require_live_confirm: bool = Field(
        default=False,
        description="Require explicit LIVE confirmation (set to True after all smoke tests PASS)",
    )

    # LLM API Keys (for agent generation, catalyst, and process factory)
    gemini_api_key: str = Field(default="", description="Google Gemini API key for agent generation and catalyst")
    openai_api_key: str = Field(default="", description="OpenAI API key for process factory and agent generation")
    openrouter_api_key: str = Field(default="", description="OpenRouter API key (Qwen3-Max-Thinking for brain + agent gen)")

    # Claude Brain
    anthropic_api_key: str = Field(default="", description="Anthropic API key for Claude Brain analysis")
    brain_analysis_interval: int = Field(default=60, gt=5, description="Brain analysis interval in seconds")
    brain_enabled: bool = Field(default=True, description="Enable Claude Brain analysis module")

    # Sovereign Brain (independent, no Anthropic dependency)
    sovereign_brain_enabled: bool = Field(
        default=False,
        description="Force SovereignBrain even if anthropic_api_key is present",
    )
    groq_api_key: str = Field(default="", description="Groq API key (Llama-3.3-70B, free tier 14k req/day)")
    deepseek_api_key: str = Field(default="", description="DeepSeek API key (deepseek-reasoner, $0.55/M tokens)")
    brain_local_model: str = Field(
        default="deepseek-r1:14b",
        description="Ollama model for SovereignBrain local analysis (default: deepseek-r1:14b)",
    )
    coinglass_api_key: str = Field(default="", description="CoinGlass API key (OI, liquidations, L/S ratio)")
    glassnode_api_key: str = Field(default="", description="Glassnode API key (SOPR, MVRV Z-Score, exchange flows)")
    coingecko_api_key: str = Field(default="", description="CoinGecko API key (BTC dominance, optional)")
    fred_api_key: str = Field(default="", description="FRED API key (DXY, Fed Funds Rate)")
    brain_signal_refresh_interval: int = Field(
        default=300, gt=30,
        description="How often to refresh data collectors in SovereignBrain (seconds, default 300)",
    )
    brain_ensemble_threshold: float = Field(
        default=0.55, ge=0.30, le=0.95,
        description="Minimum weighted-score magnitude to emit a directional signal (default 0.55)",
    )
    brain_accuracy_tracking: bool = Field(
        default=True,
        description="Enable Brier Score accuracy tracking with DuckDB persistence",
    )

    # Ollama Sentiment (local LLM, free, no API key required)
    ollama_enabled: bool = Field(
        default=True,
        description="Enable Ollama local LLM sentiment analysis (runs alongside Brain)",
    )
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL (default: http://localhost:11434)",
    )
    ollama_model: str = Field(
        default="llama3.2:3b",
        description="Ollama model to use for sentiment analysis (default: llama3.2:3b)",
    )
    ollama_sentiment_interval: int = Field(
        default=300,
        gt=5,
        description="Seconds between Ollama sentiment analyses (default 300 = 5 min)",
    )

    # RL Risk Manager (trained PPO agent for dynamic risk adjustment)
    rl_risk_enabled: bool = Field(
        default=True,
        description="Enable RL-based risk manager (graceful fallback to rule-based if no model)",
    )
    rl_risk_model_path: str = Field(
        default="",
        description="Path to trained PPO model for RL risk manager (.zip file)",
    )
    rl_risk_adjust_interval: int = Field(
        default=60,
        gt=5,
        description="Seconds between RL risk parameter adjustments (default 60)",
    )

    # Dynamic Oracle Weighting
    oracle_dynamic_weighting: bool = Field(
        default=True,
        description="Enable dynamic AI/ML model weighting based on market conditions",
    )
    oracle_min_fresh_quorum: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Minimum fresh+agreeing sources required to publish oracle signal (Quorum Kill-Switch)",
    )
    oracle_quorum_freshness_window_sec: int = Field(
        default=300,
        gt=0,
        description="Max age in seconds for a source to be considered fresh for quorum check",
    )
    oracle_regime_silencing_enabled: bool = Field(
        default=True,
        description="Silence irrelevant sources by regime: TCN=0 in ICE phase, Brain=0 in VAPOR phase",
    )
    tcn_credibility_constant: float = Field(
        default=30.0,
        gt=0,
        description="Credibility constant K in Z=n/(n+K). Lower K = faster credibility gain (actuarial)",
    )

    # Physics-Aware Position Sizing
    physics_aware_sizing: bool = Field(
        default=True,
        description="Enable physics-aware position sizing (adjusts size based on temperature, entropy, phase)",
    )

    # Strategy Capital Allocator
    strategy_capital_allocation: bool = Field(
        default=True,
        description="Enable dynamic capital allocation across strategies based on performance + regime",
    )
    capital_allocation_method: str = Field(
        default="hybrid",
        description="Capital allocation method: 'performance_weighted', 'phase_matched', or 'hybrid'",
    )

    # Source-of-truth for specialized services (local engine or external Redis microservices)
    physics_source: Literal["local", "redis"] = Field(
        default="local",
        description="Physics source: 'local' uses in-process PhysicsEngine, 'redis' consumes physics:* streams.",
    )
    brain_source: Literal["local", "redis"] = Field(
        default="local",
        description="Brain source: 'local' uses in-process ClaudeBrainClient, 'redis' consumes brain:analysis stream.",
    )
    risk_source: Literal["local", "redis"] = Field(
        default="local",
        description="Risk source: 'local' uses in-process risk gates, 'redis' also consumes risk:approved stream.",
    )
    microservices_group_prefix: str = Field(
        default="hean-core",
        description="Redis stream consumer group prefix for MicroservicesBridge.",
    )

    # EventBus configuration
    bus_circuit_open_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Queue utilization fraction at which EventBus circuit breaker opens (default 0.95 = 95%)",
    )
    bus_circuit_close_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Queue utilization fraction at which EventBus circuit breaker closes (default 0.70 = 70%)",
    )
    bus_degraded_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Queue utilization fraction at which EventBus enters degraded state (default 0.80 = 80%)",
    )
    bus_max_queue_size: int = Field(
        default=50000,
        gt=0,
        description="Maximum EventBus queue size across all priority levels (anti-memory-leak guard)",
    )

    # AI Factory (Shadow → Canary → Production pipeline)
    ai_factory_enabled: bool = Field(
        default=True,
        description="Enable AI Factory for automated strategy testing and promotion",
    )
    canary_percent: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Percentage of traffic for canary testing (default 10%)",
    )

    # AI Council (multi-model periodic system review)
    council_enabled: bool = Field(
        default=True,
        description="Enable AI Council for periodic multi-model system review",
    )
    council_review_interval: int = Field(
        default=21600,
        gt=300,
        description="Seconds between council review sessions (default: 6 hours)",
    )
    council_auto_apply_safe: bool = Field(
        default=True,
        description="Auto-apply safe parameter/strategy recommendations",
    )

    # Trade Council 2.0 (real-time adversarial signal evaluation)
    trade_council_enabled: bool = Field(
        default=True,
        description="Enable adversarial Trade Council for real-time signal evaluation",
    )
    trade_council_entry_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum weighted confidence to approve a trade (0-1)",
    )
    trade_council_exit_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Below this confidence, consider early exit (0-1)",
    )

    # ── Temporal Event Fabric ─────────────────────────────────────────
    fabric_enabled: bool = Field(
        default=True,
        description="Enable Temporal Event Fabric (Event DNA + EEV scoring)",
    )

    self_insight_enabled: bool = Field(
        default=True,
        description="Enable self-telemetry collection for Brain/Council analysis",
    )
    self_insight_interval: int = Field(
        default=60,
        gt=10,
        description="Seconds between publishing self-analytics snapshots",
    )

    # API Authentication (CRITICAL: Enable in production!)
    api_auth_enabled: bool = Field(
        default=False,
        description="Enable API authentication (set to True in production)",
    )
    api_auth_key: str = Field(
        default="",
        description="API key for authentication (generate with: python -c 'import secrets; print(secrets.token_hex(32))')",
    )
    api_auth_key_grace_period: int = Field(
        default=3600,
        gt=0,
        description=(
            "Grace period in seconds for the old API key after rotation. "
            "During this window the old key is stored in Redis under "
            "'hean:auth:secondary_key' and is accepted alongside the new primary key. "
            "Requests using the old key receive X-Key-Deprecated: true."
        ),
    )
    jwt_secret: str = Field(
        default="",
        description="JWT secret for token signing (auto-generated if not set)",
    )
    ws_allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
        description="Allowed origins for WebSocket connections (CORS)",
    )

    @field_validator("ws_allowed_origins", mode="before")
    @classmethod
    def parse_ws_allowed_origins(cls, v: Any) -> Any:
        """Parse ws_allowed_origins from env values."""
        if v is None or v == "":
            return ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"]
        return parse_list_env(v)

    # Observability — OpenTelemetry distributed tracing
    otel_enabled: bool = Field(
        default=False,
        description="Enable OpenTelemetry distributed tracing (requires opentelemetry-sdk installed)",
    )
    otel_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP gRPC collector endpoint (Jaeger all-in-one default: http://localhost:4317)",
    )
    otel_service_name: str = Field(
        default="hean",
        description="Service name emitted in trace spans (visible in Jaeger UI)",
    )
    otel_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of traces to sample (1.0=100%, 0.1=10%). Uses ParentBased sampler.",
    )

    # Observability — logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="text",
        description=(
            "Log output format: 'json' for structured JSON (Loki/Datadog/CloudWatch) "
            "or 'text' for human-readable console output (local development). "
            "Defaults to 'text'; set LOG_FORMAT=json in Docker/production environments."
        ),
    )
    health_check_port: int = Field(default=8080, gt=0, le=65535, description="Health check port")
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode (bypasses safety checks, verbose logging). WARNING: Only for development!",
    )

    # Redis (distributed event bus / AFO streaming)
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis URL for AFO event streaming and shared state (reads from REDIS_URL env var, defaults to redis://redis:6379/0)",
    )

    # Backtest
    backtest_initial_capital: float = Field(
        default=10000.0, gt=0, description="Initial capital for backtesting"
    )
    backtest_maker_fee: float = Field(
        default=0.00005,
        ge=0,
        le=0.01,
        description="Maker fee for backtesting (default 0.005% = 0.5 bps, reduced from 0.01%)",
    )
    backtest_taker_fee: float = Field(
        default=0.0003,
        ge=0,
        le=0.01,
        description="Taker fee for backtesting (default 0.03% = 3 bps, reduced from 0.06%)",
    )

    # Execution Policy - MAKER-ONLY for small capital!
    maker_first: bool = Field(
        default=True,
        description="Use maker-first execution (post-only limit orders) - ALWAYS TRUE!",
    )
    maker_ttl_ms: int = Field(
        default=150,  # Phase 2 Optimization: 150ms (was 8000ms) for HFT execution
        gt=0,
        description="Time-to-live for maker orders in milliseconds (Phase 2: 150ms optimal for HFT, +5-15 bps per execution)",
    )
    allow_taker_fallback: bool = Field(
        default=True,
        description="Allow taker fallback if maker order expires and edge is still positive (smart fallback with edge check)",
    )
    maker_price_offset_bps: int = Field(
        default=1,
        ge=0,
        description="Price offset in basis points for maker orders (best bid/ask ± offset, reduced to 1 bps for better fill rate)",
    )
    maker_fill_tolerance_bps: float = Field(
        default=0.5,
        ge=0,
        description="Tolerance in bps for maker order fills - allows fills when price is within this range of limit (default 0.5 bps)",
    )

    # Execution Edge Estimator (optimized for small capital)
    min_edge_bps_normal: float = Field(
        default=5.0,
        ge=0,
        description="Minimum edge in bps for NORMAL regime (lowered for more trades)",
    )
    min_edge_bps_impulse: float = Field(
        default=8.0,
        ge=0,
        description="Minimum edge in bps for IMPULSE regime (lowered for more trades)",
    )
    min_edge_bps_range: float = Field(
        default=3.0,
        ge=0,
        description="Minimum edge in bps for RANGE regime (lowered for more trades)",
    )

    # Income streams - enable/disable and budgets
    stream_funding_enabled: bool = Field(default=True, description="Enable funding income stream")
    stream_maker_rebate_enabled: bool = Field(
        default=True, description="Enable maker rebate income stream"
    )
    stream_basis_enabled: bool = Field(default=True, description="Enable basis hedge income stream")
    stream_volatility_enabled: bool = Field(
        default=True, description="Enable volatility harvest income stream"
    )

    stream_funding_capital_pct: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Capital allocation fraction for funding income stream",
    )
    stream_maker_rebate_capital_pct: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Capital allocation fraction for maker rebate income stream",
    )
    stream_basis_capital_pct: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description="Capital allocation fraction for basis hedge income stream",
    )
    stream_volatility_capital_pct: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Capital allocation fraction for volatility harvest income stream",
    )

    stream_funding_max_positions: int = Field(
        default=2,
        ge=0,
        description="Maximum concurrent positions for funding income stream",
    )
    stream_maker_rebate_max_positions: int = Field(
        default=10,
        ge=0,
        description="Maximum concurrent positions for maker rebate income stream",
    )
    stream_basis_max_positions: int = Field(
        default=4,
        ge=0,
        description="Maximum concurrent positions for basis hedge income stream",
    )
    stream_volatility_max_positions: int = Field(
        default=3,
        ge=0,
        description="Maximum concurrent positions for volatility harvest income stream",
    )

    # Strategy / Decision Memory
    memory_window_trades: int = Field(
        default=30,
        gt=0,
        description="Number of trades to look back for rolling profit factor in memory layers",
    )
    regime_blacklist_duration_days: int = Field(
        default=7, gt=0, description="Duration in days to blacklist a regime after repeated losses"
    )
    drawdown_threshold_pct: float = Field(
        default=15.0,
        gt=0,
        le=100,
        description="Strategy-level drawdown threshold percentage to trigger cooldown",
    )
    pf_penalty_threshold: float = Field(
        default=1.0, gt=0, description="Profit factor threshold below which penalties apply"
    )

    # Decision Memory specific
    memory_loss_streak: int = Field(
        default=3,
        ge=1,
        description="Consecutive losses in a context before temporarily blocking it",
    )
    memory_block_hours: int = Field(
        default=24,
        gt=0,
        description="Number of hours to block a bad context after loss streak/drawdown",
    )
    memory_drawdown_threshold_pct: float = Field(
        default=12.0,
        gt=0,
        le=100,
        description="Context-level drawdown threshold percentage to trigger cooldown block",
    )

    # Process Factory (experimental extension layer)
    process_factory_enabled: bool = Field(
        default=False,
        description="Enable Process Factory extension layer (experimental, disabled by default)",
    )
    process_factory_allow_actions: bool = Field(
        default=False,
        description="Allow Bybit actions from Process Factory (disabled by default, requires explicit enable)",
    )

    # Dry Run - REMOVED (No paper trading - Bybit testnet only)
    dry_run: bool = Field(
        default=False,
        description="DEPRECATED - No longer used. System uses Bybit testnet for all trading.",
    )

    # Execution Smoke Test
    execution_smoke_test_enabled: bool = Field(
        default=False,
        description="Enable execution smoke test (disabled by default)",
    )
    execution_smoke_test_symbol: str = Field(
        default="BTCUSDT",
        description="Symbol for smoke test (default BTCUSDT)",
    )
    execution_smoke_test_notional_usd: float = Field(
        default=5.0,
        gt=0,
        description="Notional value in USD for smoke test order (default 5)",
    )
    execution_smoke_test_side: Literal["BUY", "SELL"] = Field(
        default="BUY",
        description="Side for smoke test order (default BUY)",
    )
    execution_smoke_test_mode: Literal["PLACE_CANCEL"] = Field(
        default="PLACE_CANCEL",
        description="Smoke test mode (default PLACE_CANCEL: place and immediately cancel)",
    )

    # Paper Trade Assist - REMOVED (No paper trading mode)
    paper_trade_assist: bool = Field(
        default=False,
        description="DEPRECATED - No longer used. System uses Bybit testnet only.",
    )

    # Profit Capture (AFO-Director feature)
    profit_capture_enabled: bool = Field(
        default=True,
        description="Enable profit capture feature. Automatically locks profits when target is reached.",
    )
    profit_capture_target_usd: float = Field(
        default=1000.0,
        gt=0,
        description="Dollar profit threshold to trigger capture (default $1000). Closes profitable positions when unrealized profit >= this amount.",
    )
    profit_capture_target_pct: float = Field(
        default=20.0,
        gt=0,
        description="Profit capture target percentage (default 20%). Triggers when equity grows by this % from start.",
    )
    profit_capture_trail_pct: float = Field(
        default=10.0,
        gt=0,
        description="Profit capture trail percentage (default 10%). Triggers when drawdown from peak reaches this %.",
    )
    profit_capture_mode: Literal["partial", "full"] = Field(
        default="full",
        description="Profit capture mode: 'full' closes all positions and cancels orders, 'partial' reduces exposure.",
    )
    profit_capture_after_action: Literal["pause", "continue"] = Field(
        default="continue",
        description="Action after profit capture: 'pause' stops trading, 'continue' continues trading with new capital base.",
    )
    profit_capture_continue_risk_mult: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Risk multiplier when continuing after profit capture (default 0.5 = 50% of normal risk).",
    )
    paper_trade_assist_micro_trade_interval_sec: int = Field(
        default=60,
        gt=0,
        description="Interval in seconds for fallback micro-trades when PAPER_TRADE_ASSIST=true (default 60)",
    )
    paper_trade_assist_micro_trade_notional_usd: float = Field(
        default=10.0,
        gt=0,
        description="Notional value in USD for fallback micro-trades (default 10)",
    )
    paper_trade_assist_micro_trade_tp_pct: float = Field(
        default=0.3,
        gt=0,
        le=5.0,
        description="Take profit percentage for micro-trades (default 0.3%)",
    )
    paper_trade_assist_micro_trade_sl_pct: float = Field(
        default=0.3,
        gt=0,
        le=5.0,
        description="Stop loss percentage for micro-trades (default 0.3%)",
    )
    paper_trade_assist_micro_trade_max_time_min: int = Field(
        default=5,
        gt=0,
        description="Maximum time in minutes for micro-trade before forced exit (default 5)",
    )

    # Phase 5: Statistical Arbitrage & Anti-Fragile Architecture
    phase5_correlation_engine_enabled: bool = Field(
        default=True,
        description="Enable Correlation Engine for pair trading (Phase 5)",
    )
    phase5_safety_net_enabled: bool = Field(
        default=True,
        description="Enable Global Safety Net (Black Swan Protection) (Phase 5)",
    )
    phase5_kelly_criterion_enabled: bool = Field(
        default=True,
        description="Enable Kelly Criterion for position sizing (Phase 5)",
    )
    phase5_kelly_fractional: float = Field(
        default=0.25,
        ge=0.1,
        le=1.0,
        description="Fractional Kelly to use (0.25 = quarter Kelly, recommended) (Phase 5)",
    )
    phase5_self_healing_enabled: bool = Field(
        default=True,
        description="Enable Self-Healing Middleware for system monitoring (Phase 5)",
    )
    phase5_correlation_min_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum correlation threshold for pair trading (default 0.7 = 70%) (Phase 5)",
    )
    phase5_correlation_gap_threshold: float = Field(
        default=2.0,
        gt=0,
        description="Price gap threshold in standard deviations for pair trading (default 2.0) (Phase 5)",
    )
    phase5_entropy_spike_threshold: float = Field(
        default=3.0,
        gt=1.0,
        description="Market entropy spike threshold (300% = 3.0x baseline) for safety net (Phase 5)",
    )
    phase5_emergency_size_multiplier: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Position size multiplier during emergency (80% reduction = 0.2x) (Phase 5)",
    )

    # Phase 2: Execution Cost Optimization
    twap_enabled: bool = Field(
        default=True,
        description="Enable TWAP execution for large orders",
    )
    twap_threshold_usd: float = Field(
        default=500.0,
        gt=0,
        description="Minimum order size (USD) to use TWAP (default $500)",
    )
    twap_duration_sec: int = Field(
        default=300,
        gt=0,
        description="TWAP execution duration in seconds (default 300 = 5 min)",
    )
    twap_num_slices: int = Field(
        default=10,
        gt=0,
        description="Number of slices for TWAP execution (default 10)",
    )
    smart_order_selection_enabled: bool = Field(
        default=True,
        description="Enable smart order type selection (limit vs market based on edge)",
    )

    # Phase 2: Physics Integration
    physics_sizing_enabled: bool = Field(
        default=True,
        description="Enable physics-aware position sizing (temperature, entropy, phase)",
    )
    physics_filter_enabled: bool = Field(
        default=True,
        description="Enable physics-based signal filtering",
    )
    physics_filter_strict: bool = Field(
        default=True,
        description="If True, blocks counter-phase signals; if False, only penalizes",
    )

    # Physics engine publish throttling and normalisation
    physics_publish_min_interval_ms: int = Field(
        default=100,
        ge=0,
        description=(
            "Minimum milliseconds between PHYSICS_UPDATE publishes per symbol. "
            "Publish always occurs on phase change regardless of this setting. "
            "Set to 0 to disable throttling (publish every tick)."
        ),
    )
    physics_temperature_normalize_window: int = Field(
        default=1000,
        ge=10,
        description="Rolling window size (in ticks) for per-symbol temperature z-score normalisation.",
    )
    physics_entropy_normalize_window: int = Field(
        default=1000,
        ge=10,
        description="Rolling window size (in ticks) for per-symbol entropy z-score normalisation.",
    )

    # Digital Organism: MarketGenomeDetector (Stage 1)
    market_genome_enabled: bool = Field(
        default=True,
        description="Enable MarketGenomeDetector (unified market state synthesis)",
    )
    market_genome_interval: float = Field(
        default=10.0,
        gt=1.0,
        description="Seconds between MARKET_GENOME_UPDATE publications per symbol",
    )

    # Digital Organism: DoomsdaySandbox (Stage 2)
    doomsday_sandbox_enabled: bool = Field(
        default=True,
        description="Enable DoomsdaySandbox catastrophe simulation engine",
    )
    doomsday_interval_sec: int = Field(
        default=3600,
        gt=60,
        description="Seconds between automatic stress test runs",
    )
    doomsday_auto_protect: bool = Field(
        default=True,
        description="Auto-trigger SOFT_BRAKE when survival_score < threshold",
    )
    doomsday_survival_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Survival score threshold for auto-protection (0.0-1.0)",
    )
    doomsday_run_on_physics_alert: bool = Field(
        default=True,
        description="Auto-run simulations when physics detects danger signals",
    )

    # Digital Organism: MetaStrategyBrain (Stage 3)
    meta_brain_enabled: bool = Field(
        default=True,
        description="Enable MetaStrategyBrain for dynamic strategy lifecycle management",
    )
    meta_brain_evaluation_interval: int = Field(
        default=300,
        gt=30,
        description="Seconds between strategy fitness evaluations",
    )

    # Phase 2: Symbiont X Bridge
    symbiont_x_enabled: bool = Field(
        default=True,
        description="Enable Symbiont X GA optimization bridge",
    )
    symbiont_x_generations: int = Field(
        default=50,
        gt=0,
        description="Number of GA generations per optimization cycle (default 50)",
    )
    symbiont_x_population_size: int = Field(
        default=20,
        gt=0,
        description="Population size for GA (default 20)",
    )
    symbiont_x_mutation_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Mutation rate for GA (default 0.1 = 10%)",
    )
    symbiont_x_reoptimize_interval: int = Field(
        default=3600,
        gt=60,
        description="Seconds between optimization cycles (default 3600 = 1 hour)",
    )

    # Phase 3: SovereignSymbiont — full evolutionary bridge
    symbiont_enabled: bool = Field(
        default=False,
        description=(
            "Enable SovereignSymbiont full evolutionary bridge "
            "(EvolutionEngine + immune gate + live STRATEGY_PARAMS_UPDATED). "
            "Distinct from symbiont_x_enabled (simpler GA bridge)."
        ),
    )
    symbiont_evolution_interval: int = Field(
        default=300,
        gt=30,
        description="Seconds between SovereignSymbiont evolution cycles (default 300 = 5 min)",
    )
    symbiont_population_size: int = Field(
        default=50,
        gt=0,
        description="Population size for SovereignSymbiont EvolutionEngine (default 50)",
    )
    symbiont_elite_size: int = Field(
        default=5,
        gt=0,
        description="Elite size preserved unchanged per generation (default 5)",
    )
    symbiont_min_wfa_efficiency: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum Walk-Forward Efficiency ratio for genome promotion. "
            "Genomes below this threshold are never promoted to live strategies."
        ),
    )
    symbiont_min_promotion_fitness: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum composite fitness score for genome promotion "
            "(weighted blend of wfa, calmar, omega, sortino — all normalised to [0,1])."
        ),
    )

    # ARCHON Brain-Orchestrator
    archon_enabled: bool = Field(default=True, description="Enable ARCHON orchestrator")
    archon_signal_pipeline_enabled: bool = Field(
        default=True, description="Enable signal lifecycle tracking"
    )
    archon_reconciliation_enabled: bool = Field(
        default=True, description="Enable periodic state reconciliation"
    )
    archon_cortex_enabled: bool = Field(
        default=True, description="Enable Cortex decision engine"
    )
    archon_cortex_interval_sec: int = Field(
        default=30, description="Cortex decision loop interval"
    )
    archon_heartbeat_interval_sec: float = Field(
        default=5.0, description="Component heartbeat interval"
    )
    archon_signal_timeout_sec: float = Field(
        default=10.0, description="Signal stage timeout before dead-letter"
    )
    archon_max_active_signals: int = Field(
        default=1000, description="Max concurrent tracked signals"
    )
    archon_reconciliation_interval_sec: int = Field(
        default=30, description="Position reconciliation interval"
    )
    archon_chronicle_enabled: bool = Field(
        default=True, description="Enable audit trail"
    )
    archon_chronicle_max_memory: int = Field(
        default=10000, description="Max in-memory chronicle entries"
    )

    # ------------------------------------------------------------------
    # Secrets integration — highest-priority settings source
    # ------------------------------------------------------------------

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Inject SecretsManager as the highest-priority configuration source.

        Resolution order (first match wins):
        1. SecretsManager   — Docker secrets, K8s secrets, env-var fallback
        2. env_settings     — standard environment variables (pydantic-settings)
        3. dotenv_settings  — .env file (pydantic-settings)
        4. init_settings    — values passed to HEANSettings() constructor

        Sensitive fields resolved through SecretsManager:
            ``bybit_api_key``, ``bybit_api_secret``, ``anthropic_api_key``,
            ``api_auth_key``, ``gemini_api_key``, ``openai_api_key``,
            ``openrouter_api_key``, ``jwt_secret``
        """
        # Import here to avoid circular imports at module level (config.py is
        # imported very early; secrets.py imports hean.logging which may import
        # config.py indirectly in some code paths).
        from hean.core.secrets import SecretsSettingsSource, secrets as _secrets_mgr

        _SENSITIVE_FIELDS: tuple[str, ...] = (
            "bybit_api_key",
            "bybit_api_secret",
            "anthropic_api_key",
            "api_auth_key",
            "gemini_api_key",
            "openai_api_key",
            "openrouter_api_key",
            "jwt_secret",
        )

        secrets_source = SecretsSettingsSource(
            settings_cls=settings_cls,
            manager=_secrets_mgr,
            secret_fields=_SENSITIVE_FIELDS,
        )

        return (
            secrets_source,   # 1. Docker/K8s/env secrets — highest priority
            env_settings,     # 2. Plain environment variables
            dotenv_settings,  # 3. .env file
            init_settings,    # 4. Constructor kwargs — lowest priority
        )

    def update_safe(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Apply a batch of config updates, enforcing the hot-reload whitelist.

        Each key is validated against ``SAFE_RELOAD_KEYS`` and ``BLOCKED_KEYS``.
        Values are coerced to the declared Pydantic field type before being
        written via ``object.__setattr__`` (which bypasses the frozen-model
        guard while still keeping the runtime object mutated in-place).

        Args:
            updates: Mapping of ``{field_name: new_value}``.

        Returns:
            Mapping of ``{field_name: new_value}`` containing only the keys
            that were actually changed (skips unchanged and rejected keys).
        """
        import json as _json

        applied: dict[str, Any] = {}

        for key, raw in updates.items():
            # Hard security boundary – always rejected
            if key in BLOCKED_KEYS:
                continue

            # Must be in the whitelist
            if key not in SAFE_RELOAD_KEYS:
                continue

            # Unknown field on the model – skip
            if key not in self.model_fields:
                continue

            field_info = self.model_fields[key]
            annotation = field_info.annotation

            # Attempt type coercion
            try:
                origin = getattr(annotation, "__origin__", None)
                if annotation is bool:
                    if isinstance(raw, bool):
                        new_val: Any = raw
                    elif isinstance(raw, str):
                        new_val = raw.lower() in ("true", "1", "yes")
                    else:
                        new_val = bool(raw)
                elif annotation is int:
                    new_val = int(raw)
                elif annotation is float:
                    new_val = float(raw)
                elif annotation is str:
                    new_val = str(raw)
                elif origin is list:
                    if isinstance(raw, list):
                        new_val = raw
                    elif isinstance(raw, str):
                        try:
                            parsed = _json.loads(raw)
                            new_val = parsed if isinstance(parsed, list) else [raw]
                        except _json.JSONDecodeError:
                            new_val = [item.strip() for item in raw.split(",") if item.strip()]
                    else:
                        new_val = list(raw)
                else:
                    new_val = raw
            except (TypeError, ValueError):
                continue

            # Skip if value is unchanged
            current = getattr(self, key, None)
            if new_val == current:
                continue

            object.__setattr__(self, key, new_val)
            applied[key] = new_val

        return applied

    def model_post_init(self, __context: Any) -> None:
        """Validate trading mode after initialization."""
        # CRITICAL SAFETY CHECK: Prevent LIVE trading without explicit confirmation
        is_attempting_live = (
            not self.dry_run
            and self.live_confirm == "YES"
            and not self.bybit_testnet
        )

        if is_attempting_live and not self.require_live_confirm:
            raise ValueError(
                "🚨 LIVE TRADING BLOCKED 🚨\n"
                "LIVE trading mode requires REQUIRE_LIVE_CONFIRM=true.\n"
                "This safety check ensures all smoke tests have PASSED before enabling real trading.\n"
                f"Current config: DRY_RUN={self.dry_run}, LIVE_CONFIRM={self.live_confirm}, "
                f"BYBIT_TESTNET={self.bybit_testnet}, REQUIRE_LIVE_CONFIRM={self.require_live_confirm}\n"
                "To enable LIVE trading:\n"
                "1. Run all smoke tests and verify PASS\n"
                "2. Set REQUIRE_LIVE_CONFIRM=true in backend.env\n"
                "3. Set BYBIT_TESTNET=false, DRY_RUN=false, LIVE_CONFIRM=YES"
            )

        # Check environment for LIVE_CONFIRM if not already set
        live_confirm_env = os.getenv("LIVE_CONFIRM", "")
        if live_confirm_env == "YES" and self.trading_mode == "paper":
            # Update trading mode if LIVE_CONFIRM is set
            object.__setattr__(self, "trading_mode", "live")
            object.__setattr__(self, "live_confirm", "YES")

        # Auto-set live_confirm to YES if trading_mode is live and dry_run is False
        if self.trading_mode == "live" and not self.dry_run and self.live_confirm != "YES":
            object.__setattr__(self, "live_confirm", "YES")

        # Set dry_run default from environment if not explicitly set
        # DRY_RUN env var: "true"/"1"/"yes" -> True, "false"/"0"/"no" -> False, unset -> True (default)
        dry_run_env = os.getenv("DRY_RUN", "").lower()
        if dry_run_env and dry_run_env not in ("true", "1", "yes", "false", "0", "no"):
            # Only override if explicitly set in env and value is valid
            from hean.logging import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Invalid DRY_RUN value: {dry_run_env!r}. Ignoring.")
        elif dry_run_env in ("false", "0", "no"):
            object.__setattr__(self, "dry_run", False)

        # Paper trade assist is deprecated - no validation needed
        # System always uses Bybit testnet for safety

    @model_validator(mode='after')
    def validate_risk_limits(self) -> 'HEANSettings':
        """Validate cross-field constraints to catch dangerous configuration combinations."""
        # max_trade_risk_pct * max_open_positions should not exceed 100%
        # (both stored as percent, e.g. 1.0 = 1%)
        combined_risk = (self.max_trade_risk_pct / 100.0) * self.max_open_positions
        if combined_risk > 1.0:
            import warnings
            warnings.warn(
                f"max_trade_risk_pct ({self.max_trade_risk_pct:.1f}%) * "
                f"max_open_positions ({self.max_open_positions}) = "
                f"{combined_risk:.1%} > 100%. This allows theoretically >100% capital at risk.",
                stacklevel=2,
            )

        # bus thresholds must be ordered correctly
        if self.bus_circuit_close_threshold >= self.bus_circuit_open_threshold:
            import warnings
            warnings.warn(
                f"bus_circuit_close_threshold ({self.bus_circuit_close_threshold}) must be "
                f"less than bus_circuit_open_threshold ({self.bus_circuit_open_threshold}). "
                "Circuit breaker may oscillate.",
                stacklevel=2,
            )

        return self

    @property
    def is_live(self) -> bool:
        """Check if system is in live trading mode (Always True - Bybit testnet)."""
        return True  # Always live mode with Bybit testnet


# Global settings instance
settings = HEANSettings()
