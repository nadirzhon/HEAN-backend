"""Data Collectors for Sovereign Brain HEAN.

9 authoritative external sources, all free or free-tier:
  - Fear & Greed Index (alternative.me)
  - CoinGlass: OI, liquidations, L/S ratio
  - Glassnode: SOPR, MVRV Z-Score, exchange flows
  - Binance public: funding rate
  - DeFiLlama: TVL flows
  - Blockchain.info: hash rate, mempool
  - CoinGecko: BTC dominance, market caps
  - FRED API: DXY, Fed Funds Rate
  - Google Trends: retail interest (pytrends)
"""

from .base import BaseCollector
from .binance_public import BinancePublicCollector
from .blockchain_info import BlockchainInfoCollector
from .coingecko import CoinGeckoCollector
from .coinglass import CoinGlassCollector
from .collector_manager import DataCollectorManager
from .defillama import DefiLlamaCollector
from .fear_greed import FearGreedCollector
from .fred_api import FredApiCollector
from .glassnode import GlassnodeCollector
from .google_trends import GoogleTrendsCollector

__all__ = [
    "BaseCollector",
    "BinancePublicCollector",
    "BlockchainInfoCollector",
    "CoinGeckoCollector",
    "CoinGlassCollector",
    "DataCollectorManager",
    "DefiLlamaCollector",
    "FearGreedCollector",
    "FredApiCollector",
    "GlassnodeCollector",
    "GoogleTrendsCollector",
]
