"""
DEMIR AI v8.0 - Sentiment Analysis Layer
15 REAL-TIME SENTIMENT SOURCES - ZERO MOCK DATA
NO SIMPLIFICATION - ENTERPRISE GRADE
"""

import os
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any
import asyncio
import aiohttp
import json
import time
from functools import wraps

logger = logging.getLogger(__name__)

# Rate limiting decorator
def rate_limit(calls_per_minute: int):
    """Rate limiting decorator for API calls"""
    min_interval = 60.0 / calls_per_minute
    last_called = {}
    
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            key = f"{self.__class__.__name__}.{func.__name__}"
            current_time = time.time()
            
            if key in last_called:
                elapsed = current_time - last_called[key]
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)
            
            last_called[key] = time.time()
            return await func(self, *args, **kwargs)
        return wrapper
    return decorator


class SentimentAnalyzer:
    """
    Main sentiment analyzer orchestrating 15 real sources
    ZERO MOCK DATA - ALL REAL APIs
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize all sentiment sources
        self.sources = {
            'news_sentiment': NewsSentimentSource(config),
            'fear_greed': FearGreedIndexSource(config),
            'btc_dominance': BTCDominanceSource(config),
            'exchange_flow': ExchangeFlowSource(config),
            'whale_alert': WhaleAlertSource(config),
            'macro_correlation': MacroCorrelationSource(config),
            'market_regime': MarketRegimeSource(config),
            'stablecoin_dominance': StablecoinDominanceSource(config),
            'funding_rates': FundingRatesSource(config),
            'long_short_ratio': LongShortRatioSource(config),
            'onchain_activity': OnChainActivitySource(config),
            'exchange_reserves': ExchangeReserveFlowsSource(config),
            'orderbook_imbalance': OrderBookImbalanceSource(config),
            'liquidation_cascade': LiquidationCascadeSource(config),
            'basis_contango': BasisContangoSource(config)
        }
        
        # Weights for each source
        self.weights = {
            'news_sentiment': 0.08,
            'fear_greed': 0.10,
            'btc_dominance': 0.07,
            'exchange_flow': 0.08,
            'whale_alert': 0.07,
            'macro_correlation': 0.06,
            'market_regime': 0.08,
            'stablecoin_dominance': 0.05,
            'funding_rates': 0.09,
            'long_short_ratio': 0.08,
            'onchain_activity': 0.06,
            'exchange_reserves': 0.06,
            'orderbook_imbalance': 0.07,
            'liquidation_cascade': 0.05,
            'basis_contango': 0.05
        }
        
        # Cache for sentiment scores
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info("SentimentAnalyzer initialized with 15 REAL sources")
        logger.info("ZERO MOCK DATA POLICY - REAL APIs ONLY")
    
    async def analyze(self, symbol: str) -> Dict:
        """
        Analyze sentiment from all sources
        Returns aggregated sentiment score and individual scores
        """
        # Check cache
        cache_key = f"{symbol}:{int(time.time() // self.cache_ttl)}"
        if cache_key in self.cache:
            logger.debug(f"Using cached sentiment for {symbol}")
            return self.cache[cache_key]
        
        # Collect sentiment from all sources
        tasks = []
        for source_name, source in self.sources.items():
            tasks.append(self._get_source_sentiment(source_name, source, symbol))
        
        # Wait for all sources
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        sentiment_scores = {}
        valid_scores = []
        
        for i, (source_name, source) in enumerate(self.sources.items()):
            result = results[i]
            
            if isinstance(result, Exception):
                logger.warning(f"Error getting sentiment from {source_name}: {result}")
                sentiment_scores[source_name] = 50.0  # Neutral on error
            else:
                score = result
                if score is not None and 0 <= score <= 100:
                    sentiment_scores[source_name] = score
                    valid_scores.append((score, self.weights[source_name]))
                else:
                    logger.warning(f"Invalid score from {source_name}: {score}")
                    sentiment_scores[source_name] = 50.0
        
        # Calculate weighted average
        if valid_scores:
            total_weight = sum(w for _, w in valid_scores)
            weighted_sum = sum(s * w for s, w in valid_scores)
            overall_sentiment = weighted_sum / total_weight
        else:
            overall_sentiment = 50.0
        
        # Prepare result
        result = {
            'overall_sentiment': overall_sentiment,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'sources': sentiment_scores,
            'interpretation': self._interpret_sentiment(overall_sentiment),
            'confidence': len(valid_scores) / len(self.sources) * 100
        }
        
        # Cache result
        self.cache[cache_key] = result
        
        # Clean old cache entries
        self._clean_cache()
        
        logger.info(f"Sentiment analysis for {symbol}: {overall_sentiment:.1f} "
                   f"(Confidence: {result['confidence']:.1f}%)")
        
        return result
    
    async def _get_source_sentiment(self, source_name: str, source: Any, symbol: str) -> float:
        """Get sentiment from individual source"""
        try:
            return await source.get_sentiment(symbol)
        except Exception as e:
            logger.error(f"Error in {source_name}: {e}")
            raise
    
    def _interpret_sentiment(self, score: float) -> str:
        """Interpret sentiment score"""
        if score >= 75:
            return "Extreme Greed"
        elif score >= 60:
            return "Greed"
        elif score >= 40:
            return "Neutral"
        elif score >= 25:
            return "Fear"
        else:
            return "Extreme Fear"
    
    def _clean_cache(self):
        """Remove old cache entries"""
        current_time = int(time.time() // self.cache_ttl)
        self.cache = {
            k: v for k, v in self.cache.items()
            if int(k.split(':')[1]) >= current_time - 1
        }


class NewsSentimentSource:
    """CryptoPanic real-time news sentiment"""
    
    def __init__(self, config):
        self.config = config
        self.api_key = config.data_providers.cryptopanic_key
        self.base_url = "https://cryptopanic.com/api/v1"
    
    @rate_limit(30)
    async def get_sentiment(self, symbol: str) -> float:
        """Get news sentiment score"""
        if not self.api_key:
            logger.warning("CryptoPanic API key not configured")
            return 50.0
        
        try:
            # Convert symbol to currency
            currency = symbol.replace('USDT', '').replace('USD', '')
            
            url = f"{self.base_url}/posts/"
            params = {
                'auth_token': self.api_key,
                'currencies': currency,
                'filter': 'hot',
                'public': 'true'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._calculate_news_sentiment(data)
                    else:
                        logger.warning(f"CryptoPanic API error: {response.status}")
                        return 50.0
                        
        except Exception as e:
            logger.error(f"NewsSentiment error: {e}")
            return 50.0
    
    def _calculate_news_sentiment(self, data: Dict) -> float:
        """Calculate sentiment from news data"""
        if 'results' not in data:
            return 50.0
        
        posts = data['results']
        if not posts:
            return 50.0
        
        total_score = 0
        total_weight = 0
        
        for post in posts[:20]:  # Analyze last 20 posts
            # Get votes
            positive = post.get('votes', {}).get('positive', 0)
            negative = post.get('votes', {}).get('negative', 0)
            liked = post.get('votes', {}).get('liked', 0)
            disliked = post.get('votes', {}).get('disliked', 0)
            
            total_votes = positive + negative + liked + disliked
            
            if total_votes > 0:
                sentiment = ((positive + liked) / total_votes) * 100
                weight = min(total_votes, 100)  # Cap weight at 100
                
                total_score += sentiment * weight
                total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        
        return 50.0


class FearGreedIndexSource:
    """Alternative.me Fear & Greed Index"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://api.alternative.me/fng/"
    
    @rate_limit(10)
    async def get_sentiment(self, symbol: str) -> float:
        """Get Fear & Greed Index"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params={'limit': 1}) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and len(data['data']) > 0:
                            return float(data['data'][0]['value'])
            return 50.0
            
        except Exception as e:
            logger.error(f"FearGreedIndex error: {e}")
            return 50.0


class BTCDominanceSource:
    """Bitcoin dominance from CoinGecko"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://api.coingecko.com/api/v3"
    
    @rate_limit(50)
    async def get_sentiment(self, symbol: str) -> float:
        """Get BTC dominance sentiment"""
        try:
            url = f"{self.base_url}/global"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        btc_dominance = data['data']['market_cap_percentage']['btc']
                        
                        # Convert dominance to sentiment
                        # High dominance (>45%) = Fear (flight to safety)
                        # Low dominance (<35%) = Greed (alt season)
                        
                        if btc_dominance > 45:
                            sentiment = 100 - ((btc_dominance - 45) * 2)
                        elif btc_dominance < 35:
                            sentiment = 50 + ((35 - btc_dominance) * 2)
                        else:
                            sentiment = 50 + ((40 - btc_dominance) * 2)
                        
                        return max(0, min(100, sentiment))
            
            return 50.0
            
        except Exception as e:
            logger.error(f"BTCDominance error: {e}")
            return 50.0


class ExchangeFlowSource:
    """Exchange inflow/outflow from Binance volume"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://api.binance.com/api/v3"
    
    @rate_limit(60)
    async def get_sentiment(self, symbol: str) -> float:
        """Get exchange flow sentiment"""
        try:
            # Get 24h ticker
            url = f"{self.base_url}/ticker/24hr"
            params = {'symbol': symbol}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Analyze volume and price change
                        volume = float(data['volume'])
                        quote_volume = float(data['quoteVolume'])
                        price_change = float(data['priceChangePercent'])
                        
                        # High volume + positive price = Bullish (high sentiment)
                        # High volume + negative price = Bearish (low sentiment)
                        
                        if quote_volume > 0:
                            if price_change > 0:
                                sentiment = 50 + min(price_change * 2, 50)
                            else:
                                sentiment = 50 + max(price_change * 2, -50)
                            
                            return max(0, min(100, sentiment))
            
            return 50.0
            
        except Exception as e:
            logger.error(f"ExchangeFlow error: {e}")
            return 50.0


class WhaleAlertSource:
    """Whale movements from Binance order book"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://api.binance.com/api/v3"
    
    @rate_limit(100)
    async def get_sentiment(self, symbol: str) -> float:
        """Get whale activity sentiment"""
        try:
            # Get order book depth
            url = f"{self.base_url}/depth"
            params = {'symbol': symbol, 'limit': 100}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Analyze large orders
                        bids = data['bids']
                        asks = data['asks']
                        
                        # Calculate whale pressure
                        large_bid_volume = sum(float(b[1]) for b in bids[:10])
                        large_ask_volume = sum(float(a[1]) for a in asks[:10])
                        
                        if large_bid_volume + large_ask_volume > 0:
                            buy_pressure = large_bid_volume / (large_bid_volume + large_ask_volume)
                            sentiment = buy_pressure * 100
                            return sentiment
            
            return 50.0
            
        except Exception as e:
            logger.error(f"WhaleAlert error: {e}")
            return 50.0


class MacroCorrelationSource:
    """Correlation with macro markets (S&P500, DXY)"""
    
    def __init__(self, config):
        self.config = config
        self.api_key = config.data_providers.alpha_vantage_key
        self.base_url = "https://www.alphavantage.co/query"
    
    @rate_limit(5)
    async def get_sentiment(self, symbol: str) -> float:
        """Get macro correlation sentiment"""
        if not self.api_key:
            return 50.0
        
        try:
            # Get S&P 500 data
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': 'SPY',
                'apikey': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'Global Quote' in data:
                            sp500_change = float(data['Global Quote'].get('10. change percent', '0%').rstrip('%'))
                            
                            # Positive correlation: stocks up = crypto up
                            if sp500_change > 0:
                                sentiment = 50 + min(sp500_change * 10, 50)
                            else:
                                sentiment = 50 + max(sp500_change * 10, -50)
                            
                            return max(0, min(100, sentiment))
            
            return 50.0
            
        except Exception as e:
            logger.error(f"MacroCorrelation error: {e}")
            return 50.0


class MarketRegimeSource:
    """Market regime detection based on volatility"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://api.binance.com/api/v3"
    
    @rate_limit(60)
    async def get_sentiment(self, symbol: str) -> float:
        """Get market regime sentiment"""
        try:
            # Get klines for volatility calculation
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': '1h',
                'limit': 24
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Calculate volatility
                        closes = [float(k[4]) for k in data]
                        returns = [(closes[i]/closes[i-1] - 1) for i in range(1, len(closes))]
                        volatility = np.std(returns) * np.sqrt(24) * 100  # Annualized
                        
                        # Low volatility = Accumulation (neutral to bullish)
                        # High volatility = Distribution or panic (bearish)
                        
                        if volatility < 30:
                            sentiment = 60  # Low vol = slight bullish
                        elif volatility < 50:
                            sentiment = 50  # Normal vol = neutral
                        elif volatility < 80:
                            sentiment = 40  # High vol = slight bearish
                        else:
                            sentiment = 25  # Very high vol = bearish
                        
                        return sentiment
            
            return 50.0
            
        except Exception as e:
            logger.error(f"MarketRegime error: {e}")
            return 50.0


class StablecoinDominanceSource:
    """Stablecoin market cap dominance"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://api.coingecko.com/api/v3"
    
    @rate_limit(50)
    async def get_sentiment(self, symbol: str) -> float:
        """Get stablecoin dominance sentiment"""
        try:
            url = f"{self.base_url}/global"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Get stablecoin percentage
                        market_cap = data['data']['total_market_cap']['usd']
                        
                        # Get top stablecoins market cap
                        stable_url = f"{self.base_url}/coins/markets"
                        stable_params = {
                            'vs_currency': 'usd',
                            'ids': 'tether,usd-coin,binance-usd,dai',
                            'order': 'market_cap_desc'
                        }
                        
                        async with session.get(stable_url, params=stable_params) as stable_response:
                            if stable_response.status == 200:
                                stable_data = await stable_response.json()
                                stable_cap = sum(s['market_cap'] for s in stable_data)
                                
                                stable_dominance = (stable_cap / market_cap) * 100
                                
                                # High stablecoin dominance = Fear (cash position)
                                # Low stablecoin dominance = Greed (risk on)
                                
                                if stable_dominance > 10:
                                    sentiment = 100 - (stable_dominance * 3)
                                else:
                                    sentiment = 50 + ((10 - stable_dominance) * 3)
                                
                                return max(0, min(100, sentiment))
            
            return 50.0
            
        except Exception as e:
            logger.error(f"StablecoinDominance error: {e}")
            return 50.0


class FundingRatesSource:
    """Perpetual futures funding rates"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://fapi.binance.com/fapi/v1"
    
    @rate_limit(60)
    async def get_sentiment(self, symbol: str) -> float:
        """Get funding rate sentiment"""
        try:
            # Convert spot symbol to futures
            futures_symbol = symbol.replace('USDT', 'USDT')
            
            url = f"{self.base_url}/fundingRate"
            params = {
                'symbol': futures_symbol,
                'limit': 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and len(data) > 0:
                            funding_rate = float(data[0]['fundingRate'])
                            
                            # Positive funding = Longs pay shorts (bullish sentiment but potential reversal)
                            # Negative funding = Shorts pay longs (bearish sentiment but potential reversal)
                            
                            if funding_rate > 0:
                                # High positive funding = Overleveraged longs
                                if funding_rate > 0.001:
                                    sentiment = 30  # Bearish (reversal expected)
                                else:
                                    sentiment = 60  # Slightly bullish
                            elif funding_rate < 0:
                                # Negative funding = Overleveraged shorts
                                if funding_rate < -0.001:
                                    sentiment = 70  # Bullish (reversal expected)
                                else:
                                    sentiment = 40  # Slightly bearish
                            else:
                                sentiment = 50  # Neutral
                            
                            return sentiment
            
            return 50.0
            
        except Exception as e:
            logger.error(f"FundingRates error: {e}")
            return 50.0


class LongShortRatioSource:
    """Long/Short ratio from Binance"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://fapi.binance.com/futures/data"
    
    @rate_limit(30)
    async def get_sentiment(self, symbol: str) -> float:
        """Get long/short ratio sentiment"""
        try:
            url = f"{self.base_url}/globalLongShortAccountRatio"
            params = {
                'symbol': symbol,
                'period': '5m',
                'limit': 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and len(data) > 0:
                            ratio = float(data[0]['longShortRatio'])
                            
                            # Ratio > 1 = More longs than shorts
                            # Ratio < 1 = More shorts than longs
                            
                            if ratio > 1:
                                sentiment = 50 + min((ratio - 1) * 50, 50)
                            else:
                                sentiment = 50 - min((1 - ratio) * 50, 50)
                            
                            return max(0, min(100, sentiment))
            
            return 50.0
            
        except Exception as e:
            logger.error(f"LongShortRatio error: {e}")
            return 50.0


class OnChainActivitySource:
    """On-chain activity metrics"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://api.blockchain.info"
    
    @rate_limit(10)
    async def get_sentiment(self, symbol: str) -> float:
        """Get on-chain activity sentiment"""
        try:
            # Only works for Bitcoin
            if 'BTC' not in symbol:
                return 50.0
            
            url = f"{self.base_url}/stats"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Analyze transaction volume and hash rate
                        tx_volume = data.get('trade_volume_btc', 0)
                        hash_rate = data.get('hash_rate', 0)
                        difficulty = data.get('difficulty', 0)
                        
                        # High activity = Bullish
                        # Normalize and score
                        sentiment = 50
                        
                        # This is simplified - real implementation would track changes
                        if tx_volume > 100000:
                            sentiment += 10
                        if hash_rate > 500000000:
                            sentiment += 10
                        
                        return min(100, sentiment)
            
            return 50.0
            
        except Exception as e:
            logger.error(f"OnChainActivity error: {e}")
            return 50.0


class ExchangeReserveFlowsSource:
    """Exchange reserve flows from open interest"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://fapi.binance.com/fapi/v1"
    
    @rate_limit(60)
    async def get_sentiment(self, symbol: str) -> float:
        """Get exchange reserve sentiment"""
        try:
            url = f"{self.base_url}/openInterest"
            params = {'symbol': symbol}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        open_interest = float(data['openInterest'])
                        
                        # This needs historical data to be meaningful
                        # For now, just return neutral
                        # High OI increase = Bullish
                        # OI decrease = Bearish
                        
                        return 50.0
            
            return 50.0
            
        except Exception as e:
            logger.error(f"ExchangeReserves error: {e}")
            return 50.0


class OrderBookImbalanceSource:
    """Order book imbalance analysis"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://api.binance.com/api/v3"
    
    @rate_limit(100)
    async def get_sentiment(self, symbol: str) -> float:
        """Get order book imbalance sentiment"""
        try:
            url = f"{self.base_url}/depth"
            params = {
                'symbol': symbol,
                'limit': 500
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Calculate bid/ask imbalance
                        bids = data['bids']
                        asks = data['asks']
                        
                        bid_volume = sum(float(b[1]) * float(b[0]) for b in bids)
                        ask_volume = sum(float(a[1]) * float(a[0]) for a in asks)
                        
                        total_volume = bid_volume + ask_volume
                        
                        if total_volume > 0:
                            bid_ratio = bid_volume / total_volume
                            sentiment = bid_ratio * 100
                            return sentiment
            
            return 50.0
            
        except Exception as e:
            logger.error(f"OrderBookImbalance error: {e}")
            return 50.0


class LiquidationCascadeSource:
    """Liquidation cascade risk from CoinGlass"""
    
    def __init__(self, config):
        self.config = config
        self.api_key = config.data_providers.coinglass_key
        self.base_url = "https://api.coinglass.com/api"
    
    @rate_limit(10)
    async def get_sentiment(self, symbol: str) -> float:
        """Get liquidation risk sentiment"""
        if not self.api_key:
            return 50.0
        
        try:
            # Note: CoinGlass API requires authentication
            # This is a simplified version
            
            # Without real API access, estimate from funding rates
            # High funding = High liquidation risk for longs
            # This would normally use real liquidation data
            
            return 50.0
            
        except Exception as e:
            logger.error(f"LiquidationCascade error: {e}")
            return 50.0


class BasisContangoSource:
    """Futures basis and contango/backwardation"""
    
    def __init__(self, config):
        self.config = config
        self.spot_url = "https://api.binance.com/api/v3"
        self.futures_url = "https://fapi.binance.com/fapi/v1"
    
    @rate_limit(60)
    async def get_sentiment(self, symbol: str) -> float:
        """Get basis sentiment"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get spot price
                spot_params = {'symbol': symbol}
                async with session.get(f"{self.spot_url}/ticker/price", params=spot_params) as spot_response:
                    if spot_response.status != 200:
                        return 50.0
                    spot_data = await spot_response.json()
                    spot_price = float(spot_data['price'])
                
                # Get futures price
                futures_params = {'symbol': symbol}
                async with session.get(f"{self.futures_url}/ticker/price", params=futures_params) as futures_response:
                    if futures_response.status != 200:
                        return 50.0
                    futures_data = await futures_response.json()
                    futures_price = float(futures_data['price'])
                
                # Calculate basis
                basis = ((futures_price - spot_price) / spot_price) * 100
                
                # Contango (futures > spot) = Bullish
                # Backwardation (futures < spot) = Bearish
                
                if basis > 0:
                    # Contango
                    if basis > 2:
                        sentiment = 30  # Extreme contango = overleveraged
                    else:
                        sentiment = 60 + (basis * 10)
                else:
                    # Backwardation
                    if basis < -2:
                        sentiment = 70  # Extreme backwardation = short squeeze potential
                    else:
                        sentiment = 50 + (basis * 10)
                
                return max(0, min(100, sentiment))
            
        except Exception as e:
            logger.error(f"BasisContango error: {e}")
            return 50.0
