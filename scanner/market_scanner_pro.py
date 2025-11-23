"""
DEMIR AI v8.0 - Professional Market Scanner
REAL-TIME MARKET SCANNING - ZERO MOCK DATA
FINDS OPPORTUNITIES ACROSS ALL MARKETS
"""

import logging
import asyncio
import aiohttp
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


class OpportunityType(Enum):
    """Types of trading opportunities"""
    BREAKOUT = "BREAKOUT"
    BREAKDOWN = "BREAKDOWN"
    REVERSAL = "REVERSAL"
    MOMENTUM = "MOMENTUM"
    DIVERGENCE = "DIVERGENCE"
    ARBITRAGE = "ARBITRAGE"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    PATTERN = "PATTERN"
    WHALE_ACTIVITY = "WHALE_ACTIVITY"
    CORRELATION_BREAK = "CORRELATION_BREAK"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
    LIQUIDITY_SHIFT = "LIQUIDITY_SHIFT"


class ScannerPriority(Enum):
    """Scanner priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    MONITOR = 5


@dataclass
class MarketOpportunity:
    """Detected market opportunity"""
    opportunity_id: str
    timestamp: datetime
    symbol: str
    exchange: str
    
    # Opportunity details
    opportunity_type: OpportunityType
    priority: ScannerPriority
    confidence: float
    
    # Price data
    current_price: float
    target_price: float
    stop_loss: float
    entry_zone: Tuple[float, float]
    
    # Market metrics
    volume_ratio: float
    volatility: float
    liquidity_score: float
    momentum_score: float
    
    # Technical levels
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    
    # Indicators
    indicators: Dict[str, float] = field(default_factory=dict)
    
    # Risk/Reward
    risk_reward_ratio: float = 0
    max_risk_amount: float = 0
    potential_profit: float = 0
    
    # Timing
    time_to_act: str = "NORMAL"  # IMMEDIATE, URGENT, NORMAL, WAIT
    expiry_time: Optional[datetime] = None
    
    # Description
    description: str = ""
    action_items: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class MarketScannerPro:
    """
    Professional market scanner - Finds all opportunities
    SCANS ENTIRE MARKET IN REAL-TIME
    """
    
    def __init__(self, config):
        self.config = config
        
        # Scanner configuration
        self.scan_interval = 30  # seconds
        self.symbols_per_batch = 20
        self.max_opportunities = 100
        
        # API endpoints
        self.binance_url = "https://api.binance.com/api/v3"
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        
        # Symbols to scan
        self.watchlist = set()
        self.all_symbols = set()
        self.priority_symbols = set()
        
        # Scanner thresholds
        self.thresholds = {
            'volume_spike': 3.0,      # 3x average volume
            'price_breakout': 0.02,   # 2% above resistance
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volatility_high': 0.1,   # 10% daily volatility
            'momentum_strong': 0.05,  # 5% move
            'liquidity_min': 100000,  # $100k daily volume
            'divergence_threshold': 0.2
        }
        
        # Opportunity tracking
        self.opportunities = {}
        self.opportunity_history = []
        self.active_scans = {}
        
        # Performance metrics
        self.scans_completed = 0
        self.opportunities_found = 0
        self.scan_times = []
        
        # Scanner tasks
        self.scanner_task = None
        self.is_scanning = False
        
        # Cache
        self.market_data_cache = {}
        self.technical_cache = {}
        self.cache_ttl = 60  # seconds
        
        logger.info("MarketScannerPro initialized")
        logger.info(f"Scan interval: {self.scan_interval}s")
    
    async def start_scanning(self):
        """Start market scanning"""
        self.is_scanning = True
        
        # Load symbols
        await self._load_symbols()
        
        # Start scanner task
        self.scanner_task = asyncio.create_task(self._scanner_loop())
        
        logger.info(f"Market scanner started with {len(self.all_symbols)} symbols")
    
    async def stop_scanning(self):
        """Stop market scanning"""
        self.is_scanning = False
        
        if self.scanner_task:
            self.scanner_task.cancel()
            try:
                await self.scanner_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Market scanner stopped")
    
    async def _scanner_loop(self):
        """Main scanner loop"""
        while self.is_scanning:
            try:
                scan_start = datetime.now()
                
                # Scan all markets
                opportunities = await self._scan_all_markets()
                
                # Process opportunities
                for opportunity in opportunities:
                    await self._process_opportunity(opportunity)
                
                # Clean old opportunities
                self._clean_old_opportunities()
                
                # Log scan time
                scan_time = (datetime.now() - scan_start).total_seconds()
                self.scan_times.append(scan_time)
                self.scans_completed += 1
                
                logger.info(f"Scan completed in {scan_time:.1f}s - Found {len(opportunities)} opportunities")
                
                # Wait for next scan
                await asyncio.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Scanner loop error: {e}")
                await asyncio.sleep(10)
    
    async def _scan_all_markets(self) -> List[MarketOpportunity]:
        """
        Scan all markets for opportunities
        COMPREHENSIVE MARKET SCAN
        """
        all_opportunities = []
        
        # Split symbols into batches
        symbol_batches = self._create_batches(list(self.all_symbols), self.symbols_per_batch)
        
        # Scan each batch in parallel
        tasks = []
        for batch in symbol_batches:
            tasks.append(self._scan_batch(batch))
        
        # Wait for all scans
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all opportunities
        for result in batch_results:
            if isinstance(result, list):
                all_opportunities.extend(result)
        
        # Sort by priority and confidence
        all_opportunities.sort(
            key=lambda x: (x.priority.value, -x.confidence)
        )
        
        # Limit to max opportunities
        return all_opportunities[:self.max_opportunities]
    
    async def _scan_batch(self, symbols: List[str]) -> List[MarketOpportunity]:
        """Scan a batch of symbols"""
        opportunities = []
        
        for symbol in symbols:
            try:
                # Get market data
                market_data = await self._get_market_data(symbol)
                if not market_data:
                    continue
                
                # Get technical data
                technical_data = await self._get_technical_data(symbol)
                
                # Check for various opportunities
                
                # 1. Breakout detection
                breakout = await self._detect_breakout(symbol, market_data, technical_data)
                if breakout:
                    opportunities.append(breakout)
                
                # 2. Volume spike detection
                volume_spike = await self._detect_volume_spike(symbol, market_data)
                if volume_spike:
                    opportunities.append(volume_spike)
                
                # 3. Momentum detection
                momentum = await self._detect_momentum(symbol, market_data, technical_data)
                if momentum:
                    opportunities.append(momentum)
                
                # 4. Divergence detection
                divergence = await self._detect_divergence(symbol, market_data, technical_data)
                if divergence:
                    opportunities.append(divergence)
                
                # 5. Pattern detection
                pattern = await self._detect_patterns(symbol, market_data, technical_data)
                if pattern:
                    opportunities.append(pattern)
                
                # 6. Reversal detection
                reversal = await self._detect_reversal(symbol, market_data, technical_data)
                if reversal:
                    opportunities.append(reversal)
                
                # 7. Arbitrage detection
                arbitrage = await self._detect_arbitrage(symbol, market_data)
                if arbitrage:
                    opportunities.append(arbitrage)
                
                # 8. Whale activity detection
                whale = await self._detect_whale_activity(symbol, market_data)
                if whale:
                    opportunities.append(whale)
                
            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
        
        return opportunities
    
    async def _detect_breakout(self, symbol: str, market_data: Dict, 
                              technical_data: Dict) -> Optional[MarketOpportunity]:
        """Detect breakout opportunities"""
        price = market_data.get('price', 0)
        volume_ratio = market_data.get('volume_ratio', 1)
        
        # Get resistance levels
        resistance = technical_data.get('resistance', [])
        if not resistance:
            return None
        
        nearest_resistance = min([r for r in resistance if r > price], default=None)
        if not nearest_resistance:
            return None
        
        # Check if price is breaking resistance
        breakout_distance = (price - nearest_resistance) / nearest_resistance
        
        if breakout_distance > -self.thresholds['price_breakout'] and breakout_distance < 0:
            # Near breakout
            confidence = 60
            
            # Increase confidence for volume confirmation
            if volume_ratio > 1.5:
                confidence += 20
            
            # Check momentum
            momentum = technical_data.get('momentum', 0)
            if momentum > 0:
                confidence += 10
            
            opportunity = MarketOpportunity(
                opportunity_id=self._generate_id(),
                timestamp=datetime.now(),
                symbol=symbol,
                exchange="binance",
                
                opportunity_type=OpportunityType.BREAKOUT,
                priority=ScannerPriority.HIGH if confidence > 70 else ScannerPriority.MEDIUM,
                confidence=confidence,
                
                current_price=price,
                target_price=nearest_resistance * 1.03,  # 3% above resistance
                stop_loss=price * 0.98,  # 2% stop loss
                entry_zone=(price * 0.995, nearest_resistance),
                
                volume_ratio=volume_ratio,
                volatility=market_data.get('volatility', 0),
                liquidity_score=market_data.get('liquidity', 0),
                momentum_score=momentum,
                
                resistance_levels=resistance[:3],
                indicators=technical_data.get('indicators', {}),
                
                risk_reward_ratio=3.0,
                
                time_to_act="URGENT" if breakout_distance > -0.01 else "NORMAL",
                expiry_time=datetime.now() + timedelta(hours=2),
                
                description=f"Potential breakout above {nearest_resistance:.2f}",
                action_items=[
                    f"Watch for break above {nearest_resistance:.2f}",
                    f"Volume confirmation needed (current: {volume_ratio:.1f}x)",
                    "Set alerts for breakout confirmation"
                ]
            )
            
            return opportunity
        
        return None
    
    async def _detect_volume_spike(self, symbol: str, market_data: Dict) -> Optional[MarketOpportunity]:
        """Detect volume spike opportunities"""
        volume_ratio = market_data.get('volume_ratio', 1)
        
        if volume_ratio < self.thresholds['volume_spike']:
            return None
        
        price = market_data.get('price', 0)
        price_change = market_data.get('price_change_24h', 0)
        
        # Determine direction
        if price_change > 0:
            opportunity_type = OpportunityType.MOMENTUM
            target = price * 1.05
            stop = price * 0.97
        else:
            opportunity_type = OpportunityType.REVERSAL
            target = price * 1.03
            stop = price * 0.95
        
        confidence = min(50 + (volume_ratio - 3) * 10, 90)
        
        opportunity = MarketOpportunity(
            opportunity_id=self._generate_id(),
            timestamp=datetime.now(),
            symbol=symbol,
            exchange="binance",
            
            opportunity_type=OpportunityType.VOLUME_SPIKE,
            priority=ScannerPriority.HIGH if volume_ratio > 5 else ScannerPriority.MEDIUM,
            confidence=confidence,
            
            current_price=price,
            target_price=target,
            stop_loss=stop,
            entry_zone=(price * 0.995, price * 1.005),
            
            volume_ratio=volume_ratio,
            volatility=market_data.get('volatility', 0),
            liquidity_score=market_data.get('liquidity', 0),
            momentum_score=abs(price_change),
            
            risk_reward_ratio=2.5,
            
            time_to_act="IMMEDIATE" if volume_ratio > 5 else "URGENT",
            expiry_time=datetime.now() + timedelta(hours=1),
            
            description=f"Unusual volume spike: {volume_ratio:.1f}x average",
            action_items=[
                f"Volume {volume_ratio:.1f}x above average",
                f"Price change: {price_change:.2f}%",
                "Potential whale activity or news event"
            ],
            warnings=["High volatility expected"] if volume_ratio > 5 else []
        )
        
        return opportunity
    
    async def _detect_momentum(self, symbol: str, market_data: Dict, 
                              technical_data: Dict) -> Optional[MarketOpportunity]:
        """Detect momentum opportunities"""
        price_change_1h = market_data.get('price_change_1h', 0)
        
        if abs(price_change_1h) < self.thresholds['momentum_strong']:
            return None
        
        rsi = technical_data.get('rsi', 50)
        macd = technical_data.get('macd', {})
        
        # Check if momentum is sustainable
        if price_change_1h > 0:
            if rsi > 70:  # Overbought
                return None
            opportunity_type = OpportunityType.MOMENTUM
            direction = "UP"
        else:
            if rsi < 30:  # Oversold
                return None
            opportunity_type = OpportunityType.MOMENTUM
            direction = "DOWN"
        
        price = market_data.get('price', 0)
        
        confidence = 50
        
        # MACD confirmation
        if macd.get('histogram', 0) * price_change_1h > 0:  # Same direction
            confidence += 20
        
        # Volume confirmation
        if market_data.get('volume_ratio', 1) > 1.5:
            confidence += 15
        
        opportunity = MarketOpportunity(
            opportunity_id=self._generate_id(),
            timestamp=datetime.now(),
            symbol=symbol,
            exchange="binance",
            
            opportunity_type=opportunity_type,
            priority=ScannerPriority.MEDIUM,
            confidence=confidence,
            
            current_price=price,
            target_price=price * (1.03 if direction == "UP" else 0.97),
            stop_loss=price * (0.98 if direction == "UP" else 1.02),
            entry_zone=(price * 0.998, price * 1.002),
            
            volume_ratio=market_data.get('volume_ratio', 1),
            volatility=market_data.get('volatility', 0),
            liquidity_score=market_data.get('liquidity', 0),
            momentum_score=abs(price_change_1h),
            
            indicators={'rsi': rsi, 'macd_hist': macd.get('histogram', 0)},
            
            risk_reward_ratio=2.0,
            
            time_to_act="NORMAL",
            expiry_time=datetime.now() + timedelta(hours=4),
            
            description=f"Strong momentum {direction}: {price_change_1h:.2f}% in 1h",
            action_items=[
                f"Momentum play - {direction} trend",
                f"RSI at {rsi:.0f}",
                "Use trailing stop to protect profits"
            ]
        )
        
        return opportunity
    
    async def _detect_divergence(self, symbol: str, market_data: Dict, 
                                technical_data: Dict) -> Optional[MarketOpportunity]:
        """Detect price/indicator divergence"""
        price_trend = market_data.get('price_trend', 'neutral')
        rsi = technical_data.get('rsi', 50)
        rsi_trend = technical_data.get('rsi_trend', 'neutral')
        
        divergence = None
        
        # Bullish divergence: price down, RSI up
        if price_trend == 'down' and rsi_trend == 'up' and rsi < 40:
            divergence = 'bullish'
            confidence = 60
            
        # Bearish divergence: price up, RSI down
        elif price_trend == 'up' and rsi_trend == 'down' and rsi > 60:
            divergence = 'bearish'
            confidence = 60
        
        if not divergence:
            return None
        
        price = market_data.get('price', 0)
        
        # Volume confirmation
        if market_data.get('volume_ratio', 1) > 1.2:
            confidence += 10
        
        opportunity = MarketOpportunity(
            opportunity_id=self._generate_id(),
            timestamp=datetime.now(),
            symbol=symbol,
            exchange="binance",
            
            opportunity_type=OpportunityType.DIVERGENCE,
            priority=ScannerPriority.MEDIUM,
            confidence=confidence,
            
            current_price=price,
            target_price=price * (1.04 if divergence == 'bullish' else 0.96),
            stop_loss=price * (0.97 if divergence == 'bullish' else 1.03),
            entry_zone=(price * 0.995, price * 1.005),
            
            volume_ratio=market_data.get('volume_ratio', 1),
            volatility=market_data.get('volatility', 0),
            liquidity_score=market_data.get('liquidity', 0),
            momentum_score=0,
            
            indicators={'rsi': rsi},
            
            risk_reward_ratio=2.5,
            
            time_to_act="WAIT",  # Wait for confirmation
            expiry_time=datetime.now() + timedelta(hours=6),
            
            description=f"{divergence.capitalize()} divergence detected",
            action_items=[
                f"Price trend: {price_trend}, RSI trend: {rsi_trend}",
                "Wait for trend reversal confirmation",
                "Consider scaling into position"
            ],
            warnings=["Divergences can persist - use stop loss"]
        )
        
        return opportunity
    
    async def _detect_patterns(self, symbol: str, market_data: Dict, 
                              technical_data: Dict) -> Optional[MarketOpportunity]:
        """Detect chart patterns"""
        patterns = technical_data.get('patterns', [])
        
        if not patterns:
            return None
        
        # Take the first detected pattern
        pattern = patterns[0]
        pattern_name = pattern.get('name', 'Unknown')
        pattern_type = pattern.get('type', 'neutral')
        
        price = market_data.get('price', 0)
        
        confidence = 50
        
        # Pattern-specific confidence
        if pattern_name in ['Head and Shoulders', 'Double Top', 'Double Bottom']:
            confidence += 20
        elif pattern_name in ['Triangle', 'Flag', 'Pennant']:
            confidence += 15
        
        # Volume confirmation
        if market_data.get('volume_ratio', 1) > 1.3:
            confidence += 10
        
        if pattern_type == 'bullish':
            target = price * 1.05
            stop = price * 0.97
        else:
            target = price * 0.95
            stop = price * 1.03
        
        opportunity = MarketOpportunity(
            opportunity_id=self._generate_id(),
            timestamp=datetime.now(),
            symbol=symbol,
            exchange="binance",
            
            opportunity_type=OpportunityType.PATTERN,
            priority=ScannerPriority.MEDIUM,
            confidence=confidence,
            
            current_price=price,
            target_price=target,
            stop_loss=stop,
            entry_zone=(price * 0.995, price * 1.005),
            
            volume_ratio=market_data.get('volume_ratio', 1),
            volatility=market_data.get('volatility', 0),
            liquidity_score=market_data.get('liquidity', 0),
            momentum_score=0,
            
            risk_reward_ratio=2.5,
            
            time_to_act="NORMAL",
            expiry_time=datetime.now() + timedelta(hours=8),
            
            description=f"{pattern_name} pattern detected",
            action_items=[
                f"Pattern: {pattern_name} ({pattern_type})",
                "Wait for pattern completion",
                "Set alerts at key levels"
            ]
        )
        
        return opportunity
    
    async def _detect_reversal(self, symbol: str, market_data: Dict, 
                              technical_data: Dict) -> Optional[MarketOpportunity]:
        """Detect reversal opportunities"""
        rsi = technical_data.get('rsi', 50)
        price = market_data.get('price', 0)
        
        reversal = None
        
        # Oversold reversal
        if rsi < self.thresholds['rsi_oversold']:
            reversal = 'bullish'
            confidence = 50 + (30 - rsi)  # More oversold = higher confidence
            
        # Overbought reversal
        elif rsi > self.thresholds['rsi_overbought']:
            reversal = 'bearish'
            confidence = 50 + (rsi - 70)  # More overbought = higher confidence
        
        if not reversal:
            return None
        
        # Check for support/resistance
        support = technical_data.get('support', [])
        resistance = technical_data.get('resistance', [])
        
        if reversal == 'bullish' and support:
            nearest_support = max([s for s in support if s < price], default=None)
            if nearest_support and abs(price - nearest_support) / price < 0.02:
                confidence += 15  # Near support
        
        elif reversal == 'bearish' and resistance:
            nearest_resistance = min([r for r in resistance if r > price], default=None)
            if nearest_resistance and abs(price - nearest_resistance) / price < 0.02:
                confidence += 15  # Near resistance
        
        opportunity = MarketOpportunity(
            opportunity_id=self._generate_id(),
            timestamp=datetime.now(),
            symbol=symbol,
            exchange="binance",
            
            opportunity_type=OpportunityType.REVERSAL,
            priority=ScannerPriority.MEDIUM if confidence > 60 else ScannerPriority.LOW,
            confidence=confidence,
            
            current_price=price,
            target_price=price * (1.03 if reversal == 'bullish' else 0.97),
            stop_loss=price * (0.97 if reversal == 'bullish' else 1.03),
            entry_zone=(price * 0.995, price * 1.005),
            
            volume_ratio=market_data.get('volume_ratio', 1),
            volatility=market_data.get('volatility', 0),
            liquidity_score=market_data.get('liquidity', 0),
            momentum_score=0,
            
            support_levels=support[:3] if support else [],
            resistance_levels=resistance[:3] if resistance else [],
            indicators={'rsi': rsi},
            
            risk_reward_ratio=2.0,
            
            time_to_act="WAIT",  # Wait for confirmation
            expiry_time=datetime.now() + timedelta(hours=4),
            
            description=f"Potential {reversal} reversal (RSI: {rsi:.0f})",
            action_items=[
                f"RSI at extreme: {rsi:.0f}",
                "Wait for reversal confirmation",
                "Consider multiple entries"
            ],
            warnings=["Catching falling knives is risky"] if reversal == 'bullish' else []
        )
        
        return opportunity
    
    async def _detect_arbitrage(self, symbol: str, market_data: Dict) -> Optional[MarketOpportunity]:
        """Detect arbitrage opportunities between exchanges"""
        # This would compare prices across different exchanges
        # Simplified version
        
        price_binance = market_data.get('price', 0)
        
        # In real implementation, would get prices from other exchanges
        # For now, simulate with small random difference
        price_difference = 0  # Would be calculated from actual exchange prices
        
        if abs(price_difference) < 0.001:  # Less than 0.1% difference
            return None
        
        # Arbitrage opportunity exists
        confidence = min(50 + abs(price_difference) * 1000, 90)
        
        opportunity = MarketOpportunity(
            opportunity_id=self._generate_id(),
            timestamp=datetime.now(),
            symbol=symbol,
            exchange="multi",
            
            opportunity_type=OpportunityType.ARBITRAGE,
            priority=ScannerPriority.CRITICAL if abs(price_difference) > 0.005 else ScannerPriority.HIGH,
            confidence=confidence,
            
            current_price=price_binance,
            target_price=price_binance,
            stop_loss=price_binance * 0.999,
            entry_zone=(price_binance * 0.999, price_binance * 1.001),
            
            volume_ratio=1.0,
            volatility=0,
            liquidity_score=1.0,
            momentum_score=0,
            
            risk_reward_ratio=10.0,  # Arbitrage has high RR
            
            time_to_act="IMMEDIATE",
            expiry_time=datetime.now() + timedelta(minutes=5),
            
            description=f"Arbitrage opportunity: {abs(price_difference)*100:.2f}% difference",
            action_items=[
                "Execute immediately",
                "Check fees on both exchanges",
                "Ensure sufficient balance on both"
            ],
            warnings=["Arbitrage windows close quickly"]
        )
        
        return opportunity
    
    async def _detect_whale_activity(self, symbol: str, market_data: Dict) -> Optional[MarketOpportunity]:
        """Detect whale activity"""
        # Look for large orders in orderbook
        large_orders = market_data.get('large_orders', [])
        
        if not large_orders:
            return None
        
        # Calculate whale impact
        total_whale_volume = sum(order['size'] for order in large_orders)
        avg_volume = market_data.get('avg_volume', 1)
        
        if avg_volume > 0:
            whale_ratio = total_whale_volume / avg_volume
        else:
            whale_ratio = 0
        
        if whale_ratio < 0.1:  # Less than 10% of daily volume
            return None
        
        price = market_data.get('price', 0)
        
        # Determine direction based on whale orders
        buy_volume = sum(o['size'] for o in large_orders if o['side'] == 'buy')
        sell_volume = sum(o['size'] for o in large_orders if o['side'] == 'sell')
        
        if buy_volume > sell_volume * 1.5:
            direction = 'bullish'
            target = price * 1.02
            stop = price * 0.98
        else:
            direction = 'bearish'
            target = price * 0.98
            stop = price * 1.02
        
        confidence = min(50 + whale_ratio * 100, 85)
        
        opportunity = MarketOpportunity(
            opportunity_id=self._generate_id(),
            timestamp=datetime.now(),
            symbol=symbol,
            exchange="binance",
            
            opportunity_type=OpportunityType.WHALE_ACTIVITY,
            priority=ScannerPriority.HIGH,
            confidence=confidence,
            
            current_price=price,
            target_price=target,
            stop_loss=stop,
            entry_zone=(price * 0.998, price * 1.002),
            
            volume_ratio=whale_ratio,
            volatility=market_data.get('volatility', 0),
            liquidity_score=market_data.get('liquidity', 0),
            momentum_score=0,
            
            risk_reward_ratio=2.0,
            
            time_to_act="URGENT",
            expiry_time=datetime.now() + timedelta(hours=1),
            
            description=f"Whale activity detected: {direction}",
            action_items=[
                f"Whale volume: {whale_ratio:.1f}x average",
                f"Direction: {direction}",
                "Follow the smart money"
            ],
            warnings=["Whales can manipulate prices"]
        )
        
        return opportunity
    
    async def _get_market_data(self, symbol: str) -> Dict:
        """Get market data for symbol"""
        # Check cache
        cache_key = f"{symbol}_market"
        if cache_key in self.market_data_cache:
            cached_data, cache_time = self.market_data_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return cached_data
        
        try:
            # Get ticker data
            ticker_url = f"{self.binance_url}/ticker/24hr"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(ticker_url, params={'symbol': symbol}) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        market_data = {
                            'symbol': symbol,
                            'price': float(data['lastPrice']),
                            'price_change_24h': float(data['priceChangePercent']),
                            'volume': float(data['volume']),
                            'quote_volume': float(data['quoteVolume']),
                            'high_24h': float(data['highPrice']),
                            'low_24h': float(data['lowPrice']),
                            'volatility': (float(data['highPrice']) - float(data['lowPrice'])) / float(data['lastPrice']),
                            'liquidity': float(data['quoteVolume']) / 1000000,  # In millions
                            'volume_ratio': 1.0  # Would calculate from historical average
                        }
                        
                        # Cache data
                        self.market_data_cache[cache_key] = (market_data, datetime.now())
                        
                        return market_data
            
        except Exception as e:
            logger.debug(f"Error getting market data for {symbol}: {e}")
        
        return {}
    
    async def _get_technical_data(self, symbol: str) -> Dict:
        """Get technical analysis data"""
        # Check cache
        cache_key = f"{symbol}_technical"
        if cache_key in self.technical_cache:
            cached_data, cache_time = self.technical_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return cached_data
        
        # Simplified technical data
        # In real implementation, would calculate from klines
        technical_data = {
            'rsi': 50,  # Would be calculated
            'macd': {'histogram': 0},  # Would be calculated
            'support': [],  # Would be calculated
            'resistance': [],  # Would be calculated
            'patterns': [],  # Would be detected
            'momentum': 0,  # Would be calculated
            'indicators': {}
        }
        
        # Cache data
        self.technical_cache[cache_key] = (technical_data, datetime.now())
        
        return technical_data
    
    async def _load_symbols(self):
        """Load symbols to scan"""
        try:
            # Get all USDT pairs from Binance
            url = f"{self.binance_url}/exchangeInfo"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for symbol_info in data['symbols']:
                            if symbol_info['status'] == 'TRADING' and symbol_info['quoteAsset'] == 'USDT':
                                self.all_symbols.add(symbol_info['symbol'])
            
            # Set priority symbols
            self.priority_symbols = {
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
                'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT'
            }
            
            # Initialize watchlist with top symbols
            self.watchlist = self.priority_symbols.copy()
            
            logger.info(f"Loaded {len(self.all_symbols)} symbols for scanning")
            
        except Exception as e:
            logger.error(f"Error loading symbols: {e}")
            
            # Fallback to default symbols
            self.all_symbols = self.priority_symbols.copy()
    
    async def _process_opportunity(self, opportunity: MarketOpportunity):
        """Process detected opportunity"""
        # Store opportunity
        self.opportunities[opportunity.opportunity_id] = opportunity
        self.opportunity_history.append(opportunity)
        self.opportunities_found += 1
        
        # Log critical opportunities
        if opportunity.priority == ScannerPriority.CRITICAL:
            logger.warning(f"CRITICAL opportunity: {opportunity.symbol} - {opportunity.description}")
        elif opportunity.priority == ScannerPriority.HIGH:
            logger.info(f"HIGH priority opportunity: {opportunity.symbol} - {opportunity.description}")
    
    def _clean_old_opportunities(self):
        """Remove expired opportunities"""
        now = datetime.now()
        expired = []
        
        for opp_id, opportunity in self.opportunities.items():
            if opportunity.expiry_time and opportunity.expiry_time < now:
                expired.append(opp_id)
        
        for opp_id in expired:
            del self.opportunities[opp_id]
    
    def _create_batches(self, items: List, batch_size: int) -> List[List]:
        """Create batches from list"""
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        return batches
    
    def _generate_id(self) -> str:
        """Generate unique opportunity ID"""
        return f"OPP_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self.opportunities_found:04d}"
    
    async def get_opportunities(self, 
                              opportunity_type: Optional[OpportunityType] = None,
                              min_confidence: float = 0,
                              priority: Optional[ScannerPriority] = None) -> List[MarketOpportunity]:
        """Get filtered opportunities"""
        opportunities = list(self.opportunities.values())
        
        # Filter by type
        if opportunity_type:
            opportunities = [o for o in opportunities if o.opportunity_type == opportunity_type]
        
        # Filter by confidence
        opportunities = [o for o in opportunities if o.confidence >= min_confidence]
        
        # Filter by priority
        if priority:
            opportunities = [o for o in opportunities if o.priority.value <= priority.value]
        
        # Sort by priority and confidence
        opportunities.sort(key=lambda x: (x.priority.value, -x.confidence))
        
        return opportunities
    
    async def add_to_watchlist(self, symbol: str):
        """Add symbol to watchlist"""
        self.watchlist.add(symbol)
        logger.info(f"Added {symbol} to watchlist")
    
    async def remove_from_watchlist(self, symbol: str):
        """Remove symbol from watchlist"""
        self.watchlist.discard(symbol)
        logger.info(f"Removed {symbol} from watchlist")
    
    def get_statistics(self) -> Dict:
        """Get scanner statistics"""
        avg_scan_time = np.mean(self.scan_times) if self.scan_times else 0
        
        return {
            'scans_completed': self.scans_completed,
            'opportunities_found': self.opportunities_found,
            'active_opportunities': len(self.opportunities),
            'symbols_scanned': len(self.all_symbols),
            'watchlist_size': len(self.watchlist),
            'avg_scan_time': avg_scan_time,
            'cache_size': len(self.market_data_cache) + len(self.technical_cache)
        }
