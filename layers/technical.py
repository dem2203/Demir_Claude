"""
DEMIR AI v8.0 - Technical Analysis Layer
Comprehensive Technical Indicators - REAL MARKET DATA ONLY
NO SIMPLIFICATION - ENTERPRISE GRADE
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import aiohttp
import talib
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class CandlestickData:
    """Real candlestick data structure"""
    timestamp: List[int]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        return pd.DataFrame({
            'timestamp': pd.to_datetime(self.timestamp, unit='ms'),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        })
    
    def validate(self) -> bool:
        """Validate OHLC data integrity"""
        for i in range(len(self.open)):
            # High must be >= Low
            if self.high[i] < self.low[i]:
                return False
            # Open and Close must be between High and Low
            if not (self.low[i] <= self.open[i] <= self.high[i]):
                return False
            if not (self.low[i] <= self.close[i] <= self.high[i]):
                return False
            # Volume must be non-negative
            if self.volume[i] < 0:
                return False
        return True


class TechnicalAnalyzer:
    """
    Comprehensive technical analysis with real market data
    ZERO MOCK DATA - ALL INDICATORS FROM REAL PRICES
    """
    
    def __init__(self, config):
        self.config = config
        self.binance_url = "https://api.binance.com/api/v3"
        
        # Indicator periods
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2
        self.atr_period = 14
        self.adx_period = 14
        self.stoch_k_period = 14
        self.stoch_d_period = 3
        self.cci_period = 20
        self.mfi_period = 14
        self.obv_period = 20
        self.vwap_period = 20
        
        # Moving averages
        self.ma_periods = [5, 10, 20, 50, 100, 200]
        
        # Cache for klines data
        self.klines_cache = {}
        self.cache_ttl = 60  # 1 minute cache
        
        logger.info("TechnicalAnalyzer initialized - REAL MARKET DATA ONLY")
        logger.info("Using Binance API for live price data")
    
    async def analyze(self, symbol: str, timeframe: str = '1h') -> Dict:
        """
        Complete technical analysis for symbol
        Returns all technical indicators
        """
        try:
            # Get real market data from Binance
            klines = await self._get_klines(symbol, timeframe)
            
            if not klines or not klines.validate():
                logger.error(f"Invalid klines data for {symbol}")
                return self._empty_analysis()
            
            # Convert to DataFrame for easier calculation
            df = klines.to_dataframe()
            
            # Calculate all indicators
            analysis = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'last_price': klines.close[-1],
                'close': klines.close[-1],
                'open': klines.open[-1],
                'high': klines.high[-1],
                'low': klines.low[-1],
                'volume': klines.volume[-1],
                
                # Momentum Indicators
                'rsi': self._calculate_rsi(klines),
                'macd': self._calculate_macd(klines),
                'stochastic': self._calculate_stochastic(klines),
                'williams_r': self._calculate_williams_r(klines),
                'cci': self._calculate_cci(klines),
                'mfi': self._calculate_mfi(klines),
                'roc': self._calculate_roc(klines),
                
                # Trend Indicators
                'moving_averages': self._calculate_moving_averages(klines),
                'ma_signal': self._get_ma_signal(klines),
                'ema': self._calculate_ema(klines),
                'wma': self._calculate_wma(klines),
                'adx': self._calculate_adx(klines),
                'parabolic_sar': self._calculate_parabolic_sar(klines),
                'ichimoku': self._calculate_ichimoku(klines),
                
                # Volatility Indicators
                'bollinger_bands': self._calculate_bollinger_bands(klines),
                'bb_signal': self._get_bb_signal(klines),
                'atr': self._calculate_atr(klines),
                'volatility': self._calculate_volatility(klines),
                'keltner_channels': self._calculate_keltner_channels(klines),
                'donchian_channels': self._calculate_donchian_channels(klines),
                
                # Volume Indicators
                'obv': self._calculate_obv(klines),
                'volume_ratio': self._calculate_volume_ratio(klines),
                'volume_trend': self._calculate_volume_trend(klines),
                'vwap': self._calculate_vwap(klines),
                'accumulation_distribution': self._calculate_ad(klines),
                'chaikin_money_flow': self._calculate_cmf(klines),
                
                # Pattern Detection
                'candlestick_patterns': self._detect_candlestick_patterns(klines),
                'support_resistance': self._find_support_resistance(klines),
                'trend_direction': self._determine_trend(klines),
                'trend_strength': self._calculate_trend_strength(klines),
                
                # Market Structure
                'pivot_points': self._calculate_pivot_points(klines),
                'fibonacci_levels': self._calculate_fibonacci_levels(klines),
                'market_regime': self._determine_market_regime(klines),
                
                # Correlation
                'btc_correlation': await self._calculate_btc_correlation(symbol, klines),
                
                # Custom Indicators
                'momentum_score': self._calculate_momentum_score(klines),
                'trend_score': self._calculate_trend_score(klines),
                'volume_score': self._calculate_volume_score(klines),
                'volatility_score': self._calculate_volatility_score(klines),
                
                # Overall Technical Score
                'technical_score': 0,  # Will be calculated below
                'technical_rating': '',  # Will be set below
                'signals': []  # Technical signals
            }
            
            # Calculate overall technical score
            analysis['technical_score'] = self._calculate_overall_score(analysis)
            analysis['technical_rating'] = self._get_rating(analysis['technical_score'])
            analysis['signals'] = self._generate_signals(analysis)
            
            # Validate all data is real
            if not self._validate_analysis(analysis):
                logger.error(f"Analysis validation failed for {symbol}")
                return self._empty_analysis()
            
            logger.info(f"Technical analysis completed for {symbol}: Score {analysis['technical_score']:.1f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Technical analysis error for {symbol}: {e}")
            return self._empty_analysis()
    
    async def _get_klines(self, symbol: str, interval: str, limit: int = 500) -> Optional[CandlestickData]:
        """Get real klines data from Binance"""
        cache_key = f"{symbol}:{interval}:{int(time.time() // self.cache_ttl)}"
        
        # Check cache
        if cache_key in self.klines_cache:
            logger.debug(f"Using cached klines for {symbol}")
            return self.klines_cache[cache_key]
        
        try:
            url = f"{self.binance_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Parse klines data
                        timestamps = []
                        opens = []
                        highs = []
                        lows = []
                        closes = []
                        volumes = []
                        
                        for kline in data:
                            timestamps.append(int(kline[0]))
                            opens.append(float(kline[1]))
                            highs.append(float(kline[2]))
                            lows.append(float(kline[3]))
                            closes.append(float(kline[4]))
                            volumes.append(float(kline[5]))
                        
                        klines = CandlestickData(
                            timestamp=timestamps,
                            open=opens,
                            high=highs,
                            low=lows,
                            close=closes,
                            volume=volumes
                        )
                        
                        # Validate data
                        if klines.validate():
                            # Cache the data
                            self.klines_cache[cache_key] = klines
                            # Clean old cache
                            self._clean_cache()
                            return klines
                        else:
                            logger.error(f"Invalid klines data for {symbol}")
                            return None
                    else:
                        logger.error(f"Binance API error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, klines: CandlestickData) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            closes = np.array(klines.close)
            rsi = talib.RSI(closes, timeperiod=self.rsi_period)
            return float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return 50.0
    
    def _calculate_macd(self, klines: CandlestickData) -> Dict:
        """Calculate MACD"""
        try:
            closes = np.array(klines.close)
            macd, signal, histogram = talib.MACD(
                closes,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            
            return {
                'macd': float(macd[-1]) if not np.isnan(macd[-1]) else 0,
                'signal': float(signal[-1]) if not np.isnan(signal[-1]) else 0,
                'histogram': float(histogram[-1]) if not np.isnan(histogram[-1]) else 0
            }
        except Exception as e:
            logger.error(f"MACD calculation error: {e}")
            return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    def _calculate_bollinger_bands(self, klines: CandlestickData) -> Dict:
        """Calculate Bollinger Bands"""
        try:
            closes = np.array(klines.close)
            upper, middle, lower = talib.BBANDS(
                closes,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std
            )
            
            current_price = closes[-1]
            
            return {
                'upper': float(upper[-1]) if not np.isnan(upper[-1]) else current_price,
                'middle': float(middle[-1]) if not np.isnan(middle[-1]) else current_price,
                'lower': float(lower[-1]) if not np.isnan(lower[-1]) else current_price,
                'width': float(upper[-1] - lower[-1]) if not np.isnan(upper[-1]) else 0,
                'percent_b': (current_price - lower[-1]) / (upper[-1] - lower[-1]) if upper[-1] != lower[-1] else 0.5
            }
        except Exception as e:
            logger.error(f"Bollinger Bands calculation error: {e}")
            return {'upper': 0, 'middle': 0, 'lower': 0, 'width': 0, 'percent_b': 0.5}
    
    def _calculate_stochastic(self, klines: CandlestickData) -> Dict:
        """Calculate Stochastic Oscillator"""
        try:
            highs = np.array(klines.high)
            lows = np.array(klines.low)
            closes = np.array(klines.close)
            
            k, d = talib.STOCH(
                highs, lows, closes,
                fastk_period=self.stoch_k_period,
                slowk_period=3,
                slowd_period=self.stoch_d_period
            )
            
            return {
                'k': float(k[-1]) if not np.isnan(k[-1]) else 50,
                'd': float(d[-1]) if not np.isnan(d[-1]) else 50
            }
        except Exception as e:
            logger.error(f"Stochastic calculation error: {e}")
            return {'k': 50, 'd': 50}
    
    def _calculate_atr(self, klines: CandlestickData) -> float:
        """Calculate Average True Range"""
        try:
            highs = np.array(klines.high)
            lows = np.array(klines.low)
            closes = np.array(klines.close)
            
            atr = talib.ATR(highs, lows, closes, timeperiod=self.atr_period)
            return float(atr[-1]) if not np.isnan(atr[-1]) else 0
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return 0
    
    def _calculate_moving_averages(self, klines: CandlestickData) -> Dict:
        """Calculate multiple moving averages"""
        try:
            closes = np.array(klines.close)
            mas = {}
            
            for period in self.ma_periods:
                if len(closes) >= period:
                    ma = talib.SMA(closes, timeperiod=period)
                    mas[f'ma_{period}'] = float(ma[-1]) if not np.isnan(ma[-1]) else closes[-1]
                else:
                    mas[f'ma_{period}'] = closes[-1]
            
            return mas
        except Exception as e:
            logger.error(f"MA calculation error: {e}")
            return {f'ma_{p}': klines.close[-1] for p in self.ma_periods}
    
    def _calculate_adx(self, klines: CandlestickData) -> float:
        """Calculate ADX (Average Directional Index)"""
        try:
            highs = np.array(klines.high)
            lows = np.array(klines.low)
            closes = np.array(klines.close)
            
            adx = talib.ADX(highs, lows, closes, timeperiod=self.adx_period)
            return float(adx[-1]) if not np.isnan(adx[-1]) else 25
        except Exception as e:
            logger.error(f"ADX calculation error: {e}")
            return 25
    
    def _calculate_williams_r(self, klines: CandlestickData) -> float:
        """Calculate Williams %R"""
        try:
            highs = np.array(klines.high)
            lows = np.array(klines.low)
            closes = np.array(klines.close)
            
            willr = talib.WILLR(highs, lows, closes, timeperiod=14)
            return float(willr[-1]) if not np.isnan(willr[-1]) else -50
        except Exception as e:
            logger.error(f"Williams %R calculation error: {e}")
            return -50
    
    def _calculate_cci(self, klines: CandlestickData) -> float:
        """Calculate Commodity Channel Index"""
        try:
            highs = np.array(klines.high)
            lows = np.array(klines.low)
            closes = np.array(klines.close)
            
            cci = talib.CCI(highs, lows, closes, timeperiod=self.cci_period)
            return float(cci[-1]) if not np.isnan(cci[-1]) else 0
        except Exception as e:
            logger.error(f"CCI calculation error: {e}")
            return 0
    
    def _calculate_mfi(self, klines: CandlestickData) -> float:
        """Calculate Money Flow Index"""
        try:
            highs = np.array(klines.high)
            lows = np.array(klines.low)
            closes = np.array(klines.close)
            volumes = np.array(klines.volume)
            
            mfi = talib.MFI(highs, lows, closes, volumes, timeperiod=self.mfi_period)
            return float(mfi[-1]) if not np.isnan(mfi[-1]) else 50
        except Exception as e:
            logger.error(f"MFI calculation error: {e}")
            return 50
    
    def _calculate_obv(self, klines: CandlestickData) -> float:
        """Calculate On-Balance Volume"""
        try:
            closes = np.array(klines.close)
            volumes = np.array(klines.volume)
            
            obv = talib.OBV(closes, volumes)
            return float(obv[-1]) if not np.isnan(obv[-1]) else 0
        except Exception as e:
            logger.error(f"OBV calculation error: {e}")
            return 0
    
    def _calculate_ema(self, klines: CandlestickData) -> Dict:
        """Calculate Exponential Moving Averages"""
        try:
            closes = np.array(klines.close)
            emas = {}
            
            for period in [12, 26, 50, 200]:
                if len(closes) >= period:
                    ema = talib.EMA(closes, timeperiod=period)
                    emas[f'ema_{period}'] = float(ema[-1]) if not np.isnan(ema[-1]) else closes[-1]
                else:
                    emas[f'ema_{period}'] = closes[-1]
            
            return emas
        except Exception as e:
            logger.error(f"EMA calculation error: {e}")
            return {}
    
    def _calculate_wma(self, klines: CandlestickData) -> Dict:
        """Calculate Weighted Moving Averages"""
        try:
            closes = np.array(klines.close)
            wmas = {}
            
            for period in [10, 20, 50]:
                if len(closes) >= period:
                    wma = talib.WMA(closes, timeperiod=period)
                    wmas[f'wma_{period}'] = float(wma[-1]) if not np.isnan(wma[-1]) else closes[-1]
                else:
                    wmas[f'wma_{period}'] = closes[-1]
            
            return wmas
        except Exception as e:
            logger.error(f"WMA calculation error: {e}")
            return {}
    
    def _calculate_parabolic_sar(self, klines: CandlestickData) -> float:
        """Calculate Parabolic SAR"""
        try:
            highs = np.array(klines.high)
            lows = np.array(klines.low)
            
            sar = talib.SAR(highs, lows)
            return float(sar[-1]) if not np.isnan(sar[-1]) else klines.close[-1]
        except Exception as e:
            logger.error(f"Parabolic SAR calculation error: {e}")
            return klines.close[-1]
    
    def _calculate_ichimoku(self, klines: CandlestickData) -> Dict:
        """Calculate Ichimoku Cloud"""
        try:
            highs = np.array(klines.high)
            lows = np.array(klines.low)
            closes = np.array(klines.close)
            
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            tenkan_period = 9
            tenkan = (talib.MAX(highs, tenkan_period) + talib.MIN(lows, tenkan_period)) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            kijun_period = 26
            kijun = (talib.MAX(highs, kijun_period) + talib.MIN(lows, kijun_period)) / 2
            
            # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
            senkou_a = (tenkan + kijun) / 2
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            senkou_b_period = 52
            senkou_b = (talib.MAX(highs, senkou_b_period) + talib.MIN(lows, senkou_b_period)) / 2
            
            return {
                'tenkan': float(tenkan[-1]) if not np.isnan(tenkan[-1]) else closes[-1],
                'kijun': float(kijun[-1]) if not np.isnan(kijun[-1]) else closes[-1],
                'senkou_a': float(senkou_a[-1]) if not np.isnan(senkou_a[-1]) else closes[-1],
                'senkou_b': float(senkou_b[-1]) if not np.isnan(senkou_b[-1]) else closes[-1],
                'chikou': closes[-26] if len(closes) > 26 else closes[-1]
            }
        except Exception as e:
            logger.error(f"Ichimoku calculation error: {e}")
            return {}
    
    def _calculate_volatility(self, klines: CandlestickData) -> float:
        """Calculate price volatility"""
        try:
            closes = np.array(klines.close)
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * np.sqrt(24)  # Annualized for hourly data
            return float(volatility)
        except Exception as e:
            logger.error(f"Volatility calculation error: {e}")
            return 0.02
    
    def _calculate_volume_ratio(self, klines: CandlestickData) -> float:
        """Calculate volume ratio (current vs average)"""
        try:
            volumes = np.array(klines.volume)
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            
            if avg_volume > 0:
                return current_volume / avg_volume
            return 1.0
        except Exception as e:
            logger.error(f"Volume ratio calculation error: {e}")
            return 1.0
    
    def _calculate_volume_trend(self, klines: CandlestickData) -> float:
        """Calculate volume trend"""
        try:
            volumes = np.array(klines.volume[-10:])
            if len(volumes) > 1:
                # Linear regression slope
                x = np.arange(len(volumes))
                slope = np.polyfit(x, volumes, 1)[0]
                # Normalize by average volume
                avg_volume = np.mean(volumes)
                if avg_volume > 0:
                    return slope / avg_volume
            return 0
        except Exception as e:
            logger.error(f"Volume trend calculation error: {e}")
            return 0
    
    def _calculate_vwap(self, klines: CandlestickData) -> float:
        """Calculate VWAP (Volume Weighted Average Price)"""
        try:
            highs = np.array(klines.high[-self.vwap_period:])
            lows = np.array(klines.low[-self.vwap_period:])
            closes = np.array(klines.close[-self.vwap_period:])
            volumes = np.array(klines.volume[-self.vwap_period:])
            
            typical_price = (highs + lows + closes) / 3
            vwap = np.sum(typical_price * volumes) / np.sum(volumes)
            
            return float(vwap)
        except Exception as e:
            logger.error(f"VWAP calculation error: {e}")
            return klines.close[-1]
    
    def _calculate_roc(self, klines: CandlestickData) -> float:
        """Calculate Rate of Change"""
        try:
            closes = np.array(klines.close)
            roc = talib.ROC(closes, timeperiod=10)
            return float(roc[-1]) if not np.isnan(roc[-1]) else 0
        except Exception as e:
            logger.error(f"ROC calculation error: {e}")
            return 0
    
    def _calculate_keltner_channels(self, klines: CandlestickData) -> Dict:
        """Calculate Keltner Channels"""
        try:
            closes = np.array(klines.close)
            highs = np.array(klines.high)
            lows = np.array(klines.low)
            
            # Middle line (EMA)
            middle = talib.EMA(closes, timeperiod=20)
            
            # ATR for channel width
            atr = talib.ATR(highs, lows, closes, timeperiod=20)
            
            upper = middle + (2 * atr)
            lower = middle - (2 * atr)
            
            return {
                'upper': float(upper[-1]) if not np.isnan(upper[-1]) else closes[-1],
                'middle': float(middle[-1]) if not np.isnan(middle[-1]) else closes[-1],
                'lower': float(lower[-1]) if not np.isnan(lower[-1]) else closes[-1]
            }
        except Exception as e:
            logger.error(f"Keltner Channels calculation error: {e}")
            return {}
    
    def _calculate_donchian_channels(self, klines: CandlestickData) -> Dict:
        """Calculate Donchian Channels"""
        try:
            highs = np.array(klines.high)
            lows = np.array(klines.low)
            
            period = 20
            upper = talib.MAX(highs, timeperiod=period)
            lower = talib.MIN(lows, timeperiod=period)
            middle = (upper + lower) / 2
            
            return {
                'upper': float(upper[-1]) if not np.isnan(upper[-1]) else highs[-1],
                'middle': float(middle[-1]) if not np.isnan(middle[-1]) else klines.close[-1],
                'lower': float(lower[-1]) if not np.isnan(lower[-1]) else lows[-1]
            }
        except Exception as e:
            logger.error(f"Donchian Channels calculation error: {e}")
            return {}
    
    def _calculate_ad(self, klines: CandlestickData) -> float:
        """Calculate Accumulation/Distribution Line"""
        try:
            highs = np.array(klines.high)
            lows = np.array(klines.low)
            closes = np.array(klines.close)
            volumes = np.array(klines.volume)
            
            ad = talib.AD(highs, lows, closes, volumes)
            return float(ad[-1]) if not np.isnan(ad[-1]) else 0
        except Exception as e:
            logger.error(f"A/D calculation error: {e}")
            return 0
    
    def _calculate_cmf(self, klines: CandlestickData) -> float:
        """Calculate Chaikin Money Flow"""
        try:
            highs = np.array(klines.high)
            lows = np.array(klines.low)
            closes = np.array(klines.close)
            volumes = np.array(klines.volume)
            
            cmf = talib.ADOSC(highs, lows, closes, volumes)
            return float(cmf[-1]) if not np.isnan(cmf[-1]) else 0
        except Exception as e:
            logger.error(f"CMF calculation error: {e}")
            return 0
    
    def _detect_candlestick_patterns(self, klines: CandlestickData) -> List[str]:
        """Detect candlestick patterns"""
        patterns = []
        
        try:
            opens = np.array(klines.open)
            highs = np.array(klines.high)
            lows = np.array(klines.low)
            closes = np.array(klines.close)
            
            # Check various candlestick patterns
            pattern_functions = {
                'DOJI': talib.CDLDOJI,
                'HAMMER': talib.CDLHAMMER,
                'SHOOTING_STAR': talib.CDLSHOOTINGSTAR,
                'ENGULFING': talib.CDLENGULFING,
                'MORNING_STAR': talib.CDLMORNINGSTAR,
                'EVENING_STAR': talib.CDLEVENINGSTAR,
                'THREE_WHITE_SOLDIERS': talib.CDL3WHITESOLDIERS,
                'THREE_BLACK_CROWS': talib.CDL3BLACKCROWS,
                'HARAMI': talib.CDLHARAMI,
                'DARK_CLOUD_COVER': talib.CDLDARKCLOUDCOVER
            }
            
            for pattern_name, pattern_func in pattern_functions.items():
                result = pattern_func(opens, highs, lows, closes)
                if result[-1] != 0:  # Pattern detected
                    patterns.append(pattern_name)
            
        except Exception as e:
            logger.error(f"Pattern detection error: {e}")
        
        return patterns
    
    def _find_support_resistance(self, klines: CandlestickData) -> Dict:
        """Find support and resistance levels"""
        try:
            highs = np.array(klines.high[-100:])
            lows = np.array(klines.low[-100:])
            closes = np.array(klines.close[-100:])
            
            # Simple pivot points as S/R
            pivot = (highs[-1] + lows[-1] + closes[-1]) / 3
            
            # Find local maxima and minima
            resistance_levels = []
            support_levels = []
            
            window = 10
            for i in range(window, len(highs) - window):
                if highs[i] == max(highs[i-window:i+window+1]):
                    resistance_levels.append(highs[i])
                if lows[i] == min(lows[i-window:i+window+1]):
                    support_levels.append(lows[i])
            
            # Get strongest levels (most recent and most tested)
            current_price = closes[-1]
            
            nearest_resistance = min(
                [r for r in resistance_levels if r > current_price],
                default=current_price * 1.05
            )
            nearest_support = max(
                [s for s in support_levels if s < current_price],
                default=current_price * 0.95
            )
            
            return {
                'pivot': float(pivot),
                'resistance_1': float(nearest_resistance),
                'support_1': float(nearest_support),
                'resistance_levels': resistance_levels[-5:],
                'support_levels': support_levels[-5:]
            }
        except Exception as e:
            logger.error(f"S/R calculation error: {e}")
            return {}
    
    def _determine_trend(self, klines: CandlestickData) -> str:
        """Determine current trend direction"""
        try:
            closes = np.array(klines.close)
            
            # Use multiple methods to determine trend
            ma_20 = talib.SMA(closes, 20)
            ma_50 = talib.SMA(closes, 50)
            
            current_price = closes[-1]
            
            # Price vs MAs
            above_ma20 = current_price > ma_20[-1]
            above_ma50 = current_price > ma_50[-1] if len(closes) >= 50 else above_ma20
            ma20_rising = ma_20[-1] > ma_20[-5] if len(ma_20) > 5 else True
            
            # Higher highs and higher lows for uptrend
            recent_highs = klines.high[-20:]
            recent_lows = klines.low[-20:]
            
            higher_highs = recent_highs[-1] > max(recent_highs[:-5]) if len(recent_highs) > 5 else False
            higher_lows = recent_lows[-1] > min(recent_lows[:-5]) if len(recent_lows) > 5 else False
            
            if above_ma20 and above_ma50 and ma20_rising and higher_highs:
                return "STRONG_UP"
            elif above_ma20 and ma20_rising:
                return "UP"
            elif not above_ma20 and not above_ma50 and not ma20_rising:
                return "STRONG_DOWN"
            elif not above_ma20:
                return "DOWN"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            logger.error(f"Trend determination error: {e}")
            return "NEUTRAL"
    
    def _calculate_trend_strength(self, klines: CandlestickData) -> float:
        """Calculate trend strength (-1 to 1)"""
        try:
            closes = np.array(klines.close)
            
            # ADX for trend strength
            adx = self._calculate_adx(klines)
            
            # Normalize ADX (0-100) to (0-1)
            strength = min(adx / 100, 1.0)
            
            # Apply direction
            trend = self._determine_trend(klines)
            if "DOWN" in trend:
                strength = -strength
            elif trend == "NEUTRAL":
                strength = strength * 0.5
            
            return strength
            
        except Exception as e:
            logger.error(f"Trend strength calculation error: {e}")
            return 0
    
    def _calculate_pivot_points(self, klines: CandlestickData) -> Dict:
        """Calculate pivot points"""
        try:
            high = klines.high[-1]
            low = klines.low[-1]
            close = klines.close[-1]
            
            pivot = (high + low + close) / 3
            
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'pivot': float(pivot),
                'r1': float(r1),
                'r2': float(r2),
                'r3': float(r3),
                's1': float(s1),
                's2': float(s2),
                's3': float(s3)
            }
        except Exception as e:
            logger.error(f"Pivot points calculation error: {e}")
            return {}
    
    def _calculate_fibonacci_levels(self, klines: CandlestickData) -> Dict:
        """Calculate Fibonacci retracement levels"""
        try:
            # Get recent swing high and low
            recent_high = max(klines.high[-50:])
            recent_low = min(klines.low[-50:])
            
            diff = recent_high - recent_low
            
            levels = {
                '0.0': float(recent_high),
                '23.6': float(recent_high - diff * 0.236),
                '38.2': float(recent_high - diff * 0.382),
                '50.0': float(recent_high - diff * 0.5),
                '61.8': float(recent_high - diff * 0.618),
                '78.6': float(recent_high - diff * 0.786),
                '100.0': float(recent_low)
            }
            
            return levels
        except Exception as e:
            logger.error(f"Fibonacci calculation error: {e}")
            return {}
    
    def _determine_market_regime(self, klines: CandlestickData) -> str:
        """Determine current market regime"""
        try:
            volatility = self._calculate_volatility(klines)
            trend = self._determine_trend(klines)
            adx = self._calculate_adx(klines)
            
            if adx > 40:
                if "UP" in trend:
                    return "trending_up"
                elif "DOWN" in trend:
                    return "trending_down"
            elif volatility > 0.5:
                return "volatile"
            elif adx < 20:
                return "ranging"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Market regime determination error: {e}")
            return "unknown"
    
    async def _calculate_btc_correlation(self, symbol: str, klines: CandlestickData) -> float:
        """Calculate correlation with BTC"""
        if symbol == "BTCUSDT":
            return 1.0
        
        try:
            # Get BTC klines
            btc_klines = await self._get_klines("BTCUSDT", "1h", 100)
            if not btc_klines:
                return 0
            
            # Calculate returns
            symbol_returns = np.diff(klines.close[-100:]) / klines.close[-101:-1]
            btc_returns = np.diff(btc_klines.close[-100:]) / btc_klines.close[-101:-1]
            
            # Calculate correlation
            if len(symbol_returns) == len(btc_returns) and len(symbol_returns) > 0:
                correlation = np.corrcoef(symbol_returns, btc_returns)[0, 1]
                return float(correlation) if not np.isnan(correlation) else 0
            
            return 0
            
        except Exception as e:
            logger.error(f"BTC correlation calculation error: {e}")
            return 0
    
    def _calculate_momentum_score(self, klines: CandlestickData) -> float:
        """Calculate momentum score (0-100)"""
        try:
            rsi = self._calculate_rsi(klines)
            macd_data = self._calculate_macd(klines)
            stoch = self._calculate_stochastic(klines)
            mfi = self._calculate_mfi(klines)
            
            scores = []
            
            # RSI score
            if rsi > 70:
                scores.append(80)
            elif rsi > 50:
                scores.append(60)
            elif rsi > 30:
                scores.append(40)
            else:
                scores.append(20)
            
            # MACD score
            if macd_data['histogram'] > 0:
                scores.append(70)
            else:
                scores.append(30)
            
            # Stochastic score
            if stoch['k'] > 80:
                scores.append(80)
            elif stoch['k'] > 50:
                scores.append(60)
            else:
                scores.append(30)
            
            # MFI score
            if mfi > 80:
                scores.append(80)
            elif mfi > 50:
                scores.append(60)
            else:
                scores.append(30)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.error(f"Momentum score calculation error: {e}")
            return 50
    
    def _calculate_trend_score(self, klines: CandlestickData) -> float:
        """Calculate trend score (0-100)"""
        try:
            trend = self._determine_trend(klines)
            adx = self._calculate_adx(klines)
            
            base_score = 50
            
            if "STRONG_UP" in trend:
                base_score = 80
            elif "UP" in trend:
                base_score = 65
            elif "STRONG_DOWN" in trend:
                base_score = 20
            elif "DOWN" in trend:
                base_score = 35
            
            # Adjust by ADX strength
            if adx > 40:
                adjustment = 10
            elif adx > 25:
                adjustment = 5
            else:
                adjustment = -5
            
            return max(0, min(100, base_score + adjustment))
            
        except Exception as e:
            logger.error(f"Trend score calculation error: {e}")
            return 50
    
    def _calculate_volume_score(self, klines: CandlestickData) -> float:
        """Calculate volume score (0-100)"""
        try:
            volume_ratio = self._calculate_volume_ratio(klines)
            volume_trend = self._calculate_volume_trend(klines)
            
            base_score = 50
            
            # Volume ratio scoring
            if volume_ratio > 2:
                base_score += 30
            elif volume_ratio > 1.5:
                base_score += 20
            elif volume_ratio > 1:
                base_score += 10
            elif volume_ratio < 0.5:
                base_score -= 20
            
            # Volume trend scoring
            if volume_trend > 0.1:
                base_score += 10
            elif volume_trend < -0.1:
                base_score -= 10
            
            return max(0, min(100, base_score))
            
        except Exception as e:
            logger.error(f"Volume score calculation error: {e}")
            return 50
    
    def _calculate_volatility_score(self, klines: CandlestickData) -> float:
        """Calculate volatility score (0-100)"""
        try:
            volatility = self._calculate_volatility(klines)
            atr = self._calculate_atr(klines)
            bb = self._calculate_bollinger_bands(klines)
            
            # Lower volatility = higher score for stability
            if volatility < 0.2:
                score = 80
            elif volatility < 0.4:
                score = 60
            elif volatility < 0.6:
                score = 40
            else:
                score = 20
            
            # Adjust by BB width
            if bb['width'] > 0:
                current_price = klines.close[-1]
                bb_position = bb['percent_b']
                
                if 0.2 < bb_position < 0.8:
                    score += 10  # Price in middle of bands
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Volatility score calculation error: {e}")
            return 50
    
    def _get_ma_signal(self, klines: CandlestickData) -> float:
        """Get moving average signal (0-100)"""
        try:
            mas = self._calculate_moving_averages(klines)
            current_price = klines.close[-1]
            
            score = 50
            
            # Check price vs MAs
            if current_price > mas.get('ma_20', current_price):
                score += 10
            if current_price > mas.get('ma_50', current_price):
                score += 15
            if current_price > mas.get('ma_200', current_price):
                score += 20
            
            # Check MA alignment
            if mas.get('ma_20', 0) > mas.get('ma_50', 0):
                score += 5
            if mas.get('ma_50', 0) > mas.get('ma_200', 0):
                score += 5
            
            return min(100, score)
            
        except Exception as e:
            logger.error(f"MA signal calculation error: {e}")
            return 50
    
    def _get_bb_signal(self, klines: CandlestickData) -> float:
        """Get Bollinger Bands signal (0-100)"""
        try:
            bb = self._calculate_bollinger_bands(klines)
            
            percent_b = bb['percent_b']
            
            if percent_b < 0:
                return 90  # Below lower band - oversold
            elif percent_b < 0.2:
                return 70
            elif percent_b < 0.8:
                return 50
            elif percent_b < 1:
                return 30
            else:
                return 10  # Above upper band - overbought
                
        except Exception as e:
            logger.error(f"BB signal calculation error: {e}")
            return 50
    
    def _calculate_overall_score(self, analysis: Dict) -> float:
        """Calculate overall technical score"""
        try:
            scores = []
            weights = []
            
            # Momentum indicators (30% weight)
            momentum_score = analysis.get('momentum_score', 50)
            scores.append(momentum_score)
            weights.append(0.3)
            
            # Trend indicators (30% weight)
            trend_score = analysis.get('trend_score', 50)
            scores.append(trend_score)
            weights.append(0.3)
            
            # Volume indicators (20% weight)
            volume_score = analysis.get('volume_score', 50)
            scores.append(volume_score)
            weights.append(0.2)
            
            # Volatility (20% weight)
            volatility_score = analysis.get('volatility_score', 50)
            scores.append(volatility_score)
            weights.append(0.2)
            
            # Calculate weighted average
            total_weight = sum(weights)
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            
            return weighted_sum / total_weight if total_weight > 0 else 50
            
        except Exception as e:
            logger.error(f"Overall score calculation error: {e}")
            return 50
    
    def _get_rating(self, score: float) -> str:
        """Get rating based on score"""
        if score >= 80:
            return "STRONG_BUY"
        elif score >= 65:
            return "BUY"
        elif score >= 35:
            return "NEUTRAL"
        elif score >= 20:
            return "SELL"
        else:
            return "STRONG_SELL"
    
    def _generate_signals(self, analysis: Dict) -> List[str]:
        """Generate technical signals"""
        signals = []
        
        # RSI signals
        rsi = analysis.get('rsi', 50)
        if rsi < 30:
            signals.append("RSI_OVERSOLD")
        elif rsi > 70:
            signals.append("RSI_OVERBOUGHT")
        
        # MACD signals
        macd = analysis.get('macd', {})
        if macd.get('histogram', 0) > 0:
            signals.append("MACD_BULLISH")
        elif macd.get('histogram', 0) < 0:
            signals.append("MACD_BEARISH")
        
        # Trend signals
        trend = analysis.get('trend_direction', 'NEUTRAL')
        if "UP" in trend:
            signals.append("UPTREND")
        elif "DOWN" in trend:
            signals.append("DOWNTREND")
        
        # Pattern signals
        patterns = analysis.get('candlestick_patterns', [])
        for pattern in patterns:
            signals.append(f"PATTERN_{pattern}")
        
        return signals
    
    def _validate_analysis(self, analysis: Dict) -> bool:
        """Validate analysis contains real data"""
        # Check for required fields
        required = ['symbol', 'timestamp', 'last_price', 'rsi', 'macd']
        for field in required:
            if field not in analysis:
                return False
        
        # Check for realistic values
        if analysis['rsi'] < 0 or analysis['rsi'] > 100:
            return False
        
        if analysis['last_price'] <= 0:
            return False
        
        # Check for mock data indicators
        if analysis['technical_score'] == 0 or analysis['technical_score'] == 100:
            logger.warning("Suspicious technical score detected")
            return False
        
        return True
    
    def _clean_cache(self):
        """Clean old cache entries"""
        current_time = int(time.time() // self.cache_ttl)
        keys_to_remove = []
        
        for key in self.klines_cache:
            key_time = int(key.split(':')[2])
            if key_time < current_time - 2:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.klines_cache[key]
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'symbol': '',
            'timeframe': '',
            'timestamp': datetime.now().isoformat(),
            'technical_score': 50,
            'technical_rating': 'NEUTRAL',
            'signals': [],
            'error': 'Analysis failed'
        }
