"""
DEMIR AI v8.0 - Signal Generator
Advanced signal generation with multi-layer analysis
ZERO MOCK DATA - Only real market data
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    """Signal strength levels"""
    ULTRA_STRONG = "ultra_strong"  # 90-100% confidence
    STRONG = "strong"              # 75-90% confidence  
    MEDIUM = "medium"              # 60-75% confidence
    WEAK = "weak"                  # 45-60% confidence
    NEUTRAL = "neutral"            # <45% confidence

class SignalAction(Enum):
    """Signal actions"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    timestamp: datetime
    symbol: str
    action: SignalAction
    strength: SignalStrength
    confidence: float
    
    # Entry & Exit
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    
    # Risk Management
    position_size: float
    risk_reward_ratio: float
    max_loss_amount: float
    
    # Component Scores
    technical_score: float
    sentiment_score: float
    volume_score: float
    ml_score: float
    
    # Market Context
    market_regime: str
    volatility: float
    trend_strength: float
    
    # Metadata
    reasons: List[str]
    warnings: List[str]
    timeframe: str

class SignalGenerator:
    """Advanced signal generation system"""
    
    def __init__(self, config):
        self.config = config
        self.min_confidence = config.trading.min_signal_confidence
        self.risk_per_trade = config.trading.max_risk_per_trade
        
        # Component weights for signal generation
        self.weights = {
            'technical': 0.35,
            'sentiment': 0.25,
            'volume': 0.20,
            'ml': 0.20
        }
        
        logger.info("SignalGenerator initialized")
    
    async def generate(self, symbol: str, analysis_data: Dict) -> Optional[Dict]:
        """
        Generate trading signal from analysis data
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            analysis_data: Combined analysis results
            
        Returns:
            Trading signal or None
        """
        try:
            # Extract component scores
            technical_score = self._calculate_technical_score(
                analysis_data.get('technical', {})
            )
            sentiment_score = self._calculate_sentiment_score(
                analysis_data.get('sentiment', {})
            )
            volume_score = self._calculate_volume_score(
                analysis_data.get('technical', {})
            )
            ml_score = analysis_data.get('ml', {}).get('confidence', 50)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                technical_score,
                sentiment_score,
                volume_score,
                ml_score
            )
            
            # Check minimum confidence
            if composite_score < self.min_confidence:
                logger.debug(f"Signal confidence too low for {symbol}: {composite_score:.1f}%")
                return None
            
            # Determine signal action and strength
            action = self._determine_action(composite_score, technical_score, sentiment_score)
            strength = self._determine_strength(composite_score)
            
            # Get current price (from technical data)
            current_price = analysis_data.get('technical', {}).get('close', 0)
            if current_price <= 0:
                logger.warning(f"Invalid price for {symbol}: {current_price}")
                return None
            
            # Calculate entry and exit points
            atr = analysis_data.get('technical', {}).get('atr', current_price * 0.02)
            entry_price, stop_loss, tp_levels = self._calculate_levels(
                current_price, atr, action, strength
            )
            
            # Calculate position size
            position_size = self._calculate_position_size(
                composite_score,
                analysis_data.get('technical', {}).get('volatility', 0.02)
            )
            
            # Risk/Reward ratio
            risk_reward = abs(tp_levels[1] - entry_price) / abs(entry_price - stop_loss)
            
            # Generate signal reasons and warnings
            reasons = self._generate_reasons(
                technical_score, sentiment_score, volume_score, ml_score
            )
            warnings = self._generate_warnings(analysis_data)
            
            # Create signal dictionary
            signal = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action.value,
                'strength': strength.value,
                'confidence': composite_score,
                
                # Prices
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit_1': tp_levels[0],
                'take_profit_2': tp_levels[1],
                'take_profit_3': tp_levels[2],
                
                # Risk
                'position_size': position_size,
                'risk_reward_ratio': risk_reward,
                'max_loss_amount': position_size * abs(entry_price - stop_loss),
                
                # Scores
                'technical_score': technical_score,
                'sentiment_score': sentiment_score,
                'volume_score': volume_score,
                'ml_score': ml_score,
                
                # Context
                'market_regime': analysis_data.get('technical', {}).get('regime', 'unknown'),
                'volatility': analysis_data.get('technical', {}).get('volatility', 0),
                'trend_strength': analysis_data.get('technical', {}).get('trend_strength', 0),
                
                # Meta
                'reasons': reasons,
                'warnings': warnings,
                'timeframe': self.config.analysis.primary_timeframe
            }
            
            logger.info(f"📊 Signal generated for {symbol}: {action.value} "
                       f"(Confidence: {composite_score:.1f}%)")
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            return None
    
    def _calculate_technical_score(self, technical_data: Dict) -> float:
        """Calculate technical analysis score"""
        if not technical_data:
            return 50.0
        
        score = 0
        count = 0
        
        # RSI
        rsi = technical_data.get('rsi')
        if rsi is not None:
            if rsi < 30:
                score += 85  # Oversold
            elif rsi > 70:
                score += 15  # Overbought
            else:
                score += 50
            count += 1
        
        # MACD
        macd = technical_data.get('macd', {})
        if macd:
            if macd.get('histogram', 0) > 0:
                score += 70
            else:
                score += 30
            count += 1
        
        # Moving Averages
        ma_signal = technical_data.get('ma_signal')
        if ma_signal is not None:
            score += ma_signal
            count += 1
        
        # Bollinger Bands
        bb_signal = technical_data.get('bb_signal')
        if bb_signal is not None:
            score += bb_signal
            count += 1
        
        return (score / count) if count > 0 else 50.0
    
    def _calculate_sentiment_score(self, sentiment_data: Dict) -> float:
        """Calculate sentiment score from multiple sources"""
        if not sentiment_data:
            return 50.0
        
        scores = []
        
        # Fear & Greed Index
        fear_greed = sentiment_data.get('fear_greed_index')
        if fear_greed is not None:
            # Invert: Low fear/greed = buy opportunity
            scores.append(100 - fear_greed)
        
        # News Sentiment
        news_sentiment = sentiment_data.get('news_sentiment')
        if news_sentiment is not None:
            scores.append(news_sentiment)
        
        # Social Sentiment
        social_sentiment = sentiment_data.get('social_sentiment')
        if social_sentiment is not None:
            scores.append(social_sentiment)
        
        # Funding Rates
        funding_rate = sentiment_data.get('funding_rate')
        if funding_rate is not None:
            # Negative funding = bullish
            if funding_rate < 0:
                scores.append(70)
            elif funding_rate > 0.001:
                scores.append(30)
            else:
                scores.append(50)
        
        return np.mean(scores) if scores else 50.0
    
    def _calculate_volume_score(self, technical_data: Dict) -> float:
        """Calculate volume-based score"""
        if not technical_data:
            return 50.0
        
        volume_ratio = technical_data.get('volume_ratio', 1.0)
        volume_trend = technical_data.get('volume_trend', 0)
        
        score = 50.0
        
        # High volume confirmation
        if volume_ratio > 1.5:
            score += 20
        elif volume_ratio < 0.5:
            score -= 20
        
        # Volume trend
        score += volume_trend * 30
        
        return max(0, min(100, score))
    
    def _calculate_composite_score(self, technical: float, sentiment: float,
                                  volume: float, ml: float) -> float:
        """Calculate weighted composite score"""
        composite = (
            technical * self.weights['technical'] +
            sentiment * self.weights['sentiment'] +
            volume * self.weights['volume'] +
            ml * self.weights['ml']
        )
        
        return max(0, min(100, composite))
    
    def _determine_action(self, composite: float, technical: float,
                         sentiment: float) -> SignalAction:
        """Determine trading action based on scores"""
        
        # Strong signals
        if composite >= 80 and technical >= 75:
            return SignalAction.STRONG_BUY
        elif composite <= 20 and technical <= 25:
            return SignalAction.STRONG_SELL
        
        # Regular signals
        elif composite >= 60:
            return SignalAction.BUY
        elif composite <= 40:
            return SignalAction.SELL
        
        # Neutral
        else:
            return SignalAction.HOLD
    
    def _determine_strength(self, score: float) -> SignalStrength:
        """Determine signal strength from score"""
        if score >= 90:
            return SignalStrength.ULTRA_STRONG
        elif score >= 75:
            return SignalStrength.STRONG
        elif score >= 60:
            return SignalStrength.MEDIUM
        elif score >= 45:
            return SignalStrength.WEAK
        else:
            return SignalStrength.NEUTRAL
    
    def _calculate_levels(self, price: float, atr: float,
                         action: SignalAction,
                         strength: SignalStrength) -> Tuple[float, float, List[float]]:
        """Calculate entry, stop loss, and take profit levels"""
        
        # Entry price (slightly better than market)
        if action in [SignalAction.BUY, SignalAction.STRONG_BUY]:
            entry = price * 0.999  # 0.1% below market
            
            # Stop loss based on strength
            sl_multipliers = {
                SignalStrength.ULTRA_STRONG: 1.5,
                SignalStrength.STRONG: 2.0,
                SignalStrength.MEDIUM: 2.5,
                SignalStrength.WEAK: 3.0,
                SignalStrength.NEUTRAL: 3.5
            }
            stop_loss = entry - (atr * sl_multipliers.get(strength, 2.5))
            
            # Take profit levels
            tp1 = entry + (atr * self.config.trading.tp1_multiplier)
            tp2 = entry + (atr * self.config.trading.tp2_multiplier)
            tp3 = entry + (atr * self.config.trading.tp3_multiplier)
            
        else:  # SELL signals
            entry = price * 1.001  # 0.1% above market
            
            sl_multipliers = {
                SignalStrength.ULTRA_STRONG: 1.5,
                SignalStrength.STRONG: 2.0,
                SignalStrength.MEDIUM: 2.5,
                SignalStrength.WEAK: 3.0,
                SignalStrength.NEUTRAL: 3.5
            }
            stop_loss = entry + (atr * sl_multipliers.get(strength, 2.5))
            
            tp1 = entry - (atr * self.config.trading.tp1_multiplier)
            tp2 = entry - (atr * self.config.trading.tp2_multiplier)
            tp3 = entry - (atr * self.config.trading.tp3_multiplier)
        
        return entry, stop_loss, [tp1, tp2, tp3]
    
    def _calculate_position_size(self, confidence: float, volatility: float) -> float:
        """Calculate position size based on confidence and volatility"""
        
        # Base position size
        base_size = self.config.trading.default_position_size
        
        # Adjust for confidence
        confidence_multiplier = confidence / 100
        
        # Adjust for volatility (inverse relationship)
        volatility_multiplier = max(0.5, min(1.5, 1.0 / (1 + volatility)))
        
        # Final position size
        position_size = base_size * confidence_multiplier * volatility_multiplier
        
        # Apply limits
        position_size = max(0.01, min(0.10, position_size))  # 1-10% of portfolio
        
        return round(position_size, 3)
    
    def _generate_reasons(self, technical: float, sentiment: float,
                         volume: float, ml: float) -> List[str]:
        """Generate reasons for the signal"""
        reasons = []
        
        if technical >= 70:
            reasons.append(f"Strong technical indicators ({technical:.0f}%)")
        elif technical >= 60:
            reasons.append(f"Positive technical signals ({technical:.0f}%)")
        
        if sentiment >= 70:
            reasons.append(f"Positive market sentiment ({sentiment:.0f}%)")
        elif sentiment <= 30:
            reasons.append(f"Extreme fear detected ({sentiment:.0f}%)")
        
        if volume >= 70:
            reasons.append(f"High volume confirmation ({volume:.0f}%)")
        
        if ml >= 70:
            reasons.append(f"AI models predict movement ({ml:.0f}%)")
        
        return reasons if reasons else ["Multiple indicators aligned"]
    
    def _generate_warnings(self, analysis_data: Dict) -> List[str]:
        """Generate risk warnings"""
        warnings = []
        
        # Check volatility
        volatility = analysis_data.get('technical', {}).get('volatility', 0)
        if volatility > 0.05:
            warnings.append(f"High volatility: {volatility:.1%}")
        
        # Check volume
        volume_ratio = analysis_data.get('technical', {}).get('volume_ratio', 1)
        if volume_ratio < 0.5:
            warnings.append("Low volume - possible false signal")
        
        # Check correlation
        btc_correlation = analysis_data.get('technical', {}).get('btc_correlation', 0)
        if abs(btc_correlation) > 0.8:
            warnings.append(f"High BTC correlation: {btc_correlation:.2f}")
        
        return warnings if warnings else ["Standard market conditions"]
