"""
DEMIR AI v8.0 - Signal Validator
Comprehensive signal validation with integrity checking
ZERO MOCK DATA - REAL SIGNALS ONLY
NO SIMPLIFICATION - ENTERPRISE GRADE
"""

import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import json

logger = logging.getLogger(__name__)

class SignalIntegrityChecker:
    """
    Signal integrity and consistency checker
    CRITICAL: Ensures signal structure and data quality
    """
    
    def __init__(self):
        # Required signal fields
        self.required_fields = [
            'timestamp', 'symbol', 'action', 'strength', 'confidence',
            'entry_price', 'stop_loss', 'take_profit_1', 'take_profit_2', 'take_profit_3',
            'position_size', 'risk_reward_ratio', 'max_loss_amount',
            'technical_score', 'sentiment_score', 'volume_score', 'ml_score',
            'market_regime', 'volatility', 'trend_strength',
            'reasons', 'warnings', 'timeframe'
        ]
        
        # Valid actions
        self.valid_actions = ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']
        
        # Valid strengths
        self.valid_strengths = ['ultra_strong', 'strong', 'medium', 'weak', 'neutral']
        
        # Valid timeframes
        self.valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
        
        # Valid market regimes
        self.valid_regimes = ['trending_up', 'trending_down', 'ranging', 'volatile', 'breakout', 'breakdown', 'accumulation', 'distribution', 'unknown']
        
        logger.info("SignalIntegrityChecker initialized - STRICT MODE")
    
    def check_integrity(self, signal: Dict) -> Tuple[bool, List[str]]:
        """
        Check signal integrity
        Returns (is_valid, errors)
        """
        errors = []
        
        # Check required fields
        missing_fields = self._check_required_fields(signal)
        if missing_fields:
            errors.extend([f"Missing field: {field}" for field in missing_fields])
        
        # Check data types
        type_errors = self._check_data_types(signal)
        if type_errors:
            errors.extend(type_errors)
        
        # Check value ranges
        range_errors = self._check_value_ranges(signal)
        if range_errors:
            errors.extend(range_errors)
        
        # Check logical consistency
        logic_errors = self._check_logical_consistency(signal)
        if logic_errors:
            errors.extend(logic_errors)
        
        # Check price relationships
        price_errors = self._check_price_relationships(signal)
        if price_errors:
            errors.extend(price_errors)
        
        # Check score consistency
        score_errors = self._check_score_consistency(signal)
        if score_errors:
            errors.extend(score_errors)
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"Signal integrity check failed: {errors}")
        
        return is_valid, errors
    
    def _check_required_fields(self, signal: Dict) -> List[str]:
        """Check for missing required fields"""
        missing = []
        for field in self.required_fields:
            if field not in signal:
                missing.append(field)
        return missing
    
    def _check_data_types(self, signal: Dict) -> List[str]:
        """Check field data types"""
        errors = []
        
        # Numeric fields
        numeric_fields = [
            'confidence', 'entry_price', 'stop_loss', 
            'take_profit_1', 'take_profit_2', 'take_profit_3',
            'position_size', 'risk_reward_ratio', 'max_loss_amount',
            'technical_score', 'sentiment_score', 'volume_score', 'ml_score',
            'volatility', 'trend_strength'
        ]
        
        for field in numeric_fields:
            if field in signal:
                if not isinstance(signal[field], (int, float)):
                    errors.append(f"{field} must be numeric, got {type(signal[field]).__name__}")
        
        # String fields
        string_fields = ['symbol', 'action', 'strength', 'timeframe', 'market_regime']
        for field in string_fields:
            if field in signal:
                if not isinstance(signal[field], str):
                    errors.append(f"{field} must be string, got {type(signal[field]).__name__}")
        
        # List fields
        list_fields = ['reasons', 'warnings']
        for field in list_fields:
            if field in signal:
                if not isinstance(signal[field], list):
                    errors.append(f"{field} must be list, got {type(signal[field]).__name__}")
        
        # Datetime field
        if 'timestamp' in signal:
            if not isinstance(signal['timestamp'], (datetime, str)):
                errors.append(f"timestamp must be datetime or string")
        
        return errors
    
    def _check_value_ranges(self, signal: Dict) -> List[str]:
        """Check if values are in valid ranges"""
        errors = []
        
        # Confidence must be 0-100
        if 'confidence' in signal:
            conf = signal['confidence']
            if conf < 0 or conf > 100:
                errors.append(f"Confidence out of range: {conf}")
        
        # Scores must be 0-100
        score_fields = ['technical_score', 'sentiment_score', 'volume_score', 'ml_score']
        for field in score_fields:
            if field in signal:
                score = signal[field]
                if score < 0 or score > 100:
                    errors.append(f"{field} out of range: {score}")
        
        # Position size must be 0-1 (0-100% of portfolio)
        if 'position_size' in signal:
            size = signal['position_size']
            if size <= 0 or size > 0.2:  # Max 20% per position
                errors.append(f"Position size out of range: {size}")
        
        # Risk reward ratio must be positive
        if 'risk_reward_ratio' in signal:
            rr = signal['risk_reward_ratio']
            if rr <= 0 or rr > 100:
                errors.append(f"Risk/reward ratio invalid: {rr}")
        
        # Volatility must be 0-1
        if 'volatility' in signal:
            vol = signal['volatility']
            if vol < 0 or vol > 1:
                errors.append(f"Volatility out of range: {vol}")
        
        # Trend strength must be -1 to 1
        if 'trend_strength' in signal:
            trend = signal['trend_strength']
            if trend < -1 or trend > 1:
                errors.append(f"Trend strength out of range: {trend}")
        
        # Check enum values
        if 'action' in signal:
            if signal['action'] not in self.valid_actions:
                errors.append(f"Invalid action: {signal['action']}")
        
        if 'strength' in signal:
            if signal['strength'] not in self.valid_strengths:
                errors.append(f"Invalid strength: {signal['strength']}")
        
        if 'timeframe' in signal:
            if signal['timeframe'] not in self.valid_timeframes:
                errors.append(f"Invalid timeframe: {signal['timeframe']}")
        
        if 'market_regime' in signal:
            if signal['market_regime'] not in self.valid_regimes:
                errors.append(f"Invalid market regime: {signal['market_regime']}")
        
        return errors
    
    def _check_logical_consistency(self, signal: Dict) -> List[str]:
        """Check logical consistency of signal"""
        errors = []
        
        # High confidence should match strong strength
        if 'confidence' in signal and 'strength' in signal:
            conf = signal['confidence']
            strength = signal['strength']
            
            if conf >= 90 and strength not in ['ultra_strong', 'strong']:
                errors.append(f"High confidence ({conf}) but weak strength ({strength})")
            
            if conf <= 45 and strength in ['ultra_strong', 'strong']:
                errors.append(f"Low confidence ({conf}) but strong strength ({strength})")
        
        # Action should match scores
        if 'action' in signal:
            action = signal['action']
            tech_score = signal.get('technical_score', 50)
            sent_score = signal.get('sentiment_score', 50)
            
            if action in ['STRONG_BUY', 'BUY']:
                if tech_score < 40 and sent_score < 40:
                    errors.append(f"Buy signal with low scores: tech={tech_score}, sent={sent_score}")
            
            if action in ['STRONG_SELL', 'SELL']:
                if tech_score > 60 and sent_score > 60:
                    errors.append(f"Sell signal with high scores: tech={tech_score}, sent={sent_score}")
        
        # Position size should match confidence
        if 'position_size' in signal and 'confidence' in signal:
            size = signal['position_size']
            conf = signal['confidence']
            
            expected_size = (conf / 100) * 0.05  # Base 5% position
            if abs(size - expected_size) > 0.05:
                logger.debug(f"Position size ({size}) doesn't match confidence ({conf})")
        
        return errors
    
    def _check_price_relationships(self, signal: Dict) -> List[str]:
        """Check price relationships (entry, SL, TP)"""
        errors = []
        
        if not all(k in signal for k in ['entry_price', 'stop_loss', 'take_profit_1']):
            return errors  # Skip if prices missing
        
        entry = signal['entry_price']
        sl = signal['stop_loss']
        tp1 = signal['take_profit_1']
        tp2 = signal.get('take_profit_2', 0)
        tp3 = signal.get('take_profit_3', 0)
        
        if entry <= 0:
            errors.append(f"Invalid entry price: {entry}")
        
        if sl <= 0:
            errors.append(f"Invalid stop loss: {sl}")
        
        # Check based on action
        if 'action' in signal:
            action = signal['action']
            
            if action in ['STRONG_BUY', 'BUY']:
                # For buy: SL < Entry < TP1 < TP2 < TP3
                if sl >= entry:
                    errors.append(f"Buy signal: SL ({sl}) >= Entry ({entry})")
                
                if tp1 <= entry:
                    errors.append(f"Buy signal: TP1 ({tp1}) <= Entry ({entry})")
                
                if tp2 > 0 and tp2 <= tp1:
                    errors.append(f"Buy signal: TP2 ({tp2}) <= TP1 ({tp1})")
                
                if tp3 > 0 and tp3 <= tp2:
                    errors.append(f"Buy signal: TP3 ({tp3}) <= TP2 ({tp2})")
            
            elif action in ['STRONG_SELL', 'SELL']:
                # For sell: TP3 < TP2 < TP1 < Entry < SL
                if sl <= entry:
                    errors.append(f"Sell signal: SL ({sl}) <= Entry ({entry})")
                
                if tp1 >= entry:
                    errors.append(f"Sell signal: TP1 ({tp1}) >= Entry ({entry})")
                
                if tp2 > 0 and tp2 >= tp1:
                    errors.append(f"Sell signal: TP2 ({tp2}) >= TP1 ({tp1})")
                
                if tp3 > 0 and tp3 >= tp2:
                    errors.append(f"Sell signal: TP3 ({tp3}) >= TP2 ({tp2})")
        
        # Check risk/reward ratio calculation
        if 'risk_reward_ratio' in signal:
            risk = abs(entry - sl)
            reward = abs(tp1 - entry)
            
            if risk > 0:
                calculated_rr = reward / risk
                given_rr = signal['risk_reward_ratio']
                
                if abs(calculated_rr - given_rr) > 0.5:
                    errors.append(f"Risk/reward mismatch: calculated={calculated_rr:.2f}, given={given_rr:.2f}")
        
        return errors
    
    def _check_score_consistency(self, signal: Dict) -> List[str]:
        """Check if component scores match overall confidence"""
        errors = []
        
        scores = {
            'technical': signal.get('technical_score', 50),
            'sentiment': signal.get('sentiment_score', 50),
            'volume': signal.get('volume_score', 50),
            'ml': signal.get('ml_score', 50)
        }
        
        # Calculate weighted average
        weights = {
            'technical': 0.35,
            'sentiment': 0.25,
            'volume': 0.20,
            'ml': 0.20
        }
        
        calculated_conf = sum(scores[k] * weights[k] for k in scores)
        given_conf = signal.get('confidence', 0)
        
        # Allow 10% tolerance
        if abs(calculated_conf - given_conf) > 10:
            errors.append(f"Confidence mismatch: calculated={calculated_conf:.1f}, given={given_conf:.1f}")
        
        # Check if any score is 0 (likely mock data)
        for name, score in scores.items():
            if score == 0:
                errors.append(f"{name}_score is 0 - possible mock data")
        
        return errors


class SignalValidator:
    """
    Main signal validator combining all validation checks
    COMPREHENSIVE - NO SIMPLIFICATION
    REAL SIGNALS ONLY - ZERO MOCK DATA
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize sub-validators
        from core.data_validator import MockDataDetector, RealDataVerifier
        self.mock_detector = MockDataDetector()
        self.real_verifier = RealDataVerifier(config)
        self.integrity_checker = SignalIntegrityChecker()
        
        # Signal history for duplicate detection
        self.signal_history = []
        self.max_history = 1000
        
        # Statistics
        self.total_signals = 0
        self.valid_signals = 0
        self.mock_signals = 0
        self.invalid_signals = 0
        self.duplicate_signals = 0
        
        # Validation cache (to avoid re-validating same signal)
        self.validation_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info("SignalValidator initialized - COMPREHENSIVE MODE")
        logger.info("ZERO MOCK DATA POLICY - STRICTLY ENFORCED")
    
    async def validate(self, signal: Dict) -> bool:
        """
        Comprehensive signal validation
        Returns True only if signal passes ALL checks
        """
        self.total_signals += 1
        
        try:
            # Generate signal hash for caching
            signal_hash = self._generate_signal_hash(signal)
            
            # Check cache
            if signal_hash in self.validation_cache:
                cache_time, is_valid = self.validation_cache[signal_hash]
                if time.time() - cache_time < self.cache_ttl:
                    logger.debug(f"Using cached validation result for signal")
                    return is_valid
            
            # Step 1: Mock data detection
            if self._detect_mock_data(signal):
                self.mock_signals += 1
                logger.error(f"❌ MOCK SIGNAL DETECTED - {signal.get('symbol', 'UNKNOWN')}")
                self._cache_result(signal_hash, False)
                return False
            
            # Step 2: Integrity check
            is_valid, errors = self.integrity_checker.check_integrity(signal)
            if not is_valid:
                self.invalid_signals += 1
                logger.error(f"❌ SIGNAL INTEGRITY CHECK FAILED - {errors}")
                self._cache_result(signal_hash, False)
                return False
            
            # Step 3: Real data verification
            if not await self._verify_real_data(signal):
                self.invalid_signals += 1
                logger.error(f"❌ SIGNAL REAL DATA VERIFICATION FAILED")
                self._cache_result(signal_hash, False)
                return False
            
            # Step 4: Duplicate detection
            if self._is_duplicate(signal):
                self.duplicate_signals += 1
                logger.warning(f"⚠️ DUPLICATE SIGNAL DETECTED - {signal.get('symbol', 'UNKNOWN')}")
                self._cache_result(signal_hash, False)
                return False
            
            # Step 5: Exchange-specific validation
            if not await self._validate_exchange_specific(signal):
                self.invalid_signals += 1
                logger.error(f"❌ EXCHANGE-SPECIFIC VALIDATION FAILED")
                self._cache_result(signal_hash, False)
                return False
            
            # Step 6: Risk validation
            if not self._validate_risk_parameters(signal):
                self.invalid_signals += 1
                logger.error(f"❌ RISK PARAMETERS VALIDATION FAILED")
                self._cache_result(signal_hash, False)
                return False
            
            # Step 7: Market hours validation
            if not self._validate_market_hours(signal):
                logger.warning(f"⚠️ Signal generated outside market hours")
                # Don't reject, just warn
            
            # Step 8: Confidence threshold check
            if not self._validate_confidence_threshold(signal):
                self.invalid_signals += 1
                logger.info(f"Signal confidence below threshold")
                self._cache_result(signal_hash, False)
                return False
            
            # All validations passed
            self.valid_signals += 1
            self._add_to_history(signal)
            self._cache_result(signal_hash, True)
            
            logger.info(f"✅ SIGNAL VALIDATED - {signal.get('symbol')} {signal.get('action')} "
                       f"(Confidence: {signal.get('confidence', 0):.1f}%)")
            
            # Log statistics periodically
            if self.total_signals % 100 == 0:
                self._log_statistics()
            
            return True
            
        except Exception as e:
            logger.error(f"Signal validation error: {e}")
            self.invalid_signals += 1
            return False
    
    def _detect_mock_data(self, signal: Dict) -> bool:
        """Detect if signal contains mock data"""
        # Check signal itself
        if self.mock_detector.detect(signal):
            return True
        
        # Check individual fields
        suspicious_fields = []
        
        # Check symbol
        if 'symbol' in signal:
            if self.mock_detector.detect(signal['symbol']):
                suspicious_fields.append('symbol')
        
        # Check prices
        price_fields = ['entry_price', 'stop_loss', 'take_profit_1', 'take_profit_2', 'take_profit_3']
        for field in price_fields:
            if field in signal:
                if self.mock_detector.detect(signal[field]):
                    suspicious_fields.append(field)
        
        # Check scores
        score_fields = ['technical_score', 'sentiment_score', 'volume_score', 'ml_score']
        for field in score_fields:
            if field in signal:
                score = signal[field]
                # Exactly 0, 50, or 100 are suspicious
                if score in [0.0, 50.0, 100.0]:
                    suspicious_fields.append(field)
        
        # Check reasons and warnings
        if 'reasons' in signal:
            for reason in signal['reasons']:
                if self.mock_detector.detect(reason):
                    suspicious_fields.append('reasons')
                    break
        
        if suspicious_fields:
            logger.warning(f"Suspicious fields detected: {suspicious_fields}")
            return True
        
        return False
    
    async def _verify_real_data(self, signal: Dict) -> bool:
        """Verify signal is based on real data"""
        # Verify timestamp is recent
        if 'timestamp' in signal:
            if isinstance(signal['timestamp'], datetime):
                age = (datetime.now() - signal['timestamp']).total_seconds()
            else:
                # Try to parse string timestamp
                try:
                    ts = datetime.fromisoformat(str(signal['timestamp']))
                    age = (datetime.now() - ts).total_seconds()
                except:
                    logger.warning("Invalid timestamp format")
                    return False
            
            # Signal should be less than 5 minutes old
            if age > 300:
                logger.warning(f"Signal too old: {age:.0f} seconds")
                return False
            
            # Signal shouldn't be from future
            if age < -60:  # Allow 60 seconds clock drift
                logger.warning(f"Future timestamp detected")
                return False
        
        # Verify prices are realistic
        if 'symbol' in signal and 'entry_price' in signal:
            result = await self.real_verifier.verify(
                signal['symbol'],
                {'price': signal['entry_price'], 'timestamp': time.time()}
            )
            if not result:
                return False
        
        return True
    
    def _is_duplicate(self, signal: Dict) -> bool:
        """Check if signal is duplicate"""
        # Create signature from key fields
        signature = self._create_signal_signature(signal)
        
        # Check against history
        for hist_signal in self.signal_history[-100:]:  # Check last 100 signals
            hist_signature = self._create_signal_signature(hist_signal)
            
            if signature == hist_signature:
                # Check time difference
                time_diff = abs((signal['timestamp'] - hist_signal['timestamp']).total_seconds())
                if time_diff < 300:  # Within 5 minutes
                    return True
        
        return False
    
    def _create_signal_signature(self, signal: Dict) -> str:
        """Create unique signature for signal"""
        key_fields = [
            signal.get('symbol', ''),
            signal.get('action', ''),
            str(signal.get('entry_price', 0))[:6],  # First 6 digits
            str(signal.get('stop_loss', 0))[:6],
            str(signal.get('confidence', 0))[:4]
        ]
        return '|'.join(key_fields)
    
    async def _validate_exchange_specific(self, signal: Dict) -> bool:
        """Validate signal for specific exchange requirements"""
        symbol = signal.get('symbol', '')
        
        # Binance validation
        if self.config.exchange.primary_exchange == 'binance':
            # Symbol must end with USDT or BTC
            if not (symbol.endswith('USDT') or symbol.endswith('BTC')):
                logger.warning(f"Invalid Binance symbol: {symbol}")
                return False
            
            # Check minimum notional value
            if 'entry_price' in signal and 'position_size' in signal:
                notional = signal['entry_price'] * signal['position_size']
                if notional < 10:  # Minimum $10 for Binance
                    logger.warning(f"Position size below minimum notional: ${notional:.2f}")
                    return False
        
        # Bybit validation
        elif self.config.exchange.primary_exchange == 'bybit':
            # Bybit-specific checks
            pass
        
        # Coinbase validation
        elif self.config.exchange.primary_exchange == 'coinbase':
            # Coinbase-specific checks
            pass
        
        return True
    
    def _validate_risk_parameters(self, signal: Dict) -> bool:
        """Validate risk management parameters"""
        # Check position size
        if 'position_size' in signal:
            if signal['position_size'] > self.config.trading.max_risk_per_trade:
                logger.warning(f"Position size exceeds max risk: {signal['position_size']}")
                return False
        
        # Check risk/reward ratio
        if 'risk_reward_ratio' in signal:
            if signal['risk_reward_ratio'] < 1.0:  # Minimum 1:1 RR
                logger.warning(f"Risk/reward ratio too low: {signal['risk_reward_ratio']}")
                return False
        
        # Check max loss amount
        if 'max_loss_amount' in signal:
            portfolio_value = 10000  # This should come from portfolio manager
            max_allowed_loss = portfolio_value * self.config.trading.max_risk_per_trade
            
            if signal['max_loss_amount'] > max_allowed_loss:
                logger.warning(f"Max loss exceeds limit: ${signal['max_loss_amount']:.2f}")
                return False
        
        return True
    
    def _validate_market_hours(self, signal: Dict) -> bool:
        """Validate signal timing based on market hours"""
        # Crypto markets are 24/7, but check for maintenance windows
        current_hour = datetime.now().hour
        
        # Binance maintenance usually around 03:00-04:00 UTC
        if current_hour == 3:
            logger.debug("Warning: Possible exchange maintenance hour")
            # Don't reject, just warn
        
        return True
    
    def _validate_confidence_threshold(self, signal: Dict) -> bool:
        """Check if signal meets minimum confidence threshold"""
        confidence = signal.get('confidence', 0)
        
        if confidence < self.config.trading.min_signal_confidence:
            logger.debug(f"Signal confidence {confidence:.1f}% below threshold {self.config.trading.min_signal_confidence}%")
            return False
        
        return True
    
    def _add_to_history(self, signal: Dict):
        """Add signal to history"""
        self.signal_history.append(signal)
        
        # Limit history size
        if len(self.signal_history) > self.max_history:
            self.signal_history = self.signal_history[-self.max_history:]
    
    def _generate_signal_hash(self, signal: Dict) -> str:
        """Generate hash for signal caching"""
        # Create stable string representation
        key_data = {
            'symbol': signal.get('symbol'),
            'action': signal.get('action'),
            'entry': str(signal.get('entry_price', 0))[:8],
            'sl': str(signal.get('stop_loss', 0))[:8],
            'confidence': str(signal.get('confidence', 0))[:5]
        }
        
        json_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _cache_result(self, signal_hash: str, is_valid: bool):
        """Cache validation result"""
        self.validation_cache[signal_hash] = (time.time(), is_valid)
        
        # Clean old cache entries
        current_time = time.time()
        self.validation_cache = {
            k: v for k, v in self.validation_cache.items()
            if current_time - v[0] < self.cache_ttl
        }
    
    def _log_statistics(self):
        """Log validation statistics"""
        success_rate = (self.valid_signals / max(1, self.total_signals)) * 100
        mock_rate = (self.mock_signals / max(1, self.total_signals)) * 100
        duplicate_rate = (self.duplicate_signals / max(1, self.total_signals)) * 100
        
        logger.info("="*50)
        logger.info("SIGNAL VALIDATION STATISTICS")
        logger.info(f"Total Signals: {self.total_signals}")
        logger.info(f"Valid Signals: {self.valid_signals} ({success_rate:.1f}%)")
        logger.info(f"Mock Signals Detected: {self.mock_signals} ({mock_rate:.1f}%)")
        logger.info(f"Invalid Signals: {self.invalid_signals}")
        logger.info(f"Duplicate Signals: {self.duplicate_signals} ({duplicate_rate:.1f}%)")
        logger.info("="*50)
    
    def get_statistics(self) -> Dict:
        """Get validation statistics"""
        return {
            'total_signals': self.total_signals,
            'valid_signals': self.valid_signals,
            'mock_signals': self.mock_signals,
            'invalid_signals': self.invalid_signals,
            'duplicate_signals': self.duplicate_signals,
            'success_rate': (self.valid_signals / max(1, self.total_signals)) * 100,
            'cache_size': len(self.validation_cache),
            'history_size': len(self.signal_history)
        }
