"""
DEMIR AI v8.0 - Data Validator
CRITICAL: Mock Data Detector & Real Data Verifier
ZERO MOCK DATA POLICY - STRICTLY ENFORCED
"""

import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import re

logger = logging.getLogger(__name__)

class MockDataDetector:
    """
    Detects mock/fake/test/fallback/hardcoded data
    CRITICAL COMPONENT - ZERO TOLERANCE FOR FAKE DATA
    """
    
    def __init__(self):
        # Forbidden patterns that indicate mock data
        self.mock_indicators = [
            'mock', 'fake', 'test', 'demo', 'sample', 'example',
            'placeholder', 'dummy', 'fallback', 'hardcoded', 'prototype',
            'todo', 'fixme', 'xxx', 'temp', 'tmp', 'debug'
        ]
        
        # Suspicious values
        self.suspicious_values = [
            0.0, 1.0, 100.0, 1000.0, 10000.0,  # Round numbers
            12345, 123456, 111111, 999999,      # Pattern numbers
            0.123456789,                         # Too precise
            42, 69, 420, 1337,                  # Common test values
        ]
        
        # Suspicious patterns in strings
        self.suspicious_patterns = [
            r'^test_',
            r'^fake_',
            r'^mock_',
            r'_test$',
            r'_fake$',
            r'_mock$',
            r'\d{5,}',  # Long sequences of digits
            r'([a-z])\1{4,}',  # Repeated characters
            r'lorem\s*ipsum',  # Lorem ipsum text
        ]
        
        # Valid exchange symbols (whitelist)
        self.valid_symbols = {
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
            'SOLUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'AVAXUSDT',
            'UNIUSDT', 'LTCUSDT', 'ALGOUSDT', 'ATOMUSDT', 'FTMUSDT',
            'XLMUSDT', 'VETUSDT', 'TRXUSDT', 'ETCUSDT', 'THETAUSDT'
        }
        
        logger.info("MockDataDetector initialized - ZERO TOLERANCE MODE")
    
    def detect(self, data: Any) -> bool:
        """
        Detect if data contains mock/fake/test values
        Returns True if mock data detected
        """
        try:
            if data is None:
                return True
            
            # Check different data types
            if isinstance(data, dict):
                return self._check_dict(data)
            elif isinstance(data, list):
                return self._check_list(data)
            elif isinstance(data, str):
                return self._check_string(data)
            elif isinstance(data, (int, float)):
                return self._check_number(data)
            
            return False
            
        except Exception as e:
            logger.error(f"Error in mock detection: {e}")
            # Err on the side of caution - assume mock if error
            return True
    
    def _check_dict(self, data: Dict) -> bool:
        """Check dictionary for mock data"""
        # Check keys
        for key in data.keys():
            if isinstance(key, str):
                key_lower = key.lower()
                for indicator in self.mock_indicators:
                    if indicator in key_lower:
                        logger.warning(f"Mock indicator '{indicator}' found in key: {key}")
                        return True
        
        # Check values recursively
        for value in data.values():
            if self.detect(value):
                return True
        
        # Check for suspicious patterns
        if self._has_suspicious_patterns(data):
            return True
        
        return False
    
    def _check_list(self, data: List) -> bool:
        """Check list for mock data"""
        # Empty lists are suspicious
        if len(data) == 0:
            return True
        
        # Check each item
        for item in data:
            if self.detect(item):
                return True
        
        # Check for repeated values (common in mock data)
        if len(data) > 3:
            unique_ratio = len(set(map(str, data))) / len(data)
            if unique_ratio < 0.3:  # Less than 30% unique values
                logger.warning(f"Suspicious repeated values in list")
                return True
        
        return False
    
    def _check_string(self, data: str) -> bool:
        """Check string for mock indicators"""
        data_lower = data.lower()
        
        # Check for mock indicators
        for indicator in self.mock_indicators:
            if indicator in data_lower:
                logger.warning(f"Mock indicator '{indicator}' found in string: {data}")
                return True
        
        # Check suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, data_lower):
                logger.warning(f"Suspicious pattern found in string: {data}")
                return True
        
        # Check if it's a valid symbol
        if data.upper() in self.valid_symbols:
            return False
        
        # Check for placeholder formats
        if data.startswith('your_') or data.endswith('_here'):
            logger.warning(f"Placeholder format detected: {data}")
            return True
        
        return False
    
    def _check_number(self, data: float) -> bool:
        """Check number for suspicious values"""
        # Check for suspicious exact values
        if data in self.suspicious_values:
            logger.warning(f"Suspicious value detected: {data}")
            return True
        
        # Check for too many decimal places (common in random data)
        if isinstance(data, float):
            decimal_str = str(data).split('.')[-1]
            if len(decimal_str) > 8:  # More than 8 decimal places
                logger.warning(f"Suspicious precision: {data}")
                return True
        
        # Check for repeating patterns
        data_str = str(data)
        if len(data_str) > 4:
            for i in range(1, len(data_str) // 2):
                pattern = data_str[:i]
                if data_str == pattern * (len(data_str) // len(pattern)):
                    logger.warning(f"Repeating pattern in number: {data}")
                    return True
        
        return False
    
    def _has_suspicious_patterns(self, data: Dict) -> bool:
        """Check for suspicious data patterns"""
        # Check for all values being identical
        values = list(data.values())
        if len(values) > 3:
            if all(v == values[0] for v in values):
                logger.warning("All values in dict are identical")
                return True
        
        # Check for sequential values (common in mock data)
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        if len(numeric_values) > 3:
            diffs = [numeric_values[i+1] - numeric_values[i] 
                    for i in range(len(numeric_values)-1)]
            if all(d == diffs[0] for d in diffs):
                logger.warning("Sequential numeric values detected")
                return True
        
        return False


class RealDataVerifier:
    """
    Verifies data comes from real sources
    Checks timestamps, price ranges, data freshness
    """
    
    def __init__(self, config):
        self.config = config
        
        # Valid price ranges for crypto (USDT pairs)
        self.price_ranges = {
            'BTCUSDT': (10000, 100000),
            'ETHUSDT': (500, 10000),
            'BNBUSDT': (50, 1000),
            'SOLUSDT': (5, 500),
            'XRPUSDT': (0.1, 5),
            'ADAUSDT': (0.1, 5),
            'DOTUSDT': (2, 100),
            'MATICUSDT': (0.1, 5),
            'LINKUSDT': (2, 100),
            'AVAXUSDT': (5, 200)
        }
        
        # Maximum age for real-time data (seconds)
        self.max_data_age = {
            'price': 60,           # 1 minute for prices
            'orderbook': 30,       # 30 seconds for orderbook
            'trades': 60,          # 1 minute for trades
            'sentiment': 3600,     # 1 hour for sentiment
            'technical': 300,      # 5 minutes for indicators
        }
        
        logger.info("RealDataVerifier initialized")
    
    async def verify(self, source: str, data: Dict) -> bool:
        """
        Verify data is from real source
        Returns True if data is verified as real
        """
        try:
            # Check data structure
            if not self._verify_structure(source, data):
                return False
            
            # Check timestamp freshness
            if not self._verify_timestamp(source, data):
                return False
            
            # Check value ranges
            if not self._verify_ranges(source, data):
                return False
            
            # Check data consistency
            if not self._verify_consistency(source, data):
                return False
            
            # Source-specific verification
            if source == 'binance_api':
                return await self._verify_binance_data(data)
            elif source == 'sentiment_sources':
                return await self._verify_sentiment_data(data)
            elif source == 'technical_indicators':
                return await self._verify_technical_data(data)
            
            return True
            
        except Exception as e:
            logger.error(f"Real data verification error: {e}")
            return False
    
    def _verify_structure(self, source: str, data: Dict) -> bool:
        """Verify data has expected structure"""
        # Check for required fields based on source
        required_fields = {
            'price': ['symbol', 'price', 'timestamp'],
            'orderbook': ['bids', 'asks', 'timestamp'],
            'trades': ['price', 'quantity', 'time'],
            'sentiment': ['score', 'source', 'timestamp'],
            'technical': ['indicator', 'value', 'timestamp']
        }
        
        # Determine data type from source
        data_type = self._get_data_type(source)
        
        if data_type in required_fields:
            for field in required_fields[data_type]:
                if field not in str(data):  # Basic check
                    logger.warning(f"Missing required field '{field}' in {source}")
                    return False
        
        return True
    
    def _verify_timestamp(self, source: str, data: Dict) -> bool:
        """Verify timestamp is recent"""
        # Extract timestamp
        timestamp = self._extract_timestamp(data)
        
        if timestamp is None:
            logger.warning(f"No timestamp found in {source}")
            return False
        
        # Check age
        data_type = self._get_data_type(source)
        max_age = self.max_data_age.get(data_type, 300)
        
        current_time = time.time()
        age = current_time - timestamp
        
        if age > max_age:
            logger.warning(f"Data from {source} is too old: {age:.0f}s")
            return False
        
        # Check if timestamp is in future (indicates fake data)
        if age < -60:  # Allow 60 seconds for clock drift
            logger.warning(f"Future timestamp detected in {source}")
            return False
        
        return True
    
    def _verify_ranges(self, source: str, data: Dict) -> bool:
        """Verify values are within realistic ranges"""
        # Extract prices
        prices = self._extract_prices(data)
        
        for symbol, price in prices.items():
            if symbol in self.price_ranges:
                min_price, max_price = self.price_ranges[symbol]
                
                if price < min_price * 0.1:  # Allow 90% drop
                    logger.warning(f"Price too low for {symbol}: {price}")
                    return False
                
                if price > max_price * 10:  # Allow 10x increase
                    logger.warning(f"Price too high for {symbol}: {price}")
                    return False
        
        return True
    
    def _verify_consistency(self, source: str, data: Dict) -> bool:
        """Verify internal data consistency"""
        # Check orderbook consistency
        if 'bids' in data and 'asks' in data:
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            
            if bids and asks:
                # Best bid should be lower than best ask
                best_bid = float(bids[0][0]) if bids[0] else 0
                best_ask = float(asks[0][0]) if asks[0] else 0
                
                if best_bid >= best_ask and best_ask > 0:
                    logger.warning(f"Invalid orderbook: bid >= ask")
                    return False
        
        # Check OHLC consistency
        if all(k in data for k in ['open', 'high', 'low', 'close']):
            o, h, l, c = data['open'], data['high'], data['low'], data['close']
            
            if not (l <= o <= h and l <= c <= h):
                logger.warning(f"Invalid OHLC data")
                return False
        
        return True
    
    async def _verify_binance_data(self, data: Dict) -> bool:
        """Verify data from Binance API"""
        # Check for Binance-specific fields
        binance_fields = ['symbol', 'price', 'bidPrice', 'askPrice']
        
        # Additional Binance-specific checks
        if 'symbol' in data:
            symbol = data['symbol']
            if not symbol.endswith('USDT') and not symbol.endswith('BTC'):
                logger.warning(f"Suspicious Binance symbol: {symbol}")
                return False
        
        return True
    
    async def _verify_sentiment_data(self, data: Dict) -> bool:
        """Verify sentiment data"""
        # Check sentiment scores are in valid range
        for key, value in data.items():
            if 'score' in str(key).lower() or 'sentiment' in str(key).lower():
                if isinstance(value, (int, float)):
                    if value < 0 or value > 100:
                        logger.warning(f"Invalid sentiment score: {value}")
                        return False
        
        return True
    
    async def _verify_technical_data(self, data: Dict) -> bool:
        """Verify technical indicator data"""
        # Check RSI range
        if 'rsi' in data:
            rsi = data['rsi']
            if isinstance(rsi, (int, float)):
                if rsi < 0 or rsi > 100:
                    logger.warning(f"Invalid RSI value: {rsi}")
                    return False
        
        # Check MACD values
        if 'macd' in data:
            macd = data['macd']
            if isinstance(macd, dict):
                # MACD values should be reasonable
                for key in ['macd', 'signal', 'histogram']:
                    if key in macd:
                        value = macd[key]
                        if isinstance(value, (int, float)):
                            if abs(value) > 10000:  # Arbitrary large number
                                logger.warning(f"Suspicious MACD value: {value}")
                                return False
        
        return True
    
    def _get_data_type(self, source: str) -> str:
        """Determine data type from source"""
        if 'price' in source.lower() or 'ticker' in source.lower():
            return 'price'
        elif 'orderbook' in source.lower() or 'depth' in source.lower():
            return 'orderbook'
        elif 'trade' in source.lower():
            return 'trades'
        elif 'sentiment' in source.lower():
            return 'sentiment'
        elif 'technical' in source.lower() or 'indicator' in source.lower():
            return 'technical'
        
        return 'unknown'
    
    def _extract_timestamp(self, data: Dict) -> Optional[float]:
        """Extract timestamp from data"""
        # Common timestamp fields
        timestamp_fields = [
            'timestamp', 'time', 'datetime', 'created_at', 
            'updated_at', 'ts', 't', 'eventTime'
        ]
        
        for field in timestamp_fields:
            if field in data:
                ts = data[field]
                
                # Convert to seconds if milliseconds
                if isinstance(ts, (int, float)):
                    if ts > 1e12:  # Likely milliseconds
                        return ts / 1000
                    return ts
                
                # Parse string timestamp
                if isinstance(ts, str):
                    try:
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        return dt.timestamp()
                    except:
                        pass
        
        return None
    
    def _extract_prices(self, data: Dict) -> Dict[str, float]:
        """Extract prices from data"""
        prices = {}
        
        # Direct price field
        if 'price' in data and 'symbol' in data:
            try:
                prices[data['symbol']] = float(data['price'])
            except:
                pass
        
        # Multiple symbols
        if isinstance(data, dict):
            for key, value in data.items():
                if key in self.price_ranges:
                    if isinstance(value, (int, float)):
                        prices[key] = float(value)
                    elif isinstance(value, dict) and 'price' in value:
                        try:
                            prices[key] = float(value['price'])
                        except:
                            pass
        
        return prices


class DataValidator:
    """
    Main data validator combining mock detection and real verification
    CRITICAL: This is the final gatekeeper for data quality
    """
    
    def __init__(self, config):
        self.config = config
        self.mock_detector = MockDataDetector()
        self.real_verifier = RealDataVerifier(config)
        
        # Statistics
        self.total_validations = 0
        self.mock_detections = 0
        self.verification_failures = 0
        
        logger.info("DataValidator initialized - ZERO MOCK DATA POLICY ACTIVE")
    
    async def validate(self, symbol: str, data: Optional[Dict] = None) -> bool:
        """
        Complete validation of data
        Returns True only if data is real and valid
        """
        self.total_validations += 1
        
        # Check if data exists
        if data is None:
            logger.warning(f"No data provided for {symbol}")
            return False
        
        # Step 1: Mock detection
        if self.mock_detector.detect(data):
            self.mock_detections += 1
            logger.error(f"❌ MOCK DATA DETECTED for {symbol}")
            logger.error(f"Mock detection rate: {self.mock_detections}/{self.total_validations}")
            return False
        
        # Step 2: Real data verification
        if not await self.real_verifier.verify(symbol, data):
            self.verification_failures += 1
            logger.error(f"❌ REAL DATA VERIFICATION FAILED for {symbol}")
            logger.error(f"Verification failure rate: {self.verification_failures}/{self.total_validations}")
            return False
        
        # Step 3: Symbol validation
        if symbol not in self.mock_detector.valid_symbols:
            logger.warning(f"Unknown symbol: {symbol}")
            # Don't reject, but log for monitoring
        
        logger.debug(f"✅ Data validated for {symbol}")
        return True
    
    def get_statistics(self) -> Dict:
        """Get validation statistics"""
        return {
            'total_validations': self.total_validations,
            'mock_detections': self.mock_detections,
            'verification_failures': self.verification_failures,
            'success_rate': (self.total_validations - self.mock_detections - self.verification_failures) / max(1, self.total_validations)
        }
