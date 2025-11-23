"""
DEMIR AI v8.0 - Configuration Management
Central configuration for all components
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

@dataclass
class TradingConfig:
    """Trading parameters configuration"""
    
    # Symbols
    symbols: List[str] = field(default_factory=lambda: 
        os.getenv('TRADING_SYMBOLS', 'BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT').split(',')
    )
    
    # Position Management
    max_positions: int = int(os.getenv('MAX_POSITIONS', '10'))
    max_risk_per_trade: float = float(os.getenv('MAX_RISK_PER_TRADE', '0.02'))
    default_position_size: float = 0.02  # 2% of portfolio
    
    # Risk Limits
    max_daily_loss: float = float(os.getenv('MAX_DAILY_LOSS', '0.05'))
    max_drawdown: float = float(os.getenv('MAX_DRAWDOWN', '0.15'))
    emergency_stop_loss: float = 0.20  # 20% emergency stop
    
    # Signal Parameters
    min_signal_confidence: float = float(os.getenv('MIN_SIGNAL_CONFIDENCE', '60'))
    signal_check_interval: int = 30  # seconds
    
    # Order Types
    use_limit_orders: bool = True
    limit_order_offset: float = 0.001  # 0.1% from market price
    
    # Take Profit Levels (ATR multipliers)
    tp1_multiplier: float = 1.5
    tp2_multiplier: float = 3.0
    tp3_multiplier: float = 4.5
    
    # Stop Loss (ATR multiplier)
    sl_multiplier: float = 2.0
    use_trailing_stop: bool = True
    trailing_stop_activation: float = 0.02  # Activate after 2% profit

@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    
    primary_exchange: str = 'binance'
    enabled_exchanges: List[str] = field(default_factory=lambda: ['binance'])
    
    # Binance
    binance_api_key: str = os.getenv('BINANCE_API_KEY', '')
    binance_api_secret: str = os.getenv('BINANCE_API_SECRET', '')
    binance_testnet: bool = False
    
    # Bybit
    bybit_api_key: str = os.getenv('BYBIT_API_KEY', '')
    bybit_api_secret: str = os.getenv('BYBIT_API_SECRET', '')
    bybit_testnet: bool = False
    
    # Coinbase
    coinbase_api_key: str = os.getenv('COINBASE_API_KEY', '')
    coinbase_api_secret: str = os.getenv('COINBASE_API_SECRET', '')
    
    # Rate Limits (requests per minute)
    rate_limit_orders: int = 50
    rate_limit_public: int = 1200

@dataclass
class DatabaseConfig:
    """Database configuration"""
    
    # PostgreSQL
    database_url: str = os.getenv('DATABASE_URL', 'postgresql://localhost/demir_ai')
    pool_size: int = 20
    max_overflow: int = 40
    pool_timeout: int = 30
    
    # Redis
    redis_url: str = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    redis_ttl: int = 3600  # 1 hour default TTL
    redis_max_connections: int = 50

@dataclass
class AlertConfig:
    """Alert system configuration"""
    
    # Telegram
    telegram_enabled: bool = bool(os.getenv('TELEGRAM_TOKEN'))
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Discord
    discord_enabled: bool = bool(os.getenv('DISCORD_WEBHOOK'))
    discord_webhook: str = os.getenv('DISCORD_WEBHOOK', '')
    
    # Alert Levels
    alert_on_signal: bool = True
    alert_on_trade: bool = True
    alert_on_error: bool = True
    alert_on_daily_summary: bool = True
    
    # Rate Limiting
    max_alerts_per_minute: int = 10
    alert_cooldown: int = 60  # seconds between similar alerts

@dataclass
class AnalysisConfig:
    """Analysis layers configuration"""
    
    # Feature Toggles
    enable_sentiment: bool = os.getenv('ENABLE_SENTIMENT', 'true').lower() == 'true'
    enable_technical: bool = os.getenv('ENABLE_TECHNICAL', 'true').lower() == 'true'
    enable_ml: bool = os.getenv('ENABLE_ML', 'true').lower() == 'true'
    enable_onchain: bool = os.getenv('ENABLE_ONCHAIN', 'true').lower() == 'true'
    
    # Sentiment Sources (15 active)
    sentiment_sources: List[str] = field(default_factory=lambda: [
        'NewsSentiment',
        'FearGreedIndex',
        'BTCDominance',
        'ExchangeFlow',
        'WhaleAlert',
        'MacroCorrelation',
        'MarketRegime',
        'StablecoinDominance',
        'FundingRates',
        'LongShortRatio',
        'OnChainActivity',
        'ExchangeReserveFlows',
        'OrderBookImbalance',
        'LiquidationCascade',
        'BasisContango'
    ])
    
    # Technical Indicators
    technical_indicators: List[str] = field(default_factory=lambda: [
        'RSI', 'MACD', 'BollingerBands', 'ATR', 'SMA', 'EMA',
        'StochasticRSI', 'OBV', 'VWAP', 'FibonacciRetracement'
    ])
    
    # ML Models (5 active)
    ml_models: List[str] = field(default_factory=lambda: [
        'LSTM', 'XGBoost', 'RandomForest', 'GradientBoosting', 'KMeans'
    ])
    
    # Timeframes
    analysis_timeframes: List[str] = field(default_factory=lambda: ['1h', '4h', '1d'])
    primary_timeframe: str = '1h'

@dataclass
class SystemConfig:
    """System configuration"""
    
    # Environment
    environment: str = os.getenv('ENVIRONMENT', 'development')
    version: str = os.getenv('VERSION', '8.0')
    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    advisory_mode: bool = os.getenv('ADVISORY_MODE', 'true').lower() == 'true'
    
    # API Server
    api_enabled: bool = os.getenv('SERVE_API', 'true').lower() == 'true'
    api_port: int = int(os.getenv('PORT', '8000'))
    api_host: str = '0.0.0.0'
    
    # Monitoring
    health_check_interval: int = int(os.getenv('HEALTH_CHECK_INTERVAL', '60'))
    metrics_update_interval: int = int(os.getenv('METRICS_UPDATE_INTERVAL', '5'))
    
    # Logging
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_to_file: bool = True
    log_file_path: str = 'logs/demir_ai.log'
    log_rotation: str = '1 day'
    log_retention: int = 30  # days
    
    # Performance
    max_workers: int = 10
    async_timeout: int = 30
    request_timeout: int = 30
    
    # Paths
    data_dir: str = 'data'
    models_dir: str = 'models'
    logs_dir: str = 'logs'

@dataclass
class DataProviderConfig:
    """External data provider configuration"""
    
    # Market Data
    alpha_vantage_key: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    coinglass_key: str = os.getenv('COINGLASS_API_KEY', '')
    coinmarketcap_key: str = os.getenv('COINMARKETCAP_API_KEY', '')
    
    # News & Sentiment
    cryptopanic_key: str = os.getenv('CRYPTOALERT_API_KEY', '')
    newsapi_key: str = os.getenv('NEWSAPI_KEY', '')
    
    # Financial Data
    finnhub_key: str = os.getenv('FINNHUB_API_KEY', '')
    fred_key: str = os.getenv('FRED_API_KEY', '')
    twelve_data_key: str = os.getenv('TWELVE_DATA_API_KEY', '')
    
    # Alternative Data
    dexcheck_key: str = os.getenv('DEXCHECK_API_KEY', '')
    opensea_key: str = os.getenv('OPENSEA_API_KEY', '')
    
    # Social Media
    twitter_api_key: str = os.getenv('TWITTER_API_KEY', '')
    twitter_api_secret: str = os.getenv('TWITTER_API_SECRET', '')
    twitter_bearer_token: str = os.getenv('TWITTER_BEARER_TOKEN', '')

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.trading = TradingConfig()
        self.exchange = ExchangeConfig()
        self.database = DatabaseConfig()
        self.alerts = AlertConfig()
        self.analysis = AnalysisConfig()
        self.system = SystemConfig()
        self.data_providers = DataProviderConfig()
        
        # Validate configuration
        self._validate()
        
        # Create necessary directories
        self._create_directories()
    
    def _validate(self):
        """Validate configuration"""
        # Check for required API keys
        if not self.system.advisory_mode:
            if not self.exchange.binance_api_key:
                raise ValueError("Binance API key required for live trading")
        
        # Validate trading parameters
        if self.trading.max_risk_per_trade > 0.1:
            raise ValueError("Max risk per trade too high (>10%)")
        
        if self.trading.max_daily_loss > 0.2:
            raise ValueError("Max daily loss too high (>20%)")
        
        # Validate symbols
        if not self.trading.symbols:
            raise ValueError("No trading symbols configured")
    
    def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.system.data_dir, exist_ok=True)
        os.makedirs(self.system.models_dir, exist_ok=True)
        os.makedirs(self.system.logs_dir, exist_ok=True)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'trading': self.trading.__dict__,
            'exchange': {k: v for k, v in self.exchange.__dict__.items() 
                        if 'secret' not in k and 'key' not in k},
            'database': {k: v for k, v in self.database.__dict__.items() 
                        if 'url' not in k},
            'alerts': {k: v for k, v in self.alerts.__dict__.items() 
                      if 'token' not in k},
            'analysis': self.analysis.__dict__,
            'system': self.system.__dict__
        }
    
    def print_config(self):
        """Print configuration (for debugging)"""
        if self.system.debug_mode:
            print("\n" + "="*50)
            print("DEMIR AI v8.0 - Configuration")
            print("="*50)
            print(f"Environment: {self.system.environment}")
            print(f"Version: {self.system.version}")
            print(f"Advisory Mode: {self.system.advisory_mode}")
            print(f"Debug Mode: {self.system.debug_mode}")
            print(f"Trading Symbols: {', '.join(self.trading.symbols)}")
            print(f"Max Positions: {self.trading.max_positions}")
            print(f"Risk Per Trade: {self.trading.max_risk_per_trade*100:.1f}%")
            print(f"Analysis Layers: S={self.analysis.enable_sentiment}, "
                  f"T={self.analysis.enable_technical}, "
                  f"ML={self.analysis.enable_ml}")
            print("="*50 + "\n")

# Global configuration instance
config = Config()
