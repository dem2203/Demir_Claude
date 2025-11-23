"""
DEMIR AI v8.0 - PostgreSQL Database Client
REAL DATABASE OPERATIONS - NO MOCK DATA
ENTERPRISE GRADE DATABASE MANAGEMENT
"""

import logging
import asyncio
import asyncpg
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from contextlib import asynccontextmanager
import uuid

logger = logging.getLogger(__name__)


class PostgresClient:
    """
    PostgreSQL database client
    REAL DATABASE OPERATIONS - ZERO MOCK DATA
    """
    
    def __init__(self, config):
        self.config = config
        self.database_url = config.database.database_url
        self.pool_size = config.database.pool_size
        self.max_overflow = config.database.max_overflow
        self.pool_timeout = config.database.pool_timeout
        
        self.pool = None
        self.is_connected = False
        
        # Table names
        self.tables = {
            'signals': 'trading_signals',
            'trades': 'trades',
            'positions': 'positions',
            'candles': 'candles',
            'orders': 'orders',
            'balances': 'account_balances',
            'performance': 'performance_metrics',
            'alerts': 'alerts',
            'system_logs': 'system_logs',
            'ml_predictions': 'ml_predictions',
            'sentiment_data': 'sentiment_data',
            'technical_data': 'technical_data',
            'risk_metrics': 'risk_metrics',
            'backtest_results': 'backtest_results'
        }
        
        logger.info(f"PostgresClient initialized with pool size {self.pool_size}")
    
    async def connect(self):
        """
        Connect to PostgreSQL database
        REAL DATABASE CONNECTION
        """
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=10,
                max_size=self.pool_size,
                max_queries=50000,
                max_inactive_connection_lifetime=300,
                timeout=self.pool_timeout,
                command_timeout=60
            )
            
            self.is_connected = True
            
            # Create tables if not exist
            await self._create_tables()
            
            logger.info("Connected to PostgreSQL database")
            
            # Get database info
            async with self.pool.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                logger.info(f"PostgreSQL version: {version}")
                
                # Get database size
                db_size = await conn.fetchval(
                    "SELECT pg_size_pretty(pg_database_size(current_database()))"
                )
                logger.info(f"Database size: {db_size}")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from database"""
        if self.pool:
            await self.pool.close()
            self.is_connected = False
            logger.info("Disconnected from PostgreSQL")
    
    async def close(self):
        """Alias for disconnect"""
        await self.disconnect()
    
    @asynccontextmanager
    async def transaction(self):
        """
        Database transaction context manager
        REAL TRANSACTIONS WITH ROLLBACK
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    async def _create_tables(self):
        """
        Create database tables if not exist
        REAL TABLE CREATION
        """
        try:
            async with self.pool.acquire() as conn:
                # Trading signals table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS trading_signals (
                        id SERIAL PRIMARY KEY,
                        signal_id VARCHAR(100) UNIQUE NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        action VARCHAR(20) NOT NULL,
                        confidence FLOAT NOT NULL,
                        strength VARCHAR(20),
                        entry_price DECIMAL(20, 8) NOT NULL,
                        stop_loss DECIMAL(20, 8) NOT NULL,
                        take_profit_1 DECIMAL(20, 8),
                        take_profit_2 DECIMAL(20, 8),
                        take_profit_3 DECIMAL(20, 8),
                        position_size FLOAT,
                        risk_reward_ratio FLOAT,
                        technical_score FLOAT,
                        sentiment_score FLOAT,
                        volume_score FLOAT,
                        ml_score FLOAT,
                        reasons JSONB,
                        warnings JSONB,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_signals_symbol (symbol),
                        INDEX idx_signals_timestamp (timestamp)
                    )
                """)
                
                # Trades table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        trade_id VARCHAR(100) UNIQUE NOT NULL,
                        signal_id VARCHAR(100),
                        timestamp TIMESTAMP NOT NULL,
                        exchange VARCHAR(20) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        order_type VARCHAR(20) NOT NULL,
                        quantity DECIMAL(20, 8) NOT NULL,
                        price DECIMAL(20, 8) NOT NULL,
                        fee DECIMAL(20, 8),
                        fee_currency VARCHAR(10),
                        status VARCHAR(20),
                        order_id VARCHAR(100),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_trades_symbol (symbol),
                        INDEX idx_trades_timestamp (timestamp),
                        FOREIGN KEY (signal_id) REFERENCES trading_signals(signal_id)
                    )
                """)
                
                # Positions table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        id SERIAL PRIMARY KEY,
                        position_id VARCHAR(100) UNIQUE NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        exchange VARCHAR(20) NOT NULL,
                        position_type VARCHAR(10) NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        entry_price DECIMAL(20, 8) NOT NULL,
                        entry_quantity DECIMAL(20, 8) NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        current_price DECIMAL(20, 8),
                        current_quantity DECIMAL(20, 8),
                        exit_price DECIMAL(20, 8),
                        exit_time TIMESTAMP,
                        exit_reason VARCHAR(50),
                        realized_pnl DECIMAL(20, 8),
                        unrealized_pnl DECIMAL(20, 8),
                        total_fees DECIMAL(20, 8),
                        max_drawdown DECIMAL(20, 8),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_positions_symbol (symbol),
                        INDEX idx_positions_status (status)
                    )
                """)
                
                # Candles/OHLCV table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS candles (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        interval VARCHAR(10) NOT NULL,
                        open_time TIMESTAMP NOT NULL,
                        open DECIMAL(20, 8) NOT NULL,
                        high DECIMAL(20, 8) NOT NULL,
                        low DECIMAL(20, 8) NOT NULL,
                        close DECIMAL(20, 8) NOT NULL,
                        volume DECIMAL(20, 8) NOT NULL,
                        close_time TIMESTAMP NOT NULL,
                        quote_volume DECIMAL(20, 8),
                        trades INTEGER,
                        taker_buy_volume DECIMAL(20, 8),
                        taker_buy_quote_volume DECIMAL(20, 8),
                        UNIQUE(symbol, interval, open_time),
                        INDEX idx_candles_symbol_interval (symbol, interval),
                        INDEX idx_candles_time (open_time)
                    )
                """)
                
                # Orders table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id SERIAL PRIMARY KEY,
                        order_id VARCHAR(100) UNIQUE NOT NULL,
                        exchange VARCHAR(20) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        order_type VARCHAR(20) NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        quantity DECIMAL(20, 8) NOT NULL,
                        price DECIMAL(20, 8),
                        executed_quantity DECIMAL(20, 8),
                        average_price DECIMAL(20, 8),
                        stop_price DECIMAL(20, 8),
                        time_in_force VARCHAR(10),
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB,
                        INDEX idx_orders_symbol (symbol),
                        INDEX idx_orders_status (status)
                    )
                """)
                
                # Account balances table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS account_balances (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        exchange VARCHAR(20) NOT NULL,
                        asset VARCHAR(10) NOT NULL,
                        free DECIMAL(20, 8) NOT NULL,
                        locked DECIMAL(20, 8) NOT NULL,
                        total DECIMAL(20, 8) NOT NULL,
                        usd_value DECIMAL(20, 8),
                        btc_value DECIMAL(20, 8),
                        INDEX idx_balances_timestamp (timestamp),
                        INDEX idx_balances_exchange_asset (exchange, asset)
                    )
                """)
                
                # Performance metrics table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        metric_type VARCHAR(50) NOT NULL,
                        value DECIMAL(20, 8) NOT NULL,
                        period VARCHAR(20),
                        metadata JSONB,
                        INDEX idx_metrics_timestamp (timestamp),
                        INDEX idx_metrics_type (metric_type)
                    )
                """)
                
                # ML predictions table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS ml_predictions (
                        id SERIAL PRIMARY KEY,
                        prediction_id VARCHAR(100) UNIQUE NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        model_name VARCHAR(50) NOT NULL,
                        prediction DECIMAL(20, 8) NOT NULL,
                        confidence FLOAT NOT NULL,
                        actual_value DECIMAL(20, 8),
                        error DECIMAL(20, 8),
                        features JSONB,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_predictions_symbol (symbol),
                        INDEX idx_predictions_model (model_name)
                    )
                """)
                
                # Sentiment data table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS sentiment_data (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        source VARCHAR(50) NOT NULL,
                        sentiment_score FLOAT NOT NULL,
                        data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_sentiment_symbol (symbol),
                        INDEX idx_sentiment_timestamp (timestamp)
                    )
                """)
                
                # Technical indicators table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS technical_data (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        timeframe VARCHAR(10) NOT NULL,
                        rsi FLOAT,
                        macd JSONB,
                        bollinger_bands JSONB,
                        moving_averages JSONB,
                        volume_indicators JSONB,
                        other_indicators JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_technical_symbol (symbol),
                        INDEX idx_technical_timestamp (timestamp)
                    )
                """)
                
                # Alerts table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id SERIAL PRIMARY KEY,
                        alert_id VARCHAR(100) UNIQUE NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        alert_type VARCHAR(20) NOT NULL,
                        priority VARCHAR(20) NOT NULL,
                        title VARCHAR(200) NOT NULL,
                        message TEXT,
                        channels JSONB,
                        data JSONB,
                        sent BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_alerts_type (alert_type),
                        INDEX idx_alerts_timestamp (timestamp)
                    )
                """)
                
                # System logs table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        level VARCHAR(20) NOT NULL,
                        component VARCHAR(50) NOT NULL,
                        message TEXT NOT NULL,
                        data JSONB,
                        INDEX idx_logs_timestamp (timestamp),
                        INDEX idx_logs_level (level),
                        INDEX idx_logs_component (component)
                    )
                """)
                
                logger.info("Database tables created/verified")
                
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    async def save_signal(self, signal: Dict) -> bool:
        """
        Save trading signal to database
        REAL SIGNAL STORAGE
        """
        try:
            async with self.pool.acquire() as conn:
                signal_id = signal.get('signal_id', f"sig_{uuid.uuid4().hex[:8]}")
                
                await conn.execute("""
                    INSERT INTO trading_signals (
                        signal_id, timestamp, symbol, action, confidence, strength,
                        entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3,
                        position_size, risk_reward_ratio,
                        technical_score, sentiment_score, volume_score, ml_score,
                        reasons, warnings, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                    ON CONFLICT (signal_id) DO NOTHING
                """, 
                    signal_id,
                    signal.get('timestamp', datetime.now()),
                    signal['symbol'],
                    signal['action'],
                    signal['confidence'],
                    signal.get('strength', ''),
                    signal['entry_price'],
                    signal['stop_loss'],
                    signal.get('take_profit_1'),
                    signal.get('take_profit_2'),
                    signal.get('take_profit_3'),
                    signal.get('position_size'),
                    signal.get('risk_reward_ratio'),
                    signal.get('technical_score'),
                    signal.get('sentiment_score'),
                    signal.get('volume_score'),
                    signal.get('ml_score'),
                    json.dumps(signal.get('reasons', [])),
                    json.dumps(signal.get('warnings', [])),
                    json.dumps(signal.get('metadata', {}))
                )
                
                logger.debug(f"Signal saved: {signal_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            return False
    
    async def save_trade(self, trade: Dict) -> bool:
        """
        Save trade to database
        REAL TRADE STORAGE
        """
        try:
            async with self.pool.acquire() as conn:
                trade_id = trade.get('trade_id', f"trd_{uuid.uuid4().hex[:8]}")
                
                await conn.execute("""
                    INSERT INTO trades (
                        trade_id, signal_id, timestamp, exchange, symbol,
                        side, order_type, quantity, price, fee, fee_currency,
                        status, order_id, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (trade_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        metadata = EXCLUDED.metadata
                """,
                    trade_id,
                    trade.get('signal_id'),
                    trade.get('timestamp', datetime.now()),
                    trade['exchange'],
                    trade['symbol'],
                    trade['side'],
                    trade['order_type'],
                    trade['quantity'],
                    trade['price'],
                    trade.get('fee'),
                    trade.get('fee_currency'),
                    trade.get('status'),
                    trade.get('order_id'),
                    json.dumps(trade.get('metadata', {}))
                )
                
                logger.debug(f"Trade saved: {trade_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            return False
    
    async def save_position(self, position: Dict) -> bool:
        """
        Save or update position
        REAL POSITION STORAGE
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO positions (
                        position_id, symbol, exchange, position_type, status,
                        entry_price, entry_quantity, entry_time,
                        current_price, current_quantity,
                        exit_price, exit_time, exit_reason,
                        realized_pnl, unrealized_pnl, total_fees, max_drawdown,
                        metadata, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                    ON CONFLICT (position_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        current_price = EXCLUDED.current_price,
                        current_quantity = EXCLUDED.current_quantity,
                        exit_price = EXCLUDED.exit_price,
                        exit_time = EXCLUDED.exit_time,
                        exit_reason = EXCLUDED.exit_reason,
                        realized_pnl = EXCLUDED.realized_pnl,
                        unrealized_pnl = EXCLUDED.unrealized_pnl,
                        total_fees = EXCLUDED.total_fees,
                        max_drawdown = EXCLUDED.max_drawdown,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                """,
                    position['position_id'],
                    position['symbol'],
                    position['exchange'],
                    position['position_type'],
                    position['status'],
                    position['entry_price'],
                    position['entry_quantity'],
                    position['entry_time'],
                    position.get('current_price'),
                    position.get('current_quantity'),
                    position.get('exit_price'),
                    position.get('exit_time'),
                    position.get('exit_reason'),
                    position.get('realized_pnl', 0),
                    position.get('unrealized_pnl', 0),
                    position.get('total_fees', 0),
                    position.get('max_drawdown', 0),
                    json.dumps(position.get('metadata', {})),
                    datetime.now()
                )
                
                logger.debug(f"Position saved: {position['position_id']}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving position: {e}")
            return False
    
    async def save_candles(self, symbol: str, interval: str, candles: List[Dict]) -> int:
        """
        Save multiple candles to database
        BULK CANDLE STORAGE
        """
        if not candles:
            return 0
        
        try:
            async with self.pool.acquire() as conn:
                # Prepare data for bulk insert
                values = []
                for candle in candles:
                    values.append((
                        symbol,
                        interval,
                        candle['open_time'],
                        candle['open'],
                        candle['high'],
                        candle['low'],
                        candle['close'],
                        candle['volume'],
                        candle['close_time'],
                        candle.get('quote_volume'),
                        candle.get('trades'),
                        candle.get('taker_buy_volume'),
                        candle.get('taker_buy_quote_volume')
                    ))
                
                # Bulk insert with ON CONFLICT
                result = await conn.executemany("""
                    INSERT INTO candles (
                        symbol, interval, open_time, open, high, low, close, volume,
                        close_time, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (symbol, interval, open_time) DO NOTHING
                """, values)
                
                count = int(result.split()[-1]) if result else 0
                logger.debug(f"Saved {count} candles for {symbol} ({interval})")
                return count
                
        except Exception as e:
            logger.error(f"Error saving candles: {e}")
            return 0
    
    async def get_signals(self, symbol: Optional[str] = None, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         limit: int = 100) -> List[Dict]:
        """
        Get signals from database
        REAL SIGNAL RETRIEVAL
        """
        try:
            async with self.pool.acquire() as conn:
                query = "SELECT * FROM trading_signals WHERE 1=1"
                params = []
                param_count = 0
                
                if symbol:
                    param_count += 1
                    query += f" AND symbol = ${param_count}"
                    params.append(symbol)
                
                if start_time:
                    param_count += 1
                    query += f" AND timestamp >= ${param_count}"
                    params.append(start_time)
                
                if end_time:
                    param_count += 1
                    query += f" AND timestamp <= ${param_count}"
                    params.append(end_time)
                
                query += f" ORDER BY timestamp DESC LIMIT {limit}"
                
                rows = await conn.fetch(query, *params)
                
                signals = []
                for row in rows:
                    signal = dict(row)
                    # Parse JSON fields
                    signal['reasons'] = json.loads(signal.get('reasons', '[]'))
                    signal['warnings'] = json.loads(signal.get('warnings', '[]'))
                    signal['metadata'] = json.loads(signal.get('metadata', '{}'))
                    signals.append(signal)
                
                return signals
                
        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            return []
    
    async def get_positions(self, status: Optional[str] = None,
                           symbol: Optional[str] = None) -> List[Dict]:
        """
        Get positions from database
        REAL POSITION RETRIEVAL
        """
        try:
            async with self.pool.acquire() as conn:
                query = "SELECT * FROM positions WHERE 1=1"
                params = []
                param_count = 0
                
                if status:
                    param_count += 1
                    query += f" AND status = ${param_count}"
                    params.append(status)
                
                if symbol:
                    param_count += 1
                    query += f" AND symbol = ${param_count}"
                    params.append(symbol)
                
                query += " ORDER BY updated_at DESC"
                
                rows = await conn.fetch(query, *params)
                
                positions = []
                for row in rows:
                    position = dict(row)
                    position['metadata'] = json.loads(position.get('metadata', '{}'))
                    positions.append(position)
                
                return positions
                
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    async def get_performance_metrics(self, metric_type: Optional[str] = None,
                                     period: Optional[str] = None,
                                     start_time: Optional[datetime] = None) -> List[Dict]:
        """
        Get performance metrics
        REAL METRICS RETRIEVAL
        """
        try:
            async with self.pool.acquire() as conn:
                query = "SELECT * FROM performance_metrics WHERE 1=1"
                params = []
                param_count = 0
                
                if metric_type:
                    param_count += 1
                    query += f" AND metric_type = ${param_count}"
                    params.append(metric_type)
                
                if period:
                    param_count += 1
                    query += f" AND period = ${param_count}"
                    params.append(period)
                
                if start_time:
                    param_count += 1
                    query += f" AND timestamp >= ${param_count}"
                    params.append(start_time)
                
                query += " ORDER BY timestamp DESC LIMIT 1000"
                
                rows = await conn.fetch(query, *params)
                
                metrics = []
                for row in rows:
                    metric = dict(row)
                    metric['metadata'] = json.loads(metric.get('metadata', '{}'))
                    metrics.append(metric)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return []
    
    async def save_performance_metric(self, metric_type: str, value: float,
                                     period: Optional[str] = None,
                                     metadata: Optional[Dict] = None) -> bool:
        """
        Save performance metric
        REAL METRIC STORAGE
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO performance_metrics (
                        timestamp, metric_type, value, period, metadata
                    ) VALUES ($1, $2, $3, $4, $5)
                """,
                    datetime.now(),
                    metric_type,
                    value,
                    period,
                    json.dumps(metadata or {})
                )
                
                logger.debug(f"Performance metric saved: {metric_type} = {value}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving performance metric: {e}")
            return False
    
    async def cleanup_old_data(self, days: int = 30) -> int:
        """
        Clean up old data from database
        REAL DATA CLEANUP
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            total_deleted = 0
            
            async with self.pool.acquire() as conn:
                # Clean old logs
                result = await conn.execute(
                    "DELETE FROM system_logs WHERE timestamp < $1",
                    cutoff_date
                )
                count = int(result.split()[-1]) if result else 0
                total_deleted += count
                logger.info(f"Deleted {count} old system logs")
                
                # Clean old candles (keep only last 90 days)
                candle_cutoff = datetime.now() - timedelta(days=90)
                result = await conn.execute(
                    "DELETE FROM candles WHERE open_time < $1",
                    candle_cutoff
                )
                count = int(result.split()[-1]) if result else 0
                total_deleted += count
                logger.info(f"Deleted {count} old candles")
                
                # Clean old alerts
                result = await conn.execute(
                    "DELETE FROM alerts WHERE timestamp < $1 AND sent = true",
                    cutoff_date
                )
                count = int(result.split()[-1]) if result else 0
                total_deleted += count
                logger.info(f"Deleted {count} old alerts")
            
            return total_deleted
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0
    
    async def get_statistics(self) -> Dict:
        """
        Get database statistics
        REAL DATABASE STATS
        """
        try:
            async with self.pool.acquire() as conn:
                stats = {}
                
                # Table row counts
                for table_name, table in self.tables.items():
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table_name}_count"] = count
                
                # Database size
                stats['database_size'] = await conn.fetchval(
                    "SELECT pg_size_pretty(pg_database_size(current_database()))"
                )
                
                # Connection stats
                stats['active_connections'] = self.pool.get_size()
                stats['idle_connections'] = self.pool.get_idle_size()
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {}
    
    async def execute_query(self, query: str, *params) -> List[Dict]:
        """
        Execute custom query
        REAL QUERY EXECUTION
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
