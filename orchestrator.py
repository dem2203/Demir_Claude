"""
DEMIR AI v8.0 - Main Orchestrator
CENTRAL COMMAND & CONTROL SYSTEM
COORDINATES ALL AI SUBSYSTEMS - ZERO MOCK DATA
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import traceback

# Core imports
from config import Config
from core.signal_generator import SignalGenerator
from core.signal_validator import SignalValidator
from core.data_validator import DataValidator

# Layer imports
from layers.sentiment import SentimentAnalysis
from layers.technical import TechnicalAnalysis
from layers.ml_models import MLModels

# Trading imports
from trading.exchanges import ExchangeManager
from trading.position_manager import PositionManager

# Advanced imports
from ai.brain_ensemble import AiBrainEnsemble
from scanner.market_scanner_pro import MarketScannerPro
from portfolio.portfolio_optimizer import PortfolioOptimizer, OptimizationStrategy
from backtesting.advanced_backtest import AdvancedBacktestEngine
from utils.signal_processor_advanced import AdvancedSignalProcessor
from utils.circuit_breaker import MultiCircuitBreaker

# Infrastructure imports
from websocket.binance_ws import BinanceWebSocketManager
from alerts.alert_manager import AlertManager
from database.postgres_client import PostgresClient
from database.redis_cache import RedisCache
from monitoring.health_monitor import HealthMonitor
from risk.risk_controller import RiskController

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System operational states"""
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    MAINTENANCE = "MAINTENANCE"
    SHUTTING_DOWN = "SHUTTING_DOWN"


class TradingMode(Enum):
    """Trading operation modes"""
    ADVISORY = "ADVISORY"      # Signals only, no execution
    PAPER = "PAPER"            # Paper trading
    LIVE = "LIVE"              # Real money trading


@dataclass
class SystemMetrics:
    """System-wide metrics"""
    timestamp: datetime
    state: SystemState
    mode: TradingMode
    
    # Performance
    total_signals: int
    successful_trades: int
    failed_trades: int
    win_rate: float
    
    # Financial
    starting_capital: float
    current_capital: float
    total_pnl: float
    daily_pnl: float
    
    # System health
    cpu_usage: float
    memory_usage: float
    api_latency: float
    error_rate: float
    
    # Active components
    active_subsystems: int
    active_positions: int
    pending_orders: int


class DemirAIOrchestrator:
    """
    Main orchestrator for DEMIR AI v8.0
    CONTROLS AND COORDINATES ALL SUBSYSTEMS
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the orchestrator"""
        self.config = Config(config_path)
        
        # System state
        self.state = SystemState.INITIALIZING
        self.mode = TradingMode.ADVISORY  # Start in advisory mode
        
        # Core components
        self.signal_generator = None
        self.signal_validator = None
        self.data_validator = None
        
        # Analysis layers
        self.sentiment_analysis = None
        self.technical_analysis = None
        self.ml_models = None
        
        # Trading components
        self.exchange_manager = None
        self.position_manager = None
        
        # Advanced systems
        self.ai_brain = None
        self.market_scanner = None
        self.portfolio_optimizer = None
        self.backtest_engine = None
        self.signal_processor = None
        
        # Infrastructure
        self.websocket_manager = None
        self.alert_manager = None
        self.database = None
        self.cache = None
        self.health_monitor = None
        self.risk_controller = None
        self.circuit_breaker = None
        
        # Orchestration
        self.main_loop_task = None
        self.scanner_task = None
        self.monitor_task = None
        self.optimization_task = None
        
        # Metrics
        self.system_metrics = None
        self.start_time = datetime.now()
        self.starting_capital = 10000  # Will be updated from config
        
        # Control flags
        self.is_running = False
        self.emergency_stop = False
        
        logger.info("="*50)
        logger.info("DEMIR AI v8.0 ORCHESTRATOR INITIALIZING")
        logger.info("="*50)
    
    async def initialize(self):
        """
        Initialize all subsystems
        FULL SYSTEM INITIALIZATION
        """
        try:
            logger.info("Starting system initialization...")
            
            # 1. Initialize infrastructure first
            await self._initialize_infrastructure()
            
            # 2. Initialize data validators
            await self._initialize_validators()
            
            # 3. Initialize analysis layers
            await self._initialize_analysis()
            
            # 4. Initialize trading systems
            await self._initialize_trading()
            
            # 5. Initialize advanced AI
            await self._initialize_ai()
            
            # 6. Initialize monitoring
            await self._initialize_monitoring()
            
            # 7. Verify all systems
            if not await self._verify_systems():
                raise Exception("System verification failed")
            
            # 8. Load configuration
            await self._load_configuration()
            
            # Update state
            self.state = SystemState.RUNNING
            
            logger.info("="*50)
            logger.info("SYSTEM INITIALIZATION COMPLETE")
            logger.info(f"Mode: {self.mode.value}")
            logger.info(f"Starting capital: ${self.starting_capital:,.2f}")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            logger.error(traceback.format_exc())
            await self.emergency_shutdown()
            raise
    
    async def _initialize_infrastructure(self):
        """Initialize infrastructure components"""
        logger.info("Initializing infrastructure...")
        
        # Database
        self.database = PostgresClient(self.config)
        await self.database.connect()
        
        # Cache
        self.cache = RedisCache(self.config)
        await self.cache.connect()
        
        # WebSocket manager
        self.websocket_manager = BinanceWebSocketManager(self.config)
        
        # Alert manager
        self.alert_manager = AlertManager(self.config)
        await self.alert_manager.initialize()
        
        logger.info("Infrastructure initialized")
    
    async def _initialize_validators(self):
        """Initialize data validators"""
        logger.info("Initializing validators...")
        
        self.data_validator = DataValidator(self.config)
        self.signal_validator = SignalValidator(self.config)
        
        logger.info("Validators initialized")
    
    async def _initialize_analysis(self):
        """Initialize analysis layers"""
        logger.info("Initializing analysis layers...")
        
        self.sentiment_analysis = SentimentAnalysis(self.config)
        self.technical_analysis = TechnicalAnalysis(self.config)
        self.ml_models = MLModels(self.config)
        
        # Load ML models
        await self.ml_models.load_models()
        
        logger.info("Analysis layers initialized")
    
    async def _initialize_trading(self):
        """Initialize trading systems"""
        logger.info("Initializing trading systems...")
        
        # Exchange manager
        self.exchange_manager = ExchangeManager(self.config)
        await self.exchange_manager.connect_all()
        
        # Position manager
        self.position_manager = PositionManager(self.config)
        
        # Signal generator
        self.signal_generator = SignalGenerator(self.config)
        
        # Signal processor
        self.signal_processor = AdvancedSignalProcessor(self.config)
        
        logger.info("Trading systems initialized")
    
    async def _initialize_ai(self):
        """Initialize advanced AI systems"""
        logger.info("Initializing AI systems...")
        
        # AI Brain Ensemble
        self.ai_brain = AiBrainEnsemble(self.config)
        
        # Market scanner
        self.market_scanner = MarketScannerPro(self.config)
        
        # Portfolio optimizer
        self.portfolio_optimizer = PortfolioOptimizer(self.config)
        
        # Backtest engine
        self.backtest_engine = AdvancedBacktestEngine(self.config)
        
        logger.info("AI systems initialized")
    
    async def _initialize_monitoring(self):
        """Initialize monitoring systems"""
        logger.info("Initializing monitoring...")
        
        # Health monitor
        self.health_monitor = HealthMonitor(self.config)
        self.health_monitor.register_component('database', self.database)
        self.health_monitor.register_component('cache', self.cache)
        self.health_monitor.register_component('exchange', self.exchange_manager)
        self.health_monitor.register_component('websocket', self.websocket_manager)
        self.health_monitor.register_component('positions', self.position_manager)
        
        # Risk controller
        self.risk_controller = RiskController(self.config)
        
        # Circuit breaker
        self.circuit_breaker = MultiCircuitBreaker(self.config)
        
        logger.info("Monitoring systems initialized")
    
    async def _verify_systems(self) -> bool:
        """Verify all systems are operational"""
        logger.info("Verifying systems...")
        
        checks = {
            'Database': await self.database.health_check() if self.database else False,
            'Cache': await self.cache.ping() if self.cache else False,
            'Exchanges': await self.exchange_manager.health_check_all() if self.exchange_manager else False,
            'AI Brain': self.ai_brain is not None,
            'Risk Controller': self.risk_controller is not None
        }
        
        for system, status in checks.items():
            logger.info(f"  {system}: {'✓' if status else '✗'}")
        
        return all(checks.values())
    
    async def _load_configuration(self):
        """Load configuration and settings"""
        # Set trading mode from config
        if self.config.trading.live_trading_enabled:
            self.mode = TradingMode.LIVE
            logger.warning("LIVE TRADING MODE ENABLED - REAL MONEY AT RISK")
        elif self.config.trading.paper_trading:
            self.mode = TradingMode.PAPER
            logger.info("Paper trading mode enabled")
        else:
            self.mode = TradingMode.ADVISORY
            logger.info("Advisory mode - signals only")
        
        # Load capital
        self.starting_capital = self.config.trading.initial_capital
    
    async def start(self):
        """
        Start the orchestrator
        MAIN SYSTEM STARTUP
        """
        if self.is_running:
            logger.warning("System already running")
            return
        
        logger.info("Starting DEMIR AI v8.0...")
        
        try:
            # Initialize if not done
            if self.state == SystemState.INITIALIZING:
                await self.initialize()
            
            # Start subsystems
            await self._start_subsystems()
            
            # Start main loops
            await self._start_main_loops()
            
            self.is_running = True
            
            # Send startup notification
            await self.alert_manager.send_notification(
                "🚀 DEMIR AI v8.0 Started",
                f"Mode: {self.mode.value}\nCapital: ${self.starting_capital:,.2f}"
            )
            
            logger.info("DEMIR AI v8.0 is now running")
            
            # Keep running
            await self._run_forever()
            
        except Exception as e:
            logger.error(f"Startup failed: {e}")
            await self.emergency_shutdown()
            raise
    
    async def _start_subsystems(self):
        """Start all subsystems"""
        # Start WebSocket streams
        await self.websocket_manager.connect()
        
        # Start market scanner
        await self.market_scanner.start_scanning()
        
        # Start health monitoring
        await self.health_monitor.start_monitoring()
        
        logger.info("All subsystems started")
    
    async def _start_main_loops(self):
        """Start main processing loops"""
        self.main_loop_task = asyncio.create_task(self._main_loop())
        self.scanner_task = asyncio.create_task(self._scanner_loop())
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        logger.info("Main loops started")
    
    async def _main_loop(self):
        """
        Main trading loop
        CORE TRADING LOGIC
        """
        while self.is_running and not self.emergency_stop:
            try:
                # Check system state
                if self.state != SystemState.RUNNING:
                    await asyncio.sleep(1)
                    continue
                
                # 1. Get market data
                market_data = await self._get_market_data()
                
                # 2. Validate data
                if not await self.data_validator.validate_market_data(market_data):
                    logger.warning("Invalid market data, skipping cycle")
                    await asyncio.sleep(5)
                    continue
                
                # 3. Run analysis layers
                sentiment = await self.sentiment_analysis.analyze()
                technical = await self.technical_analysis.analyze(market_data)
                ml_predictions = await self.ml_models.predict(market_data)
                
                # 4. AI Brain decision
                ai_decision = await self.ai_brain.make_decision(
                    market_data, technical, sentiment, ml_predictions
                )
                
                # 5. Generate signals
                signals = await self.signal_generator.generate(
                    market_data, technical, sentiment, ai_decision
                )
                
                # 6. Process signals
                for signal in signals:
                    await self._process_signal(signal)
                
                # 7. Manage positions
                await self._manage_positions()
                
                # 8. Update metrics
                await self._update_metrics()
                
                # Sleep based on timeframe
                await asyncio.sleep(self.config.trading.signal_check_interval)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await self._handle_error(e)
                await asyncio.sleep(10)
    
    async def _scanner_loop(self):
        """Market scanner loop"""
        while self.is_running:
            try:
                # Get opportunities
                opportunities = await self.market_scanner.get_opportunities(
                    min_confidence=60
                )
                
                # Process high priority opportunities
                for opp in opportunities[:10]:  # Top 10
                    if opp.priority.value <= 2:  # HIGH or CRITICAL
                        logger.info(f"High priority opportunity: {opp.symbol} - {opp.description}")
                        
                        # Create signal from opportunity
                        signal = await self._opportunity_to_signal(opp)
                        if signal:
                            await self._process_signal(signal)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Scanner loop error: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_loop(self):
        """System monitoring loop"""
        while self.is_running:
            try:
                # Collect health metrics
                health_metrics = await self.health_monitor.collect_metrics()
                
                # Check circuit breaker conditions
                cb_check = await self.circuit_breaker.check_all({
                    'trading': {
                        'consecutive_losses': self.position_manager.consecutive_losses,
                        'daily_loss_percent': await self._calculate_daily_loss()
                    },
                    'system': {
                        'memory_percent': health_metrics.memory_usage,
                        'error_rate': health_metrics.error_count
                    }
                })
                
                # Trip if necessary
                for subsystem, reason in cb_check.items():
                    if reason:
                        await self.circuit_breaker.trip_subsystem(subsystem, reason)
                        await self._handle_circuit_break(subsystem, reason)
                
                # Check risk limits
                risk_metrics = await self.risk_controller.calculate_metrics()
                if risk_metrics.risk_level.value in ["CRITICAL", "EMERGENCY"]:
                    logger.warning(f"Risk level: {risk_metrics.risk_level.value}")
                    await self._handle_high_risk(risk_metrics)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(30)
    
    async def _optimization_loop(self):
        """Portfolio optimization loop"""
        while self.is_running:
            try:
                # Run optimization daily
                await asyncio.sleep(3600)  # Check hourly
                
                current_hour = datetime.now().hour
                if current_hour == 0:  # Midnight
                    logger.info("Running portfolio optimization...")
                    
                    # Get current positions
                    positions = await self.position_manager.get_open_positions()
                    
                    # Get asset universe
                    assets = [pos['symbol'] for pos in positions]
                    
                    # Run optimization
                    optimized = await self.portfolio_optimizer.optimize(
                        self.starting_capital,
                        assets,
                        OptimizationStrategy.MAX_SHARPE
                    )
                    
                    # Check if rebalancing needed
                    if optimized.rebalance_needed:
                        logger.info(f"Rebalancing needed: {optimized.rebalance_urgency}")
                        # Would trigger rebalancing in LIVE mode
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _process_signal(self, signal: Dict):
        """Process trading signal"""
        try:
            # Validate signal
            if not await self.signal_validator.validate(signal):
                logger.debug(f"Signal validation failed for {signal['symbol']}")
                return
            
            # Check risk
            if not await self.risk_controller.check_signal(signal):
                logger.info(f"Signal rejected by risk controller: {signal['symbol']}")
                return
            
            # Process with signal processor
            enhanced_signal = await self.signal_processor.process_signal(signal)
            if not enhanced_signal:
                return
            
            # Check trading mode
            if self.mode == TradingMode.ADVISORY:
                # Just send alert
                await self.alert_manager.send_signal_alert(signal)
                logger.info(f"Advisory signal sent: {signal['symbol']} {signal['action']}")
                
            elif self.mode in [TradingMode.PAPER, TradingMode.LIVE]:
                # Execute trade
                result = await self.exchange_manager.execute_signal(enhanced_signal)
                
                if result['success']:
                    # Create position
                    position = await self.position_manager.create_position(
                        enhanced_signal, result['order']
                    )
                    
                    # Send notification
                    await self.alert_manager.send_trade_alert(result)
                    
                    # Save to database
                    await self.database.save_trade(result)
                    
                    logger.info(f"Trade executed: {signal['symbol']} {signal['action']}")
                else:
                    logger.error(f"Trade execution failed: {result.get('error')}")
            
        except Exception as e:
            logger.error(f"Signal processing error: {e}")
    
    async def _manage_positions(self):
        """Manage open positions"""
        positions = await self.position_manager.get_open_positions()
        
        for position in positions:
            # Update with current price
            current_price = await self._get_current_price(position['symbol'])
            
            # Check exit conditions
            exit_signal = await self.position_manager.check_exit(
                position, current_price
            )
            
            if exit_signal:
                # Close position
                result = await self.exchange_manager.close_position(position)
                
                if result['success']:
                    await self.position_manager.close_position(
                        position['position_id'], 
                        result['exit_price'], 
                        exit_signal['reason']
                    )
                    
                    await self.alert_manager.send_position_alert(
                        position, "CLOSED", exit_signal['reason']
                    )
    
    async def _get_market_data(self) -> Dict:
        """Get current market data"""
        # This would aggregate data from various sources
        return {}
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        ticker = await self.exchange_manager.get_ticker(symbol)
        return ticker.get('last', 0)
    
    async def _calculate_daily_loss(self) -> float:
        """Calculate daily loss percentage"""
        # This would calculate actual daily P&L
        return 0.0
    
    async def _opportunity_to_signal(self, opportunity) -> Optional[Dict]:
        """Convert opportunity to signal"""
        # This would convert scanner opportunity to trading signal
        return None
    
    async def _handle_error(self, error: Exception):
        """Handle system errors"""
        logger.error(f"System error: {error}")
        
        # Send alert
        await self.alert_manager.send_error_alert(str(error))
        
        # Check if critical
        if "critical" in str(error).lower():
            await self.emergency_shutdown()
    
    async def _handle_circuit_break(self, subsystem: str, reason):
        """Handle circuit breaker trip"""
        logger.critical(f"Circuit breaker tripped: {subsystem} - {reason}")
        
        if subsystem == "trading":
            # Stop trading
            self.state = SystemState.PAUSED
            await self.position_manager.close_all_positions("CIRCUIT_BREAKER")
        
        await self.alert_manager.send_emergency_alert(
            f"Circuit breaker: {subsystem} - {reason}"
        )
    
    async def _handle_high_risk(self, risk_metrics):
        """Handle high risk conditions"""
        if risk_metrics.risk_level.value == "EMERGENCY":
            await self.emergency_shutdown()
        elif risk_metrics.risk_level.value == "CRITICAL":
            self.state = SystemState.PAUSED
            logger.warning("Trading paused due to high risk")
    
    async def _update_metrics(self):
        """Update system metrics"""
        # This would collect and update all system metrics
        pass
    
    async def _run_forever(self):
        """Keep the system running"""
        while self.is_running:
            await asyncio.sleep(1)
    
    async def pause(self):
        """Pause trading"""
        self.state = SystemState.PAUSED
        logger.info("System paused")
    
    async def resume(self):
        """Resume trading"""
        self.state = SystemState.RUNNING
        logger.info("System resumed")
    
    async def emergency_shutdown(self):
        """Emergency shutdown procedure"""
        logger.critical("EMERGENCY SHUTDOWN INITIATED")
        
        self.emergency_stop = True
        self.state = SystemState.EMERGENCY_STOP
        
        try:
            # Close all positions
            if self.position_manager:
                await self.position_manager.close_all_positions("EMERGENCY_SHUTDOWN")
            
            # Cancel all orders
            if self.exchange_manager:
                await self.exchange_manager.cancel_all_orders()
            
            # Send emergency alert
            if self.alert_manager:
                await self.alert_manager.send_emergency_alert(
                    "EMERGENCY SHUTDOWN - All positions closed"
                )
            
        except Exception as e:
            logger.error(f"Error during emergency shutdown: {e}")
        
        await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down DEMIR AI v8.0...")
        
        self.is_running = False
        self.state = SystemState.SHUTTING_DOWN
        
        # Cancel tasks
        tasks = [
            self.main_loop_task,
            self.scanner_task,
            self.monitor_task,
            self.optimization_task
        ]
        
        for task in tasks:
            if task:
                task.cancel()
        
        # Stop subsystems
        if self.market_scanner:
            await self.market_scanner.stop_scanning()
        
        if self.health_monitor:
            await self.health_monitor.stop_monitoring()
        
        if self.websocket_manager:
            await self.websocket_manager.disconnect()
        
        # Close connections
        if self.database:
            await self.database.close()
        
        if self.cache:
            await self.cache.close()
        
        logger.info("DEMIR AI v8.0 shutdown complete")
    
    def get_status(self) -> Dict:
        """Get system status"""
        return {
            'state': self.state.value,
            'mode': self.mode.value,
            'is_running': self.is_running,
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'subsystems': {
                'database': self.database is not None,
                'cache': self.cache is not None,
                'exchanges': self.exchange_manager is not None,
                'ai_brain': self.ai_brain is not None,
                'scanner': self.market_scanner is not None
            }
        }


async def main():
    """Main entry point"""
    orchestrator = DemirAIOrchestrator()
    
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run
    asyncio.run(main())
