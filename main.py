#!/usr/bin/env python3
"""
🚀 DEMIR AI v8.0 - Main Orchestrator
Professional Cryptocurrency Trading Bot
Railway Production Ready
"""

import os
import sys
import asyncio
import logging
import signal
import traceback
from datetime import datetime
from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.system.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.system.log_file_path) if config.system.log_to_file else logging.NullHandler()
    ]
)
logger = logging.getLogger('DEMIR_AI')

# Suppress verbose logs
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('ccxt').setLevel(logging.WARNING)

class DEMIRAIOrchestrator:
    """Main orchestrator for DEMIR AI Trading Bot"""
    
    def __init__(self):
        """Initialize the orchestrator"""
        self.is_running = False
        self.start_time = datetime.now()
        self.components = {}
        self.tasks = []
        
        # Print startup banner
        self._print_banner()
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info("✅ DEMIR AI Orchestrator initialized successfully")
    
    def _print_banner(self):
        """Print startup banner"""
        banner = f"""
        ╔══════════════════════════════════════════════════════════╗
        ║                 🚀 DEMIR AI v{config.system.version}                    ║
        ╠══════════════════════════════════════════════════════════╣
        ║  Environment: {config.system.environment:<20}              ║
        ║  Advisory Mode: {str(config.system.advisory_mode):<18}              ║
        ║  Debug Mode: {str(config.system.debug_mode):<21}              ║
        ║  Trading Symbols: {len(config.trading.symbols)} active                       ║
        ╚══════════════════════════════════════════════════════════╝
        """
        print(banner)
        logger.info(f"Starting DEMIR AI v{config.system.version}")
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Core components
            from core.signal_generator import SignalGenerator
            from core.signal_validator import SignalValidator
            from core.data_validator import DataValidator
            
            self.components['signal_generator'] = SignalGenerator(config)
            self.components['signal_validator'] = SignalValidator(config)
            self.components['data_validator'] = DataValidator(config)
            logger.info("✅ Core components initialized")
            
            # Analysis layers
            if config.analysis.enable_sentiment:
                from layers.sentiment import SentimentAnalyzer
                self.components['sentiment'] = SentimentAnalyzer(config)
                logger.info("✅ Sentiment analyzer initialized")
            
            if config.analysis.enable_technical:
                from layers.technical import TechnicalAnalyzer
                self.components['technical'] = TechnicalAnalyzer(config)
                logger.info("✅ Technical analyzer initialized")
            
            if config.analysis.enable_ml:
                from layers.ml_models import MLPredictor
                self.components['ml'] = MLPredictor(config)
                logger.info("✅ ML models initialized")
            
            # Trading components
            if not config.system.advisory_mode:
                from trading.exchanges import ExchangeManager
                from trading.position_manager import PositionManager
                
                self.components['exchange'] = ExchangeManager(config)
                self.components['positions'] = PositionManager(config)
                logger.info("✅ Trading components initialized")
            
            # Risk management
            from risk.risk_controller import RiskController
            self.components['risk'] = RiskController(config)
            logger.info("✅ Risk controller initialized")
            
            # Alert system
            if config.alerts.telegram_enabled or config.alerts.discord_enabled:
                from alerts.alert_manager import AlertManager
                self.components['alerts'] = AlertManager(config)
                logger.info("✅ Alert system initialized")
            
            # Database
            from database.postgres_client import PostgresClient
            from database.redis_cache import RedisCache
            
            self.components['database'] = PostgresClient(config)
            self.components['cache'] = RedisCache(config)
            logger.info("✅ Database connections established")
            
            # Monitoring
            from monitoring.health_monitor import HealthMonitor
            from monitoring.performance import PerformanceTracker
            
            self.components['health'] = HealthMonitor(config)
            self.components['performance'] = PerformanceTracker(config)
            logger.info("✅ Monitoring systems initialized")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _setup_signal_handlers(self):
        """Setup system signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"⚠️ Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self):
        """Main execution loop"""
        logger.info("🚀 Starting main execution loop")
        self.is_running = True
        
        try:
            # Initialize components
            await self._start_components()
            
            # Send startup notification
            if 'alerts' in self.components:
                await self.components['alerts'].send_notification(
                    f"✅ DEMIR AI v{config.system.version} started\n"
                    f"Environment: {config.system.environment}\n"
                    f"Advisory: {config.system.advisory_mode}\n"
                    f"Symbols: {', '.join(config.trading.symbols)}"
                )
            
            # Create main tasks
            self.tasks = [
                asyncio.create_task(self._market_analysis_loop()),
                asyncio.create_task(self._signal_generation_loop()),
                asyncio.create_task(self._risk_monitoring_loop()),
                asyncio.create_task(self._health_monitoring_loop()),
            ]
            
            if not config.system.advisory_mode:
                self.tasks.append(asyncio.create_task(self._trading_loop()))
            
            # Wait for shutdown
            while self.is_running:
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ Critical error in main loop: {e}")
            logger.error(traceback.format_exc())
        finally:
            await self.cleanup()
    
    async def _start_components(self):
        """Start all async components"""
        # Connect to database
        if 'database' in self.components:
            await self.components['database'].connect()
        
        if 'cache' in self.components:
            await self.components['cache'].connect()
        
        logger.info("✅ All components started")
    
    async def _market_analysis_loop(self):
        """Continuous market analysis"""
        logger.info("🔍 Market analysis loop started")
        
        while self.is_running:
            try:
                for symbol in config.trading.symbols:
                    # Validate data first
                    if 'data_validator' in self.components:
                        is_valid = await self.components['data_validator'].validate(symbol)
                        if not is_valid:
                            logger.warning(f"Invalid data for {symbol}, skipping")
                            continue
                    
                    analysis_results = {}
                    
                    # Sentiment analysis
                    if 'sentiment' in self.components:
                        sentiment = await self.components['sentiment'].analyze(symbol)
                        analysis_results['sentiment'] = sentiment
                    
                    # Technical analysis
                    if 'technical' in self.components:
                        technical = await self.components['technical'].analyze(symbol)
                        analysis_results['technical'] = technical
                    
                    # ML predictions
                    if 'ml' in self.components:
                        predictions = await self.components['ml'].predict(symbol)
                        analysis_results['ml'] = predictions
                    
                    # Cache results
                    if 'cache' in self.components:
                        await self.components['cache'].set(
                            f"analysis:{symbol}",
                            analysis_results,
                            ttl=60
                        )
                    
                    logger.debug(f"Analysis completed for {symbol}")
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Market analysis error: {e}")
                await asyncio.sleep(60)
    
    async def _signal_generation_loop(self):
        """Generate and validate trading signals"""
        logger.info("📊 Signal generation loop started")
        
        while self.is_running:
            try:
                for symbol in config.trading.symbols:
                    # Get cached analysis
                    if 'cache' in self.components:
                        analysis = await self.components['cache'].get(f"analysis:{symbol}")
                        if not analysis:
                            continue
                    
                    # Generate signal
                    if 'signal_generator' in self.components:
                        signal = await self.components['signal_generator'].generate(
                            symbol, analysis
                        )
                        
                        if signal and signal['confidence'] >= config.trading.min_signal_confidence:
                            # Validate signal
                            if 'signal_validator' in self.components:
                                is_valid = await self.components['signal_validator'].validate(signal)
                                
                                if is_valid:
                                    # Check risk
                                    if 'risk' in self.components:
                                        risk_approved = await self.components['risk'].check_signal(signal)
                                        
                                        if risk_approved:
                                            await self._process_signal(signal)
                                        else:
                                            logger.info(f"Signal rejected by risk controller: {symbol}")
                
                await asyncio.sleep(config.trading.signal_check_interval)
                
            except Exception as e:
                logger.error(f"Signal generation error: {e}")
                await asyncio.sleep(30)
    
    async def _process_signal(self, signal: Dict):
        """Process validated signal"""
        symbol = signal['symbol']
        action = signal['action']
        confidence = signal['confidence']
        
        logger.info(f"🎯 Signal: {symbol} {action} (Confidence: {confidence:.1f}%)")
        
        # Send alert
        if 'alerts' in self.components:
            await self.components['alerts'].send_signal_alert(signal)
        
        # Execute trade if not in advisory mode
        if not config.system.advisory_mode and 'exchange' in self.components:
            await self.components['exchange'].execute_signal(signal)
        
        # Save to database
        if 'database' in self.components:
            await self.components['database'].save_signal(signal)
    
    async def _trading_loop(self):
        """Trading execution loop (only if not advisory mode)"""
        logger.info("💹 Trading loop started")
        
        while self.is_running:
            try:
                if 'positions' in self.components:
                    # Update open positions
                    positions = await self.components['positions'].get_open_positions()
                    
                    for position in positions:
                        # Check exit conditions
                        should_exit = await self.components['positions'].check_exit(position)
                        
                        if should_exit:
                            await self.components['exchange'].close_position(position)
                        else:
                            # Update trailing stop if needed
                            await self.components['positions'].update_trailing_stop(position)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(30)
    
    async def _risk_monitoring_loop(self):
        """Monitor portfolio risk"""
        logger.info("🛡️ Risk monitoring loop started")
        
        while self.is_running:
            try:
                if 'risk' in self.components:
                    risk_metrics = await self.components['risk'].calculate_metrics()
                    
                    # Check daily loss
                    if risk_metrics['daily_loss'] > config.trading.max_daily_loss:
                        logger.warning("⚠️ Daily loss limit reached, stopping trading")
                        await self._emergency_stop()
                    
                    # Check drawdown
                    if risk_metrics['drawdown'] > config.trading.max_drawdown:
                        logger.warning("⚠️ Max drawdown reached")
                        if 'alerts' in self.components:
                            await self.components['alerts'].send_emergency_alert(
                                f"Max drawdown reached: {risk_metrics['drawdown']:.2%}"
                            )
                    
                    # Update metrics
                    if 'performance' in self.components:
                        await self.components['performance'].update_metrics(risk_metrics)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _health_monitoring_loop(self):
        """Monitor system health"""
        logger.info("💓 Health monitoring loop started")
        
        while self.is_running:
            try:
                if 'health' in self.components:
                    health_status = await self.components['health'].check_all()
                    
                    if not all(health_status.values()):
                        unhealthy = [k for k, v in health_status.items() if not v]
                        logger.warning(f"⚠️ Unhealthy components: {unhealthy}")
                        
                        # Try to restart unhealthy components
                        for component in unhealthy:
                            await self._restart_component(component)
                
                await asyncio.sleep(config.system.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _restart_component(self, component_name: str):
        """Restart a failed component"""
        logger.info(f"🔄 Attempting to restart {component_name}")
        # Implementation depends on component type
        pass
    
    async def _emergency_stop(self):
        """Emergency stop trading"""
        logger.critical("🚨 EMERGENCY STOP ACTIVATED")
        
        # Close all positions
        if 'positions' in self.components and not config.system.advisory_mode:
            await self.components['positions'].close_all_positions()
        
        # Send emergency alert
        if 'alerts' in self.components:
            await self.components['alerts'].send_emergency_alert(
                "🚨 EMERGENCY STOP - All trading halted"
            )
        
        # Set advisory mode
        config.system.advisory_mode = True
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Initiating graceful shutdown...")
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Send shutdown notification
        if 'alerts' in self.components:
            try:
                await self.components['alerts'].send_notification(
                    f"🛑 DEMIR AI v{config.system.version} shutting down"
                )
            except:
                pass
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up resources...")
        
        # Close database connections
        if 'database' in self.components:
            await self.components['database'].close()
        
        if 'cache' in self.components:
            await self.components['cache'].close()
        
        logger.info("✅ Cleanup completed")

# FastAPI for health checks (Railway requirement)
app = FastAPI(title="DEMIR AI API", version=config.system.version)

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "version": config.system.version,
            "environment": config.system.environment,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "DEMIR AI",
        "version": config.system.version,
        "status": "running",
        "advisory_mode": config.system.advisory_mode
    }

@app.get("/metrics")
async def metrics():
    """Get system metrics"""
    # This would return actual metrics from the performance tracker
    return {
        "uptime": "0h",
        "signals_generated": 0,
        "active_positions": 0,
        "daily_pnl": 0.0
    }

def run_api_server():
    """Run FastAPI server in a separate thread"""
    uvicorn.run(
        app,
        host=config.system.api_host,
        port=config.system.api_port,
        log_level="warning"
    )

def main():
    """Main entry point"""
    try:
        # Print configuration if in debug mode
        if config.system.debug_mode:
            config.print_config()
        
        # Start API server in background thread if enabled
        if config.system.api_enabled:
            api_thread = threading.Thread(target=run_api_server, daemon=True)
            api_thread.start()
            logger.info(f"✅ API server started on port {config.system.api_port}")
        
        # Create and run orchestrator
        orchestrator = DEMIRAIOrchestrator()
        
        # Run the main event loop
        asyncio.run(orchestrator.run())
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        logger.info("DEMIR AI terminated")

if __name__ == "__main__":
    main()
