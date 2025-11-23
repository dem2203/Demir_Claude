"""
DEMIR AI v8.0 - Professional Logging Configuration
ENTERPRISE GRADE LOGGING SYSTEM
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
import json
import traceback
from typing import Dict, Any

# Color codes for terminal output
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[34m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: green + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName', 'relativeCreated',
                          'stack_info', 'thread', 'threadName', 'exc_info', 'exc_text']:
                log_obj[key] = value
        
        return json.dumps(log_obj)


class PerformanceLogFilter(logging.Filter):
    """Filter to add performance metrics to logs"""
    
    def filter(self, record):
        # Add performance metrics
        import psutil
        
        record.cpu_percent = psutil.cpu_percent()
        record.memory_percent = psutil.virtual_memory().percent
        
        return True


class TradingLogFilter(logging.Filter):
    """Filter to add trading context to logs"""
    
    def __init__(self, orchestrator=None):
        super().__init__()
        self.orchestrator = orchestrator
    
    def filter(self, record):
        # Add trading context if available
        if self.orchestrator:
            try:
                status = self.orchestrator.get_status()
                record.trading_mode = status.get('mode', 'unknown')
                record.system_state = status.get('state', 'unknown')
            except:
                pass
        
        return True


def setup_logger(name: str = None, level: str = None) -> logging.Logger:
    """
    Setup professional logging system
    ENTERPRISE GRADE LOGGING
    """
    
    # Get logger
    logger = logging.getLogger(name or 'demirai')
    
    # Clear existing handlers
    logger.handlers = []
    
    # Set level from environment or default
    log_level = level or os.getenv('LOG_LEVEL', 'INFO')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Console Handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    if sys.stdout.isatty():  # Check if running in terminal
        console_handler.setFormatter(ColoredFormatter())
    else:
        # Use simple format for non-terminal (e.g., Railway logs)
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    
    logger.addHandler(console_handler)
    
    # File Handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'demirai.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)
    
    # JSON File Handler for structured logs
    json_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'demirai.json',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    json_handler.setLevel(logging.INFO)
    json_handler.setFormatter(JsonFormatter())
    logger.addHandler(json_handler)
    
    # Error File Handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'errors.log',
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(exc_info)s')
    )
    logger.addHandler(error_handler)
    
    # Trading Activity Handler
    trading_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'trading.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=20
    )
    trading_handler.setLevel(logging.INFO)
    trading_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    trading_handler.addFilter(lambda record: 'trade' in record.getMessage().lower() or 
                                            'position' in record.getMessage().lower() or
                                            'signal' in record.getMessage().lower())
    logger.addHandler(trading_handler)
    
    # Performance Metrics Handler (if enabled)
    if os.getenv('ENABLE_PERFORMANCE_LOGGING', 'false') == 'true':
        perf_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'performance.log',
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(
            logging.Formatter('%(asctime)s - CPU: %(cpu_percent).1f%% - MEM: %(memory_percent).1f%% - %(message)s')
        )
        perf_handler.addFilter(PerformanceLogFilter())
        logger.addHandler(perf_handler)
    
    # Syslog Handler (if configured)
    syslog_host = os.getenv('SYSLOG_HOST')
    if syslog_host:
        try:
            syslog_handler = logging.handlers.SysLogHandler(
                address=(syslog_host, int(os.getenv('SYSLOG_PORT', 514)))
            )
            syslog_handler.setLevel(logging.WARNING)
            syslog_handler.setFormatter(
                logging.Formatter('demirai: %(levelname)s - %(message)s')
            )
            logger.addHandler(syslog_handler)
        except Exception as e:
            logger.warning(f"Failed to setup syslog handler: {e}")
    
    # Set up module-specific loggers
    setup_module_loggers()
    
    logger.info("Logging system initialized")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log directory: {log_dir.absolute()}")
    
    return logger


def setup_module_loggers():
    """Configure module-specific logging levels"""
    
    # Reduce noise from external libraries
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    # Set specific levels for our modules
    module_levels = {
        'demirai.trading': 'INFO',
        'demirai.signals': 'INFO',
        'demirai.risk': 'INFO',
        'demirai.ai': 'INFO',
        'demirai.websocket': 'WARNING',
        'demirai.database': 'WARNING'
    }
    
    for module, level in module_levels.items():
        logging.getLogger(module).setLevel(getattr(logging, level))


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)


class LoggerContext:
    """Context manager for temporary log level changes"""
    
    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = logger.level
    
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def log_trade(logger: logging.Logger, trade_data: Dict[str, Any]):
    """Log trade with structured format"""
    
    trade_log = {
        'action': 'TRADE',
        'symbol': trade_data.get('symbol'),
        'side': trade_data.get('side'),
        'price': trade_data.get('price'),
        'quantity': trade_data.get('quantity'),
        'value': trade_data.get('value'),
        'timestamp': datetime.utcnow().isoformat()
    }
    
    logger.info(f"TRADE: {json.dumps(trade_log)}")


def log_signal(logger: logging.Logger, signal_data: Dict[str, Any]):
    """Log signal with structured format"""
    
    signal_log = {
        'action': 'SIGNAL',
        'symbol': signal_data.get('symbol'),
        'signal_action': signal_data.get('action'),
        'confidence': signal_data.get('confidence'),
        'entry_price': signal_data.get('entry_price'),
        'stop_loss': signal_data.get('stop_loss'),
        'take_profit': signal_data.get('take_profit_1'),
        'timestamp': datetime.utcnow().isoformat()
    }
    
    logger.info(f"SIGNAL: {json.dumps(signal_log)}")


def log_error(logger: logging.Logger, error: Exception, context: Dict[str, Any] = None):
    """Log error with full context"""
    
    error_log = {
        'action': 'ERROR',
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context or {},
        'timestamp': datetime.utcnow().isoformat()
    }
    
    logger.error(f"ERROR: {json.dumps(error_log)}", exc_info=True)


def log_metric(logger: logging.Logger, metric_name: str, value: float, tags: Dict[str, str] = None):
    """Log metric for monitoring"""
    
    metric_log = {
        'action': 'METRIC',
        'name': metric_name,
        'value': value,
        'tags': tags or {},
        'timestamp': datetime.utcnow().isoformat()
    }
    
    logger.info(f"METRIC: {json.dumps(metric_log)}")
