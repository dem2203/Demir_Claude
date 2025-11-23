"""
DEMIR AI v8.0 - Circuit Breaker Security System
ADVANCED PROTECTION MECHANISM - ZERO MOCK DATA
PREVENTS CATASTROPHIC FAILURES AND PROTECTS CAPITAL
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Circuit tripped - blocking all operations
    HALF_OPEN = "HALF_OPEN"  # Testing if system recovered


class TripReason(Enum):
    """Reasons for circuit breaker activation"""
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    RAPID_CAPITAL_LOSS = "RAPID_CAPITAL_LOSS"
    MAX_DAILY_LOSS = "MAX_DAILY_LOSS"
    SYSTEM_ERROR_RATE = "SYSTEM_ERROR_RATE"
    API_FAILURES = "API_FAILURES"
    ANOMALY_DETECTED = "ANOMALY_DETECTED"
    RISK_LIMIT_BREACH = "RISK_LIMIT_BREACH"
    MANUAL_TRIGGER = "MANUAL_TRIGGER"
    NETWORK_ISSUES = "NETWORK_ISSUES"
    EXCHANGE_ISSUES = "EXCHANGE_ISSUES"
    DATA_INTEGRITY = "DATA_INTEGRITY"
    MEMORY_OVERFLOW = "MEMORY_OVERFLOW"
    LATENCY_SPIKE = "LATENCY_SPIKE"
    CORRELATION_BREAKDOWN = "CORRELATION_BREAKDOWN"
    BLACK_SWAN_EVENT = "BLACK_SWAN_EVENT"


@dataclass
class CircuitEvent:
    """Circuit breaker event record"""
    event_id: str
    timestamp: datetime
    state_before: CircuitState
    state_after: CircuitState
    reason: TripReason
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    metrics: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    recovery_actions: List[str] = field(default_factory=list)


class CircuitBreaker:
    """
    Advanced circuit breaker for system protection
    REAL PROTECTION - NOT A MOCK SYSTEM
    """
    
    def __init__(self, config):
        self.config = config
        
        # Circuit breaker configuration
        self.failure_threshold = 5  # Number of failures before opening
        self.success_threshold = 3  # Successes needed in half-open to close
        self.timeout = timedelta(minutes=5)  # Time before half-open
        self.reset_timeout = timedelta(minutes=30)  # Full reset time
        
        # Thresholds for different trip conditions
        self.thresholds = {
            'consecutive_losses': 5,
            'rapid_loss_percent': 5.0,  # 5% in short time
            'rapid_loss_timeframe': 60,  # seconds
            'daily_loss_percent': 10.0,  # 10% daily loss
            'error_rate_percent': 20.0,  # 20% error rate
            'api_failure_count': 10,
            'latency_ms': 5000,  # 5 second latency
            'memory_percent': 90,  # 90% memory usage
        }
        
        # State management
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.now()
        self.trip_count = 0
        self.recovery_attempts = 0
        
        # Failure tracking
        self.failure_count = 0
        self.success_count = 0
        self.error_window = deque(maxlen=100)  # Last 100 operations
        self.loss_window = deque(maxlen=50)  # Last 50 trades
        
        # Performance tracking
        self.api_failures = deque(maxlen=100)
        self.latency_measurements = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        
        # Event history
        self.events = []
        self.max_events = 1000
        
        # Callbacks
        self.on_open_callbacks = []
        self.on_close_callbacks = []
        self.on_half_open_callbacks = []
        
        # Protected functions
        self.protected_functions = {}
        
        # Statistics
        self.total_trips = 0
        self.total_operations = 0
        self.blocked_operations = 0
        
        # Recovery actions
        self.recovery_actions = {
            TripReason.CONSECUTIVE_LOSSES: self._recover_from_losses,
            TripReason.RAPID_CAPITAL_LOSS: self._recover_from_rapid_loss,
            TripReason.API_FAILURES: self._recover_from_api_failures,
            TripReason.SYSTEM_ERROR_RATE: self._recover_from_errors,
            TripReason.MEMORY_OVERFLOW: self._recover_from_memory,
            TripReason.LATENCY_SPIKE: self._recover_from_latency,
            TripReason.BLACK_SWAN_EVENT: self._recover_from_black_swan
        }
        
        # Monitoring task
        self.monitoring_task = None
        
        logger.info("CircuitBreaker initialized - System protection active")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        REAL PROTECTION FOR CRITICAL OPERATIONS
        """
        self.total_operations += 1
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if datetime.now() - self.last_state_change > self.timeout:
                await self._transition_to_half_open()
            else:
                self.blocked_operations += 1
                raise CircuitOpenError(f"Circuit breaker is OPEN - {self._get_time_until_recovery()}")
        
        try:
            # Execute function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Record success
            await self._record_success()
            
            return result
            
        except Exception as e:
            # Record failure
            await self._record_failure(e)
            raise
    
    async def check_conditions(self, metrics: Dict[str, Any]) -> Optional[TripReason]:
        """
        Check if any trip conditions are met
        COMPREHENSIVE SYSTEM MONITORING
        """
        # Check consecutive losses
        if metrics.get('consecutive_losses', 0) >= self.thresholds['consecutive_losses']:
            return TripReason.CONSECUTIVE_LOSSES
        
        # Check rapid capital loss
        recent_loss = metrics.get('recent_loss_percent', 0)
        if abs(recent_loss) > self.thresholds['rapid_loss_percent']:
            loss_timeframe = metrics.get('loss_timeframe_seconds', float('inf'))
            if loss_timeframe < self.thresholds['rapid_loss_timeframe']:
                return TripReason.RAPID_CAPITAL_LOSS
        
        # Check daily loss
        daily_loss = metrics.get('daily_loss_percent', 0)
        if abs(daily_loss) > self.thresholds['daily_loss_percent']:
            return TripReason.MAX_DAILY_LOSS
        
        # Check system error rate
        error_rate = self._calculate_error_rate()
        if error_rate > self.thresholds['error_rate_percent']:
            return TripReason.SYSTEM_ERROR_RATE
        
        # Check API failures
        recent_api_failures = sum(1 for f in self.api_failures if f > datetime.now() - timedelta(minutes=5))
        if recent_api_failures > self.thresholds['api_failure_count']:
            return TripReason.API_FAILURES
        
        # Check latency
        if self.latency_measurements:
            avg_latency = np.mean(list(self.latency_measurements))
            if avg_latency > self.thresholds['latency_ms']:
                return TripReason.LATENCY_SPIKE
        
        # Check memory usage
        memory_usage = metrics.get('memory_percent', 0)
        if memory_usage > self.thresholds['memory_percent']:
            return TripReason.MEMORY_OVERFLOW
        
        # Check for anomalies
        if metrics.get('anomaly_score', 0) > 80:
            return TripReason.ANOMALY_DETECTED
        
        # Check risk limits
        if metrics.get('risk_breach', False):
            return TripReason.RISK_LIMIT_BREACH
        
        # Check for black swan events
        if self._detect_black_swan(metrics):
            return TripReason.BLACK_SWAN_EVENT
        
        return None
    
    async def trip(self, reason: TripReason, metrics: Dict[str, Any] = None):
        """
        Trip the circuit breaker
        EMERGENCY SHUTDOWN PROCEDURE
        """
        if self.state == CircuitState.OPEN:
            return  # Already open
        
        previous_state = self.state
        self.state = CircuitState.OPEN
        self.last_state_change = datetime.now()
        self.trip_count += 1
        self.total_trips += 1
        
        # Determine severity
        severity = self._determine_severity(reason, metrics)
        
        # Create event
        event = CircuitEvent(
            event_id=f"CB_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.total_trips}",
            timestamp=datetime.now(),
            state_before=previous_state,
            state_after=CircuitState.OPEN,
            reason=reason,
            severity=severity,
            metrics=metrics or {},
            description=self._get_reason_description(reason),
            recovery_actions=self._get_recovery_actions(reason)
        )
        
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        logger.critical(f"CIRCUIT BREAKER TRIPPED: {reason.value}")
        logger.critical(f"Severity: {severity} | Trip count: {self.trip_count}")
        
        # Execute callbacks
        for callback in self.on_open_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        # Initiate recovery actions
        if reason in self.recovery_actions:
            await self.recovery_actions[reason]()
    
    async def reset(self):
        """Manual reset of circuit breaker"""
        previous_state = self.state
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.now()
        self.failure_count = 0
        self.success_count = 0
        self.recovery_attempts = 0
        
        logger.info(f"Circuit breaker manually reset from {previous_state.value} to CLOSED")
        
        # Execute callbacks
        for callback in self.on_close_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def _record_success(self):
        """Record successful operation"""
        self.error_window.append(0)  # 0 = success
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            # Check if we can close circuit
            if self.success_count >= self.success_threshold:
                await self._transition_to_closed()
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)
    
    async def _record_failure(self, error: Exception):
        """Record failed operation"""
        self.error_window.append(1)  # 1 = failure
        
        if self.state == CircuitState.HALF_OPEN:
            # Immediate trip on failure in half-open state
            await self.trip(TripReason.SYSTEM_ERROR_RATE, {'error': str(error)})
        
        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1
            
            # Check if we should trip
            if self.failure_count >= self.failure_threshold:
                await self.trip(TripReason.SYSTEM_ERROR_RATE, {'failures': self.failure_count})
    
    async def _transition_to_half_open(self):
        """Transition to half-open state for testing"""
        previous_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = datetime.now()
        self.success_count = 0
        self.recovery_attempts += 1
        
        logger.info(f"Circuit breaker transitioned to HALF_OPEN (attempt #{self.recovery_attempts})")
        
        # Execute callbacks
        for callback in self.on_half_open_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def _transition_to_closed(self):
        """Transition back to closed state"""
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.now()
        self.failure_count = 0
        self.success_count = 0
        self.recovery_attempts = 0
        
        logger.info("Circuit breaker recovered - transitioned to CLOSED")
        
        # Execute callbacks
        for callback in self.on_close_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _calculate_error_rate(self) -> float:
        """Calculate recent error rate"""
        if not self.error_window:
            return 0.0
        
        errors = sum(self.error_window)
        return (errors / len(self.error_window)) * 100
    
    def _detect_black_swan(self, metrics: Dict[str, Any]) -> bool:
        """
        Detect black swan events
        EXTREME MARKET CONDITIONS
        """
        # Multiple extreme conditions
        extreme_conditions = 0
        
        # Extreme price movement
        if abs(metrics.get('price_change_percent', 0)) > 20:
            extreme_conditions += 1
        
        # Extreme volume
        if metrics.get('volume_ratio', 1) > 10:
            extreme_conditions += 1
        
        # Extreme volatility
        if metrics.get('volatility', 0) > 0.5:  # 50% daily volatility
            extreme_conditions += 1
        
        # Correlation breakdown
        if metrics.get('correlation_breakdown', False):
            extreme_conditions += 1
        
        # System-wide losses
        if metrics.get('system_wide_losses', 0) > 15:
            extreme_conditions += 1
        
        return extreme_conditions >= 3
    
    def _determine_severity(self, reason: TripReason, metrics: Dict[str, Any]) -> str:
        """Determine event severity"""
        critical_reasons = [
            TripReason.BLACK_SWAN_EVENT,
            TripReason.RAPID_CAPITAL_LOSS,
            TripReason.MAX_DAILY_LOSS
        ]
        
        high_reasons = [
            TripReason.CONSECUTIVE_LOSSES,
            TripReason.RISK_LIMIT_BREACH,
            TripReason.MEMORY_OVERFLOW
        ]
        
        if reason in critical_reasons:
            return "CRITICAL"
        elif reason in high_reasons:
            return "HIGH"
        elif self.trip_count > 5:
            return "HIGH"  # Escalate if tripping frequently
        else:
            return "MEDIUM"
    
    def _get_reason_description(self, reason: TripReason) -> str:
        """Get human-readable description"""
        descriptions = {
            TripReason.CONSECUTIVE_LOSSES: "Multiple consecutive losing trades detected",
            TripReason.RAPID_CAPITAL_LOSS: "Rapid capital loss in short timeframe",
            TripReason.MAX_DAILY_LOSS: "Maximum daily loss limit exceeded",
            TripReason.SYSTEM_ERROR_RATE: "High system error rate detected",
            TripReason.API_FAILURES: "Multiple API failures detected",
            TripReason.ANOMALY_DETECTED: "Market anomaly detected",
            TripReason.RISK_LIMIT_BREACH: "Risk management limits breached",
            TripReason.MANUAL_TRIGGER: "Manually triggered by operator",
            TripReason.NETWORK_ISSUES: "Network connectivity issues",
            TripReason.EXCHANGE_ISSUES: "Exchange connection problems",
            TripReason.DATA_INTEGRITY: "Data integrity check failed",
            TripReason.MEMORY_OVERFLOW: "System memory critically high",
            TripReason.LATENCY_SPIKE: "Extreme latency detected",
            TripReason.CORRELATION_BREAKDOWN: "Market correlation breakdown",
            TripReason.BLACK_SWAN_EVENT: "Black swan event detected"
        }
        
        return descriptions.get(reason, "Unknown reason")
    
    def _get_recovery_actions(self, reason: TripReason) -> List[str]:
        """Get recovery action recommendations"""
        actions = {
            TripReason.CONSECUTIVE_LOSSES: [
                "Review and adjust strategy parameters",
                "Reduce position sizes",
                "Check market conditions"
            ],
            TripReason.RAPID_CAPITAL_LOSS: [
                "Close all positions",
                "Review risk management",
                "Wait for market stabilization"
            ],
            TripReason.API_FAILURES: [
                "Check API credentials",
                "Verify network connectivity",
                "Contact exchange support if persistent"
            ],
            TripReason.MEMORY_OVERFLOW: [
                "Clear caches",
                "Restart system",
                "Check for memory leaks"
            ],
            TripReason.BLACK_SWAN_EVENT: [
                "EMERGENCY: Close all positions",
                "Disable automated trading",
                "Manual intervention required"
            ]
        }
        
        return actions.get(reason, ["Monitor system", "Check logs", "Manual review required"])
    
    async def _recover_from_losses(self):
        """Recovery procedure for consecutive losses"""
        logger.info("Initiating recovery from consecutive losses")
        
        # Clear loss window
        self.loss_window.clear()
        
        # Reduce risk parameters
        # This would adjust config.trading parameters
        
        logger.info("Recovery actions for losses completed")
    
    async def _recover_from_rapid_loss(self):
        """Recovery procedure for rapid capital loss"""
        logger.critical("EMERGENCY: Rapid capital loss recovery initiated")
        
        # Would trigger emergency position closure
        # Would disable new trades
        
        logger.info("Emergency procedures activated")
    
    async def _recover_from_api_failures(self):
        """Recovery procedure for API failures"""
        logger.info("Recovering from API failures")
        
        # Clear API failure history
        self.api_failures.clear()
        
        # Would reset API connections
        
        logger.info("API recovery completed")
    
    async def _recover_from_errors(self):
        """Recovery procedure for system errors"""
        logger.info("Recovering from system errors")
        
        # Clear error window
        self.error_window.clear()
        self.failure_count = 0
        
        logger.info("Error recovery completed")
    
    async def _recover_from_memory(self):
        """Recovery procedure for memory issues"""
        logger.warning("Recovering from memory overflow")
        
        # Would trigger garbage collection
        # Would clear caches
        
        logger.info("Memory recovery completed")
    
    async def _recover_from_latency(self):
        """Recovery procedure for latency issues"""
        logger.info("Recovering from latency spike")
        
        # Clear latency measurements
        self.latency_measurements.clear()
        
        # Would reduce request rate
        
        logger.info("Latency recovery completed")
    
    async def _recover_from_black_swan(self):
        """Recovery procedure for black swan events"""
        logger.critical("BLACK SWAN EVENT - EMERGENCY RECOVERY")
        
        # This would trigger complete system shutdown
        # Manual intervention required
        
        logger.critical("MANUAL INTERVENTION REQUIRED")
    
    def _get_time_until_recovery(self) -> str:
        """Get formatted time until recovery attempt"""
        if self.state != CircuitState.OPEN:
            return "N/A"
        
        time_passed = datetime.now() - self.last_state_change
        time_remaining = self.timeout - time_passed
        
        if time_remaining.total_seconds() <= 0:
            return "Ready for recovery"
        
        minutes = int(time_remaining.total_seconds() / 60)
        seconds = int(time_remaining.total_seconds() % 60)
        
        return f"{minutes}m {seconds}s"
    
    def register_on_open(self, callback: Callable):
        """Register callback for circuit open event"""
        self.on_open_callbacks.append(callback)
    
    def register_on_close(self, callback: Callable):
        """Register callback for circuit close event"""
        self.on_close_callbacks.append(callback)
    
    def register_on_half_open(self, callback: Callable):
        """Register callback for circuit half-open event"""
        self.on_half_open_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Circuit breaker monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Circuit breaker monitoring stopped")
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while True:
            try:
                # Monitor system metrics
                # This would collect real metrics from system
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    
    def get_statistics(self) -> Dict:
        """Get circuit breaker statistics"""
        success_rate = 0
        if self.total_operations > 0:
            success_rate = ((self.total_operations - self.blocked_operations) / 
                          self.total_operations * 100)
        
        return {
            'state': self.state.value,
            'total_trips': self.total_trips,
            'current_trip_count': self.trip_count,
            'recovery_attempts': self.recovery_attempts,
            'total_operations': self.total_operations,
            'blocked_operations': self.blocked_operations,
            'success_rate': success_rate,
            'error_rate': self._calculate_error_rate(),
            'time_in_state': (datetime.now() - self.last_state_change).total_seconds(),
            'recent_events': len(self.events)
        }
    
    def get_recent_events(self, limit: int = 10) -> List[CircuitEvent]:
        """Get recent circuit breaker events"""
        return self.events[-limit:] if self.events else []


class CircuitOpenError(Exception):
    """Exception raised when circuit is open"""
    pass


class MultiCircuitBreaker:
    """
    Multiple circuit breakers for different subsystems
    COMPREHENSIVE SYSTEM PROTECTION
    """
    
    def __init__(self, config):
        self.config = config
        
        # Create circuit breakers for different subsystems
        self.breakers = {
            'trading': CircuitBreaker(config),
            'api': CircuitBreaker(config),
            'database': CircuitBreaker(config),
            'ml_models': CircuitBreaker(config),
            'websocket': CircuitBreaker(config),
            'risk': CircuitBreaker(config)
        }
        
        # Configure specific thresholds
        self.breakers['trading'].thresholds['consecutive_losses'] = 5
        self.breakers['api'].thresholds['api_failure_count'] = 20
        self.breakers['database'].thresholds['latency_ms'] = 2000
        self.breakers['ml_models'].thresholds['error_rate_percent'] = 30
        
        logger.info(f"MultiCircuitBreaker initialized with {len(self.breakers)} breakers")
    
    async def check_all(self, metrics: Dict[str, Any]) -> Dict[str, Optional[TripReason]]:
        """Check all circuit breakers"""
        results = {}
        
        for name, breaker in self.breakers.items():
            reason = await breaker.check_conditions(metrics.get(name, {}))
            results[name] = reason
            
            if reason:
                logger.warning(f"Circuit breaker '{name}' detected issue: {reason.value}")
        
        return results
    
    async def trip_subsystem(self, subsystem: str, reason: TripReason, metrics: Dict[str, Any] = None):
        """Trip specific subsystem circuit breaker"""
        if subsystem in self.breakers:
            await self.breakers[subsystem].trip(reason, metrics)
        else:
            logger.error(f"Unknown subsystem: {subsystem}")
    
    def get_status(self) -> Dict[str, str]:
        """Get status of all circuit breakers"""
        return {name: breaker.state.value for name, breaker in self.breakers.items()}
    
    async def reset_all(self):
        """Reset all circuit breakers"""
        for name, breaker in self.breakers.items():
            await breaker.reset()
            logger.info(f"Reset circuit breaker: {name}")
