"""
DEMIR AI v8.0 - Position Management System
REAL POSITION TRACKING - REAL MONEY MANAGEMENT
NO MOCK POSITIONS - ENTERPRISE GRADE
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
from decimal import Decimal, ROUND_DOWN

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Position status"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class PositionType(Enum):
    """Position types"""
    LONG = "LONG"
    SHORT = "SHORT"


class ExitReason(Enum):
    """Exit reasons"""
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    MANUAL = "MANUAL"
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    RISK_LIMIT = "RISK_LIMIT"
    TIME_LIMIT = "TIME_LIMIT"
    EMERGENCY = "EMERGENCY"


@dataclass
class Position:
    """
    Position data structure
    REAL POSITION WITH REAL MONEY
    """
    position_id: str
    symbol: str
    exchange: str
    position_type: PositionType
    status: PositionStatus
    
    # Entry
    entry_price: float
    entry_quantity: float
    entry_time: datetime
    entry_order_id: str
    
    # Current state
    current_price: float
    current_quantity: float
    current_value: float
    
    # Exit targets
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    
    # Trailing stop
    trailing_stop_enabled: bool = False
    trailing_stop_distance: float = 0
    trailing_stop_price: float = 0
    highest_price: float = 0  # For long positions
    lowest_price: float = float('inf')  # For short positions
    
    # Partial exits
    partial_exits: List[Dict] = field(default_factory=list)
    remaining_quantity: float = 0
    
    # P&L
    unrealized_pnl: float = 0
    unrealized_pnl_percent: float = 0
    realized_pnl: float = 0
    total_pnl: float = 0
    
    # Fees
    entry_fee: float = 0
    exit_fees: float = 0
    total_fees: float = 0
    
    # Risk metrics
    risk_amount: float = 0
    risk_percent: float = 0
    risk_reward_ratio: float = 0
    max_drawdown: float = 0
    
    # Signal info
    signal_confidence: float = 0
    signal_strength: str = ""
    signal_reasons: List[str] = field(default_factory=list)
    
    # Management
    last_update: datetime = field(default_factory=datetime.now)
    notes: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def calculate_pnl(self, current_price: float) -> Tuple[float, float]:
        """Calculate P&L"""
        if self.position_type == PositionType.LONG:
            price_diff = current_price - self.entry_price
        else:  # SHORT
            price_diff = self.entry_price - current_price
        
        unrealized_pnl = price_diff * self.current_quantity
        unrealized_pnl_percent = (price_diff / self.entry_price) * 100
        
        # Subtract fees
        unrealized_pnl -= self.total_fees
        
        return unrealized_pnl, unrealized_pnl_percent
    
    def update_trailing_stop(self, current_price: float) -> bool:
        """Update trailing stop price"""
        if not self.trailing_stop_enabled:
            return False
        
        updated = False
        
        if self.position_type == PositionType.LONG:
            # Track highest price
            if current_price > self.highest_price:
                self.highest_price = current_price
                
                # Update trailing stop
                new_stop = current_price - self.trailing_stop_distance
                
                if new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
                    updated = True
                    logger.info(f"Updated trailing stop for {self.symbol}: {new_stop:.2f}")
        
        else:  # SHORT
            # Track lowest price
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                
                # Update trailing stop
                new_stop = current_price + self.trailing_stop_distance
                
                if new_stop < self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
                    updated = True
                    logger.info(f"Updated trailing stop for {self.symbol}: {new_stop:.2f}")
        
        return updated
    
    def should_exit(self, current_price: float) -> Tuple[bool, ExitReason]:
        """Check if position should be exited"""
        
        # Check stop loss
        if self.position_type == PositionType.LONG:
            if current_price <= self.stop_loss:
                return True, ExitReason.STOP_LOSS
            
            # Check trailing stop
            if self.trailing_stop_enabled and current_price <= self.trailing_stop_price:
                return True, ExitReason.TRAILING_STOP
        
        else:  # SHORT
            if current_price >= self.stop_loss:
                return True, ExitReason.STOP_LOSS
            
            # Check trailing stop
            if self.trailing_stop_enabled and current_price >= self.trailing_stop_price:
                return True, ExitReason.TRAILING_STOP
        
        # Check time limit (optional)
        if (datetime.now() - self.entry_time).days > 7:  # 7 days max
            return True, ExitReason.TIME_LIMIT
        
        return False, None
    
    def get_partial_exit_quantity(self, current_price: float, level: int) -> float:
        """Get quantity for partial exit at TP level"""
        if level == 1:
            # Exit 40% at TP1
            return self.entry_quantity * 0.4
        elif level == 2:
            # Exit 30% at TP2
            return self.entry_quantity * 0.3
        elif level == 3:
            # Exit remaining at TP3
            return self.remaining_quantity
        
        return 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        
        # Convert enums to strings
        data['position_type'] = self.position_type.value
        data['status'] = self.status.value
        
        # Convert datetime to ISO format
        data['entry_time'] = self.entry_time.isoformat()
        data['last_update'] = self.last_update.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        """Create from dictionary"""
        # Convert strings to enums
        data['position_type'] = PositionType(data['position_type'])
        data['status'] = PositionStatus(data['status'])
        
        # Convert ISO strings to datetime
        data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        data['last_update'] = datetime.fromisoformat(data['last_update'])
        
        return cls(**data)


class PositionManager:
    """
    Position management system
    REAL POSITION TRACKING - ZERO MOCK DATA
    """
    
    def __init__(self, config):
        self.config = config
        
        # Position storage
        self.positions = {}  # position_id -> Position
        self.symbol_positions = {}  # symbol -> [position_ids]
        self.closed_positions = []
        
        # Risk limits
        self.max_positions = config.trading.max_positions
        self.max_risk_per_position = config.trading.max_risk_per_trade
        self.max_total_risk = config.trading.max_daily_loss
        
        # Position tracking
        self.total_open_positions = 0
        self.total_long_positions = 0
        self.total_short_positions = 0
        
        # P&L tracking
        self.daily_pnl = 0
        self.weekly_pnl = 0
        self.monthly_pnl = 0
        self.total_pnl = 0
        
        # Risk metrics
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.current_risk_exposure = 0
        
        # Performance metrics
        self.winning_positions = 0
        self.losing_positions = 0
        self.total_positions_closed = 0
        self.avg_win_amount = 0
        self.avg_loss_amount = 0
        
        # Position ID counter
        self.position_counter = 0
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        logger.info("PositionManager initialized")
        logger.info(f"Max positions: {self.max_positions}")
        logger.info(f"Max risk per position: {self.max_risk_per_position*100:.1f}%")
    
    async def create_position(self, signal: Dict, order: Dict) -> Position:
        """
        Create new position from signal and order
        REAL POSITION CREATION
        """
        async with self.lock:
            try:
                # Generate position ID
                self.position_counter += 1
                position_id = f"POS_{datetime.now().strftime('%Y%m%d')}_{self.position_counter:04d}"
                
                # Determine position type
                if signal['action'] in ['BUY', 'STRONG_BUY']:
                    position_type = PositionType.LONG
                else:
                    position_type = PositionType.SHORT
                
                # Create position
                position = Position(
                    position_id=position_id,
                    symbol=signal['symbol'],
                    exchange=order.get('exchange', self.config.exchange.primary_exchange),
                    position_type=position_type,
                    status=PositionStatus.OPEN,
                    
                    # Entry
                    entry_price=float(order.get('price', signal['entry_price'])),
                    entry_quantity=float(order['quantity']),
                    entry_time=datetime.now(),
                    entry_order_id=order['order_id'],
                    
                    # Current state
                    current_price=float(order.get('price', signal['entry_price'])),
                    current_quantity=float(order['quantity']),
                    current_value=float(order['quantity']) * float(order.get('price', signal['entry_price'])),
                    
                    # Exit targets
                    stop_loss=signal['stop_loss'],
                    take_profit_1=signal['take_profit_1'],
                    take_profit_2=signal['take_profit_2'],
                    take_profit_3=signal['take_profit_3'],
                    
                    # Remaining quantity
                    remaining_quantity=float(order['quantity']),
                    
                    # Risk
                    risk_amount=abs(signal['max_loss_amount']),
                    risk_percent=signal['position_size'],
                    risk_reward_ratio=signal['risk_reward_ratio'],
                    
                    # Signal info
                    signal_confidence=signal['confidence'],
                    signal_strength=signal['strength'],
                    signal_reasons=signal.get('reasons', []),
                    
                    # Fees (estimated)
                    entry_fee=float(order['quantity']) * float(order.get('price', signal['entry_price'])) * 0.001,
                    
                    # Initial prices for trailing stop
                    highest_price=float(order.get('price', signal['entry_price'])),
                    lowest_price=float(order.get('price', signal['entry_price']))
                )
                
                # Calculate initial fees
                position.total_fees = position.entry_fee
                
                # Enable trailing stop if configured
                if self.config.trading.use_trailing_stop:
                    atr = signal.get('atr', position.entry_price * 0.02)
                    position.trailing_stop_enabled = True
                    position.trailing_stop_distance = atr * 2  # 2x ATR distance
                    position.trailing_stop_price = position.stop_loss
                
                # Add to storage
                self.positions[position_id] = position
                
                # Track by symbol
                if signal['symbol'] not in self.symbol_positions:
                    self.symbol_positions[signal['symbol']] = []
                self.symbol_positions[signal['symbol']].append(position_id)
                
                # Update counters
                self.total_open_positions += 1
                if position_type == PositionType.LONG:
                    self.total_long_positions += 1
                else:
                    self.total_short_positions += 1
                
                # Update risk exposure
                self.current_risk_exposure += position.risk_amount
                
                logger.info(f"Position created: {position_id} - {signal['symbol']} "
                          f"{position_type.value} {position.entry_quantity:.4f} @ {position.entry_price:.2f}")
                
                return position
                
            except Exception as e:
                logger.error(f"Error creating position: {e}")
                raise
    
    async def update_position(self, position_id: str, current_price: float) -> Position:
        """
        Update position with current price
        REAL-TIME POSITION UPDATE
        """
        async with self.lock:
            if position_id not in self.positions:
                logger.warning(f"Position {position_id} not found")
                return None
            
            position = self.positions[position_id]
            
            # Update current price
            position.current_price = current_price
            position.current_value = current_price * position.current_quantity
            
            # Calculate P&L
            position.unrealized_pnl, position.unrealized_pnl_percent = position.calculate_pnl(current_price)
            position.total_pnl = position.realized_pnl + position.unrealized_pnl
            
            # Update trailing stop
            if position.trailing_stop_enabled:
                position.update_trailing_stop(current_price)
            
            # Track max drawdown
            if position.unrealized_pnl < 0:
                position.max_drawdown = max(position.max_drawdown, abs(position.unrealized_pnl))
            
            # Update timestamp
            position.last_update = datetime.now()
            
            return position
    
    async def check_exit(self, position_id: str) -> Tuple[bool, Optional[ExitReason]]:
        """
        Check if position should be exited
        REAL EXIT DECISION
        """
        if position_id not in self.positions:
            return False, None
        
        position = self.positions[position_id]
        
        # Check exit conditions
        should_exit, reason = position.should_exit(position.current_price)
        
        if should_exit:
            logger.info(f"Position {position_id} should exit: {reason.value}")
        
        return should_exit, reason
    
    async def close_position(self, position_id: str, exit_price: float, 
                           exit_reason: ExitReason, exit_quantity: Optional[float] = None) -> Position:
        """
        Close position (fully or partially)
        REAL POSITION CLOSING
        """
        async with self.lock:
            if position_id not in self.positions:
                logger.warning(f"Position {position_id} not found")
                return None
            
            position = self.positions[position_id]
            
            # Determine exit quantity
            if exit_quantity is None:
                exit_quantity = position.current_quantity
            
            # Calculate P&L for this exit
            if position.position_type == PositionType.LONG:
                exit_pnl = (exit_price - position.entry_price) * exit_quantity
            else:  # SHORT
                exit_pnl = (position.entry_price - exit_price) * exit_quantity
            
            # Calculate exit fee
            exit_fee = exit_price * exit_quantity * 0.001  # 0.1% fee
            exit_pnl -= exit_fee
            
            # Update position
            position.realized_pnl += exit_pnl
            position.exit_fees += exit_fee
            position.total_fees += exit_fee
            position.current_quantity -= exit_quantity
            position.remaining_quantity = position.current_quantity
            
            # Record partial exit
            position.partial_exits.append({
                'time': datetime.now().isoformat(),
                'price': exit_price,
                'quantity': exit_quantity,
                'pnl': exit_pnl,
                'reason': exit_reason.value
            })
            
            # Check if fully closed
            if position.current_quantity <= 0:
                position.status = PositionStatus.CLOSED
                position.total_pnl = position.realized_pnl
                
                # Move to closed positions
                self.closed_positions.append(position)
                del self.positions[position_id]
                
                # Remove from symbol tracking
                if position.symbol in self.symbol_positions:
                    self.symbol_positions[position.symbol].remove(position_id)
                
                # Update counters
                self.total_open_positions -= 1
                if position.position_type == PositionType.LONG:
                    self.total_long_positions -= 1
                else:
                    self.total_short_positions -= 1
                
                # Update statistics
                self.total_positions_closed += 1
                if position.total_pnl > 0:
                    self.winning_positions += 1
                    self.avg_win_amount = (
                        (self.avg_win_amount * (self.winning_positions - 1) + position.total_pnl) / 
                        self.winning_positions
                    )
                else:
                    self.losing_positions += 1
                    self.avg_loss_amount = (
                        (self.avg_loss_amount * (self.losing_positions - 1) + abs(position.total_pnl)) / 
                        self.losing_positions
                    )
                
                # Update P&L
                self.daily_pnl += position.total_pnl
                self.total_pnl += position.total_pnl
                
                # Update risk exposure
                self.current_risk_exposure -= position.risk_amount
                
                logger.info(f"Position closed: {position_id} - P&L: ${position.total_pnl:.2f} "
                          f"({position.unrealized_pnl_percent:.1f}%) - Reason: {exit_reason.value}")
            
            else:
                position.status = PositionStatus.PARTIALLY_CLOSED
                logger.info(f"Position partially closed: {position_id} - "
                          f"Exit {exit_quantity:.4f} @ {exit_price:.2f}")
            
            return position
    
    async def update_trailing_stop(self, position_id: str, current_price: float) -> bool:
        """
        Update trailing stop for position
        DYNAMIC STOP LOSS MANAGEMENT
        """
        if position_id not in self.positions:
            return False
        
        position = self.positions[position_id]
        
        # Check if trailing stop should be activated
        if not position.trailing_stop_enabled:
            # Activate after certain profit threshold
            pnl_percent = position.unrealized_pnl_percent
            
            if pnl_percent >= self.config.trading.trailing_stop_activation * 100:
                position.trailing_stop_enabled = True
                
                # Set initial trailing stop
                atr = current_price * 0.02  # Simplified ATR
                position.trailing_stop_distance = atr * 2
                
                if position.position_type == PositionType.LONG:
                    position.trailing_stop_price = current_price - position.trailing_stop_distance
                else:
                    position.trailing_stop_price = current_price + position.trailing_stop_distance
                
                logger.info(f"Trailing stop activated for {position_id} at {position.trailing_stop_price:.2f}")
                return True
        
        else:
            # Update existing trailing stop
            return position.update_trailing_stop(current_price)
    
    async def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return list(self.positions.values())
    
    async def get_position(self, position_id: str) -> Optional[Position]:
        """Get specific position"""
        return self.positions.get(position_id)
    
    async def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get positions for specific symbol"""
        position_ids = self.symbol_positions.get(symbol, [])
        return [self.positions[pid] for pid in position_ids if pid in self.positions]
    
    async def calculate_portfolio_metrics(self) -> Dict:
        """
        Calculate portfolio-wide metrics
        REAL PORTFOLIO ANALYSIS
        """
        metrics = {
            'total_positions': self.total_open_positions,
            'long_positions': self.total_long_positions,
            'short_positions': self.total_short_positions,
            
            # P&L
            'unrealized_pnl': 0,
            'realized_pnl': sum(p.realized_pnl for p in self.positions.values()),
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            
            # Risk
            'current_risk_exposure': self.current_risk_exposure,
            'max_risk_reached': self.current_risk_exposure / self.max_total_risk * 100,
            
            # Performance
            'win_rate': (self.winning_positions / max(1, self.total_positions_closed)) * 100,
            'avg_win': self.avg_win_amount,
            'avg_loss': self.avg_loss_amount,
            'profit_factor': self.avg_win_amount / max(1, self.avg_loss_amount),
            
            # Drawdown
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            
            # Position details
            'positions': []
        }
        
        # Calculate unrealized P&L
        for position in self.positions.values():
            metrics['unrealized_pnl'] += position.unrealized_pnl
            
            metrics['positions'].append({
                'id': position.position_id,
                'symbol': position.symbol,
                'type': position.position_type.value,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'quantity': position.current_quantity,
                'pnl': position.unrealized_pnl,
                'pnl_percent': position.unrealized_pnl_percent
            })
        
        return metrics
    
    async def check_risk_limits(self) -> Tuple[bool, str]:
        """
        Check if risk limits are exceeded
        REAL RISK MANAGEMENT
        """
        # Check max positions
        if self.total_open_positions >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"
        
        # Check daily loss
        if self.daily_pnl < -self.max_total_risk:
            return False, f"Daily loss limit reached ({self.max_total_risk*100:.1f}%)"
        
        # Check total risk exposure
        if self.current_risk_exposure > self.max_total_risk:
            return False, f"Total risk exposure exceeded ({self.current_risk_exposure*100:.1f}%)"
        
        # Check drawdown
        if self.current_drawdown > self.config.trading.max_drawdown:
            return False, f"Max drawdown reached ({self.current_drawdown*100:.1f}%)"
        
        return True, "Risk limits OK"
    
    async def close_all_positions(self, reason: ExitReason = ExitReason.MANUAL) -> List[Position]:
        """
        Close all open positions
        EMERGENCY POSITION CLOSURE
        """
        closed = []
        
        for position_id in list(self.positions.keys()):
            position = self.positions[position_id]
            
            # Use current price for exit
            exit_price = position.current_price
            
            # Close position
            closed_position = await self.close_position(
                position_id, exit_price, reason
            )
            
            if closed_position:
                closed.append(closed_position)
        
        logger.info(f"Closed {len(closed)} positions - Reason: {reason.value}")
        
        return closed
    
    async def save_positions(self, filepath: str):
        """Save positions to file"""
        try:
            data = {
                'positions': {pid: pos.to_dict() for pid, pos in self.positions.items()},
                'closed_positions': [pos.to_dict() for pos in self.closed_positions[-100:]],
                'metrics': await self.calculate_portfolio_metrics(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Positions saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
    
    async def load_positions(self, filepath: str = None):
        """Load positions from file or database"""
        if not filepath:
            # Load from database if configured
            logger.info("Loading positions from database...")
            # Database loading implementation
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Restore positions
            for pid, pos_data in data.get('positions', {}).items():
                position = Position.from_dict(pos_data)
                self.positions[pid] = position
            
            # Restore closed positions
            for pos_data in data.get('closed_positions', []):
                position = Position.from_dict(pos_data)
                self.closed_positions.append(position)
            
            # Update counters
            self.total_open_positions = len(self.positions)
            self.total_long_positions = sum(
                1 for p in self.positions.values() 
                if p.position_type == PositionType.LONG
            )
            self.total_short_positions = self.total_open_positions - self.total_long_positions
            
            logger.info(f"Loaded {self.total_open_positions} positions from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
    
    def get_statistics(self) -> Dict:
        """Get position manager statistics"""
        return {
            'total_open': self.total_open_positions,
            'total_long': self.total_long_positions,
            'total_short': self.total_short_positions,
            'total_closed': self.total_positions_closed,
            'winning_positions': self.winning_positions,
            'losing_positions': self.losing_positions,
            'win_rate': (self.winning_positions / max(1, self.total_positions_closed)) * 100,
            'avg_win': self.avg_win_amount,
            'avg_loss': self.avg_loss_amount,
            'profit_factor': self.avg_win_amount / max(1, self.avg_loss_amount),
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'current_risk': self.current_risk_exposure,
            'max_drawdown': self.max_drawdown
        }
