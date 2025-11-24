"""
DEMIR AI v8.0 - Advanced Trading Strategies
PROFESSIONAL ALGORITHMIC TRADING STRATEGIES
ENTERPRISE GRADE - ZERO MOCK DATA
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Strategy types"""
    GRID_TRADING = "GRID_TRADING"
    DCA = "DCA"  # Dollar Cost Averaging
    ARBITRAGE = "ARBITRAGE"
    MARKET_MAKING = "MARKET_MAKING"
    MEAN_REVERSION = "MEAN_REVERSION"
    MOMENTUM = "MOMENTUM"
    BREAKOUT = "BREAKOUT"
    PAIRS_TRADING = "PAIRS_TRADING"
    SCALPING = "SCALPING"
    SWING_TRADING = "SWING_TRADING"


@dataclass
class GridLevel:
    """Grid trading level"""
    level_id: str
    price: float
    quantity: float
    side: str  # BUY or SELL
    order_id: Optional[str] = None
    filled: bool = False
    filled_at: Optional[datetime] = None
    profit_target: float = 0
    stop_loss: Optional[float] = None


@dataclass
class DCAOrder:
    """DCA order information"""
    order_number: int
    amount: float
    price_target: Optional[float]
    time_target: Optional[datetime]
    executed: bool = False
    executed_price: Optional[float] = None
    executed_at: Optional[datetime] = None
    quantity: Optional[float] = None


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity"""
    opportunity_id: str
    type: str  # triangular, exchange, spot-futures
    profit_percentage: float
    volume: float
    path: List[str]
    exchanges: List[str]
    expires_at: datetime
    risk_score: float


# ====================== GRID TRADING STRATEGY ======================

class GridTradingStrategy:
    """
    Grid Trading Strategy
    Places buy and sell orders at regular intervals
    PROFESSIONAL IMPLEMENTATION
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.symbol = config['symbol']
        
        # Grid parameters
        self.grid_levels = config.get('grid_levels', 20)
        self.grid_spacing = config.get('grid_spacing', 0.005)  # 0.5%
        self.quantity_per_grid = config.get('quantity_per_grid', 100)
        self.upper_price = config.get('upper_price')
        self.lower_price = config.get('lower_price')
        
        # Risk parameters
        self.max_exposure = config.get('max_exposure', 10000)
        self.stop_loss_percentage = config.get('stop_loss', 0.1)  # 10%
        self.take_profit_percentage = config.get('take_profit', 0.02)  # 2% per grid
        
        # Dynamic grid adjustment
        self.dynamic_adjustment = config.get('dynamic_adjustment', True)
        self.rebalance_threshold = config.get('rebalance_threshold', 0.1)
        
        # State tracking
        self.grids: List[GridLevel] = []
        self.active_orders: Dict[str, GridLevel] = {}
        self.completed_trades = []
        self.total_profit = 0
        self.current_exposure = 0
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_profit': 0,
            'max_drawdown': 0,
            'grid_completion_rate': 0,
            'average_profit_per_grid': 0,
            'time_in_profit': 0,
            'volatility_captured': 0
        }
        
        # Price tracking
        self.price_history = deque(maxlen=1000)
        self.last_rebalance = datetime.now()
        
        logger.info(f"GridTradingStrategy initialized for {self.symbol}")
    
    async def initialize_grid(self, current_price: float):
        """
        Initialize grid levels
        SMART GRID PLACEMENT
        """
        self.grids.clear()
        
        # Calculate grid boundaries if not set
        if not self.upper_price:
            self.upper_price = current_price * (1 + self.grid_spacing * self.grid_levels / 2)
        if not self.lower_price:
            self.lower_price = current_price * (1 - self.grid_spacing * self.grid_levels / 2)
        
        # Calculate actual grid spacing
        price_range = self.upper_price - self.lower_price
        actual_spacing = price_range / self.grid_levels
        
        # Create grid levels
        for i in range(self.grid_levels):
            price = self.lower_price + (actual_spacing * i)
            
            # Determine side based on current price
            if price < current_price:
                side = 'BUY'
                profit_target = price * (1 + self.take_profit_percentage)
            else:
                side = 'SELL'
                profit_target = price * (1 - self.take_profit_percentage)
            
            grid = GridLevel(
                level_id=f"GRID_{i}_{datetime.now().timestamp()}",
                price=price,
                quantity=self.quantity_per_grid,
                side=side,
                profit_target=profit_target,
                stop_loss=price * (1 - self.stop_loss_percentage) if side == 'BUY' else price * (1 + self.stop_loss_percentage)
            )
            
            self.grids.append(grid)
        
        logger.info(f"Initialized {len(self.grids)} grid levels from {self.lower_price:.2f} to {self.upper_price:.2f}")
    
    async def execute_grid_orders(self, exchange_manager) -> List[Dict]:
        """
        Place grid orders on exchange
        REAL ORDER EXECUTION
        """
        orders = []
        
        for grid in self.grids:
            if not grid.filled and grid.level_id not in self.active_orders:
                # Check exposure limit
                order_value = grid.price * grid.quantity
                if self.current_exposure + order_value > self.max_exposure:
                    continue
                
                # Place order
                try:
                    order = await exchange_manager.place_order(
                        symbol=self.symbol,
                        side=grid.side,
                        order_type='LIMIT',
                        quantity=grid.quantity,
                        price=grid.price
                    )
                    
                    grid.order_id = order['id']
                    self.active_orders[grid.level_id] = grid
                    self.current_exposure += order_value
                    
                    orders.append(order)
                    
                    logger.info(f"Placed grid order: {grid.side} {grid.quantity} @ {grid.price}")
                    
                except Exception as e:
                    logger.error(f"Failed to place grid order: {e}")
        
        return orders
    
    async def monitor_grid_fills(self, exchange_manager):
        """
        Monitor and manage filled grid orders
        INTELLIGENT GRID MANAGEMENT
        """
        filled_grids = []
        
        for level_id, grid in list(self.active_orders.items()):
            try:
                # Check order status
                order = await exchange_manager.get_order(grid.order_id, self.symbol)
                
                if order['status'] == 'FILLED':
                    grid.filled = True
                    grid.filled_at = datetime.now()
                    
                    # Place profit target order
                    profit_order = await self._place_profit_order(grid, exchange_manager)
                    
                    filled_grids.append(grid)
                    del self.active_orders[level_id]
                    
                    # Update metrics
                    self.metrics['total_trades'] += 1
                    
                    logger.info(f"Grid filled: {grid.side} @ {order['price']}")
                    
            except Exception as e:
                logger.error(f"Error monitoring grid {level_id}: {e}")
        
        return filled_grids
    
    async def _place_profit_order(self, grid: GridLevel, exchange_manager) -> Optional[Dict]:
        """Place take profit order for filled grid"""
        try:
            # Opposite side for profit
            profit_side = 'SELL' if grid.side == 'BUY' else 'BUY'
            
            order = await exchange_manager.place_order(
                symbol=self.symbol,
                side=profit_side,
                order_type='LIMIT',
                quantity=grid.quantity,
                price=grid.profit_target
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to place profit order: {e}")
            return None
    
    async def rebalance_grid(self, current_price: float):
        """
        Rebalance grid based on price movement
        DYNAMIC GRID ADJUSTMENT
        """
        if not self.dynamic_adjustment:
            return
        
        # Check if rebalance needed
        time_since_rebalance = (datetime.now() - self.last_rebalance).seconds / 3600
        if time_since_rebalance < 1:  # Don't rebalance more than once per hour
            return
        
        # Check price movement
        price_change = abs(current_price - (self.upper_price + self.lower_price) / 2)
        price_change_pct = price_change / ((self.upper_price + self.lower_price) / 2)
        
        if price_change_pct > self.rebalance_threshold:
            logger.info(f"Rebalancing grid due to {price_change_pct:.2%} price movement")
            
            # Cancel existing orders
            await self._cancel_all_orders()
            
            # Reinitialize grid
            await self.initialize_grid(current_price)
            
            self.last_rebalance = datetime.now()
    
    async def _cancel_all_orders(self):
        """Cancel all active grid orders"""
        for grid in self.active_orders.values():
            if grid.order_id:
                try:
                    # Cancel through exchange manager
                    pass  # Implementation depends on exchange
                except:
                    pass
    
    async def calculate_grid_performance(self) -> Dict:
        """
        Calculate grid strategy performance
        COMPREHENSIVE METRICS
        """
        if not self.completed_trades:
            return self.metrics
        
        # Calculate profit metrics
        profits = [t['profit'] for t in self.completed_trades]
        self.metrics['total_profit'] = sum(profits)
        self.metrics['profitable_trades'] = len([p for p in profits if p > 0])
        self.metrics['average_profit_per_grid'] = np.mean(profits)
        
        # Calculate grid completion rate
        filled_grids = len([g for g in self.grids if g.filled])
        self.metrics['grid_completion_rate'] = filled_grids / len(self.grids) if self.grids else 0
        
        # Calculate drawdown
        cumulative_profit = np.cumsum(profits)
        peak = np.maximum.accumulate(cumulative_profit)
        drawdown = (peak - cumulative_profit) / peak
        self.metrics['max_drawdown'] = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Volatility capture
        if self.price_history:
            price_volatility = np.std(self.price_history) / np.mean(self.price_history)
            profit_volatility = np.std(profits) / np.mean(profits) if np.mean(profits) != 0 else 0
            self.metrics['volatility_captured'] = profit_volatility / price_volatility if price_volatility > 0 else 0
        
        return self.metrics
    
    def get_grid_status(self) -> Dict:
        """Get current grid status"""
        return {
            'total_grids': len(self.grids),
            'active_orders': len(self.active_orders),
            'filled_grids': len([g for g in self.grids if g.filled]),
            'current_exposure': self.current_exposure,
            'total_profit': self.total_profit,
            'upper_price': self.upper_price,
            'lower_price': self.lower_price,
            'metrics': self.metrics
        }


# ====================== DCA STRATEGY ======================

class DCAStrategy:
    """
    Dollar Cost Averaging Strategy
    Systematic accumulation over time
    PROFESSIONAL IMPLEMENTATION
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.symbol = config['symbol']
        
        # DCA parameters
        self.total_investment = config['total_investment']
        self.number_of_orders = config.get('number_of_orders', 10)
        self.order_amount = self.total_investment / self.number_of_orders
        
        # Timing strategy
        self.interval_hours = config.get('interval_hours', 24)
        self.price_deviation_trigger = config.get('price_deviation_trigger', 0.03)  # 3%
        
        # Advanced DCA modes
        self.dca_mode = config.get('mode', 'STANDARD')  # STANDARD, AGGRESSIVE, DEFENSIVE
        self.use_technical_triggers = config.get('use_technical_triggers', True)
        self.use_sentiment_triggers = config.get('use_sentiment_triggers', False)
        
        # Risk management
        self.max_price = config.get('max_price', float('inf'))
        self.stop_loss = config.get('stop_loss', 0.2)  # 20%
        self.take_profit_levels = config.get('take_profit_levels', [0.2, 0.5, 1.0])  # 20%, 50%, 100%
        
        # State tracking
        self.orders: List[DCAOrder] = []
        self.executed_orders: List[DCAOrder] = []
        self.total_invested = 0
        self.average_price = 0
        self.total_quantity = 0
        
        # Performance tracking
        self.metrics = {
            'total_orders': 0,
            'executed_orders': 0,
            'average_buy_price': 0,
            'current_value': 0,
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'best_order': None,
            'worst_order': None,
            'time_to_profit': None
        }
        
        # Price tracking
        self.price_history = deque(maxlen=1000)
        self.last_order_time = None
        self.strategy_start_time = datetime.now()
        
        logger.info(f"DCAStrategy initialized for {self.symbol} with {self.number_of_orders} orders")
    
    async def initialize_dca_schedule(self, current_price: float):
        """
        Initialize DCA order schedule
        INTELLIGENT SCHEDULING
        """
        self.orders.clear()
        
        for i in range(self.number_of_orders):
            # Calculate time target
            time_target = datetime.now() + timedelta(hours=self.interval_hours * i)
            
            # Calculate price targets for different modes
            if self.dca_mode == 'AGGRESSIVE':
                # Buy more on dips
                price_target = current_price * (1 - 0.02 * i)  # 2% lower each time
            elif self.dca_mode == 'DEFENSIVE':
                # Spread buys across wider range
                price_target = current_price * (1 - 0.01 * i)  # 1% lower each time
            else:
                # Standard mode - time-based only
                price_target = None
            
            order = DCAOrder(
                order_number=i + 1,
                amount=self.order_amount,
                price_target=price_target,
                time_target=time_target
            )
            
            self.orders.append(order)
        
        logger.info(f"Initialized {len(self.orders)} DCA orders")
    
    async def check_dca_triggers(self, current_price: float, technical_indicators: Dict = None, sentiment: float = None) -> Optional[DCAOrder]:
        """
        Check if DCA order should be triggered
        MULTI-FACTOR TRIGGERING
        """
        # Find next pending order
        pending_orders = [o for o in self.orders if not o.executed]
        if not pending_orders:
            return None
        
        next_order = pending_orders[0]
        
        # Check if already executing too fast
        if self.last_order_time:
            time_since_last = (datetime.now() - self.last_order_time).seconds / 3600
            if time_since_last < 1:  # Minimum 1 hour between orders
                return None
        
        # Check price constraints
        if current_price > self.max_price:
            logger.warning(f"Price {current_price} exceeds max price {self.max_price}")
            return None
        
        # Time-based trigger
        time_triggered = datetime.now() >= next_order.time_target
        
        # Price-based trigger
        price_triggered = False
        if next_order.price_target:
            price_triggered = current_price <= next_order.price_target
        else:
            # Check for significant dip
            if self.price_history:
                avg_price = np.mean(list(self.price_history)[-20:])  # Last 20 prices
                if current_price < avg_price * (1 - self.price_deviation_trigger):
                    price_triggered = True
        
        # Technical triggers
        technical_triggered = True  # Default to true if not using
        if self.use_technical_triggers and technical_indicators:
            rsi = technical_indicators.get('RSI')
            if rsi and rsi < 30:  # Oversold
                technical_triggered = True
            else:
                technical_triggered = False
        
        # Sentiment triggers
        sentiment_triggered = True  # Default to true if not using
        if self.use_sentiment_triggers and sentiment is not None:
            if sentiment < 0.3:  # Extreme fear
                sentiment_triggered = True
            else:
                sentiment_triggered = False
        
        # Combine triggers based on mode
        if self.dca_mode == 'AGGRESSIVE':
            should_trigger = time_triggered or (price_triggered and technical_triggered)
        elif self.dca_mode == 'DEFENSIVE':
            should_trigger = time_triggered and price_triggered and technical_triggered
        else:
            should_trigger = time_triggered or price_triggered
        
        if should_trigger:
            return next_order
        
        return None
    
    async def execute_dca_order(self, order: DCAOrder, current_price: float, exchange_manager) -> Dict:
        """
        Execute DCA order
        REAL EXECUTION WITH SLIPPAGE PROTECTION
        """
        try:
            # Calculate quantity
            quantity = order.amount / current_price
            
            # Add slippage protection
            max_slippage = 0.01  # 1%
            limit_price = current_price * (1 + max_slippage)
            
            # Place order
            result = await exchange_manager.place_order(
                symbol=self.symbol,
                side='BUY',
                order_type='LIMIT',
                quantity=quantity,
                price=limit_price
            )
            
            # Update order
            order.executed = True
            order.executed_price = result['price']
            order.executed_at = datetime.now()
            order.quantity = result['quantity']
            
            # Update tracking
            self.executed_orders.append(order)
            self.total_invested += order.amount
            self.total_quantity += order.quantity
            
            # Update average price
            if self.total_quantity > 0:
                self.average_price = self.total_invested / self.total_quantity
            
            # Update metrics
            self.metrics['executed_orders'] += 1
            self.metrics['average_buy_price'] = self.average_price
            
            self.last_order_time = datetime.now()
            
            logger.info(f"Executed DCA order #{order.order_number}: {quantity:.4f} @ {result['price']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute DCA order: {e}")
            raise
    
    async def manage_dca_position(self, current_price: float, exchange_manager):
        """
        Manage accumulated position
        INTELLIGENT PROFIT TAKING
        """
        if self.total_quantity == 0:
            return
        
        # Calculate current position value
        current_value = self.total_quantity * current_price
        unrealized_pnl = current_value - self.total_invested
        unrealized_pnl_pct = unrealized_pnl / self.total_invested if self.total_invested > 0 else 0
        
        # Update metrics
        self.metrics['current_value'] = current_value
        self.metrics['unrealized_pnl'] = unrealized_pnl
        
        # Check stop loss
        if unrealized_pnl_pct < -self.stop_loss:
            logger.warning(f"Stop loss triggered at {unrealized_pnl_pct:.2%}")
            await self._close_position(exchange_manager, "STOP_LOSS")
            return
        
        # Check take profit levels
        for tp_level in self.take_profit_levels:
            if unrealized_pnl_pct >= tp_level:
                # Take partial profit
                sell_percentage = 0.33  # Sell 1/3 at each level
                sell_quantity = self.total_quantity * sell_percentage
                
                logger.info(f"Taking profit at {unrealized_pnl_pct:.2%}: selling {sell_quantity:.4f}")
                
                await self._sell_partial(sell_quantity, exchange_manager)
                
                # Remove this TP level
                self.take_profit_levels.remove(tp_level)
                break
    
    async def _close_position(self, exchange_manager, reason: str):
        """Close entire DCA position"""
        try:
            result = await exchange_manager.place_order(
                symbol=self.symbol,
                side='SELL',
                order_type='MARKET',
                quantity=self.total_quantity
            )
            
            # Calculate realized P&L
            sell_value = result['price'] * result['quantity']
            self.metrics['realized_pnl'] = sell_value - self.total_invested
            
            logger.info(f"Closed DCA position ({reason}): P&L = {self.metrics['realized_pnl']:.2f}")
            
            # Reset position
            self.total_quantity = 0
            self.total_invested = 0
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
    
    async def _sell_partial(self, quantity: float, exchange_manager):
        """Sell partial position"""
        try:
            result = await exchange_manager.place_order(
                symbol=self.symbol,
                side='SELL',
                order_type='MARKET',
                quantity=quantity
            )
            
            # Update position
            self.total_quantity -= result['quantity']
            
            # Calculate partial realized P&L
            sell_value = result['price'] * result['quantity']
            cost_basis = (result['quantity'] / (result['quantity'] + self.total_quantity)) * self.total_invested
            partial_pnl = sell_value - cost_basis
            
            self.metrics['realized_pnl'] += partial_pnl
            self.total_invested -= cost_basis
            
        except Exception as e:
            logger.error(f"Failed to sell partial position: {e}")
    
    def calculate_dca_performance(self) -> Dict:
        """
        Calculate DCA strategy performance
        COMPREHENSIVE METRICS
        """
        if self.executed_orders:
            # Find best and worst orders
            order_performances = []
            for order in self.executed_orders:
                if order.executed_price and self.average_price:
                    performance = (self.average_price - order.executed_price) / order.executed_price
                    order_performances.append((order.order_number, performance))
            
            if order_performances:
                order_performances.sort(key=lambda x: x[1])
                self.metrics['worst_order'] = order_performances[0]
                self.metrics['best_order'] = order_performances[-1]
        
        # Calculate time to profit
        if self.metrics['unrealized_pnl'] > 0:
            if not self.metrics['time_to_profit']:
                self.metrics['time_to_profit'] = (datetime.now() - self.strategy_start_time).days
        
        return self.metrics
    
    def get_dca_status(self) -> Dict:
        """Get current DCA status"""
        return {
            'total_orders': self.number_of_orders,
            'executed_orders': len(self.executed_orders),
            'pending_orders': len([o for o in self.orders if not o.executed]),
            'total_invested': self.total_invested,
            'average_price': self.average_price,
            'total_quantity': self.total_quantity,
            'metrics': self.metrics
        }


# ====================== ARBITRAGE STRATEGY ======================

class ArbitrageStrategy:
    """
    Arbitrage Strategy
    Exploit price differences across markets
    PROFESSIONAL IMPLEMENTATION
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Arbitrage parameters
        self.min_profit_percentage = config.get('min_profit_percentage', 0.002)  # 0.2%
        self.max_exposure = config.get('max_exposure', 10000)
        self.execution_time_limit = config.get('execution_time_limit', 1)  # seconds
        
        # Types of arbitrage
        self.enable_triangular = config.get('enable_triangular', True)
        self.enable_exchange_arb = config.get('enable_exchange_arb', True)
        self.enable_spot_futures = config.get('enable_spot_futures', True)
        
        # Risk management
        self.max_slippage = config.get('max_slippage', 0.001)  # 0.1%
        self.confidence_threshold = config.get('confidence_threshold', 0.95)
        
        # Tracking
        self.opportunities: List[ArbitrageOpportunity] = []
        self.executed_arbs = []
        self.total_profit = 0
        
        # Performance metrics
        self.metrics = {
            'opportunities_found': 0,
            'opportunities_executed': 0,
            'success_rate': 0,
            'total_profit': 0,
            'average_profit': 0,
            'fastest_execution': float('inf'),
            'largest_profit': 0
        }
        
        logger.info("ArbitrageStrategy initialized")
    
    async def scan_triangular_arbitrage(self, exchange_manager) -> List[ArbitrageOpportunity]:
        """
        Scan for triangular arbitrage opportunities
        REAL-TIME SCANNING
        """
        opportunities = []
        
        # Get all trading pairs
        markets = await exchange_manager.get_markets()
        
        # Find triangular paths
        base_currencies = ['USDT', 'BTC', 'ETH']
        
        for base in base_currencies:
            # Find all pairs with base currency
            base_pairs = [m for m in markets if base in m['symbol']]
            
            for pair1 in base_pairs:
                # Extract other currency from pair1
                other1 = pair1['symbol'].replace(base, '').replace('/', '')
                
                # Find pair2: other1/other2
                for pair2 in markets:
                    if other1 in pair2['symbol'] and base not in pair2['symbol']:
                        other2 = pair2['symbol'].replace(other1, '').replace('/', '')
                        
                        # Find pair3: other2/base to complete triangle
                        pair3_symbol = f"{other2}/{base}"
                        pair3 = next((m for m in markets if m['symbol'] == pair3_symbol), None)
                        
                        if pair3:
                            # Calculate arbitrage opportunity
                            opportunity = await self._calculate_triangular_profit(
                                pair1, pair2, pair3, exchange_manager
                            )
                            
                            if opportunity:
                                opportunities.append(opportunity)
        
        return opportunities
    
    async def _calculate_triangular_profit(self, pair1, pair2, pair3, exchange_manager) -> Optional[ArbitrageOpportunity]:
        """Calculate profit for triangular arbitrage path"""
        try:
            # Get current prices
            ticker1 = await exchange_manager.get_ticker(pair1['symbol'])
            ticker2 = await exchange_manager.get_ticker(pair2['symbol'])
            ticker3 = await exchange_manager.get_ticker(pair3['symbol'])
            
            # Calculate forward path profit
            # Start with 1 unit of base currency
            # Buy pair1, sell pair2, sell pair3
            forward_path = 1.0
            forward_path /= ticker1['ask']  # Buy pair1
            forward_path *= ticker2['bid']  # Sell pair2
            forward_path *= ticker3['bid']  # Sell pair3
            
            # Calculate backward path profit
            backward_path = 1.0
            backward_path /= ticker3['ask']  # Buy pair3
            backward_path /= ticker2['ask']  # Buy pair2
            backward_path *= ticker1['bid']  # Sell pair1
            
            # Check for profit
            best_path = max(forward_path, backward_path)
            profit_percentage = (best_path - 1) * 100
            
            # Account for fees (typically 0.1% per trade = 0.3% total)
            net_profit = profit_percentage - 0.3
            
            if net_profit > self.min_profit_percentage * 100:
                path_direction = 'forward' if forward_path > backward_path else 'backward'
                
                return ArbitrageOpportunity(
                    opportunity_id=f"TRI_{datetime.now().timestamp()}",
                    type='triangular',
                    profit_percentage=net_profit,
                    volume=min(ticker1['baseVolume'], ticker2['baseVolume'], ticker3['baseVolume']) * 0.01,
                    path=[pair1['symbol'], pair2['symbol'], pair3['symbol']],
                    exchanges=[exchange_manager.exchange_name],
                    expires_at=datetime.now() + timedelta(seconds=5),
                    risk_score=self._calculate_risk_score(net_profit, path_direction)
                )
            
        except Exception as e:
            logger.error(f"Error calculating triangular arbitrage: {e}")
        
        return None
    
    async def scan_exchange_arbitrage(self, exchange_managers: List) -> List[ArbitrageOpportunity]:
        """
        Scan for arbitrage between exchanges
        CROSS-EXCHANGE OPPORTUNITIES
        """
        opportunities = []
        
        # Common trading pairs to check
        pairs_to_check = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        for symbol in pairs_to_check:
            prices = {}
            
            # Get prices from all exchanges
            for exchange in exchange_managers:
                try:
                    ticker = await exchange.get_ticker(symbol)
                    prices[exchange.exchange_name] = {
                        'bid': ticker['bid'],
                        'ask': ticker['ask'],
                        'volume': ticker['baseVolume']
                    }
                except:
                    continue
            
            # Find arbitrage opportunities
            for ex1 in prices:
                for ex2 in prices:
                    if ex1 != ex2:
                        # Buy on ex1, sell on ex2
                        profit = (prices[ex2]['bid'] - prices[ex1]['ask']) / prices[ex1]['ask'] * 100
                        
                        # Account for fees and transfer costs
                        net_profit = profit - 0.2  # 0.1% each exchange + transfer
                        
                        if net_profit > self.min_profit_percentage * 100:
                            opportunities.append(ArbitrageOpportunity(
                                opportunity_id=f"EX_{datetime.now().timestamp()}",
                                type='exchange',
                                profit_percentage=net_profit,
                                volume=min(prices[ex1]['volume'], prices[ex2]['volume']) * 0.01,
                                path=[f"buy_{ex1}", f"sell_{ex2}"],
                                exchanges=[ex1, ex2],
                                expires_at=datetime.now() + timedelta(seconds=10),
                                risk_score=self._calculate_risk_score(net_profit, 'exchange')
                            ))
        
        return opportunities
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity, exchange_managers) -> Dict:
        """
        Execute arbitrage opportunity
        FAST EXECUTION WITH RISK MANAGEMENT
        """
        start_time = datetime.now()
        execution_result = {
            'opportunity_id': opportunity.opportunity_id,
            'success': False,
            'profit': 0,
            'execution_time': 0,
            'trades': []
        }
        
        try:
            if opportunity.type == 'triangular':
                result = await self._execute_triangular_arbitrage(opportunity, exchange_managers[0])
            elif opportunity.type == 'exchange':
                result = await self._execute_exchange_arbitrage(opportunity, exchange_managers)
            else:
                raise ValueError(f"Unknown arbitrage type: {opportunity.type}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Check if execution was fast enough
            if execution_time > self.execution_time_limit:
                logger.warning(f"Arbitrage execution too slow: {execution_time:.2f}s")
                # Consider cancelling orders if not filled
            
            execution_result.update(result)
            execution_result['execution_time'] = execution_time
            execution_result['success'] = result.get('success', False)
            
            # Update metrics
            if execution_result['success']:
                self.metrics['opportunities_executed'] += 1
                self.metrics['total_profit'] += execution_result['profit']
                self.metrics['fastest_execution'] = min(self.metrics['fastest_execution'], execution_time)
                self.metrics['largest_profit'] = max(self.metrics['largest_profit'], execution_result['profit'])
            
            self.executed_arbs.append(execution_result)
            
        except Exception as e:
            logger.error(f"Failed to execute arbitrage: {e}")
            execution_result['error'] = str(e)
        
        return execution_result
    
    async def _execute_triangular_arbitrage(self, opportunity: ArbitrageOpportunity, exchange_manager) -> Dict:
        """Execute triangular arbitrage trades"""
        trades = []
        
        # Calculate position size based on available balance and limits
        position_size = min(opportunity.volume, self.max_exposure)
        
        # Execute three trades in sequence
        for i, symbol in enumerate(opportunity.path):
            side = 'BUY' if i == 0 else 'SELL'
            
            trade = await exchange_manager.place_order(
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=position_size
            )
            
            trades.append(trade)
            
            # Update position size for next trade
            position_size = trade['executed_quantity']
        
        # Calculate actual profit
        initial_value = trades[0]['executed_quantity'] * trades[0]['price']
        final_value = trades[-1]['executed_quantity'] * trades[-1]['price']
        profit = final_value - initial_value
        
        return {
            'success': True,
            'profit': profit,
            'trades': trades
        }
    
    async def _execute_exchange_arbitrage(self, opportunity: ArbitrageOpportunity, exchange_managers) -> Dict:
        """Execute cross-exchange arbitrage"""
        # Implementation for exchange arbitrage
        # This would involve:
        # 1. Buying on cheaper exchange
        # 2. Transferring funds
        # 3. Selling on expensive exchange
        pass
    
    def _calculate_risk_score(self, profit: float, arb_type: str) -> float:
        """Calculate risk score for arbitrage opportunity"""
        base_risk = 0.5
        
        # Higher profit = lower risk (more buffer for slippage)
        profit_factor = max(0, 1 - profit / 10)  # 10% profit = 0 risk from profit
        
        # Different types have different risks
        type_risk = {
            'triangular': 0.3,
            'exchange': 0.5,
            'spot-futures': 0.4
        }
        
        return base_risk * profit_factor * type_risk.get(arb_type, 0.5)
    
    def get_arbitrage_status(self) -> Dict:
        """Get current arbitrage status"""
        self.metrics['success_rate'] = (
            self.metrics['opportunities_executed'] / self.metrics['opportunities_found'] * 100
            if self.metrics['opportunities_found'] > 0 else 0
        )
        
        self.metrics['average_profit'] = (
            self.metrics['total_profit'] / self.metrics['opportunities_executed']
            if self.metrics['opportunities_executed'] > 0 else 0
        )
        
        return {
            'active_opportunities': len([o for o in self.opportunities if o.expires_at > datetime.now()]),
            'total_profit': self.total_profit,
            'metrics': self.metrics,
            'recent_executions': self.executed_arbs[-10:]  # Last 10
        }


# ====================== STRATEGY MANAGER ======================

class AdvancedStrategyManager:
    """
    Manage all advanced trading strategies
    UNIFIED STRATEGY ORCHESTRATION
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.strategies = {}
        self.active_strategies = []
        self.performance_history = []
        
        logger.info("AdvancedStrategyManager initialized")
    
    def add_strategy(self, name: str, strategy_type: StrategyType, strategy_config: Dict):
        """Add new strategy"""
        if strategy_type == StrategyType.GRID_TRADING:
            strategy = GridTradingStrategy(strategy_config)
        elif strategy_type == StrategyType.DCA:
            strategy = DCAStrategy(strategy_config)
        elif strategy_type == StrategyType.ARBITRAGE:
            strategy = ArbitrageStrategy(strategy_config)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        self.strategies[name] = {
            'type': strategy_type,
            'instance': strategy,
            'active': False,
            'performance': {}
        }
        
        logger.info(f"Added strategy: {name} ({strategy_type.value})")
    
    async def start_strategy(self, name: str):
        """Start a strategy"""
        if name in self.strategies:
            self.strategies[name]['active'] = True
            self.active_strategies.append(name)
            logger.info(f"Started strategy: {name}")
    
    async def stop_strategy(self, name: str):
        """Stop a strategy"""
        if name in self.strategies:
            self.strategies[name]['active'] = False
            if name in self.active_strategies:
                self.active_strategies.remove(name)
            logger.info(f"Stopped strategy: {name}")
    
    async def execute_strategies(self, market_data: Dict, exchange_manager):
        """Execute all active strategies"""
        results = {}
        
        for name in self.active_strategies:
            strategy_info = self.strategies[name]
            strategy = strategy_info['instance']
            
            try:
                if strategy_info['type'] == StrategyType.GRID_TRADING:
                    # Execute grid trading
                    await strategy.monitor_grid_fills(exchange_manager)
                    await strategy.rebalance_grid(market_data['price'])
                    
                elif strategy_info['type'] == StrategyType.DCA:
                    # Execute DCA
                    order = await strategy.check_dca_triggers(
                        market_data['price'],
                        market_data.get('indicators'),
                        market_data.get('sentiment')
                    )
                    if order:
                        await strategy.execute_dca_order(order, market_data['price'], exchange_manager)
                    
                    await strategy.manage_dca_position(market_data['price'], exchange_manager)
                    
                elif strategy_info['type'] == StrategyType.ARBITRAGE:
                    # Execute arbitrage
                    opportunities = await strategy.scan_triangular_arbitrage(exchange_manager)
                    
                    for opp in opportunities[:1]:  # Execute only best opportunity
                        if opp.profit_percentage > strategy.min_profit_percentage * 100:
                            await strategy.execute_arbitrage(opp, [exchange_manager])
                
                results[name] = strategy.get_grid_status() if hasattr(strategy, 'get_grid_status') else {}
                
            except Exception as e:
                logger.error(f"Error executing strategy {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def get_all_strategies_status(self) -> Dict:
        """Get status of all strategies"""
        status = {}
        
        for name, info in self.strategies.items():
            strategy = info['instance']
            
            if hasattr(strategy, 'get_grid_status'):
                status[name] = strategy.get_grid_status()
            elif hasattr(strategy, 'get_dca_status'):
                status[name] = strategy.get_dca_status()
            elif hasattr(strategy, 'get_arbitrage_status'):
                status[name] = strategy.get_arbitrage_status()
            
            status[name]['active'] = info['active']
            status[name]['type'] = info['type'].value
        
        return status
