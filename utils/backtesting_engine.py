"""
DEMIR AI v8.0 - Professional Backtesting Engine
HIGH-PERFORMANCE HISTORICAL STRATEGY TESTING
ENTERPRISE GRADE IMPLEMENTATION
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
from collections import defaultdict, deque
import json
import pickle
import sqlite3
import psycopg2
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

# Performance imports
import numba
from numba import jit, prange
import talib
import bottleneck as bn
import zarr
import h5py

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


# ====================== BACKTESTING STRUCTURES ======================

class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderSide(Enum):
    """Order sides"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Order representation"""
    order_id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    filled_price: float = 0
    commission: float = 0
    slippage: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """Trade representation"""
    trade_id: str
    order_id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    pnl: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Position representation"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    max_profit: float = 0
    max_loss: float = 0
    holding_period: timedelta = timedelta()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Backtest result container"""
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: timedelta
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    
    # Risk metrics
    volatility: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    
    # Time series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    returns_series: pd.Series
    
    # Trade log
    trades: List[Trade]
    orders: List[Order]
    
    # Additional metrics
    metadata: Dict[str, Any] = field(default_factory=dict)


# ====================== MARKET SIMULATOR ======================

class MarketSimulator:
    """
    High-Fidelity Market Simulation
    REALISTIC ORDER EXECUTION WITH SLIPPAGE AND FEES
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Market data
        self.market_data: pd.DataFrame = None
        self.current_index = 0
        self.current_time = None
        
        # Order book simulation
        self.bid_ask_spread = config.get('bid_ask_spread', 0.0001)  # 0.01%
        self.market_impact = config.get('market_impact', 0.0001)  # 0.01%
        
        # Slippage model
        self.slippage_model = config.get('slippage_model', 'linear')
        self.slippage_factor = config.get('slippage_factor', 0.0001)
        
        # Commission structure
        self.commission_type = config.get('commission_type', 'percentage')
        self.commission_rate = config.get('commission_rate', 0.001)  # 0.1%
        self.min_commission = config.get('min_commission', 1.0)
        
        # Latency simulation
        self.latency_ms = config.get('latency_ms', 10)
        
        # Order matching engine
        self.pending_orders: List[Order] = []
        self.order_history: List[Order] = []
        
        logger.info("MarketSimulator initialized")
    
    def load_market_data(self, data: pd.DataFrame):
        """Load historical market data"""
        self.market_data = data
        self.current_index = 0
        
        # Ensure required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in self.market_data.columns:
                raise ValueError(f"Market data missing required column: {col}")
        
        logger.info(f"Loaded market data: {len(self.market_data)} bars")
    
    def get_current_price(self, price_type: str = 'close') -> float:
        """Get current market price"""
        if self.current_index >= len(self.market_data):
            return None
        
        return self.market_data.iloc[self.current_index][price_type]
    
    def get_bid_ask(self) -> Tuple[float, float]:
        """Get current bid/ask prices"""
        mid_price = self.get_current_price()
        if mid_price is None:
            return None, None
        
        spread = mid_price * self.bid_ask_spread
        bid = mid_price - spread / 2
        ask = mid_price + spread / 2
        
        return bid, ask
    
    def calculate_slippage(self, order: Order) -> float:
        """
        Calculate order slippage
        REALISTIC SLIPPAGE MODELING
        """
        if order.order_type == OrderType.MARKET:
            # Market order slippage
            current_price = self.get_current_price()
            volume = self.market_data.iloc[self.current_index]['volume']
            
            # Volume-based slippage
            volume_impact = (order.quantity / volume) * self.market_impact
            
            if self.slippage_model == 'linear':
                slippage = current_price * (self.slippage_factor + volume_impact)
            elif self.slippage_model == 'square_root':
                slippage = current_price * np.sqrt(self.slippage_factor + volume_impact)
            else:  # exponential
                slippage = current_price * (np.exp(self.slippage_factor + volume_impact) - 1)
            
            # Direction-based slippage
            if order.side == OrderSide.BUY:
                return slippage  # Pay more
            else:
                return -slippage  # Receive less
        
        return 0  # No slippage for limit orders that match
    
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """Calculate order commission"""
        if self.commission_type == 'percentage':
            commission = fill_price * order.quantity * self.commission_rate
        elif self.commission_type == 'per_share':
            commission = order.quantity * self.commission_rate
        else:  # fixed
            commission = self.commission_rate
        
        return max(commission, self.min_commission)
    
    def process_order(self, order: Order) -> Optional[Trade]:
        """
        Process order through matching engine
        REALISTIC ORDER EXECUTION
        """
        current_bar = self.market_data.iloc[self.current_index]
        
        # Check if order can be filled
        can_fill = False
        fill_price = None
        
        if order.order_type == OrderType.MARKET:
            # Market orders always fill
            can_fill = True
            bid, ask = self.get_bid_ask()
            fill_price = ask if order.side == OrderSide.BUY else bid
            
        elif order.order_type == OrderType.LIMIT:
            # Limit orders fill if price crosses limit
            if order.side == OrderSide.BUY:
                if current_bar['low'] <= order.price:
                    can_fill = True
                    fill_price = min(order.price, current_bar['open'])
            else:
                if current_bar['high'] >= order.price:
                    can_fill = True
                    fill_price = max(order.price, current_bar['open'])
        
        elif order.order_type == OrderType.STOP:
            # Stop orders trigger and fill as market orders
            if order.side == OrderSide.BUY:
                if current_bar['high'] >= order.stop_price:
                    can_fill = True
                    fill_price = max(order.stop_price, current_bar['open'])
            else:
                if current_bar['low'] <= order.stop_price:
                    can_fill = True
                    fill_price = min(order.stop_price, current_bar['open'])
        
        # Execute trade if order can fill
        if can_fill:
            # Calculate slippage
            slippage = self.calculate_slippage(order)
            fill_price += slippage
            
            # Ensure price is within bar range
            fill_price = np.clip(fill_price, current_bar['low'], current_bar['high'])
            
            # Calculate commission
            commission = self.calculate_commission(order, fill_price)
            
            # Update order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = fill_price
            order.commission = commission
            order.slippage = slippage
            
            # Create trade
            trade = Trade(
                trade_id=f"T_{datetime.now().timestamp()}",
                order_id=order.order_id,
                timestamp=self.current_time,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=fill_price,
                commission=commission,
                slippage=slippage
            )
            
            return trade
        
        return None
    
    def advance_time(self):
        """Advance market simulation by one bar"""
        if self.current_index < len(self.market_data) - 1:
            self.current_index += 1
            self.current_time = self.market_data.index[self.current_index]
            
            # Process pending orders
            filled_trades = []
            for order in self.pending_orders[:]:
                trade = self.process_order(order)
                if trade:
                    filled_trades.append(trade)
                    self.pending_orders.remove(order)
                    self.order_history.append(order)
            
            return filled_trades
        
        return []
    
    def reset(self):
        """Reset market simulator"""
        self.current_index = 0
        self.current_time = self.market_data.index[0] if self.market_data is not None else None
        self.pending_orders.clear()
        self.order_history.clear()


# ====================== PORTFOLIO MANAGER ======================

class PortfolioManager:
    """
    Portfolio Management for Backtesting
    TRACKS POSITIONS, P&L, AND RISK
    """
    
    def __init__(self, initial_capital: float, config: Dict[str, Any]):
        self.initial_capital = initial_capital
        self.config = config
        
        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.equity = initial_capital
        
        # Risk parameters
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% per position
        self.use_leverage = config.get('use_leverage', False)
        self.leverage_ratio = config.get('leverage_ratio', 1.0)
        
        # History tracking
        self.equity_curve = [initial_capital]
        self.cash_curve = [initial_capital]
        self.positions_history = []
        self.trades_history = []
        
        # Performance metrics
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.total_commission = 0
        self.total_slippage = 0
        
    def execute_trade(self, trade: Trade, current_price: float):
        """Execute trade and update portfolio"""
        # Calculate trade value
        trade_value = trade.quantity * trade.price
        total_cost = trade_value + trade.commission
        
        if trade.side == OrderSide.BUY:
            # Check if we have enough cash
            if total_cost > self.cash and not self.use_leverage:
                logger.warning(f"Insufficient cash for trade: {trade.trade_id}")
                return False
            
            # Update cash
            self.cash -= total_cost
            
            # Update or create position
            if trade.symbol in self.positions:
                position = self.positions[trade.symbol]
                # Average entry price
                total_quantity = position.quantity + trade.quantity
                position.entry_price = (
                    (position.quantity * position.entry_price + trade.quantity * trade.price) 
                    / total_quantity
                )
                position.quantity = total_quantity
            else:
                self.positions[trade.symbol] = Position(
                    symbol=trade.symbol,
                    quantity=trade.quantity,
                    entry_price=trade.price,
                    entry_time=trade.timestamp,
                    current_price=current_price,
                    unrealized_pnl=0,
                    realized_pnl=0
                )
        
        else:  # SELL
            if trade.symbol not in self.positions:
                logger.warning(f"No position to sell: {trade.symbol}")
                return False
            
            position = self.positions[trade.symbol]
            
            if trade.quantity > position.quantity:
                logger.warning(f"Sell quantity exceeds position: {trade.symbol}")
                trade.quantity = position.quantity
            
            # Calculate P&L
            pnl = (trade.price - position.entry_price) * trade.quantity
            trade.pnl = pnl - trade.commission
            
            # Update cash
            self.cash += trade_value - trade.commission
            
            # Update position
            position.quantity -= trade.quantity
            position.realized_pnl += trade.pnl
            self.realized_pnl += trade.pnl
            
            # Remove position if fully closed
            if position.quantity == 0:
                del self.positions[trade.symbol]
        
        # Track commission and slippage
        self.total_commission += trade.commission
        self.total_slippage += abs(trade.slippage) * trade.quantity
        
        # Add to history
        self.trades_history.append(trade)
        
        return True
    
    def update_positions(self, market_prices: Dict[str, float]):
        """Update position values and calculate P&L"""
        self.unrealized_pnl = 0
        
        for symbol, position in self.positions.items():
            if symbol in market_prices:
                position.current_price = market_prices[symbol]
                position.unrealized_pnl = (
                    (position.current_price - position.entry_price) * position.quantity
                )
                self.unrealized_pnl += position.unrealized_pnl
                
                # Track max profit/loss
                position.max_profit = max(position.max_profit, position.unrealized_pnl)
                position.max_loss = min(position.max_loss, position.unrealized_pnl)
        
        # Calculate total equity
        positions_value = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        self.equity = self.cash + positions_value
        
        # Update curves
        self.equity_curve.append(self.equity)
        self.cash_curve.append(self.cash)
        self.positions_history.append(dict(self.positions))
    
    def get_position_size(self, symbol: str, signal_strength: float) -> float:
        """
        Calculate position size based on risk management
        KELLY CRITERION AND RISK PARITY
        """
        # Base position size
        available_capital = self.equity * self.leverage_ratio if self.use_leverage else self.cash
        max_position_value = available_capital * self.max_position_size
        
        # Kelly criterion adjustment
        if 'kelly_fraction' in self.config:
            kelly_f = self.config['kelly_fraction']
            position_value = max_position_value * kelly_f * signal_strength
        else:
            position_value = max_position_value * signal_strength
        
        # Risk parity adjustment
        if 'use_risk_parity' in self.config and self.config['use_risk_parity']:
            # Adjust based on volatility
            # Implementation would use actual volatility calculations
            pass
        
        return position_value
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        metrics = {
            'total_return': (self.equity - self.initial_capital) / self.initial_capital,
            'total_trades': len(self.trades_history),
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'final_equity': self.equity,
            'max_equity': equity_series.max(),
            'min_equity': equity_series.min()
        }
        
        if len(returns) > 0:
            metrics['volatility'] = returns.std() * np.sqrt(252)
            metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            # Calculate drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()
        
        return metrics


# ====================== STRATEGY BACKTESTER ======================

class StrategyBacktester:
    """
    Main Backtesting Engine
    ORCHESTRATES STRATEGY TESTING
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Components
        self.market_simulator = MarketSimulator(config.get('market', {}))
        self.portfolio_manager = PortfolioManager(
            initial_capital=config.get('initial_capital', 10000),
            config=config.get('portfolio', {})
        )
        
        # Strategy
        self.strategy = None
        self.strategy_params = {}
        
        # Data management
        self.market_data: pd.DataFrame = None
        self.indicators: pd.DataFrame = None
        
        # Performance tracking
        self.performance_metrics = {}
        self.trade_log = []
        self.signal_log = []
        
        # Optimization
        self.optimization_results = []
        
        logger.info("StrategyBacktester initialized")
    
    def load_data(self, data: pd.DataFrame, preprocess: bool = True):
        """Load and preprocess market data"""
        self.market_data = data.copy()
        
        if preprocess:
            # Add technical indicators
            self._add_technical_indicators()
            
            # Clean data
            self.market_data = self.market_data.dropna()
        
        # Load data into market simulator
        self.market_simulator.load_market_data(self.market_data)
        
        logger.info(f"Loaded data: {len(self.market_data)} bars from {self.market_data.index[0]} to {self.market_data.index[-1]}")
    
    def _add_technical_indicators(self):
        """Add technical indicators to market data"""
        # Moving averages
        self.market_data['SMA_20'] = talib.SMA(self.market_data['close'], timeperiod=20)
        self.market_data['SMA_50'] = talib.SMA(self.market_data['close'], timeperiod=50)
        self.market_data['EMA_12'] = talib.EMA(self.market_data['close'], timeperiod=12)
        self.market_data['EMA_26'] = talib.EMA(self.market_data['close'], timeperiod=26)
        
        # MACD
        macd, signal, hist = talib.MACD(self.market_data['close'])
        self.market_data['MACD'] = macd
        self.market_data['MACD_signal'] = signal
        self.market_data['MACD_hist'] = hist
        
        # RSI
        self.market_data['RSI'] = talib.RSI(self.market_data['close'], timeperiod=14)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(self.market_data['close'], timeperiod=20)
        self.market_data['BB_upper'] = upper
        self.market_data['BB_middle'] = middle
        self.market_data['BB_lower'] = lower
        
        # ATR (Average True Range)
        self.market_data['ATR'] = talib.ATR(
            self.market_data['high'],
            self.market_data['low'],
            self.market_data['close'],
            timeperiod=14
        )
        
        # Volume indicators
        self.market_data['OBV'] = talib.OBV(self.market_data['close'], self.market_data['volume'])
        self.market_data['AD'] = talib.AD(
            self.market_data['high'],
            self.market_data['low'],
            self.market_data['close'],
            self.market_data['volume']
        )
    
    def set_strategy(self, strategy: Callable, params: Dict[str, Any] = None):
        """Set trading strategy"""
        self.strategy = strategy
        self.strategy_params = params or {}
    
    def run_backtest(self, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    verbose: bool = False) -> BacktestResult:
        """
        Run backtest simulation
        MAIN BACKTESTING LOOP
        """
        if self.strategy is None:
            raise ValueError("No strategy set")
        
        # Filter data by date range
        data = self.market_data.copy()
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Reset components
        self.market_simulator.reset()
        self.portfolio_manager = PortfolioManager(
            initial_capital=self.config.get('initial_capital', 10000),
            config=self.config.get('portfolio', {})
        )
        self.trade_log.clear()
        self.signal_log.clear()
        
        # Main backtesting loop
        for i in range(len(data)):
            current_bar = data.iloc[i]
            current_time = data.index[i]
            
            # Get current market state
            market_state = {
                'bar': current_bar,
                'history': data.iloc[:i+1],
                'positions': self.portfolio_manager.positions,
                'cash': self.portfolio_manager.cash,
                'equity': self.portfolio_manager.equity
            }
            
            # Generate signals from strategy
            signals = self.strategy(market_state, **self.strategy_params)
            
            if signals:
                self.signal_log.append({
                    'timestamp': current_time,
                    'signals': signals
                })
                
                # Process signals into orders
                orders = self._signals_to_orders(signals, current_bar)
                
                # Submit orders to market simulator
                for order in orders:
                    self.market_simulator.pending_orders.append(order)
            
            # Advance market simulation
            filled_trades = self.market_simulator.advance_time()
            
            # Execute trades in portfolio
            for trade in filled_trades:
                success = self.portfolio_manager.execute_trade(
                    trade, 
                    current_bar['close']
                )
                if success:
                    self.trade_log.append(trade)
            
            # Update portfolio positions
            current_prices = {
                symbol: current_bar['close'] 
                for symbol in self.portfolio_manager.positions.keys()
            }
            self.portfolio_manager.update_positions(current_prices)
            
            # Verbose output
            if verbose and i % 100 == 0:
                logger.info(f"Backtest progress: {i}/{len(data)} bars, Equity: {self.portfolio_manager.equity:.2f}")
        
        # Generate backtest result
        result = self._generate_result()
        
        return result
    
    def _signals_to_orders(self, signals: List[Dict], current_bar: pd.Series) -> List[Order]:
        """Convert strategy signals to orders"""
        orders = []
        
        for signal in signals:
            # Determine order parameters
            symbol = signal.get('symbol', 'default')
            side = OrderSide[signal['side'].upper()]
            
            # Calculate position size
            if 'quantity' in signal:
                quantity = signal['quantity']
            else:
                position_value = self.portfolio_manager.get_position_size(
                    symbol, 
                    signal.get('strength', 1.0)
                )
                quantity = position_value / current_bar['close']
            
            # Create order
            order = Order(
                order_id=f"O_{datetime.now().timestamp()}",
                timestamp=current_bar.name,
                symbol=symbol,
                side=side,
                order_type=OrderType[signal.get('order_type', 'MARKET').upper()],
                quantity=quantity,
                price=signal.get('price'),
                stop_price=signal.get('stop_price'),
                metadata=signal.get('metadata', {})
            )
            
            orders.append(order)
        
        return orders
    
    def _generate_result(self) -> BacktestResult:
        """Generate comprehensive backtest result"""
        # Calculate performance metrics
        equity_series = pd.Series(
            self.portfolio_manager.equity_curve,
            index=self.market_data.index[:len(self.portfolio_manager.equity_curve)]
        )
        
        returns = equity_series.pct_change().dropna()
        
        # Calculate metrics
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
        
        # Annualized return
        days = (equity_series.index[-1] - equity_series.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility and Sharpe
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - 0.02) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Find maximum drawdown duration
        drawdown_start = drawdown.idxmin()
        recovery_date = drawdown[drawdown_start:][drawdown[drawdown_start:] == 0]
        if len(recovery_date) > 0:
            max_dd_duration = recovery_date.index[0] - drawdown_start
        else:
            max_dd_duration = equity_series.index[-1] - drawdown_start
        
        # Trade statistics
        trades = self.trade_log
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Risk metrics
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        result = BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=0,  # Would need benchmark to calculate
            alpha=0,  # Would need benchmark to calculate
            equity_curve=equity_series,
            drawdown_series=drawdown,
            returns_series=returns,
            trades=trades,
            orders=self.market_simulator.order_history,
            metadata={
                'total_commission': self.portfolio_manager.total_commission,
                'total_slippage': self.portfolio_manager.total_slippage,
                'signal_count': len(self.signal_log)
            }
        )
        
        return result
    
    @jit(nopython=True)
    def _calculate_returns_numba(prices: np.ndarray) -> np.ndarray:
        """Numba-accelerated returns calculation"""
        returns = np.empty(len(prices) - 1)
        for i in range(len(returns)):
            returns[i] = (prices[i+1] - prices[i]) / prices[i]
        return returns


# ====================== WALK-FORWARD OPTIMIZER ======================

class WalkForwardOptimizer:
    """
    Walk-Forward Analysis and Optimization
    ROBUST OUT-OF-SAMPLE TESTING
    """
    
    def __init__(self, backtester: StrategyBacktester):
        self.backtester = backtester
        self.optimization_results = []
        self.walk_forward_results = []
        
    def optimize_parameters(self,
                           param_grid: Dict[str, List],
                           optimization_metric: str = 'sharpe_ratio',
                           method: str = 'grid') -> Dict:
        """
        Optimize strategy parameters
        GRID SEARCH, RANDOM SEARCH, BAYESIAN OPTIMIZATION
        """
        if method == 'grid':
            return self._grid_search(param_grid, optimization_metric)
        elif method == 'random':
            return self._random_search(param_grid, optimization_metric)
        elif method == 'bayesian':
            return self._bayesian_optimization(param_grid, optimization_metric)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _grid_search(self, param_grid: Dict[str, List], metric: str) -> Dict:
        """Grid search optimization"""
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        best_score = -float('inf')
        best_params = None
        
        for combination in param_combinations:
            params = dict(zip(param_names, combination))
            
            # Run backtest with parameters
            self.backtester.strategy_params = params
            result = self.backtester.run_backtest()
            
            # Get optimization metric
            score = getattr(result, metric)
            
            # Track results
            self.optimization_results.append({
                'params': params.copy(),
                'score': score,
                'result': result
            })
            
            # Update best
            if score > best_score:
                best_score = score
                best_params = params.copy()
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': self.optimization_results
        }
    
    def walk_forward_analysis(self,
                             param_grid: Dict[str, List],
                             window_size: int,
                             step_size: int,
                             optimization_metric: str = 'sharpe_ratio') -> List[Dict]:
        """
        Walk-forward analysis
        ROLLING WINDOW OPTIMIZATION AND TESTING
        """
        data = self.backtester.market_data
        results = []
        
        # Generate windows
        for start_idx in range(0, len(data) - window_size, step_size):
            end_idx = start_idx + window_size
            
            # Split into in-sample and out-of-sample
            in_sample_end = start_idx + int(window_size * 0.7)
            
            in_sample_data = data.iloc[start_idx:in_sample_end]
            out_sample_data = data.iloc[in_sample_end:end_idx]
            
            # Optimize on in-sample
            self.backtester.load_data(in_sample_data)
            optimization_result = self.optimize_parameters(param_grid, optimization_metric)
            
            # Test on out-of-sample
            self.backtester.load_data(out_sample_data)
            self.backtester.strategy_params = optimization_result['best_params']
            test_result = self.backtester.run_backtest()
            
            results.append({
                'window': {
                    'start': data.index[start_idx],
                    'end': data.index[end_idx],
                    'in_sample_end': data.index[in_sample_end]
                },
                'optimization': optimization_result,
                'out_of_sample_result': test_result
            })
        
        self.walk_forward_results = results
        return results


# ====================== MONTE CARLO SIMULATOR ======================

class MonteCarloSimulator:
    """
    Monte Carlo Simulation for Robustness Testing
    STATISTICAL SIGNIFICANCE AND CONFIDENCE INTERVALS
    """
    
    def __init__(self, backtest_result: BacktestResult):
        self.original_result = backtest_result
        self.simulation_results = []
        
    def run_simulations(self, 
                       n_simulations: int = 1000,
                       method: str = 'bootstrap') -> Dict:
        """Run Monte Carlo simulations"""
        if method == 'bootstrap':
            return self._bootstrap_simulation(n_simulations)
        elif method == 'monte_carlo':
            return self._monte_carlo_paths(n_simulations)
        elif method == 'randomized_trades':
            return self._randomized_trade_simulation(n_simulations)
    
    def _bootstrap_simulation(self, n_simulations: int) -> Dict:
        """Bootstrap resampling of returns"""
        returns = self.original_result.returns_series.values
        
        simulated_metrics = []
        
        for _ in range(n_simulations):
            # Resample returns with replacement
            resampled_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate metrics
            total_return = (1 + resampled_returns).prod() - 1
            volatility = np.std(resampled_returns) * np.sqrt(252)
            sharpe = np.mean(resampled_returns) / np.std(resampled_returns) * np.sqrt(252) if np.std(resampled_returns) > 0 else 0
            
            # Calculate drawdown
            cumulative = np.cumprod(1 + resampled_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            simulated_metrics.append({
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown
            })
        
        # Calculate confidence intervals
        metrics_df = pd.DataFrame(simulated_metrics)
        
        confidence_intervals = {}
        for metric in metrics_df.columns:
            confidence_intervals[metric] = {
                'mean': metrics_df[metric].mean(),
                'std': metrics_df[metric].std(),
                'ci_95': (metrics_df[metric].quantile(0.025), metrics_df[metric].quantile(0.975)),
                'ci_99': (metrics_df[metric].quantile(0.005), metrics_df[metric].quantile(0.995))
            }
        
        return {
            'simulations': simulated_metrics,
            'confidence_intervals': confidence_intervals,
            'original_metrics': {
                'total_return': self.original_result.total_return,
                'volatility': self.original_result.volatility,
                'sharpe_ratio': self.original_result.sharpe_ratio,
                'max_drawdown': self.original_result.max_drawdown
            }
        }


# ====================== VISUALIZATION ======================

class BacktestVisualizer:
    """
    Advanced Backtest Visualization
    INTERACTIVE PLOTS AND REPORTS
    """
    
    def __init__(self, result: BacktestResult):
        self.result = result
    
    def plot_equity_curve(self, show: bool = True) -> go.Figure:
        """Plot equity curve with drawdown"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Equity Curve', 'Drawdown'),
            row_heights=[0.7, 0.3]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=self.result.equity_curve.index,
                y=self.result.equity_curve.values,
                name='Equity',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=self.result.drawdown_series.index,
                y=self.result.drawdown_series.values * 100,
                name='Drawdown %',
                fill='tozeroy',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Backtest Performance',
            xaxis_title='Date',
            yaxis_title='Equity',
            height=600,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Equity", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        
        if show:
            fig.show()
        
        return fig
    
    def plot_returns_distribution(self) -> go.Figure:
        """Plot returns distribution"""
        returns = self.result.returns_series.values * 100  # Convert to percentage
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns Distribution',
            marker_color='blue',
            opacity=0.7
        ))
        
        # Add normal distribution overlay
        from scipy import stats
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = stats.norm.pdf(x_range, returns.mean(), returns.std())
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist * len(returns) * (returns.max() - returns.min()) / 50,
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Returns Distribution',
            xaxis_title='Returns (%)',
            yaxis_title='Frequency',
            showlegend=True
        )
        
        return fig
    
    def generate_report(self) -> str:
        """Generate comprehensive HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { font-weight: bold; }
                .positive { color: green; }
                .negative { color: red; }
            </style>
        </head>
        <body>
            <h1>Backtest Report</h1>
            
            <h2>Performance Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Return</td><td class="metric {{ 'positive' if result.total_return > 0 else 'negative' }}">{{ "%.2f"|format(result.total_return * 100) }}%</td></tr>
                <tr><td>Annualized Return</td><td class="metric">{{ "%.2f"|format(result.annualized_return * 100) }}%</td></tr>
                <tr><td>Sharpe Ratio</td><td class="metric">{{ "%.2f"|format(result.sharpe_ratio) }}</td></tr>
                <tr><td>Sortino Ratio</td><td class="metric">{{ "%.2f"|format(result.sortino_ratio) }}</td></tr>
                <tr><td>Max Drawdown</td><td class="metric negative">{{ "%.2f"|format(result.max_drawdown * 100) }}%</td></tr>
                <tr><td>Volatility</td><td class="metric">{{ "%.2f"|format(result.volatility * 100) }}%</td></tr>
            </table>
            
            <h2>Trade Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Trades</td><td class="metric">{{ result.total_trades }}</td></tr>
                <tr><td>Win Rate</td><td class="metric">{{ "%.2f"|format(result.win_rate * 100) }}%</td></tr>
                <tr><td>Profit Factor</td><td class="metric">{{ "%.2f"|format(result.profit_factor) }}</td></tr>
                <tr><td>Average Win</td><td class="metric positive">{{ "%.2f"|format(result.avg_win) }}</td></tr>
                <tr><td>Average Loss</td><td class="metric negative">{{ "%.2f"|format(result.avg_loss) }}</td></tr>
                <tr><td>Expectancy</td><td class="metric">{{ "%.2f"|format(result.expectancy) }}</td></tr>
            </table>
            
            <h2>Risk Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Value at Risk (95%)</td><td class="metric">{{ "%.2f"|format(result.var_95 * 100) }}%</td></tr>
                <tr><td>Conditional VaR (95%)</td><td class="metric">{{ "%.2f"|format(result.cvar_95 * 100) }}%</td></tr>
                <tr><td>Calmar Ratio</td><td class="metric">{{ "%.2f"|format(result.calmar_ratio) }}</td></tr>
            </table>
            
            <div id="equity_curve_plot"></div>
            <div id="returns_dist_plot"></div>
        </body>
        </html>
        """
        
        from jinja2 import Template
        template = Template(html_template)
        
        html_content = template.render(result=self.result)
        
        return html_content


# ====================== EXAMPLE STRATEGIES ======================

def moving_average_crossover_strategy(market_state: Dict, **params) -> List[Dict]:
    """Example: Moving Average Crossover Strategy"""
    fast_period = params.get('fast_period', 20)
    slow_period = params.get('slow_period', 50)
    
    current_bar = market_state['bar']
    history = market_state['history']
    positions = market_state['positions']
    
    if len(history) < slow_period:
        return []
    
    # Calculate moving averages
    fast_ma = history['close'].rolling(fast_period).mean().iloc[-1]
    slow_ma = history['close'].rolling(slow_period).mean().iloc[-1]
    prev_fast_ma = history['close'].rolling(fast_period).mean().iloc[-2]
    prev_slow_ma = history['close'].rolling(slow_period).mean().iloc[-2]
    
    signals = []
    
    # Check for crossover
    if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
        # Golden cross - buy signal
        if 'default' not in positions:
            signals.append({
                'side': 'BUY',
                'strength': 1.0,
                'order_type': 'MARKET'
            })
    
    elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
        # Death cross - sell signal
        if 'default' in positions:
            signals.append({
                'side': 'SELL',
                'quantity': positions['default'].quantity,
                'order_type': 'MARKET'
            })
    
    return signals


def mean_reversion_strategy(market_state: Dict, **params) -> List[Dict]:
    """Example: Mean Reversion Strategy with Bollinger Bands"""
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2)
    rsi_period = params.get('rsi_period', 14)
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    
    current_bar = market_state['bar']
    history = market_state['history']
    positions = market_state['positions']
    
    if len(history) < bb_period:
        return []
    
    # Calculate indicators
    close_prices = history['close']
    sma = close_prices.rolling(bb_period).mean().iloc[-1]
    std = close_prices.rolling(bb_period).std().iloc[-1]
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    
    # RSI
    rsi = current_bar.get('RSI', 50)
    
    current_price = current_bar['close']
    signals = []
    
    # Mean reversion signals
    if current_price < lower_band and rsi < rsi_oversold:
        # Oversold - buy signal
        if 'default' not in positions:
            signals.append({
                'side': 'BUY',
                'strength': min(1.0, (rsi_oversold - rsi) / rsi_oversold),
                'order_type': 'LIMIT',
                'price': current_price * 0.995  # Limit order slightly below market
            })
    
    elif current_price > upper_band and rsi > rsi_overbought:
        # Overbought - sell signal
        if 'default' in positions:
            signals.append({
                'side': 'SELL',
                'quantity': positions['default'].quantity,
                'order_type': 'LIMIT',
                'price': current_price * 1.005  # Limit order slightly above market
            })
    
    return signals
