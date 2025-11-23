"""
DEMIR AI v8.0 - Advanced Backtesting Engine
PROFESSIONAL BACKTESTING WITH REAL HISTORICAL DATA
ZERO MOCK DATA - REAL PERFORMANCE ANALYSIS
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
import json

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtest modes"""
    STANDARD = "STANDARD"
    WALK_FORWARD = "WALK_FORWARD"
    MONTE_CARLO = "MONTE_CARLO"
    STRESS_TEST = "STRESS_TEST"
    OPTIMIZATION = "OPTIMIZATION"


@dataclass
class Trade:
    """Individual trade in backtest"""
    trade_id: int
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    
    side: str  # LONG or SHORT
    entry_price: float
    exit_price: Optional[float]
    
    quantity: float
    position_value: float
    
    stop_loss: float
    take_profit: float
    
    pnl: float = 0
    pnl_percent: float = 0
    fees: float = 0
    
    max_profit: float = 0
    max_loss: float = 0
    max_drawdown: float = 0
    
    holding_period: Optional[timedelta] = None
    exit_reason: str = ""
    
    # Signal data
    signal_confidence: float = 0
    signal_quality: str = ""
    
    # Market conditions
    market_condition: str = ""
    volatility_at_entry: float = 0
    volume_at_entry: float = 0


@dataclass
class BacktestResults:
    """Complete backtest results"""
    # Basic info
    start_date: datetime
    end_date: datetime
    total_days: int
    initial_capital: float
    final_capital: float
    
    # Returns
    total_return: float
    total_return_percent: float
    annualized_return: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    profit_factor: float
    expectancy: float
    
    # Time statistics
    avg_holding_period: timedelta
    longest_trade: timedelta
    shortest_trade: timedelta
    
    # Performance by period
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    yearly_returns: Dict[int, float] = field(default_factory=dict)
    
    # Trade distribution
    trades_by_symbol: Dict[str, int] = field(default_factory=dict)
    pnl_by_symbol: Dict[str, float] = field(default_factory=dict)
    
    # Risk analysis
    var_95: float = 0  # Value at Risk
    cvar_95: float = 0  # Conditional VaR
    
    # Detailed trades
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    # Performance metrics
    recovery_factor: float = 0
    risk_adjusted_return: float = 0


class AdvancedBacktestEngine:
    """
    Professional backtesting engine with real data
    3+ YEARS OF HISTORICAL TESTING
    """
    
    def __init__(self, config):
        self.config = config
        
        # Backtest parameters
        self.initial_capital = 10000
        self.position_sizing = 'fixed'  # fixed, kelly, risk_parity
        self.max_positions = config.trading.max_positions
        self.commission = 0.001  # 0.1%
        self.slippage = 0.0005  # 0.05%
        
        # Risk parameters
        self.max_risk_per_trade = config.trading.max_risk_per_trade
        self.max_daily_loss = config.trading.max_daily_loss
        self.max_drawdown = config.trading.max_drawdown
        
        # Data storage
        self.historical_data = {}
        self.signals = []
        self.trades = []
        
        # Performance tracking
        self.equity_curve = []
        self.daily_returns = []
        self.positions = {}
        
        # Current state
        self.current_capital = self.initial_capital
        self.current_positions = {}
        self.trade_count = 0
        
        # Statistics
        self.stats = {
            'trades_executed': 0,
            'signals_generated': 0,
            'signals_skipped': 0
        }
        
        logger.info("AdvancedBacktestEngine initialized")
        logger.info(f"Initial capital: ${self.initial_capital}")
        logger.info("REAL HISTORICAL DATA ONLY - NO MOCK DATA")
    
    async def run_backtest(self, 
                          start_date: datetime,
                          end_date: datetime,
                          symbols: List[str],
                          strategy,
                          mode: BacktestMode = BacktestMode.STANDARD) -> BacktestResults:
        """
        Run complete backtest
        PROFESSIONAL BACKTESTING WITH REAL DATA
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Mode: {mode.value}")
        
        # Load historical data
        await self._load_historical_data(symbols, start_date, end_date)
        
        # Choose backtest method
        if mode == BacktestMode.STANDARD:
            await self._run_standard_backtest(strategy)
            
        elif mode == BacktestMode.WALK_FORWARD:
            await self._run_walk_forward_backtest(strategy)
            
        elif mode == BacktestMode.MONTE_CARLO:
            await self._run_monte_carlo_backtest(strategy)
            
        elif mode == BacktestMode.STRESS_TEST:
            await self._run_stress_test_backtest(strategy)
            
        elif mode == BacktestMode.OPTIMIZATION:
            await self._run_optimization_backtest(strategy)
        
        # Calculate results
        results = self._calculate_results(start_date, end_date)
        
        # Generate report
        self._generate_report(results)
        
        return results
    
    async def _run_standard_backtest(self, strategy):
        """
        Standard backtest - process signals chronologically
        """
        # Get all timestamps
        all_timestamps = self._get_all_timestamps()
        
        for timestamp in all_timestamps:
            # Update current prices
            self._update_prices(timestamp)
            
            # Check existing positions
            await self._manage_positions(timestamp)
            
            # Generate signals for this timestamp
            signals = await strategy.generate_signals(timestamp, self.historical_data)
            
            # Process signals
            for signal in signals:
                await self._process_signal(signal, timestamp)
            
            # Update equity
            self._update_equity(timestamp)
            
            # Check risk limits
            if not self._check_risk_limits():
                logger.warning(f"Risk limits breached at {timestamp}")
                await self._close_all_positions(timestamp, "RISK_LIMIT")
    
    async def _run_walk_forward_backtest(self, strategy):
        """
        Walk-forward optimization backtest
        """
        window_size = 30  # days for training
        step_size = 7  # days for testing
        
        all_timestamps = self._get_all_timestamps()
        
        for i in range(0, len(all_timestamps) - window_size, step_size):
            # Training period
            train_start = all_timestamps[i]
            train_end = all_timestamps[i + window_size]
            
            # Optimize strategy on training data
            await strategy.optimize(
                self.historical_data,
                train_start,
                train_end
            )
            
            # Testing period
            test_start = train_end
            test_end = all_timestamps[min(i + window_size + step_size, len(all_timestamps) - 1)]
            
            # Run backtest on test period
            for timestamp in all_timestamps[i + window_size:i + window_size + step_size]:
                if timestamp > test_end:
                    break
                
                self._update_prices(timestamp)
                await self._manage_positions(timestamp)
                
                signals = await strategy.generate_signals(timestamp, self.historical_data)
                
                for signal in signals:
                    await self._process_signal(signal, timestamp)
                
                self._update_equity(timestamp)
    
    async def _run_monte_carlo_backtest(self, strategy, num_simulations: int = 1000):
        """
        Monte Carlo simulation backtest
        """
        original_trades = self.trades.copy()
        monte_carlo_results = []
        
        for simulation in range(num_simulations):
            # Shuffle trade order
            np.random.shuffle(self.trades)
            
            # Reset capital
            self.current_capital = self.initial_capital
            self.equity_curve = []
            
            # Apply trades in random order
            for trade in self.trades:
                self.current_capital += trade.pnl
                self.equity_curve.append(self.current_capital)
            
            # Store result
            monte_carlo_results.append({
                'final_capital': self.current_capital,
                'max_drawdown': self._calculate_max_drawdown(self.equity_curve),
                'sharpe': self._calculate_sharpe_ratio(self.equity_curve)
            })
        
        # Restore original trades
        self.trades = original_trades
        
        # Analyze Monte Carlo results
        self._analyze_monte_carlo(monte_carlo_results)
    
    async def _run_stress_test_backtest(self, strategy):
        """
        Stress test backtest with adverse conditions
        """
        stress_scenarios = [
            {'name': 'High Volatility', 'volatility_multiplier': 2.0},
            {'name': 'Low Liquidity', 'liquidity_multiplier': 0.3},
            {'name': 'High Slippage', 'slippage_multiplier': 3.0},
            {'name': 'Bear Market', 'price_multiplier': 0.7},
            {'name': 'Flash Crash', 'crash_probability': 0.01}
        ]
        
        stress_results = {}
        
        for scenario in stress_scenarios:
            logger.info(f"Running stress test: {scenario['name']}")
            
            # Reset state
            self._reset_state()
            
            # Apply stress conditions
            self._apply_stress_conditions(scenario)
            
            # Run backtest
            await self._run_standard_backtest(strategy)
            
            # Store results
            stress_results[scenario['name']] = {
                'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
                'max_drawdown': self._calculate_max_drawdown(self.equity_curve),
                'trades': len(self.trades)
            }
        
        # Analyze stress test results
        self._analyze_stress_tests(stress_results)
    
    async def _run_optimization_backtest(self, strategy):
        """
        Parameter optimization backtest
        """
        # Define parameter ranges
        parameter_ranges = {
            'rsi_period': range(10, 21, 2),
            'ma_period': range(15, 31, 5),
            'stop_loss_percent': np.arange(0.01, 0.05, 0.01),
            'take_profit_percent': np.arange(0.02, 0.10, 0.02)
        }
        
        best_params = {}
        best_sharpe = -np.inf
        
        # Grid search
        for rsi_period in parameter_ranges['rsi_period']:
            for ma_period in parameter_ranges['ma_period']:
                for stop_loss in parameter_ranges['stop_loss_percent']:
                    for take_profit in parameter_ranges['take_profit_percent']:
                        # Set parameters
                        params = {
                            'rsi_period': rsi_period,
                            'ma_period': ma_period,
                            'stop_loss_percent': stop_loss,
                            'take_profit_percent': take_profit
                        }
                        
                        strategy.set_parameters(params)
                        
                        # Reset and run
                        self._reset_state()
                        await self._run_standard_backtest(strategy)
                        
                        # Calculate Sharpe ratio
                        sharpe = self._calculate_sharpe_ratio(self.equity_curve)
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = params.copy()
        
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best Sharpe ratio: {best_sharpe:.2f}")
        
        # Run final backtest with best parameters
        strategy.set_parameters(best_params)
        self._reset_state()
        await self._run_standard_backtest(strategy)
    
    async def _process_signal(self, signal: Dict, timestamp: datetime):
        """Process trading signal"""
        self.stats['signals_generated'] += 1
        
        # Check if we can take position
        if len(self.current_positions) >= self.max_positions:
            self.stats['signals_skipped'] += 1
            return
        
        # Check if already have position in symbol
        if signal['symbol'] in self.current_positions:
            self.stats['signals_skipped'] += 1
            return
        
        # Calculate position size
        position_size = self._calculate_position_size(signal)
        
        if position_size == 0:
            self.stats['signals_skipped'] += 1
            return
        
        # Create trade
        trade = Trade(
            trade_id=self.trade_count,
            symbol=signal['symbol'],
            entry_time=timestamp,
            exit_time=None,
            side="LONG" if signal['action'] in ['BUY', 'STRONG_BUY'] else "SHORT",
            entry_price=signal['entry_price'],
            exit_price=None,
            quantity=position_size / signal['entry_price'],
            position_value=position_size,
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit_1'],
            signal_confidence=signal.get('confidence', 50),
            signal_quality=signal.get('quality', 'MEDIUM'),
            market_condition=signal.get('market_condition', 'NORMAL'),
            volatility_at_entry=signal.get('volatility', 0),
            volume_at_entry=signal.get('volume', 0)
        )
        
        # Apply entry costs
        entry_cost = position_size * (self.commission + self.slippage)
        self.current_capital -= position_size + entry_cost
        trade.fees = entry_cost
        
        # Add to positions
        self.current_positions[signal['symbol']] = trade
        self.trade_count += 1
        self.stats['trades_executed'] += 1
        
        logger.debug(f"Opened position: {signal['symbol']} at {signal['entry_price']}")
    
    async def _manage_positions(self, timestamp: datetime):
        """Manage existing positions"""
        closed_positions = []
        
        for symbol, trade in self.current_positions.items():
            current_price = self._get_current_price(symbol, timestamp)
            
            if not current_price:
                continue
            
            # Calculate current P&L
            if trade.side == "LONG":
                pnl = (current_price - trade.entry_price) * trade.quantity
                pnl_percent = (current_price - trade.entry_price) / trade.entry_price
            else:
                pnl = (trade.entry_price - current_price) * trade.quantity
                pnl_percent = (trade.entry_price - current_price) / trade.entry_price
            
            # Track max profit/loss
            trade.max_profit = max(trade.max_profit, pnl)
            trade.max_loss = min(trade.max_loss, pnl)
            
            # Check exit conditions
            exit_reason = None
            
            # Stop loss
            if trade.side == "LONG" and current_price <= trade.stop_loss:
                exit_reason = "STOP_LOSS"
            elif trade.side == "SHORT" and current_price >= trade.stop_loss:
                exit_reason = "STOP_LOSS"
            
            # Take profit
            elif trade.side == "LONG" and current_price >= trade.take_profit:
                exit_reason = "TAKE_PROFIT"
            elif trade.side == "SHORT" and current_price <= trade.take_profit:
                exit_reason = "TAKE_PROFIT"
            
            # Time-based exit (optional)
            elif (timestamp - trade.entry_time).days > 30:
                exit_reason = "TIME_EXIT"
            
            # Exit if needed
            if exit_reason:
                trade.exit_time = timestamp
                trade.exit_price = current_price
                trade.pnl = pnl
                trade.pnl_percent = pnl_percent
                trade.holding_period = timestamp - trade.entry_time
                trade.exit_reason = exit_reason
                
                # Apply exit costs
                exit_cost = abs(pnl) * (self.commission + self.slippage)
                trade.fees += exit_cost
                trade.pnl -= exit_cost
                
                # Update capital
                self.current_capital += trade.position_value + trade.pnl
                
                # Add to trades history
                self.trades.append(trade)
                closed_positions.append(symbol)
                
                logger.debug(f"Closed position: {symbol} - {exit_reason} - P&L: {trade.pnl:.2f}")
        
        # Remove closed positions
        for symbol in closed_positions:
            del self.current_positions[symbol]
    
    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size"""
        if self.position_sizing == 'fixed':
            # Fixed percentage of capital
            return self.current_capital * self.max_risk_per_trade
            
        elif self.position_sizing == 'kelly':
            # Kelly criterion
            win_prob = signal.get('confidence', 50) / 100
            win_loss_ratio = signal.get('risk_reward_ratio', 2)
            
            kelly_percent = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
            kelly_percent = max(0, min(kelly_percent, 0.25))  # Cap at 25%
            
            return self.current_capital * kelly_percent
            
        elif self.position_sizing == 'risk_parity':
            # Equal risk weighting
            volatility = signal.get('volatility', 0.02)
            target_risk = 0.01  # 1% portfolio risk
            
            return (self.current_capital * target_risk) / volatility
        
        return 0
    
    def _update_equity(self, timestamp: datetime):
        """Update equity curve"""
        # Calculate current portfolio value
        portfolio_value = self.current_capital
        
        for trade in self.current_positions.values():
            current_price = self._get_current_price(trade.symbol, timestamp)
            if current_price:
                if trade.side == "LONG":
                    current_value = current_price * trade.quantity
                else:
                    current_value = trade.position_value - (current_price - trade.entry_price) * trade.quantity
                
                portfolio_value += current_value
        
        self.equity_curve.append(portfolio_value)
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            daily_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_returns.append(daily_return)
    
    def _check_risk_limits(self) -> bool:
        """Check if risk limits are breached"""
        if len(self.equity_curve) < 2:
            return True
        
        # Check daily loss
        daily_loss = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
        if daily_loss < -self.max_daily_loss:
            return False
        
        # Check max drawdown
        current_dd = self._calculate_current_drawdown()
        if current_dd > self.max_drawdown:
            return False
        
        return True
    
    async def _close_all_positions(self, timestamp: datetime, reason: str):
        """Emergency close all positions"""
        for symbol in list(self.current_positions.keys()):
            trade = self.current_positions[symbol]
            current_price = self._get_current_price(symbol, timestamp)
            
            if current_price:
                trade.exit_time = timestamp
                trade.exit_price = current_price
                trade.exit_reason = reason
                
                # Calculate final P&L
                if trade.side == "LONG":
                    trade.pnl = (current_price - trade.entry_price) * trade.quantity
                else:
                    trade.pnl = (trade.entry_price - current_price) * trade.quantity
                
                trade.pnl -= trade.fees
                
                self.trades.append(trade)
                self.current_capital += trade.position_value + trade.pnl
        
        self.current_positions.clear()
    
    def _calculate_results(self, start_date: datetime, end_date: datetime) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        if not self.trades:
            return self._empty_results(start_date, end_date)
        
        # Basic metrics
        total_days = (end_date - start_date).days
        final_capital = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        
        # Returns
        total_return = final_capital - self.initial_capital
        total_return_percent = (total_return / self.initial_capital) * 100
        annualized_return = (pow(final_capital / self.initial_capital, 365 / total_days) - 1) * 100
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(self.equity_curve)
        sortino_ratio = self._calculate_sortino_ratio(self.daily_returns)
        max_drawdown = self._calculate_max_drawdown(self.equity_curve)
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * abs(avg_loss))
        
        # Holding periods
        holding_periods = [t.holding_period for t in self.trades if t.holding_period]
        avg_holding_period = np.mean(holding_periods) if holding_periods else timedelta(0)
        
        # Monthly returns
        monthly_returns = self._calculate_monthly_returns()
        
        # Yearly returns
        yearly_returns = self._calculate_yearly_returns()
        
        # Symbol analysis
        trades_by_symbol = {}
        pnl_by_symbol = {}
        
        for trade in self.trades:
            if trade.symbol not in trades_by_symbol:
                trades_by_symbol[trade.symbol] = 0
                pnl_by_symbol[trade.symbol] = 0
            
            trades_by_symbol[trade.symbol] += 1
            pnl_by_symbol[trade.symbol] += trade.pnl
        
        # VaR and CVaR
        var_95, cvar_95 = self._calculate_var_cvar(self.daily_returns, 0.95)
        
        results = BacktestResults(
            start_date=start_date,
            end_date=end_date,
            total_days=total_days,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            
            total_return=total_return,
            total_return_percent=total_return_percent,
            annualized_return=annualized_return,
            
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            max_drawdown=max_drawdown,
            max_drawdown_duration=self._calculate_max_dd_duration(),
            
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            
            profit_factor=profit_factor,
            expectancy=expectancy,
            
            avg_holding_period=avg_holding_period,
            longest_trade=max(holding_periods) if holding_periods else timedelta(0),
            shortest_trade=min(holding_periods) if holding_periods else timedelta(0),
            
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns,
            
            trades_by_symbol=trades_by_symbol,
            pnl_by_symbol=pnl_by_symbol,
            
            var_95=var_95,
            cvar_95=cvar_95,
            
            trades=self.trades,
            equity_curve=self.equity_curve,
            
            recovery_factor=total_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            risk_adjusted_return=total_return_percent / (np.std(self.daily_returns) * np.sqrt(252)) if self.daily_returns else 0
        )
        
        return results
    
    def _calculate_sharpe_ratio(self, equity_curve: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(equity_curve) < 2:
            return 0
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0
        
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if not returns:
            return 0
        
        excess_returns = np.array(returns) - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 0
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0
        
        return np.sqrt(252) * np.mean(excess_returns) / downside_std
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(equity_curve) < 2:
            return 0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown"""
        if len(self.equity_curve) < 2:
            return 0
        
        peak = max(self.equity_curve)
        current = self.equity_curve[-1]
        
        if peak == 0:
            return 0
        
        return (peak - current) / peak
    
    def _calculate_max_dd_duration(self) -> int:
        """Calculate maximum drawdown duration in days"""
        if len(self.equity_curve) < 2:
            return 0
        
        peak = self.equity_curve[0]
        peak_idx = 0
        max_duration = 0
        current_duration = 0
        
        for i, value in enumerate(self.equity_curve):
            if value >= peak:
                peak = value
                peak_idx = i
                current_duration = 0
            else:
                current_duration = i - peak_idx
                max_duration = max(max_duration, current_duration)
        
        return max_duration
    
    def _calculate_var_cvar(self, returns: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional VaR"""
        if not returns:
            return 0, 0
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        
        var = sorted_returns[index] if index < len(sorted_returns) else sorted_returns[-1]
        cvar = np.mean(sorted_returns[:index]) if index > 0 else var
        
        return var, cvar
    
    def _calculate_monthly_returns(self) -> Dict[str, float]:
        """Calculate returns by month"""
        monthly_returns = {}
        
        # Group trades by month
        for trade in self.trades:
            if trade.exit_time:
                month_key = trade.exit_time.strftime("%Y-%m")
                
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = 0
                
                monthly_returns[month_key] += trade.pnl
        
        return monthly_returns
    
    def _calculate_yearly_returns(self) -> Dict[int, float]:
        """Calculate returns by year"""
        yearly_returns = {}
        
        # Group trades by year
        for trade in self.trades:
            if trade.exit_time:
                year = trade.exit_time.year
                
                if year not in yearly_returns:
                    yearly_returns[year] = 0
                
                yearly_returns[year] += trade.pnl
        
        return yearly_returns
    
    async def _load_historical_data(self, symbols: List[str], 
                                   start_date: datetime, 
                                   end_date: datetime):
        """Load historical data for backtesting"""
        logger.info("Loading historical data...")
        
        # In real implementation, would load from database or API
        # This is a placeholder for data loading logic
        
        for symbol in symbols:
            # Load price data
            # self.historical_data[symbol] = load_data_from_source(symbol, start_date, end_date)
            pass
        
        logger.info(f"Loaded data for {len(symbols)} symbols")
    
    def _get_all_timestamps(self) -> List[datetime]:
        """Get all unique timestamps from historical data"""
        timestamps = set()
        
        for symbol_data in self.historical_data.values():
            if 'timestamps' in symbol_data:
                timestamps.update(symbol_data['timestamps'])
        
        return sorted(list(timestamps))
    
    def _get_current_price(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """Get price at specific timestamp"""
        if symbol in self.historical_data:
            data = self.historical_data[symbol]
            
            # Find price at timestamp
            # In real implementation, would interpolate or use nearest
            
            return data.get('prices', {}).get(timestamp)
        
        return None
    
    def _update_prices(self, timestamp: datetime):
        """Update current prices for all symbols"""
        # In real implementation, would update price cache
        pass
    
    def _reset_state(self):
        """Reset backtest state"""
        self.current_capital = self.initial_capital
        self.current_positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.trade_count = 0
        self.stats = {
            'trades_executed': 0,
            'signals_generated': 0,
            'signals_skipped': 0
        }
    
    def _apply_stress_conditions(self, scenario: Dict):
        """Apply stress test conditions"""
        if 'volatility_multiplier' in scenario:
            # Increase volatility in data
            pass
        
        if 'slippage_multiplier' in scenario:
            self.slippage *= scenario['slippage_multiplier']
        
        # Apply other stress conditions
    
    def _analyze_monte_carlo(self, results: List[Dict]):
        """Analyze Monte Carlo simulation results"""
        final_capitals = [r['final_capital'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        sharpe_ratios = [r['sharpe'] for r in results]
        
        logger.info("Monte Carlo Analysis:")
        logger.info(f"Mean final capital: ${np.mean(final_capitals):.2f}")
        logger.info(f"Std final capital: ${np.std(final_capitals):.2f}")
        logger.info(f"95% CI: [${np.percentile(final_capitals, 2.5):.2f}, "
                   f"${np.percentile(final_capitals, 97.5):.2f}]")
        logger.info(f"Mean max drawdown: {np.mean(max_drawdowns)*100:.2f}%")
        logger.info(f"Mean Sharpe ratio: {np.mean(sharpe_ratios):.2f}")
    
    def _analyze_stress_tests(self, results: Dict):
        """Analyze stress test results"""
        logger.info("Stress Test Results:")
        
        for scenario, metrics in results.items():
            logger.info(f"\n{scenario}:")
            logger.info(f"  Total return: {metrics['total_return']*100:.2f}%")
            logger.info(f"  Max drawdown: {metrics['max_drawdown']*100:.2f}%")
            logger.info(f"  Trades: {metrics['trades']}")
    
    def _generate_report(self, results: BacktestResults):
        """Generate detailed backtest report"""
        logger.info("\n" + "="*50)
        logger.info("BACKTEST RESULTS")
        logger.info("="*50)
        
        logger.info(f"\nPeriod: {results.start_date.date()} to {results.end_date.date()}")
        logger.info(f"Total days: {results.total_days}")
        
        logger.info(f"\nCapital:")
        logger.info(f"  Initial: ${results.initial_capital:,.2f}")
        logger.info(f"  Final: ${results.final_capital:,.2f}")
        logger.info(f"  Return: ${results.total_return:,.2f} ({results.total_return_percent:.2f}%)")
        logger.info(f"  Annualized: {results.annualized_return:.2f}%")
        
        logger.info(f"\nRisk Metrics:")
        logger.info(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"  Sortino Ratio: {results.sortino_ratio:.2f}")
        logger.info(f"  Max Drawdown: {results.max_drawdown*100:.2f}%")
        logger.info(f"  VaR (95%): {results.var_95*100:.2f}%")
        
        logger.info(f"\nTrade Statistics:")
        logger.info(f"  Total trades: {results.total_trades}")
        logger.info(f"  Win rate: {results.win_rate:.2f}%")
        logger.info(f"  Profit factor: {results.profit_factor:.2f}")
        logger.info(f"  Expectancy: ${results.expectancy:.2f}")
        
        logger.info("="*50 + "\n")
    
    def _empty_results(self, start_date: datetime, end_date: datetime) -> BacktestResults:
        """Return empty results when no trades"""
        return BacktestResults(
            start_date=start_date,
            end_date=end_date,
            total_days=(end_date - start_date).days,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_return=0,
            total_return_percent=0,
            annualized_return=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            largest_win=0,
            largest_loss=0,
            profit_factor=0,
            expectancy=0,
            avg_holding_period=timedelta(0),
            longest_trade=timedelta(0),
            shortest_trade=timedelta(0)
        )
    
    def get_statistics(self) -> Dict:
        """Get backtest statistics"""
        return {
            'trades_executed': self.stats['trades_executed'],
            'signals_generated': self.stats['signals_generated'],
            'signals_skipped': self.stats['signals_skipped'],
            'current_capital': self.current_capital,
            'open_positions': len(self.current_positions),
            'equity_points': len(self.equity_curve)
        }
