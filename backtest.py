"""
Refactored backtest simulation engine using strategy pattern.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Optional, Dict, List, Union, Any
from alpha_models.base import AlphaModel
from signal_filter.hmm_filter import HMMRegimeFilter
from statistics import Statistics
from alpha_model_factory import AlphaModelFactory
from strategies import (
    AlphaOnlyStrategy,
    HMMOnlyStrategy,
    OracleStrategy,
    AlphaHMMFilterStrategy,
    AlphaHMMCombineStrategy,
    RegimeAdaptiveAlphaStrategy,
    AlphaOracleStrategy
)


class BacktestEngine:
    """
    Backtest engine for testing trading strategies with optional HMM filtering.
    """
    
    # Strategy registry
    STRATEGIES = {
        'alpha_only': AlphaOnlyStrategy,
        'hmm_only': HMMOnlyStrategy,
        'oracle': OracleStrategy,
        'alpha_hmm_filter': AlphaHMMFilterStrategy,
        'alpha_hmm_combine': AlphaHMMCombineStrategy,
        'regime_adaptive_alpha': RegimeAdaptiveAlphaStrategy,
        'alpha_oracle': AlphaOracleStrategy
    }
    
    def __init__(self, close: pd.Series, 
                 alpha_model: Optional[AlphaModel] = None,
                 alpha_config: Optional[Dict[str, Any]] = None,
                 hmm_filter: Optional[HMMRegimeFilter] = None,
                 bear_alpha_model: Optional[AlphaModel] = None,
                 initial_capital: float = 100000.0,
                 test_start_date: Optional[str] = None):
        """
        Initialize backtest engine.
        
        Parameters:
        -----------
        close : pd.Series
            Close price series
        alpha_model : AlphaModel, optional
            Alpha model instance for signal generation (used in bull/neutral regimes).
            Provide either this OR alpha_config, not both.
        alpha_config : Dict[str, Any], optional
            Alpha model configuration dictionary with 'type' and 'parameters'.
            Provide either this OR alpha_model, not both.
            Example: {'type': 'SMA', 'parameters': {'short_window': 50, 'long_window': 200}}
        hmm_filter : HMMRegimeFilter, optional
            HMM filter for regime-based filtering
        bear_alpha_model : AlphaModel, optional
            Alternative alpha model for bear markets (e.g., Bollinger Bands for mean-reversion).
            Used with 'regime_adaptive_alpha' strategy mode.
        initial_capital : float
            Initial capital
        test_start_date : str, optional
            User-specified test start date (format: 'YYYY-MM-DD').
            When provided, HMM results will be trimmed to start from this date.
            This is useful when extra historical data is loaded for HMM training.
            
        Raises:
        -------
        ValueError
            If both alpha_model and alpha_config are provided, or if neither is provided
        """
        self.close = close
        self.hmm_filter = hmm_filter
        self.bear_alpha_model = bear_alpha_model
        self.initial_capital = initial_capital
        self.test_start_date = test_start_date
        
        # Handle alpha model initialization
        if alpha_model is not None and alpha_config is not None:
            raise ValueError(
                "Provide either alpha_model OR alpha_config, not both"
            )
        
        if alpha_model is None and alpha_config is None:
            raise ValueError(
                "Must provide either alpha_model or alpha_config"
            )
        
        if alpha_config is not None:
            # Create alpha model from config
            self.alpha_model = AlphaModelFactory.create_from_config(alpha_config)
            self._alpha_config = alpha_config
        else:
            self.alpha_model = alpha_model
            self._alpha_config = None
        
        # Results storage
        self.positions: Optional[pd.Series] = None
        self.returns: Optional[pd.Series] = None
        self.equity_curve: Optional[pd.Series] = None
        self.metrics: Optional[Dict] = None
    
    @staticmethod
    def from_config(close: pd.Series, config: Dict[str, Any],
                   hmm_filter: Optional[HMMRegimeFilter] = None) -> 'BacktestEngine':
        """Create BacktestEngine from full configuration dictionary."""
        alpha_config = config.get('alpha_model')
        if alpha_config is None:
            raise KeyError("Configuration missing 'alpha_model' section")
        
        backtest_config = config.get('backtest', {})
        initial_capital = backtest_config.get('initial_capital', 100000.0)
        
        return BacktestEngine(
            close=close,
            alpha_config=alpha_config,
            hmm_filter=hmm_filter,
            initial_capital=initial_capital
        )
    
    @staticmethod
    def from_alpha_config(close: pd.Series, alpha_config: Dict[str, Any],
                         hmm_filter: Optional[HMMRegimeFilter] = None,
                         initial_capital: float = 100000.0) -> 'BacktestEngine':
        """Create BacktestEngine from alpha model configuration only."""
        return BacktestEngine(
            close=close,
            alpha_config=alpha_config,
            hmm_filter=hmm_filter,
            initial_capital=initial_capital
        )
    
    def get_alpha_config(self) -> Optional[Dict[str, Any]]:
        """Get the alpha model configuration used (if created from config)."""
        return self._alpha_config
        
    def run(self, 
            strategy_mode: str = 'alpha_only',
            rebalance_frequency: int = 1,
            walk_forward: bool = False,
            train_window: int = 504,
            refit_every: int = 21,
            bear_prob_threshold: float = 0.65,
            bull_prob_threshold: float = 0.65,
            transaction_cost: float = 0.0,
            enable_logging: bool = False,
            log_dir: str = 'logs') -> Dict:
        """
        Run backtest simulation using strategy pattern.
        
        Parameters:
        -----------
        strategy_mode : str
            Strategy mode (alpha_only, hmm_only, oracle, alpha_hmm_filter, 
            alpha_hmm_combine, regime_adaptive_alpha)
        ... (other parameters as before)
            
        Returns:
        --------
        Dict
            Backtest results including metrics and equity curve
        """
        # Validate strategy
        if strategy_mode not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_mode}. "
                           f"Available: {list(self.STRATEGIES.keys())}")
        
        # Oracle strategy must use all data (no walk-forward)
        if strategy_mode == 'oracle':
            walk_forward = False
        
        print(f"\n{'='*60}")
        print(f"RUNNING BACKTEST: {strategy_mode.upper()}")
        print(f"{'='*60}")
        print(f"Alpha Model: {self.alpha_model.get_name()}")
        print(f"Parameters: {self.alpha_model.get_parameters()}")
        print(f"Rebalance Frequency: {rebalance_frequency} period(s)")
        print(f"Transaction Cost: {transaction_cost*100:.3f}%")
        
        # Setup logging
        log_file = self._setup_logging(enable_logging, log_dir, strategy_mode)
        
        # Generate alpha signals
        alpha_signals = self.alpha_model.generate_signals(self.close)
        alpha_signals = self._apply_fsm_logic(alpha_signals)
        
        # Get strategy instance
        strategy = self.STRATEGIES[strategy_mode]()
        
        # Execute strategy
        if strategy_mode == 'alpha_only':
            positions = self._run_alpha_only(strategy, alpha_signals)
            log_data = self._log_alpha_only(strategy, positions, alpha_signals, enable_logging)
            state_info = None
        
        elif strategy_mode == 'alpha_oracle':
            # Oracle timing strategy - ignores alpha signals, uses price action only
            positions = self._run_alpha_oracle(strategy, alpha_signals)
            log_data = self._log_alpha_oracle(strategy, positions, alpha_signals, enable_logging)
            state_info = None
            
        elif strategy_mode in ['hmm_only', 'oracle', 'alpha_hmm_filter', 
                               'alpha_hmm_combine', 'regime_adaptive_alpha']:
            if self.hmm_filter is None:
                raise ValueError(f"{strategy_mode} requires HMM filter")
            
            positions, log_data, state_info = self._run_hmm_strategy(
                strategy, strategy_mode, alpha_signals, walk_forward,
                train_window, refit_every, bear_prob_threshold, 
                bull_prob_threshold, enable_logging
            )
        
        # Apply rebalancing
        if rebalance_frequency > 1:
            print(f"\nApplying rebalancing frequency: {rebalance_frequency}")
            positions = self._apply_rebalancing(positions, rebalance_frequency)
        
        # Calculate returns and metrics
        results = self._calculate_results(positions, transaction_cost, strategy_mode)
        
        # Add state info if available
        if state_info:
            results.update(state_info)
        
        # Write log file
        if enable_logging and log_data and log_file:
            self._write_log_file(log_data, log_file, results['equity_curve'])
        
        self._print_summary(results)
        
        return results
    
    def _setup_logging(self, enable_logging: bool, log_dir: str, strategy_mode: str):
        """Setup logging configuration."""
        if not enable_logging:
            return None
            
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"{strategy_mode}_{self.alpha_model.get_name()}_{timestamp}.csv"
        log_file = os.path.join(log_dir, log_filename)
        print(f"Logging enabled: {log_file}")
        return log_file
    
    def _run_alpha_only(self, strategy, alpha_signals: pd.Series) -> pd.Series:
        """Run alpha-only strategy."""
        positions = strategy.generate_positions(
            alpha_signals, self.close, alpha_signals.index
        )
        return positions
    
    def _log_alpha_only(self, strategy, positions: pd.Series, 
                       alpha_signals: pd.Series, enable_logging: bool):
        """Generate log data for alpha-only strategy."""
        if not enable_logging:
            return None
        
        return strategy.generate_log_data(
            positions, self.close, alpha_signals, positions.index
        )
    
    def _run_alpha_oracle(self, strategy, alpha_signals: pd.Series) -> pd.Series:
        """Run alpha oracle timing strategy."""
        positions = strategy.generate_positions(
            alpha_signals, self.close, alpha_signals.index
        )
        
        # Print pivot information
        pivot_info = strategy.get_pivot_info(self.close, alpha_signals.index)
        print(f"\nZigZag Pivot Analysis:")
        print(f"  Min Move Threshold: {pivot_info['min_move_threshold_pct']:.2f}%")
        print(f"  Total Pivots Identified: {pivot_info['num_pivots']}")
        print(f"  BUY Signals: {pivot_info['num_buys']}")
        print(f"  SELL Signals: {pivot_info['num_sells']}")
        print(f"  Average Swing Size: {pivot_info['avg_swing_pct']:.2f}%")
        
        return positions
    
    def _log_alpha_oracle(self, strategy, positions: pd.Series, 
                         alpha_signals: pd.Series, enable_logging: bool):
        """Generate log data for alpha oracle strategy."""
        if not enable_logging:
            return None
        
        return strategy.generate_log_data(
            positions, self.close, alpha_signals, positions.index
        )
    
    def _run_hmm_strategy(self, strategy, strategy_mode: str, alpha_signals: pd.Series,
                         walk_forward: bool, train_window: int, refit_every: int,
                         bear_prob_threshold: float, bull_prob_threshold: float,
                         enable_logging: bool):
        """Run HMM-based strategy."""
        print("\nApplying HMM regime filtering...")
        
        # Fit HMM
        probs, regime, switches, regime_info = self._fit_hmm(
            strategy_mode, walk_forward, train_window, refit_every
        )
        
        # Print regime info
        self._print_regime_info(regime_info, switches)
        
        # Store regime info for results
        self.regime_probs = probs
        self.regime = regime
        self.regime_info = regime_info
        
        # Align indices
        common_idx = alpha_signals.index.intersection(probs.index)
        alpha_signals_aligned = alpha_signals.loc[common_idx]
        
        # Get regime probabilities
        bear_regime = regime_info['bear_regime']
        bull_regime = regime_info['bull_regime']
        neutral_regime = regime_info['neutral_regime']
        
        bear_prob = probs[bear_regime].loc[common_idx]
        bull_prob = probs[bull_regime].loc[common_idx]
        neutral_prob = probs[neutral_regime].loc[common_idx] if neutral_regime is not None else pd.Series(0, index=common_idx)
        bull_prob_combined = bull_prob + neutral_prob if neutral_regime is not None else bull_prob.copy()
        
        # Prepare kwargs for strategy
        kwargs = {
            'bear_prob': bear_prob,
            'bull_prob': bull_prob,
            'neutral_prob': neutral_prob,
            'bull_prob_combined': bull_prob_combined,
            'bear_prob_threshold': bear_prob_threshold,
            'bull_prob_threshold': bull_prob_threshold
        }
        
        # Run specific strategy
        state_info = {}
        
        if strategy_mode == 'alpha_hmm_combine':
            positions, state_data = strategy.generate_positions(
                alpha_signals_aligned, self.close, common_idx, **kwargs
            )

            # Store momentum-based state info
            state_info = {
                'bull_prob_momentum': state_data['bull_prob_momentum'],
                'bear_prob_momentum': state_data['bear_prob_momentum'],
                'buy_signal': state_data['buy_signal'],
                'exit_signal': state_data['exit_signal']
            }
            kwargs['state_info'] = state_data
            
        elif strategy_mode == 'regime_adaptive_alpha':
            if self.bear_alpha_model is None:
                raise ValueError("regime_adaptive_alpha requires bear_alpha_model")
            
            bear_signals = self.bear_alpha_model.generate_signals(self.close)
            bear_signals_aligned = bear_signals.reindex(common_idx, fill_value=0)
            kwargs['regime'] = regime
            kwargs['bear_signals'] = bear_signals_aligned
            
            positions = strategy.generate_positions(
                alpha_signals_aligned, self.close, common_idx, **kwargs
            )
        else:
            positions = strategy.generate_positions(
                alpha_signals_aligned, self.close, common_idx, **kwargs
            )
        
        # Generate log data
        log_data = None
        if enable_logging:
            log_data = strategy.generate_log_data(
                positions, self.close, alpha_signals_aligned, common_idx, **kwargs
            )
        
        return positions, log_data, state_info
    
    def _fit_hmm(self, strategy_mode: str, walk_forward: bool, 
                 train_window: int, refit_every: int):
        """Fit HMM model based on strategy mode."""
        if strategy_mode == 'oracle' or not walk_forward:
            # Fit on ALL data (oracle mode or simple fit)
            if strategy_mode == 'oracle':
                print("  ⚠️  ORACLE MODE: Fitting HMM on entire dataset (future knowledge)")
            
            features = self.hmm_filter.make_features(self.close)
            self.hmm_filter.fit(features)
            probs_array = self.hmm_filter.filtered_state_probs(features)
            probs = pd.DataFrame(
                probs_array, 
                index=features.index, 
                columns=list(range(self.hmm_filter.n_states))
            )
            regime = self.hmm_filter.detect_regime_switches(probs)
            switches = regime[regime.ne(regime.shift(1))].dropna()
        else:
            # Walk-forward testing
            probs, regime, switches = self.hmm_filter.walkforward_filter(
                self.close, 
                train_window=train_window,
                refit_every=refit_every
            )
        
        # Trim HMM results to user-specified test start date if provided
        # This is needed when extra historical data is loaded for HMM training
        if self.test_start_date is not None:
            test_start = pd.to_datetime(self.test_start_date)
            print(f"  Trimming HMM results to start from {self.test_start_date}")
            
            # Filter all HMM outputs to start from test_start_date
            probs = probs[probs.index >= test_start]
            regime = regime[regime.index >= test_start]
            switches = switches[switches.index >= test_start]
            
            print(f"  HMM predictions now span: {probs.index[0]} to {probs.index[-1]}")
        
        regime_info = self.hmm_filter.identify_regimes(self.close, regime)
        return probs, regime, switches, regime_info
    
    def _print_regime_info(self, regime_info: Dict, switches: pd.Series):
        """Print regime detection information."""
        bear_regime = regime_info['bear_regime']
        bull_regime = regime_info['bull_regime']
        neutral_regime = regime_info['neutral_regime']
        
        print(f"  Bear regime: {bear_regime} (vol: {regime_info['regime_volatilities'][bear_regime]:.4f})")
        print(f"  Bull regime: {bull_regime} (vol: {regime_info['regime_volatilities'][bull_regime]:.4f})")
        if neutral_regime is not None:
            print(f"  Neutral regime: {neutral_regime} (vol: {regime_info['regime_volatilities'][neutral_regime]:.4f})")
        print(f"  Detected {len(switches)} regime switches")
    
    def _calculate_results(self, positions: pd.Series, transaction_cost: float, 
                          strategy_mode: str) -> Dict:
        """Calculate backtest results."""
        # Calculate returns
        price_returns = self.close.pct_change().fillna(0)
        
        # Align positions and returns
        common_idx = positions.index.intersection(price_returns.index)
        positions = positions.loc[common_idx]
        price_returns = price_returns.loc[common_idx]
        
        # Apply transaction costs
        if transaction_cost > 0:
            position_changes = positions.diff().fillna(0).abs()
            costs = position_changes * transaction_cost
            price_returns = price_returns - costs
        
        # Strategy returns
        strategy_returns = positions.shift(1).fillna(0) * price_returns
        
        # Calculate equity curve
        equity_curve = self.initial_capital * (1 + strategy_returns).cumprod()
        
        # Calculate metrics
        metrics = Statistics.calculate_all_metrics(strategy_returns.values)
        
        # Calculate additional statistics
        num_trades = int(np.sum(np.abs(positions.diff().fillna(0))) / 2)
        time_in_market = float(np.sum(positions != 0) / len(positions))
        
        # Store results
        self.positions = positions
        self.returns = strategy_returns
        self.equity_curve = equity_curve
        self.metrics = metrics
        
        # Extract final capital
        final_capital_value = equity_curve.iloc[-1]
        if isinstance(final_capital_value, pd.Series):
            final_capital_value = final_capital_value.iloc[0]
        
        # Prepare results dictionary
        results = {
            'metrics': metrics,
            'equity_curve': equity_curve,
            'positions': positions,
            'returns': strategy_returns,
            'close_prices': self.close,
            'num_trades': num_trades,
            'time_in_market': time_in_market,
            'strategy_mode': strategy_mode,
            'alpha_model': self.alpha_model.get_name(),
            'initial_capital': self.initial_capital,
            'final_capital': float(final_capital_value),
            'rebalance_frequency': 1  # Updated by caller if needed
        }
        
        # Add HMM-specific results if available
        if hasattr(self, 'regime_probs'):
            results['regime_probs'] = self.regime_probs
            results['regime'] = self.regime
            results['regime_info'] = self.regime_info
        
        return results
    
    def _write_log_file(self, log_data: List[Dict], log_file: str, equity_curve: pd.Series):
        """Write log data to CSV file."""
        # Add portfolio value to log data
        for entry in log_data:
            if entry['Date'] in equity_curve.index:
                entry['Portfolio_Value'] = equity_curve.loc[entry['Date']]
            else:
                entry['Portfolio_Value'] = self.initial_capital
        
        log_df = pd.DataFrame(log_data)
        log_df.to_csv(log_file, index=False)
        print(f"\n✓ Log file saved: {log_file}")
        print(f"  Total entries: {len(log_df)}")
    
    def _print_summary(self, results: Dict):
        """Print backtest summary."""
        print(f"\n{'='*60}")
        print("BACKTEST COMPLETE")
        print(f"{'='*60}")
        print(f"Number of Trades: {results['num_trades']}")
        print(f"Time in Market: {results['time_in_market']*100:.1f}%")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['metrics']['total_return']*100:.2f}%")
    
    def _apply_fsm_logic(self, signals: pd.Series) -> pd.Series:
        """Apply finite state machine logic to validate position transitions."""
        positions = pd.Series(0, index=signals.index)
        current_state = 0  # Start FLAT
        
        for i in range(len(signals)):
            requested_signal = signals.iloc[i]
            
            if current_state == 0:  # FLAT
                if requested_signal == 1:
                    current_state = 1  # BUY: FLAT -> LONG
                elif requested_signal == -1:
                    current_state = -1  # SHORT: FLAT -> SHORT
                    
            elif current_state == 1:  # LONG
                if requested_signal == 0:
                    current_state = 0  # SELL: LONG -> FLAT
                elif requested_signal == -1:
                    current_state = -1  # LONG -> SHORT
                    
            elif current_state == -1:  # SHORT
                if requested_signal == 0:
                    current_state = 0  # COVER: SHORT -> FLAT
                elif requested_signal == 1:
                    current_state = 1  # SHORT -> LONG
            
            positions.iloc[i] = current_state
        
        return positions
    
    def _apply_rebalancing(self, positions: pd.Series, frequency: int) -> pd.Series:
        """Apply rebalancing frequency to positions."""
        rebalanced = positions.copy()
        
        for i in range(1, len(rebalanced)):
            if i % frequency != 0:
                rebalanced.iloc[i] = rebalanced.iloc[i-1]
        
        return rebalanced
    
    def compare_with_benchmark(self) -> Dict:
        """Compare strategy performance with buy-and-hold benchmark."""
        if self.returns is None:
            raise ValueError("Run backtest first before comparison")
        
        price_returns = self.close.pct_change().fillna(0)
        benchmark_returns = price_returns.loc[self.returns.index]
        benchmark_metrics = Statistics.calculate_all_metrics(benchmark_returns)
        
        return {
            'strategy_metrics': self.metrics,
            'benchmark_metrics': benchmark_metrics,
            'total_return_diff': self.metrics['total_return'] - benchmark_metrics['total_return'],
            'sharpe_diff': self.metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio'],
            'max_drawdown_diff': self.metrics['max_drawdown'] - benchmark_metrics['max_drawdown']
        }
    
    def print_results(self, include_benchmark: bool = True) -> None:
        """Print formatted backtest results."""
        if self.metrics is None:
            raise ValueError("Run backtest first")
        
        Statistics.print_metrics(self.metrics, "STRATEGY PERFORMANCE")
        
        if include_benchmark:
            comparison = self.compare_with_benchmark()
            print("\n" + "="*60)
            print("BENCHMARK PERFORMANCE (Buy & Hold)")
            print("="*60)
            bench = comparison['benchmark_metrics']
            print(f"Total Return:        {bench['total_return']*100:>10.2f}%")
            print(f"Annualized Return:   {bench['annualized_return']*100:>10.2f}%")
            print(f"Sharpe Ratio:        {bench['sharpe_ratio']:>10.2f}")
            print(f"Max Drawdown:        {bench['max_drawdown']*100:>10.2f}%")
            
            print("\n" + "="*60)
            print("OUTPERFORMANCE")
            print("="*60)
            print(f"Total Return Diff:   {comparison['total_return_diff']*100:>10.2f}%")
            print(f"Sharpe Ratio Diff:   {comparison['sharpe_diff']:>10.2f}")
            print(f"Max Drawdown Diff:   {comparison['max_drawdown_diff']*100:>10.2f}%")
            print("="*60)
