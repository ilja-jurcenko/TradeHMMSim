"""
Backtest simulation engine with walk-forward testing and rebalancing.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union, Any
from alpha_models.base import AlphaModel
from signal_filter.hmm_filter import HMMRegimeFilter
from statistics import Statistics
from alpha_model_factory import AlphaModelFactory


class BacktestEngine:
    """
    Backtest engine for testing trading strategies with optional HMM filtering.
    """
    
    def __init__(self, close: pd.Series, 
                 alpha_model: Optional[AlphaModel] = None,
                 alpha_config: Optional[Dict[str, Any]] = None,
                 hmm_filter: Optional[HMMRegimeFilter] = None,
                 initial_capital: float = 100000.0):
        """
        Initialize backtest engine.
        
        Parameters:
        -----------
        close : pd.Series
            Close price series
        alpha_model : AlphaModel, optional
            Alpha model instance for signal generation.
            Provide either this OR alpha_config, not both.
        alpha_config : Dict[str, Any], optional
            Alpha model configuration dictionary with 'type' and 'parameters'.
            Provide either this OR alpha_model, not both.
            Example: {'type': 'SMA', 'parameters': {'short_window': 50, 'long_window': 200}}
        hmm_filter : HMMRegimeFilter, optional
            HMM filter for regime-based filtering
        initial_capital : float
            Initial capital
            
        Raises:
        -------
        ValueError
            If both alpha_model and alpha_config are provided, or if neither is provided
        """
        self.close = close
        self.hmm_filter = hmm_filter
        self.initial_capital = initial_capital
        
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
        """
        Create BacktestEngine from full configuration dictionary.
        
        Parameters:
        -----------
        close : pd.Series
            Close price series
        config : Dict[str, Any]
            Full configuration dictionary (as loaded from JSON)
        hmm_filter : HMMRegimeFilter, optional
            HMM filter for regime-based filtering
            
        Returns:
        --------
        BacktestEngine
            Configured backtest engine instance
            
        Example:
        --------
        >>> from config_loader import ConfigLoader
        >>> config = ConfigLoader.load_config('config_default.json')
        >>> engine = BacktestEngine.from_config(close_prices, config, hmm_filter)
        """
        # Extract relevant sections
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
        """
        Create BacktestEngine from alpha model configuration only.
        
        Parameters:
        -----------
        close : pd.Series
            Close price series
        alpha_config : Dict[str, Any]
            Alpha model configuration with 'type' and 'parameters'
        hmm_filter : HMMRegimeFilter, optional
            HMM filter for regime-based filtering
        initial_capital : float
            Initial capital
            
        Returns:
        --------
        BacktestEngine
            Configured backtest engine instance
        """
        return BacktestEngine(
            close=close,
            alpha_config=alpha_config,
            hmm_filter=hmm_filter,
            initial_capital=initial_capital
        )
    
    def get_alpha_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the alpha model configuration used (if created from config).
        
        Returns:
        --------
        Optional[Dict[str, Any]]
            Alpha model configuration, or None if created from model instance
        """
        return self._alpha_config
        
    def run(self, 
            strategy_mode: str = 'alpha_only',
            rebalance_frequency: int = 1,
            walk_forward: bool = False,
            train_window: int = 504,
            refit_every: int = 21,
            bear_prob_threshold: float = 0.65,
            bull_prob_threshold: float = 0.65,
            transaction_cost: float = 0.0) -> Dict:
        """
        Run backtest simulation.
        
        Parameters:
        -----------
        strategy_mode : str
            Strategy mode:
            - 'alpha_only': Alpha model signals only
            - 'hmm_only': HMM regime signals only
            - 'alpha_hmm_filter': HMM filters incorrect alpha signals (bear filter)
            - 'alpha_hmm_combine': Combine alpha and HMM signals (take position when either signals)
        rebalance_frequency : int
            Rebalancing frequency (1 = every period, 5 = every 5 periods, etc.)
        walk_forward : bool
            Use walk-forward testing for HMM (if HMM is used)
        train_window : int
            Training window for walk-forward HMM
        refit_every : int
            Refit HMM model every N periods
        bear_prob_threshold : float
            Bear regime probability threshold for exit
        bull_prob_threshold : float
            Bull regime probability threshold for entry override
        transaction_cost : float
            Transaction cost per trade (as fraction, e.g., 0.001 = 0.1%)
            
        Returns:
        --------
        Dict
            Backtest results including metrics and equity curve
        """
        print(f"\n{'='*60}")
        print(f"RUNNING BACKTEST: {strategy_mode.upper()}")
        print(f"{'='*60}")
        print(f"Alpha Model: {self.alpha_model.get_name()}")
        print(f"Parameters: {self.alpha_model.get_parameters()}")
        print(f"Rebalance Frequency: {rebalance_frequency} period(s)")
        print(f"Transaction Cost: {transaction_cost*100:.3f}%")
        
        # Generate alpha signals
        alpha_signals = self.alpha_model.generate_signals(self.close)
        
        # Initialize positions
        positions = alpha_signals.copy()
        
        # Apply HMM filtering if needed
        if strategy_mode != 'alpha_only' and self.hmm_filter is not None:
            print("\nApplying HMM regime filtering...")
            
            if walk_forward:
                probs, regime, switches = self.hmm_filter.walkforward_filter(
                    self.close, 
                    train_window=train_window,
                    refit_every=refit_every
                )
            else:
                # Simple fit and predict
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
            
            # Identify regimes
            regime_info = self.hmm_filter.identify_regimes(self.close, regime)
            bear_regime = regime_info['bear_regime']
            bull_regime = regime_info['bull_regime']
            neutral_regime = regime_info['neutral_regime']
            
            print(f"  Bear regime: {bear_regime} (vol: {regime_info['regime_volatilities'][bear_regime]:.4f})")
            print(f"  Bull regime: {bull_regime} (vol: {regime_info['regime_volatilities'][bull_regime]:.4f})")
            if neutral_regime is not None:
                print(f"  Neutral regime: {neutral_regime} (vol: {regime_info['regime_volatilities'][neutral_regime]:.4f})")
            print(f"  Detected {len(switches)} regime switches")
            
            # Align indices
            common_idx = alpha_signals.index.intersection(probs.index)
            alpha_signals_aligned = alpha_signals.loc[common_idx]
            
            # Get regime probabilities
            bear_prob = probs[bear_regime].loc[common_idx]
            bull_prob = probs[bull_regime].loc[common_idx]
            
            # Combine bull + neutral for non-bearish signal
            # Neutral markets are generally safe for trading, so treat bull+neutral as favorable
            bull_prob_combined = bull_prob.copy()
            if neutral_regime is not None:
                bull_prob_combined = bull_prob + probs[neutral_regime].loc[common_idx]
            
            # Apply strategy logic
            if strategy_mode == 'hmm_only':
                # HMM only: ignore alpha signals
                # Use combined bull+neutral probability
                positions = (bull_prob_combined > bull_prob_threshold).astype(int)
                
            elif strategy_mode == 'alpha_hmm_filter':
                # Alpha + HMM filter: HMM only filters out during bear regime
                bear_filter = (bear_prob < bear_prob_threshold).astype(int)
                positions = alpha_signals_aligned * bear_filter
                
            elif strategy_mode == 'alpha_hmm_combine':
                # Alpha + HMM combine: "Buy low" contrarian strategy
                # Core idea: Use HMM to detect regime changes early
                # 
                # Strategy:
                #   1. Follow alpha signals (trend-following base)
                #   2. ADD contrarian entries: When alpha says no position but HMM predicts bull
                #      -> Buy the dip before trend turns
                #   3. DON'T do contrarian exits (too aggressive, cuts winning trades early)
                #
                # This creates asymmetry: Enter early, exit normally
                
                # Detect strong bull regime prediction
                hmm_bull_signal = (bull_prob_combined > bull_prob_threshold).astype(bool)
                
                # Start with alpha signals as base
                positions = alpha_signals_aligned.astype(bool).copy()
                
                # Contrarian ENTRY ONLY: Alpha says no position, but HMM predicts bull -> BUY THE DIP
                # This catches early regime shifts when trend hasn't turned yet
                contrarian_entry = (~alpha_signals_aligned.astype(bool)) & hmm_bull_signal
                positions[contrarian_entry] = True
                
                positions = positions.astype(int)
            
            # Store regime info for later use
            self.regime_probs = probs
            self.regime = regime
            self.regime_info = regime_info
        
        # Apply rebalancing frequency
        if rebalance_frequency > 1:
            print(f"\nApplying rebalancing frequency: {rebalance_frequency}")
            positions = self._apply_rebalancing(positions, rebalance_frequency)
        
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
        
        # Strategy returns (only when in position)
        strategy_returns = positions.shift(1).fillna(0) * price_returns
        
        # Calculate equity curve
        equity_curve = self.initial_capital * (1 + strategy_returns).cumprod()
        
        # Calculate metrics (convert to numpy array)
        metrics = Statistics.calculate_all_metrics(strategy_returns.values)
        
        # Calculate additional statistics
        num_trades = int(np.sum(np.abs(positions.diff().fillna(0))) / 2)
        time_in_market = float(np.sum(positions > 0) / len(positions))
        
        # Store results
        self.positions = positions
        self.returns = strategy_returns
        self.equity_curve = equity_curve
        self.metrics = metrics
        
        # Extract final capital (handle both Series and DataFrame)
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
            'rebalance_frequency': rebalance_frequency
        }
        
        # Add HMM-specific results if available
        if hasattr(self, 'regime_probs'):
            results['regime_probs'] = self.regime_probs
            results['regime'] = self.regime
            results['regime_info'] = self.regime_info
        
        print(f"\n{'='*60}")
        print("BACKTEST COMPLETE")
        print(f"{'='*60}")
        print(f"Number of Trades: {num_trades}")
        print(f"Time in Market: {time_in_market*100:.1f}%")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {metrics['total_return']*100:.2f}%")
        
        return results
    
    def _apply_rebalancing(self, positions: pd.Series, frequency: int) -> pd.Series:
        """
        Apply rebalancing frequency to positions.
        
        Parameters:
        -----------
        positions : pd.Series
            Original positions
        frequency : int
            Rebalancing frequency
            
        Returns:
        --------
        pd.Series
            Rebalanced positions
        """
        rebalanced = positions.copy()
        
        # Keep position unchanged between rebalancing periods
        for i in range(1, len(rebalanced)):
            if i % frequency != 0:
                rebalanced.iloc[i] = rebalanced.iloc[i-1]
        
        return rebalanced
    
    def compare_with_benchmark(self) -> Dict:
        """
        Compare strategy performance with buy-and-hold benchmark.
        
        Returns:
        --------
        Dict
            Comparison metrics
        """
        if self.returns is None:
            raise ValueError("Run backtest first before comparison")
        
        # Buy-and-hold returns
        price_returns = self.close.pct_change().fillna(0)
        benchmark_returns = price_returns.loc[self.returns.index]
        
        # Calculate benchmark metrics
        benchmark_metrics = Statistics.calculate_all_metrics(benchmark_returns)
        
        # Calculate outperformance
        comparison = {
            'strategy_metrics': self.metrics,
            'benchmark_metrics': benchmark_metrics,
            'total_return_diff': self.metrics['total_return'] - benchmark_metrics['total_return'],
            'sharpe_diff': self.metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio'],
            'max_drawdown_diff': self.metrics['max_drawdown'] - benchmark_metrics['max_drawdown']
        }
        
        return comparison
    
    def print_results(self, include_benchmark: bool = True) -> None:
        """
        Print formatted backtest results.
        
        Parameters:
        -----------
        include_benchmark : bool
            Whether to include benchmark comparison
        """
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
