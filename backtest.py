"""
Backtest simulation engine with walk-forward testing and rebalancing.
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


class BacktestEngine:
    """
    Backtest engine for testing trading strategies with optional HMM filtering.
    """
    
    def __init__(self, close: pd.Series, 
                 alpha_model: Optional[AlphaModel] = None,
                 alpha_config: Optional[Dict[str, Any]] = None,
                 hmm_filter: Optional[HMMRegimeFilter] = None,
                 bear_alpha_model: Optional[AlphaModel] = None,
                 initial_capital: float = 100000.0):
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
            
        Raises:
        -------
        ValueError
            If both alpha_model and alpha_config are provided, or if neither is provided
        """
        self.close = close
        self.hmm_filter = hmm_filter
        self.bear_alpha_model = bear_alpha_model
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
            transaction_cost: float = 0.0,
            enable_logging: bool = False,
            log_dir: str = 'logs') -> Dict:
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
            - 'regime_adaptive_alpha': Use trend-following in bull/neutral, mean-reversion in bear
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
        enable_logging : bool
            Enable detailed logging of trading decisions
        log_dir : str
            Directory to save log files
            
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
        
        # Setup logging if enabled
        log_file = None
        if enable_logging:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"{strategy_mode}_{self.alpha_model.get_name()}_{timestamp}.csv"
            log_file = os.path.join(log_dir, log_filename)
            print(f"Logging enabled: {log_file}")
        
        # Generate alpha signals
        alpha_signals = self.alpha_model.generate_signals(self.close)
        
        # Apply finite state machine logic to validate transitions
        positions = self._apply_fsm_logic(alpha_signals)
        
        # Initialize logging data structure
        log_data = [] if enable_logging else None
        
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
                
                # Log decisions
                if enable_logging:
                    for i, idx in enumerate(common_idx):
                        prev_pos = 0 if i == 0 else positions.iloc[i-1]
                        curr_pos = positions.iloc[i]
                        action = self._determine_action(prev_pos, curr_pos)
                        
                        log_data.append({
                            'Date': idx,
                            'Price': self.close.loc[idx],
                            'Alpha_Signal': alpha_signals_aligned.iloc[i],
                            'Bear_Prob': bear_prob.iloc[i],
                            'Bull_Prob': bull_prob.iloc[i],
                            'Bull_Combined_Prob': bull_prob_combined.iloc[i],
                            'HMM_Signal': 1 if bull_prob_combined.iloc[i] > bull_prob_threshold else 0,
                            'Position': curr_pos,
                            'Action': action
                        })
                
            elif strategy_mode == 'alpha_hmm_filter':
                # Alpha + HMM filter: HMM only filters out during bear regime
                bear_filter = (bear_prob < bear_prob_threshold).astype(int)
                positions = alpha_signals_aligned * bear_filter
                
                # Log decisions
                if enable_logging:
                    for i, idx in enumerate(common_idx):
                        prev_pos = 0 if i == 0 else positions.iloc[i-1]
                        curr_pos = positions.iloc[i]
                        action = self._determine_action(prev_pos, curr_pos)
                        
                        log_data.append({
                            'Date': idx,
                            'Price': self.close.loc[idx],
                            'Alpha_Signal': alpha_signals_aligned.iloc[i],
                            'Bear_Prob': bear_prob.iloc[i],
                            'Bull_Prob': bull_prob.iloc[i],
                            'Bear_Filter': bear_filter.iloc[i],
                            'HMM_Signal': 0 if bear_prob.iloc[i] >= bear_prob_threshold else 1,
                            'Position': curr_pos,
                            'Action': action
                        })
                
            elif strategy_mode == 'alpha_hmm_combine':
                # Alpha + HMM combine: 4-State Variance-Trend Strategy
                # 
                # Combines HMM (market variance/regime) with Alpha (trend direction):
                #   - HMM detects variance: Low (bull/neutral prob > threshold) vs High (bear prob > threshold)
                #   - Alpha detects trend: Bullish vs Bearish
                #
                # Four States:
                #   State 1: Low variance + Bullish trend  → BUY (if not in position)
                #   State 2: Low variance + Bearish trend  → HOLD (no action)
                #   State 3: High variance + Bullish trend → HOLD (no action)
                #   State 4: High variance + Bearish trend → SELL (if in position)
                #
                # Trading Logic:
                #   - Enter long only in State 1 (safe bull market)
                #   - Exit long only in State 4 (dangerous bear market)
                #   - All other states: maintain current position
                
                # Detect variance regimes
                low_variance = (bull_prob_combined > bull_prob_threshold).astype(bool)
                high_variance = (bear_prob > bear_prob_threshold).astype(bool)
                
                # Detect trend direction (alpha signals: 1=bullish, 0/negative=bearish)
                bullish_trend = (alpha_signals_aligned > 0).astype(bool)
                bearish_trend = ~bullish_trend
                
                # Define the 4 states
                state_1 = low_variance & bullish_trend   # Low variance + Bullish → BUY signal
                state_2 = low_variance & bearish_trend   # Low variance + Bearish → HOLD
                state_3 = high_variance & bullish_trend  # High variance + Bullish → HOLD
                state_4 = high_variance & bearish_trend  # High variance + Bearish → SELL signal
                
                # Create state labels for each period (for visualization)
                state_labels = pd.Series('Unknown', index=common_idx)
                state_labels[state_1] = 'State 1: Low Var + Bull'
                state_labels[state_2] = 'State 2: Low Var + Bear'
                state_labels[state_3] = 'State 3: High Var + Bull'
                state_labels[state_4] = 'State 4: High Var + Bear'
                
                # Initialize positions array
                positions = pd.Series(0, index=common_idx, dtype=int)
                
                # State machine: track position and respond to state transitions
                current_position = 0  # Start flat
                # Track previous state for each period
                prev_state = None  # Assume starting in State 1
                
                for i in range(len(common_idx)):
                    idx = common_idx[i]
                    prev_position = current_position
                    
                    # Determine current state
                    if state_1.iloc[i]:
                        current_state_num = 1
                    elif state_2.iloc[i]:
                        current_state_num = 2
                    elif state_3.iloc[i]:
                        current_state_num = 3
                    elif state_4.iloc[i]:
                        current_state_num = 4
                    else:
                        current_state_num = None
                    
                    # Detect state transitions and update position
                    if prev_state is not None and current_state_num is not None:
                        # State switch from 1 to (2,3,4): SELL
                        if prev_state in [1,2,3] and current_state_num in [4] and current_position==1:
                            current_position = 0
                        # State switch from (2,3,4) to 1: BUY
                        elif prev_state in [2,3,4] and current_state_num in [1] and current_position==0:
                            current_position = 1
                    
                    # Update prev_state for next iteration
                    prev_state = current_state_num
                    
                    positions.iloc[i] = current_position
                    
                    # Log decisions
                    if enable_logging:
                        # Determine which state
                        if state_1.iloc[i]:
                            state_label = 'State_1_Low_Var_Bull'
                        elif state_2.iloc[i]:
                            state_label = 'State_2_Low_Var_Bear'
                        elif state_3.iloc[i]:
                            state_label = 'State_3_High_Var_Bull'
                        elif state_4.iloc[i]:
                            state_label = 'State_4_High_Var_Bear'
                        else:
                            state_label = 'Unknown'
                        
                        action = self._determine_action(prev_position, current_position)
                        
                        log_data.append({
                            'Date': idx,
                            'Price': self.close.loc[idx],
                            'Alpha_Signal': alpha_signals_aligned.iloc[i],
                            'Bear_Prob': bear_prob.iloc[i],
                            'Bull_Prob': bull_prob.iloc[i],
                            'Bull_Combined_Prob': bull_prob_combined.iloc[i],
                            'Low_Variance': low_variance.iloc[i],
                            'High_Variance': high_variance.iloc[i],
                            'State': state_label,
                            'Position': current_position,
                            'Action': action
                        })
                
                # Store state information for plotting
                self.state_labels = state_labels
                self.state_1 = state_1
                self.state_2 = state_2
                self.state_3 = state_3
                self.state_4 = state_4
                
                # Log state transitions for diagnostics
                num_state1 = state_1.sum()
                num_state2 = state_2.sum()
                num_state3 = state_3.sum()
                num_state4 = state_4.sum()
                print(f"  State 1 (Low var + Bull): {num_state1} periods")
                print(f"  State 2 (Low var + Bear): {num_state2} periods")
                print(f"  State 3 (High var + Bull): {num_state3} periods")
                print(f"  State 4 (High var + Bear): {num_state4} periods")
            
            elif strategy_mode == 'regime_adaptive_alpha':
                # Regime-Adaptive Alpha: Switch strategies based on market regime
                # 
                # Strategy:
                #   - Bull/Neutral: Use trend-following alpha (standard alpha_model)
                #   - Bear: Use mean-reversion alpha (bear_alpha_model, typically Bollinger Bands)
                #
                # This adapts to market conditions: ride trends up, catch bounces down
                
                if self.bear_alpha_model is None:
                    raise ValueError(
                        "regime_adaptive_alpha strategy requires bear_alpha_model to be set"
                    )
                
                # Generate signals from bear market alpha model (e.g., Bollinger Bands)
                bear_signals = self.bear_alpha_model.generate_signals(self.close)
                bear_signals_aligned = bear_signals.reindex(common_idx, fill_value=0)
                
                # Determine current regime for each period
                positions = pd.Series(0, index=common_idx)
                
                # Use trend-following in bull/neutral, mean-reversion in bear
                for i in range(len(common_idx)):
                    idx = common_idx[i]
                    current_regime = regime.loc[idx]
                    prev_pos = 0 if i == 0 else positions.iloc[i-1]
                    
                    if current_regime == 'bear':
                        # Use bear market strategy (mean-reversion)
                        positions.iloc[i] = bear_signals_aligned.iloc[i]
                        active_model = 'Bear_Alpha'
                    else:
                        # Use bull/neutral market strategy (trend-following)
                        positions.iloc[i] = alpha_signals_aligned.iloc[i]
                        active_model = 'Bull_Alpha'
                    
                    # Log decisions
                    if enable_logging:
                        action = self._determine_action(prev_pos, positions.iloc[i])
                        
                        log_data.append({
                            'Date': idx,
                            'Price': self.close.loc[idx],
                            'Alpha_Signal': alpha_signals_aligned.iloc[i],
                            'Bear_Signal': bear_signals_aligned.iloc[i],
                            'Regime': current_regime,
                            'Active_Model': active_model,
                            'Position': positions.iloc[i],
                            'Action': action
                        })
                
                print(f"  Regime switches - Trend-following: {(regime != 'bear').sum()} periods, "
                      f"Mean-reversion: {(regime == 'bear').sum()} periods")
            
            # Store regime info for later use
            self.regime_probs = probs
            self.regime = regime
            self.regime_info = regime_info
        
        # Apply rebalancing frequency
        if rebalance_frequency > 1:
            print(f"\nApplying rebalancing frequency: {rebalance_frequency}")
            positions = self._apply_rebalancing(positions, rebalance_frequency)
        
        # Log alpha_only strategy decisions
        if enable_logging and strategy_mode == 'alpha_only':
            for i in range(len(positions)):
                idx = positions.index[i]
                prev_pos = 0 if i == 0 else positions.iloc[i-1]
                curr_pos = positions.iloc[i]
                action = self._determine_action(prev_pos, curr_pos)
                
                log_data.append({
                    'Date': idx,
                    'Price': self.close.loc[idx],
                    'Alpha_Signal': alpha_signals.loc[idx],
                    'Position': curr_pos,
                    'Action': action
                })
        
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
        
        # Add portfolio value to log data
        if enable_logging and log_data:
            for i, entry in enumerate(log_data):
                if entry['Date'] in equity_curve.index:
                    entry['Portfolio_Value'] = equity_curve.loc[entry['Date']]
                else:
                    entry['Portfolio_Value'] = self.initial_capital
        
        # Calculate metrics (convert to numpy array)
        metrics = Statistics.calculate_all_metrics(strategy_returns.values)
        
        # Calculate additional statistics
        num_trades = int(np.sum(np.abs(positions.diff().fillna(0))) / 2)
        time_in_market = float(np.sum(positions != 0) / len(positions))
        
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
        
        # Add state information for alpha_hmm_combine strategy
        if hasattr(self, 'state_labels'):
            results['state_labels'] = self.state_labels
            results['state_1'] = self.state_1
            results['state_2'] = self.state_2
            results['state_3'] = self.state_3
            results['state_4'] = self.state_4
        
        print(f"\n{'='*60}")
        print("BACKTEST COMPLETE")
        print(f"{'='*60}")
        print(f"Number of Trades: {num_trades}")
        print(f"Time in Market: {time_in_market*100:.1f}%")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {metrics['total_return']*100:.2f}%")
        
        # Write log file
        if enable_logging and log_data and log_file:
            log_df = pd.DataFrame(log_data)
            log_df.to_csv(log_file, index=False)
            print(f"\n✓ Log file saved: {log_file}")
            print(f"  Total entries: {len(log_df)}")
        
        return results
    
    def _determine_action(self, prev_position: int, curr_position: int) -> str:
        """
        Determine trading action based on position change.
        
        Parameters:
        -----------
        prev_position : int
            Previous position (-1, 0, or 1)
        curr_position : int
            Current position (-1, 0, or 1)
            
        Returns:
        --------
        str
            Action taken: BUY, SELL, SHORT, COVER, or HOLD
        """
        if prev_position == curr_position:
            return 'HOLD'
        elif prev_position == 0 and curr_position == 1:
            return 'BUY'
        elif prev_position == 1 and curr_position == 0:
            return 'SELL'
        elif prev_position == 0 and curr_position == -1:
            return 'SHORT'
        elif prev_position == -1 and curr_position == 0:
            return 'COVER'
        elif prev_position == 1 and curr_position == -1:
            return 'SELL_AND_SHORT'
        elif prev_position == -1 and curr_position == 1:
            return 'COVER_AND_BUY'
        else:
            return 'UNKNOWN'
    
    def _apply_fsm_logic(self, signals: pd.Series) -> pd.Series:
        """
        Apply finite state machine logic to validate position transitions.
        
        Position States:
        - FLAT (0): No position
        - LONG (1): Holding long position
        - SHORT (-1): Holding short position
        
        Allowed Transitions:
        - FLAT -> LONG (BUY signal: 0 -> 1)
        - FLAT -> SHORT (SHORT signal: 0 -> -1)
        - LONG -> FLAT (SELL signal: 1 -> 0)
        - SHORT -> FLAT (COVER signal: -1 -> 0)
        
        Invalid Transitions (ignored):
        - LONG -> LONG (can't buy when already long)
        - SHORT -> SHORT (can't short when already short)
        - FLAT -> FLAT with SELL/COVER (can't exit when no position)
        
        Parameters:
        -----------
        signals : pd.Series
            Raw signals from alpha model
            
        Returns:
        --------
        pd.Series
            Validated positions following FSM rules
        """
        positions = pd.Series(0, index=signals.index)
        current_state = 0  # Start FLAT
        
        for i in range(len(signals)):
            requested_signal = signals.iloc[i]
            
            # State transition logic
            if current_state == 0:  # FLAT
                # Can BUY (go LONG) or SHORT
                if requested_signal == 1:
                    current_state = 1  # BUY: FLAT -> LONG
                elif requested_signal == -1:
                    current_state = -1  # SHORT: FLAT -> SHORT
                # else: stay FLAT (ignore redundant FLAT signals)
                
            elif current_state == 1:  # LONG
                # Can only SELL (go FLAT)
                if requested_signal == 0:
                    current_state = 0  # SELL: LONG -> FLAT
                elif requested_signal == -1:
                    # Direct transition LONG -> SHORT (exit long, enter short)
                    current_state = -1
                # else: stay LONG (ignore redundant LONG signals)
                
            elif current_state == -1:  # SHORT
                # Can only COVER (go FLAT)
                if requested_signal == 0:
                    current_state = 0  # COVER: SHORT -> FLAT
                elif requested_signal == 1:
                    # Direct transition SHORT -> LONG (cover short, enter long)
                    current_state = 1
                # else: stay SHORT (ignore redundant SHORT signals)
            
            positions.iloc[i] = current_state
        
        return positions
    
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
