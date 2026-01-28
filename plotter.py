"""
Visualization module for plotting backtest results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class BacktestPlotter:
    """
    Create visualizations for backtest results.
    """
    
    @staticmethod
    def plot_4state_strategy(results: dict, close: pd.Series, figsize: tuple = (14, 8), save_path: str = None) -> None:
        """
        Plot alpha_hmm_combine strategy with 4-state visualization.
        
        Shows equity price colored by the 4 states:
        - State 1: Low variance + Bullish (green) → BUY
        - State 2: Low variance + Bearish (yellow) → HOLD
        - State 3: High variance + Bullish (orange) → HOLD
        - State 4: High variance + Bearish (red) → SELL
        
        Parameters:
        -----------
        results : dict
            Backtest results dictionary from BacktestEngine with state information
        close : pd.Series
            Close price series
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure. If None, displays the plot.
        """
        if 'state_labels' not in results:
            print("Warning: No state information found in results. This plot is only for alpha_hmm_combine strategy.")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Extract data
        state_1 = results['state_1']
        state_2 = results['state_2']
        state_3 = results['state_3']
        state_4 = results['state_4']
        
        # Align close prices with state indices
        common_idx = state_1.index.intersection(close.index)
        close_aligned = close.loc[common_idx]
        
        # Color mapping
        state_colors = {
            'State 1': '#2ecc71',  # Green - BUY signal
            'State 2': '#f1c40f',  # Yellow - HOLD (low var, bearish)
            'State 3': '#e67e22',  # Orange - HOLD (high var, bullish)
            'State 4': '#e74c3c'   # Red - SELL signal
        }
        
        # Plot 1: Equity price colored by states
        ax1 = axes[0]
        
        # Plot price segments colored by state
        for i in range(len(close_aligned) - 1):
            idx = close_aligned.index[i]
            idx_next = close_aligned.index[i + 1]
            
            # Determine color based on state
            if state_1.loc[idx]:
                color = state_colors['State 1']
            elif state_2.loc[idx]:
                color = state_colors['State 2']
            elif state_3.loc[idx]:
                color = state_colors['State 3']
            elif state_4.loc[idx]:
                color = state_colors['State 4']
            else:
                color = 'gray'
            
            ax1.plot([idx, idx_next], 
                    [close_aligned.iloc[i], close_aligned.iloc[i + 1]], 
                    color=color, linewidth=1.5, alpha=0.8)
        
        ax1.set_ylabel('Equity Price ($)', fontsize=10)
        ax1.set_title('Equity Price Colored by 4-State Logic', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend([plt.Line2D([0], [0], color=state_colors['State 1'], linewidth=2),
                   plt.Line2D([0], [0], color=state_colors['State 2'], linewidth=2),
                   plt.Line2D([0], [0], color=state_colors['State 3'], linewidth=2),
                   plt.Line2D([0], [0], color=state_colors['State 4'], linewidth=2)],
                  ['State 1: Low Var + Bull (BUY)', 
                   'State 2: Low Var + Bear (HOLD)',
                   'State 3: High Var + Bull (HOLD)',
                   'State 4: High Var + Bear (SELL)'],
                  loc='upper left', fontsize=8)
        
        # Plot 2: State timeline
        ax2 = axes[1]
        
        # Create state numeric representation for visualization
        state_numeric = pd.Series(0, index=common_idx)
        state_numeric[state_1] = 1
        state_numeric[state_2] = 2
        state_numeric[state_3] = 3
        state_numeric[state_4] = 4
        
        # Plot state timeline as colored bars
        for i in range(len(state_numeric)):
            idx = state_numeric.index[i]
            state_val = state_numeric.iloc[i]
            
            if state_val == 1:
                color = state_colors['State 1']
            elif state_val == 2:
                color = state_colors['State 2']
            elif state_val == 3:
                color = state_colors['State 3']
            elif state_val == 4:
                color = state_colors['State 4']
            else:
                color = 'gray'
            
            ax2.axvline(x=idx, color=color, alpha=0.6, linewidth=0.5)
        
        ax2.set_ylabel('State', fontsize=10)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_title('State Timeline', fontsize=12, fontweight='bold')
        ax2.set_yticks([1, 2, 3, 4])
        ax2.set_yticklabels(['State 1\n(BUY)', 'State 2\n(HOLD)', 'State 3\n(HOLD)', 'State 4\n(SELL)'], fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"4-state plot saved to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_hmm_regime_colored_equity(results: dict, close: pd.Series, 
                                       bear_threshold: float = 0.65,
                                       bull_threshold: float = 0.65,
                                       figsize: tuple = (14, 10), 
                                       save_path: str = None) -> None:
        """
        Plot equity price colored by active regime for HMM_Only and Oracle strategies.
        
        Colors the equity price line based on which regime probability exceeds the threshold:
        - Green: Bull regime (bull_prob > threshold)
        - Orange: Neutral regime (neither bull nor bear exceed threshold)
        - Red: Bear regime (bear_prob > threshold)
        
        Parameters:
        -----------
        results : dict
            Backtest results dictionary from BacktestEngine with regime information
        close : pd.Series
            Close price series
        bear_threshold : float
            Bear regime probability threshold
        bull_threshold : float
            Bull regime probability threshold
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure. If None, displays the plot.
        """
        if 'regime_probs' not in results or 'regime_info' not in results:
            print("Warning: No regime information found in results. This plot is only for HMM_Only and Oracle strategies.")
            return
        
        probs = results['regime_probs']
        regime_info = results['regime_info']
        positions = results['positions']
        equity_curve = results['equity_curve']
        
        # Get regime IDs
        bear_regime = regime_info['bear_regime']
        bull_regime = regime_info['bull_regime']
        neutral_regime = regime_info.get('neutral_regime')
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Align indices
        common_idx = probs.index.intersection(close.index)
        close_aligned = close.loc[common_idx]
        probs_aligned = probs.loc[common_idx]
        
        # Get probabilities
        bear_prob = probs_aligned[bear_regime]
        bull_prob = probs_aligned[bull_regime]
        if neutral_regime is not None:
            neutral_prob = probs_aligned[neutral_regime]
            bull_combined_prob = bull_prob + neutral_prob
        else:
            bull_combined_prob = bull_prob
        
        # Determine active regime for each time point
        active_regime = pd.Series('Neutral', index=common_idx)
        active_regime[bear_prob >= bear_threshold] = 'Bear'
        active_regime[bull_combined_prob > bull_threshold] = 'Bull'
        
        # Regime colors
        regime_colors = {
            'Bull': '#2ecc71',    # Green
            'Neutral': '#f39c12',  # Orange
            'Bear': '#e74c3c'      # Red
        }
        
        # Plot 1: Equity price colored by regime
        ax1 = axes[0]
        
        for i in range(len(close_aligned) - 1):
            idx = close_aligned.index[i]
            idx_next = close_aligned.index[i + 1]
            regime = active_regime.loc[idx]
            color = regime_colors[regime]
            
            ax1.plot([idx, idx_next], 
                    [close_aligned.iloc[i], close_aligned.iloc[i + 1]], 
                    color=color, linewidth=1.5, alpha=0.8)
        
        ax1.set_ylabel('Price ($)', fontsize=10)
        ax1.set_title(f'Equity Price Colored by Active Regime ({results.get("strategy_mode", "HMM").upper()})', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend([plt.Line2D([0], [0], color=regime_colors['Bull'], linewidth=2),
                   plt.Line2D([0], [0], color=regime_colors['Neutral'], linewidth=2),
                   plt.Line2D([0], [0], color=regime_colors['Bear'], linewidth=2)],
                  ['Bull Regime (Active)', 'Neutral Regime', 'Bear Regime (Active)'],
                  loc='upper left', fontsize=9)
        
        # Plot 2: Regime probabilities with thresholds
        ax2 = axes[1]
        
        ax2.plot(probs_aligned.index, bear_prob, color='red', linewidth=1.5, 
                alpha=0.8, label='Bear Prob')
        ax2.plot(probs_aligned.index, bull_prob, color='green', linewidth=1.5, 
                alpha=0.8, label='Bull Prob')
        if neutral_regime is not None:
            ax2.plot(probs_aligned.index, neutral_prob, color='orange', linewidth=1.5, 
                    alpha=0.8, label='Neutral Prob')
        
        # Add threshold lines
        ax2.axhline(bear_threshold, color='red', linestyle='--', alpha=0.5, 
                   linewidth=1, label=f'Bear Threshold ({bear_threshold})')
        ax2.axhline(bull_threshold, color='green', linestyle='--', alpha=0.5, 
                   linewidth=1, label=f'Bull Threshold ({bull_threshold})')
        
        # Highlight active regimes as background
        for regime_name, regime_color in regime_colors.items():
            mask = active_regime == regime_name
            if mask.any():
                ax2.fill_between(common_idx, 0, 1, where=mask, 
                               alpha=0.1, color=regime_color, interpolate=True)
        
        ax2.set_ylabel('Probability', fontsize=10)
        ax2.set_title('Regime Probabilities with Thresholds', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.legend(loc='upper left', fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Positions and equity curve
        ax3 = axes[2]
        
        # Align equity curve
        equity_aligned = equity_curve.reindex(common_idx, method='ffill')
        initial_capital = results.get('initial_capital', 100000)
        equity_pct = (equity_aligned / initial_capital - 1) * 100
        
        ax3.plot(common_idx, equity_pct, color='blue', linewidth=2, label='Strategy Return')
        ax3.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # Highlight when in position
        positions_aligned = positions.reindex(common_idx, fill_value=0)
        in_position = positions_aligned > 0
        if in_position.any():
            y_min = equity_pct.min()
            y_max = equity_pct.max()
            ax3.fill_between(common_idx, y_min, y_max, where=in_position, 
                           alpha=0.15, color='green', label='In Position', interpolate=True)
        
        ax3.set_ylabel('Return (%)', fontsize=10)
        ax3.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Regime timeline
        ax4 = axes[3]
        
        # Create numeric representation for regimes
        regime_numeric = pd.Series(1, index=common_idx)  # Default neutral
        regime_numeric[active_regime == 'Bull'] = 2
        regime_numeric[active_regime == 'Bear'] = 0
        
        # Plot as colored vertical lines
        for i in range(len(regime_numeric)):
            idx = regime_numeric.index[i]
            regime_val = regime_numeric.iloc[i]
            
            if regime_val == 2:  # Bull
                color = regime_colors['Bull']
            elif regime_val == 1:  # Neutral
                color = regime_colors['Neutral']
            else:  # Bear
                color = regime_colors['Bear']
            
            ax4.axvline(x=idx, color=color, alpha=0.6, linewidth=0.5)
        
        ax4.set_ylabel('Regime', fontsize=10)
        ax4.set_xlabel('Date', fontsize=10)
        ax4.set_title('Regime Timeline', fontsize=12, fontweight='bold')
        ax4.set_yticks([0, 1, 2])
        ax4.set_yticklabels(['Bear', 'Neutral', 'Bull'], fontsize=9)
        ax4.set_ylim([-0.5, 2.5])
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"HMM regime-colored equity plot saved to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_results(results: dict, close: pd.Series, figsize: tuple = (14, 12), save_path: str = None) -> None:
        """
        Plot comprehensive backtest results with 4 subplots.
        
        Parameters:
        -----------
        results : dict
            Backtest results dictionary from BacktestEngine
        close : pd.Series
            Close price series
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure. If None, displays the plot.
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Cumulative returns comparison
        BacktestPlotter._plot_cumulative_returns(axes[0], results)
        
        # Plot 2: Price with signals and positions
        BacktestPlotter._plot_price_and_signals(axes[1], results, close)
        
        # Plot 3: Regime probabilities (if available)
        if 'regime_probs' in results:
            BacktestPlotter._plot_regime_probabilities(axes[2], results)
        else:
            BacktestPlotter._plot_positions_over_time(axes[2], results)
        
        # Plot 4: Drawdown comparison
        BacktestPlotter._plot_drawdowns(axes[3], results)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
    
    @staticmethod
    def _plot_cumulative_returns(ax, results: dict) -> None:
        """Plot cumulative returns for strategy vs benchmark."""
        equity = results['equity_curve']
        initial_capital = results.get('initial_capital', 100000)
        
        # Strategy cumulative return as percentage
        strategy_cum = (equity / initial_capital - 1) * 100
        
        # Benchmark cumulative return (Buy & Hold) as percentage
        if 'close_prices' in results:
            close_prices = results['close_prices']
            # Align with strategy index
            common_idx = strategy_cum.index
            close_aligned = close_prices.loc[common_idx]
            # Calculate buy and hold cumulative return as percentage
            benchmark_cum = (close_aligned / close_aligned.iloc[0] - 1) * 100
        else:
            # Fallback if close_prices not available
            benchmark_cum = pd.Series([0] * len(equity), index=equity.index)
        
        ax.plot(strategy_cum.index, strategy_cum.values, 
                label='Strategy', color='blue', linewidth=2)
        ax.plot(benchmark_cum.index, benchmark_cum.values, 
                label='Buy & Hold', color='gray', linewidth=2, alpha=0.7)
        
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title('Strategy vs Benchmark Performance', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    @staticmethod
    def _plot_price_and_signals(ax, results: dict, close: pd.Series) -> None:
        """Plot price with moving averages and position highlights."""
        positions = results['positions']
        strategy_mode = results.get('strategy_mode', 'unknown')
        alpha_model = results.get('alpha_model', 'Unknown')
        
        # Align close prices with positions
        common_idx = positions.index
        close_aligned = close.loc[common_idx]
        
        # Plot price
        ax.plot(common_idx, close_aligned, label='Close Price', 
                color='black', linewidth=1.5, alpha=0.8)
        
        # Highlight when in position - use actual y-values for better visualization
        in_position = positions > 0
        if in_position.any():
            # Create a mask that properly handles transitions
            y_min = close_aligned.min() * 0.98  # Slightly below minimum
            y_max = close_aligned.max() * 1.02  # Slightly above maximum
            
            ax.fill_between(common_idx, y_min, y_max, 
                           where=in_position, alpha=0.15, color='green', 
                           label='In Position', interpolate=True)
        
        # Add entry/exit points
        position_changes = positions.diff()
        entries = position_changes[position_changes > 0].index
        exits = position_changes[position_changes < 0].index
        
        if len(entries) > 0:
            ax.scatter(entries, close_aligned.loc[entries], 
                      color='green', marker='^', s=100, 
                      label='Entry', zorder=5, edgecolors='black', linewidths=1)
        if len(exits) > 0:
            ax.scatter(exits, close_aligned.loc[exits], 
                      color='red', marker='v', s=100, 
                      label='Exit', zorder=5, edgecolors='black', linewidths=1)
        
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title(f'Price & Trading Signals ({alpha_model} - {strategy_mode})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def _plot_regime_probabilities(ax, results: dict) -> None:
        """Plot regime probabilities and position overlay."""
        probs = results['regime_probs']
        positions = results['positions']
        regime_info = results.get('regime_info', {})
        
        # Get regime IDs
        bear_regime = regime_info.get('bear_regime', 2)
        bull_regime = regime_info.get('bull_regime', 0)
        neutral_regime = regime_info.get('neutral_regime', 1)
        
        # Align indices
        common_idx = probs.index.intersection(positions.index)
        probs_aligned = probs.loc[common_idx]
        positions_aligned = positions.loc[common_idx]
        
        # Plot probabilities
        colors = {bear_regime: 'red', bull_regime: 'green', neutral_regime: 'orange'}
        labels = {bear_regime: 'Bear', bull_regime: 'Bull', neutral_regime: 'Neutral'}
        
        for state in sorted(probs.columns):
            color = colors.get(state, 'blue')
            label = labels.get(state, f'State {state}')
            ax.plot(probs_aligned.index, probs_aligned[state], 
                   color=color, linewidth=1.5, alpha=0.8, label=f'{label} Regime')
        
        # Highlight when in position
        in_position = positions_aligned > 0
        if in_position.any():
            ax.fill_between(common_idx, 0, 1, where=in_position, 
                           alpha=0.15, color='blue', label='In Position')
        
        # Add threshold lines
        ax.axhline(0.65, color='black', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(0.5, color='black', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Regime Probabilities', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def _plot_positions_over_time(ax, results: dict) -> None:
        """Plot position changes over time (fallback when no HMM data)."""
        positions = results['positions']
        
        ax.fill_between(positions.index, 0, positions, 
                       alpha=0.5, color='green', label='Position')
        ax.set_ylabel('Position', fontsize=12)
        ax.set_title('Position Over Time', fontsize=14, fontweight='bold')
        ax.set_ylim([-0.1, 1.1])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def _plot_drawdowns(ax, results: dict) -> None:
        """Plot drawdown comparison."""
        equity = results['equity_curve']
        returns = results['returns']
        positions = results['positions']
        
        # Strategy drawdown
        running_max = equity.expanding().max()
        strategy_dd = (equity - running_max) / running_max
        
        # Benchmark drawdown (approximate from returns)
        if len(returns) > 0 and len(positions) > 0:
            # Recover price returns
            price_returns = returns / positions.shift(1).fillna(0).replace(0, 1)
            price_returns = price_returns.fillna(0)
            benchmark_equity = (1 + price_returns).cumprod()
            benchmark_running_max = benchmark_equity.expanding().max()
            benchmark_dd = (benchmark_equity - benchmark_running_max) / benchmark_running_max
        else:
            benchmark_dd = pd.Series([0] * len(equity), index=equity.index)
        
        ax.fill_between(strategy_dd.index, strategy_dd.values, 0, 
                       alpha=0.5, color='blue', label='Strategy DD')
        ax.fill_between(benchmark_dd.index, benchmark_dd.values, 0, 
                       alpha=0.3, color='gray', label='Benchmark DD')
        
        ax.set_ylabel('Drawdown', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def plot_regime_analysis(probs: pd.DataFrame, regime: pd.Series, 
                            close: pd.Series, switches: pd.Series = None,
                            figsize: tuple = (14, 10)) -> None:
        """
        Plot detailed regime analysis with price coloring.
        
        Parameters:
        -----------
        probs : pd.DataFrame
            Regime probabilities
        regime : pd.Series
            Active regime at each time point
        close : pd.Series
            Close prices
        switches : pd.Series, optional
            Regime switch points
        figsize : tuple
            Figure size (width, height)
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Price with regime coloring
        ax1 = axes[0]
        colors = ['green', 'orange', 'red']
        
        # Align close prices with regime index
        close_aligned = close.loc[regime.index]
        
        for state in sorted(regime.unique()):
            mask = regime == state
            state_dates = regime[mask].index
            state_prices = close_aligned[mask]
            ax1.scatter(state_dates, state_prices, 
                       c=colors[state], alpha=0.6, s=10, 
                       label=f'Regime {state}')
        
        # Mark switch points
        if switches is not None:
            for switch_date in switches.index:
                ax1.axvline(switch_date, color='black', linestyle='--', 
                           alpha=0.3, linewidth=0.5)
        
        ax1.set_ylabel('Close Price', fontsize=12)
        ax1.set_title('Price with Regime Coloring & Switch Points', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Regime probabilities
        ax2 = axes[1]
        for state in range(probs.shape[1]):
            ax2.plot(probs.index, probs[state], 
                    label=f'P(Regime {state})', alpha=0.7)
        
        ax2.axhline(0.7, color='red', linestyle='--', alpha=0.5, 
                   linewidth=1, label='Enter Threshold')
        ax2.axhline(0.55, color='orange', linestyle='--', alpha=0.5, 
                   linewidth=1, label='Exit Threshold')
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title('Filtered State Probabilities', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Active regime
        ax3 = axes[2]
        ax3.plot(regime.index, regime.values, drawstyle='steps-post', 
                color='blue', linewidth=1.5)
        ax3.set_ylabel('Active Regime', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_title('Detected Regime Over Time', fontsize=14, fontweight='bold')
        ax3.set_yticks(sorted(regime.unique()))
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_comparison(results_list: list, labels: list, 
                       figsize: tuple = (14, 8)) -> None:
        """
        Compare multiple backtest results.
        
        Parameters:
        -----------
        results_list : list
            List of backtest result dictionaries
        labels : list
            List of labels for each backtest
        figsize : tuple
            Figure size (width, height)
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Equity curves
        ax1 = axes[0]
        for results, label in zip(results_list, labels):
            equity = results['equity_curve']
            initial_capital = results.get('initial_capital', 100000)
            cum_return = (equity / initial_capital - 1) * 100
            ax1.plot(cum_return.index, cum_return.values, 
                    label=label, linewidth=2, alpha=0.8)
        
        ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax1.set_title('Strategy Comparison - Equity Curves', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Plot 2: Drawdowns
        ax2 = axes[1]
        for results, label in zip(results_list, labels):
            equity = results['equity_curve']
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max
            ax2.fill_between(drawdown.index, drawdown.values, 0, 
                           alpha=0.4, label=label)
        
        ax2.set_ylabel('Drawdown', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_title('Strategy Comparison - Drawdowns', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(results_list: list, labels: list,
                               figsize: tuple = (12, 8)) -> None:
        """
        Create bar charts comparing key metrics across strategies.
        
        Parameters:
        -----------
        results_list : list
            List of backtest result dictionaries
        labels : list
            List of labels for each backtest
        figsize : tuple
            Figure size (width, height)
        """
        # Extract metrics
        metrics_data = {
            'Total Return (%)': [],
            'Sharpe Ratio': [],
            'Max Drawdown (%)': [],
            'Win Rate (%)': []
        }
        
        for results in results_list:
            metrics = results['metrics']
            metrics_data['Total Return (%)'].append(metrics['total_return'] * 100)
            metrics_data['Sharpe Ratio'].append(metrics['sharpe_ratio'])
            metrics_data['Max Drawdown (%)'].append(metrics['max_drawdown'] * 100)
            metrics_data['Win Rate (%)'].append(metrics['win_rate'] * 100)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (metric_name, values) in enumerate(metrics_data.items()):
            ax = axes[idx]
            x_pos = np.arange(len(labels))
            colors = ['green' if v > 0 else 'red' for v in values]
            
            ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
