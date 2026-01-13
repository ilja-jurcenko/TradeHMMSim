"""
Example demonstrating short position functionality in backtest engine.
Shows how to create alpha models that generate short signals.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from backtest import BacktestEngine
from alpha_models.base import AlphaModel


class SimpleShortAlphaModel(AlphaModel):
    """
    Simple alpha model that shorts during downtrends.
    
    Strategy:
    - Long when short MA > long MA (uptrend)
    - Short when short MA < long MA (downtrend)
    """
    
    def __init__(self, short_window: int = 10, long_window: int = 30):
        """Initialize with MA parameters."""
        super().__init__(short_window, long_window)
        
    def calculate_indicators(self, close: pd.Series):
        """Calculate short and long moving averages."""
        short_ma = close.rolling(window=self.short_window).mean()
        long_ma = close.rolling(window=self.long_window).mean()
        return short_ma, long_ma
        
    def generate_signals(self, close: pd.Series) -> pd.Series:
        """
        Generate signals: 1 for long, -1 for short, 0 for flat.
        
        Returns:
        --------
        pd.Series
            Trading signals with values in {-1, 0, 1}
        """
        short_ma, long_ma = self.calculate_indicators(close)
        
        # Initialize signals
        signals = pd.Series(0, index=close.index)
        
        # Long when short MA > long MA
        signals[short_ma > long_ma] = 1
        
        # Short when short MA < long MA
        signals[short_ma < long_ma] = -1
        
        return signals
    
    def get_name(self) -> str:
        """Get model name."""
        return f"SimpleShort_{self.short_window}_{self.long_window}"


def main():
    """Run example backtests demonstrating short positions."""
    print("="*80)
    print("DEMONSTRATING SHORT POSITION FUNCTIONALITY")
    print("="*80)
    
    # Create synthetic price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Create trending data: up for first half, down for second half
    trend_up = np.linspace(100, 130, 100)
    trend_down = np.linspace(130, 100, 100)
    trend = np.concatenate([trend_up, trend_down])
    noise = np.random.randn(200) * 1
    prices = pd.Series(trend + noise, index=dates, name='close')
    
    print(f"\nTest Data: {len(prices)} periods")
    print(f"  First 100 periods: Uptrend (100 -> 130)")
    print(f"  Last 100 periods: Downtrend (130 -> 100)")
    
    # Create alpha model that goes long and short
    alpha_model = SimpleShortAlphaModel(short_window=10, long_window=30)
    
    print(f"\nAlpha Model: {alpha_model.get_name()}")
    print(f"  Strategy: Long when 10MA > 30MA, Short when 10MA < 30MA")
    
    # Run backtest
    print("\n" + "="*80)
    engine = BacktestEngine(prices, alpha_model, initial_capital=100000)
    results = engine.run(strategy_mode='alpha_only', transaction_cost=0.001)
    
    # Analyze positions
    positions = results['positions']
    long_count = (positions == 1).sum()
    short_count = (positions == -1).sum()
    flat_count = (positions == 0).sum()
    
    print("\n" + "="*80)
    print("POSITION ANALYSIS")
    print("="*80)
    print(f"Long Positions:  {long_count:>5} periods ({long_count/len(positions)*100:>5.1f}%)")
    print(f"Short Positions: {short_count:>5} periods ({short_count/len(positions)*100:>5.1f}%)")
    print(f"Flat Positions:  {flat_count:>5} periods ({flat_count/len(positions)*100:>5.1f}%)")
    print(f"Total Trades:    {results['num_trades']:>5}")
    print(f"Time in Market:  {results['time_in_market']*100:>5.1f}% (includes both long and short)")
    
    # Compare with long-only strategy
    print("\n" + "="*80)
    print("COMPARISON: LONG+SHORT vs LONG-ONLY")
    print("="*80)
    
    # Create long-only version
    class LongOnlyVersion(AlphaModel):
        """Same model but long-only (no shorts)."""
        def __init__(self):
            super().__init__(10, 30)
        def calculate_indicators(self, close):
            short_ma = close.rolling(window=10).mean()
            long_ma = close.rolling(window=30).mean()
            return short_ma, long_ma
        def generate_signals(self, close):
            short_ma, long_ma = self.calculate_indicators(close)
            signals = (short_ma > long_ma).astype(int)
            return signals
        def get_name(self):
            return "LongOnly_10_30"
    
    long_only_model = LongOnlyVersion()
    engine_long_only = BacktestEngine(prices, long_only_model, initial_capital=100000)
    results_long_only = engine_long_only.run(strategy_mode='alpha_only', transaction_cost=0.001)
    
    # Print comparison
    print(f"\nLONG+SHORT Strategy:")
    print(f"  Final Capital:   ${results['final_capital']:>12,.2f}")
    print(f"  Total Return:    {results['metrics']['total_return']*100:>11.2f}%")
    print(f"  Sharpe Ratio:    {results['metrics']['sharpe_ratio']:>11.2f}")
    print(f"  Max Drawdown:    {results['metrics']['max_drawdown']*100:>11.2f}%")
    print(f"  Number of Trades: {results['num_trades']:>10}")
    
    print(f"\nLONG-ONLY Strategy:")
    print(f"  Final Capital:   ${results_long_only['final_capital']:>12,.2f}")
    print(f"  Total Return:    {results_long_only['metrics']['total_return']*100:>11.2f}%")
    print(f"  Sharpe Ratio:    {results_long_only['metrics']['sharpe_ratio']:>11.2f}")
    print(f"  Max Drawdown:    {results_long_only['metrics']['max_drawdown']*100:>11.2f}%")
    print(f"  Number of Trades: {results_long_only['num_trades']:>10}")
    
    # Show advantage
    return_diff = (results['final_capital'] - results_long_only['final_capital'])
    print(f"\nADVANTAGE of Short Positions:")
    print(f"  Additional Profit: ${return_diff:>12,.2f}")
    print(f"  Return Improvement: {(results['metrics']['total_return'] - results_long_only['metrics']['total_return'])*100:>9.2f}%")
    
    if return_diff > 0:
        print(f"\n✅ SHORT POSITIONS PROVIDED BENEFIT in this trending market")
    else:
        print(f"\n⚠️  Long-only performed better in this scenario")
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. Backtest engine supports positions: 1 (long), 0 (flat), -1 (short)")
    print("  2. Short positions profit when prices fall")
    print("  3. Time in market counts both long and short positions")
    print("  4. Transaction costs apply to all position changes")
    print("  5. Short positions can improve returns in range-bound or falling markets")


if __name__ == '__main__':
    main()
