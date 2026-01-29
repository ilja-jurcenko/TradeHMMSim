"""
Example usage of Alpha Oracle strategy for ideal timing labels.

This demonstrates how to use the ZigZag-based oracle to identify
optimal entry/exit points based on price swings.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from portfolio import Portfolio
from backtest import BacktestEngine
from alpha_models import SMA
import matplotlib.pyplot as plt

def main():
    """Run example alpha oracle backtest."""
    
    # Load data
    print("Loading SPY data...")
    portfolio = Portfolio(['SPY'], '2020-01-01', '2023-12-31')
    portfolio.load_data()
    close = portfolio.get_close_prices('SPY')
    
    # Create alpha model (not actually used by oracle, but required by engine)
    alpha_model = SMA(short_window=10, long_window=30)
    
    # Create backtest engine
    engine = BacktestEngine(close, alpha_model)
    
    # Run alpha oracle strategy
    print("\n" + "="*80)
    print("Running Alpha Oracle Strategy (5% ZigZag)")
    print("="*80)
    results = engine.run(
        strategy_mode='alpha_oracle',
        rebalance_frequency=1,
        transaction_cost=0.001,
        enable_logging=False
    )
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total Return: {results['metrics']['total_return']*100:.2f}%")
    print(f"Annualized Return: {results['metrics']['annualized_return']*100:.2f}%")
    print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['metrics']['max_drawdown']*100:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Win Rate: {results['metrics']['win_rate']*100:.1f}%")
    
    # Plot equity curve
    print("\nPlotting equity curve...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot price and positions
    equity = results['equity_curve']
    positions = results['positions']
    
    # Price with buy/sell markers
    ax1.plot(close.index, close.values, label='SPY Price', linewidth=1.5, alpha=0.7)
    
    # Mark buy signals (position changes from 0 to 1)
    buy_signals = positions[(positions == 1) & (positions.shift(1, fill_value=0) == 0)]
    if len(buy_signals) > 0:
        ax1.scatter(buy_signals.index, close.loc[buy_signals.index], 
                   color='green', marker='^', s=100, label='BUY', zorder=5)
    
    # Mark sell signals (position changes from 1 to 0)
    sell_signals = positions[(positions == 0) & (positions.shift(1, fill_value=1) == 1)]
    if len(sell_signals) > 0:
        ax1.scatter(sell_signals.index, close.loc[sell_signals.index], 
                   color='red', marker='v', s=100, label='SELL', zorder=5)
    
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Alpha Oracle Strategy - Ideal Timing Labels (5% ZigZag)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Equity curve
    ax2.plot(equity.index, equity.values, label='Strategy Equity', linewidth=2)
    ax2.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Equity ($)')
    ax2.set_title('Equity Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('alpha_oracle_example.png', dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to alpha_oracle_example.png")
    
    # Compare with buy & hold
    bnh_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
    print(f"\nBuy & Hold Return: {bnh_return:.2f}%")
    print(f"Oracle Strategy Return: {results['metrics']['total_return']*100:.2f}%")
    print(f"Outperformance: {results['metrics']['total_return']*100 - bnh_return:.2f}%")

if __name__ == '__main__':
    main()
