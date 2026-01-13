"""
Analyze Bollinger Bands mean-reversion strategy performance.
Runs backtest on one-year period and outputs primary financial results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from portfolio import Portfolio
from backtest import BacktestEngine
from alpha_models import BollingerBands
from statistics import Statistics
import matplotlib.pyplot as plt


def analyze_bollinger_bands(ticker: str = 'SPY',
                            period: int = 20,
                            std_dev: int = 2,
                            start_date: str = None,
                            end_date: str = None,
                            transaction_cost: float = 0.001,
                            show_plot: bool = True):
    """
    Analyze Bollinger Bands strategy performance.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol to analyze
    period : int
        Bollinger Bands period (default: 20)
    std_dev : int
        Number of standard deviations for bands (default: 2)
    start_date : str
        Start date (default: one year ago from today)
    end_date : str
        End date (default: today)
    transaction_cost : float
        Transaction cost per trade
    show_plot : bool
        Whether to display charts
    """
    
    # Set default dates to one year period
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_dt = datetime.now() - timedelta(days=365)
        start_date = start_dt.strftime('%Y-%m-%d')
    
    print("="*80)
    print("BOLLINGER BANDS STRATEGY ANALYSIS")
    print("="*80)
    print(f"\nTicker: {ticker}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Bollinger Bands Parameters: Period={period}, Std Dev={std_dev}")
    print(f"Transaction Cost: {transaction_cost*100:.3f}%")
    print("\n" + "="*80)
    
    # Load data
    print("\nLoading data...")
    portfolio = Portfolio([ticker], start_date, end_date)
    portfolio.load_data()
    
    close = portfolio.get_close_prices(ticker)
    print(f"Loaded {len(close)} trading days")
    
    # Create Bollinger Bands model
    bb_model = BollingerBands(short_window=period, long_window=std_dev)
    
    # Get Bollinger Bands
    upper_band, middle_band, lower_band = bb_model.get_bands(close)
    
    # Run backtest
    print("\nRunning Bollinger Bands backtest...")
    engine = BacktestEngine(close, alpha_model=bb_model)
    results = engine.run(
        strategy_mode='alpha_only',
        rebalance_frequency=1,
        transaction_cost=transaction_cost
    )
    
    # Calculate benchmark (Buy & Hold)
    print("Calculating Buy & Hold benchmark...")
    returns = close.pct_change().fillna(0)
    benchmark_metrics = Statistics.calculate_all_metrics(returns)
    
    # Print Results
    print("\n" + "="*80)
    print("PERFORMANCE RESULTS")
    print("="*80)
    
    print("\n📊 BOLLINGER BANDS STRATEGY")
    print("-" * 40)
    metrics = results['metrics']
    print(f"Total Return:        {metrics['total_return']*100:>10.2f}%")
    print(f"Annualized Return:   {metrics['annualized_return']*100:>10.2f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
    print(f"Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
    print(f"Max Drawdown:        {metrics['max_drawdown']*100:>10.2f}%")
    print(f"Profit Factor:       {metrics['profit_factor']:>10.2f}")
    print(f"Win Rate:            {metrics['win_rate']*100:>10.2f}%")
    print(f"Number of Trades:    {results['num_trades']:>10}")
    print(f"Time in Market:      {results['time_in_market']*100:>10.2f}%")
    
    print("\n📈 BUY & HOLD BENCHMARK")
    print("-" * 40)
    print(f"Total Return:        {benchmark_metrics['total_return']*100:>10.2f}%")
    print(f"Annualized Return:   {benchmark_metrics['annualized_return']*100:>10.2f}%")
    print(f"Sharpe Ratio:        {benchmark_metrics['sharpe_ratio']:>10.2f}")
    print(f"Sortino Ratio:       {benchmark_metrics['sortino_ratio']:>10.2f}")
    print(f"Max Drawdown:        {benchmark_metrics['max_drawdown']*100:>10.2f}%")
    
    print("\n📊 STRATEGY vs BENCHMARK")
    print("-" * 40)
    outperformance = (metrics['total_return'] - benchmark_metrics['total_return']) * 100
    sharpe_diff = metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio']
    dd_improvement = (benchmark_metrics['max_drawdown'] - metrics['max_drawdown']) * 100
    
    print(f"Return Difference:   {outperformance:>10.2f}%  {'✅' if outperformance > 0 else '❌'}")
    print(f"Sharpe Difference:   {sharpe_diff:>10.2f}  {'✅' if sharpe_diff > 0 else '❌'}")
    print(f"Drawdown Reduction:  {dd_improvement:>10.2f}%  {'✅' if dd_improvement > 0 else '❌'}")
    
    # Trading Statistics
    print("\n💼 TRADING STATISTICS")
    print("-" * 40)
    positions = results['positions']
    trades = (positions.diff() != 0).sum()
    long_periods = (positions == 1).sum()
    flat_periods = (positions == 0).sum()
    
    print(f"Total Signals:       {trades:>10}")
    print(f"Long Periods:        {long_periods:>10} ({long_periods/len(positions)*100:.1f}%)")
    print(f"Flat Periods:        {flat_periods:>10} ({flat_periods/len(positions)*100:.1f}%)")
    
    # Band Statistics
    print("\n📉 BOLLINGER BANDS STATISTICS")
    print("-" * 40)
    valid_idx = ~middle_band.isna()
    band_width = (upper_band[valid_idx] - lower_band[valid_idx]) / middle_band[valid_idx] * 100
    
    print(f"Avg Band Width:      {band_width.mean():>10.2f}%")
    print(f"Max Band Width:      {band_width.max():>10.2f}%")
    print(f"Min Band Width:      {band_width.min():>10.2f}%")
    
    # Count touches
    touches_lower = (close < lower_band).sum()
    touches_upper = (close > upper_band).sum()
    
    print(f"Lower Band Touches:  {touches_lower:>10}")
    print(f"Upper Band Touches:  {touches_upper:>10}")
    
    # Plot if requested
    if show_plot:
        print("\n📈 Generating charts...")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: Price with Bollinger Bands
        ax1.plot(close.index, close.values, label='Price', linewidth=1.5, color='black')
        ax1.plot(upper_band.index, upper_band.values, '--', label='Upper Band', color='red', alpha=0.7)
        ax1.plot(middle_band.index, middle_band.values, '--', label='Middle Band', color='blue', alpha=0.7)
        ax1.plot(lower_band.index, lower_band.values, '--', label='Lower Band', color='green', alpha=0.7)
        ax1.fill_between(close.index, lower_band.values, upper_band.values, alpha=0.1, color='gray')
        
        # Mark entry/exit points
        entries = positions.diff() == 1
        exits = positions.diff() == -1
        ax1.scatter(close.index[entries], close[entries], marker='^', color='green', s=100, 
                   label='Buy Signal', zorder=5)
        ax1.scatter(close.index[exits], close[exits], marker='v', color='red', s=100, 
                   label='Sell Signal', zorder=5)
        
        ax1.set_title(f'{ticker} - Bollinger Bands Strategy ({period} period, {std_dev} std)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=11)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Equity Curves Comparison
        equity_bb = results['equity_curve']
        equity_bh = (1 + returns).cumprod() * 100000
        
        ax2.plot(equity_bb.index, equity_bb.values, label='Bollinger Bands', linewidth=2, color='blue')
        ax2.plot(equity_bh.index, equity_bh.values, label='Buy & Hold', linewidth=2, color='orange', alpha=0.7)
        ax2.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax2.set_title('Equity Curve Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        running_max = equity_bb.expanding().max()
        drawdown = (equity_bb - running_max) / running_max * 100
        
        ax3.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
        ax3.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
        ax3.set_title('Strategy Drawdown', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = f'bollinger_bands_analysis_{ticker}_{start_date}_{end_date}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Chart saved to: {output_file}")
        
        plt.show()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return results, benchmark_metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Bollinger Bands strategy performance')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--period', type=int, default=20, help='Bollinger Bands period (default: 20)')
    parser.add_argument('--std', type=int, default=2, help='Standard deviations (default: 2)')
    parser.add_argument('--start', type=str, default=None, help='Start date YYYY-MM-DD (default: 1 year ago)')
    parser.add_argument('--end', type=str, default=None, help='End date YYYY-MM-DD (default: today)')
    parser.add_argument('--cost', type=float, default=0.001, help='Transaction cost (default: 0.001)')
    parser.add_argument('--no-plot', action='store_true', help='Disable chart display')
    
    args = parser.parse_args()
    
    analyze_bollinger_bands(
        ticker=args.ticker,
        period=args.period,
        std_dev=args.std,
        start_date=args.start,
        end_date=args.end,
        transaction_cost=args.cost,
        show_plot=not args.no_plot
    )
