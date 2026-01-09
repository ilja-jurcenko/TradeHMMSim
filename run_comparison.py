"""
Main script to compare the impact of HMM model with AlphaModels.
"""

import pandas as pd
import numpy as np
from portfolio import Portfolio
from backtest import BacktestEngine
from AlphaModels import SMA, EMA, WMA, HMA, KAMA, TEMA, ZLEMA
from SignalFilter import HMMRegimeFilter
from statistics import Statistics
from plotter import BacktestPlotter


def run_comparison(ticker: str = 'SPY', 
                   start_date: str = '2018-01-01',
                   end_date: str = '2024-12-31',
                   alpha_models: list = None,
                   short_window: int = 10,
                   long_window: int = 30,
                   rebalance_frequency: int = 1,
                   transaction_cost: float = 0.001):
    """
    Run comprehensive comparison of AlphaModels with and without HMM filtering.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol to test
    start_date : str
        Start date for backtest
    end_date : str
        End date for backtest
    alpha_models : list
        List of alpha model classes to test (default: all)
    short_window : int
        Short MA window
    long_window : int
        Long MA window
    rebalance_frequency : int
        Rebalancing frequency in days
    transaction_cost : float
        Transaction cost per trade
        
    Returns:
    --------
    pd.DataFrame
        Comparison results
    """
    print("\n" + "="*80)
    print("BACKTESTING FRAMEWORK - ALPHA MODELS VS HMM COMPARISON")
    print("="*80)
    
    # Default models
    if alpha_models is None:
        alpha_models = [SMA, EMA, WMA, HMA, KAMA, TEMA, ZLEMA]
    
    # Load data
    print(f"\nLoading data for {ticker}...")
    portfolio = Portfolio([ticker], start_date, end_date)
    portfolio.load_data()
    portfolio.summary()
    
    close = portfolio.get_close_prices(ticker)
    
    # Initialize HMM filter
    print("\nInitializing HMM regime filter...")
    hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)
    
    # Storage for results
    results_list = []
    
    # Test each alpha model with different strategies
    for model_class in alpha_models:
        model_name = model_class.__name__
        print(f"\n{'='*80}")
        print(f"TESTING: {model_name}")
        print(f"{'='*80}")
        
        model = model_class(short_window=short_window, long_window=long_window)
        
        # Strategy 1: Alpha only
        print(f"\n[1/4] Running {model_name} - Alpha Only...")
        engine_alpha = BacktestEngine(close, model)
        results_alpha = engine_alpha.run(
            strategy_mode='alpha_only',
            rebalance_frequency=rebalance_frequency,
            transaction_cost=transaction_cost
        )
        
        results_list.append({
            'Model': model_name,
            'Strategy': 'Alpha Only',
            'Total Return (%)': results_alpha['metrics']['total_return'] * 100,
            'Annual Return (%)': results_alpha['metrics']['annualized_return'] * 100,
            'Sharpe Ratio': results_alpha['metrics']['sharpe_ratio'],
            'Sortino Ratio': results_alpha['metrics']['sortino_ratio'],
            'Max Drawdown (%)': results_alpha['metrics']['max_drawdown'] * 100,
            'Profit Factor': results_alpha['metrics']['profit_factor'],
            'Win Rate (%)': results_alpha['metrics']['win_rate'] * 100,
            'Num Trades': results_alpha['num_trades'],
            'Time in Market (%)': results_alpha['time_in_market'] * 100
        })
        
        # Strategy 2: HMM only
        print(f"\n[2/4] Running {model_name} - HMM Only...")
        hmm_filter_new = HMMRegimeFilter(n_states=3, random_state=42)
        engine_hmm = BacktestEngine(close, model, hmm_filter=hmm_filter_new)
        results_hmm = engine_hmm.run(
            strategy_mode='hmm_only',
            walk_forward=True,
            train_window=504,
            refit_every=21,
            rebalance_frequency=rebalance_frequency,
            transaction_cost=transaction_cost
        )
        
        results_list.append({
            'Model': model_name,
            'Strategy': 'HMM Only',
            'Total Return (%)': results_hmm['metrics']['total_return'] * 100,
            'Annual Return (%)': results_hmm['metrics']['annualized_return'] * 100,
            'Sharpe Ratio': results_hmm['metrics']['sharpe_ratio'],
            'Sortino Ratio': results_hmm['metrics']['sortino_ratio'],
            'Max Drawdown (%)': results_hmm['metrics']['max_drawdown'] * 100,
            'Profit Factor': results_hmm['metrics']['profit_factor'],
            'Win Rate (%)': results_hmm['metrics']['win_rate'] * 100,
            'Num Trades': results_hmm['num_trades'],
            'Time in Market (%)': results_hmm['time_in_market'] * 100
        })
        
        # Strategy 3: Alpha + HMM Filter
        print(f"\n[3/4] Running {model_name} - Alpha + HMM Filter...")
        hmm_filter_new = HMMRegimeFilter(n_states=3, random_state=42)
        engine_filter = BacktestEngine(close, model, hmm_filter=hmm_filter_new)
        results_filter = engine_filter.run(
            strategy_mode='alpha_hmm_filter',
            walk_forward=True,
            train_window=504,
            refit_every=21,
            rebalance_frequency=rebalance_frequency,
            transaction_cost=transaction_cost
        )
        
        results_list.append({
            'Model': model_name,
            'Strategy': 'Alpha + HMM Filter',
            'Total Return (%)': results_filter['metrics']['total_return'] * 100,
            'Annual Return (%)': results_filter['metrics']['annualized_return'] * 100,
            'Sharpe Ratio': results_filter['metrics']['sharpe_ratio'],
            'Sortino Ratio': results_filter['metrics']['sortino_ratio'],
            'Max Drawdown (%)': results_filter['metrics']['max_drawdown'] * 100,
            'Profit Factor': results_filter['metrics']['profit_factor'],
            'Win Rate (%)': results_filter['metrics']['win_rate'] * 100,
            'Num Trades': results_filter['num_trades'],
            'Time in Market (%)': results_filter['time_in_market'] * 100
        })
        
        # Strategy 4: Alpha + HMM Combine
        print(f"\n[4/4] Running {model_name} - Alpha + HMM Combine...")
        hmm_filter_new = HMMRegimeFilter(n_states=3, random_state=42)
        engine_combine = BacktestEngine(close, model, hmm_filter=hmm_filter_new)
        results_combine = engine_combine.run(
            strategy_mode='alpha_hmm_combine',
            walk_forward=True,
            train_window=504,
            refit_every=21,
            rebalance_frequency=rebalance_frequency,
            transaction_cost=transaction_cost
        )
        
        results_list.append({
            'Model': model_name,
            'Strategy': 'Alpha + HMM Combine',
            'Total Return (%)': results_combine['metrics']['total_return'] * 100,
            'Annual Return (%)': results_combine['metrics']['annualized_return'] * 100,
            'Sharpe Ratio': results_combine['metrics']['sharpe_ratio'],
            'Sortino Ratio': results_combine['metrics']['sortino_ratio'],
            'Max Drawdown (%)': results_combine['metrics']['max_drawdown'] * 100,
            'Profit Factor': results_combine['metrics']['profit_factor'],
            'Win Rate (%)': results_combine['metrics']['win_rate'] * 100,
            'Num Trades': results_combine['num_trades'],
            'Time in Market (%)': results_combine['time_in_market'] * 100
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Calculate benchmark (Buy & Hold)
    print("\n" + "="*80)
    print("CALCULATING BUY & HOLD BENCHMARK")
    print("="*80)
    returns = close.pct_change().fillna(0)
    benchmark_metrics = Statistics.calculate_all_metrics(returns)
    
    print("\nBuy & Hold Performance:")
    print(f"  Total Return: {benchmark_metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {benchmark_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {benchmark_metrics['max_drawdown']*100:.2f}%")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print("\nTop 10 Strategies by Total Return:")
    print(results_df.sort_values('Total Return (%)', ascending=False).head(10).to_string(index=False))
    
    print("\nTop 10 Strategies by Sharpe Ratio:")
    print(results_df.sort_values('Sharpe Ratio', ascending=False).head(10)[
        ['Model', 'Strategy', 'Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
    ].to_string(index=False))
    
    # Calculate average performance by strategy type
    print("\n" + "="*80)
    print("AVERAGE PERFORMANCE BY STRATEGY TYPE")
    print("="*80)
    avg_by_strategy = results_df.groupby('Strategy').agg({
        'Total Return (%)': 'mean',
        'Sharpe Ratio': 'mean',
        'Max Drawdown (%)': 'mean',
        'Num Trades': 'mean',
        'Time in Market (%)': 'mean'
    }).round(2)
    print(avg_by_strategy.to_string())
    
    # Calculate HMM impact
    print("\n" + "="*80)
    print("HMM IMPACT ANALYSIS")
    print("="*80)
    
    for model_name in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model_name]
        
        alpha_only = model_data[model_data['Strategy'] == 'Alpha Only'].iloc[0]
        hmm_only = model_data[model_data['Strategy'] == 'HMM Only'].iloc[0]
        alpha_filter = model_data[model_data['Strategy'] == 'Alpha + HMM Filter'].iloc[0]
        alpha_combine = model_data[model_data['Strategy'] == 'Alpha + HMM Combine'].iloc[0]
        
        print(f"\n{model_name}:")
        print(f"  Alpha Only Return: {alpha_only['Total Return (%)']:.2f}%")
        print(f"  HMM Only Return: {hmm_only['Total Return (%)']:.2f}%")
        print(f"  Alpha + Filter Return: {alpha_filter['Total Return (%)']:.2f}%")
        print(f"  Alpha + Combine Return: {alpha_combine['Total Return (%)']:.2f}%")
        print(f"  HMM Filter Impact: {alpha_filter['Total Return (%)'] - alpha_only['Total Return (%)']:.2f}%")
        print(f"  HMM Combine Impact: {alpha_combine['Total Return (%)'] - alpha_only['Total Return (%)']:.2f}%")
    
    # Save results
    output_file = f'backtest_comparison_{ticker}_{start_date}_{end_date}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    return results_df


if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'SPY'
    start_date = sys.argv[2] if len(sys.argv) > 2 else '2018-01-01'
    end_date = sys.argv[3] if len(sys.argv) > 3 else '2024-12-31'
    show_plots = '--plot' in sys.argv or '-p' in sys.argv
    
    # Run comparison
    results = run_comparison(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        short_window=10,
        long_window=30,
        rebalance_frequency=1,
        transaction_cost=0.001
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Optional: Generate plots for best strategies
    if show_plots:
        print("\nGenerating plots for top strategies...")
        # Get top 3 strategies by Sharpe ratio
        top_strategies = results.nlargest(3, 'Sharpe Ratio')
        print("\nPlotting top 3 strategies:")
        print(top_strategies[['Model', 'Strategy', 'Sharpe Ratio', 'Total Return (%)']])

