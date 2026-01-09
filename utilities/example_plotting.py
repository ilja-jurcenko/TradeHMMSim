"""
Example script demonstrating visualization capabilities of the backtesting framework.
"""

from portfolio import Portfolio
from backtest import BacktestEngine
from AlphaModels import SMA, EMA, KAMA
from SignalFilter import HMMRegimeFilter
from plotter import BacktestPlotter


def example_single_backtest_plot():
    """
    Example 1: Plot a single backtest result with HMM filtering.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Backtest with Detailed Visualization")
    print("="*80)
    
    # Load data
    print("\nLoading SPY data...")
    portfolio = Portfolio(['SPY'], '2020-01-01', '2024-12-31')
    portfolio.load_data()
    close = portfolio.get_close_prices('SPY')
    
    # Create alpha model and HMM filter
    print("Setting up SMA(10, 30) with HMM regime filtering...")
    alpha_model = SMA(short_window=10, long_window=30)
    hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)
    
    # Run backtest
    print("Running backtest...")
    engine = BacktestEngine(close, alpha_model, hmm_filter=hmm_filter, initial_capital=100000)
    results = engine.run(
        strategy_mode='alpha_hmm_combine',
        walk_forward=True,
        train_window=252,
        refit_every=21
    )
    
    # Print results
    engine.print_results(include_benchmark=True)
    
    # Generate comprehensive visualization
    print("\nGenerating plots...")
    BacktestPlotter.plot_results(results, close)


def example_strategy_comparison():
    """
    Example 2: Compare multiple strategies side-by-side.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Strategy Comparison Visualization")
    print("="*80)
    
    # Load data
    print("\nLoading SPY data...")
    portfolio = Portfolio(['SPY'], '2020-01-01', '2024-12-31')
    portfolio.load_data()
    close = portfolio.get_close_prices('SPY')
    
    # Create alpha model
    alpha_model = SMA(short_window=10, long_window=30)
    hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)
    
    # Run multiple strategies
    strategies = [
        ('Alpha Only', 'alpha_only', False),
        ('HMM Filter', 'alpha_hmm_filter', True),
        ('HMM Combine', 'alpha_hmm_combine', True),
        ('HMM Only', 'hmm_only', True)
    ]
    
    results_list = []
    labels = []
    
    for label, mode, use_hmm in strategies:
        print(f"\nRunning {label}...")
        
        if use_hmm:
            engine = BacktestEngine(close, alpha_model, hmm_filter=hmm_filter)
            result = engine.run(
                strategy_mode=mode,
                walk_forward=True,
                train_window=252,
                refit_every=21
            )
        else:
            engine = BacktestEngine(close, alpha_model)
            result = engine.run(strategy_mode=mode)
        
        results_list.append(result)
        labels.append(label)
        
        print(f"  Total Return: {result['metrics']['total_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {result['metrics']['sharpe_ratio']:.2f}")
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    BacktestPlotter.plot_comparison(results_list, labels)
    BacktestPlotter.plot_metrics_comparison(results_list, labels)


def example_model_comparison():
    """
    Example 3: Compare different alpha models.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Alpha Model Comparison")
    print("="*80)
    
    # Load data
    print("\nLoading SPY data...")
    portfolio = Portfolio(['SPY'], '2020-01-01', '2024-12-31')
    portfolio.load_data()
    close = portfolio.get_close_prices('SPY')
    
    # Test different models
    models = [
        ('SMA', SMA(10, 30)),
        ('EMA', EMA(10, 30)),
        ('KAMA', KAMA(10, 30))
    ]
    
    results_list = []
    labels = []
    
    for name, model in models:
        print(f"\nRunning {name}...")
        engine = BacktestEngine(close, model)
        result = engine.run(strategy_mode='alpha_only')
        
        results_list.append(result)
        labels.append(name)
        
        print(f"  Total Return: {result['metrics']['total_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {result['metrics']['sharpe_ratio']:.2f}")
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    BacktestPlotter.plot_comparison(results_list, labels)
    BacktestPlotter.plot_metrics_comparison(results_list, labels)


def example_regime_analysis():
    """
    Example 4: Visualize HMM regime detection.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: HMM Regime Analysis Visualization")
    print("="*80)
    
    # Load data
    print("\nLoading SPY data...")
    portfolio = Portfolio(['SPY'], '2018-01-01', '2024-12-31')
    portfolio.load_data()
    close = portfolio.get_close_prices('SPY')
    
    # Train HMM and get regime data
    print("Training HMM regime filter...")
    hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)
    
    # Prepare features
    import pandas as pd
    import numpy as np
    returns = close.pct_change()
    volatility = returns.rolling(20).std()
    features = pd.DataFrame({
        'returns': returns,
        'volatility': volatility
    }).dropna()
    
    # Fit HMM
    hmm_filter.fit(features.values)
    
    # Get regime probabilities and states
    probs = hmm_filter.filtered_state_probs(features.values)
    probs_df = pd.DataFrame(probs, index=features.index, columns=range(3))
    regime = pd.Series(probs.argmax(axis=1), index=features.index)
    
    # Detect switches
    switches = regime[regime.ne(regime.shift(1))].dropna()
    
    print(f"Detected {len(switches)} regime switches")
    
    # Generate regime analysis plots
    print("\nGenerating regime analysis plots...")
    BacktestPlotter.plot_regime_analysis(probs_df, regime, close, switches)


if __name__ == '__main__':
    import sys
    
    # Determine which example to run
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
    else:
        print("\nAvailable examples:")
        print("  1: Single backtest with detailed visualization")
        print("  2: Strategy comparison (Alpha Only vs HMM strategies)")
        print("  3: Alpha model comparison (SMA vs EMA vs KAMA)")
        print("  4: HMM regime analysis visualization")
        print("\nUsage: python example_plotting.py [example_number]")
        print("Or run without arguments to run all examples")
        example_num = 'all'
    
    if example_num in ['1', 'all']:
        example_single_backtest_plot()
    
    if example_num in ['2', 'all']:
        example_strategy_comparison()
    
    if example_num in ['3', 'all']:
        example_model_comparison()
    
    if example_num in ['4', 'all']:
        example_regime_analysis()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80)
