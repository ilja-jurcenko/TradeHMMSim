"""
Simple example demonstrating basic usage of the backtesting framework.
"""

from portfolio import Portfolio
from backtest import BacktestEngine
from AlphaModels import SMA, EMA
from SignalFilter import HMMRegimeFilter
from plotter import BacktestPlotter


def simple_example():
    """Run a simple backtest example."""
    
    print("="*80)
    print("SIMPLE BACKTEST EXAMPLE")
    print("="*80)
    
    # Step 1: Load data
    print("\n1. Loading SPY data...")
    portfolio = Portfolio(['SPY'], '2020-01-01', '2024-12-31')
    portfolio.load_data()
    close = portfolio.get_close_prices('SPY')
    print(f"   Loaded {len(close)} days of data")
    
    # Step 2: Create alpha model
    print("\n2. Creating SMA(10, 30) alpha model...")
    alpha_model = SMA(short_window=10, long_window=30)
    
    # Step 3: Run simple backtest (alpha only)
    print("\n3. Running backtest - Alpha Only...")
    engine = BacktestEngine(close, alpha_model, initial_capital=100000)
    results_alpha = engine.run(strategy_mode='alpha_only')
    engine.print_results(include_benchmark=True)
    
    # Step 4: Add HMM filtering
    print("\n4. Running backtest - Alpha + HMM Override...")
    hmm_filter = HMMRegimeFilter(n_states=3, random_state=42)
    engine_hmm = BacktestEngine(close, alpha_model, hmm_filter=hmm_filter, initial_capital=100000)
    results_hmm = engine_hmm.run(
        strategy_mode='alpha_hmm_override',
        walk_forward=True,
        train_window=252,  # 1 year
        refit_every=21     # Monthly
    )
    engine_hmm.print_results(include_benchmark=True)
    
    # Step 5: Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Alpha Only Total Return:     {results_alpha['metrics']['total_return']*100:.2f}%")
    print(f"Alpha + HMM Total Return:    {results_hmm['metrics']['total_return']*100:.2f}%")
    print(f"HMM Improvement:             {(results_hmm['metrics']['total_return'] - results_alpha['metrics']['total_return'])*100:.2f}%")
    print()
    print(f"Alpha Only Sharpe Ratio:     {results_alpha['metrics']['sharpe_ratio']:.2f}")
    print(f"Alpha + HMM Sharpe Ratio:    {results_hmm['metrics']['sharpe_ratio']:.2f}")
    print(f"Sharpe Improvement:          {results_hmm['metrics']['sharpe_ratio'] - results_alpha['metrics']['sharpe_ratio']:.2f}")
    print("="*80)
    
    # Step 6: Plot results
    print("\n5. Generating plots...")
    BacktestPlotter.plot_results(results_hmm, close)
    BacktestPlotter.plot_comparison(
        [results_alpha, results_hmm],
        ['Alpha Only', 'Alpha + HMM Override']
    )


def compare_alpha_models():
    """Compare different alpha models."""
    
    print("\n" + "="*80)
    print("COMPARING DIFFERENT ALPHA MODELS")
    print("="*80)
    
    # Load data
    portfolio = Portfolio(['SPY'], '2020-01-01', '2024-12-31')
    portfolio.load_data()
    close = portfolio.get_close_prices('SPY')
    
    # Test different models
    models = [
        ('SMA', SMA(10, 30)),
        ('EMA', EMA(10, 30))
    ]
    
    results = []
    for name, model in models:
        print(f"\nTesting {name}...")
        engine = BacktestEngine(close, model)
        result = engine.run(strategy_mode='alpha_only')
        results.append((name, result['metrics']['total_return'], result['metrics']['sharpe_ratio']))
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"{'Model':<10} {'Total Return':<15} {'Sharpe Ratio':<15}")
    print("-"*80)
    for name, ret, sharpe in results:
        print(f"{name:<10} {ret*100:>13.2f}%  {sharpe:>13.2f}")
    print("="*80)


if __name__ == '__main__':
    # Run simple example
    simple_example()
    
    # Run comparison
    compare_alpha_models()
    
    print("\n✓ Example complete!")
