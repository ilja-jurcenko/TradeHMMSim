"""
Example: Using Configuration Files Programmatically

This example demonstrates how to load and use configuration files
in your own scripts.
"""

from config_loader import ConfigLoader
from portfolio import Portfolio
from backtest import BacktestEngine
from alpha_models import EMA
from signal_filter import HMMRegimeFilter


def example_1_load_config():
    """Example 1: Load and print configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Load Configuration")
    print("="*60)
    
    # Load config
    config = ConfigLoader.load_config('config_optimal.json')
    
    # Print it
    ConfigLoader.print_config(config)
    
    # Extract specific parameter groups
    backtest_params = ConfigLoader.get_backtest_params(config)
    hmm_params = ConfigLoader.get_hmm_params(config)
    alpha_params = ConfigLoader.get_alpha_params(config)
    
    print("\nExtracted Parameters:")
    print("  Backtest:", backtest_params)
    print("  HMM:", hmm_params)
    print("  Alpha:", alpha_params)


def example_2_run_with_config():
    """Example 2: Run a backtest using configuration file."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Run Backtest with Config")
    print("="*60)
    
    # Load configuration
    config = ConfigLoader.load_config('config_optimal.json')
    
    # Extract parameters
    data_params = config.get('data', {})
    backtest_params = ConfigLoader.get_backtest_params(config)
    hmm_params = ConfigLoader.get_hmm_params(config)
    alpha_params = ConfigLoader.get_alpha_params(config)
    
    # Load data
    print(f"\nLoading data: {data_params['ticker']}")
    portfolio = Portfolio(
        [data_params['ticker']], 
        data_params['start_date'], 
        data_params['end_date']
    )
    portfolio.load_data()
    close = portfolio.get_close_prices(data_params['ticker'])
    
    # Create models with config parameters
    alpha_model = EMA(**alpha_params)
    hmm_filter = HMMRegimeFilter(**hmm_params)
    
    # Run backtest with config parameters
    engine = BacktestEngine(
        close, 
        alpha_model, 
        hmm_filter=hmm_filter,
        initial_capital=config['backtest']['initial_capital']
    )
    
    results = engine.run(**backtest_params)
    
    # Print results
    print("\nResults:")
    print(f"  Total Return: {results['metrics']['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {results['metrics']['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {results['metrics']['max_drawdown']*100:.2f}%")


def example_3_merge_configs():
    """Example 3: Merge configurations for custom testing."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Merge Configurations")
    print("="*60)
    
    # Load base config
    base_config = ConfigLoader.load_config('config_optimal.json')
    
    # Create override config (e.g., test with higher transaction cost)
    override_config = {
        'backtest': {
            'transaction_cost': 0.002  # Double the cost
        },
        'data': {
            'ticker': 'QQQ'  # Test on different ticker
        }
    }
    
    # Merge configs
    merged_config = ConfigLoader.merge_configs(base_config, override_config)
    
    print("\nBase Config:")
    print(f"  Transaction Cost: {base_config['backtest']['transaction_cost']}")
    print(f"  Ticker: {base_config['data']['ticker']}")
    
    print("\nOverride Config:")
    print(f"  Transaction Cost: {override_config['backtest']['transaction_cost']}")
    print(f"  Ticker: {override_config['data']['ticker']}")
    
    print("\nMerged Config:")
    print(f"  Transaction Cost: {merged_config['backtest']['transaction_cost']}")
    print(f"  Ticker: {merged_config['data']['ticker']}")
    print(f"  HMM train_window: {merged_config['hmm']['train_window']} (from base)")


def example_4_create_custom_config():
    """Example 4: Create and save a custom configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Create Custom Configuration")
    print("="*60)
    
    # Start with default config
    config = ConfigLoader.load_config('config_default.json')
    
    # Modify for high-frequency trading
    config['backtest']['transaction_cost'] = 0.0005  # Lower cost
    config['backtest']['rebalance_frequency'] = 1     # Daily rebalance
    config['hmm']['train_window'] = 126               # 6 months
    config['hmm']['refit_every'] = 21                 # Monthly refit
    config['alpha_model']['short_window'] = 5         # Shorter window
    config['alpha_model']['long_window'] = 15         # Shorter window
    config['data']['ticker'] = 'SPY'
    config['output']['output_dir'] = 'results/high_freq_test'
    
    # Save custom config
    output_path = 'config_high_freq.json'
    ConfigLoader.save_config(config, output_path)
    
    print(f"Custom configuration created: {output_path}")
    print("\nKey parameters:")
    print(f"  Transaction Cost: {config['backtest']['transaction_cost']*100:.3f}%")
    print(f"  HMM train_window: {config['hmm']['train_window']} days")
    print(f"  Alpha windows: {config['alpha_model']['short_window']}/{config['alpha_model']['long_window']}")


def example_5_test_multiple_configs():
    """Example 5: Test and compare multiple configurations."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Compare Multiple Configurations")
    print("="*60)
    
    configs = [
        ('config_default.json', 'Baseline'),
        ('config_optimal.json', 'Optimal'),
        ('config_accurate.json', 'Accurate')
    ]
    
    results_summary = []
    
    for config_file, config_name in configs:
        print(f"\nTesting {config_name}...")
        
        # Load config
        config = ConfigLoader.load_config(config_file)
        
        # Get HMM parameters
        hmm = config.get('hmm', {})
        train_window = hmm.get('train_window')
        refit_every = hmm.get('refit_every')
        
        results_summary.append({
            'Configuration': config_name,
            'train_window': train_window,
            'refit_every': refit_every,
            'transaction_cost': config['backtest']['transaction_cost']
        })
    
    print("\n" + "="*60)
    print("Configuration Summary:")
    print("="*60)
    import pandas as pd
    df = pd.DataFrame(results_summary)
    print(df.to_string(index=False))
    
    print("\nTo run full comparison, use:")
    print("  python examples/example_config_testing.py")


if __name__ == '__main__':
    # Run all examples
    example_1_load_config()
    example_2_run_with_config()
    example_3_merge_configs()
    example_4_create_custom_config()
    example_5_test_multiple_configs()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETE")
    print("="*60)
