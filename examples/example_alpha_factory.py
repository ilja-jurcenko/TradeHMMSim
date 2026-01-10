"""
Example: Using Alpha Model Factory with JSON Configuration
Demonstrates creating and testing different alpha models from config.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha_model_factory import AlphaModelFactory
from backtest import BacktestEngine
from config_loader import ConfigLoader


def example_1_basic_factory_usage():
    """Example 1: Basic factory usage."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Factory Usage")
    print("="*60)
    
    # Create from config dictionary
    alpha_config = {
        'type': 'SMA',
        'parameters': {
            'short_window': 50,
            'long_window': 200
        }
    }
    
    model = AlphaModelFactory.create_from_config(alpha_config)
    print(f"Created model: {model.get_name()}")
    print(f"Parameters: {model.get_parameters()}")
    
    # Create from type and parameters
    model2 = AlphaModelFactory.create_from_type('EMA', 12, 26)
    print(f"\nCreated model: {model2.get_name()}")
    print(f"Parameters: {model2.get_parameters()}")
    
    # Get available models
    available = AlphaModelFactory.get_available_models()
    print(f"\nAvailable models: {', '.join(available)}")


def example_2_backtest_with_config():
    """Example 2: Running backtest with alpha config."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Backtest with Alpha Config")
    print("="*60)
    
    # Create synthetic price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    prices = 100 * (1 + np.random.randn(500).cumsum() * 0.01)
    close = pd.Series(prices, index=dates, name='close')
    
    # Alpha config
    alpha_config = {
        'type': 'EMA',
        'parameters': {
            'short_window': 10,
            'long_window': 30
        }
    }
    
    # Create and run backtest
    engine = BacktestEngine(close=close, alpha_config=alpha_config)
    results = engine.run(strategy_mode='alpha_only', transaction_cost=0.001)
    
    print(f"\nModel: {results['alpha_model']}")
    print(f"Total Return: {results['metrics']['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results['metrics']['max_drawdown']*100:.2f}%")


def example_3_test_all_models():
    """Example 3: Test all alpha models."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Test All Alpha Models")
    print("="*60)
    
    # Create synthetic price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    prices = 100 * (1 + np.random.randn(500).cumsum() * 0.01)
    close = pd.Series(prices, index=dates, name='close')
    
    # Test all models
    models = AlphaModelFactory.get_available_models()
    results = {}
    
    for model_type in models:
        alpha_config = {
            'type': model_type,
            'parameters': {
                'short_window': 10,
                'long_window': 30
            }
        }
        
        engine = BacktestEngine(close=close, alpha_config=alpha_config)
        result = engine.run(strategy_mode='alpha_only', transaction_cost=0.001)
        results[model_type] = result['metrics']
    
    # Display results
    print("\nResults Summary:")
    print(f"{'Model':<8} {'Return':<10} {'Sharpe':<8} {'Max DD':<10}")
    print("-" * 40)
    
    for model_type, metrics in results.items():
        print(f"{model_type:<8} "
              f"{metrics['total_return']*100:>6.2f}%   "
              f"{metrics['sharpe_ratio']:>6.3f}  "
              f"{metrics['max_drawdown']*100:>6.2f}%")


def example_4_load_from_json_file():
    """Example 4: Load config from JSON file."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Load Config from JSON File")
    print("="*60)
    
    # Create synthetic price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    prices = 100 * (1 + np.random.randn(500).cumsum() * 0.01)
    close = pd.Series(prices, index=dates, name='close')
    
    try:
        # Load config file
        config = ConfigLoader.load_config('config_optimal.json')
        print(f"Loaded config: config_optimal.json")
        print(f"Alpha model type: {config['alpha_model']['type']}")
        print(f"Parameters: {config['alpha_model']['parameters']}")
        
        # Create engine from config
        engine = BacktestEngine.from_config(close, config)
        results = engine.run(strategy_mode='alpha_only')
        
        print(f"\nTotal Return: {results['metrics']['total_return']*100:.2f}%")
        print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.3f}")
        
    except FileNotFoundError:
        print("Config file not found. Make sure you're running from project root.")


def example_5_compare_configurations():
    """Example 5: Compare different configurations."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Compare Configurations")
    print("="*60)
    
    # Create synthetic price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    prices = 100 * (1 + np.random.randn(500).cumsum() * 0.01)
    close = pd.Series(prices, index=dates, name='close')
    
    # Different configurations to test
    configs = [
        {
            'name': 'Fast MA (10/30)',
            'type': 'SMA',
            'short_window': 10,
            'long_window': 30
        },
        {
            'name': 'Medium MA (20/50)',
            'type': 'SMA',
            'short_window': 20,
            'long_window': 50
        },
        {
            'name': 'Slow MA (50/200)',
            'type': 'SMA',
            'short_window': 50,
            'long_window': 200
        }
    ]
    
    print("\nComparing different window configurations:")
    print(f"{'Configuration':<20} {'Return':<10} {'Sharpe':<8}")
    print("-" * 40)
    
    for cfg in configs:
        alpha_config = {
            'type': cfg['type'],
            'parameters': {
                'short_window': cfg['short_window'],
                'long_window': cfg['long_window']
            }
        }
        
        engine = BacktestEngine(close=close, alpha_config=alpha_config)
        result = engine.run(strategy_mode='alpha_only', transaction_cost=0.001)
        
        print(f"{cfg['name']:<20} "
              f"{result['metrics']['total_return']*100:>6.2f}%   "
              f"{result['metrics']['sharpe_ratio']:>6.3f}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("ALPHA MODEL FACTORY EXAMPLES")
    print("="*60)
    print("\nThese examples demonstrate using the AlphaModelFactory")
    print("to create and test alpha models from JSON configuration.")
    
    example_1_basic_factory_usage()
    example_2_backtest_with_config()
    example_3_test_all_models()
    example_4_load_from_json_file()
    example_5_compare_configurations()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETE")
    print("="*60)
    print("\nFor more information, see docs/CONFIG_GUIDE.md")
