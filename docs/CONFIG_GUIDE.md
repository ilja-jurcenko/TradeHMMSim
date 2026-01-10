# Configuration System

This document explains how to use JSON configuration files to control backtest parameters.

## Overview

The backtesting framework now supports external JSON configuration files to expose all hyperparameters. This allows you to:
- Run backtests with different parameter combinations without code changes
- Share configurations for reproducible results
- Test optimal HMM configurations discovered through parameter search

## Configuration File Structure

```json
{
  "backtest": {
    "initial_capital": 100000.0,
    "transaction_cost": 0.001,
    "rebalance_frequency": 1
  },
  "strategy": {
    "strategy_mode": "alpha_hmm_combine",
    "walk_forward": true
  },
  "hmm": {
    "n_states": 3,
    "random_state": 42,
    "train_window": 252,
    "refit_every": 42,
    "bear_prob_threshold": 0.65,
    "bull_prob_threshold": 0.65
  },
  "alpha_model": {
    "short_window": 10,
    "long_window": 30
  },
  "data": {
    "ticker": "SPY",
    "start_date": "2020-01-01",
    "end_date": "2025-12-31"
  },
  "output": {
    "save_plots": true,
    "output_dir": null
  }
}
```

## Parameter Descriptions

### Backtest Parameters
- **initial_capital**: Starting capital for backtesting (default: 100000.0)
- **transaction_cost**: Cost per trade as a fraction (e.g., 0.001 = 0.1%)
- **rebalance_frequency**: Rebalance every N periods (1 = daily)

### Strategy Parameters
- **strategy_mode**: Trading strategy to use
  - `alpha_only`: Alpha model signals only
  - `hmm_only`: HMM regime signals only
  - `alpha_hmm_filter`: HMM filters out bear regime
  - `alpha_hmm_combine`: Combine alpha and HMM signals (contrarian entries)
- **walk_forward**: Use walk-forward testing for HMM (recommended: true)

### HMM Parameters
- **n_states**: Number of HMM states (default: 3 for bull/bear/neutral)
- **random_state**: Random seed for reproducibility
- **train_window**: Historical window for HMM training (in trading days)
  - 252 ≈ 1 year, 504 ≈ 2 years, 756 ≈ 3 years
- **refit_every**: Refit HMM model every N periods
- **bear_prob_threshold**: Probability threshold for bear regime detection (0-1)
- **bull_prob_threshold**: Probability threshold for bull regime detection (0-1)

### Alpha Model Parameters
- **short_window**: Short moving average window
- **long_window**: Long moving average window

### Data Parameters
- **ticker**: Stock ticker symbol to backtest
- **start_date**: Backtest start date (YYYY-MM-DD)
- **end_date**: Backtest end date (YYYY-MM-DD)

### Output Parameters
- **save_plots**: Generate and save plots (true/false)
- **output_dir**: Custom output directory (null = auto-generate timestamp)

## Pre-configured Files

### config_default.json
Baseline configuration using original parameters:
- train_window: 504 (2 years)
- refit_every: 21 (~1 month)

### config_optimal.json
**Optimal configuration** from HMM parameter search:
- train_window: 252 (1 year)
- refit_every: 42 (~2 months)
- Best balance of returns (105.56%) and Sharpe ratio (1.257)
- Recommended for production use

### config_accurate.json
**Most accurate configuration** from HMM parameter search:
- train_window: 756 (3 years)
- refit_every: 21 (~1 month)
- Highest Sharpe ratio (1.583) with 68.68% returns
- Best for risk-adjusted returns

## Usage

### Using Configuration File

Run backtest with a configuration file:
```bash
python3 run_comparison.py --config config_optimal.json
```

### Command Line Override

You can still use command line arguments (config file takes precedence):
```bash
# Using config file
python3 run_comparison.py --config config_optimal.json --save-plots

# Traditional command line (without config)
python3 run_comparison.py SPY 2020-01-01 2025-12-31 --save-plots
```

### Programmatic Usage

Use the ConfigLoader in your Python code:

```python
from config_loader import ConfigLoader
from backtest import BacktestEngine

# Load configuration
config = ConfigLoader.load_config('config_optimal.json')

# Extract parameters
backtest_params = ConfigLoader.get_backtest_params(config)
hmm_params = ConfigLoader.get_hmm_params(config)
alpha_params = ConfigLoader.get_alpha_params(config)

# Create backtest engine
from alpha_models import EMA
from signal_filter import HMMRegimeFilter

model = EMA(**alpha_params)
hmm_filter = HMMRegimeFilter(**hmm_params)
engine = BacktestEngine(close_prices, model, hmm_filter)

# Run with config parameters
results = engine.run(**backtest_params)
```

### Creating Custom Configurations

1. Copy an existing config file:
   ```bash
   cp config_optimal.json config_custom.json
   ```

2. Edit parameters in your favorite editor

3. Run with your custom config:
   ```bash
   python3 run_comparison.py --config config_custom.json
   ```

## Testing Multiple Configurations

Create a script to test multiple configurations:

```python
from config_loader import ConfigLoader

configs = [
    'config_default.json',
    'config_optimal.json',
    'config_accurate.json'
]

for config_file in configs:
    print(f"\nTesting {config_file}...")
    results, output_dir = run_comparison(config_path=config_file)
    print(f"Results saved to: {output_dir}")
```

## Best Practices

1. **Version Control**: Commit configuration files to track parameter changes
2. **Naming Convention**: Use descriptive names (e.g., `config_low_cost.json`, `config_aggressive.json`)
3. **Documentation**: Add comments in separate markdown files explaining configuration choices
4. **Reproducibility**: Always set `random_state` for reproducible results
5. **Validation**: Test new configurations on out-of-sample data

## Parameter Optimization

Based on HMM parameter search results:

| Configuration | train_window | refit_every | Total Return | Sharpe Ratio | Use Case |
|--------------|--------------|-------------|--------------|--------------|----------|
| Baseline     | 504          | 21          | 67.16%       | 1.099        | Reference |
| Optimal      | 252          | 42          | 105.56%      | 1.257        | Production (best returns) |
| Stable       | 252          | 63          | 91.83%       | 1.171        | Conservative (fewer switches) |
| Accurate     | 756          | 21          | 68.68%       | 1.583        | Risk-adjusted (highest Sharpe) |

See `../hmm_analysis/HMM_CONFIGURATION_COMPARISON.md` for detailed analysis.

## Troubleshooting

### Config file not found
```python
FileNotFoundError: Configuration file not found: config.json
```
- Ensure the file path is correct
- Use absolute path or run from project root

### Invalid JSON
```python
json.decoder.JSONDecodeError
```
- Validate JSON syntax at jsonlint.com
- Check for trailing commas, missing quotes, or brackets

### Parameter not applied
- Verify parameter is in correct section (backtest, strategy, hmm, etc.)
- Check spelling matches exactly
- Restart Python interpreter if using interactive mode

## Related Files

- `config_loader.py`: Configuration loading utility
- `backtest.py`: Core backtesting engine
- `run_comparison.py`: Comparison script with config support
- `../hmm_analysis/HMM_CONFIGURATION_COMPARISON.md`: Parameter search results
