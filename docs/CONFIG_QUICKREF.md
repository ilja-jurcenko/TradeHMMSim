# Configuration System - Quick Reference

## Available Configuration Files

| File | Description | HMM Parameters | Best For |
|------|-------------|----------------|----------|
| `config_default.json` | Baseline configuration | train_window: 504<br>refit_every: 21 | Reference/comparison |
| `config_optimal.json` | **Optimal returns** | train_window: 252<br>refit_every: 42 | **Production use** (105.56% returns, 1.257 Sharpe) |
| `config_accurate.json` | **Highest Sharpe** | train_window: 756<br>refit_every: 21 | Risk-adjusted returns (1.583 Sharpe) |

## Quick Commands

### Run with Configuration
```bash
# Optimal configuration (recommended for production)
python3 run_comparison.py --config config_optimal.json

# Accurate configuration (best risk-adjusted)
python3 run_comparison.py --config config_accurate.json

# Default/baseline configuration
python3 run_comparison.py --config config_default.json
```

### Traditional Command Line (no config file)
```bash
python3 run_comparison.py SPY 2020-01-01 2025-12-31 --save-plots
```

## Configuration Parameters

### All Available Parameters

```json
{
  "backtest": {
    "initial_capital": 100000.0,         // Starting capital
    "transaction_cost": 0.001,           // 0.1% per trade
    "rebalance_frequency": 1             // Days between rebalances
  },
  "strategy": {
    "strategy_mode": "alpha_hmm_combine", // Strategy type
    "walk_forward": true                  // Use walk-forward testing
  },
  "hmm": {
    "n_states": 3,                       // Bull/Bear/Neutral
    "random_state": 42,                  // For reproducibility
    "train_window": 252,                 // Training period (days)
    "refit_every": 42,                   // Refit frequency (days)
    "bear_prob_threshold": 0.65,         // Bear detection threshold
    "bull_prob_threshold": 0.65          // Bull detection threshold
  },
  "alpha_model": {
    "short_window": 10,                  // Short MA window
    "long_window": 30                    // Long MA window
  },
  "data": {
    "ticker": "SPY",                     // Stock ticker
    "start_date": "2020-01-01",          // Backtest start
    "end_date": "2025-12-31"             // Backtest end
  },
  "output": {
    "save_plots": true,                  // Generate plots
    "output_dir": null                   // Auto-generate timestamp dir
  }
}
```

## Strategy Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `alpha_only` | Alpha model signals only | Baseline comparison |
| `hmm_only` | HMM regime signals only | Pure regime-based trading |
| `alpha_hmm_filter` | HMM filters out bear regimes | Conservative approach |
| `alpha_hmm_combine` | Combine alpha + HMM signals | **Best performance** (contrarian entries) |

## HMM Training Windows

| Window | Days | Description |
|--------|------|-------------|
| 126 | ~6 months | Short-term, responsive |
| 252 | ~1 year | **Optimal balance** |
| 504 | ~2 years | Default/baseline |
| 756 | ~3 years | Long-term, stable |

## Common Workflows

### 1. Quick Test with Optimal Settings
```bash
python3 run_comparison.py --config config_optimal.json
```

### 2. Create Custom Configuration
```python
from config_loader import ConfigLoader

# Load and modify
config = ConfigLoader.load_config('config_optimal.json')
config['data']['ticker'] = 'QQQ'
config['backtest']['transaction_cost'] = 0.002

# Save
ConfigLoader.save_config(config, 'config_qqq.json')

# Run
python3 run_comparison.py --config config_qqq.json
```

### 3. Test Multiple Configurations
```bash
python3 examples/example_config_testing.py
```

### 4. Programmatic Usage
```python
from config_loader import ConfigLoader
from run_comparison import run_comparison

# Run with config
results, output_dir = run_comparison(config_path='config_optimal.json')
```

## Performance Comparison

Based on HMM parameter search (2020-01-01 to 2025-12-31):

| Configuration | Total Return | Sharpe Ratio | Max Drawdown | Regime Switches |
|--------------|--------------|--------------|--------------|----------------|
| Default (504,21) | 67.16% | 1.099 | Higher | 283 |
| **Optimal (252,42)** | **105.56%** | **1.257** | Moderate | 101 |
| Stable (252,63) | 91.83% | 1.171 | Lower | 82 |
| Accurate (756,21) | 68.68% | **1.583** | Lowest | 209 |

**Recommendation:** Use `config_optimal.json` for production - delivers highest returns with good Sharpe ratio.

## Tips

1. **Always set random_state** for reproducible results
2. **Use walk_forward=true** to avoid lookahead bias
3. **Test on out-of-sample data** before production
4. **Version control configs** to track parameter changes
5. **Start with optimal config** then tune for your needs

## Documentation

- **Full Guide:** [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
- **HMM Analysis:** [hmm_analysis/HMM_CONFIGURATION_COMPARISON.md](../hmm_analysis/HMM_CONFIGURATION_COMPARISON.md)
- **Main README:** [README.md](../README.md)

## Examples

- `examples/example_config_usage.py` - Programmatic config usage
- `examples/example_config_testing.py` - Test multiple configs
