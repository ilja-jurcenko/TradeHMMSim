# Multiple Alpha Models Configuration

## Overview

The `run_comparison.py` script now supports testing multiple alpha models with different parameters in a single run using a configuration file. This allows for comprehensive comparison across various model types and parameter sets.

## Configuration Format

### New Format: Multiple Models

```json
{
  "alpha_models": [
    {
      "type": "SMA",
      "parameters": {
        "short_window": 10,
        "long_window": 30
      }
    },
    {
      "type": "EMA",
      "parameters": {
        "short_window": 12,
        "long_window": 26
      }
    },
    {
      "type": "KAMA",
      "parameters": {
        "short_window": 10,
        "long_window": 30
      }
    }
  ],
  "data": {
    "ticker": "SPY",
    "start_date": "2020-01-01",
    "end_date": "2025-12-31"
  },
  "backtest": {
    "transaction_cost": 0.001,
    "rebalance_frequency": 1
  },
  "hmm": {
    "train_window": 252,
    "refit_every": 42
  },
  "output": {
    "save_plots": true
  }
}
```

### Legacy Format: Single Model (Still Supported)

```json
{
  "alpha_model": {
    "type": "SMA",
    "parameters": {
      "short_window": 10,
      "long_window": 30
    }
  },
  ...
}
```

## Features

### 1. Multiple Models with Different Parameters

Each model in the list can have its own unique parameters:

```json
{
  "alpha_models": [
    {
      "type": "SMA",
      "parameters": {
        "short_window": 5,
        "long_window": 20
      }
    },
    {
      "type": "SMA",
      "parameters": {
        "short_window": 10,
        "long_window": 30
      }
    },
    {
      "type": "EMA",
      "parameters": {
        "short_window": 12,
        "long_window": 26
      }
    }
  ]
}
```

This will test:
- SMA with 5/20 windows
- SMA with 10/30 windows  
- EMA with 12/26 windows

### 2. All Available Model Types

Supported model types:
- `SMA` - Simple Moving Average
- `EMA` - Exponential Moving Average
- `WMA` - Weighted Moving Average
- `HMA` - Hull Moving Average
- `KAMA` - Kaufman Adaptive Moving Average
- `TEMA` - Triple Exponential Moving Average
- `ZLEMA` - Zero Lag Exponential Moving Average

### 3. Backward Compatibility

The old `alpha_model` (singular) format still works for backward compatibility. If both `alpha_models` and `alpha_model` are present, `alpha_models` takes precedence.

## Usage

### Command Line

```bash
# Using the multi-model config
python3 run_comparison.py --config config_multi_model.json --save-plots

# With output directory
python3 run_comparison.py --config config_multi_model.json --output-dir my_results
```

### Example Configurations

#### Example 1: Compare All Model Types

```json
{
  "alpha_models": [
    {"type": "SMA", "parameters": {"short_window": 10, "long_window": 30}},
    {"type": "EMA", "parameters": {"short_window": 10, "long_window": 30}},
    {"type": "WMA", "parameters": {"short_window": 10, "long_window": 30}},
    {"type": "HMA", "parameters": {"short_window": 10, "long_window": 30}},
    {"type": "KAMA", "parameters": {"short_window": 10, "long_window": 30}},
    {"type": "TEMA", "parameters": {"short_window": 10, "long_window": 30}},
    {"type": "ZLEMA", "parameters": {"short_window": 10, "long_window": 30}}
  ]
}
```

#### Example 2: Parameter Optimization Study

```json
{
  "alpha_models": [
    {"type": "EMA", "parameters": {"short_window": 5, "long_window": 15}},
    {"type": "EMA", "parameters": {"short_window": 10, "long_window": 30}},
    {"type": "EMA", "parameters": {"short_window": 12, "long_window": 26}},
    {"type": "EMA", "parameters": {"short_window": 20, "long_window": 50}}
  ]
}
```

#### Example 3: Mixed Models for Comparison

```json
{
  "alpha_models": [
    {"type": "SMA", "parameters": {"short_window": 10, "long_window": 30}},
    {"type": "EMA", "parameters": {"short_window": 12, "long_window": 26}},
    {"type": "KAMA", "parameters": {"short_window": 10, "long_window": 30}},
    {"type": "HMA", "parameters": {"short_window": 10, "long_window": 30}}
  ]
}
```

## Benefits

1. **Comprehensive Testing**: Test multiple models and parameters in one run
2. **Efficient Comparison**: All models use the same data and time period
3. **Parameter Studies**: Easily compare different parameter combinations
4. **Reproducible Results**: Configuration files document exact setup
5. **Batch Analysis**: Run multiple configurations sequentially

## Output

The analysis report will show results for each model configuration:

```
Top 10 Strategies by Total Return
| Model | Strategy             | Total Return (%) | Sharpe Ratio | Max Drawdown (%) |
|-------|---------------------|------------------|--------------|------------------|
| EMA   | Alpha + HMM Combine | 45.23           | 1.85         | -12.45          |
| KAMA  | Alpha + HMM Filter  | 42.67           | 1.72         | -11.23          |
| SMA   | Alpha Only          | 38.91           | 1.45         | -15.67          |
```

## Sample Configuration File

See `config_multi_model.json` for a complete example with all available options.

## Testing

Unit tests are available in `tests/unit/test_run_comparison_multi_model.py`:

```bash
# Run the tests
python3 tests/unit/test_run_comparison_multi_model.py

# Or with unittest discovery
python3 -m unittest tests.unit.test_run_comparison_multi_model -v
```

## Migration Guide

### From Single Model to Multi-Model

**Before (single model):**
```json
{
  "alpha_model": {
    "type": "EMA",
    "parameters": {
      "short_window": 10,
      "long_window": 30
    }
  }
}
```

**After (multi-model):**
```json
{
  "alpha_models": [
    {
      "type": "EMA",
      "parameters": {
        "short_window": 10,
        "long_window": 30
      }
    }
  ]
}
```

Simply wrap the single model in an array and rename the key from `alpha_model` to `alpha_models`.

## Notes

- Each model in the list will be tested with all 4 strategies (Alpha Only, HMM Only, Alpha + HMM Filter, Alpha + HMM Combine)
- Total number of backtests = number of models × 4 strategies
- Execution time scales linearly with the number of models
- Individual plots are saved for each model/strategy combination when `save_plots: true`
