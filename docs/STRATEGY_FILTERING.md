# Strategy Filtering Guide

## Overview

The `--strategies` command-line parameter allows you to run only specific strategies instead of all 6 strategies. This is useful for:

- **Faster testing**: Run only the strategies you're interested in
- **Focused analysis**: Compare specific strategies without waiting for all 6
- **Debugging**: Isolate and test individual strategies
- **Resource management**: Reduce computation time and output file size

## Available Strategies

1. **alpha_only** - Alpha model signals only (no HMM filtering)
2. **hmm_only** - HMM regime detection only (walk-forward)
3. **oracle** - HMM with full dataset (upper bound with future knowledge)
4. **alpha_hmm_filter** - Alpha signals filtered by HMM regime
5. **alpha_hmm_combine** - Combined 4-state strategy (alpha + HMM)
6. **regime_adaptive_alpha** - Regime-adaptive (trend-following in bull, mean-reversion in bear)

## Usage

### Run a Single Strategy

```bash
# Run only Oracle strategy
python run_comparison.py SPY --strategies oracle

# Run only Alpha Only strategy
python run_comparison.py SPY --strategies alpha_only
```

### Run Multiple Strategies

```bash
# Run Oracle and Alpha Only
python run_comparison.py SPY --strategies oracle,alpha_only

# Run all HMM-based strategies
python run_comparison.py SPY --strategies hmm_only,oracle,alpha_hmm_filter,alpha_hmm_combine

# Run comparison of no-HMM vs HMM
python run_comparison.py SPY --strategies alpha_only,alpha_hmm_filter
```

### Run All Strategies (Default)

```bash
# Without --strategies parameter, all 6 strategies run
python run_comparison.py SPY

# Or explicitly specify all
python run_comparison.py SPY --strategies alpha_only,hmm_only,oracle,alpha_hmm_filter,alpha_hmm_combine,regime_adaptive_alpha
```

## Examples

### Fast Oracle-Only Test (2-3 seconds)

```bash
python run_comparison.py SPY \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --strategies oracle
```

**Output**: 7 models × 1 strategy = 7 backtests

### Compare Alpha vs Oracle

```bash
python run_comparison.py SPY \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --strategies alpha_only,oracle \
  --save-plots
```

**Output**: 7 models × 2 strategies = 14 backtests

### Full Walk-Forward Test (Slower)

```bash
python run_comparison.py SPY \
  --start-date 2021-01-01 \
  --end-date 2024-12-31 \
  --strategies hmm_only,alpha_hmm_filter,alpha_hmm_combine
```

**Output**: 7 models × 3 strategies = 21 backtests

## Strategy Characteristics

| Strategy | Walk-Forward? | HMM Training | Speed | Log Start Date* |
|----------|---------------|--------------|-------|-----------------|
| alpha_only | No | None | ⚡️ Fast | 2018-01-31 |
| hmm_only | Yes | 504 days | 🐌 Slow | 2021-02-01 |
| oracle | No | All data | ⚡️ Fast | 2019-01-31 |
| alpha_hmm_filter | Yes | 504 days | 🐌 Slow | 2021-02-01 |
| alpha_hmm_combine | Yes | 504 days | 🐌 Slow | 2021-02-01 |
| regime_adaptive_alpha | Yes | 504 days | 🐌 Slow | 2021-02-01 |

*For date range 2018-01-01 to 2025-01-01 with default config

## HMM Impact Analysis

The HMM Impact Analysis section requires **all 4 core strategies**:
- `alpha_only`
- `hmm_only`
- `alpha_hmm_filter`
- `alpha_hmm_combine`

If you run a subset of strategies, the HMM Impact Analysis will be **skipped** with a message:

```
================================================================================
HMM IMPACT ANALYSIS
================================================================================

Skipping HMM Impact Analysis - requires all of: Alpha Only, HMM Only, Alpha + HMM Filter, Alpha + HMM Combine
Available strategies: Oracle
```

To enable HMM Impact Analysis, run:

```bash
python run_comparison.py SPY \
  --strategies alpha_only,hmm_only,alpha_hmm_filter,alpha_hmm_combine
```

## Log Files

When using `--enable-logging`, only the selected strategies generate log files:

```bash
# Only oracle logs generated
python run_comparison.py SPY --strategies oracle --enable-logging

# Check logs directory
ls results/run_YYYYMMDD_HHMMSS/logs/
# Output: oracle_SMA_*.csv, oracle_EMA_*.csv, etc.
```

## Performance Tips

1. **Use Oracle for quick tests** - No walk-forward training, runs in seconds
2. **Avoid walk-forward strategies** for rapid iteration - hmm_only, alpha_hmm_filter, etc. are much slower
3. **Filter early, plot later** - Run quick tests with --strategies, then re-run with --save-plots
4. **Shorter date ranges** for development - Use 1-2 years instead of 5+ years

## Validation

The script validates strategy names and shows an error for invalid names:

```bash
python run_comparison.py SPY --strategies invalid_strategy
# Error: Unknown strategy 'invalid_strategy'
# Valid strategies: alpha_only, hmm_only, oracle, alpha_hmm_filter, alpha_hmm_combine, regime_adaptive_alpha
```

## Integration with Other Flags

The `--strategies` parameter works with all other flags:

```bash
python run_comparison.py SPY,AGG \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --config config_default.json \
  --strategies oracle,alpha_only \
  --save-plots \
  --enable-logging \
  --output-dir results/custom_run
```

## Use Cases

### 1. Quick Sanity Check
```bash
# Test if oracle beats buy-and-hold
python run_comparison.py SPY --strategies oracle
```

### 2. Alpha Model Validation
```bash
# Test alpha models without HMM overhead
python run_comparison.py SPY --strategies alpha_only
```

### 3. Walk-Forward Performance
```bash
# Compare realistic walk-forward strategies
python run_comparison.py SPY --strategies hmm_only,alpha_hmm_filter,alpha_hmm_combine
```

### 4. Oracle vs Reality
```bash
# Compare upper bound (oracle) vs walk-forward (hmm_only)
python run_comparison.py SPY --strategies oracle,hmm_only
```

### 5. Development/Debugging
```bash
# Test changes to specific strategy
python run_comparison.py SPY \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --strategies alpha_hmm_combine \
  --enable-logging
```

## Summary

The `--strategies` parameter provides:
- ✅ **Flexibility**: Run any combination of strategies
- ✅ **Speed**: Skip slow walk-forward strategies for rapid testing
- ✅ **Focus**: Analyze specific strategy comparisons
- ✅ **Validation**: Automatic strategy name validation
- ✅ **Logging**: Only selected strategies generate logs
- ✅ **Compatibility**: Works with all other command-line flags

Default behavior (no `--strategies` specified) runs all 6 strategies for comprehensive analysis.
