# Alpha Model Parameter Optimization Report

**Generated:** 2026-01-10 12:14:06

## Timeline

- **Ticker:** SPY
- **Start Date:** 2023-01-01
- **End Date:** 2024-01-01
- **Optimization Metric:** Total Return

## Benchmark Performance (Buy & Hold)

- **Total Return:** 26.71%
- **Annualized Return:** 26.95%
- **Sharpe Ratio:** 1.896
- **Max Drawdown:** -9.97%
- **Profit Factor:** 1.358
- **Volatility:** 13.04%

## Optimization Results Summary

| Model | Short | Long | Total Return | Profit Factor | Sharpe | Max DD | Trades |
|-------|-------|------|--------------|---------------|--------|--------|--------|
| EMA   |    20 |  200 |       25.74% |         1.346 |  1.839 |  -9.97% |      0 |
| HMA   |    50 |  100 |       15.34% |         2.264 |  2.643 |  -2.40% |      1 |
| KAMA  |    10 |   50 |       20.19% |         1.647 |  2.250 |  -4.79% |      1 |
| SMA   |    30 |   50 |       17.08% |         1.575 |  2.003 |  -4.70% |      2 |
| TEMA  |    20 |   50 |       20.35% |         1.585 |  2.068 |  -3.78% |      6 |
| WMA   |    10 |   50 |       18.10% |         1.591 |  2.097 |  -5.67% |      2 |
| ZLEMA |    20 |  100 |       22.61% |         1.916 |  2.707 |  -2.60% |      1 |

## Detailed Model Results

### EMA

**Model Description:**
- Exponential Moving Average - Weights recent prices more heavily

**Optimal Parameters:**
- Short Window: 20
- Long Window: 200

**Performance Metrics:**
- Total Return: 25.74%
- Annualized Return: 25.97%
- Sharpe Ratio: 1.839
- Sortino Ratio: 1.869
- Max Drawdown: -9.97%
- Profit Factor: 1.346
- Win Rate: 55.60%
- Volatility: 13.02%
- Calmar Ratio: 2.604
- Number of Trades: 0
- Time in Market: 99.6%

**Comparison to Benchmark:**
- Return Difference: -0.97%
- Sharpe Difference: -0.057
- Profit Factor Difference: -0.011

### HMA

**Model Description:**
- Hull Moving Average - Reduces lag using weighted moving averages

**Optimal Parameters:**
- Short Window: 50
- Long Window: 100

**Performance Metrics:**
- Total Return: 15.34%
- Annualized Return: 15.48%
- Sharpe Ratio: 2.643
- Sortino Ratio: 1.639
- Max Drawdown: -2.40%
- Profit Factor: 2.264
- Win Rate: 19.20%
- Volatility: 5.50%
- Calmar Ratio: 6.452
- Number of Trades: 1
- Time in Market: 30.4%

**Comparison to Benchmark:**
- Return Difference: -11.37%
- Sharpe Difference: +0.747
- Profit Factor Difference: +0.907

### KAMA

**Model Description:**
- Kaufman's Adaptive Moving Average - Adjusts to market volatility

**Optimal Parameters:**
- Short Window: 10
- Long Window: 50

**Performance Metrics:**
- Total Return: 20.19%
- Annualized Return: 20.37%
- Sharpe Ratio: 2.250
- Sortino Ratio: 1.794
- Max Drawdown: -4.79%
- Profit Factor: 1.647
- Win Rate: 32.80%
- Volatility: 8.40%
- Calmar Ratio: 4.250
- Number of Trades: 1
- Time in Market: 55.6%

**Comparison to Benchmark:**
- Return Difference: -6.52%
- Sharpe Difference: +0.354
- Profit Factor Difference: +0.290

### SMA

**Model Description:**
- Simple Moving Average - Traditional crossover strategy using arithmetic mean

**Optimal Parameters:**
- Short Window: 30
- Long Window: 50

**Performance Metrics:**
- Total Return: 17.08%
- Annualized Return: 17.23%
- Sharpe Ratio: 2.003
- Sortino Ratio: 1.538
- Max Drawdown: -4.70%
- Profit Factor: 1.575
- Win Rate: 28.80%
- Volatility: 8.10%
- Calmar Ratio: 3.669
- Number of Trades: 2
- Time in Market: 50.4%

**Comparison to Benchmark:**
- Return Difference: -9.63%
- Sharpe Difference: +0.107
- Profit Factor Difference: +0.217

### TEMA

**Model Description:**
- Triple Exponential Moving Average - Further reduces lag

**Optimal Parameters:**
- Short Window: 20
- Long Window: 50

**Performance Metrics:**
- Total Return: 20.35%
- Annualized Return: 20.53%
- Sharpe Ratio: 2.068
- Sortino Ratio: 1.671
- Max Drawdown: -3.78%
- Profit Factor: 1.585
- Win Rate: 32.40%
- Volatility: 9.24%
- Calmar Ratio: 5.425
- Number of Trades: 6
- Time in Market: 56.0%

**Comparison to Benchmark:**
- Return Difference: -6.36%
- Sharpe Difference: +0.172
- Profit Factor Difference: +0.228

### WMA

**Model Description:**
- Weighted Moving Average - Linear weighting favoring recent data

**Optimal Parameters:**
- Short Window: 10
- Long Window: 50

**Performance Metrics:**
- Total Return: 18.10%
- Annualized Return: 18.26%
- Sharpe Ratio: 2.097
- Sortino Ratio: 1.748
- Max Drawdown: -5.67%
- Profit Factor: 1.591
- Win Rate: 32.40%
- Volatility: 8.16%
- Calmar Ratio: 3.220
- Number of Trades: 2
- Time in Market: 56.4%

**Comparison to Benchmark:**
- Return Difference: -8.60%
- Sharpe Difference: +0.201
- Profit Factor Difference: +0.233

### ZLEMA

**Model Description:**
- Zero-Lag Exponential Moving Average - Minimizes lag in trend detection

**Optimal Parameters:**
- Short Window: 20
- Long Window: 100

**Performance Metrics:**
- Total Return: 22.61%
- Annualized Return: 22.81%
- Sharpe Ratio: 2.707
- Sortino Ratio: 2.212
- Max Drawdown: -2.60%
- Profit Factor: 1.916
- Win Rate: 29.60%
- Volatility: 7.70%
- Calmar Ratio: 8.785
- Number of Trades: 1
- Time in Market: 49.6%

**Comparison to Benchmark:**
- Return Difference: -4.10%
- Sharpe Difference: +0.811
- Profit Factor Difference: +0.559

## Model Rankings

### By Total Return

1. **EMA**: 25.74%
2. **ZLEMA**: 22.61%
3. **TEMA**: 20.35%
4. **KAMA**: 20.19%
5. **WMA**: 18.10%
6. **SMA**: 17.08%
7. **HMA**: 15.34%

### By Profit Factor

1. **HMA**: 2.264
2. **ZLEMA**: 1.916
3. **KAMA**: 1.647
4. **WMA**: 1.591
5. **TEMA**: 1.585
6. **SMA**: 1.575
7. **EMA**: 1.346

### By Sharpe Ratio

1. **ZLEMA**: 2.707
2. **HMA**: 2.643
3. **KAMA**: 2.250
4. **WMA**: 2.097
5. **TEMA**: 2.068
6. **SMA**: 2.003
7. **EMA**: 1.839

## Conclusion

### Key Findings

1. **Best Total Return:** EMA achieved 25.74%, underperforming the buy-and-hold benchmark by -0.97%

2. **Best Profit Factor:** HMA with 2.264, indicating strong profitability per unit of risk

3. **Best Risk-Adjusted Return:** ZLEMA with Sharpe ratio of 2.707

4. **Benchmark Comparison:** 0 out of 7 models (0%) outperformed the buy-and-hold strategy

5. **Parameter Patterns:** Average optimal short window was 23 and long window was 86

6. **Trading Frequency:** Models averaged 2 trades, with EMA executing 0 trades

### Recommendations

- **Overall:** Buy-and-hold remains competitive; trend-following requires careful model selection
- **For Maximum Returns:** Use EMA with parameters (20, 200)
- **For Risk-Adjusted Returns:** Use ZLEMA with parameters (20, 100)
- **For Consistent Profitability:** Use HMA with parameters (50, 100)

---

*Note: Past performance does not guarantee future results. These optimizations are based on historical data and should be validated on out-of-sample periods before deployment.*