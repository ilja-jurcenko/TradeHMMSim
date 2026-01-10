# Alpha Model Parameter Optimization Report

**Generated:** 2026-01-10 11:49:00

## Timeline

- **Ticker:** SPY
- **Start Date:** 2020-01-01
- **End Date:** 2025-12-31
- **Optimization Metric:** Total Return

## Benchmark Performance (Buy & Hold)

- **Total Return:** 130.77%
- **Annualized Return:** 15.01%
- **Sharpe Ratio:** 0.778
- **Max Drawdown:** -33.72%
- **Profit Factor:** 1.162
- **Volatility:** 20.74%

## Optimization Results Summary

| Model | Short | Long | Total Return | Profit Factor | Sharpe | Max DD | Trades |
|-------|-------|------|--------------|---------------|--------|--------|--------|
| EMA   |    10 |  100 |      105.14% |         1.222 |  1.039 | -21.27% |      9 |
| HMA   |    20 |   50 |       53.59% |         1.180 |  0.656 | -17.97% |     48 |
| KAMA  | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| SMA   |    10 |  100 |      110.91% |         1.240 |  1.101 | -21.93% |      8 |
| TEMA  |    10 |  200 |       54.87% |         1.180 |  0.705 | -12.59% |     31 |
| WMA   |    10 |   50 |      109.22% |         1.252 |  1.084 | -13.76% |     18 |
| ZLEMA |    50 |  100 |       79.62% |         1.213 |  0.751 | -16.73% |     19 |

## Detailed Model Results

### EMA

**Model Description:**
- Exponential Moving Average - Weights recent prices more heavily

**Optimal Parameters:**
- Short Window: 10
- Long Window: 100

**Performance Metrics:**
- Total Return: 105.14%
- Annualized Return: 12.77%
- Sharpe Ratio: 1.039
- Sortino Ratio: 0.837
- Max Drawdown: -21.27%
- Profit Factor: 1.222
- Win Rate: 43.40%
- Volatility: 12.30%
- Calmar Ratio: 0.600
- Number of Trades: 9
- Time in Market: 77.0%

**Comparison to Benchmark:**
- Return Difference: -25.63%
- Sharpe Difference: +0.260
- Profit Factor Difference: +0.060

### HMA

**Model Description:**
- Hull Moving Average - Reduces lag using weighted moving averages

**Optimal Parameters:**
- Short Window: 20
- Long Window: 50

**Performance Metrics:**
- Total Return: 53.59%
- Annualized Return: 7.44%
- Sharpe Ratio: 0.656
- Sortino Ratio: 0.458
- Max Drawdown: -17.97%
- Profit Factor: 1.180
- Win Rate: 28.33%
- Volatility: 12.06%
- Calmar Ratio: 0.414
- Number of Trades: 48
- Time in Market: 52.2%

**Comparison to Benchmark:**
- Return Difference: -77.18%
- Sharpe Difference: -0.123
- Profit Factor Difference: +0.018

### KAMA

**Status:** No valid parameter combinations found

### SMA

**Model Description:**
- Simple Moving Average - Traditional crossover strategy using arithmetic mean

**Optimal Parameters:**
- Short Window: 10
- Long Window: 100

**Performance Metrics:**
- Total Return: 110.91%
- Annualized Return: 13.29%
- Sharpe Ratio: 1.101
- Sortino Ratio: 0.879
- Max Drawdown: -21.93%
- Profit Factor: 1.240
- Win Rate: 41.87%
- Volatility: 11.99%
- Calmar Ratio: 0.606
- Number of Trades: 8
- Time in Market: 74.3%

**Comparison to Benchmark:**
- Return Difference: -19.87%
- Sharpe Difference: +0.323
- Profit Factor Difference: +0.079

### TEMA

**Model Description:**
- Triple Exponential Moving Average - Further reduces lag

**Optimal Parameters:**
- Short Window: 10
- Long Window: 200

**Performance Metrics:**
- Total Return: 54.87%
- Annualized Return: 7.59%
- Sharpe Ratio: 0.705
- Sortino Ratio: 0.494
- Max Drawdown: -12.59%
- Profit Factor: 1.180
- Win Rate: 30.86%
- Volatility: 11.28%
- Calmar Ratio: 0.603
- Number of Trades: 31
- Time in Market: 56.1%

**Comparison to Benchmark:**
- Return Difference: -75.90%
- Sharpe Difference: -0.073
- Profit Factor Difference: +0.018

### WMA

**Model Description:**
- Weighted Moving Average - Linear weighting favoring recent data

**Optimal Parameters:**
- Short Window: 10
- Long Window: 50

**Performance Metrics:**
- Total Return: 109.22%
- Annualized Return: 13.14%
- Sharpe Ratio: 1.084
- Sortino Ratio: 0.866
- Max Drawdown: -13.76%
- Profit Factor: 1.252
- Win Rate: 39.08%
- Volatility: 12.06%
- Calmar Ratio: 0.955
- Number of Trades: 18
- Time in Market: 70.1%

**Comparison to Benchmark:**
- Return Difference: -21.55%
- Sharpe Difference: +0.306
- Profit Factor Difference: +0.090

### ZLEMA

**Model Description:**
- Zero-Lag Exponential Moving Average - Minimizes lag in trend detection

**Optimal Parameters:**
- Short Window: 50
- Long Window: 100

**Performance Metrics:**
- Total Return: 79.62%
- Annualized Return: 10.29%
- Sharpe Ratio: 0.751
- Sortino Ratio: 0.536
- Max Drawdown: -16.73%
- Profit Factor: 1.213
- Win Rate: 31.59%
- Volatility: 14.43%
- Calmar Ratio: 0.615
- Number of Trades: 19
- Time in Market: 56.9%

**Comparison to Benchmark:**
- Return Difference: -51.15%
- Sharpe Difference: -0.027
- Profit Factor Difference: +0.051

## Model Rankings

### By Total Return

1. **SMA**: 110.91%
2. **WMA**: 109.22%
3. **EMA**: 105.14%
4. **ZLEMA**: 79.62%
5. **TEMA**: 54.87%
6. **HMA**: 53.59%

### By Profit Factor

1. **WMA**: 1.252
2. **SMA**: 1.240
3. **EMA**: 1.222
4. **ZLEMA**: 1.213
5. **HMA**: 1.180
6. **TEMA**: 1.180

### By Sharpe Ratio

1. **SMA**: 1.101
2. **WMA**: 1.084
3. **EMA**: 1.039
4. **ZLEMA**: 0.751
5. **TEMA**: 0.705
6. **HMA**: 0.656

## Conclusion

### Key Findings

1. **Best Total Return:** SMA achieved 110.91%, underperforming the buy-and-hold benchmark by -19.87%

2. **Best Profit Factor:** WMA with 1.252, indicating moderate profitability per unit of risk

3. **Best Risk-Adjusted Return:** SMA with Sharpe ratio of 1.101

4. **Benchmark Comparison:** 0 out of 6 models (0%) outperformed the buy-and-hold strategy

5. **Parameter Patterns:** Average optimal short window was 18 and long window was 100

6. **Trading Frequency:** Models averaged 22 trades, with SMA executing 8 trades

### Recommendations

- **Overall:** Buy-and-hold remains competitive; trend-following requires careful model selection
- **For Maximum Returns:** Use SMA with parameters (10, 100)
- **For Risk-Adjusted Returns:** Use SMA with parameters (10, 100)
- **For Consistent Profitability:** Use WMA with parameters (10, 50)

---

*Note: Past performance does not guarantee future results. These optimizations are based on historical data and should be validated on out-of-sample periods before deployment.*