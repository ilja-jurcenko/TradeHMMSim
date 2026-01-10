# Alpha Model Parameter Optimization Report

**Generated:** 2026-01-10 12:21:09

## Timeline

- **Ticker:** SPY
- **Start Date:** 2023-01-01
- **End Date:** 2024-01-01
- **Optimization Metric:** Profit Factor

## Benchmark Performance (Buy & Hold)

- **Total Return:** 26.71%
- **Annualized Return:** 26.95%
- **Sharpe Ratio:** 1.896
- **Max Drawdown:** -9.97%
- **Profit Factor:** 1.358
- **Volatility:** 13.04%

## Optimization Results Summary

| Model | Short | Long | Profit Factor | Total Return | Sharpe | Max DD | Trades | Tested |
|-------|-------|------|---------------|--------------|--------|--------|--------|--------|
| EMA   |    14 |   30 |         1.424 |       20.93% |  1.855 |  -6.99% |      3 |   1490 |
| HMA   |    30 |   94 |         2.313 |       17.59% |  2.841 |  -2.79% |      2 |   1490 |
| KAMA  |    29 |   30 |         2.944 |       16.86% |  2.976 |  -2.16% |     12 |   1490 |
| SMA   |    26 |   48 |         1.640 |       18.09% |  2.135 |  -4.70% |      2 |   1490 |
| TEMA  |    10 |   30 |         1.647 |       20.37% |  2.112 |  -4.53% |      9 |   1490 |
| WMA   |    13 |   42 |         1.672 |       19.72% |  2.290 |  -4.38% |      2 |   1490 |
| ZLEMA |    22 |   85 |         2.372 |       15.37% |  2.700 |  -2.16% |      1 |   1490 |

*Note: 'Tested' shows total parameter combinations tested for each model.*

## Detailed Model Results

### EMA

**Model Description:**
- Exponential Moving Average - Weights recent prices more heavily

**Optimal Parameters:**
- Short Window: 14
- Long Window: 30
- Total Combinations Tested: 1490
- Improving Combinations Found: 2

**Performance Metrics:**
- Total Return: 20.93%
- Annualized Return: 21.12%
- Sharpe Ratio: 1.855
- Sortino Ratio: 1.716
- Max Drawdown: -6.99%
- Profit Factor: 1.424
- Win Rate: 42.00%
- Volatility: 10.64%
- Calmar Ratio: 3.021
- Number of Trades: 3
- Time in Market: 75.2%

**Comparison to Benchmark:**
- Return Difference: -5.78%
- Sharpe Difference: -0.041
- Profit Factor Difference: +0.066

### HMA

**Model Description:**
- Hull Moving Average - Reduces lag using weighted moving averages

**Optimal Parameters:**
- Short Window: 30
- Long Window: 94
- Total Combinations Tested: 1490
- Improving Combinations Found: 13

**Performance Metrics:**
- Total Return: 17.59%
- Annualized Return: 17.74%
- Sharpe Ratio: 2.841
- Sortino Ratio: 1.895
- Max Drawdown: -2.79%
- Profit Factor: 2.313
- Win Rate: 20.80%
- Volatility: 5.81%
- Calmar Ratio: 6.351
- Number of Trades: 2
- Time in Market: 33.2%

**Comparison to Benchmark:**
- Return Difference: -9.12%
- Sharpe Difference: +0.945
- Profit Factor Difference: +0.955

### KAMA

**Model Description:**
- Kaufman's Adaptive Moving Average - Adjusts to market volatility

**Optimal Parameters:**
- Short Window: 29
- Long Window: 30
- Total Combinations Tested: 1490
- Improving Combinations Found: 8

**Performance Metrics:**
- Total Return: 16.86%
- Annualized Return: 17.01%
- Sharpe Ratio: 2.976
- Sortino Ratio: 2.001
- Max Drawdown: -2.16%
- Profit Factor: 2.944
- Win Rate: 16.40%
- Volatility: 5.33%
- Calmar Ratio: 7.877
- Number of Trades: 12
- Time in Market: 25.2%

**Comparison to Benchmark:**
- Return Difference: -9.85%
- Sharpe Difference: +1.080
- Profit Factor Difference: +1.586

### SMA

**Model Description:**
- Simple Moving Average - Traditional crossover strategy using arithmetic mean

**Optimal Parameters:**
- Short Window: 26
- Long Window: 48
- Total Combinations Tested: 1490
- Improving Combinations Found: 9

**Performance Metrics:**
- Total Return: 18.09%
- Annualized Return: 18.25%
- Sharpe Ratio: 2.135
- Sortino Ratio: 1.658
- Max Drawdown: -4.70%
- Profit Factor: 1.640
- Win Rate: 28.80%
- Volatility: 8.00%
- Calmar Ratio: 3.885
- Number of Trades: 2
- Time in Market: 50.0%

**Comparison to Benchmark:**
- Return Difference: -8.62%
- Sharpe Difference: +0.239
- Profit Factor Difference: +0.282

### TEMA

**Model Description:**
- Triple Exponential Moving Average - Further reduces lag

**Optimal Parameters:**
- Short Window: 10
- Long Window: 30
- Total Combinations Tested: 1490
- Improving Combinations Found: 1

**Performance Metrics:**
- Total Return: 20.37%
- Annualized Return: 20.55%
- Sharpe Ratio: 2.112
- Sortino Ratio: 1.617
- Max Drawdown: -4.53%
- Profit Factor: 1.647
- Win Rate: 29.20%
- Volatility: 9.05%
- Calmar Ratio: 4.538
- Number of Trades: 9
- Time in Market: 49.6%

**Comparison to Benchmark:**
- Return Difference: -6.34%
- Sharpe Difference: +0.216
- Profit Factor Difference: +0.289

### WMA

**Model Description:**
- Weighted Moving Average - Linear weighting favoring recent data

**Optimal Parameters:**
- Short Window: 13
- Long Window: 42
- Total Combinations Tested: 1490
- Improving Combinations Found: 8

**Performance Metrics:**
- Total Return: 19.72%
- Annualized Return: 19.89%
- Sharpe Ratio: 2.290
- Sortino Ratio: 1.909
- Max Drawdown: -4.38%
- Profit Factor: 1.672
- Win Rate: 32.40%
- Volatility: 8.07%
- Calmar Ratio: 4.538
- Number of Trades: 2
- Time in Market: 55.6%

**Comparison to Benchmark:**
- Return Difference: -6.99%
- Sharpe Difference: +0.394
- Profit Factor Difference: +0.314

### ZLEMA

**Model Description:**
- Zero-Lag Exponential Moving Average - Minimizes lag in trend detection

**Optimal Parameters:**
- Short Window: 22
- Long Window: 85
- Total Combinations Tested: 1490
- Improving Combinations Found: 10

**Performance Metrics:**
- Total Return: 15.37%
- Annualized Return: 15.50%
- Sharpe Ratio: 2.700
- Sortino Ratio: 1.619
- Max Drawdown: -2.16%
- Profit Factor: 2.372
- Win Rate: 18.80%
- Volatility: 5.39%
- Calmar Ratio: 7.178
- Number of Trades: 1
- Time in Market: 29.2%

**Comparison to Benchmark:**
- Return Difference: -11.34%
- Sharpe Difference: +0.804
- Profit Factor Difference: +1.015

## Model Rankings

### By Total Return

1. **EMA**: 20.93%
2. **TEMA**: 20.37%
3. **WMA**: 19.72%
4. **SMA**: 18.09%
5. **HMA**: 17.59%
6. **KAMA**: 16.86%
7. **ZLEMA**: 15.37%

### By Profit Factor

1. **KAMA**: 2.944
2. **ZLEMA**: 2.372
3. **HMA**: 2.313
4. **WMA**: 1.672
5. **TEMA**: 1.647
6. **SMA**: 1.640
7. **EMA**: 1.424

### By Sharpe Ratio

1. **KAMA**: 2.976
2. **HMA**: 2.841
3. **ZLEMA**: 2.700
4. **WMA**: 2.290
5. **SMA**: 2.135
6. **TEMA**: 2.112
7. **EMA**: 1.855

## Conclusion

### Key Findings

1. **Best Total Return:** EMA achieved 20.93%, underperforming the buy-and-hold benchmark by -5.78%

2. **Best Profit Factor:** KAMA with 2.944, indicating strong profitability per unit of risk

3. **Best Risk-Adjusted Return:** KAMA with Sharpe ratio of 2.976

4. **Benchmark Comparison:** 0 out of 7 models (0%) outperformed the buy-and-hold strategy

5. **Parameter Patterns:** Average optimal short window was 21 and long window was 51

6. **Trading Frequency:** Models averaged 4 trades, with EMA executing 3 trades

### Recommendations

- **Overall:** Buy-and-hold remains competitive; trend-following requires careful model selection
- **For Maximum Returns:** Use EMA with parameters (14, 30)
- **For Risk-Adjusted Returns:** Use KAMA with parameters (29, 30)
- **For Consistent Profitability:** Use KAMA with parameters (29, 30)

---

*Note: Past performance does not guarantee future results. These optimizations are based on historical data and should be validated on out-of-sample periods before deployment.*