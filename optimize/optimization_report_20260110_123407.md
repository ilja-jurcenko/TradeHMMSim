# Alpha Model Parameter Optimization Report

**Generated:** 2026-01-10 12:34:07

## Timeline

- **Ticker:** SPY
- **Start Date:** 2020-01-01
- **End Date:** 2025-12-31
- **Optimization Metric:** Profit Factor

## Benchmark Performance (Buy & Hold)

- **Total Return:** 130.77%
- **Annualized Return:** 15.01%
- **Sharpe Ratio:** 0.778
- **Max Drawdown:** -33.72%
- **Profit Factor:** 1.162
- **Volatility:** 20.74%

## Optimization Results Summary

| Model | Short | Long | Profit Factor | Total Return | Sharpe | Max DD | Trades | Tested |
|-------|-------|------|---------------|--------------|--------|--------|--------|--------|
| EMA   |    58 |  108 |         1.265 |      133.61% |  1.219 | -12.44% |      3 |  13005 |
| HMA   |    50 |   51 |         1.430 |       95.87% |  0.962 | -13.72% |     13 |  13005 |
| KAMA  |    51 |   55 |         1.509 |      187.70% |  1.537 |  -9.44% |     24 |  13005 |
| SMA   |    31 |  184 |         1.295 |      129.06% |  1.299 | -12.84% |      2 |  13005 |
| TEMA  |    19 |  170 |         1.250 |       80.76% |  0.940 | -11.86% |     20 |  13005 |
| WMA   |    80 |  184 |         1.298 |      130.87% |  1.306 | -11.67% |      2 |  13005 |
| ZLEMA |    38 |  111 |         1.380 |      171.14% |  1.319 | -11.35% |     14 |  13005 |

*Note: 'Tested' shows total parameter combinations tested for each model.*

## Detailed Model Results

### EMA

**Model Description:**
- Exponential Moving Average - Weights recent prices more heavily

**Optimal Parameters:**
- Short Window: 58
- Long Window: 108
- Total Combinations Tested: 13005
- Improving Combinations Found: 17

**Performance Metrics:**
- Total Return: 133.61%
- Annualized Return: 15.24%
- Sharpe Ratio: 1.219
- Sortino Ratio: 0.993
- Max Drawdown: -12.44%
- Profit Factor: 1.265
- Win Rate: 43.33%
- Volatility: 12.26%
- Calmar Ratio: 1.226
- Number of Trades: 3
- Time in Market: 76.2%

**Comparison to Benchmark:**
- Return Difference: +2.83%
- Sharpe Difference: +0.440
- Profit Factor Difference: +0.103

### HMA

**Model Description:**
- Hull Moving Average - Reduces lag using weighted moving averages

**Optimal Parameters:**
- Short Window: 50
- Long Window: 51
- Total Combinations Tested: 13005
- Improving Combinations Found: 23

**Performance Metrics:**
- Total Return: 95.87%
- Annualized Return: 11.90%
- Sharpe Ratio: 0.962
- Sortino Ratio: 0.507
- Max Drawdown: -13.72%
- Profit Factor: 1.430
- Win Rate: 13.34%
- Volatility: 12.49%
- Calmar Ratio: 0.867
- Number of Trades: 13
- Time in Market: 23.8%

**Comparison to Benchmark:**
- Return Difference: -34.90%
- Sharpe Difference: +0.184
- Profit Factor Difference: +0.268

### KAMA

**Model Description:**
- Kaufman's Adaptive Moving Average - Adjusts to market volatility

**Optimal Parameters:**
- Short Window: 51
- Long Window: 55
- Total Combinations Tested: 13005
- Improving Combinations Found: 29

**Performance Metrics:**
- Total Return: 187.70%
- Annualized Return: 19.33%
- Sharpe Ratio: 1.537
- Sortino Ratio: 1.092
- Max Drawdown: -9.44%
- Profit Factor: 1.509
- Win Rate: 28.27%
- Volatility: 11.97%
- Calmar Ratio: 2.047
- Number of Trades: 24
- Time in Market: 48.1%

**Comparison to Benchmark:**
- Return Difference: +56.92%
- Sharpe Difference: +0.758
- Profit Factor Difference: +0.347

### SMA

**Model Description:**
- Simple Moving Average - Traditional crossover strategy using arithmetic mean

**Optimal Parameters:**
- Short Window: 31
- Long Window: 184
- Total Combinations Tested: 13005
- Improving Combinations Found: 19

**Performance Metrics:**
- Total Return: 129.06%
- Annualized Return: 14.87%
- Sharpe Ratio: 1.299
- Sortino Ratio: 1.036
- Max Drawdown: -12.84%
- Profit Factor: 1.295
- Win Rate: 39.35%
- Volatility: 11.15%
- Calmar Ratio: 1.158
- Number of Trades: 2
- Time in Market: 69.5%

**Comparison to Benchmark:**
- Return Difference: -1.72%
- Sharpe Difference: +0.520
- Profit Factor Difference: +0.133

### TEMA

**Model Description:**
- Triple Exponential Moving Average - Further reduces lag

**Optimal Parameters:**
- Short Window: 19
- Long Window: 170
- Total Combinations Tested: 13005
- Improving Combinations Found: 13

**Performance Metrics:**
- Total Return: 80.76%
- Annualized Return: 10.41%
- Sharpe Ratio: 0.940
- Sortino Ratio: 0.654
- Max Drawdown: -11.86%
- Profit Factor: 1.250
- Win Rate: 30.66%
- Volatility: 11.20%
- Calmar Ratio: 0.877
- Number of Trades: 20
- Time in Market: 54.7%

**Comparison to Benchmark:**
- Return Difference: -50.02%
- Sharpe Difference: +0.162
- Profit Factor Difference: +0.088

### WMA

**Model Description:**
- Weighted Moving Average - Linear weighting favoring recent data

**Optimal Parameters:**
- Short Window: 80
- Long Window: 184
- Total Combinations Tested: 13005
- Improving Combinations Found: 26

**Performance Metrics:**
- Total Return: 130.87%
- Annualized Return: 15.02%
- Sharpe Ratio: 1.306
- Sortino Ratio: 1.044
- Max Drawdown: -11.67%
- Profit Factor: 1.298
- Win Rate: 39.15%
- Volatility: 11.20%
- Calmar Ratio: 1.287
- Number of Trades: 2
- Time in Market: 69.3%

**Comparison to Benchmark:**
- Return Difference: +0.10%
- Sharpe Difference: +0.527
- Profit Factor Difference: +0.136

### ZLEMA

**Model Description:**
- Zero-Lag Exponential Moving Average - Minimizes lag in trend detection

**Optimal Parameters:**
- Short Window: 38
- Long Window: 111
- Total Combinations Tested: 13005
- Improving Combinations Found: 27

**Performance Metrics:**
- Total Return: 171.14%
- Annualized Return: 18.15%
- Sharpe Ratio: 1.319
- Sortino Ratio: 1.034
- Max Drawdown: -11.35%
- Profit Factor: 1.380
- Win Rate: 32.51%
- Volatility: 13.32%
- Calmar Ratio: 1.599
- Number of Trades: 14
- Time in Market: 57.6%

**Comparison to Benchmark:**
- Return Difference: +40.37%
- Sharpe Difference: +0.541
- Profit Factor Difference: +0.218

## Model Rankings

### By Total Return

1. **KAMA**: 187.70%
2. **ZLEMA**: 171.14%
3. **EMA**: 133.61%
4. **WMA**: 130.87%
5. **SMA**: 129.06%
6. **HMA**: 95.87%
7. **TEMA**: 80.76%

### By Profit Factor

1. **KAMA**: 1.509
2. **HMA**: 1.430
3. **ZLEMA**: 1.380
4. **WMA**: 1.298
5. **SMA**: 1.295
6. **EMA**: 1.265
7. **TEMA**: 1.250

### By Sharpe Ratio

1. **KAMA**: 1.537
2. **ZLEMA**: 1.319
3. **WMA**: 1.306
4. **SMA**: 1.299
5. **EMA**: 1.219
6. **HMA**: 0.962
7. **TEMA**: 0.940

## Conclusion

### Key Findings

1. **Best Total Return:** KAMA achieved 187.70%, outperforming the buy-and-hold benchmark by +56.92%

2. **Best Profit Factor:** KAMA with 1.509, indicating strong profitability per unit of risk

3. **Best Risk-Adjusted Return:** KAMA with Sharpe ratio of 1.537

4. **Benchmark Comparison:** 4 out of 7 models (57%) outperformed the buy-and-hold strategy

5. **Parameter Patterns:** Average optimal short window was 47 and long window was 123

6. **Trading Frequency:** Models averaged 11 trades, with KAMA executing 24 trades

### Recommendations

- **Overall:** Trend-following strategies show promise, with majority outperforming buy-and-hold
- **For Maximum Returns:** Use KAMA with parameters (51, 55)
- **For Risk-Adjusted Returns:** Use KAMA with parameters (51, 55)
- **For Consistent Profitability:** Use KAMA with parameters (51, 55)

---

*Note: Past performance does not guarantee future results. These optimizations are based on historical data and should be validated on out-of-sample periods before deployment.*