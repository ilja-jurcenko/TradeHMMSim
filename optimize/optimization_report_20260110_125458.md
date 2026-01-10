# Alpha Model Parameter Optimization Report

**Generated:** 2026-01-10 12:54:58

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

| Model | Short | Long | Profit Factor | Total Return | Sharpe | Max DD | Trades | Tested |
|-------|-------|------|---------------|--------------|--------|--------|--------|--------|
| EMA   |    58 |  108 |         1.265 |      133.61% |  1.219 | -12.44% |      3 |  13005 |
| HMA   |    73 |   76 |         1.293 |      102.35% |  1.096 | -13.66% |     21 |  13005 |
| KAMA  |    51 |   55 |         1.509 |      187.70% |  1.537 |  -9.44% |     24 |  13005 |
| SMA   |    56 |   78 |         1.290 |      143.09% |  1.291 | -12.88% |      5 |  13005 |
| TEMA  |    19 |  170 |         1.250 |       80.76% |  0.940 | -11.86% |     20 |  13005 |
| WMA   |    69 |  123 |         1.286 |      135.65% |  1.282 | -15.11% |      4 |  13005 |
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
- Improving Combinations Found: 22

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
- Short Window: 73
- Long Window: 76
- Total Combinations Tested: 13005
- Improving Combinations Found: 19

**Performance Metrics:**
- Total Return: 102.35%
- Annualized Return: 12.51%
- Sharpe Ratio: 1.096
- Sortino Ratio: 0.779
- Max Drawdown: -13.66%
- Profit Factor: 1.293
- Win Rate: 31.25%
- Volatility: 11.35%
- Calmar Ratio: 0.916
- Number of Trades: 21
- Time in Market: 55.3%

**Comparison to Benchmark:**
- Return Difference: -28.42%
- Sharpe Difference: +0.317
- Profit Factor Difference: +0.131

### KAMA

**Model Description:**
- Kaufman's Adaptive Moving Average - Adjusts to market volatility

**Optimal Parameters:**
- Short Window: 51
- Long Window: 55
- Total Combinations Tested: 13005
- Improving Combinations Found: 18

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
- Short Window: 56
- Long Window: 78
- Total Combinations Tested: 13005
- Improving Combinations Found: 24

**Performance Metrics:**
- Total Return: 143.09%
- Annualized Return: 16.01%
- Sharpe Ratio: 1.291
- Sortino Ratio: 1.036
- Max Drawdown: -12.88%
- Profit Factor: 1.290
- Win Rate: 40.74%
- Volatility: 12.08%
- Calmar Ratio: 1.243
- Number of Trades: 5
- Time in Market: 71.7%

**Comparison to Benchmark:**
- Return Difference: +12.32%
- Sharpe Difference: +0.512
- Profit Factor Difference: +0.128

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
- Short Window: 69
- Long Window: 123
- Total Combinations Tested: 13005
- Improving Combinations Found: 23

**Performance Metrics:**
- Total Return: 135.65%
- Annualized Return: 15.41%
- Sharpe Ratio: 1.282
- Sortino Ratio: 1.020
- Max Drawdown: -15.11%
- Profit Factor: 1.286
- Win Rate: 40.94%
- Volatility: 11.72%
- Calmar Ratio: 1.020
- Number of Trades: 4
- Time in Market: 72.0%

**Comparison to Benchmark:**
- Return Difference: +4.87%
- Sharpe Difference: +0.503
- Profit Factor Difference: +0.124

### ZLEMA

**Model Description:**
- Zero-Lag Exponential Moving Average - Minimizes lag in trend detection

**Optimal Parameters:**
- Short Window: 38
- Long Window: 111
- Total Combinations Tested: 13005
- Improving Combinations Found: 33

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
3. **SMA**: 143.09%
4. **WMA**: 135.65%
5. **EMA**: 133.61%
6. **HMA**: 102.35%
7. **TEMA**: 80.76%

### By Profit Factor

1. **KAMA**: 1.509
2. **ZLEMA**: 1.380
3. **HMA**: 1.293
4. **SMA**: 1.290
5. **WMA**: 1.286
6. **EMA**: 1.265
7. **TEMA**: 1.250

### By Sharpe Ratio

1. **KAMA**: 1.537
2. **ZLEMA**: 1.319
3. **SMA**: 1.291
4. **WMA**: 1.282
5. **EMA**: 1.219
6. **HMA**: 1.096
7. **TEMA**: 0.940

## Conclusion

### Key Findings

1. **Best Total Return:** KAMA achieved 187.70%, outperforming the buy-and-hold benchmark by +56.92%

2. **Best Profit Factor:** KAMA with 1.509, indicating strong profitability per unit of risk

3. **Best Risk-Adjusted Return:** KAMA with Sharpe ratio of 1.537

4. **Benchmark Comparison:** 5 out of 7 models (71%) outperformed the buy-and-hold strategy

5. **Parameter Patterns:** Average optimal short window was 52 and long window was 103

6. **Trading Frequency:** Models averaged 13 trades, with KAMA executing 24 trades

### Recommendations

- **Overall:** Trend-following strategies show promise, with majority outperforming buy-and-hold
- **For Maximum Returns:** Use KAMA with parameters (51, 55)
- **For Risk-Adjusted Returns:** Use KAMA with parameters (51, 55)
- **For Consistent Profitability:** Use KAMA with parameters (51, 55)

---

*Note: Past performance does not guarantee future results. These optimizations are based on historical data and should be validated on out-of-sample periods before deployment.*