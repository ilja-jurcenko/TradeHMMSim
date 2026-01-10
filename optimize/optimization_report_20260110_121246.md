# Alpha Model Parameter Optimization Report

**Generated:** 2026-01-10 12:12:46

## Timeline

- **Ticker:** SPY
- **Start Date:** 2020-01-01
- **End Date:** 2021-01-01
- **Optimization Metric:** Total Return

## Benchmark Performance (Buy & Hold)

- **Total Return:** 17.24%
- **Annualized Return:** 17.16%
- **Sharpe Ratio:** 0.644
- **Max Drawdown:** -33.72%
- **Profit Factor:** 1.137
- **Volatility:** 33.32%

## Optimization Results Summary

| Model | Short | Long | Total Return | Profit Factor | Sharpe | Max DD | Trades |
|-------|-------|------|--------------|---------------|--------|--------|--------|
| EMA   |    10 |  100 |       20.13% |         1.284 |  1.249 |  -9.44% |      1 |
| HMA   |    30 |   50 |       25.09% |         1.498 |  1.487 |  -9.03% |      5 |
| KAMA  |    20 |   50 |       58.79% |         1.515 |  2.016 | -11.29% |      0 |
| SMA   |    30 |   50 |       33.03% |         1.573 |  2.082 |  -9.44% |      1 |
| TEMA  |    10 |  100 |       20.37% |         1.305 |  1.237 | -10.13% |      6 |
| WMA   |    20 |  100 |       26.60% |         1.447 |  1.708 |  -9.44% |      0 |
| ZLEMA |    50 |  100 |       36.40% |         1.339 |  1.303 | -16.73% |      1 |

## Detailed Model Results

### EMA

**Model Description:**
- Exponential Moving Average - Weights recent prices more heavily

**Optimal Parameters:**
- Short Window: 10
- Long Window: 100

**Performance Metrics:**
- Total Return: 20.13%
- Annualized Return: 20.04%
- Sharpe Ratio: 1.249
- Sortino Ratio: 0.897
- Max Drawdown: -9.44%
- Profit Factor: 1.284
- Win Rate: 42.69%
- Volatility: 15.61%
- Calmar Ratio: 2.123
- Number of Trades: 1
- Time in Market: 72.7%

**Comparison to Benchmark:**
- Return Difference: +2.89%
- Sharpe Difference: +0.606
- Profit Factor Difference: +0.146

### HMA

**Model Description:**
- Hull Moving Average - Reduces lag using weighted moving averages

**Optimal Parameters:**
- Short Window: 30
- Long Window: 50

**Performance Metrics:**
- Total Return: 25.09%
- Annualized Return: 24.98%
- Sharpe Ratio: 1.487
- Sortino Ratio: 0.978
- Max Drawdown: -9.03%
- Profit Factor: 1.498
- Win Rate: 26.48%
- Volatility: 15.84%
- Calmar Ratio: 2.767
- Number of Trades: 5
- Time in Market: 45.1%

**Comparison to Benchmark:**
- Return Difference: +7.86%
- Sharpe Difference: +0.844
- Profit Factor Difference: +0.360

### KAMA

**Model Description:**
- Kaufman's Adaptive Moving Average - Adjusts to market volatility

**Optimal Parameters:**
- Short Window: 20
- Long Window: 50

**Performance Metrics:**
- Total Return: 58.79%
- Annualized Return: 58.50%
- Sharpe Ratio: 2.016
- Sortino Ratio: 1.830
- Max Drawdown: -11.29%
- Profit Factor: 1.515
- Win Rate: 47.83%
- Volatility: 24.32%
- Calmar Ratio: 5.182
- Number of Trades: 0
- Time in Market: 80.2%

**Comparison to Benchmark:**
- Return Difference: +41.56%
- Sharpe Difference: +1.372
- Profit Factor Difference: +0.378

### SMA

**Model Description:**
- Simple Moving Average - Traditional crossover strategy using arithmetic mean

**Optimal Parameters:**
- Short Window: 30
- Long Window: 50

**Performance Metrics:**
- Total Return: 33.03%
- Annualized Return: 32.88%
- Sharpe Ratio: 2.082
- Sortino Ratio: 1.378
- Max Drawdown: -9.44%
- Profit Factor: 1.573
- Win Rate: 36.76%
- Volatility: 14.14%
- Calmar Ratio: 3.483
- Number of Trades: 1
- Time in Market: 59.3%

**Comparison to Benchmark:**
- Return Difference: +15.79%
- Sharpe Difference: +1.439
- Profit Factor Difference: +0.436

### TEMA

**Model Description:**
- Triple Exponential Moving Average - Further reduces lag

**Optimal Parameters:**
- Short Window: 10
- Long Window: 100

**Performance Metrics:**
- Total Return: 20.37%
- Annualized Return: 20.28%
- Sharpe Ratio: 1.237
- Sortino Ratio: 0.896
- Max Drawdown: -10.13%
- Profit Factor: 1.305
- Win Rate: 36.76%
- Volatility: 15.97%
- Calmar Ratio: 2.002
- Number of Trades: 6
- Time in Market: 64.0%

**Comparison to Benchmark:**
- Return Difference: +3.13%
- Sharpe Difference: +0.594
- Profit Factor Difference: +0.168

### WMA

**Model Description:**
- Weighted Moving Average - Linear weighting favoring recent data

**Optimal Parameters:**
- Short Window: 20
- Long Window: 100

**Performance Metrics:**
- Total Return: 26.60%
- Annualized Return: 26.48%
- Sharpe Ratio: 1.708
- Sortino Ratio: 1.142
- Max Drawdown: -9.44%
- Profit Factor: 1.447
- Win Rate: 36.36%
- Volatility: 14.37%
- Calmar Ratio: 2.805
- Number of Trades: 0
- Time in Market: 60.9%

**Comparison to Benchmark:**
- Return Difference: +9.36%
- Sharpe Difference: +1.064
- Profit Factor Difference: +0.309

### ZLEMA

**Model Description:**
- Zero-Lag Exponential Moving Average - Minimizes lag in trend detection

**Optimal Parameters:**
- Short Window: 50
- Long Window: 100

**Performance Metrics:**
- Total Return: 36.40%
- Annualized Return: 36.23%
- Sharpe Ratio: 1.303
- Sortino Ratio: 1.011
- Max Drawdown: -16.73%
- Profit Factor: 1.339
- Win Rate: 44.66%
- Volatility: 26.45%
- Calmar Ratio: 2.165
- Number of Trades: 1
- Time in Market: 74.7%

**Comparison to Benchmark:**
- Return Difference: +19.16%
- Sharpe Difference: +0.659
- Profit Factor Difference: +0.202

## Model Rankings

### By Total Return

1. **KAMA**: 58.79%
2. **ZLEMA**: 36.40%
3. **SMA**: 33.03%
4. **WMA**: 26.60%
5. **HMA**: 25.09%
6. **TEMA**: 20.37%
7. **EMA**: 20.13%

### By Profit Factor

1. **SMA**: 1.573
2. **KAMA**: 1.515
3. **HMA**: 1.498
4. **WMA**: 1.447
5. **ZLEMA**: 1.339
6. **TEMA**: 1.305
7. **EMA**: 1.284

### By Sharpe Ratio

1. **SMA**: 2.082
2. **KAMA**: 2.016
3. **WMA**: 1.708
4. **HMA**: 1.487
5. **ZLEMA**: 1.303
6. **EMA**: 1.249
7. **TEMA**: 1.237

## Conclusion

### Key Findings

1. **Best Total Return:** KAMA achieved 58.79%, outperforming the buy-and-hold benchmark by +41.56%

2. **Best Profit Factor:** SMA with 1.573, indicating strong profitability per unit of risk

3. **Best Risk-Adjusted Return:** SMA with Sharpe ratio of 2.082

4. **Benchmark Comparison:** 7 out of 7 models (100%) outperformed the buy-and-hold strategy

5. **Parameter Patterns:** Average optimal short window was 24 and long window was 79

6. **Trading Frequency:** Models averaged 2 trades, with KAMA executing 0 trades

### Recommendations

- **Overall:** Trend-following strategies show promise, with majority outperforming buy-and-hold
- **For Maximum Returns:** Use KAMA with parameters (20, 50)
- **For Risk-Adjusted Returns:** Use SMA with parameters (30, 50)
- **For Consistent Profitability:** Use SMA with parameters (30, 50)

---

*Note: Past performance does not guarantee future results. These optimizations are based on historical data and should be validated on out-of-sample periods before deployment.*