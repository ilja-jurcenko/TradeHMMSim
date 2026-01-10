# HMM Impact Analysis on Trend-Following Strategies

**Generated:** 2026-01-10 11:09:17

**Ticker:** SPY

**Period:** 2020-01-01 to 2025-12-31

---

## Research Question

**What is the impact of HMM regime detection on trend-following strategy performance?**

This analysis compares standalone alpha model performance (pure trend-following) against alpha models enhanced with HMM regime detection (Alpha + HMM Combine strategy). Three HMM configurations are tested to assess robustness and identify optimal parameters.

---

## Benchmark Performance (Buy & Hold)

- **Total Return:** 130.77%
- **Sharpe Ratio:** 0.78
- **Max Drawdown:** -33.72%

---

## HMM Configurations Tested

| Configuration | train_window | refit_every | Description |
|---------------|--------------|-------------|-------------|
| Baseline | 504 | 21 | Original parameters |
| Optimal | 252 | 42 | Best total returns |
| Accurate | 756 | 21 | Best Sharpe ratio |

---

## Executive Summary

### Key Findings

1. **Average Return Improvement:** +2.92%
2. **Average Sharpe Improvement:** +0.411
3. **Best Return Improvement:** +43.61%
4. **Best Sharpe Improvement:** +1.223
5. **Models with Positive Return Impact:** 4/7
6. **Models with Positive Sharpe Impact:** 7/7

**Conclusion:** HMM regime detection provides **positive impact** on trend-following strategies, improving both returns and risk-adjusted performance on average.

---

## Detailed Results by Alpha Model

### SMA

**Baseline (Alpha Only):**
- Total Return: 64.42%
- Sharpe Ratio: 0.729

**Best HMM Configuration (Return):**
- Configuration: Optimal
- Total Return: 94.92%
- Improvement: +30.50%

**Best HMM Configuration (Sharpe):**
- Configuration: Accurate
- Sharpe Ratio: 1.482
- Improvement: +0.753

**Average Impact (all HMM configs):**
- Avg Return Improvement: +5.10%
- Avg Sharpe Improvement: +0.440

### EMA

**Baseline (Alpha Only):**
- Total Return: 93.22%
- Sharpe Ratio: 0.952

**Best HMM Configuration (Return):**
- Configuration: Optimal
- Total Return: 102.88%
- Improvement: +9.67%

**Best HMM Configuration (Sharpe):**
- Configuration: Accurate
- Sharpe Ratio: 1.570
- Improvement: +0.617

**Average Impact (all HMM configs):**
- Avg Return Improvement: -14.26%
- Avg Sharpe Improvement: +0.343

### WMA

**Baseline (Alpha Only):**
- Total Return: 87.99%
- Sharpe Ratio: 0.917

**Best HMM Configuration (Return):**
- Configuration: Optimal
- Total Return: 102.46%
- Improvement: +14.47%

**Best HMM Configuration (Sharpe):**
- Configuration: Accurate
- Sharpe Ratio: 1.632
- Improvement: +0.714

**Average Impact (all HMM configs):**
- Avg Return Improvement: -12.11%
- Avg Sharpe Improvement: +0.330

### HMA

**Baseline (Alpha Only):**
- Total Return: 32.97%
- Sharpe Ratio: 0.472

**Best HMM Configuration (Return):**
- Configuration: Accurate
- Total Return: 76.58%
- Improvement: +43.61%

**Best HMM Configuration (Sharpe):**
- Configuration: Accurate
- Sharpe Ratio: 1.695
- Improvement: +1.223

**Average Impact (all HMM configs):**
- Avg Return Improvement: +22.46%
- Avg Sharpe Improvement: +0.544

### KAMA

**Baseline (Alpha Only):**
- Total Return: 69.37%
- Sharpe Ratio: 0.799

**Best HMM Configuration (Return):**
- Configuration: Optimal
- Total Return: 72.75%
- Improvement: +3.38%

**Best HMM Configuration (Sharpe):**
- Configuration: Accurate
- Sharpe Ratio: 1.616
- Improvement: +0.817

**Average Impact (all HMM configs):**
- Avg Return Improvement: -3.62%
- Avg Sharpe Improvement: +0.383

### TEMA

**Baseline (Alpha Only):**
- Total Return: 58.57%
- Sharpe Ratio: 0.713

**Best HMM Configuration (Return):**
- Configuration: Optimal
- Total Return: 87.96%
- Improvement: +29.39%

**Best HMM Configuration (Sharpe):**
- Configuration: Accurate
- Sharpe Ratio: 1.417
- Improvement: +0.704

**Average Impact (all HMM configs):**
- Avg Return Improvement: +2.17%
- Avg Sharpe Improvement: +0.305

### ZLEMA

**Baseline (Alpha Only):**
- Total Return: 40.21%
- Sharpe Ratio: 0.551

**Best HMM Configuration (Return):**
- Configuration: Accurate
- Total Return: 82.10%
- Improvement: +41.90%

**Best HMM Configuration (Sharpe):**
- Configuration: Accurate
- Sharpe Ratio: 1.774
- Improvement: +1.223

**Average Impact (all HMM configs):**
- Avg Return Improvement: +20.68%
- Avg Sharpe Improvement: +0.530

---

## Top Performers

### Best Return Improvements

| Model | Best Config | Return Improvement (%) |
|-------|-------------|------------------------|
| HMA | Accurate | +43.61% |
| ZLEMA | Accurate | +41.90% |
| SMA | Optimal | +30.50% |
| TEMA | Optimal | +29.39% |
| WMA | Optimal | +14.47% |

### Best Sharpe Improvements

| Model | Best Config | Sharpe Improvement |
|-------|-------------|--------------------|
| HMA | Accurate | +1.223 |
| ZLEMA | Accurate | +1.223 |
| KAMA | Accurate | +0.817 |
| SMA | Accurate | +0.753 |
| WMA | Accurate | +0.714 |

---

## Configuration Comparison

### Baseline Configuration (504, 21)

- **Description:** Original parameters
- **Avg Return Improvement:** -20.24%
- **Avg Sharpe Improvement:** +0.029
- **Models Improved (Return):** 0/7
- **Models Improved (Sharpe):** 3/7

### Optimal Configuration (252, 42)

- **Description:** Best total returns
- **Avg Return Improvement:** +22.06%
- **Avg Sharpe Improvement:** +0.339
- **Models Improved (Return):** 7/7
- **Models Improved (Sharpe):** 7/7

### Accurate Configuration (756, 21)

- **Description:** Best Sharpe ratio
- **Avg Return Improvement:** +6.93%
- **Avg Sharpe Improvement:** +0.865
- **Models Improved (Return):** 4/7
- **Models Improved (Sharpe):** 7/7

---

## Recommendations

1. **For Maximum Returns:** Use **Optimal** configuration
   - Most frequently delivers best returns across alpha models

2. **For Risk-Adjusted Returns:** Use **Accurate** configuration
   - Most frequently delivers best Sharpe ratios across alpha models

3. **Model-Specific Optimization:** Consider configuration on per-model basis
   - Different alpha models may benefit from different HMM parameters

---

## Files Generated

- **Summary Results:** `summary_results.csv`
- **Detailed Results:** `detailed_results.csv`
- **Visualization:** `hmm_impact_analysis.png`
- **This Report:** `HMM_IMPACT_REPORT.md`
