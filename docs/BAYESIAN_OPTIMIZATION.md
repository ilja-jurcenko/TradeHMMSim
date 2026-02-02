# Bayesian Optimization for HMM Parameters

## Overview

This document describes the Bayesian optimization approach implemented in `optimize_hmm_params.py` for finding optimal HMM parameters.

## Why Bayesian Optimization?

### Problem with Grid Search
- **Exponential complexity**: With 8 parameters, exhaustive grid search requires millions of combinations
- **Inefficient sampling**: Tests many poor parameter combinations
- **No learning**: Each evaluation is independent, doesn't use previous results
- **Time consuming**: 56 combinations took ~30 minutes

### Advantages of Bayesian Optimization
1. **Intelligent exploration**: Uses Gaussian Process to model objective function
2. **Exploitation vs exploration**: Balances testing promising regions vs discovering new areas
3. **Efficient**: Achieves good results with ~50-100 evaluations instead of millions
4. **Adaptive**: Learns from each evaluation to choose better parameters
5. **Handles noise**: Robust to stochastic evaluation metrics

## Parameters Being Optimized

The optimization searches over 8 parameters simultaneously:

### Training Windows
- `train_window`: [300, 1500] - Number of historical days for HMM training
- `refit_every`: [5, 126] - Days between model refitting

### Feature Parameters
- `short_vol_window`: [5, 30] - Short-term volatility window
- `long_vol_window`: [20, 100] - Long-term volatility window
- `short_ma_window`: [5, 30] - Short moving average window
- `long_ma_window`: [20, 100] - Long moving average window

### Regime Thresholds
- `bear_prob_threshold`: [0.50, 0.80] - Minimum probability for bear regime
- `bull_prob_threshold`: [0.50, 0.80] - Minimum probability for bull regime

## Objective Function

### Composite Score
The optimizer maximizes (minimizes negative) a weighted combination of metrics:

```
Score = 0.40 × Balanced Accuracy +
        0.30 × Cohen's Kappa (scaled) +
        0.15 × Switch Precision +
        0.15 × Switch Recall
```

### Metric Scaling
- Balanced Accuracy: [0, 100] → [0, 100]
- Cohen's Kappa: [-1, 1] → [0, 100] via `50 + 50×kappa`
- Switch Precision/Recall: [0, 100] → [0, 100]

## Implementation Details

### Algorithm: Gaussian Process Optimization
- **Method**: `gp_minimize` from scikit-optimize
- **Surrogate model**: Gaussian Process
- **Acquisition function**: Expected Improvement (default)
- **Initial sampling**: 10 random points for exploration
- **Random state**: Fixed at 42 for reproducibility

### Search Configuration
```python
from skopt import gp_minimize
from skopt.space import Integer, Real

result = gp_minimize(
    objective_function,
    space=[
        Integer(300, 1500, name='train_window'),
        Integer(5, 126, name='refit_every'),
        # ... other parameters
    ],
    n_calls=50,              # Total evaluations
    n_random_starts=10,      # Initial random exploration
    random_state=42,
    n_jobs=1                 # Sequential execution
)
```

## Usage

### Basic Usage
```bash
python optimize_hmm_params.py --n-calls 50
```

### Advanced Options
```bash
python optimize_hmm_params.py \
    --config config_optimal.json \
    --start-date 2020-01-01 \
    --end-date 2024-12-31 \
    --threshold-pct 0.05 \
    --n-calls 100
```

### Parameters
- `--config`: Base configuration file (default: config_optimal.json)
- `--start-date`: Evaluation start date (default: 2020-01-01)
- `--end-date`: Evaluation end date (default: 2024-12-31)
- `--threshold-pct`: Zigzag threshold for regime labeling (default: 0.05)
- `--n-calls`: Optimization budget - number of evaluations (default: 50)

## Output

### Results CSV
Saved to `results/hmm_bayesian_search_TIMESTAMP.csv` with columns:
- All 8 parameter values
- `composite_score`: Overall optimization objective
- `balanced_accuracy`: Balanced regime classification accuracy
- `kappa`: Cohen's Kappa score
- `switch_precision`: Precision of regime switches
- `switch_recall`: Recall of regime switches
- Additional metrics: overall_accuracy, true_switches, hmm_switches

### Optimization State
Saved to `results/hmm_bayesian_opt_TIMESTAMP.pkl`:
- Complete optimization history
- Gaussian Process model state
- Can be loaded with `skopt.load()` for analysis

### Auto-Configuration Update
The script automatically updates the base config file with the best parameters found.

## Expected Runtime

### Timing Estimates
- Each evaluation: ~30-60 seconds (depends on training window size)
- 30 evaluations: ~15-30 minutes
- 50 evaluations: ~25-50 minutes
- 100 evaluations: ~50-100 minutes

### Convergence
- Usually converges to good solution within 30-50 evaluations
- Diminishing returns after 50-100 evaluations
- Can monitor progress in terminal output

## Optimization Strategy

### Phase 1: Random Exploration (n_random_starts=10)
- First 10 evaluations are random
- Explores diverse regions of parameter space
- Builds initial GP model

### Phase 2: Guided Search (remaining evaluations)
- Uses GP model to predict promising parameters
- Balances exploitation (testing near best) vs exploration (finding new regions)
- Adaptively focuses on high-potential areas

### Phase 3: Refinement
- Later evaluations fine-tune best parameters
- Smaller parameter variations
- Confirms optimal configuration

## Interpreting Results

### Top Results Table
The script prints the top 5 parameter combinations:
```
1. Composite Score: 45.23
   train_window=400, refit_every=42
   short_vol_window=10, long_vol_window=30
   short_ma_window=10, long_ma_window=30
   bear_prob_threshold=0.66, bull_prob_threshold=0.66
   Metrics: Balanced Acc=60.98%, Kappa=0.174, Switch P/R=20.0%/20.0%
```

### Score Interpretation
- **Score > 45**: Excellent performance
- **Score 40-45**: Good performance
- **Score 35-40**: Moderate performance
- **Score < 35**: Poor performance

## Advanced Analysis

### Loading Optimization Results
```python
from skopt import load
import pandas as pd

# Load optimization state
result = load('results/hmm_bayesian_opt_TIMESTAMP.pkl')

# Load results CSV
df = pd.read_csv('results/hmm_bayesian_search_TIMESTAMP.csv')

# Analyze convergence
from skopt.plots import plot_convergence, plot_objective
plot_convergence(result)
plot_objective(result)
```

### Parameter Importance
```python
from skopt.plots import plot_evaluations

# Visualize parameter effect on objective
plot_evaluations(result)
```

## Comparison with Grid Search

| Aspect | Grid Search | Bayesian Optimization |
|--------|-------------|----------------------|
| Evaluations for 8 params | Millions | 50-100 |
| Runtime | Days/weeks | 30-90 minutes |
| Learning | No | Yes |
| Optimality | Guaranteed* | Probabilistic |
| Efficiency | Very low | High |

*Only if grid is fine enough

## Recommendations

### Quick Testing
- Use `--n-calls 20-30` for quick experiments
- Good for validating approach

### Production Optimization
- Use `--n-calls 50-100` for thorough search
- Recommended for final parameter selection

### Research/Analysis
- Use `--n-calls 100-200` for comprehensive exploration
- Good for understanding parameter landscape

## Next Steps

After optimization:

1. **Validate Results**: Run `hmm_accuracy_evaluation.py` with best parameters
2. **Backtest**: Test on different time periods
3. **Stability Check**: Run multiple optimizations to verify consistency
4. **Parameter Analysis**: Use scikit-optimize plotting tools
5. **Production Deployment**: Update production configs with best parameters

## References

- [Scikit-optimize documentation](https://scikit-optimize.github.io/)
- [Bayesian Optimization review](https://arxiv.org/abs/1807.02811)
- Original paper: Mockus, J. (1975). "On Bayesian methods for seeking the extremum"
