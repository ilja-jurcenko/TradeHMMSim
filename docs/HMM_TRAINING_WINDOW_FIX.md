# HMM Training Window Fix

## Problem

Previously, when users specified a test period (e.g., 2018-01-01 to 2024-01-01), the HMM walk-forward filter would skip the first `train_window` days because it needed historical data for initial training. This meant:

- **Specified period**: 2018-01-01 to 2024-01-01
- **Actual test period**: Started ~504 days after 2018-01-01 (depending on train_window)
- **Result**: Tests didn't run on the full user-specified date range

## Solution

The system now automatically loads extra historical data before the test period to ensure HMM has enough data for initial training:

1. **Data Loading**: Loads data from (start_date - train_window) to end_date
2. **HMM Training**: Uses the extra historical data for initial model training
3. **Test Period**: Backtest runs exactly on the user-specified date range
4. **No Data Leakage**: Walk-forward approach ensures only past data is used at each point

## Example

User specifies:
```bash
python3 run_comparison.py SPY --start-date 2020-01-01 --end-date 2020-06-30 --config config_default.json
```

With `train_window=504` in config:

**Before Fix**:
- Data loaded: 2020-01-01 to 2020-06-30
- Warning: Not enough data for training window
- Actual test start: ~2021-04-01 (after waiting for 504 days)
- Result: Missing the entire first half of 2020!

**After Fix**:
- Data loaded: 2018-08-15 to 2020-06-30 (504 extra days)
- HMM training: Uses 2018-08-15 to 2020-01-01 for initial training
- Test period: 2020-01-02 to 2020-06-30 (exactly as specified)
- Result: Full test period coverage!

## Date Calculation

```python
from datetime import timedelta
import pandas as pd

# User-specified dates
user_start_date = '2020-01-01'
user_end_date = '2020-06-30'
train_window = 504  # days

# Calculate adjusted start date for data loading
start_date_dt = pd.to_datetime(user_start_date)
adjusted_start_date = start_date_dt - timedelta(days=train_window)
data_load_start_date = adjusted_start_date.strftime('%Y-%m-%d')

# Result: data_load_start_date = '2018-08-15'
```

## Output Information

The system now clearly shows the date ranges:

```
Date range adjustment for HMM training:
  User-specified test period: 2020-01-01 to 2020-06-30
  Data loading period: 2018-08-15 to 2020-06-30
  Extra days for HMM training: 504

Data split:
  Full dataset for HMM training: 472 days (2018-08-15 to 2020-06-30)
  Test period for backtest: 125 days (2020-01-02 to 2020-06-30)
```

## Configuration Files Updated

All configuration files and reports now include:
- `test_start_date`: User-specified test start date
- `test_end_date`: User-specified test end date
- `data_load_start_date`: Adjusted date for loading training data
- `extra_training_days`: Number of days loaded before test period

## Runtime Config Example

```json
{
  "data": {
    "ticker": ["SPY"],
    "test_start_date": "2020-01-01",
    "test_end_date": "2020-06-30",
    "data_load_start_date": "2018-08-15",
    "data_load_end_date": "2020-06-30",
    "extra_training_days": 504
  }
}
```

## Walk-Forward Behavior

The HMM `walkforward_filter` function:

1. Creates features from full dataset (including extra historical data)
2. Starts predictions at `range(train_window, len(feats))`
3. This naturally aligns with user-specified start date since we loaded exactly `train_window` days before it
4. Continues walk-forward testing through the end date

```python
for t in range(train_window, len(feats)):  # Starts at user_start_date
    if ((t - train_window) % refit_every == 0):
        # Refit using previous train_window days
        X_train = feats.iloc[t - train_window : t].values
        model.fit(X_train_scaled)
    
    # Make prediction for time t
    prob_t = compute_filtered_probability(...)
```

## Benefits

1. **Exact Date Ranges**: Tests run on precisely the dates specified by the user
2. **Full Data Utilization**: No wasted data at the beginning of the test period
3. **Proper Training**: HMM has sufficient historical data for initial training
4. **Transparency**: Clear output showing both data loading and test periods
5. **No Surprises**: Results match user expectations for date coverage

## Verification

To verify the fix works, run a short test:

```bash
python3 run_comparison.py SPY \
  --start-date 2020-01-01 \
  --end-date 2020-06-30 \
  --config config_default.json \
  --strategies hmm_only \
  --save-plots
```

Check the output for:
- Data loading period starts ~504 days before test start
- Test period matches exactly what you specified
- No warnings about insufficient data
- Full test period coverage in results

## Impact on Existing Tests

This fix is **backward compatible**:
- Old test results will differ slightly because they now include more data at the start
- Performance metrics may improve since the full test period is covered
- Historical comparisons should use the new methodology going forward
