# Test Organization

This directory contains all tests for the TradeHMMSim project, organized into unit and integration tests.

## Structure

```
tests/
├── unit/              # Fast tests with mocked dependencies
│   ├── test_alpha_model_factory.py
│   ├── test_alpha_models/
│   ├── test_backtest.py
│   ├── test_backtest_with_config.py
│   ├── test_bug_fixes.py
│   ├── test_cached_loader.py
│   ├── test_config_loader.py
│   ├── test_kama_fix.py
│   ├── test_loaders.py
│   ├── test_plotter.py
│   └── test_signal_filter/
│
├── integration/       # Slower tests with real data and network calls
│   ├── test_cached_loader_real.py
│   └── test_loader_integration.py
│
├── run_unit_tests.py          # Run only unit tests
├── run_integration_tests.py   # Run only integration tests
└── run_all_tests.py           # Run all tests

```

## Running Tests

### Unit Tests Only (Fast)
Unit tests use mocked dependencies and don't require network access. They run in ~3 seconds.

```bash
# From project root
python3 tests/run_unit_tests.py

# Or directly with unittest
python3 -m unittest discover -s tests/unit -p 'test_*.py'
```

### Integration Tests Only (Slower)
Integration tests use real yfinance data and require network access. They may take 5-10 seconds.

```bash
# From project root
python3 tests/run_integration_tests.py

# Or directly with unittest
python3 -m unittest discover -s tests/integration -p 'test_*.py'
```

### All Tests
Run both unit and integration tests.

```bash
# From project root
python3 tests/run_all_tests.py

# Or directly with unittest (from project root)
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Test Guidelines

### Unit Tests (`tests/unit/`)
- **Purpose**: Verify individual components in isolation
- **Characteristics**:
  - Fast execution (< 5 seconds total)
  - No network calls
  - Use mocked dependencies
  - Deterministic results
  - Can run offline
- **Use for**: 
  - Algorithm correctness
  - Edge case handling
  - Error handling
  - Configuration validation
  - Component interfaces

### Integration Tests (`tests/integration/`)
- **Purpose**: Verify components work correctly with real data
- **Characteristics**:
  - Slower execution (5-10 seconds)
  - Make real network calls to yfinance
  - Use actual data sources
  - May have timing variability
  - Require internet connection
- **Use for**:
  - Data loading from yfinance
  - End-to-end workflows
  - Performance validation
  - Real-world scenarios

## Test Coverage

### Components Tested

#### Alpha Models
- `test_alpha_models/`: Unit tests for all moving average implementations
- `test_alpha_model_factory.py`: Factory pattern and model creation

#### Data Loading
- `test_loaders.py`: Unit tests with mocked loaders
- `test_cached_loader.py`: Unit tests for cached loader logic
- `test_loader_integration.py`: Integration tests with real yfinance
- `test_cached_loader_real.py`: Integration tests for caching behavior

#### Backtesting
- `test_backtest.py`: Unit tests for backtest engine
- `test_backtest_with_config.py`: Configuration-based backtesting

#### Signal Filtering
- `test_signal_filter/`: HMM regime filter tests

#### Configuration
- `test_config_loader.py`: Configuration loading and validation

#### Visualization
- `test_plotter.py`: Plotting functionality

#### Bug Fixes
- `test_bug_fixes.py`: Regression tests for fixed bugs
- `test_kama_fix.py`: Specific KAMA calculation fixes

## Writing New Tests

### Adding a Unit Test

1. Create test file in `tests/unit/`
2. Use mocks for external dependencies
3. Keep tests fast (< 0.1s each)
4. Make tests deterministic

Example:
```python
import unittest
from unittest.mock import Mock
from mymodule import MyClass

class TestMyClass(unittest.TestCase):
    def test_my_function(self):
        mock_data = Mock()
        result = MyClass().process(mock_data)
        self.assertEqual(result, expected_value)
```

### Adding an Integration Test

1. Create test file in `tests/integration/`
2. Use real data sources
3. Clean up any created resources
4. Be tolerant of timing variations

Example:
```python
import unittest
from loaders import YFinanceLoader

class TestRealData(unittest.TestCase):
    def test_load_real_ticker(self):
        loader = YFinanceLoader()
        data = loader.load_ticker('SPY', '2024-01-01', '2024-01-31')
        self.assertGreater(len(data), 0)
```

## CI/CD Considerations

For continuous integration:
- Run unit tests on every commit
- Run integration tests on pull requests
- Consider running integration tests on a schedule due to network dependency
- Set appropriate timeouts for integration tests

## Troubleshooting

### Tests Failing with Network Errors
- Integration tests require internet connection
- Check if yfinance is accessible
- Some failures may be due to market holidays (no data available)

### Cache Issues
- Integration tests may create temporary cache directories
- Clean up with: `rm -rf ./test_cache*`
- Tests should clean up after themselves in `tearDown()`

### Import Errors
- Make sure to run tests from project root
- Ensure all dependencies are installed: `pip install -r requirements.txt`

## Test Statistics

- **Total Tests**: 165
- **Unit Tests**: 158 (96%)
- **Integration Tests**: 7 (4%)
- **Typical Runtime**:
  - Unit tests: ~3 seconds
  - Integration tests: ~5 seconds
  - All tests: ~8 seconds
