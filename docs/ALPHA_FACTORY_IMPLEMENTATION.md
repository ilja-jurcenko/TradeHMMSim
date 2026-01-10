# Alpha Model Factory - Implementation Summary

## Overview

Extended the backtest system to support JSON-based alpha model configuration, enabling dynamic alpha model selection and parameter control without code changes.

## Implementation Date

January 10, 2026

## Key Components

### 1. AlphaModelFactory (`alpha_model_factory.py`)

Factory pattern implementation for creating alpha models from configuration.

**Features:**
- Registry of all 7 alpha models (SMA, EMA, WMA, HMA, KAMA, TEMA, ZLEMA)
- Type-safe parameter validation
- Flexible creation methods
- Model discovery utilities

**API Methods:**
```python
# Create from config dictionary
create_from_config(config: Dict[str, Any]) -> AlphaModel

# Create from type and parameters
create_from_type(model_type: str, short_window: int, long_window: int) -> AlphaModel

# Get available models
get_available_models() -> List[str]

# Get default parameters
get_default_parameters(model_type: str) -> Dict[str, int]

# Create all models
create_all_models(short_window: int = 50, long_window: int = 200) -> Dict[str, AlphaModel]
```

**Validation:**
- Model type must be recognized
- Parameters must include short_window and long_window
- Windows must be positive integers
- short_window < long_window
- Extra parameters are filtered (graceful handling)

### 2. BacktestEngine Extensions (`backtest.py`)

Enhanced initialization to accept alpha models from configuration.

**New Features:**
- Accept `alpha_config` dict in `__init__()`
- Factory methods for config-based creation
- Backward compatible with direct alpha_model instances

**API Methods:**
```python
# Initialize with config
BacktestEngine(close, alpha_config=config)

# Create from full config
BacktestEngine.from_config(close, config)

# Create from alpha config only
BacktestEngine.from_alpha_config(close, alpha_config)

# Get stored config
engine.get_alpha_config() -> Optional[Dict]
```

### 3. Configuration Format

**New Structure:**
```json
{
  "alpha_model": {
    "type": "SMA",
    "parameters": {
      "short_window": 50,
      "long_window": 200
    }
  }
}
```

**Old Structure (still supported):**
```json
{
  "alpha_model": {
    "short_window": 50,
    "long_window": 200
  }
}
```

### 4. ConfigLoader Updates (`config_loader.py`)

Modified `get_alpha_params()` to return new format:

**Returns:**
```python
{
    'type': 'SMA',
    'parameters': {
        'short_window': 50,
        'long_window': 200
    }
}
```

**Backward Compatibility:**
- Detects old format automatically
- Returns default 'SMA' type if not specified
- No breaking changes to existing code

## Testing

### Test Coverage

**Total: 61 tests (100% pass rate)**

#### test_alpha_model_factory.py (29 tests)
- ✓ Model creation for all 7 types
- ✓ Parameter validation (missing, invalid types, ranges)
- ✓ Error handling (unknown types, invalid values)
- ✓ Edge cases (floats, negatives, equals, case sensitivity)
- ✓ Utility methods (get_available_models, create_all_models)

#### test_backtest_with_config.py (18 tests)
- ✓ Initialization variations (config vs instance)
- ✓ Factory method creation
- ✓ Config loading from JSON files
- ✓ Error handling (missing keys, invalid params)
- ✓ Backtest execution with different models
- ✓ Integration with existing config files

#### test_config_loader.py (14 tests - updated)
- ✓ New alpha_model format parsing
- ✓ Backward compatibility
- ✓ Parameter extraction
- ✓ Default fallbacks

### Running Tests

```bash
# All config-related tests
python3 -m unittest tests.test_config_loader tests.test_alpha_model_factory tests.test_backtest_with_config -v

# Factory tests only
python3 -m unittest tests.test_alpha_model_factory -v

# Integration tests only
python3 -m unittest tests.test_backtest_with_config -v
```

## Examples

### Example 1: Basic Factory Usage

```python
from alpha_model_factory import AlphaModelFactory

# Create from config
alpha_config = {
    'type': 'EMA',
    'parameters': {
        'short_window': 12,
        'long_window': 26
    }
}
model = AlphaModelFactory.create_from_config(alpha_config)

# Create from type
model = AlphaModelFactory.create_from_type('SMA', 50, 200)
```

### Example 2: Backtest with Config

```python
from backtest import BacktestEngine
import yfinance as yf

# Get data
data = yf.download('SPY', start='2020-01-01', end='2025-12-31')
close = data['Close']

# Create engine with alpha config
alpha_config = {
    'type': 'HMA',
    'parameters': {
        'short_window': 10,
        'long_window': 30
    }
}
engine = BacktestEngine(close=close, alpha_config=alpha_config)
results = engine.run(strategy_mode='alpha_only')
```

### Example 3: Test All Models

```python
from alpha_model_factory import AlphaModelFactory
from backtest import BacktestEngine

models = AlphaModelFactory.get_available_models()
# Returns: ['SMA', 'EMA', 'WMA', 'HMA', 'KAMA', 'TEMA', 'ZLEMA']

for model_type in models:
    alpha_config = {
        'type': model_type,
        'parameters': {
            'short_window': 50,
            'long_window': 200
        }
    }
    engine = BacktestEngine(close=close, alpha_config=alpha_config)
    result = engine.run(strategy_mode='alpha_only')
    print(f"{model_type}: {result['metrics']['total_return']*100:.2f}%")
```

### Example 4: Load from JSON File

```python
from config_loader import ConfigLoader
from backtest import BacktestEngine

# Load config
config = ConfigLoader.load_config('config_optimal.json')

# Create engine from config
engine = BacktestEngine.from_config(close, config)
results = engine.run(strategy_mode='alpha_only')
```

## Files Modified

### New Files
- `alpha_model_factory.py` (224 lines)
- `tests/test_alpha_model_factory.py` (434 lines)
- `tests/test_backtest_with_config.py` (328 lines)
- `examples/example_alpha_factory.py` (261 lines)

### Modified Files
- `backtest.py` (+76 lines)
- `config_loader.py` (+16 lines)
- `config_default.json` (updated structure)
- `config_optimal.json` (updated structure)
- `config_accurate.json` (updated structure)
- `docs/CONFIG_GUIDE.md` (+120 lines)
- `tests/test_config_loader.py` (+6 lines)

## Benefits

### 1. Flexibility
- Change alpha model without code modification
- Test different models via JSON config
- Easy A/B testing of strategies

### 2. Maintainability
- Centralized model creation logic
- Consistent validation across all models
- Clear separation of concerns

### 3. Extensibility
- Easy to add new alpha models
- Factory pattern supports future enhancements
- Prepared for models with additional parameters

### 4. Type Safety
- Parameter validation at creation time
- Clear error messages
- Prevents runtime errors

### 5. Backward Compatibility
- Existing code continues to work
- Optional adoption of new features
- Gradual migration path

## Usage Scenarios

### Scenario 1: Research Different Models
```bash
# Test SMA
python3 run_comparison.py SPY 2020-01-01 2025-12-31 --config config_sma.json

# Test EMA
python3 run_comparison.py SPY 2020-01-01 2025-12-31 --config config_ema.json

# Test HMA
python3 run_comparison.py SPY 2020-01-01 2025-12-31 --config config_hma.json
```

### Scenario 2: Batch Testing
```python
from alpha_model_factory import AlphaModelFactory

models = AlphaModelFactory.get_available_models()
results = {}

for model_type in models:
    config = {
        'type': model_type,
        'parameters': {'short_window': 10, 'long_window': 30}
    }
    # Run backtest and store results
    results[model_type] = run_backtest(config)
```

### Scenario 3: Parameter Sweep
```python
windows = [(10, 30), (20, 50), (50, 200)]

for short, long in windows:
    config = {
        'type': 'SMA',
        'parameters': {
            'short_window': short,
            'long_window': long
        }
    }
    # Test configuration
    test_backtest(config)
```

## Future Enhancements

### Potential Extensions

1. **Additional Parameters**
   - Support model-specific parameters (e.g., KAMA fast/slow EMA)
   - Extend factory to handle varying parameter schemas

2. **Model Combinations**
   - Ensemble strategies
   - Model rotation based on market conditions

3. **Parameter Optimization**
   - Integrate with scipy.optimize
   - Grid search over parameter space

4. **Model Registry**
   - Plugin architecture for custom models
   - Dynamic model loading

5. **Config Validation**
   - JSON schema validation
   - Pre-flight checks before execution

## Documentation

### Updated Documentation
- `docs/CONFIG_GUIDE.md`: Added alpha model factory section
- `examples/example_alpha_factory.py`: 5 usage examples
- Inline docstrings for all public methods

### Additional Resources
- Unit tests serve as usage examples
- Integration tests demonstrate real-world scenarios
- Example scripts show common patterns

## Migration Guide

### For Existing Code

**Before:**
```python
from alpha_models.sma import SMA

alpha_model = SMA(short_window=50, long_window=200)
engine = BacktestEngine(close, alpha_model)
```

**After (Option 1 - Direct):**
```python
from alpha_model_factory import AlphaModelFactory

model = AlphaModelFactory.create_from_type('SMA', 50, 200)
engine = BacktestEngine(close, alpha_model=model)
```

**After (Option 2 - Config):**
```python
alpha_config = {
    'type': 'SMA',
    'parameters': {'short_window': 50, 'long_window': 200}
}
engine = BacktestEngine(close, alpha_config=alpha_config)
```

**After (Option 3 - JSON File):**
```python
from config_loader import ConfigLoader

config = ConfigLoader.load_config('my_config.json')
engine = BacktestEngine.from_config(close, config)
```

### For Config Files

**Add type field to alpha_model section:**

```json
"alpha_model": {
  "type": "SMA",
  "parameters": {
    "short_window": 50,
    "long_window": 200
  }
}
```

Old format still works but is deprecated.

## Summary

Successfully implemented a flexible, type-safe, and extensible alpha model configuration system. The factory pattern enables dynamic model selection from JSON configuration while maintaining full backward compatibility. Comprehensive test coverage (61 tests) ensures reliability. Documentation and examples facilitate adoption.

**Key Achievement:** Alpha model type is now fully configurable via JSON, enabling external control of all backtest hyperparameters including model selection.
