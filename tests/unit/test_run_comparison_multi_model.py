"""
Unit tests for run_comparison with multiple alpha models configuration.
"""

import unittest
import os
import sys
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the function to test
import run_comparison
from alpha_models import SMA, EMA, WMA, KAMA
from config_loader import ConfigLoader


class TestRunComparisonMultiModel(unittest.TestCase):
    """Test run_comparison with multiple alpha model configurations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample multi-model config
        self.multi_model_config = {
            "backtest": {
                "initial_capital": 100000.0,
                "transaction_cost": 0.001,
                "rebalance_frequency": 1
            },
            "strategy": {
                "strategy_mode": "alpha_hmm_combine",
                "walk_forward": True
            },
            "hmm": {
                "n_states": 3,
                "random_state": 42,
                "train_window": 252,
                "refit_every": 42,
                "bear_prob_threshold": 0.65,
                "bull_prob_threshold": 0.65
            },
            "alpha_models": [
                {
                    "type": "SMA",
                    "parameters": {
                        "short_window": 10,
                        "long_window": 30
                    }
                },
                {
                    "type": "EMA",
                    "parameters": {
                        "short_window": 12,
                        "long_window": 26
                    }
                }
            ],
            "data": {
                "ticker": "SPY",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31"
            },
            "output": {
                "save_plots": False,
                "output_dir": None
            }
        }
        
        # Create sample single-model config (legacy format)
        self.single_model_config = {
            "backtest": {
                "initial_capital": 100000.0,
                "transaction_cost": 0.001,
                "rebalance_frequency": 1
            },
            "strategy": {
                "strategy_mode": "alpha_only",
                "walk_forward": False
            },
            "hmm": {
                "n_states": 3,
                "random_state": 42,
                "train_window": 252,
                "refit_every": 42,
                "bear_prob_threshold": 0.65,
                "bull_prob_threshold": 0.65
            },
            "alpha_model": {
                "type": "SMA",
                "parameters": {
                    "short_window": 10,
                    "long_window": 30
                }
            },
            "data": {
                "ticker": "SPY",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31"
            },
            "output": {
                "save_plots": False,
                "output_dir": None
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _save_config(self, config, filename='test_config.json'):
        """Helper to save config to temp file."""
        config_path = os.path.join(self.temp_dir, filename)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return config_path
    
    def test_multi_model_config_parsing(self):
        """Test that multi-model config is parsed correctly."""
        config_path = self._save_config(self.multi_model_config)
        
        # Load config
        config = ConfigLoader.load_config(config_path)
        
        # Verify alpha_models list exists
        self.assertIn('alpha_models', config)
        self.assertEqual(len(config['alpha_models']), 2)
        
        # Verify first model
        self.assertEqual(config['alpha_models'][0]['type'], 'SMA')
        self.assertEqual(config['alpha_models'][0]['parameters']['short_window'], 10)
        self.assertEqual(config['alpha_models'][0]['parameters']['long_window'], 30)
        
        # Verify second model
        self.assertEqual(config['alpha_models'][1]['type'], 'EMA')
        self.assertEqual(config['alpha_models'][1]['parameters']['short_window'], 12)
        self.assertEqual(config['alpha_models'][1]['parameters']['long_window'], 26)
    
    def test_single_model_config_backward_compatibility(self):
        """Test that single alpha_model config still works (backward compatibility)."""
        config_path = self._save_config(self.single_model_config)
        
        # Load config
        config = ConfigLoader.load_config(config_path)
        
        # Verify alpha_model exists
        self.assertIn('alpha_model', config)
        self.assertEqual(config['alpha_model']['type'], 'SMA')
        self.assertEqual(config['alpha_model']['parameters']['short_window'], 10)
    
    def test_multi_model_different_parameters(self):
        """Test multiple models with different parameters."""
        config = {
            **self.multi_model_config,
            "alpha_models": [
                {"type": "SMA", "parameters": {"short_window": 5, "long_window": 20}},
                {"type": "EMA", "parameters": {"short_window": 10, "long_window": 30}},
                {"type": "WMA", "parameters": {"short_window": 8, "long_window": 25}},
                {"type": "KAMA", "parameters": {"short_window": 12, "long_window": 35}}
            ]
        }
        config_path = self._save_config(config)
        
        loaded_config = ConfigLoader.load_config(config_path)
        
        # Verify all models loaded
        self.assertEqual(len(loaded_config['alpha_models']), 4)
        
        # Verify each has different params
        params = [m['parameters'] for m in loaded_config['alpha_models']]
        self.assertEqual(params[0]['short_window'], 5)
        self.assertEqual(params[1]['short_window'], 10)
        self.assertEqual(params[2]['short_window'], 8)
        self.assertEqual(params[3]['short_window'], 12)
    
    def test_config_with_all_model_types(self):
        """Test config with all available alpha model types."""
        config = {
            **self.multi_model_config,
            "alpha_models": [
                {"type": "SMA", "parameters": {"short_window": 10, "long_window": 30}},
                {"type": "EMA", "parameters": {"short_window": 10, "long_window": 30}},
                {"type": "WMA", "parameters": {"short_window": 10, "long_window": 30}},
                {"type": "HMA", "parameters": {"short_window": 10, "long_window": 30}},
                {"type": "KAMA", "parameters": {"short_window": 10, "long_window": 30}},
                {"type": "TEMA", "parameters": {"short_window": 10, "long_window": 30}},
                {"type": "ZLEMA", "parameters": {"short_window": 10, "long_window": 30}}
            ]
        }
        config_path = self._save_config(config)
        
        loaded_config = ConfigLoader.load_config(config_path)
        
        # Verify all 7 models loaded
        self.assertEqual(len(loaded_config['alpha_models']), 7)
        
        # Verify all types
        types = [m['type'] for m in loaded_config['alpha_models']]
        expected_types = ['SMA', 'EMA', 'WMA', 'HMA', 'KAMA', 'TEMA', 'ZLEMA']
        self.assertEqual(types, expected_types)
    
    def test_invalid_model_type_in_list(self):
        """Test that invalid model type in list raises error."""
        config = {
            **self.multi_model_config,
            "alpha_models": [
                {"type": "SMA", "parameters": {"short_window": 10, "long_window": 30}},
                {"type": "INVALID", "parameters": {"short_window": 10, "long_window": 30}}
            ]
        }
        config_path = self._save_config(config)
        
        # Should raise ValueError when trying to use config
        with self.assertRaises(ValueError) as context:
            run_comparison.run_comparison(
                ticker='SPY',
                start_date='2023-01-01',
                end_date='2023-01-31',
                config_path=config_path,
                save_plots=False
            )
        
        self.assertIn('Unknown model type', str(context.exception))
    
    def test_empty_alpha_models_list(self):
        """Test that empty alpha_models list uses defaults."""
        config = {
            **self.multi_model_config,
            "alpha_models": []
        }
        config_path = self._save_config(config)
        
        loaded_config = ConfigLoader.load_config(config_path)
        self.assertEqual(len(loaded_config['alpha_models']), 0)
    
    def test_missing_parameters_uses_defaults(self):
        """Test that missing parameters in model config are handled."""
        config = {
            **self.multi_model_config,
            "alpha_models": [
                {"type": "SMA", "parameters": {}},  # Empty parameters
                {"type": "EMA"}  # No parameters key at all
            ]
        }
        config_path = self._save_config(config)
        
        loaded_config = ConfigLoader.load_config(config_path)
        
        # Should load without error
        self.assertEqual(len(loaded_config['alpha_models']), 2)
        self.assertEqual(loaded_config['alpha_models'][0]['parameters'], {})
        self.assertNotIn('parameters', loaded_config['alpha_models'][1])
    
    def test_config_validation_model_params(self):
        """Test validation of model parameters."""
        # Test with invalid parameters
        config = {
            **self.multi_model_config,
            "alpha_models": [
                {
                    "type": "SMA",
                    "parameters": {
                        "short_window": -5,  # Negative (invalid)
                        "long_window": 10
                    }
                }
            ]
        }
        config_path = self._save_config(config)
        
        # Should raise error when creating model (negative windows not allowed)
        with self.assertRaises((ValueError, TypeError)):
            run_comparison.run_comparison(
                ticker='SPY',
                start_date='2023-01-01',
                end_date='2023-01-31',
                config_path=config_path,
                save_plots=False
            )


class TestConfigFormat(unittest.TestCase):
    """Test different configuration formats."""
    
    def test_new_format_structure(self):
        """Test the new alpha_models list format structure."""
        config = {
            "alpha_models": [
                {
                    "type": "SMA",
                    "parameters": {
                        "short_window": 10,
                        "long_window": 30
                    }
                }
            ]
        }
        
        # Verify structure
        self.assertIn('alpha_models', config)
        self.assertIsInstance(config['alpha_models'], list)
        self.assertGreater(len(config['alpha_models']), 0)
        
        first_model = config['alpha_models'][0]
        self.assertIn('type', first_model)
        self.assertIn('parameters', first_model)
        self.assertIsInstance(first_model['parameters'], dict)
    
    def test_legacy_format_structure(self):
        """Test the legacy alpha_model single format structure."""
        config = {
            "alpha_model": {
                "type": "SMA",
                "parameters": {
                    "short_window": 10,
                    "long_window": 30
                }
            }
        }
        
        # Verify structure
        self.assertIn('alpha_model', config)
        self.assertIsInstance(config['alpha_model'], dict)
        self.assertIn('type', config['alpha_model'])
        self.assertIn('parameters', config['alpha_model'])
    
    def test_both_formats_exclusive(self):
        """Test that having both formats is handled correctly."""
        # Config with both formats - should prefer new format
        config = {
            "alpha_model": {
                "type": "SMA",
                "parameters": {"short_window": 10, "long_window": 30}
            },
            "alpha_models": [
                {"type": "EMA", "parameters": {"short_window": 12, "long_window": 26}}
            ]
        }
        
        # Both keys exist, but alpha_models should take precedence
        self.assertIn('alpha_models', config)
        self.assertIn('alpha_model', config)


if __name__ == '__main__':
    unittest.main()
