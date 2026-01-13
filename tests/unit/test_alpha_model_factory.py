"""
Unit tests for alpha model factory.
Tests model creation, parameter validation, and error handling.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha_model_factory import AlphaModelFactory
from alpha_models.sma import SMA
from alpha_models.ema import EMA
from alpha_models.wma import WMA
from alpha_models.hma import HMA
from alpha_models.kama import KAMA
from alpha_models.tema import TEMA
from alpha_models.zlema import ZLEMA


class TestAlphaModelFactory(unittest.TestCase):
    """Test alpha model factory functionality."""
    
    def test_create_sma_from_config(self):
        """Test creating SMA from config."""
        config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 50,
                'long_window': 200
            }
        }
        model = AlphaModelFactory.create_from_config(config)
        
        self.assertIsInstance(model, SMA)
        self.assertEqual(model.short_window, 50)
        self.assertEqual(model.long_window, 200)
    
    def test_create_ema_from_config(self):
        """Test creating EMA from config."""
        config = {
            'type': 'EMA',
            'parameters': {
                'short_window': 12,
                'long_window': 26
            }
        }
        model = AlphaModelFactory.create_from_config(config)
        
        self.assertIsInstance(model, EMA)
        self.assertEqual(model.short_window, 12)
        self.assertEqual(model.long_window, 26)
    
    def test_create_wma_from_config(self):
        """Test creating WMA from config."""
        config = {
            'type': 'WMA',
            'parameters': {
                'short_window': 10,
                'long_window': 30
            }
        }
        model = AlphaModelFactory.create_from_config(config)
        
        self.assertIsInstance(model, WMA)
        self.assertEqual(model.short_window, 10)
        self.assertEqual(model.long_window, 30)
    
    def test_create_hma_from_config(self):
        """Test creating HMA from config."""
        config = {
            'type': 'HMA',
            'parameters': {
                'short_window': 9,
                'long_window': 21
            }
        }
        model = AlphaModelFactory.create_from_config(config)
        
        self.assertIsInstance(model, HMA)
        self.assertEqual(model.short_window, 9)
        self.assertEqual(model.long_window, 21)
    
    def test_create_kama_from_config(self):
        """Test creating KAMA from config."""
        config = {
            'type': 'KAMA',
            'parameters': {
                'short_window': 10,
                'long_window': 30
            }
        }
        model = AlphaModelFactory.create_from_config(config)
        
        self.assertIsInstance(model, KAMA)
        self.assertEqual(model.short_window, 10)
        self.assertEqual(model.long_window, 30)
    
    def test_create_tema_from_config(self):
        """Test creating TEMA from config."""
        config = {
            'type': 'TEMA',
            'parameters': {
                'short_window': 5,
                'long_window': 15
            }
        }
        model = AlphaModelFactory.create_from_config(config)
        
        self.assertIsInstance(model, TEMA)
        self.assertEqual(model.short_window, 5)
        self.assertEqual(model.long_window, 15)
    
    def test_create_zlema_from_config(self):
        """Test creating ZLEMA from config."""
        config = {
            'type': 'ZLEMA',
            'parameters': {
                'short_window': 20,
                'long_window': 50
            }
        }
        model = AlphaModelFactory.create_from_config(config)
        
        self.assertIsInstance(model, ZLEMA)
        self.assertEqual(model.short_window, 20)
        self.assertEqual(model.long_window, 50)
    
    def test_create_from_type(self):
        """Test creating model from type and parameters."""
        model = AlphaModelFactory.create_from_type('EMA', 12, 26)
        
        self.assertIsInstance(model, EMA)
        self.assertEqual(model.short_window, 12)
        self.assertEqual(model.long_window, 26)
    
    def test_missing_type_key(self):
        """Test error when 'type' key is missing."""
        config = {
            'parameters': {
                'short_window': 50,
                'long_window': 200
            }
        }
        
        with self.assertRaises(KeyError) as context:
            AlphaModelFactory.create_from_config(config)
        
        self.assertIn('type', str(context.exception))
    
    def test_missing_parameters_key(self):
        """Test error when 'parameters' key is missing."""
        config = {
            'type': 'SMA'
        }
        
        with self.assertRaises(KeyError) as context:
            AlphaModelFactory.create_from_config(config)
        
        self.assertIn('parameters', str(context.exception))
    
    def test_unknown_model_type(self):
        """Test error for unknown model type."""
        config = {
            'type': 'UNKNOWN_MODEL',
            'parameters': {
                'short_window': 50,
                'long_window': 200
            }
        }
        
        with self.assertRaises(ValueError) as context:
            AlphaModelFactory.create_from_config(config)
        
        self.assertIn('Unknown alpha model type', str(context.exception))
        self.assertIn('UNKNOWN_MODEL', str(context.exception))
    
    def test_missing_short_window(self):
        """Test error when short_window is missing."""
        config = {
            'type': 'SMA',
            'parameters': {
                'long_window': 200
            }
        }
        
        with self.assertRaises(ValueError) as context:
            AlphaModelFactory.create_from_config(config)
        
        self.assertIn('short_window', str(context.exception))
    
    def test_missing_long_window(self):
        """Test error when long_window is missing."""
        config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 50
            }
        }
        
        with self.assertRaises(ValueError) as context:
            AlphaModelFactory.create_from_config(config)
        
        self.assertIn('long_window', str(context.exception))
    
    def test_invalid_short_window_type(self):
        """Test error for invalid short_window type."""
        config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 'fifty',
                'long_window': 200
            }
        }
        
        with self.assertRaises(ValueError) as context:
            AlphaModelFactory.create_from_config(config)
        
        self.assertIn('short_window', str(context.exception))
    
    def test_invalid_long_window_type(self):
        """Test error for invalid long_window type."""
        config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 50,
                'long_window': 'two hundred'
            }
        }
        
        with self.assertRaises(ValueError) as context:
            AlphaModelFactory.create_from_config(config)
        
        self.assertIn('long_window', str(context.exception))
    
    def test_negative_short_window(self):
        """Test error for negative short_window."""
        config = {
            'type': 'SMA',
            'parameters': {
                'short_window': -10,
                'long_window': 200
            }
        }
        
        with self.assertRaises(ValueError) as context:
            AlphaModelFactory.create_from_config(config)
        
        self.assertIn('positive integer', str(context.exception))
    
    def test_zero_long_window(self):
        """Test error for zero long_window."""
        config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 50,
                'long_window': 0
            }
        }
        
        with self.assertRaises(ValueError) as context:
            AlphaModelFactory.create_from_config(config)
        
        self.assertIn('positive integer', str(context.exception))
    
    def test_short_window_greater_than_long(self):
        """Test error when short_window >= long_window."""
        config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 200,
                'long_window': 50
            }
        }
        
        with self.assertRaises(ValueError) as context:
            AlphaModelFactory.create_from_config(config)
        
        self.assertIn('must be less than', str(context.exception))
    
    def test_short_window_equals_long(self):
        """Test error when short_window == long_window."""
        config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 100,
                'long_window': 100
            }
        }
        
        with self.assertRaises(ValueError) as context:
            AlphaModelFactory.create_from_config(config)
        
        self.assertIn('must be less than', str(context.exception))
    
    def test_get_available_models(self):
        """Test getting list of available models."""
        models = AlphaModelFactory.get_available_models()
        
        self.assertIsInstance(models, list)
        self.assertIn('SMA', models)
        self.assertIn('EMA', models)
        self.assertIn('WMA', models)
        self.assertIn('HMA', models)
        self.assertIn('KAMA', models)
        self.assertIn('TEMA', models)
        self.assertIn('ZLEMA', models)
        self.assertIn('BollingerBands', models)
        self.assertEqual(len(models), 8)
    
    def test_get_default_parameters(self):
        """Test getting default parameters."""
        params = AlphaModelFactory.get_default_parameters('SMA')
        
        self.assertIsInstance(params, dict)
        self.assertIn('short_window', params)
        self.assertIn('long_window', params)
        self.assertEqual(params['short_window'], 50)
        self.assertEqual(params['long_window'], 200)
    
    def test_get_default_parameters_invalid_type(self):
        """Test error for invalid model type in get_default_parameters."""
        with self.assertRaises(ValueError) as context:
            AlphaModelFactory.get_default_parameters('INVALID')
        
        self.assertIn('Unknown model type', str(context.exception))
    
    def test_create_all_models(self):
        """Test creating all models with same parameters."""
        models = AlphaModelFactory.create_all_models(short_window=10, long_window=30)
        
        self.assertIsInstance(models, dict)
        self.assertEqual(len(models), 8)
        
        # Check all models are present and have correct parameters
        for model_type in ['SMA', 'EMA', 'WMA', 'HMA', 'KAMA', 'TEMA', 'ZLEMA', 'BollingerBands']:
            self.assertIn(model_type, models)
            self.assertEqual(models[model_type].short_window, 10)
            self.assertEqual(models[model_type].long_window, 30)
    
    def test_create_all_models_default_params(self):
        """Test creating all models with default parameters."""
        models = AlphaModelFactory.create_all_models()
        
        self.assertEqual(len(models), 8)
        
        # Check default parameters
        for model in models.values():
            self.assertEqual(model.short_window, 50)
            self.assertEqual(model.long_window, 200)


class TestAlphaModelFactoryEdgeCases(unittest.TestCase):
    """Test edge cases for alpha model factory."""
    
    def test_float_windows_converted_to_int(self):
        """Test that float windows are rejected (not auto-converted)."""
        config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 50.5,
                'long_window': 200.5
            }
        }
        
        with self.assertRaises(ValueError):
            AlphaModelFactory.create_from_config(config)
    
    def test_very_small_windows(self):
        """Test creation with minimum valid windows."""
        config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 1,
                'long_window': 2
            }
        }
        
        model = AlphaModelFactory.create_from_config(config)
        self.assertEqual(model.short_window, 1)
        self.assertEqual(model.long_window, 2)
    
    def test_very_large_windows(self):
        """Test creation with very large windows."""
        config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 1000,
                'long_window': 5000
            }
        }
        
        model = AlphaModelFactory.create_from_config(config)
        self.assertEqual(model.short_window, 1000)
        self.assertEqual(model.long_window, 5000)
    
    def test_case_sensitive_model_type(self):
        """Test that model type is case-sensitive."""
        config = {
            'type': 'sma',  # lowercase
            'parameters': {
                'short_window': 50,
                'long_window': 200
            }
        }
        
        with self.assertRaises(ValueError) as context:
            AlphaModelFactory.create_from_config(config)
        
        self.assertIn('Unknown alpha model type', str(context.exception))
    
    def test_extra_parameters_ignored(self):
        """Test that extra parameters don't cause errors."""
        config = {
            'type': 'SMA',
            'parameters': {
                'short_window': 50,
                'long_window': 200,
                'extra_param': 'ignored'
            }
        }
        
        # Should succeed and ignore extra parameter
        model = AlphaModelFactory.create_from_config(config)
        self.assertIsInstance(model, SMA)


if __name__ == '__main__':
    unittest.main()
