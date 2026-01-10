"""
Unit tests for configuration loader.
"""

import unittest
import json
import os
import tempfile
from config_loader import ConfigLoader


class TestConfigLoader(unittest.TestCase):
    """Test cases for ConfigLoader class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def test_load_default_config(self):
        """Test loading default configuration file."""
        config = ConfigLoader.load_config('config_default.json')
        
        # Verify main sections exist
        self.assertIn('backtest', config)
        self.assertIn('strategy', config)
        self.assertIn('hmm', config)
        self.assertIn('alpha_model', config)
        self.assertIn('data', config)
        self.assertIn('output', config)
        
        # Verify default values
        self.assertEqual(config['hmm']['train_window'], 504)
        self.assertEqual(config['hmm']['refit_every'], 21)
        self.assertEqual(config['hmm']['n_states'], 3)
    
    def test_load_optimal_config(self):
        """Test loading optimal configuration file."""
        config = ConfigLoader.load_config('config_optimal.json')
        
        # Verify optimal parameters
        self.assertEqual(config['hmm']['train_window'], 252)
        self.assertEqual(config['hmm']['refit_every'], 42)
        self.assertEqual(config['strategy']['strategy_mode'], 'alpha_hmm_combine')
        self.assertEqual(config['strategy']['walk_forward'], True)
    
    def test_load_accurate_config(self):
        """Test loading accurate configuration file."""
        config = ConfigLoader.load_config('config_accurate.json')
        
        # Verify accurate parameters
        self.assertEqual(config['hmm']['train_window'], 756)
        self.assertEqual(config['hmm']['refit_every'], 21)
    
    def test_load_nonexistent_config(self):
        """Test loading nonexistent configuration file."""
        with self.assertRaises(FileNotFoundError):
            ConfigLoader.load_config('nonexistent_config.json')
    
    def test_get_backtest_params(self):
        """Test extracting backtest parameters."""
        config = ConfigLoader.load_config('config_optimal.json')
        params = ConfigLoader.get_backtest_params(config)
        
        # Verify all required parameters are present
        required_params = [
            'strategy_mode', 'rebalance_frequency', 'walk_forward',
            'train_window', 'refit_every', 'bear_prob_threshold',
            'bull_prob_threshold', 'transaction_cost'
        ]
        
        for param in required_params:
            self.assertIn(param, params)
        
        # Verify parameter types and values
        self.assertIsInstance(params['strategy_mode'], str)
        self.assertIsInstance(params['rebalance_frequency'], int)
        self.assertIsInstance(params['walk_forward'], bool)
        self.assertIsInstance(params['train_window'], int)
        self.assertIsInstance(params['refit_every'], int)
        self.assertIsInstance(params['bear_prob_threshold'], float)
        self.assertIsInstance(params['bull_prob_threshold'], float)
        self.assertIsInstance(params['transaction_cost'], float)
        
        # Verify optimal values
        self.assertEqual(params['train_window'], 252)
        self.assertEqual(params['refit_every'], 42)
        self.assertEqual(params['strategy_mode'], 'alpha_hmm_combine')
    
    def test_get_hmm_params(self):
        """Test extracting HMM parameters."""
        config = ConfigLoader.load_config('config_default.json')
        params = ConfigLoader.get_hmm_params(config)
        
        # Verify required HMM parameters
        self.assertIn('n_states', params)
        self.assertIn('random_state', params)
        
        # Verify types and values
        self.assertEqual(params['n_states'], 3)
        self.assertEqual(params['random_state'], 42)
    
    def test_get_alpha_params(self):
        """Test extracting alpha model parameters."""
        config = ConfigLoader.load_config('config_optimal.json')
        params = ConfigLoader.get_alpha_params(config)
        
        # Verify required alpha parameters
        self.assertIn('short_window', params)
        self.assertIn('long_window', params)
        
        # Verify types and values
        self.assertIsInstance(params['short_window'], int)
        self.assertIsInstance(params['long_window'], int)
        self.assertEqual(params['short_window'], 10)
        self.assertEqual(params['long_window'], 30)
    
    def test_merge_configs(self):
        """Test merging configuration dictionaries."""
        base_config = {
            'backtest': {
                'initial_capital': 100000.0,
                'transaction_cost': 0.001
            },
            'hmm': {
                'train_window': 504,
                'refit_every': 21
            }
        }
        
        override_config = {
            'backtest': {
                'transaction_cost': 0.002
            },
            'hmm': {
                'train_window': 252
            },
            'data': {
                'ticker': 'QQQ'
            }
        }
        
        merged = ConfigLoader.merge_configs(base_config, override_config)
        
        # Verify override values
        self.assertEqual(merged['backtest']['transaction_cost'], 0.002)
        self.assertEqual(merged['hmm']['train_window'], 252)
        
        # Verify base values that weren't overridden
        self.assertEqual(merged['backtest']['initial_capital'], 100000.0)
        self.assertEqual(merged['hmm']['refit_every'], 21)
        
        # Verify new values from override
        self.assertEqual(merged['data']['ticker'], 'QQQ')
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        # Create a temporary config
        test_config = {
            'backtest': {
                'initial_capital': 50000.0,
                'transaction_cost': 0.0005
            },
            'hmm': {
                'train_window': 126,
                'refit_every': 10
            }
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            ConfigLoader.save_config(test_config, temp_path)
            
            # Verify file exists
            self.assertTrue(os.path.exists(temp_path))
            
            # Load it back
            loaded_config = ConfigLoader.load_config(temp_path)
            
            # Verify contents match
            self.assertEqual(loaded_config['backtest']['initial_capital'], 50000.0)
            self.assertEqual(loaded_config['backtest']['transaction_cost'], 0.0005)
            self.assertEqual(loaded_config['hmm']['train_window'], 126)
            self.assertEqual(loaded_config['hmm']['refit_every'], 10)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_config_completeness(self):
        """Test that all config files have required sections."""
        config_files = ['config_default.json', 'config_optimal.json', 'config_accurate.json']
        required_sections = ['backtest', 'strategy', 'hmm', 'alpha_model', 'data', 'output']
        
        for config_file in config_files:
            with self.subTest(config_file=config_file):
                config = ConfigLoader.load_config(config_file)
                
                for section in required_sections:
                    self.assertIn(section, config, 
                                f"Missing section '{section}' in {config_file}")
    
    def test_parameter_value_ranges(self):
        """Test that parameter values are within reasonable ranges."""
        config = ConfigLoader.load_config('config_optimal.json')
        
        # Test HMM parameters
        self.assertGreater(config['hmm']['train_window'], 0)
        self.assertGreater(config['hmm']['refit_every'], 0)
        self.assertGreaterEqual(config['hmm']['bear_prob_threshold'], 0.0)
        self.assertLessEqual(config['hmm']['bear_prob_threshold'], 1.0)
        self.assertGreaterEqual(config['hmm']['bull_prob_threshold'], 0.0)
        self.assertLessEqual(config['hmm']['bull_prob_threshold'], 1.0)
        
        # Test backtest parameters
        self.assertGreater(config['backtest']['initial_capital'], 0)
        self.assertGreaterEqual(config['backtest']['transaction_cost'], 0.0)
        self.assertGreater(config['backtest']['rebalance_frequency'], 0)
        
        # Test alpha model parameters
        self.assertGreater(config['alpha_model']['short_window'], 0)
        self.assertGreater(config['alpha_model']['long_window'], 0)
        self.assertGreater(config['alpha_model']['long_window'], 
                         config['alpha_model']['short_window'])
    
    def test_strategy_mode_values(self):
        """Test that strategy_mode values are valid."""
        valid_modes = ['alpha_only', 'hmm_only', 'alpha_hmm_filter', 'alpha_hmm_combine']
        
        config_files = ['config_default.json', 'config_optimal.json', 'config_accurate.json']
        
        for config_file in config_files:
            with self.subTest(config_file=config_file):
                config = ConfigLoader.load_config(config_file)
                strategy_mode = config['strategy']['strategy_mode']
                self.assertIn(strategy_mode, valid_modes,
                            f"Invalid strategy_mode '{strategy_mode}' in {config_file}")
    
    def test_default_parameters_fallback(self):
        """Test that parameter extraction uses defaults for missing values."""
        # Create minimal config
        minimal_config = {
            'backtest': {},
            'strategy': {},
            'hmm': {}
        }
        
        # Test backtest params with defaults
        params = ConfigLoader.get_backtest_params(minimal_config)
        self.assertEqual(params['strategy_mode'], 'alpha_only')
        self.assertEqual(params['rebalance_frequency'], 1)
        self.assertEqual(params['walk_forward'], False)
        self.assertEqual(params['train_window'], 504)
        self.assertEqual(params['refit_every'], 21)
        self.assertEqual(params['transaction_cost'], 0.0)
        
        # Test HMM params with defaults
        hmm_params = ConfigLoader.get_hmm_params(minimal_config)
        self.assertEqual(hmm_params['n_states'], 3)
        self.assertEqual(hmm_params['random_state'], 42)
        
        # Test alpha params with defaults
        alpha_params = ConfigLoader.get_alpha_params(minimal_config)
        self.assertEqual(alpha_params['short_window'], 10)
        self.assertEqual(alpha_params['long_window'], 30)
    
    def test_configuration_differences(self):
        """Test that different configurations have expected differences."""
        default_config = ConfigLoader.load_config('config_default.json')
        optimal_config = ConfigLoader.load_config('config_optimal.json')
        accurate_config = ConfigLoader.load_config('config_accurate.json')
        
        # Verify train_window differences
        self.assertEqual(default_config['hmm']['train_window'], 504)
        self.assertEqual(optimal_config['hmm']['train_window'], 252)
        self.assertEqual(accurate_config['hmm']['train_window'], 756)
        
        # Verify refit_every differences
        self.assertEqual(default_config['hmm']['refit_every'], 21)
        self.assertEqual(optimal_config['hmm']['refit_every'], 42)
        self.assertEqual(accurate_config['hmm']['refit_every'], 21)


if __name__ == '__main__':
    unittest.main()
