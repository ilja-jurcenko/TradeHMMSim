"""
Configuration loader for backtest parameters.
Supports loading from JSON files and merging with defaults.
"""

import json
import os
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage backtest configuration from JSON files."""
    
    DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_default.json')
    
    @staticmethod
    def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file. If None, uses default config.
            
        Returns:
        --------
        Dict[str, Any]
            Configuration dictionary
        """
        if config_path is None:
            config_path = ConfigLoader.DEFAULT_CONFIG_PATH
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration dictionary
        output_path : str
            Path to save configuration file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {output_path}")
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries (override takes precedence).
        
        Parameters:
        -----------
        base_config : Dict[str, Any]
            Base configuration
        override_config : Dict[str, Any]
            Override configuration
            
        Returns:
        --------
        Dict[str, Any]
            Merged configuration
        """
        result = base_config.copy()
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def get_backtest_params(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract backtest parameters from config.
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Full configuration dictionary
            
        Returns:
        --------
        Dict[str, Any]
            Backtest parameters for BacktestEngine.run()
        """
        backtest = config.get('backtest', {})
        strategy = config.get('strategy', {})
        hmm = config.get('hmm', {})
        
        return {
            'strategy_mode': strategy.get('strategy_mode', 'alpha_only'),
            'rebalance_frequency': backtest.get('rebalance_frequency', 1),
            'walk_forward': strategy.get('walk_forward', False),
            'train_window': hmm.get('train_window', 504),
            'refit_every': hmm.get('refit_every', 21),
            'bear_prob_threshold': hmm.get('bear_prob_threshold', 0.65),
            'bull_prob_threshold': hmm.get('bull_prob_threshold', 0.65),
            'transaction_cost': backtest.get('transaction_cost', 0.0)
        }
    
    @staticmethod
    def get_hmm_params(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract HMM filter parameters from config.
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Full configuration dictionary
            
        Returns:
        --------
        Dict[str, Any]
            HMM filter parameters
        """
        hmm = config.get('hmm', {})
        
        return {
            'n_states': hmm.get('n_states', 3),
            'random_state': hmm.get('random_state', 42)
        }
    
    @staticmethod
    def get_alpha_params(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract alpha model parameters from config.
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Full configuration dictionary
            
        Returns:
        --------
        Dict[str, Any]
            Alpha model parameters with 'type' and 'parameters' keys
        """
        alpha = config.get('alpha_model', {})
        
        # Support both old and new format for backward compatibility
        if 'type' in alpha and 'parameters' in alpha:
            # New format with type and parameters
            return alpha
        else:
            # Old format with direct parameters (backward compatibility)
            return {
                'type': 'SMA',  # Default type
                'parameters': {
                    'short_window': alpha.get('short_window', 10),
                    'long_window': alpha.get('long_window', 30)
                }
            }
    
    @staticmethod
    def print_config(config: Dict[str, Any]) -> None:
        """
        Print configuration in a readable format.
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration dictionary
        """
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        print(json.dumps(config, indent=2))
        print("="*60)
