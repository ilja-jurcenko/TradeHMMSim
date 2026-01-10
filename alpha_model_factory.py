"""
Alpha model factory for creating alpha models from JSON configuration.
Supports all alpha model types with varying parameters.
"""

from typing import Dict, Any, Optional
from alpha_models.base import AlphaModel
from alpha_models.sma import SMA
from alpha_models.ema import EMA
from alpha_models.wma import WMA
from alpha_models.hma import HMA
from alpha_models.kama import KAMA
from alpha_models.tema import TEMA
from alpha_models.zlema import ZLEMA


class AlphaModelFactory:
    """Factory for creating alpha models from configuration."""
    
    # Registry of available alpha models
    _MODELS = {
        'SMA': SMA,
        'EMA': EMA,
        'WMA': WMA,
        'HMA': HMA,
        'KAMA': KAMA,
        'TEMA': TEMA,
        'ZLEMA': ZLEMA
    }
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> AlphaModel:
        """
        Create an alpha model from configuration dictionary.
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration dictionary with 'type' and 'parameters' keys
            
        Returns:
        --------
        AlphaModel
            Instantiated alpha model
            
        Raises:
        -------
        ValueError
            If model type is unknown or parameters are invalid
        KeyError
            If required configuration keys are missing
            
        Example:
        --------
        >>> config = {
        ...     'type': 'SMA',
        ...     'parameters': {'short_window': 50, 'long_window': 200}
        ... }
        >>> model = AlphaModelFactory.create_from_config(config)
        """
        if 'type' not in config:
            raise KeyError("Configuration must include 'type' key")
        
        if 'parameters' not in config:
            raise KeyError("Configuration must include 'parameters' key")
        
        model_type = config['type']
        parameters = config['parameters']
        
        if model_type not in AlphaModelFactory._MODELS:
            available = ', '.join(AlphaModelFactory._MODELS.keys())
            raise ValueError(
                f"Unknown alpha model type: '{model_type}'. "
                f"Available types: {available}"
            )
        
        # Validate parameters
        AlphaModelFactory._validate_parameters(model_type, parameters)
        
        # Create model instance
        model_class = AlphaModelFactory._MODELS[model_type]
        
        # Extract only the required parameters (filter out extras)
        filtered_params = {
            'short_window': parameters['short_window'],
            'long_window': parameters['long_window']
        }
        
        try:
            model = model_class(**filtered_params)
            return model
        except TypeError as e:
            raise ValueError(
                f"Invalid parameters for {model_type}: {filtered_params}. Error: {str(e)}"
            )
    
    @staticmethod
    def create_from_type(model_type: str, short_window: int, long_window: int) -> AlphaModel:
        """
        Create an alpha model from type and standard parameters.
        
        Parameters:
        -----------
        model_type : str
            Type of alpha model (e.g., 'SMA', 'EMA')
        short_window : int
            Short window parameter
        long_window : int
            Long window parameter
            
        Returns:
        --------
        AlphaModel
            Instantiated alpha model
        """
        config = {
            'type': model_type,
            'parameters': {
                'short_window': short_window,
                'long_window': long_window
            }
        }
        return AlphaModelFactory.create_from_config(config)
    
    @staticmethod
    def _validate_parameters(model_type: str, parameters: Dict[str, Any]) -> None:
        """
        Validate parameters for a specific model type.
        
        Parameters:
        -----------
        model_type : str
            Type of alpha model
        parameters : Dict[str, Any]
            Parameters to validate
            
        Raises:
        -------
        ValueError
            If parameters are invalid
        """
        # All current models require short_window and long_window
        required_params = ['short_window', 'long_window']
        
        for param in required_params:
            if param not in parameters:
                raise ValueError(
                    f"Missing required parameter '{param}' for {model_type}"
                )
        
        # Validate parameter types and values
        short_window = parameters['short_window']
        long_window = parameters['long_window']
        
        if not isinstance(short_window, int) or short_window <= 0:
            raise ValueError(
                f"short_window must be a positive integer, got: {short_window}"
            )
        
        if not isinstance(long_window, int) or long_window <= 0:
            raise ValueError(
                f"long_window must be a positive integer, got: {long_window}"
            )
        
        if short_window >= long_window:
            raise ValueError(
                f"short_window ({short_window}) must be less than "
                f"long_window ({long_window})"
            )
    
    @staticmethod
    def get_available_models() -> list:
        """
        Get list of available alpha model types.
        
        Returns:
        --------
        list
            List of available model type names
        """
        return list(AlphaModelFactory._MODELS.keys())
    
    @staticmethod
    def get_default_parameters(model_type: str) -> Dict[str, int]:
        """
        Get default parameters for a model type.
        
        Parameters:
        -----------
        model_type : str
            Type of alpha model
            
        Returns:
        --------
        Dict[str, int]
            Default parameters
        """
        if model_type not in AlphaModelFactory._MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Standard defaults
        return {
            'short_window': 50,
            'long_window': 200
        }
    
    @staticmethod
    def create_all_models(short_window: int = 50, long_window: int = 200) -> Dict[str, AlphaModel]:
        """
        Create instances of all available alpha models with same parameters.
        
        Parameters:
        -----------
        short_window : int
            Short window parameter
        long_window : int
            Long window parameter
            
        Returns:
        --------
        Dict[str, AlphaModel]
            Dictionary mapping model names to instances
        """
        models = {}
        for model_type in AlphaModelFactory._MODELS.keys():
            models[model_type] = AlphaModelFactory.create_from_type(
                model_type, short_window, long_window
            )
        return models
