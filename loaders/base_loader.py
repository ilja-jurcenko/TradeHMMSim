"""
Base data loader interface for different data providers.
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    Defines the interface that all data providers must implement.
    """
    
    @abstractmethod
    def load_ticker(self, ticker: str, start_date: str, end_date: str, 
                   progress: bool = False) -> Optional[pd.DataFrame]:
        """
        Load data for a single ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol to load
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        progress : bool
            Whether to show download progress
            
        Returns:
        --------
        pd.DataFrame or None
            OHLCV data with DatetimeIndex, or None if no data available
        """
        pass
    
    @abstractmethod
    def load_multiple_tickers(self, tickers: List[str], start_date: str, 
                             end_date: str, progress: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple tickers.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols to load
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        progress : bool
            Whether to show download progress
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping ticker symbols to their OHLCV data
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the loaded data meets basic requirements.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to validate
            
        Returns:
        --------
        bool
            True if data is valid
        """
        if data is None or len(data) == 0:
            return False
        
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        has_required = all(col in data.columns for col in required_cols)
        
        if not has_required:
            # At minimum, we need Close prices
            return 'Close' in data.columns
        
        return True
