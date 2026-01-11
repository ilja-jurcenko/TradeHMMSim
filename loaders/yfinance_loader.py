"""
Yahoo Finance data loader implementation.
"""

import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
from .base_loader import BaseDataLoader


class YFinanceLoader(BaseDataLoader):
    """
    Data loader for Yahoo Finance.
    Handles downloading historical price data from Yahoo Finance.
    """
    
    def load_ticker(self, ticker: str, start_date: str, end_date: str, 
                   progress: bool = False) -> Optional[pd.DataFrame]:
        """
        Load data for a single ticker from Yahoo Finance.
        
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
            OHLCV data with DatetimeIndex, or None if error occurs
        """
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=progress)
            
            if len(data) == 0:
                return None
            
            # Normalize the data structure - handle MultiIndex columns from yfinance
            data = self._normalize_dataframe(data)
            
            if not self.validate_data(data):
                return None
                
            return data
            
        except Exception as e:
            print(f"Error loading {ticker}: {e}")
            return None
    
    def load_multiple_tickers(self, tickers: List[str], start_date: str, 
                             end_date: str, progress: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple tickers from Yahoo Finance.
        
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
        data_dict = {}
        
        for ticker in tickers:
            data = self.load_ticker(ticker, start_date, end_date, progress)
            if data is not None:
                data_dict[ticker] = data
                
        return data_dict
    
    def _normalize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize yfinance DataFrame structure.
        Handles MultiIndex columns that sometimes appear with single ticker downloads.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw data from yfinance
            
        Returns:
        --------
        pd.DataFrame
            Normalized DataFrame with standard columns
        """
        if isinstance(data, pd.DataFrame):
            # Check if we have MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten MultiIndex by taking the first level (column names)
                data.columns = data.columns.get_level_values(0)
            
        return data
