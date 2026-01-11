"""
CSV data loader implementation.
Example implementation showing how to create custom data loaders.
"""

import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
from .base_loader import BaseDataLoader


class CSVLoader(BaseDataLoader):
    """
    Data loader for CSV files.
    Loads historical price data from CSV files.
    
    Expected CSV format:
    - Index: Date (can be named 'Date' or be the index)
    - Columns: Open, High, Low, Close, Volume (at minimum Close is required)
    
    Example usage:
    --------------
    loader = CSVLoader(data_dir='./data')
    data = loader.load_ticker('SPY', '2020-01-01', '2025-01-01')
    
    This will look for a file at './data/SPY.csv'
    """
    
    def __init__(self, data_dir: str = './data', date_column: str = 'Date'):
        """
        Initialize CSV loader.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing CSV files
        date_column : str
            Name of the date column (use 'index' if date is the index)
        """
        self.data_dir = Path(data_dir)
        self.date_column = date_column
    
    def load_ticker(self, ticker: str, start_date: str, end_date: str, 
                   progress: bool = False) -> Optional[pd.DataFrame]:
        """
        Load data for a single ticker from CSV file.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol to load (used as filename: {ticker}.csv)
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        progress : bool
            Whether to show progress (not used for CSV loading)
            
        Returns:
        --------
        pd.DataFrame or None
            OHLCV data with DatetimeIndex, or None if error occurs
        """
        try:
            csv_path = self.data_dir / f"{ticker}.csv"
            
            if not csv_path.exists():
                print(f"Error: File not found: {csv_path}")
                return None
            
            # Read CSV
            if self.date_column == 'index':
                data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            else:
                data = pd.read_csv(csv_path, parse_dates=[self.date_column])
                data.set_index(self.date_column, inplace=True)
            
            # Filter by date range
            data = data[start_date:end_date]
            
            if len(data) == 0:
                print(f"Warning: No data in date range for {ticker}")
                return None
            
            # Validate data
            if not self.validate_data(data):
                print(f"Error: Invalid data format in {csv_path}")
                return None
            
            return data
            
        except Exception as e:
            print(f"Error loading {ticker} from CSV: {e}")
            return None
    
    def load_multiple_tickers(self, tickers: List[str], start_date: str, 
                             end_date: str, progress: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple tickers from CSV files.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols to load
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        progress : bool
            Whether to show progress
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping ticker symbols to their OHLCV data
        """
        data_dict = {}
        
        for ticker in tickers:
            if progress:
                print(f"Loading {ticker}...")
            
            data = self.load_ticker(ticker, start_date, end_date, progress=False)
            if data is not None:
                data_dict[ticker] = data
                
        return data_dict
