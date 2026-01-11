"""
Cached Yahoo Finance data loader with intelligent data management.
Downloads from yfinance and caches locally in CSV files, with smart append logic.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from .yfinance_loader import YFinanceLoader
from .base_loader import BaseDataLoader


class CachedYFinanceLoader(BaseDataLoader):
    """
    Yahoo Finance loader with local CSV caching and intelligent data appending.
    
    Features:
    - Downloads from yfinance and saves to local CSV
    - Reads from local cache if available
    - Detects gaps in local data and downloads only missing periods
    - Automatically appends new data to existing CSV files
    
    Example:
    --------
    loader = CachedYFinanceLoader(cache_dir='./cache')
    data = loader.load_ticker('SPY', '2020-01-01', '2025-01-01')
    # First call: downloads from yfinance and saves to ./cache/SPY.csv
    # Second call: reads from ./cache/SPY.csv (no download)
    # Third call with later date: downloads only missing data and appends
    """
    
    def __init__(self, cache_dir: str = './cache', yfinance_loader: Optional[YFinanceLoader] = None):
        """
        Initialize cached loader.
        
        Parameters:
        -----------
        cache_dir : str
            Directory to store cached CSV files
        yfinance_loader : YFinanceLoader, optional
            YFinance loader instance. If None, creates a new one.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.yfinance_loader = yfinance_loader if yfinance_loader is not None else YFinanceLoader()
    
    def _get_cache_path(self, ticker: str) -> Path:
        """Get the cache file path for a ticker."""
        return self.cache_dir / f"{ticker}.csv"
    
    def _read_cached_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Read cached data from CSV file.
        
        Returns:
        --------
        pd.DataFrame or None
            Cached data with DatetimeIndex, or None if cache doesn't exist
        """
        cache_path = self._get_cache_path(ticker)
        
        if not cache_path.exists():
            return None
        
        try:
            data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return data
        except Exception as e:
            print(f"Warning: Failed to read cache for {ticker}: {e}")
            return None
    
    def _save_to_cache(self, ticker: str, data: pd.DataFrame) -> None:
        """
        Save data to CSV cache.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        data : pd.DataFrame
            Data to save
        """
        cache_path = self._get_cache_path(ticker)
        
        try:
            # Sort by date before saving
            data = data.sort_index()
            data.to_csv(cache_path)
        except Exception as e:
            print(f"Warning: Failed to save cache for {ticker}: {e}")
    
    def _append_to_cache(self, ticker: str, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Append new data to existing cache.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        new_data : pd.DataFrame
            New data to append
            
        Returns:
        --------
        pd.DataFrame
            Combined data (existing + new)
        """
        cached_data = self._read_cached_data(ticker)
        
        if cached_data is None:
            # No existing cache, just save the new data
            self._save_to_cache(ticker, new_data)
            return new_data
        
        # Combine data, removing duplicates (prefer new data for overlapping dates)
        combined = pd.concat([cached_data, new_data])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
        
        # Save combined data
        self._save_to_cache(ticker, combined)
        
        return combined
    
    def _detect_gaps(self, cached_data: pd.DataFrame, start_date: str, end_date: str) -> List[tuple]:
        """
        Detect gaps in cached data compared to requested date range.
        
        Parameters:
        -----------
        cached_data : pd.DataFrame
            Existing cached data
        start_date : str
            Requested start date
        end_date : str
            Requested end date
            
        Returns:
        --------
        List[tuple]
            List of (start, end) tuples representing gaps to download
        """
        if cached_data is None or len(cached_data) == 0:
            # No cached data, need to download entire range
            return [(start_date, end_date)]
        
        requested_start = pd.to_datetime(start_date)
        requested_end = pd.to_datetime(end_date)
        
        cached_start = cached_data.index.min()
        cached_end = cached_data.index.max()
        
        gaps = []
        
        # Check if we need data before the cached start
        # Add buffer to avoid single-day downloads at boundaries
        if requested_start < cached_start:
            # Download from requested_start to just before cached_start
            # Use a 3-day buffer to avoid edge issues
            gap_end_date = cached_start - timedelta(days=1)
            if (gap_end_date - requested_start).days >= 0:  # Only if meaningful gap
                gaps.append((start_date, gap_end_date.strftime('%Y-%m-%d')))
        
        # Check if we need data after the cached end
        if requested_end > cached_end:
            # Download from just after cached_end to requested_end
            # Use a 3-day buffer to avoid edge issues
            gap_start_date = cached_end + timedelta(days=1)
            if (requested_end - gap_start_date).days >= 0:  # Only if meaningful gap
                gaps.append((gap_start_date.strftime('%Y-%m-%d'), end_date))
        
        return gaps
    
    def load_ticker(self, ticker: str, start_date: str, end_date: str, 
                   progress: bool = False, force_download: bool = False) -> Optional[pd.DataFrame]:
        """
        Load data with intelligent caching and gap filling.
        
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
        force_download : bool
            If True, bypass cache and download fresh data
            
        Returns:
        --------
        pd.DataFrame or None
            OHLCV data with DatetimeIndex, or None if error occurs
        """
        if force_download:
            # Force download from yfinance
            data = self.yfinance_loader.load_ticker(ticker, start_date, end_date, progress)
            if data is not None:
                self._save_to_cache(ticker, data)
            return data
        
        # Try to read from cache first
        cached_data = self._read_cached_data(ticker)
        
        if cached_data is None:
            # No cache, download entire range
            print(f"Cache miss for {ticker}, downloading from yfinance...")
            data = self.yfinance_loader.load_ticker(ticker, start_date, end_date, progress)
            if data is not None:
                self._save_to_cache(ticker, data)
            return data
        
        # Detect gaps in cached data
        gaps = self._detect_gaps(cached_data, start_date, end_date)
        
        if len(gaps) == 0:
            # Cache has all the data we need
            print(f"Cache hit for {ticker}, using local data")
            # Filter to requested date range
            result = cached_data[start_date:end_date]
            return result
        
        # Download missing data for each gap
        print(f"Cache partial hit for {ticker}, downloading missing data...")
        downloaded_any = False
        for gap_start, gap_end in gaps:
            print(f"  Downloading gap: {gap_start} to {gap_end}")
            gap_data = self.yfinance_loader.load_ticker(ticker, gap_start, gap_end, progress)
            if gap_data is not None and len(gap_data) > 0:
                cached_data = self._append_to_cache(ticker, gap_data)
                downloaded_any = True
        
        # Return the requested date range from updated cache
        if downloaded_any:
            cached_data = self._read_cached_data(ticker)
        
        if cached_data is not None:
            result = cached_data[start_date:end_date]
            if len(result) > 0:
                return result
        
        return None
    
    def load_multiple_tickers(self, tickers: List[str], start_date: str, 
                             end_date: str, progress: bool = False,
                             force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load multiple tickers with caching.
        
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
        force_download : bool
            If True, bypass cache and download fresh data
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping ticker symbols to their OHLCV data
        """
        data_dict = {}
        
        for ticker in tickers:
            data = self.load_ticker(ticker, start_date, end_date, progress, force_download)
            if data is not None:
                data_dict[ticker] = data
                
        return data_dict
    
    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """
        Clear cached data.
        
        Parameters:
        -----------
        ticker : str, optional
            Ticker to clear. If None, clears all cached data.
        """
        if ticker is not None:
            cache_path = self._get_cache_path(ticker)
            if cache_path.exists():
                cache_path.unlink()
                print(f"Cleared cache for {ticker}")
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.csv"):
                cache_file.unlink()
            print("Cleared all cache files")
    
    def get_cache_info(self, ticker: str) -> Optional[Dict]:
        """
        Get information about cached data.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        dict or None
            Dictionary with cache info (start_date, end_date, rows), or None if no cache
        """
        cached_data = self._read_cached_data(ticker)
        
        if cached_data is None:
            return None
        
        return {
            'ticker': ticker,
            'start_date': cached_data.index.min().strftime('%Y-%m-%d'),
            'end_date': cached_data.index.max().strftime('%Y-%m-%d'),
            'rows': len(cached_data),
            'cache_path': str(self._get_cache_path(ticker))
        }
