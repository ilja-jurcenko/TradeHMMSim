"""
Portfolio management module for loading and managing asset data.
"""

import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime


class Portfolio:
    """
    Portfolio class for managing assets and loading data from Yahoo Finance.
    """
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        """
        Initialize Portfolio.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols to load
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data: Dict[str, pd.DataFrame] = {}
        self.close_prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        
    def load_data(self, progress: bool = False) -> None:
        """
        Load data from Yahoo Finance for all tickers.
        
        Parameters:
        -----------
        progress : bool
            Whether to show download progress
        """
        print(f"Loading data for {len(self.tickers)} ticker(s) from {self.start_date} to {self.end_date}...")
        
        for ticker in self.tickers:
            try:
                data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=progress)
                if len(data) > 0:
                    self.data[ticker] = data
                    print(f"  ✓ {ticker}: {len(data)} rows")
                else:
                    print(f"  ✗ {ticker}: No data available")
            except Exception as e:
                print(f"  ✗ {ticker}: Error - {e}")
        
        self._prepare_close_prices()
        self._calculate_returns()
        
    def _prepare_close_prices(self) -> None:
        """Prepare close prices DataFrame from loaded data."""
        if not self.data:
            return
            
        close_dict = {}
        for ticker, data in self.data.items():
            # Handle both single and multi-ticker downloads
            if isinstance(data, pd.DataFrame):
                if 'Close' in data.columns:
                    close_series = data['Close']
                    # If Close is a DataFrame (single ticker with MultiIndex columns)
                    if isinstance(close_series, pd.DataFrame):
                        # Get the first (and only) column as a Series
                        close_dict[ticker] = close_series.iloc[:, 0]
                    else:
                        # Already a Series
                        close_dict[ticker] = close_series
                elif len(data.columns) == 1:
                    # Single column DataFrame
                    close_dict[ticker] = data.iloc[:, 0]
            elif isinstance(data, pd.Series):
                close_dict[ticker] = data
        
        if close_dict:
            # Create DataFrame from dictionary of Series
            self.close_prices = pd.DataFrame(close_dict)
            
    def _calculate_returns(self) -> None:
        """Calculate daily returns from close prices."""
        if self.close_prices is not None:
            self.returns = self.close_prices.pct_change()
            
    def get_close_prices(self, ticker: Optional[str] = None) -> pd.Series or pd.DataFrame:
        """
        Get close prices for a specific ticker or all tickers.
        
        Parameters:
        -----------
        ticker : str, optional
            Ticker symbol. If None, returns all tickers.
            
        Returns:
        --------
        pd.Series or pd.DataFrame
            Close prices (Series if single ticker specified, DataFrame otherwise)
        """
        if self.close_prices is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if ticker:
            if ticker in self.close_prices.columns:
                # Return as Series
                return self.close_prices[ticker]
            else:
                raise ValueError(f"Ticker {ticker} not found in loaded data")
        else:
            return self.close_prices
    
    def get_returns(self, ticker: Optional[str] = None) -> pd.Series or pd.DataFrame:
        """
        Get returns for a specific ticker or all tickers.
        
        Parameters:
        -----------
        ticker : str, optional
            Ticker symbol. If None, returns all tickers.
            
        Returns:
        --------
        pd.Series or pd.DataFrame
            Returns
        """
        if ticker:
            if ticker in self.data:
                return self.data[ticker]['Close'].pct_change()
            else:
                raise ValueError(f"Ticker {ticker} not found in portfolio")
        return self.returns
    
    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        """
        Get OHLCV data for a specific ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        pd.DataFrame
            OHLCV data (Open, High, Low, Close, Volume)
        """
        if ticker not in self.data:
            raise ValueError(f"Ticker {ticker} not found in portfolio")
        return self.data[ticker]
    
    def add_ticker(self, ticker: str) -> None:
        """
        Add a new ticker to the portfolio.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol to add
        """
        if ticker not in self.tickers:
            self.tickers.append(ticker)
            try:
                data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if len(data) > 0:
                    self.data[ticker] = data
                    self._prepare_close_prices()
                    self._calculate_returns()
                    print(f"Added {ticker}: {len(data)} rows")
                else:
                    print(f"No data available for {ticker}")
            except Exception as e:
                print(f"Error adding {ticker}: {e}")
    
    def remove_ticker(self, ticker: str) -> None:
        """
        Remove a ticker from the portfolio.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol to remove
        """
        if ticker in self.tickers:
            self.tickers.remove(ticker)
            if ticker in self.data:
                del self.data[ticker]
            self._prepare_close_prices()
            self._calculate_returns()
            print(f"Removed {ticker}")
        else:
            print(f"Ticker {ticker} not found in portfolio")
    
    def get_ticker_list(self) -> List[str]:
        """
        Get list of tickers in portfolio.
        
        Returns:
        --------
        List[str]
            List of ticker symbols
        """
        return self.tickers
    
    def summary(self) -> None:
        """Print portfolio summary."""
        print("\n" + "="*60)
        print("PORTFOLIO SUMMARY")
        print("="*60)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Tickers: {len(self.tickers)}")
        print(f"Loaded assets: {len(self.data)}")
        print("\nAsset details:")
        for ticker in self.tickers:
            if ticker in self.data:
                data = self.data[ticker]
                print(f"  {ticker}: {len(data)} rows, "
                      f"first: {data.index[0].strftime('%Y-%m-%d')}, "
                      f"last: {data.index[-1].strftime('%Y-%m-%d')}")
            else:
                print(f"  {ticker}: Not loaded")
        print("="*60)
