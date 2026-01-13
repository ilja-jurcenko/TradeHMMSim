"""
Portfolio management module for loading and managing asset data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from loaders.base_loader import BaseDataLoader
from loaders.cached_yfinance_loader import CachedYFinanceLoader
class Portfolio:
    """
    Portfolio class for managing assets with pluggable data loaders.
    """
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str, 
                 loader: Optional[BaseDataLoader] = None,
                 weights: Optional[Dict[str, float]] = None):
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
        loader : BaseDataLoader, optional
            Data loader instance. If None, uses CachedYFinanceLoader with './data' cache by default.
        weights : Dict[str, float], optional
            Initial weights for each ticker. If None, equal weighting is used.
            Weights should sum to 1.0. Example: {'SPY': 0.6, 'AGG': 0.4}
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.loader = loader if loader is not None else CachedYFinanceLoader(cache_dir='./data')
        self.data: Dict[str, pd.DataFrame] = {}
        self.close_prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        
        # Initialize weights
        if weights is None:
            # Equal weighting by default
            self.weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
        else:
            self.weights = weights.copy()
            # Validate weights
            if set(weights.keys()) != set(tickers):
                raise ValueError("Weights keys must match tickers list")
            weight_sum = sum(weights.values())
            if not np.isclose(weight_sum, 1.0, atol=0.01):
                raise ValueError(f"Weights must sum to 1.0, got {weight_sum:.4f}")
        
    def load_data(self, progress: bool = False) -> None:
        """
        Load data using the configured data loader.
        
        Parameters:
        -----------
        progress : bool
            Whether to show download progress
        """
        print(f"Loading data for {len(self.tickers)} ticker(s) from {self.start_date} to {self.end_date}...")
        
        for ticker in self.tickers:
            data = self.loader.load_ticker(ticker, self.start_date, self.end_date, progress)
            if data is not None:
                self.data[ticker] = data
                print(f"  ✓ {ticker}: {len(data)} rows")
            else:
                print(f"  ✗ {ticker}: No data available")
        
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
            
    def get_close_prices(self, ticker: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
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
    
    def get_returns(self, ticker: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
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
            data = self.loader.load_ticker(ticker, self.start_date, self.end_date, progress=False)
            if data is not None:
                self.data[ticker] = data
                self._prepare_close_prices()
                self._calculate_returns()
                print(f"Added {ticker}: {len(data)} rows")
            else:
                print(f"No data available for {ticker}")
    
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
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set portfolio weights.
        
        Parameters:
        -----------
        weights : Dict[str, float]
            Dictionary of ticker: weight pairs. Must sum to 1.0.
            
        Raises:
        -------
        ValueError
            If weights don't match tickers or don't sum to 1.0
        """
        if set(weights.keys()) != set(self.tickers):
            raise ValueError(f"Weights keys {set(weights.keys())} must match tickers {set(self.tickers)}")
        
        weight_sum = sum(weights.values())
        if not np.isclose(weight_sum, 1.0, atol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum:.4f}")
        
        self.weights = weights.copy()
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get current portfolio weights.
        
        Returns:
        --------
        Dict[str, float]
            Dictionary of ticker: weight pairs
        """
        return self.weights.copy()
    
    def get_weighted_returns(self) -> pd.Series:
        """
        Calculate portfolio returns using current weights.
        
        Returns:
        --------
        pd.Series
            Weighted portfolio returns
        """
        if self.returns is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Apply weights to returns
        weighted_returns = pd.Series(0.0, index=self.returns.index)
        for ticker in self.tickers:
            if ticker in self.returns.columns:
                weighted_returns += self.returns[ticker] * self.weights[ticker]
        
        return weighted_returns
    
    def rebalance_regime_based(self, regime: str, aggressive_ticker: str = 'SPY', 
                               defensive_ticker: str = 'AGG') -> None:
        """
        Rebalance portfolio based on market regime.
        
        Parameters:
        -----------
        regime : str
            Market regime: 'bull', 'neutral', or 'bear'
        aggressive_ticker : str
            Ticker for aggressive allocation (default: 'SPY')
        defensive_ticker : str
            Ticker for defensive allocation (default: 'AGG')
            
        Rules:
        ------
        - Bull/Neutral: 100% aggressive (SPY)
        - Bear: 100% defensive (AGG)
        """
        if aggressive_ticker not in self.tickers or defensive_ticker not in self.tickers:
            raise ValueError(f"Both {aggressive_ticker} and {defensive_ticker} must be in portfolio")
        
        new_weights = {ticker: 0.0 for ticker in self.tickers}
        
        if regime in ['bull', 'neutral']:
            # 100% aggressive asset
            new_weights[aggressive_ticker] = 1.0
        elif regime == 'bear':
            # 100% defensive asset
            new_weights[defensive_ticker] = 1.0
        else:
            raise ValueError(f"Unknown regime: {regime}. Use 'bull', 'neutral', or 'bear'")
        
        self.set_weights(new_weights)
    
    def summary(self) -> None:
        """Print portfolio summary."""
        print("\n" + "="*60)
        print("PORTFOLIO SUMMARY")
        print("="*60)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Tickers: {len(self.tickers)}")
        print(f"Loaded assets: {len(self.data)}")
        print("\nCurrent Weights:")
        for ticker in self.tickers:
            weight_pct = self.weights[ticker] * 100
            print(f"  {ticker}: {weight_pct:.1f}%")
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
