"""
Statistics module for calculating financial performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Union


class Statistics:
    """
    Calculate primary financial performance metrics for trading strategies.
    """
    
    @staticmethod
    def total_return(returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate total return.
        
        Parameters:
        -----------
        returns : pd.Series or np.ndarray
            Series of returns
            
        Returns:
        --------
        float
            Total return (cumulative)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        return float(np.prod(1 + returns) - 1)
    
    @staticmethod
    def annualized_return(returns: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
        """
        Calculate annualized return.
        
        Parameters:
        -----------
        returns : pd.Series or np.ndarray
            Series of returns
        periods_per_year : int
            Number of periods per year (252 for daily, 52 for weekly, 12 for monthly)
            
        Returns:
        --------
        float
            Annualized return
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        total_ret = np.prod(1 + returns)
        n_periods = len(returns)
        if n_periods == 0:
            return 0.0
        return float(total_ret ** (periods_per_year / n_periods) - 1)
    
    @staticmethod
    def volatility(returns: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility.
        
        Parameters:
        -----------
        returns : pd.Series or np.ndarray
            Series of returns
        periods_per_year : int
            Number of periods per year
            
        Returns:
        --------
        float
            Annualized volatility (standard deviation)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        return float(np.std(returns) * np.sqrt(periods_per_year))
    
    @staticmethod
    def sharpe_ratio(returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.0, 
                     periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe Ratio.
        
        Parameters:
        -----------
        returns : pd.Series or np.ndarray
            Series of returns
        risk_free_rate : float
            Risk-free rate (annualized)
        periods_per_year : int
            Number of periods per year
            
        Returns:
        --------
        float
            Sharpe ratio
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / periods_per_year)
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess == 0:
            return 0.0
            
        return float(mean_excess / std_excess * np.sqrt(periods_per_year))
    
    @staticmethod
    def sortino_ratio(returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.0,
                      periods_per_year: int = 252) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation instead of total volatility).
        
        Parameters:
        -----------
        returns : pd.Series or np.ndarray
            Series of returns
        risk_free_rate : float
            Risk-free rate (annualized)
        periods_per_year : int
            Number of periods per year
            
        Returns:
        --------
        float
            Sortino ratio
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / periods_per_year)
        mean_excess = np.mean(excess_returns)
        
        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            downside_std = 0.0
        else:
            downside_std = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_std == 0:
            return 0.0
            
        return float(mean_excess / downside_std * np.sqrt(periods_per_year))
    
    @staticmethod
    def max_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate maximum drawdown.
        
        Parameters:
        -----------
        returns : pd.Series or np.ndarray
            Series of returns
            
        Returns:
        --------
        float
            Maximum drawdown (negative value)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            return 0.0
            
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))
    
    @staticmethod
    def profit_factor(returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate profit factor (gross profits / gross losses).
        
        Parameters:
        -----------
        returns : pd.Series or np.ndarray
            Series of returns
            
        Returns:
        --------
        float
            Profit factor
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return np.inf if gains > 0 else 0.0
            
        return float(gains / losses)
    
    @staticmethod
    def win_rate(returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate win rate (percentage of positive returns).
        
        Parameters:
        -----------
        returns : pd.Series or np.ndarray
            Series of returns
            
        Returns:
        --------
        float
            Win rate (0 to 1)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            return 0.0
            
        wins = np.sum(returns > 0)
        return float(wins / len(returns))
    
    @staticmethod
    def calmar_ratio(returns: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
        """
        Calculate Calmar Ratio (annualized return / max drawdown).
        
        Parameters:
        -----------
        returns : pd.Series or np.ndarray
            Series of returns
        periods_per_year : int
            Number of periods per year
            
        Returns:
        --------
        float
            Calmar ratio
        """
        ann_ret = Statistics.annualized_return(returns, periods_per_year)
        max_dd = abs(Statistics.max_drawdown(returns))
        
        if max_dd == 0:
            return 0.0
            
        return float(ann_ret / max_dd)
    
    @staticmethod
    def calculate_all_metrics(returns: Union[pd.Series, np.ndarray], 
                             risk_free_rate: float = 0.0,
                             periods_per_year: int = 252) -> dict:
        """
        Calculate all performance metrics.
        
        Parameters:
        -----------
        returns : pd.Series or np.ndarray
            Series of returns
        risk_free_rate : float
            Risk-free rate (annualized)
        periods_per_year : int
            Number of periods per year
            
        Returns:
        --------
        dict
            Dictionary containing all metrics
        """
        return {
            'total_return': Statistics.total_return(returns),
            'annualized_return': Statistics.annualized_return(returns, periods_per_year),
            'volatility': Statistics.volatility(returns, periods_per_year),
            'sharpe_ratio': Statistics.sharpe_ratio(returns, risk_free_rate, periods_per_year),
            'sortino_ratio': Statistics.sortino_ratio(returns, risk_free_rate, periods_per_year),
            'max_drawdown': Statistics.max_drawdown(returns),
            'profit_factor': Statistics.profit_factor(returns),
            'win_rate': Statistics.win_rate(returns),
            'calmar_ratio': Statistics.calmar_ratio(returns, periods_per_year)
        }
    
    @staticmethod
    def print_metrics(metrics: dict, title: str = "PERFORMANCE METRICS") -> None:
        """
        Print formatted performance metrics.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of metrics
        title : str
            Title for the output
        """
        print("\n" + "="*60)
        print(title)
        print("="*60)
        print(f"Total Return:        {metrics['total_return']*100:>10.2f}%")
        print(f"Annualized Return:   {metrics['annualized_return']*100:>10.2f}%")
        print(f"Volatility:          {metrics['volatility']*100:>10.2f}%")
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
        print(f"Max Drawdown:        {metrics['max_drawdown']*100:>10.2f}%")
        print(f"Profit Factor:       {metrics['profit_factor']:>10.2f}")
        print(f"Win Rate:            {metrics['win_rate']*100:>10.2f}%")
        print(f"Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")
        print("="*60)
