import numpy as np
import pandas as pd
from typing import Dict, Tuple

def returns(portfolio_values: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from portfolio values"""
    df = portfolio_values.copy()
    df['return'] = df['portfolio_value'].pct_change()
    return df

def cumulative_returns(portfolio_values: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative returns from portfolio values"""
    df = returns(portfolio_values)
    df['cumulative_return'] = (1 + df['return'].fillna(0)).cumprod() - 1
    return df

def sharpe_ratio(returns: pd.DataFrame, risk_free_rates: pd.Series) -> float:
    """Calculate Sharpe ratio (annualized)"""
    df = returns.dropna(subset=['return'])
    daily_rf = (1 + risk_free_rates) ** (1/252) - 1
    excess_returns = df['return'] - daily_rf
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def max_drawdown(returns: pd.DataFrame) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """Calculate maximum drawdown and timeframe"""
    wealth_index = (1 + returns['return'].fillna(0)).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    peak_idx = previous_peaks.iloc[:max_dd_idx].idxmax()
    
    return max_dd, returns['date'].iloc[peak_idx], returns['date'].iloc[max_dd_idx]

def evaluate_strategy(portfolio_values: pd.DataFrame, against: pd.DataFrame) -> Dict:
    """Calculate all performance metrics for a strategy"""
    portfolio_values = portfolio_values.loc[portfolio_values['date'].dt.weekday < 5].reset_index()
    returns_df = cumulative_returns(portfolio_values)
    
    total_return = returns_df['cumulative_return'].iloc[-1]
    days = (returns_df['date'].iloc[-1] - returns_df['date'].iloc[0]).days
    years = days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    against['observation_date'] = pd.to_datetime(against['observation_date'])
    risk_free_rates = returns_df.merge(against, left_on='date', right_on='observation_date', how='left').ffill()['DGS3MO']

    sharpe = sharpe_ratio(returns_df, risk_free_rates)
    max_dd, peak_date, trough_date = max_drawdown(returns_df)
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'max_drawdown_peak_date': peak_date,
        'max_drawdown_trough_date': trough_date
    }
