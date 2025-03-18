import numpy as np
import pandas as pd
from typing import Dict, Tuple

def returns(portfolio_values: pd.DataFrame) -> pd.DataFrame:
    df = portfolio_values.copy()
    df['return'] = df['portfolio_value'].pct_change()
    return df

def cumulative_returns(portfolio_values: pd.DataFrame) -> pd.DataFrame:
    df = returns(portfolio_values)
    df['cumulative_return'] = (1 + df['return'].fillna(0)).cumprod() - 1
    return df

def sharpe_ratio(returns_df: pd.DataFrame, risk_free_rates: pd.Series) -> float:
    df = returns_df.dropna(subset=['return']).copy()
    daily_rf = (1 + risk_free_rates) ** (1 / 252) - 1
    excess_returns = df['return'] - daily_rf
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def max_drawdown(returns_df: pd.DataFrame) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    df = returns_df.copy()
    wealth_index = (1 + df['return'].fillna(0)).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks

    max_dd_idx = int(drawdown.idxmin())
    peak_idx = int(previous_peaks.iloc[:max_dd_idx].idxmax())
    max_dd = drawdown.iloc[max_dd_idx]

    return max_dd, df['date'].iloc[peak_idx], df['date'].iloc[max_dd_idx]

def evaluate_strategy(portfolio_values: pd.DataFrame, against: pd.DataFrame) -> Dict:
    """
    Calculate performance metrics for a strategy.
    
    Args:
        portfolio_values: DataFrame with at least ['date', 'portfolio_value'].
                          It is expected that 'date' is a datetime column.
        against: Benchmark DataFrame with columns ['observation_date', 'DGS3MO'].
                 'DGS3MO' should be expressed as percentages (e.g. 2.5 for 2.5%).
    
    Returns:
        Dictionary containing total return, annualized return, Sharpe ratio,
        maximum drawdown, and the corresponding peak and trough dates.
    """
    portfolio_values = portfolio_values.reset_index(drop=True)
    
    returns_cum = cumulative_returns(portfolio_values)
    
    total_return = returns_cum['cumulative_return'].iloc[-1]
    days = (returns_cum['date'].iloc[-1] - returns_cum['date'].iloc[0]).days
    years = days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

    against['DGS3MO'] = against['DGS3MO'] / 100
    against['observation_date'] = pd.to_datetime(against['observation_date'])
    
    risk_free_rates = returns_cum.merge(
        against, left_on='date', right_on='observation_date', how='left'
    ).ffill()['DGS3MO']

    sharpe = sharpe_ratio(returns_cum, risk_free_rates)
    max_dd, peak_date, trough_date = max_drawdown(returns_cum)
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'max_drawdown_peak_date': peak_date,
        'max_drawdown_trough_date': trough_date
    }

