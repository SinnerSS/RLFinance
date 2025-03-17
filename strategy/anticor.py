import numpy as np
import pandas as pd
from typing import Union, List, Optional

from strategy.base_strategy import BaseStrategy

class AnticorStrategy(BaseStrategy):
    """
    Implementation of the ANTICOR (Anti-correlation) strategy.
    
    The strategy exploits negative correlation between assets over time,
    transferring wealth from assets that performed well to those that performed poorly.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        pool: Optional[List[str]] = None,
        initial_capital: float = 10000.0,
        window: int = 30,
        rebalance_freq: int = 1
    ):
        super().__init__(data, start_date, end_date, pool, initial_capital)
        self.window = window
        self.rebalance_freq = rebalance_freq
        self.last_rebalance_date = None
    
    def _update_strategy(self, date):
        if self.last_rebalance_date is None or (date - self.last_rebalance_date).days >= self.rebalance_freq:
            self._anticor_rebalance(date)
            self.last_rebalance_date = date
    
    def _anticor_rebalance(self, date):
        returns_data = self._get_historical_returns(date)
        
        if returns_data is None or returns_data.shape[1] < 2:
            return
        
        weights = self._calculate_anticor_weights(returns_data)
        
        self._rebalance_portfolio(weights, date)
    
    def _get_historical_returns(self, date):
        """Get historical returns for all stocks up to the current date"""
        lookback = 2 * self.window
        start_lookback = date - pd.Timedelta(days=lookback*2 + 1)
        
        price_data = {}
        valid_symbols = []
        
        for symbol in self.pool:
            symbol_data = self.stock_data.get(symbol)
            if symbol_data is None:
                continue
                
            hist_data = symbol_data[(symbol_data['date'] >= start_lookback) & 
                                   (symbol_data['date'] <= date)]
            
            if len(hist_data) >= lookback:
                price_data[symbol] = hist_data.set_index('date')['adj close']
                valid_symbols.append(symbol)
        
        if not valid_symbols:
            return None
            
        prices_df = pd.DataFrame(price_data)
        prices_df = prices_df.sort_index().iloc[-lookback:]
        
        if prices_df.shape[1] < 2:
            return None
            
        returns_df = prices_df.pct_change()
        return returns_df
    
    def _calculate_anticor_weights(self, returns_data):
        """Calculate ANTICOR weights based on returns data"""
        n_assets = returns_data.shape[1]
        symbols = returns_data.columns

        # Get two windows of data (as DataFrames)
        window1 = returns_data.iloc[:self.window]
        window2 = returns_data.iloc[self.window:]
        
        # Convert to numpy arrays for faster math (each column is one asset)
        X1 = window1.values  # shape: (self.window, n_assets)
        X2 = window2.values  # shape: (len(window2), n_assets)

        # 1. Compute the full correlation matrix for window1
        # (this uses pandas vectorized computation under the hood)
        C = window1.corr().values  # shape: (n_assets, n_assets)

        # 2. Compute the correlation between each asset's returns in window1 and window2.
        # We use vectorized operations:
        # Compute means and standard deviations (ddof=1 for sample std)
        mu1 = np.mean(X1, axis=0)  # shape: (n_assets,)
        mu2 = np.mean(X2, axis=0)
        std1 = np.std(X1, axis=0, ddof=1)
        std2 = np.std(X2, axis=0, ddof=1)
        # Compute covariance between corresponding columns
        # Note: subtracting mu1 and mu2 will broadcast over rows.
        cov12 = np.sum((X1 - mu1) * (X2 - mu2), axis=0) / (X1.shape[0] - 1)
        r_between = cov12 / (std1 * std2)  # shape: (n_assets,)

        # 3. Get the mean returns for window2 (as a 1D numpy array)
        m2 = window2.mean().values  # shape: (n_assets,)

        # Build a boolean mask using broadcasting.
        # For each pair (i,j):
        #   condition1: C[i,j] < 0
        #   condition2: r_between[i] > 0 and r_between[j] > 0
        #   condition3: m2[i] < m2[j]
        mask = (C < 0) & (r_between[:, None] > 0) & (r_between[None, :] > 0) & (m2[:, None] < m2[None, :])

        # Claim matrix: where mask is True, use absolute value of C; else zero.
        claim_matrix = np.where(mask, np.abs(C), 0)

        # Sum over rows and columns.
        row_sums = claim_matrix.sum(axis=1)
        col_sums = claim_matrix.sum(axis=0)
        total_claims = claim_matrix.sum()

        # Compute weights.
        weights = {}
        if total_claims > 0:
            for i, symbol in enumerate(symbols):
                weight = (col_sums[i] - row_sums[i]) / total_claims
                weights[symbol] = max(0, weight)
        else:
            for symbol in symbols:
                weights[symbol] = 1.0 / n_assets

        # Normalize weights so they sum to 1.
        total_weight = sum(weights.values())
        if total_weight > 0:
            for symbol in weights:
                weights[symbol] /= total_weight

        return weights    

    def _rebalance_portfolio(self, weights, date):
        """Rebalance portfolio according to weights"""
        current_value = self._calculate_portfolio_value(date)
        
        for symbol in list(self.portfolio.keys()):
            proceeds = self.sell_position(symbol, 1.0, date)
            self.capital += proceeds
        
        for symbol, weight in weights.items():
            if weight > 0:
                allocation = current_value * weight
                self.plan[symbol] = allocation
