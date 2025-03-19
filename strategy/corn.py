import numpy as np
import pandas as pd
from typing import Dict, Union, List, Optional

from .base_strategy import BaseStrategy

class Corn(BaseStrategy):
    """
    CORN: Correlation-driven Nonparametric Learning Approach for portfolio selection.
    
    This strategy searches for historical market windows that are similar (using a correlation 
    measure) to the current market window. When similar windows (with correlation above a 
    threshold ρ) are found, their subsequent returns are averaged to predict the next-day 
    performance. Positions are taken (long-only) on assets with a positive expected return, 
    allocating capital proportionally.
    
    Two versions are available:
      - 'slow': Loops over each candidate window (memory efficient).
      - 'fast': Uses vectorized correlation computations (≈2× speedup, but more memory).
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        pool: Optional[List[str]] = None,
        initial_capital: float = 10000.0,
        window: int = 5,
        rho: float = 0.9,
        version: str = 'slow'
    ):
        super().__init__(data, start_date, end_date, pool, initial_capital)
        self.window = window
        self.rho = rho
        if version not in ['slow', 'fast']:
            raise ValueError("version must be 'slow' or 'fast'")
        self.version = version

    def _update_strategy(self, date: pd.Timestamp):
        all_dates = self.data['adj_close'].index
        if date not in all_dates:
            self.plan = {}
            return
        current_index = all_dates.get_loc(date)
        if current_index < self.window:
            self.plan = {}
            return

        if self.version == 'slow':
            self._update_strategy_slow(date, current_index, all_dates)
        else:
            self._update_strategy_fast(date, current_index, all_dates)

    def _update_strategy_slow(self, date, current_index, all_dates):
        current_window = self.data['adj_close'].iloc[current_index - self.window: current_index]
        current_returns = current_window.pct_change().dropna()
        current_vec = current_returns.values.flatten()

        candidate_returns_list = []
        for i in range(self.window, current_index):
            candidate_window = self.data['adj_close'].iloc[i - self.window: i]
            candidate_returns = candidate_window.pct_change().dropna()
            if candidate_returns.shape[0] != current_returns.shape[0]:
                continue
            candidate_vec = candidate_returns.values.flatten()
            if candidate_vec.std() == 0 or current_vec.std() == 0:
                continue
            corr = np.corrcoef(current_vec, candidate_vec)[0, 1]
            if corr >= self.rho:
                if i < len(all_dates) - 1:
                    next_day_returns = self.data['adj_close'].iloc[i] / self.data['adj_close'].iloc[i-1] - 1
                    candidate_returns_list.append(next_day_returns)
        if not candidate_returns_list:
            self.plan = {}
            return
        
        avg_future_returns = pd.concat(candidate_returns_list, axis=1).mean(axis=1)
        positive_assets = avg_future_returns[avg_future_returns > 0]
        if positive_assets.empty:
            self.plan = {}
            return
        weights = positive_assets / positive_assets.sum()
        allocation = self.capital * weights
        self.plan = allocation.to_dict()

    def _update_strategy_fast(self, date, current_index, all_dates):
        current_window = self.data['adj_close'].iloc[current_index - self.window: current_index]
        current_returns = current_window.pct_change().dropna()
        current_vec = current_returns.values.flatten()
        d = current_vec.shape[0]
        
        candidate_matrix = []
        candidate_indices = []
        for i in range(self.window, current_index):
            candidate_window = self.data['adj_close'].iloc[i - self.window: i]
            candidate_returns = candidate_window.pct_change().dropna()
            if candidate_returns.shape[0] != current_returns.shape[0]:
                continue
            candidate_vec = candidate_returns.values.flatten()
            candidate_matrix.append(candidate_vec)
            candidate_indices.append(i)
        if not candidate_matrix:
            self.plan = {}
            return
        candidate_matrix = np.array(candidate_matrix)
        
        current_mean = current_vec.mean()
        current_std = current_vec.std()
        candidates_mean = candidate_matrix.mean(axis=1)
        candidates_std = candidate_matrix.std(axis=1)
        cov = np.mean((candidate_matrix - candidates_mean[:, None]) * (current_vec - current_mean), axis=1)
        valid = (candidates_std > 0) & (current_std > 0)
        if not np.any(valid):
            self.plan = {}
            return
        corrs = np.zeros_like(candidates_mean)
        corrs[valid] = cov[valid] / (candidates_std[valid] * current_std)
        
        similar_mask = corrs >= self.rho
        similar_indices = [candidate_indices[i] for i, flag in enumerate(similar_mask) if flag]
        candidate_returns_list = []
        for i in similar_indices:
            if i < len(all_dates) - 1:
                next_day_returns = self.data['adj_close'].iloc[i] / self.data['adj_close'].iloc[i-1] - 1
                candidate_returns_list.append(next_day_returns)
        if not candidate_returns_list:
            self.plan = {}
            return
        
        avg_future_returns = pd.concat(candidate_returns_list, axis=1).mean(axis=1)
        positive_assets = avg_future_returns[avg_future_returns > 0]
        if positive_assets.empty:
            self.plan = {}
            return
        weights = positive_assets / positive_assets.sum()
        allocation = self.capital * weights
        self.plan = allocation.to_dict()
