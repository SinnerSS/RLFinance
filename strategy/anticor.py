import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional

from .base_strategy import BaseStrategy


class Anticor(BaseStrategy):
    """
    Anticor (anti-correlation) is a heuristic portfolio selection algorithm that adjusts the
    portfolio based on the lagged correlations of asset returns. In particular, it shifts weight
    from assets with lower recent performance to those with higher performance when there is
    evidence of positive cross-correlation and negative autocorrelation.

    Reference:
        A. Borodin, R. El-Yaniv, and V. Gogan. Can we learn to beat the best stock, 2005.
        http://www.cs.technion.ac.il/~rani/el-yaniv-papers/BorodinEG03.pdf
    """

    def __init__(
        self, 
        data: Dict[str, pd.DataFrame], 
        start_date: Union[str, pd.Timestamp], 
        end_date: Union[str, pd.Timestamp], 
        pool: Optional[List[str]] = None, 
        initial_capital: float = 10000.0, 
        window: int = 30
    ):
        super().__init__(data, start_date, end_date, pool, initial_capital)
        self.window = window
        m = len(self.pool)
        self.current_weights = np.full(m, 1.0 / m)

    def _update_strategy(self, date: pd.Timestamp):
        """
        Update the strategy for the given date using the Anticor algorithm.
        If there is insufficient historical data (i.e. fewer than 2*window days),
        the strategy defaults to the current (or uniform) weights.
        Otherwise, it computes the weights adjustment using the past 2*window days.
        """
        prices = self.data['adj_close']
        try:
            pos = prices.index.get_loc(date)
        except KeyError:
            return

        m = len(self.pool)
        if pos < 2 * self.window:
            self.plan = {symbol: self.current_weights[i] * self.capital for i, symbol in enumerate(self.pool)}
            return

        window_data = prices.iloc[pos - 2 * self.window : pos]
        first_window = window_data.iloc[:self.window]
        second_window = window_data.iloc[self.window:]

        r_first = first_window.pct_change().dropna()
        r_second = second_window.pct_change().dropna()

        if r_first.shape[0] < 1 or r_second.shape[0] < 1:
            self.plan = {symbol: self.current_weights[i] * self.capital for i, symbol in enumerate(self.pool)}
            return

        M = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                ri = r_first.iloc[:, i].values
                rj = r_second.iloc[:, j].values
                if np.std(ri) == 0 or np.std(rj) == 0:
                    M[i, j] = 0.0
                else:
                    M[i, j] = np.corrcoef(ri, rj)[0, 1]

        mu = r_second.mean(axis=0).values

        claim = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                if mu[i] > mu[j] and M[i, j] > 0:
                    claim[i, j] = M[i, j]
                    if M[i, i] < 0:
                        claim[i, j] += abs(M[i, i])
                    if M[j, j] < 0:
                        claim[i, j] += abs(M[j, j])

        transfer = np.zeros((m, m))
        for i in range(m):
            total_claim = claim[i, :].sum()
            if total_claim != 0:
                transfer[i, :] = self.current_weights[i] * claim[i, :] / total_claim

        new_weights = self.current_weights + transfer.sum(axis=0) - transfer.sum(axis=1)
        new_weights[new_weights < 0] = 0
        total = new_weights.sum()
        if total > 0:
            new_weights = new_weights / total
        else:
            new_weights = np.full(m, 1.0 / m)

        self.current_weights = new_weights
        self.plan = {symbol: new_weights[i] * self.capital for i, symbol in enumerate(self.pool)}

