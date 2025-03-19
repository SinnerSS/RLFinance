import logging
import numpy as np
import pandas as pd
from scipy import optimize
from typing import Dict, List, Union, Optional

from .base_strategy import BaseStrategy

class NearestNeighbor(BaseStrategy):
    """
    Nearest neighbor based strategy that inherits from BaseStrategy.
    
    This strategy tries to find similar sequences of price in history and
    then maximize objective function (profit) on the days following them.
    
    Reference:
        L. Gyorfi, G. Lugosi, and F. Udina. Nonparametric kernel based sequential
        investment strategies. Mathematical Finance 16 (2006) 337–357.
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        pool: Optional[List[str]] = None,
        initial_capital: float = 10000.0,
        k: int = 5,
        l: int = 10,
        rebalance_freq: str = 'W-FRI',
        max_leverage: float = 1.0,
        metric: str = "return"
    ):
        super().__init__(data, start_date, end_date, pool, initial_capital)
        self.k = k
        self.l = l
        self.rebalance_freq = rebalance_freq
        self.max_leverage = max_leverage
        self.metric = metric
        self.rebalance_dates = self._get_rebalance_dates()
        self.last_rebalance_date = None
        
    def _get_rebalance_dates(self) -> pd.DatetimeIndex:
        all_dates = self.data['adj_close'].index
        if self.rebalance_freq == 'D':
            return all_dates
        else:
            rebalance_range = pd.date_range(
                start=self.start_date,
                end=self.end_date,
                freq=self.rebalance_freq
            )
            return all_dates[all_dates.isin(rebalance_range)]
    
    def _update_strategy(self, date: pd.Timestamp):
        if date not in self.rebalance_dates:
            return
        
        history = self.data['adj_close'].loc[:date, self.pool].copy()
        
        price_ratios = history.pct_change() + 1
        price_ratios = price_ratios.dropna()
        
        min_history_required = self.k + self.l
        if len(price_ratios) < min_history_required:
            return
        
        try:
            weights = self._find_optimal_weights(price_ratios)
            
            portfolio_value = self._calculate_portfolio_value(date)
            
            for symbol in list(self.portfolio.keys()):
                self.sell_position(symbol, 1.0, date)
            
            for symbol, weight in zip(self.pool, weights):
                if weight > 0:
                    allocation = portfolio_value * weight
                    self.plan[symbol] = allocation
                    
            self.last_rebalance_date = date
            
        except Exception as e:
            logging.warning(f"Error in update strategy for date {date}: {e}")
            pass
    
    def _find_optimal_weights(self, price_ratios: pd.DataFrame) -> np.ndarray:
        """
        Find optimal weights using nearest neighbors algorithm.
        
        Args:
            price_ratios: DataFrame of price ratios (returns + 1)
            
        Returns:
            Array of weights for each symbol in the pool
        """
        nn_indices = self._find_nn(price_ratios, self.k, self.l)
        
        next_day_indices = [price_ratios.index.get_loc(i) + 1 for i in nn_indices 
                           if price_ratios.index.get_loc(i) + 1 < len(price_ratios)]
        
        if not next_day_indices:
            return np.ones(len(self.pool)) / len(self.pool)
            
        J = price_ratios.iloc[next_day_indices]
        
        return self._opt_weights(J, metric=self.metric, max_leverage=self.max_leverage)
    
    def _find_nn(self, history: pd.DataFrame, k: int, l: int) -> pd.Index:
        """
        Find nearest neighbors based on price sequence similarity.
        
        Args:
            history: DataFrame of price ratios
            k: Sequence length
            l: Number of nearest neighbors
            
        Returns:
            Index of nearest neighbor dates
        """
        # Calculate distance from current sequence to every other point
        D = pd.Series(0.0, index=history.index[k:])
        
        for i in range(1, k + 1):
            current_price = history.iloc[-i]
            historical_prices = history.shift(i - 1).iloc[k-1:-1]
            squared_diff = ((historical_prices - current_price) ** 2).sum(axis=1)
            D += squared_diff
        
        D = D.sort_values()
        return D.index[:l]
    
    def _opt_weights(self, X: pd.DataFrame, metric: str = "return", max_leverage: float = 1.0,
                   rf_rate: float = 0.0, alpha: float = 0.0, freq: float = 252,
                   no_cash: bool = False, sd_factor: float = 1.0) -> np.ndarray:
        """
        Find best constant rebalanced portfolio with regards to some metric.
        
        Args:
            X: Price ratios DataFrame
            metric: Performance metric to optimize ('return', 'sharpe', 'drawdown', 'ulcer')
            max_leverage: Maximum leverage
            rf_rate: Risk-free rate for sharpe calculation
            alpha: Regularization parameter for volatility
            freq: Frequency for annualization (252 for daily data)
            no_cash: If True, must fully invest (sum of weights == max_leverage)
            sd_factor: Standard deviation factor for sharpe ratio
            
        Returns:
            Array of optimal weights
        """
        assert metric in ("return", "sharpe", "drawdown", "ulcer")
        
        if X.empty or X.isnull().any().any():
            return np.ones(len(self.pool)) / len(self.pool)
            
        x_0 = max_leverage * np.ones(X.shape[1]) / float(X.shape[1])
        
        if metric == "return":
            objective = lambda b: -np.sum(np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001)))
        elif metric == "sharpe":
            def objective(b):
                returns = np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001))
                if len(returns) <= 1:
                    return 0
                mean_return = np.mean(returns) * freq
                std_return = np.std(returns) * np.sqrt(freq) * sd_factor
                if std_return == 0:
                    return 0
                return -(mean_return - rf_rate) / (std_return + alpha)
        elif metric == "drawdown":
            def objective(b):
                R = np.dot(X - 1, b) + 1
                L = np.cumprod(R)
                if len(L) <= 1:
                    return 0
                dd = max(1 - L / np.maximum.accumulate(L))
                annual_ret = np.mean(R) ** freq - 1
                return -annual_ret / (dd + alpha)
        elif metric == "ulcer":
            def objective(b):
                returns = np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001))
                if len(returns) <= 1:
                    return 0
                    
                # Ulcer Index calculation
                wealth = np.exp(np.cumsum(returns))
                peak = np.maximum.accumulate(wealth)
                drawdown = (wealth - peak) / peak
                ulcer_idx = np.sqrt(np.mean(drawdown**2))
                if ulcer_idx == 0:
                    return 0
                    
                mean_return = np.mean(returns) * freq
                return -(mean_return - rf_rate) / ulcer_idx
        
        # Constraints
        if no_cash:
            cons = ({"type": "eq", "fun": lambda b: max_leverage - sum(b)},)
        else:
            cons = ({"type": "ineq", "fun": lambda b: max_leverage - sum(b)},)
        
        for attempt in range(3):  
            try:
                res = optimize.minimize(
                    objective,
                    x_0,
                    bounds=[(0.0, max_leverage)] * len(x_0),
                    constraints=cons,
                    method="slsqp"
                )
                
                EPS = 1e-7
                if (res.x < 0.0 - EPS).any() or (res.x > max_leverage + EPS).any():
                    X = X + np.random.randn(1)[0] * 1e-5
                    logging.debug("Optimal weights not found, trying again...")
                    continue
                elif res.success:
                    return res.x
                else:
                    if np.isnan(res.x).any():
                        logging.warning("Solution does not exist, using equal weights.")
                        return np.ones(X.shape[1]) / X.shape[1]
                    else:
                        logging.warning("Converged, but not successfully.")
                        return res.x
            except Exception as e:
                logging.warning(f"Optimization error: {e}")
                
        return np.ones(X.shape[1]) / X.shape[1]
