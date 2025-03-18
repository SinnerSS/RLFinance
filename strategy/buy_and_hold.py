import pandas as pd
from typing import List, Dict, Union, Optional

from .base_strategy import BaseStrategy

class BuyAndHold(BaseStrategy):
    """
    A class implementing the buy and hold investment strategy.
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        pool: Optional[List[str]] = None,
        initial_capital: float = 10000.0,
        equal_weight: bool = True
    ):
        """
        Initialize the buy and hold strategy.
        
        Args:
            data: Dictionary of price DataFrames with keys like 'open' and 'adj_close'
            start_date: Start date for the strategy
            end_date: End date for the strategy
            pool: Optional list of stock symbols. Defaults to all columns in 'adj_close'
            initial_capital: Starting capital for the portfolio
            equal_weight: If True, allocate equal weights to each stock in the pool.
        """
        super().__init__(data, start_date, end_date, pool, initial_capital)
        self.equal_weight = equal_weight
        
    def _update_strategy(self, date: pd.Timestamp):
        if date == self.data['open'].index[0]:
            weights = self._calculate_weights()
            for symbol, weight in weights.items():
                self.plan[symbol] = self.capital * weight
    
    def _calculate_weights(self) -> Dict[str, float]:
        weights = {}
        if self.equal_weight:
            weight_per_stock = 1.0 / len(self.pool)
            for symbol in self.pool:
                weights[symbol] = weight_per_stock
        else:
            raise NotImplementedError("Custom weighted allocation not implemented.")
        return weights

