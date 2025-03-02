import pandas as pd
from typing import List, Dict, Optional

from .base_strategy import BaseStrategy

class BuyAndHold(BaseStrategy):
    """
    A class implementing the buy and hold investment strategy.
    """
    
    def __init__(
        self,
        pool: List[str],
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        equal_weight: bool = True
    ):
        """
        Initialize the buy and hold strategy.
        
        Args:
            pool: Pool of stock symbols
            data: Stock price dataframe
            initial_capital: Starting capital
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            equal_weight: If True, equal allocation; else, weighted by Count
        """
        super().__init__(pool, data, initial_capital, start_date, end_date)
        self.equal_weight = equal_weight
        
    def _update_strategy(self, date: pd.Timestamp):
        """
        Buy and hold strategy only makes allocation decisions once at the start.
        """
        if date == self.start_date:
            weights = self._calculate_weights()
            for symbol, weight in weights.items():
                self.plan[symbol] = self.capital * weight
    
    def _calculate_weights(self) -> Dict[str, float]:
        weights = {}
        if self.equal_weight:
            weight_per_stock = 1.0 / len(self.stock_data)
            for symbol in self.stock_data:
                weights[symbol] = weight_per_stock
        else:
            pass
        return weights
