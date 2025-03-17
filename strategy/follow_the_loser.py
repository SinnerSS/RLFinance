import pandas as pd
from typing import List, Dict, Union, Optional

from .base_strategy import BaseStrategy

class MeanReversionFollowLoser(BaseStrategy):
    """
    A class implementing a 'Follow the Loser' strategy based on mean reversion.
    This strategy:
    1. Ranks stocks based on their recent underperformance
    2. Allocates capital to the worst performing stocks
    3. Periodically rebalances portfolio to capture new potential mean-reverting stocks
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        pool: Optional[List[str]] = None,
        initial_capital: float = 10000.0,
        lookback_period: int = 20,     
        bottom_n: int = 3,               
        rebalance_freq: int = 30     
    ):
        """
        Initialize the mean reversion follow-the-loser strategy.
        
        Args:
            pool: Pool of stock symbols
            data: Stock price dataframe
            initial_capital: Starting capital
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            lookback_period: Number of days to calculate performance
            bottom_n: Number of worst performers to include
            rebalance_freq: Rebalance frequency in days
        """
        super().__init__(data, start_date, end_date, pool, initial_capital)
        self.lookback_period = lookback_period
        self.bottom_n = min(bottom_n, len(self.pool))  
        self.rebalance_freq = rebalance_freq
        self.last_rebalance_date = None
    
    def _update_strategy(self, date: pd.Timestamp):
        """
        Update strategy based on mean reversion signals.
        Rebalances at specified intervals or on the start date.
        """
        if (self.last_rebalance_date is None) or \
           ((date - self.last_rebalance_date).days >= self.rebalance_freq):
            
            performance_scores = self._calculate_performance(date)
            worst_performers = self._select_worst_performers(performance_scores)
            
            if worst_performers:
                self._sell_all_positions(date)

                # Use negative performance as reversion potential 
                reversion_potential = {s: max(-performance_scores[s], 0.001) for s in worst_performers}
                total_potential = sum(reversion_potential.values())

                for symbol in worst_performers:
                    weight = reversion_potential[symbol] / total_potential
                    self.plan[symbol] = self.capital * weight
                
                self.last_rebalance_date = date
    
    def _calculate_performance(self, current_date: pd.Timestamp) -> Dict[str, float]:
        performance_scores = {}
        lookback_start = current_date - pd.Timedelta(days=self.lookback_period)
        
        for symbol, data in self.stock_data.items():
            period_data = data[(data['date'] <= current_date) & (data['date'] >= lookback_start)]
            
            if len(period_data) >= 2:  
                start_price = period_data.iloc[0]['adj close']
                end_price = period_data.iloc[-1]['adj close']
                
                performance = (end_price - start_price) / start_price
                performance_scores[symbol] = performance
        
        return performance_scores
    
    def _select_worst_performers(self, performance_scores: Dict[str, float]) -> List[str]:
        sorted_stocks = sorted(performance_scores.items(), key=lambda x: x[1])
        
        worst_performers = [symbol for symbol, _ in sorted_stocks[:self.bottom_n]]
        
        return worst_performers
    
    def _sell_all_positions(self, date: pd.Timestamp):
        for symbol in list(self.portfolio.keys()):
            proceeds = self.sell_position(symbol, 1.0, date)
            self.capital += proceeds
