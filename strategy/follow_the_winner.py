import pandas as pd
from typing import List, Dict, Union, Optional

from .base_strategy import BaseStrategy

class MomentumFollowWinner(BaseStrategy):
    """
    A class implementing a 'Follow the Winner' strategy based on momentum.
    This strategy:
    1. Ranks stocks based on their momentum (recent performance)
    2. Allocates capital to the top performing stocks
    3. Periodically rebalances portfolio to capture new momentum leaders
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        pool: Optional[List[str]] = None,
        initial_capital: float = 10000.0,
        lookback_period: int = 20,     
        top_n: int = 3,               
        rebalance_freq: int = 30     
    ):
        """
        Initialize the momentum follow-the-winner strategy.
        
        Args:
            pool: Pool of stock symbols
            data: Stock price dataframe
            initial_capital: Starting capital
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            lookback_period: Number of days to calculate momentum
            top_n: Number of top performers to include
            rebalance_freq: Rebalance frequency in days
        """
        super().__init__(data, start_date, end_date, pool, initial_capital)
        self.lookback_period = lookback_period
        self.top_n = min(top_n, len(self.pool))  
        self.rebalance_freq = rebalance_freq
        self.last_rebalance_date = None
    
    def _update_strategy(self, date: pd.Timestamp):
        """
        Update strategy based on momentum signals.
        Rebalances at specified intervals or on the start date.
        """
        if (self.last_rebalance_date is None) or \
           ((date - self.last_rebalance_date).days >= self.rebalance_freq):
            
            momentum_scores = self._calculate_momentum(date)
            top_performers = self._select_top_performers(momentum_scores)
            
            if top_performers:
                self._sell_all_positions(date)

                total_momentum = sum(momentum_scores[symbol] for symbol in top_performers)

                for symbol in top_performers:
                    weight = momentum_scores[symbol] / total_momentum
                    self.plan[symbol] = self.capital * weight
                
                self.last_rebalance_date = date    

    def _calculate_momentum(self, current_date: pd.Timestamp) -> Dict[str, float]:
        """
        Calculate momentum scores for all stocks in the pool.
        
        Returns:
            Dict mapping stock symbols to momentum scores
        """
        momentum_scores = {}
        lookback_start = current_date - pd.Timedelta(days=self.lookback_period)
        
        for symbol, data in self.stock_data.items():
            period_data = data[(data['date'] <= current_date) & (data['date'] >= lookback_start)]
            
            if len(period_data) >= 2:  
                start_price = period_data.iloc[0]['adj close']
                end_price = period_data.iloc[-1]['adj close']
                
                momentum = (end_price - start_price) / start_price
                momentum_scores[symbol] = momentum
        
        return momentum_scores
    
    def _select_top_performers(self, momentum_scores: Dict[str, float]) -> List[str]:
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_performers = [symbol for symbol, _ in sorted_stocks[:self.top_n]]
        
        return top_performers
    
    def _sell_all_positions(self, date: pd.Timestamp):
        for symbol in list(self.portfolio.keys()):
            proceeds = self.sell_position(symbol, 1.0, date)
            self.capital += proceeds
