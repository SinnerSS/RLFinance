import pandas as pd
from tqdm import tqdm
from abc import abstractmethod
from typing import List, Dict, Union, Optional

from utils.eval import evaluate_strategy
from utils.monitor import monitor_time

class BaseStrategy:
    """
    Abstract base class for investment strategies that expects preformatted data.
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        pool: Optional[List[str]] = None,
        initial_capital: float = 10000.0,
    ):
        self.capital = initial_capital
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        self.data = {
            key: df.loc[self.start_date:self.end_date] for key, df in data.items()
        }
        
        if pool is None:
            self.pool = list(self.data['adj_close'].columns)
        else:
            self.pool = pool
        
        self.portfolio = {}
        self.plan = {}

        self.history_records = [{
            'date': self.start_date - pd.Timedelta(days=1),
            'portfolio_value': self.capital
        }]
    
    @monitor_time
    def execute(self) -> pd.DataFrame:
        dates = self.data['open'].index
        for date in tqdm(dates, desc=f'Executing {self.__class__.__name__}'):
            self._update_strategy(date)
            self._update_portfolio(date)
            date_value = self._calculate_portfolio_value(date)
            self.history_records.append({'date': date, 'portfolio_value': date_value})
        self.history = pd.DataFrame(self.history_records)
        return self.history
    
    @abstractmethod
    def _update_strategy(self, date: pd.Timestamp):
        """
        Update the strategy for a given date (to be implemented by subclasses).
        """
        pass
    
    def _update_portfolio(self, date: pd.Timestamp):
        open_prices = self.data['open']
        for symbol in list(self.plan.keys()):
            if symbol in self.pool and date in open_prices.index:
                price = open_prices.loc[date, symbol]
                allocated = self.plan[symbol]
                if pd.notna(price) and allocated > 0:
                    shares = allocated / price
                    if symbol in self.portfolio:
                        prev_shares = self.portfolio[symbol]['shares']
                        prev_entry_price = self.portfolio[symbol]['entry_price']
                        new_total_shares = prev_shares + shares
                        new_entry_price = (prev_shares * prev_entry_price + shares * price) / new_total_shares
                        self.portfolio[symbol].update({
                            'shares': new_total_shares,
                            'entry_price': new_entry_price,
                            'entry_date': date
                        })
                    else:
                        self.portfolio[symbol] = {
                            'shares': shares,
                            'entry_price': price,
                            'entry_date': date
                        }
                    self.capital -= allocated
                del self.plan[symbol]
    
    def _calculate_portfolio_value(self, date: pd.Timestamp) -> float:
        total_value = self.capital
        adj_close_prices = self.data['adj_close']
        for symbol, detail in self.portfolio.items():
            price = self._find_last_price(symbol, date, adj_close_prices)
            total_value += detail['shares'] * price
        return total_value
    
    def _find_last_price(self, symbol: str, date: pd.Timestamp, price_df: pd.DataFrame) -> float:
        available = price_df.loc[:date, symbol]
        if available.empty:
            raise ValueError(f"No price data available for {symbol} on or before {date}")
        return available.iloc[-1]
    
    def sell_position(self, symbol: str, percentage: float, date: pd.Timestamp) -> float:
        if symbol not in self.portfolio or percentage <= 0:
            return 0.0
        
        position = self.portfolio[symbol]
        shares_to_sell = position['shares'] * percentage
        price = self._find_last_price(symbol, date, self.data['adj_close'])
        proceeds = shares_to_sell * price
        
        position['shares'] -= shares_to_sell
        if position['shares'] <= 0:
            del self.portfolio[symbol]
        
        self.capital += proceeds
        return proceeds
    
    def evaluate(self, against: pd.DataFrame):
        return evaluate_strategy(self.history, against)
