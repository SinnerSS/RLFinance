import pandas as pd
from abc import abstractmethod
from typing import List, Dict, Union

from utils.eval import evaluate_strategy
from utils.monitor import monitor_time

class BaseStrategy:
    """
    Abstract base class for investment strategies.
    """
    
    def __init__(
        self,
        pool: List[str],
        data: pd.DataFrame,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        initial_capital: float = 10000.0,
    ):
        """
        Initialize the base strategy.
        
        Args:
            pool: Pool of stock symbols
            data: Stock price dataframe
            initial_capital: Starting capital
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
        """
        self.pool = pool
        self.capital = initial_capital

        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)

        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        self.start_date, self.end_date = start_date, end_date
        
        filtered_data = self._filter_date_range(data, self.start_date, self.end_date)
        self.stock_data = self._prepare_stock_data(filtered_data)
        
        self.portfolio = {}
        self.plan = {}
        self.history = pd.DataFrame({'date': [self.start_date - pd.Timedelta(days=1)], 'portfolio_value': [self.capital]})

    def _filter_date_range(self, data: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        data['date'] = pd.to_datetime(data['date'])
        if start_date:
            data = data[data['date'] >= start_date]
        if end_date:
            data = data[data['date'] <= end_date]
        return data
            
    def _prepare_stock_data(self, data: pd.DataFrame) -> Dict:
        stock_data = {}
        for symbol in self.pool:
            stock_df = data[data['Stock_symbol'] == symbol]
            if not stock_df.empty:
                stock_data[symbol] = stock_df
        return stock_data

    @monitor_time
    def execute(self):
        dates = pd.date_range(self.start_date, self.end_date)
        for date in dates:
            self._update_strategy(date)
            self._update_portfolio(date)
            date_value = self._calculate_portfolio_value(date)
            self.history = pd.concat([
                self.history, 
                pd.DataFrame({'date': [date], 'portfolio_value': [date_value]})
            ], ignore_index=True)
        return self.history
    
    @abstractmethod
    def _update_strategy(self, date: pd.Timestamp):
        pass

    def _update_portfolio(self, date: pd.Timestamp):
        for symbol in list(self.plan.keys()):
            symbol_data = self.stock_data.get(symbol)
            if not symbol_data is None:
                date_data = symbol_data[symbol_data['date'] == date]
                allocated = self.plan[symbol]

                if not date_data.empty and allocated > 0:
                    price = date_data['open'].iloc[0]
                    shares = allocated / price
                    
                    if symbol in self.portfolio:
                        self.portfolio[symbol]['shares'] += shares
                        total_shares = self.portfolio[symbol]['shares']
                        old_value = self.portfolio[symbol]['entry_price'] * (total_shares - shares)
                        new_value = price * shares
                        self.portfolio[symbol]['entry_price'] = (old_value + new_value) / total_shares
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
        for symbol, detail in self.portfolio.items():
            last_price = self._find_last_price(symbol, date)
            total_value += detail['shares'] * last_price
        return total_value

    def _find_last_price(self, symbol: str, date: pd.Timestamp) -> float:
        symbol_data = self.stock_data[symbol]
        available_dates = symbol_data[symbol_data['date'] <= date]
        
        if available_dates.empty:
            min_date = symbol_data['date'].min()
            raise ValueError(f"No price data available for date {date} when earliest date is {min_date}")
        
        last_available_date = available_dates['date'].max()
        price = symbol_data.loc[symbol_data['date'] == last_available_date, 'adj close'].iloc[0]
        return price
    
    def sell_position(self, symbol: str, percentage: float, date: pd.Timestamp) -> float:
        """
        Sell a percentage of a position and return proceeds.
        
        Args:
            symbol: Stock symbol
            percentage: Percentage to sell (0-1)
            date: Date of the sale
            
        Returns:
            Amount added back to capital
        """
        if symbol not in self.portfolio or percentage <= 0:
            return 0
            
        position = self.portfolio[symbol]
        shares_to_sell = position['shares'] * percentage
        price = self._find_last_price(symbol, date)
        proceeds = shares_to_sell * price
        
        position['shares'] -= shares_to_sell
        if position['shares'] <= 0:
            del self.portfolio[symbol]
            
        return proceeds
    
    def evaluate(self, against: pd.DataFrame):
        return evaluate_strategy(self.history, against)
