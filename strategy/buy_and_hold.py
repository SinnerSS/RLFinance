import pandas as pd
from typing import List, Dict, Optional

from utils.performance import monitor_time

class BuyAndHold:
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
        self.pool = pool
        self.data = data
        self.capital = initial_capital
        self.equal_weight = equal_weight
        self.portfolio = {}
        self.stock_data = {}
        self.plan = {}
        self.history = {}
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        self._filter_date_range(self.start_date, self.end_date)
        
        self._prepare_stock_data()
        
    def _filter_date_range(self, start_date: pd.Timestamp,  end_date: pd.Timestamp) -> None:
        """Filter dataframe by date range."""
        self.data['date'] = pd.to_datetime(self.data['date'])
        if start_date:
            self.data = self.data.loc[self.data['date'] >= start_date]
        if end_date:
            self.data = self.data.loc[self.data['date'] <= end_date]
            
    def _prepare_stock_data(self) -> None:
        """Prepare individual stock data dictionaries."""
        for symbol in self.pool:
            stock_df = self.data[self.data['Stock_symbol'] == symbol]
            if not stock_df.empty:
                self.stock_data[symbol] = stock_df

    @monitor_time
    def execute(self):
        
        weights = self._calculate_weights()
        for symbol, weight in weights.items():
            self.plan[symbol] = self.capital * weight

        dates = pd.date_range(self.start_date, self.end_date)
        for date in dates: 
            self._update_portfolio(date)
            date_value = self._calculate_portfolio_value(date)
            self.history[date] = date_value

        return self.history
    
    def _calculate_weights(self) -> Dict[str, float]:
        """Calculate portfolio allocation weights."""
        weights = {}
        if self.equal_weight:
            weight_per_stock = 1.0 / len(self.stock_data)
            for symbol in self.stock_data:
                weights[symbol] = weight_per_stock
        else:
            pass
        return weights

    def _update_portfolio(self, date: pd.Timestamp):
        for symbol in list(self.plan.keys()):
            symbol_data = self.stock_data[symbol]
            date_data = symbol_data[symbol_data['date'] == date]
            allocated = self.plan[symbol]

            if not date_data.empty:
                if symbol in self.plan:
                    self.portfolio[symbol] = {
                        'shares': allocated / date_data['open'].iloc[0],
                        'entry_price': date_data['open'].iloc[0],
                        'entry_date': date
                    }
                    self.capital -= allocated
                    del self.plan[symbol]

    def build_portfolio(self) -> Dict:
        """
        Build the initial portfolio based on the strategy parameters.
        Buys each stock at the earliest date when data is available.
        
        Returns:
            Dict containing portfolio composition and leftover capital
        """
        weights = self._calculate_weights()
        self.portfolio = {}
        self.leftover_capital = self.capital
        
        for symbol, symbol_data in self.stock_data.items():
            if not symbol_data.empty and symbol in weights:
                first_date = symbol_data['date'].min()
                first_row = symbol_data[symbol_data['date'] == first_date]
                
                if not first_row.empty:
                    first_price = first_row['open'].iloc[0]
                    capital_allocated = self.capital * weights[symbol]
                    shares = capital_allocated / first_price
                    
                    self.portfolio[symbol] = {
                        'shares': shares,
                        'entry_price': first_price,
                        'entry_date': first_date
                    }
                    self.leftover_capital -= capital_allocated
        
        if self.portfolio:
            all_entry_dates = [details['entry_date'] for details in self.portfolio.values()]
            self.all_capital_allocated_date = max(all_entry_dates)
        else:
            self.all_capital_allocated_date = None
        
        return self.portfolio
 
    def _calculate_portfolio_value(self, date: pd.Timestamp):
        """
        Calculate the portfolio value over time, starting from the specified start_date.
        Includes both stock values and leftover capital.
        
        Returns:
            DataFrame with portfolio values
        """
        
        total_value = self.capital
        for symbol, detail in self.portfolio.items():
            last_price = self._find_last_price(symbol, date)
            total_value += detail['shares'] * last_price

        return total_value

    def _find_last_price(self, symbol, date: pd.Timestamp):
        """Find the last available price on or before the given date."""
        symbol_data = self.stock_data[symbol]
        available_dates = symbol_data[symbol_data['date'] <= date]
        
        if available_dates.empty:
            min_date = symbol_data['date'].min()
            raise ValueError(f"No price data available for date {date} when earliest date is {min_date}")
        
        last_available_date = available_dates['date'].max()
        
        price = symbol_data.loc[symbol_data['date'] == last_available_date, 'adj close'].iloc[0]

        return price
