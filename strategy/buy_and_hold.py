import pandas as pd
from typing import List, Dict, Optional

from utils.eval import evaluate_strategy
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
        self.capital = initial_capital
        self.equal_weight = equal_weight
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        filtered_data = self._filter_date_range(data, self.start_date, self.end_date)
        
        self.stock_data = self._prepare_stock_data(filtered_data)
        
        self.portfolio, self.plan = {}, {}
        self.history = pd.DataFrame({'date': [self.start_date - pd.Timedelta(days=1)], 'portfolio_value': [self.capital]})

    def _filter_date_range(self, data, start_date: pd.Timestamp,  end_date: pd.Timestamp) -> pd.DataFrame:
        """Filter dataframe by date range."""
        data['date'] = pd.to_datetime(data['date'])
        if start_date:
            data = data.loc[data['date'] >= start_date]
        if end_date:
            data = data.loc[data['date'] <= end_date]

        return data
            
    def _prepare_stock_data(self, data) -> Dict:
        """Prepare individual stock data dictionaries."""
        stock_data = {}
        for symbol in self.pool:
            stock_df = data[data['Stock_symbol'] == symbol]
            if not stock_df.empty:
                stock_data[symbol] = stock_df

        return stock_data

    @monitor_time
    def execute(self):
        
        weights = self._calculate_weights()
        for symbol, weight in weights.items():
            self.plan[symbol] = self.capital * weight

        dates = pd.date_range(self.start_date, self.end_date)
        for date in dates: 
            self._update_portfolio(date)
            date_value = self._calculate_portfolio_value(date)
            self.history = pd.concat([self.history, pd.DataFrame({'date': [date], 'portfolio_value': [date_value]})], ignore_index=True)

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

    def evaluate(self, against: pd.DataFrame):
        return evaluate_strategy(self.history, against)
