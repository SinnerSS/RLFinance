from typing import List, Dict, Optional
import pandas as pd

class BuyAndHoldStrategy:
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
        self.initial_capital = initial_capital
        self.equal_weight = equal_weight
        self.portfolio = {}
        self.stock_data = {}
        
        self._filter_date_range(start_date, end_date)
        
        self._prepare_stock_data()
        
    def _filter_date_range(self, start_date: Optional[str], end_date: Optional[str]) -> None:
        """Filter dataframe by date range."""
        self.data['date'] = pd.to_datetime(self.data['date'])
        if start_date:
            self.data = self.data.loc[self.data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            self.data = self.data.loc[self.data['date'] <= pd.to_datetime(end_date)]
            
    def _prepare_stock_data(self) -> None:
        """Prepare individual stock data dictionaries."""
        for symbol in self.pool:
            stock_df = self.data[self.data['Stock_symbol'] == symbol]
            if not stock_df.empty:
                self.stock_data[symbol] = stock_df
    
    def _calculate_weights(self) -> Dict[str, float]:
        """Calculate portfolio allocation weights."""
        weights = {}
        if self.equal_weight:
            weight_per_stock = 1.0 / len(self.stock_data)
            for symbol in self.stock_data:
                weights[symbol] = weight_per_stock
        else:
            # Implement alternative weighting strategy here
            # This would be implemented in a subclass or added as needed
            pass
        return weights
    
    def build_portfolio(self) -> Dict:
        """
        Build the initial portfolio based on the strategy parameters.
        
        Returns:
            Dict containing portfolio composition
        """
        weights = self._calculate_weights()
        
        for symbol, symbol_data in self.stock_data.items():
            if not symbol_data.empty:
                first_price = symbol_data['adj close'].iloc[0]
                capital_allocated = self.initial_capital * weights[symbol]
                shares = capital_allocated / first_price
                
                self.portfolio[symbol] = {
                    'shares': shares,
                    'entry_price': first_price,
                    'entry_date': symbol_data['date'].iloc[0]
                }
        
        return self.portfolio
    
    def calculate_performance(self) -> Dict:
        """
        Calculate portfolio performance over time.
        
        Returns:
            Dict containing portfolio composition and time series values
        """
        if not self.portfolio:
            self.build_portfolio()
            
        portfolio_values = self._calculate_portfolio_value()
        print(portfolio_values)
        
        return {
            'portfolio': self.portfolio,
            'values': portfolio_values
        }
    
    def _calculate_portfolio_value(self) -> pd.DataFrame:
        """
        Calculate the portfolio value over time.
        
        Returns:
            DataFrame with portfolio values
        """
        
        dates = sorted(pd.unique(self.data['date']))
        values = []
        
        for date in dates:
            daily_value = 0
            for symbol, details in self.portfolio.items():
                symbol_data = self.stock_data[symbol]
                price_on_date = symbol_data.loc[symbol_data['date'] == date, 'adj close']
                if not price_on_date.empty:
                    daily_value += details['shares'] * price_on_date.iloc[0]
            
            values.append({'date': date, 'portfolio_value': daily_value})
            
        return pd.DataFrame(values).reset_index()
