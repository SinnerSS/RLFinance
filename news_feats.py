import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load the data
data = pd.read_csv('data/Stock_news/deepseek_responses.csv')

data['date'] = pd.to_datetime(data['date'])

# Calculate daily sentiment mean for each stock
daily_sentiment = data.groupby(['date', 'stock_symbol'])['sentiment_score'].mean().reset_index()
daily_sentiment.rename(columns={'sentiment_score': 'mean_sentiment'}, inplace=True)

# Calculate daily news count for each stock
daily_news_count = data.groupby(['date', 'stock_symbol']).size().reset_index(name='news_count')

# Get all unique dates and stocks
all_dates = pd.date_range(start=data['date'].min(), end=data['date'].max())
all_stocks = data['stock_symbol'].unique()

# Create a complete grid of all dates and stocks
date_grid = pd.MultiIndex.from_product([all_dates, all_stocks], names=['date', 'stock_symbol']).to_frame(index=False)

# Function to efficiently compute rolling sentiment based on recent N news items in a given time window
def compute_rolling_sentiment(df, window_days, n_news):
    results = []
    
    # Group by stock symbol to process each stock independently
    for stock, stock_data in df.groupby('stock_symbol'):
        stock_data = stock_data.sort_values('date')
        
        # For each date in the grid
        for date in all_dates:
            # Define the window start date
            window_start = date - timedelta(days=window_days)
            
            # Get all news within the window for this stock
            window_data = stock_data[(stock_data['date'] >= window_start) & 
                                    (stock_data['date'] <= date)]
            
            # Take the most recent n_news items within the window
            recent_news = window_data.tail(n_news)
            
            # Calculate the mean sentiment if there's any news
            if len(recent_news) > 0:
                mean_sentiment = recent_news['sentiment_score'].mean()
                # Record how many news items were used for this calculation
                news_used = len(recent_news)
            else:
                mean_sentiment = np.nan
                news_used = 0
                
            results.append({
                'date': date,
                'stock_symbol': stock,
                f'mean_recent_{n_news}_news_{window_days}d': mean_sentiment,
                f'news_used_{n_news}_news_{window_days}d': news_used
            })
    
    return pd.DataFrame(results)

# Compute the rolling sentiment metrics
rolling_90d_3n = compute_rolling_sentiment(data, 90, 3)
rolling_180d_5n = compute_rolling_sentiment(data, 180, 5)

# Merge daily sentiment and news count with the date grid
result = date_grid.merge(daily_sentiment, on=['date', 'stock_symbol'], how='left')
result = result.merge(daily_news_count, on=['date', 'stock_symbol'], how='left')

# Fill NaN values for news_count with 0 (days with no news)
result['news_count'] = result['news_count'].fillna(0)

# Add rolling sentiment metrics
result = result.merge(rolling_90d_3n, on=['date', 'stock_symbol'], how='left')
result = result.merge(rolling_180d_5n, on=['date', 'stock_symbol'], how='left')

# Reset index to ensure a flat dataframe structure
result = result.reset_index(drop=True)

# Display the result
print(result.head(10))
print(f"Total rows in result: {len(result)}")
print(f"Memory usage: {result.memory_usage(deep=True).sum() / 1024:.2f} KB")
