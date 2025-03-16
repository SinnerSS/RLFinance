import pandas as pd
from pathlib import Path

def filter_stock(news_count: pd.DataFrame, price_dir: Path, strategy: str, pool_size: int):
    filtered_count = news_count[news_count['count'] >= 1000].copy()

    filtered_count['csv_exists'] = filtered_count['tic'].apply(lambda x: (price_dir / f'{x}.csv').exists())
    filtered_count = filtered_count[filtered_count['csv_exists'] == True].copy()
    filtered_count = filtered_count.drop('csv_exists', axis=1)

    sorted_count = filtered_count.sort_values(by='count', ascending=False)
    
    if len(sorted_count) < pool_size:
        raise ValueError(f'Pool size {pool_size} is larger then number of stocks {len(sorted_count)}.')

    match strategy:
        case 'mixed':
            stocks_per_category = pool_size // 3
            top_count = stocks_per_category + (pool_size % 3)
            
            top_stocks = sorted_count.head(top_count).copy()
            top_stocks['category'] = 'top'
            
            bottom_stocks = sorted_count.tail(stocks_per_category).copy()
            bottom_stocks['category'] = 'bottom'
            
            total_stocks = len(sorted_count)
            mid_start = (total_stocks - stocks_per_category) // 2
            mid_end = mid_start + stocks_per_category
            
            mid_stocks = sorted_count.iloc[mid_start:mid_end].copy()
            mid_stocks['category'] = 'middle'
            
            combined = pd.concat([top_stocks, mid_stocks, bottom_stocks])
            
            return combined
        case 'top':
            return sorted_count.head(pool_size)

        case 'bot':
            return sorted_count.tail(pool_size)

        case _:
            raise NotImplementedError(f'Strategy {strategy} not implemented.')
