import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import Counter

import config

def news_by_stock(news: pd.DataFrame) -> Counter:

    all_counts = Counter()
    
    counts = news['Stock_symbol'].value_counts().to_dict()
    all_counts.update(counts)

    return all_counts

def filter_stock(news_count: pd.DataFrame, price_dir: Path):

    filtered_count = news_count[news_count['Count'] >= 1000].copy()

    filtered_count['csv_exists'] = filtered_count['Stock_symbol'].apply(lambda x: (price_dir / f'{x}.csv').exists())
    filtered_count = filtered_count[filtered_count['csv_exists'] == True].copy()
    filtered_count = filtered_count.drop('csv_exists', axis=1)

    sorted_count = filtered_count.sort_values(by='Count', ascending=False)
    
    if len(sorted_count) < config.POOL_SIZE:
        raise ValueError(f'Pool size {config.POOL_SIZE} is larger then number of stocks {len(sorted_count)}.')

    match config.POOL_STRATEGY:
        case 'mixed':
            stocks_per_category = config.POOL_SIZE // 3
            top_count = stocks_per_category + (config.POOL_SIZE % 3)
            
            top_stocks = sorted_count.head(top_count).copy()
            top_stocks['Category'] = 'Top'
            
            bottom_stocks = sorted_count.tail(stocks_per_category).copy()
            bottom_stocks['Category'] = 'Bottom'
            
            total_stocks = len(sorted_count)
            mid_start = (total_stocks - stocks_per_category) // 2
            mid_end = mid_start + stocks_per_category
            
            mid_stocks = sorted_count.iloc[mid_start:mid_end].copy()
            mid_stocks['Category'] = 'Middle'
            
            combined = pd.concat([top_stocks, mid_stocks, bottom_stocks])
            
            return combined
        case 'top':
            return sorted_count.head(config.POOL_SIZE)

        case 'bot':
            return sorted_count.tail(config.POOL_SIZE)

        case _:
            raise NotImplementedError(f'Strategy {config.POOL_STRATEGY} not implemented.')

def capitalize_files(data_dir: Path):
    count = 0
    for file_path in data_dir.iterdir():
        if file_path.is_file():
            stem = file_path.stem.upper()
            suffix = file_path.suffix
            
            new_filename = stem + suffix
            new_file_path = data_dir / new_filename
            
            file_path.rename(new_file_path)
            count += 1
    return count

def load_price_data(pool: pd.DataFrame, data_dir: Path):

    stock_symbols = pool['Stock_symbol'].tolist()
    
    stock_list = []
    for symbol in stock_symbols:
        file_path = data_dir / f'{symbol}.csv'
        stock_price = pd.read_csv(file_path)
        stock_price['Stock_symbol'] = symbol
        stock_list.append(stock_price)

    price_data = pd.concat(stock_list, ignore_index=True)

    return price_data

def main():

    post_path = Path.cwd() / config.DATA_PATH / 'post'
    price_path = Path.cwd() / config.DATA_PATH / 'Stock_price/full_history'
    if not post_path.is_dir():
        post_path.mkdir()

    news_path = Path.cwd() / config.DATA_PATH / 'Stock_news/nasdaq_exteral_data.csv'

    all_counts = Counter()
    for chunk in tqdm(pd.read_csv(news_path, chunksize=10000), desc='Counting news by stock'):
        all_counts.update(news_by_stock(chunk))

    news_counts = pd.DataFrame(all_counts.items(), columns=['Stock_symbol', 'Count'])
    news_counts.to_csv(post_path / 'news_counts.csv', index=False)

    capitalize_files(price_path)
    pool = filter_stock(news_counts, price_path)

    pool.to_csv(post_path / 'pool.csv', index=False) 

    price_data = load_price_data(pool, price_path)

    price_data.to_csv(post_path / 'price.csv', index=False) 

if __name__ == '__main__':
    main()
