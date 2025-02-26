import pickle
import pandas as pd
from pathlib import Path

import config

from pre.pool import process_pool
from pre.price import process_stock_price, capitalize_files
from strategy.buy_and_hold import BuyAndHoldStrategy
from utils.plot import plot_performance

def main():
    cwd = Path.cwd()
    pool_path = cwd / config.POOL_PATH
    price_path = cwd / config.PRICE_PATH
    interim_path = cwd / config.INTERIM_PATH
    if not interim_path.is_dir():
        interim_path.mkdir()
        pool = pd.read_csv(pool_path)
        pool = process_pool(pool, interim_path / 'pool.pkl')
        capitalize_files(price_path)
        price_data = process_stock_price(pool, price_path, interim_path / 'price.parquet') 
    else: 
        with open(interim_path / 'pool.pkl', 'rb') as f:
            pool = pickle.load(f)
        price_data = pd.read_parquet(interim_path / 'price.parquet')

    strategy = BuyAndHoldStrategy(
        pool,
        price_data,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
    )

    performance = strategy.calculate_performance()
    plot_performance(performance)
if __name__ == '__main__':
    main()
