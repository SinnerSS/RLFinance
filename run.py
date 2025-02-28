import pickle
import pandas as pd
from pathlib import Path

import config

from strategy.buy_and_hold import BuyAndHoldStrategy
from utils.plot import plot_performance

def main():
    cwd = Path.cwd()
    post_path  = cwd / config.DATA_PATH / 'post'
    pool_path = post_path / 'pool.csv'
    price_path = post_path / 'price.csv'
    interim_path = cwd / config.INTERIM_PATH

    pool = pd.read_csv(pool_path)
    price_data = pd.read_csv(price_path)

    pool = pool['Stock_symbol'].tolist()
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
