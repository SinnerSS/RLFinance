import pandas as pd
from pathlib import Path

import config

from strategy.buy_and_hold import BuyAndHold
from utils.plot import plot_values

def main():
    cwd = Path.cwd()
    post_path  = cwd / config.DATA_PATH / 'post'
    pool_path = post_path / 'pool.csv'
    price_path = post_path / 'price.csv'
    interim_path = cwd / config.INTERIM_PATH

    pool = pd.read_csv(pool_path)
    price_data = pd.read_csv(price_path)

    pool = pool['Stock_symbol'].tolist()
    strategy = BuyAndHold(
        pool,
        price_data,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
    )

    history = strategy.execute()
    plot_values(history)
if __name__ == '__main__':
    main()
