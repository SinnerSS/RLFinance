import pandas as pd
from pathlib import Path

import config

from strategy.buy_and_hold import BuyAndHold
from utils.plot import plot_values

def main():
    cwd = Path.cwd()
    post_path  = cwd / config.DATA_PATH / 'post'
    result_path = cwd / config.RESULT_PATH

    if not result_path.is_dir():
        result_path.mkdir()
    pool_path = post_path / 'pool.csv'
    price_path = post_path / 'price.csv'
    baseline_path = post_path / 'DGS3MO.csv'

    pool = pd.read_csv(pool_path)
    price_data = pd.read_csv(price_path)
    baseline = pd.read_csv(baseline_path)

    pool = pool['Stock_symbol'].tolist()
    strategy = BuyAndHold(
        pool,
        price_data,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
    )

    history = strategy.execute()
    history.to_csv(result_path / 'history.csv', index=False)
    plot_values(history)

if __name__ == '__main__':
    main()
