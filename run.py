import pandas as pd
from pathlib import Path

from config import Config
from utils.plot import plot_values
from strategy.buy_and_hold import BuyAndHold

def main():
    cf = Config()
    post_path = cf.data_path / 'post'

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
        start_date=cf.start_date,
        end_date=cf.end_date,
    )

    history = strategy.execute()
    history.to_csv(cf.result_path / 'history.csv', index=False)
    print(strategy.evaluate(against=baseline))
    plot_values(history)

if __name__ == '__main__':
    main()
