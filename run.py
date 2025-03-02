import pandas as pd
from pathlib import Path

from config import Config
from utils.plot import plot_values
from strategy.buy_and_hold import BuyAndHold
from strategy.follow_the_winner import MomentumFollowWinner

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
    bah_pool = BuyAndHold(
        pool,
        price_data,
        start_date=cf.start_date,
        end_date=cf.end_date,
    )

    bah_snp = BuyAndHold(
        ['SPY'],
        price_data,
        start_date=cf.start_date,
        end_date=cf.end_date
    )
    
    ftw = MomentumFollowWinner(
        pool,
        price_data,
        start_date=cf.start_date,
        end_date=cf.end_date,
        top_n=50
    )

    history1 = bah_pool.execute()
    history2 = bah_snp.execute()
    history3 = ftw.execute()
    print(bah_pool.evaluate(against=baseline))
    print(bah_snp.evaluate(against=baseline))
    print(ftw.evaluate(against=baseline))
    plot_values(history1)
    plot_values(history2)
    plot_values(history3)

if __name__ == '__main__':
    main()
