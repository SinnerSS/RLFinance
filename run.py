import pandas as pd
from pathlib import Path

from config import Config
from utils.plot import plot_values
from loader.pool_loader import filter_stock
from loader.price_loader import load_price_data
from strategy import BuyAndHold, Anticor, UniversalPortfolio, NearestNeighbor

def main():
    cf = Config()

    post_path = cf.data_path / 'post'
    baseline_path = post_path / 'DGS3MO.csv'
    price_path = cf.data_path / 'Stock_price/full_history'


    # news_count = pd.read_csv(post_path / 'news_counts.csv')
    baseline = pd.read_csv(baseline_path)

    with open("default_pool.txt", "r") as file:
        pool = [line.strip() for line in file if line.strip()]

    # pool = filter_stock(news_count, price_path, strategy='mixed', pool_size=500)
    price_data = load_price_data(pool + ['SPY'], price_path)

    bah_pool = BuyAndHold(
        price_data,
        start_date=cf.start_date,
        end_date=cf.end_date,
    )

    bah_snp = BuyAndHold(
        price_data,
        start_date=cf.start_date,
        end_date=cf.end_date,
        pool=['SPY']
    )

    up = UniversalPortfolio(
        price_data,
        start_date=cf.start_date,
        end_date=cf.end_date,
        num_candidates=100,
        beta=2,
        seed=42,
    )

    bnn = NearestNeighbor(
        price_data,
        start_date=cf.start_date,
        end_date=cf.end_date,
        k=5,
        l=10,
        rebalance_freq='W-FRI',
        max_leverage=1.0,
        metric='return'
    )

    anticor = Anticor(
        price_data,
        start_date=cf.start_date,
        end_date=cf.end_date,
    )


    # history1 = bah_pool.execute()
    # history2 = bah_snp.execute()
    history3 = up.execute()
    history4 = nearest_neighbor.execute()
    history5 = anticor.execute()

    # print(bah_pool.evaluate(against=baseline))
    # print(bah_snp.evaluate(against=baseline))
    print(up.evaluate(against=baseline))
    print(nearest_neighbor.evaluate(against=baseline))
    print(anticor.evaluate(against=baseline))

    # plot_values(history1)
    # plot_values(history2)
    plot_values(history3)
    plot_values(history4) 
    plot_values(history5)

if __name__ == '__main__':
    main()
