import pandas as pd
from pathlib import Path

from config import Config
from utils.plot import plot_values
from loader.pool_loader import filter_stock
from loader.price_loader import load_price_data
from strategy.buy_and_hold import BuyAndHold
from strategy.follow_the_winner import MomentumFollowWinner
from strategy.follow_the_loser import MeanReversionFollowLoser

def main():
    cf = Config()

    post_path = cf.data_path / 'post'
    baseline_path = post_path / 'DGS3MO.csv'
    price_path = cf.data_path / 'Stock_price/full_history'

    news_count = pd.read_csv(post_path / 'news_counts.csv')
    baseline = pd.read_csv(baseline_path)

    pool = filter_stock(news_count, price_path, strategy='mixed', pool_size=500)
    price_data = load_price_data(pool, price_path)

    pool = pool['tic'].tolist()
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

    ftl = MeanReversionFollowLoser(
        pool,
        price_data,
        start_date=cf.start_date,
        end_date=cf.end_date,
        bottom_n=50
    )

    history1 = bah_pool.execute()
    history2 = bah_snp.execute()
    history3 = ftw.execute()
    history4 = ftl.execute()
    print(bah_pool.evaluate(against=baseline))
    print(bah_snp.evaluate(against=baseline))
    print(ftw.evaluate(against=baseline))
    print(ftl.evaluate(against=baseline))
    plot_values(history1)
    plot_values(history2)
    plot_values(history3)
    plot_values(history4)

if __name__ == '__main__':
    main()
