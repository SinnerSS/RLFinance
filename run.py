import pandas as pd
from pathlib import Path

from config import Config
from utils.plot import plot_values
from loader.pool_loader import filter_stock
from loader.price_loader import load_price_data
from strategy import BuyAndHold, MomentumFollowWinner, MeanReversionFollowLoser, AnticorStrategy

def main():
    cf = Config()

    post_path = cf.data_path / 'post'
    baseline_path = post_path / 'DGS3MO.csv'
    price_path = cf.data_path / 'Stock_price/full_history'

    news_count = pd.read_csv(post_path / 'news_counts.csv')
    baseline = pd.read_csv(baseline_path)

    pool = filter_stock(news_count, price_path, strategy='mixed', pool_size=500)
    price_data = load_price_data(pool, price_path)

    # bah_pool = BuyAndHold(
    #     price_data,
    #     start_date=cf.start_date,
    #     end_date=cf.end_date,
    # )
    #
    # bah_snp = BuyAndHold(
    #     price_data,
    #     start_date=cf.start_date,
    #     end_date=cf.end_date,
    #     pool=['SPY']
    # )
    # 
    # ftw = MomentumFollowWinner(
    #     price_data,
    #     start_date=cf.start_date,
    #     end_date=cf.end_date,
    #     top_n=50
    # )
    #
    # ftl = MeanReversionFollowLoser(
    #     price_data,
    #     start_date=cf.start_date,
    #     end_date=cf.end_date,
    #     bottom_n=50
    # )

    anticor = AnticorStrategy(
        price_data,
        start_date=cf.start_date,
        end_date=cf.end_date,
    )


    # history1 = bah_pool.execute()
    # history2 = bah_snp.execute()
    # history3 = ftw.execute()
    # history4 = ftl.execute()
    history5 = anticor.execute()
    # print(bah_pool.evaluate(against=baseline))
    # print(bah_snp.evaluate(against=baseline))
    # print(ftw.evaluate(against=baseline))
    # print(ftl.evaluate(against=baseline))
    print(anticor.evaluate(against=baseline))
    # plot_values(history1)
    # plot_values(history2)
    # plot_values(history3)
    # plot_values(history4)
    plot_values(history5)

if __name__ == '__main__':
    main()
