import argparse
import pandas as pd

from config import Config
from utils.plot import plot_values
from loader.pool_loader import filter_stock
from loader.price_loader import load_price_data
from strategy import BuyAndHold, Anticor, UniversalPortfolio, NearestNeighbor, Corn

def main():
    parser = argparse.ArgumentParser(description='Run trading strategies analysis.')
    parser.add_argument('--strategy', action='store_true', help='Execute all available strategy')
    args = parser.parse_args()
    cf = Config()

    post_path = cf.data_path / 'post'
    baseline_path = post_path / 'DGS3MO.csv'
    price_path = cf.data_path / 'Stock_price/full_history'


    if args.strategy:
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

        nn = NearestNeighbor(
            price_data,
            start_date=cf.start_date,
            end_date=cf.end_date,
            k=10,
            l=30,
            rebalance_freq='W-FRI',
            max_leverage=1.0,
            metric='return'
        )

        anticor = Anticor(
            price_data,
            start_date=cf.start_date,
            end_date=cf.end_date,
        )

        corn = Corn(
            price_data,
            start_date=cf.start_date,
            end_date=cf.end_date,
            rho=0.1,
            window=10,
            version='fast'
        )


        history1 = bah_pool.execute()
        history2 = bah_snp.execute()
        history3 = up.execute()
        history4 = nn.execute()
        history5 = anticor.execute()
        history6 = corn.execute()

        print(bah_pool.evaluate(against=baseline))
        print(bah_snp.evaluate(against=baseline))
        print(up.evaluate(against=baseline))
        print(nn.evaluate(against=baseline))
        print(anticor.evaluate(against=baseline))
        print(corn.evaluate(against=baseline))

        plot_values(history1)
        plot_values(history2)
        plot_values(history3)
        plot_values(history4) 
        plot_values(history5)
        plot_values(history6)

if __name__ == '__main__':
    main()
