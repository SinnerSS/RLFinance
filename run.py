import argparse
import pandas as pd

import torch
from sklearn.preprocessing import MaxAbsScaler
from finrl.meta.preprocessor.preprocessors import GroupByScaler
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv

from config import Config
from utils.plot import plot_values
from loader.price_loader import load_price_strategy, load_price_model
from strategy import BuyAndHold, Anticor, UniversalPortfolio, NearestNeighbor, Corn

def main():
    parser = argparse.ArgumentParser(description='Run trading strategies analysis.')
    parser.add_argument('--strategy', action='store_true', help='Execute all available strategy')
    parser.add_argument('--model', action='store_true', help='Execute model')
    args = parser.parse_args()
    cf = Config()

    post_path = cf.data_path / 'post'
    baseline_path = post_path / 'DGS3MO.csv'
    price_path = cf.data_path / 'Stock_price/full_history'
    baseline = pd.read_csv(baseline_path)
    with open("default_pool.txt", "r") as file:
        pool = [line.strip() for line in file if line.strip()]


    if args.strategy:
        price_data = load_price_strategy(pool + ['SPY'], price_path)

        bah_pool = BuyAndHold(
            price_data,
            start_date=cf.start_test,
            end_date=cf.end_test,
        )

        bah_snp = BuyAndHold(
            price_data,
            start_date=cf.start_test,
            end_date=cf.end_test,
            pool=['SPY']
        )

        up = UniversalPortfolio(
            price_data,
            start_date=cf.start_test,
            end_date=cf.end_test,
            num_candidates=100,
            beta=2,
            seed=42,
        )

        nn = NearestNeighbor(
            price_data,
            start_date=cf.start_test,
            end_date=cf.end_test,
            k=10,
            l=30,
            rebalance_freq='W-FRI',
            max_leverage=1.0,
            metric='return'
        )

        anticor = Anticor(
            price_data,
            start_date=cf.start_test,
            end_date=cf.end_test,
        )

        corn = Corn(
            price_data,
            start_date=cf.start_test,
            end_date=cf.end_test,
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
    
    if args.model:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        price_data = load_price_model(pool, price_path)
        price_norm_data = GroupByScaler(by="tic", scaler=MaxAbsScaler).fit_transform(price_data)

        price_norm_data = price_norm_data[['date', 'tic', 'close', 'high', 'low']]
        price_norm_data['date'] = pd.to_datetime(price_norm_data['date'])


        train_data = price_norm_data[(price_norm_data['date'] >= cf.start_train) & (price_norm_data['date'] <= cf.end_train)]
        test_data = price_norm_data[(price_norm_data['date'] >= cf.start_test) & (price_norm_data['date'] <= cf.end_test)] 
        
        train_env = PortfolioOptimizationEnv(
            train_data,
            initial_amount=10000,
            time_window=50,
            features=['close', 'high', 'low'],
            normalize_df=None
        )
        test_env = PortfolioOptimizationEnv(
            test_data,
            initial_amount=10000,
            time_window=50,
            features=['close', 'high', 'low'],
            normalize_df=None
        )

        model_kwargs = {
            "lr": 0.01,
            "policy": EIIE,
        }

        policy_kwargs = {
            "k_size": 3,
            "time_window": 50,
        }

        model = DRLAgent(train_env).get_model("pg", device, model_kwargs, policy_kwargs)
        DRLAgent.train_model(model, episodes=100)

        
        DRLAgent.DRL_validation(model, test_env)

if __name__ == '__main__':
    main()
