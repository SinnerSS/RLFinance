import csv 
import argparse
import pandas as pd
from pathlib import Path

import torch
from sklearn.preprocessing import MaxAbsScaler
from finrl.config import INDICATORS
from finrl.agents.portfolio_optimization.architectures import EIIE
from finrl.meta.preprocessor.preprocessors import GroupByScaler, FeatureEngineer

from config import Config
from utils.plot import plot_values
from loader.price_loader import load_price_strategy, load_price_model
from strategy import BuyAndHold, Anticor, UniversalPortfolio, NearestNeighbor, Corn
from agent.ppo import PPOAgent
from env.env import LoggedPortfolioOptimizationEnv

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


        csv_file = cf.result_path / "evaluation_metrics.csv"

        data = [
            {"parameter_name": "bah_pool", **bah_pool.evaluate(against=baseline)},
            {"parameter_name": "bah_snp", **bah_snp.evaluate(against=baseline)},
            {"parameter_name": "up", **up.evaluate(against=baseline)},
            {"parameter_name": "nn", **nn.evaluate(against=baseline)},
            {"parameter_name": "anticor", **anticor.evaluate(against=baseline)},
            {"parameter_name": "corn", **corn.evaluate(against=baseline)},
        ]

        headers = list(data[0].keys())

        with open(csv_file, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)

        history1.to_csv(cf.result_path / 'bahp_history.csv')
        history2.to_csv(cf.result_path / 'bahs_history.csv')
        history3.to_csv(cf.result_path / 'up_history.csv')
        history4.to_csv(cf.result_path / 'nn_history.csv')
        history5.to_csv(cf.result_path / 'anticor_history.csv')
        history6.to_csv(cf.result_path / 'corn_history.csv')

    
    if args.model:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        price_data = load_price_model(pool, price_path)
        price_data['date'] = pd.to_datetime(price_data['date'])

        feature_engineer = FeatureEngineer(
           use_technical_indicator=True,
           tech_indicator_list=INDICATORS,
           use_turbulence=True
        )

        price_data = feature_engineer.preprocess_data(price_data)
        print(price_data.isna().any(axis=None))

        train_data = price_data[(price_data['date'] >= cf.start_train) & (price_data['date'] <= cf.end_train)]
        val_data = price_data[(price_data['date'] >= cf.start_val) & (price_data['date'] <= cf.end_val)]
        test_data = price_data[(price_data['date'] >= cf.start_test) & (price_data['date'] <= cf.end_test)] 
        
        env_kwargs = {
            "initial_amount": 100000,
            "features": ["close", "high", "low", "turbulence"] + INDICATORS,
            "valuation_feature": "close",
            "time_window": 50,
            "return_last_action": True,
            "new_gym_api": True,
            "order_df": False,
            "cwd": "./portfolio_ppo_results"
        }

        train_env = LoggedPortfolioOptimizationEnv(df=train_data.copy(), **env_kwargs)
        val_env = LoggedPortfolioOptimizationEnv(df=val_data.copy(), **env_kwargs)
        test_env = LoggedPortfolioOptimizationEnv(df=test_data.copy(), **env_kwargs)

        policy_kwargs_eii_e = {
            "initial_features": len(env_kwargs["features"]),
            "time_window": env_kwargs["time_window"],
        }

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log_directory = "./result/portfolio_ppo"

        agent = PPOAgent(
            env=train_env,
            validation_env=val_env,
            policy_class=EIIE,
            policy_kwargs=policy_kwargs_eii_e,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            lambda_gae=0.95,
            clip_epsilon=0.2,
            lr_actor=1e-4,
            lr_critic=1e-3,
            entropy_coef=0.0001,
            max_grad_norm=0.5,
            device=device,
            log_dir=log_directory,
            validation_freq=2048,
            use_pvm=False
        )

        print(f"Starting training... Device: {agent.device}")
        print(f"Check TensorBoard logs in: {agent.log_dir}")
        print(f"To view logs: tensorboard --logdir {Path(log_directory).parent}")
        agent.train(total_timesteps=200000) # Adjust total steps

        agent.test(test_env)
if __name__ == '__main__':
    main()
