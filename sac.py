import pandas as pd
import numpy as np
import os
import logging
from typing import List
from stable_baselines3 import SAC # Changed import
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from config import Config # Assuming config.py exists
from env import StockTradingEnv # Assuming env.py exists

LOG_DIR = "./result"
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, "tensorboard")
MODEL_SAVE_DIR = os.path.join(LOG_DIR, "models")
REPORT_SAVE_DIR = os.path.join(LOG_DIR, "eval/reports") # For env's internal reporting
TEST_SAVE_DIR = os.path.join(LOG_DIR, "test/")

SELECTED_TICKERS = 'all'
TECHNICAL_FEATURES = [
    'open_log_ret', 'high_log_ret', 'low_log_ret', 'close_log_ret',
    'close', 'volume_log',
    'macd_norm', 'cci_30_norm', 'bb_pctB',
    'sma_short_ratio', 'sma_long_ratio',
    'rsi_30_norm', 'dx_30_norm'
]

# NEWS_FEATURES = [
#     'mean_sentiment_norm',
#     'news_count_norm',
#     'mean_recent_3_news_90d_norm',
#     'mean_recent_5_news_180d_norm'
# ]

FEATURES = TECHNICAL_FEATURES #+ NEWS_FEATURES
EVALUATE_BY = 'close'
LOOKBACK = 30
INITIAL_CAPITAL = 100000.0
MAX_EPISODE_STEPS = None
REWARD_SCALING = 1
LOG_METRICS_ENV = True

SAC_PARAMS = {
    "policy": "MlpPolicy",      
    "learning_rate": 3e-4,     
    "buffer_size": 100_000,       # Size of the replay buffer (adjust based on memory)
    "learning_starts": 10000,     # How many steps to collect data before starting training (important!)
    "batch_size": 256,            # Minibatch size for SAC updates
    "tau": 0.005,                 # Soft update coefficient for target networks
    "gamma": 0.99,                # Discount factor
    "train_freq": 1,              # Update the model every 'train_freq' steps
    "gradient_steps": 1,          # How many gradient steps to perform per update
    # "action_noise": None,       # SAC uses stochastic policy for exploration, explicit noise usually not needed unless specified type
    "ent_coef": 'auto',           # Entropy regularization coefficient ('auto' learns it)
    "target_update_interval": 1,  # Interval (in updates) to update target networks
    # "target_entropy": 'auto',   # Target entropy ('auto' calculates based on action space)
    "use_sde": False,             # State-Dependent Exploration (can sometimes help)
    "sde_sample_freq": -1,        # Frequency for sampling SDE noise (-1 disables default sampling)
    "tensorboard_log": TENSORBOARD_LOG_DIR,
    "verbose": 1,
}

TOTAL_TIMESTEPS = 200000
EVAL_FREQ = 5000 
N_EVAL_EPISODES = 1
CHECKPOINT_FREQ = 20000

def load_data(file_path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['date', 'tic']).reset_index(drop=True)
        logger.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def make_env(data, tickers, features, evaluate_by, lookback, initial_capital, max_episode_steps, reward_scaling, log_metrics, log_dir, seed=0, train=True):
    def _init():
        env = StockTradingEnv(
            data=data,
            tic_symbols=tickers,
            features=features,
            evaluate_by=evaluate_by,
            lookback=lookback,
            initial_capital=initial_capital,
            max_episode_step=max_episode_steps,
            reward_scaling=reward_scaling,
            log_metrics=log_metrics,
            log_dir=log_dir
        )
        if not train:
             monitor_path = os.path.join(TEST_SAVE_DIR, f"monitor_sac_{seed}.csv") # Changed prefix
             os.makedirs(os.path.dirname(monitor_path), exist_ok=True)
             env = Monitor(env,
                          filename=monitor_path,
                          allow_early_resets=True,
                          info_keywords=("date", "portfolio_value"))
        return env
    return _init
# --- End Shared Functions ---

if __name__ == "__main__":
    config = Config()
    data_path = config.cwd / "data"
    train_path = data_path / "train.csv"
    eval_path = data_path / "eval.csv"
    test_path = data_path / "test.csv"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(REPORT_SAVE_DIR, exist_ok=True)
    os.makedirs(TEST_SAVE_DIR, exist_ok=True)

    try:
        train_data = load_data(train_path)
        eval_data = load_data(eval_path)
        test_data = load_data(test_path)
        def normalize_features(
            df: pd.DataFrame,
            price_cols: list = ['open', 'high', 'low', 'close'],
            vol_col: str = 'volume',
            macd_col: str = 'macd',
            cci_col: str = 'cci_30',
            bb_ub: str = 'boll_ub',
            bb_lb: str = 'boll_lb',
            sma_short: str = 'close_30_sma',
            sma_long: str = 'close_60_sma',
            rsi_col: str = 'rsi_30',
            dx_col: str = 'dx_30',
            #news_col: List[str] = ['mean_sentiment', 'news_count', 'mean_recent_3_news_90d', 'mean_recent_5_news_180d'],
            window: int = 30
            ):
            epsilon = 1e-8
            for col in price_cols:
                df[f'{col}_log_ret'] = np.log(df[col] / (df[col].shift(1) + epsilon))
                df[f'{col}_log_ret'] = df[f'{col}_log_ret'].ffill().bfill()

            df['volume_log'] = np.log1p(df[vol_col])
            vol_mean = df['volume_log'].rolling(window).mean().shift(1)
            vol_std = df['volume_log'].rolling(window).std(ddof=0).shift(1)
            df['volume_norm'] = (df['volume_log'] - vol_mean) / (vol_std + epsilon)
            df['volume_norm'] = df['volume_norm'].ffill().bfill()

            for col in [macd_col, cci_col]:
                m_mean = df[col].rolling(window).mean().shift(1)
                m_std = df[col].rolling(window).std(ddof=0).shift(1)
                df[f'{col}_norm'] = (df[col] - m_mean) / (m_std + epsilon)
                df[f'{col}_norm'] = df[f'{col}_norm'].ffill().bfill()

            df['bb_pctB'] = (df['close'] - df[bb_lb]) / (df[bb_ub] - df[bb_lb] + epsilon)
            df['bb_pctB'] = df['bb_pctB'].ffill().bfill()

            df['sma_short_ratio'] = df['close'] / (df[sma_short] + epsilon) - 1
            df['sma_long_ratio'] = df['close'] / (df[sma_long] + epsilon) - 1
            df['sma_short_ratio'] = df['sma_short_ratio'].ffill().bfill()
            df['sma_long_ratio'] = df['sma_long_ratio'].ffill().bfill()

            df[f'{rsi_col}_norm'] = df[rsi_col] / 100
            df[f'{dx_col}_norm'] = df[dx_col] / 100
            # for col in news_col:
            #     m_mean = df[col].rolling(window).mean().shift(1)
            #     m_std = df[col].rolling(window).std(ddof=0).shift(1)
            #     df[f'{col}_norm'] = (df[col] - m_mean) / (m_std + epsilon)
            #     df[f'{col}_norm'] = df[f'{col}_norm'].ffill().bfill()

            assert not df.isnull().values.any(), "Null values found after normalization"
            return df

        train_data = normalize_features(train_data, window=LOOKBACK)
        eval_data = normalize_features(eval_data, window=LOOKBACK)
        test_data = normalize_features(test_data, window=LOOKBACK)


        env = DummyVecEnv([make_env(train_data, "all", FEATURES, EVALUATE_BY, LOOKBACK,
                                   INITIAL_CAPITAL, MAX_EPISODE_STEPS, REWARD_SCALING,
                                   log_metrics=False,
                                   log_dir=None)])

        eval_env = DummyVecEnv([make_env(eval_data, "all", FEATURES, EVALUATE_BY, LOOKBACK,
                                       INITIAL_CAPITAL, MAX_EPISODE_STEPS, reward_scaling=1.0,
                                       log_metrics=LOG_METRICS_ENV, 
                                       log_dir=REPORT_SAVE_DIR)])

        test_env = DummyVecEnv([make_env(test_data, "all", FEATURES, EVALUATE_BY, LOOKBACK,
                                       INITIAL_CAPITAL, MAX_EPISODE_STEPS, reward_scaling=1.0,
                                       log_metrics=LOG_METRICS_ENV, 
                                       log_dir=TEST_SAVE_DIR)])

        best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model")

        checkpoint_callback = CheckpointCallback(
            save_freq=max(CHECKPOINT_FREQ // env.num_envs, 1),
            save_path=MODEL_SAVE_DIR,
            name_prefix="sac_stock_trader",
            save_replay_buffer=False,
            save_vecnormalize=False
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_path,
            log_path=os.path.join(LOG_DIR, "eval"),
            eval_freq=max(EVAL_FREQ // env.num_envs, 1), 
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True, 
            render=False,
        )


        model = SAC(
            env=env,
            seed=42,
            **SAC_PARAMS
        )

        logger.info("Starting SAC training...")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            log_interval=1,
            progress_bar=True
        )
        logger.info("Training finished.")

        final_model_path = os.path.join(MODEL_SAVE_DIR, "sac_stock_trader_final")
        model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

        logger.info("Running final test on the trained model...")
        model = SAC.load(os.path.join(best_model_path, "best_model.zip"), env=test_env)
        obs = test_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = test_env.step(action)
            done = any(dones)


        logger.info("Final evaluation complete. Check report files if generated.")


    except FileNotFoundError:
        logger.error("Critical error: Data file not found. Exiting.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)

