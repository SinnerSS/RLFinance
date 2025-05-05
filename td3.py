import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from pathlib import Path
from typing import List
from stable_baselines3 import TD3 # Changed import
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise # TD3 uses action noise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from config import Config # Assuming config.py exists
from env import StockTradingEnv # Assuming env.py exists

LOG_DIR = "./result_td3" # Changed log dir
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

NEWS_FEATURES = [
    'mean_sentiment_norm',
    'news_count_norm',
    'mean_recent_3_news_90d_norm',
    'mean_recent_5_news_180d_norm'
]

FEATURES = TECHNICAL_FEATURES + NEWS_FEATURES
EVALUATE_BY = 'close'
LOOKBACK = 1 
INITIAL_CAPITAL = 100000.0
MAX_EPISODE_STEPS = None
REWARD_SCALING = 1
LOG_METRICS_ENV = True


action_dim = 30 
action_noise_sigma = 0.1 # Standard deviation for noise - TUNE THIS CAREFULLY

action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=action_noise_sigma * np.ones(action_dim))

TD3_PARAMS = {
    "policy": "MlpPolicy",           # Using standard MlpPolicy
    "learning_rate": 1e-3,           # Often 1e-3 or 3e-4 for TD3
    "buffer_size": 1_000_000,        # Size of the replay buffer (can be large for TD3)
    "learning_starts": 10000,        # Steps before training starts (important!)
    "batch_size": 100,               # Minibatch size (often 100 or 256)
    "tau": 0.005,                    # Soft update coefficient
    "gamma": 0.99,                   # Discount factor
    "train_freq": (1, "step"),       # Train every step
    "gradient_steps": 1,             # How many gradient steps per update
    "action_noise": action_noise,    # Action noise for exploration
    "policy_delay": 2,               # TD3 specific: Delay policy updates (every 2 Q updates)
    "target_policy_noise": 0.2,      # TD3 specific: Noise added to target policy actions
    "target_noise_clip": 0.5,        # TD3 specific: Clip target policy noise
    # "optimize_memory_usage": False,
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
    news_col: List[str] = ['mean_sentiment', 'news_count', 'mean_recent_3_news_90d', 'mean_recent_5_news_180d'],
    window: int = 30
    ):
    """Normalizes features, handling NaNs carefully using grouped operations."""
    epsilon = 1e-8
    df = df.copy() # Work on a copy to avoid SettingWithCopyWarning

    # Group by ticker for rolling calculations to prevent lookahead bias across tickers
    grouped = df.groupby('tic', group_keys=False) # group_keys=False avoids adding tic index

    # Log returns (handle potential zero/negative prices)
    for col in price_cols:
        # Use transform to apply function within each group and align output with original df index
        df[f'{col}_log_ret'] = grouped[col].transform(
            lambda x: np.log(x / (x.shift(1).replace(0, epsilon)) + epsilon) # Add epsilon inside log for safety
        )

    # Log Volume + Rolling Normalization
    df['volume_log'] = np.log1p(df[vol_col])
    vol_mean = grouped['volume_log'].transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
    vol_std = grouped['volume_log'].transform(lambda x: x.rolling(window=window, min_periods=1).std(ddof=0).shift(1))
    df['volume_norm'] = (df['volume_log'] - vol_mean) / (vol_std + epsilon)

    # Rolling Normalization for MACD, CCI
    for col in [macd_col, cci_col]:
        m_mean = grouped[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        m_std = grouped[col].transform(lambda x: x.rolling(window=window, min_periods=1).std(ddof=0).shift(1))
        df[f'{col}_norm'] = (df[col] - m_mean) / (m_std + epsilon)

    # Bollinger Bands %B
    df['bb_pctB'] = (df['close'] - df[bb_lb]) / (df[bb_ub] - df[bb_lb] + epsilon)

    # SMA Ratios
    df['sma_short_ratio'] = df['close'] / (df[sma_short] + epsilon) - 1
    df['sma_long_ratio'] = df['close'] / (df[sma_long] + epsilon) - 1

    # RSI, DX Normalization (scale to 0-1)
    df[f'{rsi_col}_norm'] = df[rsi_col] / 100.0
    df[f'{dx_col}_norm'] = df[dx_col] / 100.0

    # News Features Rolling Normalization
    for col in news_col:
        n_mean = grouped[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        n_std = grouped[col].transform(lambda x: x.rolling(window=window, min_periods=1).std(ddof=0).shift(1))
        df[f'{col}_norm'] = (df[col] - n_mean) / (n_std + epsilon)

    # --- Handle NaNs arising from rolling calculations/shifts ---
    # Identify all feature columns generated (including intermediate like volume_log if needed later)
    all_feature_cols = FEATURES # Use the list defined globally

    # Fill NaNs within each group - bfill first for initial NaNs, then ffill
    df[all_feature_cols] = grouped[all_feature_cols].transform(lambda x: x.fillna(method='bfill').fillna(method='ffill'))

    # Final check for any remaining NaNs in the features we will actually use
    if df[FEATURES].isnull().values.any():
        nan_counts = df[FEATURES].isnull().sum()
        logger.warning(f"NaN values remain in features after normalization and filling: \n{nan_counts[nan_counts > 0]}")
        logger.warning("Attempting global ffill/bfill as fallback...")
        # As a fallback, apply global fill, but this is less ideal
        df[FEATURES] = df[FEATURES].fillna(method='bfill').fillna(method='ffill')
        if df[FEATURES].isnull().values.any():
             logger.error("CRITICAL: NaN values still present in features. Check data source and normalization logic.")
             # Optionally drop rows with NaNs, but this might break time series continuity
             # df.dropna(subset=FEATURES, inplace=True)
             # Or raise an error:
             raise ValueError("Persistent NaN values found in features after all filling attempts.")

    logger.info("Feature normalization and NaN handling complete.")
    return df


def make_env(data, tickers, features, evaluate_by, lookback, initial_capital, max_episode_steps, reward_scaling, log_metrics, log_dir, seed=0, train=True):
    def _init():
        env = StockTradingEnv(
            data=data,
            tic_symbols=tickers,
            features=features,
            evaluate_by=evaluate_by,
            lookback=lookback, # Env needs lookback even if policy doesn't use LSTM state
            initial_capital=initial_capital,
            max_episode_step=max_episode_steps,
            reward_scaling=reward_scaling,
            log_metrics=log_metrics,
            log_dir=log_dir
        )
        # Important: Ensure Action Space is Box for TD3
        if not isinstance(env.action_space, gym.spaces.Box):
             logger.warning(f"Environment action space is {type(env.action_space)}, not Box. TD3 requires Box space.")
             # Consider adding wrappers or modifying env if needed

        if not train:
             monitor_path = os.path.join(TEST_SAVE_DIR, f"monitor_td3_test_{seed}.csv") # Changed prefix
             os.makedirs(os.path.dirname(monitor_path), exist_ok=True)
             env = Monitor(env,
                          filename=monitor_path,
                          allow_early_resets=True,
                          info_keywords=("date", "portfolio_value"))
        env.seed(seed)
        return env
    return _init
# --- End Shared Functions ---

if __name__ == "__main__":
    config = Config()
    data_path = config.cwd / "data"
    train_path = data_path / "new_train.csv"
    eval_path = data_path / "new_eval.csv"
    test_path = data_path / "new_test.csv"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Checking action space dimension for TD3 noise setup...")
    try:
        # Load minimal data just for env instantiation
        temp_env_data = load_data(train_path)
        # Need enough unique dates for lookback + 1 step for a single ticker
        unique_dates = temp_env_data['date'].unique()
        if len(unique_dates) < LOOKBACK + 1:
             raise ValueError(f"Train data has only {len(unique_dates)} unique dates, need at least {LOOKBACK + 1} for lookback.")
        first_ticker = temp_env_data['tic'].unique()[0]
        temp_env_data_slice = temp_env_data[
            (temp_env_data['tic'] == first_ticker) &
            (temp_env_data['date'].isin(unique_dates[:LOOKBACK+1]))
        ].copy()

        # Normalize the small slice
        temp_env_data_slice = normalize_features(temp_env_data_slice, window=LOOKBACK) # Use refined function

        # Drop any NaNs resulting from normalization if they exist at the very start
        temp_env_data_slice.dropna(subset=FEATURES, inplace=True)
        if len(temp_env_data_slice) < LOOKBACK + 1:
             raise ValueError("Not enough data points remain after normalization for env init.")

        temp_env = StockTradingEnv(
                    data=temp_env_data_slice.tail(LOOKBACK + 1), # Ensure exactly lookback + 1 rows
                    tic_symbols=[first_ticker], # Use one ticker
                    features=FEATURES,
                    evaluate_by=EVALUATE_BY,
                    lookback=LOOKBACK,
                    initial_capital=INITIAL_CAPITAL,
                    max_episode_step=MAX_EPISODE_STEPS,
                    reward_scaling=REWARD_SCALING,
                    log_metrics=False,
                    log_dir=None)

        if not isinstance(temp_env.action_space, gym.spaces.Box):
            raise TypeError(f"Environment action space is {type(temp_env.action_space)}, TD3 requires gym.spaces.Box.")

        action_dim = temp_env.action_space.shape[-1]
        logger.info(f"Detected action dimension: {action_dim}")
        del temp_env_data
        del temp_env_data_slice
        del temp_env

        action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=action_noise_sigma * np.ones(action_dim))
        TD3_PARAMS["action_noise"] = action_noise
        logger.info(f"TD3 action noise configured for {action_dim} dimensions.")

    except Exception as e:
        logger.error(f"Could not automatically determine action dimension or init temp env: {e}. Using placeholder value: {action_dim}. THIS MAY CAUSE ERRORS.", exc_info=True)
    if action_dim <= 0:
        logger.error("Action dimension is invalid. Exiting.")
        exit()


    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(REPORT_SAVE_DIR, exist_ok=True)
    os.makedirs(TEST_SAVE_DIR, exist_ok=True)

    try:
        train_data = load_data(train_path)
        eval_data = load_data(eval_path)
        test_data = load_data(test_path)

        logger.info("Normalizing training data...")
        train_data = normalize_features(train_data, window=LOOKBACK)
        logger.info("Normalizing evaluation data...")
        eval_data = normalize_features(eval_data, window=LOOKBACK)
        logger.info("Normalizing test data...")
        test_data = normalize_features(test_data, window=LOOKBACK)

        # Drop any remaining NaNs just before creating environments, although normalize_features should handle it
        train_data.dropna(subset=FEATURES, inplace=True)
        eval_data.dropna(subset=FEATURES, inplace=True)
        test_data.dropna(subset=FEATURES, inplace=True)
        logger.info("Removed any potential remaining NaN rows before env creation.")


        # Create vectorized environments (typically n_envs=1 for TD3)
        n_envs = 1
        logger.info(f"Creating {n_envs} training environment(s)...")
        env = DummyVecEnv([make_env(train_data, "all", FEATURES, EVALUATE_BY, LOOKBACK,
                                   INITIAL_CAPITAL, MAX_EPISODE_STEPS, REWARD_SCALING,
                                   log_metrics=False, log_dir=None, seed=i, train=True)
                                   for i in range(n_envs)])

        logger.info("Creating evaluation environment...")
        eval_env = DummyVecEnv([make_env(eval_data, "all", FEATURES, EVALUATE_BY, LOOKBACK,
                                       INITIAL_CAPITAL, MAX_EPISODE_STEPS, reward_scaling=1.0,
                                       log_metrics=LOG_METRICS_ENV,
                                       log_dir=REPORT_SAVE_DIR, seed=100, train=False)])

        logger.info("Creating test environment...")
        test_env = DummyVecEnv([make_env(test_data, "all", FEATURES, EVALUATE_BY, LOOKBACK,
                                       INITIAL_CAPITAL, MAX_EPISODE_STEPS, reward_scaling=1.0,
                                       log_metrics=LOG_METRICS_ENV,
                                       log_dir=TEST_SAVE_DIR, seed=200, train=False)])

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=max(CHECKPOINT_FREQ // n_envs, 1),
            save_path=MODEL_SAVE_DIR,
            name_prefix="td3_stock_trader", # Changed prefix
            save_replay_buffer=True,       # Save replay buffer for TD3
            save_vecnormalize=False
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(MODEL_SAVE_DIR, "best_model"),
            log_path=os.path.join(LOG_DIR, "eval"),
            eval_freq=max(EVAL_FREQ // n_envs, 1),
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True, # Evaluate deterministic policy (no action noise)
            render=False,
        )

        model = TD3(
            env=env,
            seed=42,
            device="auto", # Or "cuda", "cpu"
            **TD3_PARAMS
        )

        logger.info("Starting TD3 training...")
        logger.info(f"Hyperparameters: {TD3_PARAMS}")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            log_interval=4, # Log every 4 episodes
            progress_bar=True
        )
        logger.info("Training finished.")

        final_model_path = os.path.join(MODEL_SAVE_DIR, "td3_stock_trader_final")
        model.save(final_model_path)
        logger.info(f"Final TD3 model saved to {final_model_path}")

        logger.info("Running final test on the trained TD3 model...")
        try:
            best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model/best_model.zip")
            if os.path.exists(best_model_path):
                model = TD3.load(best_model_path, env=test_env)
                logger.info(f"Loaded best model from {best_model_path} for final test.")
            else:
                 logger.warning("Best model not found at %s, testing with the final model.", best_model_path)
                 model = TD3.load(final_model_path, env=test_env)
        except Exception as e:
             logger.error(f"Error loading model for final test: {e}. Testing with final model.", exc_info=True)
             model = TD3.load(final_model_path, env=test_env)


        obs = test_env.reset()
        dones = [False] * test_env.num_envs
        while not all(dones):
            # Use deterministic=True for evaluation (no action noise)
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = test_env.step(action)


        logger.info("Final evaluation complete. Check report files and monitor logs in test directory.")
        test_env.close() # Close the env and save monitor file


    except FileNotFoundError:
        logger.error("Critical error: Data file not found. Exiting.")
    except ValueError as e: # Catch specific errors like NaN issues
        logger.error(f"Data validation error: {e}", exc_info=True)
    except TypeError as e: # Catch action space errors
         logger.error(f"Environment configuration error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during TD3 training: {e}", exc_info=True)
