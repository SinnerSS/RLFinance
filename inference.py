import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import List

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import Config # Assuming you have a config.py
from env import StockTradingEnv # Use the provided StockTradingEnv class

LOG_DIR = Path("./result")
MODEL_SAVE_DIR = LOG_DIR / "models/PPO"
BEST_MODEL_PATH = MODEL_SAVE_DIR / "best_model30040126" / "best_model.zip" # Example path
TEST_SAVE_DIR = LOG_DIR / "test"

# Features and Env Params (Must match training)
SELECTED_TICKERS = 'all' # Or your specific list ['AAPL', 'MSFT', ...]
TECHNICAL_FEATURES = [
    'open_log_ret', 'high_log_ret', 'low_log_ret', 'close_log_ret',
    'close', 'volume_log', # Assuming 'close' itself is needed if evaluate_by='close'
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
EVALUATE_BY = 'close' # The price column used for valuation
LOOKBACK = 1 # Example: Number of days history for observation (must match training)
INITIAL_CAPITAL = 100000.0
MAX_EPISODE_STEPS = None # Use None if the env should run till the end of data
REWARD_SCALING = 1 # Reward scaling doesn't affect deterministic evaluation much

def load_data(file_path) -> pd.DataFrame:
    """Loads data, converts date, sorts."""
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        # IMPORTANT: Sort before any group-by or shift operations
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
    for col in news_col:
        m_mean = df[col].rolling(window).mean().shift(1)
        m_std = df[col].rolling(window).std(ddof=0).shift(1)
        df[f'{col}_norm'] = (df[col] - m_mean) / (m_std + epsilon)
        df[f'{col}_norm'] = df[f'{col}_norm'].ffill().bfill()

    assert not df.isnull().values.any(), "Null values found after normalization"
    return df


def make_env(data, tickers, features, evaluate_by, lookback, initial_capital, max_episode_steps, reward_scaling, log_metrics, log_dir, is_test, seed=0):
    # No Monitor wrapper needed for basic inference unless step logs are desired
    def _init():
        # Use the provided StockTradingEnv class
        env = StockTradingEnv(
            data=data,
            tic_symbols=tickers,
            features=features,
            evaluate_by=evaluate_by,
            lookback=lookback,
            initial_capital=initial_capital,
            max_episode_step=max_episode_steps,
            reward_scaling=reward_scaling,
            log_metrics=log_metrics, # True to generate env's own report
            log_dir=log_dir,         # Directory for env's report
            is_test=is_test
        )
        # env.seed(seed) # Seeding deprecated in Gymnasium, use reset(seed=...)
        return env
    return _init

# --- Main Execution ---
if __name__ == "__main__":
    config = Config()
    data_path = config.cwd / "data"
    test_path = data_path / "new_test.csv" # Make sure this is your test dataset

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Create necessary directories
    TEST_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Load and preprocess test data
        logger.info(f"Loading test data from: {test_path}")
        test_data = load_data(test_path)

        logger.info("Normalizing test data features...")
        # Use the same NORMALIZATION_WINDOW as used during training
        test_data = normalize_features(test_data, window=LOOKBACK)

        # --- Include 'close' price in FEATURES if evaluate_by='close' ---
        # The environment needs the evaluation price available.
        # If 'close' is not already in FEATURES, add it.
        # However, the normalize_features function uses 'close' internally,
        # so it should exist in the dataframe. Check if it's in FEATURES list.
        if EVALUATE_BY not in FEATURES and EVALUATE_BY in test_data.columns:
             logger.warning(f"'{EVALUATE_BY}' not in FEATURES list, but required by env. Adding it temporarily for env init. Make sure your model wasn't trained expecting this feature unless intended.")
             # This is mainly for the env to find the column. The model won't use it if not trained on it.
             env_features = FEATURES + [EVALUATE_BY]
        elif EVALUATE_BY not in test_data.columns:
             raise ValueError(f"Evaluation column '{EVALUATE_BY}' not found in the prepared data.")
        else:
             env_features = FEATURES # Use the original FEATURES list

        # Verify all features exist after normalization
        missing_final_features = [f for f in env_features if f not in test_data.columns]
        if missing_final_features:
            raise ValueError(f"Features missing after normalization: {missing_final_features}. Check normalization function.")


        # 2. Create the test environment
        logger.info("Creating test environment...")
        test_env = DummyVecEnv([make_env(test_data, SELECTED_TICKERS, env_features, EVALUATE_BY, LOOKBACK,
                                       INITIAL_CAPITAL, MAX_EPISODE_STEPS, REWARD_SCALING,
                                       log_metrics=True, # Generate env report
                                       log_dir=TEST_SAVE_DIR, is_test=True)])

        # 3. Load the trained model
        if not BEST_MODEL_PATH.exists():
            logger.error(f"Best model not found at {BEST_MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {BEST_MODEL_PATH}")

        logger.info(f"Loading best model from: {BEST_MODEL_PATH}")
        logger.info("Starting inference on test data...")
        model = PPO.load(BEST_MODEL_PATH, env=test_env)
        obs = test_env.reset()

        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = test_env.step(action)
            done = any(dones)

    except FileNotFoundError as e:
        logger.error(f"Fatal Error: Required file not found. {e}")
    except KeyError as e:
         logger.error(f"Fatal Error: Missing expected column or key: {e}. Check feature lists, data loading, and env logic.", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during testing: {e}", exc_info=True) # Log traceback
