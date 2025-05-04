import logging
import numpy as np
import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Union, List, Literal, Optional, Dict, Tuple

import gymnasium as gym

logger = logging.getLogger(__name__)
class StockTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    EPSILON = 1e-8

    def __init__(
            self,
            data: pd.DataFrame,
            tic_symbols: Union[List[str], Literal['all']],
            features: List[str],
            evaluate_by: str,
            lookback: int,
            initial_capital: float,
            max_episode_step: Optional[int],
            reward_scaling: float = 1.0,
            log_metrics: bool = False,
            log_dir: Optional[str] = None,
            is_test: bool = False
        ):
        assert set(features + [evaluate_by, 'tic', 'date']).issubset(data.columns), "Expected {features} columns. Found {data.columns} columns."

        data = data.copy()

        data_tic_list = list(data['tic'].unique().tolist())
        self.tic_list = data_tic_list if tic_symbols == 'all' else tic_symbols

        missing_tics = [tic for tic in self.tic_list if tic not in data_tic_list]
        assert not missing_tics, f"Missing tics: {missing_tics}"

        self.evaluate_by = evaluate_by
        self.features = features
        self.look_back = lookback
        self.initial_capital = initial_capital
        self.max_episode_step = max_episode_step
        self.reward_scaling = reward_scaling
        self.log_metrics = log_metrics
        if self.log_metrics and log_dir:
            self.log_dir = Path(log_dir)
        self.is_test = is_test

        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1 + len(self.tic_list),), dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.look_back,
                len(self.tic_list),
                len(self.features),
            )
        )

        self._unique_dates = np.sort(data['date'].unique())
        self._date_to_step = {date: i for i, date in enumerate(self._unique_dates)}
        self._step_to_date = {i: date for i, date in enumerate(self._unique_dates)}

        self._start_tick = self.look_back 
        self._end_tick = len(self._unique_dates) - 1

        data = data.set_index('date')
        self._data = {}
        self.feature_arrays = {}
        for feature in self.features:
            try:
                pivoted_df = data.pivot(columns='tic', values=feature)
                pivoted_df = pivoted_df.reindex(index=self._unique_dates, columns=self.tic_list)
                if pivoted_df.isnull().values.any():
                    logger.warning(f"NaN values remains in feature {feature}!") 
                self._data[feature] = pivoted_df.to_numpy()
            except Exception as e:
                raise RuntimeError(f"Error pivoting feature {feature}: {e}")
        try:
            price_df = data.pivot(columns='tic', values=evaluate_by)
            price_df = price_df.reindex(index=self._unique_dates, columns=self.tic_list)
            if price_df.isnull().values.any():
                logger.warning(f"NaN values remains in feature {evaluate_by}!")
            price_df[price_df <= self.EPSILON] = self.EPSILON
            self._price_array = price_df.to_numpy()
        except Exception as e:
            raise RuntimeError(f"Error pivoting feature {evaluate_by}: {e}")
                
        self._current_step: int = self._start_tick
        self._capital: float = 0.0
        self._asset_holdings: np.ndarray = np.zeros(len(self.tic_list), dtype=np.float32) 
        self._portfolio_value: float = 0.0
        self._terminated: bool = False
        self._truncated: bool = False
        self._info_history: List[Dict] = []
        self._episode_count: int = 0

    def _get_obs(self) -> np.ndarray:
        start_idx = self._current_step - self.look_back
        end_idx = self._current_step
        obs = np.stack([self._data[feature][start_idx:end_idx] for feature in self.features], axis=-1).astype(np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self._current_step,
            "date": self._step_to_date[self._current_step],
            "portfolio_value": self._portfolio_value,
            "capital": self._capital,
            "asset_holdings": self._asset_holdings.copy(),
        }

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._current_step = self._start_tick
        self._capital = self.initial_capital
        self._asset_holdings.fill(0.0)
        self._portfolio_value = self.initial_capital
        self._terminated = False
        self._info_history = []
        self._episode_count += 1

        logger.debug(f"Environment reset. Start step: {self._current_step}, Initial capital: {self._capital:.2f}")

        initial_obs = self._get_obs()
        initial_info = self._get_info()

        self._info_history.append(initial_info)

        return initial_obs, initial_info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        if self._terminated or self._truncated:
            logger.warning(f"Step called after episode was terminated or truncated. Resetting required.")
            obs = self._get_obs()
            info = self._get_info()
            return obs, 0.0, self._terminated, self._truncated, info
        
        current_portfolio_value = self._portfolio_value
        current_prices = self._price_array[self._current_step, :]

        action = np.clip(action, 0.0, 1.0)
        action_sum = np.sum(action)
        if action_sum > self.EPSILON:
            weights = action / action_sum
        else:
            weights = np.zeros_like(action)
            weights[0] = 1.0
            logger.debug("Action sum close to zero at step {self._current_step}. Default to all cash.")

        target_cash_weight = weights[0]
        target_asset_weights = weights[1:]
        
        target_asset_value = current_portfolio_value * target_asset_weights
        target_asset_holding = target_asset_value / current_prices

        shares_trade = target_asset_holding - self._asset_holdings
        trade_value = shares_trade * current_prices

        self._capital -= np.sum(trade_value)
        self._asset_holdings = target_asset_holding
        
        self._capital = max(self._capital, 0.0)

        self._current_step += 1
        next_prices = self._price_array[self._current_step, :]
        asset_value = np.sum(self._asset_holdings * next_prices)
        self._portfolio_value = self._capital + asset_value 
        
        reward = (np.log(self._portfolio_value / (current_portfolio_value + self.EPSILON))) * self.reward_scaling

        if self._current_step >= self._end_tick:
            self._terminated = True
            logger.info(f"Episode terminated: End of data reached at step {self._current_step}.")
            if self.log_metrics and self._info_history:
                self._generate_report()

        elif self._portfolio_value <= self.initial_capital * 0.1:
            self._terminated = True
            logger.info(f"Episode terminated: Portfolio value ({self._portfolio_value:.2f}) fell below threshold.")
            reward -= 100 
            if self.log_metrics and self._info_history:
                self._generate_report()


        if self.max_episode_step and (self._current_step - self._start_tick) >= self.max_episode_step:
            self._truncated = True
            logger.info(f"Episode truncated: Max steps ({self.max_episode_step}) reached.")

        observation = self._get_obs()
        info = self._get_info()
        self._info_history.append(info)
        
        return observation, reward, self._terminated, self._truncated, info

    def _generate_report(self):
        if not self._info_history:
            logger.warning("Cannot generate report: No history recorded.")
            return

        logger.info(f"Generating performance report for Episode {self._episode_count}...")

        history_df = pd.DataFrame(self._info_history)
        history_df['date'] = pd.to_datetime(history_df['date'])
        history_df.set_index('date', inplace=True)

        if history_df.empty or len(history_df) < 2:
             logger.warning("Cannot generate report: History is too short.")
             return

        try:
            returns = history_df['portfolio_value'].pct_change().fillna(0)
            returns.name = "Strategy" # Name the series for QuantStats

            report_path = self.log_dir / f"quantstats_report_ep_{self._episode_count}.html"
            qs.reports.html(returns, output=report_path, title=f"Stock Trading Performance - Episode {self._episode_count}")
            logger.info(f"QuantStats report saved to: {report_path}")

        except Exception as e:
            logger.error(f"Failed to generate QuantStats report: {e}", exc_info=True)

        try:
            holdings_plot_path = self.log_dir / f"holdings_plot_ep_{self._episode_count}.png"

            cash_values = history_df['capital']
            asset_values_df = pd.DataFrame(history_df['asset_holdings'].tolist(), index=history_df.index, columns=self.tic_list)

            plot_df = asset_values_df.copy()
            plot_df['Cash'] = cash_values

            plot_df = plot_df[['Cash'] + self.tic_list]

            fig, ax = plt.subplots(figsize=(12, 6))
            plot_df.plot.area(ax=ax, stacked=True, linewidth=0.5)

            ax.set_title(f'Portfolio Holdings Over Time - Episode {self._episode_count}')
            ax.set_ylabel('Value ($)')
            ax.set_xlabel('Date')
            ax.grid(True, linestyle='--', alpha=0.5)

            plt.savefig(holdings_plot_path)
            plt.close(fig) # Close the figure to free memory
            logger.info(f"Holdings plot saved to: {holdings_plot_path}")

        except Exception as e:
            logger.error(f"Failed to generate holdings plot: {e}", exc_info=True)

    def _output(self):
        if self.log_metrics and self._info_history:
            self._generate_report()
        if self.is_test:
            history_path = self.log_dir / "history.csv"
            history_df = pd.DataFrame(self._info_history)
            history_df.to_csv(history_path, index=False)
            logger.info(f"History saved to: {history_path}")
