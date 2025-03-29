
from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym # Changed import from gym to gymnasium
import matplotlib
import numpy as np
import pandas as pd
from gymnasium.utils import seeding # Keep for now, though reset handles main seeding

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import quantstats as qs
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        """QuantStats module not found, environment can't plot results and calculate indicadors.
        This module is not installed with FinRL. Install by running one of the options:
        pip install quantstats --upgrade --no-cache-dir
        conda install -c ranaroussi quantstats
        """
    )


class PortfolioOptimizationEnv(gym.Env): # Inherit from gymnasium.Env
    """A portfolio allocation environment for OpenAI Gymnasium.

    This environment simulates the interactions between an agent and the financial market
    based on data provided by a dataframe. The dataframe contains the time series of
    features defined by the user (such as closing, high and low prices) and must have
    a time and a tic column with a list of datetimes and ticker symbols respectively.
    An example of dataframe is shown below::

            date        high            low             close           tic
        0   2020-12-23  0.157414        0.127420        0.136394        ADA-USD
        1   2020-12-23  34.381519       30.074295       31.097898       BNB-USD
        2   2020-12-23  24024.490234    22802.646484    23241.345703    BTC-USD
        3   2020-12-23  0.004735        0.003640        0.003768        DOGE-USD
        4   2020-12-23  637.122803      560.364258      583.714600      ETH-USD
        ... ...         ...             ...             ...             ...

    Based on this dataframe, the environment will create an observation space that can
    be a Dict or a Box. The Box observation space is a three-dimensional array of shape
    (f, n, t), where f is the number of features, n is the number of stocks in the
    portfolio and t is the user-defined time window. If the environment is created with
    the parameter return_last_action set to True, the observation space is a Dict with
    the following keys::

        {
        "state": three-dimensional Box (f, n, t) representing the time series,
        "last_action": one-dimensional Box (n+1,) representing the portfolio weights
        }

    Note that the action space of this environment is an one-dimensional Box with size
    n + 1 because the portfolio weights must contains the weights related to all the
    stocks in the portfolio and to the remaining cash.

    Attributes:
        action_space: Action space.
        observation_space: Observation space.
        episode_length: Number of timesteps of an episode.
        portfolio_size: Number of stocks in the portfolio.
        np_random: The random number generator for the environment.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30} # Added render_modes

    def __init__(
        self,
        df: pd.DataFrame,
        initial_amount: float,
        order_df: bool = True,
        return_last_action: bool = False,
        normalize_df: str | callable | None = "by_previous_time",
        reward_scaling: float = 1,
        comission_fee_model: str | None = "trf",
        comission_fee_pct: float = 0,
        features: list[str] = ["close", "high", "low"],
        valuation_feature: str = "close",
        time_column: str = "date",
        time_format: str = "%Y-%m-%d",
        tic_column: str = "tic",
        tics_in_portfolio: list[str] | str = "all",
        time_window: int = 1,
        cwd: str | Path = "./",
        render_mode: str | None = None, # Added render_mode for Gymnasium
    ):
        """Initializes environment's instance.

        Args:
            df: Dataframe with market information over a period of time.
            initial_amount: Initial amount of cash available to be invested.
            order_df: If True input dataframe is ordered by time.
            return_last_action: If True, observations also return the last performed
                action. Note that, in that case, the observation space is a Dict.
            normalize_df: Defines the normalization method applied to input dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.
            reward_scaling: A scaling factor to multiply the reward function. This
                factor can help training.
            comission_fee_model: Model used to simulate comission fee. Possible values
                are "trf" (for transaction remainder factor model) and "wvm" (for weights
                vector modifier model). If None, commission fees are not considered.
            comission_fee_pct: Percentage to be used in comission fee. It must be a value
                between 0 and 1.
            features: List of features to be considered in the observation space. The
                items of the list must be names of columns of the input dataframe.
            valuation_feature: Feature to be considered in the portfolio value calculation.
            time_column: Name of the dataframe's column that contain the datetimes that
                index the dataframe.
            time_format: Formatting string of time column.
            tic_name: Name of the dataframe's column that contain ticker symbols.
            tics_in_portfolio: List of ticker symbols to be considered as part of the
                portfolio. If "all", all tickers of input data are considered.
            time_window: Size of time window.
            cwd: Local repository in which resulting graphs will be saved.
            render_mode: The rendering mode ('human' or None). Added for Gymnasium compatibility.
        """
        self.np_random = None # Initialize RNG, will be seeded in reset
        self._time_window = time_window
        self._time_index = time_window - 1 # Start index adjusted for lookback
        self._time_column = time_column
        self._time_format = time_format
        self._tic_column = tic_column
        self._df = df.copy() # Work on a copy to avoid modifying original df
        self._initial_amount = initial_amount
        self._return_last_action = return_last_action
        self._reward_scaling = reward_scaling
        self._comission_fee_pct = comission_fee_pct
        self._comission_fee_model = comission_fee_model
        self._features = features
        self._valuation_feature = valuation_feature
        self._cwd = Path(cwd)
        self.render_mode = render_mode # Store render_mode

        # results file setup
        self._results_file_dir = self._cwd / "results" / "rl" # Renamed for clarity
        self._results_file_dir.mkdir(parents=True, exist_ok=True)

        # initialize price variation placeholder
        self._df_price_variation = None

        # preprocess data
        self._preprocess_data(order_df, normalize_df, tics_in_portfolio)

        # dims and spaces
        self._tic_list = self._df[self._tic_column].unique()
        self.portfolio_size = len(self._tic_list)

        action_dim = 1 + self.portfolio_size # Cash + assets

        # sort datetimes and define episode length
        self._sorted_times = sorted(list(set(self._df[time_column]))) # Ensure list
        # Episode ends one step *before* the last possible index due to lookback/next state calculation
        self.episode_length = len(self._sorted_times) - time_window

        # define action space (portfolio weights including cash)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(action_dim,), dtype=np.float32)

        # define observation space
        state_shape = (len(self._features), self.portfolio_size, self._time_window)
        if self._return_last_action:
            # if last action must be returned, a dict observation space is defined
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32
                    ),
                    "last_action": gym.spaces.Box(
                        low=0, high=1, shape=(action_dim,), dtype=np.float32
                    ),
                }
            )
        else:
            # if information about last action is not relevant, a 3D observation space is defined
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32
            )

        # Internal state variables reset in self.reset()
        self._portfolio_value = self._initial_amount
        self._terminated = False # Use Gymnasium terminology
        self._truncated = False # Add Gymnasium truncated flag
        self._current_step = 0 # Track steps within an episode

        # Memory buffers reset in self._reset_memory() called by self.reset()
        self._state = None
        self._reward = 0.0
        self._info = {}
        self._asset_memory = None
        self._portfolio_return_memory = None
        self._portfolio_reward_memory = None
        self._actions_memory = None
        self._final_weights = None
        self._date_memory = None
        self._data = None # Holds data for the current window
        self._price_variation = None # Holds price variation for the current step

        # Seed the action space for reproducibility if needed (Box usually doesn't strictly require it)
        # self.action_space.seed(self.seed()) # Seeding now primarily handled in reset


    def step(self, action: np.ndarray):
        """Performs a simulation step.

        Args:
            action: An array containing the new target portfolio weights (including cash).

        Returns:
            A tuple containing:
            - observation (object): The agent's observation of the current environment.
            - reward (float): The amount of reward returned after previous action.
            - terminated (bool): Whether the episode has ended (e.g., reached the end of the dataset).
            - truncated (bool): Whether the episode was ended prematurely (e.g. time limit, not used here).
            - info (dict): Contains auxiliary diagnostic information.
        """
        self._terminated = self._time_index >= len(self._sorted_times) - 1
        self._truncated = False # This env doesn't truncate based on time limits internally

        if self._terminated:
            # --- Final step post-processing ---
            # Ensure metrics_df calculation happens *before* returning
            if self._date_memory and len(self._date_memory) > 1: # Need at least 2 points for calculations
                metrics_df = pd.DataFrame(
                    {
                        "date": self._date_memory[1:], # Skip initial state date
                        "returns": self._portfolio_return_memory[1:], # Skip initial 0 return
                        "rewards": self._portfolio_reward_memory[1:], # Skip initial 0 reward
                        "portfolio_values": self._asset_memory["final"][1:], # Skip initial amount
                    }
                )
                # Ensure 'date' is in datetime format before setting index
                metrics_df["date"] = pd.to_datetime(metrics_df["date"])
                metrics_df.set_index("date", inplace=True)

                # --- Plotting ---
                plt.figure() # Create a new figure
                plt.plot(metrics_df.index, metrics_df["portfolio_values"], "r")
                plt.title("Portfolio Value Over Time")
                plt.xlabel("Time")
                plt.ylabel("Portfolio Value")
                plt.savefig(self._results_file_dir / "portfolio_value.png")
                plt.close() # Close the figure

                plt.figure() # Create a new figure
                plt.plot(metrics_df.index, metrics_df["rewards"], "r") # Use rewards memory
                plt.title("Reward Over Time")
                plt.xlabel("Time")
                plt.ylabel("Reward")
                plt.savefig(self._results_file_dir / "reward.png")
                plt.close() # Close the figure

                plt.figure() # Create a new figure
                # Plot actions (excluding initial action if desired, or keep all)
                actions_toplot = np.array(self._actions_memory[1:]) # Skip initial action
                plt.plot(metrics_df.index, actions_toplot)
                action_labels = ["Cash"] + self._tic_list.tolist()
                plt.legend(action_labels)
                plt.title("Actions (Portfolio Weights)")
                plt.xlabel("Time")
                plt.ylabel("Weight")
                plt.savefig(self._results_file_dir / "actions.png")
                plt.close() # Close the figure

                # --- Print Stats ---
                print("=================================")
                initial_val = self._asset_memory["final"][0]
                final_val = self._portfolio_value
                print(f"Initial portfolio value: {initial_val:.2f}")
                print(f"Final portfolio value: {final_val:.2f}")
                if initial_val != 0:
                    print(f"Final accumulative portfolio value: {final_val / initial_val:.4f}")
                # Calculate QuantStats only if there are returns
                if not metrics_df.empty and not metrics_df["returns"].isnull().all():
                     try:
                        print(f"Maximum Drawdown: {qs.stats.max_drawdown(metrics_df['portfolio_values']):.4f}")
                        print(f"Sharpe ratio: {qs.stats.sharpe(metrics_df['returns']):.4f}")
                        # Generate QuantStats report plot
                        qs.plots.snapshot(
                            metrics_df["returns"],
                            title="Portfolio Performance",
                            show=False, # Don't display interactively
                            savefig=self._results_file_dir / "portfolio_summary.png",
                        )
                     except Exception as e:
                        print(f"Could not calculate QuantStats: {e}")
                else:
                    print("QuantStats calculation skipped (no valid returns data).")
                print("=================================")

            # Return the *last* valid state before termination
            # If _get_state was called at the terminating step, self._state holds it
            # Otherwise, need to ensure a valid state is returned (might need adjustment)
            # In Gymnasium, it's standard to return the observation *leading* to termination
            return self._state, self._reward, self._terminated, self._truncated, self._info

        else:
            # --- Regular step ---
            # Ensure action is numpy array
            actions = np.array(action, dtype=np.float32)

            # Normalize weights if they don't sum to 1 or contain negatives
            if not (math.isclose(np.sum(actions), 1, abs_tol=1e-6) and np.all(actions >= 0)):
                weights = self._softmax_normalization(actions)
            else:
                weights = actions

            # Save the *intended* weights for this time step
            self._actions_memory.append(weights)

            # Get previous step's final weights and portfolio value
            last_final_weights = self._final_weights[-1]
            last_portfolio_value = self._portfolio_value # Value *before* this step's transactions/market move

            # --- Apply commission fees (using value *before* market move) ---
            effective_portfolio_value = last_portfolio_value # Start with value before fees
            transaction_cost_mu = 1.0 # Factor representing reduction due to costs

            if self._comission_fee_model == "wvm":
                 # WVM applies cost *after* market move, which is complex.
                 # Simplified approach: Calculate cost based on intended trade *before* market move.
                 delta_weights = weights[1:] - last_final_weights[1:] # Change in asset weights
                 # Value of assets being traded (absolute change)
                 trade_value = np.sum(np.abs(delta_weights) * last_portfolio_value)
                 fees = trade_value * self._comission_fee_pct
                 # Check if fees exceed cash available *after* allocating to assets
                 cash_after_alloc = weights[0] * last_portfolio_value
                 if fees > cash_after_alloc :
                      # Fee exceeds available cash, revert to previous weights (no trade)
                      weights = last_final_weights
                      # Optionally add a penalty signal here
                      self._info["commission_revert"] = True
                      effective_portfolio_value = last_portfolio_value # No change if trade fails
                 else:
                     # Deduct fee from the portfolio value *before* market move
                     effective_portfolio_value = last_portfolio_value - fees
                     # Note: WVM in original code adjusted weights *after* finding fees could be paid from cash.
                     # This version deducts from total value *before* market move for simplicity.
                     # Weights remain the target weights here, value is reduced.

            elif self._comission_fee_model == "trf":
                # Transaction Remainder Factor Model
                # Calculate mu factor based on target weights and previous weights
                last_mu = 1.0
                mu = 1 - 2 * self._comission_fee_pct + self._comission_fee_pct**2
                epsilon = 1e-10 # Convergence tolerance
                max_iter = 100 # Prevent infinite loops
                iter_count = 0

                # Iteratively solve for mu
                while abs(mu - last_mu) > epsilon and iter_count < max_iter:
                    last_mu = mu
                    # Calculate sum part: max(0, w_{t-1} - mu * w_t) for assets only
                    sum_term = np.sum(np.maximum(last_final_weights[1:] - mu * weights[1:], 0))
                    # Calculate mu based on the formula
                    numerator = 1 - self._comission_fee_pct * weights[0] - \
                                (2 * self._comission_fee_pct - self._comission_fee_pct**2) * sum_term
                    denominator = (1 - self._comission_fee_pct * weights[0])
                    if abs(denominator) < epsilon: # Avoid division by zero
                        mu = last_mu # Keep previous mu if denominator is near zero
                        break
                    mu = numerator / denominator
                    iter_count += 1

                self._info["trf_mu"] = mu
                transaction_cost_mu = mu # Store the calculated factor
                effective_portfolio_value = transaction_cost_mu * last_portfolio_value # Apply cost factor

            # Store the portfolio value *before* the market move but *after* fees
            self._asset_memory["initial"].append(effective_portfolio_value)

            # --- Advance time and get new market data ---
            self._time_index += 1
            self._current_step += 1
            # Get state for the *next* time step (t+1) and price variation from t to t+1
            self._state, self._info = self._get_state_and_info_from_time_index(
                self._time_index
            ) # _info now contains price_variation

            # --- Apply market price variation ---
            # Calculate portfolio value *after* market move
            # price_variation includes cash (element 0 is 1)
            portfolio_after_market = effective_portfolio_value * (weights * self._info["price_variation"])

            # New portfolio value is the sum of asset values after market move
            self._portfolio_value = np.sum(portfolio_after_market)

            # Calculate new final weights based on the value *after* the market move
            # Handle potential division by zero if portfolio value becomes ~0
            if self._portfolio_value < epsilon:
                # Assign equal weights or revert to cash if value is negligible
                final_weights = np.array([1] + [0] * self.portfolio_size, dtype=np.float32)
                self._portfolio_value = 0.0 # Set value to zero
            else:
                final_weights = portfolio_after_market / self._portfolio_value

            # Ensure final weights sum to 1 (due to potential float precision issues)
            final_weights = final_weights / np.sum(final_weights)


            # Save final portfolio value and weights of this time step
            self._asset_memory["final"].append(self._portfolio_value)
            self._final_weights.append(final_weights)

            # --- Calculate reward ---
            # Use log return for reward, based on value *after* fees and market move
            previous_final_value = self._asset_memory["final"][-2] # Value from end of t-1
            current_final_value = self._asset_memory["final"][-1]  # Value from end of t

            if previous_final_value <= epsilon: # Avoid log(0) or division by zero
                 portfolio_reward = 0.0
                 portfolio_return = 0.0
            else:
                rate_of_return = current_final_value / previous_final_value
                portfolio_return = rate_of_return - 1
                # Use log return, handle rate_of_return <= 0 case
                portfolio_reward = np.log(rate_of_return) if rate_of_return > 0 else -np.inf

            # Scale reward
            self._reward = portfolio_reward * self._reward_scaling

            # Save memory
            self._date_memory.append(self._info["end_time"])
            self._portfolio_return_memory.append(portfolio_return)
            self._portfolio_reward_memory.append(portfolio_reward)


            # Update state representation if needed (e.g., if last action is part of state)
            self._state = self._standardize_state(self._state, weights) # Pass current action for dict state

            # --- Rendering ---
            if self.render_mode == "human":
                self.render()

            return self._state, self._reward, self._terminated, self._truncated, self._info


    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the environment to its initial state.

        Args:
            seed: The seed to use for the environment's random number generator.
            options: Additional options for resetting the environment (not used here).

        Returns:
            A tuple containing:
            - observation (object): The initial observation.
            - info (dict): Auxiliary information accompanying the initial observation.
        """
        # Seed the environment's random number generator
        super().reset(seed=seed) # Important for gymnasium compatibility
        if seed is not None:
             self.np_random = np.random.default_rng(seed=seed) # Use numpy's default_rng
        else:
             # Ensure np_random is initialized even if seed is None
             self.np_random = np.random.default_rng()

        # Reset time and step counters
        self._time_index = self._time_window - 1
        self._current_step = 0

        # Reset internal memory buffers
        self._reset_memory()

        # Get the initial state and info
        self._state, self._info = self._get_state_and_info_from_time_index(
            self._time_index
        )

        # Reset portfolio value and termination flags
        self._portfolio_value = self._initial_amount
        self._terminated = False
        self._truncated = False
        self._reward = 0.0 # Reset reward for the first step

        # Update state representation if needed (pass initial action)
        self._state = self._standardize_state(self._state, self._actions_memory[-1])

        # --- Rendering ---
        if self.render_mode == "human":
            self.render()

        return self._state, self._info

    def _get_state_and_info_from_time_index(self, time_index: int):
        """Gets state and information given a time index. It also calculates
        the price variation for the *next* step (from time_index to time_index + 1).

        Args:
            time_index: The current end index of the time window in _sorted_times.

        Returns:
            A tuple with the following form: (state_data, info).

            state_data: The state data (numpy array) for the current time window.
                        This is before standardization (adding last action if needed).
            info: A dictionary with info about the current step. Crucially, it now
                  contains 'price_variation' which is the change from the *end* of
                  this window (time_index) to the start of the *next* window
                  (time_index + 1).
        """
        # Ensure time_index is within valid bounds for accessing data window
        if time_index < self._time_window - 1 or time_index >= len(self._sorted_times):
             raise IndexError(f"time_index {time_index} is out of bounds.")

        end_time = self._sorted_times[time_index]
        start_time_index = time_index - (self._time_window - 1)
        start_time = self._sorted_times[start_time_index]

        # --- Get state data for the window [start_time, end_time] ---
        current_data_window = self._df[
            (self._df[self._time_column] >= start_time)
            & (self._df[self._time_column] <= end_time)
        ][[self._time_column, self._tic_column] + self._features].copy()

        state_data = [] # Use list for easier appending
        for tic in self._tic_list:
            tic_data = current_data_window[current_data_window[self._tic_column] == tic]
            # Ensure data matches time window length (handle potential missing data carefully if needed)
            if len(tic_data) != self._time_window:
                 # This case needs careful handling depending on desired behavior
                 # Option 1: Forward fill, backward fill, error out, or use zeros/mean
                 # Current implementation assumes data is complete for the window
                 print(f"Warning: Data for tic {tic} at time {end_time} has length {len(tic_data)}, expected {self._time_window}. Check data integrity.")
                 # Example: Pad with last known value (simple approach)
                 if len(tic_data) > 0:
                    padding = pd.concat([tic_data.iloc[[-1]]] * (self._time_window - len(tic_data)), ignore_index=True)
                    tic_data = pd.concat([tic_data, padding], ignore_index=True)
                 else: # No data at all, pad with zeros (or other strategy)
                    # Create a zero DataFrame matching structure
                    zero_data = pd.DataFrame(0, index=np.arange(self._time_window), columns=tic_data.columns)
                    # Need to fill tic and time if required elsewhere, though features are main state
                    tic_data = zero_data
                 # Ensure correct length after padding/filling
                 tic_data = tic_data.head(self._time_window)


            tic_features = tic_data[self._features].to_numpy().T # Shape (features, time_window)
            state_data.append(tic_features)

        # Stack along the 'tic' dimension (axis=1) -> (features, tics, time_window)
        state_data_np = np.stack(state_data, axis=1).astype(np.float32)


        # --- Get price variation for the *next* step (from end_time to next_time) ---
        price_variation_step = np.ones(1 + self.portfolio_size, dtype=np.float32) # Default to 1 (no change) including cash

        # Check if there is a next time step available
        if time_index + 1 < len(self._sorted_times):
            next_time = self._sorted_times[time_index + 1]
            try:
                # Get the variation from the precalculated variation dataframe
                price_variation_assets = self._df_price_variation[
                    self._df_price_variation[self._time_column] == next_time
                ][self._valuation_feature].to_numpy()

                # Ensure we got the correct number of assets
                if len(price_variation_assets) == self.portfolio_size:
                    price_variation_step[1:] = price_variation_assets.astype(np.float32)
                else:
                     print(f"Warning: Price variation data at {next_time} has length {len(price_variation_assets)}, expected {self.portfolio_size}. Using 1.0.")
                     # Keep default ones if mismatch occurs
            except KeyError:
                # Handle case where the next_time might not be in _df_price_variation (e.g., end of data)
                print(f"Warning: Price variation data not found for {next_time}. Using 1.0.")
                # Keep default ones
        # else: # If it's the very last time_index, there's no 'next step' variation
             # print("Last time index reached, no further price variation calculated.")


        info = {
            "tics": self._tic_list,
            "start_time": start_time,
            "start_time_index": start_time_index,
            "end_time": end_time,
            "end_time_index": time_index,
            "data_window": current_data_window, # The data used for the state
            "price_variation": price_variation_step, # Variation for the *next* transition
        }
        return state_data_np, info


    def render(self):
        """Renders the environment.

        Currently, 'human' mode prints basic information at each step.
        Could be extended to provide plots or more detailed output.
        """
        if self.render_mode == "human":
            if self._current_step > 0: # Avoid printing before first step is complete
                print("-" * 30)
                print(f"Step: {self._current_step}")
                print(f"Time: {self._info.get('end_time', 'N/A')}")
                print(f"Portfolio Value: {self._portfolio_value:.2f}")
                last_action = self._actions_memory[-1] if self._actions_memory else "N/A"
                print(f"Action (weights): {np.round(last_action, 3)}")
                final_weights = self._final_weights[-1] if self._final_weights else "N/A"
                print(f"Final Weights: {np.round(final_weights, 3)}")
                print(f"Reward: {self._reward:.4f}")
                print(f"Price Variation Used: {np.round(self._info.get('price_variation', 'N/A'), 4)}")
            else:
                print("Environment Reset.")
                print(f"Initial Portfolio Value: {self._initial_amount:.2f}")
        # If other render modes were supported (like 'rgb_array'), they would be handled here.


    def _softmax_normalization(self, actions: np.ndarray) -> np.ndarray:
        """Normalizes the action vector using softmax function.

        Ensures weights are positive and sum to 1.

        Args:
            actions: The raw action vector from the agent.

        Returns:
            Normalized action vector (portfolio weights).
        """
        # Subtract max for numerical stability
        exp_actions = np.exp(actions - np.max(actions))
        softmax_output = exp_actions / np.sum(exp_actions)
        return softmax_output.astype(np.float32)


    def enumerate_portfolio(self):
        """Prints the mapping from index to ticker symbol for the portfolio assets."""
        print("Portfolio Asset Mapping:")
        print("Index: 0, Tic: Cash")
        for index, tic in enumerate(self._tic_list):
            print(f"Index: {index + 1}, Tic: {tic}")


    def _preprocess_data(
        self,
        order: bool,
        normalize: str | callable | None,
        tics_in_portfolio: list[str] | str
        ):
        """Orders, filters, normalizes the environment's dataframe, and calculates price variations.

        Args:
            order: If true, the dataframe will be ordered by ticker list and datetime.
            normalize: Defines the normalization method applied to the main dataframe used for state.
            tics_in_portfolio: List of ticker symbols to use, or "all".
        """
        df_processed = self._df.copy()

        # --- Filtering ---
        if isinstance(tics_in_portfolio, list):
             df_processed = df_processed[df_processed[self._tic_column].isin(tics_in_portfolio)]
             self._tic_list = sorted(list(tics_in_portfolio)) # Ensure consistent order
             print(f"Filtered dataframe to include tics: {self._tic_list}")
        elif tics_in_portfolio == "all":
             self._tic_list = sorted(df_processed[self._tic_column].unique())
             print("Using all tics found in the dataframe.")
        else:
             raise ValueError("tics_in_portfolio must be a list of strings or 'all'")

        # Ensure valuation feature is in the feature list if specified differently
        if self._valuation_feature not in self._features:
             self._features.append(self._valuation_feature)
             print(f"Added '{self._valuation_feature}' to features list as it's needed for valuation.")
        # Ensure features are valid columns
        for feature in self._features:
            if feature not in df_processed.columns:
                raise ValueError(f"Feature '{feature}' not found in dataframe columns: {df_processed.columns}")


        # --- Ordering ---
        if order:
            df_processed = df_processed.sort_values(by=[self._tic_column, self._time_column])
            df_processed.reset_index(drop=True, inplace=True)
            print("Sorted dataframe by tic and time.")


        # --- Calculate Price Variation Dataframe ---
        # This variation is based on the original (unnormalized) valuation feature
        # It calculates Var(t) = Price(t) / Price(t-1)
        # Important: Calculate variation *before* normalizing the main df
        self._df_price_variation = self._calculate_price_variation(df_processed)


        # --- Normalization of State Dataframe ---
        # Apply normalization *only* to the df used for constructing the state (self._df)
        if normalize:
             df_normalized = self._normalize_dataframe(df_processed, normalize) # Pass the potentially ordered/filtered df
             print(f"Applied normalization: {normalize}")
        else:
             df_normalized = df_processed.copy() # Use the processed df directly if no normalization
             print("No normalization applied to state features.")

        # Final assignment to self._df (used for state construction)
        self._df = df_normalized

        # --- Data Type Conversion ---
        # Convert time column to datetime objects
        try:
             self._df[self._time_column] = pd.to_datetime(self._df[self._time_column], format=self._time_format)
             self._df_price_variation[self._time_column] = pd.to_datetime(self._df_price_variation[self._time_column], format=self._time_format)
        except ValueError as e:
            raise ValueError(f"Could not parse time column '{self._time_column}' with format '{self._time_format}'. Error: {e}")


        # Convert feature columns to float32 for efficiency and compatibility
        self._df[self._features] = self._df[self._features].astype(np.float32)
        # Price variation df only needs valuation feature as float32, others can be dropped or kept
        self._df_price_variation[self._valuation_feature] = self._df_price_variation[self._valuation_feature].astype(np.float32)
        # Optionally drop other features from price variation df to save memory
        self._df_price_variation = self._df_price_variation[[self._time_column, self._tic_column, self._valuation_feature]]

        print("Data preprocessing complete.")


    def _reset_memory(self):
        """Resets the environment's internal memory buffers used for tracking episode progress."""
        # Get the start datetime based on the initial time_index
        start_datetime = self._sorted_times[self._time_index]

        # Initial portfolio value (before first step)
        self._asset_memory = {
            "initial": [], # Will store value *before* market move (after fees)
            "final": [self._initial_amount], # Stores value *after* market move
        }
        # Memory for returns and rewards (length matches number of steps)
        self._portfolio_return_memory = [0.0] # Initial return is 0
        self._portfolio_reward_memory = [0.0] # Initial reward is 0

        # Initial action: all money in cash
        initial_action = np.array([1.0] + [0.0] * self.portfolio_size, dtype=np.float32)
        self._actions_memory = [initial_action]

        # Initial final weights (matches initial action)
        self._final_weights = [initial_action]

        # Memory for dates (starts with the date of the initial state)
        self._date_memory = [start_datetime]


    def _standardize_state(self, state_data: np.ndarray, last_action: np.ndarray) -> np.ndarray | dict:
        """Formats the state observation based on the environment's configuration.

        Args:
            state_data: The raw numpy array representing the market features window.
                        Shape: (features, tics, time_window).
            last_action: The last action taken by the agent (portfolio weights).
                         Shape: (1 + num_tics,).

        Returns:
            The formatted observation, either a numpy array or a dictionary,
            matching the defined observation_space.
        """
        if self._return_last_action:
            # Return a dictionary containing state data and the last action
            return {
                "state": state_data.astype(np.float32),
                "last_action": last_action.astype(np.float32)
                }
        else:
            # Return only the state data array
            return state_data.astype(np.float32)


    def _normalize_dataframe(self, df_to_normalize: pd.DataFrame, normalize: str | callable) -> pd.DataFrame:
        """Normalizes features in the provided dataframe based on the specified method.

        Args:
            df_to_normalize: The dataframe to apply normalization to.
            normalize: The normalization method ("by_previous_time", "by_fist_time_window_value",
                       "by_COLUMN_NAME", or a custom function).

        Returns:
            A new dataframe with normalized features.
        """
        df_normalized = df_to_normalize.copy()

        if isinstance(normalize, str):
            if normalize == "by_fist_time_window_value":
                # Normalize by the value at the beginning of the *overall* time window (tricky)
                # A more common interpretation is normalizing by the value at the *start* of the *current* lookback window.
                # FinRL's original implementation seems ambiguous here. Let's assume normalize by T-k+1 value.
                # This requires careful grouping and shifting.
                 print(f"Normalizing {self._features} by value at start of {self._time_window}-step window...")
                 periods = self._time_window -1 if self._time_window > 1 else 0
                 if periods > 0:
                      for col in self._features:
                          # Group by tic, shift to get the value 'periods' steps ago
                          start_window_val = df_normalized.groupby(self._tic_column)[col].shift(periods)
                          # Avoid division by zero, fill NaN with 1 (no change) or ffill/bfill
                          df_normalized[col] = (df_normalized[col] / start_window_val.replace(0, np.nan)).fillna(1.0)
                 # else: No normalization needed if time_window is 1

            elif normalize == "by_previous_time":
                 print(f"Normalizing {self._features} by previous time step value (calculating rate of change)...")
                 df_normalized = self._calculate_temporal_variation(df_normalized, periods=1)

            elif normalize.startswith("by_"):
                 # Normalize selected features by another specified column
                 normalizer_column = normalize[3:]
                 if normalizer_column not in df_normalized.columns:
                      raise ValueError(f"Normalization column '{normalizer_column}' not found in dataframe.")
                 print(f"Normalizing {self._features} by column '{normalizer_column}'...")
                 for col in self._features:
                      # Avoid division by zero
                      df_normalized[col] = (df_normalized[col] / df_normalized[normalizer_column].replace(0, np.nan)).fillna(1.0) # Fill NaN ratios with 1
                 # Remove the normalizer column if it's not part of the state features? Optional.

            else:
                 print(f"Warning: Unknown string normalization method '{normalize}'. No normalization applied.")

        elif callable(normalize):
            print("Applying custom normalization function...")
            try:
                df_normalized = normalize(df_normalized)
            except Exception as e:
                raise RuntimeError(f"Custom normalization function failed: {e}")
        else:
             # Should not happen if type hints are followed, but good practice
             raise TypeError(f"Unsupported normalize type: {type(normalize)}")

        return df_normalized


    def _calculate_temporal_variation(self, df_input: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """Calculates the temporal variation rate ( V(t) / V(t-periods) ) for features.

        Args:
            df_input: The input dataframe (should be sorted by tic and time).
            periods: The number of time steps to look back for calculating the ratio.

        Returns:
            A dataframe where feature columns contain the variation rate.
            The first 'periods' rows for each tic will have NaN/1.0 values.
        """
        df_var = df_input.copy()
        if periods <= 0:
            return df_var # No change if periods is non-positive

        for column in self._features:
            # Group by tic and shift to get the previous value
            prev_value = df_var.groupby(self._tic_column)[column].shift(periods)
            # Calculate ratio: current / previous
            # Replace 0s in denominator with NaN to avoid division by zero errors
            # Fill resulting NaNs/Infs with 1.0 (implies no change where prev=0 or at start)
            df_var[column] = (df_var[column] / prev_value.replace(0, np.nan)).fillna(1.0)

        return df_var

    def _calculate_price_variation(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Calculates the price variation V(t)/V(t-1) specifically for the valuation feature.

        This is stored separately and used in the step function to update portfolio value.
        It operates on the potentially unnormalized data.

        Args:
            df_input: Dataframe (sorted by tic, time) containing at least time, tic, and valuation feature.

        Returns:
            Dataframe with columns [time, tic, valuation_feature], where the
            valuation_feature column holds the V(t)/V(t-1) ratio.
        """
        df_pr_var = df_input[[self._time_column, self._tic_column, self._valuation_feature]].copy()
        col = self._valuation_feature
        # Group by tic, shift to get previous value
        prev_value = df_pr_var.groupby(self._tic_column)[col].shift(1)
        # Calculate ratio, handle division by zero and initial NaNs -> fill with 1.0
        df_pr_var[col] = (df_pr_var[col] / prev_value.replace(0, np.nan)).fillna(1.0)
        return df_pr_var


    # Remove the old _seed method, seeding is handled by reset now.
    # def _seed(self, seed=None): ...


    def get_sb_env(self, env_number: int = 1):
        """Creates a Stable Baselines3 compatible vectorized environment.

        Args:
            env_number: Number of parallel environments to create (default 1).

        Returns:
            A tuple containing:
            - The vectorized environment (DummyVecEnv).
            - The initial observation from the reset environment(s).
        """
        # Create a list of environment-making functions
        env_fns = [lambda: self] * env_number
        # Wrap them using DummyVecEnv
        e = DummyVecEnv(env_fns)
        # Reset the vectorized environment to get the initial observation
        obs = e.reset()
        return e, obs

    def close(self):
        """Closes the environment and cleans up resources (e.g., closing plot figures)."""
        plt.close("all") # Close all matplotlib figures
        print("PortfolioOptimizationEnv closed.")
