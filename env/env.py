import math
import numpy as np
import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt
from gym import spaces
from pathlib import Path

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv

class CustomPortfolioOptimizationEnv(PortfolioOptimizationEnv):
    EPSILON = 1e-8  # Define a small constant to avoid division by zero
    def __init__(
        self,
        df,
        initial_amount,
        order_df=True,
        return_last_action=False,
        normalize_df="by_previous_time",
        normalize_features=None,
        reward_scaling=1,
        comission_fee_model="trf",
        comission_fee_pct=0,
        features=["close", "high", "low"],
        valuation_feature="close",
        time_column="date",
        time_format="%Y-%m-%d",
        tic_column="tic",
        tics_in_portfolio="all",
        time_window=1,
        cwd="./",
        new_gym_api=False,
        log_metrics=False,
    ):
        self._time_window = time_window
        self._time_index = time_window - 1
        self._time_column = time_column
        self._time_format = time_format
        self._tic_column = tic_column
        self._df = df
        self._initial_amount = initial_amount
        self._return_last_action = return_last_action
        self._reward_scaling = reward_scaling
        self._comission_fee_pct = comission_fee_pct
        self._comission_fee_model = comission_fee_model
        self._features = features
        self._valuation_feature = valuation_feature
        self._cwd = Path(cwd)
        self._new_gym_api = new_gym_api
        self._log_metrics = log_metrics

        self._normalize_features = normalize_features if normalize_features else []

        # results file
        self._results_file = self._cwd / "results" / "rl"
        self._results_file.mkdir(parents=True, exist_ok=True)

        # initialize price variation
        self._df_price_variation = None

        # preprocess data
        self._preprocess_data(order_df, normalize_df, tics_in_portfolio)

        # dims and spaces
        self._tic_list = self._df[self._tic_column].unique()
        self.portfolio_size = (
            len(self._tic_list)
            if tics_in_portfolio == "all"
            else len(tics_in_portfolio)
        )
        action_space = 1 + self.portfolio_size

        # sort datetimes and define episode length
        self._sorted_times = sorted(set(self._df[time_column]))
        self.episode_length = len(self._sorted_times) - time_window + 1

        # define action space
        self.action_space = spaces.Box(low=0, high=1, shape=(action_space,))

        # define observation state
        if self._return_last_action:
            # if  last action must be returned, a dict observation
            # is defined
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(
                            len(self._features),
                            len(self._tic_list),
                            self._time_window,
                        ),
                    ),
                    "last_action": spaces.Box(low=0, high=1, shape=(action_space,)),
                }
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self._features), len(self._tic_list), self._time_window),
            )

        self._reset_memory()

        self._portfolio_value = self._initial_amount
        self._terminal = False

    def step(self, actions):
        self._terminal = self._time_index >= len(self._sorted_times) - 1

        if self._terminal:
            metrics_df = pd.DataFrame(
                {
                    "date": self._date_memory,
                    "returns": self._portfolio_return_memory,
                    "rewards": self._portfolio_reward_memory,
                    "portfolio_values": self._asset_memory["final"],
                }
            )
            metrics_df.set_index("date", inplace=True)

            plt.plot(metrics_df["portfolio_values"], "r")
            plt.title("Portfolio Value Over Time")
            plt.xlabel("Time")
            plt.ylabel("Portfolio value")
            plt.savefig(self._results_file / "portfolio_value.png")
            plt.close()

            plt.plot(self._portfolio_reward_memory, "r")
            plt.title("Reward Over Time")
            plt.xlabel("Time")
            plt.ylabel("Reward")
            plt.savefig(self._results_file / "reward.png")
            plt.close()

            plt.plot(self._actions_memory)
            plt.title("Actions performed")
            plt.xlabel("Time")
            plt.ylabel("Weight")
            plt.savefig(self._results_file / "actions.png")
            plt.close()

            print("=================================")
            print("Initial portfolio value:{}".format(self._asset_memory["final"][0]))
            print(f"Final portfolio value: {self._portfolio_value}")
            print(
                "Final accumulative portfolio value: {}".format(
                    self._portfolio_value / self._asset_memory["final"][0]
                )
            )
            print(
                "Maximum DrawDown: {}".format(
                    qs.stats.max_drawdown(metrics_df["portfolio_values"])
                )
            )
            print("Sharpe ratio: {}".format(qs.stats.sharpe(metrics_df["returns"])))
            print("=================================")

            qs.plots.snapshot(
                metrics_df["returns"],
                show=False,
                savefig=self._results_file / "portfolio_summary.png",
            )
            final_portfolio_value = self._portfolio_value
            sharpe = qs.stats.sharpe(metrics_df["returns"])
            max_dd = qs.stats.max_drawdown(metrics_df["portfolio_values"])
            initial_portfolio_value = self._asset_memory["final"][0]
            acc_value = final_portfolio_value / initial_portfolio_value
            self._info["initial_portfolio_value"] = initial_portfolio_value
            self._info["final_portfolio_value"] = final_portfolio_value
            self._info["accumulative_return"] = acc_value - 1
            self._info["sharpe_ratio"] = sharpe
            self._info["max_drawdown"] = max_dd

            if self._log_metrics:
                history_df = metrics_df[["date", "portfolio_values"]].copy()
                history_df.to_csv(self._results_file / "history.csv")
                final_metrics = {
                    "Metric": ["Initial Portfolio Value",
                               "Final Portfolio Value",
                               "Accumulative Portfolio Value (Factor)",
                               "Max Drawdown",
                               "Sharpe Ratio (Annualized)"],
                    "Value": [initial_portfolio_value,
                              final_portfolio_value,
                              acc_value,
                              max_dd,
                              sharpe]
                }
                final_metrics_df = pd.DataFrame(final_metrics)
                final_metrics_df.to_csv(self._results_file / "metrics.csv", index=False)
            if self._new_gym_api:
                return self._state, self._reward, self._terminal, False, self._info
            return self._state, self._reward, self._terminal, self._info

        else:
            # transform action to numpy array (if it's a list)
            actions = np.array(actions, dtype=np.float32)

            # if necessary, normalize weights
            if math.isclose(np.sum(actions), 1, abs_tol=1e-6) and np.min(actions) >= 0:
                weights = actions
            else:
                weights = self._softmax_normalization(actions)

            # save initial portfolio weights for this time step
            self._actions_memory.append(weights)

            # get last step final weights and portfolio_value
            last_weights = self._final_weights[-1]

            # load next state
            self._time_index += 1
            self._state, self._info = self._get_state_and_info_from_time_index(
                self._time_index
            )

            # if using weights vector modifier, we need to modify weights vector
            if self._comission_fee_model == "wvm":
                delta_weights = weights - last_weights
                delta_assets = delta_weights[1:]  # disconsider
                # calculate fees considering weights modification
                fees = np.sum(np.abs(delta_assets * self._portfolio_value))
                if fees > weights[0] * self._portfolio_value:
                    weights = last_weights
                    # maybe add negative reward
                else:
                    portfolio = weights * self._portfolio_value
                    portfolio[0] -= fees
                    self._portfolio_value = np.sum(portfolio)  # new portfolio value
                    weights = portfolio / self._portfolio_value  # new weights
            elif self._comission_fee_model == "trf":
                last_mu = 1
                mu = 1 - 2 * self._comission_fee_pct + self._comission_fee_pct**2
                while abs(mu - last_mu) > 1e-10:
                    last_mu = mu
                    mu = (
                        1
                        - self._comission_fee_pct * weights[0]
                        - (2 * self._comission_fee_pct - self._comission_fee_pct**2)
                        * np.sum(np.maximum(last_weights[1:] - mu * weights[1:], 0))
                    ) / (1 - self._comission_fee_pct * weights[0])
                self._info["trf_mu"] = mu
                self._portfolio_value = mu * self._portfolio_value

            # save initial portfolio value of this time step
            self._asset_memory["initial"].append(self._portfolio_value)

            # time passes and time variation changes the portfolio distribution
            portfolio = self._portfolio_value * (weights * self._price_variation)

            # calculate new portfolio value and weights
            self._portfolio_value = np.sum(portfolio)
            weights = portfolio / self._portfolio_value

            # save final portfolio value and weights of this time step
            self._asset_memory["final"].append(self._portfolio_value)
            self._final_weights.append(weights)

            # save date memory
            self._date_memory.append(self._info["end_time"])

            # define portfolio return
            rate_of_return = (
                self._asset_memory["final"][-1] / self._asset_memory["final"][-2]
            )
            portfolio_return = rate_of_return - 1
            portfolio_reward = np.log(rate_of_return)

            # save portfolio return memory
            self._portfolio_return_memory.append(portfolio_return)
            self._portfolio_reward_memory.append(portfolio_reward)

            # Define portfolio return
            self._reward = portfolio_reward
            self._reward = self._reward * self._reward_scaling

        if self._new_gym_api:
            return self._state, self._reward, self._terminal, False, self._info
        return self._state, self._reward, self._terminal, self._info

    def _preprocess_data(self, order, normalize, tics_in_portfolio):
        """Orders and normalizes the environment's dataframe.

        Args:
            order: If true, the dataframe will be ordered by ticker list
                and datetime.
            normalize: Defines the normalization method applied to the dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.
            tics_in_portfolio: List of ticker symbols to be considered as part of the
                portfolio. If "all", all tickers of input data are considered.
        """
        # order time dataframe by tic and time
        if order:
            self._df = self._df.sort_values(by=[self._tic_column, self._time_column])
        # defining price variation after ordering dataframe
        self._df_price_variation = self._temporal_variation_df(self._df, [self._valuation_feature])
        # select only stocks in portfolio
        if tics_in_portfolio != "all":
            self._df_price_variation = self._df_price_variation[
                self._df_price_variation[self._tic_column].isin(tics_in_portfolio)
            ]
        # apply normalization
        if normalize:
            self._normalize_dataframe(normalize)
        print(self._df.columns)
        # transform str to datetime
        self._df[self._time_column] = pd.to_datetime(self._df[self._time_column])
        self._df_price_variation[self._time_column] = pd.to_datetime(
            self._df_price_variation[self._time_column]
        )
        # transform numeric variables to float32 (compatibility with pytorch)
        self._df[self._features] = self._df[self._features].astype("float32")
        self._df_price_variation[self._valuation_feature] = self._df_price_variation[
            self._valuation_feature
        ].astype("float32")

    def _temporal_variation_df(self, df, features, periods=1):
            """Calculates the temporal variation dataframe. For each feature, this
            dataframe contains the rate of the current feature's value and the last
            feature's value given a period. It's used to normalize the dataframe.

            Args:
                periods: Periods (in time indexes) to calculate temporal variation.

            Returns:
                Temporal variation dataframe.
            """
            df_temporal_variation = df[[self._time_column, self._tic_column] + features].copy()

            df_temporal_variation = df_temporal_variation.sort_values(by=[self._tic_column, self._time_column])

            for column in features:
                prev_column = f"prev_{column}"
                df_temporal_variation[prev_column] = df_temporal_variation.groupby(
                    self._tic_column
                )[column].shift(periods=periods)

                denominator = df_temporal_variation[prev_column] + self.EPSILON

                df_temporal_variation[column] = np.divide(
                    df_temporal_variation[column],
                    denominator,
                    out=np.full_like(df_temporal_variation[column], 1.0),
                    where=denominator.notna() & (np.abs(denominator) > self.EPSILON / 10)
                )

                mask_initial_nan = df_temporal_variation[prev_column].isna()
                df_temporal_variation.loc[mask_initial_nan, column] = 1.0

                df_temporal_variation = df_temporal_variation.drop(columns=[prev_column])

            return df_temporal_variation[[self._time_column, self._tic_column] + features]

    def _normalize_dataframe(self, normalize):
        """ "Normalizes the environment's dataframe.

        Args:
            normalize: Defines the normalization method applied to the dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.

        Note:
            If a custom function is used in the normalization, it must have an
            argument representing the environment's dataframe.
        """
        if type(normalize) == str:
            if normalize == "by_fist_time_window_value":
                print(
                    "Normalizing {} by first time window value...".format(
                        self._normalize_features
                    )
                )
                normalized_features = self._temporal_variation_df(self._df, self._normalize_features, self._time_window - 1)
                self._df.update(normalized_features[self._normalize_features])
            elif normalize == "by_previous_time":
                print(f"Normalizing {self._normalize_features} by previous time...")
                normalized_features = self._temporal_variation_df(self._df, self._normalize_features)
                self._df.update(normalized_features[self._normalize_features])
            elif normalize.startswith("by_"):
                normalizer_column = normalize[3:]
                print(f"Normalizing {self._features} by {normalizer_column}")
                for column in self._features:
                    self._df[column] = self._df[column] / self._df[normalizer_column]
        elif callable(normalize):
            print("Applying custom normalization function...")
            self._df = normalize(self._df)
        else:
            print("No normalization was performed.")
