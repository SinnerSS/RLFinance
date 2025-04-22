import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LSTMExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 1024,
        num_lstm_layers: int = 1,
    ):
        super().__init__(observation_space, features_dim)

        if len(observation_space.shape) != 3:
             raise ValueError(f"Expected observation space shape of (lookback, num_tickers, num_features), got {observation_space.shape}")

        lookback, num_tickers, num_features = observation_space.shape
        self.lookback = lookback
        self.num_tickers = num_tickers
        self.num_features = num_features

        lstm_input_size = num_tickers * num_features
        self.lstm_hidden_size = features_dim # LSTM hidden size is the output feature dim

        print(f"LSTMExtractor Initialized:")
        print(f"  Obs Shape (Lookback, Tickers, Features): ({lookback}, {num_tickers}, {num_features})")
        print(f"  LSTM Input Size (Tickers * Features): {lstm_input_size}")
        print(f"  LSTM Hidden Size (Features Dim): {self.lstm_hidden_size}")
        print(f"  Num LSTM Layers: {num_lstm_layers}")


        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        lstm_input = observations.reshape(batch_size, self.lookback, -1)

        lstm_output, (hn, cn) = self.lstm(lstm_input)
        extracted_features = hn[-1] # Shape: (batch_size, features_dim)

        return extracted_features

class LSTMPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        features_dim: int = 1024,
        num_lstm_layers: int = 1,
        **kwargs,
    ):
        self.features_dim_lstm = features_dim
        self.num_lstm_layers = num_lstm_layers

        kwargs["features_extractor_class"] = LSTMExtractor
        kwargs["features_extractor_kwargs"] = dict(
            features_dim=self.features_dim_lstm,
            num_lstm_layers=self.num_lstm_layers,
        )

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )
