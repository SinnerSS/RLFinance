import gym
import numpy as np
import torch
import torch.nn as nn

class MLPCritic(nn.Module):
    """Simple MLP Critic Network"""
    def __init__(self, state_shape, hidden_dim=128, device="cpu"):
        super().__init__()
        # Calculate flat input size from state_shape (features, num_assets, time_window)
        if isinstance(state_shape, gym.spaces.Dict):
             # If observation space is Dict, adjust based on the actual 'state' key shape
             input_dim = np.prod(state_shape['state'].shape)
             # Potentially add dimension for last_action if needed
             # Example: input_dim += state_shape['last_action'].shape[0]
        elif isinstance(state_shape, tuple):
             # Example: (features, num_assets, time_window) -> features * num_assets * time_window
            input_dim = np.prod(state_shape)
        elif isinstance(state_shape, gym.spaces.Box): # Added check for Box
             input_dim = np.prod(state_shape.shape)
        else:
             raise ValueError(f"Unsupported state_shape type: {type(state_shape)}")

        self.device = device
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)

    def _preprocess_state(self, observation):
        """Flattens and prepares the observation for the MLP"""
        if isinstance(observation, dict):
            state_tensor = torch.from_numpy(observation['state']).float().to(self.device)
            # last_action_tensor = torch.from_numpy(observation['last_action']).float().to(self.device)
            # Flatten state and potentially concatenate last_action
            flat_features = state_tensor.reshape(-1)
            # input_tensor = torch.cat((flat_state, last_action_tensor), dim=-1) # If using last_action
            input_tensor = flat_features.unsqueeze(0) # If only using state tensor
        elif isinstance(observation, np.ndarray): # Added check for numpy array (Box)
            state_tensor = torch.from_numpy(observation).float().to(self.device)
            flat_features = state_tensor.reshape(-1) # Flatten
            input_tensor = flat_features.unsqueeze(0) # Add batch dimension
        elif isinstance(observation, torch.Tensor):
            state_tensor = observation.to(self.device)
            # Assume shape is (Batch, F, N, T) or similar - we need (Batch, Flattened_Features)
            # The first dimension is the batch size. Flatten all subsequent dimensions.
            if state_tensor.ndim > 1: # Check if it has more than one dimension (includes batch)
                input_tensor = state_tensor.reshape(state_tensor.size(0), -1) # Shape: (Batch, F*N*T)
            else:
                input_tensor = state_tensor.reshape(1, -1) # Shape: (1, F*N*T)
        else:
            raise TypeError(f"Unsupported observation type for critic preprocessing: {type(observation)}")

        return input_tensor

    def forward(self, observation):
        processed_obs = self._preprocess_state(observation)
        return self.critic(processed_obs)
