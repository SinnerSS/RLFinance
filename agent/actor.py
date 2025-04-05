import torch
import torch.nn as nn

class CustomEIIE(nn.Module):
    def __init__(
        self,
        num_assets,
        initial_features=3,
        k_size=3, 
        conv_mid_features=32,
        conv_final_features=64,
        temporal_kernel=5,
        time_window=50,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.num_assets = num_assets
        self.time_window = time_window
        self.initial_features = initial_features

        self.conv1 = nn.Conv2d(
            in_channels=initial_features,
            out_channels=conv_mid_features,
            kernel_size=(1, k_size),
        )
        self.relu1 = nn.ReLU()

        padding_l2 = (0, temporal_kernel // 2)
        self.conv2 = nn.Conv2d(
            in_channels=conv_mid_features,
            out_channels=conv_final_features,
            kernel_size=(1, temporal_kernel),
            padding=padding_l2
        )
        self.relu2 = nn.ReLU()

        with torch.no_grad():
             dummy_state = torch.zeros(1, initial_features, num_assets, time_window)
             dummy_x = self.relu1(self.conv1(dummy_state))
             dummy_x = self.relu2(self.conv2(dummy_x))
             self.output_time_dim = dummy_x.shape[3]

        self.final_convolution = nn.Conv2d(
            in_channels=conv_final_features + 1,
            out_channels=1,
            kernel_size=(1, 1)
        )

    def mu(self, state_tensor, last_action_tensor):
        batch_size = state_tensor.shape[0]
        expected_state_shape = (batch_size, self.initial_features, self.num_assets, self.time_window)
        expected_action_shape = (batch_size, self.num_assets + 1)
        if state_tensor.shape != expected_state_shape:
             if state_tensor.shape == (batch_size, self.num_assets, self.initial_features, self.time_window):
                 print(f"Warning: state_tensor shape {state_tensor.shape} suggests Assets and Features might be swapped. Permuting to expected {expected_state_shape}.")
                 state_tensor = state_tensor.permute(0, 2, 1, 3)
             else:
                raise ValueError(f"Unexpected state_tensor shape. Got {state_tensor.shape}, expected {expected_state_shape}")
        if last_action_tensor.shape != expected_action_shape:
            raise ValueError(f"Unexpected last_action_tensor shape. Got {last_action_tensor.shape}, expected {expected_action_shape}")

        x = self.relu1(self.conv1(state_tensor))
        x = self.relu2(self.conv2(x))
        # x shape: (Batch, conv_final_features, num_assets, self.output_time_dim)

        last_action_asset_weights = last_action_tensor[:, 1:] # Shape: (Batch, Assets)
        last_action_reshaped = last_action_asset_weights.view(
            batch_size, 1, self.num_assets, 1
        ).expand(
            -1, -1, -1, self.output_time_dim
        )
        x_concat = torch.cat((x, last_action_reshaped), dim=1)
        # x_concat shape: (Batch, conv_final_features + 1, num_assets, self.output_time_dim)

        final_scores = self.final_convolution(x_concat)
        # final_scores shape: (Batch, 1, num_assets, self.output_time_dim)

        final_scores_collapsed = final_scores[:, :, :, -1] # Shape: (Batch, 1, Assets)
        asset_logits = final_scores_collapsed.squeeze(1) # Shape: (Batch, Assets)

        cash_logit = torch.zeros(batch_size, 1, device=self.device)
        logits_with_cash = torch.cat((cash_logit, asset_logits), dim=1)
        # logits_with_cash shape: (Batch, Assets + 1)

        return logits_with_cash

    def forward(self, state_tensor, last_action_tensor):
        return self.mu(state_tensor, last_action_tensor)
