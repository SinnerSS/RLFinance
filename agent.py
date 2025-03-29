import os
import time
import random
import argparse
from distutils.util import strtobool

import gymnasium as gym 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard.writer import SummaryWriter

from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

from env import PortfolioOptimizationEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=True`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanrl",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    parser.add_argument("--data-path", type=str, required=True, help="path to the csv file containing stock data")
    parser.add_argument("--total-timesteps", type=int, default=2000000, # Adjust as needed
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4, 
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1, # Portfolio env usually doesn't parallelize well naively unless data is split
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048, # PPO buffer size per env
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0, # Often 0 or small for portfolio tasks
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")

    parser.add_argument("--train-start-date", type=str, default="2016-01-01", # Example start date
                        help="Start date for the training period (YYYY-MM-DD)")
    parser.add_argument("--train-end-date", type=str, default="2020-12-31", # Example end date
                        help="End date for the training period (YYYY-MM-DD)")
    parser.add_argument("--test-start-date", type=str, default="2021-01-01", # Example start date
                        help="Start date for the testing period (YYYY-MM-DD)")
    parser.add_argument("--test-end-date", type=str, default="2022-12-31", # Example end date
                        help="End date for the testing period (YYYY-MM-DD)")
    parser.add_argument("--initial-amount", type=float, default=1000000,
                        help="initial portfolio value")
    parser.add_argument("--time-window", type=int, default=10, 
                        help="size of the observation time window")
    parser.add_argument("--commission-fee-pct", type=float, default=0.001, 
                        help="commission fee percentage")
    parser.add_argument("--reward-scaling", type=float, default=1e-4, # Adjust based on reward magnitudes
                        help="scaling factor for the reward")
    parser.add_argument("--features", nargs="+", default=["close", "high", "low", "volume"], # Default, adjust if FE adds more
                        help="list of features to use from the dataframe")
    parser.add_argument("--valuation-feature", type=str, default="close",
                        help="feature used for portfolio valuation")
    parser.add_argument("--time-column", type=str, default="date",
                        help="name of the time column in the csv")
    parser.add_argument("--tic-column", type=str, default="tic",
                        help="name of the ticker column in the csv")
    parser.add_argument("--normalize-df", type=str, default="by_previous_time",
                        choices=["by_previous_time", "by_fist_time_window_value", "none"], # Add more options if needed
                        help="normalization method for the dataframe")
    parser.add_argument("--return-last-action", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether the environment observation includes the last action")

    # Feature Engineering arguments (optional, depends on FeatureEngineer class)
    parser.add_argument("--use-technical-indicator", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                         help="whether to use technical indicators")
    # Add more FeatureEngineer args if needed (e.g., tech_indicator_list, use_vix)

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def make_env(args, seed, idx, run_name, data_period=None, eval_mode=False):
    def thunk():
        print(f"Loading data from: {args.data_path}")
        raw_df = pd.read_csv(args.data_path)

        time_col = args.time_column
        raw_df[time_col] = pd.to_datetime(raw_df[time_col]) # Ensure datetime format

        if data_period:
            start_date_str, end_date_str = data_period
            print(f"Filtering data for period: {start_date_str} to {end_date_str}")
        else: # Default to training period if not specified
            start_date_str, end_date_str = args.train_start_date, args.train_end_date
            print(f"Filtering data for TRAINING period: {start_date_str} to {end_date_str}")

        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)

        # Filter the dataframe
        filtered_df = raw_df[(raw_df[time_col] >= start_date) & (raw_df[time_col] <= end_date)].copy()

        if filtered_df.empty:
             raise ValueError(f"No data found for the specified period: {start_date_str} to {end_date_str}. Check dates and data file.")
        # --- END ADDED ---
        # --- Feature Engineering ---
        # Assuming FeatureEngineer needs to be instantiated and used
        # If FeatureEngineer has arguments, pass them from args
        fe = FeatureEngineer(
            use_technical_indicator=args.use_technical_indicator,
            tech_indicator_list=INDICATORS,
            # use_vix=False, # Example, control via args if needed
            # use_turbulence=False, # Example, control via args if needed
            # user_defined_feature=False # Example, control via args if needed
        )
        print("Preprocessing data with FeatureEngineer...")
        processed_df = fe.preprocess_data(filtered_df)
        print("Data preprocessing complete.")

        # --- Determine Features ---
        # If FE added features, update the list
        # Example: Detect technical indicators added by default FE setup
        current_features = list(args.features)
        # This part needs customization based on how FeatureEngineer exactly modifies the df
        if args.use_technical_indicator:
            for indicator in INDICATORS:
                if indicator in processed_df.columns and indicator not in current_features:
                    current_features.append(indicator)
        print(f"Using features: {current_features}")

        env_log_suffix = "eval" if eval_mode else f"train_{idx}"
        env_kwargs = {
            "df": processed_df,
            "initial_amount": args.initial_amount,
            "time_window": args.time_window,
            "comission_fee_pct": args.commission_fee_pct,
            "reward_scaling": args.reward_scaling,
            "features": current_features,
            "valuation_feature": args.valuation_feature,
            "time_column": args.time_column,
            "tic_column": args.tic_column,
            "normalize_df": args.normalize_df if args.normalize_df != "none" else None,
            "return_last_action": args.return_last_action,
            "order_df": False, # FE likely handles sorting
            "comission_fee_model": "trf", # or 'wvm' or None
            "cwd": f"runs/{run_name}/env_logs_{env_log_suffix}", 
        }
        env = PortfolioOptimizationEnv(**env_kwargs)

        # --- Wrappers ---
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # Add other wrappers if needed, e.g., for video recording
        # if args.capture_video and idx == 0:
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        env.action_space.seed(seed)
        # Note: PortfolioOptimizationEnv handles its own observation seeding internally if needed
        # env.observation_space.seed(seed) # Not standard for Box/Dict spaces
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        # --- Determine Observation Shape ---
        # Handle Box or Dict observation space
        if isinstance(envs.single_observation_space, gym.spaces.Dict):
            # Example: Flatten the 'state' part of the Dict space
            # Adjust this logic if the Dict structure is different or needs specific handling
            state_shape = envs.single_observation_space["state"].shape
            # TODO: Handle 'last_action' if present and needed by the network
            # obs_shape = np.prod(state_shape) + envs.single_observation_space["last_action"].shape[0]
            obs_shape = np.prod(state_shape) # Flatten only the state for simplicity here
            print(f"Using Dict observation space. Flattened state shape: {state_shape} -> {obs_shape}")
        elif isinstance(envs.single_observation_space, gym.spaces.Box):
            obs_shape = np.prod(envs.single_observation_space.shape)
            print(f"Using Box observation space. Shape: {envs.single_observation_space.shape} -> Flattened: {obs_shape}")
        else:
            raise ValueError(f"Unsupported observation space type: {type(envs.single_observation_space)}")

        # --- Determine Action Shape ---
        action_shape = np.prod(envs.single_action_space.shape)
        print(f"Action space shape: {envs.single_action_space.shape} -> Flattened: {action_shape}")

        # --- Define Networks ---
        # Simple MLP Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Simple MLP Actor (outputs mean of a Normal distribution)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_shape), std=0.01), # Smaller std for action output layer
        )

        # Learnable standard deviation for continuous actions
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))

    def _preprocess_obs(self, obs):
        """Handles Dict or Box observations and flattens them."""
        if isinstance(obs, dict):
            # Assuming 'state' is the primary observation tensor
            # TODO: Incorporate 'last_action' if args.return_last_action is True
            state_obs = obs["state"]
            # Flatten the state part: B x F x N x T -> B x (F*N*T)
            return state_obs.reshape(state_obs.shape[0], -1)
        else:
            # Flatten the Box observation: B x F x N x T -> B x (F*N*T)
            return obs.reshape(obs.shape[0], -1)


    def get_value(self, x):
        x_flat = self._preprocess_obs(x)
        return self.critic(x_flat)

    def get_action_and_value(self, x, action=None):
        x_flat = self._preprocess_obs(x)
        action_mean = self.actor_mean(x_flat)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        # Calculate log probability and entropy
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        # Get value from critic
        value = self.critic(x_flat)
        # Return: action, log_prob, entropy, value
        # Note: The environment handles softmax normalization if needed.
        # The agent outputs raw continuous values.
        return action, log_prob, entropy, value


if __name__ == "__main__":
    args = parse_args()
    run_name = f"PortfolioOpt_{args.seed}_{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # --- Environment Setup ---
    print("\n" + "="*20 + " STARTING TRAINING PHASE " + "="*20)
    print(f"Training Period: {args.train_start_date} to {args.train_end_date}")
    train_envs = gym.vector.SyncVectorEnv(
        [make_env(args, args.seed + i, i, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(train_envs.single_action_space, gym.spaces.Box), "This PPO script only supports Box action space (continuous actions)"

    print("Observation Space:", train_envs.single_observation_space)
    print("Action Space:", train_envs.single_action_space)


    # --- Agent Setup ---
    agent = Agent(train_envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)


    # --- Storage Setup (for PPO Rollouts) ---
    obs_space = train_envs.single_observation_space
    action_space = train_envs.single_action_space

    # Adjust storage based on observation space type
    if isinstance(obs_space, gym.spaces.Dict):
        # Store each component of the Dict space separately if needed, or handle appropriately
        # Simple approach: store the raw dict observation
        obs_storage = {key: torch.zeros((args.num_steps, args.num_envs) + space.shape).to(device)
                       for key, space in obs_space.spaces.items()}
        next_obs_storage = {key: torch.zeros((args.num_steps, args.num_envs) + space.shape).to(device)
                           for key, space in obs_space.spaces.items()}
    else: # Box space
        obs_storage = torch.zeros((args.num_steps, args.num_envs) + obs_space.shape).to(device)
        next_obs_storage = torch.zeros((args.num_steps, args.num_envs) + obs_space.shape).to(device)

    actions = torch.zeros((args.num_steps, args.num_envs) + action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)


    # --- Training Loop ---
    global_step = 0
    start_time = time.time()
    next_obs, _ = train_envs.reset(seed=args.seed) # Use new API reset

    # Convert initial observation to tensor and move to device
    if isinstance(obs_space, gym.spaces.Dict):
         next_obs = {k: torch.Tensor(v).to(device) for k, v in next_obs.items()}
    else:
         next_obs = torch.Tensor(next_obs).to(device)

    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    print(f"Starting training for {num_updates} updates...")

    for update in range(1, num_updates + 1):
        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # --- Rollout Phase ---
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs

            # Store current observation and done flag
            if isinstance(obs_space, gym.spaces.Dict):
                for key in obs_space.spaces.keys():
                    obs_storage[key][step] = next_obs[key]
            else:
                obs_storage[step] = next_obs
            dones[step] = next_done

            # Get action and value from agent
            with torch.no_grad():
                # Pass the correct observation structure (dict or tensor)
                current_obs_for_agent = {k: next_obs[k] for k in obs_space.spaces.keys()} if isinstance(obs_space, gym.spaces.Dict) else next_obs
                action, logprob, _, value = agent.get_action_and_value(current_obs_for_agent)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute action in environment
            # Ensure action is in numpy cpu format for env.step
            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = train_envs.step(action_np) # New API returns 5 values
            done = np.logical_or(terminated, truncated) # Combine termination conditions

            rewards[step] = torch.tensor(reward).to(device).view(-1)

            # Convert next observation and done flag to tensor and move to device
            if isinstance(obs_space, gym.spaces.Dict):
                next_obs = {k: torch.Tensor(v).to(device) for k, v in next_obs.items()}
            else:
                 next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)


            # Log episode statistics if available (from RecordEpisodeStatistics wrapper)
            if "final_info" in info:
                 for item in info["final_info"]:
                     if item and "episode" in item: # Check if item is not None and contains 'episode'
                        print(f"global_step={global_step}, episodic_return={item['episode']['r']:.2f}, episodic_length={item['episode']['l']}")
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        # Log final portfolio value if the env provides it in info
                        if "final_portfolio_value" in item:
                             writer.add_scalar("charts/final_portfolio_value", item["final_portfolio_value"], global_step)
                        break # Process only the first valid episode info if multiple envs finish


        # --- Advantage Calculation (GAE) ---
        with torch.no_grad():
             # Pass the correct observation structure (dict or tensor)
            next_obs_for_agent = {k: next_obs[k] for k in obs_space.spaces.keys()} if isinstance(obs_space, gym.spaces.Dict) else next_obs
            next_value = agent.get_value(next_obs_for_agent).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # --- Learning Phase ---
        # Flatten batch for training
        if isinstance(obs_space, gym.spaces.Dict):
             # Flatten each component separately if needed, or handle the agent's preprocessing
             # b_obs becomes a dict of flattened tensors
             b_obs = {k: obs_storage[k].reshape((-1,) + obs_space[k].shape) for k in obs_space.keys()}
             # Agent's _preprocess_obs should handle flattening this dict internally
        else:
             b_obs = obs_storage.reshape((-1,) + obs_space.shape)

        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Get data for minibatch, handling dict obs if necessary
                if isinstance(obs_space, gym.spaces.Dict):
                    mb_obs = {k: b_obs[k][mb_inds] for k in obs_space.keys()}
                else:
                    mb_obs = b_obs[mb_inds]

                # Pass correct obs structure to agent
                newaction, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # --- Logging ---
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        print(f"SPS: {int(global_step / (time.time() - start_time))}")

    # --- Cleanup ---
    train_envs.close()
    print("Training finished.")

    # --- ======== TESTING PHASE ======== ---
    print("\n" + "="*20 + " STARTING TESTING PHASE " + "="*20)
    print(f"Testing Period: {args.test_start_date} to {args.test_end_date}")

    # Create a single test environment
    # Use a different seed for test env if desired, e.g., args.seed + args.num_envs
    test_env = make_env(args, args.seed + args.num_envs, 0, run_name,
                       data_period=(args.test_start_date, args.test_end_date),
                       eval_mode=True)() # Call the thunk to get the env instance

    agent.eval()  # Set agent to evaluation mode (e.g., affects dropout, batch norm)

    # Reset test environment
    obs, info = test_env.reset(seed=args.seed + args.num_envs) # Use new API reset
    terminated = False
    truncated = False
    total_test_reward = 0

    while not (terminated or truncated):
        with torch.no_grad():
            # Convert obs to tensor, handle dict if necessary
            if isinstance(test_env.observation_space, gym.spaces.Dict):
                obs_tensor = {k: torch.Tensor(v).unsqueeze(0).to(device) for k, v in obs.items()} # Add batch dim
            else:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device) # Add batch dim

            # Get deterministic action (usually mean for continuous PPO) or sample
            # For simplicity, let's use the sampling mechanism as in training but without grads
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action_np = action.squeeze(0).cpu().numpy() # Remove batch dim

        # Step in the test environment
        obs, reward, terminated, truncated, info = test_env.step(action_np) # New API
        total_test_reward += reward

    print("Testing finished.")
    # The environment itself prints performance metrics at the end of the episode
    # You can capture these from the final `info` dict if needed, or add custom logging
    print(f"Total Test Reward (sum of log returns): {total_test_reward:.4f}")

    # Example: Access final info if provided by the wrapper/env
    if "episode" in info:
        print(f"Test Episode Return: {info['episode']['r']:.4f}")
        print(f"Test Episode Length: {info['episode']['l']}")
        # Log to tensorboard/wandb if desired
        writer.add_scalar("charts/test_episodic_return", info['episode']['r'], global_step) # Log at end of training

    # --- Cleanup ---
    test_env.close()
    writer.close()

    print("Script finished.")
