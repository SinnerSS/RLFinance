import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import time
import os 
import gym
import pandas as pd

from pathlib import Path
from finrl.agents.portfolio_optimization.architectures import EIIE
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv

from .critic import MLPCritic, CNNCritic

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    SummaryWriter = None

class PPOAgent:
    def __init__(
        self,
        env: PortfolioOptimizationEnv,
        policy_class=EIIE, 
        policy_kwargs=None,
        critic_class=CNNCritic,
        critic_kwargs=None,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        lambda_gae=0.95,
        clip_epsilon=0.2,
        lr_actor=3e-4,
        lr_critic=1e-3,
        entropy_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        optimizer=AdamW,
        use_pvm=False,
        device="cpu",
        log_dir=None,
        validation_env=None,
        validation_freq=20480,
    ):
        if SummaryWriter is None and log_dir is not None:
            raise ImportError("TensorBoard not installed. Please install it: pip install tensorboard")

        self.env = env
        self.validation_env = validation_env
        self.validation_freq = validation_freq
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.use_pvm = use_pvm

        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.critic_kwargs = {} if critic_kwargs is None else critic_kwargs

        self.actor = policy_class(**self.policy_kwargs).to(self.device)
        action_dim = self.env.action_space.shape[0]
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32, device=device))

        if isinstance(env.observation_space, gym.spaces.Dict):
             state_shape = env.observation_space.spaces['state'].shape
        else: 
             state_shape = env.observation_space.shape
        self.critic =  critic_class(state_shape, **self.critic_kwargs)

        self.optimizer_actor = optimizer(
            list(self.actor.parameters()) + [self.log_std], lr=lr_actor
        )
        self.optimizer_critic = optimizer(self.critic.parameters(), lr=lr_critic)

        if self.use_pvm:
             self.pvm = PVM(self.env.episode_length, self.env.portfolio_size + 1)

        self._setup_rollout_buffer()

        maybe_obs, maybe_info = self.env.reset()
        if isinstance(maybe_obs, tuple):
             self.current_obs = maybe_obs[0]
        else:
             self.current_obs = maybe_obs
        self.current_episode_steps = 0
        self.total_episodes_done = 0

        self.log_dir = log_dir
        self.writer = None
        if self.log_dir is not None:
            run_name = f"PPO_{time.strftime('%Y%m%d-%H%M%S')}"
            self.log_dir = os.path.join(log_dir, run_name)
            print(f"Logging to: {self.log_dir}")
            self.writer = SummaryWriter(log_dir=self.log_dir)
        self.global_step = 0
        self.last_validation_step = 0

        self.ep_info_buffer = []


    def _setup_rollout_buffer(self):
        """Initializes lists to store rollout data."""
        self.buffer = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
            "values": [],
            "last_actions_env": [],
            "returns": torch.zeros(self.n_steps, device=self.device),
            "advantages": torch.zeros(self.n_steps, device=self.device),
        }
        self.buffer_ptr = 0
        self.buffer_full = False

    def _add_to_buffer(self, obs, action, reward, done, log_prob, value, last_action_env):
        """Adds a transition to the buffer."""
        self.buffer["observations"].append(obs)
        self.buffer["actions"].append(action)
        self.buffer["rewards"].append(reward)
        self.buffer["dones"].append(done)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["values"].append(value)
        self.buffer["last_actions_env"].append(last_action_env)

        self.buffer_ptr = (self.buffer_ptr + 1) % self.n_steps
        self.buffer_full = self.buffer_full or self.buffer_ptr == 0

    def _clear_buffer(self):
        """Clears the buffer lists."""
        for key in ["observations", "actions", "rewards", "dones", "log_probs", "values", "last_actions_env"]:
            self.buffer[key].clear()
        self.buffer_ptr = 0
        self.buffer_full = False

    def _get_action_and_value(self, obs, last_action_for_policy=None):
        """Gets action, log_prob, entropy, and value estimate from networks."""
        if isinstance(obs, dict):
             obs_state_np = obs['state']
             last_action_np = obs.get('last_action', None)
             if last_action_np is None:
                  action_dim = self.env.action_space.shape[0]
                  last_action_np = np.zeros(action_dim, dtype=np.float32)
                  last_action_np[0] = 1.0
             obs_for_critic = obs
        elif isinstance(obs, np.ndarray):
             obs_state_np = obs
             if self.use_pvm:
                  last_action_np = self.pvm.retrieve()
             else:
                  action_dim = self.env.action_space.shape[0]
                  last_action_np = np.zeros(action_dim, dtype=np.float32)
                  last_action_np[0] = 1.0
             obs_for_critic = obs_state_np
        else:
            raise ValueError(f"Unsupported observation type in _get_action_and_value: {type(obs)}")


        obs_state_batch = np.expand_dims(obs_state_np, axis=0)
        last_action_batch = np.expand_dims(last_action_np, axis=0)

        obs_state_tensor = torch.from_numpy(obs_state_batch).float().to(self.device)
        last_action_tensor = torch.from_numpy(last_action_batch).float().to(self.device)

        action_logits = self.actor.mu(obs_state_tensor, last_action_tensor)
        action_std = torch.exp(self.log_std)
        distribution = Normal(action_logits, action_std)
        raw_action = distribution.sample()
        log_prob = distribution.log_prob(raw_action).sum(axis=-1)
        entropy = distribution.entropy().sum(axis=-1)
        final_action_weights = torch.softmax(raw_action, dim=-1)
        final_action_weights_np = final_action_weights.cpu().detach().numpy().squeeze()
        raw_action_detached = raw_action.detach()

        value = self.critic(obs_for_critic).squeeze()

        return raw_action_detached, final_action_weights_np, log_prob, entropy, value


    def _calculate_gae(self, rewards, dones, values):
        """Calculates Generalized Advantage Estimation (GAE)."""
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0.0

        with torch.no_grad():
            next_obs = self.current_obs 
            next_value = self.critic(next_obs).squeeze()

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - dones[t] # Use done flag of the last step collected (D_T)
                next_values = next_value           # Use V(S_{T})
            else:
                next_non_terminal = 1.0 - dones[t+1] # Use done flag D_{t+1}
                next_values = values[t+1]          # Use V(S_{t+1})

            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.lambda_gae * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values
        return advantages, returns


    def collect_experiences(self):
        """Collects experiences from the environment for n_steps."""
        self._clear_buffer()
        self.actor.eval()
        self.critic.eval()

        current_rollout_ep_rewards = []
        current_rollout_ep_lengths = []
        current_ep_reward = 0
        current_ep_length = 0

        for step in range(self.n_steps):
            current_obs_for_policy = self.current_obs

            with torch.no_grad():
                action_raw, action_final_weights, log_prob, _, value = \
                    self._get_action_and_value(current_obs_for_policy)
                action_raw_cpu = action_raw.cpu()
                log_prob_cpu = log_prob.cpu()
                value_cpu = value.cpu()

            step_result = self.env.step(action_final_weights)

            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                next_obs, reward, done, info = step_result
            else:
                raise ValueError("Unexpected number of return values from env.step()")

            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done_tensor = torch.tensor(float(done), dtype=torch.float32, device=self.device)

            if self.use_pvm:
                 self.pvm.add(action_final_weights)

            self._add_to_buffer(
                 current_obs_for_policy, action_raw_cpu, reward_tensor,
                 done_tensor, log_prob_cpu, value_cpu, action_final_weights
             )

            self.current_obs = next_obs
            self.current_episode_steps += 1
            current_ep_reward += reward
            current_ep_length += 1

            if done:
                self.total_episodes_done += 1
                current_rollout_ep_rewards.append(current_ep_reward)
                current_rollout_ep_lengths.append(current_ep_length)
                self.ep_info_buffer.append({'reward': current_ep_reward, 'length': current_ep_length})
                current_ep_reward = 0
                current_ep_length = 0

                maybe_obs, maybe_info = self.env.reset()
                if isinstance(maybe_obs, tuple):
                    self.current_obs = maybe_obs[0]
                else:
                    self.current_obs = maybe_obs
                self.current_episode_steps = 0
                if self.use_pvm:
                    self.pvm.reset()

        if self.writer and current_rollout_ep_rewards:
            avg_ep_reward = np.mean(current_rollout_ep_rewards)
            avg_ep_length = np.mean(current_rollout_ep_lengths)
            self.writer.add_scalar("Rollout/Avg_Episode_Reward", avg_ep_reward, self.global_step)
            self.writer.add_scalar("Rollout/Avg_Episode_Length", avg_ep_length, self.global_step)
            self.writer.add_scalar("Rollout/Total_Episodes_Done", self.total_episodes_done, self.global_step)
        if self.writer:
             std_mean = torch.exp(self.log_std).mean().item()
             self.writer.add_scalar("Stats/Action_Std_Mean", std_mean, self.global_step)


    def _update_networks(self):
        """Performs PPO update steps and logs training metrics."""
        self.actor.train()
        self.critic.train()

        if isinstance(self.buffer["observations"][0], dict):
             obs_states = np.array([o['state'] for o in self.buffer["observations"]])
             initial_last_action = np.zeros(self.env.action_space.shape[0], dtype=np.float32); initial_last_action[0] = 1.0
             batch_last_actions = np.array([o.get('last_action', initial_last_action) for o in self.buffer["observations"]], dtype=np.float32)
             batch_obs_tensor = torch.tensor(obs_states, dtype=torch.float32).to(self.device)
             batch_last_actions_tensor = torch.tensor(batch_last_actions, dtype=torch.float32).to(self.device)
             # Combine obs state and last action for dataset if critic/actor need both unified
             # For now, keep separate and handle in dataloader loop
             dataset_tensors = (batch_obs_tensor, batch_last_actions_tensor) # Pass both state and last_action
        else: # Box observations
             batch_obs = np.array(self.buffer["observations"])
             batch_obs_tensor = torch.tensor(batch_obs, dtype=torch.float32).to(self.device)
             # Handle last_action if needed for Box (complex, assume not needed or policy adapted for now)
             # If EIIE/EI3 absolutely need last_action with Box env, PVM or other mechanism required
             batch_last_actions_tensor = torch.zeros((self.n_steps, self.env.action_space.shape[0]), dtype=torch.float32, device=self.device) # Placeholder
             dataset_tensors = (batch_obs_tensor,) # Only pass state if last_action not handled/needed


        batch_actions = torch.stack(self.buffer["actions"]).to(self.device)
        batch_log_probs_old = torch.stack(self.buffer["log_probs"]).to(self.device)
        batch_rewards = torch.stack(self.buffer["rewards"]).to(self.device)
        batch_dones = torch.stack(self.buffer["dones"]).to(self.device)
        batch_values = torch.stack(self.buffer["values"]).to(self.device)

        advantages, returns = self._calculate_gae(batch_rewards, batch_dones, batch_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        full_dataset_tensors = dataset_tensors + (batch_actions, batch_log_probs_old, batch_values, advantages, returns)
        dataset = TensorDataset(*full_dataset_tensors)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        all_policy_losses, all_value_losses, all_entropies, all_clip_fractions = [], [], [], []

        for epoch in range(self.n_epochs):
            for minibatch_data in dataloader:
                if len(dataset_tensors) == 2:
                    obs_mini, last_actions_mini, actions_mini, log_probs_old_mini, values_old_mini, advantages_mini, returns_mini = minibatch_data
                    obs_for_critic = {'state': obs_mini.cpu().numpy()}
                else: 
                    obs_mini, actions_mini, log_probs_old_mini, values_old_mini, advantages_mini, returns_mini = minibatch_data
                    last_actions_mini = None
                    obs_for_critic = obs_mini.cpu().numpy()


                if last_actions_mini is None and self.policy_class in [EIIE, EI3]:
                     raise ValueError("EIIE/EI3 policy requires last_action, but it's not available in minibatch (possibly Box env without PVM?)")

                action_logits = self.actor.mu(obs_mini, last_actions_mini)
                action_std = torch.exp(self.log_std)
                distribution = Normal(action_logits, action_std)
                log_probs_new = distribution.log_prob(actions_mini).sum(axis=-1)
                entropy = distribution.entropy().sum(axis=-1)

                ratio = torch.exp(log_probs_new - log_probs_old_mini)
                surr1 = ratio * advantages_mini
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_mini
                policy_loss = -torch.min(surr1, surr2).mean()

                values_new = self.critic(obs_mini).squeeze()
                values_clipped = values_old_mini + torch.clamp(values_new - values_old_mini, -self.clip_epsilon, self.clip_epsilon)

                vf_loss1 = nn.functional.mse_loss(values_new, returns_mini)
                vf_loss2 = nn.functional.mse_loss(values_clipped, returns_mini)
                value_loss = 0.5 * torch.mean(torch.maximum(vf_loss1, vf_loss2))

                loss = policy_loss - self.entropy_coef * entropy.mean() + self.vf_coef * value_loss

                self.optimizer_actor.zero_grad()
                # Separate backward passes are generally cleaner for actor-critic
                actor_loss_component = policy_loss - self.entropy_coef * entropy.mean()
                actor_loss_component.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + [self.log_std], self.max_grad_norm)
                self.optimizer_actor.step()

                # --- Critic Update ---
                values_new_for_critic_update = self.critic(obs_mini).squeeze()
                values_clipped_for_critic_update = values_old_mini + torch.clamp(
                     values_new_for_critic_update - values_old_mini, -self.clip_epsilon, self.clip_epsilon
                )
                vf_loss1_crit = nn.functional.mse_loss(values_new_for_critic_update, returns_mini, reduction='none')
                vf_loss2_crit = nn.functional.mse_loss(values_clipped_for_critic_update, returns_mini, reduction='none')
                value_loss_final = 0.5 * torch.mean(torch.maximum(vf_loss1_crit, vf_loss2_crit))

                self.optimizer_critic.zero_grad()
                value_loss_final.backward() # Use the separately computed critic loss
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

                # --- Store metrics for logging ---
                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropies.append(entropy.mean().item())
                # Calculate clip fraction (optional but useful PPO metric)
                with torch.no_grad():
                    clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float()).item()
                    all_clip_fractions.append(clip_fraction)

        if self.writer:
            avg_policy_loss = np.mean(all_policy_losses)
            avg_value_loss = np.mean(all_value_losses)
            avg_entropy = np.mean(all_entropies)
            avg_clip_fraction = np.mean(all_clip_fractions)

            self.writer.add_scalar("Train/Policy_Loss", avg_policy_loss, self.global_step)
            self.writer.add_scalar("Train/Value_Loss", avg_value_loss, self.global_step)
            self.writer.add_scalar("Train/Entropy", avg_entropy, self.global_step)
            self.writer.add_scalar("Train/Clip_Fraction", avg_clip_fraction, self.global_step)
            self.writer.add_scalar("Train/Learning_Rate_Actor", self.optimizer_actor.param_groups[0]['lr'], self.global_step)
            self.writer.add_scalar("Train/Learning_Rate_Critic", self.optimizer_critic.param_groups[0]['lr'], self.global_step)

            self.writer.add_scalar("Stats/Advantage_Mean", advantages.mean().item(), self.global_step)
            self.writer.add_scalar("Stats/Return_Mean", returns.mean().item(), self.global_step)


    def train(self, total_timesteps):
        """Main training loop with logging and validation."""
        timesteps_elapsed = 0
        pbar = tqdm(total=total_timesteps)

        while timesteps_elapsed < total_timesteps:
            start_time = time.time()

            self.collect_experiences()

            if self.buffer_full:
                self._update_networks() 
                pbar.update(self.n_steps)
                timesteps_elapsed += self.n_steps
                self.global_step = timesteps_elapsed 

                end_time = time.time()
                fps = int(self.n_steps / (end_time - start_time + 1e-8))
                if self.writer:
                    self.writer.add_scalar("Time/FPS", fps, self.global_step)

                if self.validation_env and self.global_step >= self.last_validation_step + self.validation_freq:
                     print(f"\n--- Running Validation @ step {self.global_step} ---")
                     self.run_validation()
                     self.last_validation_step = self.global_step
                     print("--- Validation Complete ---")

            else:
                 print("Warning: Buffer not full after collection.")
                 break

        pbar.close()
        if self.writer:
            self.writer.close()
        print("Training finished.")

    def run_validation(self, deterministic=True):
        """Runs evaluation on the validation environment and logs results."""
        if not self.validation_env or not self.writer:
            print("Validation environment or TensorBoard writer not configured.")
            return

        print("Running validation...")
        val_env = self.validation_env
       
        maybe_obs, maybe_info = val_env.reset()
        if isinstance(maybe_obs, tuple):
             obs = maybe_obs[0]
        else:
             obs = maybe_obs
        done = False
        episode_reward = 0
        episode_steps = 0

        test_pvm = None
        if self.use_pvm:
            test_pvm = PVM(val_env.episode_length, val_env.portfolio_size + 1)

        self.actor.eval()

        while not done:
            current_obs_for_policy = obs
            if isinstance(obs, dict):
                last_action_np = obs.get('last_action', None)
                if last_action_np is None:
                    action_dim=val_env.action_space.shape[0]; last_action_np=np.zeros(action_dim,dtype=np.float32); last_action_np[0]=1.0
                obs_state_batch = np.expand_dims(obs['state'], axis=0)
            else: # Box
                if self.use_pvm and test_pvm: last_action_np = test_pvm.retrieve()
                else: action_dim=val_env.action_space.shape[0]; last_action_np=np.zeros(action_dim,dtype=np.float32); last_action_np[0]=1.0
                obs_state_batch = np.expand_dims(obs, axis=0)

            last_action_batch = np.expand_dims(last_action_np, axis=0)

            with torch.no_grad():
                action_logits = self.actor.mu(obs_state_batch, last_action_batch)
                if deterministic: action_raw = action_logits
                else: action_std = torch.exp(self.log_std); distribution = Normal(action_logits, action_std); action_raw = distribution.sample()
                final_action_weights = torch.softmax(action_raw, dim=-1)
                final_action_weights_np = final_action_weights.cpu().numpy().squeeze()

            step_result = val_env.step(final_action_weights_np)
            if len(step_result) == 5: next_obs, reward, terminated, truncated, info = step_result; done = terminated or truncated
            elif len(step_result) == 4: next_obs, reward, done, info = step_result
            else: raise ValueError("Unexpected number of return values from env.step()")


            if self.use_pvm and test_pvm: test_pvm.add(final_action_weights_np)
            obs = next_obs
            episode_reward += reward
            episode_steps += 1

            if done:
                print(f"Validation Episode Finished. Steps: {episode_steps}, Total Reward: {episode_reward:.4f}")
                final_value = info.get("final_portfolio_value", None)
                sharpe_ratio = info.get("sharpe_ratio", None)
                max_drawdown = info.get("max_drawdown", None)

                if final_value is not None:
                    self.writer.add_scalar("Validation/Final_Portfolio_Value", final_value, self.global_step)
                if sharpe_ratio is not None:
                    self.writer.add_scalar("Validation/Sharpe_Ratio", sharpe_ratio, self.global_step)
                if max_drawdown is not None:
                    self.writer.add_scalar("Validation/Max_Drawdown", max_drawdown, self.global_step)
                self.writer.add_scalar("Validation/Episode_Reward", episode_reward, self.global_step)
                self.writer.add_scalar("Validation/Episode_Length", episode_steps, self.global_step)
                break

        self.actor.train()


    def test(self, test_env, deterministic=True):
      """
      Tests the learned policy on a given environment.
      Note: This test function only runs one episode and relies on the environment's
            internal printing for detailed results. For logging test results over
            multiple runs or episodes, consider adapting the run_validation logic.
      """
      print("--- Running Final Test ---")
      maybe_obs, maybe_info = test_env.reset()
      if isinstance(maybe_obs, tuple): obs = maybe_obs[0]
      else: obs = maybe_obs
      done = False
      episode_reward = 0
      episode_steps = 0

      test_pvm = None
      if self.use_pvm:
          test_pvm = PVM(test_env.episode_length, test_env.portfolio_size + 1)

      self.actor.eval()

      while not done:
          current_obs_for_policy = obs
          if isinstance(obs, dict):
              last_action_np = obs.get('last_action', None)
              if last_action_np is None: action_dim=test_env.action_space.shape[0]; last_action_np=np.zeros(action_dim,dtype=np.float32); last_action_np[0]=1.0
              obs_state_batch = np.expand_dims(obs['state'], axis=0)
          else: # Box
              if self.use_pvm and test_pvm: last_action_np = test_pvm.retrieve()
              else: action_dim=test_env.action_space.shape[0]; last_action_np=np.zeros(action_dim,dtype=np.float32); last_action_np[0]=1.0
              obs_state_batch = np.expand_dims(obs, axis=0)
          last_action_batch = np.expand_dims(last_action_np, axis=0)

          with torch.no_grad():
              action_logits = self.actor.mu(obs_state_batch, last_action_batch)
              if deterministic: action_raw = action_logits
              else: action_std = torch.exp(self.log_std); distribution = Normal(action_logits, action_std); action_raw = distribution.sample()
              final_action_weights = torch.softmax(action_raw, dim=-1)
              final_action_weights_np = final_action_weights.cpu().numpy().squeeze()

          step_result = test_env.step(final_action_weights_np)
          if len(step_result) == 5: next_obs, reward, terminated, truncated, info = step_result; done = terminated or truncated
          elif len(step_result) == 4: next_obs, reward, done, info = step_result
          else: raise ValueError("Unexpected number of return values from env.step()")

          if self.use_pvm and test_pvm: test_pvm.add(final_action_weights_np)
          obs = next_obs
          episode_reward += reward
          episode_steps += 1

          if done:
              print(f"Test Episode Finished. Steps: {episode_steps}, Total Reward: {episode_reward:.4f}")
              # The environment's print statements will show detailed results
              # You could potentially log these final test metrics too if needed
              # final_value = info.get("final_portfolio_value", None) etc.
              # if self.writer and final_value is not None:
              #    self.writer.add_scalar("Test/Final_Value_Run", final_value, self.global_step)
              break

      self.actor.train()
      print("--- Test Complete ---")
