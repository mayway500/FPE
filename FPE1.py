#@title Instantiate environment
import torch
import torch.nn as nn
import pandas as pd
import os
import networkx as nx
import random
import numpy as np

from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from torchrl.data import Unbounded, Categorical, Composite, DiscreteTensorSpec, MultiCategorical as MultiCategoricalSpec
from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    CompositeSpec as DataCompositeSpec,
    UnboundedDiscreteTensorSpec,
    DiscreteTensorSpec as DataDiscreteTensorSpec,
    MultiDiscreteTensorSpec as DataMultiDiscreteTensorSpec,
    BoxList as DataBoxList,
    CategoricalBox as DataCategoricalBox,
    UnboundedContinuous as DataUnboundedContinuous
)
from typing import Optional
from torchrl.envs.utils import check_env_specs

# Define the AnFuelpriceEnv class, inheriting from EnvBase
class AnFuelpriceEnv(EnvBase):
    # Accept pre-loaded data_tensor as an argument
    def __init__(self, data_tensor: torch.Tensor, num_envs, seed, device, num_agents=13, **kwargs):
        self.episode_length = kwargs.get('episode_length', 100)
        self.num_agents = num_agents
        self.allow_repeat_data = kwargs.get('allow_repeat_data', False)
        self.num_envs = num_envs
        self.current_data_index = torch.zeros(num_envs, dtype=torch.int64, device=device)

        # Use the pre-loaded data_tensor
        self.combined_data = data_tensor
        self.device = device

        # Ensure data was loaded successfully before proceeding
        if self.combined_data is None:
             print("AnFuelpriceEnv.__init__: Error: data_tensor is None. Data loading likely failed before environment instantiation.") # Debug print
             raise RuntimeError("Data loading failed before environment instantiation.")

        # Assuming the first num_agents columns of combined_data are the node features
        self.node_feature_dim = 13 # Based on the data structure described previously
        self.num_individual_actions = 3 # 3 categories per action feature (down/hold/up)
        self.num_individual_actions_features = 13 # 13 action features per agent

        # Calculate bounds assuming first self.node_feature_dim columns are observation features
        if self.combined_data.shape[1] < self.node_feature_dim:
             print(f"AnFuelpriceEnv.__init__: Error: combined_data has {self.combined_data.shape[1]} columns, but expected at least {self.node_feature_dim} for node features.")
             raise ValueError("Combined data shape mismatch with node feature dimension.")

        # Bounds for observation features (the first self.node_feature_dim columns)
        self.obs_min = torch.min(self.combined_data[:, :self.node_feature_dim], dim=0)[0].to(self.device)
        self.obs_max = torch.max(self.combined_data[:, :self.node_feature_dim], dim=0)[0].to(self.device)

        # Check if bounds were set
        if self.obs_min is None or self.obs_max is None:
             print("AnFuelpriceEnv.__init__: Error: Observation bounds (obs_min/obs_max) were not set after data loading.") # Debug print
             raise RuntimeError("Observation bounds (obs_min/obs_max) were not set after data loading.")

        # Define graph structure (fixed for now, e.g., fully connected or ring)
        self.num_nodes_per_graph = self.num_agents
        # Assuming a fully connected graph for simplicity, adjust if needed.
        # For fully connected, each node connects to every other node.
        # Number of directed edges is num_nodes * (num_nodes - 1)
        self.num_edges_per_graph = self.num_agents * (self.num_agents - 1) if self.num_agents > 1 else 0

        if self.num_agents > 1:
            # Create a fully connected graph adjacency list
            sources = torch.arange(self.num_agents).repeat(self.num_agents)
            targets = torch.arange(self.num_agents).repeat_interleave(self.num_agents)
            # Filter out self-loops if they are not intended
            non_self_loop_mask = sources != targets
            sources = sources[non_self_loop_mask]
            targets = targets[non_self_loop_mask]
            self._fixed_edge_index_single = torch.stack([sources, targets], dim=0).to(torch.long)
            self._fixed_num_edges_single = self._fixed_edge_index_single.shape[1]
        else:
            self._fixed_edge_index_single = torch.empty(2, 0, dtype=torch.long)
            self._fixed_num_edges_single = 0
        print(f"AnFuelpriceEnv.__init__: num_edges_per_graph calculated as {self._fixed_num_edges_single} (assuming fully connected without self-loops).")


        super().__init__(device=device, batch_size=[num_envs])

        self._make_specs()

    # Add the _set_seed method
    def _set_seed(self, seed: Optional[int] = None):
        # Implement seeding logic here if needed
        if seed is not None:
            self.seed = seed
        else:
            self.seed = torch.seed() # Use torch's current seed if none provided
        return seed

    # Implement the actual _is_terminal method
    def _is_terminal(self) -> torch.Tensor:
        """
        Determines if the current state is a terminal state.

        Returns:
            torch.Tensor: A boolean tensor of shape [num_envs] indicating
                          whether each environment is in a terminal state.
        """
        # For now, keeping it as always False, allowing episodes to end only by truncation.
        # Replace this with your specific termination conditions if any.
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated


    # Implement the actual _batch_reward method
    def _batch_reward(self, data_indices: torch.Tensor, tensordict: TensorDictBase) -> TensorDictBase:
        # Check if combined_data is available
        if self.combined_data is None:
            print("AnFuelpriceEnv._batch_reward: Error: combined_data is None. Data loading likely failed.")
            # Return a tensordict with zero rewards and appropriate batch size
            original_batch_shape = data_indices.shape
            num_agents = self.num_agents
            rewards_reshaped = torch.zeros(*original_batch_shape, num_agents, 1, dtype=torch.float32, device=self.device)
            return TensorDict({("agents", "reward"): rewards_reshaped}, batch_size=original_batch_shape, device=self.device)


        # data_indices shape: [num_envs] (for _step) or [num_envs, num_steps] (for _reset)
        # tensordict is the input tensordict passed to step() by the collector (containing the action)

        # Determine the flat batch size based on data_indices batch size
        original_batch_shape = data_indices.shape
        if len(original_batch_shape) == 2:
            num_envs_current_batch = original_batch_shape[0]
            num_steps_current_batch = original_batch_shape[1]
            flat_batch_size = num_envs_current_batch * num_steps_current_batch
            # Flatten data_indices for combined access
            data_indices_flat = data_indices.view(flat_batch_size)
        elif len(original_batch_shape) == 1:
            num_envs_current_batch = original_batch_shape[0]
            num_steps_current_batch = 1
            flat_batch_size = num_envs_current_batch
            data_indices_flat = data_indices # Already flat
        else:
            raise ValueError(f"Unexpected data_indices batch size dimensions: {len(original_batch_shape)}")


        num_agents = self.num_agents
        num_action_features = self.num_individual_actions_features # Should be 13

        # Ensure data_indices are within bounds, handle out-of-bounds gracefully
        valid_mask = (data_indices_flat >= 0) & (data_indices_flat < self.combined_data.shape[0]) # Shape [flat_batch_size]

        # Initialize rewards tensor with shape [flat_batch_size, num_agents, 1]
        rewards_flat = torch.zeros(flat_batch_size, num_agents, 1, dtype=torch.float32, device=self.device)

        # Calculate rewards only for valid environments/timesteps
        valid_indices_flat = torch.where(valid_mask)[0]

        if valid_indices_flat.numel() > 0:
            # Select the relevant slices from rewards and other tensors using valid_indices_flat
            valid_data_indices_flat = data_indices_flat[valid_indices_flat] # Shape [num_valid_flat]

            # Get the actual changes at the current timestep for the reward calculation.
            # Assuming changes are columns 13 to 25 (13 columns) in the combined data.
            # This indexing aligns with the data structure created in the data loading cell.
            # Shape: [num_valid_flat, 13]
            actual_changes_current_step = self.combined_data[valid_data_indices_flat][:, self.node_feature_dim : self.node_feature_dim + num_action_features]


            # Extract actions from the input tensordict for valid data points and restructure them
            # The actions tensor from the policy has shape [batch_size, num_agents, num_individual_actions_features]
            # The batch_size here corresponds to the 'num_envs' dimension of the environment.
            # When called from _step, tensordict has batch_size=[num_envs].
            # When called from _reset (for initial reward calculation if any), tensordict might be None or have a different structure.
            # We only calculate reward based on actions taken *after* the environment is stepped.
            # The collector calls _batch_reward within _step *after* getting the action.

            # Check if actions are available in the input tensordict
            actions_tensor_input = tensordict.get(("agents", "action"), None)

            if actions_tensor_input is not None:
                 # Actions tensor shape from policy: [num_envs, num_agents, num_individual_actions_features]
                 # Need to flatten to match valid_indices_flat if data_indices_flat was flattened.
                 # If original_batch_shape was [num_envs], actions_tensor_input shape is [num_envs, num_agents, num_action_features].
                 # If original_batch_shape was [num_envs, num_steps], actions_tensor_input shape is [num_envs, num_steps, num_agents, num_action_features].

                 if len(original_batch_shape) == 2: # Called from GAE with [num_envs, num_steps]
                      actions_tensor_flat = actions_tensor_input.view(flat_batch_size, num_agents, num_action_features)
                 elif len(original_batch_shape) == 1: # Called from _step with [num_envs]
                      actions_tensor_flat = actions_tensor_input # Already [num_envs, num_agents, num_action_features], which is flat_batch_size, num_agents, num_action_features

                 actions_for_reward = actions_tensor_flat[valid_indices_flat].contiguous() # Shape [num_valid_flat, num_agents, num_action_features]


                 # Calculate reward for each agent.
                 # Assuming reward for agent j is based on their action features [num_action_features]
                 # and the actual changes at the current timestep [num_action_features].
                 # Assuming reward for agent j is sum over action features i:
                 # reward_ji = (action_ji == 2) * change_i - (action_ji == 0) * change_i - (action_ji == 1) * 0.01 * abs(change_i)
                 # Then sum reward_ji over i for agent j.

                 # Expand actual_changes_current_step to match actions_for_reward shape for broadcasting
                 # Shape [num_valid_flat, 1, num_action_features]
                 changes_broadcastable = actual_changes_current_step.unsqueeze(1)

                 # Create masks based on the actions [num_valid_flat, num_agents, num_action_features]
                 down_mask = (actions_for_reward == 0) # Action 0: Down
                 hold_mask = (actions_for_reward == 1) # Action 1: Hold
                 up_mask = (actions_for_reward == 2)   # Action 2: Up

                 # Calculate reward contributions for each action feature comparison
                 # Shape [num_valid_flat, num_agents, num_action_features]
                 reward_contributions_down = -changes_broadcastable * down_mask.float()
                 reward_contributions_up = changes_broadcastable * up_mask.float()
                 reward_contributions_hold = -0.01 * torch.abs(changes_broadcastable) * hold_mask.float() # Small penalty for holding

                 # Sum the reward contributions across the action features dimension for each agent
                 # Shape [num_valid_flat, num_agents]
                 agent_rewards_valid = (reward_contributions_down + reward_contributions_up + reward_contributions_hold).sum(dim=-1)

                 # Add the last dimension to match the expected output shape [num_valid_flat, num_agents, 1]
                 agent_rewards_valid = agent_rewards_valid.unsqueeze(-1)

                 # Assign calculated rewards to the selected slice of the rewards_flat tensor
                 rewards_flat[valid_indices_flat] = agent_rewards_valid

            else:
                 # If actions are not in the input tensordict (e.g., called during reset), rewards remain zero.
                 pass

        # Reshape rewards_flat back to the original input batch size
        rewards_reshaped = rewards_flat.view(*original_batch_shape, num_agents, 1)

        # Return rewards wrapped in a TensorDict
        return TensorDict({("agents", "reward"): rewards_reshaped}, batch_size=original_batch_shape, device=self.device)


    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Check if combined_data is available
        if self.combined_data is None:
            print("AnFuelpriceEnv._step: Error: combined_data is None. Data loading likely failed.")
            # Return a tensordict with done flags set to True and dummy values for other keys
            num_envs = tensordict.batch_size[0] if tensordict.batch_size else self.num_envs
            # Ensure dummy edge_index has the correct shape based on _fixed_num_edges_single
            dummy_edge_index = torch.zeros(num_envs, 2, self._fixed_num_edges_single, dtype=torch.int64, device=self.device)

            dummy_obs_td = TensorDict({
                 "x": torch.zeros(num_envs, self.num_agents, self.node_feature_dim, dtype=torch.float32, device=self.device),
                 "edge_index": dummy_edge_index, # Use the correctly shaped dummy edge index
                 "batch": torch.zeros(num_envs, self.num_agents, dtype=torch.int64, device=self.device),
                 "time": torch.zeros(num_envs, 1, dtype=torch.int64, device=self.device),
            }, batch_size=[num_envs], device=self.device)
            return TensorDict({
                "next": Composite({
                     ("agents", "data"): dummy_obs_td,
                     ('agents', 'reward'): torch.zeros(num_envs, self.num_agents, 1, dtype=torch.float32, device=self.device),
                     "terminated": torch.ones(num_envs, 1, dtype=torch.bool, device=self.device),
                     "truncated": torch.zeros(num_envs, 1, dtype=torch.bool, device=self.device),
                     "done": torch.ones(num_envs, 1, dtype=torch.bool, device=self.device),
                }),
                "terminated": torch.ones(num_envs, 1, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(num_envs, 1, dtype=torch.bool, device=self.device),
                "done": torch.ones(num_envs, 1, dtype=torch.bool, device=self.device),
            }, batch_size=[num_envs], device=self.device)

        action=tensordict.get(("agents", "action"))


        self.current_data_index += 1

        terminated = self._is_terminal()
        truncated = (self.current_data_index >= self.episode_length)

        # The action is now expected to be in the input tensordict passed to step() by the collector
        # Pass the input tensordict directly to _batch_reward
        # _batch_reward will need to extract the action from this tensordict
        # Get the reward tensordict from _batch_reward
        reward_td = self._batch_reward(self.current_data_index, tensordict) # Pass the input tensordict here


        # Logic previously in _get_state_at for next state
        num_envs = self.current_data_index.shape[0]

        # Get data indices for the next step, handling boundaries
        # Use the current_data_index for the next state's data
        data_indices_for_next_state = self.current_data_index
        # Clamp data indices to prevent out-of-bounds access, especially if episode_length is near data end
        # Ensure we don't go beyond the last available data point index
        data_indices_for_next_state = torch.min(data_indices_for_next_state, torch.as_tensor(self.combined_data.shape[0] - 1, device=self.device))

        # Extract the first node_feature_dim columns for each environment
        # If all agents share the same features, extract and then expand
        x_data_time_step = self.combined_data[data_indices_for_next_state, :self.node_feature_dim] # Shape: [num_envs, node_feature_dim]
        # Expand to [num_envs, num_agents, node_feature_dim]
        x_data = x_data_time_step.unsqueeze(1).expand(-1, self.num_agents, -1)

        # Use fixed edge index repeated for the batch, only if there are edges
        if self._fixed_num_edges_single > 0:
            edge_index_data = self._fixed_edge_index_single.unsqueeze(0).repeat(num_envs, 1, 1).to(self.device)
        else:
            # Create an empty edge index tensor with the correct batch size
            edge_index_data = torch.empty(num_envs, 2, 0, dtype=torch.int64, device=self.device)


        next_state_tensordict_data = TensorDict({
             "x": x_data, # Use actual data for node features
             "edge_index": edge_index_data, # Use the fixed edge indices
             "batch": torch.arange(num_envs, device=self.device).repeat_interleave(self.num_agents).view(num_envs, self.num_agents), # Create batch tensor
             "time": self.current_data_index.unsqueeze(-1).to(self.device), # Use the current_data_index as timestamp for the next state
        }, batch_size=[num_envs], device=self.device)

        print("AnFuelpriceEnv._step: Returning tensordict with next observation under ('agents', 'data'). Structure:") # Diagnostic print
        print(next_state_tensordict_data) # Diagnostic print


        # Create the output tensordict containing the next observation, reward, and done flags
        # The next observation and reward should be nested under "next" by the collector.
        # The done flags should be at the root level of the tensordict returned by _step.
        # Structure the output tensordict with reward and done flags at the root level
        output_tensordict = TensorDict({
            # Include the next observation structure under ("agents", "data")
            "next": Composite({
                 ("agents", "data"): next_state_tensordict_data,
                 # Include the reward tensordict directly under the key expected by reward_spec
                 ('agents', 'reward'): reward_td.get(('agents', 'reward')), # Include the reward tensor here
                 "terminated": terminated.unsqueeze(-1).to(self.device),
                 "truncated": truncated.unsqueeze(-1).to(self.device),
                 "done": (terminated | truncated).unsqueeze(-1).to(self.device),
                 # Include placeholders for next recurrent states (will be filled by collector)
                 ('agents', 'rnn_hidden_state'): torch.zeros(num_envs, self.num_agents, 64, dtype=torch.float32, device=self.device),
                 ('agents', 'rnn_hidden_state_forecast'): torch.zeros(num_envs, self.num_agents, 64, dtype=torch.float32, device=self.device),
                 ('agents', 'rnn_hidden_state_value'): torch.zeros(num_envs, self.num_agents, 64, dtype=torch.float32, device=self.device),
            }),
            # Include the done flags at the root level
            "terminated": terminated.to(self.device), # Shape [num_envs]
            "truncated": truncated.to(self.device), # Shape [num_envs]
            "done": (terminated | truncated).to(self.device), # Shape [num_envs]

        }, batch_size=[self.num_envs], device=self.device)

        print("AnFuelpriceEnv._step: Full output tensordict structure:") # Diagnostic print
        print(output_tensordict) # Diagnostic print


        return output_tensordict

    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        # Check if combined_data is available
        if self.combined_data is None:
            print("AnFuelpriceEnv._reset: Error: combined_data is None. Data loading failed during environment initialization.")
            raise RuntimeError("Cannot reset environment: Data loading failed during environment initialization.")


        if self.allow_repeat_data and self.combined_data is not None:
             # Calculate max_start_index based on data length and episode length
             # Max start index allows for at least episode_length steps.
             # If starting at index i, the last step is i + episode_length - 1.
             # The data needed for step i is at index i. The data needed for reward for step i is at index i+1.
             # The state at the end of the episode (after episode_length steps) is at index i + episode_length.
             # Need data up to index i + episode_length.
             # So max start index should be combined_data.shape[0] - episode_length.
             max_start_index = self.combined_data.shape[0] - self.episode_length
             if max_start_index < 0:
                  # If data length is less than episode length + 1, start from index 0 and warn
                  self.current_data_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
                  print("Warning: Data length is less than episode_length + 1. Starting episodes from index 0.")
             else:
                  # Randomly sample start index for each environment
                  self.current_data_index = torch.randint(0, max_start_index + 1, (self.num_envs,), dtype=torch.int64, device=self.device)
        else:
             # If not allowing repeat data, always start from index 0 for all environments
             self.current_data_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        num_envs = self.current_data_index.shape[0]
        data_indices = self.current_data_index
        # Clamp data indices to prevent out-of-bounds access, especially if the start index is near the end
        data_indices = torch.min(data_indices, torch.as_tensor(self.combined_data.shape[0] - 1, device=self.device))

        # Modified: Ensure x_data has shape [num_envs, num_agents, node_feature_dim]
        # Extract node features (first node_feature_dim columns) from combined_data at data_indices
        x_data_time_step = self.combined_data[data_indices, :self.node_feature_dim] # Shape [num_envs, node_feature_dim]
        # Expand to match the number of agents: [num_envs, 1, node_feature_dim] -> [num_envs, num_agents, node_feature_dim]
        x_data = x_data_time_step.unsqueeze(1).expand(-1, self.num_agents, -1)


        # Use fixed edge index repeated for the batch, only if there are edges
        # _fixed_edge_index_single has shape [2, num_edges_per_graph]
        # Unsqueeze to [1, 2, num_edges_per_graph], repeat to [num_envs, 2, num_edges_per_graph]
        if self._fixed_num_edges_single > 0:
            edge_index_data = self._fixed_edge_index_single.unsqueeze(0).repeat(num_envs, 1, 1).to(self.device)
        else:
            # Create an empty edge index tensor with the correct batch size
            edge_index_data = torch.empty(num_envs, 2, 0, dtype=torch.int64, device=self.device)

        print(f"_reset: Using fixed edge index. Generated edge_index_data shape = {edge_index_data.shape}")

        # Create the initial observation tensordict structure
        initial_observation_data_td = TensorDict({
             "x": x_data, # Use actual data for node features
             "edge_index": edge_index_data, # Use the fixed edge indices
             # Create a batch tensor for torch_geometric: [num_envs * num_agents] where each agent in env i has batch index i
             # Reshape to [num_envs, num_agents] to match the structure expected by the policy/value/forecasting modules
             "batch": torch.arange(num_envs, device=self.device).repeat_interleave(self.num_agents).view(num_envs, self.num_agents),
             "time": self.current_data_index.unsqueeze(-1).to(self.device), # Use the current_data_index as timestamp
        }, batch_size=[num_envs], device=self.device)

        print("AnFuelpriceEnv._reset: Returning tensordict with initial observation under ('agents', 'data'). Structure:") # Diagnostic print
        print(initial_observation_data_td) # Diagnostic print


        # Create the output tensordict to return, containing the initial observation and done flags
        # The initial observation should be nested under ("agents", "data")

        output_tensordict = TensorDict({
            # Include the initial observation structure
            ("agents", "data"): initial_observation_data_td,
            # Set initial done, terminated, truncated flags to False at the root level
            "terminated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "truncated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "done": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            # Add initial recurrent states (zeros) at the root level, matching spec shape [num_envs, num_agents, hidden_rnn_dim]
            ('agents', 'rnn_hidden_state'): torch.zeros(self.num_envs, self.num_agents, 64, dtype=torch.float32, device=self.device),
            ('agents', 'rnn_hidden_state_forecast'): torch.zeros(self.num_envs, self.num_agents, 64, dtype=torch.float32, device=self.device),
            ('agents', 'rnn_hidden_state_value'): torch.zeros(self.num_envs, self.num_agents, 64, dtype=torch.float32, device=self.device),
        }, batch_size=[self.num_envs], device=self.device)

        print("AnFuelpriceEnv._reset: Full output tensordict structure:") # Diagnostic print
        print(output_tensordict) # Diagnostic print


        return output_tensordict


    def _make_specs(self):
        print("Debug: Entering _make_specs.")
        # Define the state_spec to match the structure of the tensordict
        # that will be collected by a collector after a step. This tensordict
        # will contain the current state, action, reward, done, and the *next* state.
        self.state_spec = Composite(
             {
                 # Define the keys for the current state, changed 'observation' to 'data'
                 ("agents", "data"): Composite({ # Nested under "agents"
                     "x": Unbounded( # Node features [num_envs, num_agents, node_feature_dim]
                         shape=torch.Size([self.num_envs, self.num_agents, self.node_feature_dim]),
                         dtype=torch.float32,
                         device=self.device
                     ),
                     # Use the fixed number of edges for the spec shape
                     "edge_index": Unbounded( # Edge indices [num_envs, 2, _fixed_num_edges_single]
                         shape=torch.Size([self.num_envs, 2, self._fixed_num_edges_single]),
                         dtype=torch.int64,
                         device=self.device
                     ),
                     # Add 'batch' key to the observation spec
                     "batch": Unbounded( # Batch tensor [num_envs, num_agents]
                         shape=torch.Size([self.num_envs, self.num_agents]),
                         dtype=torch.int64,
                         device=self.device
                     ),
                     # Add timestamp to observation spec
                     "time": Unbounded( # Timestamp (e.g., data index) [num_envs, 1]
                          shape=torch.Size([self.num_envs, 1]),
                          dtype=torch.int64, # Using int64 for data index
                          device=self.device
                      ),
                 }),
                 # Added top-level done, terminated, truncated keys for the current state
                 "done": Categorical(n=2,
                      shape=torch.Size([self.num_envs]), # Shape from _reset and _step output
                      dtype=torch.bool,
                      device=self.device),
                 "terminated": Categorical(n=2,
                      shape=torch.Size([self.num_envs]), # Shape from _reset and _step output
                      dtype=torch.bool,
                      device=self.device),
                 "truncated": Categorical(n=2,
                      shape=torch.Size([self.num_envs]), # Shape from _reset and _step output
                      dtype=torch.bool,
                      device=self.device),

                 # Add recurrent state keys to the state spec
                 # These keys are where the collector will store and retrieve the recurrent states.
                 # The shapes should match the output of the base modules after reshaping
                 # to [num_envs, num_agents, hidden_rnn_dim]. Assuming hidden_rnn_dim=64.
                 ('agents', 'rnn_hidden_state'): Unbounded( # Policy RNN hidden state
                      shape=torch.Size([self.num_envs, self.num_agents, 64]),
                      dtype=torch.float32,
                      device=self.device
                 ),
                 ('agents', 'rnn_hidden_state_forecast'): Unbounded( # Forecasting RNN hidden state
                      shape=torch.Size([self.num_envs, self.num_agents, 64]),
                      dtype=torch.float32,
                      device=self.device
                 ),
                 ('agents', 'rnn_hidden_state_value'): Unbounded( # Value RNN hidden state
                      shape=torch.Size([self.num_envs, self.num_agents, 64]),
                      dtype=torch.float32,
                      device=self.device
                 ),


                 # Define the keys for the next state, nested under "next", changed 'observation' to 'data'
                 "next": Composite({
                      ("agents", "data"): Composite({ # Nested under "agents"
                          "x": Unbounded( # Node features [num_envs, num_agents, node_feature_dim]
                              shape=torch.Size([self.num_envs, self.num_agents, self.node_feature_dim]),
                              dtype=torch.float32,
                              device=self.device
                          ),
                          # Use the fixed number of edges for the spec shape
                          "edge_index": Unbounded( # Edge indices [num_envs, 2, _fixed_num_edges_single]
                              shape=torch.Size([self.num_envs, 2, self._fixed_num_edges_single]),
                              dtype=torch.int64,
                              device=self.device
                          ),
                           # Add 'batch' key to the next observation spec
                          "batch": Unbounded( # Batch tensor [num_envs, num_agents]
                              shape=torch.Size([self.num_envs, self.num_agents]),
                              dtype=torch.int64,
                              device=self.device
                          ),
                          # Add timestamp to next observation spec
                          "time": Unbounded( # Timestamp (e.g., data index) [num_envs, 1]
                               shape=torch.Size([self.num_envs, 1]),
                               dtype=torch.int64, # Using int64 for data index
                               device=self.device
                           ),
                      }),
                      # Also include reward key under ('agents',) under 'next'
                      ('agents', 'reward'): Unbounded(shape=torch.Size([self.num_envs, self.num_agents, 1]), dtype=torch.float32, device=self.device),
                     # Add top-level done, terminated, truncated keys to the "next" composite
                     "done": Categorical(n=2,
                          shape=torch.Size([self.num_envs, 1]), # Corrected shape to match _step output
                          dtype=torch.bool,
                          device=self.device),
                     "terminated": Categorical(n=2,
                          shape=torch.Size([self.num_envs, 1]), # Corrected shape
                           dtype=torch.bool,
                           device=self.device),
                     "truncated": Categorical(n=2,
                          shape=torch.Size([self.num_envs, 1]), # Corrected shape
                      dtype=torch.bool,
                      device=self.device),
                      # Add recurrent state keys to the next state spec
                      # These keys are where the collector will store the next recurrent states.
                      ('agents', 'rnn_hidden_state'): Unbounded(
                           shape=torch.Size([self.num_envs, self.num_agents, 64]),
                           dtype=torch.float32,
                           device=self.device
                      ),
                      ('agents', 'rnn_hidden_state_forecast'): Unbounded(
                           shape=torch.Size([self.num_envs, self.num_agents, 64]),
                           dtype=torch.float32,
                           device=self.device
                      ),
                      ('agents', 'rnn_hidden_state_value'): Unbounded(
                           shape=torch.Size([self.num_envs, self.num_agents, 64]),
                           dtype=torch.float32,
                           device=self.device
                      ),
                 }),
             },
             # The batch size of the state spec is the number of environments
             batch_size=self.batch_size,
             device=self.device,
         )
        print(f"State specification defined with single graph per env structure and batch shape {self.state_spec.shape}.")


        # Corrected nvec to be a 1D tensor of size num_individual_actions_features with value num_individual_actions
        nvec_list = [self.num_individual_actions] * self.num_individual_actions_features
        nvec_tensor = torch.tensor(nvec_list, dtype=torch.int64, device=self.device)


        # Define the action specification using the MultiCategorical SPEC class from torchrl.data
        # The shape should be [num_agents, num_individual_actions_features] for an unbatched environment
        nvec_unbatched = nvec_tensor.repeat(self.num_agents).view(self.num_agents, self.num_individual_actions_features)

        # Use DiscreteTensorSpec for the action if it's a single discrete value per agent
        # If each agent outputs a single discrete action from a set of categories:
        # self.action_spec_unbatched = Composite(
        #       {('agents','action'): DiscreteTensorSpec(n=self.num_individual_actions, # Total categories for a single action
        #                                               shape=torch.Size([self.num_agents]), # Shape [num_agents] for a single action per agent
        #                                               dtype=torch.int64,
        #                                               device=self.device)},
        #       batch_size=[],
        #       device=self.device
        # )

        # If each agent outputs num_individual_actions_features discrete values:
        # This seems to be the case based on the policy module output structure.
        # Using MultiCategoricalSpec from torchrl.data is appropriate here.
        self.action_spec_unbatched = Composite(
              {('agents','action'): MultiCategoricalSpec( # Use the MultiCategorical SPEC class
                                                      # The shape here defines the shape of the action tensor for a single environment: [num_agents, num_individual_actions_features]
                                                      shape=torch.Size([self.num_agents, self.num_individual_actions_features]),
                                                      dtype=torch.int64,
                                                      device=self.device,
                                                      nvec=nvec_unbatched # nvec should be defined in the MultiCategorical SPEC
                                                      )},
              batch_size=[], # Unbatched environment has no batch size at the root
              device=self.device
            )


        print("\nUnbatched Multi-Agent Action specification defined using nested Composites and MultiCategoricalSpec.")
        print(f"Unbatched Environment action_spec: {self.action_spec_unbatched}")
        # The batched action_spec is automatically derived by EnvBase
        print(f"Batched Environment action_spec: {self.action_spec}")


        # Restored original reward spec
        self.reward_spec = Composite(
             {('agents', 'reward'): Unbounded(shape=torch.Size([self.num_envs, self.num_agents, 1]), dtype=torch.float32, device=self.device)},
             batch_size=[self.num_envs],
             device=self.device,
        )
        print(f"Agent-wise Reward specification defined with batch shape {self.reward_spec.shape}.")

        # Define the done_spec to match the structure of the done keys
        # returned by _step.
        self.done_spec = Composite(
            {
                "done": Categorical(n=2,
                     shape=torch.Size([self.num_envs, 1]), # Corrected shape to match _step output
                     dtype=torch.bool,
                     device=self.device),
                "terminated": Categorical(n=2,
                     shape=torch.Size([self.num_envs, 1]), # Corrected shape
                     dtype=torch.bool,
                     device=self.device),
                "truncated": Categorical(n=2,
                     shape=torch.Size([self.num_envs, 1]), # Corrected shape
                      dtype=torch.bool,
                      device=self.device),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )
        print(f"Done specification defined with batch shape {self.done_spec.shape}.")

        self.state_spec.unlock_(recurse=True)
        self.action_spec.unlock_(recurse=True) # Keep action_spec unlocked as it is batched
        self.reward_spec.unlock_(recurse=True)
        self.done_spec.unlock_(recurse=True)
        print("Debug: Exiting _make_specs.")


# Instantiate the environment ONLY if data_tensor is available
print("Instantiating the AnFuelpriceEnv environment...")
# Ensure num_envs, seed, device, and num_agents are defined (assuming previous cells ran)
try:
    _ = num_envs
    _ = seed
    _ = device
    _ = num_agents
    _ = data_tensor # Check if data_tensor is defined from the preprocessing step
except NameError:
    print("Please run the hyperparameters cell and the data loading/preprocessing cell first.")
    # Set essential variables to None or default if not defined
    num_envs = 3
    seed = 42
    device = 'cpu'
    num_agents = 13
    data_tensor = None
    print("Setting default values and data_tensor to None due to missing dependencies.")


if data_tensor is not None:
    try:
        # Use the variables defined in the hyperparameters cell and the loaded data_tensor
        # Pass episode_length and other kwargs as needed
        env = AnFuelpriceEnv(data_tensor=data_tensor, num_envs=num_envs, seed=seed, device=device, episode_length=10, num_agents=num_agents) # Reduced episode_length for a quick test
        print("\nEnvironment instantiated successfully.")

        # Check environment specs
        print("\nChecking environment specs...")
        #check_env_specs(env)
        print("Environment specs checked successfully.")

    except Exception as e:
        print(f"\nAn error occurred during environment instantiation or spec check: {e}")
        env = None # Set env to None if instantiation fails

else:
    env = None
    print("\nEnvironment not instantiated because data_tensor is None.")


    # @title Refactored Module Definitions and Instantiation


import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv
from tensordict.nn import TensorDictModule
import torch.nn.functional as F # Import F for log_softmax

# Define the base policy module with GCN and RNN
class SimpleMultiAgentPolicyModuleGCN(nn.Module):
     def __init__(self, input_x_dim, num_agents, num_individual_actions_features, num_action_categories, hidden_rnn_dim=64, gcn_hidden_dim=64):
         super().__init__()
         self.num_agents = num_agents
         self.num_individual_actions_features = num_individual_actions_features
         self.num_action_categories = num_action_categories
         self.hidden_rnn_dim = hidden_rnn_dim
         self.gcn_hidden_dim = gcn_hidden_dim # Hidden dimension for GCN

         # GCN layer: maps node features to gcn_hidden_dim
         self.gcn1 = GCNConv(input_x_dim, gcn_hidden_dim)

         # RNN layer: processes GCN output sequences
         # Input to RNN is [sequence_length, batch_size, input_size]
         # In our case, sequence_length=1 (single step), batch_size = num_envs * num_agents, input_size = gcn_hidden_dim
         self.rnn = nn.GRU(gcn_hidden_dim, hidden_rnn_dim, batch_first=False)

         # Linear layer: maps flattened RNN output to action logits
         # Input to linear is [batch_size, num_agents * hidden_rnn_dim]
         # Output is [batch_size, num_agents * num_individual_actions_features * num_action_categories]
         self.linear = nn.Linear(hidden_rnn_dim * num_agents,
                                 num_agents * num_individual_actions_features * num_action_categories)

         # Initial hidden state buffer (internal to the module, not directly used by collector)
         # Shape [num_layers * num_directions, batch_size, hidden_size]
         # For a single-layer, unidirectional GRU, this is [1, batch_size, hidden_rnn_dim]
         # Initialize with a dummy batch_size of 1, will be expanded in forward pass
         self.register_buffer("_initial_rnn_hidden_state", torch.zeros(1, 1, self.hidden_rnn_dim))

     # Corrected forward signature to accept inputs as separate arguments
     def forward(self, x, edge_index, prev_rnn_hidden_state=None):
         # x shape: [num_envs, num_agents, input_x_dim]
         # edge_index shape: [num_envs, 2, num_edges_per_graph]
         # prev_rnn_hidden_state shape: [num_envs, num_agents, hidden_rnn_dim] (from collector)

         num_envs = x.shape[0]
         num_agents = self.num_agents # Get num_agents from self
         batch_size_flat = num_envs * num_agents # Flatten batch size for GCN and RNN input


         # Reshape x for GCN: [num_envs * num_agents, input_x_dim]
         x_flat = x.reshape(batch_size_flat, -1)


         # Reshape edge_index for GCN: [num_envs, 2, num_edges_per_graph] -> [2, num_envs * num_edges_per_graph]
         # The edge index needs to be adjusted for the flattened node indices.
         # Assuming edge_index is [num_envs, 2, num_edges_per_graph] where indices are relative to each graph.
         # We need to add an offset to the node indices for each graph.
         # The offset for env i is i * num_agents.
         # Reshape to [num_envs, num_edges_per_graph, 2]
         edge_index_permuted = edge_index.permute(0, 2, 1) # Shape [num_envs, num_edges_per_graph, 2]
         num_edges_per_graph = edge_index_permuted.shape[1]

         # Create offsets: [num_envs, 1, 1]
         offsets = torch.arange(num_envs, device=x.device).view(-1, 1, 1) * num_agents

         # Add offset to edge indices: [num_envs, num_edges_per_graph, 2]
         edge_index_offset = edge_index_permuted + offsets

         # Flatten the edge index for GCN: [num_envs * num_edges_per_graph, 2] -> [2, num_envs * num_edges_per_graph]
         edge_index_flat = edge_index_offset.reshape(-1, 2).permute(1, 0);


         # Apply GCN layers
         # Input: x_flat [num_envs * num_agents, input_x_dim], edge_index_flat [2, num_envs * num_edges_per_graph]
         gcn_output_flat = self.gcn1(x_flat, edge_index_flat) # Shape [num_envs * num_agents, gcn_hidden_dim]


         # Reshape GCN output for RNN: [1, num_envs * num_agents, gcn_hidden_dim]
         # Sequence length = 1 (single step at a time)
         rnn_input = gcn_output_flat.reshape(1, batch_size_flat, self.gcn_hidden_dim)


         # Reshape input state from collector format [num_envs, num_agents, hidden_rnn_dim]
         # to RNN input format [1, num_envs * num_agents, hidden_rnn_dim]
         rnn_input_state = prev_rnn_hidden_state
         if rnn_input_state is not None:
             # Reshape from [num_envs, num_agents, hidden_rnn_dim] to [1, num_envs * num_agents, hidden_rnn_dim]
             rnn_input_state = rnn_input_state.reshape(1, batch_size_flat, self.hidden_rnn_dim)
         else:
             # If prev_rnn_hidden_state is None (first step), initialize with the correct shape
             # using the buffer and expanding to match the current flattened batch size.
             rnn_input_state = self._initial_rnn_hidden_state.expand(1, batch_size_flat, -1).to(x.device)


         # Pass through RNN
         # rnn_output shape: [1, num_envs * num_agents, hidden_rnn_dim]
         # next_rnn_hidden_state_rnn_format shape: [1, num_envs * num_agents, hidden_rnn_dim]
         rnn_output, next_rnn_hidden_state_rnn_format = self.rnn(rnn_input, rnn_input_state.contiguous())


         # Reshape RNN output to [num_envs, num_agents, hidden_rnn_dim] for the linear layer input
         rnn_output_reshaped = rnn_output.reshape(num_envs, num_agents, self.hidden_rnn_dim)

         # Flatten the reshaped RNN output for the linear layer
         flattened_rnn_output = rnn_output_reshaped.reshape(num_envs, -1) # Shape: [num_envs, num_agents * hidden_rnn_dim]


         # Get logits from the linear layer
         # Output shape: [num_envs, num_agents * num_individual_actions_features * num_action_categories]
         flattened_logits = self.linear(flattened_rnn_output)


         # Reshape the next hidden state from RNN format [1, flat_batch, hidden_dim]
         # back to collector format [num_envs, num_agents, hidden_rnn_dim]
         next_rnn_hidden_state = next_rnn_hidden_state_rnn_format.reshape(num_envs, num_agents, self.hidden_rnn_dim)


         # The TensorDictModule expects the output as a tuple matching the out_keys.
         # The policy module outputs action logits and the next RNN hidden state.
         # The TensorDictModule will handle mapping the flattened logits to the correct action spec shape.
         # The action spec is [num_agents, num_individual_actions_features] with num_action_categories.
         # The TensorDictModule for policy expects the output logits to be shape [batch_size, num_agents * num_individual_actions_features * num_action_categories].
         # The TensorDictModule will then apply log_softmax and sample/compute log_prob.
         # The next hidden state should be outputted as [num_envs, num_agents, hidden_rnn_dim].

         return flattened_logits, next_rnn_hidden_state


# Define the base value module with GCN and RNN
class SimpleMultiAgentValueModuleGCN(nn.Module):
    def __init__(self, input_x_dim, num_agents, hidden_rnn_dim=64, gcn_hidden_dim=64):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_rnn_dim = hidden_rnn_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.input_x_dim = input_x_dim # Store input_x_dim

        # Add a linear layer before GCNConv if input_x_dim is not gcn_hidden_dim
        self.pre_gcn_linear = None
        if self.input_x_dim != self.gcn_hidden_dim:
            # print(f"Value Module: Adding pre-GCN linear layer from {self.input_x_dim} to {self.gcn_hidden_dim}") # Debug print
            self.pre_gcn_linear = nn.Linear(self.input_x_dim, self.gcn_hidden_dim)
            gcn_input_dim = self.gcn_hidden_dim
        else:
            gcn_input_dim = self.input_x_dim

        # GCN layer: maps node features to gcn_hidden_dim
        self.gcn1 = GCNConv(gcn_input_dim, gcn_hidden_dim)

        # RNN layer: processes GCN output sequences
        self.rnn = nn.GRU(gcn_hidden_dim, hidden_rnn_dim, batch_first=False)

        # Linear layer: maps flattened RNN output to a single value estimate per environment
        self.linear = nn.Linear(self.hidden_rnn_dim * self.num_agents, 1)

        # Initial hidden state buffer
        self.register_buffer("_initial_rnn_hidden_state", torch.zeros(1, 1, self.hidden_rnn_dim))


    # Corrected forward signature to accept inputs as separate arguments
    def forward(self, x, edge_index, prev_rnn_hidden_state=None):
        # x shape: [num_envs, num_agents, input_x_dim]
        # edge_index shape: [num_envs, 2, num_edges_per_graph]
        # prev_rnn_hidden_state shape: [num_envs, num_agents, hidden_rnn_dim] (from collector)

        num_envs = x.shape[0]
        num_agents = self.num_agents
        batch_size_flat = num_envs * num_agents # Flatten batch size for GCN and RNN input


        # Reshape x: [num_envs * num_agents, input_x_dim]
        x_flat = x.reshape(batch_size_flat, -1)


        # Pass through pre-GCN linear layer if it exists
        if self.pre_gcn_linear is not None:
            # print(f"Value Module: Applying pre-GCN linear layer to x_flat (shape {x_flat.shape})") # Debug print
            x_flat = self.pre_gcn_linear(x_flat)
            # print(f"Value Module: x_flat shape after linear = {x_flat.shape}") # Debug print


        # Reshape edge_index for GCN (with offset)
        edge_index_permuted = edge_index.permute(0, 2, 1)
        num_edges_per_graph = edge_index_permuted.shape[1]
        offsets = torch.arange(num_envs, device=x.device).view(-1, 1, 1) * num_agents
        edge_index_offset = edge_index_permuted + offsets
        edge_index_flat = edge_index_offset.reshape(-1, 2).permute(1, 0);

        # Debugging prints for edge_index reshaping (can be commented out after verification)
        # print(f"Value Module: x shape = {x.shape}, edge_index shape = {edge_index.shape}")
        # print(f"Value Module: edge_index_permuted shape = {edge_index_permuted.shape}")
        # print(f"Value Module: offsets shape = {offsets.shape}")
        # print(f"Value Module: edge_index_offset shape = {edge_index_offset.shape}")
        # print(f"Value Module: edge_index_flat shape = {edge_index_flat.shape}")
        # print(f"Value Module: edge_index_flat dtype = {edge_index_flat.dtype}")


        # Apply GCN layers FIRST
        # Input: x_flat [num_envs * num_agents, gcn_input_dim], edge_index_flat [2, num_envs * num_edges_per_graph]
        # print(f"Value Module: Passing to GCNConv.") # Debug print
        # print(f"Value Module: x_flat shape = {x_flat.shape}") # Debug print
        # print(f"Value Module: edge_index_flat shape = {edge_index_flat.shape}") # Debug print
        # print(f"Value Module: x_flat dtype = {x_flat.dtype}, x_flat is_contiguous = {x_flat.is_contiguous()}") # Debug print
        # print(f"Value Module: edge_index_flat dtype = {edge_index_flat.dtype}, edge_index_flat is_contiguous = {edge_index_flat.is_contiguous()}") # Debug print

        # Debugging print for GCNConv parameter shapes (can be commented out)
        # if hasattr(self.gcn1, 'lin') and hasattr(self.gcn1.lin, 'weight'):
        #      print(f"Value Module: GCNConv internal linear weight shape = {self.gcn1.lin.weight.shape}")
        #      print(f"Value Module: GCNConv internal linear weight dtype = {self.gcn1.lin.weight.dtype}")
        # else:
        #      print("Value Module: GCNConv internal linear layer or weight not found directly.")

        gcn_output_flat = self.gcn1(x_flat, edge_index_flat) # Shape [num_envs * num_agents, gcn_hidden_dim]
        # print(f"Value Module: GCNConv output shape = {gcn_output_flat.shape}") # Debug print


        # Reshape GCN output for RNN: [1, num_envs * num_agents, gcn_hidden_dim]
        # Sequence length = 1 (single step at a time)
        rnn_input = gcn_output_flat.reshape(1, batch_size_flat, self.gcn_hidden_dim)

        # Reshape input state from collector format [num_envs, num_agents, hidden_rnn_dim]
        # to RNN input format [1, num_envs * num_agents, hidden_rnn_dim]
        rnn_input_state = prev_rnn_hidden_state
        if rnn_input_state is not None:
             # Reshape from [num_envs, num_agents, hidden_rnn_dim] to [1, num_envs * num_agents, hidden_rnn_dim]
             rnn_input_state = rnn_input_state.reshape(1, batch_size_flat, self.hidden_rnn_dim)
        else:
             # If prev_rnn_hidden_state is None (first step), initialize with the correct shape
             # using the buffer and expanding to match the current flattened batch size.
             rnn_input_state = self._initial_rnn_hidden_state.expand(1, batch_size_flat, -1).to(x.device)


        # Pass through RNN SECOND
        # rnn_output shape: [1, num_envs * num_agents, hidden_rnn_dim]
        # next_rnn_hidden_state_rnn_format shape: [1, num_envs * num_agents, hidden_rnn_dim]
        rnn_output, next_rnn_hidden_state_rnn_format = self.rnn(rnn_input, rnn_input_state.contiguous())


        # Reshape RNN output to [num_envs, num_agents, hidden_rnn_dim] for the linear layer input
        rnn_output_reshaped = rnn_output.reshape(num_envs, num_agents, self.hidden_rnn_dim)

        # Flatten the reshaped RNN output for the linear layer (for single value per env)
        flattened_rnn_output = rnn_output_reshaped.reshape(num_envs, -1) # Shape: [num_envs, num_agents * hidden_rnn_dim]


        # Get the value from the linear layer
        # Output shape: [num_envs, 1]
        value = self.linear(flattened_rnn_output)


        # Reshape the next hidden state from RNN format [1, flat_batch, hidden_dim]
        # back to collector format [num_envs, num_agents, hidden_rnn_dim]
        next_rnn_hidden_state = next_rnn_hidden_state_rnn_format.reshape(num_envs, num_agents, self.hidden_rnn_dim)


        # The TensorDictModule expects the output as a tuple matching the out_keys.
        # The value module outputs the value estimate and the next RNN hidden state.
        return value, next_rnn_hidden_state


class SimpleMultiAgentForecastingModuleGCN(nn.Module):
    """
    A simple GCN-RNN based module for forecasting input features for multiple agents.
    """
    def __init__(self, input_x_dim, num_agents, forecast_horizon, hidden_rnn_dim=64, gcn_hidden_dim=64):
        super().__init__()
        self.num_agents = num_agents
        self.forecast_horizon = forecast_horizon
        self.hidden_rnn_dim = hidden_rnn_dim
        self.input_x_dim = input_x_dim
        self.gcn_hidden_dim = gcn_hidden_dim

        # GCN layer: maps node features to gcn_hidden_dim
        self.gcn1 = GCNConv(input_x_dim, gcn_hidden_dim)

        # RNN layer: processes GCN output sequences
        self.rnn = nn.GRU(gcn_hidden_dim, hidden_rnn_dim, batch_first=False)

        # Linear layer: maps flattened RNN output to the forecast
        # Output shape: [num_envs, num_agents * input_x_dim * forecast_horizon]
        self.linear = nn.Linear(self.hidden_rnn_dim * self.num_agents,
                                self.num_agents * self.input_x_dim * self.forecast_horizon)

        # Initial hidden state buffer
        self.register_buffer("_initial_rnn_hidden_state", torch.zeros(1, 1, self.hidden_rnn_dim))

    # Corrected forward signature to accept inputs as separate arguments
    def forward(self, x, edge_index, prev_rnn_hidden_state=None):
        # x shape: [num_envs, num_agents, input_x_dim]
        # edge_index shape: [num_envs, 2, num_edges_per_graph]
        # prev_rnn_hidden_state shape: [num_envs, num_agents, hidden_rnn_dim] (from collector)

        num_envs = x.shape[0]
        num_agents = self.num_agents
        batch_size_flat = num_envs * num_agents # Flatten batch size for GCN and RNN input


        # Reshape x for GCN: [num_envs * num_agents, input_x_dim]
        x_flat = x.reshape(batch_size_flat, -1);

        # Reshape edge_index for GCN (with offset)
        edge_index_permuted = edge_index.permute(0, 2, 1);
        num_edges_per_graph = edge_index_permuted.shape[1];
        offsets = torch.arange(num_envs, device=x.device).view(-1, 1, 1) * num_agents;
        edge_index_offset = edge_index_permuted + offsets;
        edge_index_flat = edge_index_offset.reshape(-1, 2).permute(1, 0);

        # Debugging prints for edge_index reshaping (can be commented out)
        # print(f"Forecasting Module: x shape = {x.shape}, edge_index shape = {edge_index.shape}")
        # print(f"Forecasting Module: edge_index_permuted shape = {edge_index_permuted.shape}")
        # print(f"Forecasting Module: offsets shape = {offsets.shape}")
        # print(f"Forecasting Module: edge_index_offset shape = {edge_index_offset.shape}")
        # print(f"Forecasting Module: edge_index_flat shape = {edge_index_flat.shape}")
        # print(f"Forecasting Module: edge_index_flat dtype = {edge_index_flat.dtype}")


        # Apply GCN layers FIRST
        # Input: x_flat [num_envs * num_agents, input_x_dim], edge_index_flat [2, num_envs * num_edges_per_graph]
        # print(f"Forecasting Module: Passing to GCNConv.") # Debug print
        # print(f"Forecasting Module: x_flat shape = {x_flat.shape}") # Debug print
        # print(f"Forecasting Module: edge_index_flat shape = {edge_index_flat.shape}") # Debug print
        # print(f"Forecasting Module: x_flat dtype = {x_flat.dtype}, x_flat is_contiguous = {x_flat.is_contiguous()}") # Debug print
        # print(f"Forecasting Module: edge_index_flat dtype = {edge_index_flat.dtype}, edge_index_flat is_contiguous = {edge_index_flat.is_contiguous()}") # Debug print

        # Debugging print for GCNConv parameter shapes (can be commented out)
        # if hasattr(self.gcn1, 'lin') and hasattr(self.gcn1.lin, 'weight'):
        #      print(f"Forecasting Module: GCNConv internal linear weight shape = {self.gcn1.lin.weight.shape}")
        #      print(f"Forecasting Module: GCNConv internal linear weight dtype = {self.gcn1.lin.weight.dtype}")
        # else:
        #      print("Forecasting Module: GCNConv internal linear layer or weight not found directly.")

        gcn_output_flat = self.gcn1(x_flat, edge_index_flat); # Shape [num_envs * num_agents, gcn_hidden_dim]
        # print(f"Forecasting Module: GCNConv output shape = {gcn_output_flat.shape}") # Debug print


        # Reshape GCN output for RNN: [1, num_envs * num_agents, gcn_hidden_dim]
        # Sequence length = 1 (single step at a time)
        rnn_input = gcn_output_flat.reshape(1, batch_size_flat, self.gcn_hidden_dim);


        # Reshape input state from collector format [num_envs, num_agents, hidden_rnn_dim]
        # to RNN input format [1, num_envs * num_agents, hidden_rnn_dim]
        rnn_input_state = prev_rnn_hidden_state
        if rnn_input_state is not None:
             # Reshape from [num_envs, num_agents, hidden_rnn_dim] to [1, num_envs * num_agents, hidden_rnn_dim]
             rnn_input_state = rnn_input_state.reshape(1, batch_size_flat, self.hidden_rnn_dim);
        else:
             # If prev_rnn_hidden_state is None (first step), initialize with the correct shape
             # using the buffer and expanding to match the current flattened batch size.
             rnn_input_state = self._initial_rnn_hidden_state.expand(1, batch_size_flat, -1).to(x.device);


        # Pass through RNN SECOND
        # rnn_output shape: [1, num_envs * num_agents, hidden_rnn_dim]
        # next_rnn_hidden_state_rnn_format shape: [1, num_envs * num_agents, hidden_rnn_dim]
        rnn_output, next_rnn_hidden_state_rnn_format = self.rnn(rnn_input, rnn_input_state.contiguous());

        # Reshape RNN output to [num_envs, num_agents, hidden_rnn_dim] for the linear layer input
        rnn_output_reshaped = rnn_output.reshape(num_envs, num_agents, self.hidden_rnn_dim);

        # Flatten the reshaped RNN output for the linear layer
        flattened_rnn_output = rnn_output_reshaped.reshape(num_envs, -1);

        # Get the raw forecast output from the linear layer
        # Output shape: [num_envs, num_agents * input_x_dim * forecast_horizon]
        raw_forecast = self.linear(flattened_rnn_output);

        # Reshape the raw forecast output to [num_envs, num_agents, input_x_dim, forecast_horizon]
        forecast = raw_forecast.reshape(num_envs, num_agents, self.input_x_dim, self.forecast_horizon);

        # Reshape the next hidden state from RNN format [1, flat_batch, hidden_dim]
        # back to collector format [num_envs, num_agents, hidden_rnn_dim]
        next_rnn_hidden_state = next_rnn_hidden_state_rnn_format.reshape(num_envs, num_agents, self.hidden_rnn_dim);


        # The TensorDictModule expects the output as a tuple matching the out_keys.
        # The forecasting module outputs the forecast and the next RNN hidden state.
        return forecast, next_rnn_hidden_state


# Define the custom combined module as nn.Module with corrected initial state handling
class CombinedPolicyForecastingBase(nn.Module):
    """
    A base nn.Module to combine policy and forecasting networks' logic,
    handling RNN hidden states internally and using reshape.
    Corrects initial state handling for collector.
    """
    def __init__(self, policy_module_base, forecasting_module_base, hidden_rnn_dim, num_agents, device):
        super().__init__()
        self.policy_module_base = policy_module_base
        self.forecasting_module_base = forecasting_module_base
        self.hidden_rnn_dim = hidden_rnn_dim
        self.num_agents = num_agents
        self.device = device

    # Corrected forward signature to accept inputs as separate arguments
    def forward(self, x, edge_index, prev_policy_rnn_hidden_state=None, prev_forecast_rnn_hidden_state=None):
        # x shape: [num_envs, num_agents, input_x_dim]
        # edge_index shape: [num_envs, 2, num_edges_per_graph]
        # State shapes: [num_envs, num_agents, hidden_rnn_dim] as expected by collector in tensordict

        num_envs = x.shape[0]
        batch_size_flat = num_envs * self.num_agents # Flatten batch size for RNN input


        # Need to reshape input state from collector format [num_envs, num_agents, hidden_rnn_dim]
        # to RNN input format [1, num_envs * num_agents, hidden_rnn_dim]
        # Handle initial None state: If None, the base policy/forecasting modules
        # will handle creating the zero state in the correct RNN format.
        # If not None, reshape the state provided by the collector.

        # Note: The base modules' forward methods handle the initial state if None is passed.
        # We pass the collector-provided state here, which might be None on the very first step.
        # The base modules will then correctly initialize the RNN if the input state is None.
        # So no need to reshape prev_rnn_hidden_state to RNN format *before* passing to base modules.
        # Pass the collector-provided state directly.

        # Run policy module base
        # policy_module_base forward expects (x, edge_index, prev_policy_rnn_hidden_state in collector format or None)
        action_logits, next_policy_rnn_hidden_state = self.policy_module_base(x, edge_index, prev_policy_rnn_hidden_state);


        # Run forecasting module base
        # forecasting_module_base forward expects (x, edge_index, prev_forecast_rnn_hidden_state in collector format or None)
        forecast, next_forecast_rnn_hidden_state = self.forecasting_module_base(x, edge_index, prev_forecast_rnn_hidden_state);


        # The TensorDictModule expects the output as a tuple matching the out_keys.
        # The combined module outputs policy logits, next policy hidden state, forecast, and next forecasting hidden state.
        # The base modules are designed to output next hidden states in collector format [num_envs, num_agents, hidden_rnn_dim].
        return action_logits, next_policy_rnn_hidden_state, forecast, next_forecast_rnn_hidden_state


# Instantiate the base policy, value, and forecasting nn.Modules
# Ensure env, device, num_agents, forecast_horizon are defined (assuming previous cells ran)
try:
    _ = env
    _ = device
    _ = num_agents
    _ = forecast_horizon
except NameError:
    print("Environment or essential variables not defined. Please run previous cells.")
    # Set essential variables to None or default if not defined to prevent errors
    env = None
    device = 'cpu'
    num_agents = 13
    forecast_horizon = 5
    print("Setting essential variables to default/None due to missing dependencies.")


if env is not None:
    input_x_dim = env.node_feature_dim
    num_agents = env.num_agents
    num_individual_actions_features = env.num_individual_actions_features
    # Get num_action_categories from the action_spec as before
    action_spec = env.action_spec['agents', 'action']
    if hasattr(action_spec, 'nvec'):
        unique_categories = torch.unique(action_spec.nvec)
        if len(unique_categories) == 1:
            num_action_categories = unique_categories.item()
        else:
             raise ValueError(f"Expected all action categories to have the same number of options, but found {unique_categories}")
    else:
         raise ValueError("Action spec does not have a 'nvec' attribute to determine number of categories.")

    hidden_rnn_dim = 64
    gcn_hidden_dim = 64
    forecast_horizon = 5

    # Instantiate the base nn.Modules (using corrected GCN definitions)
    print("\nInstantiating Base nn.Modules...")
    try:
        policy_module_gcn = SimpleMultiAgentPolicyModuleGCN(
            input_x_dim=input_x_dim,
            num_agents=num_agents,
            num_individual_actions_features=num_individual_actions_features,
            num_action_categories=num_action_categories,
            hidden_rnn_dim=hidden_rnn_dim,
            gcn_hidden_dim=gcn_hidden_dim
        ).to(device)

        value_module_gcn = SimpleMultiAgentValueModuleGCN(
            input_x_dim=input_x_dim,
            num_agents=num_agents,
            hidden_rnn_dim=hidden_rnn_dim,
            gcn_hidden_dim=gcn_hidden_dim
        ).to(device)

        forecasting_module_gcn = SimpleMultiAgentForecastingModuleGCN(
            input_x_dim=input_x_dim,
            num_agents=num_agents,
            forecast_horizon=forecast_horizon,
            hidden_rnn_dim=hidden_rnn_dim,
            gcn_hidden_dim=gcn_hidden_dim
        ).to(device)
        print("Base nn.Modules instantiated successfully.")


        # Instantiate the custom combined base module
        print("\nInstantiating Custom Combined Base nn.Module...")
        combined_module_base_gcn = CombinedPolicyForecastingBase(
            policy_module_base=policy_module_gcn, # Pass the GCN base nn.Module
            forecasting_module_base=forecasting_module_gcn, # Pass the GCN base nn.Module
            hidden_rnn_dim=hidden_rnn_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        print("Custom Combined Base nn.Module instantiated successfully.")

        # Now wrap the base modules in TensorDictModules
        # Define the input and output keys for the wrapped module
        # Input keys correspond to the arguments of the base module's forward method.
        # Output keys correspond to the tuple returned by the base module's forward method.
        print("\nWrapping Base Modules in TensorDictModules...")
        combined_module_wrapped_gcn = TensorDictModule(
            module=combined_module_base_gcn,
            in_keys=[('agents', 'data', 'x'), # Corresponds to x arg
                     ('agents', 'data', 'edge_index'), # Corresponds to edge_index arg
                     ('agents', 'rnn_hidden_state'), # Corresponds to prev_policy_rnn_hidden_state arg
                     ('agents', 'rnn_hidden_state_forecast')], # Corresponds to prev_forecast_rnn_hidden_state arg
            out_keys=[('agents', 'action'), # Output 0 from base module (logits)
                      ('agents', 'rnn_hidden_state'), # Output 1 from base module (next policy hidden state)
                      'forecast', # Output 2 from base module (forecast)
                      ('agents', 'rnn_hidden_state_forecast')] # Output 3 from base module (next forecasting hidden state)
        ).to(device)

        # Also wrap the value module in a TensorDictModule
        # Input keys correspond to the arguments of the base value module's forward method.
        # Output keys correspond to the tuple returned by the base value module's forward method.
        value_net_gcn = TensorDictModule(
            module=value_module_gcn,
            in_keys=[('agents', 'data', 'x'), # Corresponds to x arg
                     ('agents', 'data', 'edge_index'), # Corresponds to edge_index arg
                     ('agents', 'rnn_hidden_state_value')], # Corresponds to prev_rnn_hidden_state arg
            out_keys=['value', # Output 0 from base module (value)
                      ('agents', 'rnn_hidden_state_value')] # Output 1 from base module (next value hidden state)
        ).to(device)

        print("TensorDictModules wrapped successfully.")

        print("\nVerification of in_keys and out_keys:")
        print(f"combined_module_wrapped_gcn in_keys: {combined_module_wrapped_gcn.in_keys}")
        print(f"combined_module_wrapped_gcn out_keys: {combined_module_wrapped_gcn.out_keys}")
        print(f"value_net_gcn in_keys: {value_net_gcn.in_keys}")
        print(f"value_net_gcn out_keys: {value_net_gcn.out_keys}")


    except Exception as e:
        print(f"\nAn error occurred during module instantiation or wrapping: {e}")
        policy_module_gcn = None
        value_module_gcn = None
        forecasting_module_gcn = None
        combined_module_base_gcn = None
        combined_module_wrapped_gcn = None
        value_net_gcn = None

else:
    policy_module_gcn = None
    value_module_gcn = None
    forecasting_module_gcn = None
    combined_module_base_gcn = None
    combined_module_wrapped_gcn = None
    value_net_gcn = None
    print("\nEnvironment is None. Cannot instantiate modules.")


# Define the gradient hook function (re-defining for self-containment)
def log_grad(name):
    def hook(grad):
        if grad is not None:
            grad_norm = torch.linalg.norm(grad.cpu())
            # print(f"Gradient norm for {name}: {grad_norm.item()}") # Commented out to reduce output
        else:
            pass # print(f"Gradient is None for {name}") # Commented out


    # Return the hook function
    return hook

# Check if the modules were successfully instantiated
if combined_module_base_gcn is not None and value_net_gcn is not None:
    print("\nRegistering gradient hooks...")

    # List to store hook handles
    hook_handles = []

    def register_module_hooks(module, name_prefix):
        # print(f"Attempting to register hooks for module {name_prefix}...") # Debug print
        if hasattr(module, 'named_parameters'):
            module_params = list(module.named_parameters())
            if len(module_params) > 0:
                for param_name, param in module_params:
                     if param.requires_grad:
                          hook_handles.append(param.register_hook(log_grad(f"{name_prefix}.{param_name}")))
            else:
                # print(f"No parameters found in module {name_prefix}.") # Debug print
                pass
        else:
            # print(f"Module {name_prefix} does not have named_parameters().") # Debug print
            pass

    # Register hooks for the parameters of the base modules within the wrapped modules
    if hasattr(combined_module_base_gcn, 'policy_module_base') and combined_module_base_gcn.policy_module_base is not None:
        register_module_hooks(combined_module_base_gcn.policy_module_base, "policy_module_base")
    if hasattr(combined_module_base_gcn, 'forecasting_module_base') and combined_module_base_gcn.forecasting_module_base is not None:
        register_module_hooks(combined_module_base_gcn.forecasting_module_base, "forecasting_module_base")
    if hasattr(value_net_gcn, 'module') and value_net_gcn.module is not None:
        register_module_hooks(value_net_gcn.module, "value_module_base")

    print(f"Registered {len(hook_handles)} gradient hooks.")
else:
    print("Modules not available. Skipping hook registration.")


# --- Test Forward and Backward Pass ---
# This section tests if the modules can process data and if gradients flow.
# It's included here for verification after refactoring the module definitions.

if combined_module_wrapped_gcn is not None and value_net_gcn is not None and env is not None:
    print("\nPerforming test forward and backward pass...")

    try:
        # Get an initial observation from the environment
        # Use a collector to get data that includes initial states and will handle
        # the initial hidden states for the recurrent layers.
        # We need to create a temporary collector for just one step.
        print("Collecting initial data from environment...")
        temp_collector = SyncDataCollector(
            env,
            combined_module_wrapped_gcn, # Pass the policy/forecasting module for action sampling
            frames_per_batch=1, # Collect just one frame/step
            max_frames_per_traj=-1, # No trajectory limit
            device=device,
            # Add required recurrent state keys to the collector's state_spec
            # These should match the keys expected by the combined module's forward method
            # and present in the environment's state_spec.
            # The collector will automatically add and manage these if they are in the env.state_spec.
        )

        # Reset the collector to get the initial state and collect the first step
        # The initial_tensordict from reset will contain the initial observation and initial RNN states.
        initial_tensordict = temp_collector.reset()


        # Collect one step of data using the collector
        # This will run the forward pass of combined_module_wrapped_gcn to sample an action
        # and then call env.step() with that action.
        # The collected tensordict will contain the 'next' state, reward, and done flags,
        # as well as the original 'observation' (at t=0) and the 'action' sampled (at t=0).
        print("Collecting one step using the temporary collector...")
        # The rollout method of SyncDataCollector returns a tensordict of shape [T, B]
        collected_tensordict = temp_collector.rollout(1) # Collect 1 step


        print("\nCollected Tensordict structure after one step:")
        print(collected_tensordict.keys(include_nested=True))
        print(f"Shape of collected tensordict: {collected_tensordict.shape}")

        # Extract relevant data for the backward pass
        # We need the value estimates from the value network for the collected states
        # and a dummy loss based on these value estimates or the reward.
        # For a simple test, let's use the value estimates and backpropagate a dummy loss.

        # The collected_tensordict has shape [T, B] where T=1 and B=num_envs.
        # The value estimates are expected to be under the 'value' key after the value_net_gcn forward pass.
        # We need to run the value_net_gcn forward pass on the collected states.
        # Let's use the state at T=0 from the collected tensordict for this simple test.
        # The value network needs ('agents', 'data', 'x'), ('agents', 'data', 'edge_index'), and ('agents', 'rnn_hidden_state_value').
        # These should be available at T=0 in the collected_tensordict.

        # Select the data at T=0 (the only step) from the collected tensordict for the value network input.
        states_at_t0 = collected_tensordict[0] # Shape: [B]

        print(f"\nInput tensordict for value network (shape {states_at_t0.shape}):")
        print(states_at_t0.keys(include_nested=True))

        # Pass the states through the value network
        # The value network will output 'value' and update 'rnn_hidden_state_value' in the tensordict.
        print("\nRunning value network forward pass...")
        # Select only the required keys from states_at_t0 and clone to avoid modifying original data
        value_input_td = states_at_t0.select(*value_net_gcn.in_keys).clone()
        value_output_td = value_net_gcn(value_input_td)

        print(f"Value network output tensordict keys: {value_output_td.keys(include_nested=True)}")
        print(f"Shape of value output tensordict: {value_output_td.shape}")
        print(f"Value estimates shape: {value_output_td.get('value').shape}")

        # Create a dummy loss for backpropagation
        # A simple loss could be based on the sum of values.
        # This loss should be differentiable with respect to the value network parameters.
        dummy_value_loss = value_output_td.get('value').sum()

        print(f"\nDummy value loss: {dummy_value_loss.item()}")

        # Perform backward pass for the value network
        print("Performing backward pass for value network...")
        value_net_gcn.zero_grad() # Ensure gradients are zeroed before backward
        dummy_value_loss.backward()
        print("Backward pass for value network complete.")

        # Check if any gradients were computed for value network parameters
        print("\nChecking gradients for Value Network after backward pass:")
        for name, param in value_net_gcn.module.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    print(f"  Gradient exists for {name}. Norm: {torch.linalg.norm(param.grad.cpu()).item()}")
                else:
                    # This might happen if the parameter was not part of the computation graph that led to the loss
                    print(f"  Gradient is None for {name}.")
            else:
                 print(f"  {name} does not require gradients.")


        # Now, let's do a similar test for the combined policy/forecasting module.
        # This module outputs action_logits, policy hidden state, forecast, and forecast hidden state.
        # We need a dummy loss that depends on these outputs to trigger backprop.
        # For policy, we could use the sum of logits. For forecasting, the sum of forecasts.
        # The combined_module_wrapped_gcn was already run during temp_collector.rollout(1).
        # Its outputs should be available in the collected_tensordict under the keys
        # specified in the combined_module_wrapped_gcn's out_keys.

        print(f"\nPolicy/Forecasting module outputs in collected_tensordict:")
        print(collected_tensordict.keys(include_nested=True))

        # The combined_module_wrapped_gcn was run on the state at T=0.
        # Its outputs are stored in the collected_tensordict at T=0.
        # The first output of combined_module_wrapped_gcn is action_logits, mapped to ('agents', 'action').
        # The second output is next_policy_rnn_hidden_state, mapped to ('agents', 'rnn_hidden_state').
        # The third output is forecast, mapped to 'forecast'.
        # The fourth output is next_forecast_rnn_hidden_state, mapped to ('agents', 'rnn_hidden_state_forecast').

        # Create dummy losses for backpropagation
        # Dummy policy loss based on sum of action logits (which are stored under ('agents', 'action') by the TensorDictModule)
        # Note: In a real RL setup, you'd use a proper policy loss (e.g., negative log probability).
        # We are using sum() here just to create a differentiable path for testing gradients.
        dummy_policy_loss = collected_tensordict[0].get(('agents', 'action')).sum()

        # Dummy forecasting loss based on sum of forecasts
        # The forecast is stored under the 'forecast' key in the collected_tensordict at T=0.
        dummy_forecasting_loss = collected_tensordict[0].get('forecast').sum()


        print(f"\nDummy policy loss: {dummy_policy_loss.item()}")
        print(f"Dummy forecasting loss: {dummy_forecasting_loss.item()}")


        # Perform backward pass for the combined module
        # We can sum the losses and backpropagate through both simultaneously.
        total_dummy_loss = dummy_policy_loss + dummy_forecasting_loss

        print("Performing backward pass for combined policy/forecasting module...")
        combined_module_wrapped_gcn.zero_grad() # Ensure gradients are zeroed before backward
        total_dummy_loss.backward()
        print("Backward pass for combined policy/forecasting module complete.")


        # Check if any gradients were computed for combined module parameters
        print("\nChecking gradients for Combined Policy/Forecasting Module after backward pass:")
        # Iterate through the parameters of the base modules within the wrapped module
        print("  Policy Module Parameters:")
        for name, param in combined_module_base_gcn.policy_module_base.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    print(f"    Gradient exists for {name}. Norm: {torch.linalg.norm(param.grad.cpu()).item()}")
                else:
                    print(f"    Gradient is None for {name}.")
            else:
                 print(f"    {name} does not require gradients.")

        print("  Forecasting Module Parameters:")
        for name, param in combined_module_base_gcn.forecasting_module_base.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    print(f"    Gradient exists for {name}. Norm: {torch.linalg.norm(param.grad.cpu()).item()}")
                else:
                    print(f"    Gradient is None for {name}.")
            else:
                 print(f"    {name} does not require gradients.")


        # You can now examine the output logs from the registered hooks during the backward passes.

        print("\nTest forward and backward pass complete. Examine the console output for gradient norms.")


    except Exception as e:
        print(f"\nAn error occurred during the test forward/backward pass: {e}")
        print("Could not complete the test pass.")

else:
    print("\nCombined module, value module, or environment not available. Skipping test forward/backward pass.")

# hook_handles list will contain the handles if registration was successful.
# You can now proceed with a test forward and backward pass in the next step
# and then use a similar code block to remove the hooks using hook_handles.
