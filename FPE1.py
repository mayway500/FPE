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
        self.episode_length = kwargs.get('episode_length', 10000)
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
        print("AnFuelpriceEnv._step: Entering _step method.") # Debug print
        print(f"AnFuelpriceEnv._step: Input tensordict structure: {tensordict.keys(include_nested=True)}") # Debug print
        print(f"AnFuelpriceEnv._step: Input tensordict batch size: {tensordict.batch_size}") # Debug print
        print(f"AnFuelpriceEnv._step: Current data index before increment: {self.current_data_index}") # Debug print


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
        print(f"AnFuelpriceEnv._step: Action extracted. Shape: {action.shape if action is not None else 'None'}") # Debug print


        terminated = self._is_terminal()
        truncated = (self.current_data_index >= self.episode_length)
        print(f"AnFuelpriceEnv._step: Done flags calculated. Terminated: {terminated}, Truncated: {truncated}") # Debug print


        # The action is now expected to be in the input tensordict passed to step() by the collector
        # Pass the input tensordict directly to _batch_reward
        # _batch_reward will need to extract the action from this tensordict
        # Get the reward tensordict from _batch_reward
        print("AnFuelpriceEnv._step: Calling _batch_reward...") # Debug print
        reward_td = self._batch_reward(self.current_data_index, tensordict) # Pass the input tensordict here
        print(f"AnFuelpriceEnv._step: _batch_reward returned. Reward tensordict structure: {reward_td.keys(include_nested=True)}") # Debug print


        # Increment the data index for the next step BEFORE calculating the next state
        # The next state should correspond to the data at the *next* time step.
        self.current_data_index += 1
        print(f"AnFuelpriceEnv._step: current_data_index incremented to {self.current_data_index}") # Debug print


        # Logic previously in _get_state_at for next state

        num_envs = self.current_data_index.shape[0]

        # Get data indices for the next step, handling boundaries
        # Use the current_data_index (which is now the index for the next step) for the next state's data
        data_indices_for_next_state = self.current_data_index
        # Clamp data indices to prevent out-of-bounds access, especially if episode_length is near data end
        # Ensure we don't go beyond the last available data point index
        # Check if combined_data is None before accessing shape
        if self.combined_data is not None:
             data_indices_for_next_state = torch.min(data_indices_for_next_state, torch.as_tensor(self.combined_data.shape[0] - 1, device=self.device))
             print(f"AnFuelpriceEnv._step: Data indices for next state (clamped): {data_indices_for_next_state}") # Debug print
        else:
             print("AnFuelpriceEnv._step: Error: combined_data is None when calculating next state data index.") # Debug print
             # Handle this error case, maybe return a terminal state immediately?
             # For now, let's proceed with clamped index assuming the error was caught earlier.
             # This might still lead to errors if combined_data is truly None.
             pass # The check at the beginning of _step should handle this.


        # Extract the first node_feature_dim columns for each environment
        # If all agents share the same features, extract and then expand
        # Check combined_data again before indexing
        if self.combined_data is not None:
            print(f"AnFuelpriceEnv._step: Accessing self.combined_data with indices {data_indices_for_next_state}") # Debug print
            x_data_time_step = self.combined_data[data_indices_for_next_state, :self.node_feature_dim] # Shape: [num_envs, node_feature_dim]
            # Expand to [num_envs, num_agents, node_feature_dim]
            x_data = x_data_time_step.unsqueeze(1).expand(-1, self.num_agents, -1)
            print(f"AnFuelpriceEnv._step: x_data for next state shape: {x_data.shape}") # Debug print

        else:
            print("AnFuelpriceEnv._step: Error: combined_data is None when extracting x_data for next state.") # Debug print
            # Return dummy data if combined_data is None
            x_data = torch.zeros(num_envs, self.num_agents, self.node_feature_dim, dtype=torch.float32, device=self.device)
            print("AnFuelpriceEnv._step: Using dummy x_data for next state.") # Debug print


        # Use fixed edge index repeated for the batch, only if there are edges
        if self._fixed_num_edges_single > 0:
            edge_index_data = self._fixed_edge_index_single.unsqueeze(0).repeat(num_envs, 1, 1).to(self.device)
        else:
            # Create an empty edge index tensor with the correct batch size
            edge_index_data = torch.empty(num_envs, 2, 0, dtype=torch.int64, device=self.device)

        print(f"AnFuelpriceEnv._step: Generated edge_index_data shape: {edge_index_data.shape}") # Debug print


        next_state_tensordict_data = TensorDict({
             "x": x_data, # Use actual data for node features
             "edge_index": edge_index_data, # Use the fixed edge indices
             "batch": torch.arange(num_envs, device=self.device).repeat_interleave(self.num_agents).view(num_envs, self.num_agents), # Create batch tensor
             "time": self.current_data_index.unsqueeze(-1).to(self.device), # Use the current_data_index as timestamp for the next state
        }, batch_size=[num_envs], device=self.device)

        print("AnFuelpriceEnv._step: Populated next observation data tensordict ('agents', 'data'). Structure:") # Diagnostic print
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
                 # These should match the structure expected by the state_spec
                 ('agents', 'rnn_hidden_state'): torch.zeros(num_envs, self.num_agents, 64, dtype=torch.float32, device=self.device),
                 ('agents', 'rnn_hidden_state_forecast'): torch.zeros(num_envs, self.num_agents, 64, dtype=torch.float32, device=self.device),
                 ('agents', 'rnn_hidden_state_value'): torch.zeros(num_envs, self.num_agents, 64, dtype=torch.float32, device=self.device),
            }),
            # Include the done flags at the root level
            "terminated": terminated.to(self.device), # Shape [num_envs]
            "truncated": truncated.to(self.device), # Shape [num_envs]
            "done": (terminated | truncated).to(self.device), # Shape [num_envs]

            # Include the current observation data in the root level for the collector to use
            # This is what the collector needs to prepare the input for the *next* step's policy/value forward pass.
            # The structure here should match the state_spec's expectation for the current state.
            ("agents", "data"): tensordict.get(("agents", "data")), # Get the current observation data from the input tensordict
             ('agents', 'rnn_hidden_state'): tensordict.get(('agents', 'rnn_hidden_state')), # Get current RNN states from input
             ('agents', 'rnn_hidden_state_forecast'): tensordict.get(('agents', 'rnn_hidden_state_forecast')),
             ('agents', 'rnn_hidden_state_value'): tensordict.get(('agents', 'rnn_hidden_state_value')),


        }, batch_size=[self.num_envs], device=self.device)

        print("AnFuelpriceEnv._step: Full output tensordict structure:") # Diagnostic print
        print(output_tensordict) # Diagnostic print


        return output_tensordict

    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        # Check if combined_data is available
        if self.combined_data is None:
            print("AnFuelpriceEnv._reset: Error: combined_data is None. Data loading failed during environment initialization.")
            raise RuntimeError("Cannot reset environment: Data loading failed during environment initialization.")

        num_envs = self.num_envs # Get num_envs from self for clarity

        # Ensure edge_index_data is always initialized at the beginning of the method
        edge_index_data = torch.empty(num_envs, 2, 0, dtype=torch.int64, device=self.device)
        print(f"_reset: Initialized edge_index_data shape = {edge_index_data.shape}")


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

        data_indices = self.current_data_index
        # Clamp data indices to prevent out-of-bounds access, especially if the start index is near the end
        data_indices = torch.min(data_indices, torch.as_tensor(self.combined_data.shape[0] - 1, device=self.device))

        # Use fixed edge index repeated for the batch, only if there are edges
        # _fixed_edge_index_single has shape [2, num_edges_per_graph]
        # Unsqueeze to [1, 2, num_edges_per_graph], repeat to [num_envs, 2, num_edges_per_graph]
        if self._fixed_num_edges_single > 0:
            edge_index_data = self._fixed_edge_index_single.unsqueeze(0).repeat(num_envs, 1, 1).to(self.device)
            print(f"_reset: Fixed edge index used. Updated edge_index_data shape = {edge_index_data.shape}")


        # Modified: Ensure x_data has shape [num_envs, num_agents, node_feature_dim]
        # Extract node features (first node_feature_dim columns) from combined_data at data_indices
        x_data_time_step = self.combined_data[data_indices, :self.node_feature_dim] # Shape [num_envs, node_feature_dim]
        # Expand to match the number of agents: [num_envs, 1, node_feature_dim] -> [num_envs, num_agents, node_feature_dim]
        x_data = x_data_time_step.unsqueeze(1).expand(-1, self.num_agents, -1)


        # Create the initial observation tensordict structure
        initial_observation_data_td = TensorDict({
             "x": x_data, # Use actual data for node features
             "edge_index": edge_index_data, # Use the fixed edge indices
             # Create a batch tensor for torch_geometric: [num_envs * num_agents] where each agent in env i has batch index i
             # Reshape to [num_envs, num_agents] to match the structure expected by the policy/value/forecasting modules
             "batch": torch.arange(num_envs, device=self.device).repeat_interleave(self.num_agents).view(num_envs, self.num_agents),
             "time": self.current_data_index.unsqueeze(-1).to(self.device), # Use the current_data_index as timestamp
        }, batch_size=[num_envs], device=self.device)

        # Add print statement for initial_observation_data_td
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

        # Add print statement for the final output_tensordict
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
         # Sequence length = 1 (single step)
