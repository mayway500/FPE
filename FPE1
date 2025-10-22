import torch.nn as nn
import pandas as pd
import os
import networkx as nx
import random
import torch
import torch_geometric
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool # Import global_mean_pool
from torch_geometric.nn import SplineConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
# Import MultiCategorical specification class from torchrl.data
from torchrl.data import Unbounded, Categorical, Composite, DiscreteTensorSpec, MultiCategorical as MultiCategoricalSpec # Import Composite, DiscreteTensorSpec, and rename MultiCategorical
from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    CompositeSpec as DataCompositeSpec, # Rename to avoid conflict with torchrl.envs.CompositeSpec
    UnboundedDiscreteTensorSpec,
    DiscreteTensorSpec as DataDiscreteTensorSpec, # Rename to conflict
    MultiDiscreteTensorSpec as DataMultiDiscreteTensorSpec, # Rename to conflict
    BoxList as DataBoxList, # Rename to conflict
    CategoricalBox as DataCategoricalBox, # Rename to conflict
    UnboundedContinuous as DataUnboundedContinuous # Rename to conflict
)

from tensordict.nn import TensorDictModule # Import TensorDictModule


from typing import Optional
import numpy as np
import torch # Import torch
import torch.nn as nn # Explicitly import torch.nn again here
from tensordict.nn import TensorDictModuleBase # Import TensorDictModuleBase
from torchrl.envs.utils import check_env_specs # Keep check_env_specs


# Define the FuelpriceenvfeatureGraph class (from 6ryKoyyHkPbi)
class FuelpriceenvfeatureGraph():

    def __init__(self, device='cpu', allow_repeat_data=False): # Added device and allow_repeat_data to init
        self.graph = nx.DiGraph()
        self.graph.graph["graph_attr_1"] = random.random() * 10
        self.graph.graph["graph_attr_2"] = random.random() * 5.
        self.num_nodes_per_graph = 13 # Initialize here
        self.device = device # Store device
        self.allow_repeat_data = allow_repeat_data # Store allow_repeat_data
        self.combined_data = None # Initialize combined_data to None
        self.node_feature_dim = 0 # Initialize node_feature_dim here


    def _load_data(self):
        print("FuelpriceenvfeatureGraph._load_data: Starting data loading.")
        from scipy.stats import zscore
        import pandas as pd
        import numpy as np
        import os # Import os
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True) # Added force_remount=True

        # Update this path with the correct location of your CSV file
        local_data_path = '/content/drive/MyDrive/deep learning codes/EIAAPI_DOWNLOAD/solutions/mergedata/Cleaneddata.csv' # Corrected path to a local file in Google Drive
        if os.path.exists(local_data_path):
            print(f"FuelpriceenvfeatureGraph._load_data: Attempting to load data from {local_data_path}")
            try:
                print("FuelpriceenvfeatureGraph._load_data: Reading CSV...")
                # Removed the 'names' argument and rely on 'header=0' to read existing column names
                dffff = pd.read_csv(
                    local_data_path,
                    header=0,
                    parse_dates=["('Date',)"]
                )
                print(f"FuelpriceenvfeatureGraph._load_data: Columns after initial read: {dffff.columns.tolist()}") # Debug print
                print(f"FuelpriceenvfeatureGraph._load_data: DataFrame shape after initial read: {dffff.shape}") # Debug print


                print("FuelpriceenvfeatureGraph._load_data: Setting index and handling NaNs...")
                # Corrected line to set the index using the correct column name
                dffff = dffff.set_index("('Date',)")
                dffff = dffff.ffill()
                dffff.dropna(axis=0, how='any', inplace=True)
                print("FuelpriceenvfeatureGraph._load_data: NaN handling complete.")
                print(f"FuelpriceenvfeatureGraph._load_data: DataFrame shape after NaN handling: {dffff.shape}") # Debug print


                # Simplify feature column identification: include any numeric column except the date index
                feature_columns = [col for col in dffff.columns if dffff[col].dtype in [np.number]]


                print(f"FuelpriceenvfeatureGraph._load_data: DataFrame columns before filtering: {dffff.columns.tolist()}") # Debug print
                print(f"FuelpriceenvfeatureGraph._load_data: Identified feature columns: {feature_columns}") # Debug print

                # Ensure features_df is created even if feature_columns is empty
                if feature_columns:
                     features_df = dffff[feature_columns]
                else:
                     features_df = pd.DataFrame() # Create an empty DataFrame if no features found

                print(f"FuelpriceenvfeatureGraph._load_data: Shape of features_df after filtering: {features_df.shape}") # Debug print


                # Check if features_df is empty or has unexpected structure
                if features_df.empty or features_df.shape[1] == 0:
                    print("FuelpriceenvfeatureGraph._load_data: Error: No numeric feature columns found after filtering. Ensure your CSV has numeric data besides the date column.")
                    self.combined_data = None
                    self.node_feature_dim = 0
                    return None # Return None immediately on error
                    # Removed the raise here to prevent it from being caught by the outer except

                numberr_np = features_df.values
                print(f"FuelpriceenvfeatureGraph._load_data: Shape of numberr_np: {numberr_np.shape}") # Debug print
                if numberr_np.shape[0] < 2:
                     print("FuelpriceenvfeatureGraph._load_data: Error: Insufficient data points to calculate returns and actions (need at least 2 rows).")
                     self.combined_data = None
                     self.node_feature_dim = 0
                     return None

                print("FuelpriceenvfeatureGraph._load_data: Calculating returns and actions...")
                def returns(x):
                  x = np.array(x)
                  return x[1:, :] - x[:-1, :]
                RRRR = returns(numberr_np)
                print(f"FuelpriceenvfeatureGraph._load_data: Shape of RRRR (returns): {RRRR.shape}") # Debug print

                def actionspace(x):
                  x = np.array(x)
                  differences = x[1:, :] - x[:-1, :]
                  yxx = np.zeros_like(differences)
                  yxx[differences > 0] = 2
                  yxx[differences < 0] = 0
                  yxx[differences == 0] = 1
                  return yxx
                action = actionspace(numberr_np)
                print(f"FuelpriceenvfeatureGraph._load_data: Shape of action: {action.shape}") # Debug print
                print("FuelpriceenvfeatureGraph._load_data: Returns and actions calculation complete.")


                # Check shapes before hstack
                print(f"FuelpriceenvfeatureGraph._load_data: Shape of RRRR before hstack: {RRRR.shape}")
                print(f"FuelpriceenvfeatureGraph._load_data: Shape of action before hstack: {action.shape}")
                if RRRR.shape[0] != action.shape[0]:
                    print("FuelpriceenvfeatureGraph._load_data: Error: Mismatch in number of rows between returns and actions.")
                    self.combined_data = None
                    self.node_feature_dim = 0
                    return None

                print("FuelpriceenvfeatureGraph._load_data: Stacking returns and actions...")
                Indep = np.hstack((RRRR, action))
                print(f"FuelpriceenvfeatureGraph._load_data: Shape of Indep (returns and actions stacked): {Indep.shape}") # Debug print
                print("FuelpriceenvfeatureGraph._load_data: Stacking complete.")


                features_aligned_np = numberr_np[1:, :]
                print(f"FuelpriceenvfeatureGraph._load_data: Shape of features_aligned_np: {features_aligned_np.shape}") # Debug print

                # Check shapes before final hstack
                print(f"FuelpriceenvfeatureGraph._load_data: Shape of features_aligned_np before final hstack: {features_aligned_np.shape}")
                print(f"FuelpriceenvfeatureGraph._load_data: Shape of Indep before final hstack: {Indep.shape}")
                if features_aligned_np.shape[0] != Indep.shape[0]:
                    print("FuelpriceenvfeatureGraph._load_data: Error: Mismatch in number of rows between features_aligned_np and Indep.")
                    self.combined_data = None
                    self.node_feature_dim = 0
                    return None

                # Perform the final hstack only if shapes are compatible and not empty
                if features_aligned_np.shape[0] > 0 and Indep.shape[0] > 0 and features_aligned_np.shape[1] + Indep.shape[1] > 0:
                     print("FuelpriceenvfeatureGraph._load_data: Performing final hstack...")
                     self.combined_data = np.hstack([features_aligned_np, Indep])
                     print(f"FuelpriceenvfeatureGraph._load_data: Shape of combined_data after numpy hstack: {self.combined_data.shape}") # Debug print
                     print("FuelpriceenvfeatureGraph._load_data: Final hstack complete.")
                else:
                     print("FuelpriceenvfeatureGraph._load_data: Error: Resulting numpy arrays are empty or have zero total features. Cannot form combined_data.")
                     self.combined_data = None
                     self.node_feature_dim = 0
                     return None


                if np.isnan(self.combined_data).any() or np.isinf(self.combined_data).any():
                    print("FuelpriceenvfeatureGraph._load_data: Warning: NaNs or Infs found in combined_data numpy array before tensor conversion. Attempting to clean.")
                    self.combined_data[np.isnan(self.combined_data)] = 0.0
                    self.combined_data[np.isinf(self.combined_data)] = 0.0
                    if np.isnan(self.combined_data).any() or np.isinf(self.combined_data).any():
                         print("FuelpriceenvfeatureGraph._load_data: Error: NaNs or Infs still present after cleaning.")
                         raise ValueError("Invalid values in combined_data after numpy cleaning.")
                    else:
                         print("FuelpriceenvfeatureGraph._load_data: NaNs and Infs successfully replaced in numpy array.")

                print(f"FuelpriceenvfeatureGraph._load_data: Shape of combined_data after replacement (numpy): {self.combined_data.shape}")
                print(f"FuelpriceenvfeatureGraph._load_data: Attempting to convert combined_data to torch tensor on device {self.device}...")
                try:
                    self.combined_data = torch.as_tensor(self.combined_data, dtype=torch.float32, device=self.device)
                    print("FuelpriceenvfeatureGraph._load_data: Conversion to torch tensor successful.")
                except Exception as e:
                    print(f"FuelpriceenvfeatureGraph._load_data: Error converting data to torch tensor: {e}")
                    raise e

                # Corrected the isinf check
                if torch.isnan(self.combined_data).any() or torch.isinf(self.combined_data).any():
                     print("FuelpriceenvfeatureGraph._load_data: Error: NaNs or Infs found in the PyTorch tensor after conversion.")
                     raise ValueError("Invalid values in PyTorch tensor after conversion.")
                else:
                     print("FuelpriceenvfeatureGraph._load_data: No NaNs or Infs found in the PyTorch tensor.")

                # Ensure node_feature_dim is set based on the actual loaded features
                self.node_feature_dim = features_df.shape[1]
                print(f"FuelpriceenvfeatureGraph._load_data: Set node_feature_dim to: {self.node_feature_dim}")

                if self.combined_data is not None and self.combined_data.shape[0] > 0 and self.node_feature_dim > 0: # Added checks for shape and dim
                    self.obs_min = torch.min(self.combined_data[:, :self.node_feature_dim], dim=0)[0].to(self.device)
                    self.obs_max = torch.max(self.combined_data[:, :self.node_feature_dim], dim=0)[0].to(self.device)
                    print("FuelpriceenvfeatureGraph._load_data: Calculated observation bounds.")
                    print(f"FuelpriceenvfeatureGraph._load_data: obs_min shape: {self.obs_min.shape}, obs_max shape: {self.obs_max.shape}") # Debug print
                else:
                    self.obs_min = None
                    self.obs_max = None
                    print("FuelpriceenvfeatureGraph._load_data: combined_data is None, empty, or node_feature_dim is zero, skipping observation bounds calculation.")


                print("FuelpriceenvfeatureGraph._load_data: Data loading finished.")
                print(f"FuelpriceenvfeatureGraph._load_data: Final combined_data is None: {self.combined_data is None}") # Debug print
                print(f"FuelpriceenvfeatureGraph._load_data: Final obs_min is None: {self.obs_min is None}") # Debug print
                print(f"FuelpriceenvfeatureGraph._load_data: Final obs_max is None: {self.obs_max is None}") # Debug print


                return self.combined_data # Return the loaded data

            except Exception as e:
                 print(f"FuelpriceenvfeatureGraph._load_data: An error occurred during data processing or conversion: {e}")
                 # Add more specific error printing here
                 import traceback
                 traceback.print_exc()
                 self.combined_data = None # Ensure combined_data is None on error
                 self.node_feature_dim = 0 # Reset node_feature_dim on error
                 self.obs_min = None # Reset bounds on error
                 self.obs_max = None
                 raise # Re-raise the exception after printing


        else:
            print(f"FuelpriceenvfeatureGraph._load_data: Error: Data file not found at {local_data_path}. Please ensure it was downloaded.")
            self.combined_data = None # Ensure combined_data is None if file not found
            self.node_feature_dim = 0 # Reset node_feature_dim if file not found
            self.obs_min = None # Reset bounds if file not found
            self.obs_max = None
            raise FileNotFoundError(f"Data file not found at {local_data_path}") # Raise error if file not found


# Define the AnFuelpriceEnv class (from 6ryKoyyHkPbi)
class FuelpriceEnv(EnvBase):
    def __init__(self, num_envs, seed, device, num_agents=13, **kwargs): # Removed dataframe argument
        self.episode_length = kwargs.get('episode_length', 100)
        self.num_agents = num_agents
        self.allow_repeat_data = kwargs.get('allow_repeat_data', False)
        self.num_envs = num_envs
        self.current_data_index = torch.zeros(num_envs, dtype=torch.int64, device=device)

        self.graph_generator = FuelpriceenvfeatureGraph(device=device, allow_repeat_data=self.allow_repeat_data) # Pass device and allow_repeat_data

        # Pass num_agents to graph_generator BEFORE loading data
        self.graph_generator.num_nodes_per_graph = self.num_agents

        # Data loading now happens inside the graph_generator init
        self.graph_generator._load_data() # Uncommented this line to load data

        # Check if data loading was successful in the graph generator
        if self.graph_generator.combined_data is None:
             print("FuelpriceEnv.__init__: Error: graph_generator combined_data is None after _load_data.") # Debug print
             raise RuntimeError("Data loading failed in FuelpriceenvfeatureGraph.")

        self.combined_data = self.graph_generator.combined_data # Get combined_data from the generator
        self.node_feature_dim = self.graph_generator.node_feature_dim # Get node_feature_dim from generator
        self.obs_min = self.graph_generator.obs_min # Get bounds from generator
        self.obs_max = self.graph_generator.obs_max

        # Add a check here to see if obs_min and obs_max were set
        if self.obs_min is None or self.obs_max is None:
             print("FuelpriceEnv.__init__: Error: Observation bounds (obs_min/obs_max) were not set after data loading.") # Debug print
             raise RuntimeError("Observation bounds (obs_min/obs_max) were not set after data loading.")


        # Check if data loading was successful (redundant with the check above, but kept for clarity)
        if self.combined_data is None:
            raise RuntimeError("Data loading failed during environment initialization.")


        # num_agents is now initialized from the constructor argument
        # self.num_agents = num_agents # Removed redundant initialization
        self.num_individual_actions = 3 # Defined inside the class
        self.num_individual_actions_features = 13 # Still 13 action features per agent


        # In the new single-graph-per-environment structure, num_nodes_per_graph is num_agents
        self.num_nodes_per_graph = self.num_agents
        # Re-calculate num_edges_per_graph based on a potential graph structure among agents.
        # Assuming a fully connected graph among agents.
        self.num_edges_per_graph = self.num_agents * (self.num_agents - 1) if self.num_agents > 1 else 0
        print(f"AnFuelpriceEnv.__init__: num_edges_per_graph calculated as {self.num_edges_per_graph}")

        # Define a simple, fixed edge index for a single graph (e.g., a ring)
        # This is for debugging the policy's graph processing
        if self.num_agents > 1:
            # Create a ring graph: 0->1, 1->2, ..., 12->0
            sources = torch.arange(self.num_agents)
            targets = (torch.arange(self.num_agents) + 1) % self.num_agents
            self._fixed_edge_index_single = torch.stack([sources, targets], dim=0).to(torch.long)
            self._fixed_num_edges_single = self._fixed_edge_index_single.shape[1]
        else:
            self._fixed_edge_index_single = torch.empty(2, 0, dtype=torch.long)
            self._fixed_num_edges_single = 0


        # Node feature dimension needs to be defined
        # obs_dim is the dimension of the observation space for a single agent, or the relevant feature dimensions.
        # The observation spec should now reflect the variable number of agents.
        # The shape of 'x' should be [num_envs, num_agents, node_feature_dim].

        super().__init__(device=device, batch_size=[num_envs])

        self._make_specs()

    # Add the _set_seed method
    def _set_seed(self, seed: Optional[int] = None):
        # Implement seeding logic here if needed
        # For a simple implementation, you can just store the seed
        if seed is not None:
            self.seed = seed
        else:
            self.seed = torch.seed() # Use torch's current seed if none provided
        # Note: For proper reproducibility, you might need to seed other components
        # like random number generators used in _reset or _step.
        return seed

    # Removed _get_state_at method as its logic will be embedded in _reset and _step


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
                     # Removed graph_attributes from observation spec
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
                 # Removed global_reward_in_state from state_spec
                 # Added top-level done, terminated, truncated keys for the current state
                 "done": Categorical(n=2, # Added n=2 back
                      # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                      shape=torch.Size([self.num_envs]), # Corrected shape
                      dtype=torch.bool,
                      device=self.device),
                 "terminated": Categorical(n=2, # Added n=2 back
                      # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                      shape=torch.Size([self.num_envs]), # Corrected shape
                      dtype=torch.bool,
                      device=self.device),
                 "truncated": Categorical(n=2, # Added n=2 back
                      # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                      shape=torch.Size([self.num_envs]), # Corrected shape
                      dtype=torch.bool,
                      device=self.device),
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
                          # Removed graph_attributes from next observation spec
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
                      # Removed global_reward_in_state from next state spec
                      # Also include reward key under ('agents',) under 'next'
                      ('agents', 'reward'): Unbounded(shape=torch.Size([self.num_envs, self.num_agents, 1]), dtype=torch.float32, device=self.device),
                       # Removed nested done, terminated, truncated keys from state spec
                       # ("agents", "terminated"): Categorical(n=2, # Added n=2 back
                       #      # shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1] as done/terminated/truncated are per env
                       #      shape=torch.Size([self.num_envs]), # Corrected shape
                       #      dtype=torch.bool,
                       #      device=self.device),
                       # ("agents", "truncated"):  Categorical(n=2, # Added n=2 back
                       #      # shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1]
                       #      shape=torch.Size([self.num_envs]), # Corrected shape
                       #       dtype=torch.bool,
                       #       device=self.device),
                       # ("agents", "done"):  Categorical(n=2, # Added n=2 back
                       #      # shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1]
                       #      shape=torch.Size([self.num_envs]), # Corrected shape
                       #       dtype=torch.bool,
                       #       device=self.device),
                     # Add top-level done, terminated, truncated keys to the "next" composite
                     "done": Categorical(n=2, # Added n=2 back
                          # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                          shape=torch.Size([self.num_envs]), # Corrected shape
                          dtype=torch.bool,
                          device=self.device),
                     "terminated": Categorical(n=2, # Added n=2 back
                          # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                          shape=torch.Size([self.num_envs]), # Corrected shape
                          dtype=torch.bool,
                          device=self.device),
                     "truncated": Categorical(n=2, # Added n=2 back
                          # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                          shape=torch.Size([self.num_envs]), # Corrected shape
                      dtype=torch.bool,
                      device=self.device),
                 }),
             },
             # The batch size of the state spec is the number of environments
             batch_size=self.batch_size,
             device=self.device,
         )
        print(f"State specification defined with single graph per env structure and batch shape {self.state_spec.shape}.")


        # Corrected nvec to be a 1D tensor of size num_individual_actions_features with value num_individual_actions
        # This tensor defines the number of categories for each of the individual action features (13 features, each with 3 categories).
        # Ensure nvec is correctly shaped [self.num_individual_actions_features]
        # nvec should be a tensor of shape [self.num_individual_actions_features] where each element is self.num_individual_actions (3).
        # Explicitly creating a list and then a tensor to ensure 1D shape.
        nvec_list = [self.num_individual_actions] * self.num_individual_actions_features
        nvec_tensor = torch.tensor(nvec_list, dtype=torch.int64, device=self.device)


        # Define the action specification using the MultiCategorical SPEC class from torchrl.data
        # The shape should be [num_agents, num_individual_actions_features] for an unbatched environment
        # The nvec should be [num_agents, num_individual_actions_features]
        nvec_unbatched = nvec_tensor.repeat(self.num_agents).view(self.num_agents, self.num_individual_actions_features)

        self.action_spec_unbatched = Composite(
              {("agents","action"): MultiCategoricalSpec( # Use the MultiCategorical SPEC class
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
        # Removed nested done, terminated, truncated keys from done_spec
        self.done_spec = Composite(
            {
                # ("agents", "done"):  Categorical(n=2, # Added n=2 back
                #       # shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1]
                #       shape=torch.Size([self.num_envs]), # Corrected shape
                #       dtype=torch.bool,
                #       device=self.device),

                # ("agents", "terminated"): Categorical(n=2, # Added n=2 back
                #       # shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1]
                #       shape=torch.Size([self.num_envs]), # Corrected shape
                #       dtype=torch.bool,
                #       device=self.device),
                # ("agents", "truncated"):  Categorical(n=2, # Added n=2 back
                #      # shape=torch.Size([self.num_envs, 1]), # Shape should be [num_envs, 1]
                #      shape=torch.Size([self.num_envs]), # Corrected shape
                #       dtype=torch.bool,
                #       device=self.device),
                # Add top-level done, terminated, truncated keys to the done_spec
                "done": Categorical(n=2, # Added n=2 back
                     # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                     shape=torch.Size([self.num_envs]), # Corrected shape
                     dtype=torch.bool,
                     device=self.device),
                "terminated": Categorical(n=2, # Added n=2 back
                     # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                     shape=torch.Size([self.num_envs]), # Corrected shape
                     dtype=torch.bool,
                     device=self.device),
                "truncated": Categorical(n=2, # Added n=2 back
                     # shape=torch.Size([self.num_envs, 1]), # Corrected shape
                     shape=torch.Size([self.num_envs]), # Corrected shape
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
        print("Debug: Exiting _make_specs.")


    # Implement the actual _is_terminal method
    def _is_terminal(self) -> torch.Tensor:
        """
        Determines if the current state is a terminal state.

        Returns:
            torch.Tensor: A boolean tensor of shape [num_envs] indicating
                          whether each environment is in a terminal state.
        """
        # --- Implement your actual episode termination logic here ---
        # Example: Terminate if a certain prediction error threshold is crossed,
        #          or if the episode has gone on for too long (handled by truncated).

        # For now, keeping it as always False, allowing episodes to end only by truncation.
        # Replace this with your specific termination conditions.
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Example termination based on a hypothetical error threshold (replace with your logic):
        # if self.current_data_index > 0: # Avoid checking at step 0
        #     # Assuming you have access to predicted and actual prices somewhere
        #     # Example: hypothetical 'predicted_price' and 'actual_price' in your state or data
        #     # (You'll need to adapt this based on your actual data structure and how predictions are made)
        #     predicted_price = ... # Get predicted price based on current state/data and actions
        #     actual_price = ...    # Get actual price from your data at current_data_index + 1 (next step)
        #     error = torch.abs(predicted_price - actual_price)
        #     error_threshold = 10.0 # Define your threshold
        #     terminated = error > error_threshold

        return terminated


    # Implement the actual _batch_reward method

    def _batch_reward(self, data_indices: torch.Tensor, tensordict: TensorDict) -> TensorDict: # Changed actions type hint to TensorDict
        # Check if combined_data is available
        if self.combined_data is None:
            print("FuelpriceEnv._batch_reward: Error: combined_data is None. Data loading likely failed.")
            # Return a tensordict with zero rewards and appropriate batch size
            original_batch_shape = data_indices.shape
            num_agents = self.num_agents
            rewards_reshaped = torch.zeros(*original_batch_shape, num_agents, 1, dtype=torch.float32, device=self.device)
            return TensorDict({("agents", "reward"): rewards_reshaped}, batch_size=original_batch_shape, device=self.device)


        # data_indices shape: [num_envs] or [num_envs, num_steps]
        # tensordict is the input tensordict passed to step() by the collector

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

            # Get the returns for the valid data points
            # Assuming returns are columns 13 to 25 (13 columns) in the combined data.
            # Corrected indexing for returns based on self.node_feature_dim
            returns_data_valid_flat = self.combined_data[valid_data_indices_flat][:, self.node_feature_dim : self.node_feature_dim + self.num_agents] # Shape [num_valid_flat, num_agents]


            # Extract actions from the input tensordict for valid data points and restructure them
            # The input tensordict's batch size will match the batch size of data_indices,
            # which can be [num_envs] for a single step or [num_envs, num_steps] for a rollout.
            # We need to flatten the input tensordict to match the flat_batch_size of data_indices_flat.

            # Let's check the shape of the action tensor in the input tensordict directly.
            # The input tensordict has batch_size matching the original batch_shape of data_indices.
            # The action key is ('agents', 'action').
            actions_tensor_input = tensordict.get(("agents", "action"))
            print(f"Debug: Input tensordict batch shape: {tensordict.batch_size}")
            print(f"Debug: Action tensor shape in input tensordict: {actions_tensor_input.shape}")

            # Now flatten the actions tensor to match the flat_batch_size of data_indices_flat
            # The flattened shape should be [flat_batch_size, num_agents, num_action_features]
            expected_action_flat_shape = torch.Size([flat_batch_size, num_agents, num_action_features])
            if actions_tensor_input.shape == expected_action_flat_shape:
                 actions_tensor_flat = actions_tensor_input # Already flat (batch size was flat_batch_size)
            elif actions_tensor_input.shape[:len(original_batch_shape)] == original_batch_shape:
                 # The action tensor has the original batch shape and then the agent/action dimensions.
                 # Flatten the batch dimensions.
                 actions_tensor_flat = actions_tensor_input.view(flat_batch_size, num_agents, num_action_features)
            else:
                 print(f"Error: Unexpected action tensor shape in input tensordict: {actions_tensor_input.shape}. Expected batch_shape + [{num_agents}, {num_action_features}]. Using zeros for rewards.")
                 # Keep the corresponding slices in rewards_flat as zeros (initialized)
                 valid_indices_flat = torch.tensor([], dtype=torch.int64, device=self.device) # Treat all as invalid to skip calculation
                 actions_tensor_flat = None # Set to None to prevent use


            if valid_indices_flat.numel() > 0 and actions_tensor_flat is not None:
                try:
                    # Select the relevant slices from the flattened actions tensor using valid_indices_flat
                    actions_tensor_valid_flat = actions_tensor_flat[valid_indices_flat].contiguous() # Shape [num_valid_flat, num_agents, num_action_features]
                    print(f"Debug: Action tensor shape after flattening and slicing: {actions_tensor_valid_flat.shape}")


                    # The action tensor now has shape [num_valid_flat, num_agents, num_action_features].
                    # We need to calculate the reward for each agent based on their specific action features.
                    # The returns_data_valid_flat has shape [num_valid_flat, num_agents] (returns for 13 agents).
                    # We need to compare the action features for each agent against their corresponding return.

                    # Let's assume for simplicity that the first action feature (index 0) determines the 'down', 'hold', 'up' action.
                    # If your action features have a different meaning, this logic needs adjustment.
                    # Assuming actions_tensor_valid_flat[:, :, 0] is the 'down' (0), 'hold' (1), 'up' (2) action for each agent.
                    # This assumes the MultiCategorical output [num_agents, 13] means each of the 13 values is an independent action (0, 1, or 2) for that agent.
                    # If the 13 values define a single complex action per agent, the reward logic needs to be more complex.

                    # Based on the MultiCategorical definition (nvec with 13 elements, each 3 categories),
                    # it seems each agent has 13 independent categorical actions with 3 choices each.
                    # The reward should then be a function of the returns for each agent and all 13 of their chosen action features.
                    # This requires a more complex reward calculation than the simple 'down/hold/up' based on returns.

                    # For simplicity, let's calculate a reward for each agent by comparing *each* of their 13 action features
                    # to the corresponding 13 return values for that step.

                    # returns_data_valid_flat shape: [num_valid_flat, num_agents] (returns for num_agents)
                    # actions_tensor_valid_flat shape: [num_valid_flat, num_agents, num_action_features] (actions for num_agents, each with num_action_features)

                    # We need to compare actions_tensor_valid_flat[:, j, i] with returns_data_valid_flat[:, j]
                    # for each agent j (0 to num_agents-1) and each action feature/return i (0 to num_action_features-1).

                    # Reshape returns_data_valid_flat for broadcasting: [num_valid_flat, num_agents, 1]
                    returns_broadcastable = returns_data_valid_flat.unsqueeze(-1)

                    # Create masks based on the actions [num_valid_flat, num_agents, num_action_features]
                    down_mask_valid_flat = (actions_tensor_valid_flat == 0)
                    up_mask_valid_flat = (actions_tensor_valid_flat == 2)
                    hold_mask_valid_flat = (actions_tensor_valid_flat == 1)

                    # Calculate rewards for each action feature comparison [num_valid_flat, num_agents, num_action_features]
                    # If action feature i for agent j is 'down' (0), reward contribution is -returns_data_valid_flat[:, j]
                    # If action feature i for agent j is 'up' (2), reward contribution is +returns_data_valid_flat[:, j]
                    # If action feature i for agent j is 'hold' (1), reward contribution is -0.01 * torch.abs(returns_repeated[:, :, i]) # Corrected indexing


                    # We need to broadcast returns_data_valid_flat [num_valid_flat, num_agents] to [num_valid_flat, num_agents, num_action_features]
                    # by repeating along the action feature dimension.
                    returns_repeated = returns_data_valid_flat.unsqueeze(-1).repeat(1, 1, num_action_features) # Shape [num_valid_flat, num_agents, num_action_features]


                    reward_contributions_down = -returns_repeated * down_mask_valid_flat.float()
                    reward_contributions_up = returns_repeated * up_mask_valid_flat.float()
                    # Corrected the indexing here
                    reward_contributions_hold = -0.01 * torch.abs(returns_repeated) * hold_mask_valid_flat.float()

                    # Sum the reward contributions across the action features dimension for each agent
                    agent_rewards_valid_flat = (reward_contributions_down + reward_contributions_up + reward_contributions_hold).sum(dim=-1) # Shape [num_valid_flat, num_agents]


                    # Add the last dimension to match the expected output shape [flat_batch_size, num_agents, 1]
                    agent_rewards_valid_flat = agent_rewards_valid_flat.unsqueeze(-1) # Shape [num_valid_flat, num_agents, 1]

                    # Assign calculated rewards to the selected slice of the rewards_flat tensor
                    rewards_flat[valid_indices_flat] = agent_rewards_valid_flat # Assign to the slice


                except KeyError:
                     print("Error: Action key ('agents', 'action') not found in the input tensordict to _batch_reward.")
                     # Keep the corresponding slices in rewards_flat as zeros (initialized)
                     pass
                except Exception as e:
                     print(f"An error occurred during reward calculation in _batch_reward: {e}")
                     # Keep the corresponding slices in rewards_flat as zeros (initialized)
                     pass


        # Reshape rewards_flat back to the original input batch size [num_envs, num_steps, num_agents, 1] or [num_envs, num_agents, 1]
        # The output TensorDict batch size should match the original batch_shape of data_indices.
        rewards_reshaped = rewards_flat.view(*original_batch_shape, num_agents, 1)


        # Return rewards wrapped in a TensorDict with the expected key and original input batch size
        # The batch size of the output TensorDict should match the batch size of data_indices
        return TensorDict({("agents", "reward"): rewards_reshaped}, batch_size=original_batch_shape, device=self.device) # Use original_batch_shape for batch size





    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Check if combined_data is available
        if self.combined_data is None:
            print("FuelpriceEnv._step: Error: combined_data is None. Data loading likely failed.")
            # Return a tensordict with done flags set to True and dummy values for other keys
            num_envs = tensordict.batch_size[0] if tensordict.batch_size else self.num_envs
            dummy_obs_td = TensorDict({
                 "x": torch.zeros(num_envs, self.num_agents, self.node_feature_dim, dtype=torch.float32, device=self.device),
                 "edge_index": torch.zeros(num_envs, 2, self._fixed_num_edges_single, dtype=torch.int64, device=self.device),
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


        self.current_data_index += 1

        terminated = self._is_terminal()
        truncated = (self.current_data_index >= self.episode_length)

        # The action is now expected to be in the input tensordict passed to step() by the collector
        # It should be at tensordict[('agents', 'action')]
        actions=tensordict.get(('agents', 'action')) # Use .get() here as well
        # Pass the input tensordict directly to _batch_reward
        # _batch_reward will need to extract the action from this tensordict
        # Get the reward tensordict from _batch_reward
        reward_td = self._batch_reward(self.current_data_index, tensordict) # Pass the input tensordict here


        # Logic previously in _get_state_at for next state
        num_envs = self.current_data_index.shape[0]

        # Get data indices for the next step, handling boundaries
        data_indices = torch.min(self.current_data_index + 1, torch.as_tensor(self.combined_data.shape[0] - 1, device=self.device))
        # Extract the first node_feature_dim columns for each environment
        # If all agents share the same features, extract and then expand
        x_data_time_step = self.combined_data[data_indices, :self.node_feature_dim] # Shape: [num_envs, node_feature_dim]
        # Expand to [num_envs, num_agents, node_feature_dim]
        x_data = x_data_time_step.unsqueeze(1).expand(-1, self.num_agents, -1)

        # Use fixed edge index repeated for the batch
        edge_index_data = self._fixed_edge_index_single.unsqueeze(0).repeat(num_envs, 1, 1).to(self.device)
        print(f"_step: Using fixed edge index. Generated edge_index_data shape = {edge_index_data.shape}")


        next_state_tensordict_data = TensorDict({
             "x": x_data, # Use actual data for node features
             "edge_index": edge_index_data, # Placeholder edge indices
             "batch": torch.arange(num_envs, device=self.device).repeat_interleave(self.num_agents).view(num_envs, self.num_agents), # Placeholder batch tensor
             "time": self.current_data_index.unsqueeze(-1).to(self.device), # Use the provided data_index as timestamp
        }, batch_size=[num_envs], device=self.device)


        # Create the output tensordict containing the next observation, reward, and done flags
        # The next observation and reward should be nested under "next" by the collector.
        # The done flags should be at the root level of the tensordict returned by _step.
        # Structure the output tensordict with reward and done flags at the root level
        output_tensordict = TensorDict({
            # Include the next observation structure under ("agents", "data")
            "next": Composite({
                 ("agents", "data"): next_state_tensordict_data,
                 # Include the reward tensordict directly under the key expected by reward_spec
                 # This structure should match the reward_spec defined in _make_specs.
                 # reward_spec is Composite({('agents', 'reward'): Unbounded(...)}) with batch_size=[num_envs]
                 # The _batch_reward function returns a TensorDict structured as {('agents', 'reward'): reward_tensor}
                 # So, we need to include this entire reward_td under the key where reward is expected in 'next'.
                 # Based on the error and torchrl's conventions, the reward tensor itself, not the tensordict containing it,
                 # might be expected directly under ('agents', 'reward') within 'next'.
                 # Reverting to the previous approach of including the reward tensor directly.
                 # The issue might lie elsewhere, possibly related to how specs are defined or used by the collector.
                 ('agents', 'reward'): reward_td.get(('agents', 'reward')), # Include the reward tensor here
                 "terminated": terminated.unsqueeze(-1).to(self.device),
                 "truncated": truncated.unsqueeze(-1).to(self.device),
                 "done": (terminated | truncated).unsqueeze(-1).to(self.device),
            }),
            # Include the done flags at the root level
            "terminated": terminated.unsqueeze(-1).to(self.device), # Ensure shape matches spec [num_envs, 1]
            "truncated": truncated.unsqueeze(-1).to(self.device), # Ensure shape matches spec [num_envs, 1]
            "done": (terminated | truncated).unsqueeze(-1).to(self.device), # Ensure shape matches spec [num_envs, 1]

        }, batch_size=[self.num_envs], device=self.device)


        # Return the output_tensordict
        return output_tensordict

    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        # Check if combined_data is available
        if self.combined_data is None:
            print("FuelpriceEnv._reset: Error: combined_data is None. Data loading likely failed during environment initialization.")
            raise RuntimeError("Cannot reset environment: Data loading failed during environment initialization.")


        if self.allow_repeat_data and self.combined_data is not None:
             max_start_index = self.combined_data.shape[0] - self.episode_length -1
             if max_start_index < 0:
                  self.current_data_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
                  print("Warning: Data length is less than episode_length + 1. Starting episodes from index 0.")
             else:
                  self.current_data_index = torch.randint(0, max_start_index + 1, (self.num_envs,), dtype=torch.int64, device=self.device)
        else:
             self.current_data_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        num_envs = self.current_data_index.shape[0]
        data_indices = self.current_data_index
        data_indices = torch.min(data_indices, torch.as_tensor(self.combined_data.shape[0] - 1, device=self.device))

        # Modified: Ensure x_data has shape [num_envs, num_agents, node_feature_dim]
        x_data_time_step = self.combined_data[data_indices, :self.node_feature_dim] # Shape [num_envs, node_feature_dim]
        x_data = x_data_time_step.unsqueeze(1).expand(-1, self.num_agents, -1) # Shape [num_envs, 1, node_feature_dim] -> [num_envs, num_agents, node_feature_dim]


        edge_index_data = self._fixed_edge_index_single.unsqueeze(0).repeat(num_envs, 1, 1).to(self.device)
        print(f"_reset: Using fixed edge index. Generated edge_index_data shape = {edge_index_data.shape}")


        initial_observation_data_td = TensorDict({
             "x": x_data, # Use actual data for node features
             "edge_index": edge_index_data, # Placeholder edge indices
             "batch": torch.arange(num_envs, device=self.device).repeat_interleave(self.num_agents).view(num_envs, self.num_agents), # Placeholder batch tensor
             "time": self.current_data_index.unsqueeze(-1).to(self.device), # Use the provided data_index as timestamp
        }, batch_size=[num_envs], device=self.device)


        # Create the tensordict to return, containing the initial observation and done flags
        # The initial observation should be nested under ("agents", "data")

        output_tensordict = TensorDict({
            # Include the initial observation structure
            ("agents", "data"): initial_observation_data_td,
            # Set initial done, terminated, truncated flags to False at the root level
            # Corrected shapes to [num_envs, 1] to match done_spec
            "terminated": torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
            "truncated": torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
            "done": torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
             # Removed nested done, terminated, truncated keys from reset output
            # ("agents", "terminated"): torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
            # ("agents", "truncated"): torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
            # ("agents", "done"): torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
        }, batch_size=[self.num_envs], device=self.device)


        # Removed all debugging print statements from _reset
        # print("\n--- Environment _reset output tensordict ---")
        # print(f"Output tensordict keys: {output_tensordict.keys(include_nested=True)}") # Added include_nested=True
        # # Also print keys of nested observation tensordict, changed 'observation' to 'data'
        # if ("agents", "data") in output_tensordict.keys(include_nested=True): # Added include_nested=True
        #      nested_obs_td = output_tensordict.get(("agents", "data"))
        #      print(f"  Nested ('agents', 'data') keys: {nested_obs_td.keys(include_nested=True)}") # Added include_nested=True
        #      print(f"  Nested ('agents', 'data') shape: {nested_obs_td.shape}")
        #      if ("x") in nested_obs_td.keys():
        #          print(f"  Nested ('agents', 'data', 'x') shape: {output_tensordict.get(('agents', 'data', 'x')).shape}")
        #          print(f"  Nested ('agents', 'data', 'x') dtype: {output_tensordict.get(('agents', 'data', 'x')).dtype}")
        #      else:
        #           print("  Nested ('agents', 'data', 'x') key NOT found.")
        #      # Check for timestamp key
        #      if "time" in nested_obs_td.keys():
        #           print(f"  Nested ('agents', 'data', 'time') value sample: {output_tensordict.get(('agents', 'data', 'time'))[0][:5]}...")
        #      else:
        #           print("  Nested ('agents', 'data', 'time') key NOT found.")

        # else:
        #      print("  Nested ('agents', 'data') key NOT found in output_tensordict.")
        # # Print done keys to verify their location
        # if ("done") in output_tensordict.keys():
        #      print(f"  ('done') value: {output_tensordict.get(('done')))}")
        # if ("terminated") in output_tensordict.keys():
        #      # Fixed the unmatched parenthesis here
        #      print(f"  ('terminated') value: {output_tensordict.get(('terminated')))}")
        # if ("truncated") in output_tensordict.keys():
        #      # Fixed the unmatched parenthesis here
        #      print(f"  ('truncated') value: {output_tensordict.get(('truncated')))}")
        # if ("agents", "done") in output_tensordict.keys(include_nested=True):
        #      print(f"  ('agents', 'done') value: {output_tensordict.get(('agents', 'done')))}")
        # if ("agents", "terminated") in output_tensordict.keys(include_nested=True):
        #      print(f"  ('agents', 'terminated') value: {output_tensordict.get(('agents', 'terminated')))}")
        # if ("agents", "truncated") in output_tensordict.keys(include_nested=True):
        #      print(f"  ('truncated') value: {output_tensordict.get(('truncated')))}")
        # # Print batch key to verify its location, changed 'observation' to 'data'
        # if ('agents', 'data', 'batch') in output_tensordict.keys(include_nested=True):
        #      print(f"  ('agents', 'data', 'batch') value sample: {output_tensordict.get(('agents', 'data', 'batch'))[0][:5]}...") # Print sample of batch tensor


        # print("-------------------------------------------")


        return output_tensordict


# Add code to instantiate and check the environment here
# You can get num_envs, seed, device, and num_agents from the hyperparameters cell (XNDTTPwfRY0f)
# Make sure to run the hyperparameters cell first.
# from XNDTTPwfRY0f import num_envs, seed, device, num_agents # This would require importing from another cell, which is not directly supported.
# Instead, redefine them or ensure the user runs the hyperparameters cell first.

# Assuming the user runs cell XNDTTPwfRY0f first, these variables will be defined.
# If you want to define them here explicitly for self-containment, you can uncomment and set them:
# num_envs = 3
# seed = 42
# device = 'cpu' # or 'cuda'
# num_agents = 13


# Check if num_envs, seed, device, and num_agents are defined (assuming XNDTTPwfRY0f has been run)
try:
    # Attempt to access variables defined in XNDTTPwfRY0f
    _ = num_envs
    _ = seed
    _ = device
    _ = num_agents
except NameError:
    print("Please run the hyperparameters cell (XNDTTPwfRY0f) first to define num_envs, seed, device, and num_agents.")
    # You could exit or raise an error here if these variables are critical
    # exit()
    # raise


# Instantiate the environment
print("Instantiating the FuelpriceEnv environment...")
try:
    # Use the variables defined in the hyperparameters cell
    env = FuelpriceEnv(num_envs=num_envs, seed=seed, device=device, episode_length=10, num_agents=num_agents) # Reduced episode_length for a quick test
    print("\nEnvironment instantiated successfully.")

    # Check environment specs
    print("\nChecking environment specs...")
    check_env_specs(env)
    print("Environment specs checked successfully.")

except Exception as e:
    print(f"\nAn error occurred during environment instantiation or spec check: {e}")
