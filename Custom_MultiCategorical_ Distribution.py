# @title  Custom Distributions & Flatten modules


from typing import Optional, Union, List

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
# from torch.distributions.utils import broadcast_all # Not needed for MultiCategorical in this structure
from torch.distributions.categorical import Categorical as TorchCategorical
from torchrl.data import CompositeSpec, DiscreteTensorSpec # Import DiscreteTensorSpec
from tensordict.nn import TensorDictModule # Import TensorDictModule


# --- Define Custom MultiCategorical Distribution (Inheriting from Distribution) ---
# Reverting to direct inheritance from Distribution as requested.

class MultiCategorical(Distribution):
    """
    A MultiCategorical distribution for multiple agents with multiple independent categorical action features.

    Args:
        logits (torch.Tensor): The logits tensor.
            Expected shape: `[..., num_agents, num_individual_actions_features, num_action_categories]`
            Can also accept a flattened tensor of shape `[..., num_agents * num_individual_actions_features * num_action_categories]`
        action_spec (DiscreteTensorSpec): The action specification of the environment.
            Used to infer num_agents, num_individual_actions_features, and dtype.
            The spec should have shape `[..., num_agents, num_individual_actions_features]`.
            The nvec should be defined in the spec with shape `[..., num_agents, num_individual_actions_features]`.
    """
    arg_constraints = {"logits": constraints.real} # Define argument constraints if necessary
    has_enumerate_support = False # Enumerating support for multi-categorical is complex and likely not needed

    def __init__(self, logits: torch.Tensor, action_spec: DiscreteTensorSpec, validate_args: Optional[bool] = None):
        # print("MultiCategorical.__init__ called.") # Debug print
        self.action_spec = action_spec

        # Infer shapes from action_spec
        # action_spec is expected to be a DiscreteTensorSpec
        # with shape [..., num_agents, num_individual_actions_features] and nvec defined.
        action_spec_shape = self.action_spec.shape
        action_spec_nvec = self.action_spec.nvec

        if len(action_spec_shape) < 2:
             raise ValueError(f"Unexpected action_spec shape: {action_spec_shape}. Expected at least 2 dimensions (num_agents, num_individual_actions_features) for the event shape.")
        if len(action_spec_nvec.shape) != len(action_spec_shape):
             raise ValueError(f"Action_spec nvec shape {action_spec_nvec.shape} must match action_spec shape {action_spec_shape}.")


        self.num_agents = action_spec_shape[-2] # Second to last dimension is num_agents
        self.num_individual_actions_features = action_spec_shape[-1] # Last dimension is num_individual_actions_features
        # Assuming all individual action features for all agents have the same number of categories
        # This is consistent with how nvec is created in the environment.
        # Get category count from the last dimension of nvec.
        # All elements in the last dimension of nvec should be the same.
        unique_categories = torch.unique(action_spec_nvec)
        if len(unique_categories) != 1:
             raise ValueError(f"Expected all category counts in action_spec.nvec to be the same, but found {unique_categories}")
        self.num_action_categories = unique_categories.item() # Get category count


        # Handle potentially flattened logits
        expected_flat_size_per_graph = self.num_agents * self.num_individual_actions_features * self.num_action_categories

        if logits.shape[-1] == expected_flat_size_per_graph and len(logits.shape) > 1:
             # Logits are flattened, reshape them
             batch_shape_logits = logits.shape[:-1] # Batch shape excluding the flattened dimension
             self.logits = logits.view(batch_shape_logits + torch.Size([self.num_agents, self.num_individual_actions_features, self.num_action_categories]))
             # print(f"Debug: Reshaped flattened logits to: {self.logits.shape}") # Debug print
        else:
             # Assume logits have the expected non-flattened shape, validate it
             expected_shape_suffix = (self.num_agents, self.num_individual_actions_features, self.num_action_categories)
             if logits.shape[-len(expected_shape_suffix):] != expected_shape_suffix:
                  raise ValueError(f"Logits shape mismatch with inferred shapes and expected flattened size. Expected shape ending with {expected_shape_suffix} or flattened size {expected_flat_size_per_graph}, but got {logits.shape}. Full logits shape: {logits.shape}")
             self.logits = logits


        # The batch shape of this distribution is the batch shape of the input logits (excluding the last 3 dims).
        batch_shape = self.logits.shape[:-3] # Exclude num_agents, num_individual_actions_features, num_action_categories

        # The event shape is [num_agents, num_individual_actions_features]
        event_shape = torch.Size([self.num_agents, self.num_individual_actions_features])

        super(MultiCategorical, self).__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)

        # Store dtype for deterministic sample from the action spec
        self._action_spec_dtype = action_spec.dtype if hasattr(action_spec, 'dtype') else torch.long


    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultiCategorical, _instance)
        batch_shape = torch.Size(batch_shape)
        # Expand logits to the new batch shape
        new.logits = self.logits.expand(batch_shape + self.event_shape + (self.num_action_categories,)) # Expand batch and keep event/category dims
        # The action_spec also needs to be expanded to the new batch shape.
        new.action_spec = self.action_spec.expand(batch_shape + self.action_spec.shape) # Expand action_spec batch

        new.num_agents = self.num_agents # Keep original inferred values
        new.num_individual_actions_features = self.num_individual_actions_features
        new.num_action_categories = self.num_action_categories
        new._action_spec_dtype = self._action_spec_dtype

        super(MultiCategorical, new).__init__(batch_shape, new.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self.logits.new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=2) # Event dim is 2 for [num_agents, num_individual_actions_features]
    def support(self):
        # Support for MultiCategorical is a product of individual categorical supports.
        # This is complex to represent concisely. Returning a general constraint.
        # Each element in the event shape [num_agents, num_individual_actions_features]
        # has a support of {0, 1, ..., num_action_categories - 1}.
        # For simplicity and alignment with torch.distributions, we might not define a specific support constraint
        # if it's too complex. Returning a dummy constraint for now.
        return constraints.greater_equal(0) # Dummy constraint


    @property
    def probs(self) -> torch.Tensor:
        """
        Returns the probabilities derived from the logits.
        Shape: `[..., num_agents, num_individual_actions_features, num_action_categories]`
        """
        return torch.softmax(self.logits, dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        """
        Returns the mode of the distribution as a stand-in for mean for deterministic evaluation.
        This method is explicitly defined to satisfy torchrl's requirements.
        """
        # print("MultiCategorical.mean property accessed.") # Debug print
        # The mode of a Categorical distribution is the index of the highest logit.
        # self.logits has shape [..., num_agents, num_individual_actions_features, num_action_categories]
        # We need to take argmax over the last dimension.
        mode = torch.argmax(self.logits, dim=-1) # Shape [..., num_agents, num_individual_actions_features]
        return mode.to(self._action_spec_dtype) # Ensure dtype matches action spec


    @property
    def mode(self) -> torch.Tensor:
        """
        Returns the mode of the MultiCategorical distribution.
        This corresponds to the action with the highest probability for each
        of the individual categorical choices.
        """
        # print("MultiCategorical.mode property accessed.") # Debug print
        # The mode is already computed in the mean property for this structure.
        return self.mean # Use the mean property which is the mode

    @property
    def variance(self) -> Tensor:
         # Variance for MultiCategorical is complex as it's a product distribution.
         # Variance of a categorical is p * (1-p). For a product, it's the product of variances
         # for the log_prob (sum of log_probs).
         # However, the variance of the *sample* itself is more complex.
         # It might be better to not implement this if not strictly needed.
         raise NotImplementedError("Variance for MultiCategorical is not implemented.")


    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape batch of samples.

        Returns:
            torch.Tensor: The sampled actions.
                Shape: `[sample_shape, ..., num_agents, num_individual_actions_features]`
        """
        shape = self._extended_shape(sample_shape)
        # We need to sample independently for each agent and each action feature.
        # The logits are already structured for this:
        # [..., num_agents, num_individual_actions_features, num_action_categories]
        # We can use torch.distributions.Categorical directly on the last dimension.

        # Create a base Categorical distribution for sampling.
        # Need to reshape logits to [..., num_action_categories] for the base Categorical.
        # The batch shape for this temporary Categorical will be
        # [..., num_agents, num_individual_actions_features]
        original_shape = self.logits.shape
        reshaped_logits = self.logits.view(-1, self.num_action_categories) # Flatten batch and event dims

        # Create a temporary Categorical distribution
        base_categorical = TorchCategorical(logits=reshaped_logits)

        # Sample from the base categorical distribution
        # The sample method of the base categorical returns a tensor with shape:
        # [sample_shape, batch_shape of base_categorical]
        # In our case: [sample_shape, flattened_batch_and_event_dims]
        # where flattened_batch_and_event_dims is product of batch shape and event shape

        # The batch shape of the base_categorical is the shape of reshaped_logits[:-1], which is just [flattened_batch_and_event_dims]
        # The sample shape requested is `sample_shape`
        sampled_flat = base_categorical.sample(sample_shape) # Shape: [sample_shape, flattened_batch_and_event_dims]

        # Reshape the sampled actions back to the desired shape
        # Desired shape: [sample_shape, ..., num_agents, num_individual_actions_features]
        output_shape = sample_shape + self.batch_shape + self.event_shape
        sampled_action = sampled_flat.view(output_shape)

        # Validate sampled action against action spec if validate_args is True
        if self._validate_args:
             # Need to ensure the action spec used for validation has the correct batch shape.
             # The action_spec stored in the distribution instance should have the correct batch shape
             # after expand is called.
             # Or, we can get the spec from the distribution itself: self.action_spec
             self.action_spec.assert_acceptable(sampled_action)


        return sampled_action.to(self._action_spec_dtype)


    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability density of a given sample.

        Args:
            value (torch.Tensor): The action tensor for which to compute the log probability.
                Expected shape: `[..., num_agents, num_individual_actions_features]`

        Returns:
            torch.Tensor: The log probability of the action.
                Shape: `[..., 1]` (summed log probs over action features and agents)
        """
        if self._validate_args:
             self._validate_sample(value)

        # We need to compute the log probability for each individual categorical choice
        # and then sum them up.
        # The value tensor has shape [..., num_agents, num_individual_actions_features]
        # The logits tensor has shape [..., num_agents, num_individual_actions_features, num_action_categories]

        # Create a base Categorical distribution for log_prob calculation.
        # Reshape logits to [..., num_action_categories] for the base Categorical.
        # The batch shape for this temporary Categorical will be
        # [..., num_agents, num_individual_actions_features]
        original_logits_shape = self.logits.shape
        reshaped_logits = self.logits.view(-1, self.num_action_categories) # Flatten batch and event dims

        # Create a temporary Categorical distribution
        base_categorical = TorchCategorical(logits=reshaped_logits)

        # Reshape the value tensor to match the batch shape of the base Categorical
        # Value shape: [..., num_agents, num_individual_actions_features]
        # Desired shape for base_categorical.log_prob input: [..., flattened_batch_and_event_dims]
        original_value_shape = value.shape
        # Need to flatten the value tensor to match the flattened batch shape of the logits
        reshaped_value = value.view(-1) # Flatten all dimensions


        # The log_prob method of the base categorical expects a value tensor with shape:
        # [..., batch_shape of base_categorical]
        # In our case, after flattening, the batch shape is the product of the original batch shape and event shape.
        # The value tensor should have this flattened shape.
        # The log_prob returns a tensor with the same batch shape: [..., flattened_batch_and_event_dims]
        log_prob_flat = base_categorical.log_prob(reshaped_value) # Shape [flattened_batch_and_event_dims]

        # Reshape the log probabilities back to [..., num_agents, num_individual_actions_features]
        # This will have the same batch shape as the original input tensors (logits and value)
        log_prob_per_choice = log_prob_flat.view(self.batch_shape + self.event_shape) # Shape [..., num_agents, num_individual_actions_features]


        # Sum over the event dimensions ([num_agents, num_individual_actions_features])
        # to get the log probability for the entire multi-categorical action.
        log_prob_summed = log_prob_per_choice.sum(dim=(-1, -2)) # Sum over num_individual_actions_features and num_agents

        # Need to unsqueeze the output to match the expected shape [..., 1] for log_prob
        return log_prob_summed.unsqueeze(-1)

    def deterministic_sample(self) -> torch.Tensor:
        """
        Returns a deterministic sample from the distribution (the mode).
        This method is explicitly added to satisfy ProbabilisticActor in evaluation mode.
        """
        # print("MultiCategorical.deterministic_sample called.") # Debug print
        return self.mode # Return the mode


# Maker function for ProbabilisticActor
# Re-defining the maker function here to ensure it uses the MultiCategorical defined in this cell.
# The maker function should return a callable that takes the output `spec` and `**dist_kwargs`.
# The `nvec` should be implicitly defined in the `action_spec` within the output `spec`.
def multi_categorical_maker():
    """
    A maker function to create the MultiCategorical distribution for ProbabilisticActor.
    This function returns a callable that takes the output `spec` and `**dist_kwargs`
    and returns a MultiCategorical instance.
    The `action_spec` within the output `spec` is expected to be a CompositeSpec
    containing the DiscreteTensorSpec for the action at ('agents', 'action').
    The `logits` in `dist_kwargs` can be flattened or have the expected MultiCategorical shape.
    """
    # The maker function should return a callable that takes the distribution parameters
    # and returns a Distribution instance.
    # The ProbabilisticActor will pass dist_kwargs derived from dist_keys and the output `spec`
    # to this callable.

    def get_multi_categorical_with_spec_and_kwargs(spec: CompositeSpec = None, **dist_kwargs) -> MultiCategorical:
         if 'logits' not in dist_kwargs:
             raise ValueError("multi_categorical_maker expects 'logits' in dist_kwargs")
         if spec is None:
              raise ValueError("multi_categorical_maker requires 'spec' argument from ProbabilisticActor")

         logits = dist_kwargs['logits']

         # Get the action spec from the overall output spec
         # This should be a CompositeSpec containing the DiscreteTensorSpec at ('agents', 'action')
         if ('agents', 'action') not in spec.keys(include_nested=True):
              raise ValueError("Action spec not found at ('agents', 'action') in the provided spec")

         # Access the nested DiscreteTensorSpec within the CompositeSpec
         action_spec_composite = spec[('agents', 'action')]
         if ('agents', 'action') not in action_spec_composite.keys(include_nested=True):
             raise ValueError("Nested action spec not found at ('agents', 'action') within the CompositeSpec")

         action_spec = action_spec_composite[('agents', 'action')]


         # The action_spec should be a DiscreteTensorSpec for MultiCategorical
         if not isinstance(action_spec, DiscreteTensorSpec):
              raise TypeError(f"Expected nested action spec to be DiscreteTensorSpec, but got {type(action_spec)}")


         # Infer shapes from action_spec to potentially reshape flattened logits
         action_spec_shape = action_spec.shape
         action_spec_nvec = action_spec.nvec

         if len(action_spec_shape) < 1: # Corrected check, should have at least one dimension for features
              raise ValueError(f"Unexpected action_spec shape during maker inference: {action_spec_shape}")
         # nvec should have the same number of dimensions as action_spec shape
         if len(action_spec_nvec.shape) != len(action_spec_shape):
              raise ValueError(f"Action_spec nvec shape {action_spec_nvec.shape} must match action_spec shape {action_spec_shape}.")

         # Infer shapes based on the action_spec dimensions
         # The action spec shape is [..., num_agents, num_individual_actions_features]
         num_agents = action_spec_shape[-2]
         num_individual_actions_features = action_spec_shape[-1]

         unique_categories = torch.unique(action_spec_nvec)
         if len(unique_categories) != 1:
              raise ValueError(f"Expected all category counts in action_spec.nvec to be the same, but found {unique_categories}")
         num_action_categories = unique_categories.item()

         # Check if logits need reshaping (if they are flattened)
         # The logits shape should match the batch shape of the spec + [num_agents, num_individual_actions_features, num_action_categories]
         # The spec's batch shape is its shape excluding its event shape [num_agents, num_individual_actions_features]
         spec_batch_shape = action_spec_shape[:-2] # Exclude num_agents and num_individual_actions_features

         expected_non_flattened_shape = spec_batch_shape + torch.Size([num_agents, num_individual_actions_features, num_action_categories])
         expected_flattened_size = num_agents * num_individual_actions_features * num_action_categories


         if logits.shape[-1] == expected_flattened_size and logits.shape[:-1] == spec_batch_shape:
              # Logits are flattened, reshape them.
              reshaped_logits = logits.view(expected_non_flattened_shape)
              # print(f"Debug maker: Reshaped flattened logits to: {reshaped_logits.shape}") # Debug print
         elif logits.shape == expected_non_flattened_shape:
              # Logits already have the expected shape
              reshaped_logits = logits
              # print(f"Debug maker: Logits already have expected shape: {reshaped_logits.shape}") # Debug print
         else:
              raise ValueError(f"Logits shape mismatch in multi_categorical_maker. Expected shape {expected_non_flattened_shape} or flattened size {expected_flattened_size} with batch shape {spec_batch_shape}, but got logits shape {logits.shape}.")


         # Now initialize MultiCategorical with the (potentially reshaped) logits and the extracted action_spec
         return MultiCategorical(logits=reshaped_logits, action_spec=action_spec)


    # Return the inner callable
    return get_multi_categorical_with_spec_and_kwargs


class FlattenObservationTransformer(TensorDictModule):
    """
    A TensorDictModule to flatten the observation ('agents', 'data', 'x')
    from shape [..., num_agents, node_feature_dim] to [..., num_agents * node_feature_dim].
    """
    def __init__(self, num_agents, node_feature_dim, in_keys, out_keys):
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.num_agents = num_agents
        self.node_feature_dim = node_feature_dim

    def forward(self, tensordict):
        # Get the observation data
        x = tensordict.get(('agents', 'data', 'x'))
        if x is None:
            raise ValueError("Input tensordict must contain ('agents', 'data', 'x')")

        # Flatten the last two dimensions
        # Shape before: [..., num_agents, node_feature_dim]
        # Shape after: [..., num_agents * node_feature_dim]
        flattened_x = x.view(x.shape[:-2] + (-1,))

        # Update the tensordict with the flattened observation
        tensordict.set(('agents', 'data', 'x_flattened'), flattened_x)

        # Optionally, remove the original non-flattened key if no longer needed
        # tensordict.pop(('agents', 'data', 'x'))

        return tensordict

class FlattenActionTransformer(TensorDictModule):
    """
    A TensorDictModule to flatten the action ('agents', 'action')
    from shape [..., num_agents, num_individual_actions_features] to [..., num_agents * num_individual_actions_features].
    """
    def __init__(self, num_agents, num_individual_actions_features, in_keys, out_keys):
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.num_agents = num_agents
        self.num_individual_actions_features = num_individual_actions_features

    def forward(self, tensordict):
        # Get the action data
        action = tensordict.get(('agents', 'action'))
        if action is None:
            raise ValueError("Input tensordict must contain ('agents', 'action')")

        # Flatten the last two dimensions
        # Shape before: [..., num_agents, num_individual_actions_features]
        # Shape after: [..., num_agents * num_individual_actions_features]
        flattened_action = action.view(action.shape[:-2] + (-1,))

        # Update the tensordict with the flattened action
        tensordict.set(('agents', 'action_flattened'), flattened_action)

        # Optionally, remove the original non-flattened key if no longer needed
        # tensordict.pop(('agents', 'action'))

        return tensordict

# Example usage (assuming env is defined and has num_agents and node_feature_dim attributes):
# if env is not None:
#     num_agents = env.num_agents
#     node_feature_dim = env.node_feature_dim
#     num_individual_actions_features = env.num_individual_actions_features # Assuming this is defined in env

#     obs_transformer = FlattenObservationTransformer(
#         num_agents=num_agents,
#         node_feature_dim=node_feature_dim,
#         in_keys=[('agents', 'data', 'x')],
#         out_keys=[('agents', 'data', 'x_flattened')]
#     )

#     action_transformer = FlattenActionTransformer(
#          num_agents=num_agents,
#          num_individual_actions_features=num_individual_actions_features,
#          in_keys=[('agents', 'action')],
#          out_keys=[('agents', 'action_flattened')]
#     )

#     # You can now use these transformers in your policy or value network modules
#     # or as part of a sequence of TensorDictModules.
