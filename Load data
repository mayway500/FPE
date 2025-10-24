#@title Load data
from google.colab import drive
# drive.mount('/content/drive') # Removed redundant mount call
from scipy.stats import zscore

import pandas as pd
import numpy as np
import os
import torch


def load_and_preprocess_data(file_path, device):
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}. Please ensure it was downloaded.")
        # Do not raise FileNotFoundError here, just return None to allow the rest of the notebook to run
        return None

    try:
        # Read the CSV, ensuring the correct Date column is used as index
        # The previous data download step named the date column "('Date',)"
        data = pd.read_csv(file_path, index_col="('Date',)", parse_dates=True)
        print(f"Successfully loaded data from {file_path}")
    except Exception as e:
         print(f"Error loading data: {e}")
         # Return None if loading fails
         return None

    # Ensure index is datetime and drop rows with NaT index (should be handled by parse_dates)
    data.index = pd.to_datetime(data.index)
    data = data[data.index.notna()]

    # Select all columns except the index as features
    feature_columns = data.columns.tolist()
    feature_data = data[feature_columns].copy()

    # Impute missing values with the mean of each column
    mean_values = feature_data.mean()
    feature_data = feature_data.fillna(mean_values)
    print("NaN values replaced with column means.")

    # Handle outliers using Z-score
    numeric_cols_feature_data = feature_data.select_dtypes(include=np.number)
    # Calculate Z-scores only for numeric columns
    abs_z_scores_feature_data = numeric_cols_feature_data.apply(lambda x: np.abs(zscore(x, ddof=0)) if x.std() != 0 else pd.Series(0, index=x.index))
    threshold = 3
    outliers_feature_data = abs_z_scores_feature_data > threshold
    # Replace outliers with 0 in the original feature_data DataFrame
    feature_data.loc[:, numeric_cols_feature_data.columns][outliers_feature_data] = 0
    print("Outliers replaced with 0.")

    # Calculate the change in features between consecutive time steps
    # This will be used as the 'actual_changes' for the reward function
    # Use .shift(1) to get the previous row, and fill NaNs in the first row with 0
    feature_changes = feature_data.diff().fillna(0)
    print("Calculated feature changes.")
    # Ensure the change data has the same column names as the feature data, or use a different naming convention if needed.
    # For now, assuming the order matches and we just need the numerical values.


    # Combine the original features with the calculated changes
    # The combined data will have shape [num_samples, node_feature_dim + num_action_features]
    # where the first node_feature_dim columns are the features (observations)
    # and the next num_action_features columns are the changes (for reward calculation)
    # Assuming node_feature_dim == num_action_features (which is 13 based on previous cells)
    # The number of features is 13, regardless of the number of agents.
    node_feature_dim = 13 # Hardcoded based on the data structure
    num_action_features = 13 # Hardcoded based on the data structure

    if feature_data.shape[1] != node_feature_dim:
         print(f"Warning: Feature data has {feature_data.shape[1]} columns, expected {node_feature_dim}.")
         # Adjust feature_data to only include the first node_feature_dim columns if it has more
         if feature_data.shape[1] > node_feature_dim:
             print(f"Truncating feature_data to first {node_feature_dim} columns.")
             feature_data = feature_data.iloc[:, :node_feature_dim]
         # If it has less, padding might be needed or an error should be raised depending on expected behavior.
         # For now, assuming 13 features is the correct input size.

    if feature_changes.shape[1] != num_action_features:
        print(f"Warning: Feature changes has {feature_changes.shape[1]} columns, expected {num_action_features}.")
        # Adjust feature_changes similarly
        if feature_changes.shape[1] > num_action_features:
            print(f"Truncating feature_changes to first {num_action_features} columns.")
            feature_changes = feature_changes.iloc[:, :num_action_features]


    # Concatenate feature_data and feature_changes along axis 1 (columns)
    # Make sure to align based on index (Date)
    combined_processed_data = pd.concat([feature_data, feature_changes], axis=1, join='inner') # Use inner join to drop any non-matching dates (shouldn't happen with diff())
    print(f"Combined feature data and changes. Shape: {combined_processed_data.shape}")

    # Save the combined_processed_data DataFrame to Google Drive
    output_csv_path = '/content/drive/MyDrive/deep learning codes/EIAAPI_DOWNLOAD/solutions/mergedata/combined_processed_data.csv'
    try:
        combined_processed_data.to_csv(output_csv_path, index=True) # Include index (Date) in the CSV
        print(f"Saved combined_processed_data to {output_csv_path}")
    except Exception as e:
        print(f"Error saving combined_processed_data to Google Drive: {e}")


    # Convert the combined processed data to a numpy array
    processed_data_np = combined_processed_data.values

    # Check for NaNs or Infs in numpy array after processing
    if np.isnan(processed_data_np).any() or np.isinf(processed_data_np).any():
        print("Warning: NaNs or Infs found in processed_data numpy array before tensor conversion.")
        processed_data_np = np.nan_to_num(processed_data_np, nan=0.0, posinf=0.0, neginf=0.0)
        print("Remaining NaNs and Infs replaced with 0 in numpy array.")

    # Update node_feature_dim to reflect the number of actual features used as observation
    # This should still be 13, as we are only using the first 13 columns as observation features in the environment.
    # The total number of columns in the tensor will be node_feature_dim + num_action_features (13 + 13 = 26)
    # but the environment's observation spec and module inputs should still only expect node_feature_dim (13) features per node.
    # The extra columns for changes are only used internally by the _batch_reward function.
    print(f"  Shape of processed_data (numpy): {processed_data_np.shape}")

    print(f"Attempting to convert processed_data to torch tensor on device {device}...")
    try:
        data_tensor = torch.tensor(processed_data_np, dtype=torch.float32, device=device)
        print("Conversion to torch tensor successful.")
    except Exception as e:
        print(f"Error converting data to torch tensor: {e}")
        # Return None if tensor conversion fails
        return None

    # Final check for NaNs or Infs in the PyTorch tensor
    if torch.isnan(data_tensor).any() or torch.isinf(data_tensor).any():
        print("Warning: NaNs or Infs found in the PyTorch tensor after conversion.")
        # This should ideally not happen if np.nan_to_num was effective, but as a safeguard:
        data_tensor = torch.nan_to_num(data_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        print("Remaining NaNs and Infs in tensor replaced with 0.")
    else:
        print("No NaNs or Infs found in the PyTorch tensor.")

    return data_tensor

# Example usage:
DATA_FILE_PATH = '/content/drive/MyDrive/deep learning codes/EIAAPI_DOWNLOAD/solutions/mergedata/Cleaneddata.csv'
# Assuming 'device' is defined in a previous cell (e.g., the hyperparameters cell)
# If not, define it here:
try:
    _ = device
except NameError:
    device = 'cpu'
    print(f"Using default device: {device} as 'device' was not defined.")

print("Loading and preprocessing data using the defined function...")
try:
    data_tensor = load_and_preprocess_data(DATA_FILE_PATH, device)
    if data_tensor is not None:
        print("Data loading and preprocessing complete.")
        # The shape should now be [num_samples, 2 * node_feature_dim] = [num_samples, 26]
        print(f"Resulting data_tensor shape: {data_tensor.shape}, dtype: {data_tensor.dtype}, device: {data_tensor.device}")
    else:
        print("Data loading or preprocessing failed. data_tensor is None.")

except Exception as e:
    print(f"An error occurred during data loading or preprocessing: {e}")
    data_tensor = None

# Ensure num_action_features is defined if needed for subsequent cells
try:
    _ = num_action_features
except NameError:
    # Assuming num_action_features is equal to node_feature_dim based on the reward function logic
    num_action_features = 13 # Or retrieve from a hyperparameters cell if available
    print(f"Using default num_action_features: {num_action_features} as it was not defined.")
