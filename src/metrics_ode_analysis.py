import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import OrdinalEncoder
from scipy.spatial.distance import jensenshannon

def measure_jsd_similarity(original_df, synthetic_df, numerical_features, categorical_features, num_bins=30):
    """
    Measure the similarity between the original and synthetic datasets using Jensen-Shannon Divergence.

    Parameters:
    original_df (pd.DataFrame): Original dataset
    synthetic_df (pd.DataFrame): Synthetic dataset
    numerical_features (list): List of numerical features to compare
    categorical_features (list): List of categorical features to compare
    num_bins (int): Number of bins to use for numerical features (default is 30)

    Returns:
    dict: Dictionary with feature names as keys and Jensen-Shannon Divergences as values
    """
    distances = {}

    # Handle numerical features
    for feature in numerical_features:
        original_values = original_df[feature].values
        synthetic_values = synthetic_df[feature].values

        # Discretize the values into bins
        min_val = min(original_values.min(), synthetic_values.min())
        max_val = max(original_values.max(), synthetic_values.max())
        bins = np.linspace(min_val, max_val, num_bins)
        
        original_hist, _ = np.histogram(original_values, bins=bins, density=True)
        synthetic_hist, _ = np.histogram(synthetic_values, bins=bins, density=True)

        # Normalize histograms to get probabilities
        original_hist = original_hist / original_hist.sum()
        synthetic_hist = synthetic_hist / synthetic_hist.sum()

        # Compute JSD
        jsd = jensenshannon(original_hist, synthetic_hist)
        distances[feature] = jsd

    # Handle categorical features
    encoder = OrdinalEncoder()
    for feature in categorical_features:
        original_values = original_df[feature].values.reshape(-1, 1)
        synthetic_values = synthetic_df[feature].values.reshape(-1, 1)
        
        combined_values = np.concatenate([original_values, synthetic_values])
        encoder.fit(combined_values)
        
        original_encoded = encoder.transform(original_values).ravel().astype(int)
        synthetic_encoded = encoder.transform(synthetic_values).ravel().astype(int)
        
        original_probs = np.bincount(original_encoded) / len(original_encoded)
        synthetic_probs = np.bincount(synthetic_encoded) / len(synthetic_encoded)
        
        # Pad the shorter array with zeros
        if len(original_probs) > len(synthetic_probs):
            synthetic_probs = np.pad(synthetic_probs, (0, len(original_probs) - len(synthetic_probs)))
        else:
            original_probs = np.pad(original_probs, (0, len(synthetic_probs) - len(original_probs)))
        
        # Compute JSD
        jsd = jensenshannon(original_probs, synthetic_probs)
        distances[feature] = jsd

    return distances

def measure_similarity(original_df, synthetic_df, features, categorical_features, use_case):
    """
    Measure the similarity between the original and synthetic datasets using Wasserstein distance.

    Parameters:
    original_df (pd.DataFrame): Original dataset
    synthetic_df (pd.DataFrame): Synthetic dataset
    features (list): List of numerical features to compare
    categorical_features (list): List of categorical features to compare

    Returns:
    dict: Dictionary with feature names as keys and Wasserstein distances as values
    """
    distances = {}
    
    # Handle numerical features
    for feature in features:
        original_values = original_df[feature].values
        synthetic_values = synthetic_df[feature].values
        distance = wasserstein_distance(original_values, synthetic_values)
        distances[feature] = distance
    
    # Handle categorical features
    encoder = OrdinalEncoder()
    for feature in categorical_features:
        original_values = original_df[feature].values.reshape(-1, 1)
        synthetic_values = synthetic_df[feature].values.reshape(-1, 1)
        combined_values = np.concatenate([original_values, synthetic_values])
        encoder.fit(combined_values)
        original_encoded = encoder.transform(original_values)
        synthetic_encoded = encoder.transform(synthetic_values)
        distance = wasserstein_distance(original_encoded.ravel(), synthetic_encoded.ravel())
        distances[feature] = distance
    
    return distances


# Example usage
df = pd.read_parquet('data/prefiltered/customer_journey/{use_case}_all_events_neural_ode_raw.parquet')
all_synthetic_data = pd.read_parquet('data/prefiltered/customer_journey/synthetic_selected_customers_events_{use_case}.parquet')
all_synthetic_data.rename(columns={'event_type': 'TYPE'}, inplace=True)

all_synthetic_data['date'] = pd.to_datetime(all_synthetic_data['event_time'])
df['date'] = pd.to_datetime(df['date'])

features_to_compare = ['date', 'amount_net']
categorical_features_to_compare = ['TYPE']

# Wasserstein
similarity_distances = measure_similarity(df, all_synthetic_data, features_to_compare, categorical_features_to_compare)
print(similarity_distances)


# JSD
numerical_features_to_compare = ['amount_net']
categorical_features_to_compare = ['TYPE']

# Assuming `df` is the original dataframe and `all_synthetic_data` is the synthetic dataframe
jsd_distances = measure_jsd_similarity(df, all_synthetic_data, numerical_features_to_compare, categorical_features_to_compare)
print(jsd_distances)