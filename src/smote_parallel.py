import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define file paths for all three use cases
file_info = {
    "telesales": {
        "train": "res/tm_train_data_sagemaker.csv",
        "test": "res/tm_test_data_sagemaker.csv",
        "sampling_strategy": lambda n0, n1: {0: n0, 1: n1}
    },
    "print": {
        "train": "res/print_train_data_sagemaker.csv",
        "test": "res/print_test_data_sagemaker.csv",
        "sampling_strategy": lambda n0, n1: {0: n0, 1: n1}
    },
    "webinar": {
        "train": "res/webinar_train_data_sagemaker.csv",
        "test": "res/webinar_test_data_sagemaker.csv",
        "sampling_strategy": lambda n0, n1: {0: n0, 1: n1}
    }
}

# Define the grid search parameters
k_neighbors_range = [5, 10, 15]
noise_levels = [0.01, 0.05, 0.1]

def process_combination(use_case, X, y, original_negative_count, original_positive_count, k_neighbors, noise_level):
    # Initialize SMOTE with the current k_neighbors
    smote = SMOTE(sampling_strategy={0: original_negative_count * 2, 1: original_positive_count * 2}, k_neighbors=k_neighbors)
    
    # Apply SMOTE to generate synthetic data
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Extract synthetic data
    if len(X_resampled) > len(X):
        synthetic_data = X_resampled[len(X):].copy()
        synthetic_targets = y_resampled[len(X):].copy()

        # Introduce Gaussian noise to synthetic data
        noise = np.random.normal(0, noise_level, synthetic_data.shape)
        synthetic_data_noisy = synthetic_data + noise

        # Combine synthetic features and target
        synthetic_data_noisy = pd.concat([synthetic_data_noisy, synthetic_targets], axis=1)
        synthetic_data_noisy.columns = list(X.columns) + ['target']

        # Combine original and noisy synthetic data
        combined_data = pd.concat([X, synthetic_data_noisy.drop(columns=['target'])], ignore_index=True)
        
        # Visualization using t-SNE
        tsne_results = TSNE(n_components=2, random_state=42).fit_transform(combined_data)

        # Calculate Silhouette Score
        kmeans = KMeans(n_clusters=2, random_state=42).fit(tsne_results)
        silhouette_avg = silhouette_score(tsne_results, kmeans.labels_)

        # Calculate Cosine Similarity
        cosine_similarities = cosine_similarity(X, synthetic_data_noisy.drop(columns=['target']))
        max_cosine_sim = cosine_similarities.max(axis=1)
        average_cosine_sim = max_cosine_sim.mean()

        return (silhouette_avg, average_cosine_sim, k_neighbors, noise_level, synthetic_data_noisy, tsne_results)
    else:
        return None

def main():
    # Process each use case
    for use_case, paths in file_info.items():
        print(f"Processing {use_case}...")

        # Load the training data
        train_df = pd.read_csv(paths['train'])
        test_df = pd.read_csv(paths['test'])

        # Define features and target
        target_column = 'target'
        if 'telesales' in use_case:
            X = train_df.drop(columns=[target_column, 'identifier', 'telesales_campaign_id'])

        if 'print' in use_case:
            X = train_df.drop(columns=[target_column, 'identifier', 'campaign_id'])

        if 'webinar' in use_case:
            X = train_df.drop(columns=[target_column, 'identifier', 'webinarid'])
        
        y = train_df[target_column]

        # Capture the original counts of positive and negative samples
        original_positive_count = sum(y == 1)
        original_negative_count = sum(y == 0)
        
        # Verbose output before SMOTE
        print(f"Original dataset size: {X.shape[0]} samples")
        print(f"Original positive samples: {original_positive_count}, Original negative samples: {original_negative_count}")

        # Initialize variables to track the best results
        best_silhouette_score = -1
        best_cosine_similarity = float('inf')
        best_params = {}
        best_synthetic_data_noisy = None
        best_tsne_results = None

        # Use ProcessPoolExecutor to parallelize the grid search
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = []
            for k_neighbors in k_neighbors_range:
                for noise_level in noise_levels:
                    futures.append(executor.submit(process_combination, use_case, X, y, original_negative_count, original_positive_count, k_neighbors, noise_level))
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    silhouette_avg, average_cosine_sim, k_neighbors, noise_level, synthetic_data_noisy, tsne_results = result
                    # Check if this combination is the best so far
                    if silhouette_avg > best_silhouette_score and average_cosine_sim < best_cosine_similarity:
                        best_silhouette_score = silhouette_avg
                        best_cosine_similarity = average_cosine_sim
                        best_params = {
                            'k_neighbors': k_neighbors,
                            'noise_level': noise_level
                        }
                        best_synthetic_data_noisy = synthetic_data_noisy.copy()
                        best_tsne_results = tsne_results.copy()

        # Output the best parameters and corresponding scores
        print(f"Best Silhouette Score for {use_case}: {best_silhouette_score:.4f}")
        print(f"Best Average Cosine Similarity for {use_case}: {best_cosine_similarity:.4f}")
        print(f"Best Parameters for {use_case}: {best_params}")

        # Save the best synthetic data and TSNE plot based on the optimal parameters
        if best_synthetic_data_noisy is not None:
            config_name = f"k{best_params['k_neighbors']}_noise{best_params['noise_level']}"
            
            # Save the synthetic dataset
            best_synthetic_data_noisy.to_csv(f'res/smote_features_synthetic_{use_case}_{config_name}.csv', index=False)
            
            # Save the TSNE plot
            plt.figure(figsize=(16, 9))
            plt.scatter(best_tsne_results[:len(X), 0], best_tsne_results[:len(X), 1], label='Original', marker='o', alpha=0.7)
            plt.scatter(best_tsne_results[len(X):, 0], best_tsne_results[len(X):, 1], label='Synthetic', marker='x', alpha=0.2)
            plt.legend()
            plt.title(f't-SNE visualization of original and synthetic data ({use_case})\nBest Config: {config_name}')
            plt.savefig(f'res/{use_case}_tsne_{config_name}.png')
            plt.close()

            print(f"Best synthetic data and image for {use_case} saved successfully.")

        print(f"Finished processing {use_case}.\n")

if __name__ == "__main__":
    main()

print("SMOTE processing, grid search, and visualization completed for all use cases.")