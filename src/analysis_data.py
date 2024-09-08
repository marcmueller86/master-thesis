import pandas as pd
import json
import IPython


# Define the file paths and load the datasets
file_info = {
    "telesales": {
        "train": "res/tm_train_data_sagemaker.csv",
        "test": "res/tm_test_data_sagemaker.csv",
        "summary": "res/tm_metrics_xgboost_test_summary.json"
    },
    "print": {
        "train": "res/print_train_data_sagemaker.csv",
        "test": "res/print_test_data_sagemaker.csv",
        "summary": "res/print_metrics_xgboost_test_summary.json"
    },
    "webinar": {
        "train": "res/webinar_train_data_sagemaker.csv",
        "test": "res/webinar_test_data_sagemaker.csv",
        "summary": "res/webinar_metrics_xgboost_test_summary.json"
    }
}

# Initialize a list to hold the rows for the LaTeX table
metrics_table = []
duplicates_table = []

# Process each use case
for use_case, paths in file_info.items():
    # Load the train and test datasets
    train_df = pd.read_csv(paths["train"])
    test_df = pd.read_csv(paths["test"])
    
    # Exclude 'target_date' and 'customer_id' columns for duplicate check
    if 'telesales' in use_case:
        features_train = train_df.drop(columns=['identifier', 'telesales_campaign_id', 'target'])
        features_test = test_df.drop(columns=['identifier', 'telesales_campaign_id', 'target'])
    if 'print' in use_case:
        features_train = train_df.drop(columns=['identifier', 'person_campaign_id', 'target'])
        features_test = test_df.drop(columns=['identifier', 'person_campaign_id', 'target'])
    if 'webinar' in use_case:
        print (train_df.columns)
        features_train = train_df.drop(columns=['identifier', 'webinarid', 'target'])
        features_test = test_df.drop(columns=['identifier', 'webinarid', 'target'])

    # Calculate duplicate counts
    duplicate_counts_train = features_train.duplicated(keep=False)
    duplicate_counts_test = features_test.duplicated(keep=False)

    # Count unique rows that have duplicates
    train_with_duplicates = features_train[duplicate_counts_train].drop_duplicates().shape[0]
    test_with_duplicates = features_test[duplicate_counts_test].drop_duplicates().shape[0]

    # Count rows that have duplicates occurring more than 10 times
    train_duplicate_more_than_10 = features_train[duplicate_counts_train].groupby(list(features_train.columns)).size().gt(10).sum()
    test_duplicate_more_than_10 = features_test[duplicate_counts_test].groupby(list(features_test.columns)).size().gt(10).sum()

    # Load the summary metrics
    with open(paths["summary"], 'r') as f:
        summary_metrics = json.load(f)

    # Calculate the number of positive and negative examples in the train and test sets
    train_target_counts = train_df['target'].value_counts()
    test_target_counts = test_df['target'].value_counts()

    # Extract relevant metrics from the summary
    total_train = summary_metrics['train_size']['0']
    total_test = summary_metrics['test_size']['0']
    f1 = summary_metrics['f1_grid_test']['0']
    balanced_acc = summary_metrics['balanced_acc_grid_test']['0']

    # Confusion matrix can be used to infer the counts directly from the test set
    confusion_matrix = summary_metrics['confusion_matrix']['0']
    test_positives = confusion_matrix[1][1] + confusion_matrix[1][0]
    test_negatives = confusion_matrix[0][0] + confusion_matrix[0][1]

    # Construct a row for the metrics LaTeX table
    metrics_row = f"{use_case} & {train_target_counts.get(1, 0)} & {train_target_counts.get(0, 0)} & {total_train} & {test_positives} & {test_negatives} & {total_test} & {f1:.2f} & {balanced_acc:.2f} \\\\"
    metrics_table.append(metrics_row)

    # Total number of events in train and test datasets
    total_train_events = len(features_train)
    total_test_events = len(features_test)

    # Calculate percentages
    percent_with_duplicates_train = (train_with_duplicates / total_train_events) * 100
    percent_with_duplicates_test = (test_with_duplicates / total_test_events) * 100
    
    percent_more_than_10_train = (train_duplicate_more_than_10 / total_train_events) * 100
    percent_more_than_10_test = (test_duplicate_more_than_10 / total_test_events) * 100

    # Construct a row for the duplicates LaTeX table, including both percentage and count
    duplicates_row = (f"{use_case} & "
                      f"{total_train_events} & "
                      f"{percent_with_duplicates_train:.2f}\\% | {train_with_duplicates} & "
                      f"{percent_more_than_10_train:.2f}\\% | {train_duplicate_more_than_10} & "
                      f"{total_test_events} & "
                      f"{percent_with_duplicates_test:.2f}\\% | {test_with_duplicates} & "
                      f"{percent_more_than_10_test:.2f}\\% | {test_duplicate_more_than_10} \\\\")
    duplicates_table.append(duplicates_row)

# Convert the metrics list to a LaTeX table format
metrics_table_str = "\\begin{tabular}{lcccccccc}\n"
metrics_table_str += "Use Case & Train (1) & Train (0) & Total Train & Test (1) & Test (0) & Total Test & F1 Score & Balanced Accuracy \\\\\n"
metrics_table_str += "\\hline\n"
metrics_table_str += "\n".join(metrics_table)
metrics_table_str += "\n\\end{tabular}"

# Convert the duplicates list to a LaTeX table format
duplicates_table_str = "\\begin{tabular}{lcccccc}\n"
duplicates_table_str += "Use Case & Total Train & Train Duplicates (≥1) & Train Duplicates (≥10) & Total Test & Test Duplicates (≥1) & Test Duplicates (≥10) \\\\\n"
duplicates_table_str += "\\hline\n"
duplicates_table_str += "\n".join(duplicates_table)
duplicates_table_str += "\n\\end{tabular}"

# Print the LaTeX tables
print(metrics_table_str)
print("\n")
print(duplicates_table_str)