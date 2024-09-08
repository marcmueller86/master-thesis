import pandas as pd
import matplotlib.pyplot as plt
import os

# person_campaign_id


def get_metrics(df, identifier, use_case):

    total_rows = len(df)

    # 2. Number of unique customers identified by `company_id_telesales_campaign_id`
    unique_customers = df[identifier].nunique()
    # 3. Group by `company_id_telesales_campaign_id` to calculate various statistics
    print("start grouping 1 ... ")
    grouped = df.groupby(identifier).size()

    # 4. Minimum events per customer
    min_events_per_customer = grouped.min()

    # 5. Maximum events per customer
    max_events_per_customer = grouped.max()

    # 6. Average events per customer
    avg_events_per_customer = grouped.mean()

    # 7. Median events per customer
    median_events_per_customer = grouped.median()

    print("start grouping 2 ... ")

    type_counts_overall = df['TYPE'].value_counts().reset_index()

    # Rename columns for clarity
    type_counts_overall.columns = ['TYPE', 'count']

    min_type = type_counts_overall.loc[type_counts_overall['count'].idxmin()]
    max_type = type_counts_overall.loc[type_counts_overall['count'].idxmax()]
    print(f"Usecase: {use_case}")

    # Display the result
    print("TYPE with the minimum overall occurrences:")
    print(
        f"TYPE: {min_type['TYPE']}, Count: {min_type['count']} Percentage: {min_type['count']/total_rows}")

    print("\nTYPE with the maximum overall occurrences:")
    # Rename the columns for clarity
    print(
        f"TYPE: {max_type['TYPE']}, Count: {max_type['count']} Percentage: {max_type['count']/total_rows}")

    # Display the calculated statistics
    print(f"Total number of rows: {total_rows}")
    print(f"Number of unique customers: {unique_customers}")
    print(f"Minimum events per customer: {min_events_per_customer}")
    print(f"Maximum events per customer: {max_events_per_customer}")
    print(f"Average events per customer: {avg_events_per_customer:.2f}")
    print(f"Median events per customer: {median_events_per_customer}")

    # Create the histogram
    plt.figure(figsize=(16, 9))
    plt.hist(grouped, bins=20, edgecolor='black', alpha=0.7)
    plt.yscale('log')
    plt.title(f'Distribution of Events per Customer for {use_case}')
    plt.xlabel('Number of Events')
    plt.ylabel('Number of Customers')

    # Define the output file path
    output_file_path = f"res/histo_{use_case}_type.png"

    # Save the histogram as a PNG file
    plt.savefig(output_file_path)

    # Close the plot to avoid display in non-interactive environments
    plt.close()

    print(f"Histogram saved to {output_file_path}")
    print("##########################")


# Path to the Parquet file on local disk
use_cases = ["print", "telesales", "webinar"]

for use_case in use_cases:
    print(f"processing {use_case}")

    file_path = f"res/{use_case}_all_events_neural_ode_raw.parquet"
    file_path_sampled = f"res/{use_case}_cutoff_dates_sampled.csv"

    # Load the Parquet file into a DataFrame
    df = pd.read_parquet(file_path)

    # Ensure that the columns `person_id` and `campaign_id` exist in your DataFrame
    # Create the new column `person_id_campaign_id`
    if 'telesales' in use_case:
        identifier = 'company_campaign_id'

        df[identifier] = df['company_id'].astype(
            str) + '_' + df['telesales_campaign_id'].astype(str)
        df_sampled = pd.read_csv(file_path_sampled)
        print(df_sampled.head())
        print(df.head())

        df = df[df[identifier].isin(df_sampled[identifier])]

        get_metrics(df, identifier, use_case)
    if 'webinar' in use_case:
        identifier = 'primary_key'
        df_sampled = pd.read_csv(file_path_sampled)
        df_sampled.rename(
            columns={'email_webinar_id': 'primary_key'}, inplace=True)

        print(df_sampled.head())
        print(df.head())

        df = df[df[identifier].isin(df_sampled[identifier])]
        get_metrics(df, identifier, use_case)

    if 'print' in use_case:
        identifier = 'person_campaign_id'
        df[identifier] = df['person_id'].astype(
            str) + '_' + df['campaign_id'].astype(str)
        df_sampled = pd.read_csv(file_path_sampled)
        print(df_sampled.head())
        print(df.head())

        df = df[df[identifier].isin(df_sampled[identifier])]

        get_metrics(df, identifier, use_case)

    # Split the DataFrame into two based on the target column
    df_target_1 = df[df['target'] == 1].reset_index(drop=True)
    df_target_0 = df[df['target'] == 0].reset_index(drop=True)

    # Display the first few rows of the split DataFrames
    print("DataFrame with target=1:")
    print(len(df_target_1))

    print("\nDataFrame with target=0:")
    print(len(df_target_0))

    df_target_1.to_parquet(f"res/{use_case}_target_1.parquet")
    df_target_0.to_parquet(f"res/{use_case}_target_0.parquet")
