import os
import pandas as pd
import IPython

# Load and process the datasets
use_cases = ['print','telesales', 'webinar']
targets = [0, 1]

for use_case in use_cases:
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame to hold the combined data
    
    # Combine data from both target classes
    for target in targets:
        file_path = f"res/synthetic_selected_customers_events_{use_case}_{target}.csv"
        df = pd.read_csv(file_path)
        df['target'] = target  # Add the target column to distinguish between the two classes
        
        # Add target as a prefix to customer_id
        df['customer_id'] = df['customer_id'].apply(lambda x: f"{target}_{x}")
        
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'], format='%Y-%m-%d %H:%M:%S.%f')

    # Split the combined dataset by event_type and save them to respective directories
    output_base_dir = "res/synthetic"
    
    for event_type, group in combined_df.groupby('event_type'):
        event_dir = os.path.join(output_base_dir, use_case, str(event_type), "filtered_events")
        os.makedirs(event_dir, exist_ok=True)
        output_file = os.path.join(event_dir, f"filtered_events_{event_type}.csv")
        group.to_csv(output_file, index=False)
    
    # Create a DataFrame with the max date per unique customer
    max_date_df = combined_df.groupby('customer_id').agg({'date': 'max'}).reset_index()
    max_date_df['target'] = combined_df.groupby('customer_id')['target'].first().values  # Preserve the target
    
    # Save the max date DataFrame
    max_date_output_path = f"res/synthetic/{use_case}_max_date_target.csv"
    os.makedirs(os.path.dirname(max_date_output_path), exist_ok=True)
    max_date_df.to_csv(max_date_output_path, index=False)
    
    # Display final DataFrames for verification (only showing first few rows)
    print(f"Max date DataFrame for {use_case}:")
    print(max_date_df.head())
    
    # Count the number of occurrences for each event_type
    event_counts = combined_df['event_type'].value_counts()
    
    # Calculate the percentage of total events for each event_type
    total_events = event_counts.sum()
    event_percentages = (event_counts / total_events) * 100
    
    # Identify the event_type with the most and least counts
    most_common_event = event_counts.idxmax()
    least_common_event = event_counts.idxmin()
    
    # Output the event type statistics
    print(f"\nEvent Type Statistics for {use_case}:")
    print(event_counts)
    print("\nEvent Type Percentages:")
    print(event_percentages)
    print(f"\nMost common event type: {most_common_event} with {event_counts[most_common_event]} occurrences ({event_percentages[most_common_event]:.2f}%)")
    print(f"Least common event type: {least_common_event} with {event_counts[least_common_event]} occurrences ({event_percentages[least_common_event]:.2f}%)")
