import boto3
import awswrangler as wr
import pandas as pd
import os

# Initialize boto3 session
session = boto3.Session()
bucket = ""  # Add your bucket name here
# List of event types and use cases
event_types = [
    'activity', 'bank_charge_booked', 'cancellation', 'cancellation_booked', 'contract_created', 
    'conversion', 'dunning_booked', 'invoice_received', 'email_subscribe', 'email_click', 'letter', 
    'payment_booked', 'return_debit_booked', 'return_remittance_booked', 
    'revenue_booked', 'tm_called_on', 'write_off_booked', 'email_unsubscribe', 
    'webinar_attendee', 'webinar_registration'
]
use_cases = ["telesales", "print", "webinar"]

# Directory to store the output files
output_dir = "res"
os.makedirs(output_dir, exist_ok=True)

# Loop over use cases and event types
for use_case in use_cases:
    data_frames = []
    
    for event_type in event_types:
        print (f"Processing {use_case} - {event_type}...")
        s3_path = f"s3://{bucket}/{use_case}/prefiltered_events/train/{event_type}/filtered_{event_type}.parquet"
        
        # Load data from S3 into a DataFrame
        try:
            df = wr.s3.read_parquet(path=s3_path, boto3_session=session)
            data_frames.append(df)
            print(f"Loaded data from {s3_path}")
        except Exception as e:
            print(f"Failed to load data from {s3_path}: {e}")
    
    # Concatenate all DataFrames for the current use case
    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        # Define the output file path
        output_path = os.path.join(output_dir, f"{use_case}_all_events_neural_ode_raw.parquet")
        
        # Store the combined DataFrame to local disk
        combined_df.to_parquet(output_path)
        print(f"Saved combined DataFrame to {output_path}")
    else:
        print(f"No data loaded for use case {use_case}. Skipping file creation.")