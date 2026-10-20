import pandas as pd
import numpy as np

def preprocess_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    # 1. Handle Duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_count - len(df)} duplicate rows.")

    # 2. Handle Missing Values
    # Filling numeric missing values with median and categorical with mode
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    print("Missing values handled.")

    # 3. Datetime Parsing and Feature Engineering
    datetime_cols = [
        'scheduled_departure', 'actual_departure',
        'scheduled_port_arrival', 'actual_port_arrival',
        'scheduled_final_arrival', 'actual_final_delivery'
    ]

    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col])

    # Feature: Departure Month
    df['departure_month'] = df['scheduled_departure'].dt.month

    # Feature: Departure Weekday (0=Monday, 6=Sunday)
    df['departure_weekday'] = df['scheduled_departure'].dt.weekday

    # Feature: Scheduled Travel Duration (Hours)
    # Duration from scheduled departure to scheduled final arrival
    df['scheduled_duration_hours'] = (df['scheduled_final_arrival'] - df['scheduled_departure']).dt.total_seconds() / 3600

    print("Datetime features created.")

    # 4. Prevent Data Leakage
    # Target variable
    target = 'final_delay_hours'

    # Columns that contain information known ONLY after the shipment has arrived/been delivered
    # These must be removed to ensure the model predicts based on available data at departure/during transit
    leakage_cols = [
        'actual_departure',
        'actual_port_arrival',
        'actual_final_delivery',
        'actual_port_processing_hours',
        'actual_customs_clearance'
    ]

    # We keep 'final_delay_hours' as our target, but remove the others
    cols_to_drop = [col for col in leakage_cols if col in df.columns]
    df_processed = df.drop(columns=cols_to_drop)

    print(f"Removed leakage columns: {cols_to_drop}")

    # 5. Save Processed Data
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(f"Final dataset shape: {df_processed.shape}")

if __name__ == "__main__":
    INPUT_FILE = 'data/shipments.csv'
    OUTPUT_FILE = 'data/processed_shipments.csv'
    preprocess_data(INPUT_FILE, OUTPUT_FILE)
