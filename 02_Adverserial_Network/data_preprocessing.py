import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm  # Import tqdm for progress bar
import yaml
import time
from sklearn.model_selection import train_test_split

# Load configuration
config_path = Path('01_Data_Augmentation/250901-base_config.yaml')
config2_path = Path('02_Adverserial_Network/250902-config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
with open(config2_path, 'r') as f:
    config2 = yaml.safe_load(f)

# Number of events to process
NUM_EVENTS = config.get('NUM_EVENTS')
if NUM_EVENTS == -1:
    NUM_EVENTS = 200000

def extract_sequences(meta_df, data_df, label, num_events):
    # Filter meta_df to first num_events
    meta_df = meta_df.head(num_events)
    event_ids = meta_df['event_id'].values

    # Filter data_df to only relevant events
    data_df = data_df[data_df['event_id'].isin(event_ids)]

    # Group by event_id and extract pulse arrays
    grouped = data_df.groupby('event_id')
    sequences = []
    labels = []
    for event_id, group in grouped:
        pulses = group[['sensor_id', 'time', 'charge', 'auxiliary']].astype({
            'sensor_id': np.int16,
            'time': np.float64,
            'charge': np.float64,
            'auxiliary': np.int8  # Mapping: True -> 1, False -> 0
        }).values
        if len(pulses) > 0:
            sequences.append(pulses)
            labels.append(label)
    return sequences, labels

def preprocess_data(meta_path, data_path, original_meta_path, original_data_path, output_paths):
    # Load data
    print("Loading data...")
    meta_df = pd.read_parquet(meta_path)
    data_df = pd.read_parquet(data_path)
    original_meta_df = pd.read_parquet(original_meta_path)
    original_data_df = pd.read_parquet(original_data_path)

    # Reset indices to make 'event_id' a column (if it's currently the index)
    data_df.reset_index(inplace=True)
    original_data_df.reset_index(inplace=True)

    print("Normalizing time & charge columns...")
    # Normalize time columns
    for df in [data_df, original_data_df]:
        df['time'] = (df['time'] - df['time'].mean()) / df['time'].std()
        # use log scaling for charge column
        df['charge'] = np.log1p(df['charge'])
        df['charge'] = (df['charge'] - df['charge'].mean()) / df['charge'].std()

    # Vectorized extraction
    print("Extracting sequences...")
    start_time = time.time()
    orig_sequences, orig_labels = extract_sequences(original_meta_df, original_data_df, label=0, num_events=NUM_EVENTS)
    end_time = time.time()
    print(f"Extracted {len(orig_sequences)} original sequences in {end_time - start_time:.2f} seconds")
    aug_sequences, aug_labels = extract_sequences(meta_df, data_df, label=1, num_events=NUM_EVENTS)

    # Concatenate
    sequences = orig_sequences + aug_sequences
    labels = orig_labels + aug_labels

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=config2['TRAIN_TEST_SPLIT'], random_state=42)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=config2['TRAIN_TEST_SPLIT'], random_state=42)

    # Save sequences and labels
    print("Saving preprocessed data...")
    with open(output_paths[0], 'wb') as f:
        pickle.dump({'sequences': X_train, 'labels': y_train}, f)
    print(f"Preprocessed {len(X_train)} training sequences saved to {output_paths[0]}")

    with open(output_paths[1], 'wb') as f:
        pickle.dump({'sequences': X_val, 'labels': y_val}, f)
    print(f"Preprocessed {len(X_val)} validation sequences saved to {output_paths[1]}")

    with open(output_paths[2], 'wb') as f:
        pickle.dump({'sequences': X_test, 'labels': y_test}, f)
    print(f"Preprocessed {len(X_test)} testing sequences saved to {output_paths[2]}")

if __name__ == "__main__":
    preprocess_data(
        meta_path="data/filtered_meta_batch_1.parquet",
        data_path="data/augmented_data_df.parquet",
        original_meta_path="data/train_meta_batch_1.parquet",
        original_data_path="data/batch_1.parquet",
        output_paths=("data/preprocessed_sequences_train.pkl", "data/preprocessed_sequences_val.pkl", "data/preprocessed_sequences_test.pkl")
    )
