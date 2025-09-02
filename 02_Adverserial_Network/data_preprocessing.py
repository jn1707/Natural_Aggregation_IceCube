import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm  # Import tqdm for progress bar
import yaml
import time

# Load configuration
config_path = Path('01_Data_Augmentation/250901-base_config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

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
            'time': np.int64,
            'charge': np.float64,
            'auxiliary': np.int8  # or np.float32
        }).values
        if len(pulses) > 0:
            sequences.append(pulses)
            labels.append(label)
    return sequences, labels

def preprocess_data(meta_path, data_path, original_meta_path, original_data_path, output_path):
    # Load data
    print("Loading data...")
    meta_df = pd.read_parquet(meta_path)
    data_df = pd.read_parquet(data_path)
    original_meta_df = pd.read_parquet(original_meta_path)
    original_data_df = pd.read_parquet(original_data_path)

    # Reset indices to make 'event_id' a column (if it's currently the index)
    data_df.reset_index(inplace=True)
    original_data_df.reset_index(inplace=True)
    
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

    # Save sequences and labels
    print("Saving preprocessed data...")
    with open(output_path, 'wb') as f:
        pickle.dump({'sequences': sequences, 'labels': labels}, f)
    print(f"Preprocessed {len(sequences)} sequences saved to {output_path}")


if __name__ == "__main__":
    preprocess_data(
        meta_path="data/filtered_meta_batch_1.parquet",
        data_path="data/augmented_data_df.parquet",
        original_meta_path="data/train_meta_batch_1.parquet",
        original_data_path="data/batch_1.parquet",
        output_path="data/preprocessed_sequences.pkl"
    )
