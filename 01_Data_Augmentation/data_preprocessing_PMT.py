import polars as pl
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import yaml
import time
from sklearn.model_selection import train_test_split

# Load configuration
config_path = Path('01_Data_Augmentation/250901-base_config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

NUM_EVENTS = config.get('NUM_EVENTS')
if NUM_EVENTS == -1:
    NUM_EVENTS = 200000
NUM_EVENTS_PMT = config.get('NUM_EVENTS_PMT', -1)
TRAIN_TEST_SPLIT = config.get('TRAIN_TEST_SPLIT', 0.1)

def extract_sensor_sequences(meta_df, data_df, label, num_events, num_events_pmt):
    meta_df = meta_df.head(num_events)
    event_ids = meta_df['event_id'].to_numpy()
    data_df = data_df.filter(pl.col('event_id').is_in(event_ids))
    # Normalize columns
    data_df = data_df.with_columns([
        ((pl.col('time') - pl.col('time').mean()) / pl.col('time').std()).alias('time'),
        (pl.col('charge').log1p()).alias('charge')
    ])
    data_df = data_df.with_columns([
        ((pl.col('charge') - pl.col('charge').mean()) / pl.col('charge').std()).alias('charge')
    ])
    # Group by event_id and sensor_id
    groups = data_df.group_by(['event_id', 'sensor_id'])
    sequences = []
    labels = []
    event_sensor_ids = []
    count = 0
    for (event_id, sensor_id), group in groups:
        pulses = group.select(['time', 'charge', 'auxiliary']).to_numpy()
        if pulses.shape[0] > 0:
            sequences.append(pulses.astype(np.float32))
            labels.append(label)
            event_sensor_ids.append((event_id, sensor_id))
            count += 1
        if num_events_pmt != -1 and count >= num_events_pmt:
            break
    return sequences, labels, event_sensor_ids

def preprocess_data(meta_path, data_path, original_meta_path, original_data_path, output_paths, eventid_paths):
    print("Loading data...")
    meta_df = pl.read_parquet(meta_path)
    data_df = pl.read_parquet(data_path)
    original_meta_df = pl.read_parquet(original_meta_path)
    original_data_df = pl.read_parquet(original_data_path)

    print("Extracting original sequences...")
    start_time = time.time()
    orig_sequences, orig_labels, orig_event_sensor_ids = extract_sensor_sequences(
        original_meta_df, original_data_df, label=0, num_events=NUM_EVENTS, num_events_pmt=NUM_EVENTS_PMT
    )
    end_time = time.time()
    print(f"Extracted {len(orig_sequences)} original sequences in {end_time - start_time:.2f} seconds")

    print("Extracting augmented sequences...")
    start_time = time.time()
    aug_sequences, aug_labels, aug_event_sensor_ids = extract_sensor_sequences(
        meta_df, data_df, label=1, num_events=NUM_EVENTS, num_events_pmt=NUM_EVENTS_PMT
    )
    end_time = time.time()
    print(f"Extracted {len(aug_sequences)} augmented sequences in {end_time - start_time:.2f} seconds")

    # Concatenate
    sequences = orig_sequences + aug_sequences
    labels = orig_labels + aug_labels
    event_sensor_ids = orig_event_sensor_ids + aug_event_sensor_ids

    # Train-test split
    X_train, X_test, y_train, y_test, event_sensor_ids_train, event_sensor_ids_test = train_test_split(
        sequences, labels, event_sensor_ids, test_size=TRAIN_TEST_SPLIT, random_state=42
    )

    # Train-validation split
    X_train, X_val, y_train, y_val, event_sensor_ids_train, event_sensor_ids_val = train_test_split(
        X_train, y_train, event_sensor_ids_train, test_size=TRAIN_TEST_SPLIT, random_state=42
    )

    # Save sequences and labels
    print("Saving preprocessed data...")
    with open(output_paths[0], 'wb') as f:
        pickle.dump({'sequences': X_train, 'labels': y_train}, f)
    with open(output_paths[1], 'wb') as f:
        pickle.dump({'sequences': X_val, 'labels': y_val}, f)
    with open(output_paths[2], 'wb') as f:
        pickle.dump({'sequences': X_test, 'labels': y_test}, f)
    # Save event_sensor_ids for each split
    with open(eventid_paths[0], 'wb') as f:
        pickle.dump(event_sensor_ids_train, f)
    with open(eventid_paths[1], 'wb') as f:
        pickle.dump(event_sensor_ids_val, f)
    with open(eventid_paths[2], 'wb') as f:
        pickle.dump(event_sensor_ids_test, f)
    print(f"Preprocessed {len(X_train)} training sequences saved to {output_paths[0]}")
    print(f"Preprocessed {len(X_val)} validation sequences saved to {output_paths[1]}")
    print(f"Preprocessed {len(X_test)} testing sequences saved to {output_paths[2]}")
    print(f"Event Sensor IDs for test set saved to {eventid_paths[2]}")

if __name__ == "__main__":
    preprocess_data(
        meta_path="data/filtered_meta_batch_1.parquet",
        data_path="data/augmented_data_df.parquet",
        original_meta_path="data/train_meta_batch_1.parquet",
        original_data_path="data/batch_1.parquet",
        output_paths=("data/preprocessed_sequences_PMT_train.pkl", "data/preprocessed_sequences_PMT_val.pkl", "data/preprocessed_sequences_PMT_test.pkl"),
        eventid_paths=("data/event_ids_PMT_train.pkl", "data/event_ids_PMT_val.pkl", "data/event_ids_PMT_test.pkl")
    )