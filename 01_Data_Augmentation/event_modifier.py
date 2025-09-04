import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, SelectMultiple
import pyarrow.parquet as pq
from pathlib import Path
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bar
import yaml
import polars as pl
import time

print("Script started.")
script_start_time = time.time()

config_path = Path('01_Data_Augmentation/250901-base_config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Path to data files
meta_path = Path('data/train_meta_batch_1.parquet')
data_path = Path('data/batch_1.parquet')
sensor_geometry_path = Path('data/sensor_geometry.csv')

# Load as Polars DataFrame
meta_df = pl.read_parquet(meta_path)
data_df = pl.read_parquet(data_path)

if 'event_id' not in data_df.columns:
    data_df = data_df.with_row_index('event_id')

# Number of events to process
NUM_EVENTS = config.get('NUM_EVENTS')
if NUM_EVENTS == -1:
    NUM_EVENTS = data_df['event_id'].n_unique()
    
CHARGE_PERC = config['LATE_PULSES']['CHARGE_PERC']

# Load sensor geometry data
sensor_geometry_df = pl.read_csv(sensor_geometry_path)

# Precompute random values (vectorized approach)
np.random.seed(42)
random_poisson_late = np.random.poisson(lam=config['LATE_PULSES']['POISSON_MEAN'], size=NUM_EVENTS)
random_norm_late = np.random.normal(loc=0, scale=config['LATE_PULSES']['CHARGE_NOISE_STD'], size=NUM_EVENTS * 5)  # Extra samples to ensure enough
random_exp_late = np.random.exponential(scale=config['LATE_PULSES']['MEAN_TIME_OFFSET'], size=NUM_EVENTS * 5)  # Extra samples to ensure enough

# Vectorized processing using group_by
def process_late_pulses(group, idx):
    """
    Create late pulses by sampling from existing charges and modifying them.
    """
    num_samples = random_poisson_late[idx]
    if num_samples == 0:
        return pl.DataFrame({
            'sensor_id': [],
            'time': [],
            'charge': [],
            'auxiliary': [],
            'event_id': [],
            'late_pulse': [],
            'after_pulse': [],
            'noise': []
        }, schema={
            'sensor_id': pl.Int16,
            'time': pl.Int64,
            'charge': pl.Float64,
            'auxiliary': pl.Boolean,
            'event_id': pl.Int64,
            'late_pulse': pl.Boolean,
            'after_pulse': pl.Boolean,
            'noise': pl.Boolean
        })
    
    # Sample vectorized
    sampled = group.sample(n=num_samples, seed=42)

    # Extract as numpy arrays for fast numpy computations
    sampled_charges = sampled['charge'].to_numpy()
    sampled_times = sampled['time'].to_numpy()
    sampled_sensor_ids = sampled['sensor_id'].to_numpy()
    sampled_auxiliary = sampled['auxiliary'].to_numpy()
    
    # Vectorized modifications
    modified_charges = (sampled_charges * CHARGE_PERC + random_norm_late[:len(sampled)]).clip(0)
    modified_charges = (modified_charges * 40).round() / 40.0
    # Add 0.025 to .X00 or .X50 values to avoid exact, using numpy
    mask = (modified_charges * 10) % 0.5 == 0
    modified_charges[mask] += 0.025
    modified_charges = np.round(modified_charges, 3)
    
    modified_times = np.round(sampled_times + random_exp_late[:len(sampled)]).astype(np.int64)

    return pl.DataFrame({
        'sensor_id': sampled_sensor_ids.astype(np.int16),
        'time': modified_times,
        'charge': modified_charges,
        'auxiliary': sampled_auxiliary,
        'event_id': np.int64(group['event_id'][0]),
        'late_pulse': True,
        'after_pulse': False,
        'noise': False
    },
    orient='row'
    )

# Apply to each group (fast with index)
late_pulses_list = []
start_time = time.time()

for idx, (event_id, group) in enumerate(data_df.group_by('event_id')):
    late_pulses_list.append(process_late_pulses(group, idx))

    # Stop after NUM_EVENTS
    if idx + 1 >= NUM_EVENTS:
        break

end_time = time.time()
print(f"Late pulse generation took {end_time - start_time:.2f} seconds.")

late_pulses_df = pl.concat(late_pulses_list)


# Precompute random values for after pulses
random_poisson_after = np.random.poisson(lam=config['AFTERPULSES']['POISSON_MEAN'], size=NUM_EVENTS)
random_poisson_after_per_pulse = np.random.poisson(lam=config['AFTERPULSES']['POISSON_MEAN_AFTERPULSES'], size=NUM_EVENTS * 10)
random_norm_after_charge = np.random.normal(loc=config['AFTERPULSES']['CHARGE_MEAN'], scale=config['AFTERPULSES']['CHARGE_NOISE_STD'], size=NUM_EVENTS * 20)
random_norm_after_time = np.random.normal(loc=config['AFTERPULSES']['MEAN_TIME_OFFSET'], scale=config['AFTERPULSES']['TIME_OFFSET_STD'], size=NUM_EVENTS * 20)

def process_after_pulses(group, idx):
    """
    Create after pulses by sampling from existing charges and modifying them.
    """
    num_samples = random_poisson_after[idx]
    if num_samples == 0:
        return pl.DataFrame({
            'sensor_id': [],
            'time': [],
            'charge': [],
            'auxiliary': [],
            'event_id': [],
            'late_pulse': [],
            'after_pulse': [],
            'noise': []
        }, schema={
            'sensor_id': pl.Int16,
            'time': pl.Int64,
            'charge': pl.Float64,
            'auxiliary': pl.Boolean,
            'event_id': pl.Int64,
            'late_pulse': pl.Boolean,
            'after_pulse': pl.Boolean,
            'noise': pl.Boolean
        })
    
    # Sample vectorized
    sampled = group.sample(n=num_samples, seed=42)
    
    # Extract as numpy arrays
    sampled_charges = sampled['charge'].to_numpy()
    sampled_times = sampled['time'].to_numpy()
    sampled_sensor_ids = sampled['sensor_id'].to_numpy()
    sampled_auxiliary = sampled['auxiliary'].to_numpy()
    
    after_pulse_rows = []
    pulse_idx = 0
    
    for i in range(len(sampled_charges)):
        charge = sampled_charges[i]
        charge_time = sampled_times[i]
        sensor_id = sampled_sensor_ids[i]
        auxiliary = sampled_auxiliary[i]
        
        # Number of after pulses for this pulse
        num_after_pulses = random_poisson_after_per_pulse[pulse_idx % len(random_poisson_after_per_pulse)]
        pulse_idx += 1
        
        for j in range(num_after_pulses):
            if pulse_idx >= len(random_norm_after_charge):
                break
                
            # Modify charge
            modified_charge = charge * random_norm_after_charge[pulse_idx]
            modified_charge = max(modified_charge, 0)
            modified_charge = np.round(modified_charge * 40) / 40.0
            
            # Add 0.025 to avoid exact values
            if modified_charge * 10 % 0.5 == 0:
                modified_charge += 0.025
            modified_charge = np.round(modified_charge, 3)
            
            # Modify time
            modified_time = np.round(charge_time + random_norm_after_time[pulse_idx])
            pulse_idx += 1
            
            after_pulse_rows.append([
                sensor_id,
                modified_time,
                modified_charge,
                auxiliary,
                group['event_id'][0],
                False,  # late_pulse
                True,   # after_pulse
                False   # noise
            ])
    
    if not after_pulse_rows:
        return pl.DataFrame()
    
    return pl.DataFrame(
        after_pulse_rows,
        schema={
            'sensor_id': pl.Int16,
            'time': pl.Int64,
            'charge': pl.Float64,
            'auxiliary': pl.Boolean,
            'event_id': pl.Int64,
            'late_pulse': pl.Boolean,
            'after_pulse': pl.Boolean,
            'noise': pl.Boolean
        },
        orient='row'
    )

# Apply after pulses processing
after_pulses_list = []
start_time = time.time()

for idx, (event_id, group) in enumerate(data_df.group_by('event_id')):
    after_pulses_list.append(process_after_pulses(group, idx))
    
    if idx + 1 >= NUM_EVENTS:
        break

end_time = time.time()
print(f"After pulse generation took {end_time - start_time:.2f} seconds.")

after_pulses_df = pl.concat(after_pulses_list) if after_pulses_list else pl.DataFrame()

# Precompute random values for noise
random_norm_noise_charge = np.random.normal(loc=config['NOISE']['CHARGE_MEAN'], scale=config['NOISE']['CHARGE_NOISE_STD'], size=NUM_EVENTS * 50)
random_uniform_noise_time = np.random.uniform(0, 1, size=NUM_EVENTS * 50)

# Convert sensor_geometry to Polars for consistency
sensor_ids_array = sensor_geometry_df['sensor_id'].to_numpy()

def process_noise(group, idx):
    """
    Add noise hits based on event time duration and noise rate.
    """
    # Calculate event duration
    time_min = group['time'].min()
    time_max = group['time'].max()
    event_duration = time_max - time_min
    
    # Calculate number of noise hits based on duration and rate
    # NOISE_RATE is hits per nanosecond, so multiply by duration
    expected_noise_hits = config['NOISE']['NOISE_RATE'] * event_duration
    num_noise_hits = np.random.poisson(lam=expected_noise_hits)
    
    if num_noise_hits == 0:
        return pl.DataFrame({
            'sensor_id': [],
            'time': [],
            'charge': [],
            'auxiliary': [],
            'event_id': [],
            'late_pulse': [],
            'after_pulse': [],
            'noise': []
        }, schema={
            'sensor_id': pl.Int16,
            'time': pl.Int64,
            'charge': pl.Float64,
            'auxiliary': pl.Boolean,
            'event_id': pl.Int64,
            'late_pulse': pl.Boolean,
            'after_pulse': pl.Boolean,
            'noise': pl.Boolean
        })
    
    # Sample sensor IDs
    sampled_sensor_ids = np.random.choice(sensor_ids_array, size=num_noise_hits, replace=True).astype(np.int16)
    
    # Generate random times within extended event window
    time_range_start = int(time_min - 100)
    time_range_end = int(time_max + 100)
    random_times = np.random.uniform(time_range_start, time_range_end, size=num_noise_hits)
    random_times = np.round(random_times).astype(np.int64)

    
    # Generate random charges
    start_idx = (idx * 50) % len(random_norm_noise_charge)
    end_idx = min(start_idx + num_noise_hits, len(random_norm_noise_charge))
    actual_samples = end_idx - start_idx
    
    if actual_samples < num_noise_hits:
        # If we don't have enough precomputed values, generate on the fly
        random_charges = np.random.normal(config['NOISE']['CHARGE_MEAN'], config['NOISE']['CHARGE_NOISE_STD'], num_noise_hits)
    else:
        random_charges = random_norm_noise_charge[start_idx:end_idx]
    
    # Process charges
    random_charges = np.maximum(random_charges, 0)  # Ensure non-negative
    random_charges = np.round(random_charges * 40) / 40.0
    
    # Add 0.025 to avoid exact values
    mask = (random_charges * 10) % 0.5 == 0
    random_charges[mask] += 0.025
    random_charges = np.round(random_charges, 3)
    
    # Create noise DataFrame
    noise_data = []
    for i in range(num_noise_hits):
        noise_data.append([
            sampled_sensor_ids[i],
            random_times[i],
            random_charges[i],
            True,  # auxiliary
            group['event_id'][0],
            False,  # late_pulse
            False,  # after_pulse
            True    # noise
        ])
    
    return pl.DataFrame(
        noise_data,
        schema={
            'sensor_id': pl.Int16,
            'time': pl.Int64,
            'charge': pl.Float64,
            'auxiliary': pl.Boolean,
            'event_id': pl.Int64,
            'late_pulse': pl.Boolean,
            'after_pulse': pl.Boolean,
            'noise': pl.Boolean
        },
        orient='row'
    )

# Apply noise processing
noise_list = []
start_time = time.time()

for idx, (event_id, group) in enumerate(data_df.group_by('event_id')):
    noise_list.append(process_noise(group, idx))
    
    if idx + 1 >= NUM_EVENTS:
        break

end_time = time.time()
print(f"Noise generation took {end_time - start_time:.2f} seconds.")

noise_df = pl.concat(noise_list) if noise_list else pl.DataFrame()

#------------------------------------------------------
print(f"Adding flag columns...")
flag_start_time = time.time()
# Add columns to original data
data_df_with_flags = data_df.with_columns([
    pl.lit(False).alias('late_pulse'),
    pl.lit(False).alias('after_pulse'), 
    pl.lit(False).alias('noise')
])
flag_end_time = time.time()
print(f"Flag column addition took {flag_end_time - flag_start_time:.2f} seconds.")

print(f"Filtering data_df to first NUM_EVENTS...")
filter_start_time = time.time()
# Filter data_df to first NUM_EVENTS 
unique_event_ids = data_df['event_id'].unique().sort()[:NUM_EVENTS]
data_df_filtered = data_df_with_flags.filter(pl.col('event_id').is_in(pl.Series(unique_event_ids).implode()))
filter_end_time = time.time()
print(f"Filtering took {filter_end_time - filter_start_time:.2f} seconds.")

print(f'Sorting augmented data by event_id and time...')
concat_sort_start_time = time.time()
# Combine all DataFrames in Polars
dataframes_to_combine = [data_df_filtered]
dataframes_to_combine.append(late_pulses_df)
dataframes_to_combine.append(after_pulses_df)
dataframes_to_combine.append(noise_df)

# Concatenate and sort
augmented_data_pl = pl.concat(dataframes_to_combine, how="vertical")
augmented_data_pl = augmented_data_pl.sort(['event_id', 'time'])
concat_sort_end_time = time.time()
print(f"Combining and sorting took {concat_sort_end_time - concat_sort_start_time:.2f} seconds.")

print(f'Updating metadata...')
metadata_start_time = time.time()
# Calculate first and last pulse indices for each event_id
metadata_updates = (
    augmented_data_pl
    .filter(pl.col('event_id').is_in(pl.Series(unique_event_ids).implode()))
    .with_row_index('row_idx')  # Add row indices
    .group_by('event_id')
    .agg([
        pl.col('row_idx').min().alias('first_pulse_index'),
        pl.col('row_idx').max().alias('last_pulse_index')
    ])
    .sort('event_id')
)

# Filter meta_df to only relevant event_ids
filtered_meta_batch = meta_df.filter(pl.col('event_id').is_in(pl.Series(unique_event_ids).implode()))

# Drop old pulse index columns if present
filtered_meta_batch = filtered_meta_batch.drop(['first_pulse_index', 'last_pulse_index'])

# Merge with metadata_updates (Polars join)
filtered_meta_batch = filtered_meta_batch.join(
    metadata_updates.select(['event_id', 'first_pulse_index', 'last_pulse_index']),
    on='event_id',
    how='left'
)
metadata_end_time = time.time()
print(f"Metadata update took {metadata_end_time - metadata_start_time:.2f} seconds.")

#------------------------------------------------------
print(f"Saving files...")
save_start_time = time.time()
filtered_meta_batch_path = Path('data/filtered_meta_batch_1.parquet')
augmented_data_df_path = Path('data/augmented_data_df.parquet')

filtered_meta_batch.write_parquet(filtered_meta_batch_path)
augmented_data_pl.write_parquet(augmented_data_df_path)
save_end_time = time.time()
print(f"Saving files took {save_end_time - save_start_time:.2f} seconds.")

script_end_time = time.time()
print(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds.")

print(f"Filtered meta data for first {NUM_EVENTS} events saved to {filtered_meta_batch_path}")
print(f"Augmented event data saved to {augmented_data_df_path}")