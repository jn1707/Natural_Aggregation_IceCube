import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, SelectMultiple
import pyarrow.parquet as pq
from pathlib import Path
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bar

# Number of events to process
NUM_EVENTS = 100000

meta_path = Path('data/train_meta_batch_1.parquet')
data_path = Path('data/batch_1.parquet')
sensor_geometry_path = Path('data/sensor_geometry.csv')

meta_table = pq.read_table(meta_path)
data_table = pq.read_table(data_path)
meta_df = meta_table.to_pandas()
data_df = data_table.to_pandas()

# Reset the index to make 'event_id' a regular column
data_df.reset_index(inplace=True)

# Load sensor geometry data
sensor_geometry_df = pd.read_csv(sensor_geometry_path)

"""
Create late pulses by sampling from existing charges and modifying them.
"""
# Initialize an empty list to store new rows
late_pulse_rows = []

# Loop through each unique event
for i in tqdm(range(0, min(NUM_EVENTS, data_df['event_id'].nunique())), desc="Processing Late Pulses"):
    # Filter data for the current event
    event_data = data_df[data_df['event_id'] == meta_df['event_id'].iloc[i]]
    
    # Determine the number of samples using a Poisson distribution with mean=1
    num_samples = np.random.poisson(lam=1)
    
    # If no samples are drawn, skip this event
    if num_samples == 0:
        continue
    
    # Sample charges from the event data
    sampled_data = event_data[['charge', 'time', 'sensor_id', 'auxiliary']].sample(n=num_samples, random_state=42)

    # Create a modified pulse for each sampled charge
    for _, row in sampled_data.iterrows():
        charge = row['charge']
        charge_time = row['time']

        # Modify the charge to be around 20% of the original with some noise
        modified_charge = charge * 0.2 + np.random.normal(0, 0.1)

        modified_charge = max(modified_charge, 0)  # Ensure charge is non-negative

        # Round charge to .X25 or .X75 to simulate quantization
        modified_charge = round(modified_charge * 40) / 40.0
        # Add 0.025 to .X00 or .X50 values to avoid exact zeros
        if modified_charge * 10 % 0.5 == 0:
            modified_charge += 0.025
            modified_charge = float(f"{modified_charge:.3f}")

        # Modify the time of this charge to be exponentially later than its original time with some noise
        modified_charge_time = round(charge_time + np.random.exponential(scale=50))
        
        # Create a new row with the modified values
        new_row = {
            'event_id': event_data['event_id'].iloc[0],
            'sensor_id': row['sensor_id'],
            'time': modified_charge_time,
            'charge': modified_charge,
            'auxiliary': row['auxiliary'],
            'late_pulse': True,
            'after_pulse': False,
            'noise': False
        }
        
        # Append the new row to the list
        late_pulse_rows.append(new_row)

# Combine all new rows into a new DataFrame
late_pulses_df = pd.DataFrame(late_pulse_rows)

"""
Create after pulses by sampling from existing charges and modifying them.
"""
# Initialize an empty list to store new rows
after_pulse_rows = []

# Loop through each unique event
for i in tqdm(range(0, min(NUM_EVENTS, data_df['event_id'].nunique())), desc="Processing After Pulses"):
    # Filter data for the current event
    event_data = data_df[data_df['event_id'] == meta_df['event_id'].iloc[i]]
    
    # Determine the number of samples using a Poisson distribution with mean=1
    num_samples = np.random.poisson(lam=1)
    
    # If no samples are drawn, skip this event
    if num_samples == 0:
        continue
    
    # Sample charges from the event data
    sampled_data = event_data[['charge', 'time', 'sensor_id', 'auxiliary']].sample(n=num_samples, random_state=42)

    # Create a modified pulse for each sampled charge
    for _, row in sampled_data.iterrows():
        
        charge = row['charge']
        charge_time = row['time']
        sensor_id = row['sensor_id']
        auxiliary = row['auxiliary']
        
        # Create a  number of after pulses based on a Poisson distribution with mean=2
        num_after_pulses = np.random.poisson(lam=2)
        for _ in range(num_after_pulses):
            # Modify the charge to be around 100% of the original with some noise
            modified_charge = charge  * np.random.normal(1, 2)

            modified_charge = max(modified_charge, 0)  # Ensure charge is non-negative

            # Round charge to .X25 or .X75 to simulate quantization
            modified_charge = round(modified_charge * 40) / 40.0
            # Add 0.025 to .X00 or .X50 values to avoid exact
            if modified_charge * 10 % 0.5 == 0:
                modified_charge += 0.025
                modified_charge = float(f"{modified_charge:.3f}")
            
            # Modify the time of this charge to be significantly later than its original time with some noise
            modified_charge_time = round(charge_time + np.random.normal(200, 100))
            
            # Create a new row with the modified values
            new_row = {
                'event_id': event_data['event_id'].iloc[0],
                'sensor_id': sensor_id,
                'time': modified_charge_time,
                'charge': modified_charge,
                'auxiliary': auxiliary,
                'late_pulse': False,
                'after_pulse': True,
                'noise': False
            }
            
            # Append the new row to the list
            after_pulse_rows.append(new_row)

# Combine all new rows into a new DataFrame
after_pulses_df = pd.DataFrame(after_pulse_rows)

"""
Add noise to the event data
"""
noise_rows = []
# Loop through each unique event
for i in tqdm(range(0, min(NUM_EVENTS, data_df['event_id'].nunique())), desc="Adding noise"):
    # Filter data for the current event
    event_data = data_df[data_df['event_id'] == meta_df['event_id'].iloc[i]]
    
    # Determine the number of noise hits to add using a Poisson distribution with mean=5
    num_noise_hits = np.random.poisson(lam=5)
    
    # If no noise hits are to be added, skip this event
    if num_noise_hits == 0:
        continue
    
    # Randomly select sensor_ids from the sensor geometry data
    sampled_sensors = sensor_geometry_df['sensor_id'].sample(n=num_noise_hits, replace=True, random_state=42) # Allow sensor to be sampled multiple times

    # Pick time randomly 
    random_time = int(np.random.uniform(event_data['time'].min()-100, event_data['time'].max()+100))

    # Pick a random charge around 3 with some noise
    random_charge = max(np.random.normal(3, 3), 0)
    # Round charge to .X25 or .X75 to simulate quantization
    random_charge = round(random_charge * 40) / 40.0
    # Add 0.025 to .X00 or .X50 values to avoid exact
    if random_charge * 10 % 0.5 == 0:
        random_charge += 0.025
        random_charge = float(f"{random_charge:.3f}")

    for sensor_id in sampled_sensors:
        # Create a new noise hit
        new_row = {
            'event_id': event_data['event_id'].iloc[0],
            'sensor_id': sensor_id,
            'time': random_time,
            'charge': random_charge,
            'auxiliary': False,
            'late_pulse': False,
            'after_pulse': False,
            'noise': True
        }
        
        # Append the new row to the list
        noise_rows.append(new_row)

# Combine all new rows into a new DataFrame
noise_df = pd.DataFrame(noise_rows)


# Add 'late_pulse', 'after_pulse', and 'noise' columns to the original data
data_df['late_pulse'] = False
data_df['after_pulse'] = False
data_df['noise'] = False

# Combine the original data with the new late and after pulses
augmented_data_df = pd.concat([data_df, late_pulses_df, after_pulses_df, noise_df], ignore_index=True)

# Create a new augmented_meta_batch_1 DataFrame
augmented_meta_batch_1 = meta_df.copy()

print(f'Sorting augmented data by event_id and time...')
# Sort the augmented data by event_id and time to ensure proper indexing
augmented_data_df = augmented_data_df.sort_values(by=['event_id', 'time']).reset_index(drop=True)


# Filter the meta data to include only the first NUM_EVENTS events
filtered_meta_batch = augmented_meta_batch_1[augmented_meta_batch_1['event_id'].isin(meta_df['event_id'][:NUM_EVENTS])].copy()

# Initialize lists to store the first and last pulse indices
first_pulse_indices = []
last_pulse_indices = []

# Update metadata for the first NUM_EVENTS events
for event_id in tqdm(filtered_meta_batch['event_id'], desc=f"Updating Meta Data for First {NUM_EVENTS} Events"):
    # Get the indices of the first and last pulses for the current event
    event_indices = augmented_data_df[augmented_data_df['event_id'] == event_id].index
    first_pulse_indices.append(event_indices.min())
    last_pulse_indices.append(event_indices.max())

# Update the filtered_meta_batch DataFrame
filtered_meta_batch['first_pulse_index'] = first_pulse_indices
filtered_meta_batch['last_pulse_index'] = last_pulse_indices

# Save the filtered metadata to a new file
filtered_meta_batch_path = Path('data/filtered_meta_batch_1.parquet')
filtered_meta_batch.to_parquet(filtered_meta_batch_path, index=False)

# Save the augmented data (optional, if not already saved)
augmented_data_df_path = Path('data/augmented_data_df.parquet')
augmented_data_df.to_parquet(augmented_data_df_path, index=False)

print(f"Filtered meta data for first {NUM_EVENTS} events saved to {filtered_meta_batch_path}")
print(f"Augmented event data saved to {augmented_data_df_path}")