import pickle
import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

# --- Load configuration and extract date ---
config_path = Path('02_Adverserial_Network/250904-config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
MAX_SEQ_LENGTH = config['DATA']['MAX_SEQ_LENGTH']

config_date = config_path.stem.split('-')[0]
plot_dir = Path(f"data/{config_date}")
plot_dir.mkdir(parents=True, exist_ok=True)

# --- Load predictions and test event IDs ---
with open("data/test_predictions.pkl", "rb") as f:
    test_results = pickle.load(f)
preds = np.array(test_results['preds'])
labels = np.array(test_results['labels'])

with open("data/event_ids_test.pkl", "rb") as f:
    test_event_ids = pickle.load(f)
test_event_ids = np.array(test_event_ids)

# --- Load original and augmented data / metadata ---
original_data_df = pd.read_parquet("data/batch_1.parquet")
augmented_data_df = pd.read_parquet("data/augmented_data_df.parquet")
all_data_df = pd.concat([original_data_df, augmented_data_df], ignore_index=True)

original_metadata_df = pd.read_parquet("data/train_meta_batch_1.parquet")
augmented_metadata_df = pd.read_parquet("data/filtered_meta_batch_1.parquet")
all_metadata_df = pd.concat([original_metadata_df, augmented_metadata_df], ignore_index=True)

# --- Filter test events in all data ---
test_data = all_data_df[all_data_df['event_id'].isin(test_event_ids)]

# --- Get noise flags for each test event ---
flags = test_data.groupby('event_id')[['late_pulse', 'after_pulse', 'noise']].max()  # max will convert True/False to 1/0
event_noise_flags = (
    pd.DataFrame({'event_id': test_event_ids})
    .set_index('event_id')
    .join(flags, how='left')
    .fillna(False) # Fill NaNs for events without late-, after pulses or noise with False
    .infer_objects(copy=False)
    .reset_index()
)
# Ensure the order matches test_event_ids
event_noise_flags = event_noise_flags.set_index('event_id').loc[test_event_ids].reset_index()

# --- Identify original events in test set ---
is_original = event_noise_flags['event_id'] > 0

# --- ROC Curves ---
plt.figure(figsize=(8, 6))
y_true = event_noise_flags[['late_pulse', 'after_pulse', 'noise']].any(axis=1).astype(int)
count = y_true.sum()
print(f"Number of test events with any noise type: {int(count)} out of {len(y_true)}")
fpr, tpr, _ = roc_curve(y_true, preds)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'Any noise type (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Any Noise Type')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_dir / f"roc_curve_any_noise.png")
plt.close()

plt.figure(figsize=(8, 6))
for noise_type in ['late_pulse', 'after_pulse', 'noise']:
    # Create a mask for the current noise type and original events
    mask = (event_noise_flags[noise_type] == 1) | (is_original)

    y_true = event_noise_flags[mask][noise_type].astype(int)
    count = y_true.sum()
    print(f"Number of test events with {noise_type}: {int(count)}")
    fpr, tpr, _ = roc_curve(y_true, preds[mask])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{noise_type} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Noise Types')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_dir / f"roc_curves.png")
plt.close()

# --- Probability Distributions ---
plt.figure(figsize=(10, 6))
for noise_type in ['late_pulse', 'after_pulse', 'noise']:
    y_true = event_noise_flags[noise_type].astype(int).values
    plt.hist(
        preds[y_true == 1],
        bins=30,
        density=True,
        histtype='step',
        label=f"{noise_type} (events={int(y_true.sum())})"
    )
# Add original data (label=0) in test set
plt.hist(
    preds[(is_original.values) & (labels == 0)],
    bins=30,
    density=True,
    histtype='step',
    color='black',
    label="Original data (label=0)"
)
plt.xlabel("Predicted Probability to be any noise")
plt.ylabel("Density")
plt.title("Density of Prediction Probabilities for Noise Types and Original Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_dir / f"probability_distributions.png")
plt.close()

# --- Compute sequence length from metadata ---
# Map event_id to sequence length using meta data
meta_event_lengths = (
    all_metadata_df.set_index('event_id')[['first_pulse_index', 'last_pulse_index']]
    .apply(lambda row: row['last_pulse_index'] - row['first_pulse_index'] + 1, axis=1)
)
event_noise_flags['seq_length'] = event_noise_flags['event_id'].map(meta_event_lengths)

# --- Masks for short and long events ---
short_mask = event_noise_flags['seq_length'] <= MAX_SEQ_LENGTH
long_mask = event_noise_flags['seq_length'] > MAX_SEQ_LENGTH

for mask, label in zip([short_mask, long_mask], ["short (≤MAX_SEQ_LENGTH)", "long (>MAX_SEQ_LENGTH)"]):
    print(f"\n--- {label} events: {mask.sum()} ---")

    # ROC Curve for any noise type
    plt.figure(figsize=(8, 6))
    y_true = event_noise_flags[mask][['late_pulse', 'after_pulse', 'noise']].any(axis=1).astype(int)
    preds_masked = preds[mask]
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, preds_masked)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Any noise type (AUC = {roc_auc:.2f})')
    else:
        print(f"Not enough variation for ROC curve in {label} events.")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Any Noise Type ({label})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"roc_curve_any_noise_{label.replace(' ', '').replace('(≤','').replace('(>','').replace('MAX_SEQ_LENGTH)','')}.png")

    plt.figure(figsize=(8, 6))
    for noise_type in ['late_pulse', 'after_pulse', 'noise']:
        mask_type = (event_noise_flags[mask][noise_type] == 1) | (is_original[mask])
        y_true = event_noise_flags[mask][mask_type][noise_type].astype(int)
        preds_type = preds[mask][mask_type]
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, preds_type)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{noise_type} (AUC = {roc_auc:.2f})')
        else:
            print(f"Not enough variation for ROC curve for {noise_type} in {label} events.")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for Noise Type {noise_type} ({label})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"roc_curves_{label.replace(' ', '').replace('(≤','').replace('(>','').replace('MAX_SEQ_LENGTH)','')}.png")
    plt.close()

    # Probability Distributions
    plt.figure(figsize=(10, 6))
    for noise_type in ['late_pulse', 'after_pulse', 'noise']:
        y_true = event_noise_flags[noise_type].astype(int).values
        plt.hist(
            preds[(y_true == 1) & mask.values],
            bins=30,
            density=True,
            histtype='step',
            label=f"{noise_type} (events={int(((y_true == 1) & mask.values).sum())})"
        )
    # Add original data (label=0) in test set
    plt.hist(
        preds[(is_original.values) & (labels == 0) & mask.values],
        bins=30,
        density=True,
        histtype='step',
        color='black',
        label="Original data (label=0)"
    )
    plt.xlabel("Predicted Probability to be any noise")
    plt.ylabel("Density")
    plt.title(f"Density of Prediction Probabilities ({label})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"probability_distributions_{label.replace(' ', '').replace('(≤','').replace('(>','').replace('MAX_SEQ_LENGTH)','')}.png")
    plt.close()