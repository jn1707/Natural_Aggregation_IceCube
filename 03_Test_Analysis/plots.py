import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# --- Load predictions and test event IDs ---
with open("data/test_predictions.pkl", "rb") as f:
    test_results = pickle.load(f)
preds = np.array(test_results['preds'])
labels = np.array(test_results['labels'])

with open("data/event_ids_test.pkl", "rb") as f:
    test_event_ids = pickle.load(f)
test_event_ids = np.array(test_event_ids)

# --- Load original and augmented data ---
original_data_df = pd.read_parquet("data/batch_1.parquet")
augmented_data_df = pd.read_parquet("data/augmented_data_df.parquet")
all_data_df = pd.concat([original_data_df, augmented_data_df], ignore_index=True)

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

# --- ROC Curve ---
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
plt.savefig("data/roc_curve_any_noise.png")
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
plt.savefig("data/roc_curves.png")
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
plt.savefig("data/probability_distributions.png")
plt.close()