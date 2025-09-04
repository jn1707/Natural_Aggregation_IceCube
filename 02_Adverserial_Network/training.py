import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from data_loader import get_data_loader
from lightning_module import LitTransformerClassifier
import yaml
from pathlib import Path

# Load config
config_path = Path('02_Adverserial_Network/250902-config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config['BATCH_SIZE']
EPOCHS = config['EPOCHS']
LEARNING_RATE = config['LEARNING_RATE']
INPUT_DIM = config['INPUT_DIM']
OUTPUT_DIM = config['OUTPUT_DIM']

# Paths to your data splits
train_path = "data/preprocessed_sequences_train.pkl"
val_path = "data/preprocessed_sequences_val.pkl"
test_path = "data/preprocessed_sequences_test.pkl"

def main():
    # Data loaders
    train_loader = get_data_loader(train_path, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_data_loader(val_path, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_data_loader(test_path, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = LitTransformerClassifier(input_dim=INPUT_DIM, num_classes=OUTPUT_DIM, lr=float(LEARNING_RATE))

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename='best_model')

    # Trainer
    trainer = Trainer(
        max_epochs=EPOCHS,
        callbacks=[early_stop, checkpoint],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16,
        log_every_n_steps=10
    )

    # Training & validation
    trainer.fit(model, train_loader, val_loader)

    # Testing (loads best weights automatically)
    results = trainer.test(model, test_loader, ckpt_path='best')
    print("Test results:", results)

    # Save predictions on test set
    preds = []
    labels = []
    model = model.cuda().to(torch.float16)
    for batch in test_loader:
        x, y, attention_mask = batch
        x = x.cuda().to(torch.float16)
        attention_mask = attention_mask.cuda()
        logits = model(x, attention_mask)
        batch_preds = logits[:, 1].detach().cpu().numpy()
        preds.extend(batch_preds)
        labels.extend(y.cpu().numpy())
    import pickle
    with open("data/test_predictions.pkl", "wb") as f:
        pickle.dump({'preds': preds, 'labels': labels}, f)

if __name__ == "__main__":
    main()