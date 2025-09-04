import torch
import yaml
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy
from transformer_model import TransformerClassifier
from pathlib import Path

# Load configuration
config_path = Path('02_Adverserial_Network/250902-config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

class LitTransformerClassifier(pl.LightningModule):
    def __init__(self, input_dim, num_classes, lr=config['LEARNING_RATE']):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes)
        self.model = self.model.cuda()
        self.lr = float(lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

    def forward(self, x, attention_mask=None):
        x = x.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        return self.model(x, attention_mask=attention_mask)
    
    def training_step(self, batch, batch_idx):
        x, y, attention_mask = batch
        x = x.cuda()
        y = y.cuda()
        attention_mask = attention_mask.cuda()
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or Inf detected in input!")
        logits = self(x, attention_mask)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("NaN or Inf detected in logits!")
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, attention_mask = batch
        x = x.cuda()
        y = y.cuda()
        attention_mask = attention_mask.cuda()
        logits = self(x, attention_mask)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y, attention_mask = batch
        x = x.cuda()
        y = y.cuda()
        attention_mask = attention_mask.cuda()
        logits = self(x, attention_mask)
        preds = torch.argmax(logits, dim=1)
        return {'preds': preds, 'labels': y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)