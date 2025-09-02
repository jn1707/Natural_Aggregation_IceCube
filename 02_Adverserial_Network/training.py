import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loader
from transformer_model import TransformerClassifier

def train_model(data_path, epochs=10, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = get_data_loader(data_path, batch_size=16)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")
    
    torch.save(model.state_dict(), "../models/transformer_model.pth")

if __name__ == "__main__":
    train_model("../data/preprocessed_sequences.pkl")