import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

os.makedirs('models', exist_ok=True)
os.makedirs('src/plots', exist_ok=True)

class AirlineNet(nn.Module):
    def __init__(self, input_dim):
        super(AirlineNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def train(X_train, y_train, X_val, y_val):

    X_train_t = torch.FloatTensor(X_train.copy())
    y_train_t = torch.FloatTensor(y_train.values.copy()).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val.copy())
    y_val_t = torch.FloatTensor(y_val.values.copy()).unsqueeze(1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    input_dim = X_train.shape[1]
    model = AirlineNet(input_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    patience = 10
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    print("Training Neural Network...")
    for epoch in range(200):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_weights)
    print(f"Best Val Loss: {best_val_loss:.4f}")

    # Loss curves plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')
    ax.set_title('Neural Network - Training & Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.tight_layout()
    plt.savefig('src/plots/nn_loss_curves.png')
    plt.close()

    torch.save(model.state_dict(), 'models/neural_network.pt')
    print("Neural Network saved ✓")

    return model

if __name__ == '__main__':
    from SRC import preprocessing
    (X_train, X_val, X_test,
     y_train, y_val, y_test, feature_names) = preprocessing.run()
    train(X_train, y_train, X_val, y_val)