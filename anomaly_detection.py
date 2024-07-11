import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import device

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

def lstm_anomaly_detection(X, feature_columns, num_anomalies=10, epochs=100, batch_size=64):
    # Prepare data
    X = torch.FloatTensor(X).to(device)

    # Split data into train and validation sets
    train_size = int(0.85 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]

    # Create model
    model = LSTMAutoencoder(input_size=len(feature_columns)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Training
    patience = 1
    best_val_loss = float('inf')
    counter = 0

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output_train = model(X_train.unsqueeze(0))
        loss_train = criterion(output_train, X_train)
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output_val = model(X_val.unsqueeze(0))
            loss_val = criterion(output_val, X_val)

        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}')

        # Early stopping
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Anomaly detection
    model.eval()
    with torch.no_grad():
        reconstructed = model(X.unsqueeze(0)).squeeze(0).cpu().numpy()

    mse = np.mean(np.power(X.cpu().numpy() - reconstructed, 2), axis=1)

    top_indices = mse.argsort()[-num_anomalies:][::-1]
    anomalies = np.zeros(len(mse), dtype=bool)
    anomalies[top_indices] = True

    return anomalies, mse, top_indices, model, train_losses, val_losses