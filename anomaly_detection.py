import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class VAE(nn.Module):
    def __init__(self, input_size, latent_dim=32):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_size)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size * seq_len, -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded.view(batch_size, seq_len, -1), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def anomaly_detection(X_embeddings, X_posture, X_voice, epochs=200, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize posture
    scaler_posture = MinMaxScaler()
    X_posture_scaled = scaler_posture.fit_transform(X_posture.reshape(-1, 1))

    # Process facial embeddings
    X_embeddings = torch.FloatTensor(X_embeddings).to(device)
    if X_embeddings.dim() == 2:
        X_embeddings = X_embeddings.unsqueeze(0)

    # Process posture
    X_posture_scaled = torch.FloatTensor(X_posture_scaled).to(device)
    if X_posture_scaled.dim() == 2:
        X_posture_scaled = X_posture_scaled.unsqueeze(0)

    # Process voice embeddings
    X_voice = torch.FloatTensor(X_voice).to(device)
    if X_voice.dim() == 2:
        X_voice = X_voice.unsqueeze(0)

    model_embeddings = VAE(input_size=X_embeddings.shape[2]).to(device)
    model_posture = VAE(input_size=X_posture_scaled.shape[2]).to(device)
    model_voice = VAE(input_size=X_voice.shape[2]).to(device)

    optimizer_embeddings = optim.Adam(model_embeddings.parameters())
    optimizer_posture = optim.Adam(model_posture.parameters())
    optimizer_voice = optim.Adam(model_voice.parameters())

    # Train models
    for epoch in range(int(epochs)):  # Ensure epochs is an integer
        for model, optimizer, X in [(model_embeddings, optimizer_embeddings, X_embeddings),
                                    (model_posture, optimizer_posture, X_posture_scaled),
                                    (model_voice, optimizer_voice, X_voice)]:
            model.train()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(X)
            loss = vae_loss(recon_batch, X, mu, logvar)
            loss.backward()
            optimizer.step()

    # Compute reconstruction error for embeddings, posture, and voice
    model_embeddings.eval()
    model_posture.eval()
    model_voice.eval()
    with torch.no_grad():
        recon_embeddings, _, _ = model_embeddings(X_embeddings)
        recon_posture, _, _ = model_posture(X_posture_scaled)
        recon_voice, _, _ = model_voice(X_voice)
        mse_embeddings = F.mse_loss(recon_embeddings, X_embeddings, reduction='none').mean(
            dim=2).cpu().numpy().squeeze()
        mse_posture = F.mse_loss(recon_posture, X_posture_scaled, reduction='none').mean(dim=2).cpu().numpy().squeeze()
        mse_voice = F.mse_loss(recon_voice, X_voice, reduction='none').mean(dim=2).cpu().numpy().squeeze()

    return mse_embeddings, mse_posture, mse_voice


def determine_anomalies(mse_values, threshold):
    mean = np.mean(mse_values)
    std = np.std(mse_values)
    anomalies = mse_values > (mean + threshold * std)
    return anomalies