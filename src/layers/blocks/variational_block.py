import torch
from torch import nn


class VariationalBlock(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.mu_layer = nn.Linear(input_dim, latent_dim)
        self.logvar_layer = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        logvar = torch.clamp(logvar, -10, 10)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + std * eps

        return z, mu, logvar