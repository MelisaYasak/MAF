import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
import plotly.graph_objects as go
from IPython.display import HTML, display

from made_implementation import MADE
from setup_and_data import dataloaders

import warnings
warnings.filterwarnings('ignore')

# GPU/CPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""## Adım 3.1: MAF Sınıfı"""

class MAF(nn.Module):
    """Masked Autoregressive Flow - Birden fazla MADE layer'ı stack eder"""

    def __init__(self, input_dim, hidden_dims, n_layers=5):
        super(MAF, self).__init__()

        self.input_dim = input_dim
        self.n_layers = n_layers

        # Create multiple MADE blocks
        self.made_blocks = nn.ModuleList([
            MADE(input_dim, hidden_dims)
            for _ in range(n_layers)
        ])

        # Base distribution (isotropic Gaussian)
        self.register_buffer('base_mu', torch.zeros(input_dim))
        self.register_buffer('base_sigma', torch.ones(input_dim))

    def forward(self, z):
        """
        Generate samples: z₀ ~ N(0,I) -> x

        Args:
            z: Base samples (batch_size, input_dim)

        Returns:
            x: Generated samples
        """
        x = z

        for made in self.made_blocks:
            x, _, _ = made.forward(x)

        return x

    def inverse(self, x):
        """
        Compute latent: x -> z₀

        Args:
            x: Data samples (batch_size, input_dim)

        Returns:
            z: Latent variables
            log_det_jacobian: Log determinant
        """
        z = x
        log_det_jacobian = 0

        # Apply inverse of each MADE block (in reverse order)
        for made in reversed(self.made_blocks):
            z, log_det = made.inverse(z)
            log_det_jacobian += log_det

        return z, log_det_jacobian

    def log_probs(self, x):
        """
        Compute log p(x) using change of variables

        Args:
            x: Data samples (batch_size, input_dim)

        Returns:
            log_probs: Log probabilities
        """
        # Inverse transform
        z, log_det_jacobian = self.inverse(x)

        # Base distribution log probability
        log_pz = -0.5 * (
            self.input_dim * np.log(2 * np.pi) +
            torch.sum(z ** 2, dim=1)
        )

        # Change of variables formula
        log_px = log_pz + log_det_jacobian

        return log_px

    def sample(self, n_samples):
        """
        Generate samples from the model

        Args:
            n_samples: Number of samples

        Returns:
            samples: Generated samples
        """
        self.eval()

        with torch.no_grad():
            # Sample from base distribution
            z = torch.randn(n_samples, self.input_dim).to(
                self.base_mu.device
            )

            # Transform through flow
            x = self.forward(z)

        return x

# Create MAF model
maf = MAF(
    input_dim=2,
    hidden_dims=[128, 128],
    n_layers=5
).to(device)

print(f"MAF model: {sum(p.numel() for p in maf.parameters())} parameters")
print(f"Number of flows: {maf.n_layers}")

"""## Adım 3.2: Loss Function

"""

def compute_loss(model, x):
    """
    Compute negative log-likelihood loss

    Args:
        model: MAF model
        x: Data batch (batch_size, input_dim)

    Returns:
        loss: Scalar loss value
        log_probs: Log probabilities for monitoring
    """
    log_probs = model.log_probs(x)
    loss = -torch.mean(log_probs)  # Negative log-likelihood

    return loss, log_probs

# Test loss computation
x_batch = next(iter(dataloaders['moons']))[0].to(device)
loss, log_probs = compute_loss(maf, x_batch)

print(f"Batch size: {x_batch.shape[0]}")
print(f"Loss: {loss.item():.4f} nats")
print(f"Mean log prob: {log_probs.mean().item():.4f} nats")

