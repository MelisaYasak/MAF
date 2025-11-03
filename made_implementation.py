import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
import plotly.graph_objects as go
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')

# GPU/CPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""## Adım 2.1: Masking Fonksiyonları

"""

class MADEMask:
    """MADE için masking operations"""

    @staticmethod
    def create_masks(input_dim, hidden_dims, output_dim):
        """
        MADE için autoregressive masking oluştur

        Args:
            input_dim: Input boyutu
            hidden_dims: Hidden layer boyutları listesi
            output_dim: Output boyutu

        Returns:
            masks: Her layer için mask listesi
        """
        masks = []

        # Input numbering: m(x) = [1, 2, ..., D]
        m_x = torch.arange(1, input_dim + 1)

        # Hidden layer numberings
        m_h = []
        for hidden_dim in hidden_dims:
            # Her hidden unit'e [1, D-1] arası random sayı ata
            m = torch.randint(1, input_dim, (hidden_dim,))
            m_h.append(m)

        # Create masks for each layer
        # Input to first hidden
        masks.append((m_x.unsqueeze(-1) <= m_h[0].unsqueeze(0)).float())

        # Hidden to hidden
        for i in range(len(hidden_dims) - 1):
            masks.append((m_h[i].unsqueeze(-1) <= m_h[i+1].unsqueeze(0)).float())

        # Last hidden to output
        # Output'u μ ve α için ikiye böl
        m_out = torch.cat([m_x, m_x])  # [1,2,...,D, 1,2,...,D]
        masks.append((m_h[-1].unsqueeze(-1) < m_out.unsqueeze(0)).float())

        return masks

# Test masking
input_dim = 2
hidden_dims = [64, 64]
output_dim = 2 * input_dim  # μ ve α için

masks = MADEMask.create_masks(input_dim, hidden_dims, output_dim)

print("Mask shapes:")
for i, mask in enumerate(masks):
    print(f"  Layer {i}: {mask.shape}")

# İlk mask'i görselleştir
plt.figure(figsize=(10, 8))
plt.imshow(masks[0].numpy(), cmap='RdBu', aspect='auto')
plt.colorbar(label='Mask Value')
plt.title('First Layer Mask (Input to Hidden)')
plt.xlabel('Hidden Units')
plt.ylabel('Input Features')
plt.show()

"""## Adım 2.2: MADE Network Sınıfı"""

class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation"""

    def __init__(self, input_dim, hidden_dims, activation=nn.ReLU):
        super(MADE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = 2 * input_dim  # μ ve log(σ)

        # Create masks
        self.masks = MADEMask.create_masks(
            input_dim, hidden_dims, self.output_dim
        )

        # Build network layers
        dims = [input_dim] + hidden_dims + [self.output_dim]
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i+1])
            self.layers.append(layer)

            if i < len(dims) - 2:  # Son layer'da activation yok
                self.activations.append(activation())

        # Register masks as buffers (GPU'ya taşınabilir)
        for i, mask in enumerate(self.masks):
            self.register_buffer(f'mask_{i}', mask)

    def forward(self, z):
        """
        Forward pass: z -> x

        Args:
            z: Latent variables (batch_size, input_dim)

        Returns:
            x: Generated samples (batch_size, input_dim)
        """
        x = z

        # Apply masked layers
        for i, layer in enumerate(self.layers):
            # Apply mask to weights
            mask = getattr(self, f'mask_{i}')
            weight = layer.weight * mask.t()

            # Linear transformation
            x = torch.nn.functional.linear(x, weight, layer.bias)

            # Activation (except last layer)
            if i < len(self.layers) - 1:
                x = self.activations[i](x)

        # Split output into μ and log(σ)
        mu, log_sigma = x.chunk(2, dim=1)

        # Compute x = μ + z * exp(log_σ)
        x = mu + z * torch.exp(log_sigma)

        return x, mu, log_sigma

    def inverse(self, x):
        """
        Inverse pass: x -> z

        Args:
            x: Data samples (batch_size, input_dim)

        Returns:
            z: Latent variables (batch_size, input_dim)
            log_det_jacobian: Log determinant of Jacobian
        """
        batch_size = x.shape[0]
        z = torch.zeros_like(x)
        log_det_jacobian = 0

        # Autoregressive inverse computation
        for i in range(self.input_dim):
            # Compute μ and log_σ for dimension i
            h = z if i == 0 else z

            for j, layer in enumerate(self.layers):
                mask = getattr(self, f'mask_{j}')
                weight = layer.weight * mask.t()
                h = torch.nn.functional.linear(h, weight, layer.bias)

                if j < len(self.layers) - 1:
                    h = self.activations[j](h)

            mu, log_sigma = h.chunk(2, dim=1)

            # Extract values for dimension i
            mu_i = mu[:, i]
            log_sigma_i = log_sigma[:, i]

            # Compute z_i
            z[:, i] = (x[:, i] - mu_i) / torch.exp(log_sigma_i)

            # Add to log determinant
            log_det_jacobian -= log_sigma_i

        return z, log_det_jacobian.sum(dim=1)

# Test MADE
made = MADE(input_dim=2, hidden_dims=[64, 64]).to(device)
print(f"MADE model: {sum(p.numel() for p in made.parameters())} parameters")

# Test forward pass
z_test = torch.randn(10, 2).to(device)
x_test, mu, log_sigma = made.forward(z_test)
print(f"\nForward pass:")
print(f"  Input z: {z_test.shape}")
print(f"  Output x: {x_test.shape}")
print(f"  μ: {mu.shape}, log(σ): {log_sigma.shape}")

"""## Adım 2.3: MADE Görselleştirme"""

def visualize_made_transform(made, n_samples=1000):
    """MADE'in z->x dönüşümünü görselleştir"""
    made.eval()

    with torch.no_grad():
        # Gaussian z samples
        z = torch.randn(n_samples, 2).to(device)
        x, _, _ = made.forward(z)

        z_np = z.cpu().numpy()
        x_np = x.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot z (input)
    axes[0].scatter(z_np[:, 0], z_np[:, 1], alpha=0.3, s=1)
    axes[0].set_title('Input: z ~ N(0, I)')
    axes[0].set_xlabel('z₁')
    axes[0].set_ylabel('z₂')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')

    # Plot x (output)
    axes[1].scatter(x_np[:, 0], x_np[:, 1], alpha=0.3, s=1)
    axes[1].set_title('Output: x = MADE(z)')
    axes[1].set_xlabel('x₁')
    axes[1].set_ylabel('x₂')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.show()

visualize_made_transform(made)

