# -*- coding: utf-8 -*-
"""
# Adım 4.1: Training Loop
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
import plotly.graph_objects as go
from IPython.display import HTML, display

import setup_and_data as sad
from maf_model import MAF, compute_loss
from setup_and_data import datasets

import warnings
warnings.filterwarnings('ignore')

# GPU/CPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MAFTrainer:
    """MAF model trainer with logging and visualization"""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_log_prob': [],
            'val_log_prob': []
        }

    def train_epoch(self, dataloader, optimizer):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_log_prob = 0
        n_batches = 0

        for batch in dataloader:
            x = batch[0].to(self.device)

            # Forward pass
            loss, log_probs = compute_loss(self.model, x)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_log_prob += log_probs.mean().item()
            n_batches += 1

        return epoch_loss / n_batches, epoch_log_prob / n_batches

    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        epoch_loss = 0
        epoch_log_prob = 0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)

                loss, log_probs = compute_loss(self.model, x)

                epoch_loss += loss.item()
                epoch_log_prob += log_probs.mean().item()
                n_batches += 1

        return epoch_loss / n_batches, epoch_log_prob / n_batches

    def train(self, train_loader, val_loader, n_epochs, lr=1e-3):
        """Full training loop"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(n_epochs):
            # Train
            train_loss, train_log_prob = self.train_epoch(
                train_loader, optimizer
            )

            # Validate
            val_loss, val_log_prob = self.validate(val_loader)

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_log_prob'].append(train_log_prob)
            self.history['val_log_prob'].append(val_log_prob)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}")
                print(f"  Train Loss: {train_loss:.4f} nats")
                print(f"  Val Loss: {val_loss:.4f} nats")
                print(f"  Val Log Prob: {val_log_prob:.4f} nats")

        return self.history

"""# Adım 4.2: Training Pipeline"""

# Veri setini train/val'a böl
def split_dataset(data, train_ratio=0.8):
    """Split data into train and validation sets"""
    n_train = int(len(data) * train_ratio)
    indices = torch.randperm(len(data))

    train_data = data[indices[:n_train]]
    val_data = data[indices[n_train:]]

    return train_data, val_data

# Her veri seti için model train et
trained_models = {}
training_histories = {}

for dataset_name in ['moons']:  # Başlangıç için sadece moons
    print(f"\n{'='*50}")
    print(f"Training MAF on {dataset_name} dataset")
    print(f"{'='*50}\n")

    # Split data
    data = datasets[dataset_name]
    train_data, val_data = split_dataset(data)

    # Create dataloaders
    train_loader = sad.create_dataloader(train_data, batch_size=256)
    val_loader = sad.create_dataloader(val_data, batch_size=256, shuffle=False)

    # Create model
    model = MAF(
        input_dim=2,
        hidden_dims=[128, 128],
        n_layers=5
    ).to(device)

    # Train
    trainer = MAFTrainer(model, device=device)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=50,
        lr=1e-3
    )

    # Store
    trained_models[dataset_name] = model
    training_histories[dataset_name] = history

    print(f"\nFinal validation loss: {history['val_loss'][-1]:.4f} nats")

"""# Adım 4.3: Training Curves Görselleştirme

"""

def plot_training_curves(history, title="Training History"):
    """Plot loss and log probability curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    axes[0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (nats)')
    axes[0].set_title('Negative Log-Likelihood')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Log probability curves
    axes[1].plot(epochs, history['train_log_prob'], label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_log_prob'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Log Probability (nats)')
    axes[1].set_title('Average Log Probability')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Görselleştir
for dataset_name, history in training_histories.items():
    plot_training_curves(
        history,
        title=f"MAF Training on {dataset_name.capitalize()} Dataset"
    )

"""# Adım 4.4: Sample Generation ve Karşılaştırma

"""

def compare_real_vs_generated(model, real_data, n_samples=2000, title=""):
    """Compare real data distribution with generated samples"""
    model.eval()

    # Generate samples
    with torch.no_grad():
        generated = model.sample(n_samples).cpu().numpy()

    real_data_np = real_data.cpu().numpy()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Real data
    axes[0].scatter(real_data_np[:n_samples, 0], real_data_np[:n_samples, 1],
                   alpha=0.3, s=1, c='blue')
    axes[0].set_title('Real Data')
    axes[0].set_xlabel('x₁')
    axes[0].set_ylabel('x₂')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')

    # Generated data
    axes[1].scatter(generated[:, 0], generated[:, 1],
                   alpha=0.3, s=1, c='red')
    axes[1].set_title('Generated Samples')
    axes[1].set_xlabel('x₁')
    axes[1].set_ylabel('x₂')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')

    # Overlay
    axes[2].scatter(real_data_np[:n_samples, 0], real_data_np[:n_samples, 1],
                   alpha=0.3, s=1, c='blue', label='Real')
    axes[2].scatter(generated[:, 0], generated[:, 1],
                   alpha=0.3, s=1, c='red', label='Generated')
    axes[2].set_title('Overlay')
    axes[2].set_xlabel('x₁')
    axes[2].set_ylabel('x₂')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Karşılaştır
for dataset_name, model in trained_models.items():
    compare_real_vs_generated(
        model,
        datasets[dataset_name],
        title=f"MAF on {dataset_name.capitalize()} Dataset"
    )

"""# Adım 4.5: Density Heatmap"""

def plot_learned_density(model, xlim=(-3, 3), ylim=(-3, 3), n_points=200):
    """Plot the learned probability density as a heatmap"""
    model.eval()

    # Create grid
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)

    # Flatten grid
    grid = np.stack([X.flatten(), Y.flatten()], axis=1)
    grid_tensor = torch.FloatTensor(grid).to(device)

    # Compute log probabilities in batches
    batch_size = 1000
    log_probs = []

    with torch.no_grad():
        for i in range(0, len(grid_tensor), batch_size):
            batch = grid_tensor[i:i+batch_size]
            log_prob = model.log_probs(batch)
            log_probs.append(log_prob.cpu())

    log_probs = torch.cat(log_probs).numpy()
    probs = np.exp(log_probs).reshape(X.shape)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, probs, levels=20, cmap='viridis')
    plt.colorbar(label='Probability Density')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Learned Probability Density')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

# Görselleştir
for dataset_name, model in trained_models.items():
    print(f"\nDensity for {dataset_name}:")
    plot_learned_density(model)

