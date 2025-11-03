# -*- coding: utf-8 -*-
"""
## Adım 1.1: Environment Setup
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
import warnings
warnings.filterwarnings('ignore')

# GPU/CPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""## Adım 1.2: Veri Setleri Oluşturma"""

class DatasetGenerator:
    """Farklı 2D veri setleri oluşturmak için sınıf"""

    @staticmethod
    def make_moons(n_samples=10000, noise=0.05):
        """Moons veri seti"""
        X, _ = make_moons(n_samples=n_samples, noise=noise)
        return torch.FloatTensor(X)

    @staticmethod
    def make_circles(n_samples=10000, noise=0.05):
        """İç içe circles veri seti"""
        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5)
        return torch.FloatTensor(X)

    @staticmethod
    def make_swiss_roll_2d(n_samples=10000, noise=0.5):
        """Swiss roll'u 2D'ye project et"""
        X, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
        X = X[:, [0, 2]]  # x ve z koordinatlarını al
        # Normalize
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        return torch.FloatTensor(X)

    @staticmethod
    def make_pinwheel(n_samples=10000, n_arms=5, noise=0.1):
        """Pinwheel/Yıldız şekli"""
        rng = np.random.RandomState(42)
        radial = rng.randn(n_samples, 1) * 0.3 + 1.0
        tangent = rng.randn(n_samples, 1) * 0.05

        rate = 0.25
        angles = rng.rand(n_samples, 1) * 2 * np.pi / n_arms

        for i in range(n_arms):
            angle_shift = 2 * np.pi * i / n_arms
            mask = (angles + angle_shift) < (2 * np.pi)
            x = radial[mask.flatten()] * np.cos(angles[mask.flatten()] + angle_shift)
            y = radial[mask.flatten()] * np.sin(angles[mask.flatten()] + angle_shift)

            if i == 0:
                data = np.hstack([x, y])
            else:
                data = np.vstack([data, np.hstack([x, y])])

        data = data + rng.randn(*data.shape) * noise
        return torch.FloatTensor(data[:n_samples])

# Veri setlerini oluştur
datasets = {
    'moons': DatasetGenerator.make_moons(),
    'circles': DatasetGenerator.make_circles(),
    'swiss_roll': DatasetGenerator.make_swiss_roll_2d(),
    'pinwheel': DatasetGenerator.make_pinwheel()
}

"""## Adım 1.3: Veri Görselleştirme"""

def visualize_datasets(datasets, figsize=(16, 4)):
    """Tüm veri setlerini görselleştir"""
    fig, axes = plt.subplots(1, len(datasets), figsize=figsize)

    for idx, (name, data) in enumerate(datasets.items()):
        ax = axes[idx]
        ax.scatter(data[:, 0], data[:, 1], alpha=0.3, s=1)
        ax.set_title(f'{name.capitalize()} Dataset')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

visualize_datasets(datasets)

"""## Adım 1.4: DataLoader Oluşturma"""

from torch.utils.data import TensorDataset, DataLoader

def create_dataloader(data, batch_size=256, shuffle=True):
    """PyTorch DataLoader oluştur"""
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Tüm veri setleri için DataLoader'lar
dataloaders = {
    name: create_dataloader(data)
    for name, data in datasets.items()
}

print("DataLoader'lar hazır:")
for name, dl in dataloaders.items():
    print(f"  {name}: {len(dl)} batches")

