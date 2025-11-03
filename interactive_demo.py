import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
import plotly.graph_objects as go
from IPython.display import HTML, display
import os 

from training_and_visualization import trained_models, training_histories, split_dataset
from setup_and_data import datasets, create_dataloader
from maf_model import MAF

import warnings
warnings.filterwarnings('ignore')

# GPU/CPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def animate_flow_layers(model, n_samples=1000, save_gif=False):
    """Animate the transformation through each flow layer"""
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    model.eval()

    # Start with Gaussian samples
    with torch.no_grad():
        z = torch.randn(n_samples, 2).to(device)

        # Track transformation through each layer
        transformations = [z.cpu().numpy()]
        x = z
        for i, made_block in enumerate(model.made_blocks):
            x, _, _ = made_block.forward(x)
            transformations.append(x.cpu().numpy())

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 10))

    def update(frame):
        ax.clear()
        data = transformations[frame]
        ax.scatter(data[:, 0], data[:, 1], alpha=0.3, s=1, c='blue')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('x₁', fontsize=12)
        ax.set_ylabel('x₂', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        if frame == 0:
            ax.set_title('Layer 0: z ~ N(0, I)', fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Layer {frame}: After MADE Block {frame}',
                        fontsize=14, fontweight='bold')

    anim = FuncAnimation(fig, update, frames=len(transformations),
                        interval=800, repeat=True)

    plt.close()

    if save_gif:
        anim.save('maf_flow_animation.gif', writer='pillow', fps=1)
        print("Animation saved as maf_flow_animation.gif")

    return HTML(anim.to_jshtml())

# Animasyonu göster
for dataset_name, model in trained_models.items():
    print(f"\nFlow transformation for {dataset_name}:")
    display(animate_flow_layers(model, n_samples=2000))

import ipywidgets as widgets
from IPython.display import display, clear_output

def interactive_sampling_demo(model):
    """Interactive widget for controlling sample generation"""

    # Widgets
    n_samples_slider = widgets.IntSlider(
        value=1000,
        min=100,
        max=5000,
        step=100,
        description='Samples:',
        style={'description_width': '100px'}
    )

    temperature_slider = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=2.0,
        step=0.1,
        description='Temperature:',
        style={'description_width': '100px'}
    )

    regenerate_button = widgets.Button(
        description='Regenerate',
        button_style='success',
        icon='refresh'
    )

    output = widgets.Output()

    def plot_samples(n_samples, temperature):
        """Generate and plot samples"""
        model.eval()

        with torch.no_grad():
            # Sample with temperature
            z = torch.randn(n_samples, 2).to(device) * temperature
            x = model.forward(z)
            samples = x.cpu().numpy()

        with output:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1)
            ax.set_xlabel('x₁', fontsize=12)
            ax.set_ylabel('x₂', fontsize=12)
            ax.set_title(f'Generated Samples (n={n_samples}, T={temperature:.1f})',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.show()

    def on_button_click(b):
        plot_samples(n_samples_slider.value, temperature_slider.value)

    def on_slider_change(change):
        plot_samples(n_samples_slider.value, temperature_slider.value)

    regenerate_button.on_click(on_button_click)
    n_samples_slider.observe(on_slider_change, names='value')
    temperature_slider.observe(on_slider_change, names='value')

    # Initial plot
    plot_samples(n_samples_slider.value, temperature_slider.value)

    # Display widgets
    display(widgets.VBox([
        widgets.HBox([n_samples_slider, temperature_slider, regenerate_button]),
        output
    ]))

# İnteraktif demo
print("Interactive Sampling Demo:")
for dataset_name, model in trained_models.items():
    print(f"\n{dataset_name.upper()}:")
    interactive_sampling_demo(model)

def latent_interpolation(model, n_steps=50, n_samples=5):
    """Interpolate between random points in latent space"""
    model.eval()

    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))
    if n_samples == 1:
        axes = [axes]

    with torch.no_grad():
        for idx in range(n_samples):
            # Random start and end points
            z_start = torch.randn(1, 2).to(device)
            z_end = torch.randn(1, 2).to(device)

            # Interpolate
            alphas = torch.linspace(0, 1, n_steps).to(device)
            z_interp = torch.stack([
                alpha * z_end + (1 - alpha) * z_start
                for alpha in alphas
            ]).squeeze(1)

            # Transform
            x_interp = model.forward(z_interp).cpu().numpy()

            # Plot
            ax = axes[idx]

            # Plot trajectory
            ax.plot(x_interp[:, 0], x_interp[:, 1],
                   'b-', alpha=0.6, linewidth=2)

            # Plot start and end
            ax.scatter(x_interp[0, 0], x_interp[0, 1],
                      c='green', s=100, marker='o',
                      label='Start', zorder=5)
            ax.scatter(x_interp[-1, 0], x_interp[-1, 1],
                      c='red', s=100, marker='s',
                      label='End', zorder=5)

            # Formatting
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.set_title(f'Interpolation {idx+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

    plt.suptitle('Latent Space Interpolation',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# Görselleştir
for dataset_name, model in trained_models.items():
    print(f"\nInterpolation for {dataset_name}:")
    latent_interpolation(model, n_steps=50, n_samples=5)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_3d_density_interactive(model, xlim=(-3, 3), ylim=(-3, 3), n_points=100):
    """Interactive 3D density visualization with Plotly"""
    model.eval()

    # Create grid
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)

    # Compute densities
    grid = np.stack([X.flatten(), Y.flatten()], axis=1)
    grid_tensor = torch.FloatTensor(grid).to(device)

    with torch.no_grad():
        log_probs = []
        batch_size = 1000

        for i in range(0, len(grid_tensor), batch_size):
            batch = grid_tensor[i:i+batch_size]
            log_prob = model.log_probs(batch)
            log_probs.append(log_prob.cpu())

        log_probs = torch.cat(log_probs).numpy()

    Z = np.exp(log_probs).reshape(X.shape)

    # Create 3D surface plot
    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            name='Density',
            showscale=True
        )
    ])

    fig.update_layout(
        title='3D Probability Density',
        scene=dict(
            xaxis_title='x₁',
            yaxis_title='x₂',
            zaxis_title='p(x)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        width=800,
        height=700
    )

    fig.show()

# İnteraktif 3D görselleştirme
for dataset_name, model in trained_models.items():
    print(f"\n3D Density for {dataset_name}:")
    plot_3d_density_interactive(model)

def create_comparison_dashboard(models_dict, datasets_dict, n_samples=1000):
    """Create comprehensive comparison dashboard"""
    n_models = len(models_dict)

    fig = make_subplots(
        rows=2, cols=n_models,
        subplot_titles=[f'{name.capitalize()} - Real' for name in models_dict.keys()] +
                       [f'{name.capitalize()} - Generated' for name in models_dict.keys()],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    for idx, (name, model) in enumerate(models_dict.items(), 1):
        model.eval()

        # Real data
        real_data = datasets_dict[name].numpy()
        fig.add_trace(
            go.Scatter(
                x=real_data[:n_samples, 0],
                y=real_data[:n_samples, 1],
                mode='markers',
                marker=dict(size=2, opacity=0.5, color='blue'),
                name=f'{name} Real',
                showlegend=False
            ),
            row=1, col=idx
        )

        # Generated data
        with torch.no_grad():
            generated = model.sample(n_samples).cpu().numpy()

        fig.add_trace(
            go.Scatter(
                x=generated[:, 0],
                y=generated[:, 1],
                mode='markers',
                marker=dict(size=2, opacity=0.5, color='red'),
                name=f'{name} Generated',
                showlegend=False
            ),
            row=2, col=idx
        )

    fig.update_xaxes(title_text="x₁")
    fig.update_yaxes(title_text="x₂")

    fig.update_layout(
        title_text="MAF Model Comparison Dashboard",
        height=800,
        showlegend=False
    )

    fig.show()

# Dashboard oluştur
if len(trained_models) > 1:
    create_comparison_dashboard(trained_models, datasets)

def compute_evaluation_metrics(model, test_data, n_samples=1000):
    """Compute comprehensive evaluation metrics"""
    model.eval()

    metrics = {}

    with torch.no_grad():
        # 1. Test log-likelihood
        test_loader = create_dataloader(test_data, batch_size=256, shuffle=False)
        log_probs = []

        for batch in test_loader:
            x = batch[0].to(device)
            log_prob = model.log_probs(x)
            log_probs.append(log_prob)

        log_probs = torch.cat(log_probs)
        metrics['test_log_likelihood'] = log_probs.mean().item()
        metrics['test_log_likelihood_std'] = log_probs.std().item()

        # 2. Sample quality - coverage
        generated = model.sample(n_samples).cpu()

        # Compute pairwise distances (simplified coverage metric)
        from scipy.spatial.distance import cdist

        real_subset = test_data[:n_samples].cpu().numpy()
        gen_subset = generated.numpy()

        # Minimum distance from each generated point to real data
        distances = cdist(gen_subset, real_subset)
        min_distances = distances.min(axis=1)

        metrics['mean_min_distance'] = min_distances.mean()
        metrics['coverage_threshold_01'] = (min_distances < 0.1).mean()

        # 3. Forward KL estimate (if available)
        sample_log_probs = model.log_probs(generated.to(device))
        metrics['sample_log_likelihood'] = sample_log_probs.mean().item()

    return metrics

def display_metrics_table(metrics_dict):
    """Display metrics in a formatted table"""
    import pandas as pd

    df = pd.DataFrame(metrics_dict).T
    df = df.round(4)

    print("\n" + "="*80)
    print("MODEL EVALUATION METRICS")
    print("="*80)
    print(df.to_string())
    print("="*80)

    return df

# Compute metrics for all models
all_metrics = {}

for dataset_name, model in trained_models.items():
    _, val_data = split_dataset(datasets[dataset_name])
    metrics = compute_evaluation_metrics(model, val_data)
    all_metrics[dataset_name] = metrics

# Display
metrics_df = display_metrics_table(all_metrics)

def save_model(model, history, dataset_name, save_dir='maf_models'):
    """Save trained model and training history"""
    import os
    import pickle

    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    model_path = os.path.join(save_dir, f'maf_{dataset_name}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.input_dim,
            'n_layers': model.n_layers,
        }
    }, model_path)

    # Save history
    history_path = os.path.join(save_dir, f'history_{dataset_name}.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    print(f"Model saved to {model_path}")
    print(f"History saved to {history_path}")

def load_model(dataset_name, hidden_dims=[128, 128], save_dir='maf_models'):
    """Load trained model"""
    import pickle

    model_path = os.path.join(save_dir, f'maf_{dataset_name}.pth')
    history_path = os.path.join(save_dir, f'history_{dataset_name}.pkl')

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create model
    model = MAF(
        input_dim=checkpoint['model_config']['input_dim'],
        hidden_dims=hidden_dims,
        n_layers=checkpoint['model_config']['n_layers']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    # Load history
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    print(f"Model loaded from {model_path}")
    return model, history

# Save all trained models
for dataset_name, model in trained_models.items():
    save_model(model, training_histories[dataset_name], dataset_name)

# Example: Load model
# loaded_model, loaded_history = load_model('moons')