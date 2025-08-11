# Helper function: javascript visualization
import numpy as np
import json
from IPython.display import display, HTML
import string
import random

# Helper function to plot attention patterns, hide this function

import matplotlib.pyplot as plt
import numpy as np


def plot_attn_heads(
    total_activations,
    n_heads=12,
    n_layers=12,
    img_shape=50,
    idx=0,
    figsize=(20, 20),
    global_min_max=False,
    global_normalize=False,
    fourier_transform_local=False,
    log_transform=False,
    fourier_transform_global=False,
    graph_type="imshow_graph",
    cmap="viridis",
):

    # New shape handling: total_activations is now expected to be of shape [n_layers*n_heads, img_shape, img_shape]
    total_data = np.zeros((n_layers * n_heads, img_shape, img_shape))

    # Adjusted processing for flattened layer-heads structure
    if global_min_max or global_normalize or fourier_transform_global:
        for i in range(n_layers * n_heads):
            data = total_activations[i, :, :]
            if log_transform:
                data = np.log10(np.maximum(data, 1e-6))  # log10_stable equivalent
            if fourier_transform_global:
                data = np.abs(np.fft.fftshift(np.fft.fft2(data)))
            total_data[i, :, :] = data

        total_min, total_max = np.min(total_data), np.max(total_data)
        print(f"Total Min: {total_min}, Total Max: {total_max}")

        if global_normalize:
            total_data = -1 + 2 * (total_data - total_min) / (total_max - total_min)

    fig, axes = plt.subplots(
        n_layers, n_heads, figsize=figsize, squeeze=False
    )  # Ensure axes is always 2D array
    total_data_dict = {}

    for i in range(n_layers):
        total_data_dict[f"Layer_{i}"] = {}
        for j in range(n_heads):
            # Adjust indexing for the flattened layer-head structure
            linear_idx = i * n_heads + j
            data = total_data[linear_idx, :, :]

            if graph_type == "histogram_graph":
                data = data.flatten()
                axes[i, j].hist(data, bins=100, log=log_transform, cmap=cmap)
            elif graph_type == "imshow_graph":
                if fourier_transform_local:
                    data = np.abs(np.fft.fftshift(np.fft.fft2(data)))
                vmin, vmax = (
                    (total_min, total_max)
                    if (global_min_max or global_normalize)
                    else (data.min(), data.max())
                )
                im = axes[i, j].imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
                axes[i, j].axis("off")
                total_data_dict[f"Layer_{i}"][f"Head_{j}"] = data.tolist()

            axes[i, j].set_title(f"Head {j}", fontsize=12, pad=5) if i == 0 else None
            if j == 0:
                axes[i, j].text(
                    -0.3,
                    0.5,
                    f"Layer {i}",
                    fontsize=12,
                    rotation=90,
                    ha="center",
                    va="center",
                    transform=axes[i, j].transAxes,
                )

    # Add colorbar for imshow_graph
    if graph_type == "imshow_graph" and (global_min_max or global_normalize):
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        cbar_ax.set_title("Attention", size=12)

    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.suptitle(f"Attention for Image Idx {idx}", fontsize=20, y=0.93)
    plt.show()


def plot_single_attn_head(
    total_activations,
    layer_idx,
    head_idx,
    n_heads=12,
    img_shape=50,
    idx=0,
    figsize=(8, 6),
    fourier_transform_local=False,
    log_transform=False,
    graph_type="imshow_graph",
    cmap="viridis",
    image=None,
    show_side_by_side=False,
):
    """
    Plot a single attention head from specified layer and head indices.
    
    Args:
        total_activations: Attention data of shape [n_layers*n_heads, img_shape, img_shape]
        layer_idx: Index of the layer (0-based)
        head_idx: Index of the head within the layer (0-based)
        n_heads: Number of heads per layer
        img_shape: Shape of the attention map (assumed square)
        idx: Image index for title
        figsize: Figure size tuple
        fourier_transform_local: Whether to apply Fourier transform
        log_transform: Whether to apply log transform
        graph_type: Type of plot ("imshow_graph" or "histogram_graph")
        cmap: Colormap for visualization
        image: Original image to display side by side (optional)
        show_side_by_side: Whether to show image and attention side by side
    """
    
    # Calculate linear index for the flattened layer-head structure
    linear_idx = layer_idx * n_heads + head_idx
    
    # Extract data for the specific head
    data = total_activations[linear_idx, :, :]
    
    # Apply transformations if requested
    if log_transform:
        data = np.log10(np.maximum(data, 1e-6))
    
    if fourier_transform_local:
        data = np.abs(np.fft.fftshift(np.fft.fft2(data)))
    
    # Create figure
    if show_side_by_side and image is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 1.5, figsize[1]))
        
        # Plot original image
        if len(image.shape) == 3:  # RGB image
            ax1.imshow(image)
        else:  # Grayscale image
            ax1.imshow(image, cmap='gray')
        ax1.axis("off")
        ax1.set_title(f"Original Image (Idx {idx})")
        
        # Plot attention on the second subplot
        ax = ax2
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if graph_type == "histogram_graph":
        data_flat = data.flatten()
        ax.hist(data_flat, bins=100, log=log_transform, color='skyblue', alpha=0.7)
        ax.set_xlabel("Attention Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Attention Distribution - Layer {layer_idx}, Head {head_idx}")
    elif graph_type == "imshow_graph":
        im = ax.imshow(data, cmap=cmap)
        ax.axis("off")
        ax.set_title(f"Attention Pattern - Layer {layer_idx}, Head {head_idx}")
        
        # Add colorbar
        if show_side_by_side and image is not None:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention", rotation=270, labelpad=15)
    
    if not (show_side_by_side and image is not None):
        plt.suptitle(f"Image Idx {idx}", fontsize=14, y=0.95)
    plt.tight_layout()
    plt.show()
    
    return data

