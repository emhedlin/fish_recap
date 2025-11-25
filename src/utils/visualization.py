"""Visualization utilities for QA plots and matching visualization.

This module provides functions for:
- QA plots showing detected ruler ticks and inferred scale
- Visualization of matching results
- Other diagnostic visualizations
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def visualize_ruler_detection(
    ruler_roi: np.ndarray,
    tick_positions: np.ndarray,
    pixels_per_mm: float,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Create QA visualization showing detected ticks and inferred scale.
    
    Args:
        ruler_roi: Ruler region of interest image
        tick_positions: Array of x-coordinates of detected ticks
        pixels_per_mm: Calculated pixels per millimeter
        output_path: Path to save the visualization
        dpi: Resolution for saved figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Display ruler ROI
    if ruler_roi.ndim == 3:
        ax.imshow(ruler_roi)
    else:
        ax.imshow(ruler_roi, cmap='gray')
    
    # Overlay tick positions
    h, w = ruler_roi.shape[:2]
    for tick_x in tick_positions:
        ax.axvline(x=tick_x, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add text annotation with scale
    ax.text(
        0.02, 0.98,
        f'Detected {len(tick_positions)} ticks\n'
        f'Pixels per mm: {pixels_per_mm:.2f}\n'
        f'Median tick spacing: {pixels_per_mm:.2f} px',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )
    
    ax.set_title('Ruler Tick Detection', fontsize=12, fontweight='bold')
    ax.set_xlabel('X position (pixels)')
    ax.set_ylabel('Y position (pixels)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.debug(f"Saved ruler detection visualization to {output_path}")


def visualize_fish_mask(
    image: np.ndarray,
    fish_mask: np.ndarray,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Visualize fish mask overlaid on original image.
    
    Args:
        image: Original image
        fish_mask: Binary fish mask
        output_path: Path to save visualization
        dpi: Resolution for saved figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Mask overlay
    overlay = image.copy()
    if overlay.ndim == 3:
        # Create colored mask overlay
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 1] = fish_mask * 255  # Green channel
        overlay = np.clip(overlay * 0.7 + mask_colored * 0.3, 0, 255).astype(np.uint8)
        axes[1].imshow(overlay)
    else:
        axes[1].imshow(image, cmap='gray')
        axes[1].imshow(fish_mask, alpha=0.5, cmap='Greens')
    
    axes[1].set_title('Fish Mask Overlay', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.debug(f"Saved fish mask visualization to {output_path}")


def plot_scale_distribution(
    pixels_per_mm_values: list[float],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Plot distribution of pixels_per_mm across all images.
    
    Args:
        pixels_per_mm_values: List of pixels_per_mm values for each image
        output_path: Path to save the plot
        dpi: Resolution for saved figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    values = np.array(pixels_per_mm_values)
    valid_values = values[values > 0]
    
    if len(valid_values) == 0:
        axes[0].text(0.5, 0.5, 'No valid data', ha='center', va='center')
        axes[1].text(0.5, 0.5, 'No valid data', ha='center', va='center')
    else:
        # Histogram
        axes[0].hist(valid_values, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(np.median(valid_values), color='red', linestyle='--', 
                       label=f'Median: {np.median(valid_values):.2f}')
        axes[0].axvline(np.mean(valid_values), color='blue', linestyle='--', 
                       label=f'Mean: {np.mean(valid_values):.2f}')
        axes[0].set_xlabel('Pixels per mm')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Scale Values', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(valid_values, vert=True)
        axes[1].set_ylabel('Pixels per mm')
        axes[1].set_title('Scale Values Box Plot', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = (
            f'Count: {len(valid_values)}\n'
            f'Mean: {np.mean(valid_values):.2f}\n'
            f'Median: {np.median(valid_values):.2f}\n'
            f'Std: {np.std(valid_values):.2f}\n'
            f'Min: {np.min(valid_values):.2f}\n'
            f'Max: {np.max(valid_values):.2f}'
        )
        axes[1].text(1.1, np.median(valid_values), stats_text,
                    verticalalignment='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.debug(f"Saved scale distribution plot to {output_path}")


def visualize_standardization(
    original_image: np.ndarray,
    standardized_image: np.ndarray,
    original_mask: np.ndarray,
    standardized_mask: np.ndarray,
    metadata: dict,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Visualize before/after standardization for QA.
    
    Args:
        original_image: Original image before standardization
        standardized_image: Standardized passport photo
        original_mask: Original fish mask
        standardized_mask: Standardized fish mask
        metadata: Metadata dictionary from standardization
        output_path: Path to save visualization
        dpi: Resolution for saved figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Original image with mask overlay
    overlay_orig = original_image.copy()
    if overlay_orig.ndim == 3:
        mask_colored = np.zeros_like(overlay_orig)
        mask_colored[:, :, 1] = original_mask.astype(np.uint8) * 255
        overlay_orig = np.clip(overlay_orig * 0.7 + mask_colored * 0.3, 0, 255).astype(np.uint8)
        axes[0, 0].imshow(overlay_orig)
    else:
        axes[0, 0].imshow(overlay_orig, cmap='gray')
        axes[0, 0].imshow(original_mask, alpha=0.5, cmap='Greens')
    axes[0, 0].set_title('Original Image + Mask', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Standardized image with mask overlay
    overlay_std = standardized_image.copy()
    if overlay_std.ndim == 3:
        mask_colored = np.zeros_like(overlay_std)
        mask_colored[:, :, 1] = standardized_mask.astype(np.uint8) * 255
        overlay_std = np.clip(overlay_std * 0.7 + mask_colored * 0.3, 0, 255).astype(np.uint8)
        axes[0, 1].imshow(overlay_std)
    else:
        axes[0, 1].imshow(overlay_std, cmap='gray')
        axes[0, 1].imshow(standardized_mask, alpha=0.5, cmap='Greens')
    axes[0, 1].set_title('Standardized Passport Photo', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Original image only
    if original_image.ndim == 3:
        axes[1, 0].imshow(original_image)
    else:
        axes[1, 0].imshow(original_image, cmap='gray')
    axes[1, 0].set_title('Original Image', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Standardized image only
    if standardized_image.ndim == 3:
        axes[1, 1].imshow(standardized_image)
    else:
        axes[1, 1].imshow(standardized_image, cmap='gray')
    axes[1, 1].set_title('Standardized Image', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add metadata text with confidence scores
    rotation_angle = metadata.get('rotation_angle', metadata.get('adjusted_angle', metadata.get('initial_orientation', 0)))
    head_conf = metadata.get('head_confidence', 'N/A')
    dorsal_conf = metadata.get('dorsal_confidence', 'N/A')
    validation_passed = metadata.get('validation_passed', 'N/A')
    validation_score = metadata.get('validation_score', 'N/A')
    
    metadata_text = (
        f"Rotation: {rotation_angle:.2f}°\n"
        f"Method: {metadata.get('rotation_method', 'unknown')}\n"
        f"Head on left: {metadata.get('head_on_left', 'N/A')}\n"
        f"Head confidence: {head_conf if isinstance(head_conf, str) else f'{head_conf:.2f}'}\n"
        f"Dorsal on top: {metadata.get('dorsal_on_top', 'N/A')}\n"
        f"Dorsal confidence: {dorsal_conf if isinstance(dorsal_conf, str) else f'{dorsal_conf:.2f}'}\n"
        f"Validation: {validation_passed if isinstance(validation_passed, bool) else 'N/A'}\n"
        f"Validation score: {validation_score if isinstance(validation_score, str) else f'{validation_score:.2f}' if validation_score is not None else 'N/A'}\n"
        f"Original size: {metadata.get('original_shape', 'N/A')}\n"
        f"Standardized size: {metadata.get('cropped_shape', 'N/A')}"
    )
    fig.text(0.5, 0.02, metadata_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.debug(f"Saved standardization visualization to {output_path}")
