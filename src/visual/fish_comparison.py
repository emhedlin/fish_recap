"""Functions for comparing and visualizing fish images."""

import pickle
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image
from sklearn.decomposition import PCA


def extract_side(image_name: str) -> Optional[str]:
    """Extract side from image name (L or R).
    
    Args:
        image_name: Image name containing side indicator
        
    Returns:
        'L' or 'R' if found, None otherwise
    """
    match = re.search(r'([LR])', image_name)
    return match.group(1) if match else None


def compare_fish(
    fish_id_1: str,
    fish_id_2: str,
    standardized_dir: Path,
    matches_df: pl.DataFrame,
    layout: str = "vertical",
) -> None:
    """Compare two fish images side by side or stacked vertically.
    
    Args:
        fish_id_1: First fish ID (e.g., "CSDD017R")
        fish_id_2: Second fish ID (e.g., "CSDD029R")
        standardized_dir: Path to directory containing standardized images
        matches_df: DataFrame containing match information
        layout: "vertical" (top/bottom) or "horizontal" (side by side)
    
    The function will automatically find the standardized images for both fish IDs.
    It searches for images matching the pattern: {fish_id}_standardized.png
    """
    # Construct image paths
    # Note: fish_id should be like "CSDD017R", image file is "CSDD017R_standardized.png"
    img1_path = standardized_dir / f"{fish_id_1}_standardized.png"
    img2_path = standardized_dir / f"{fish_id_2}_standardized.png"
    
    if not img1_path.exists():
        print(f"Error: Image not found for {fish_id_1}")
        print(f"  Expected path: {img1_path}")
        return
    
    if not img2_path.exists():
        print(f"Error: Image not found for {fish_id_2}")
        print(f"  Expected path: {img2_path}")
        return
    
    # Load images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Get match information if available
    # Note: CSV stores IDs without '_standardized' suffix
    match_info = None
    
    # Try to find match info (fish_id_1 as query, fish_id_2 as match)
    match_row = matches_df.filter(
        (pl.col("query_id") == fish_id_1) & (pl.col("match_id") == fish_id_2)
    )
    
    if len(match_row) > 0:
        match_info = match_row.to_dicts()[0]
    
    # Also try reverse (fish_id_2 as query, fish_id_1 as match)
    if match_info is None:
        match_row = matches_df.filter(
            (pl.col("query_id") == fish_id_2) & (pl.col("match_id") == fish_id_1)
        )
        if len(match_row) > 0:
            match_info = match_row.to_dicts()[0]
    
    if layout == "vertical":
        fig, axes = plt.subplots(2, 1, figsize=(10, 12), facecolor='black')
    else:  # horizontal
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='black')
    
    # Set axes background to black
    for ax in axes:
        ax.set_facecolor('black')
    
    # First image
    axes[0].imshow(img1)
    title1 = f"{fish_id_1}"
    if match_info:
        title1 += f"\nScore: {match_info['score']:.1f} | Inliers: {match_info['inliers']} | Ratio: {match_info['inlier_ratio']:.2%}"
    axes[0].set_title(title1, fontsize=14, fontweight='bold', color='white')
    axes[0].axis('off')
    
    # Second image
    axes[1].imshow(img2)
    title2 = f"{fish_id_2}"
    if match_info:
        title2 += f"\nTotal Matches: {match_info['total_matches']}"
    axes[1].set_title(title2, fontsize=14, fontweight='bold', color='white')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    if match_info:
        print(f"\nMatch Information:")
        print(f"  Score: {match_info['score']:.1f}")
        print(f"  Inliers: {match_info['inliers']}")
        print(f"  Total Matches: {match_info['total_matches']}")
        print(f"  Inlier Ratio: {match_info['inlier_ratio']:.2%}")
    else:
        print(f"\nNo match information found between {fish_id_1} and {fish_id_2}")
        print("  (These fish may not have been matched, or match score was below threshold)")


def get_top_matches(
    fish_id: str,
    matches_df: pl.DataFrame,
    top_n: int = 5,
) -> pl.DataFrame:
    """Get top N matches for a given fish ID.
    
    Args:
        fish_id: Fish ID (e.g., "CSDD017R") - without '_standardized' suffix
        matches_df: DataFrame containing match information
        top_n: Number of top matches to return
    
    Returns:
        DataFrame with top matches
    """
    # Note: CSV stores IDs without '_standardized' suffix
    top_matches = (
        matches_df
        .filter(pl.col("query_id") == fish_id)
        .sort("score", descending=True)
        .head(top_n)
    )
    
    return top_matches


def compare_with_top_match(
    fish_id: str,
    standardized_dir: Path,
    matches_df: pl.DataFrame,
    rank: int = 1,
) -> None:
    """Compare a fish with its top-ranked match.
    
    Args:
        fish_id: Fish ID (e.g., "CSDD017R")
        standardized_dir: Path to directory containing standardized images
        matches_df: DataFrame containing match information
        rank: Rank of match to compare (1 = best match, 2 = second best, etc.)
    """
    top_matches = get_top_matches(fish_id, matches_df, top_n=rank)
    
    if len(top_matches) < rank:
        print(f"Fish {fish_id} has fewer than {rank} matches")
        return
    
    match_row = top_matches.to_dicts()[rank - 1]
    match_fish_id = match_row["match_id"]
    
    # CSV already stores IDs without '_standardized' suffix
    print(f"Comparing {fish_id} with its #{rank} match: {match_fish_id}")
    compare_fish(fish_id, match_fish_id, standardized_dir, matches_df)


def dict_to_keypoints(kp_dicts: List[Dict[str, Any]]) -> List[cv2.KeyPoint]:
    """Convert dict format back to cv2.KeyPoint objects."""
    keypoints = []
    for kp_dict in kp_dicts:
        kp = cv2.KeyPoint()
        kp.pt = tuple(kp_dict["pt"])
        kp.size = kp_dict["size"]
        kp.angle = kp_dict["angle"]
        kp.response = kp_dict["response"]
        kp.octave = kp_dict["octave"]
        kp.class_id = kp_dict["class_id"]
        keypoints.append(kp)
    return keypoints


def load_features_from_pickle(
    image_path: Path,
    features_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Load features from pickle file.
    
    Args:
        image_path: Path to image file
        features_dir: Directory containing feature pickle files
        
    Returns:
        Dictionary with keypoints and descriptors, or None if not found
    """
    feature_path = features_dir / f"{image_path.stem}_features.pkl"
    
    if not feature_path.exists():
        return None
        
    try:
        with open(feature_path, 'rb') as f:
            features = pickle.load(f)
            # Convert keypoints back to cv2.KeyPoint objects if needed
            if "keypoints_list" in features and features.get("keypoints") is None:
                features["keypoints"] = dict_to_keypoints(features["keypoints_list"])
            elif isinstance(features.get("keypoints"), list) and len(features.get("keypoints", [])) > 0:
                # Check if first element is dict (serialized format)
                if isinstance(features["keypoints"][0], dict):
                    features["keypoints"] = dict_to_keypoints(features["keypoints"])
            return features
    except Exception as e:
        print(f"Failed to load features for {image_path.name}: {e}")
        return None


def _scale_keypoints(keypoints: List[cv2.KeyPoint], scale_factor: float) -> List[cv2.KeyPoint]:
    """Scale keypoint coordinates and sizes by a factor.
    
    Args:
        keypoints: List of cv2.KeyPoint objects
        scale_factor: Scaling factor (e.g., 0.5 for half size)
        
    Returns:
        List of scaled cv2.KeyPoint objects
    """
    scaled_kps = []
    for kp in keypoints:
        scaled_kp = cv2.KeyPoint()
        scaled_kp.pt = (kp.pt[0] * scale_factor, kp.pt[1] * scale_factor)
        scaled_kp.size = kp.size * scale_factor
        scaled_kp.angle = kp.angle
        scaled_kp.response = kp.response
        scaled_kp.octave = kp.octave
        scaled_kp.class_id = kp.class_id
        scaled_kps.append(scaled_kp)
    return scaled_kps


def visualize_single_image_features(
    image_path: Path,
    features_dir: Path,
    extractor: Any = None,
    force_recompute: bool = False,
    panels: Optional[List[str]] = None,
    resolution: float = 1.0,
    save_path: Optional[Path] = None,  # Add this parameter
    dpi: int = 150,  # Optional: control resolution of saved image
) -> None:
    """Visualize SIFT features for a single fish image.
    
    Creates three visualizations:
    1. Standard keypoints (location & scale)
    2. Feature similarity via PCA (128-dim → RGB colors)
    3. Feature density heatmap
    
    Args:
        image_path: Path to standardized fish image
        features_dir: Directory containing feature pickle files
        extractor: FeatureExtractor instance (optional, for on-demand extraction)
        force_recompute: Force re-extraction if features not found
        panels: List of panels to show. Options: 'location', 'similarity', 'density'.
                Default is None which shows all panels. Can also pass 'all'.
        resolution: Resolution scale factor (0.0 to 1.0). 1.0 = original resolution,
                    lower values scale down the image for better viewing. Default is 1.0.
        save_path: Optional path to save the figure. If None, only displays.
                   Supports common formats: .png, .jpg, .pdf, .svg
        dpi: Resolution for saved image (dots per inch). Default is 150.
    """
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    # Validate resolution
    if not 0 < resolution <= 1.0:
        print(f"Error: resolution must be between 0 and 1.0, got {resolution}")
        return
    
    # Validate and set panels
    valid_panels = {'location', 'similarity', 'density'}
    if panels is None or panels == 'all':
        panels = ['location', 'similarity', 'density']
    elif isinstance(panels, str):
        panels = [panels]
    
    # Validate panel names
    panels = [p.lower() for p in panels]
    invalid = set(panels) - valid_panels
    if invalid:
        print(f"Warning: Invalid panel names: {invalid}. Valid options: {valid_panels}")
        panels = [p for p in panels if p in valid_panels]
    
    if not panels:
        print("Error: No valid panels specified")
        return
    
    # Load image
    img = np.array(Image.open(image_path))
    if img.ndim == 2:  # Convert grayscale to RGB for plotting
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Scale image if resolution < 1.0
    original_shape = img.shape[:2]
    if resolution < 1.0:
        new_width = int(img.shape[1] * resolution)
        new_height = int(img.shape[0] * resolution)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Scaled image from {original_shape} to {img.shape[:2]} (resolution={resolution:.2f})")
    
    # Load features
    features = load_features_from_pickle(image_path, features_dir)
    
    if features is None:
        if extractor is None:
            print(f"Error: Features not found for {image_path.name} and no extractor provided")
            return
        # Extract features on-demand
        features = extractor.process_image(image_path)
    
    keypoints = features.get("keypoints", [])
    descriptors = features.get("descriptors", np.array([]))
    
    if len(keypoints) == 0:
        print(f"No features found for {image_path.name}")
        return
    
    # Scale keypoints if image was scaled
    if resolution < 1.0:
        keypoints = _scale_keypoints(keypoints, resolution)
    
    print(f"Found {len(keypoints)} features.")
    
    # Prepare visualizations
    visualizations = {}
    
    # --- Visualization 1: Basic Keypoints (Location & Scale) ---
    if 'location' in panels:
        vis_basic = img.copy()
        vis_basic = cv2.drawKeypoints(
            vis_basic,
            keypoints,
            None,
            flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )
        visualizations['location'] = {
            'image': vis_basic,
            'title': "Location & Scale\n(Circle size = Feature scale)"
        }
    
    # --- Visualization 2: Feature Similarity (PCA) ---
    if 'similarity' in panels:
        vis_pca = img.copy()
        if len(descriptors) > 3:
            # Normalize descriptors
            normalized_descs = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-7)
            
            # Apply PCA
            pca = PCA(n_components=3)
            reduced_descs = pca.fit_transform(normalized_descs)
            
            # Min-Max scale to 0-255 for color
            d_min, d_max = reduced_descs.min(axis=0), reduced_descs.max(axis=0)
            colors = ((reduced_descs - d_min) / (d_max - d_min + 1e-7) * 255).astype(int)
            
            # Draw keypoints with PCA colors
            for kp, color in zip(keypoints, colors):
                x, y = int(kp.pt[0]), int(kp.pt[1])
                r = max(int(kp.size / 2), 2)
                # cv2 uses BGR, PCA gave us RGB-like vectors
                c = (int(color[2]), int(color[1]), int(color[0]))
                
                # Draw filled circle with color
                cv2.circle(vis_pca, (x, y), r, c, 2)
                cv2.circle(vis_pca, (x, y), 2, c, -1)  # Center dot
            
            vis_pca = cv2.cvtColor(vis_pca, cv2.COLOR_BGR2RGB)
        visualizations['similarity'] = {
            'image': vis_pca,
            'title': "Feature Similarity\n(Same Color = Similar Texture)"
        }
    
    # --- Visualization 3: Density Heatmap ---
    if 'density' in panels:
        heatmap_data = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                heatmap_data[y, x] += 1
        
        # Blur to create heatmap effect (scale blur kernel with resolution)
        blur_size = max(1, int(41 * resolution))
        # Ensure odd number for GaussianBlur
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        heatmap_data = cv2.GaussianBlur(heatmap_data, (blur_size, blur_size), 0)
        
        # Normalize and color map
        if heatmap_data.max() > 0:
            heatmap_data = heatmap_data / (heatmap_data.max() + 1e-7)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_data), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        visualizations['density'] = {
            'image': overlay,
            'title': "Feature Density\n(Red = High Concentration)"
        }
    
    # Create figure with requested panels
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.67 * n_panels, 8))
    
    # Handle single panel case (axes is not iterable)
    if n_panels == 1:
        axes = [axes]
    
    # Display requested panels in order
    for idx, panel_name in enumerate(panels):
        if panel_name in visualizations:
            axes[idx].imshow(visualizations[panel_name]['image'])
            axes[idx].set_title(visualizations[panel_name]['title'], fontsize=12)
            axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save if path provided (before show, using figure object)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.1)
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_matched_features(
    image_path_1: Path,
    image_path_2: Path,
    features_dir: Path,
    matches: List[cv2.DMatch],
    extractor: Any = None,
    match_info: Optional[Dict[str, Any]] = None,
    save_path: Optional[Path] = None,  # Add this parameter
    dpi: int = 150,  # Optional: control resolution of saved image
) -> None:
    """Visualize matched features between two fish images.
    
    Shows:
    1. Side-by-side images with matched keypoints connected by lines
    2. Individual feature visualizations for each image
    
    Args:
        image_path_1: Path to first standardized fish image
        image_path_2: Path to second standardized fish image
        features_dir: Directory containing feature pickle files
        matches: List of cv2.DMatch objects connecting keypoints
        extractor: FeatureExtractor instance (optional, for on-demand extraction)
        match_info: Optional dictionary with match statistics (score, inliers, etc.)
        save_path: Optional path to save the figure. If None, only displays.
                   Supports common formats: .png, .jpg, .pdf, .svg
        dpi: Resolution for saved image (dots per inch). Default is 150.
    """
    if not image_path_1.exists():
        print(f"Error: Image not found at {image_path_1}")
        return
    if not image_path_2.exists():
        print(f"Error: Image not found at {image_path_2}")
        return
    
    # Load images
    img1 = np.array(Image.open(image_path_1))
    img2 = np.array(Image.open(image_path_2))
    
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    
    # Load features
    features1 = load_features_from_pickle(image_path_1, features_dir)
    features2 = load_features_from_pickle(image_path_2, features_dir)
    
    if features1 is None:
        if extractor is None:
            print(f"Error: Features not found for {image_path_1.name}")
            return
        features1 = extractor.process_image(image_path_1)
    
    if features2 is None:
        if extractor is None:
            print(f"Error: Features not found for {image_path_2.name}")
            return
        features2 = extractor.process_image(image_path_2)
    
    keypoints1 = features1.get("keypoints", [])
    keypoints2 = features2.get("keypoints", [])
    
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        print("Error: No features found in one or both images")
        return
    
    # Create matched features visualization
    img_matches = cv2.drawMatches(
        img1, keypoints1,
        img2, keypoints2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0),  # Green lines
        singlePointColor=(255, 0, 0),  # Red keypoints
    )
    
    # Convert BGR to RGB for matplotlib
    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    
    # Create figure with matched features on top, individual visualizations below
    fig = plt.figure(figsize=(20, 12))
    
    # Top row: Matched features
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(img_matches)
    title = f"Matched Features: {image_path_1.stem} ↔ {image_path_2.stem}"
    if match_info:
        title += f"\nScore: {match_info.get('score', 0):.1f} | "
        title += f"Inliers: {match_info.get('inliers', 0)} | "
        title += f"Total Matches: {match_info.get('total_matches', len(matches))}"
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Bottom row: Individual feature visualizations
    # Left: Image 1 features
    ax2 = plt.subplot(2, 2, 3)
    vis1 = img1.copy()
    vis1 = cv2.drawKeypoints(
        vis1, keypoints1, None,
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )
    ax2.imshow(vis1)
    ax2.set_title(f"{image_path_1.stem}\n{len(keypoints1)} features", fontsize=12)
    ax2.axis('off')
    
    # Right: Image 2 features
    ax3 = plt.subplot(2, 2, 4)
    vis2 = img2.copy()
    vis2 = cv2.drawKeypoints(
        vis2, keypoints2, None,
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )
    ax3.imshow(vis2)
    ax3.set_title(f"{image_path_2.stem}\n{len(keypoints2)} features", fontsize=12)
    ax3.axis('off')
    
    plt.tight_layout()
    
    # Save if path provided (before show, using figure object)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.1)
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    if match_info:
        print(f"\nMatch Information:")
        print(f"  Score: {match_info.get('score', 0):.1f}")
        print(f"  Inliers: {match_info.get('inliers', 0)}")
        print(f"  Total Matches: {match_info.get('total_matches', len(matches))}")
        if match_info.get('total_matches', 0) > 0:
            inlier_ratio = match_info.get('inliers', 0) / match_info.get('total_matches', 1)
            print(f"  Inlier Ratio: {inlier_ratio:.2%}")

