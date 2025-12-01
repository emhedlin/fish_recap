"""Functions for comparing and visualizing fish images."""

import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import polars as pl
from PIL import Image


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

