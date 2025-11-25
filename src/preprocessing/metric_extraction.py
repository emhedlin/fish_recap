"""Metric extraction module for ruler detection and scale calculation.

This module handles:
- ROI extraction using ruler mask
- Tick detection using Sobel filters
- Scale calculation (pixels per mm)
- Fish length measurement (medial axis or bounding box approach)
- QA visualization of detected ticks and inferred scale
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.ndimage import label
from scipy.signal import find_peaks
from skimage import measure, morphology

logger = logging.getLogger(__name__)


def extract_ruler_roi(
    image: np.ndarray,
    ruler_mask: np.ndarray,
    margin: int = 10,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Extract ruler region of interest from image using ruler mask.
    
    Args:
        image: Original image (H, W, 3) or (H, W)
        ruler_mask: Binary mask of ruler (H, W) with values 0 or 1
        margin: Additional margin in pixels around bounding box
    
    Returns:
        Tuple of (ruler_roi, bbox) where bbox is (x_min, y_min, x_max, y_max)
    
    Raises:
        ValueError: If image and mask dimensions don't match
    """
    # Ensure mask is binary
    if ruler_mask.dtype != bool:
        ruler_mask = ruler_mask > 0.5
    
    # Validate dimensions match
    image_h, image_w = image.shape[:2]
    mask_h, mask_w = ruler_mask.shape[:2]
    
    if image_h != mask_h or image_w != mask_w:
        raise ValueError(
            f"Image and mask dimensions don't match: "
            f"image={image_h}x{image_w}, mask={mask_h}x{mask_w}. "
            f"This may indicate an EXIF rotation issue."
        )
    
    # Find bounding box of ruler
    coords = np.column_stack(np.where(ruler_mask))
    
    if len(coords) == 0:
        logger.warning("Ruler mask is empty (no ruler detected), cannot extract ROI")
        return np.array([]), (0, 0, 0, 0)
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Add margin
    h, w = ruler_mask.shape
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w, x_max + margin)
    y_max = min(h, y_max + margin)
    
    # Extract ROI
    if image.ndim == 3:
        ruler_roi = image[y_min:y_max, x_min:x_max, :]
    else:
        ruler_roi = image[y_min:y_max, x_min:x_max]
    
    bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
    logger.debug(f"Extracted ruler ROI: bbox={bbox}, shape={ruler_roi.shape}")
    
    return ruler_roi, bbox


def detect_ruler_ticks(
    ruler_roi: np.ndarray,
    method: str = "sobel",
    min_tick_spacing: int = 15,
    max_tick_spacing: int = 100,
) -> Tuple[np.ndarray, float]:
    """Detect tick marks on ruler and calculate pixels per millimeter.
    
    Args:
        ruler_roi: Ruler region of interest image
        method: Detection method ("sobel" or "hough")
        min_tick_spacing: Minimum expected spacing between ticks in pixels
        max_tick_spacing: Maximum expected spacing between ticks in pixels
    
    Returns:
        Tuple of (tick_positions, pixels_per_mm) where tick_positions is array of x-coordinates
    """
    # Convert to grayscale if needed
    if ruler_roi.ndim == 3:
        gray = cv2.cvtColor(ruler_roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = ruler_roi.copy()
    
    if method == "sobel":
        # Apply vertical Sobel filter to detect vertical tick marks
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y = np.abs(sobel_y)
        
        # Threshold to find strong vertical edges
        _, binary = cv2.threshold(
            sobel_y.astype(np.uint8),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Find vertical lines (ticks)
        # Project onto horizontal axis (sum along columns)
        horizontal_projection = np.sum(binary, axis=0)
        
        # --- FIX: Use Peak Finding instead of raw thresholding ---
        # Previous logic selected every pixel above threshold (contiguous blocks).
        # We want the PEAK of those blocks.
        
        # Calculate dynamic height threshold based on the signal
        height_threshold = np.percentile(horizontal_projection, 70)
        
        tick_positions, _ = find_peaks(
            horizontal_projection, 
            height=height_threshold, 
            distance=min_tick_spacing
        )
        
    else:
        raise ValueError(f"Unknown detection method: {method}")
    
    if len(tick_positions) < 2:
        logger.warning(f"Found only {len(tick_positions)} tick positions, cannot calculate scale")
        return np.array([]), 0.0
    
    # Calculate distances between consecutive ticks
    tick_distances = np.diff(np.sort(tick_positions))
    
    # Filter out distances that are too small or too large
    # We accept a wider range of valid distances now that peak finding is robust
    valid_distances = tick_distances[
        (tick_distances >= min_tick_spacing) & (tick_distances <= max_tick_spacing)
    ]
    
    if len(valid_distances) == 0:
        logger.warning("No valid tick distances found")
        return tick_positions, 0.0
    
    # Calculate median distance (most robust to outliers)
    median_distance = np.median(valid_distances)
    
    # Assume ticks are millimeter marks
    # pixels_per_mm = median_distance_px (since distance between mm ticks is 1mm)
    pixels_per_mm = float(median_distance)
    
    logger.debug(
        f"Detected {len(tick_positions)} ticks, "
        f"median spacing: {median_distance:.2f} px, "
        f"pixels_per_mm: {pixels_per_mm:.2f}"
    )
    
    return tick_positions, pixels_per_mm


def measure_fish_length_medial_axis(
    fish_mask: np.ndarray,
    pixels_per_mm: float,
) -> Tuple[float, float]:
    """Measure fish length using medial axis (skeletonization).
    
    Args:
        fish_mask: Binary mask of fish (H, W) with values 0 or 1
        pixels_per_mm: Scale factor from ruler detection
    
    Returns:
        Tuple of (length_px, length_mm)
    """
    # Ensure mask is binary
    if fish_mask.dtype != bool:
        fish_mask = fish_mask > 0.5
    
    if np.sum(fish_mask) == 0:
        logger.warning("Fish mask is empty, cannot measure length")
        return 0.0, 0.0
    
    try:
        # Skeletonize the fish mask to get medial axis
        # Convert to uint8 to ensure compatibility
        skeleton = morphology.skeletonize(fish_mask.astype(np.uint8))
    except Exception as e:
        logger.warning(f"Skeletonization failed: {e}. Falling back to bounding box method.")
        return measure_fish_length_bounding_box(fish_mask, pixels_per_mm)
    
    # --- FIX: Robust Length Calculation ---
    # Previous logic failed if the skeleton was fragmented (common).
    # It would pick the "longest component" which might be a small rib or noise (146px).
    
    # Geodesic / Pixel Counting approach
    # For a 1-pixel wide skeleton, the area (sum) is a decent approximation of length.
    total_skeleton_pixels = int(np.sum(skeleton))  # Convert to Python int to avoid numpy scalar issues
    
    if total_skeleton_pixels == 0:
        logger.warning("Skeletonization produced no features")
        return 0.0, 0.0
    
    # Fallback sanity check: Compare with Major Axis
    # Skeleton length should be roughly similar to the Ellipse Major Axis
    contours = measure.find_contours(fish_mask.astype(float), 0.5)
    major_axis_length = 0.0
    if contours:
        largest_contour = max(contours, key=len)
        if len(largest_contour) >= 5:
            contour_xy = np.column_stack([largest_contour[:, 1], largest_contour[:, 0]]).astype(np.float32)
            ellipse = cv2.fitEllipse(contour_xy)
            major_axis_length = float(max(ellipse[1]))  # Convert to Python float
    
    # If skeleton is severely broken (length < 50% of major axis), use major axis
    if major_axis_length > 0 and total_skeleton_pixels < (0.5 * major_axis_length):
        logger.warning(
            f"Skeleton fragmented ({total_skeleton_pixels}px vs axis {major_axis_length:.0f}px). "
            f"Falling back to major axis."
        )
        length_px = float(major_axis_length)
    else:
        # Refined skeleton measurement: Count pixels
        # A perfect line of 10 pixels has length 10. Diagonals add distance.
        # A simple heuristic is sufficient here given the noise.
        length_px = float(total_skeleton_pixels)
    
    # Convert to mm
    if pixels_per_mm > 0:
        length_mm = length_px / pixels_per_mm
    else:
        length_mm = 0.0
        logger.warning("pixels_per_mm is 0, cannot convert to mm")
    
    logger.debug(
        f"Medial axis length: {length_px:.2f} px = {length_mm:.2f} mm"
    )
    
    return length_px, length_mm


def measure_fish_length_bounding_box(
    fish_mask: np.ndarray,
    pixels_per_mm: float,
    method: str = "major_axis",
) -> Tuple[float, float]:
    """Measure fish length using bounding box or major axis.
    
    Args:
        fish_mask: Binary mask of fish (H, W) with values 0 or 1
        pixels_per_mm: Scale factor from ruler detection
        method: "bounding_box" or "major_axis"
    
    Returns:
        Tuple of (length_px, length_mm)
    """
    # Ensure mask is binary
    if fish_mask.dtype != bool:
        fish_mask = fish_mask > 0.5
    
    if method == "bounding_box":
        # Use bounding box diagonal or longest side
        coords = np.column_stack(np.where(fish_mask))
        if len(coords) == 0:
            return 0.0, 0.0
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Use longer dimension as length
        length_px = float(max(width, height))
    
    elif method == "major_axis":
        # Use major axis length from image moments
        # Find contours
        contours = measure.find_contours(fish_mask.astype(float), 0.5)
        
        if not contours:
            return 0.0, 0.0
        
        # Use largest contour
        largest_contour = max(contours, key=len)
        
        # Fit ellipse to get major axis
        if len(largest_contour) >= 5:
            # Convert to (x, y) format
            contour_xy = np.column_stack([largest_contour[:, 1], largest_contour[:, 0]])
            
            # Fit ellipse
            ellipse = cv2.fitEllipse(contour_xy.astype(np.float32))
            (center, axes, angle) = ellipse
            
            # Major axis is the longer of the two axes
            major_axis = max(axes)
            length_px = float(major_axis)
        else:
            # Fallback to bounding box
            coords = np.column_stack(np.where(fish_mask))
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            length_px = float(max(x_max - x_min, y_max - y_min))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert to mm
    if pixels_per_mm > 0:
        length_mm = length_px / pixels_per_mm
    else:
        length_mm = 0.0
        logger.warning("pixels_per_mm is 0, cannot convert to mm")
    
    logger.debug(
        f"Bounding box length ({method}): {length_px:.2f} px = {length_mm:.2f} mm"
    )
    
    return length_px, length_mm


def extract_metrics(
    image_path: Path,
    fish_mask_path: Path,
    ruler_mask_path: Path,
    length_method: str = "medial_axis",
) -> dict:
    """Extract metrics from fish and ruler masks.
    
    Args:
        image_path: Path to original image
        fish_mask_path: Path to fish mask
        ruler_mask_path: Path to ruler mask
        length_method: Method for measuring fish length ("medial_axis" or "bounding_box")
    
    Returns:
        Dictionary containing extracted metrics with error flags:
        - "ruler_detected": bool - Whether ruler was detected in mask
        - "extraction_error": Optional[str] - Error message if extraction failed
    """
    # Load image with EXIF rotation handling
    # SAM3 respects EXIF rotation, so we must apply it here too
    pil_image = Image.open(image_path)
    pil_image = ImageOps.exif_transpose(pil_image)
    image = np.array(pil_image)
    
    # Load masks (masks are generated from SAM3 which handles EXIF, so they should match)
    fish_mask = np.array(Image.open(fish_mask_path)) > 0
    ruler_mask = np.array(Image.open(ruler_mask_path)) > 0
    
    # Check if ruler mask is empty (no ruler detected by SAM3)
    ruler_mask_pixels = np.sum(ruler_mask > 0)
    if ruler_mask_pixels == 0:
        logger.warning(
            f"No ruler detected in image {image_path.name}. "
            f"Ruler mask is empty (all zeros). "
            f"This image may not contain a ruler."
        )
        return {
            "pixels_per_mm": 0.0,
            "fish_length_px": 0.0,
            "fish_length_mm": 0.0,
            "ruler_bbox": None,
            "ruler_detected": False,
            "extraction_error": "no_ruler_detected",
        }
    
    # Extract ruler ROI
    try:
        ruler_roi, ruler_bbox = extract_ruler_roi(image, ruler_mask)
    except ValueError as e:
        # Dimension mismatch - likely EXIF rotation issue
        logger.error(
            f"Dimension mismatch when extracting ruler ROI from {image_path.name}: {e}"
        )
        return {
            "pixels_per_mm": 0.0,
            "fish_length_px": 0.0,
            "fish_length_mm": 0.0,
            "ruler_bbox": None,
            "ruler_detected": True,
            "extraction_error": "dimension_mismatch",
        }
    
    if ruler_roi.size == 0:
        logger.warning(
            f"Could not extract ruler ROI from {image_path.name}. "
            f"Ruler mask has {ruler_mask_pixels} pixels but ROI extraction failed."
        )
        return {
            "pixels_per_mm": 0.0,
            "fish_length_px": 0.0,
            "fish_length_mm": 0.0,
            "ruler_bbox": None,
            "ruler_detected": True,
            "extraction_error": "roi_extraction_failed",
        }
    
    # Detect ticks and calculate scale
    tick_positions, pixels_per_mm = detect_ruler_ticks(ruler_roi)
    
    if pixels_per_mm == 0.0:
        logger.warning(
            f"Could not detect ruler ticks in {image_path.name}. "
            f"Found {len(tick_positions)} tick positions."
        )
        return {
            "pixels_per_mm": 0.0,
            "fish_length_px": 0.0,
            "fish_length_mm": 0.0,
            "ruler_bbox": ruler_bbox,
            "ruler_detected": True,
            "extraction_error": "no_ticks_detected",
        }
    
    # Measure fish length
    if length_method == "medial_axis":
        length_px, length_mm = measure_fish_length_medial_axis(fish_mask, pixels_per_mm)
    elif length_method == "bounding_box":
        length_px, length_mm = measure_fish_length_bounding_box(fish_mask, pixels_per_mm)
    else:
        raise ValueError(f"Unknown length method: {length_method}")
    
    return {
        "pixels_per_mm": float(pixels_per_mm),
        "fish_length_px": float(length_px),
        "fish_length_mm": float(length_mm),
        "ruler_bbox": ruler_bbox,
        "length_method": length_method,
        "ruler_detected": True,
        "extraction_error": None,
    }


def analyze_scale_distribution(
    metrics_list: list[dict],
    outlier_threshold: float = 2.0,
) -> dict:
    """Analyze distribution of pixels_per_mm across images and identify outliers.
    
    Args:
        metrics_list: List of metric dictionaries from extract_metrics()
        outlier_threshold: Number of standard deviations for outlier detection
    
    Returns:
        Dictionary containing statistics and outlier information
    """
    pixels_per_mm_values = [m.get("pixels_per_mm", 0.0) for m in metrics_list]
    valid_values = np.array([v for v in pixels_per_mm_values if v > 0])
    
    if len(valid_values) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "outliers": [],
            "outlier_indices": [],
        }
    
    mean = float(np.mean(valid_values))
    median = float(np.median(valid_values))
    std = float(np.std(valid_values))
    
    # Identify outliers using z-score
    z_scores = np.abs((valid_values - mean) / std) if std > 0 else np.zeros_like(valid_values)
    outlier_mask = z_scores > outlier_threshold
    
    # Map back to original indices
    outlier_indices = []
    valid_idx = 0
    for i, value in enumerate(pixels_per_mm_values):
        if value > 0:
            if outlier_mask[valid_idx]:
                outlier_indices.append(i)
            valid_idx += 1
    
    outliers = [metrics_list[i] for i in outlier_indices]
    
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "min": float(np.min(valid_values)),
        "max": float(np.max(valid_values)),
        "count": len(valid_values),
        "outliers": outliers,
        "outlier_indices": outlier_indices,
    }
