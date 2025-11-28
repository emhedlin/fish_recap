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
import networkx as nx
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.ndimage import label
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform
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
    max_tick_spacing: int = 150,
) -> Tuple[np.ndarray, float]:
    """Detect tick marks on ruler and calculate pixels per millimeter.
    
    Uses deskewing to handle rotated rulers, Canny edge detection for cleaner edges,
    and robust spacing calculation with outlier filtering.
    
    Args:
        ruler_roi: Ruler region of interest image
        method: Detection method (deprecated, kept for backward compatibility)
        min_tick_spacing: Minimum expected spacing between ticks in pixels
        max_tick_spacing: Maximum expected spacing between ticks in pixels
    
    Returns:
        Tuple of (tick_positions, pixels_per_mm) where tick_positions is array of x-coordinates
    """
    # 1. Pre-processing: Deskew the ruler ROI
    # Convert to grayscale
    if ruler_roi.ndim == 3:
        gray = cv2.cvtColor(ruler_roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = ruler_roi.copy()
    
    # Otsu threshold to get binary ruler content
    _, binary_roi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours of the markings
    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the rotated rectangle of all points
        all_points = np.vstack(contours)
        rect = cv2.minAreaRect(all_points)
        angle = rect[-1]
        
        # Fix cv2.minAreaRect angle quirks
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Rotate image to make ticks perfectly vertical
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # 2. Enhanced Edge Detection
    # Use Canny instead of just Sobel for cleaner edges
    edges = cv2.Canny(gray, 50, 150)
    
    # 3. Projection
    # Sum along columns to get vertical signal
    projection = np.sum(edges, axis=0)
    
    # Smooth signal to merge double-edges (left and right of a single line)
    # A tick line has thickness. Canny gives 2 edges. Smoothing merges them into 1 peak.
    smoothed = ndimage.gaussian_filter1d(projection, sigma=2.0)
    
    # 4. Find Peaks
    # Dynamic height threshold
    height_threshold = np.percentile(smoothed, 50)
    peaks, _ = find_peaks(smoothed, height=height_threshold, distance=min_tick_spacing)
    
    if len(peaks) < 2:
        logger.warning(f"Found only {len(peaks)} tick positions, cannot calculate scale")
        return np.array([]), 0.0
    
    # 5. Robust Spacing Calculation
    diffs = np.diff(np.sort(peaks))
    
    # Filter statistical outliers using IQR
    q1 = np.percentile(diffs, 25)
    q3 = np.percentile(diffs, 75)
    iqr = q3 - q1
    valid_diffs = diffs[(diffs >= q1 - 1.5 * iqr) & (diffs <= q3 + 1.5 * iqr)]
    
    if len(valid_diffs) == 0:
        valid_diffs = diffs
    
    # Also filter by min/max spacing constraints
    valid_diffs = valid_diffs[
        (valid_diffs >= min_tick_spacing) & (valid_diffs <= max_tick_spacing)
    ]
    
    if len(valid_diffs) == 0:
        logger.warning("No valid tick distances found after filtering")
        return peaks, 0.0
    
    # Get median
    median_px = np.median(valid_diffs)
    
    # Assume ticks are millimeter marks
    # pixels_per_mm = median_distance_px (since distance between mm ticks is 1mm)
    pixels_per_mm = float(median_px)
    
    logger.debug(
        f"Detected {len(peaks)} ticks, "
        f"median spacing: {median_px:.2f} px, "
        f"pixels_per_mm: {pixels_per_mm:.2f}"
    )
    
    return peaks, pixels_per_mm


def measure_fish_length_medial_axis(
    fish_mask: np.ndarray,
    pixels_per_mm: float,
) -> Tuple[float, float]:
    """Measure fish length using graph-based analysis of the medial axis.
    
    Calculates Euclidean distance along the longest path, correcting for diagonal pixels.
    Uses networkx to build a graph from skeleton pixels and find the longest shortest path
    between endpoints, effectively removing spurs and noise.
    
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
        # 1. Skeletonize
        skeleton = morphology.skeletonize(fish_mask.astype(np.uint8))
        
        # 2. Get coordinates of skeleton pixels
        y_coords, x_coords = np.where(skeleton)
        if len(y_coords) < 2:
            logger.warning("Skeletonization produced fewer than 2 pixels")
            return 0.0, 0.0
        
        # 3. Build a graph
        # Points are nodes. Edges exist between 8-connected neighbors.
        # Weights are Euclidean distances (1.0 or 1.414)
        points = np.column_stack((y_coords, x_coords))
        G = nx.Graph()
        
        # Add nodes
        for i in range(len(points)):
            G.add_node(i, pos=points[i])
        
        # Fast neighbor finding using image structure
        # Create a map from (y,x) to node_index
        coord_to_idx = {tuple(p): i for i, p in enumerate(points)}
        
        for r, c in points:
            curr_idx = coord_to_idx[(r, c)]
            # Check 8 neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = (r + dr, c + dc)
                    if neighbor in coord_to_idx:
                        neigh_idx = coord_to_idx[neighbor]
                        # Euclidean distance
                        dist = np.sqrt(dr**2 + dc**2)
                        G.add_edge(curr_idx, neigh_idx, weight=dist)
        
        # 4. Find the endpoints (nodes with degree 1)
        endpoints = [n for n, d in G.degree() if d == 1]
        
        # If skeleton is a loop or has no endpoints, pick arbitrary nodes
        if len(endpoints) < 2:
            endpoints = list(G.nodes())
        
        # 5. Find the longest shortest path (the main spine)
        # We calculate shortest path between all pairs of endpoints
        max_length = 0.0
        
        # If too many endpoints (noisy skeleton), just take the two farthest points roughly
        # to avoid computing paths between adjacent noisy spurs
        if len(endpoints) > 10:
            # Optimization: only check paths between the two geometrically farthest pixels
            pts = np.array([G.nodes[n]['pos'] for n in endpoints])
            D = squareform(pdist(pts))
            i, j = np.unravel_index(np.argmax(D), D.shape)
            endpoints = [endpoints[i], endpoints[j]]
        
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                try:
                    length = nx.shortest_path_length(
                        G, source=endpoints[i], target=endpoints[j], weight='weight'
                    )
                    if length > max_length:
                        max_length = length
                except nx.NetworkXNoPath:
                    continue
        
        length_px = float(max_length)
    
    except Exception as e:
        logger.warning(f"Graph skeleton measurement failed: {e}. Falling back to bounding box method.")
        return measure_fish_length_bounding_box(fish_mask, pixels_per_mm)
    
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
