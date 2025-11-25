"""Image standardization module for creating aligned 'passport photos' of fish.

This module handles:
- Rotation alignment using image moments or PCA
- Cropping to fish bounding box with margin
- Background removal (setting wood background to solid color)
- Enhanced orientation detection using image-based features (eyes, fins)
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from scipy import ndimage
from scipy.signal import find_peaks
from skimage import morphology

logger = logging.getLogger(__name__)


def detect_eyes(
    image: np.ndarray,
    fish_mask: np.ndarray,
    orientation_angle: float,
) -> Tuple[List[Tuple[float, float]], float]:
    """Detect fish eyes using image-based features.
    
    Eyes are typically dark circular regions in the anterior portion of the fish.
    
    Args:
        image: Grayscale or color image (H, W) or (H, W, 3)
        fish_mask: Binary mask of fish
        orientation_angle: Current orientation angle in degrees
    
    Returns:
        Tuple of (eye_locations, confidence) where:
        - eye_locations: List of (x, y) coordinates of detected eyes
        - confidence: Confidence score (0-1) for eye detection
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Clean mask
    fish_mask = clean_fish_mask(fish_mask)
    if fish_mask.dtype != bool:
        fish_mask = fish_mask > 0.5
    
    # Rotate image and mask to horizontal for analysis
    center = tuple(np.array(fish_mask.shape[::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -orientation_angle, 1.0)
    rotated_mask = cv2.warpAffine(
        fish_mask.astype(np.uint8),
        rotation_matrix,
        fish_mask.shape[::-1],
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    
    rotated_gray = cv2.warpAffine(
        gray,
        rotation_matrix,
        gray.shape[::-1],
        flags=cv2.INTER_LINEAR,
    )
    
    # Mask the rotated image
    masked_gray = rotated_gray.copy()
    masked_gray[~rotated_mask] = 255  # Set background to white
    
    # Find bounding box of fish
    coords = np.column_stack(np.where(rotated_mask))
    if len(coords) == 0:
        return [], 0.0
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    fish_length = x_max - x_min
    
    # Focus on anterior 30% of fish (head region)
    head_region_x_max = x_min + int(fish_length * 0.3)
    head_region = masked_gray[y_min:y_max, x_min:head_region_x_max]
    head_mask = rotated_mask[y_min:y_max, x_min:head_region_x_max]
    
    if head_region.size == 0:
        return [], 0.0
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(head_region, (5, 5), 0)
    
    # Threshold to find dark regions (eyes are darker than surrounding scales)
    # Use adaptive threshold or percentile-based threshold
    masked_head = blurred[head_mask]
    if len(masked_head) == 0:
        return [], 0.0
    
    # Use lower percentile for threshold (eyes are among darkest pixels)
    threshold_value = np.percentile(masked_head, 15)
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Mask out non-fish regions
    binary[~head_mask] = 0
    
    # Find contours (potential eyes)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    eye_locations = []
    eye_confidences = []
    
    for contour in contours:
        # Filter by area (eyes should be reasonably sized)
        area = cv2.contourArea(contour)
        if area < 5 or area > 500:  # Too small or too large
            continue
        
        # Check circularity (eyes are roughly circular)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity < 0.3:  # Not circular enough
            continue
        
        # Get center of contour
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Convert back to original image coordinates
        # First adjust for head region offset
        orig_x = x_min + cx
        orig_y = y_min + cy
        
        # Rotate back to original orientation
        # (Inverse rotation)
        angle_rad = np.radians(orientation_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Translate to origin, rotate, translate back
        rel_x = orig_x - center[0]
        rel_y = orig_y - center[1]
        
        rot_x = rel_x * cos_a - rel_y * sin_a + center[0]
        rot_y = rel_x * sin_a + rel_y * cos_a + center[1]
        
        eye_locations.append((rot_x, rot_y))
        
        # Confidence based on circularity and darkness
        darkness = 1.0 - (blurred[cy, cx] / 255.0)
        eye_confidences.append(circularity * darkness)
    
    # Calculate overall confidence
    if len(eye_confidences) > 0:
        confidence = min(1.0, np.mean(eye_confidences))
    else:
        confidence = 0.0
    
    logger.debug(f"Detected {len(eye_locations)} eyes with confidence {confidence:.2f}")
    
    return eye_locations, confidence


def detect_dorsal_fin(
    image: np.ndarray,
    fish_mask: np.ndarray,
    orientation_angle: float,
) -> Tuple[Optional[Tuple[float, float]], float]:
    """Detect dorsal fin location using image-based features.
    
    Dorsal fin creates a distinct vertical protrusion on the top edge of the fish.
    
    Args:
        image: Grayscale or color image (H, W) or (H, W, 3)
        fish_mask: Binary mask of fish
        orientation_angle: Current orientation angle in degrees
    
    Returns:
        Tuple of (fin_location, confidence) where:
        - fin_location: (x, y) coordinate of fin tip, or None if not detected
        - confidence: Confidence score (0-1) for fin detection
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Clean mask
    fish_mask = clean_fish_mask(fish_mask)
    if fish_mask.dtype != bool:
        fish_mask = fish_mask > 0.5
    
    # Rotate image and mask to horizontal for analysis
    center = tuple(np.array(fish_mask.shape[::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -orientation_angle, 1.0)
    rotated_mask = cv2.warpAffine(
        fish_mask.astype(np.uint8),
        rotation_matrix,
        fish_mask.shape[::-1],
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    
    rotated_gray = cv2.warpAffine(
        gray,
        rotation_matrix,
        gray.shape[::-1],
        flags=cv2.INTER_LINEAR,
    )
    
    # Find bounding box
    coords = np.column_stack(np.where(rotated_mask))
    if len(coords) == 0:
        return None, 0.0
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Extract top edge profile
    top_edge_profile = []
    top_edge_x_coords = []
    
    for col_idx in range(x_min, x_max + 1):
        col = rotated_mask[:, col_idx]
        if np.any(col):
            top_idx = np.argmax(col)
            top_edge_profile.append(top_idx)
            top_edge_x_coords.append(col_idx)
        else:
            top_edge_profile.append(np.nan)
            top_edge_x_coords.append(col_idx)
    
    top_edge_profile = np.array(top_edge_profile)
    valid = ~np.isnan(top_edge_profile)
    
    if np.sum(valid) < 3:
        return None, 0.0
    
    # Find local minima in top edge (protrusions upward = fins)
    # Actually, we want to find points where the edge protrudes upward (lower y values)
    # So we look for local minima in the y-coordinate
    smoothed = ndimage.gaussian_filter1d(top_edge_profile[valid].astype(float), sigma=2.0)
    
    # Find local minima (protrusions)
    # Invert to find minima as peaks
    inverted = -smoothed
    peaks, properties = find_peaks(inverted, prominence=3, distance=10)
    
    if len(peaks) == 0:
        return None, 0.0
    
    # Get the most prominent peak (largest protrusion)
    prominences = properties['prominences']
    best_peak_idx = peaks[np.argmax(prominences)]
    
    # Get x coordinate
    valid_x_coords = np.array(top_edge_x_coords)[valid]
    fin_x = valid_x_coords[best_peak_idx]
    fin_y = top_edge_profile[valid][best_peak_idx]
    
    # Calculate confidence based on prominence
    max_prominence = np.max(prominences)
    confidence = min(1.0, max_prominence / 20.0)  # Normalize
    
    # Convert back to original image coordinates
    angle_rad = np.radians(orientation_angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    rel_x = fin_x - center[0]
    rel_y = fin_y - center[1]
    
    rot_x = rel_x * cos_a - rel_y * sin_a + center[0]
    rot_y = rel_x * sin_a + rel_y * cos_a + center[1]
    
    logger.debug(f"Detected dorsal fin at ({rot_x:.1f}, {rot_y:.1f}) with confidence {confidence:.2f}")
    
    return (rot_x, rot_y), confidence


def clean_fish_mask(fish_mask: np.ndarray) -> np.ndarray:
    """Clean fish mask by keeping only the largest connected component.
    
    Removes noise, disconnected ruler parts, etc.
    
    Args:
        fish_mask: Binary mask of fish
        
    Returns:
        Cleaned binary mask
    """
    # Ensure mask is uint8
    if fish_mask.dtype != bool:
        mask_uint8 = (fish_mask > 0.5).astype(np.uint8)
    else:
        mask_uint8 = fish_mask.astype(np.uint8)
        
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    if num_labels <= 1:
        return fish_mask # No background or only background
        
    # Find largest component (ignoring background label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create new mask
    cleaned_mask = (labels == largest_label)
    
    # Log if we removed significant area
    original_area = np.sum(mask_uint8)
    new_area = np.sum(cleaned_mask)
    if original_area > 0 and (original_area - new_area) / original_area > 0.05:
        logger.info(
            f"Cleaned mask: kept largest component ({new_area} px), "
            f"removed {original_area - new_area} px ({100*(original_area-new_area)/original_area:.1f}%)"
        )
        
    return cleaned_mask


def get_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Calculate centroid of a binary mask using image moments.
    
    Args:
        mask: Binary mask (H, W)
    
    Returns:
        Tuple of (cx, cy) centroid coordinates
    """
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = 0.0, 0.0
    return (cx, cy)


def get_bbox_center(mask: np.ndarray) -> Tuple[float, float]:
    """Calculate bounding box center of a binary mask.
    
    Args:
        mask: Binary mask (H, W)
    
    Returns:
        Tuple of (cx, cy) bounding box center coordinates
    """
    coords = np.column_stack(np.where(mask))
    if len(coords) == 0:
        return (0.0, 0.0)
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return (cx, cy)


def rotate_masks_180(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rotate all masks in dictionary 180 degrees.
    
    Args:
        masks: Dictionary mapping names to mask arrays
    
    Returns:
        Dictionary with rotated masks
    """
    rotated = {}
    for name, mask in masks.items():
        rotated[name] = np.rot90(mask, k=2)
    return rotated


def flip_masks_vertically(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Flip all masks in dictionary vertically.
    
    Args:
        masks: Dictionary mapping names to mask arrays
    
    Returns:
        Dictionary with flipped masks
    """
    flipped = {}
    for name, mask in masks.items():
        flipped[name] = np.flipud(mask)
    return flipped


def check_fish_curvature(
    body_mask: np.ndarray,
    threshold: float = 50.0
) -> Tuple[bool, float]:
    """Check if fish is curved beyond acceptable threshold.
    
    Uses skeletonization to find the medial axis, then fits a straight line
    and calculates the mean squared error. High MSE indicates curvature.
    
    Args:
        body_mask: Binary mask of fish body (H, W)
        threshold: Maximum acceptable MSE (default: 50.0)
    
    Returns:
        Tuple of (is_straight, mse) where:
        - is_straight: True if fish is acceptably straight (MSE < threshold)
        - mse: Mean squared error of skeleton vs fitted line
    """
    # Ensure mask is binary
    if body_mask.dtype != bool:
        body_mask = body_mask > 0.5
    
    if np.sum(body_mask) == 0:
        logger.warning("Empty body mask, cannot check curvature")
        return True, 0.0
    
    try:
        # Skeletonize the body mask to get medial axis
        skeleton = morphology.skeletonize(body_mask.astype(np.uint8))
        
        # Get skeleton pixel coordinates
        skeleton_coords = np.column_stack(np.where(skeleton))
        
        if len(skeleton_coords) < 3:
            logger.debug("Skeleton too short for curvature check")
            return True, 0.0
        
        # Convert to (x, y) format (column, row)
        coords_xy = np.column_stack([skeleton_coords[:, 1], skeleton_coords[:, 0]])
        
        # Fit a straight line using least squares
        # Line equation: y = mx + b
        # Using normal equation: (X^T X)^(-1) X^T y
        X = np.column_stack([coords_xy[:, 0], np.ones(len(coords_xy))])
        y = coords_xy[:, 1]
        
        # Solve for m and b
        try:
            params = np.linalg.lstsq(X, y, rcond=None)[0]
            m, b = params[0], params[1]
        except np.linalg.LinAlgError:
            logger.warning("Failed to fit line for curvature check")
            return True, 0.0
        
        # Calculate predicted y values
        y_pred = m * coords_xy[:, 0] + b
        
        # Calculate MSE (mean squared error)
        mse = np.mean((coords_xy[:, 1] - y_pred) ** 2)
        
        is_straight = mse < threshold
        
        logger.debug(f"Curvature check: MSE={mse:.2f}, threshold={threshold:.2f}, is_straight={is_straight}")
        
        return is_straight, float(mse)
        
    except Exception as e:
        logger.warning(f"Curvature check failed: {e}")
        return True, 0.0


def detect_head_direction_mask_based(
    body_mask: np.ndarray,
    eye_mask: Optional[np.ndarray],
    orientation_angle: float,
) -> Tuple[bool, float]:
    """Detect head direction using SAM 3 eye mask (primary) or center of mass (fallback).
    
    Args:
        body_mask: Binary mask of fish body (already rotated to horizontal)
        eye_mask: Optional binary mask of fish eye (may be None or empty)
        orientation_angle: Current orientation angle (for logging)
    
    Returns:
        Tuple of (head_is_right, confidence) where:
        - head_is_right: True if head is on the right side
        - confidence: Confidence score (0-1)
    """
    # Ensure masks are binary
    if body_mask.dtype != bool:
        body_mask = body_mask > 0.5
    
    body_center = get_centroid(body_mask)
    
    # Primary method: Use SAM 3 eye mask
    if eye_mask is not None:
        if eye_mask.dtype != bool:
            eye_mask = eye_mask > 0.5
        
        eye_pixels = np.sum(eye_mask)
        
        if eye_pixels > 10:  # Minimum threshold for valid eye detection
            eye_center = get_centroid(eye_mask)
            
            # If eye is to the right of body center, head is on right
            head_is_right = eye_center[0] > body_center[0]
            
            # Confidence based on how far eye is from body center
            distance = abs(eye_center[0] - body_center[0])
            body_width = np.sum(np.any(body_mask, axis=0))
            if body_width > 0:
                normalized_distance = distance / body_width
                confidence = min(1.0, normalized_distance * 2.0)  # Scale to 0-1
            else:
                confidence = 0.7  # Default confidence if we can't normalize
            
            logger.debug(
                f"Head detection (eye mask): eye_center={eye_center}, "
                f"body_center={body_center}, head_right={head_is_right}, "
                f"confidence={confidence:.2f}"
            )
            
            return head_is_right, confidence
    
    # Fallback method: Center of Mass vs Geometric Center
    # Heads are denser/wider, so mass center is usually closer to head
    geom_center = get_bbox_center(body_mask)
    mass_center = body_center  # Already calculated above
    
    head_is_right = mass_center[0] > geom_center[0]
    
    # Lower confidence for fallback method
    distance = abs(mass_center[0] - geom_center[0])
    body_width = np.sum(np.any(body_mask, axis=0))
    if body_width > 0:
        normalized_distance = distance / body_width
        confidence = min(0.7, normalized_distance * 1.5)  # Max 0.7 for fallback
    else:
        confidence = 0.5
    
    logger.debug(
        f"Head detection (fallback): mass_center={mass_center}, "
        f"geom_center={geom_center}, head_right={head_is_right}, "
        f"confidence={confidence:.2f}"
    )
    
    return head_is_right, confidence


def detect_dorsal_orientation_mask_based(
    body_mask: np.ndarray,
    eye_mask: Optional[np.ndarray],
    fin_mask: Optional[np.ndarray],
    image: Optional[np.ndarray],
    orientation_angle: float,
) -> Tuple[bool, float]:
    """Detect dorsal orientation using SAM 3 eye mask (primary), fin mask (secondary), or intensity gradient (fallback).
    
    Eyes are almost always closer to the dorsal (back) side of the fish, making eye position
    a reliable indicator of dorsal orientation.
    
    Args:
        body_mask: Binary mask of fish body (already rotated to horizontal)
        eye_mask: Optional binary mask of fish eye (may be None or empty)
        fin_mask: Optional binary mask of dorsal fin (may be None or empty)
        image: Optional grayscale image for intensity analysis fallback
        orientation_angle: Current orientation angle (for logging)
    
    Returns:
        Tuple of (dorsal_is_up, confidence) where:
        - dorsal_is_up: True if dorsal side is on top
        - confidence: Confidence score (0-1)
    """
    # Ensure mask is binary
    if body_mask.dtype != bool:
        body_mask = body_mask > 0.5
    
    body_center = get_centroid(body_mask)
    
    # Primary method: Use SAM 3 eye mask (most reliable)
    if eye_mask is not None:
        if eye_mask.dtype != bool:
            eye_mask = eye_mask > 0.5
        
        eye_pixels = np.sum(eye_mask)
        
        if eye_pixels > 10:  # Minimum threshold for valid eye detection
            eye_center = get_centroid(eye_mask)
            eye_x = int(eye_center[0])
            
            # Instead of global y_min/y_max, we look at the vertical slice
            # of the body mask specifically at the eye's X location.
            
            # Extract a vertical column from the body mask at the eye's x-coordinate
            # We use a small window (e.g., 5px) to be robust against noise/mask gaps
            x_start = max(0, eye_x - 2)
            x_end = min(body_mask.shape[1], eye_x + 3)
            
            # Get indices where the body mask is True in this vertical strip
            # dimensions: (row_indices, col_indices)
            body_y_indices, _ = np.where(body_mask[:, x_start:x_end])
            
            if len(body_y_indices) > 0:
                # Find the local top (skin above eye) and local bottom (skin below eye)
                local_y_min = np.min(body_y_indices)  # Forehead
                local_y_max = np.max(body_y_indices)  # Jaw
                
                # Calculate distances
                distance_to_top = eye_center[1] - local_y_min
                distance_to_bottom = local_y_max - eye_center[1]
                
                # If eye is closer to top edge, dorsal is up
                dorsal_is_up = distance_to_top < distance_to_bottom
                
                # Confidence calculation
                local_head_height = local_y_max - local_y_min
                if local_head_height > 0:
                    distance_diff = abs(distance_to_top - distance_to_bottom)
                    normalized_diff = distance_diff / local_head_height
                    # Eyes are usually VERY high up, so this signal is strong.
                    confidence = min(1.0, normalized_diff * 3.0)
                else:
                    confidence = 0.8
                
                logger.debug(
                    f"Dorsal detection (eye mask): eye_y={eye_center[1]:.1f}, "
                    f"local_top={local_y_min}, local_bottom={local_y_max}, "
                    f"dist_top={distance_to_top:.1f}, dist_bot={distance_to_bottom:.1f}, "
                    f"dorsal_up={dorsal_is_up}, confidence={confidence:.2f}"
                )
                
                return dorsal_is_up, confidence
            else:
                logger.warning(f"Eye detected at x={eye_x} but body mask is empty at that column.")
    
    # Secondary method: Use SAM 3 fin mask
    if fin_mask is not None:
        if fin_mask.dtype != bool:
            fin_mask = fin_mask > 0.5
        
        fin_pixels = np.sum(fin_mask)
        
        if fin_pixels > 10:  # Minimum threshold for valid fin detection
            fin_center = get_centroid(fin_mask)
            
            # In image coordinates, Y increases downwards
            # If fin Y < body Y, fin is above (dorsal is up)
            dorsal_is_up = fin_center[1] < body_center[1]
            
            # Confidence based on how far fin is from body center
            distance = abs(fin_center[1] - body_center[1])
            body_height = np.sum(np.any(body_mask, axis=1))
            if body_height > 0:
                normalized_distance = distance / body_height
                confidence = min(1.0, normalized_distance * 2.0)  # Scale to 0-1
            else:
                confidence = 0.7  # Default confidence
            
            logger.debug(
                f"Dorsal detection (fin mask): fin_center={fin_center}, "
                f"body_center={body_center}, dorsal_up={dorsal_is_up}, "
                f"confidence={confidence:.2f}"
            )
            
            return dorsal_is_up, confidence
    
    # Fallback method: Intensity Gradient (counter-shading)
    # Assumption: Backs are dark, bellies are light
    if image is not None:
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Get body pixel coordinates
        coords = np.column_stack(np.where(body_mask))
        if len(coords) == 0:
            return True, 0.5  # Default with low confidence
        
        y_coords = coords[:, 0]
        y_min, y_max = y_coords.min(), y_coords.max()
        body_height = y_max - y_min
        
        if body_height > 0:
            # Extract top 25% and bottom 25% of body pixels
            top_threshold = y_min + int(body_height * 0.25)
            bottom_threshold = y_max - int(body_height * 0.25)
            
            top_pixels = gray[coords[(y_coords <= top_threshold), 0], coords[(y_coords <= top_threshold), 1]]
            bottom_pixels = gray[coords[(y_coords >= bottom_threshold), 0], coords[(y_coords >= bottom_threshold), 1]]
            
            if len(top_pixels) > 0 and len(bottom_pixels) > 0:
                top_intensity = np.mean(top_pixels)
                bottom_intensity = np.mean(bottom_pixels)
                
                # Backs are darker (lower value), so if top is darker, dorsal is up
                dorsal_is_up = top_intensity < bottom_intensity
                
                # Confidence based on intensity difference
                intensity_diff = abs(top_intensity - bottom_intensity)
                confidence = min(0.7, intensity_diff / 50.0)  # Max 0.7 for fallback
                
                logger.debug(
                    f"Dorsal detection (fallback): top_intensity={top_intensity:.1f}, "
                    f"bottom_intensity={bottom_intensity:.1f}, dorsal_up={dorsal_is_up}, "
                    f"confidence={confidence:.2f}"
                )
                
                return dorsal_is_up, confidence
    
    # Default fallback: assume dorsal is up with low confidence
    logger.warning("Dorsal detection: no eye mask, fin mask, or image available, defaulting to dorsal_up=True")
    return True, 0.5


def calculate_fish_orientation(
    fish_mask: np.ndarray,
    method: str = "moments",
) -> float:
    """Calculate the orientation angle of the fish's main body axis.
    
    Args:
        fish_mask: Binary mask of fish (H, W) with values 0 or 1
        method: Method to use ("moments" or "pca")
    
    Returns:
        Angle in degrees (0-180) representing the fish's orientation
    """
    # Clean mask first
    fish_mask = clean_fish_mask(fish_mask)
    
    # Ensure mask is binary
    if fish_mask.dtype != bool:
        fish_mask = fish_mask > 0.5
    
    # Get coordinates of fish pixels
    coords = np.column_stack(np.where(fish_mask))
    
    if len(coords) == 0:
        logger.warning("Empty fish mask, cannot calculate orientation")
        return 0.0
    
    if method == "moments":
        # Calculate image moments
        moments = cv2.moments(fish_mask.astype(np.uint8))
        
        if moments["mu02"] == 0:
            logger.warning("Cannot calculate orientation from moments")
            return 0.0
        
        # Calculate orientation angle from central moments
        # Angle is in radians, convert to degrees
        angle_rad = 0.5 * np.arctan2(
            2 * moments["mu11"],
            moments["mu20"] - moments["mu02"]
        )
        angle_deg = np.degrees(angle_rad)
        
    elif method == "pca":
        # Use PCA to find principal axis
        # Note: coords are in (row, col) format, need to convert to (x, y)
        coords_xy = np.column_stack([coords[:, 1], coords[:, 0]])  # (x, y) format
        
        # Center the coordinates
        mean = np.mean(coords_xy, axis=0)
        centered = coords_xy - mean
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca.fit(centered)
        
        # Get principal component (first component)
        principal_component = pca.components_[0]
        
        # Calculate angle from principal component
        angle_rad = np.arctan2(principal_component[1], principal_component[0])
        angle_deg = np.degrees(angle_rad)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize angle to 0-180 range
    angle_deg = angle_deg % 180
    if angle_deg < 0:
        angle_deg += 180
    
    logger.debug(f"Calculated fish orientation: {angle_deg:.2f} degrees (method: {method})")
    
    return angle_deg


def detect_head_direction(
    fish_mask: np.ndarray,
    orientation_angle: float,
    image: Optional[np.ndarray] = None,
    use_image_features: bool = True,
) -> Tuple[float, bool, float]:
    """Detect which end of the fish is the head.
    
    Uses multiple heuristics:
    1. Eye detection (image-based, most reliable)
    2. V-shape detection (tail has V-shape with acute angle, head is rounded)
    3. Center of Mass vs Geometric Center (Head is heavier/bulkier)
    4. Width analysis (Head/Body is wider than Tail Peduncle)
    5. Peduncle detection (Tail fin flares out after narrowing, head tapers)
    6. Contour curvature analysis (fallback when V-shape unclear)
    
    Args:
        fish_mask: Binary mask of fish
        orientation_angle: Current orientation angle in degrees
        image: Optional grayscale or color image for eye detection
        use_image_features: Whether to use image-based features (eyes)
    
    Returns:
        Tuple of (adjusted_angle, head_on_left, confidence) where:
        - adjusted_angle: Angle adjusted so head points left (0-180)
        - head_on_left: Boolean indicating if head is on the left side
        - confidence: Confidence score (0-1) for the detection
    """
    # Clean mask first
    fish_mask = clean_fish_mask(fish_mask)
    
    # Ensure mask is binary
    if fish_mask.dtype != bool:
        fish_mask = fish_mask > 0.5
    
    # Rotate mask to horizontal orientation for analysis
    center = tuple(np.array(fish_mask.shape[::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -orientation_angle, 1.0)
    rotated_mask = cv2.warpAffine(
        fish_mask.astype(np.uint8),
        rotation_matrix,
        fish_mask.shape[::-1],
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    
    # --- Heuristic 0: Eye Detection (image-based, most reliable) ---
    eye_vote_left = None
    eye_confidence = 0.0
    if use_image_features and image is not None:
        eye_locations, eye_conf = detect_eyes(image, fish_mask, orientation_angle)
        
        if len(eye_locations) > 0 and eye_conf > 0.3:
            # Eyes should be in anterior portion (left side if head is on left)
            # Get bounding box of rotated mask
            coords = np.column_stack(np.where(rotated_mask))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                fish_length = x_max - x_min
                
                # Check if eyes are in anterior 30% (left side)
                eye_x_coords = [eye[0] for eye in eye_locations]
                # Rotate eye coordinates to match rotated mask
                for i, (ex, ey) in enumerate(eye_locations):
                    angle_rad = np.radians(orientation_angle)
                    cos_a = np.cos(angle_rad)
                    sin_a = np.sin(angle_rad)
                    rel_x = ex - center[0]
                    rel_y = ey - center[1]
                    rot_x = rel_x * cos_a - rel_y * sin_a + center[0]
                    rot_y = rel_x * sin_a + rel_y * cos_a + center[1]
                    eye_x_coords[i] = rot_x
                
                # Check if majority of eyes are in left half
                left_eyes = sum(1 for ex in eye_x_coords if ex < (x_min + fish_length * 0.3))
                eye_vote_left = left_eyes > len(eye_locations) / 2
                eye_confidence = eye_conf
                logger.debug(f"Eye detection: {len(eye_locations)} eyes found, {left_eyes} in left region, confidence {eye_conf:.2f}")
    
    # --- Heuristic 1: Center of Mass (Centroid) vs Geometric Center ---
    # Calculate moments of the ROTATED mask
    moments = cv2.moments(rotated_mask.astype(np.uint8))
    if moments["m00"] != 0:
        centroid_x = moments["m10"] / moments["m00"]
    else:
        centroid_x = rotated_mask.shape[1] / 2
        
    # Find bounding box of the fish in the rotated mask
    coords = np.column_stack(np.where(rotated_mask))
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        bbox_center_x = (x_min + x_max) / 2
    else:
        bbox_center_x = rotated_mask.shape[1] / 2
        
    # If Centroid is to the left of Bbox Center, mass is on left -> Head on Left
    mass_on_left = centroid_x < bbox_center_x
    
    # --- Heuristic 2 & 3: Width Profile Analysis ---
    horizontal_projection = np.sum(rotated_mask, axis=0)
    
    if len(horizontal_projection) == 0:
        return orientation_angle, True
    
    # Trim empty space
    non_zero_indices = np.where(horizontal_projection > 0)[0]
    if len(non_zero_indices) > 0:
        start_idx = non_zero_indices[0]
        end_idx = non_zero_indices[-1]
        fish_profile = horizontal_projection[start_idx:end_idx+1]
    else:
        fish_profile = horizontal_projection

    mid = len(fish_profile) // 2
    left_half = fish_profile[:mid]
    right_half = fish_profile[mid:]
    
    left_avg = np.mean(left_half) if len(left_half) > 0 else 0
    right_avg = np.mean(right_half) if len(right_half) > 0 else 0
    
    # Heuristic 2: Average Width (Head is usually wider than tail)
    width_on_left = left_avg > right_avg
    
    # Heuristic 3: Tail Flare / Peduncle Detection
    tip_len = max(1, int(len(fish_profile) * 0.10))
    neck_len = max(1, int(len(fish_profile) * 0.10))
    
    # Left Side Analysis
    left_tip = fish_profile[:tip_len]
    left_neck = fish_profile[tip_len:tip_len+neck_len]
    left_tip_avg = np.mean(left_tip) if len(left_tip) > 0 else 0
    left_neck_avg = np.mean(left_neck) if len(left_neck) > 0 else 0
    
    # Right Side Analysis
    right_tip = fish_profile[-tip_len:]
    right_neck = fish_profile[-(tip_len+neck_len):-tip_len]
    right_tip_avg = np.mean(right_tip) if len(right_tip) > 0 else 0
    right_neck_avg = np.mean(right_neck) if len(right_neck) > 0 else 0
    
    # Check for flaring (Tip > Neck) - increased threshold for robustness
    flare_threshold = 1.2  # Increased from 1.1
    left_flares = left_tip_avg > left_neck_avg * flare_threshold
    right_flares = right_tip_avg > right_neck_avg * flare_threshold
    
    # --- Heuristic 4: V-Shape Detection (tail has V-shape, head is rounded) ---
    # Find contour of rotated mask
    contours, _ = cv2.findContours(
        rotated_mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    contour_vote_left = None
    vshape_confidence = 0.0
    
    if len(contours) > 0:
        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        contour_len = len(largest_contour)
        
        if contour_len > 10:
            # Analyze the tip regions (last 15% of contour points)
            # Tail typically has a V-shape (acute angle), head is rounded (obtuse angle)
            tip_region_size = max(5, contour_len // 7)  # ~15% of contour
            
            # Left end: first tip_region_size points
            left_tip_points = largest_contour[:tip_region_size]
            # Right end: last tip_region_size points
            right_tip_points = largest_contour[-tip_region_size:]
            
            def detect_vshape(points):
                """Detect V-shape at tip by measuring angle between edges.
                
                Returns:
                    Tuple of (is_vshape, angle_degrees, confidence)
                    - is_vshape: True if V-shaped (acute angle < 90°)
                    - angle_degrees: Angle at tip in degrees
                    - confidence: Confidence in detection (0-1)
                """
                if len(points) < 5:
                    return False, 180.0, 0.0
                
                # Get tip point (furthest point from center)
                points_array = np.array([p[0] for p in points])
                center_point = np.mean(points_array, axis=0)
                
                # Find tip (point furthest from center)
                distances = np.linalg.norm(points_array - center_point, axis=1)
                tip_idx = np.argmax(distances)
                tip_point = points_array[tip_idx]
                
                # Get points before and after tip
                if tip_idx < len(points) - 1 and tip_idx > 0:
                    # Use points on either side of tip
                    before_idx = max(0, tip_idx - 2)
                    after_idx = min(len(points) - 1, tip_idx + 2)
                    
                    before_point = points_array[before_idx]
                    after_point = points_array[after_idx]
                    
                    # Calculate vectors from tip
                    vec1 = before_point - tip_point
                    vec2 = after_point - tip_point
                    
                    # Calculate angle between vectors
                    if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle_rad = np.arccos(cos_angle)
                        angle_deg = np.degrees(angle_rad)
                        
                        # V-shape: acute angle (< 90°), rounded: obtuse angle (> 90°)
                        is_vshape = angle_deg < 90.0
                        
                        # Confidence based on how acute/obtuse the angle is
                        # Very acute (< 60°) = high confidence V-shape
                        # Very obtuse (> 120°) = high confidence rounded
                        if angle_deg < 60:
                            confidence = 1.0 - (angle_deg / 60.0)  # 1.0 at 0°, 0.0 at 60°
                        elif angle_deg > 120:
                            confidence = (angle_deg - 120) / 60.0  # 0.0 at 120°, 1.0 at 180°
                        else:
                            confidence = 0.5  # Uncertain in middle range
                        
                        return is_vshape, angle_deg, confidence
                
                return False, 180.0, 0.0
            
            # Detect V-shape at both ends
            left_is_vshape, left_angle, left_conf = detect_vshape(left_tip_points)
            right_is_vshape, right_angle, right_conf = detect_vshape(right_tip_points)
            
            # Tail has V-shape, head is rounded
            # If left is V-shaped and right is rounded -> left is tail, right is head
            # If right is V-shaped and left is rounded -> right is tail, left is head
            
            if left_is_vshape and not right_is_vshape:
                # Left is tail (V-shaped), right is head (rounded)
                contour_vote_left = False  # Head is on right
                vshape_confidence = (left_conf + (1.0 - right_conf)) / 2.0
            elif right_is_vshape and not left_is_vshape:
                # Right is tail (V-shaped), left is head (rounded)
                contour_vote_left = True  # Head is on left
                vshape_confidence = (right_conf + (1.0 - left_conf)) / 2.0
            else:
                # Both similar or unclear - use curvature as fallback
                def calculate_curvature(points):
                    """Calculate average curvature (smoothness)."""
                    if len(points) < 3:
                        return 0.0
                    angles = []
                    for i in range(1, len(points) - 1):
                        p1 = points[i-1][0]
                        p2 = points[i][0]
                        p3 = points[i+1][0]
                        v1 = p2 - p1
                        v2 = p3 - p2
                        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle = np.arccos(cos_angle)
                            angles.append(angle)
                    return np.mean(angles) if len(angles) > 0 else 0.0
                
                left_curvature = calculate_curvature(left_tip_points)
                right_curvature = calculate_curvature(right_tip_points)
                
                # Head is more rounded (lower curvature = smoother)
                contour_vote_left = left_curvature < right_curvature
                vshape_confidence = 0.3  # Lower confidence for fallback method
            
            logger.debug(
                f"V-shape detection: Left(V={left_is_vshape}, angle={left_angle:.1f}°, conf={left_conf:.2f}), "
                f"Right(V={right_is_vshape}, angle={right_angle:.1f}°, conf={right_conf:.2f}), "
                f"Vote: head_left={contour_vote_left}, confidence={vshape_confidence:.2f}"
            )
        
    # Voting Logic with weights
    score_left = 0
    score_right = 0
    
    # Eye detection gets highest weight (if available)
    if eye_vote_left is not None:
        weight = int(eye_confidence * 5) + 3  # Weight 3-8 based on confidence
        if eye_vote_left:
            score_left += weight
        else:
            score_right += weight
    
    # Mass (weight: 1)
    score_left += 1 if mass_on_left else 0
    score_right += 1 if not mass_on_left else 0
    
    # Width (weight: 1)
    score_left += 1 if width_on_left else 0
    score_right += 1 if not width_on_left else 0
    
    # Tail flare (weight: 2)
    if (right_flares and not left_flares):
        score_left += 2  # Strong indicator Right is Tail
    elif (left_flares and not right_flares):
        score_right += 2  # Strong indicator Left is Tail
    
    # V-shape/Contour analysis (weight: 2-3 based on confidence)
    # This is a strong indicator: tail has V-shape, head is rounded
    if contour_vote_left is not None:
        weight = int(vshape_confidence * 2) + 2  # Weight 2-4 based on confidence
        if contour_vote_left:
            score_left += weight
        else:
            score_right += weight
        
    head_on_left = score_left > score_right
    
    # Calculate confidence based on vote margin and signal strength
    total_votes = score_left + score_right
    vote_margin = abs(score_left - score_right) / max(total_votes, 1)
    
    # Combine vote margin with eye confidence and V-shape confidence (if available)
    if eye_confidence > 0:
        # Eye detection is most reliable
        confidence = 0.5 * eye_confidence + 0.3 * vshape_confidence + 0.2 * vote_margin
    elif vshape_confidence > 0.5:
        # V-shape detection is very reliable when confident
        confidence = 0.6 * vshape_confidence + 0.4 * vote_margin
    else:
        confidence = vote_margin
    
    # Boost confidence if tail flare is strong signal
    if (right_flares and not left_flares) or (left_flares and not right_flares):
        confidence = min(1.0, confidence * 1.2)
    
    # Boost confidence if V-shape detection is very confident
    if vshape_confidence > 0.7:
        confidence = min(1.0, confidence * 1.15)
    
    logger.debug(
        f"Head detection votes: Left={score_left}, Right={score_right}, Confidence={confidence:.2f}. "
        f"(Eye: {eye_vote_left if eye_vote_left is not None else 'N/A'}[{eye_confidence:.2f}], "
        f"MassL={mass_on_left}, WidthL={width_on_left}, "
        f"LFla={left_flares}[{left_tip_avg:.1f}vs{left_neck_avg:.1f}], "
        f"RFla={right_flares}[{right_tip_avg:.1f}vs{right_neck_avg:.1f}], "
        f"VShape: {contour_vote_left if contour_vote_left is not None else 'N/A'}[{vshape_confidence:.2f}])"
    )
    
    # Calculate adjusted angle (always returning angle for Head Left state)
    if not head_on_left:
        adjusted_angle = orientation_angle + 180
    else:
        adjusted_angle = orientation_angle

    return adjusted_angle, head_on_left, confidence


def detect_dorsal_orientation(
    fish_mask: np.ndarray,
    orientation_angle: float,
    image: Optional[np.ndarray] = None,
    use_image_features: bool = True,
) -> Tuple[bool, float]:
    """Detect if dorsal (back) side is on top.
    
    Uses multiple heuristics:
    1. Eye position detection (eyes are closer to dorsal edge - most reliable when head is correct)
    2. Fin detection (image-based, reliable)
    3. Variation (Standard Deviation): Dorsal side (fins) has more variation
    4. Max Width Position: Dorsal fin creates a peak width often in the top half
    5. Convexity analysis
    6. Edge roughness analysis
    
    Args:
        fish_mask: Binary mask of fish
        orientation_angle: Current orientation angle in degrees
        image: Optional grayscale or color image for fin detection
        use_image_features: Whether to use image-based features (fins)
    
    Returns:
        Tuple of (dorsal_on_top, confidence) where:
        - dorsal_on_top: True if dorsal is on top, False if ventral is on top
        - confidence: Confidence score (0-1) for the detection
    """
    # Clean mask first
    fish_mask = clean_fish_mask(fish_mask)
    
    # Ensure mask is binary
    if fish_mask.dtype != bool:
        fish_mask = fish_mask > 0.5
    
    # Rotate mask to horizontal orientation for analysis
    center = tuple(np.array(fish_mask.shape[::-1]) / 2)
    if orientation_angle != 0:
        rotation_matrix = cv2.getRotationMatrix2D(center, -orientation_angle, 1.0)
        rotated_mask = cv2.warpAffine(
            fish_mask.astype(np.uint8),
            rotation_matrix,
            fish_mask.shape[::-1],
            flags=cv2.INTER_NEAREST,
        ).astype(bool)
    else:
        rotated_mask = fish_mask
    
    # --- Heuristic 0: Eye Position Detection (eyes are closer to dorsal edge) ---
    # This is the most reliable indicator: eyes are always closer to dorsal than ventral
    eye_vote_top = None
    eye_position_confidence = 0.0
    
    if use_image_features and image is not None:
        # Detect eyes in the rotated/horizontal fish
        eye_locations, eye_conf = detect_eyes(image, fish_mask, orientation_angle)
        
        if len(eye_locations) > 0 and eye_conf > 0.3:
            # Get fish bounding box in rotated coordinates
            coords = np.column_stack(np.where(rotated_mask))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                fish_height = y_max - y_min
                fish_center_y = (y_min + y_max) / 2
                
                # Convert eye locations to rotated coordinates
                eye_y_positions = []
                angle_rad = np.radians(orientation_angle)
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)
                
                for eye_x, eye_y in eye_locations:
                    # Rotate eye position to match rotated mask
                    rel_x = eye_x - center[0]
                    rel_y = eye_y - center[1]
                    rot_x = rel_x * cos_a - rel_y * sin_a + center[0]
                    rot_y = rel_x * sin_a + rel_y * cos_a + center[1]
                    eye_y_positions.append(rot_y)
                
                if len(eye_y_positions) > 0:
                    # Calculate average eye Y position
                    avg_eye_y = np.mean(eye_y_positions)
                    
                    # Calculate distances to top and bottom edges
                    dist_to_top = avg_eye_y - y_min
                    dist_to_bottom = y_max - avg_eye_y
                    
                    # Eyes should be closer to dorsal (top) edge
                    eyes_closer_to_top = dist_to_top < dist_to_bottom
                    
                    # Confidence based on how much closer eyes are to top
                    # If eyes are much closer to top, high confidence
                    if dist_to_top > 0 and dist_to_bottom > 0:
                        ratio = dist_to_top / dist_to_bottom
                        if ratio < 0.7:  # Eyes significantly closer to top
                            eye_position_confidence = min(1.0, 0.5 + (0.7 - ratio) * 2.0)
                        elif ratio > 1.4:  # Eyes significantly closer to bottom (unexpected)
                            eye_position_confidence = 0.3  # Low confidence, might be wrong orientation
                        else:
                            eye_position_confidence = 0.6  # Moderate confidence
                    
                    eye_vote_top = eyes_closer_to_top
                    
                    logger.debug(
                        f"Eye position detection: avg_y={avg_eye_y:.1f}, "
                        f"dist_to_top={dist_to_top:.1f}, dist_to_bottom={dist_to_bottom:.1f}, "
                        f"ratio={dist_to_top/dist_to_bottom if dist_to_bottom > 0 else 'inf':.2f}, "
                        f"vote_top={eye_vote_top}, confidence={eye_position_confidence:.2f}"
                    )
    
    # --- Heuristic 1: Fin Detection (image-based, reliable) ---
    fin_vote_top = None
    fin_confidence = 0.0
    if use_image_features and image is not None:
        fin_location, fin_conf = detect_dorsal_fin(image, fish_mask, orientation_angle)
        
        if fin_location is not None and fin_conf > 0.3:
            # Check if fin is in top half of rotated fish
            coords = np.column_stack(np.where(rotated_mask))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                fish_height = y_max - y_min
                fish_center_y = (y_min + y_max) / 2
                
                # Rotate fin location to match rotated mask
                angle_rad = np.radians(orientation_angle)
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)
                rel_x = fin_location[0] - center[0]
                rel_y = fin_location[1] - center[1]
                rot_x = rel_x * cos_a - rel_y * sin_a + center[0]
                rot_y = rel_x * sin_a + rel_y * cos_a + center[1]
                
                # Check if fin is in top half
                fin_vote_top = rot_y < fish_center_y
                fin_confidence = fin_conf
                logger.debug(f"Fin detection: fin at ({rot_x:.1f}, {rot_y:.1f}), vote_top={fin_vote_top}, confidence={fin_conf:.2f}")

    # Analyze vertical profile (sum along columns)
    vertical_projection = np.sum(rotated_mask, axis=1)
    
    if len(vertical_projection) == 0:
        return True, 0.5  # Default to top with low confidence
    
    # Find top and bottom edges of the fish to crop empty space
    if not np.any(vertical_projection > 0):
        return True, 0.5  # Default to top with low confidence
        
    top_edge = np.argmax(vertical_projection > 0)
    bottom_edge = len(vertical_projection) - 1 - np.argmax(vertical_projection[::-1] > 0)
    
    fish_height = bottom_edge - top_edge
    if fish_height <= 0:
        return True, 0.5  # Default to top with low confidence
        
    # Extract only the fish part for analysis
    fish_profile = vertical_projection[top_edge:bottom_edge+1]
    mid_point = len(fish_profile) // 2
    top_half = fish_profile[:mid_point]
    bottom_half = fish_profile[mid_point:]
    
    # --- Heuristic 2: Variation (Fins create jagged profile) ---
    top_std = np.std(top_half) if len(top_half) > 0 else 0
    bottom_std = np.std(bottom_half) if len(bottom_half) > 0 else 0
    vote_std_top = top_std > bottom_std
    
    # --- Heuristic 3: Max Width Location ---
    max_width_idx = np.argmax(fish_profile)
    # If max width is in top half -> Top is heavier/fin-side?
    # Note: This assumes dorsal fin adds width.
    vote_max_width_top = max_width_idx < mid_point
    
    # --- Heuristic 4: Convexity Analysis ---
    # Dorsal side (with fins) is typically more convex/curved than ventral (belly)
    # We can measure this by looking at the curvature of the profile
    # For top edge: find the top boundary of the fish
    top_edge_profile = []
    bottom_edge_profile = []
    
    # Get the top and bottom edges for each column
    for col_idx in range(rotated_mask.shape[1]):
        col = rotated_mask[:, col_idx]
        if np.any(col):
            top_idx = np.argmax(col)
            bottom_idx = len(col) - 1 - np.argmax(col[::-1])
            # Adjust to fish region (subtract top_edge)
            top_edge_profile.append(top_idx - top_edge)
            bottom_edge_profile.append(bottom_idx - top_edge)
        else:
            top_edge_profile.append(np.nan)
            bottom_edge_profile.append(np.nan)
    
    top_edge_profile = np.array(top_edge_profile)
    bottom_edge_profile = np.array(bottom_edge_profile)
    
    # Remove NaN values
    valid_top = ~np.isnan(top_edge_profile)
    valid_bottom = ~np.isnan(bottom_edge_profile)
    
    convexity_top = 0.0
    convexity_bottom = 0.0
    
    if np.sum(valid_top) > 2:
        # Calculate second derivative (curvature) of top edge
        # Positive curvature = convex upward (dorsal fin)
        top_diff = np.diff(top_edge_profile[valid_top])
        top_diff2 = np.diff(top_diff)
        # More negative values = more convex upward (curves up)
        convexity_top = -np.mean(top_diff2) if len(top_diff2) > 0 else 0
    
    if np.sum(valid_bottom) > 2:
        # Calculate second derivative of bottom edge
        bottom_diff = np.diff(bottom_edge_profile[valid_bottom])
        bottom_diff2 = np.diff(bottom_diff)
        # More positive values = more convex downward (belly bulges)
        convexity_bottom = np.mean(bottom_diff2) if len(bottom_diff2) > 0 else 0
    
    # Dorsal side (top) should be more convex upward (more negative curvature)
    # Ventral side (bottom) should be flatter or convex downward
    vote_convexity_top = convexity_top < convexity_bottom  # More negative = more convex upward
    
    # --- Heuristic 5: Profile Asymmetry ---
    # Dorsal fin often creates asymmetry - top half has more variation in shape
    # Compare the "roughness" of top vs bottom edges
    top_edge_roughness = np.std(top_edge_profile[valid_top]) if np.sum(valid_top) > 0 else 0
    bottom_edge_roughness = np.std(bottom_edge_profile[valid_bottom]) if np.sum(valid_bottom) > 0 else 0
    vote_roughness_top = top_edge_roughness > bottom_edge_roughness
    
    # Voting with weights
    score_top = 0
    score_bottom = 0
    
    # Eye position gets highest weight (most reliable when head is correctly oriented)
    if eye_vote_top is not None:
        weight = int(eye_position_confidence * 5) + 4  # Weight 4-9 based on confidence
        if eye_vote_top:
            score_top += weight
        else:
            score_bottom += weight
    
    # Fin detection gets high weight (if available)
    if fin_vote_top is not None:
        weight = int(fin_confidence * 5) + 3  # Weight 3-8 based on confidence
        if fin_vote_top:
            score_top += weight
        else:
            score_bottom += weight
    
    # Heuristic 2: Variation (weight: 2)
    if vote_std_top:
        score_top += 2
    else:
        score_bottom += 2
    
    # Heuristic 3: Max Width Position (weight: 1)
    if vote_max_width_top:
        score_top += 1
    else:
        score_bottom += 1
    
    # Heuristic 4: Convexity (weight: 1)
    if vote_convexity_top:
        score_top += 1
    else:
        score_bottom += 1
    
    # Heuristic 5: Edge Roughness (weight: 1)
    if vote_roughness_top:
        score_top += 1
    else:
        score_bottom += 1
    
    # If Std Diff is large, it overrides everything (strong signal)
    std_diff_ratio = abs(top_std - bottom_std) / (max(top_std, bottom_std) + 1e-6)
    if std_diff_ratio > 0.3:  # 30% difference (increased threshold for more robustness)
        dorsal_on_top = top_std > bottom_std
        # High confidence if std difference is large
        std_confidence = min(1.0, std_diff_ratio)
    else:
        # Use voting with all heuristics
        dorsal_on_top = score_top > score_bottom
        std_confidence = 0.5
    
    # Calculate overall confidence
    total_votes = score_top + score_bottom
    vote_margin = abs(score_top - score_bottom) / max(total_votes, 1)
    
    # Combine vote margin with eye position confidence, fin confidence, and std confidence
    if eye_position_confidence > 0:
        # Eye position is most reliable when head is correctly oriented
        confidence = 0.5 * eye_position_confidence + 0.25 * fin_confidence + 0.15 * std_confidence + 0.1 * vote_margin
    elif fin_confidence > 0:
        confidence = 0.5 * fin_confidence + 0.3 * std_confidence + 0.2 * vote_margin
    else:
        confidence = 0.6 * std_confidence + 0.4 * vote_margin
    
    # Boost confidence if std difference is very large
    if std_diff_ratio > 0.5:
        confidence = min(1.0, confidence * 1.3)
    
    # Boost confidence if eye position is very confident
    if eye_position_confidence > 0.8:
        confidence = min(1.0, confidence * 1.2)
        
    logger.debug(
        f"Dorsal orientation: {'top' if dorsal_on_top else 'bottom'}, Confidence={confidence:.2f} "
        f"(Eye: {eye_vote_top if eye_vote_top is not None else 'N/A'}[{eye_position_confidence:.2f}], "
        f"Fin: {fin_vote_top if fin_vote_top is not None else 'N/A'}[{fin_confidence:.2f}], "
        f"Top Std: {top_std:.1f}, Bot Std: {bottom_std:.1f}, Ratio: {std_diff_ratio:.2f}, "
        f"Max Width Pos: {max_width_idx}/{len(fish_profile)}, "
        f"Convexity Top: {convexity_top:.2f}, Bot: {convexity_bottom:.2f}, "
        f"Roughness Top: {top_edge_roughness:.1f}, Bot: {bottom_edge_roughness:.1f}, "
        f"Scores: Top={score_top}, Bot={score_bottom})"
    )
    
    return dorsal_on_top, confidence


def validate_standardization(
    standardized_image: np.ndarray,
    standardized_mask: np.ndarray,
    expected_head_right: bool = True,
    head_confidence: float = 0.0,
    dorsal_confidence: float = 0.0,
) -> Tuple[bool, float]:
    """Validate that standardization produced correct orientation.
    
    Checks if the standardized image has expected properties:
    - Head is on the correct side (right if expected_head_right=True)
    - Dorsal side is on top
    - Image quality is reasonable
    
    Args:
        standardized_image: Standardized image after rotation and cropping
        standardized_mask: Standardized mask
        expected_head_right: Whether head should be on right side
        head_confidence: Confidence score from head detection
        dorsal_confidence: Confidence score from dorsal detection
    
    Returns:
        Tuple of (is_valid, validation_score) where:
        - is_valid: True if orientation appears correct
        - validation_score: Overall validation score (0-1)
    """
    if standardized_image.size == 0 or standardized_mask.size == 0:
        return False, 0.0
    
    # Ensure mask is binary
    if standardized_mask.dtype != bool:
        standardized_mask = standardized_mask > 0.5
    
    # Check if mask is reasonable size
    mask_area = np.sum(standardized_mask)
    image_area = standardized_image.shape[0] * standardized_image.shape[1]
    if mask_area < image_area * 0.1:  # Mask too small
        logger.warning(f"Validation: Mask area ({mask_area}/{image_area}) too small")
        return False, 0.3
    
    # Check orientation using simple heuristics on standardized image
    # Head should be wider than tail (if head is on right)
    horizontal_projection = np.sum(standardized_mask, axis=0)
    if len(horizontal_projection) > 0:
        non_zero = np.where(horizontal_projection > 0)[0]
        if len(non_zero) > 0:
            fish_length = len(non_zero)
            left_half = horizontal_projection[non_zero[0]:non_zero[0] + fish_length // 2]
            right_half = horizontal_projection[non_zero[0] + fish_length // 2:non_zero[-1] + 1]
            
            left_avg = np.mean(left_half) if len(left_half) > 0 else 0
            right_avg = np.mean(right_half) if len(right_half) > 0 else 0
            
            # Head is typically wider than tail
            if expected_head_right:
                head_wider = right_avg > left_avg
            else:
                head_wider = left_avg > right_avg
            
            orientation_score = 1.0 if head_wider else 0.5
        else:
            orientation_score = 0.5
    else:
        orientation_score = 0.5
    
    # Combine with detection confidence scores
    # Weight: 40% detection confidence, 40% orientation check, 20% image quality
    detection_confidence = (head_confidence + dorsal_confidence) / 2.0
    validation_score = 0.4 * detection_confidence + 0.4 * orientation_score + 0.2 * 1.0
    
    is_valid = validation_score > 0.6
    
    logger.debug(
        f"Validation: is_valid={is_valid}, score={validation_score:.2f} "
        f"(head_conf={head_confidence:.2f}, dorsal_conf={dorsal_confidence:.2f}, "
        f"orientation_score={orientation_score:.2f})"
    )
    
    return is_valid, validation_score


def rotate_fish_image(
    image: np.ndarray,
    fish_mask: np.ndarray,
    angle: float,
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate image and mask so fish is horizontal.
    
    Args:
        image: Original image (H, W, 3) or (H, W)
        fish_mask: Binary mask of fish (H, W)
        angle: Rotation angle in degrees (counterclockwise)
        background_color: RGB color for background fill
    
    Returns:
        Tuple of (rotated_image, rotated_mask)
    """
    # Ensure mask is uint8
    if fish_mask.dtype != bool:
        mask_uint8 = (fish_mask > 0.5).astype(np.uint8) * 255
    else:
        mask_uint8 = fish_mask.astype(np.uint8) * 255
    
    # Get image dimensions
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    # Calculate new dimensions to avoid cropping
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new center
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Rotate image
    if image.ndim == 3:
        rotated_image = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            borderValue=background_color,
            flags=cv2.INTER_LINEAR,
        )
    else:
        rotated_image = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            borderValue=0,
            flags=cv2.INTER_LINEAR,
        )
    
    # Rotate mask
    rotated_mask = cv2.warpAffine(
        mask_uint8,
        rotation_matrix,
        (new_w, new_h),
        borderValue=0,
        flags=cv2.INTER_NEAREST,
    )
    
    # Convert mask back to binary
    rotated_mask = rotated_mask > 127
    
    logger.debug(f"Rotated image from {image.shape} to {rotated_image.shape} at angle {angle:.2f}°")
    
    return rotated_image, rotated_mask


def crop_to_fish_bbox(
    image: np.ndarray,
    fish_mask: np.ndarray,
    margin: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Crop image and mask to fish bounding box with margin.
    
    Args:
        image: Image to crop (H, W, 3) or (H, W)
        fish_mask: Binary mask of fish (H, W)
        margin: Fraction of bounding box size to add as margin
    
    Returns:
        Tuple of (cropped_image, cropped_mask, bbox) where bbox is (x_min, y_min, x_max, y_max)
    """
    # Clean mask first - NO, cleaning already done in standardize_fish_image (implicit? No)
    # standardize_fish_image loads raw mask.
    # But calculate_fish_orientation cleans it.
    # We should clean it ONCE at start.
    # Let's add cleaning to standardize_fish_image
    
    # Ensure mask is binary
    if fish_mask.dtype != bool:
        fish_mask = fish_mask > 0.5
    
    # Find bounding box
    coords = np.column_stack(np.where(fish_mask))
    
    if len(coords) == 0:
        logger.warning("Empty fish mask, cannot crop")
        return image, fish_mask, (0, 0, image.shape[1], image.shape[0])
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Calculate margin in pixels
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    margin_x = int(bbox_width * margin)
    margin_y = int(bbox_height * margin)
    
    # Apply margin
    h, w = image.shape[:2]
    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(w, x_max + margin_x)
    y_max = min(h, y_max + margin_y)
    
    # Crop image and mask
    if image.ndim == 3:
        cropped_image = image[y_min:y_max, x_min:x_max, :]
    else:
        cropped_image = image[y_min:y_max, x_min:x_max]
    
    cropped_mask = fish_mask[y_min:y_max, x_min:x_max]
    bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
    
    logger.debug(f"Cropped to bbox {bbox}, new size: {cropped_image.shape}")
    
    return cropped_image, cropped_mask, bbox


def remove_background(
    image: np.ndarray,
    fish_mask: np.ndarray,
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Remove background by setting non-fish pixels to solid color.
    
    Args:
        image: Image to process (H, W, 3) or (H, W)
        fish_mask: Binary mask of fish (H, W) where True = fish, False = background
        background_color: RGB color for background (or grayscale value if image is grayscale)
    
    Returns:
        Image with background removed
    """
    # Ensure mask is boolean
    if fish_mask.dtype != bool:
        fish_mask = fish_mask > 0.5
    
    # Create output image
    result = image.copy()
    
    if image.ndim == 3:
        # Color image
        if len(background_color) == 3:
            bg_color = np.array(background_color, dtype=image.dtype)
        else:
            bg_color = np.array([background_color[0]] * 3, dtype=image.dtype)
        
        # Set background pixels
        for c in range(3):
            result[:, :, c][~fish_mask] = bg_color[c]
    else:
        # Grayscale image
        bg_value = background_color[0] if isinstance(background_color, (list, tuple)) else background_color
        result[~fish_mask] = bg_value
    
    logger.debug(f"Removed background, set to color {background_color}")
    
    return result


def standardize_fish_image(
    image_path: Path,
    body_mask_path: Path,
    eye_mask_path: Optional[Path] = None,
    fin_mask_path: Optional[Path] = None,
    rotation_method: str = "pca",
    crop_margin: float = 0.1,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    ensure_head_right: bool = True,
    use_image_features: bool = True,
    confidence_threshold: float = 0.7,
    enable_validation: bool = True,
    enable_double_check: bool = True,
    curvature_threshold: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Standardize fish image: rotate, crop, and remove background.
    
    Uses SAM 3 semantic masks (eye, fin) for robust orientation detection.
    Falls back to image-based detection if masks are not available.
    
    Args:
        image_path: Path to original image
        body_mask_path: Path to fish body mask (from "whole fish" or "fish" prompt)
        eye_mask_path: Optional path to eye mask (from "fish eye" prompt)
        fin_mask_path: Optional path to dorsal fin mask (from "dorsal fin" prompt)
        rotation_method: Method for rotation ("moments" or "pca")
        crop_margin: Fraction of bbox size for margin
        background_color: RGB color for background
        ensure_head_right: Whether to ensure head points right
        use_image_features: Whether to use image-based features as fallback
        confidence_threshold: Minimum confidence to accept detection
        enable_validation: Whether to validate standardization result
        enable_double_check: Whether to try both orientations for low confidence
        curvature_threshold: Maximum acceptable MSE for curvature check
    
    Returns:
        Tuple of (standardized_image, standardized_mask, metadata_dict)
    """
    # Load image and body mask
    image = np.array(Image.open(image_path))
    body_mask = np.array(Image.open(body_mask_path)) > 0
    
    # Clean mask first to remove noise/ruler fragments
    body_mask = clean_fish_mask(body_mask)
    
    # Load eye and fin masks if provided
    eye_mask = None
    fin_mask = None
    eye_mask_used = False
    fin_mask_used = False
    
    if eye_mask_path is not None and eye_mask_path.exists():
        eye_mask = np.array(Image.open(eye_mask_path)) > 0
        eye_mask_used = True
        logger.debug(f"Loaded eye mask from {eye_mask_path}")
    elif use_image_features:
        logger.debug("Eye mask not provided, will use image-based detection as fallback")
    
    if fin_mask_path is not None and fin_mask_path.exists():
        fin_mask = np.array(Image.open(fin_mask_path)) > 0
        fin_mask_used = True
        logger.debug(f"Loaded fin mask from {fin_mask_path}")
    elif use_image_features:
        logger.debug("Fin mask not provided, will use image-based detection as fallback")
    
    # Convert image to grayscale for feature detection if needed
    if use_image_features:
        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
    else:
        gray_image = None
    
    metadata = {
        "original_shape": image.shape,
        "rotation_method": rotation_method,
        "use_image_features": use_image_features,
        "eye_mask_used": eye_mask_used,
        "fin_mask_used": fin_mask_used,
    }
    
    # Step 0: Check curvature (before any transformations)
    is_straight, curvature_mse = check_fish_curvature(body_mask, threshold=curvature_threshold)
    metadata["curvature_warning"] = not is_straight
    metadata["curvature_mse"] = float(curvature_mse)
    if not is_straight:
        logger.warning(f"Fish curvature detected: MSE={curvature_mse:.2f} (threshold={curvature_threshold:.2f})")
    
    # Step 1: PCA Axis Alignment (rotate to horizontal)
    orientation_angle = calculate_fish_orientation(body_mask, method=rotation_method)
    metadata["initial_orientation"] = float(orientation_angle)
    
    # Rotate image and all masks to horizontal
    rotated_image, rotated_body_mask = rotate_fish_image(
        image,
        body_mask,
        orientation_angle,
        background_color=background_color,
    )
    
    # Get the rotation matrix and dimensions used for body mask
    # (so we can apply the same transformation to eye/fin masks)
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -orientation_angle, 1.0)
    
    # Calculate new dimensions (same as in rotate_fish_image)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new center
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Rotate eye and fin masks to match rotated_image dimensions
    rotated_eye_mask = None
    rotated_fin_mask = None
    if eye_mask is not None:
        # Rotate using same matrix and dimensions as body mask
        rotated_eye_mask = cv2.warpAffine(
            eye_mask.astype(np.uint8) * 255,
            rotation_matrix,
            (new_w, new_h),
            borderValue=0,
            flags=cv2.INTER_NEAREST,
        ) > 127
    
    if fin_mask is not None:
        # Rotate using same matrix and dimensions as body mask
        rotated_fin_mask = cv2.warpAffine(
            fin_mask.astype(np.uint8) * 255,
            rotation_matrix,
            (new_w, new_h),
            borderValue=0,
            flags=cv2.INTER_NEAREST,
        ) > 127
    
    metadata["rotated_shape"] = rotated_image.shape
    
    # Step 2: Head-Tail Resolution (ensure head is right)
    # Use mask-based detection if available, otherwise fall back to image-based
    if eye_mask_used and rotated_eye_mask is not None:
        head_is_right, head_confidence = detect_head_direction_mask_based(
            rotated_body_mask,
            rotated_eye_mask,
            orientation_angle,
        )
        head_detection_method = "eye_mask"
    elif use_image_features and gray_image is not None:
        # Fallback to image-based detection
        rotated_gray, _ = rotate_fish_image(
            gray_image,
            body_mask,
            orientation_angle,
            background_color=(255, 255, 255),
        )
        _, head_on_left, head_confidence = detect_head_direction(
            rotated_body_mask,
            0.0,  # Already rotated
            image=rotated_gray,
            use_image_features=True,
        )
        head_is_right = not head_on_left
        head_detection_method = "image_based"
    else:
        # Default fallback: assume head is right with low confidence
        head_is_right = True
        head_confidence = 0.5
        head_detection_method = "default"
        logger.warning("No eye mask or image features available, defaulting to head_right=True")
    
    metadata["head_detection_method"] = head_detection_method
    metadata["head_is_right"] = head_is_right
    metadata["head_confidence"] = float(head_confidence)
    
    # Apply 180° rotation if head is not on right (and we want it on right)
    if ensure_head_right and not head_is_right:
        rotated_image = np.rot90(rotated_image, k=2)
        rotated_body_mask = np.rot90(rotated_body_mask, k=2)
        if rotated_eye_mask is not None:
            rotated_eye_mask = np.rot90(rotated_eye_mask, k=2)
        if rotated_fin_mask is not None:
            rotated_fin_mask = np.rot90(rotated_fin_mask, k=2)
        logger.debug("Rotated 180° to ensure head is on right")
        metadata["rotated_180"] = True
    else:
        metadata["rotated_180"] = False
    
    # Step 3: Dorsal-Ventral Resolution (ensure dorsal is up)
    # Use mask-based detection if available, otherwise fall back to image-based
    rotated_gray_for_fin = None
    if use_image_features and gray_image is not None:
        rotated_gray_for_fin, _ = rotate_fish_image(
            gray_image,
            body_mask,
            orientation_angle,
            background_color=(255, 255, 255),
        )
        if metadata.get("rotated_180", False):
            rotated_gray_for_fin = np.rot90(rotated_gray_for_fin, k=2)
    
    # Detection priority: eye_mask > fin_mask > image_based > default
    if eye_mask_used and rotated_eye_mask is not None:
        dorsal_is_up, dorsal_confidence = detect_dorsal_orientation_mask_based(
            rotated_body_mask,
            rotated_eye_mask,
            rotated_fin_mask,
            rotated_gray_for_fin,
            orientation_angle,
        )
        dorsal_detection_method = "eye_mask"
    elif fin_mask_used and rotated_fin_mask is not None:
        dorsal_is_up, dorsal_confidence = detect_dorsal_orientation_mask_based(
            rotated_body_mask,
            rotated_eye_mask,  # May be None, but function handles it
            rotated_fin_mask,
            rotated_gray_for_fin,
            orientation_angle,
        )
        dorsal_detection_method = "fin_mask"
    elif use_image_features and rotated_gray_for_fin is not None:
        # Fallback to image-based detection
        dorsal_is_up, dorsal_confidence = detect_dorsal_orientation(
            rotated_body_mask,
            0.0,  # Already rotated
            image=rotated_gray_for_fin,
            use_image_features=True,
        )
        dorsal_detection_method = "image_based"
    else:
        # Default fallback: assume dorsal is up with low confidence
        dorsal_is_up = True
        dorsal_confidence = 0.5
        dorsal_detection_method = "default"
        logger.warning("No fin mask or image features available, defaulting to dorsal_up=True")
    
    metadata["dorsal_detection_method"] = dorsal_detection_method
    metadata["dorsal_is_up"] = dorsal_is_up
    metadata["dorsal_confidence"] = float(dorsal_confidence)
    
    # Flip vertically if dorsal is not on top
    if not dorsal_is_up:
        rotated_image = np.flipud(rotated_image)
        rotated_body_mask = np.flipud(rotated_body_mask)
        logger.debug("Flipped vertically to ensure dorsal side is on top")
        metadata["flipped_vertically"] = True
    else:
        metadata["flipped_vertically"] = False
    
    # Step 4: Crop to bounding box
    cropped_image, cropped_mask, bbox = crop_to_fish_bbox(
        rotated_image,
        rotated_body_mask,
        margin=crop_margin,
    )
    metadata["bbox"] = bbox
    metadata["cropped_shape"] = cropped_image.shape
    
    # Step 5: Remove background
    standardized_image = remove_background(
        cropped_image,
        cropped_mask,
        background_color=background_color,
    )
    
    # Step 6: Validate standardization (if enabled)
    if enable_validation:
        is_valid, validation_score = validate_standardization(
            standardized_image,
            cropped_mask,
            expected_head_right=ensure_head_right,
            head_confidence=head_confidence,
            dorsal_confidence=dorsal_confidence,
        )
        metadata["validation_passed"] = is_valid
        metadata["validation_score"] = float(validation_score)
        
        if not is_valid:
            logger.warning(
                f"Standardization validation failed (score: {validation_score:.2f}). "
                f"Head confidence: {head_confidence:.2f}, Dorsal confidence: {dorsal_confidence:.2f}"
            )
    else:
        metadata["validation_passed"] = None
        metadata["validation_score"] = None
    
    logger.info(
        f"Standardized image: {image.shape} -> {standardized_image.shape}, "
        f"rotation: {orientation_angle:.2f}°, "
        f"head_conf: {head_confidence:.2f} ({head_detection_method}), "
        f"dorsal_conf: {dorsal_confidence:.2f} ({dorsal_detection_method}), "
        f"curvature_MSE: {curvature_mse:.2f}"
    )
    
    return standardized_image, cropped_mask, metadata
