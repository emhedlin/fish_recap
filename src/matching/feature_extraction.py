"""Feature extraction module using SIFT/ROOTSIFT for local feature detection.

This module implements Scale-Invariant Feature Transform (SIFT) for detecting
keypoints on fish bodies. SIFT is robust to scale and rotation changes, making
it ideal for matching natural spot patterns across different images.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Class for extracting local features (SIFT/RootSIFT) from images."""
    
    def __init__(
        self,
        method: str = "sift",
        n_features: int = 0,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10,
        sigma: float = 1.6,
    ):
        """Initialize feature extractor.
        
        Args:
            method: Feature extraction method ("sift" or "rootsift")
            n_features: Number of best features to retain (0 = all)
            contrast_threshold: Threshold for filtering weak features
            edge_threshold: Threshold for filtering edge-like features
            sigma: Sigma of the Gaussian applied to the input image
        """
        self.method = method.lower()
        self.n_features = n_features
        
        if "sift" in self.method:
            self.detector = cv2.SIFT_create(
                nfeatures=n_features,
                contrastThreshold=contrast_threshold,
                edgeThreshold=edge_threshold,
                sigma=sigma,
            )
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")
            
    def compute(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect keypoints and compute descriptors.
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            mask: Optional binary mask (H, W) where features should be detected
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Ensure mask is uint8
        if mask is not None:
            if mask.dtype != np.uint8:
                # Convert boolean or other types to uint8 0-255
                mask = mask.astype(np.uint8)
                if mask.max() <= 1:
                    mask = mask * 255
        
        # Detect and compute
        keypoints, descriptors = self.detector.detectAndCompute(gray, mask)
        
        if descriptors is None:
            logger.debug("No descriptors found.")
            return [], np.array([])
            
        # RootSIFT normalization (L1 normalize -> sqrt)
        if self.method == "rootsift":
            # L1 normalize
            eps = 1e-7
            descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
            # Square root
            descriptors = np.sqrt(descriptors)
            
        return list(keypoints), descriptors

    def process_image(
        self, 
        image_path: Path, 
        mask_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Load image and extract features.
        
        Args:
            image_path: Path to image file
            mask_path: Path to mask file (optional)
            
        Returns:
            Dictionary containing keypoints, descriptors, and metadata
        """
        try:
            image = np.array(Image.open(image_path))
            mask = None
            if mask_path and mask_path.exists():
                mask = np.array(Image.open(mask_path))
                
            keypoints, descriptors = self.compute(image, mask)
            
            # Convert keypoints to serializable format (for storage/debug)
            kp_list = []
            for kp in keypoints:
                kp_list.append({
                    "pt": kp.pt,
                    "size": kp.size,
                    "angle": kp.angle,
                    "response": kp.response,
                    "octave": kp.octave,
                    "class_id": kp.class_id,
                })
                
            return {
                "keypoints": keypoints,  # Keep cv2 objects for internal use
                "keypoints_list": kp_list,  # Serializable version
                "descriptors": descriptors,
                "n_features": len(keypoints),
                "image_path": str(image_path),
            }
            
        except Exception as e:
            logger.error(f"Failed to extract features from {image_path}: {e}")
            return {
                "keypoints": [],
                "keypoints_list": [],
                "descriptors": np.array([]),
                "n_features": 0,
                "image_path": str(image_path),
                "error": str(e)
            }
