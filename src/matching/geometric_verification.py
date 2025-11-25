"""Geometric verification module using RANSAC for transformation estimation.

This module uses Random Sample Consensus (RANSAC) to find geometric
transformations (Homography or Affine) that map spots on Fish A to spots on Fish B.
Verifies that matching spots form a consistent geometric pattern, filtering out
false matches from wood grain or other background features.

Affine transformations are less strict than Homography and better suited for
flexible/curved fish bodies, while Homography is better for rigid planar objects.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GeometricVerifier:
    """Class for verifying geometric consistency of feature matches."""
    
    def __init__(
        self,
        ransac_threshold: float = 5.0,
        min_inliers: int = 10,
        confidence: float = 0.99,
        max_iterations: int = 2000,
        model: str = "homography",
    ):
        """Initialize geometric verifier.
        
        Args:
            ransac_threshold: Maximum reprojection error to classify as inlier
            min_inliers: Minimum number of inliers required to accept match
            confidence: RANSAC confidence level
            max_iterations: Maximum number of RANSAC iterations
            model: Transformation model ("homography" or "affine")
                - "homography": 8 DOF projective transformation (requires 4+ points)
                - "affine": 6 DOF affine transformation (requires 3+ points, better for flexible bodies)
        """
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.confidence = confidence
        self.max_iterations = max_iterations
        
        if model not in ["homography", "affine"]:
            raise ValueError(f"Unknown model: {model}. Must be 'homography' or 'affine'")
        self.model = model
        
    def verify(
        self, 
        kp1: List[cv2.KeyPoint], 
        kp2: List[cv2.KeyPoint], 
        matches: List[cv2.DMatch]
    ) -> Tuple[int, Optional[np.ndarray], List[cv2.DMatch]]:
        """Verify matches using RANSAC with specified transformation model.
        
        Args:
            kp1: Keypoints from query image
            kp2: Keypoints from train image
            matches: List of matches to verify
            
        Returns:
            Tuple of (num_inliers, transformation_matrix, inlier_matches)
            - For homography: 3x3 matrix
            - For affine: 2x3 matrix
        """
        # Check minimum points required
        min_points = 4 if self.model == "homography" else 3
        if len(matches) < min_points:
            logger.debug(f"Not enough matches for {self.model} RANSAC: {len(matches)} (need {min_points})")
            return 0, None, []
            
        # Extract points from matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        try:
            if self.model == "affine":
                # Estimate Affine Transform (2D rotation + scale + translation + shear)
                # Less strict than Homography, better for flexible/curved bodies
                M, mask = cv2.estimateAffine2D(
                    src_pts,
                    dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.ransac_threshold,
                    confidence=self.confidence,
                    maxIters=self.max_iterations,
                )
            else:  # homography
                # Find Homography with RANSAC (projective transformation)
                M, mask = cv2.findHomography(
                    src_pts, 
                    dst_pts, 
                    cv2.RANSAC, 
                    self.ransac_threshold,
                    confidence=self.confidence,
                    maxIters=self.max_iterations
                )
            
            if M is None:
                return 0, None, []
                
            # Count inliers
            matchesMask = mask.ravel().tolist()
            inlier_matches = [m for i, m in enumerate(matches) if matchesMask[i]]
            num_inliers = len(inlier_matches)
            
            # Check against minimum threshold
            if num_inliers < self.min_inliers:
                # Consider it a failed match if not enough inliers
                # But we still return the inliers found so far, caller can decide
                pass
                
            return num_inliers, M, inlier_matches
            
        except Exception as e:
            logger.warning(f"RANSAC ({self.model}) failed: {e}")
            return 0, None, []

    def calculate_score(self, num_inliers: int, total_matches: int) -> float:
        """Calculate a match score based on inliers.
        
        Args:
            num_inliers: Number of geometrically consistent matches
            total_matches: Total number of putative matches
            
        Returns:
            Score between 0 and 1 (or higher)
        """
        if total_matches == 0:
            return 0.0
            
        # Simple score: number of inliers
        # Or weighted by ratio: num_inliers * (num_inliers / total_matches)
        # For now, return number of inliers as it's robust
        return float(num_inliers)
