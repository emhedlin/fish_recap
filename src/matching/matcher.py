"""FLANN-based matcher for comparing fish features.

This module implements Fast Library for Approximate Nearest Neighbors (FLANN)
matching to find similar spots between query fish and database fish.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FishMatcher:
    """Class for matching local feature descriptors using FLANN.
    
    .. deprecated:: 
        This class implements pairwise matching. Use GlobalFishMatcher for 
        scalable one-vs-many matching with LNBNN scoring.
    """
    
    def __init__(
        self,
        flann_index_type: int = 1,  # FLANN_INDEX_KDTREE = 1
        flann_trees: int = 5,
        flann_checks: int = 50,
        ratio_threshold: float = 0.75,
    ):
        """Initialize FLANN matcher.
        
        Args:
            flann_index_type: Algorithm index (1 for KDTree, 6 for LSH)
            flann_trees: Number of trees for KDTree
            flann_checks: Number of checks during search
            ratio_threshold: Lowe's ratio test threshold
        """
        self.index_params = dict(algorithm=flann_index_type, trees=flann_trees)
        self.search_params = dict(checks=flann_checks)
        self.ratio_threshold = ratio_threshold
        
        # Initialize matcher
        self.matcher = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        
    def match(
        self, 
        query_descriptors: np.ndarray, 
        train_descriptors: np.ndarray
    ) -> List[cv2.DMatch]:
        """Match descriptors between two images using KNN and ratio test.
        
        Args:
            query_descriptors: Descriptors from query image (N, D)
            train_descriptors: Descriptors from database image (M, D)
            
        Returns:
            List of good matches passing the ratio test
        """
        # Basic validation
        if query_descriptors is None or train_descriptors is None:
            return []
            
        if len(query_descriptors) < 2 or len(train_descriptors) < 2:
            # Need at least 2 for KNN check
            return []
            
        try:
            # KNN match with k=2
            knn_matches = self.matcher.knnMatch(query_descriptors, train_descriptors, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in knn_matches:
                if len(match_pair) < 2:
                    continue
                    
                m, n = match_pair
                # If the closest match is significantly better than the second closest
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
                    
            return good_matches
            
        except Exception as e:
            logger.warning(f"FLANN matching failed: {e}")
            return []

    def match_batch(
        self,
        query_descriptors: np.ndarray,
        database_descriptors: Dict[str, np.ndarray]
    ) -> Dict[str, List[cv2.DMatch]]:
        """Match query against multiple database entries.
        
        Args:
            query_descriptors: Query features
            database_descriptors: Dictionary mapping ID to features
            
        Returns:
            Dictionary mapping ID to list of matches
        """
        results = {}
        for fish_id, db_desc in database_descriptors.items():
            matches = self.match(query_descriptors, db_desc)
            results[fish_id] = matches
        return results


class GlobalFishMatcher:
    """Global matcher for one-vs-many matching using LNBNN scoring.
    
    Implements HotSpotter's scalable matching approach:
    - Builds a single FLANN index for all database descriptors
    - Uses Local Naive Bayes Nearest Neighbor (LNBNN) scoring
    - Handles ambiguous features naturally by comparing distances
    """
    
    def __init__(
        self,
        flann_index_type: int = 1,  # FLANN_INDEX_KDTREE = 1
        flann_trees: int = 5,
        flann_checks: int = 50,
    ):
        """Initialize global FLANN matcher.
        
        Args:
            flann_index_type: Algorithm index (1 for KDTree, 6 for LSH)
            flann_trees: Number of trees for KDTree
            flann_checks: Number of checks during search
        """
        self.index_params = dict(algorithm=flann_index_type, trees=flann_trees)
        self.search_params = dict(checks=flann_checks)
        
        # Initialize matcher (will be built later)
        self.matcher = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        
        # Metadata storage
        self.labels: Optional[np.ndarray] = None  # (N_total,) - image ID indices
        self.label_map: Optional[Dict[int, str]] = None  # int -> image_id string
        self.is_built = False
        
    def build_index(
        self,
        all_descriptors: np.ndarray,
        all_labels: np.ndarray,
        label_map: Dict[int, str],
    ) -> None:
        """Build global FLANN index from aggregated descriptors.
        
        Args:
            all_descriptors: Stacked descriptors from all images (N_total, 128)
            all_labels: Image ID index for each descriptor (N_total,)
            label_map: Mapping from image index to image ID string
            
        Raises:
            ValueError: If inputs are invalid or empty
        """
        if all_descriptors is None or len(all_descriptors) == 0:
            raise ValueError("Cannot build index with empty descriptors")
            
        if len(all_labels) != len(all_descriptors):
            raise ValueError(f"Labels length {len(all_labels)} != descriptors length {len(all_descriptors)}")
            
        if all_descriptors.shape[1] != 128:
            raise ValueError(f"Expected 128-dimensional descriptors, got {all_descriptors.shape[1]}")
            
        logger.info(f"Building global FLANN index with {len(all_descriptors)} descriptors from {len(label_map)} images")
        
        # Ensure descriptors are float32 for FLANN
        if all_descriptors.dtype != np.float32:
            all_descriptors = all_descriptors.astype(np.float32)
            
        # Build FLANN index
        self.matcher.add([all_descriptors])
        self.matcher.train()
        
        # Store metadata
        self.labels = all_labels.astype(np.int32)
        self.label_map = label_map
        self.is_built = True
        
        logger.info("Global FLANN index built successfully")
        
    def query_lnbnn(
        self,
        query_descriptors: np.ndarray,
        k: int = 5,
    ) -> Dict[str, float]:
        """Query using Local Naive Bayes Nearest Neighbor (LNBNN) scoring.
        
        For each query descriptor:
        1. Find k+1 nearest neighbors in global index
        2. For each candidate image X:
           - d_target = min distance to any descriptor in image X
           - d_other = min distance to any descriptor NOT in image X
           - score_X += max(0, d_other - d_target)
        
        Args:
            query_descriptors: Query descriptors (N_query, 128)
            k: Number of nearest neighbors to consider (default: 5)
            
        Returns:
            Dictionary mapping image_id to LNBNN score
            
        Raises:
            RuntimeError: If index is not built
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
            
        if query_descriptors is None or len(query_descriptors) == 0:
            logger.warning("Empty query descriptors")
            return {}
            
        # Ensure float32
        if query_descriptors.dtype != np.float32:
            query_descriptors = query_descriptors.astype(np.float32)
            
        # Determine actual k (may be limited by database size)
        max_k = min(k + 1, len(self.labels))
        if max_k < 2:
            logger.warning("Database too small for LNBNN (need at least 2 descriptors)")
            return {}
            
        try:
            # KNN search: find k+1 nearest neighbors for each query descriptor
            knn_matches = self.matcher.knnMatch(query_descriptors, k=max_k)
            
            # Initialize score accumulator
            scores: Dict[str, float] = {}
            
            # Process each query descriptor
            for query_idx, match_list in enumerate(knn_matches):
                if len(match_list) < 2:
                    continue  # Need at least 2 neighbors
                    
                # Extract distances and indices
                distances = np.array([m.distance for m in match_list])
                indices = np.array([m.trainIdx for m in match_list])
                
                # Get image IDs for these neighbors
                neighbor_img_indices = self.labels[indices]
                
                # Find unique images in the neighborhood
                unique_img_indices = np.unique(neighbor_img_indices)
                
                # For each candidate image
                for img_idx in unique_img_indices:
                    # Find closest descriptor in this image
                    mask_target = (neighbor_img_indices == img_idx)
                    if not np.any(mask_target):
                        continue
                    d_target = np.min(distances[mask_target])
                    
                    # Find closest descriptor NOT in this image
                    mask_other = (neighbor_img_indices != img_idx)
                    if not np.any(mask_other):
                        continue  # All neighbors are from same image
                    d_other = np.min(distances[mask_other])
                    
                    # LNBNN score: how much closer is this feature to target than to others?
                    # Only add positive scores (distinctive features)
                    if d_target < d_other:
                        score_increment = d_other - d_target
                        img_id = self.label_map[img_idx]
                        scores[img_id] = scores.get(img_id, 0.0) + score_increment
                        
            return scores
            
        except Exception as e:
            logger.error(f"LNBNN query failed: {e}")
            return {}
