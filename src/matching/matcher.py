"""FAISS-based matcher for comparing fish features.

This module implements FAISS with Product Quantization (IVFPQ) for scalable
one-vs-many matching, following the HotSpotter paper methodology. FAISS reduces
memory usage by ~16x compared to FLANN.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import faiss  # Requires: pip install faiss-cpu

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
    """Global matcher for one-vs-many matching using LNBNN scoring with FAISS.
    
    Implements HotSpotter's scalable matching approach:
    - Builds a single FAISS index with Product Quantization (IVFPQ) for all database descriptors
    - Uses Local Naive Bayes Nearest Neighbor (LNBNN) scoring with squared distances
    - Preserves specific descriptor correspondences for spatial verification
    - Handles ambiguous features naturally by comparing distances globally
    - Reduces memory usage by ~16x compared to FLANN (as per HotSpotter paper Section 5)
    
    This follows the HotSpotter paper methodology where LNBNN matches are used
    directly for geometric verification, preserving the advantage of one-vs-many
    matching over pairwise Ratio Test matching.
    """
    
    def __init__(
        self,
        use_pq: bool = True,
        flann_index_type: int = 1,  # Deprecated: kept for backward compatibility
        flann_trees: int = 5,  # Deprecated: kept for backward compatibility
        flann_checks: int = 50,  # Deprecated: kept for backward compatibility
    ):
        """Initialize global FAISS matcher.
        
        Args:
            use_pq: If True, uses Product Quantization (IVFPQ) as per HotSpotter paper.
                    If False, uses exact search (IndexFlatL2) - only for small datasets.
            flann_index_type: Deprecated, kept for backward compatibility
            flann_trees: Deprecated, kept for backward compatibility
            flann_checks: Deprecated, kept for backward compatibility
        """
        self.index: Optional[faiss.Index] = None
        self.labels: Optional[np.ndarray] = None  # (N_total,) - image ID indices
        self.label_map: Optional[Dict[int, str]] = None  # int -> image_id string
        self.id_map: Optional[Dict[str, int]] = None  # image_id string -> int (reverse lookup)
        self.descriptor_offsets: Optional[Dict[int, int]] = None  # image_idx -> offset in global array
        self.use_pq = use_pq
        self.is_built = False
        
    def build_index(
        self,
        all_descriptors: np.ndarray,
        all_labels: np.ndarray,
        label_map: Dict[int, str],
    ) -> None:
        """Build global FAISS index from aggregated descriptors.
        
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
            
        logger.info(f"Building FAISS index for {len(all_descriptors)} descriptors from {len(label_map)} images. PQ={self.use_pq}")
        
        # FAISS expects float32
        if all_descriptors.dtype != np.float32:
            all_descriptors = all_descriptors.astype(np.float32)
        
        d = all_descriptors.shape[1]  # Dimension (128 for SIFT)
        
        if self.use_pq:
            # --- Implementation of HotSpotter Section 5 (Product Quantization) ---
            # We use IVFPQ: Inverted File with Product Quantization
            
            # nlist: Number of Voronoi cells (clusters). Rule of thumb: 4 * sqrt(N)
            nlist = int(4 * np.sqrt(len(all_descriptors)))
            
            # m: Number of sub-quantizers. 128 dims / 8 bits = 16 (as per paper)
            m = 16
            
            # nbits: 8 bits per code
            quantizer = faiss.IndexFlatL2(d)
            self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
            
            logger.info("Training Product Quantizer (this might take a moment)...")
            # PQ requires training on a subset of data to learn centroids
            # FAISS recommends ~39 * nlist training points for proper cluster training.
            # With nlist = 4*sqrt(N), for large datasets (e.g., 31M descriptors -> nlist ~22k),
            # we need ~900k training points. We cap at 1.5M to ensure high-quality clusters.
            train_size = min(len(all_descriptors), 1500000)
            self.index.train(all_descriptors[:train_size])
            
            logger.info("Adding vectors to index...")
            self.index.add(all_descriptors)
        else:
            # Exact search (brute force) - fast for < 100k total descriptors
            self.index = faiss.IndexFlatL2(d)
            self.index.add(all_descriptors)
        
        # Store metadata
        self.labels = all_labels.astype(np.int32)
        self.label_map = label_map
        
        # Build reverse lookup map for fast image_id -> image_idx conversion
        self.id_map = {v: k for k, v in label_map.items()}
        
        # Build descriptor offsets: track where each image's descriptors start in global array
        self.descriptor_offsets = {}
        current_offset = 0
        for img_idx in sorted(label_map.keys()):
            self.descriptor_offsets[img_idx] = current_offset
            # Count descriptors for this image
            n_desc = np.sum(all_labels == img_idx)
            current_offset += n_desc
        
        self.is_built = True
        logger.info("FAISS index built successfully")
        
    def query_lnbnn(
        self,
        query_descriptors: np.ndarray,
        k: int = 5,
        ignore_id: Optional[str] = None,
    ) -> Tuple[Dict[str, float], Dict[str, List[cv2.DMatch]]]:
        """Query using Local Naive Bayes Nearest Neighbor (LNBNN) scoring.
        
        For each query descriptor:
        1. Find k+1 nearest neighbors in global index
        2. For each candidate image X:
           - d_target = min distance to any descriptor in image X
           - d_other = min distance to any descriptor NOT in image X
           - score_X += max(0, d_other^2 - d_target^2)  # Squared distances per paper
           - Store match correspondence for spatial verification
        
        This preserves the specific descriptor matches found during LNBNN scoring,
        which is critical for the HotSpotter algorithm. These matches are used directly
        for geometric verification instead of re-matching with Ratio Test.
        
        Args:
            query_descriptors: Query descriptors (N_query, 128)
            k: Number of nearest neighbors to consider (default: 5)
            ignore_id: Optional image ID to exclude from neighbor search (prevents self-matching)
            
        Returns:
            Tuple of:
            - Dictionary mapping image_id to LNBNN score
            - Dictionary mapping image_id to list of cv2.DMatch objects
              (trainIdx in matches refers to global descriptor index)
            
        Raises:
            RuntimeError: If index is not built
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
            
        if query_descriptors is None or len(query_descriptors) == 0:
            logger.warning("Empty query descriptors")
            return {}, {}
            
        if query_descriptors.dtype != np.float32:
            query_descriptors = query_descriptors.astype(np.float32)
        
        # Get integer index to ignore (for self-matching prevention)
        ignore_idx = -1
        if ignore_id is not None and self.id_map is not None:
            ignore_idx = self.id_map.get(ignore_id, -1)
        
        # Determine actual k (may be limited by database size)
        # If ignoring an image, request extra neighbors to ensure enough after filtering
        search_k = k + 1
        if ignore_idx != -1:
            search_k += 5  # Buffer to ensure we have enough neighbors after filtering
        max_k = min(search_k, len(self.labels))
        if max_k < 2:
            logger.warning("Database too small for LNBNN (need at least 2 descriptors)")
            return {}, {}
        
        try:
            # FAISS search: returns distances and indices
            # D: distances (squared L2), I: indices
            D, I = self.index.search(query_descriptors, max_k)
            
            scores: Dict[str, float] = {}
            matches_by_image: Dict[str, List[cv2.DMatch]] = {}
            
            # Process each query descriptor
            for q_idx in range(len(query_descriptors)):
                indices = I[q_idx]
                dists = D[q_idx]
                
                # Filter out -1 indices (if using IVFPQ and not enough neighbors found)
                valid_mask = indices != -1
                indices = indices[valid_mask]
                dists = dists[valid_mask]
                
                if len(indices) < 2:
                    continue
                
                # Map global descriptor indices to image IDs
                neighbor_img_indices = self.labels[indices]
                
                # Filter out the query image itself (prevents self-matching)
                if ignore_idx != -1:
                    is_not_self = neighbor_img_indices != ignore_idx
                    indices = indices[is_not_self]
                    dists = dists[is_not_self]
                    neighbor_img_indices = neighbor_img_indices[is_not_self]
                
                if len(indices) < 2:
                    continue
                
                unique_img_indices = np.unique(neighbor_img_indices)
                
                # For each candidate image
                for img_idx in unique_img_indices:
                    mask = (neighbor_img_indices == img_idx)
                    
                    # Get distance to closest descriptor in THIS image (d_target)
                    target_dists = dists[mask]
                    d_target = np.min(target_dists)
                    
                    # Get index of that specific descriptor (for geometric verification)
                    local_hit_idx = np.where(mask)[0][np.argmin(target_dists)]
                    global_desc_idx = indices[local_hit_idx]
                    
                    # Get distance to closest descriptor in ANY OTHER image (d_other)
                    mask_other = (neighbor_img_indices != img_idx)
                    if not np.any(mask_other):
                        continue
                    d_other = np.min(dists[mask_other])
                    
                    # HotSpotter LNBNN Equation
                    # FAISS returns squared L2 distances already
                    if d_target < d_other:
                        score_increment = d_other - d_target
                        img_id = self.label_map[img_idx]
                        scores[img_id] = scores.get(img_id, 0.0) + score_increment
                        
                        # Store match
                        match = cv2.DMatch()
                        match.queryIdx = q_idx
                        match.trainIdx = int(global_desc_idx)
                        match.distance = float(d_target)
                        
                        if img_id not in matches_by_image:
                            matches_by_image[img_id] = []
                        matches_by_image[img_id].append(match)
            
            return scores, matches_by_image
            
        except Exception as e:
            logger.error(f"LNBNN query failed: {e}")
            return {}, {}
    
    def map_global_to_local_matches(
        self,
        matches: List[cv2.DMatch],
        image_id: str,
    ) -> List[cv2.DMatch]:
        """Map matches from global descriptor indices to local indices for a specific image.
        
        Args:
            matches: List of DMatch objects with trainIdx referring to global indices
            image_id: Image ID to map matches for
            
        Returns:
            List of DMatch objects with trainIdx referring to local indices within the image
        """
        if not self.is_built or self.descriptor_offsets is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Find image_idx for this image_id (fast lookup using id_map)
        if self.id_map and image_id in self.id_map:
            image_idx = self.id_map[image_id]
        else:
            # Fallback (slower) - should not happen if id_map is properly built
            image_idx = None
            for idx, img_id in self.label_map.items():
                if img_id == image_id:
                    image_idx = idx
                    break
        
        if image_idx is None:
            logger.warning(f"Image ID {image_id} not found in label_map")
            return []
        
        # Get offset for this image
        offset = self.descriptor_offsets.get(image_idx)
        if offset is None:
            logger.warning(f"No offset found for image_idx {image_idx}")
            return []
        
        # Map matches: subtract offset from global trainIdx to get local trainIdx
        local_matches = []
        for match in matches:
            global_train_idx = match.trainIdx
            # Verify this match belongs to the correct image
            if global_train_idx < len(self.labels) and self.labels[global_train_idx] == image_idx:
                local_match = cv2.DMatch()
                local_match.queryIdx = match.queryIdx
                local_match.trainIdx = global_train_idx - offset  # Map to local index
                local_match.distance = match.distance
                local_matches.append(local_match)
        
        return local_matches
