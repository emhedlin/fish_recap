"""CLI script for running the fish re-identification matching engine.

This script handles:
1. Feature extraction (SIFT/RootSIFT) from standardized images
2. Building a feature database
3. Running matching queries (one-vs-all or all-vs-all)
4. Geometric verification of matches
5. Generating results
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.feature_extraction import FeatureExtractor
from src.matching.matcher import FishMatcher, GlobalFishMatcher
from src.matching.geometric_verification import GeometricVerifier
from src.matching.deep_metric import DeepMetricExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def keypoints_to_dict(keypoints: List[cv2.KeyPoint]) -> List[Dict[str, Any]]:
    """Convert cv2.KeyPoint objects to serializable dict format."""
    return [
        {
            "pt": kp.pt,
            "size": kp.size,
            "angle": kp.angle,
            "response": kp.response,
            "octave": kp.octave,
            "class_id": kp.class_id,
        }
        for kp in keypoints
    ]


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


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_or_extract_features(
    image_path: Path,
    features_dir: Path,
    extractor: FeatureExtractor,
    force_recompute: bool = False
) -> Dict[str, Any]:
    """Load features from disk or extract them if missing."""
    feature_path = features_dir / f"{image_path.stem}_features.pkl"
    
    if not force_recompute and feature_path.exists():
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
            logger.warning(f"Failed to load features for {image_path.name}: {e}")
            
    # Extract features
    logger.debug(f"Extracting features for {image_path.name}")
    features = extractor.process_image(image_path)
    
    # Convert keypoints to serializable format before saving
    features_to_save = features.copy()
    if "keypoints" in features_to_save:
        features_to_save["keypoints"] = keypoints_to_dict(features["keypoints"])
    
    # Save features
    features_dir.mkdir(parents=True, exist_ok=True)
    with open(feature_path, 'wb') as f:
        pickle.dump(features_to_save, f)
    
    # Return original features with cv2.KeyPoint objects for immediate use
    return features


def run_pairwise_matching(
    database: Dict[str, Dict[str, Any]],
    queries: List[Path],
    matcher: FishMatcher,
    verifier: GeometricVerifier,
    geo_config: dict,
    output_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run pairwise (one-vs-one) matching pipeline.
    
    Args:
        database: Dictionary mapping image_id to features
        queries: List of query image paths
        matcher: Pairwise FLANN matcher
        verifier: Geometric verifier
        geo_config: Geometric verification configuration
        output_dir: Output directory for results
        
    Returns:
        Dictionary mapping query_id to list of match results
    """
    logger.info("Running pairwise matching (one-vs-one)...")
    all_results = {}
    
    for query_path in tqdm(queries, desc="Matching (pairwise)"):
        query_id = query_path.stem
        query_feat = database.get(query_id)
        
        if not query_feat:
            continue
            
        query_kp = query_feat["keypoints"]
        query_desc = query_feat["descriptors"]
        
        query_results = []
        
        for db_id, db_feat in database.items():
            if db_id == query_id:
                continue
                
            db_kp = db_feat["keypoints"]
            db_desc = db_feat["descriptors"]
            
            # Step A: Feature Matching
            matches = matcher.match(query_desc, db_desc)
            
            # Step B: Geometric Verification
            num_inliers, H, inlier_matches = verifier.verify(query_kp, db_kp, matches)
            
            if num_inliers >= geo_config.get("min_inliers", 10):
                score = verifier.calculate_score(num_inliers, len(matches))
                query_results.append({
                    "match_id": db_id,
                    "score": score,
                    "inliers": num_inliers,
                    "total_matches": len(matches)
                })
                
        # Sort by score
        query_results.sort(key=lambda x: x["score"], reverse=True)
        all_results[query_id] = query_results
        
    return all_results


def run_hotspotter_matching(
    database: Dict[str, Dict[str, Any]],
    queries: List[Path],
    global_matcher: GlobalFishMatcher,
    verifier: GeometricVerifier,
    match_config: dict,
    geo_config: dict,
    output_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run HotSpotter one-vs-many matching pipeline.
    
    This implementation follows the HotSpotter paper methodology:
    1. LNBNN scoring finds matches across all database images
    2. The specific descriptor correspondences from LNBNN are preserved
    3. These matches are used directly for geometric verification (no re-matching)
    
    This preserves the advantage of the one-vs-many algorithm, where matches
    that might fail the Ratio Test can still be valid and distinctive for
    a specific animal when compared globally.
    
    Args:
        database: Dictionary mapping image_id to features
        queries: List of query image paths
        global_matcher: Global FLANN matcher with LNBNN scoring
        verifier: Geometric verifier
        match_config: Matching configuration
        geo_config: Geometric verification configuration
        output_dir: Output directory for results
        
    Returns:
        Dictionary mapping query_id to list of match results
    """
    logger.info("Running HotSpotter matching (one-vs-many)...")
    all_results = {}
    
    lnbnn_k = match_config.get("lnbnn_k", 5)
    top_candidates = match_config.get("top_candidates", 20)
    
    for query_path in tqdm(queries, desc="Matching (HotSpotter)"):
        query_id = query_path.stem
        query_feat = database.get(query_id)
        
        if not query_feat:
            continue
            
        query_kp = query_feat["keypoints"]
        query_desc = query_feat["descriptors"]
        
        # Step A: LNBNN Scoring (fast, checks all images)
        # Returns both scores and the specific matches that generated them
        lnbnn_scores, lnbnn_matches = global_matcher.query_lnbnn(query_desc, k=lnbnn_k)
        
        if not lnbnn_scores:
            all_results[query_id] = []
            continue
            
        # Step B: Sort and take top candidates
        sorted_candidates = sorted(
            lnbnn_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_candidates]
        
        # Step C: Spatial Verification on top candidates using LNBNN matches
        query_results = []
        for candidate_id, lnbnn_score in sorted_candidates:
            candidate_feat = database.get(candidate_id)
            if not candidate_feat:
                continue
                
            candidate_kp = candidate_feat["keypoints"]
            
            # Get matches found during LNBNN scoring for this candidate
            # These matches have trainIdx referring to global descriptor indices
            global_matches = lnbnn_matches.get(candidate_id, [])
            
            if not global_matches:
                # No matches found for this candidate during LNBNN scoring
                continue
            
            # Map global descriptor indices to local indices for geometric verification
            local_matches = global_matcher.map_global_to_local_matches(
                global_matches, candidate_id
            )
            
            if not local_matches:
                # Failed to map matches (shouldn't happen, but handle gracefully)
                continue
            
            # Geometric Verification using LNBNN matches directly
            # This preserves the advantage of the one-vs-many algorithm
            num_inliers, H, inlier_matches = verifier.verify(
                query_kp, candidate_kp, local_matches
            )
            
            if num_inliers >= geo_config.get("min_inliers", 10):
                # Combine LNBNN score with inlier count
                # Weighted combination: LNBNN score + normalized inlier count
                inlier_score = verifier.calculate_score(num_inliers, len(local_matches))
                # Normalize LNBNN score by query descriptor count for fairness
                normalized_lnbnn = lnbnn_score / max(len(query_desc), 1)
                combined_score = normalized_lnbnn + inlier_score
                
                query_results.append({
                    "match_id": candidate_id,
                    "score": combined_score,
                    "lnbnn_score": lnbnn_score,
                    "inliers": num_inliers,
                    "total_matches": len(local_matches)
                })
                
        # Sort by combined score
        query_results.sort(key=lambda x: x["score"], reverse=True)
        all_results[query_id] = query_results
        
    return all_results


def run_matching_pipeline(
    standardized_dir: Path,
    features_dir: Path,
    output_dir: Path,
    config: dict,
    query_image: Optional[Path] = None,
    force_recompute: bool = False,
    matching_method: Optional[str] = None,
) -> None:
    """Run the full matching pipeline.
    
    Args:
        standardized_dir: Directory containing standardized images
        features_dir: Directory for feature storage
        output_dir: Directory for output results
        config: Configuration dictionary
        query_image: Optional specific query image
        force_recompute: Force re-extraction of features
        matching_method: Override matching method ("hotspotter" or "pairwise")
    """
    
    # 1. Setup components
    feat_config = config.get("feature_extraction", {})
    match_config = config.get("matching", {})
    geo_config = config.get("geometric_verification", {})
    
    # Determine matching method
    method = matching_method or match_config.get("method", "hotspotter")
    if method not in ["hotspotter", "pairwise"]:
        logger.warning(f"Unknown method '{method}', defaulting to 'hotspotter'")
        method = "hotspotter"
    
    logger.info(f"Using matching method: {method}")
    
    logger.info("Initializing feature extractor...")
    extractor = FeatureExtractor(
        method=feat_config.get("method", "sift"),
        n_features=feat_config.get("n_features", 0),
        contrast_threshold=feat_config.get("contrast_threshold", 0.04),
        edge_threshold=feat_config.get("edge_threshold", 10),
        sigma=feat_config.get("sigma", 1.6),
    )
    
    logger.info("Initializing geometric verifier...")
    verifier = GeometricVerifier(
        ransac_threshold=geo_config.get("ransac_threshold", 5.0),
        min_inliers=geo_config.get("min_inliers", 10),
        confidence=geo_config.get("confidence", 0.99),
        max_iterations=geo_config.get("max_iterations", 2000),
        model=geo_config.get("model", "affine"),
    )
    
    # 2. Build Database (Extract features for all images)
    image_files = sorted(standardized_dir.glob("*_standardized.png"))
    if not image_files:
        logger.error(f"No standardized images found in {standardized_dir}")
        return
        
    logger.info(f"Building feature database for {len(image_files)} images...")
    database = {}
    valid_images = []
    
    for img_path in tqdm(image_files, desc="Extracting features"):
        features = load_or_extract_features(
            img_path, features_dir, extractor, force_recompute
        )
        
        if features.get("n_features", 0) > 0:
            database[img_path.stem] = features
            valid_images.append(img_path)
        else:
            logger.warning(f"No features extracted for {img_path.name}")
            
    logger.info(f"Database built: {len(database)} valid images")
    
    # 3. Define queries
    queries = []
    if query_image:
        # Single query
        if query_image.exists():
            queries = [query_image]
        else:
            # Try finding it in standardized dir
            found = list(standardized_dir.glob(f"{query_image}*_standardized.png"))
            if found:
                queries = [found[0]]
            else:
                logger.error(f"Query image not found: {query_image}")
                return
    else:
        # All-vs-all
        queries = valid_images
        
    # 4. Run Matching
    logger.info(f"Running matching for {len(queries)} queries...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if method == "hotspotter":
        # Build global index
        logger.info("Building global FLANN index...")
        all_descriptors = []
        all_labels = []
        label_map = {}
        
        for img_idx, (img_id, feat) in enumerate(database.items()):
            desc = feat["descriptors"]
            if len(desc) == 0:
                continue
                
            all_descriptors.append(desc)
            n_desc = len(desc)
            label_map[img_idx] = img_id
            labels = np.full(n_desc, img_idx, dtype=np.int32)
            all_labels.append(labels)
            
        if len(all_descriptors) == 0:
            logger.error("No valid descriptors found in database")
            return
            
        global_descriptors = np.vstack(all_descriptors)
        global_labels = np.concatenate(all_labels)
        
        global_matcher = GlobalFishMatcher(
            flann_index_type=match_config.get("flann_index_type", 1),
            flann_trees=match_config.get("flann_trees", 5),
            flann_checks=match_config.get("flann_checks", 50),
        )
        global_matcher.build_index(global_descriptors, global_labels, label_map)
        
        all_results = run_hotspotter_matching(
            database, queries, global_matcher, verifier, match_config, geo_config, output_dir
        )
    else:  # pairwise
        matcher = FishMatcher(
            flann_index_type=match_config.get("flann_index_type", 1),
            flann_trees=match_config.get("flann_trees", 5),
            flann_checks=match_config.get("flann_checks", 50),
            ratio_threshold=match_config.get("ratio_threshold", 0.75),
        )
        
        all_results = run_pairwise_matching(
            database, queries, matcher, verifier, geo_config, output_dir
        )
        
    # 5. Save Results
    results_path = output_dir / "matching_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
        
    logger.info(f"Matching complete. Results saved to {results_path}")
    
    # Print top match for first few queries
    for i, (qid, matches) in enumerate(all_results.items()):
        if i >= 5: break
        if matches:
            top = matches[0]
            logger.info(f"Query {qid} -> Top Match: {top['match_id']} (Score: {top['score']:.2f})")
        else:
            logger.info(f"Query {qid} -> No matches found")


def main():
    parser = argparse.ArgumentParser(description="Fish Re-Identification Matching Engine")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Specific image ID or path to query (optional, default: all-vs-all)"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force re-extraction of features"
    )
    parser.add_argument(
        "--standardized-dir",
        type=Path,
        help="Directory containing standardized images"
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        help="Directory to save/load features"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save matching results"
    )
    parser.add_argument(
        "--matching-method",
        type=str,
        choices=["hotspotter", "pairwise"],
        help="Matching method: 'hotspotter' (one-vs-many) or 'pairwise' (one-vs-one). Overrides config."
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    project_root = Path(__file__).parent.parent
    
    # Support both absolute paths (anywhere on system) and relative paths (to project root)
    def resolve_path(config_path: str) -> Path:
        """Resolve path from config, handling both absolute and relative paths."""
        path = Path(config_path)
        return path if path.is_absolute() else project_root / path
    
    standardized_dir = args.standardized_dir or resolve_path(config["data"]["standardized_dir"])
    features_dir = args.features_dir or resolve_path(config["data"]["features_dir"])
    output_dir = args.output_dir or (project_root / "data" / "results")
    
    run_matching_pipeline(
        standardized_dir=standardized_dir,
        features_dir=features_dir,
        output_dir=output_dir,
        config=config,
        query_image=Path(args.query) if args.query else None,
        force_recompute=args.force_recompute,
        matching_method=args.matching_method
    )


if __name__ == "__main__":
    main()
