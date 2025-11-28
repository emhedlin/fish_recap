"""CLI script for running the preprocessing pipeline on raw fish images.

This script processes raw images through segmentation, metric extraction, and
standardization to create standardized 'passport photos' ready for matching.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.metric_extraction import (
    analyze_scale_distribution,
    detect_ruler_ticks,
    extract_metrics,
    extract_ruler_roi,
)
from src.preprocessing.segmentation import SAM3Segmenter, save_mask
from src.preprocessing.standardization import standardize_fish_image
from src.utils.visualization import (
    plot_scale_distribution,
    visualize_ruler_detection,
    visualize_standardization,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_metrics_atomic(
    all_results: list[dict],
    metadata_path: Path,
) -> None:
    """Save metrics JSON atomically to prevent corruption on interruption.
    
    Writes to a temporary file first, then atomically renames it to the final
    file. This ensures the JSON file is never left in a half-written state.
    
    Args:
        all_results: List of result dictionaries to save
        metadata_path: Path to the metrics.json file
    """
    if metadata_path is None:
        return
    
    try:
        # Create parent directory if needed
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first
        temp_path = metadata_path.with_suffix('.json.tmp')
        with open(temp_path, 'w') as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)
        
        # Atomically rename temp file to final file
        temp_path.replace(metadata_path)
        logger.debug(f"Incrementally saved metrics to {metadata_path}")
    except Exception as e:
        logger.warning(f"Failed to save metrics incrementally: {e}")


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_image_progress(
    image_stem: str,
    masks_dir: Path,
    standardized_dir: Optional[Path],
    metadata_dir: Optional[Path],
    qa_dir: Optional[Path],
) -> dict:
    """Check what processing steps have been completed for an image.
    
    Args:
        image_stem: Image filename without extension
        masks_dir: Directory containing masks
        standardized_dir: Directory containing standardized images
        metadata_dir: Directory containing metadata JSON
        qa_dir: Directory containing QA plots
    
    Returns:
        Dictionary with status of each step:
        {
            "segmentation": bool,
            "metrics": bool,
            "standardization": bool,
            "qa_plots": bool
        }
    """
    status = {
        "segmentation": False,
        "metrics": False,
        "standardization": False,
        "qa_plots": False,
    }
    
    # Check segmentation (masks)
    # Support both old ("fish") and new ("whole fish") prompt names
    fish_mask = masks_dir / f"{image_stem}_fish_mask.png"
    whole_fish_mask = masks_dir / f"{image_stem}_whole fish_mask.png"
    ruler_mask = masks_dir / f"{image_stem}_ruler_mask.png"
    # Segmentation is complete if we have either fish mask variant and ruler mask
    status["segmentation"] = (fish_mask.exists() or whole_fish_mask.exists()) and ruler_mask.exists()
    
    # Check metrics (check if metadata JSON exists and contains this image)
    if metadata_dir:
        metadata_path = metadata_dir / "metrics.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    try:
                        metadata = json.load(f)
                        if not isinstance(metadata, list):
                            metadata = []
                    except json.JSONDecodeError:
                        metadata = []
                        
                # Check if this image is in the metadata
                image_found = any(
                    item.get("image", "").endswith(f"{image_stem}.jpg") or
                    item.get("image", "").endswith(f"{image_stem}.JPG")
                    for item in metadata
                )
                if image_found:
                    # Also check if metrics data exists
                    for item in metadata:
                        if (item.get("image", "").endswith(f"{image_stem}.jpg") or
                            item.get("image", "").endswith(f"{image_stem}.JPG")):
                            if item.get("metrics") and item["metrics"].get("pixels_per_mm", 0) > 0:
                                status["metrics"] = True
                                break
            except Exception as e:
                logger.debug(f"Could not check metrics in metadata: {e}")
    
    # Check standardization
    if standardized_dir:
        std_image = standardized_dir / f"{image_stem}_standardized.png"
        std_mask = standardized_dir / f"{image_stem}_standardized_mask.png"
        status["standardization"] = std_image.exists() and std_mask.exists()
    
    # Check QA plots
    if qa_dir:
        ruler_qa = qa_dir / f"{image_stem}_ruler_qa.png"
        std_qa = qa_dir / f"{image_stem}_standardization_qa.png"
        status["qa_plots"] = ruler_qa.exists() or std_qa.exists()
    
    return status


def determine_required_steps(
    progress: dict,
    extract_metrics_flag: bool,
    standardize_flag: bool,
    force_reprocess: bool = False,
) -> dict:
    """Determine which steps need to be run based on progress and flags.
    
    Args:
        progress: Progress status dictionary from check_image_progress
        extract_metrics_flag: Whether metrics extraction is requested
        standardize_flag: Whether standardization is requested
        force_reprocess: If True, reprocess even if already done
    
    Returns:
        Dictionary indicating which steps to run:
        {
            "segmentation": bool,
            "metrics": bool,
            "standardization": bool
        }
    """
    required = {
        "segmentation": False,
        "metrics": False,
        "standardization": False,
    }
    
    if force_reprocess:
        # Reprocess everything that's requested
        required["segmentation"] = True
        required["metrics"] = extract_metrics_flag
        required["standardization"] = standardize_flag
    else:
        # Only run what's missing
        # Only require segmentation if it's needed for requested operations
        
        # Metrics requires segmentation
        if extract_metrics_flag:
            if not progress["segmentation"]:
                # Need segmentation first
                required["segmentation"] = True
                required["metrics"] = True
            elif not progress["metrics"]:
                # Segmentation done, metrics missing
                required["metrics"] = True
        
        # Standardization requires segmentation
        if standardize_flag:
            if not progress["standardization"]:
                # Standardization is missing, check if segmentation is needed
                if not progress["segmentation"]:
                    # Need segmentation first
                    required["segmentation"] = True
                required["standardization"] = True
            # If standardization is already done, don't require anything (image will be skipped)
    
    return required


def process_single_image(
    image_path: Path,
    segmenter: Optional[SAM3Segmenter],
    masks_dir: Path,
    prompts: list[str],
    required_steps: dict,
    qa_dir: Optional[Path] = None,
    standardized_dir: Optional[Path] = None,
    length_method: str = "medial_axis",
    standardization_config: Optional[dict] = None,
) -> dict:
    """Process a single image through requested steps only.
    
    Args:
        image_path: Path to input image
        segmenter: Initialized SAM3 segmenter (can be None if segmentation not needed)
        masks_dir: Directory to save masks
        prompts: List of prompts to segment
        required_steps: Dictionary indicating which steps to run
        qa_dir: Directory to save QA visualizations (optional)
        standardized_dir: Directory to save standardized images (optional)
        length_method: Method for measuring fish length
        standardization_config: Configuration dict for standardization
    
    Returns:
        Dictionary containing mask paths, metrics, and standardization results
    """
    try:
        image_stem = image_path.stem
        result = {
            "image": str(image_path),
        }
        
        # Step 1: Segmentation
        output_paths = {}
        if required_steps["segmentation"]:
            if segmenter is None:
                raise ValueError("Segmentation requested but segmenter not initialized")
            logger.info(f"Running segmentation for {image_path.name}")
            masks = segmenter.segment(image_path, prompts)
            
            # Save masks
            for prompt, mask in masks.items():
                output_filename = f"{image_stem}_{prompt}_mask.png"
                output_path = masks_dir / output_filename
                save_mask(mask, output_path)
                output_paths[prompt] = output_path
            
            result["masks"] = {k: str(v) for k, v in output_paths.items()}
        else:
            # Load existing masks (support both old and new prompt names)
            fish_mask_path = masks_dir / f"{image_stem}_fish_mask.png"
            whole_fish_mask_path = masks_dir / f"{image_stem}_whole fish_mask.png"
            ruler_mask_path = masks_dir / f"{image_stem}_ruler_mask.png"
            
            # Determine which body mask exists (prefer new name)
            body_mask_path = None
            if whole_fish_mask_path.exists():
                body_mask_path = whole_fish_mask_path
                body_mask_key = "whole fish"
            elif fish_mask_path.exists():
                body_mask_path = fish_mask_path
                body_mask_key = "fish"
            
            if body_mask_path is not None and ruler_mask_path.exists():
                logger.debug(f"Using existing masks for {image_path.name}")
                output_paths = {
                    body_mask_key: body_mask_path,
                    "ruler": ruler_mask_path,
                }
                
                # Also check for eye and fin masks (optional)
                eye_mask_path = masks_dir / f"{image_stem}_fish eye_mask.png"
                fin_mask_path = masks_dir / f"{image_stem}_dorsal fin_mask.png"
                
                if eye_mask_path.exists():
                    output_paths["fish eye"] = eye_mask_path
                if fin_mask_path.exists():
                    output_paths["dorsal fin"] = fin_mask_path
                
                result["masks"] = {k: str(v) for k, v in output_paths.items()}
            else:
                logger.warning(f"Masks not found for {image_path.name} but segmentation not required. Skipping image.")
                return result
        
        # Step 2: Extract metrics if requested
        # Support both old ("fish") and new ("whole fish") body mask names
        body_mask_key = "whole fish" if "whole fish" in output_paths else "fish"
        if required_steps["metrics"] and body_mask_key in output_paths and "ruler" in output_paths:
            try:
                metrics = extract_metrics(
                    image_path,
                    output_paths[body_mask_key],
                    output_paths["ruler"],
                    length_method=length_method,
                )
                result["metrics"] = metrics
                
                # Create QA visualization if requested
                if qa_dir and metrics.get("pixels_per_mm", 0) > 0:
                    qa_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Load ruler ROI for visualization
                    image = np.array(Image.open(image_path))
                    ruler_mask = np.array(Image.open(output_paths["ruler"])) > 0
                    ruler_roi, _ = extract_ruler_roi(image, ruler_mask)
                    
                    if ruler_roi.size > 0:
                        # Detect ticks for visualization
                        tick_positions, _ = detect_ruler_ticks(ruler_roi)
                        if len(tick_positions) > 0:
                            qa_path = qa_dir / f"{image_stem}_ruler_qa.png"
                            visualize_ruler_detection(
                                ruler_roi,
                                tick_positions,
                                metrics["pixels_per_mm"],
                                qa_path,
                            )
                            result["qa_plot"] = str(qa_path)
            except Exception as e:
                logger.warning(f"Failed to extract metrics for {image_path.name}: {e}")
                result["metrics"] = None
        
        # Step 3: Standardize image if requested
        # Support both old ("fish") and new ("whole fish") body mask names
        body_mask_key = "whole fish" if "whole fish" in output_paths else "fish"
        if required_steps["standardization"] and body_mask_key in output_paths:
            try:
                if standardization_config is None:
                    standardization_config = {}
                
                # Get eye and fin mask paths if available
                eye_mask_path = output_paths.get("fish eye")
                fin_mask_path = output_paths.get("dorsal fin")
                
                standardized_image, standardized_mask, std_metadata = standardize_fish_image(
                    image_path,
                    output_paths[body_mask_key],
                    eye_mask_path=eye_mask_path,
                    fin_mask_path=fin_mask_path,
                    rotation_method=standardization_config.get("rotation_method", "pca"),
                    crop_margin=standardization_config.get("crop_margin", 0.1),
                    background_color=tuple(standardization_config.get("background_color", [0, 0, 0])),
                    ensure_head_right=standardization_config.get("ensure_head_right", True),
                    use_image_features=standardization_config.get("use_image_features", True),
                    confidence_threshold=standardization_config.get("confidence_threshold", 0.7),
                    enable_validation=standardization_config.get("enable_validation", True),
                    enable_double_check=standardization_config.get("enable_double_check", True),
                    curvature_threshold=standardization_config.get("curvature_threshold", 50.0),
                )
                
                # Save standardized image and mask
                if standardized_dir:
                    standardized_dir.mkdir(parents=True, exist_ok=True)
                    
                    std_image_path = standardized_dir / f"{image_stem}_standardized.png"
                    std_mask_path = standardized_dir / f"{image_stem}_standardized_mask.png"
                    
                    Image.fromarray(standardized_image.astype(np.uint8)).save(std_image_path)
                    Image.fromarray((standardized_mask * 255).astype(np.uint8)).save(std_mask_path)
                    
                    result["standardized"] = {
                        "image": str(std_image_path),
                        "mask": str(std_mask_path),
                        "metadata": std_metadata,
                    }
                    
                    # Create QA visualization
                    if qa_dir:
                        qa_dir.mkdir(parents=True, exist_ok=True)
                        original_image = np.array(Image.open(image_path))
                        original_mask = np.array(Image.open(output_paths[body_mask_key])) > 0
                        
                        qa_path = qa_dir / f"{image_stem}_standardization_qa.png"
                        visualize_standardization(
                            original_image,
                            standardized_image,
                            original_mask,
                            standardized_mask,
                            std_metadata,
                            qa_path,
                        )
                        result["standardization_qa"] = str(qa_path)
            except Exception as e:
                logger.warning(f"Failed to standardize {image_path.name}: {e}")
                result["standardized"] = None
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {e}")
        raise


def process_all_images(
    raw_dir: Path,
    masks_dir: Path,
    config: dict,
    skip_existing: bool = True,
    extract_metrics_flag: bool = True,
    standardize_flag: bool = True,
    qa_dir: Optional[Path] = None,
    standardized_dir: Optional[Path] = None,
    metadata_dir: Optional[Path] = None,
) -> None:
    """Process all images in raw directory.
    
    Args:
        raw_dir: Directory containing raw images
        masks_dir: Directory to save processed masks
        config: Configuration dictionary
        skip_existing: Skip images that already have masks
        extract_metrics_flag: Whether to extract metrics
        qa_dir: Directory to save QA visualizations
        metadata_dir: Directory to save metadata JSON
    """
    # Get all image files
    image_files = sorted(raw_dir.glob("*.jpg")) + sorted(raw_dir.glob("*.JPG"))
    
    if not image_files:
        logger.error(f"No images found in {raw_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    logger.info(f"Skip existing: {skip_existing}, Extract metrics: {extract_metrics_flag}, Standardize: {standardize_flag}")
    logger.info(f"Masks directory: {masks_dir}")
    logger.info(f"Standardized directory: {standardized_dir}")
    logger.info(f"Metadata directory: {metadata_dir}")
    
    # Check if we need segmentation at all before loading model
    needs_segmentation = False
    if not skip_existing:
        needs_segmentation = True
    else:
        # Check a few images to see if any need segmentation
        sample_checks = min(5, len(image_files))
        for image_path in image_files[:sample_checks]:
            image_stem = image_path.stem
            progress = check_image_progress(
                image_stem,
                masks_dir,
                standardized_dir,
                metadata_dir,
                qa_dir,
            )
            if not progress["segmentation"]:
                needs_segmentation = True
                break
    
    # Initialize segmenter only if needed (model loaded once, reused for all images)
    segmenter = None
    if needs_segmentation:
        logger.info("Loading SAM3 model (segmentation needed)...")
        seg_config = config.get("segmentation", {})
        segmenter = SAM3Segmenter(
            confidence_threshold=seg_config.get("confidence_threshold", 0.5),
            device=seg_config.get("device", "cuda"),
        )
    else:
        logger.info("Skipping SAM3 model loading (all images already segmented)")
    
    # Ensure we ask for the specific parts we need for the new logic
    default_prompts = ["whole fish", "fish eye", "dorsal fin", "ruler"]
    prompts = config.get("segmentation", {}).get("prompts", default_prompts)
    metric_config = config.get("metric_extraction", {})
    length_method = metric_config.get("length_measurement_method", "medial_axis")
    standardization_config = config.get("standardization", {})
    
    # Store metadata_path early for incremental saving (even if metadata_dir is None initially)
    metadata_path = metadata_dir / "metrics.json" if metadata_dir else None
    
    # Load existing metrics if available to preserve history and speed up skipping
    existing_results = {}
    if metadata_dir and metadata_path:
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                image_path_str = item.get("image", "")
                                if image_path_str:
                                    stem = Path(image_path_str).stem
                                    existing_results[stem] = item
                    except json.JSONDecodeError:
                        pass
                logger.info(f"Loaded {len(existing_results)} existing metrics entries")
            except Exception as e:
                 logger.warning(f"Could not load existing metrics: {e}")
    
    # Pre-filter images to only those needing work
    logger.info("Pre-scanning images to determine work needed...")
    images_to_process = []
    skipped_pre_count = 0
    all_results = []
    all_metrics = []
    
    for image_path in tqdm(image_files, desc="Pre-scanning images", leave=False):
        image_stem = image_path.stem
        progress = check_image_progress(
            image_stem,
            masks_dir,
            standardized_dir,
            metadata_dir,
            qa_dir,
        )
        required_steps = determine_required_steps(
            progress,
            extract_metrics_flag,
            standardize_flag,
            force_reprocess=not skip_existing,
        )
        if any(required_steps.values()):
            images_to_process.append(image_path)
        else:
            skipped_pre_count += 1
            # Still add skipped images to all_results for completeness
            if image_stem in existing_results:
                all_results.append(existing_results[image_stem])
                if extract_metrics_flag and existing_results[image_stem].get("metrics"):
                    all_metrics.append(existing_results[image_stem]["metrics"])
            elif extract_metrics_flag and progress["metrics"]:
                # Try to load metrics for skipped images
                try:
                    fish_mask_path = masks_dir / f"{image_stem}_fish_mask.png"
                    whole_fish_mask_path = masks_dir / f"{image_stem}_whole fish_mask.png"
                    ruler_mask_path = masks_dir / f"{image_stem}_ruler_mask.png"
                    
                    # Determine which body mask exists
                    body_mask_path = None
                    if whole_fish_mask_path.exists():
                        body_mask_path = whole_fish_mask_path
                    elif fish_mask_path.exists():
                        body_mask_path = fish_mask_path
                    
                    if body_mask_path and ruler_mask_path.exists():
                        metrics = extract_metrics(
                            image_path,
                            body_mask_path,
                            ruler_mask_path,
                            length_method=length_method,
                        )
                        all_metrics.append(metrics)
                        all_results.append({
                            "image": str(image_path),
                            "metrics": metrics
                        })
                except Exception as e:
                    logger.debug(f"Could not load metrics for {image_path.name}: {e}")
    
    logger.info(f"Found {len(images_to_process)} images needing processing out of {len(image_files)} total ({skipped_pre_count} already complete)")
    
    # Save metrics for pre-scanned skipped images if any were added
    if extract_metrics_flag and metadata_path and all_results:
        save_metrics_atomic(all_results, metadata_path)
    
    # Process each image
    successful = 0
    failed = 0
    skipped = skipped_pre_count  # Start with pre-scanned skipped count
    # all_results and all_metrics already populated from pre-scan
    
    # Summary statistics
    step_counts = {
        "segmentation": 0,
        "metrics": 0,
        "standardization": 0,
    }
    
    try:
        for image_path in tqdm(images_to_process, desc="Processing images"):
            image_stem = image_path.stem
            
            # Check current progress (we already know this needs work, but check again for safety)
            progress = check_image_progress(
                image_stem,
                masks_dir,
                standardized_dir,
                metadata_dir,
                qa_dir,
            )
            
            # Log progress status (at INFO level for first few images, DEBUG for rest)
            if successful + failed < 3:
                logger.info(
                    f"{image_path.name} progress: "
                    f"segmentation={progress['segmentation']}, "
                    f"metrics={progress['metrics']}, "
                    f"standardization={progress['standardization']}"
                )
            
            # Determine what needs to be done
            required_steps = determine_required_steps(
                progress,
                extract_metrics_flag,
                standardize_flag,
                force_reprocess=not skip_existing,
            )
            
            # Double-check that work is still needed (should always be true after pre-scan)
            if not any(required_steps.values()):
                logger.debug(f"Skipping {image_path.name} (all requested steps already complete)")
                skipped += 1
                continue
            
            # Log what will be done
            steps_to_run = [step for step, needed in required_steps.items() if needed]
            logger.debug(f"Processing {image_path.name}: running steps {', '.join(steps_to_run)}")
            
            # Update step counts
            for step, needed in required_steps.items():
                if needed:
                    step_counts[step] += 1
            
            try:
                # Only pass segmenter if segmentation is needed
                if required_steps["segmentation"] and segmenter is None:
                    logger.warning(f"Segmentation needed for {image_path.name} but segmenter not loaded. Loading now...")
                    seg_config = config.get("segmentation", {})
                    segmenter = SAM3Segmenter(
                        confidence_threshold=seg_config.get("confidence_threshold", 0.5),
                        device=seg_config.get("device", "cuda"),
                    )
                
                result = process_single_image(
                    image_path,
                    segmenter,  # Can be None if segmentation not needed
                    masks_dir,
                    prompts,
                    required_steps=required_steps,
                    qa_dir=qa_dir,
                    standardized_dir=standardized_dir,
                    length_method=length_method,
                    standardization_config=standardization_config,
                )
                all_results.append(result)
                if extract_metrics_flag and result.get("metrics"):
                    all_metrics.append(result["metrics"])
                    # Save metrics incrementally after extraction
                    if extract_metrics_flag and metadata_path:
                        save_metrics_atomic(all_results, metadata_path)
                successful += 1
                logger.debug(f"Successfully processed {image_path.name}")
            except Exception as e:
                failed += 1
                logger.error(f"Failed to process {image_path.name}: {e}")
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user (KeyboardInterrupt)")
        # Save metrics before exiting
        if extract_metrics_flag and metadata_path and all_results:
            logger.info("Saving metrics before exit...")
            save_metrics_atomic(all_results, metadata_path)
            logger.info(f"Metrics saved to {metadata_path}")
        raise
    
    # Log summary
    logger.info(f"Processing summary:")
    logger.info(f"  Total images: {len(image_files)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Skipped (already complete): {skipped}")
    logger.info(f"  Steps run:")
    logger.info(f"    - Segmentation: {step_counts['segmentation']}")
    logger.info(f"    - Metrics: {step_counts['metrics']}")
    logger.info(f"    - Standardization: {step_counts['standardization']}")
    
    logger.info(f"Processing complete: {successful} successful, {failed} failed")
    
    # Save metadata and analyze distribution
    if extract_metrics_flag and all_metrics:
        if metadata_dir:
            metadata_dir.mkdir(parents=True, exist_ok=True)
            
            # Save individual metrics
            metadata_path = metadata_dir / "metrics.json"
            with open(metadata_path, 'w') as f:
                json.dump(all_results, f, indent=2, cls=NumpyEncoder)
            logger.info(f"Saved metrics to {metadata_path}")
            
            # Analyze distribution
            distribution = analyze_scale_distribution(all_metrics)
            
            # Save distribution analysis
            dist_path = metadata_dir / "scale_distribution.json"
            with open(dist_path, 'w') as f:
                json.dump(distribution, f, indent=2, cls=NumpyEncoder)
            logger.info(f"Saved distribution analysis to {dist_path}")
            
            # Create distribution plot
            if qa_dir:
                pixels_per_mm_values = [m.get("pixels_per_mm", 0.0) for m in all_metrics]
                plot_path = qa_dir / "scale_distribution.png"
                plot_scale_distribution(pixels_per_mm_values, plot_path)
                logger.info(f"Saved distribution plot to {plot_path}")
            
            # Log outliers
            if distribution.get("outliers"):
                logger.warning(
                    f"Found {len(distribution['outliers'])} outliers in scale distribution. "
                    f"These may indicate mis-detected rulers."
                )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process raw fish images through segmentation pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Directory containing raw images (overrides config)",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        help="Directory to save masks (overrides config)",
    )
    parser.add_argument(
        "--qa-dir",
        type=Path,
        help="Directory to save QA visualizations",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        help="Directory to save metadata JSON",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip metric extraction",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Skip standardization",
    )
    parser.add_argument(
        "--standardized-dir",
        type=Path,
        help="Directory to save standardized images",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess all steps even if they're already complete (forces full reprocessing). "
             "By default, the script intelligently skips completed steps and only runs what's needed.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config
    config = load_config(args.config)
    
    # Determine paths
    # Support both absolute paths (anywhere on system) and relative paths (to project root)
    project_root = Path(__file__).parent.parent
    
    def resolve_path(config_path: str) -> Path:
        """Resolve path from config, handling both absolute and relative paths."""
        path = Path(config_path)
        # If absolute path, use as-is; otherwise join with project root
        return path if path.is_absolute() else project_root / path
    
    raw_dir = args.raw_dir or resolve_path(config["data"]["raw_dir"])
    masks_dir = args.masks_dir or resolve_path(config["data"]["masks_dir"])
    standardized_dir = args.standardized_dir or resolve_path(config["data"]["standardized_dir"])
    metadata_dir = args.metadata_dir or resolve_path(config["data"]["metadata_dir"])
    # QA dir is relative to processed_dir
    processed_dir = resolve_path(config["data"]["processed_dir"])
    qa_dir = args.qa_dir or processed_dir / "qa"
    
    # Create output directories
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    process_all_images(
        raw_dir=raw_dir,
        masks_dir=masks_dir,
        config=config,
        skip_existing=not args.no_skip_existing,
        extract_metrics_flag=not args.no_metrics,
        standardize_flag=not args.no_standardize,
        qa_dir=qa_dir if (not args.no_metrics or not args.no_standardize) else None,
        standardized_dir=standardized_dir if not args.no_standardize else None,
        metadata_dir=metadata_dir if not args.no_metrics else None,
    )


if __name__ == "__main__":
    main()
