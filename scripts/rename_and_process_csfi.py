"""One-time script to rename JAVR images based on detected head direction.

This script:
1. Processes images in data/raw/CSFI/ directory
2. For images without L/R suffix (JAVR*.jpg), detects head direction
3. Renames them to add L or R suffix based on detected orientation
4. Saves a mapping file preserving original names
5. Optionally runs the standard processing pipeline
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.segmentation import SAM3Segmenter, save_mask
from src.preprocessing.standardization import (
    calculate_fish_orientation,
    clean_fish_mask,
    detect_head_direction,
    detect_head_direction_mask_based,
    rotate_fish_image,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_head_direction_for_renaming(
    image_path: Path,
    body_mask: np.ndarray,
    eye_mask: Optional[np.ndarray],
    orientation_angle: float,
    use_image_features: bool = True,
) -> tuple[bool, float]:
    """Detect head direction for renaming purposes.
    
    Determines if head is on left or right side BEFORE any standardization.
    This is used to determine the L/R suffix for renaming.
    
    Args:
        image_path: Path to original image
        body_mask: Binary mask of fish body (original orientation)
        eye_mask: Optional binary mask of fish eye (original orientation)
        orientation_angle: Orientation angle calculated from body mask
        use_image_features: Whether to use image-based detection as fallback
    
    Returns:
        Tuple of (head_on_left, confidence) where:
        - head_on_left: True if head is on the left side (add "L" suffix)
        - confidence: Confidence score (0-1)
    """
    # Rotate mask to horizontal for analysis (same as standardization does)
    image = np.array(Image.open(image_path))
    rotated_image, rotated_body_mask = rotate_fish_image(
        image,
        body_mask,
        orientation_angle,
        background_color=(255, 255, 255),
    )
    
    # Rotate eye mask if available (using same rotation matrix as body mask)
    rotated_eye_mask = None
    if eye_mask is not None:
        # Calculate rotation matrix (same as standardization code)
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
        
        # Rotate eye mask using same matrix and dimensions as body mask
        rotated_eye_mask = cv2.warpAffine(
            eye_mask.astype(np.uint8) * 255,
            rotation_matrix,
            (new_w, new_h),
            borderValue=0,
            flags=cv2.INTER_NEAREST,
        ) > 127
    
    # Detect head direction using mask-based method if eye mask available
    if rotated_eye_mask is not None and np.sum(rotated_eye_mask) > 10:
        head_is_right, confidence = detect_head_direction_mask_based(
            rotated_body_mask,
            rotated_eye_mask,
            orientation_angle,
        )
        # Convert to head_on_left (opposite of head_is_right)
        head_on_left = not head_is_right
        logger.debug(f"Head detection (eye mask): head_on_left={head_on_left}, confidence={confidence:.2f}")
        return head_on_left, confidence
    
    # Fallback to image-based detection
    if use_image_features:
        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        rotated_gray, _ = rotate_fish_image(
            gray_image,
            body_mask,
            orientation_angle,
            background_color=(255, 255, 255),
        )
        
        _, head_on_left, confidence = detect_head_direction(
            rotated_body_mask,
            0.0,  # Already rotated
            image=rotated_gray,
            use_image_features=True,
        )
        logger.debug(f"Head detection (image-based): head_on_left={head_on_left}, confidence={confidence:.2f}")
        return head_on_left, confidence
    
    # Default fallback: assume head on left with low confidence
    logger.warning("No eye mask or image features available, defaulting to head_on_left=True")
    return True, 0.5


def process_image_for_renaming(
    image_path: Path,
    segmenter: SAM3Segmenter,
    masks_dir: Path,
    prompts: list[str],
    use_image_features: bool = True,
) -> Optional[dict]:
    """Process a single image to determine head direction for renaming.
    
    Args:
        image_path: Path to image file
        segmenter: Initialized SAM3 segmenter
        masks_dir: Directory to save masks (temporary)
        prompts: List of prompts for segmentation
        use_image_features: Whether to use image-based detection
    
    Returns:
        Dictionary with head direction info, or None if processing failed
    """
    try:
        image_stem = image_path.stem
        
        # Run segmentation
        logger.info(f"Segmenting {image_path.name}...")
        masks = segmenter.segment(image_path, prompts)
        
        # Save masks temporarily (we'll clean them up later)
        mask_paths = {}
        for prompt, mask in masks.items():
            output_filename = f"{image_stem}_{prompt}_mask.png"
            output_path = masks_dir / output_filename
            save_mask(mask, output_path)
            mask_paths[prompt] = output_path
        
        # Get body mask (support both "whole fish" and "fish")
        body_mask_key = "whole fish" if "whole fish" in masks else "fish"
        if body_mask_key not in masks:
            logger.warning(f"No body mask found for {image_path.name}")
            return None
        
        body_mask = masks[body_mask_key]
        body_mask = clean_fish_mask(body_mask)
        
        # Get eye mask if available
        eye_mask = masks.get("fish eye")
        if eye_mask is not None:
            eye_mask = eye_mask > 0.5
        
        # Calculate orientation
        orientation_angle = calculate_fish_orientation(body_mask, method="pca")
        
        # Detect head direction
        head_on_left, confidence = detect_head_direction_for_renaming(
            image_path,
            body_mask,
            eye_mask,
            orientation_angle,
            use_image_features=use_image_features,
        )
        
        return {
            "head_on_left": head_on_left,
            "confidence": float(confidence),
            "orientation_angle": float(orientation_angle),
            "mask_paths": {k: str(v) for k, v in mask_paths.items()},
        }
    
    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {e}")
        return None


def rename_images_in_directory(
    csfi_dir: Path,
    masks_dir: Path,
    config: dict,
    dry_run: bool = False,
    run_processing: bool = False,
) -> None:
    """Rename JAVR images based on detected head direction.
    
    Args:
        csfi_dir: Directory containing CSFI images
        masks_dir: Directory to save temporary masks
        config: Configuration dictionary
        dry_run: If True, don't actually rename files (just show what would happen)
        run_processing: If True, run standard processing pipeline after renaming
    """
    # Get all image files
    image_files = sorted(csfi_dir.glob("*.jpg")) + sorted(csfi_dir.glob("*.JPG"))
    
    if not image_files:
        logger.error(f"No images found in {csfi_dir}")
        return
    
    # Filter to images without L/R suffix (JAVR*.jpg pattern)
    images_to_rename = []
    for img_path in image_files:
        stem = img_path.stem.upper()  # Normalize to uppercase for comparison
        # Check if it ends with L or R
        if not (stem.endswith('L') or stem.endswith('R')):
            # Check if it matches JAVR pattern
            if stem.startswith('JAVR'):
                images_to_rename.append(img_path)
    
    if not images_to_rename:
        logger.info("No images found that need renaming (all already have L/R suffix)")
        # If no images to rename but processing requested, skip to processing
        if run_processing and not dry_run:
            logger.info("Skipping renaming, proceeding directly to processing pipeline...")
            segmenter = None  # No segmenter needed since no renaming
        else:
            return
    else:
        logger.info(f"Found {len(images_to_rename)} images to rename")
        
        # Initialize segmenter only if we have images to rename
        seg_config = config.get("segmentation", {})
        logger.info("Loading SAM3 model...")
        segmenter = SAM3Segmenter(
            confidence_threshold=seg_config.get("confidence_threshold", 0.5),
            device=seg_config.get("device", "cuda"),
        )
    
    # Get prompts
    default_prompts = ["whole fish", "fish eye", "dorsal fin", "ruler"]
    prompts = config.get("segmentation", {}).get("prompts", default_prompts)
    use_image_features = config.get("standardization", {}).get("use_image_features", True)
    
    # Ensure masks directory exists
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image (only if there are images to rename)
    rename_mapping = {}
    successful = 0
    failed = 0
    
    if images_to_rename:
        for image_path in tqdm(images_to_rename, desc="Processing images"):
            result = process_image_for_renaming(
                image_path,
                segmenter,
                masks_dir,
                prompts,
                use_image_features=use_image_features,
            )
            
            if result is None:
                failed += 1
                continue
            
            # Determine new name
            stem = image_path.stem
            suffix = "L" if result["head_on_left"] else "R"
            new_name = f"{stem}{suffix}.jpg"
            new_path = image_path.parent / new_name
            
            # Check if new name already exists
            if new_path.exists():
                logger.warning(f"Target name {new_name} already exists, skipping {image_path.name}")
                failed += 1
                continue
            
            rename_mapping[str(image_path.name)] = {
                "new_name": new_name,
                "head_direction": suffix,
                "confidence": result["confidence"],
                "orientation_angle": result["orientation_angle"],
            }
            
            # Rename file
            if not dry_run:
                image_path.rename(new_path)
                logger.info(f"Renamed {image_path.name} -> {new_name} (head_on_left={result['head_on_left']}, confidence={result['confidence']:.2f})")
            else:
                logger.info(f"[DRY RUN] Would rename {image_path.name} -> {new_name} (head_on_left={result['head_on_left']}, confidence={result['confidence']:.2f})")
            
            successful += 1
        
        # Save mapping file (only if we renamed anything)
        if rename_mapping:
            mapping_path = csfi_dir / "rename_mapping.json"
            if not dry_run:
                with open(mapping_path, 'w') as f:
                    json.dump(rename_mapping, f, indent=2)
                logger.info(f"Saved rename mapping to {mapping_path}")
            else:
                logger.info(f"[DRY RUN] Would save mapping to {mapping_path}")
        
        # Summary (only if we processed images)
        logger.info(f"Renaming summary:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total: {len(images_to_rename)}")
    
    # Optionally run processing pipeline
    if run_processing and not dry_run:
        # Clear GPU memory before running processing pipeline
        # This prevents OOM errors from having two SAM3Segmenter instances in memory
        logger.info("Clearing GPU memory before processing pipeline...")
        
        # Delete segmenter instance to free model memory (if it was created)
        if segmenter is not None:
            del segmenter
            segmenter = None
        
        # Clear PyTorch CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # Convert to MB
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**2)  # Convert to MB
            free_memory = total_memory - allocated_memory
            logger.info(f"GPU memory cleared. Free memory: {free_memory:.2f} MB / {total_memory:.2f} MB total")
        
        # Force garbage collection to ensure Python objects are freed
        gc.collect()
        
        logger.info("Running standard processing pipeline...")
        from scripts.process_images import process_all_images, load_config
        
        # Update config to point to CSFI directory
        config_copy = config.copy()
        config_copy["data"]["raw_dir"] = str(csfi_dir)
        
        process_all_images(
            raw_dir=csfi_dir,
            masks_dir=masks_dir,
            config=config_copy,
            skip_existing=True,
            extract_metrics_flag=True,
            standardize_flag=True,
            qa_dir=None,
            standardized_dir=Path(config["data"]["standardized_dir"]),
            metadata_dir=Path(config["data"]["metadata_dir"]),
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Rename JAVR images based on detected head direction"
    )
    parser.add_argument(
        "--csfi-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw" / "CSFI",
        help="Directory containing CSFI images",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        help="Directory to save temporary masks (default: data/processed/masks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually rename files, just show what would happen",
    )
    parser.add_argument(
        "--run-processing",
        action="store_true",
        help="Run standard processing pipeline after renaming",
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
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine masks directory
    if args.masks_dir:
        masks_dir = args.masks_dir
    else:
        project_root = Path(__file__).parent.parent
        masks_dir = project_root / config["data"]["masks_dir"]
    
    # Run renaming
    rename_images_in_directory(
        csfi_dir=args.csfi_dir,
        masks_dir=masks_dir,
        config=config,
        dry_run=args.dry_run,
        run_processing=args.run_processing,
    )


if __name__ == "__main__":
    main()

