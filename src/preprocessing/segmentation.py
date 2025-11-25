"""SAM 3 segmentation module for extracting fish and ruler masks from raw images.

This module integrates Meta's SAM 3 model to segment fish and rulers from raw JPEG images
using text prompts. Outputs binary masks for both objects.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

try:
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    raise ImportError(
        f"SAM3 not installed: {e}. Install with: pip install -e /path/to/sam3"
    )

try:
    from huggingface_hub.errors import GatedRepoError
except ImportError:
    GatedRepoError = Exception

logger = logging.getLogger(__name__)


class SAM3Segmenter:
    """SAM3-based segmenter for fish and ruler detection."""
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        device: str = "cuda",
        bpe_path: Optional[str] = None,
    ):
        """Initialize SAM3 segmenter.
        
        Args:
            confidence_threshold: Minimum confidence score for detections
            device: Device to run model on ("cuda" or "cpu")
            bpe_path: Optional path to BPE vocabulary file
        """
        self.confidence_threshold = confidence_threshold
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model = None
        self.processor = None
        self.bpe_path = bpe_path
        
        # Enable TensorFloat-32 for Ampere GPUs if available
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _load_model(self) -> None:
        """Load SAM3 model if not already loaded."""
        if self.model is not None:
            return
        
        logger.info("Loading SAM3 model...")
        
        # Determine BPE path
        if self.bpe_path is None:
            sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
            self.bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
        
        # Build model
        try:
            if os.path.exists(self.bpe_path):
                logger.debug(f"Using BPE file at {self.bpe_path}")
                self.model = build_sam3_image_model(bpe_path=self.bpe_path)
            else:
                logger.warning(f"BPE file not found at {self.bpe_path}, using default")
                self.model = build_sam3_image_model()
        except GatedRepoError:
            self._raise_access_error()
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "gated" in error_msg.lower() or "authorized list" in error_msg.lower():
                self._raise_access_error()
            raise
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create processor
        self.processor = Sam3Processor(self.model, confidence_threshold=self.confidence_threshold)
        logger.info("SAM3 model loaded successfully")
    
    def _raise_access_error(self) -> None:
        """Raise informative error about SAM3 access requirements."""
        error_msg = (
            "\n" + "="*70 + "\n"
            "ERROR: SAM3 Model Access Required\n"
            "="*70 + "\n\n"
            "The SAM3 model repository is gated and requires access approval.\n\n"
            "To request access:\n"
            "  1. Visit: https://huggingface.co/facebook/sam3\n"
            "  2. Click 'Request access' or 'Agree and access repository'\n"
            "  3. Wait for approval (usually quick)\n"
            "  4. Ensure you're logged in: huggingface-cli login\n\n"
            "Once approved, run again.\n"
            "="*70
        )
        raise RuntimeError(error_msg)
    
    def segment(
        self,
        image_path: Path,
        prompts: list[str],
    ) -> dict[str, np.ndarray]:
        """Segment objects from image using SAM3 text prompts.
        
        Args:
            image_path: Path to input image
            prompts: List of text prompts (e.g., ["fish", "ruler"])
        
        Returns:
            Dictionary mapping prompt names to binary mask arrays (0/1)
        """
        # Load model if needed
        self._load_model()
        
        # Load image
        logger.debug(f"Loading image from {image_path}")
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        # Set image in processor
        inference_state = self.processor.set_image(image)
        
        # Process each prompt
        masks = {}
        
        for prompt in prompts:
            logger.debug(f"Processing prompt: '{prompt}'")
            
            # Reset prompts and set text prompt
            self.processor.reset_all_prompts(inference_state)
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
            
            detected_masks = output["masks"]
            scores = output["scores"]
            
            logger.debug(f"Found {len(detected_masks)} detections for '{prompt}'")
            if len(scores) > 0:
                logger.debug(f"Confidence scores: {scores[:5]}")
            
            if len(detected_masks) == 0:
                logger.warning(f"No masks found for prompt '{prompt}'")
                mask = np.zeros((height, width), dtype=np.uint8)
            else:
                # Combine all masks (union) for this prompt
                combined_mask = np.zeros((height, width), dtype=bool)
                for mask_item in detected_masks:
                    if isinstance(mask_item, torch.Tensor):
                        mask_np = mask_item.cpu().numpy()
                    else:
                        mask_np = np.array(mask_item)
                    
                    # Ensure mask is 2D and boolean
                    if mask_np.ndim > 2:
                        mask_np = mask_np.squeeze()
                    if mask_np.dtype != bool:
                        mask_np = mask_np > 0.5
                    
                    # Ensure mask matches image dimensions
                    if mask_np.shape != (height, width):
                        mask_pil = Image.fromarray(mask_np.astype(np.uint8))
                        mask_pil = mask_pil.resize((width, height), Image.NEAREST)
                        mask_np = np.array(mask_pil) > 0
                    
                    combined_mask = combined_mask | mask_np
                
                mask = combined_mask.astype(np.uint8)
            
            masks[prompt] = mask
        
        return masks
    
    def segment_fish_and_ruler(
        self,
        image_path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Segment fish and ruler from image.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Tuple of (fish_mask, ruler_mask) as binary arrays (0/1)
        """
        masks = self.segment(image_path, prompts=["fish", "ruler"])
        return masks["fish"], masks["ruler"]


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    """Save mask as binary PNG image.
    
    Args:
        mask: Binary mask array (0 or 1)
        output_path: Path to save the mask
    """
    # Convert to uint8 (0 or 255)
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_uint8, mode='L')
    mask_image.save(output_path)
    logger.debug(f"Saved mask to {output_path}")


def segment_fish_and_ruler(
    image_path: Path,
    output_dir: Path,
    prompts: Optional[list[str]] = None,
    confidence_threshold: float = 0.5,
    device: str = "cuda",
) -> dict[str, Path]:
    """Segment fish and ruler from image and save masks.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output masks
        prompts: List of prompts (default: ["fish", "ruler"])
        confidence_threshold: Minimum confidence score
        device: Device to use ("cuda" or "cpu")
    
    Returns:
        Dictionary mapping prompt names to output mask paths
    """
    if prompts is None:
        prompts = ["fish", "ruler"]
    
    # Create segmenter
    segmenter = SAM3Segmenter(
        confidence_threshold=confidence_threshold,
        device=device,
    )
    
    # Segment image
    masks = segmenter.segment(image_path, prompts)
    
    # Save masks
    output_paths = {}
    image_stem = image_path.stem  # filename without extension
    
    for prompt, mask in masks.items():
        output_filename = f"{image_stem}_{prompt}_mask.png"
        output_path = output_dir / output_filename
        save_mask(mask, output_path)
        output_paths[prompt] = output_path
    
    return output_paths
