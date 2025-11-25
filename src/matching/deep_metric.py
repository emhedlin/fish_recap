"""Deep metric learning module using DINOv2 as alternative matching approach.

This module implements Meta's DINOv2 for extracting feature embeddings from
standardized fish images. Uses cosine similarity for matching when local feature
matching fails (e.g., spots are too faint).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)


class DeepMetricExtractor:
    """Class for extracting deep embeddings using DINOv2."""
    
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str = "cuda",
    ):
        """Initialize DINOv2 model.
        
        Args:
            model_name: Name of DINOv2 model to load (e.g., dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)
            device: Device to run model on ("cuda" or "cpu")
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model_name = model_name
        
        logger.info(f"Loading DINOv2 model: {model_name} on {self.device}")
        try:
            # Load from torch hub
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load DINOv2 model: {e}")
            raise
            
        # Standard ImageNet normalization
        self.transform = T.Compose([
            T.Resize((224, 224)),  # Resize to standard size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract feature embedding from image.
        
        Args:
            image: Input image (H, W, 3) RGB
            
        Returns:
            Feature embedding vector (D,)
        """
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Preprocess
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
            
        # Return as numpy array (flattened)
        return features.cpu().numpy().flatten()
        
    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """Load image and extract embedding.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with embedding and metadata
        """
        try:
            image = np.array(Image.open(image_path))
            embedding = self.extract(image)
            
            return {
                "embedding": embedding,
                "image_path": str(image_path),
                "model": self.model_name,
            }
        except Exception as e:
            logger.error(f"Failed to extract embedding for {image_path}: {e}")
            return {
                "embedding": np.array([]),
                "image_path": str(image_path),
                "error": str(e),
            }

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            emb1: Embedding vector 1
            emb2: Embedding vector 2
            
        Returns:
            Cosine similarity score (-1.0 to 1.0)
        """
        if emb1.size == 0 or emb2.size == 0:
            return 0.0
            
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
