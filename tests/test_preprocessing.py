"""Unit tests for preprocessing modules."""

import unittest
from pathlib import Path
import numpy as np
import cv2

from src.preprocessing.standardization import (
    detect_eyes,
    detect_dorsal_fin,
    detect_head_direction,
    detect_dorsal_orientation,
    validate_standardization,
    clean_fish_mask,
)


class TestStandardization(unittest.TestCase):
    """Test standardization functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test image (grayscale)
        self.test_image = np.ones((100, 200), dtype=np.uint8) * 128
        
        # Create a simple fish mask (horizontal ellipse)
        self.fish_mask = np.zeros((100, 200), dtype=bool)
        cv2.ellipse(
            self.fish_mask,
            (100, 50),
            (80, 30),
            0,
            0,
            360,
            True,
            -1
        )
        
        # Add some "eyes" (dark circles) near the left end
        cv2.circle(self.test_image, (30, 50), 5, 50, -1)
        cv2.circle(self.test_image, (40, 50), 5, 50, -1)
    
    def test_clean_fish_mask(self):
        """Test mask cleaning."""
        # Add some noise
        noisy_mask = self.fish_mask.copy()
        noisy_mask[10, 10] = True  # Small disconnected component
        
        cleaned = clean_fish_mask(noisy_mask)
        
        # Should remove small disconnected components
        self.assertIsInstance(cleaned, np.ndarray)
        self.assertEqual(cleaned.dtype, bool)
    
    def test_detect_eyes(self):
        """Test eye detection."""
        eye_locations, confidence = detect_eyes(
            self.test_image,
            self.fish_mask,
            orientation_angle=0.0,
        )
        
        # Should detect eyes
        self.assertIsInstance(eye_locations, list)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_detect_dorsal_fin(self):
        """Test dorsal fin detection."""
        # Add a "fin" (protrusion) on top
        fin_image = self.test_image.copy()
        fin_mask = self.fish_mask.copy()
        cv2.rectangle(fin_image, (80, 20), (100, 30), 100, -1)
        fin_mask[20:30, 80:100] = True
        
        fin_location, confidence = detect_dorsal_fin(
            fin_image,
            fin_mask,
            orientation_angle=0.0,
        )
        
        # Should detect fin or return None
        if fin_location is not None:
            self.assertIsInstance(fin_location, tuple)
            self.assertEqual(len(fin_location), 2)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_detect_head_direction(self):
        """Test head direction detection."""
        # Test with mask only
        adjusted_angle, head_on_left, confidence = detect_head_direction(
            self.fish_mask,
            orientation_angle=0.0,
            use_image_features=False,
        )
        
        self.assertIsInstance(adjusted_angle, float)
        self.assertIsInstance(head_on_left, bool)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Test with image features
        adjusted_angle2, head_on_left2, confidence2 = detect_head_direction(
            self.fish_mask,
            orientation_angle=0.0,
            image=self.test_image,
            use_image_features=True,
        )
        
        self.assertIsInstance(adjusted_angle2, float)
        self.assertIsInstance(head_on_left2, bool)
        self.assertIsInstance(confidence2, float)
    
    def test_detect_dorsal_orientation(self):
        """Test dorsal orientation detection."""
        # Test with mask only
        dorsal_on_top, confidence = detect_dorsal_orientation(
            self.fish_mask,
            orientation_angle=0.0,
            use_image_features=False,
        )
        
        self.assertIsInstance(dorsal_on_top, bool)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Test with image features
        dorsal_on_top2, confidence2 = detect_dorsal_orientation(
            self.fish_mask,
            orientation_angle=0.0,
            image=self.test_image,
            use_image_features=True,
        )
        
        self.assertIsInstance(dorsal_on_top2, bool)
        self.assertIsInstance(confidence2, float)
    
    def test_validate_standardization(self):
        """Test standardization validation."""
        # Create a standardized image (simple case)
        standardized_image = np.ones((50, 100, 3), dtype=np.uint8) * 128
        standardized_mask = np.zeros((50, 100), dtype=bool)
        standardized_mask[10:40, 20:80] = True
        
        is_valid, validation_score = validate_standardization(
            standardized_image,
            standardized_mask,
            expected_head_right=True,
            head_confidence=0.8,
            dorsal_confidence=0.7,
        )
        
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(validation_score, float)
        self.assertGreaterEqual(validation_score, 0.0)
        self.assertLessEqual(validation_score, 1.0)


if __name__ == '__main__':
    unittest.main()
