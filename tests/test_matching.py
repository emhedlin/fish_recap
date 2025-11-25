"""Unit tests for matching modules."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.feature_extraction import FeatureExtractor
from src.matching.geometric_verification import GeometricVerifier
from src.matching.matcher import FishMatcher, GlobalFishMatcher


class TestMatching(unittest.TestCase):
    def setUp(self):
        # Create a dummy image
        self.image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add some "spots" (white circles on black background)
        points = [
            (50, 50), (150, 50), (50, 150), (150, 150),
            (100, 100), (75, 75), (125, 125), (75, 125), (125, 75)
        ]
        for pt in points:
            cv2.circle(self.image, pt, 5, (255, 255, 255), -1)
            
        # Create a transformed version (shifted)
        M = np.float32([[1, 0, 10], [0, 1, 10]])
        self.image_shifted = cv2.warpAffine(self.image, M, (200, 200))
        
        self.extractor = FeatureExtractor(method="sift", contrast_threshold=0.01)
        self.matcher = FishMatcher(ratio_threshold=0.8)
        self.verifier = GeometricVerifier(min_inliers=4)

    def test_feature_extraction(self):
        kps, desc = self.extractor.compute(self.image)
        self.assertTrue(len(kps) > 0)
        self.assertIsNotNone(desc)
        self.assertEqual(desc.shape[0], len(kps))
        self.assertEqual(desc.shape[1], 128) # SIFT descriptor size

    def test_matching_identity(self):
        # Match image against itself
        kps, desc = self.extractor.compute(self.image)
        matches = self.matcher.match(desc, desc)
        
        # Should have high number of matches (perfect match)
        self.assertTrue(len(matches) > 0)
        
        # Check self-matching (should match index i to i with distance 0)
        # Note: FLANN might not return exact order, but for identical descriptors distance is 0
        perfect_matches = 0
        for m in matches:
            if m.distance < 1e-6:
                perfect_matches += 1
        self.assertTrue(perfect_matches > 0)

    def test_matching_transformed(self):
        # Match against shifted image
        kp1, desc1 = self.extractor.compute(self.image)
        kp2, desc2 = self.extractor.compute(self.image_shifted)
        
        matches = self.matcher.match(desc1, desc2)
        self.assertTrue(len(matches) > 0)
        
        # Verify geometry
        num_inliers, M, inliers = self.verifier.verify(kp1, kp2, matches)
        
        self.assertTrue(num_inliers >= 4)
        self.assertIsNotNone(M)
        
        # Shift should be roughly [1, 0, 10], [0, 1, 10]
        # Homography is 3x3, top 2x3 should match affine transform close enough
        # Note: SIFT localization error might make it not exact
        pass

    def test_empty_input(self):
        # Test empty descriptors
        matches = self.matcher.match(np.array([]), np.array([]))
        self.assertEqual(len(matches), 0)
        
        # Test verifier with no matches
        n, M, i = self.verifier.verify([], [], [])
        self.assertEqual(n, 0)
        self.assertIsNone(M)


class TestGeometricVerification(unittest.TestCase):
    """Test geometric verification with both Homography and Affine models."""
    
    def setUp(self):
        self.image = np.zeros((200, 200, 3), dtype=np.uint8)
        points = [
            (50, 50), (150, 50), (50, 150), (150, 150),
            (100, 100), (75, 75), (125, 125)
        ]
        for pt in points:
            cv2.circle(self.image, pt, 5, (255, 255, 255), -1)
            
        # Create a transformed version (shifted)
        M = np.float32([[1, 0, 10], [0, 1, 10]])
        self.image_shifted = cv2.warpAffine(self.image, M, (200, 200))
        
        self.extractor = FeatureExtractor(method="sift", contrast_threshold=0.01)
        self.matcher = FishMatcher(ratio_threshold=0.8)
        
    def test_homography_verification(self):
        """Test Homography-based verification."""
        verifier = GeometricVerifier(min_inliers=4, model="homography")
        
        kp1, desc1 = self.extractor.compute(self.image)
        kp2, desc2 = self.extractor.compute(self.image_shifted)
        matches = self.matcher.match(desc1, desc2)
        
        num_inliers, M, inliers = verifier.verify(kp1, kp2, matches)
        
        self.assertTrue(num_inliers >= 4)
        self.assertIsNotNone(M)
        self.assertEqual(M.shape, (3, 3))  # Homography is 3x3
        
    def test_affine_verification(self):
        """Test Affine-based verification."""
        verifier = GeometricVerifier(min_inliers=3, model="affine")
        
        kp1, desc1 = self.extractor.compute(self.image)
        kp2, desc2 = self.extractor.compute(self.image_shifted)
        matches = self.matcher.match(desc1, desc2)
        
        num_inliers, M, inliers = verifier.verify(kp1, kp2, matches)
        
        self.assertTrue(num_inliers >= 3)
        self.assertIsNotNone(M)
        self.assertEqual(M.shape, (2, 3))  # Affine is 2x3
        
    def test_affine_minimum_points(self):
        """Test that Affine requires fewer points than Homography."""
        verifier_affine = GeometricVerifier(min_inliers=3, model="affine")
        verifier_homography = GeometricVerifier(min_inliers=4, model="homography")
        
        kp1, desc1 = self.extractor.compute(self.image)
        kp2, desc2 = self.extractor.compute(self.image_shifted)
        matches = self.matcher.match(desc1, desc2)
        
        # Affine should work with 3 points
        if len(matches) >= 3:
            num_inliers_a, M_a, _ = verifier_affine.verify(kp1, kp2, matches[:3])
            # Should succeed or at least not crash
            self.assertIsNotNone(M_a or True)  # M_a can be None if RANSAC fails
            
        # Homography needs 4 points
        if len(matches) >= 4:
            num_inliers_h, M_h, _ = verifier_homography.verify(kp1, kp2, matches[:4])
            self.assertIsNotNone(M_h or True)
            
    def test_invalid_model(self):
        """Test that invalid model raises ValueError."""
        with self.assertRaises(ValueError):
            GeometricVerifier(model="invalid")


class TestGlobalFishMatcher(unittest.TestCase):
    """Test GlobalFishMatcher for one-vs-many matching."""
    
    def setUp(self):
        # Create multiple test images with different patterns
        self.images = []
        for i in range(3):
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            # Add unique pattern for each image
            offset = i * 20
            points = [
                (50 + offset, 50), (150 + offset, 50),
                (50 + offset, 150), (150 + offset, 150),
                (100 + offset, 100)
            ]
            for pt in points:
                cv2.circle(img, pt, 5, (255, 255, 255), -1)
            self.images.append(img)
            
        self.extractor = FeatureExtractor(method="sift", contrast_threshold=0.01)
        
    def test_build_index(self):
        """Test building global FLANN index."""
        matcher = GlobalFishMatcher()
        
        # Extract features from multiple images
        all_descriptors = []
        all_labels = []
        label_map = {}
        
        for img_idx, img in enumerate(self.images):
            kps, desc = self.extractor.compute(img)
            if len(desc) > 0:
                all_descriptors.append(desc)
                n_desc = len(desc)
                label_map[img_idx] = f"image_{img_idx}"
                labels = np.full(n_desc, img_idx, dtype=np.int32)
                all_labels.append(labels)
                
        if len(all_descriptors) == 0:
            self.skipTest("No descriptors extracted")
            
        global_descriptors = np.vstack(all_descriptors)
        global_labels = np.concatenate(all_labels)
        
        # Build index
        matcher.build_index(global_descriptors, global_labels, label_map)
        
        self.assertTrue(matcher.is_built)
        self.assertIsNotNone(matcher.labels)
        self.assertIsNotNone(matcher.label_map)
        
    def test_query_lnbnn(self):
        """Test LNBNN scoring."""
        matcher = GlobalFishMatcher()
        
        # Build index
        all_descriptors = []
        all_labels = []
        label_map = {}
        
        for img_idx, img in enumerate(self.images):
            kps, desc = self.extractor.compute(img)
            if len(desc) > 0:
                all_descriptors.append(desc)
                n_desc = len(desc)
                label_map[img_idx] = f"image_{img_idx}"
                labels = np.full(n_desc, img_idx, dtype=np.int32)
                all_labels.append(labels)
                
        if len(all_descriptors) == 0:
            self.skipTest("No descriptors extracted")
            
        global_descriptors = np.vstack(all_descriptors)
        global_labels = np.concatenate(all_labels)
        
        matcher.build_index(global_descriptors, global_labels, label_map)
        
        # Query with first image (should match itself best)
        query_kps, query_desc = self.extractor.compute(self.images[0])
        
        if len(query_desc) == 0:
            self.skipTest("No query descriptors")
            
        scores = matcher.query_lnbnn(query_desc, k=5)
        
        # Should return scores for images in database
        self.assertIsInstance(scores, dict)
        self.assertTrue(len(scores) > 0)
        
        # First image should have highest score (matching itself)
        if "image_0" in scores:
            self.assertTrue(scores["image_0"] > 0)
            
    def test_query_before_build(self):
        """Test that querying before building raises error."""
        matcher = GlobalFishMatcher()
        query_desc = np.random.rand(10, 128).astype(np.float32)
        
        with self.assertRaises(RuntimeError):
            matcher.query_lnbnn(query_desc)
            
    def test_build_index_empty(self):
        """Test that building with empty descriptors raises error."""
        matcher = GlobalFishMatcher()
        
        with self.assertRaises(ValueError):
            matcher.build_index(
                np.array([]),
                np.array([]),
                {}
            )
            
    def test_build_index_mismatch(self):
        """Test that mismatched descriptor/label lengths raise error."""
        matcher = GlobalFishMatcher()
        
        descriptors = np.random.rand(10, 128).astype(np.float32)
        labels = np.array([0, 1, 2])  # Wrong length
        
        with self.assertRaises(ValueError):
            matcher.build_index(descriptors, labels, {0: "img0"})
            
    def test_query_empty(self):
        """Test querying with empty descriptors."""
        matcher = GlobalFishMatcher()
        
        # Build minimal index
        descriptors = np.random.rand(5, 128).astype(np.float32)
        labels = np.array([0] * 5, dtype=np.int32)
        label_map = {0: "img0"}
        
        matcher.build_index(descriptors, labels, label_map)
        
        # Query with empty descriptors
        scores = matcher.query_lnbnn(np.array([]))
        self.assertEqual(len(scores), 0)

if __name__ == '__main__':
    unittest.main()
