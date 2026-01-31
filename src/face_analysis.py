"""face_analysis.py
Extract facial features from an image using MediaPipe Face Mesh.
Returns a dictionary of facial metrics compatible with the recommendation system.
"""
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Optional, Dict


class FaceAnalyzer:
    """Analyze facial features from images using MediaPipe Face Mesh."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def extract_face_features(self, image_path: str) -> Optional[Dict[str, float]]:
        """
        Extract facial features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with facial metrics or None if no face detected
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Convert to RGB (MediaPipe uses RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get landmarks (468 points)
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape
        
        # Convert normalized landmarks to pixel coordinates
        points = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
        
        # Calculate facial metrics
        features = self._calculate_metrics(points, w, h)
        
        return features
    
    def _calculate_metrics(self, points: np.ndarray, img_width: int, img_height: int) -> Dict[str, float]:
        """
        Calculate the 6 required facial metrics from landmarks.
        
        MediaPipe Face Mesh key landmark indices:
        - Left eye: 33, 133, 159, 145
        - Right eye: 362, 263, 386, 374
        - Nose tip: 1
        - Left face contour: 234
        - Right face contour: 454
        - Chin: 152
        - Forehead: 10
        - Left eyebrow: 70
        - Right eyebrow: 300
        - Upper lip: 13
        """
        
        # Eye centers
        left_eye_center = np.mean(points[[33, 133, 159, 145]], axis=0)
        right_eye_center = np.mean(points[[362, 263, 386, 374]], axis=0)
        
        # Key facial points
        nose_tip = points[1]
        left_face = points[234]
        right_face = points[454]
        chin = points[152]
        forehead = points[10]
        left_eyebrow = points[70]
        right_eyebrow = points[300]
        upper_lip = points[13]
        
        # Calculate eye spacing
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
        
        # Face width (at eye level)
        face_width = np.linalg.norm(left_face - right_face)
        
        # Face height
        face_height = np.linalg.norm(forehead - chin)
        
        # Jaw width (lower face)
        left_jaw = points[172]
        right_jaw = points[397]
        jaw_width = np.linalg.norm(left_jaw - right_jaw)
        
        # 1. FacialSymmetry: measure left-right symmetry
        # Compare distances from centerline to left and right features
        center_x = (left_eye_center[0] + right_eye_center[0]) / 2
        left_deviation = abs(left_eye_center[0] - center_x)
        right_deviation = abs(right_eye_center[0] - center_x)
        facial_symmetry = abs(left_deviation - right_deviation) / face_width
        
        # 2. GoldenRatioDeviation: how close face proportions are to 1.618
        golden_ratio = 1.618
        face_ratio = face_height / face_width if face_width > 0 else 0
        golden_ratio_deviation = abs(face_ratio - golden_ratio) / golden_ratio
        
        # 3. EyeSpacingRatio: eye distance / face width
        eye_spacing_ratio = eye_distance / face_width if face_width > 0 else 0
        
        # 4. JawlineWidthRatio: jaw width / face width
        jawline_width_ratio = jaw_width / face_width if face_width > 0 else 0
        
        # 5. BrowToEyeDistance: vertical distance from eyebrow to eye
        left_brow_to_eye = abs(left_eyebrow[1] - left_eye_center[1]) / face_height
        right_brow_to_eye = abs(right_eyebrow[1] - right_eye_center[1]) / face_height
        brow_to_eye_distance = (left_brow_to_eye + right_brow_to_eye) / 2
        
        # 6. LipToNoseDistance: vertical distance from nose to upper lip
        lip_to_nose_distance = abs(upper_lip[1] - nose_tip[1]) / face_height
        
        # Return normalized metrics (no clipping - let the model handle the full range)
        return {
            'FacialSymmetry': round(facial_symmetry, 4),
            'GoldenRatioDeviation': round(golden_ratio_deviation, 4),
            'EyeSpacingRatio': round(eye_spacing_ratio, 4),
            'JawlineWidthRatio': round(jawline_width_ratio, 4),
            'BrowToEyeDistance': round(brow_to_eye_distance, 4),
            'LipToNoseDistance': round(lip_to_nose_distance, 4)
        }
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


def extract_face_features(image_path: str) -> Optional[Dict[str, float]]:
    """
    Convenience function to extract facial features from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with facial metrics or None if no face detected
    """
    analyzer = FaceAnalyzer()
    return analyzer.extract_face_features(image_path)


if __name__ == '__main__':
    # Test with example image
    import sys
    if len(sys.argv) > 1:
        result = extract_face_features(sys.argv[1])
        if result:
            print("Facial features extracted:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print("Error: No face detected in the image")
    else:
        print("Usage: python face_analysis.py <image_path>")
