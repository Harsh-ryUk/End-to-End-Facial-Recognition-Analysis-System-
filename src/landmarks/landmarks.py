import cv2
import dlib
import numpy as np

class LandmarkDetector:
    """
    68-point Facial Landmark Detector using Dlib.
    """
    def __init__(self, model_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)
    
    def detect(self, image, face_box):
        """
        Detect landmarks for a specific face box.
        
        Args:
            image (np.ndarray): Input image.
            face_box (np.ndarray): Bounding box [x1, y1, x2, y2].
            
        Returns:
            np.ndarray: 68x2 array of (x, y) coordinates.
        """
        if image is None:
            return None
        
        # Convert bbox to dlib rectangle
        x1, y1, x2, y2 = map(int, face_box[:4])
        rect = dlib.rectangle(x1, y1, x2, y2)
        
        # Predict
        shape = self.predictor(image, rect)
        
        # Convert to numpy
        landmarks = np.zeros((68, 2), dtype=int)
        for i in range(68):
            landmarks[i] = (shape.part(i).x, shape.part(i).y)
            
        return landmarks
