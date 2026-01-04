import numpy as np
from scipy.spatial import distance as dist

class BlinkDetector:
    def __init__(self, eye_aspect_ratio_threshold=0.25, consecutive_frames=3):
        self.thresh = eye_aspect_ratio_threshold
        self.consecutive_frames = consecutive_frames
        self.counter = 0
        self.total_blinks = 0
        
        # Dlib landmark indices
        self.left_eye_idxs = list(range(36, 42))
        self.right_eye_idxs = list(range(42, 48))

    def eye_aspect_ratio(self, eye):
        # Vertical distances
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # Horizontal distance
        C = dist.euclidean(eye[0], eye[3])
        
        ear = (A + B) / (2.0 * C)
        return ear

    def detect(self, shape):
        """
        Check for blink in the *current frame*.
        Must be called sequentially to track blink states.
        
        Args:
            shape (list/np.array): 68 landmarks (x, y).
            
        Returns:
            dict: {is_blinking: bool, blinks_total: int, ear: float}
        """
        if shape is None or len(shape) < 68:
            return {'is_blinking': False, 'blinks_total': self.total_blinks, 'ear': 0.0}
            
        left_eye = np.array([shape[i] for i in self.left_eye_idxs])
        right_eye = np.array([shape[i] for i in self.right_eye_idxs])
        
        ear_left = self.eye_aspect_ratio(left_eye)
        ear_right = self.eye_aspect_ratio(right_eye)
        
        ear = (ear_left + ear_right) / 2.0
        
        is_blinking = False
        
        if ear < self.thresh:
            self.counter += 1
        else:
            if self.counter >= self.consecutive_frames:
                self.total_blinks += 1
                is_blinking = True
            self.counter = 0
            
        return {
            'is_blinking': is_blinking,
            'blinks_total': self.total_blinks,
            'ear': ear
        }
