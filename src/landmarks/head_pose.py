import cv2
import numpy as np

class HeadPoseEstimator:
    """
    Head Pose Estimation using PnP with 68-point landmarks.
    """
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        # Generic 3D model points (standard for 68-point Dlib model)
        # Using 6 key points: Nose tip, Chin, Left Eye Left Corner, Right Eye Right Corner, Left Mouth Corner, Right Mouth Corner
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (30)
            (0.0, -330.0, -65.0),        # Chin (8)
            (-225.0, 170.0, -135.0),     # Left eye left corner (36)
            (225.0, 170.0, -135.0),      # Right eye right corner (45)
            (-150.0, -150.0, -125.0),    # Left Mouth corner (48)
            (150.0, -150.0, -125.0)      # Right mouth corner (54)
        ])
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        if self.camera_matrix is None:
            # Approximate camera matrix if not provided
            # Focal length typically image width
            self.focal_length = 640 
            self.center = (320, 240)
            self.camera_matrix = np.array(
                [[self.focal_length, 0, self.center[0]],
                 [0, self.focal_length, self.center[1]],
                 [0, 0, 1]], dtype="double"
            )
            
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion

    def estimate(self, image, landmarks):
        """
        Estimate Yaw, Pitch, Roll.
        
        Args:
            image (np.ndarray): Image (used for size if cam matrix needs update).
            landmarks (np.ndarray): 68x2 landmarks.
            
        Returns:
            tuple: (pitch, yaw, roll) in degrees.
        """
        if landmarks is None or len(landmarks) != 68:
            return None
        
        # Update camera matrix if image size changes significantly (optional)
        if image is not None:
            h, w = image.shape[:2]
            self.center = (w/2, h/2)
            self.camera_matrix[0, 2] = self.center[0]
            self.camera_matrix[1, 2] = self.center[1]
        
        # 2D image points
        image_points = np.array([
            landmarks[30],     # Nose tip
            landmarks[8],      # Chin
            landmarks[36],     # Left eye left corner
            landmarks[45],     # Right eye right corner
            landmarks[48],     # Left Mouth corner
            landmarks[54]      # Right mouth corner
        ], dtype="double")
        
        # Solve PnP
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, 
            image_points, 
            self.camera_matrix, 
            self.dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
            
        # Project a 3D point (nose) to see if it works (optional for vis)
        # (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, self.camera_matrix, self.dist_coeffs)
        
        # Calculate Euler angles
        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rmat, translation_vector))
        
        # Decompose projection matrix
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        
        pitch, yaw, roll = [math.squeeze() for math in euler_angles]
        
        # Adjust signs/offsets to match standard conventions if needed
        return pitch, yaw, roll # Degrees
