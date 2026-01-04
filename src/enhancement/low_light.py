import cv2
import numpy as np

class LowLightEnhancer:
    """
    Implements Contrast Limited Adaptive Histogram Equalization (CLAHE)
    for low-light image enhancement.
    """
    
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        """
        Initialize CLAHE Enhancer.
        
        Args:
            clip_limit (float): Threshold for contrast limiting.
            tile_grid_size (tuple): Size of grid for histogram equalization.
        """
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the Value channel of HSV (or LAB) color space
        to enhance brightness/contrast without shifting colors.
        
        Args:
            image (np.ndarray): BGR image.
        
        Returns:
            np.ndarray: Enhanced BGR image.
        """
        if image is None:
            return None
        
        # Convert to LAB color space (L channel is lightness)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        l_enhanced = self.clahe.apply(l)
        
        # Merge and convert back to BGR
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Denoising (Bilateral Filter)
        # Removes noise ("fake wrinkles") while keeping edges sharp
        enhanced_image = cv2.bilateralFilter(enhanced_image, d=5, sigmaColor=50, sigmaSpace=50)
        
        return enhanced_image
