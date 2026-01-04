import cv2
import numpy as np
import onnxruntime
from ..utils.helpers import face_alignment

class ArcFaceEmbedder:
    """
    ArcFace: Deep Face Embedding
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.input_size = (112, 112)
        self.session = onnxruntime.InferenceSession(self.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def get_embedding(self, image, kps):
        """
        Get normalized 512-D face embedding
        """
        if image is None or kps is None:
            return None
            
        # Align face using 5-point landmarks (kps) from SCRFD
        aligned_face = face_alignment(image, kps)
        
        # Preprocess
        # ArcFace expects RGB, [0,255], usually normalized by subtracting 127.5 and dividing by 127.5
        # Input shape: (1, 3, 112, 112)
        blob = cv2.dnn.blobFromImage(aligned_face, 1.0/127.5, self.input_size, (127.5, 127.5, 127.5), swapRB=True)
        
        embedding = self.session.run([self.output_name], {self.input_name: blob})[0]
        embedding = embedding.flatten()
        
        # L2 Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
            
        return embedding
