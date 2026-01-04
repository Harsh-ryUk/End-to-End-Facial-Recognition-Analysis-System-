import cv2
import numpy as np
import onnxruntime

class AttributeEstimator:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (96, 96) # Standard for InsightFace GenderAge

    def estimate(self, frame, face_bbox):
        """
        Estimate Age and Gender for a face.
        
        Args:
            frame: Full BGR image.
            face_bbox: [x1, y1, x2, y2]
            
        Returns:
            dict: {gender: str, age: int} with 'Male'/'Female'
        """
        x1, y1, x2, y2 = map(int, face_bbox)
        
        # Padding to capture full head often helps
        h, w, _ = frame.shape
        pad_x = max(0, int((x2-x1)*0.1))
        pad_y = max(0, int((y2-y1)*0.1))
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
             return {'gender': 'Unknown', 'age': 0}

        # Preprocessing
        blob = cv2.resize(face_img, self.input_shape)
        blob = blob.transpose(2, 0, 1) # HWC -> CHW
        blob = np.expand_dims(blob, axis=0).astype(np.float32)
        
        # Standardize (if needed, usually InsightFace just takes RGB/BGR raw or standard norm)
        # InsightFace models often expect RGB, and input is BGR from cv2
        # But this specific model works reasonably with standard normalization
        
        outputs = self.session.run(None, {self.input_name: blob})
        # Output format depends on specific ONNX export, usually [gender, age] or similar
        # For standard antelopev2 genderage: 
        # Output 0: Gender/Age combined or separate. 
        # Usually it returns a (1, 3) or similar vector.
        # Let's assume standard InsightFace output: [1, 2, 1, 1]? 
        # Actually for 'genderage.onnx' in antelopev2:
        # returns [gender_prob, age]
        
        ret = outputs[0][0]
        gender = "Male" if ret[0:2].argmax() == 1 else "Female"
        age = int(ret[2] * 100)
        
        return {'gender': gender, 'age': age}
