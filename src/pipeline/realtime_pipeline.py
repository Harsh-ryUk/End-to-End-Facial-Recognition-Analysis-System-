import cv2
import numpy as np
from ..enhancement.low_light import LowLightEnhancer
from ..detector.face_detector import SCRFD
from ..recognition.arcface_embedder import ArcFaceEmbedder
from ..recognition.reid_matcher import ReidMatcher
from ..landmarks.landmarks import LandmarkDetector
from ..landmarks.head_pose import HeadPoseEstimator
from ..utils.timing import FPSMeter, LatencyTracker
from ..utils.visualizer import draw_results

from ..liveness.blink_detector import BlinkDetector
from ..analysis.attribute_estimator import AttributeEstimator

class RealTimePipeline:
    def __init__(self, config):
        print("Initializing Pipeline...")
        self.enhancer = LowLightEnhancer()
        print("Loading Detector...")
        self.detector = SCRFD(config['det_model'])
        print("Loading Recognizer...")
        self.recognizer = ArcFaceEmbedder(config['rec_model'])
        print("Loading Matcher...")
        self.matcher = ReidMatcher(config['db_path'])
        print("Loading Look-at-Dat (Landmarks)...")
        self.landmark_detector = LandmarkDetector(config['land_model'])
        self.head_pose = HeadPoseEstimator()
        
        # New Liveness & Analytics
        self.blink_detector = BlinkDetector()
        self.attr_estimator = AttributeEstimator('models/genderage.onnx')
        
        self.fps_meter = FPSMeter()
        self.tracker = LatencyTracker()
        
        self.last_frame_data = None 
        
        self.frame_count = 0
        self.last_results = None # (faces, landmarks, poses, names, scores, blinks, attributes)
        
    def process_frame(self, frame):
        self.tracker.start('total')
        self.frame_count += 1
        
        # 1. Enhancement (Enabled)
        self.tracker.start('enhance')
        enhanced_frame = self.enhancer.enhance(frame)
        self.tracker.end('enhance')
        
        if self.frame_count % 5 == 0 or self.last_results is None:
            # Full Inference
            self.tracker.start('detect')
            faces, kpss = self.detector.detect(enhanced_frame, max_num=10)
            self.tracker.end('detect')
            
            names = []
            scores = []
            landmarks_detailed = [] 
            poses = []
            blinks = [] 
            attributes = [] # New
            current_embeddings = []
            
            if faces is not None:
                for i, face in enumerate(faces):
                    kps_5 = kpss[i] if kpss is not None else None
                    
                    self.tracker.start('recog')
                    embedding = self.recognizer.get_embedding(enhanced_frame, kps_5)
                    name, score = self.matcher.match(embedding)
                    names.append(name)
                    scores.append(score)
                    self.tracker.end('recog')
                    
                    current_embeddings.append(embedding)
                    
                    self.tracker.start('pose')
                    lms_68 = self.landmark_detector.detect(enhanced_frame, face[:4])
                    landmarks_detailed.append(lms_68)
                    
                    pose = self.head_pose.estimate(enhanced_frame, lms_68)
                    poses.append(pose)
                    self.tracker.end('pose')
                    
                    # Liveness Check
                    blink_data = self.blink_detector.detect(lms_68)
                    blinks.append(blink_data)
                    
                    # Attribute Check
                    attr = self.attr_estimator.estimate(enhanced_frame, face[:4])
                    attributes.append(attr)
            
            self.last_results = (faces, landmarks_detailed, poses, names, scores, blinks, attributes)
            
            self.last_frame_data = {
                'embeddings': current_embeddings,
                'faces': faces
            }
            
        else:
            if self.last_results:
                faces, landmarks_detailed, poses, names, scores, blinks, attributes = self.last_results
            else:
                faces, landmarks_detailed, poses, names, scores, blinks, attributes = [], [], [], [], [], [], []
        
        fps, latency = self.fps_meter.update()
        total_lat = self.tracker.end('total')
        
        vis_frame = draw_results(enhanced_frame, faces if faces is not None else [], landmarks_detailed, poses, names, scores, fps, total_lat, blinks, attributes)
        
        return vis_frame
