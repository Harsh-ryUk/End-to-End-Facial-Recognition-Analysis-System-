import cv2
import numpy as np

def draw_results(frame, faces, landmarks_list, poses, names, scores, fps, latency):
    """
    Draw all overlays on the frame.
    """
    # Draw FPS/Latency
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Latency: {latency:.1f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    for i, face in enumerate(faces):
        bbox = face[:4].astype(int)
        score = face[4]
        name = names[i]
        match_score = scores[i]
        
        # BBox
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"{name} ({match_score:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Landmarks (68 points)
        if landmarks_list is not None and len(landmarks_list) > i:
            lms = landmarks_list[i]
            for (lx, ly) in lms:
                cv2.circle(frame, (lx, ly), 1, (255, 255, 0), -1)
                
        # Head Pose Axis
        if poses is not None and len(poses) > i and poses[i] is not None:
            pitch, yaw, roll = poses[i]
            draw_axis(frame, yaw, pitch, roll, tdx=(x1+x2)/2, tdy=(y1+y2)/2, size=50)

    return frame

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
        # Referenced from standard Head Pose viz
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    # Z-Axis (out of screen) drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img
