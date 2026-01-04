import cv2
import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline.realtime_pipeline import RealTimePipeline

def main():
    parser = argparse.ArgumentParser(description="Real-Time Facial Analysis System")
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--det_model', type=str, default='models/scrfd_10g_bnkps.onnx', help='Path to SCRFD ONNX model')
    parser.add_argument('--rec_model', type=str, default='models/w600k_r50.onnx', help='Path to ArcFace ONNX model')
    parser.add_argument('--land_model', type=str, default='models/shape_predictor_68_face_landmarks.dat', help='Path to Dlib Landmark DAT')
    parser.add_argument('--db_path', type=str, default='models/face_db.pkl', help='Path to Face Database')
    
    args = parser.parse_args()
    
    # Check models
    if not os.path.exists(args.det_model) or not os.path.exists(args.rec_model) or not os.path.exists(args.land_model):
        print("ERROR: Models not found! Please run 'python download_models.py' first.")
        return

    config = {
        'det_model': args.det_model,
        'rec_model': args.rec_model,
        'land_model': args.land_model,
        'db_path': args.db_path
    }
    
    pipeline = RealTimePipeline(config)
    
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Input State Variables
    input_mode = False
    input_text = ""
    pending_embedding = None # Store embedding while typing
    
    print("Starting Video Stream... Press 'q' to exit, 'r' to register.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        vis_frame = pipeline.process_frame(frame)
        
        # --- DRAW INPUT UI IF ACTIVE ---
        if input_mode:
            # Darken invalid area
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 480), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, vis_frame, 0.5, 0, vis_frame)
            
            # Draw Input Box
            h, w, _ = vis_frame.shape
            cx, cy = w // 2, h // 2
            
            cv2.rectangle(vis_frame, (cx - 150, cy - 40), (cx + 150, cy + 40), (255, 255, 255), -1)
            cv2.rectangle(vis_frame, (cx - 150, cy - 40), (cx + 150, cy + 40), (0, 0, 0), 2)
            
            # Text
            text_size = cv2.getTextSize(input_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2
            
            cv2.putText(vis_frame, input_text + "|", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(vis_frame, "Enter Name:", (cx - 150, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, "[Enter] Save  [Esc] Cancel", (cx - 120, cy + 70), cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 200, 200), 1)

        cv2.imshow("Advanced Face System", vis_frame)
        
        if cv2.getWindowProperty("Advanced Face System", cv2.WND_PROP_VISIBLE) < 1:
            break
            
        key = cv2.waitKey(1) & 0xFF
        
        if input_mode:
            # --- INPUT HANDLING ---
            if key == 13: # Enter
                if input_text.strip() and pending_embedding is not None:
                    pipeline.matcher.register(input_text.strip(), pending_embedding)
                    print(f"[INFO] Registered: {input_text}")
                input_mode = False
                input_text = ""
                pending_embedding = None
                
            elif key == 27: # Esc
                input_mode = False
                input_text = ""
                pending_embedding = None
                print("[INFO] Registration Cancelled")
                
            elif key == 8: # Backspace
                input_text = input_text[:-1]
                
            elif 32 <= key <= 126: # Printable chars
                if len(input_text) < 20: # Limit length
                    input_text += chr(key)
                    
        else:
            # --- NORMAL MODE HANDLING ---
            if key == ord('q') or key == ord('Q') or key == 27: # q, Q, or Esc
                break
            elif key == ord('r'):
                 if pipeline.last_frame_data and pipeline.last_frame_data['embeddings']:
                     emb = pipeline.last_frame_data['embeddings'][0]
                     
                     # Check if already registered
                     existing_name, score = pipeline.matcher.match(emb)
                     if existing_name != "Unknown":
                          print(f"\n[INFO] Already registered as '{existing_name}'")
                     else:
                         print("[INFO] Enter name on screen...")
                         input_mode = True
                         input_text = ""
                         pending_embedding = emb
                 else:
                     print("[INFO] No face detected to register!")
            
            elif key == ord('d'):
                if pipeline.last_results:
                    names = pipeline.last_results[3]
                    if names:
                        target_name = names[0]
                        if target_name != "Unknown":
                            pipeline.matcher.delete(target_name)
                            pipeline.last_results = None 
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
