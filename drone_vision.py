import cv2
import time
import numpy as np
from modules.camera import StereoCamera
from modules.depth import DepthEstimator
from ultralytics import YOLO

# Constants
TARGET_CLASS_ID = 0  # Example: 0 for 'person' (COCO). Change this for your custom trained model.
LOCKED_DISTANCE = 2.0 # Meters to engage payload or stop
FPS_LIMIT = 10 # Cap FPS for thermal management on Pi

def main():
    print("[INFO] Starting Drone Vision System...")
    
    # 1. Initialize Hardware
    # Note: On RPi, use left_id=0, right_id=2 or check /dev/video*
    camera = StereoCamera(left_id=0, right_id=1, width=640, height=480) 
    camera.start()
    
    # 2. Initialize Depth
    depth_estimator = DepthEstimator()
    
    # 3. Load Model (Use TFLite/NCNN on RPi for performance)
    print("[INFO] Loading Model...")
    try:
        # Load exported tflite model if available, else standard pt
        model = YOLO("yolov10n.pt") 
        # model = YOLO("yolov10n_ncnn_model") # Recommended for RPi 4
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    print("[INFO] System Ready. Waiting for acquisition...")
    
    prev_time = 0

    try:
        while True:
            # Frame limiting logic to prevent CPU saturation
            now = time.time()
            elapsed = now - prev_time
            if elapsed < (1.0 / FPS_LIMIT):
                time.sleep((1.0 / FPS_LIMIT) - elapsed)
                now = time.time() # Update time after sleep
            prev_time = now
            
            # Read frames
            ret_l, frame_l, ret_r, frame_r = camera.read()
            if not ret_l or frame_l is None:
                print("[WARN] Lost Left Camera")
                continue
            if not ret_r or frame_r is None:
                # If right camera missing, we lose depth but keep tracking
                pass

            # Detection (Left Frame)
            results = model(frame_l, stream=True, verbose=False)
            
            target_found = False
            target_bbox = None
            
            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls_id == TARGET_CLASS_ID and conf > 0.4:
                        # Found target
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        target_bbox = [x1, y1, x2, y2]
                        target_found = True
                        
                        # Draw bounding box
                        cv2.rectangle(frame_l, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        
                        # Calculate center
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        h, w, _ = frame_l.shape
                        
                        # Flight Logic Error (deviation from center)
                        error_x = cx - (w // 2)
                        error_y = cy - (h // 2)
                        
                        cv2.circle(frame_l, (cx, cy), 5, (0, 0, 255), -1)
                        
                        # Depth Estimation (if Right camera attached)
                        dist_msg = "Dist: N/A"
                        if ret_r and frame_r is not None:
                            distance = depth_estimator.estimate_distance_to_object(frame_l, frame_r, target_bbox)
                            if distance > 0:
                                dist_msg = f"Dist: {distance:.2f}m"
                                if distance < 30.0:
                                     cv2.putText(frame_l, "LOCKING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        cv2.putText(frame_l, f"Target Locked! {dist_msg}", (int(x1), int(y1)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # --- OUTPUT TO FLIGHT CONTROLLER HERE ---
                        # send_mavlink(error_x, error_y, distance)
                        # ----------------------------------------
                        
                        break # Track only primary target
                
                if target_found:
                    break
            
            # Display (Turn off for RPi Headless)
            cv2.imshow("Drone Vision (Left)", frame_l)
            if ret_r and frame_r is not None:
                cv2.imshow("Drone Vision (Right)", frame_r)
            
            # Check for 'q' key OR window close button (X)
            if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty("Drone Vision (Left)", cv2.WND_PROP_VISIBLE) < 1):
                break
                
    except KeyboardInterrupt:
        print("[INFO] Stopping...")
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
