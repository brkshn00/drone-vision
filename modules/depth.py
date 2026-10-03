import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, baseline=0.06, focal_length=700):
        # baseline: Distance between cameras in meters
        # focal_length: Focal length in pixels (depends on camera calibration)
        self.baseline = baseline
        self.focal_length = focal_length
        self.stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
        
        # Optimization: WLS Filter for smoother disparity (Optional, costs FPS)
        # self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo)
        
    def compute_depth(self, left_img, right_img, roi_rect=None):
        """
        Compute depth map or ROI depth.
        roi_rect: (x, y, w, h) - if provided, only checks this area to save CPU.
        """
        # Convert to grayscale
        gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        # Note: Compute full image disparity for now as ROI matching is complex without rectification
        # For true optimization, we would crop BEFORE disparity, but that requires perfect rectification.
        disparity = self.stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
        
        if roi_rect:
            x, y, w, h = roi_rect
            # Safety checks
            h_img, w_img = disparity.shape
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            
            roi_disp = disparity[y:y+h, x:x+w]
            
            # Simple average or median of valid disparities (disparity > 0)
            valid_disp = roi_disp[roi_disp > 0]
            
            if len(valid_disp) > 0:
                mean_disp = np.median(valid_disp)
                if mean_disp > 0:
                    depth = (self.focal_length * self.baseline) / mean_disp
                    return depth
            return -1.0 # Invalid
            
        return disparity

    def estimate_distance_to_object(self, left_frame, right_frame, bbox):
        """
        Detects object distance based on bounding box from YOLO.
        bbox: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = map(int, bbox)
        w = x2 - x1
        h = y2 - y1
        roi = (x1, y1, w, h)
        
        return self.compute_depth(left_frame, right_frame, roi_rect=roi)
