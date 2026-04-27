"""
config.py - Cấu hình tham số cho vision module
"""

import cv2
import os
from pathlib import Path

# Đường dẫn gốc của dự án
PROJECT_ROOT = Path(__file__).parent.parent

class VisionConfig:
    """Cấu hình cho module thị giác"""
    
    # ========== Chế độ hoạt động ==========
    # Có thể chọn: "safe", "balanced", "agile"
    MODE = "balanced" 
    
    # Profile cho từng chế độ (được áp dụng trong __init__)
    PROFILES = {
        "safe": {      # Chậm, an toàn, ít false positive
            'SPEED_STOPPED': 1.5,
            'SPEED_FAST': 20.0,
            'ANGLE_TURNING': 25.0,
            'YOLO_CONF_THRESH': 0.6,
            'TRACK_MAX_AGE': 10,
        },
        "balanced": {  # Cân bằng (mặc định)
            'SPEED_STOPPED': 2.0,
            'SPEED_FAST': 15.0,
            'ANGLE_TURNING': 30.0,
            'YOLO_CONF_THRESH': 0.5,
            'TRACK_MAX_AGE': 5,
        },
        "agile": {     # NHANH - phản ứng nhanh với vật cản
            'SPEED_STOPPED': 3.0,    # Dễ phát hiện stopped hơn
            'SPEED_FAST': 10.0,      # Phát hiện fast SỚM hơn (ngưỡng thấp hơn)
            'ANGLE_TURNING': 20.0,   # Nhạy hơn với việc chuyển hướng
            'YOLO_CONF_THRESH': 0.35, # Bắt nhiều vật thể hơn (chấp nhận nhiễu nhẹ)
            'TRACK_MAX_AGE': 3,      # Quên nhanh để tránh ID bị nhảy
        }
    }

    def __init__(self, mode: str = None):
        if mode:
            self.MODE = mode
        
        # Load profile tương ứng
        profile = self.PROFILES.get(self.MODE, self.PROFILES["balanced"])
        for key, value in profile.items():
            setattr(self, key, value)

    # ========== Đường dẫn model ==========
    YOLO_MODEL_PATH = str(PROJECT_ROOT / "models" / "yolov8n.pt")
    SNN_MODEL_PATH = str(PROJECT_ROOT / "models" / "snn_behavior.pth")
    
    # ========== Tham số YOLO ==========
    YOLO_CONF_THRESH = 0.5      # Sẽ bị ghi đè bởi profile
    YOLO_IOU_THRESH = 0.45      # Ngưỡng IOU cho NMS
    
    # ========== Tham số tracking ==========
    TRAJ_MAX_LEN = 30           # Tăng lên để dự đoán collision chính xác hơn
    TRACK_MAX_AGE = 5           # Sẽ bị ghi đè bởi profile
    TRACK_MIN_HITS = 3          
    ENABLE_KALMAN = True        # Sử dụng bộ lọc Kalman cho tracking
    
    # ========== Tham số SNN behavior ==========
    SNN_NUM_STEPS = 20          
    SNN_BEHAVIORS = ['normal', 'fast', 'stopped', 'turning']
    TRAJ_FEATURE_DIM = 5        
    
    # ========== Camera ==========
    CAMERA_ID = 0               
    CAMERA_BACKEND = cv2.CAP_ANY 
    FRAME_WIDTH = 320           # Giảm từ 640 xuống để tăng FPS
    FRAME_HEIGHT = 240          # Giảm từ 480 xuống để tăng FPS
    FPS_TARGET = 30             
    
    # ========== Danger Zone (Robot Body) ==========
    ROBOT_WIDTH_PX = 60         # Điều chỉnh theo độ phân giải mới
    ROBOT_HEIGHT_PX = 80        # Điều chỉnh theo độ phân giải mới
    ENABLE_DANGER_ZONE = True   
    
    # ========== Thresholds (Sẽ bị ghi đè bởi profile) ==========
    SPEED_STOPPED = 1.0         # Nhạy hơn ở độ phân giải thấp
    SPEED_FAST = 8.0            
    ANGLE_TURNING = 30.0        
