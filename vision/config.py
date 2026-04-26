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
    
    # ========== Đường dẫn model ==========
    YOLO_MODEL_PATH = str(PROJECT_ROOT / "models" / "yolov8n.pt")  # hoặc yolo11n.pt
    SNN_MODEL_PATH = str(PROJECT_ROOT / "models" / "snn_behavior.pth")
    
    # ========== Tham số YOLO ==========
    YOLO_CONF_THRESH = 0.5      # Ngưỡng confidence
    YOLO_IOU_THRESH = 0.45      # Ngưỡng IOU cho NMS
    
    # ========== Tham số tracking ==========
    TRACK_HISTORY_LEN = 30      # Số điểm lưu tối đa cho mỗi object
    TRACK_MAX_AGE = 5           # Số frame cho phép mất ID
    TRACK_MIN_HITS = 3          # Số frame tối thiểu để tạo ID mới
    
    # ========== Tham số SNN behavior ==========
    SNN_NUM_STEPS = 20          # Số timestep cho spike encoding
    SNN_BEHAVIORS = ['normal', 'fast', 'stopped', 'turning']
    
    # ========== Tham số trajectory encoding ==========
    TRAJ_MAX_LEN = 20           # Số điểm quỹ đạo lưu (1 giây @20fps)
    TRAJ_FEATURE_DIM = 5        # [dx, dy, speed, angle_change, acceleration]
    
    # ========== Camera ==========
    CAMERA_ID = 0               # ID camera (0: webcam mặc định)
    CAMERA_BACKEND = cv2.CAP_DSHOW # Backend (DSHOW ổn định hơn trên Windows)
    FRAME_WIDTH = 640           # Chiều rộng khung hình
    FRAME_HEIGHT = 480          # Chiều cao khung hình
    FPS_TARGET = 20             # FPS mục tiêu
    
    # ========== Thresholds cho behavior ==========
    SPEED_STOPPED = 2.0         # Vận tốc < 2 pixel/frame -> stopped
    SPEED_FAST = 15.0           # Vận tốc > 15 pixel/frame -> fast
    ANGLE_TURNING = 30.0        # Góc thay đổi > 30 độ -> turning
