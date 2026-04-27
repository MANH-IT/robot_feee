"""
Cấu hình tối ưu cực độ cho CPU - Max FPS
"""
from vision.config import VisionConfig

class FastVisionConfig(VisionConfig):
    # Giảm độ phân giải tối đa để CPU gánh được YOLO
    FRAME_WIDTH = 320
    FRAME_HEIGHT = 240
    
    # Tối ưu tracking
    TRAJ_MAX_LEN = 10
    ENABLE_KALMAN = False # Tắt Kalman để tiết kiệm CPU
    
    # Giảm kích thước vùng nguy hiểm tương ứng
    ROBOT_WIDTH_PX = 50
    ROBOT_HEIGHT_PX = 70

print("🚀 Đã nạp cấu hình SIÊU TỐC (CPU Optimized)")
