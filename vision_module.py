"""
vision_module.py - Giao diện chính cho vision module
Sử dụng các class từ package vision/
"""

import time
import cv2
import numpy as np
from typing import Generator, Tuple, List, Dict

from vision import (
    VisionConfig,
    YOLODetector,
    ByteTracker,
    BehaviorSNN,
    draw_bbox
)


class VisionModule:
    """Module thị giác tổng hợp cho robot"""
    
    def __init__(self, config: VisionConfig = None, mode: str = "balanced"):
        """
        Khởi tạo VisionModule
        
        Args:
            config: Cấu hình vision
            mode: Chế độ "safe", "balanced", hoặc "agile"
        """
        # Nếu đã truyền config thì dùng config, nếu không tạo mới với mode tương ứng
        if config:
            self.config = config
        else:
            self.config = VisionConfig(mode=mode)
            
        # Khởi tạo các thành phần
        self.detector = YOLODetector(self.config)
        self.tracker = ByteTracker(
            history_len=self.config.TRAJ_MAX_LEN,
            max_age=self.config.TRACK_MAX_AGE
        )
        self.behavior_snn = BehaviorSNN(self.config)
        
        # Thử mở camera với nhiều index và backend khác nhau
        self.cap = None
        indices_to_try = [self.config.CAMERA_ID, 0, 1, 2]
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        
        for backend in backends:
            for idx in indices_to_try:
                print(f"[ThịGiác] Thử mở Camera ID {idx} với backend {backend}...")
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    # Thử đọc vài frame để "mồi" camera
                    for _ in range(5):
                        ret, _ = cap.read()
                        if ret: break
                    
                    if ret:
                        self.cap = cap
                        print(f"[ThịGiác] KẾT NỐI THÀNH CÔNG Camera ID {idx}")
                        break
                    else:
                        cap.release()
            if self.cap: break
            
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
        else:
            print("[ThịGiác] LỖI: Không thể truy cập Camera phần cứng. Vui lòng kiểm tra kết nối hoặc quyền hạn.")
            # Khởi tạo VideoCapture rỗng để tránh crash
            self.cap = cv2.VideoCapture() 
        
        # Biến FPS tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Trạng thái log để tránh flooding
        self.last_risk_log_time = {} # obj_id -> timestamp
        
        print(f"[ThịGiác] Khởi tạo thành công ở chế độ '{self.config.MODE}'")
    
    def process_video(self) -> Generator[Tuple[np.ndarray, List[dict]], None, None]:
        """
        Generator xử lý video liên tục. 
        Nếu không có camera thật, sẽ tự động chuyển sang 'Simulation Mode' (giả lập).
        """
        is_simulation = not self.cap.isOpened()
        if is_simulation:
            print("[ThịGiác] Đang chạy ở chế độ GIẢ LẬP (Simulation Mode)")
            
        while True:
            ret, frame = False, None
            if not is_simulation:
                ret, frame = self.cap.read()
            
            # Nếu camera hỏng hoặc đang giả lập
            if not ret or frame is None:
                # Tạo frame giả lập (đen/xám)
                frame = np.zeros((self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.rectangle(frame, (0, 0), (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT), (20, 20, 20), -1)
                
                # Thêm hiệu ứng lưới (grid) cho chuyên nghiệp
                for i in range(0, self.config.FRAME_WIDTH, 40):
                    cv2.line(frame, (i, 0), (i, self.config.FRAME_HEIGHT), (40, 40, 40), 1)
                for i in range(0, self.config.FRAME_HEIGHT, 40):
                    cv2.line(frame, (0, i), (self.config.FRAME_WIDTH, i), (40, 40, 40), 1)
                
                # Tạo vật thể giả lập di chuyển
                t = time.time()
                obj_x = int(self.config.FRAME_WIDTH / 2 + np.cos(t) * 100)
                obj_y = int(self.config.FRAME_HEIGHT / 2 + np.sin(t * 1.5) * 50)
                
                # Vẽ một vòng tròn đại diện vật thể giả lập
                cv2.circle(frame, (obj_x, obj_y), 20, (0, 255, 255), 2)
                cv2.putText(frame, "SIMULATION", (10, self.config.FRAME_HEIGHT - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Giả lập dữ liệu object cho tracker
                objects_raw = [{
                    'id': 99,
                    'bbox': [obj_x - 20, obj_y - 20, obj_x + 20, obj_y + 20],
                    'confidence': 0.95,
                    'class_id': 0,
                    'class_name': 'person'
                }]
                timestamp = t
            else:
                timestamp = time.time()
                # 1. Phát hiện vật thể thật
                results = self.detector.detect(frame)
                objects_raw = self.detector.extract_objects(results)
            
            # 2. Cập nhật tracker
            self.tracker.update(objects_raw, timestamp)
            
            # 3. Phân tích hành vi và rủi ro
            trajectories = self.tracker.get_all_trajectories()
            behaviors = self.behavior_snn.classify_batch(trajectories)
            
            processed_objects = []
            for obj in objects_raw:
                obj_id = obj['id']
                obj['behavior'] = behaviors.get(obj_id, 'unknown')
                
                if obj_id in trajectories:
                    risk_info = self.behavior_snn.detect_collision_risk(trajectories[obj_id])
                    obj['risk'] = risk_info
                
                processed_objects.append(obj)
            
            yield frame, processed_objects
            
            # Giới hạn tốc độ giả lập
            if is_simulation:
                time.sleep(0.03) # ~30 FPS
    
    def _update_fps(self):
        """Cập nhật FPS"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def release(self):
        """Giải phóng camera"""
        self.cap.release()
        print("[ThịGiác] Đã giải phóng camera")
    
    def get_frame(self) -> np.ndarray:
        """Lấy một frame (không qua xử lý)"""
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
