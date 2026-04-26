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
    
    def __init__(self, config: VisionConfig = None):
        self.config = config or VisionConfig()
        
        # Khởi tạo các thành phần
        self.detector = YOLODetector(self.config)
        self.tracker = ByteTracker(
            history_len=self.config.TRAJ_MAX_LEN,
            max_age=self.config.TRACK_MAX_AGE
        )
        self.behavior_snn = BehaviorSNN(self.config)
        
        # Mở camera với backend ổn định
        self.cap = cv2.VideoCapture(self.config.CAMERA_ID, self.config.CAMERA_BACKEND)
        
        # Kiểm tra và thử các camera ID khác nếu không mở được
        if not self.cap.isOpened():
            print(f"[VisionModule] WARNING: Cannot open camera ID {self.config.CAMERA_ID}")
            for i in range(5):
                if i == self.config.CAMERA_ID: continue
                self.cap = cv2.VideoCapture(i, self.config.CAMERA_BACKEND)
                if self.cap.isOpened():
                    print(f"[VisionModule] Using camera ID {i} instead")
                    break
        
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
        else:
            print("[VisionModule] ERROR: No camera available")
        
        # Biến FPS tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        print("[VisionModule] Initialized successfully")
    
    def process_video(self) -> Generator[Tuple[np.ndarray, List[dict]], None, None]:
        """
        Generator xử lý video liên tục
        
        Yields:
            frame: Ảnh đã vẽ bounding box
            objects: List các object dict với thông tin đầy đủ
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[VisionModule] Failed to read frame")
                break
            
            start_time = time.time()
            
            # Detect objects
            results = self.detector.detect(frame)
            objects_raw = self.detector.extract_objects(results)
            
            # Update tracker
            self.tracker.update(objects_raw, start_time)
            
            # Get trajectories and classify behaviors
            trajectories = self.tracker.get_all_trajectories()
            behaviors = self.behavior_snn.classify_batch(trajectories)
            
            # Merge behavior info
            objects = []
            for obj in objects_raw:
                obj_id = obj['id']
                if obj_id in behaviors:
                    obj['behavior'] = behaviors[obj_id]
                else:
                    obj['behavior'] = 'unknown'
                objects.append(obj)
            
            # Vẽ bounding box
            annotated_frame = draw_bbox(frame, objects)
            
            # Thêm FPS info
            self._update_fps()
            cv2.putText(annotated_frame, f"FPS: {self.current_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            yield annotated_frame, objects
    
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
        print("[VisionModule] Released camera")
    
    def get_frame(self) -> np.ndarray:
        """Lấy một frame (không qua xử lý)"""
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
