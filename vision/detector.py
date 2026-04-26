"""
detector.py - YOLO detector (load model, predict)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from .config import VisionConfig


class YOLODetector:
    """YOLO object detector với tracking"""
    
    def __init__(self, config: VisionConfig = None):
        """
        Khởi tạo detector
        
        Args:
            config: Cấu hình vision module
        """
        self.config = config or VisionConfig()
        
        # Load YOLO model
        print(f"[YOLO] Loading model from {self.config.YOLO_MODEL_PATH}")
        self.model = YOLO(self.config.YOLO_MODEL_PATH)
        print("[YOLO] Model loaded successfully")
        
        # Lấy danh sách class names
        self.class_names = self.model.names
        
        # Biến lưu tracking state
        self.track_history = {}
        
    def detect(self, frame: np.ndarray, track: bool = True):
        """
        Phát hiện object trong frame
        
        Args:
            frame: Ảnh BGR từ camera
            track: Có sử dụng tracking hay không
            
        Returns:
            results: Kết quả từ YOLO
        """
        if track:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.config.YOLO_CONF_THRESH,
                iou=self.config.YOLO_IOU_THRESH,
                verbose=False
            )
        else:
            results = self.model(
                frame,
                conf=self.config.YOLO_CONF_THRESH,
                iou=self.config.YOLO_IOU_THRESH,
                verbose=False
            )
        return results
    
    def extract_objects(self, results) -> list:
        """
        Trích xuất thông tin object từ kết quả YOLO
        
        Args:
            results: Kết quả từ model.predict() hoặc model.track()
            
        Returns:
            List các dict chứa thông tin object
        """
        objects = []
        
        if not results or results[0].boxes is None:
            return objects
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
        # Lấy ID nếu có tracking
        ids = None
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
            obj_id = ids[i] if ids is not None else -1
            objects.append({
                'id': int(obj_id),
                'bbox': box.tolist(),
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': self.class_names[int(cls)]
            })
        
        return objects
    
    def get_class_name(self, class_id: int) -> str:
        """Lấy tên class từ ID"""
        return self.class_names.get(class_id, "unknown")
    
    def get_annotated_frame(self, results) -> np.ndarray:
        """Lấy frame đã được vẽ bounding box"""
        if results and len(results) > 0:
            return results[0].plot()
        return None
