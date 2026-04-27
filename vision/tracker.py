"""
tracker.py - Quản lý ID và trajectory cho các object
"""

import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from .data_structs import ObjectInfo, Trajectory
from .utils import KalmanFilter


class ByteTracker:
    """
    Quản lý theo dõi object và trajectory.
    Hoạt động song song với YOLO tracking.
    """
    
    def __init__(self, history_len: int = 30, max_age: int = 5):
        """
        Khởi tạo tracker
        
        Args:
            history_len: Số điểm trajectory tối đa lưu cho mỗi object
            max_age: Số frame cho phép mất ID trước khi xóa
        """
        self.history_len = history_len
        self.max_age = max_age
        
        # Lưu trajectory cho mỗi object
        self.trajectories: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        
        # Lưu thời gian last seen của mỗi object
        self.last_seen: Dict[int, float] = {}
        
        # Lưu object info đầy đủ
        self.objects: Dict[int, ObjectInfo] = {}
        
        # Lưu Kalman Filters cho mỗi object
        self.kalman_filters: Dict[int, KalmanFilter] = {}
        
    def update(self, objects: List[dict], timestamp: float = None):
        """
        Cập nhật trajectory cho các object hiện tại
        
        Args:
            objects: List từ YOLODetector.extract_objects()
            timestamp: Thời gian hiện tại (giây)
        """
        if timestamp is None:
            timestamp = time.time()
        
        current_ids = set()
        
        for obj in objects:
            obj_id = obj['id']
            if obj_id < 0:
                continue
                
            current_ids.add(obj_id)
            self.last_seen[obj_id] = timestamp
            
            # Tính tâm của bounding box
            x1, y1, x2, y2 = obj['bbox']
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Khởi tạo hoặc cập nhật Kalman Filter
            if obj_id not in self.kalman_filters:
                self.kalman_filters[obj_id] = KalmanFilter()
                # Khởi tạo state với vị trí đầu tiên
                self.kalman_filters[obj_id].X[0] = center[0]
                self.kalman_filters[obj_id].X[1] = center[1]
            
            self.kalman_filters[obj_id].update(center)
            predicted_pos = self.kalman_filters[obj_id].predict()
            
            # Sử dụng vị trí đã được filter cho trajectory (hoặc mix)
            # Ở đây ta ưu tiên filter để giảm rung lắc
            self.trajectories[obj_id].append((float(predicted_pos[0]), float(predicted_pos[1])))
            
            # Giới hạn độ dài lịch sử
            if len(self.trajectories[obj_id]) > self.history_len:
                self.trajectories[obj_id].pop(0)
            
            # Cập nhật hoặc tạo mới ObjectInfo
            if obj_id not in self.objects:
                self.objects[obj_id] = ObjectInfo(
                    id=obj_id,
                    class_name=obj['class_name'],
                    class_id=obj['class_id'],
                    bbox=obj['bbox'],
                    center=center,
                    confidence=obj['confidence']
                )
            else:
                self.objects[obj_id].update(obj['bbox'], obj['confidence'], timestamp)
        
        # Xóa các object không còn trong frame và đã quá hạn
        current_time = timestamp
        to_remove = []
        for obj_id in list(self.objects.keys()):
            if obj_id not in current_ids:
                if current_time - self.last_seen.get(obj_id, 0) > self.max_age:
                    to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.objects[obj_id]
            if obj_id in self.trajectories:
                del self.trajectories[obj_id]
            if obj_id in self.last_seen:
                del self.last_seen[obj_id]
            if obj_id in self.kalman_filters:
                del self.kalman_filters[obj_id]
    
    def get_trajectory(self, obj_id: int) -> List[Tuple[float, float]]:
        """Lấy trajectory của một object"""
        return self.trajectories.get(obj_id, [])
    
    def get_all_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        """Lấy tất cả trajectory hiện tại"""
        return dict(self.trajectories)
    
    def get_object_info(self, obj_id: int) -> Optional[ObjectInfo]:
        """Lấy thông tin đầy đủ của object"""
        return self.objects.get(obj_id)
    
    def get_all_objects(self) -> Dict[int, ObjectInfo]:
        """Lấy tất cả object info hiện tại"""
        return dict(self.objects)
    
    def clear(self):
        """Xóa tất cả dữ liệu tracking"""
        self.trajectories.clear()
        self.last_seen.clear()
        self.objects.clear()
