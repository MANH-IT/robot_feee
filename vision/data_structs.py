"""
data_structs.py - Định nghĩa các lớp dữ liệu cho vision module
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import numpy as np


@dataclass
class TrajectoryPoint:
    """Một điểm trên quỹ đạo"""
    x: float
    y: float
    timestamp: float  # thời gian tính bằng giây


@dataclass
class Trajectory:
    """Lịch sử di chuyển của một object"""
    object_id: int
    points: List[TrajectoryPoint] = field(default_factory=list)
    
    def add_point(self, x: float, y: float, timestamp: float):
        """Thêm một điểm vào quỹ đạo"""
        self.points.append(TrajectoryPoint(x, y, timestamp))
    
    def get_positions(self) -> List[Tuple[float, float]]:
        """Lấy danh sách vị trí (x, y)"""
        return [(p.x, p.y) for p in self.points]
    
    def get_velocities(self) -> List[float]:
        """Tính vận tốc tại các điểm (pixel/giây)"""
        velocities = []
        for i in range(1, len(self.points)):
            dx = self.points[i].x - self.points[i-1].x
            dy = self.points[i].y - self.points[i-1].y
            dt = self.points[i].timestamp - self.points[i-1].timestamp
            if dt > 0:
                v = np.sqrt(dx*dx + dy*dy) / dt
                velocities.append(v)
        return velocities
    
    def get_average_velocity(self) -> float:
        """Tính vận tốc trung bình (pixel/giây)"""
        velocities = self.get_velocities()
        return np.mean(velocities) if velocities else 0.0
    
    def get_turning_angles(self) -> List[float]:
        """Tính góc thay đổi hướng tại các điểm (độ)"""
        angles = []
        for i in range(2, len(self.points)):
            v1 = (self.points[i-1].x - self.points[i-2].x,
                  self.points[i-1].y - self.points[i-2].y)
            v2 = (self.points[i].x - self.points[i-1].x,
                  self.points[i].y - self.points[i-1].y)
            
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            norm1 = np.sqrt(v1[0]**2 + v1[1]**2)
            norm2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = max(-1, min(1, dot / (norm1 * norm2)))
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
        return angles
    
    def get_average_turning_angle(self) -> float:
        """Tính góc xoay trung bình (độ)"""
        angles = self.get_turning_angles()
        return np.mean(angles) if angles else 0.0
    
    def get_length(self) -> int:
        """Số điểm trên quỹ đạo"""
        return len(self.points)
    
    def clear_old_points(self, max_points: int = 30):
        """Giữ lại tối đa max_points điểm gần nhất"""
        if len(self.points) > max_points:
            self.points = self.points[-max_points:]


@dataclass
class ObjectInfo:
    """Thông tin của một đối tượng được theo dõi"""
    id: int
    class_name: str
    class_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]               # tâm bounding box
    confidence: float
    trajectory: Trajectory = None
    behavior: str = "unknown"
    last_update: float = 0.0
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = Trajectory(self.id)
    
    def update(self, bbox: Tuple[float, float, float, float], 
               confidence: float, timestamp: float):
        """Cập nhật thông tin mới cho object"""
        self.bbox = bbox
        self.confidence = confidence
        self.last_update = timestamp
        
        # Cập nhật tâm
        x1, y1, x2, y2 = bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Thêm vào trajectory
        self.trajectory.add_point(self.center[0], self.center[1], timestamp)
    
    def to_dict(self) -> dict:
        """Chuyển thành dictionary để gửi qua web socket"""
        return {
            'id': self.id,
            'class': self.class_name,
            'bbox': list(self.bbox),
            'behavior': self.behavior,
            'confidence': self.confidence
        }
