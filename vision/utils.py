"""
utils.py - Các hàm tiện ích cho vision module
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Optional


def draw_bbox(frame: np.ndarray, objects: List[dict]) -> np.ndarray:
    """
    Vẽ bounding box và thông tin lên frame
    
    Args:
        frame: Ảnh BGR gốc
        objects: List các object dict từ detector/extract_objects
        
    Returns:
        frame: Ảnh đã được vẽ
    """
    # Màu sắc theo behavior
    color_map = {
        'normal': (0, 255, 0),      # xanh lá
        'fast': (0, 165, 255),       # cam
        'stopped': (0, 0, 255),      # đỏ
        'turning': (255, 0, 0),      # xanh dương
        'unknown': (128, 128, 128)   # xám
    }
    
    for obj in objects:
        x1, y1, x2, y2 = [int(v) for v in obj['bbox']]
        class_name = obj.get('class_name', 'unknown')
        obj_id = obj.get('id', -1)
        behavior = obj.get('behavior', 'unknown')
        confidence = obj.get('confidence', 0)
        
        # Chọn màu sắc
        color = color_map.get(behavior, (255, 255, 255))
        
        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Tạo label
        label = f"ID:{obj_id}|{class_name}"
        if behavior != 'unknown':
            label += f"|{behavior}"
        label += f"|{confidence:.2f}"
        
        # Vẽ nền cho text
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
        
        # Vẽ text
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame


def draw_trajectory(frame: np.ndarray, trajectory: List[Tuple[float, float]], 
                    color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """
    Vẽ quỹ đạo (trajectory) lên frame
    
    Args:
        frame: Ảnh BGR
        trajectory: List các điểm (x, y)
        color: Màu vẽ (BGR)
    """
    if len(trajectory) < 2:
        return frame
    
    # Vẽ các điểm
    for i, (x, y) in enumerate(trajectory):
        cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    
    # Vẽ đường nối
    points = np.array([(int(x), int(y)) for x, y in trajectory], dtype=np.int32)
    cv2.polylines(frame, [points], False, color, 2)
    
    return frame


def encode_trajectory_to_spike(trajectory: List[Tuple[float, float]], 
                                num_steps: int = 20) -> np.ndarray:
    """
    Mã hóa trajectory thành spike train (Poisson encoding)
    
    Args:
        trajectory: List các điểm (x, y)
        num_steps: Số timestep
    
    Returns:
        spikes: numpy array shape [num_steps, 5] với các features:
                [dx, dy, speed, angle_change, acceleration]
    """
    if len(trajectory) < 2:
        return np.zeros((num_steps, 5))
    
    # Tính toán features từ trajectory
    features = []
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i-1][0]
        dy = trajectory[i][1] - trajectory[i-1][1]
        speed = np.sqrt(dx*dx + dy*dy)
        
        # Góc thay đổi (nếu có đủ 3 điểm)
        angle_change = 0
        if i >= 2:
            v1 = (trajectory[i-1][0] - trajectory[i-2][0],
                  trajectory[i-1][1] - trajectory[i-2][1])
            v2 = (dx, dy)
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            norm1 = np.sqrt(v1[0]**2 + v1[1]**2)
            norm2 = np.sqrt(v2[0]**2 + v2[1]**2)
            if norm1 > 0 and norm2 > 0:
                cos_angle = max(-1, min(1, dot / (norm1 * norm2)))
                angle_change = np.arccos(cos_angle)
        
        # Gia tốc (thay đổi vận tốc)
        acceleration = 0
        if i >= 2:
            prev_speed = np.sqrt((trajectory[i-1][0] - trajectory[i-2][0])**2 +
                                 (trajectory[i-1][1] - trajectory[i-2][1])**2)
            acceleration = speed - prev_speed
        
        features.append([dx, dy, speed, angle_change, acceleration])
    
    # Interpolate về num_steps
    if len(features) != num_steps:
        # Sử dụng linear interpolation
        indices = np.linspace(0, len(features) - 1, num_steps)
        interpolated = []
        for idx in indices:
            i0 = int(np.floor(idx))
            i1 = min(i0 + 1, len(features) - 1)
            if i0 == i1:
                interpolated.append(features[i0])
            else:
                alpha = idx - i0
                interp = [features[i0][j] * (1 - alpha) + features[i1][j] * alpha 
                         for j in range(5)]
                interpolated.append(interp)
        features = interpolated
    
    # Poisson encoding: chuyển feature thành spike probability
    features = np.array(features)
    max_vals = np.max(np.abs(features), axis=0)
    max_vals[max_vals == 0] = 1
    norm_features = features / max_vals  # normalize to [0, 1]
    norm_features = np.clip(norm_features, 0, 1)
    
    # Tạo spike train
    spikes = np.random.rand(num_steps, 5) < norm_features
    
    return spikes.astype(np.float32)


def resize_frame(frame: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    """Thay đổi kích thước frame"""
    return cv2.resize(frame, (width, height))


def frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Chuyển frame thành JPEG bytes để stream"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()


def get_fps_estimate(start_time: float, frame_count: int) -> float:
    """Tính FPS ước lượng"""
    elapsed = time.time() - start_time
    if elapsed > 0:
        return frame_count / elapsed
    return 0
