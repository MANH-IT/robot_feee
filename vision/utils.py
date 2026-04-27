"""
utils.py - Các hàm tiện ích cho vision module (Phiên bản Premium)
"""

import cv2
import numpy as np
import time
import os
from typing import List, Tuple, Dict, Optional, Any
from PIL import Image, ImageDraw, ImageFont

# Cache cho font để không phải load lại mỗi frame (tăng FPS)
_FONT_CACHE = {}

def get_font(size: int = 18):
    """Lấy font từ cache hoặc load mới"""
    if size not in _FONT_CACHE:
        # Danh sách các đường dẫn font phổ biến trên Linux và Windows
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "C:\\Windows\\Fonts\\arial.ttf" 
        ]
        font = None
        for path in font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except:
                    continue
        if font is None:
            font = ImageFont.load_default()
        _FONT_CACHE[size] = font
    return _FONT_CACHE[size]

def draw_text_vn(frame: np.ndarray, text: str, pos: Tuple[int, int], 
                 size: int = 18, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Vẽ tiếng Việt lên frame OpenCV dùng Pillow (Đã tối ưu cache)"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font = get_font(size)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_panel(frame: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], 
               title: str = "", alpha: float = 0.6) -> np.ndarray:
    """Vẽ một panel bán trong suốt với tiêu đề kiểu hiện đại"""
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Bo viền mỏng
    cv2.rectangle(frame, pt1, pt2, (150, 150, 150), 1)
    
    if title:
        frame = draw_text_vn(frame, title, (pt1[0] + 10, pt1[1] + 5), 14, (0, 255, 255))
        cv2.line(frame, (pt1[0], pt1[1] + 28), (pt2[0], pt1[1] + 28), (100, 100, 100), 1)
        
    return frame

def draw_bbox(frame: np.ndarray, objects: List[dict]) -> np.ndarray:
    """Vẽ bounding box nghệ thuật hơn"""
    color_map = {
        'normal': (0, 255, 0),      # xanh lá
        'fast': (0, 165, 255),       # cam
        'stopped': (0, 0, 255),      # đỏ
        'turning': (255, 0, 0),      # xanh dương
        'unknown': (128, 128, 128)   # xám
    }
    
    for obj in objects:
        bbox = obj.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in bbox]
        behavior = obj.get('behavior', 'unknown')
        color = color_map.get(behavior, (255, 255, 255))
        
        # Vẽ các góc (Corner bracket style)
        d = 15
        t = 2 # thickness
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + d, y1), color, t)
        cv2.line(frame, (x1, y1), (x1, y1 + d), color, t)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - d, y1), color, t)
        cv2.line(frame, (x2, y1), (x2, y1 + d), color, t)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + d, y2), color, t)
        cv2.line(frame, (x1, y2), (x1, y2 - d), color, t)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - d, y2), color, t)
        cv2.line(frame, (x2, y2), (x2, y2 - d), color, t)
        
        # Thêm bóng mờ nhẹ bên trong box
        if x2 > x1 and y2 > y1:
            overlay = frame[y1:y2, x1:x2].copy()
            cv2.rectangle(overlay, (0, 0), (x2-x1, y2-y1), color, -1)
            frame[y1:y2, x1:x2] = cv2.addWeighted(overlay, 0.1, frame[y1:y2, x1:x2], 0.9, 0)

    return frame

def draw_trajectory(frame: np.ndarray, trajectory: List[Tuple[float, float]], 
                    color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """Vẽ quỹ đạo với hiệu ứng gradient mờ dần"""
    if len(trajectory) < 2:
        return frame
    
    for i in range(1, len(trajectory)):
        # Tính toán độ đậm nhạt theo thời gian
        alpha = i / len(trajectory)
        curr_color = tuple([int(c * alpha) for c in color])
        
        p1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
        p2 = (int(trajectory[i][0]), int(trajectory[i][1]))
        cv2.line(frame, p1, p2, curr_color, 1 + int(2 * alpha))
        
    return frame

def encode_trajectory_to_spike(trajectory: List[Tuple[float, float]], 
                                num_steps: int = 20, 
                                method: str = 'poisson') -> np.ndarray:
    """Mã hóa trajectory thành spike train"""
    if len(trajectory) < 2:
        return np.zeros((num_steps, 5))
    
    features = []
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i-1][0]
        dy = trajectory[i][1] - trajectory[i-1][1]
        speed = np.sqrt(dx*dx + dy*dy)
        
        angle_change = 0
        if i >= 2:
            v1 = (trajectory[i-1][0] - trajectory[i-2][0], trajectory[i-1][1] - trajectory[i-2][1])
            v2 = (dx, dy)
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            norm1 = np.sqrt(v1[0]**2 + v1[1]**2)
            norm2 = np.sqrt(v2[0]**2 + v2[1]**2)
            if norm1 > 0 and norm2 > 0:
                cos_angle = max(-1, min(1, dot / (norm1 * norm2)))
                angle_change = np.arccos(cos_angle)
        
        acceleration = 0
        if i >= 2:
            prev_speed = np.sqrt((trajectory[i-1][0] - trajectory[i-2][0])**2 + (trajectory[i-1][1] - trajectory[i-2][1])**2)
            acceleration = speed - prev_speed
        
        features.append([dx, dy, speed, angle_change, acceleration])
    
    if len(features) != num_steps:
        indices = np.linspace(0, len(features) - 1, num_steps)
        interpolated = []
        for idx in indices:
            i0 = int(np.floor(idx))
            i1 = min(i0 + 1, len(features) - 1)
            alpha = idx - i0
            interp = [features[i0][j] * (1 - alpha) + features[i1][j] * alpha for j in range(5)]
            interpolated.append(interp)
        features = np.array(interpolated)
    else:
        features = np.array(features)

    if method == 'delta':
        threshold = 0.1
        spikes = np.zeros_like(features)
        prev_val = features[0]
        for t in range(1, num_steps):
            diff = features[t] - prev_val
            spikes[t] = np.abs(diff) > threshold
            prev_val[spikes[t] > 0] = features[t][spikes[t] > 0]
        return spikes.astype(np.float32)
    else:
        max_vals = np.max(np.abs(features), axis=0)
        max_vals[max_vals == 0] = 1
        norm_features = np.clip(features / max_vals, 0, 1)
        spikes = np.random.rand(num_steps, 5) < norm_features
        return spikes.astype(np.float32)

class KalmanFilter:
    """Bộ lọc Kalman đơn giản cho tracking 2D"""
    def __init__(self, dt: float = 1.0/20, process_noise: float = 0.1, measurement_noise: float = 0.5):
        self.dt = dt
        self.X = np.zeros((4, 1))
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.P = np.eye(4) * 1000.0
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        
    def predict(self):
        self.X = np.dot(self.F, self.X)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.X[:2].flatten()
        
    def update(self, measurement: Tuple[float, float]):
        Z = np.array([[measurement[0]], [measurement[1]]])
        Y = Z - np.dot(self.H, self.X)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.X = self.X + np.dot(K, Y)
        self.P = np.dot(np.eye(4) - np.dot(K, self.H), self.P)

def resize_frame(frame: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    return cv2.resize(frame, (width, height))

def frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()

def get_fps_estimate(start_time: float, frame_count: int) -> float:
    elapsed = time.time() - start_time
    return frame_count / elapsed if elapsed > 0 else 0
