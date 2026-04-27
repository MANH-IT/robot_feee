"""
ui.py - Quản lý giao diện hiển thị Dashboard cho Vision Module
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Tuple
from .utils import draw_text_vn, draw_panel, draw_bbox, draw_trajectory

class VisionUI:
    """Lớp quản lý giao diện Dashboard chuyên nghiệp"""
    
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        
        # Mappings chuyển ngữ
        self.behavior_map = {
            'normal': 'Bình thường',
            'fast': 'Nhanh ⚡',
            'stopped': 'Dừng 🛑',
            'turning': 'Đang rẽ ↩️',
            'unknown': 'Chưa rõ'
        }
        
        self.class_map = {
            'person': 'Người 👤',
            'bicycle': 'Xe đạp 🚲',
            'car': 'Ô tô 🚗',
            'motorbike': 'Xe máy 🏍️',
            'laptop': 'Máy tính 💻',
            'cell phone': 'Điện thoại 📱',
            'bottle': 'Chai nước 🍾'
        }
        
    def render(self, frame: np.ndarray, objects: List[Dict], fps: float, mode: str) -> np.ndarray:
        """Render toàn bộ Dashboard lên frame"""
        h, w = frame.shape[:2]
        
        # 1. Vẽ Bounding Boxes và Trajectories (Lớp nền)
        frame = draw_bbox(frame, objects)
        for obj in objects:
            if 'trajectory' in obj and obj['trajectory']:
                frame = draw_trajectory(frame, obj['trajectory'])
        
        # 2. Panel Thông tin hệ thống (Góc trên trái)
        frame = draw_panel(frame, (10, 10), (220, 110), "HỆ THỐNG")
        frame = draw_text_vn(frame, f"FPS: {fps:.1f}", (20, 40), 16, (0, 255, 0))
        frame = draw_text_vn(frame, f"Vật thể: {len(objects)}", (20, 65), 14, (255, 255, 0))
        frame = draw_text_vn(frame, f"Chế độ: {mode.upper()}", (20, 85), 14, (0, 255, 255))
        
        # 3. Panel Danh sách vật thể (Bên trái)
        if objects:
            y_offset = 120
            panel_h = min(len(objects) * 25 + 40, h - y_offset - 40)
            frame = draw_panel(frame, (10, y_offset), (250, y_offset + panel_h), "DANH SÁCH")
            
            for i, obj in enumerate(objects[:8]): # Hiển thị tối đa 8 vật thể
                if y_offset + 35 + i * 25 > h - 50: break
                
                obj_id = obj.get('id', -1)
                name = self.class_map.get(obj.get('class_name'), obj.get('class_name', '??'))
                behavior = self.behavior_map.get(obj.get('behavior'), '??')
                
                risk = obj.get('risk', {})
                risk_level = risk.get('risk_level', 'none')
                
                color = (255, 255, 255)
                if risk_level == 'high': color = (0, 0, 255)
                elif risk_level == 'medium': color = (0, 165, 255)
                
                text = f"#{obj_id} {name} | {behavior}"
                frame = draw_text_vn(frame, text, (20, y_offset + 35 + i * 25), 13, color)
                
        # 4. Cảnh báo nguy hiểm (Giữa màn hình nếu có High Risk)
        high_risk_count = sum(1 for o in objects if o.get('risk', {}).get('risk_level') == 'high')
        if high_risk_count > 0:
            # Vẽ viền cảnh báo đỏ toàn màn hình
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 3)
            # Banner cảnh báo
            overlay = frame.copy()
            cv2.rectangle(overlay, (w//2 - 150, 10), (w//2 + 150, 45), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            frame = draw_text_vn(frame, f"⚠️ NGUY HIỂM! ({high_risk_count})", (w//2 - 100, 15), 18, (255, 255, 255))

        # 5. Thanh trạng thái dưới cùng
        cv2.rectangle(frame, (0, h-25), (w, h), (20, 20, 20), -1)
        frame = draw_text_vn(frame, "ROBOT FEEE VISION SYSTEM | Nhấn 'q' để thoát", (w//2 - 150, h-20), 12, (150, 150, 150))
        
        return frame
