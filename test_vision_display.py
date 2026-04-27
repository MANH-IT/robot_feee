#!/usr/bin/env python3
"""
HỆ THỐNG KIỂM TRA THỊ GIÁC ROBOT (PHIÊN BẢN CHUẨN & ĐẸP)
- Tách biệt logic xử lý và hiển thị (MVC pattern)
- Giao diện Dashboard chuyên nghiệp
- Tối ưu hiệu năng font và rendering
"""

import cv2
import time
import sys
import numpy as np

sys.path.insert(0, '.')

from vision_module import VisionModule
from vision.config import VisionConfig
from vision.ui import VisionUI

def main():
    print("=" * 70)
    print("🔍 ROBOT FEEE - VISION DASHBOARD v2.0")
    print("=" * 70)
    
    # 1. Khởi tạo cấu hình và module lõi
    config = VisionConfig()
    config.MODE = "balanced" # Có thể đổi sang 'safe' hoặc 'agile'
    
    vision = VisionModule(config=config)
    ui = VisionUI(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)
    
    # Biến tính toán FPS
    fps_calc = {
        'count': 0,
        'time': time.time(),
        'val': 0
    }
    
    print("🚀 Hệ thống đang chạy... Nhấn 'q' tại cửa sổ camera để thoát.")

    try:
        # 2. Vòng lặp xử lý chính
        for frame, objects in vision.process_video():
            # Tính toán FPS
            fps_calc['count'] += 1
            if time.time() - fps_calc['time'] >= 1.0:
                fps_calc['val'] = fps_calc['count'] / (time.time() - fps_calc['time'])
                fps_calc['count'] = 0
                fps_calc['time'] = time.time()
            
            # Thêm trajectory vào object list để UI render (nếu có)
            for obj in objects:
                obj_id = obj.get('id', -1)
                if obj_id != -1:
                    obj['trajectory'] = vision.tracker.get_trajectory(obj_id)
            
            # 3. Giao cho lớp UI render toàn bộ Dashboard
            display_frame = ui.render(
                frame=frame, 
                objects=objects, 
                fps=fps_calc['val'], 
                mode=config.MODE
            )
            
            # 4. Hiển thị
            cv2.imshow('Robot Feee - Professional Vision Dashboard', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")
    finally:
        vision.release()
        cv2.destroyAllWindows()
        print("\n✅ Hệ thống đã đóng an toàn.")

if __name__ == "__main__":
    main()
