#!/usr/bin/env python3
"""
TEST ĐƠN GIẢN - Chỉ hiển thị bounding box và behavior
"""

import cv2
import sys
sys.path.insert(0, '.')
from vision_module import VisionModule

print("=" * 50)
print("TEST NHẬN DIỆN - XEM ROBOT THẤY GÌ?")
print("Nhấn 'q' để thoát")
print("=" * 50)

vision = VisionModule()

try:
    for frame, objects in vision.process_video():
        # Hiển thị số object
        cv2.putText(frame, f"Objects: {len(objects)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Robot Vision', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    vision.release()
    cv2.destroyAllWindows()
