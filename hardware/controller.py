"""
controller.py - Lớp điều khiển Robot giao tiếp qua cổng Serial
"""

import serial
import time
import threading
from typing import Optional

from .config import HardwareConfig


class RobotController:
    """Điều khiển phần cứng robot qua Serial"""
    
    def __init__(self, config: HardwareConfig = None, mock_mode: bool = False):
        self.config = config or HardwareConfig()
        self.mock_mode = mock_mode
        self.serial_conn: Optional[serial.Serial] = None
        self.lock = threading.Lock()
        self.is_connected = False
        
        self.connect()

    def connect(self):
        """Khởi tạo kết nối với vi điều khiển"""
        if self.mock_mode:
            print(f"[Hardware] MOCK MODE enabled. Simulating connection to {self.config.SERIAL_PORT}")
            self.is_connected = True
            return True
            
        try:
            self.serial_conn = serial.Serial(
                port=self.config.SERIAL_PORT,
                baudrate=self.config.BAUD_RATE,
                timeout=self.config.TIMEOUT
            )
            time.sleep(2)  # Đợi Arduino khởi động lại khi mở Serial
            self.is_connected = True
            print(f"[Hardware] Successfully connected to {self.config.SERIAL_PORT}")
            return True
        except serial.SerialException as e:
            self.is_connected = False
            print(f"[Hardware] ERROR: Cannot connect to {self.config.SERIAL_PORT}. Details: {e}")
            print("[Hardware] Switching to MOCK MODE...")
            self.mock_mode = True
            self.is_connected = True
            return False

    def send_action(self, action: str):
        """
        Dịch lệnh ngữ nghĩa thành bản tin Serial và gửi đi
        
        Args:
            action (str): Hành động từ hệ thống (VD: 'MOVE_LEFT')
        """
        if not self.is_connected:
            return False
            
        # Dịch action thành command frame
        command_frame = self.config.COMMANDS.get(action, self.config.COMMANDS['IDLE'])
        
        with self.lock:
            if self.mock_mode:
                print(f"[Hardware-Mock] Sent: {command_frame.strip()}")
            else:
                try:
                    self.serial_conn.write(command_frame.encode('utf-8'))
                    self.serial_conn.flush()
                    print(f"[Hardware] Sent: {command_frame.strip()}")
                except Exception as e:
                    print(f"[Hardware] Error sending command: {e}")
                    self.is_connected = False
                    
        return True

    def read_feedback(self) -> Optional[str]:
        """Đọc phản hồi từ cảm biến siêu âm/encoder của vi điều khiển"""
        if self.mock_mode or not self.is_connected or not self.serial_conn:
            return None
            
        try:
            if self.serial_conn.in_waiting > 0:
                data = self.serial_conn.readline().decode('utf-8').strip()
                return data
        except Exception:
            pass
        return None

    def close(self):
        """Ngắt kết nối an toàn"""
        if self.serial_conn and self.serial_conn.is_open:
            # Gửi lệnh dừng xe trước khi tắt
            self.send_action('STOP')
            time.sleep(0.1)
            self.serial_conn.close()
            print("[Hardware] Serial connection closed.")
