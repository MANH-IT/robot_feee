"""
config.py - Cấu hình cổng giao tiếp với mạch vi điều khiển (Arduino/ESP)
"""

class HardwareConfig:
    # ========== Serial Configuration ==========
    # Trên Windows thường là COM3, COM4... Trên Linux là /dev/ttyUSB0
    SERIAL_PORT = "COM3" 
    BAUD_RATE = 115200
    TIMEOUT = 1.0  # seconds
    
    # ========== Motor Constraints ==========
    DEFAULT_SPEED = 150       # Tốc độ PWM mặc định (0-255)
    MAX_SPEED = 255           # Tốc độ tối đa
    TURN_SPEED = 120          # Tốc độ khi vào cua
    
    # ========== Command Mappings ==========
    # Protocol: COMMAND_TYPE, PARAM1, PARAM2\n
    # M: Motor, S: Stop, B: Buzzer/Feedback
    COMMANDS = {
        'MOVE_STRAIGHT': f"M,F,{DEFAULT_SPEED}\n",  # Motor, Forward
        'MOVE_LEFT': f"M,L,{TURN_SPEED}\n",         # Motor, Left
        'MOVE_RIGHT': f"M,R,{TURN_SPEED}\n",        # Motor, Right
        'STOP': "S,0,0\n",                          # Stop Normal
        'STOP_SAFETY': "S,E,0\n",                   # Stop Emergency (Phanh gấp)
        'IDLE': "S,0,0\n",
        'GREET': "B,1,0\n",                         # Buzzer 1 tiếng (Chào)
        'ERROR': "B,3,0\n"                          # Buzzer 3 tiếng (Lỗi)
    }
