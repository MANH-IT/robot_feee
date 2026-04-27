import serial
import time
import serial.tools.list_ports

def list_available_ports():
    ports = serial.tools.list_ports.comports()
    print("\n📡 Available COM ports:")
    if not ports:
        print("   Không tìm thấy cổng COM nào. Bạn có cắm Arduino chưa?")
    for port in ports:
        print(f"   {port.device} - {port.description}")
    return ports

def test_arduino_connection(port='COM3', baudrate=115200):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)
        ser.write(b"VEL 0.5 0.0\n")
        response = ser.readline()
        print(f"✅ Arduino connected on {port}: {response}")
        ser.close()
        return True
    except Exception as e:
        print(f"❌ Arduino not found on {port}: {e}")
        return False

if __name__ == "__main__":
    ports = list_available_ports()
    if ports:
        test_arduino_connection(ports[0].device)
