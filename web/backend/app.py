"""
app.py - Flask backend tích hợp vision và NLP modules
"""

import sys
import os
import json
from datetime import datetime

# Thêm đường dẫn gốc để import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import threading

# Import modules của dự án
from vision_module import VisionModule
from nlp_module import NLPModule
from decision_fusion import DecisionFusion

app = Flask(__name__, 
            template_folder='../frontend',
            static_folder='../frontend/assets',
            static_url_path='/assets')
socketio = SocketIO(app, cors_allowed_origins="*")

# Khởi tạo các module
vision = VisionModule()
nlp = NLPModule(use_mock=True)  # use_mock=True để test không cần microphone
fusion = DecisionFusion()

# Biến lưu trạng thái
current_frame = None
current_objects = []
current_command = None
current_action = None

# Socket events
@socketio.on('connect')
def handle_connect():
    print('[WebSocket] Client connected')
    emit('connected', {'status': 'ok'})

@socketio.on('voice_command')
def handle_voice_command(data):
    """Nhận lệnh voice từ client (qua WebSocket)"""
    global current_command, current_action
    
    text = data.get('text', '')
    if not text:
        return
    
    print(f"[Web] Voice command: {text}")
    
    # Parse lệnh
    action, command_obj = nlp.parse_command(text)
    
    if command_obj:
        current_command = command_obj
        current_action = action
        
        # Kết hợp với vision để quyết định
        # (Lưu ý: Bạn có thể cần triển khai fusion.decide trong decision_fusion.py)
        final_action = fusion.fuse(action, current_objects)
        
        # Gửi kết quả về client
        emit('command_result', {
            'success': True,
            'text': text,
            'intent': command_obj.intent,
            'action': action,
            'final_action': final_action,
            'confidence': command_obj.confidence,
            'entities': command_obj.entities
        })
    else:
        emit('command_result', {
            'success': False,
            'text': text,
            'error': 'Cannot parse command'
        })

@socketio.on('feedback')
def handle_feedback(data):
    """Nhận phản hồi từ người dùng (đúng/sai) để cập nhật grammar"""
    success = data.get('success', False)
    command_text = data.get('command', '')
    
    if current_command and current_command.original_text == command_text:
        nlp.update_feedback(current_command, success)
        print(f"[Web] Feedback received: {success} for '{command_text}'")
        emit('feedback_result', {'success': True})

@socketio.on('get_status')
def handle_get_status():
    """Gửi trạng thái hiện tại"""
    emit('status_update', {
        'objects': current_objects,
        'current_command': current_command.to_dict() if current_command else None,
        'current_action': current_action
    })

# Routes
@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/camera')
def camera():
    """Trang camera"""
    return render_template('pages/camera.html')

@app.route('/video_feed')
def video_feed():
    """Stream video từ camera"""
    def generate():
        global current_frame, current_objects
        while True:
            if current_frame is not None:
                # Vẽ bounding box lên frame
                from vision.utils import draw_bbox
                frame_with_boxes = draw_bbox(current_frame.copy(), current_objects)
                _, jpeg = cv2.imencode('.jpg', frame_with_boxes)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/objects')
def get_objects():
    """API lấy danh sách object hiện tại"""
    return jsonify({'objects': current_objects})

@app.route('/api/command', methods=['POST'])
def post_command():
    """API nhận lệnh text (thay thế voice)"""
    data = request.get_json()
    text = data.get('text', '')
    
    action, command_obj = nlp.parse_command(text)
    
    if command_obj:
        return jsonify({
            'success': True,
            'text': text,
            'intent': command_obj.intent,
            'action': action,
            'entities': command_obj.entities
        })
    return jsonify({'success': False, 'error': 'Cannot parse'}), 400

# Vision processing thread
def vision_loop():
    global current_frame, current_objects
    
    print("[Vision] Starting vision processing thread...")
    for frame, objects in vision.process_video():
        current_frame = frame
        current_objects = objects
        
        # Gửi update qua WebSocket
        socketio.emit('vision_update', {
            'num_objects': len(objects),
            'objects': [
                {'id': obj.get('id'), 'class': obj.get('class_name'), 
                 'behavior': obj.get('behavior'), 'confidence': obj.get('confidence')}
                for obj in objects
            ]
        })

if __name__ == '__main__':
    # Chạy vision loop trong thread riêng
    vision_thread = threading.Thread(target=vision_loop, daemon=True)
    vision_thread.start()
    
    # Chạy web server
    print("[Web] Starting server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)