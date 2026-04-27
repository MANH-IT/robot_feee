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
from hardware import RobotController

app = Flask(__name__, 
            template_folder='../frontend',
            static_folder='../frontend/assets',
            static_url_path='/assets')
socketio = SocketIO(app, cors_allowed_origins="*")

# Khởi tạo các module
vision = VisionModule()
nlp = NLPModule(use_mock=False)  # Đã có Model Vosk & Stanza, chạy thật!
fusion = DecisionFusion()
controller = RobotController(mock_mode=True)  # Đặt mock_mode=False khi cắm board Arduino thật

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
        final_action = fusion.fuse(action, current_objects)
        
        # Gửi lệnh xuống phần cứng (Arduino)
        controller.send_action(final_action)
        
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

@socketio.on('ask_question')
def handle_ask_question(data):
    """Xử lý câu hỏi qua WebSocket"""
    question = data.get('question', '')
    if question:
        answer, confidence = nlp.answer_question(question)
        emit('answer_response', {
            'question': question,
            'answer': answer,
            'confidence': confidence
        })

# Routes
@app.route('/')
def index():
    """Trang chủ"""
    return render_template('pages/home.html')

@app.route('/camera')
def camera():
    """Trang camera"""
    return render_template('pages/camera.html')

@app.route('/chat')
def chat():
    return render_template('pages/chat.html')

@app.route('/map')
def map_page():
    return render_template('pages/map.html')

@app.route('/news')
def news():
    return render_template('pages/news.html')

@app.route('/team')
def team():
    return render_template('pages/team.html')

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

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """API nhận câu hỏi và trả lời"""
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'success': False, 'error': 'No question provided'}), 400
    
    answer, confidence = nlp.answer_question(question)
    
    return jsonify({
        'success': True,
        'question': question,
        'answer': answer,
        'confidence': confidence
    })

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

# Audio processing thread
def audio_loop():
    print("[Audio] Starting offline audio listening thread...")
    while True:
        try:
            text = nlp.listen_once()
            if text:
                print(f"[Audio] Nhận diện giọng nói: {text}")
                
                action, command_obj = nlp.parse_command(text)
                
                if command_obj:
                    global current_command, current_action
                    current_command = command_obj
                    current_action = action
                    
                    final_action = fusion.fuse(action, current_objects)
                    controller.send_action(final_action)
                    
                    socketio.emit('command_result', {
                        'success': True,
                        'text': text,
                        'intent': command_obj.intent,
                        'action': action,
                        'final_action': final_action,
                        'confidence': command_obj.confidence,
                        'entities': command_obj.entities
                    })
        except Exception as e:
            print(f"[Audio] Lỗi vòng lặp nghe: {e}")
            import time
            time.sleep(1)

if __name__ == '__main__':
    # Chạy vision loop trong thread riêng
    vision_thread = threading.Thread(target=vision_loop, daemon=True)
    vision_thread.start()
    
    # Chạy audio loop trong thread riêng
    audio_thread = threading.Thread(target=audio_loop, daemon=True)
    audio_thread.start()
    
    # Chạy web server
    print("[Web] Starting server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)