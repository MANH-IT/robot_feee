"""
utils.py - Hàm tiện ích cho NLP module
"""

import re
from typing import List, Dict, Tuple
import json


def preprocess_text(text: str) -> str:
    """
    Tiền xử lý văn bản tiếng Việt
    - lowercase
    - bỏ dấu câu
    - chuẩn hóa khoảng trắng
    """
    # lowercase
    text = text.lower()
    
    # bỏ dấu câu
    text = re.sub(r'[^\w\s]', '', text)
    
    # chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_action_from_intent(intent: str, entities: Dict, intent_to_action: Dict) -> str:
    """Chuyển ý định và thực thể thành hành động robot"""
    base_action = intent_to_action.get(intent, 'IDLE')
    
    # Thêm chi tiết dựa trên entities
    if base_action == 'MOVE':
        direction = entities.get('direction', 'straight')
        return f"MOVE_{direction.upper()}"
    
    return base_action


def format_response(intent: str, action: str, entities: Dict) -> str:
    """Tạo phản hồi dạng text cho người dùng"""
    responses = {
        'di_chuyen': f"Đã hiểu, robot sẽ {entities.get('direction', 'đi thẳng')}",
        'dung_lai': "Đã dừng robot",
        'bam_theo': f"Đang bám theo {entities.get('target', 'đối tượng')}",
        'chao_hoi': "Xin chào! Tôi có thể giúp gì cho bạn?",
        'hoi_thong_tin': "Đang tra cứu thông tin...",
        'khac': "Tôi chưa hiểu lệnh, bạn có thể nói lại không?"
    }
    return responses.get(intent, "Đã nhận lệnh")


def save_command_log(log_path: str, command_data: Dict):
    """Lưu lịch sử câu lệnh vào file"""
    import os
    from datetime import datetime
    
    logs = []
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    
    command_data['timestamp'] = datetime.now().isoformat()
    logs.append(command_data)
    
    # Giữ tối đa 1000 log
    if len(logs) > 1000:
        logs = logs[-1000:]
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def load_command_history(log_path: str) -> List[Dict]:
    """Đọc lịch sử câu lệnh từ file"""
    import os
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []
