"""
config.py - Cấu hình cho NLP module
"""

import os
from pathlib import Path

# Đường dẫn gốc của dự án
PROJECT_ROOT = Path(__file__).parent.parent

class NLPConfig:
    """Cấu hình cho module xử lý ngôn ngữ"""
    
    # ========== Đường dẫn model ==========
    VOSK_MODEL_PATH = str(PROJECT_ROOT / "models" / "vosk-model-small-vi-0.22")
    STANZA_MODEL_DIR = str(PROJECT_ROOT / "models" / "stanza_vi")
    
    # ========== Vosk STT ==========
    SAMPLE_RATE = 16000          # Tần số lấy mẫu
    MIC_DEVICE_INDEX = 0         # Index microphone
    
    # ========== Stanza ==========
    STANZA_LANG = 'vi'
    STANZA_PROCESSORS = 'tokenize,pos,depparse'
    
    # ========== Adaptive Grammar ==========
    DEFAULT_GRAMMAR_WEIGHT = 1.0
    MIN_GRAMMAR_WEIGHT = 0.0
    MAX_GRAMMAR_WEIGHT = 2.0
    LEARNING_RATE = 0.1          # Delta cập nhật trọng số
    
    # ========== Intent Classification (SNN) ==========
    INTENT_CLASSES = [
        'di_chuyen',     # di chuyển (rẽ trái, phải, đi thẳng)
        'dung_lai',      # dừng lại
        'bam_theo',      # bám theo đối tượng
        'hoi_thong_tin', # hỏi thông tin về trường
        'chao_hoi',      # chào hỏi
        'khac'           # không xác định
    ]
    
    # ========== Command mapping ==========
    # Ánh xạ từ ý định sang hành động robot
    INTENT_TO_ACTION = {
        'di_chuyen': 'MOVE',
        'dung_lai': 'STOP',
        'bam_theo': 'FOLLOW',
        'hoi_thong_tin': 'ANSWER',
        'chao_hoi': 'GREET',
        'khac': 'IDLE'
    }
    
    # Từ khóa cho từng ý định (fallback khi SNN chưa train)
    INTENT_KEYWORDS = {
        'di_chuyen': ['rẽ', 'quẹo', 'đi', 'chạy', 'tiến', 'lùi', 'thẳng', 'trái', 'phải'],
        'dung_lai': ['dừng', 'ngừng', 'dừng lại', 'ngừng lại'],
        'bam_theo': ['bám', 'theo', 'đuổi', 'bám theo', 'đi theo'],
        'hoi_thong_tin': ['ở đâu', 'học gì', 'làm gì', 'thông tin', 'bao nhiêu', 'khi nào'],
        'chao_hoi': ['xin chào', 'hello', 'chào', 'hi']
    }
