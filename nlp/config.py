# -*- coding: utf-8 -*-
"""
config.py - Cấu hình cho module xử lý ngôn ngữ tự nhiên
Tác giả: Robot EEEC Team
Phiên bản: 3.0 - Nâng cao
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

# Dinh dang duong dan goc cua du an
PROJECT_ROOT = Path(__file__).parent.parent

# Tao thu muc models neu chua ton tai
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class NLPConfig:
    """
    Cau hinh tong hop cho he thong NLP
    Bao gom: duong dan model, tham so xu ly, mapping intent
    """
    
    # ========== Duong dan model ==========
    # Vosk model cho nhan dang gioi noi offline (tieng Viet)
    vosk_model_path: str = str(MODELS_DIR / "vosk_vn")
    
    # Stanza model cho phan tich cau phap tieng Viet
    stanza_model_dir: str = str(MODELS_DIR / "stanza_vi")
    
    # SNN model cho phan loai y dinh
    snn_intent_path: str = str(MODELS_DIR / "snn_intent_phobert.pth")
    
    # File chua tri thuc ve truong UTC
    knowledge_file: str = str(PROJECT_ROOT / "nlp" / "data" / "utc_knowledge_advanced.json")
    
    # File chua thong tin toa nha 15 tang
    building_file: str = str(PROJECT_ROOT / "data" / "building_data.json")
    
    # ========== Tham so xu ly am thanh ==========
    sample_rate: int = 16000           # Tan so lay mau (Hz)
    mic_device_index: int = 0          # Chi so microphone mac dinh
    audio_chunk_size: int = 4000       # Kich thuoc chunk am thanh
    silence_threshold: int = 500       # Nguong nhan dang im lang
    
    # ========== Tham so Stanza NLP ==========
    stanza_lang: str = 'vi'            # Ngon ngu tieng Viet
    stanza_processors: str = 'tokenize,pos,lemma,depparse'  # Bo xu ly can dung
    
    # ========== Tham so PhoBERT ==========
    use_phobert: bool = True           # Su dung PhoBERT thay cho BoW

    # ========== Tham so SNN Intent ==========
    snn_input_dim: int = 768           # 768 neu dung PhoBERT, 256 neu dung BoW
    snn_hidden_dim: int = 128          # Chieu tang an
    snn_num_steps: int = 20            # So buoc thoi gian cho SNN
    
    # ========== Danh sach y dinh ==========
    intent_classes: List[str] = None
    
    # ========== Mapping tu y dinh sang hanh dong robot ==========
    intent_to_action: Dict[str, str] = None
    
    # ========== Tu khoa cho tung y dinh (fallback) ==========
    intent_keywords: Dict[str, List[str]] = None
    
    # ========== Nguong tin cay ==========
    confidence_threshold: float = 0.6   # Nguong chap nhan cua SNN
    
    def __post_init__(self):
        """Khoi tao cac mapping va danh sach sau khi khoi tao"""
        
        # Danh sach y dinh (8 classes)
        self.intent_classes = [
            'di_chuyen',        # Di chuyen: re trai, re phai, di thang
            'dung_lai',         # Dung lai: dung khau cap
            'bam_theo',         # Bam theo: di theo nguoi hoac vat
            'hoi_thong_tin',    # Hoi thong tin: ve truong, khoa, nganh
            'chao_hoi',         # Chao hoi: xin chao, hello
            'cam_on',           # Cam on
            'tam_biet',         # Tam biet
            'khac'              # Khong xac dinh
        ]
        
        # Mapping y dinh sang hanh dong robot
        self.intent_to_action = {
            'di_chuyen': 'MOVE',
            'dung_lai': 'STOP',
            'bam_theo': 'FOLLOW',
            'hoi_thong_tin': 'ANSWER',
            'chao_hoi': 'GREET',
            'cam_on': 'THANK',
            'tam_biet': 'BYE',
            'khac': 'IDLE'
        }
        
        # Tu khoa cho tung y dinh (su dung khi SNN chua san sang)
        self.intent_keywords = {
            'di_chuyen': ['re', 'queo', 'di', 'chay', 'tien', 'lui', 'thang', 'trai', 'phai', 'quay'],
            'dung_lai': ['dung', 'ngung', 'dung lai', 'ngung lai', 'dung ngay'],
            'bam_theo': ['bam', 'theo', 'duoi', 'bam theo', 'di theo', 'chay theo'],
            'hoi_thong_tin': ['o dau', 'hoc gi', 'lam gi', 'thong tin', 'bao nhieu', 'khi nao', 'the nao'],
            'chao_hoi': ['xin chao', 'hello', 'chao', 'hi', 'chao ban', 'chao cau'],
            'cam_on': ['cam on', 'thank', 'thanks', 'cam on ban'],
            'tam_biet': ['tam biet', 'bye', 'goodbye', 'hen gap lai', 'chao tam biet']
        }


# Khoi tao cau hinh mac dinh de cac module khac co the import
default_config = NLPConfig()