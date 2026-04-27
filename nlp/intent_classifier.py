# -*- coding: utf-8 -*-
"""
intent_classifier.py - Phan loai y dinh su dung Spiking Neural Network
Ket hop: BoW Text Vectorizer + SNN (snntorch.Leaky) + Softmax
Phiên bản: 3.0 - Hoc thuc te, khong con gia lap
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List, Optional
import re
import os

import snntorch as snn
from snntorch import surrogate

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from .config import default_config, NLPConfig
except ImportError:
    # Ho tro chay truc tiep file bang cach import cung thu muc
    from config import default_config, NLPConfig


class SpikingIntentNeuron(nn.Module):
    """
    Mang noron xung (SNN) cho phan loai y dinh - Phien ban Nang cap 4.0
    Su dung noron Leaky Integrate-and-Fire (LIF) voi Surrogate Gradient, BatchNorm, Dropout
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, 
                 num_classes: int = 8, num_steps: int = 20):
        """Khoi tao mang SNN nang cap"""
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes
        
        spike_grad = surrogate.fast_sigmoid(slope=15.0)
        dropout_rate = 0.3
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad, output=True)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad, output=True)
        
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.lif3 = snn.Leaky(beta=0.95, spike_grad=spike_grad, output=True)
    
    def forward(self, x_sequence: torch.Tensor) -> torch.Tensor:
        """Duyet mang noron xung theo thoi gian"""
        memory1 = self.lif1.init_leaky()
        memory2 = self.lif2.init_leaky()
        memory3 = self.lif3.init_leaky()
        
        spike_outputs = []
        
        for step in range(self.num_steps):
            cur1 = self.fc1(x_sequence[step])
            cur1 = self.bn1(cur1)
            spike1, memory1 = self.lif1(cur1, memory1)
            spike1 = self.dropout1(spike1)
            
            cur2 = self.fc2(spike1)
            cur2 = self.bn2(cur2)
            spike2, memory2 = self.lif2(cur2, memory2)
            spike2 = self.dropout2(spike2)
            
            cur3 = self.fc3(spike2)
            spike3, memory3 = self.lif3(cur3, memory3)
            
            spike_outputs.append(spike3)
        
        return torch.stack(spike_outputs, dim=0)
    
    def predict_from_features(self, features: torch.Tensor) -> Tuple[int, float]:
        """Dua ra du doan tu vector dac trung"""
        self.eval()
        
        # Tao sequence goi ham forward qua cac buoc thoi gian
        batch_features = features.unsqueeze(0).unsqueeze(1)
        features_seq = batch_features.repeat(self.num_steps, 1, 1)
        
        with torch.no_grad():
            spike_outputs = self.forward(features_seq)
            spike_counts = spike_outputs.sum(dim=0).squeeze(0)
            probabilities = torch.softmax(spike_counts, dim=0)
            confidence, class_index = probabilities.max(dim=0)
            
        return class_index.item(), confidence.item()
        return class_index.item(), confidence.item()


class BaseVectorizer:
    def transform(self, text: str) -> torch.Tensor:
        pass

class PhoBERTVectorizer(BaseVectorizer):
    """Vectorizer su dung PhoBERT (768-dim)"""
    def __init__(self):
        print("[PhoBERT] Dang tai model tu vinai/phobert-base...")
        if not HAS_TRANSFORMERS:
            raise ImportError("Vui long 'pip install transformers'")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.model = AutoModel.from_pretrained("vinai/phobert-base")
        self.model.eval()
        self.output_dim = 768
        
    def transform(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_vector = outputs.last_hidden_state[:, 0, :]
        return cls_vector.squeeze(0)

class TextVectorizer(BaseVectorizer):
    """
    Vectorizer nang cap: Bag-of-Words + Bigram + TF-IDF weighting
    """
    
    def __init__(self, output_dim: int = 256, config: NLPConfig = None):
        """Khoi tao bo chuyen doi van ban"""
        self.output_dim = output_dim
        self.config = config or default_config
        self.use_bigrams = True
        
        # Xay dung tu dien tu khoa tu cau hinh
        self.keywords = self._build_keyword_vocab()
        self.bigram_vocab = self._build_bigram_vocab() if self.use_bigrams else {}
        
        # Trong so TF-IDF
        self.weights = {}
        self._init_weights()
    
    def _build_keyword_vocab(self) -> Dict[str, int]:
        """Xay dung tu vung tu khoa mo rong"""
        keywords = {
            'di': 1, 'chay': 2, 'tien': 3, 'lui': 4, 'thang': 5,
            're': 6, 'queo': 7, 'trai': 8, 'phai': 9, 'quay': 10,
            'dung': 11, 'ngung': 12, 'bam': 13, 'theo': 14, 'duoi': 15,
            'phong': 20, 'tang': 21, 'khoa': 22, 'vien': 23, 'nganh': 24,
            'hoc': 25, 'truong': 26, 'thu vien': 27, 'giang duong': 28,
            'thanh lap': 29, 'nam': 30, 'dia chi': 31, 'so dien thoai': 32,
            'o dau': 33, 'cntt': 34, 'cong nghe thong tin': 35,
            'chao': 40, 'hello': 41, 'hi': 42, 'xin chao': 43,
            'cam on': 50, 'thank': 51, 'thanks': 55,
            'tam biet': 60, 'bye': 61, 'goodbye': 62,
            'robot': 100, 'fe': 101
        }
        return keywords
    
    def _build_bigram_vocab(self) -> Dict[str, int]:
        """Xay dung tu vung bigram"""
        bigrams = [
            'di thang', 're trai', 're phai', 'dung lai',
            'bam theo', 'di theo', 'cam on', 'tam biet',
            'phong hoc', 'tang may', 'khoa nao', 'nganh gi'
        ]
        return {bg: idx + 200 for idx, bg in enumerate(bigrams)}
    
    def _init_weights(self):
        """Khoi tao trong so TF-IDF"""
        high_weight_keywords = ['re', 'trai', 'phai', 'dung', 'bam', 'theo']
        for kw in high_weight_keywords:
            if kw in self.keywords:
                self.weights[self.keywords[kw]] = 2.0
        
        medium_weight_keywords = ['di', 'chay', 'quay', 'phong', 'tang', 'khoa']
        for kw in medium_weight_keywords:
            if kw in self.keywords:
                self.weights[self.keywords[kw]] = 1.5
        
        default_weight = 1.0
        for idx in self.keywords.values():
            if idx not in self.weights:
                self.weights[idx] = default_weight
    
    def _remove_accents(self, text: str) -> str:
        import unicodedata
        text = unicodedata.normalize('NFD', text)
        text = re.sub(r'[\u0300-\u036f]', '', text)
        return text.replace('đ', 'd').replace('Đ', 'D')
        
    def _extract_keywords(self, text: str) -> List[int]:
        """Trich xuat tu khoa va bigram tu van ban"""
        text_lower = text.lower()
        text_lower = self._remove_accents(text_lower)
        text_lower = re.sub(r'[^\w\s]', ' ', text_lower)
        text_lower = re.sub(r'\s+', ' ', text_lower).strip()
        
        keywords_found = []
        
        if self.use_bigrams:
            for bigram, idx in self.bigram_vocab.items():
                if bigram in text_lower:
                    keywords_found.append(idx)
                    text_lower = text_lower.replace(bigram, '')
        
        for keyword, idx in self.keywords.items():
            if keyword in text_lower:
                keywords_found.append(idx)
        
        return keywords_found
    
    def transform(self, text: str) -> torch.Tensor:
        """Chuyen doi van ban thanh vector dac trung nang cao"""
        features = torch.zeros(self.output_dim)
        keyword_indices = self._extract_keywords(text)
        
        for idx in keyword_indices:
            if idx < self.output_dim:
                weight = self.weights.get(idx, 1.0)
                features[idx] += weight
        
        if features.sum() > 0:
            features = features / features.sum()
        else:
            features[0] = 1.0
        
        return features


class IntentClassifier:
    """
    Bo phan loai y dinh chinh
    Ket hop: TextVectorizer + SpikingIntentNeuron
    """
    
    def __init__(self, config: NLPConfig = None):
        """
        Khoi tao bo phan loai y dinh
        
        Tham so:
            config: Cau hinh NLP
        """
        self.config = config or default_config
        
        # Khoi tao bo chuyen doi van ban
        if getattr(self.config, 'use_phobert', False):
            print("[IntentClassifier] Su dung PhoBERT Vectorizer...")
            self.vectorizer = PhoBERTVectorizer()
        else:
            self.vectorizer = TextVectorizer(output_dim=self.config.snn_input_dim, config=self.config)
        
        # Khoi tao mang SNN
        self.model = SpikingIntentNeuron(
            input_dim=self.config.snn_input_dim,
            hidden_dim=self.config.snn_hidden_dim,
            num_classes=len(self.config.intent_classes),
            num_steps=self.config.snn_num_steps
        )
        
        # Load trong so da duoc train (neu co)
        self._load_weights()
        
        # Che do su dung: True = SNN, False = Rule-based (fallback)
        self.use_snn = self._check_model_ready()
        
        if self.use_snn:
            print(f"[IntentClassifier] SNN model ready, {len(self.config.intent_classes)} intent classes")
        else:
            print(f"[IntentClassifier] SNN not available, using rule-based fallback")
    
    def _load_weights(self):
        """Load trong so cua SNN tu file da train"""
        if os.path.exists(self.config.snn_intent_path):
            try:
                checkpoint = torch.load(self.config.snn_intent_path, map_location='cpu')
                
                # Kiem tra cau truc checkpoint
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.eval()
                print(f"[IntentClassifier] Loaded weights from {self.config.snn_intent_path}")
                
                # Hien thi thong tin them neu co
                if isinstance(checkpoint, dict) and 'val_acc' in checkpoint:
                    print(f"[IntentClassifier] Validation accuracy: {checkpoint['val_acc']:.2f}%")
                    
            except Exception as error:
                print(f"[IntentClassifier] Error loading weights: {error}")
        else:
            print(f"[IntentClassifier] No weights found at {self.config.snn_intent_path}")
    
    def _check_model_ready(self) -> bool:
        """
        Kiem tra xem SNN da san sang su dung chua
        
        Tra ve:
            True neu co the dung SNN, False neu phai dung fallback
        """
        # Kiem tra model co trong so khong
        if not os.path.exists(self.config.snn_intent_path):
            return False
        
        # Kiem tra cac layer co duoc khoi tao khong
        try:
            test_features = torch.rand(self.config.snn_input_dim)
            self.model.predict_from_features(test_features)
            return True
        except Exception:
            return False
    
    def _rule_based_predict(self, text: str) -> Tuple[int, float]:
        """
        Phan loai y dinh bang phuong phap rule-based (fallback)
        
        Tham so:
            text: Van ban dau vao
        
        Tra ve:
            (class_index, confidence)
        """
        text_lower = text.lower()
        
        # Duyet qua tung y dinh va tu khoa tuong ung
        for intent_idx, intent_name in enumerate(self.config.intent_classes):
            keywords = self.config.intent_keywords.get(intent_name, [])
            for keyword in keywords:
                if keyword in text_lower:
                    confidence = 0.6  # Tin cay mac dinh cho rule-based
                    return intent_idx, confidence
        
        # Khong tim thay tu khoa nao
        default_intent_idx = self.config.intent_classes.index('khac')
        return default_intent_idx, 0.3
    
    def predict(self, text: str) -> str:
        """
        Du doan y dinh tu van ban
        
        Tham so:
            text: Van ban dau vao
        
        Tra ve:
            Ten y dinh (string)
        """
        intent_name, _ = self.predict_with_confidence(text)
        return intent_name
    
    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        """Du doan y dinh kem do tin cay"""
        text_lower = text.lower()
        
        # Heuristic dac biet cho cau hoi
        question_words = ['ở đâu', 'ở tầng mấy', 'năm nào', 'là ai', 'bao nhiêu', 'thế nào']
        for qw in question_words:
            if qw in text_lower:
                return 'hoi_thong_tin', 1.0
                
        features = self.vectorizer.transform(text)
        
        if self.use_snn:
            try:
                class_index, confidence = self.model.predict_from_features(features)
                intent_name = self.config.intent_classes[class_index]
                return intent_name, confidence
            except Exception as error:
                print(f"[IntentClassifier] SNN inference error: {error}")
                return self._fallback_predict(text)
        
        return self._fallback_predict(text)
    
    def _fallback_predict(self, text: str) -> Tuple[str, float]:
        """
        Phuong thuc fallback khi SNN khong hoat dong
        
        Tham so:
            text: Van ban dau vao
        
        Tra ve:
            (ten_y_dinh, do_tin_cay)
        """
        class_index, confidence = self._rule_based_predict(text)
        intent_name = self.config.intent_classes[class_index]
        return intent_name, confidence
    
    def extract_entities(self, text: str, intent: str = None) -> Dict[str, str]:
        """
        Trich xuat cac thuc the tu van ban
        
        Tham so:
            text: Van ban dau vao
            intent: Y dinh (neu biet) de trich xuat chinh xac hon
        
        Tra ve:
            Dictionary chua cac thuc the
        """
        text_lower = text.lower()
        entities = {}
        
        # Trich xuat huong di chuyen (trai, phai, thang)
        if 'trai' in text_lower:
            entities['direction'] = 'left'
        elif 'phai' in text_lower:
            entities['direction'] = 'right'
        elif 'thang' in text_lower:
            entities['direction'] = 'straight'
        
        # Trich xuat so tang (vi du: tang 3, tang 7)
        floor_pattern = re.search(r'tang\s*(\d+)', text_lower)
        if floor_pattern:
            entities['floor'] = floor_pattern.group(1)
        
        # Trich xuat doi tuong can bam theo
        if intent == 'bam_theo' or 'bam' in text_lower or 'theo' in text_lower:
            if 'nguoi' in text_lower:
                entities['target'] = 'person'
            elif 'xe' in text_lower:
                entities['target'] = 'vehicle'
            elif 'robot' in text_lower:
                entities['target'] = 'robot'
        
        # Trich xuat ma phong (vi du: phong 303, phong A101)
        room_pattern = re.search(r'phong\s*([A-Za-z]?\d{2,4})', text_lower)
        if room_pattern:
            entities['room_code'] = room_pattern.group(1)
        
        return entities
    
    def get_action(self, text: str) -> Tuple[str, float, Dict]:
        """
        Lay hanh dong robot tuong ung tu cau lenh
        
        Tham so:
            text: Van ban cau lenh
        
        Tra ve:
            (hanh_dong, do_tin_cay, thuc_the)
        """
        # Buoc 1: Phan loai y dinh
        intent, confidence = self.predict_with_confidence(text)
        
        # Buoc 2: Trich xuat thuc the
        entities = self.extract_entities(text, intent)
        
        # Buoc 3: Chuyen y dinh thanh hanh dong
        action = self.config.intent_to_action.get(intent, 'IDLE')
        
        # Bo sung huong di chuyen
        if action == 'MOVE' and 'direction' in entities:
            direction = entities['direction'].upper()
            action = f"MOVE_{direction}"
        
        return action, confidence, entities


# Tao instance mac dinh de cac module khac co the import
default_classifier = IntentClassifier()

if __name__ == "__main__":
    print("\n--- TEST INTENT CLASSIFIER ---")
    test_texts = ["Robot rẽ trái đi", "Dừng lại ngay", "Đi theo người áo đỏ", "Phòng 303 ở đâu vậy?"]
    for text in test_texts:
        intent, conf = default_classifier.predict_with_confidence(text)
        action, _, entities = default_classifier.get_action(text)
        print(f"Text: '{text}'\n -> Intent: {intent} ({conf*100:.1f}%)\n -> Action: {action}\n -> Entities: {entities}\n")