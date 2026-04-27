"""
intent_classifier.py - Phân loại ý định sử dụng SNN hoặc rule-based
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
import re

from .config import NLPConfig


class SpikingIntentNeuron(nn.Module):
    """Neuron LIF cho intent classification"""
    
    def __init__(self, beta: float = 0.9, threshold: float = 1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
    
    def forward(self, x, mem):
        mem = self.beta * mem + x
        spike = (mem >= self.threshold).float()
        mem = mem - spike * self.threshold
        return spike, mem


class IntentSNN(nn.Module):
    """
    Mạng SNN phân loại ý định từ câu văn bản.
    Input: word embedding features (từ PhoBERT hoặc TF-IDF)
    Output: intent class
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, 
                 num_classes: int = 6, num_steps: int = 10):
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self.lif1 = SpikingIntentNeuron()
        self.lif2 = SpikingIntentNeuron()
    
    def forward(self, x_seq):
        """
        x_seq: [steps, batch, input_dim]
        """
        batch_size = x_seq.shape[1]
        mem1 = torch.zeros(batch_size, 64)
        mem2 = torch.zeros(batch_size, self.num_classes)
        
        spike_outputs = []
        
        for step in range(self.num_steps):
            cur1 = self.fc1(x_seq[step])
            spike1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spike1)
            spike2, mem2 = self.lif2(cur2, mem2)
            spike_outputs.append(spike2)
        
        return torch.stack(spike_outputs, dim=0)


class IntentClassifier:
    """Phân loại ý định (fallback rule-based + SNN)"""
    
    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        self.snn_model = None
        self._init_snn()
    
    def _init_snn(self):
        """Khởi tạo SNN model (mock, chưa train)"""
        # TODO: Load model đã train từ file
        self.snn_model = IntentSNN(
            input_dim=128,
            hidden_dim=64,
            num_classes=len(self.config.INTENT_CLASSES)
        )
        print("[Intent] SNN model initialized (not trained yet)")
    
    def predict(self, text: str) -> str:
        """
        Dự đoán ý định từ câu lệnh
        
        Hiện tại dùng rule-based (keyword matching) cho độ chính xác cao hơn
        """
        text_lower = text.lower()
        
        # Kiểm tra theo thứ tự ưu tiên
        for intent, keywords in self.config.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return intent
        
        return "khac"
    
    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        """Dự đoán kèm độ tin cậy"""
        text_lower = text.lower()
        
        best_intent = "khac"
        best_score = 0.0
        
        for intent, keywords in self.config.INTENT_KEYWORDS.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1.0
            score = min(1.0, score / 2)  # normalize
            
            if score > best_score:
                best_score = score
                best_intent = intent
        
        return best_intent, best_score
    
    def extract_entities(self, text: str, intent: str) -> Dict[str, str]:
        """Trích xuất thực thể từ câu lệnh"""
        text_lower = text.lower()
        entities = {}
        
        # Trích xuất hướng
        if any(w in text_lower for w in ['trái', 'trái ạ']):
            entities['direction'] = 'left'
        elif any(w in text_lower for w in ['phải', 'phải ạ']):
            entities['direction'] = 'right'
        elif 'thẳng' in text_lower:
            entities['direction'] = 'straight'
        
        # Trích xuất tầng
        floor_match = re.search(r'tầng\s*(\d+)', text_lower)
        if floor_match:
            entities['floor'] = floor_match.group(1)
        
        # Trích xuất đối tượng cần bám
        if intent == 'bam_theo':
            if 'người' in text_lower:
                entities['target'] = 'person'
            elif 'xe' in text_lower:
                entities['target'] = 'vehicle'
        
        return entities
