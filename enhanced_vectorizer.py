import torch
import numpy as np
from typing import List, Dict

class EnhancedTextVectorizer:
    """
    Vectorizer nâng cấp:
    - Bag-of-Words (từ đơn)
    - Bigram (cụm 2 từ)
    - TF-IDF weighting
    - Position encoding cơ bản
    """
    
    def __init__(self, output_dim: int = 256, use_bigrams: bool = True):
        self.output_dim = output_dim
        self.use_bigrams = use_bigrams
        
        # Từ điển từ khóa mở rộng
        self.keywords = self._build_keyword_vocab()
        self.bigram_vocab = self._build_bigram_vocab() if use_bigrams else {}
        
        # Trọng số TF-IDF (khởi tạo mặc định)
        self.weights = {}
        self._init_weights()
    
    def _build_keyword_vocab(self) -> Dict[str, int]:
        """Xây dựng từ vựng từ khóa mở rộng"""
        keywords = {
            # Hành động
            'di': 1, 'chay': 2, 'tien': 3, 'lui': 4, 'thang': 5,
            're': 6, 'queo': 7, 'trai': 8, 'phai': 9, 'quay': 10,
            'dung': 11, 'ngung': 12, 'bam': 13, 'theo': 14, 'duoi': 15,
            
            # Thông tin
            'phong': 20, 'tang': 21, 'khoa': 22, 'vien': 23, 'nganh': 24,
            'hoc': 25, 'truong': 26, 'thu vien': 27, 'giang duong': 28,
            'thanh lap': 29, 'nam': 30, 'dia chi': 31, 'so dien thoai': 32,
            
            # Chào hỏi
            'chao': 40, 'hello': 41, 'hi': 42, 'xin chao': 43,
            
            # Cảm ơn
            'cam on': 50, 'thank': 51, 'thanks': 55,
            
            # Tạm biệt
            'tam biet': 60, 'bye': 61, 'goodbye': 62,
            
            # Từ khóa đặc biệt
            'robot': 100, 'fe': 101
        }
        return keywords
    
    def _build_bigram_vocab(self) -> Dict[str, int]:
        """Xây dựng từ vựng bigram"""
        bigrams = [
            'di thang', 're trai', 're phai', 'dung lai',
            'bam theo', 'di theo', 'cam on', 'tam biet',
            'phong hoc', 'tang may', 'khoa nao', 'nganh gi'
        ]
        return {bg: idx + 200 for idx, bg in enumerate(bigrams)}
    
    def _init_weights(self):
        """Khởi tạo trọng số TF-IDF"""
        # Trọng số cao cho từ khóa quan trọng
        high_weight_keywords = ['re', 'trai', 'phai', 'dung', 'bam', 'theo']
        for kw in high_weight_keywords:
            if kw in self.keywords:
                self.weights[self.keywords[kw]] = 2.0
        
        # Trọng số trung bình
        medium_weight_keywords = ['di', 'chay', 'quay', 'phong', 'tang', 'khoa']
        for kw in medium_weight_keywords:
            if kw in self.keywords:
                self.weights[self.keywords[kw]] = 1.5
        
        # Trọng số mặc định
        default_weight = 1.0
        for idx in self.keywords.values():
            if idx not in self.weights:
                self.weights[idx] = default_weight
    
    def _extract_keywords(self, text: str) -> List[int]:
        """Trích xuất từ khóa và bigram từ văn bản"""
        text_lower = text.lower()
        keywords_found = []
        
        # Tìm bigram trước
        if self.use_bigrams:
            for bigram, idx in self.bigram_vocab.items():
                if bigram in text_lower:
                    keywords_found.append(idx)
                    text_lower = text_lower.replace(bigram, '')
        
        # Tìm từ khóa đơn
        for keyword, idx in self.keywords.items():
            if keyword in text_lower:
                keywords_found.append(idx)
        
        return keywords_found
    
    def transform(self, text: str) -> torch.Tensor:
        """Chuyển văn bản thành vector đặc trưng nâng cao"""
        features = torch.zeros(self.output_dim)
        
        # Trích xuất từ khóa
        keyword_indices = self._extract_keywords(text)
        
        # Đánh dấu và áp dụng trọng số
        for idx in keyword_indices:
            if idx < self.output_dim:
                weight = self.weights.get(idx, 1.0)
                features[idx] += weight
        
        # Chuẩn hóa vector về [0, 1]
        if features.sum() > 0:
            features = features / features.sum()
        else:
            # Tạo vector mặc định cho câu lạ
            features[0] = 1.0
        
        return features
