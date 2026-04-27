# -*- coding: utf-8 -*-
"""
knowledge_retriever.py - Truy xuat tri thuc tu kho du lieu
Hỗ trợ: Tra cuu khoa, giang vien, phong hoc, tin tuc
Phiên bản: 3.0 - Tich hop building data
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional
from difflib import get_close_matches
import torch
from typing import List, Dict, Optional, Tuple

try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

try:
    from .config import default_config, NLPConfig
except ImportError:
    from config import default_config, NLPConfig

try:
    from .intent_classifier import PhoBERTVectorizer
except ImportError:
    from nlp.intent_classifier import PhoBERTVectorizer


class KnowledgeRetriever:
    """
    Bo truy xuat tri thuc cho robot
    Du lieu bao gom: thong tin truong, khoa, giang vien, toa nha, tin tuc
    """
    
    def __init__(self, config: NLPConfig = None):
        """
        Khoi tao bo truy xuat tri thuc
        
        Tham so:
            config: Cau hinh NLP (chua duong dan file)
        """
        self.config = config or default_config
        
        # Khoi tao cac bien luu tru du lieu
        self.knowledge_data = None
        self.building_data = None
        self.custom_qa = []
        
        # Load du lieu
        self._load_knowledge_base()
        self._load_building_data()
        self.custom_qa = self._load_custom_qa()
        
        # Khoi tao PhoBERT cho Semantic Search
        self.use_phobert = getattr(self.config, 'use_phobert', False)
        if self.use_phobert:
            print("[KnowledgeRetriever] Dang khoi tao PhoBERT cho Semantic Search...")
            self.vectorizer = PhoBERTVectorizer()
            self.qa_embeddings = self._precompute_qa_embeddings()
        else:
            self.vectorizer = None
            self.qa_embeddings = None
        
        # Dinh nghia cac chu de va tu khoa tuong ung
        self.topic_keywords = {
            "gioi_thieu": ["gioi thieu", "truong gi", "ngoi truong", "thong tin chung"],
            "nam_thanh_lap": ["thanh lap", "nam thanh lap", "bao nhieu tuoi", "1962"],
            "khoa": ["khoa", "vien", "bo mon", "nganh hoc", "chuyen nganh"],
            "giang_vien": ["giang vien", "thay", "co", "giao su", "tien si"],
            "nghien_cuu": ["nghien cuu", "de tai", "du an", "khoa hoc"],
            "tuyen_sinh": ["tuyen sinh", "xet tuyen", "chi tieu", "diem chuan", "thi"],
            "tin_tuc": ["tin tuc", "su kien", "hoat dong", "moi nhat"],
            "phong_hoc": ["phong", "tang", "phong hoc", "giang duong"]
        }
        
        # Thong ke
        self.stats = {
            "pages": 0,
            "departments": 0,
            "lecturers": 0,
            "research_projects": 0,
            "rooms": 0
        }
        
        self._update_stats()
    
    def _load_knowledge_base(self):
        """Load du lieu tri thuc tu file JSON"""
        if not os.path.exists(self.config.knowledge_file):
            print(f"[KnowledgeRetriever] Khong tim thay file: {self.config.knowledge_file}")
            return
        
        try:
            with open(self.config.knowledge_file, 'r', encoding='utf-8') as file:
                self.knowledge_data = json.load(file)
            print(f"[KnowledgeRetriever] Da load du lieu tri thuc")
        except Exception as error:
            print(f"[KnowledgeRetriever] Loi load du lieu: {error}")
            self.knowledge_data = None
    
    def _load_building_data(self):
        """Load du lieu toa nha 15 tang tu file JSON"""
        if not os.path.exists(self.config.building_file):
            print(f"[KnowledgeRetriever] Khong tim thay file: {self.config.building_file}")
            return
        
        try:
            with open(self.config.building_file, 'r', encoding='utf-8') as file:
                self.building_data = json.load(file)
            
            # Xay dung chi muc phong de truy xuat nhanh
            self.room_index = {}
            for floor in self.building_data.get('floors', []):
                floor_number = floor.get('floor')
                for room in floor.get('rooms', []):
                    room_code = room.get('code', '').lower()
                    self.room_index[room_code] = {
                        'code': room.get('code'),
                        'name': room.get('name'),
                        'floor': floor_number,
                        'floor_name': floor.get('name', '')
                    }
            
            print(f"[KnowledgeRetriever] Da load du lieu toa nha: {len(self.room_index)} phong")
        except Exception as error:
            print(f"[KnowledgeRetriever] Loi load toa nha: {error}")
            self.building_data = None
            self.room_index = {}
    
    def _load_custom_qa(self):
        """Load cau hoi thuong gap tu file custom"""
        custom_qa_path = os.path.join(os.path.dirname(__file__), "data", "qa_custom.json")
        if os.path.exists(custom_qa_path):
            try:
                with open(custom_qa_path, 'r', encoding='utf-8') as file:
                    return json.load(file)
            except Exception as e:
                print(f"[KnowledgeRetriever] Loi load QA custom: {e}")
                return []
        return []
    
    def _precompute_qa_embeddings(self):
        """Pre-calculate embeddings cho tat ca cau hoi trong custom_qa"""
        if not self.custom_qa or not self.vectorizer:
            return None
            
        print(f"[KnowledgeRetriever] Dang ma hoa {len(self.custom_qa)} cau hoi tri thuc...")
        embeddings = []
        for qa in self.custom_qa:
            emb = self.vectorizer.transform(qa['question'])
            embeddings.append(emb)
        
        return torch.stack(embeddings) if embeddings else None
                
    def _update_stats(self):
        """Cap nhat thong ke du lieu"""
        if self.knowledge_data:
            metadata = self.knowledge_data.get('metadata', {})
            self.stats['pages'] = metadata.get('total_pages', 0)
            self.stats['departments'] = metadata.get('total_departments', len(self.knowledge_data.get('departments', [])))
            self.stats['lecturers'] = metadata.get('total_lecturers', len(self.knowledge_data.get('lecturers', [])))
            self.stats['research_projects'] = metadata.get('total_research', len(self.knowledge_data.get('research_projects', [])))
        
        if self.building_data:
            for floor in self.building_data.get('floors', []):
                self.stats['rooms'] += len(floor.get('rooms', []))
    
    def _remove_accents(self, text: str) -> str:
        import unicodedata
        text = unicodedata.normalize('NFD', text)
        text = re.sub(r'[\u0300-\u036f]', '', text)
        return text.replace('đ', 'd').replace('Đ', 'D')
        
    def detect_topic(self, question: str) -> str:
        """
        Phat hien chu de cua cau hoi
        
        Tham so:
            question: Cau hoi tu nguoi dung
        
        Tra ve:
            Ten chu de (string)
        """
        question_lower = self._remove_accents(question.lower())
        
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in question_lower:
                    return topic
        
        return "khac"
    
    def search_departments(self, query: str) -> List[Dict]:
        """
        Tim kiem khoa/vien theo tu khoa
        
        Tham so:
            query: Tu khoa tim kiem
        
        Tra ve:
            Danh sach cac khoa tim duoc
        """
        if not self.knowledge_data:
            return []
        
        query_lower = self._remove_accents(query.lower())
        results = []
        
        for department in self.knowledge_data.get('departments', []):
            name = self._remove_accents(department.get('name', '').lower())
            description = self._remove_accents(department.get('description', '').lower())
            
            if name and name in query_lower:
                results.append(department)
            elif query_lower in name or query_lower in description:
                results.append(department)
        
        # Neu khong tim thay, dung fuzzy matching
        if not results:
            department_names = [d.get('name', '') for d in self.knowledge_data.get('departments', [])]
            matches = get_close_matches(query_lower, [n.lower() for n in department_names], n=3, cutoff=0.6)
            
            for match in matches:
                for department in self.knowledge_data.get('departments', []):
                    if department.get('name', '').lower() == match:
                        results.append(department)
        
        return results[:5]
    
    def search_lecturers(self, query: str) -> List[Dict]:
        """
        Tim kiem giang vien theo ten hoac linh vuc
        
        Tham so:
            query: Tu khoa tim kiem
        
        Tra ve:
            Danh sach giang vien tim duoc
        """
        if not self.knowledge_data:
            return []
        
        query_lower = self._remove_accents(query.lower())
        results = []
        
        for lecturer in self.knowledge_data.get('lecturers', []):
            name = lecturer.get('name', '').lower()
            title = lecturer.get('title', '').lower()
            research = lecturer.get('research_area', '').lower()
            
            if query_lower in name or query_lower in title or query_lower in research:
                results.append(lecturer)
        
        return results[:5]
    
    def search_room(self, query: str) -> Optional[Dict]:
        """
        Tim kiem thong tin phong hoc theo ma phong
        
        Tham so:
            query: Cau hoi chua ma phong (vi du: "phong 303")
        
        Tra ve:
            Thong tin phong hoac None neu khong tim thay
        """
        # Tim ma phong trong cau hoi
        query_no_accents = self._remove_accents(query.lower())
        room_pattern = re.search(r'phong\s*([A-Za-z]?\d{2,4})', query_no_accents)
        if not room_pattern:
            return None
        
        room_code = room_pattern.group(1).lower()
        
        if room_code in self.room_index:
            return self.room_index[room_code]
        
        return None
    
    def _get_semantic_similarity(self, question: str, candidate_idx: int = -1, candidate_text: str = "") -> float:
        """Tinh do tuong dong ngu nghia (Cosine Similarity neu co PhoBERT, Jaccard neu khong)"""
        
        # Neu dung PhoBERT
        if self.use_phobert and self.vectorizer:
            try:
                q_emb = self.vectorizer.transform(question)
                if candidate_idx >= 0 and self.qa_embeddings is not None:
                    c_emb = self.qa_embeddings[candidate_idx]
                else:
                    c_emb = self.vectorizer.transform(candidate_text)
                
                # Cosine Similarity
                return torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), c_emb.unsqueeze(0)).item()
            except Exception as e:
                print(f"[KnowledgeRetriever] Loi tinh similarity PhoBERT: {e}")
                return 0.0

        # Fallback xuong Jaccard Similarity (dem tu)
        if HAS_PYVI:
            q_words = set(ViTokenizer.tokenize(question.lower()).split())
            c_words = set(ViTokenizer.tokenize(candidate_text.lower()).split())
        else:
            q_words = set(question.lower().split())
            c_words = set(candidate_text.lower().split())
            
        intersection = q_words.intersection(c_words)
        union = q_words.union(c_words)
        return len(intersection) / len(union) if union else 0

    def get_answer(self, question: str) -> Tuple[str, float]:
        """
        Tra loi cau hoi dua tren knowledge base
        
        Tham so:
            question: Cau hoi tu nguoi dung
        
        Tra ve:
            (cau_tra_loi, do_tin_cay)
        """
        if not self.knowledge_data:
            return "Xin loi, toi chua co du lieu ve truong.", 0.0
            
        # Tim trong custom QA voi PhoBERT Semantic Search
        if self.custom_qa:
            best_answer = None
            best_score = 0.0
            
            for i, qa in enumerate(self.custom_qa):
                # Neu dung PhoBERT, truyen index vao de lay embedding co san
                score = self._get_semantic_similarity(question, candidate_idx=i, candidate_text=qa['question'])
                
                if score > best_score:
                    best_score = score
                    best_answer = qa['answer']
            
            # Nguong chap nhan (PhoBERT thuong cao > 0.8, Jaccard > 0.4)
            threshold = 0.85 if self.use_phobert else 0.4
            
            if best_score > threshold:
                return best_answer, 0.95
        
        question_lower = self._remove_accents(question.lower())
        topic = self.detect_topic(question)
        
        # Truong hop 1: Hoi ve phong hoc trong toa nha
        room_info = self.search_room(question)
        if room_info:
            response = f"Phong {room_info['code']} - {room_info['name']} nam o tang {room_info['floor']}"
            return response, 0.95
        
        # Truong hop 2: Hoi ve nam thanh lap
        if topic == "nam_thanh_lap" or "1962" in question_lower:
            return "Truong Dai hoc Giao thong van tai duoc thanh lap nam 1962.", 0.95
        
        # Truong hop 3: Hoi gioi thieu chung
        if topic == "gioi_thieu":
            return ("Truong Dai hoc Giao thong van tai (UTC) la truong dai hoc trong diem "
                    "ve linh vuc giao thong van tai tai Viet Nam, duoc thanh lap nam 1962."), 0.90
        
        # Truong hop 4: Hoi ve khoa
        if topic == "khoa":
            # Tim khoa bang search_departments (co fuzzy matching)
            dept_results = self.search_departments(question)
            if dept_results:
                best_dept = dept_results[0]
                department_name = best_dept.get('name', '')
                description = best_dept.get('description', '')
                if description:
                    return f"Khoa {department_name}: {description[:300]}", 0.85
                else:
                    return f"Khoa {department_name}. Ban co thong tin tren website.", 0.70
            
            # Neu khong tim thay khoa cu the, liet ke cac khoa
            department_names = [d.get('name', '') for d in self.knowledge_data.get('departments', [])[:10]]
            if department_names:
                department_list = ", ".join(department_names)
                return f"Truong co cac khoa/vien: {department_list}...", 0.80
        
        # Truong hop 5: Hoi ve giang vien
        if topic == "giang_vien":
            # Tim ten giang vien cu the
            for lecturer in self.knowledge_data.get('lecturers', []):
                lecturer_name = lecturer.get('name', '')
                if lecturer_name.lower() in question_lower:
                    title = lecturer.get('title', '')
                    research = lecturer.get('research_area', '')
                    response = f"Giang vien {lecturer_name}"
                    if title:
                        response += f" - {title}"
                    if research:
                        response += f". Linh vuc nghien cuu: {research}"
                    return response, 0.85
            
            # Dem so giang vien
            total_lecturers = len(self.knowledge_data.get('lecturers', []))
            return f"Truong co {total_lecturers} giang vien, bao gom nhieu PGS, TS dau nganh.", 0.70
        
        # Truong hop 6: Hoi ve nghien cuu
        if topic == "nghien_cuu":
            research_items = self.knowledge_data.get('research_projects', [])
            if research_items:
                item = research_items[0]
                title = item.get('title', '')
                description = item.get('description', '')
                return f"De tai: {title}. {description[:200]}", 0.75
            else:
                return "Truong co nhieu de tai nghien cuu khoa hoc ve giao thong van tai.", 0.60
        
        # Truong hop 7: Hoi ve tin tuc
        if topic == "tin_tuc":
            news_items = self.knowledge_data.get('news', [])
            if news_items:
                item = news_items[0]
                title = item.get('title', '')
                url = item.get('url', '')
                return f"Tin moi nhat: {title}. Xem chi tiet tai: {url}", 0.70
        
        # Truong hop 8: Hoi ve tuyen sinh
        if topic == "tuyen_sinh":
            admissions = self.knowledge_data.get('admissions', [])
            if admissions:
                item = admissions[0]
                title = item.get('title', '')
                return f"Thong tin tuyen sinh: {title}. Chi tiet tren website truong.", 0.75
            else:
                return ("Thong tin tuyen sinh duoc cap nhat tren website chinh thuc cua truong. "
                        "Ban co the xem tai https://utc.edu.vn/tuyen-sinh"), 0.70
        
        # Truong hop 9: Cau hoi tong quat khong khop
        return ("Toi chua co thong tin chi tiet ve cau hoi nay. "
                "Ban co the tham khao website chinh thuc: https://utc.edu.vn"), 0.50
    
    def get_statistics(self) -> Dict:
        """
        Lay thong ke ve kho tri thuc
        
        Tra ve:
            Dictionary chua thong ke
        """
        return {
            'total_pages': self.stats['pages'],
            'total_departments': self.stats['departments'],
            'total_lecturers': self.stats['lecturers'],
            'total_research': self.stats['research_projects'],
            'total_rooms': self.stats['rooms'],
            'has_building_data': self.building_data is not None,
            'has_knowledge_data': self.knowledge_data is not None
        }
    
    def get_all_departments(self) -> List[str]:
        """Lay danh sach tat ca khoa"""
        if not self.knowledge_data:
            return []
        return [d.get('name', '') for d in self.knowledge_data.get('departments', []) if d.get('name')]
    
    def get_all_lecturers(self) -> List[Dict]:
        """Lay danh sach tat ca giang vien"""
        if not self.knowledge_data:
            return []
        return self.knowledge_data.get('lecturers', [])


# Tao instance mac dinh de cac module khac co the import
default_retriever = KnowledgeRetriever()

if __name__ == "__main__":
    print("\n--- TEST KNOWLEDGE RETRIEVER ---")
    retriever = KnowledgeRetriever()
    print(f"Thống kê: {retriever.get_statistics()}")
    
    test_questions = [
        "Trường Giao thông vận tải thành lập năm nào?",
        "Cho tôi biết thông tin về khoa Công nghệ thông tin",
        "Phòng 303 ở đâu?",
        "Trường có bao nhiêu giảng viên?"
    ]
    
    for q in test_questions:
        ans, conf = retriever.get_answer(q)
        print(f"\nHỏi: '{q}'\nĐáp: {ans} (Tin cậy: {conf*100:.1f}%)")
    print()