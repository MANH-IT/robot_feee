import torch
import numpy as np
from typing import List, Tuple, Dict
import random

class EnhancedIntentDataset:
    """Tạo dataset chất lượng cao cho 8 intent classes"""
    
    INTENT_TEMPLATES = {
        'di_chuyen': [
            # Di chuyển cơ bản
            "di thang", "chay thang", "tien len", "di tiep", "thang tien",
            # Rẽ trái
            "re trai", "queo trai", "ru trai", "re sang trai", "quay sang trai",
            # Rẽ phải
            "re phai", "queo phai", "ru phai", "re sang phai", "quay sang phai",
            # Câu lệnh đầy đủ
            "robot di thang", "hay di thang", "di thang di", "cho robot di thang",
            "robot re trai", "robot re phai", "xin moi re trai", "xin moi re phai",
            "di thang ve phia truoc", "tien ve phia truoc", "cu di thang",
            "re trai ngay", "re phai ngay", "quay dau xe", "lui lai", "di lui"
        ],
        'dung_lai': [
            "dung lai", "dung ngay", "dung khung cap", "dung yen", "dung",
            "robot dung lai", "hay dung lai", "dung lai ngay", "dung ngay lap tuc",
            "dung lai di", "dung lai cho toi", "khoan da", "dung nghe"
        ],
        'bam_theo': [
            "bam theo", "di theo", "chay theo", "bam theo toi", "di theo toi",
            "bam theo tao", "di theo sau", "bam theo xe", "di theo nguoi",
            "hay bam theo", "bam theo day", "di theo robot khac"
        ],
        'hoi_thong_tin': [
            "phong 303 o dau", "khoa cntt o tang may", "thu vien mo luc may gio",
            "truong o dau", "co so vat chat the nao", "truong co nhung khoa nao",
            "giang vien gioi nhat", "nganh nao hot nhat", "diem chuan nam nay",
            "truong thanh lap nam nao", "hieu truong la ai", "co bao nhieu sinh vien"
        ],
        'chao_hoi': [
            "xin chao", "hello", "hi", "chao", "chao robot", "chao cau",
            "chao ban", "chao buoi sang", "chao buoi chieu", "chao buoi toi",
            "xin chao robot", "hello robot", "hi robot", "chao than than"
        ],
        'cam_on': [
            "cam on", "cam on ban", "cam on robot", "thank you", "thanks",
            "cam on nhieu", "cam on rat nhieu", "thank", "cam on nhe"
        ],
        'tam_biet': [
            "tam biet", "bye", "goodbye", "hen gap lai", "chao tam biet",
            "tam biet robot", "bye bye", "hen gap lai sau", "di day nhe"
        ],
        'khac': [
            "troi hom nay dep qua", "toi thich an pho", "hom nay la thu may",
            "ban co khoe khong", "toi met qua", "hom nay buon ngu qua"
        ]
    }
    
    @classmethod
    def generate_sample(cls, intent: str, template: str) -> Dict:
        """Tạo một mẫu dữ liệu"""
        # Tạo biến thể của câu lệnh
        text = template
        
        # Thêm nhiễu (từ sai chính tả)
        if random.random() < 0.1:
            noise_words = ['a', 'e', 'o', 'u', 'i']
            if random.random() < 0.5:
                text = text + " " + random.choice(noise_words)
        
        # Thêm tiền tố ngẫu nhiên
        prefixes = ['', 'oi', 'nhe', 'this is', 'hay', 'xin moi']
        if random.random() < 0.15:
            text = random.choice(prefixes) + " " + text if prefixes[0] else text
        
        return {
            'text': text,
            'intent': intent,
            'template': template
        }
    
    @classmethod
    def create_dataset(cls, samples_per_intent: int = 1500) -> List[Dict]:
        """
        Tạo dataset với số mẫu chỉ định cho mỗi intent
        
        Tham số:
            samples_per_intent: Số mẫu cho mỗi intent
        
        Trả về:
            Danh sách các mẫu dữ liệu
        """
        dataset = []
        
        for intent, templates in cls.INTENT_TEMPLATES.items():
            samples_needed = samples_per_intent
            
            # Nhân bản templates nếu cần
            while samples_needed > 0:
                for template in templates:
                    if samples_needed <= 0:
                        break
                    
                    # Tạo 3-5 biến thể từ mỗi template
                    for variation in range(random.randint(3, 5)):
                        if samples_needed <= 0:
                            break
                        
                        sample = cls.generate_sample(intent, template)
                        dataset.append(sample)
                        samples_needed -= 1
        
        # Xáo trộn dữ liệu
        random.shuffle(dataset)
        
        print(f"[Dataset] Đã tạo {len(dataset)} mẫu cho {len(cls.INTENT_TEMPLATES)} intents")
        return dataset
