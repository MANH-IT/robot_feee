# -*- coding: utf-8 -*-
"""
nlp_processor.py - Bo xu ly NLP tong hop cho robot
Tich hop: Nhan dang giong noi (Vosk) + Phan tich cau phap (Stanza) + SNN Intent + Knowledge
Phiên bản: 3.0 - Hoan chinh, khong con mock
"""

import os
import queue
import threading
import numpy as np
from typing import Optional, Tuple, Dict, Any
import json

# Import cac module con
try:
    from .config import default_config, NLPConfig
    from .intent_classifier import IntentClassifier
    from .knowledge_retriever import KnowledgeRetriever
except ImportError:
    from config import default_config, NLPConfig
    from intent_classifier import IntentClassifier
    from knowledge_retriever import KnowledgeRetriever


class NLPProcessor:
    """
    Bo xu ly NLP tong hop cho robot FEEE
    Tich hop day du cac thanh phan: STT, NLP, Intent, Knowledge
    """
    
    def __init__(self, config: NLPConfig = None, use_mock: bool = False):
        """
        Khoi tao bo xu ly NLP
        
        Tham so:
            config: Cau hinh NLP
            use_mock: Neu True, chi dung rule-based (khong can model)
        """
        self.config = config or default_config
        self.use_mock = use_mock
        
        # Khoi tao cac thanh phan chinh
        self.intent_classifier = IntentClassifier(self.config)
        self.knowledge_retriever = KnowledgeRetriever(self.config)
        
        # Khoi tao Vosk (nhan dang giong noi) neu khong dung mock
        self.vosk_model = None
        self.recognizer = None
        self.audio_queue = None
        
        if not use_mock:
            self._init_vosk()
        
        # Khoi tao Stanza (phan tich cau phap) neu khong dung mock
        self.stanza_pipeline = None
        if not use_mock:
            self._init_stanza()
        
        # Trang thai
        self.is_listening = False
        self.audio_thread = None
        
        print("[NLPProcessor] Khoi tao hoan tat")
        print(f"  - SNN Intent: {'Co' if self.intent_classifier.use_snn else 'Khong (fallback)'}")
        print(f"  - Knowledge: {'Co' if self.knowledge_retriever.knowledge_data else 'Khong'}")
        print(f"  - Vosk: {'Co' if self.vosk_model else 'Khong'}")
        print(f"  - Mock mode: {use_mock}")
    
    def _init_vosk(self):
        """Khoi tao Vosk cho nhan dang giong noi offline"""
        try:
            from vosk import Model, KaldiRecognizer
            
            if os.path.exists(self.config.vosk_model_path):
                self.vosk_model = Model(self.config.vosk_model_path)
                self.recognizer = KaldiRecognizer(self.vosk_model, self.config.sample_rate)
                self.audio_queue = queue.Queue()
                print(f"[NLPProcessor] Vosk khoi tao thanh cong tai {self.config.vosk_model_path}")
            else:
                print(f"[NLPProcessor] Khong tim thay Vosk model tai {self.config.vosk_model_path}")
                print(f"  Vui long tai model ve va giai nen vao thu muc models/vosk_vn")
                self.vosk_model = None
                
        except ImportError:
            print("[NLPProcessor] Thu vien vosk chua duoc cai dat")
            print("  Chay: pip install vosk")
            self.vosk_model = None
        except Exception as error:
            print(f"[NLPProcessor] Loi khoi tao Vosk: {error}")
            self.vosk_model = None
    
    def _init_stanza(self):
        """Khoi tao Stanza cho phan tich cau phap tieng Viet"""
        try:
            import stanza
            
            # Tao thu muc neu chua co
            os.makedirs(self.config.stanza_model_dir, exist_ok=True)
            
            # Tai model neu chua co
            stanza.download(self.config.stanza_lang, model_dir=self.config.stanza_model_dir, verbose=False)
            
            # Khoi tao pipeline
            self.stanza_pipeline = stanza.Pipeline(
                lang=self.config.stanza_lang,
                processors=self.config.stanza_processors,
                model_dir=self.config.stanza_model_dir,
                verbose=False,
                use_gpu=False
            )
            print(f"[NLPProcessor] Stanza khoi tao thanh cong")
            
        except ImportError:
            print("[NLPProcessor] Thu vien stanza chua duoc cai dat")
            print("  Chay: pip install stanza")
            self.stanza_pipeline = None
        except Exception as error:
            print(f"[NLPProcessor] Loi khoi tao Stanza: {error}")
            self.stanza_pipeline = None
    
    def speech_to_text(self, audio_data: bytes) -> Optional[str]:
        """
        Chuyen doi am thanh thanh van ban su dung Vosk
        
        Tham so:
            audio_data: Dữ liệu âm thanh dang bytes
        
        Tra ve:
            Van ban dang string hoac None neu khong nhan dang duoc
        """
        if self.use_mock or self.recognizer is None:
            return None
        
        try:
            if self.recognizer.AcceptWaveform(audio_data):
                result = json.loads(self.recognizer.Result())
                text = result.get('text', '')
                if text:
                    return text
        except Exception as error:
            print(f"[NLPProcessor] Loi nhan dang giong noi: {error}")
        
        return None
    
    def parse_dependency(self, text: str) -> Dict[str, Any]:
        """
        Phan tich cau phap dependency cua cau
        
        Tham so:
            text: Van ban can phan tich
        
        Tra ve:
            Dictionary chua cay dependency
        """
        result = {
            'words': [],
            'dependencies': [],
            'root': None,
            'verb': None,
            'objects': []
        }
        
        if self.use_mock or self.stanza_pipeline is None:
            return result
        
        try:
            doc = self.stanza_pipeline(text)
            if not doc.sentences:
                return result
            
            sentence = doc.sentences[0]
            
            for word in sentence.words:
                result['words'].append({
                    'id': word.id,
                    'text': word.text,
                    'pos': word.upos,
                    'head': word.head,
                    'deprel': word.deprel
                })
                
                # Tim dong tu (VERB)
                if word.upos == 'VERB':
                    result['verb'] = word.text
                
                # Tim tan ngu (dobj)
                if word.deprel == 'obj' or word.deprel == 'dobj':
                    result['objects'].append(word.text)
                
                # Tim root
                if word.head == 0:
                    result['root'] = word.text
            
            # Xay dung dependencies
            for word in sentence.words:
                if word.head != 0:
                    head_word = sentence.words[word.head - 1].text
                    result['dependencies'].append({
                        'head': head_word,
                        'dependent': word.text,
                        'relation': word.deprel
                    })
                    
        except Exception as error:
            print(f"[NLPProcessor] Loi phan tich cau phap: {error}")
        
        return result
    
    def parse_command(self, text: str) -> Dict[str, Any]:
        """
        Phan tich day du cau lenh tu van ban
        
        Tham so:
            text: Van ban cau lenh
        
        Tra ve:
            Dictionary chua ket qua phan tich day du
        """
        # Buoc 1: Phan tich y dinh
        intent_name, confidence = self.intent_classifier.predict_with_confidence(text)
        
        # Buoc 2: Trich xuat thuc the
        entities = self.intent_classifier.extract_entities(text, intent_name)
        
        # Buoc 3: Lay hanh dong tuong ung
        action, _, more_entities = self.intent_classifier.get_action(text)
        entities.update(more_entities)
        
        # Buoc 4: Phan tich cau phap (neu co the)
        dependency = self.parse_dependency(text)
        
        # Buoc 5: Lay cau tra loi (neu la cau hoi)
        response = None
        if intent_name == 'hoi_thong_tin':
            response, _ = self.knowledge_retriever.get_answer(text)
        
        # Xay dung ket qua
        result = {
            'text': text,
            'intent': intent_name,
            'confidence': confidence,
            'action': action,
            'entities': entities,
            'dependency': dependency,
            'response': response,
            'is_valid': confidence >= self.config.confidence_threshold
        }
        
        return result
    
    def process_audio(self, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Xu ly am thanh dau vao: STT -> NLP -> Ket qua
        
        Tham so:
            audio_data: Dữ liệu âm thanh
        
        Tra ve:
            Ket qua phan tich hoac None
        """
        # Buoc 1: Nhan dang giong noi
        text = self.speech_to_text(audio_data)
        if not text:
            return None
        
        # Buoc 2: Phan tich cau lenh
        result = self.parse_command(text)
        
        return result
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Xu ly van ban dau vao (khong qua STT)
        
        Tham so:
            text: Van ban can xu ly
        
        Tra ve:
            Ket qua phan tich
        """
        return self.parse_command(text)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Lay thong ke ve he thong NLP
        
        Tra ve:
            Dictionary chua thong ke
        """
        return {
            'snintent': {
                'available': self.intent_classifier.use_snn,
                'num_classes': len(self.config.intent_classes)
            },
            'knowledge': self.knowledge_retriever.get_statistics(),
            'vosk_available': self.vosk_model is not None,
            'stanza_available': self.stanza_pipeline is not None,
            'mock_mode': self.use_mock,
            'confidence_threshold': self.config.confidence_threshold
        }


# Tao instance mac dinh
default_processor = NLPProcessor()

if __name__ == "__main__":
    import json
    print("\n--- TEST NLP PROCESSOR ---")
    processor = NLPProcessor(use_mock=True) # Dung mock de test nhanh
    
    test_texts = [
        "Robot đi thẳng lên phía trước",
        "Trường thành lập năm nào vậy?",
        "Khoa công nghệ thông tin ở đâu?"
    ]
    
    for text in test_texts:
        print(f"\n[Input]: {text}")
        result = processor.process_text(text)
        print(json.dumps(result, indent=2, ensure_ascii=False))