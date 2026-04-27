"""
dependency_parser.py - Phân tích cú pháp dependency bằng Stanza
"""

import stanza
import networkx as nx
from typing import Tuple, Optional, Dict, List

from .config import NLPConfig
from .data_structs import ParsedCommand


class DependencyParser:
    """Phân tích cú pháp dependency cho tiếng Việt"""
    
    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        self.nlp_pipeline = None
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load Stanza pipeline cho tiếng Việt"""
        try:
            print("[Stanza] Loading Vietnamese pipeline...")
            self.nlp_pipeline = stanza.Pipeline(
                lang=self.config.STANZA_LANG,
                processors=self.config.STANZA_PROCESSORS,
                dir=self.config.STANZA_MODEL_DIR,
                verbose=False
            )
            print("[Stanza] Pipeline loaded successfully")
        except Exception as e:
            print(f"[Stanza] Error loading pipeline: {e}")
            print("[Stanza] Will download models on first run...")
            stanza.download(self.config.STANZA_LANG)
            self.nlp_pipeline = stanza.Pipeline(
                lang=self.config.STANZA_LANG,
                processors=self.config.STANZA_PROCESSORS,
                verbose=False
            )
    
    def parse(self, text: str) -> Tuple[nx.DiGraph, Dict]:
        """
        Phân tích câu, trả về đồ thị dependency và các thành phần
        
        Args:
            text: Câu tiếng Việt
        
        Returns:
            graph: NetworkX DiGraph biểu diễn dependency
            info: Dict chứa thông tin bổ sung (verb, objects, ...)
        """
        doc = self.nlp_pipeline(text)
        
        if not doc.sentences:
            return nx.DiGraph(), {}
        
        sent = doc.sentences[0]
        
        # Xây dựng đồ thị dependency
        graph = nx.DiGraph()
        
        # Thêm các node (từ)
        for word in sent.words:
            graph.add_node(word.id, text=word.text, upos=word.upos, xpos=word.xpos)
        
        # Thêm các cạnh (dependency)
        for word in sent.words:
            if word.head != 0:  # không phải root
                head_text = sent.words[word.head - 1].text
                graph.add_edge(head_text, word.text, rel=word.deprel)
        
        # Trích xuất thông tin quan trọng
        info = self._extract_info(sent)
        
        return graph, info
    
    def _extract_info(self, sent) -> Dict:
        """Trích xuất verb, object, direction từ câu"""
        info = {
            'verb': None,
            'object': None,
            'direction': None,
            'modifier': None,
            'entities': {}
        }
        
        for word in sent.words:
            # Tìm động từ (VERB)
            if word.upos == 'VERB' or word.xpos in ['V', 'Vvb', 'Vvc']:
                info['verb'] = word.text
            
            # Tìm danh từ làm object
            if word.deprel in ['obj', 'dobj', 'nsubj']:
                info['object'] = word.text
            
            # Tìm hướng (trái, phải, thẳng)
            if word.text.lower() in ['trái', 'phải', 'thẳng', 'lên', 'xuống']:
                info['direction'] = word.text
            
            # Tìm tầng số
            if 'tầng' in word.text or word.text.isdigit():
                info['entities']['floor'] = word.text
        
        return info
    
    def get_dependency_tree_str(self, text: str) -> str:
        """Lấy cây dependency dạng text (debug)"""
        doc = self.nlp_pipeline(text)
        if not doc.sentences:
            return ""
        
        sent = doc.sentences[0]
        lines = []
        for word in sent.words:
            lines.append(f"{word.id}: {word.text} -> head {word.head} ({word.deprel})")
        
        return "\n".join(lines)


# Mock parser cho testing
class MockDependencyParser:
    """Mock parser dùng rule đơn giản, không cần Stanza"""
    
    def parse(self, text: str) -> Tuple[nx.DiGraph, Dict]:
        text_lower = text.lower()
        graph = nx.DiGraph()
        
        info = {'verb': None, 'object': None, 'direction': None, 'entities': {}}
        
        # Rule-based parsing đơn giản
        if 'rẽ' in text_lower or 'quẹo' in text_lower:
            info['verb'] = 'rẽ'
            if 'trái' in text_lower:
                info['direction'] = 'trái'
                graph.add_edge('rẽ', 'trái', rel='direction')
            elif 'phải' in text_lower:
                info['direction'] = 'phải'
                graph.add_edge('rẽ', 'phải', rel='direction')
        
        elif 'dừng' in text_lower:
            info['verb'] = 'dừng'
            graph.add_node('dừng')
        
        elif 'đi' in text_lower or 'chạy' in text_lower:
            info['verb'] = 'đi'
            if 'thẳng' in text_lower:
                info['direction'] = 'thẳng'
                graph.add_edge('đi', 'thẳng', rel='direction')
        
        return graph, info
