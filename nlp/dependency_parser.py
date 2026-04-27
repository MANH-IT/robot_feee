# -*- coding: utf-8 -*-
"""
dependency_parser.py - Phan tich cau phap dependency su dung Stanza
Ket hop: Stanza + NetworkX de xay dung do thi dependency
"""

import stanza
import networkx as nx
from typing import Tuple, Dict, List, Optional

try:
    from .config import default_config, NLPConfig
    from .data_structs import ParsedCommand
except ImportError:
    from config import default_config, NLPConfig
    from data_structs import ParsedCommand


class DependencyParser:
    """
    Phan tich cau phap dependency cho tieng Viet
    Su dung Stanza de lay cay dependency va chuyen thanh do thi NetworkX
    """
    
    def __init__(self, config: NLPConfig = None):
        """
        Khoi tao bo phan tich cau phap
        
        Tham so:
            config: Cau hinh NLP
        """
        self.config = config or default_config
        self.nlp_pipeline = None
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load Stanza pipeline cho tieng Viet"""
        try:
            print("[DependencyParser] Dang load Stanza pipeline...")
            
            # Tao thu muc neu chua co
            import os
            os.makedirs(self.config.stanza_model_dir, exist_ok=True)
            
            # Tai model neu chua co
            stanza.download(self.config.stanza_lang, dir=self.config.stanza_model_dir, verbose=False)
            
            # Khoi tao pipeline
            self.nlp_pipeline = stanza.Pipeline(
                lang=self.config.stanza_lang,
                processors=self.config.stanza_processors,
                dir=self.config.stanza_model_dir,
                verbose=False,
                use_gpu=False
            )
            print("[DependencyParser] Stanza pipeline da san sang")
            
        except ImportError as error:
            print(f"[DependencyParser] Loi: stanza chua duoc cai dat - {error}")
            print("  Chay: pip install stanza")
            self.nlp_pipeline = None
        except Exception as error:
            print(f"[DependencyParser] Loi load Stanza: {error}")
            self.nlp_pipeline = None
    
    def parse(self, text: str) -> Tuple[nx.DiGraph, Dict]:
        """
        Phan tich cau, tra ve do thi dependency va cac thanh phan
        
        Tham so:
            text: Cau tieng Viet can phan tich
        
        Tra ve:
            graph: NetworkX DiGraph bieu dien dependency
            info: Dict chua thong tin bo sung (verb, objects, ...)
        """
        graph = nx.DiGraph()
        info = {
            'verb': None,
            'object': None,
            'direction': None,
            'modifier': None,
            'entities': {}
        }
        
        if self.nlp_pipeline is None:
            return graph, info
        
        try:
            doc = self.nlp_pipeline(text)
            
            if not doc.sentences:
                return graph, info
            
            sentence = doc.sentences[0]
            
            # Them cac node (tu) vao do thi
            for word in sentence.words:
                graph.add_node(word.id, text=word.text, upos=word.upos, xpos=word.xpos)
            
            # Them cac canh (dependency)
            for word in sentence.words:
                if word.head != 0:
                    head_text = sentence.words[word.head - 1].text
                    graph.add_edge(head_text, word.text, relation=word.deprel)
            
            # Trich xuat thong tin quan trong
            for word in sentence.words:
                # Tim dong tu (VERB)
                if word.upos == 'VERB' or word.xpos in ['V', 'Vvb', 'Vvc']:
                    info['verb'] = word.text
                
                # Tim danh tu lam object
                if word.deprel in ['obj', 'dobj', 'nsubj']:
                    info['object'] = word.text
                
                # Tim huong (trai, phai, thang)
                if word.text.lower() in ['trai', 'phai', 'thang', 'len', 'xuong']:
                    info['direction'] = word.text
                
                # Tim tang so
                if 'tang' in word.text or word.text.isdigit():
                    info['entities']['floor'] = word.text
            
        except Exception as error:
            print(f"[DependencyParser] Loi phan tich: {error}")
        
        return graph, info
    
    def get_dependency_tree_string(self, text: str) -> str:
        """
        Lay cay dependency dang chuoi (de debug)
        
        Tham so:
            text: Cau can phan tich
        
        Tra ve:
            Chuoi mo ta cay dependency
        """
        doc = self.nlp_pipeline(text)
        if not doc.sentences:
            return ""
        
        sentence = doc.sentences[0]
        lines = []
        for word in sentence.words:
            lines.append(f"{word.id}: {word.text} -> head {word.head} ({word.deprel})")
        
        return "\n".join(lines)


class MockDependencyParser:
    """
    Bo phan tich gia lap (mock) cho testing
    Khong can Stanza, su dung rule don gian
    """
    
    def parse(self, text: str) -> Tuple[nx.DiGraph, Dict]:
        """
        Phan tich bang rule don gian
        
        Tham so:
            text: Cau can phan tich
        
        Tra ve:
            (graph, info)
        """
        text_lower = text.lower()
        graph = nx.DiGraph()
        
        info = {
            'verb': None,
            'object': None,
            'direction': None,
            'modifier': None,
            'entities': {}
        }
        
        # Nhan dien bang rule
        if 're' in text_lower or 'queo' in text_lower:
            info['verb'] = 're'
            if 'trai' in text_lower:
                info['direction'] = 'trai'
                graph.add_edge('re', 'trai', relation='direction')
            elif 'phai' in text_lower:
                info['direction'] = 'phai'
                graph.add_edge('re', 'phai', relation='direction')
        
        elif 'dung' in text_lower:
            info['verb'] = 'dung'
            graph.add_node('dung')
        
        elif 'di' in text_lower or 'chay' in text_lower:
            info['verb'] = 'di'
            if 'thang' in text_lower:
                info['direction'] = 'thang'
                graph.add_edge('di', 'thang', relation='direction')
        return graph, info

if __name__ == "__main__":
    print("\n--- TEST DEPENDENCY PARSER ---")
    parser = DependencyParser()
    test_text = "Robot hãy đi theo anh Mạnh đến phòng 303 ở tầng 3"
    print(f"\nPhân tích câu: '{test_text}'")
    print(parser.get_dependency_tree_string(test_text))
    
    graph, info = parser.parse(test_text)
    print(f"\nTrích xuất thông tin: {info}\n")